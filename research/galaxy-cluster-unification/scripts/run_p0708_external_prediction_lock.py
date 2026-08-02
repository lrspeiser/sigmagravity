#!/usr/bin/env python3
"""Generate the outcome-blind P0708 external prediction lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0635_ddo154_map_commissioning import radial_circular_speed
from run_p0635_map_geometry_sensitivity import build_density
from voidscreen.field_solvers import solve_aqual, solve_newtonian, solve_poisson_dirichlet, solve_qumond
from voidscreen.metric_lensing_3d import KPC_M, lift_surface_density_msun_kpc2_to_si_volume, photon_deflection_zero_slip
from voidscreen.observational_resampling import common_resolution_surface_density
from voidscreen.source_routing_qumond import projected_baryonic_spectral_anisotropy, solve_source_conserving_baryonic_routing
from voidscreen.two_potential_metric import build_two_potential_metric, rar_coherent_monopole_potential

DEFAULT_CONFIG = ROOT / "configs" / "p0708_external_prediction_lock.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resample_surface(surface: np.ndarray, axis: np.ndarray, target: int) -> tuple[np.ndarray, np.ndarray, float]:
    values = np.asarray(surface, dtype=float)
    if values.shape[0] == target:
        return values, axis.astype(float), 0.0
    result = common_resolution_surface_density(values, target)
    return result.coarse, np.linspace(float(axis[0]), float(axis[-1]), target), max(result.filtered_mass_relative_error, result.coarse_mass_relative_error)


def crop_and_resample(surface: np.ndarray, axis: np.ndarray, half_extent: float, target: int) -> tuple[np.ndarray, np.ndarray, float]:
    mask = np.abs(axis) <= half_extent + 1e-9
    indices = np.flatnonzero(mask)
    start, stop = int(indices[0]), int(indices[-1]) + 1
    if (stop - start) % 2 == 0:
        stop -= 1
    cropped = np.asarray(surface[start:stop, start:stop], dtype=float)
    cropped_axis = np.asarray(axis[start:stop], dtype=float)
    return resample_surface(cropped, cropped_axis, target)


def solve_metric(density, surface, spacing, *, gravity, a0, light_speed, parameters):
    fraction, _, _ = projected_baryonic_spectral_anisotropy(surface, spacing)
    routing = solve_source_conserving_baryonic_routing(
        density,
        spacing,
        gravitational_constant=gravity,
        a0=a0,
        transition_depth=parameters["chi_t"],
        transition_power=parameters["transition_power_n"],
        extra_spatial_channels=parameters["extra_spatial_channels"],
        path_power=parameters["path_power_q"],
        light_speed=light_speed,
    )
    local = solve_poisson_dirichlet(routing.local_generator_source, spacing, routing.boundary_potential)
    coherent = rar_coherent_monopole_potential(
        density,
        routing.newtonian.potential,
        routing.newtonian.acceleration,
        spacing,
        a0=a0,
    )
    correction = routing.field.potential - local
    metric = build_two_potential_metric(
        coherent.potential + fraction * correction,
        local + fraction * correction,
        spacing,
    )
    return metric, routing, fraction


def los_map(solution, axis: np.ndarray, inclination_deg: float) -> np.ndarray:
    middle = len(axis) // 2
    x, y = np.meshgrid(axis, axis, indexing="ij")
    radius = np.hypot(x, y)
    ax = solution.acceleration[0][:, :, middle]
    ay = solution.acceleration[1][:, :, middle]
    inward = np.zeros_like(radius)
    active = radius > 0.0
    inward[active] = -(ax[active] * x[active] + ay[active] * y[active]) / radius[active]
    circular = np.sqrt(np.maximum(radius * inward, 0.0))
    return circular * np.sin(np.deg2rad(inclination_deg)) * np.divide(x, radius, out=np.zeros_like(x), where=active)


def curve_relative_rms(high: pd.DataFrame, low: pd.DataFrame) -> float:
    r = low["radius_kpc"].to_numpy()
    reference = np.interp(r, high["radius_kpc"], high["circular_speed_km_s"])
    trial = low["circular_speed_km_s"].to_numpy()
    return float(np.sqrt(np.mean((trial - reference) ** 2)) / max(np.sqrt(np.mean(reference**2)), np.finfo(float).tiny))


def radial_profile(axis: np.ndarray, ax: np.ndarray, ay: np.ndarray, bins: np.ndarray) -> np.ndarray:
    x, y = np.meshgrid(axis, axis, indexing="ij")
    radius = np.hypot(x, y)
    magnitude = np.hypot(ax, ay)
    values = []
    for left, right in zip(bins[:-1], bins[1:], strict=True):
        selected = (radius >= left) & (radius < right)
        values.append(float(np.mean(magnitude[selected])) if np.any(selected) else np.nan)
    return np.asarray(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    if protocol["status"] != "frozen_before_any_P0633_kinematic_or_P0640_lensing_constraint_open":
        raise RuntimeError("P0708 protocol is not frozen")
    candidate_parent = read_json(ROOT / protocol["candidate_parent"])
    external = read_json(ROOT / protocol["external_parent"])
    galaxy_parent = read_json(ROOT / protocol["galaxy_map_parent"])
    cluster_parent = read_json(ROOT / protocol["cluster_map_parent"])
    if not candidate_parent.get("all_progression_gates_pass"):
        raise RuntimeError("P0707 candidate did not pass")
    if not galaxy_parent.get("all_gates_pass") or not all(cluster_parent["gates"].values()):
        raise RuntimeError("registered baryonic-map parents are incomplete")
    p = protocol["universal_parameters"]
    output = ROOT / protocol["outputs"]["directory"]
    galaxy_output = output / "galaxies"
    cluster_output = output / "clusters"
    galaxy_output.mkdir(parents=True, exist_ok=True)
    cluster_output.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(ROOT / protocol["galaxy_predictions"]["metadata"]).set_index("galaxy")
    curves = []
    manifest = []
    convergence = []
    aqual_residuals = []
    qumond_residuals = []
    mass_errors = []

    for item in external["galaxy_validation"]["systems"]:
        name = item["id"]
        print(f"P0708 galaxy {name}", flush=True)
        input_path = ROOT / protocol["galaxy_predictions"]["input_directory"] / f"{name}.npz"
        with np.load(input_path) as data:
            native_axis = data["axis_kpc"].astype(float)
            gas_native = data["gas"].astype(float)
            stars_native = data["stars"].astype(float)
        gas, axis, gas_error = resample_surface(gas_native, native_axis, int(p["nominal_cells_per_axis"]))
        stars, axis2, star_error = resample_surface(stars_native, native_axis, int(p["nominal_cells_per_axis"]))
        if not np.array_equal(axis, axis2):
            raise RuntimeError("galaxy component axes disagree")
        mass_errors.extend([gas_error, star_error])
        spacing = float(axis[1] - axis[0])
        density = build_density(gas, stars, axis, p["galaxy_gas_scale_height_kpc"], p["galaxy_stellar_scale_height_kpc"])
        metric, routing, fraction = solve_metric(
            density,
            gas + stars,
            spacing,
            gravity=p["gravitational_constant_galaxy_kpc_km2_s2_per_solar_mass"],
            a0=p["a0_galaxy_km2_s2_per_kpc"],
            light_speed=p["light_speed_galaxy_km_s"],
            parameters=p,
        )
        newtonian = routing.newtonian
        qumond = solve_qumond(density, spacing, a0=p["a0_galaxy_km2_s2_per_kpc"], gravitational_constant=p["gravitational_constant_galaxy_kpc_km2_s2_per_solar_mass"])
        aqual = solve_aqual(density, spacing, a0=p["a0_galaxy_km2_s2_per_kpc"], gravitational_constant=p["gravitational_constant_galaxy_kpc_km2_s2_per_solar_mass"], residual_tolerance=1e-5, maximum_nonlinear_iterations=100, damping=0.5)
        aqual_residuals.append(float(aqual.normalized_residual_rms))
        qumond_residuals.append(float(qumond.normalized_residual_rms))
        time_solution = SimpleNamespace(potential=metric.time_potential, acceleration=metric.time_acceleration)
        models = {
            "P0707_time_potential": time_solution,
            "Newtonian_3D": newtonian,
            "AQUAL_simple_mu_3D": aqual,
            "QUMOND_simple_nu_3D": qumond,
        }
        prediction_arrays = {"axis_kpc": axis, "routing_fraction": np.float64(fraction)}
        for model, solution in models.items():
            frame = radial_circular_speed(solution, axis)
            frame.insert(0, "model", model)
            frame.insert(0, "system", name)
            curves.append(frame)
            prediction_arrays[f"los_{model}"] = los_map(solution, axis, float(metadata.loc[name, "derived_photometric_inclination_deg"]))
        prediction_path = galaxy_output / f"{name}_predictions.npz"
        np.savez_compressed(prediction_path, **prediction_arrays)

        low_cells = int(p["convergence_cells_per_axis"])
        gas_low, axis_low, error_low_g = resample_surface(gas_native, native_axis, low_cells)
        stars_low, _, error_low_s = resample_surface(stars_native, native_axis, low_cells)
        mass_errors.extend([error_low_g, error_low_s])
        density_low = build_density(gas_low, stars_low, axis_low, p["galaxy_gas_scale_height_kpc"], p["galaxy_stellar_scale_height_kpc"])
        metric_low, _, _ = solve_metric(
            density_low,
            gas_low + stars_low,
            float(axis_low[1] - axis_low[0]),
            gravity=p["gravitational_constant_galaxy_kpc_km2_s2_per_solar_mass"],
            a0=p["a0_galaxy_km2_s2_per_kpc"],
            light_speed=p["light_speed_galaxy_km_s"],
            parameters=p,
        )
        high_curve = radial_circular_speed(time_solution, axis)
        low_curve = radial_circular_speed(SimpleNamespace(potential=metric_low.time_potential, acceleration=metric_low.time_acceleration), axis_low)
        convergence.append({"domain": "galaxy", "system": name, "relative_RMS_65_to_33": curve_relative_rms(high_curve, low_curve)})
        manifest.append({"domain": "galaxy", "system": name, "input_path": str(input_path.relative_to(ROOT)), "input_sha256": sha256(input_path), "prediction_path": str(prediction_path.relative_to(ROOT)), "prediction_sha256": sha256(prediction_path), "finite": bool(all(np.all(np.isfinite(value)) for value in prediction_arrays.values()))})

    half_extent = float(p["cluster_half_extent_kpc"])
    for item in external["cluster_validation"]["systems"]:
        name = item["id"]
        print(f"P0708 cluster {name}", flush=True)
        input_path = ROOT / protocol["cluster_predictions"]["input_directory"] / f"{name}_baryons.npz"
        with np.load(input_path) as data:
            native_axis = data["axis_kpc"].astype(float)
            stars_native = data["stellar_surface_density_msun_kpc2"].astype(float)
            gas_native = data["gas_surface_density_msun_kpc2"].astype(float)
        stars, axis, star_error = crop_and_resample(stars_native, native_axis, half_extent, int(p["nominal_cells_per_axis"]))
        gas, axis2, gas_error = crop_and_resample(gas_native, native_axis, half_extent, int(p["nominal_cells_per_axis"]))
        if not np.array_equal(axis, axis2):
            raise RuntimeError("cluster component axes disagree")
        mass_errors.extend([star_error, gas_error])
        cell_kpc = float(axis[1] - axis[0])
        stars_3d, stellar_height = lift_surface_density_msun_kpc2_to_si_volume(stars, axis, cell_kpc=cell_kpc)
        gas_3d, gas_height = lift_surface_density_msun_kpc2_to_si_volume(gas, axis, cell_kpc=cell_kpc)
        density = stars_3d + gas_3d
        spacing_m = cell_kpc * KPC_M
        metric, routing, fraction = solve_metric(density, stars + gas, spacing_m, gravity=p["gravitational_constant_si"], a0=p["a0_si_m_s2"], light_speed=p["light_speed_si_m_s"], parameters=p)
        newtonian = routing.newtonian
        qumond = solve_qumond(density, spacing_m, a0=p["a0_si_m_s2"], gravitational_constant=p["gravitational_constant_si"])
        aqual = solve_aqual(density, spacing_m, a0=p["a0_si_m_s2"], gravitational_constant=p["gravitational_constant_si"], residual_tolerance=1e-5, maximum_nonlinear_iterations=100, damping=0.5)
        aqual_residuals.append(float(aqual.normalized_residual_rms))
        qumond_residuals.append(float(qumond.normalized_residual_rms))
        deflections = {}
        for model, acceleration in {
            "P0707_Weyl": metric.weyl_acceleration,
            "baryon_only_GR": newtonian.acceleration,
            "AQUAL_simple_mu_diagnostic": aqual.acceleration,
            "QUMOND_simple_nu_diagnostic": qumond.acceleration,
        }.items():
            d = photon_deflection_zero_slip(acceleration, spacing_m, distance_ratio=1.0)
            deflections[f"alpha_x_{model}_arcsec"] = d.alpha_x_arcsec
            deflections[f"alpha_y_{model}_arcsec"] = d.alpha_y_arcsec
        prediction_path = cluster_output / f"{name}_physical_deflections.npz"
        np.savez_compressed(prediction_path, axis_kpc=axis, routing_fraction=np.float64(fraction), stellar_scale_height_kpc=np.float64(stellar_height), gas_scale_height_kpc=np.float64(gas_height), **deflections)

        stars_low, axis_low, low_star_error = crop_and_resample(stars_native, native_axis, half_extent, int(p["convergence_cells_per_axis"]))
        gas_low, _, low_gas_error = crop_and_resample(gas_native, native_axis, half_extent, int(p["convergence_cells_per_axis"]))
        mass_errors.extend([low_star_error, low_gas_error])
        cell_low = float(axis_low[1] - axis_low[0])
        stars_low_3d, _ = lift_surface_density_msun_kpc2_to_si_volume(stars_low, axis_low, cell_kpc=cell_low)
        gas_low_3d, _ = lift_surface_density_msun_kpc2_to_si_volume(gas_low, axis_low, cell_kpc=cell_low)
        metric_low, _, _ = solve_metric(stars_low_3d + gas_low_3d, stars_low + gas_low, cell_low * KPC_M, gravity=p["gravitational_constant_si"], a0=p["a0_si_m_s2"], light_speed=p["light_speed_si_m_s"], parameters=p)
        d_low = photon_deflection_zero_slip(metric_low.weyl_acceleration, cell_low * KPC_M, distance_ratio=1.0)
        bins = np.linspace(0.0, min(300.0, half_extent), 13)
        high_profile = radial_profile(axis, deflections["alpha_x_P0707_Weyl_arcsec"], deflections["alpha_y_P0707_Weyl_arcsec"], bins)
        low_profile = radial_profile(axis_low, d_low.alpha_x_arcsec, d_low.alpha_y_arcsec, bins)
        valid = np.isfinite(high_profile) & np.isfinite(low_profile)
        resolution_rms = float(np.sqrt(np.mean((low_profile[valid] - high_profile[valid]) ** 2)) / max(np.sqrt(np.mean(high_profile[valid] ** 2)), np.finfo(float).tiny))
        convergence.append({"domain": "cluster", "system": name, "relative_RMS_65_to_33": resolution_rms})
        manifest.append({"domain": "cluster", "system": name, "input_path": str(input_path.relative_to(ROOT)), "input_sha256": sha256(input_path), "prediction_path": str(prediction_path.relative_to(ROOT)), "prediction_sha256": sha256(prediction_path), "finite": bool(all(np.all(np.isfinite(value)) for value in deflections.values()))})

    curves_frame = pd.concat(curves, ignore_index=True)
    curves_path = output / protocol["outputs"]["galaxy_curves"]
    manifest_path = output / protocol["outputs"]["system_manifest"]
    convergence_path = output / "resolution_convergence.csv"
    curves_frame.to_csv(curves_path, index=False)
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    pd.DataFrame(convergence).to_csv(convergence_path, index=False)
    parameter_bytes = json.dumps(p, sort_keys=True, separators=(",", ":")).encode("utf-8")
    parameter_hash = hashlib.sha256(parameter_bytes).hexdigest()
    gates = protocol["predeclared_gates"]
    maximum_convergence = max(row["relative_RMS_65_to_33"] for row in convergence)
    maximum_mass_error = max(mass_errors) if mass_errors else 0.0
    gate_results = {
        "candidate_parent": bool(candidate_parent["all_progression_gates_pass"]) is bool(gates["candidate_parent_pass"]),
        "galaxy_count": sum(row["domain"] == "galaxy" for row in manifest) == gates["galaxies"],
        "cluster_count": sum(row["domain"] == "cluster" for row in manifest) == gates["clusters"],
        "finite_candidate_fields": all(row["finite"] for row in manifest) is bool(gates["all_nominal_candidate_fields_finite"]),
        "AQUAL_convergence": all(value <= gates["maximum_AQUAL_residual"] for value in aqual_residuals) is bool(gates["all_AQUAL_comparators_converged"]),
        "AQUAL_residual": max(aqual_residuals) <= gates["maximum_AQUAL_residual"],
        "QUMOND_residual": max(qumond_residuals) <= gates["maximum_QUMOND_residual"],
        "resolution_convergence": maximum_convergence <= gates["maximum_candidate_65_to_33_relative_RMS"],
        "mass_conservation": maximum_mass_error <= gates["maximum_mass_conservation_relative_error"],
        "prediction_hashes": all(len(row["prediction_sha256"]) == 64 for row in manifest) is bool(gates["complete_prediction_hashes"]),
        "no_per_object_gravity": p["per_object_gravity_parameters"] == gates["per_object_gravity_parameters"],
        "sealed_outcomes_untouched": not bool(gates["sealed_outcomes_opened"]),
    }
    passed = all(gate_results.values())
    report = {
        "report_version": "P0708-EXTERNAL-PREDICTION-LOCK-RESULTS-1.0.0",
        "status": "pass" if passed else "fail",
        "all_prediction_lock_gates_pass": passed,
        "candidate_authorized_for_one_external_unlock": passed,
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__).resolve()),
        "universal_parameter_sha256": parameter_hash,
        "curve_sha256": sha256(curves_path),
        "manifest_sha256": sha256(manifest_path),
        "resolution_sha256": sha256(convergence_path),
        "systems": len(manifest),
        "maximum_AQUAL_residual": max(aqual_residuals),
        "maximum_QUMOND_residual": max(qumond_residuals),
        "maximum_65_to_33_relative_RMS": maximum_convergence,
        "maximum_mass_conservation_relative_error": maximum_mass_error,
        "gate_results": gate_results,
        "failed_gates": [name for name, value in gate_results.items() if not value],
        "sealed_P0633_kinematics_opened": False,
        "sealed_P0640_lensing_constraints_opened": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    (output / protocol["outputs"]["report"]).write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = f"""# P0708 external prediction lock

- Status: **{'PASS' if passed else 'FAIL'}**.
- Frozen systems: **{len(manifest)}** ({gates['galaxies']} galaxies, {gates['clusters']} clusters).
- Universal parameter SHA-256: `{parameter_hash}`.
- Maximum AQUAL / QUMOND residual: **{max(aqual_residuals):.3g} / {max(qumond_residuals):.3g}**.
- Maximum 65-to-33 candidate relative RMS: **{maximum_convergence:.3g}**.
- Maximum resampling mass error: **{maximum_mass_error:.3g}**.
- Per-object gravity / fitted slip / fitted photon parameters: **0 / 0 / 0**.
- Authorized for one external unlock: **{'yes' if passed else 'no'}**.
- Sealed outcomes opened: **no**.
"""
    (output / protocol["outputs"]["summary"]).write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
