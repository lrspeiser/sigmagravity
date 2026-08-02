#!/usr/bin/env python3
"""Build the frozen absolute RX J2129 baryonic 3D field inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0554_all_baryon_route_screen import (
    chandra_paths,
    hst_products,
    prepare_hst_map,
    prepare_xray_maps,
)
from run_p0660_exact_tensor_activation_audit import manifest_sha256, sha256
from run_rxj2129_raw_theory_lensing import load_baryonic_anchors, load_images

from voidscreen.amplitude_activation_3d import exact_amplitude_multipole_activation_3d
from voidscreen.field_solvers import simple_mond_monopole_boundary
from voidscreen.metric_lensing_3d import (
    KPC_M,
    M_SUN_KG,
    lift_surface_density_msun_kpc2_to_si_volume,
)
from voidscreen.observational_resampling import common_resolution_surface_density

G_SI = 6.67430e-11
DEFAULT_CONFIG = ROOT / "configs" / "p0670_spent_rxj2129_absolute_3d_map_build.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_surface(shape: np.ndarray, mass_msun: float, cell_kpc: float) -> np.ndarray:
    values = np.maximum(np.asarray(shape, dtype=float), 0.0)
    integral = float(np.sum(values) * float(cell_kpc) ** 2)
    if integral <= 0.0 or mass_msun <= 0.0:
        raise ValueError("surface shape and target mass must be positive")
    return values * float(mass_msun) / integral


def relative_error(measured: float, expected: float) -> float:
    return abs(float(measured) / float(expected) - 1.0)


def mass_weighted(values: np.ndarray, density: np.ndarray) -> float:
    return float(np.sum(np.asarray(values) * density) / np.sum(density))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    if protocol.get("status") != "frozen_before_any_P0670_map_or_field_score":
        raise RuntimeError("P0670 protocol is not frozen")
    parent = read_json(ROOT / protocol["parent_result"])
    if not parent["all_progression_gates_pass"]:
        raise RuntimeError("P0669 parent no longer passes")

    inputs = protocol["inputs"]
    raw = read_json(ROOT / inputs["raw_lensing_protocol"])
    map_protocol = read_json(ROOT / inputs["map_protocol"])
    acquisition = read_json(ROOT / inputs["map_acquisition"])
    reused = read_json(ROOT / inputs["reused_hst_acquisition"])
    settings = protocol["physical_map"]
    native_axis_arcsec = np.arange(
        float(settings["native_axis_min_arcsec"]),
        float(settings["native_axis_max_arcsec"])
        + 0.5 * float(settings["native_grid_spacing_arcsec"]),
        float(settings["native_grid_spacing_arcsec"]),
    )
    images = load_images(raw)
    context = SimpleNamespace(label=str(inputs["system"]), local=raw)
    star_yx, star_audit = prepare_hst_map(
        map_protocol,
        acquisition,
        reused,
        context,
        images,
        native_axis_arcsec,
    )
    _gas_unmasked_yx, gas_masked_yx, gas_audit = prepare_xray_maps(
        map_protocol,
        acquisition,
        context,
        native_axis_arcsec,
    )
    scale = float(raw["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
    native_axis_kpc = native_axis_arcsec * scale
    x_native, y_native = np.meshgrid(native_axis_kpc, native_axis_kpc, indexing="ij")
    radius_native = np.hypot(x_native, y_native)
    normalization_radius = float(settings["normalization_radius_kpc"])
    support = radius_native <= normalization_radius
    star_shape = np.maximum(np.asarray(star_yx, dtype=float).T, 0.0)
    gas_rate = np.maximum(np.asarray(gas_masked_yx, dtype=float).T, 0.0)
    star_shape[~support] = 0.0
    depth_kpc = 2.0 * np.sqrt(
        np.maximum(normalization_radius**2 - radius_native**2, 0.0)
    )
    gas_shape = np.sqrt(gas_rate * depth_kpc)
    gas_shape[~support] = 0.0

    cells = int(settings["common_cubic_grid_cells"])
    star_resampling = common_resolution_surface_density(star_shape, cells)
    gas_resampling = common_resolution_surface_density(gas_shape, cells)
    axis_kpc = np.linspace(native_axis_kpc[0], native_axis_kpc[-1], cells)
    cell_kpc = float(axis_kpc[1] - axis_kpc[0])
    anchors = load_baryonic_anchors(raw)
    anchor = anchors[np.isclose(anchors.radius_kpc, normalization_radius)]
    if len(anchor) != 1:
        raise RuntimeError("normalization radius is not one exact baryonic anchor")
    gbar = float(10.0 ** float(anchor.iloc[0].log_gbar))
    target_mass_msun = (
        gbar * (normalization_radius * KPC_M) ** 2 / G_SI / M_SUN_KG
    )
    star_fraction = float(settings["stellar_mass_fraction"])
    gas_fraction = float(settings["gas_mass_fraction"])
    if not np.isclose(star_fraction + gas_fraction, 1.0):
        raise RuntimeError("component mass fractions do not sum to one")
    star_mass_target = star_fraction * target_mass_msun
    gas_mass_target = gas_fraction * target_mass_msun
    star_surface = normalize_surface(star_resampling.coarse, star_mass_target, cell_kpc)
    gas_surface = normalize_surface(gas_resampling.coarse, gas_mass_target, cell_kpc)
    star_surface_mass = float(np.sum(star_surface) * cell_kpc**2)
    gas_surface_mass = float(np.sum(gas_surface) * cell_kpc**2)

    star_volume, star_scale_height = lift_surface_density_msun_kpc2_to_si_volume(
        star_surface,
        axis_kpc,
        cell_kpc=cell_kpc,
    )
    gas_volume, gas_scale_height = lift_surface_density_msun_kpc2_to_si_volume(
        gas_surface,
        axis_kpc,
        cell_kpc=cell_kpc,
    )
    spacing_m = cell_kpc * KPC_M
    cell_volume_m3 = spacing_m**3
    star_volume_mass = float(np.sum(star_volume) * cell_volume_m3 / M_SUN_KG)
    gas_volume_mass = float(np.sum(gas_volume) * cell_volume_m3 / M_SUN_KG)
    gravity = protocol["gravity"]
    activation = exact_amplitude_multipole_activation_3d(
        star_volume,
        gas_volume,
        spacing_m,
        a0=float(gravity["a0_m_s2"]),
        coherence_length=float(gravity["coherence_length_kpc"]) * KPC_M,
        coherence_power=float(gravity["coherence_power"]),
    )
    total_volume = star_volume + gas_volume
    boundary = simple_mond_monopole_boundary(
        total_volume,
        spacing_m,
        gravitational_constant=G_SI,
        a0=float(gravity["a0_m_s2"]),
    )
    component_surface_error = max(
        relative_error(star_surface_mass, star_mass_target),
        relative_error(gas_surface_mass, gas_mass_target),
    )
    component_volume_error = max(
        relative_error(star_volume_mass, star_mass_target),
        relative_error(gas_volume_mass, gas_mass_target),
    )
    weighted_sigma = mass_weighted(activation.sigma, total_volume)
    maximum_strong_lens_radius = max(
        raw["baryonic_inputs"]["strong_lens_impact_radius_range_kpc_expected"]
    )
    strong_lens_cells = float(maximum_strong_lens_radius / cell_kpc)
    positive_star_cells = int(np.sum(star_surface > 0.0))
    positive_gas_cells = int(np.sum(gas_surface > 0.0))
    scale_heights_valid = bool(
        np.isfinite(star_scale_height)
        and np.isfinite(gas_scale_height)
        and star_scale_height > 0.0
        and gas_scale_height > 0.0
    )
    bounded_sigma = bool(
        np.all(np.isfinite(activation.sigma))
        and float(np.min(activation.sigma)) >= 0.0
        and float(np.max(activation.sigma)) <= 1.0
    )
    minimum_eigenvalue = float(np.min(activation.minimum_eigenvalue_proxy))
    boundary_valid = bool(
        np.all(np.isfinite(boundary))
        and float(np.max(boundary) - np.min(boundary)) > 0.0
    )
    gates = protocol["predeclared_progression_gates"]
    gate_results = {
        "P0669_parent": bool(parent["all_progression_gates_pass"])
        is bool(gates["P0669_all_progression_gates_pass"]),
        "hst_inputs": int(star_audit["input_count"]) == int(gates["hst_input_count"]),
        "chandra_inputs": int(gas_audit["input_count"])
        == int(gates["chandra_input_count"]),
        "chandra_exposure": float(gas_audit["total_exposure_ks"])
        >= gates["chandra_total_exposure_ks_min"],
        "chandra_events": int(gas_audit["soft_events_on_grid"])
        >= int(gates["chandra_soft_events_on_grid_min"]),
        "positive_stellar_surface": positive_star_cells
        >= int(gates["positive_stellar_surface_cells_min"]),
        "positive_gas_surface": positive_gas_cells
        >= int(gates["positive_gas_surface_cells_min"]),
        "surface_mass": component_surface_error
        <= gates["surface_component_mass_relative_error_max"],
        "volume_mass": component_volume_error
        <= gates["volume_component_mass_relative_error_max"],
        "scale_heights": scale_heights_valid
        is bool(gates["all_scale_heights_finite_positive"]),
        "bounded_sigma": bounded_sigma
        is bool(gates["sigma_finite_and_in_closed_unit_interval"]),
        "positive_eigenvalue": bool(minimum_eigenvalue > 0.0)
        is bool(gates["minimum_constitutive_eigenvalue_strictly_positive"]),
        "cluster_sigma": weighted_sigma
        >= gates["spent_cluster_mass_weighted_sigma_min"],
        "amplitude_gate": activation.amplitude_gate
        >= gates["spent_cluster_amplitude_gate_min"],
        "strong_lens_sampling": strong_lens_cells
        >= gates["strong_lens_maximum_radius_grid_cells_min"],
        "boundary": boundary_valid
        is bool(gates["boundary_finite_and_nonconstant"]),
        "no_new_constants": int(gravity["new_universal_constants_after_P0659"])
        == int(gates["new_universal_constants_after_P0659"]),
        "no_per_object_parameters": int(gravity["per_object_gravity_parameters"])
        == int(gates["per_object_gravity_parameters"]),
        "no_raw_lens_score": not bool(gates["raw_lens_score_computed"]),
        "sealed_targets_untouched": not bool(gates["sealed_target_outcomes_opened"]),
    }
    all_pass = bool(all(gate_results.values()))
    hst_science, hst_weight = hst_products(
        map_protocol,
        acquisition,
        reused,
        context.label,
    )
    input_paths = [hst_science, hst_weight, *chandra_paths(acquisition, context.label)]
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    map_path = output / protocol["outputs"]["map"]
    np.savez_compressed(
        map_path,
        axis_kpc=axis_kpc,
        stellar_surface_density_msun_kpc2=star_surface,
        gas_surface_density_msun_kpc2=gas_surface,
        stellar_volume_density_kg_m3=star_volume,
        gas_volume_density_kg_m3=gas_volume,
        sigma=activation.sigma,
        transport_direction_x=activation.local.transport_direction[0],
        transport_direction_y=activation.local.transport_direction[1],
        transport_direction_z=activation.local.transport_direction[2],
        simple_mond_boundary_m2_s2=boundary,
        a0_m_s2=np.float64(gravity["a0_m_s2"]),
        target_baryon_mass_msun=np.float64(target_mass_msun),
        normalization_radius_kpc=np.float64(normalization_radius),
    )
    metrics = {
        "normalization_gbar_m_s2": gbar,
        "target_baryon_mass_msun": target_mass_msun,
        "stellar_mass_target_msun": star_mass_target,
        "gas_mass_target_msun": gas_mass_target,
        "stellar_surface_mass_msun": star_surface_mass,
        "gas_surface_mass_msun": gas_surface_mass,
        "stellar_volume_mass_msun": star_volume_mass,
        "gas_volume_mass_msun": gas_volume_mass,
        "maximum_component_surface_mass_relative_error": component_surface_error,
        "maximum_component_volume_mass_relative_error": component_volume_error,
        "stellar_scale_height_kpc": star_scale_height,
        "gas_scale_height_kpc": gas_scale_height,
        "positive_stellar_surface_cells": positive_star_cells,
        "positive_gas_surface_cells": positive_gas_cells,
        "multipole_power_gate": activation.multipole.gate,
        "multipole_amplitude_gate": activation.amplitude_gate,
        "dipole_squared": activation.multipole.dipole_squared,
        "quadrupole_squared": activation.multipole.quadrupole_squared,
        "mass_weighted_sigma": weighted_sigma,
        "sigma_maximum": float(np.max(activation.sigma)),
        "minimum_constitutive_eigenvalue_proxy": minimum_eigenvalue,
        "strong_lens_maximum_radius_grid_cells": strong_lens_cells,
        "grid_cells": cells,
        "cell_kpc": cell_kpc,
        "boundary_range_m2_s2": float(np.max(boundary) - np.min(boundary)),
    }
    report = {
        "report_version": "P0670-SPENT-RXJ2129-ABSOLUTE-3D-MAP-BUILD-RESULTS-1.0.0",
        "status": "pass" if all_pass else "fail",
        "all_progression_gates_pass": all_pass,
        "candidate_advanced_to_spent_scalar_tensor_field_solve": all_pass,
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__).resolve()),
        "activation_source_sha256": sha256(
            ROOT / "src/voidscreen/amplitude_activation_3d.py"
        ),
        "input_manifest_sha256": manifest_sha256(input_paths),
        "map_sha256": sha256(map_path),
        "map_audits": {"hst": star_audit, "chandra": gas_audit},
        "metrics": metrics,
        "gate_results": gate_results,
        "spent_image_coordinates_used_only_for_HST_mask": True,
        "raw_lens_score_computed": False,
        "sealed_P0633_kinematics_opened": False,
        "sealed_P0640_lensing_constraints_opened": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    extent = [axis_kpc[0], axis_kpc[-1], axis_kpc[0], axis_kpc[-1]]
    for axis, values, title in zip(
        axes[:2],
        (star_surface, gas_surface),
        ("stellar surface mass", "gas surface mass"),
        strict=True,
    ):
        positive = values[values > 0.0]
        floor = float(np.quantile(positive, 0.05)) if positive.size else 1.0
        image = axis.imshow(
            np.log10(values.T + floor),
            origin="lower",
            extent=extent,
            cmap="magma",
        )
        axis.set(title=title, xlabel="x (kpc)", ylabel="y (kpc)")
        figure.colorbar(image, ax=axis, shrink=0.75)
    center = cells // 2
    image = axes[2].imshow(
        activation.sigma[:, :, center].T,
        origin="lower",
        extent=extent,
        cmap="viridis",
    )
    axes[2].set(title="central tensor sigma", xlabel="x (kpc)", ylabel="y (kpc)")
    figure.colorbar(image, ax=axes[2], shrink=0.75)
    figure.tight_layout()
    figure.savefig(output / "p0670_absolute_3d_map.png", dpi=180)
    plt.close(figure)
    failed = [name for name, passed in gate_results.items() if not passed]
    summary = f"""# P0670 spent RX J2129 absolute 3D map build

- Status: **{'PASS' if all_pass else 'FAIL'}**.
- Independent baryonic mass inside 200 kpc: **{target_mass_msun:.6g} Msun**.
- Surface/volume mass errors: **{component_surface_error:.3g} / {component_volume_error:.3g}**.
- Multipole amplitude gate: **{activation.amplitude_gate:.6g}**.
- Mass-weighted tensor sigma: **{weighted_sigma:.6g}**.
- 3D cell size and strong-lens sampling: **{cell_kpc:.4g} kpc / {strong_lens_cells:.3g} cells**.
- Failed frozen gates: **{', '.join(failed) if failed else 'none'}**.
- Raw lens score computed: **no**; sealed targets opened: **no**.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
