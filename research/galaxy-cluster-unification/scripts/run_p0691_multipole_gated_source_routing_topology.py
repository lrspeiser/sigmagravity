#!/usr/bin/env python3
"""Run the frozen P0691 multipole-gated 3D field and raw topology test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_adaptive_route_raw_rxj2129 import json_safe
from run_p0660_exact_tensor_activation_audit import sha256
from run_p0672_spent_rxj2129_absolute_raw_topology import (
    AbsoluteGridLens,
    PhysicalDeflectionGrid,
    exact_fit,
    global_topology,
    load_images,
    near_bound_count,
    split_images,
    topology_summary,
)

from voidscreen.field_solvers import boundary_mask
from voidscreen.metric_lensing_3d import (
    KPC_M,
    normalized_deflection_curl,
    photon_deflection_zero_slip,
)
from voidscreen.source_routing_qumond import solve_multipole_gated_source_routing

G_SI = 6.67430e-11
DEFAULT_CONFIG = ROOT / "configs" / "p0691_multipole_gated_source_routing_topology.json"
MODEL = "multipole_gated_routing_P0691"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def vector_rms(x_values: np.ndarray, y_values: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x_values[mask] ** 2 + y_values[mask] ** 2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    if protocol.get("status") != "frozen_before_any_P0691_mixed_field_photon_or_topology_score":
        raise RuntimeError("P0691 protocol is not frozen")
    failure_path = ROOT / protocol["failure_parent"]
    failure = read_json(failure_path)
    map_parent_path = ROOT / protocol["map_parent"]
    map_parent = read_json(map_parent_path)
    expected = protocol["predeclared_integrity_gates"]
    if failure.get("status") != expected["P0690_status"]:
        raise RuntimeError("P0690 status changed")
    if bool(failure.get("candidate_advanced_to_real_2D_and_robustness")) != bool(
        expected["P0690_candidate_advanced_to_real_2D_and_robustness"]
    ):
        raise RuntimeError("P0690 advancement state changed")
    if not map_parent["all_progression_gates_pass"]:
        raise RuntimeError("P0670 physical-map parent no longer passes")

    map_path = ROOT / protocol["map_input"]
    with np.load(map_path) as data:
        axis_kpc = data["axis_kpc"].astype(float)
        density = data["stellar_volume_density_kg_m3"].astype(float) + data[
            "gas_volume_density_kg_m3"
        ].astype(float)
        map_a0 = float(data["a0_m_s2"])
    equation = protocol["equation"]
    a0 = float(equation["a0_m_s2"])
    if not np.isclose(a0, map_a0, rtol=0.0, atol=0.0):
        raise RuntimeError("P0691 a0 no longer matches the physical map")
    spacing_m = float(axis_kpc[1] - axis_kpc[0]) * KPC_M

    print("solving multipole-gated source routing", flush=True)
    solution = solve_multipole_gated_source_routing(
        density,
        spacing_m,
        gravitational_constant=G_SI,
        a0=a0,
        transition_depth=float(equation["chi_t"]),
        transition_power=float(equation["transition_power_n"]),
        extra_spatial_channels=float(equation["extra_spatial_channels"]),
        path_power=float(equation["path_power_q"]),
    )
    q_b = solution.quadrupole_fraction
    expected_source = (
        1.0 - q_b
    ) * solution.routing.local_generator_source + q_b * solution.routing.routed_source
    identity_rms = float(np.sqrt(np.mean(np.square(solution.mixed_source - expected_source))))
    identity_scale = max(
        float(np.sqrt(np.mean(np.square(expected_source)))),
        np.finfo(float).tiny,
    )
    identity_relative = identity_rms / identity_scale
    deflection = photon_deflection_zero_slip(
        solution.field.acceleration,
        spacing_m,
        distance_ratio=1.0,
    )
    magnitude = np.hypot(deflection.alpha_x_arcsec, deflection.alpha_y_arcsec)
    x_kpc, y_kpc = np.meshgrid(axis_kpc, axis_kpc, indexing="ij")
    annulus = (np.hypot(x_kpc, y_kpc) >= 15.8) & (np.hypot(x_kpc, y_kpc) <= 76.5)
    field_median = float(np.median(magnitude[annulus]))
    field_rms = vector_rms(
        deflection.alpha_x_arcsec,
        deflection.alpha_y_arcsec,
        annulus,
    )
    field_curl = normalized_deflection_curl(
        deflection.alpha_x_arcsec,
        deflection.alpha_y_arcsec,
        float(axis_kpc[1] - axis_kpc[0]),
    )
    edge = boundary_mask(density.shape)
    boundary_scale = max(
        float(np.max(np.abs(solution.routing.boundary_potential[edge]))),
        np.finfo(float).tiny,
    )
    boundary_mismatch = float(
        np.max(np.abs(solution.field.potential[edge] - solution.routing.boundary_potential[edge]))
        / boundary_scale
    )
    finite = bool(
        np.all(np.isfinite(solution.mixed_source))
        and np.all(np.isfinite(solution.field.potential))
        and all(np.all(np.isfinite(item)) for item in solution.field.acceleration)
        and np.all(np.isfinite(magnitude))
    )

    raw = read_json(ROOT / protocol["raw_protocol"])
    images = load_images(raw)
    training, heldout = split_images(images, raw)
    scale = float(raw["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
    grid = PhysicalDeflectionGrid(
        axis_kpc / scale,
        deflection.alpha_x_arcsec,
        deflection.alpha_y_arcsec,
    )
    lens = AbsoluteGridLens(raw, {MODEL: grid})
    fit = exact_fit(lens, MODEL, training, heldout, protocol, seed_offset=0)
    prediction = pd.concat(
        [fit["training_prediction"], fit["heldout_prediction"]],
        ignore_index=True,
    )
    fit_score = {
        "model": MODEL,
        "training_RMS_arcsec": fit["training_score"]["exact_radial_RMS_arcsec"],
        "training_roots_converged": fit["training_score"]["converged_roots"],
        "heldout_RMS_arcsec": fit["heldout_score"]["exact_radial_RMS_arcsec"],
        "heldout_roots_converged": fit["heldout_score"]["converged_roots"],
        "optimizer_cost": fit["optimizer_cost"],
        "nuisance_parameters_near_bound": near_bound_count(fit["parameters"]),
    }
    parameter_rows = [
        {
            "model": MODEL,
            "parameter": label,
            "value": float(value),
            "lower": float(lower),
            "upper": float(upper),
        }
        for label, value, lower, upper in zip(
            AbsoluteGridLens.labels,
            fit["parameters"],
            AbsoluteGridLens.lower,
            AbsoluteGridLens.upper,
            strict=True,
        )
    ]
    roots, assignments, families, critical_maps = global_topology(
        lens,
        MODEL,
        fit,
        images,
        protocol["global_topology"],
    )
    topology = topology_summary(families)
    compact_report = read_json(ROOT / protocol["compact_halo_comparator_report"])
    compact_halo = float(
        compact_report["model_scores"]["GR_plus_cluster_halo"]["heldout"]["exact_radial_RMS_arcsec"]
    )
    heldout_rms = float(fit_score["heldout_RMS_arcsec"])
    halo_ratio = heldout_rms / compact_halo if np.isfinite(heldout_rms) else float("inf")

    gates = protocol["predeclared_advancement_gates"]
    gate_results = {
        "P0690_parent": failure.get("status") == expected["P0690_status"],
        "P0670_parent": bool(map_parent["all_progression_gates_pass"])
        is bool(expected["P0670_all_progression_gates_pass"]),
        "coverage": len(training) == int(expected["training_images"])
        and len(heldout) == int(expected["spent_heldout_images"])
        and int(images.source_family.nunique()) == int(expected["source_families"]),
        "quadrupole_lower": q_b >= float(gates["quadrupole_fraction_min"]),
        "quadrupole_upper": q_b <= float(gates["quadrupole_fraction_max"]),
        "mixed_source_identity": identity_relative
        <= float(gates["mixed_source_linear_identity_relative_RMS_max"]),
        "field_residual": solution.field.normalized_residual_rms
        <= float(gates["field_normalized_residual_RMS_max"]),
        "boundary": boundary_mismatch <= float(gates["boundary_maximum_relative_mismatch_max"]),
        "finite": finite
        is bool(gates["all_sources_potentials_accelerations_and_deflections_finite"]),
        "field_amplitude_lower": field_median
        >= float(gates["strong_lens_median_physical_deflection_arcsec_min"]),
        "field_amplitude_upper": field_median
        <= float(gates["strong_lens_median_physical_deflection_arcsec_max"]),
        "field_curl": field_curl <= float(gates["normalized_deflection_curl_RMS_max"]),
        "training_roots": int(fit_score["training_roots_converged"])
        == int(gates["training_roots_converged"]),
        "heldout_roots": int(fit_score["heldout_roots_converged"])
        == int(gates["heldout_roots_converged"]),
        "training_RMS": float(fit_score["training_RMS_arcsec"])
        <= float(gates["training_RMS_arcsec_max"]),
        "heldout_RMS": heldout_rms <= float(gates["heldout_RMS_arcsec_max"]),
        "compact_halo_comparison": halo_ratio
        <= float(gates["candidate_to_compact_halo_heldout_RMS_ratio_max"]),
        "no_missing_multiplicity": topology["missing_multiplicity_families"]
        <= int(gates["missing_multiplicity_families_max"]),
        "observable_surplus": topology["potentially_observable_surplus_families"]
        <= int(gates["potentially_observable_surplus_families_max"]),
        "acceptable_multiplicity": topology["exact_or_demagnified_only_families"]
        >= int(gates["exact_or_demagnified_only_families_min"]),
        "parity_diversity": topology["parity_diverse_families"]
        >= int(gates["parity_diverse_families_min"]),
        "critical_curves": topology["critical_curve_present_families"]
        >= int(gates["critical_curve_present_families_min"]),
        "nuisance_bounds": int(fit_score["nuisance_parameters_near_bound"])
        <= int(gates["nuisance_parameters_near_bound_max"]),
        "no_new_constants": int(equation["new_universal_constants"])
        == int(gates["new_universal_constants"]),
        "no_fitted_gravity": int(equation["gravity_parameters_fit_to_RXJ2129"])
        == int(gates["gravity_parameters_fit_to_RXJ2129"]),
        "no_fitted_photon_amplitude": int(equation["photon_amplitudes_fit_to_RXJ2129"])
        == int(gates["photon_amplitudes_fit_to_RXJ2129"]),
        "sealed_targets_untouched": not bool(gates["sealed_target_outcomes_opened"]),
    }
    all_pass = bool(all(gate_results.values()))
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    field_path = output / protocol["outputs"]["field"]
    np.savez_compressed(
        field_path,
        axis_kpc=axis_kpc,
        mixed_potential_m2_s2=solution.field.potential,
        mixed_source_s2=solution.mixed_source,
        alpha_x_physical_arcsec=deflection.alpha_x_arcsec,
        alpha_y_physical_arcsec=deflection.alpha_y_arcsec,
        baryonic_covariance_m2=solution.covariance,
        quadrupole_fraction=q_b,
    )
    pd.DataFrame([fit_score]).to_csv(output / "fit_scores.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(output / "nuisance_parameters.csv", index=False)
    prediction.to_csv(output / "exact_predictions.csv", index=False)
    roots.to_csv(output / "global_roots.csv", index=False)
    assignments.to_csv(output / "global_assignments.csv", index=False)
    families.to_csv(output / "family_topology.csv", index=False)
    report = {
        "report_version": "P0691-MULTIPOLE-GATED-SOURCE-ROUTING-RESULTS-1.0.0",
        "status": "pass" if all_pass else "fail",
        "all_progression_gates_pass": all_pass,
        "candidate_advanced_to_real_2D_galaxy_and_robustness": all_pass,
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__).resolve()),
        "operator_source_sha256": sha256(ROOT / "src/voidscreen/source_routing_qumond.py"),
        "map_sha256": sha256(map_path),
        "field_sha256": sha256(field_path),
        "quadrupole": {
            "fraction": q_b,
            "covariance_m2": solution.covariance.tolist(),
            "eigenvalues_m2": np.linalg.eigvalsh(solution.covariance).tolist(),
        },
        "field": {
            "normalized_residual_RMS": solution.field.normalized_residual_rms,
            "mixed_source_identity_relative_RMS": identity_relative,
            "boundary_maximum_relative_mismatch": boundary_mismatch,
            "strong_lens_median_physical_deflection_arcsec": field_median,
            "strong_lens_RMS_physical_deflection_arcsec": field_rms,
            "normalized_deflection_curl_RMS": field_curl,
        },
        "inherited_spherical_limit": protocol["spherical_limit"],
        "fit_score": fit_score,
        "comparisons": {
            "compact_halo_heldout_RMS_arcsec": compact_halo,
            "candidate_to_compact_halo_heldout_RMS_ratio": halo_ratio,
        },
        "topology": topology,
        "gate_results": gate_results,
        "sealed_P0633_kinematics_opened": False,
        "sealed_P0640_lensing_constraints_opened": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    (output / "report.json").write_text(
        json.dumps(json_safe(report), indent=2),
        encoding="utf-8",
    )

    figure, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    extent = [axis_kpc[0], axis_kpc[-1], axis_kpc[0], axis_kpc[-1]]
    image = axes[0].imshow(magnitude.T, origin="lower", extent=extent, cmap="viridis")
    axes[0].set(title=f"mixed deflection, q={q_b:.3f}", xlabel="x (kpc)", ylabel="y (kpc)")
    figure.colorbar(image, ax=axes[0], shrink=0.75, label="arcsec")
    axes[1].bar(
        ["training", "heldout"],
        [fit_score["training_roots_converged"], fit_score["heldout_roots_converged"]],
    )
    axes[1].scatter(
        ["training", "heldout"],
        [len(training), len(heldout)],
        color="black",
        label="required",
    )
    axes[1].set(title="Exact roots", ylabel="recovered")
    axes[1].legend()
    class_counts = families.multiplicity_classification.value_counts()
    axes[2].bar(class_counts.index, class_counts.values)
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].set(title="Multiplicity", ylabel="families")
    determinant = critical_maps[1]
    half = float(protocol["global_topology"]["critical_grid_half_width_arcsec"])
    sign_image = axes[3].imshow(
        np.sign(determinant).T,
        origin="lower",
        extent=[-half, half, -half, half],
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    axes[3].set(title="Family 1 Jacobian sign", xlabel="x (arcsec)", ylabel="y (arcsec)")
    figure.colorbar(sign_image, ax=axes[3], shrink=0.75)
    figure.tight_layout()
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)

    failed = [name for name, passed in gate_results.items() if not passed]
    summary = f"""# P0691 multipole-gated source-routing topology

- Status: **{"PASS" if all_pass else "FAIL"}**.
- Baryonic quadrupole fraction: **{q_b:.6g}**.
- Field residual / mixed-source identity: **{solution.field.normalized_residual_rms:.3g} / {identity_relative:.3g}**.
- Strong-lens median/RMS physical deflection: **{field_median:.4g} / {field_rms:.4g} arcsec**.
- Training / heldout exact roots: **{fit_score["training_roots_converged"]}/15 / {fit_score["heldout_roots_converged"]}/7**.
- Missing / acceptable / parity-diverse / critical families: **{topology["missing_multiplicity_families"]} / {topology["exact_or_demagnified_only_families"]} / {topology["parity_diverse_families"]} / {topology["critical_curve_present_families"]}**.
- Failed frozen gates: **{", ".join(failed) if failed else "none"}**.
- Sealed P0633/P0640 targets opened: **no**.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
