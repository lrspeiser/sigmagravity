#!/usr/bin/env python3
"""Run the frozen P0689 source-conserving routing audit without lens scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0660_exact_tensor_activation_audit import sha256

from voidscreen.field_solvers import boundary_mask
from voidscreen.metric_lensing_3d import KPC_M
from voidscreen.source_routing_qumond import solve_source_conserving_baryonic_routing

G_SI = 6.67430e-11
DEFAULT_CONFIG = ROOT / "configs" / "p0689_source_conserving_baryonic_routing_audit.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    if protocol.get("status") != "frozen_before_any_P0689_source_or_field_metric":
        raise RuntimeError("P0689 protocol is not frozen")
    failure_path = ROOT / protocol["failure_parent"]
    failure = read_json(failure_path)
    map_parent_path = ROOT / protocol["map_parent"]
    map_parent = read_json(map_parent_path)
    gates = protocol["predeclared_progression_gates"]
    if failure.get("status") != gates["P0688_status"]:
        raise RuntimeError("P0688 status changed")
    if bool(failure.get("candidate_advanced_to_3D_potential_shell_freeze")) != bool(
        gates["P0688_candidate_advanced_to_3D_potential_shell_freeze"]
    ):
        raise RuntimeError("P0688 advancement state changed")
    if not map_parent["all_progression_gates_pass"]:
        raise RuntimeError("P0670 physical map no longer passes")

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
        raise RuntimeError("P0689 a0 no longer matches the frozen physical map")
    spacing_m = float(axis_kpc[1] - axis_kpc[0]) * KPC_M
    cell_volume = spacing_m**3

    print("solving source-conserving baryonic routing audit", flush=True)
    solution = solve_source_conserving_baryonic_routing(
        density,
        spacing_m,
        gravitational_constant=G_SI,
        a0=a0,
        transition_depth=float(equation["chi_t"]),
        transition_power=float(equation["transition_power_n"]),
        extra_spatial_channels=float(equation["extra_spatial_channels"]),
        path_power=float(equation["path_power_q"]),
    )
    edge = boundary_mask(density.shape)
    interior = ~edge
    positive_strength = float(np.sum(solution.positive_routed_source) * cell_volume)
    negative_strength = float(np.sum(solution.negative_shell_source) * cell_volume)
    net_added = float(
        np.sum(solution.positive_routed_source - solution.negative_shell_source) * cell_volume
    )
    reconstruction_cancellation = float(
        np.sum(solution.routed_source - solution.base_source) * cell_volume
    )
    strength_scale = max(solution.positive_generator_strength, np.finfo(float).tiny)
    strength_mismatch = abs(positive_strength - negative_strength) / strength_scale
    net_relative = abs(net_added) / strength_scale
    shell_sum = float(np.sum(solution.transition_shell_weight))
    shell_interior_fraction = float(np.sum(solution.transition_shell_weight[interior])) / max(
        shell_sum,
        np.finfo(float).tiny,
    )
    shell_threshold = float(np.max(solution.transition_shell_weight)) * 1e-12
    shell_positive_cells = int(np.sum(solution.transition_shell_weight > shell_threshold))
    positive_total = float(np.sum(solution.positive_routed_source))
    baryon_support_fraction = float(
        np.sum(solution.positive_routed_source[density > 0.0])
        / max(positive_total, np.finfo(float).tiny)
    )
    boundary_scale = max(
        float(np.max(np.abs(solution.boundary_potential[edge]))),
        np.finfo(float).tiny,
    )
    boundary_mismatch = float(
        np.max(np.abs(solution.field.potential[edge] - solution.boundary_potential[edge]))
        / boundary_scale
    )
    finite = bool(
        np.all(np.isfinite(solution.base_source))
        and np.all(np.isfinite(solution.local_generator_source))
        and np.all(np.isfinite(solution.local_extra_source))
        and np.all(np.isfinite(solution.positive_routed_source))
        and np.all(np.isfinite(solution.negative_shell_source))
        and np.all(np.isfinite(solution.routed_source))
        and np.all(np.isfinite(solution.transition_shell_weight))
        and np.all(np.isfinite(solution.field.potential))
        and all(np.all(np.isfinite(item)) for item in solution.field.acceleration)
    )
    gate_results = {
        "P0688_parent": failure.get("status") == gates["P0688_status"],
        "P0670_parent": bool(map_parent["all_progression_gates_pass"])
        is bool(gates["P0670_all_progression_gates_pass"]),
        "newtonian_residual": solution.newtonian.normalized_residual_rms
        <= float(gates["newtonian_normalized_residual_RMS_max"]),
        "routed_residual": solution.field.normalized_residual_rms
        <= float(gates["routed_field_normalized_residual_RMS_max"]),
        "routed_convergence": solution.field.converged is bool(gates["routed_field_converged"]),
        "boundary": boundary_mismatch <= float(gates["boundary_maximum_relative_mismatch_max"]),
        "finite": finite is bool(gates["all_sources_weights_potentials_and_accelerations_finite"]),
        "positive_generator": bool(solution.positive_generator_strength > 0.0)
        is bool(gates["positive_generator_strength_strictly_positive"]),
        "positive_shell": bool(shell_sum > 0.0)
        is bool(gates["transition_shell_integral_strictly_positive"]),
        "shell_support": shell_positive_cells >= int(gates["transition_shell_positive_cells_min"]),
        "shell_interior": shell_interior_fraction
        >= float(gates["transition_shell_interior_weight_fraction_min"]),
        "baryonic_positive_route": baryon_support_fraction
        >= float(gates["positive_route_on_positive_baryon_cells_fraction_min"]),
        "balanced_routed_strength": strength_mismatch
        <= float(gates["positive_negative_routed_strength_relative_mismatch_max"]),
        "net_source_conservation": net_relative
        <= float(gates["net_added_source_relative_to_positive_strength_max"]),
        "no_new_constants": int(equation["new_universal_constants"])
        == int(gates["new_universal_constants"]),
        "no_fitted_gravity": int(equation["gravity_parameters_fit_to_RXJ2129"])
        == int(gates["gravity_parameters_fit_to_RXJ2129"]),
        "no_fitted_photon_amplitude": int(equation["photon_amplitudes_fit_to_RXJ2129"])
        == int(gates["photon_amplitudes_fit_to_RXJ2129"]),
        "no_photon_deflection": not bool(gates["photon_deflection_computed"]),
        "no_observation_score": not bool(gates["radial_or_raw_lens_score_computed"]),
        "sealed_targets_untouched": not bool(gates["sealed_target_outcomes_opened"]),
    }
    all_pass = bool(all(gate_results.values()))
    metrics = {
        "newtonian_normalized_residual_RMS": solution.newtonian.normalized_residual_rms,
        "routed_field_normalized_residual_RMS": solution.field.normalized_residual_rms,
        "boundary_maximum_relative_mismatch": boundary_mismatch,
        "positive_generator_strength_m3_s2": solution.positive_generator_strength,
        "positive_routed_strength_m3_s2": positive_strength,
        "negative_shell_strength_m3_s2": negative_strength,
        "positive_negative_relative_mismatch": strength_mismatch,
        "net_added_source_m3_s2": net_added,
        "net_added_source_relative_to_positive_strength": net_relative,
        "large_source_subtraction_cancellation_m3_s2": reconstruction_cancellation,
        "transition_shell_positive_cells": shell_positive_cells,
        "transition_shell_interior_weight_fraction": shell_interior_fraction,
        "positive_route_on_positive_baryon_cells_fraction": baryon_support_fraction,
        "local_extra_source_positive_fraction": float(np.mean(solution.local_extra_source > 0.0)),
        "local_extra_source_negative_fraction": float(np.mean(solution.local_extra_source < 0.0)),
    }

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    field_path = output / protocol["outputs"]["field"]
    np.savez_compressed(
        field_path,
        axis_kpc=axis_kpc,
        density_kg_m3=density,
        newtonian_potential_m2_s2=solution.newtonian.potential,
        routed_potential_m2_s2=solution.field.potential,
        base_source_s2=solution.base_source,
        local_extra_source_s2=solution.local_extra_source,
        positive_routed_source_s2=solution.positive_routed_source,
        negative_shell_source_s2=solution.negative_shell_source,
        routed_source_s2=solution.routed_source,
        transition_shell_weight_m_inv=solution.transition_shell_weight,
        potential_depth=solution.potential_depth,
        local_channel_exponent=solution.local_channel_exponent,
    )
    report = {
        "report_version": "P0689-SOURCE-CONSERVING-BARYONIC-ROUTING-AUDIT-RESULTS-1.0.0",
        "status": "pass" if all_pass else "fail",
        "all_progression_gates_pass": all_pass,
        "operator_advanced_to_frozen_empirical_screen": all_pass,
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__).resolve()),
        "operator_source_sha256": sha256(ROOT / "src/voidscreen/source_routing_qumond.py"),
        "map_sha256": sha256(map_path),
        "field_sha256": sha256(field_path),
        "metrics": metrics,
        "gate_results": gate_results,
        "photon_deflection_computed": False,
        "radial_or_raw_lens_score_computed": False,
        "sealed_P0633_kinematics_opened": False,
        "sealed_P0640_lensing_constraints_opened": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    mid = density.shape[2] // 2
    extent = [axis_kpc[0], axis_kpc[-1], axis_kpc[0], axis_kpc[-1]]
    figure, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    panels = (
        (np.log10(np.maximum(density[:, :, mid], 1e-40)), "log10 baryon density"),
        (solution.local_extra_source[:, :, mid], "local extra source"),
        (solution.positive_routed_source[:, :, mid], "positive baryon route"),
        (-solution.negative_shell_source[:, :, mid], "negative transition shell"),
    )
    for axis, (values, title) in zip(axes, panels, strict=True):
        image = axis.imshow(values.T, origin="lower", extent=extent, cmap="coolwarm")
        axis.set(title=title, xlabel="x (kpc)", ylabel="y (kpc)")
        figure.colorbar(image, ax=axis, shrink=0.75)
    figure.tight_layout()
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)

    failed = [name for name, passed in gate_results.items() if not passed]
    summary = f"""# P0689 source-conserving baryonic routing audit

- Status: **{"PASS" if all_pass else "FAIL"}**.
- Newtonian/routed residual: **{solution.newtonian.normalized_residual_rms:.4g} / {solution.field.normalized_residual_rms:.4g}**.
- Positive/negative routed strength mismatch: **{strength_mismatch:.3g}**.
- Net added source / positive strength: **{net_relative:.3g}**.
- Transition-shell positive cells / interior weight: **{shell_positive_cells} / {shell_interior_fraction:.3f}**.
- Failed frozen gates: **{", ".join(failed) if failed else "none"}**.
- Photon, radial, or raw-lens score computed: **no**; sealed targets opened: **no**.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
