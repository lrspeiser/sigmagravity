#!/usr/bin/env python3
"""Run the frozen, non-promotable P0694 DDO154 routing continuum."""

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
from run_p0635_ddo154_map_commissioning import radial_circular_speed, score_curve
from run_p0635_map_geometry_sensitivity import build_density
from run_p0660_exact_tensor_activation_audit import sha256

from voidscreen.data import load_curves
from voidscreen.field_solvers import boundary_mask
from voidscreen.source_routing_qumond import (
    solve_linear_routing_mixture,
    solve_source_conserving_baryonic_routing,
)

DEFAULT_CONFIG = ROOT / "configs" / "p0694_spent_ddo154_routing_continuum.json"
SPARC = ROOT / "data" / "raw" / "sparc"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    if protocol.get("status") != (
        "frozen_before_any_P0694_intermediate_fraction_field_or_rotation_score"
    ):
        raise RuntimeError("P0694 protocol is not frozen")
    failure_path = ROOT / protocol["failure_parent"]
    galaxy_parent_path = ROOT / protocol["galaxy_parent"]
    failure = read_json(failure_path)
    galaxy_parent = read_json(galaxy_parent_path)
    expected = protocol["predeclared_integrity_gates"]
    fractions = np.asarray(protocol["routing_fractions"], dtype=float)
    marker = float(protocol["registered_marker"]["fraction"])
    integrity = {
        "P0693_status": failure.get("status") == expected["P0693_status"],
        "P0693_not_advanced": bool(failure.get("candidate_advanced_to_robustness"))
        is bool(expected["P0693_candidate_advanced_to_robustness"]),
        "P0635_status": galaxy_parent.get("status") == expected["P0635_status"],
        "P0635_no_velocity_product": bool(
            galaxy_parent["data_boundary"]["little_things_velocity_products_downloaded"]
        )
        is bool(expected["P0635_velocity_products_downloaded"]),
        "fraction_range": bool(
            np.all(fractions >= float(expected["routing_fraction_min"]))
            and np.all(fractions <= float(expected["routing_fraction_max"]))
        ),
        "fractions_unique": len(np.unique(fractions)) == len(fractions),
        "fractions_strictly_increasing": bool(np.all(np.diff(fractions) > 0.0)),
        "registered_marker_present": bool(
            np.any(np.isclose(fractions, marker, rtol=0.0, atol=1e-15))
        ),
        "no_selected_gravity_parameter": int(
            expected["gravity_parameters_selected_from_atlas"]
        )
        == 0,
        "sealed_targets_untouched": not bool(expected["sealed_target_outcomes_opened"]),
    }
    if not all(integrity.values()):
        raise RuntimeError(f"P0694 integrity failure before scores: {integrity}")

    map_path = ROOT / protocol["galaxy_map_input"]
    with np.load(map_path) as data:
        axis = data["axis_kpc"].astype(float)
        gas = data["gas_surface_density_solar_kpc2"].astype(float)
        stars = data["stellar_surface_density_solar_kpc2"].astype(float)
    spacing = float(axis[1] - axis[0])
    density = build_density(
        gas,
        stars,
        axis,
        float(protocol["galaxy"]["gas_scale_height_kpc"]),
        float(protocol["galaxy"]["stellar_scale_height_kpc"]),
    )
    equation = protocol["equation"]
    print("P0694: solving shared DDO154 source endpoints", flush=True)
    routing = solve_source_conserving_baryonic_routing(
        density,
        spacing,
        gravitational_constant=float(
            equation["gravitational_constant_kpc_km2_s2_per_solar_mass"]
        ),
        a0=float(equation["a0_km2_s2_per_kpc"]),
        transition_depth=float(equation["chi_t"]),
        transition_power=float(equation["transition_power_n"]),
        extra_spatial_channels=float(equation["extra_spatial_channels"]),
        path_power=float(equation["path_power_q"]),
        light_speed=float(equation["light_speed_km_s"]),
    )
    edge = boundary_mask(density.shape)
    boundary_scale = max(
        float(np.max(np.abs(routing.boundary_potential[edge]))),
        np.finfo(float).tiny,
    )
    sparc_curve = next(curve for curve in load_curves(SPARC) if curve.metadata.name == "DDO154")
    algebraic = galaxy_parent["spent_DDO154_rotation_scores"]["algebraic_simple_mond"]
    qumond = galaxy_parent["spent_DDO154_rotation_scores"]["QUMOND_3d_map"]
    gates = protocol["diagnostic_viability_gates"]
    rows = []
    curve_frames = []
    for fraction in fractions:
        print(f"P0694 f={fraction:.12g}", flush=True)
        solution = solve_linear_routing_mixture(routing, spacing, float(fraction))
        expected_source = (
            (1.0 - fraction) * routing.local_generator_source
            + fraction * routing.routed_source
        )
        identity_scale = max(
            float(np.sqrt(np.mean(np.square(expected_source)))),
            np.finfo(float).tiny,
        )
        identity = float(
            np.sqrt(np.mean(np.square(solution.mixed_source - expected_source)))
            / identity_scale
        )
        boundary = float(
            np.max(np.abs(solution.field.potential[edge] - routing.boundary_potential[edge]))
            / boundary_scale
        )
        finite = bool(
            np.all(np.isfinite(solution.mixed_source))
            and np.all(np.isfinite(solution.field.potential))
            and all(np.all(np.isfinite(item)) for item in solution.field.acceleration)
        )
        curve = radial_circular_speed(solution.field, axis)
        curve.insert(0, "routing_fraction", float(fraction))
        curve_frames.append(curve)
        score = score_curve(
            curve.radius_kpc.to_numpy(),
            curve.circular_speed_km_s.to_numpy(),
            sparc_curve.radius_kpc,
            sparc_curve.velocity_observed_kms,
            sparc_curve.velocity_error_kms,
        )
        comparisons = {
            "RMSE_to_algebraic_MOND_ratio": score["RMSE_km_s"]
            / float(algebraic["RMSE_km_s"]),
            "weighted_RMSE_to_algebraic_MOND_ratio": score["weighted_RMSE_km_s"]
            / float(algebraic["weighted_RMSE_km_s"]),
            "RMSE_to_3D_QUMOND_ratio": score["RMSE_km_s"]
            / float(qumond["RMSE_km_s"]),
        }
        row_gates = {
            "mixed_source_identity": identity
            <= float(gates["mixed_source_linear_identity_relative_RMS_max"]),
            "field_residual": solution.field.normalized_residual_rms
            <= float(gates["field_normalized_residual_RMS_max"]),
            "boundary": boundary <= float(gates["boundary_maximum_relative_mismatch_max"]),
            "finite": finite is bool(gates["all_sources_potentials_accelerations_finite"]),
            "rotation_points": int(score["points"]) == int(gates["rotation_points"]),
            "RMSE_vs_MOND": comparisons["RMSE_to_algebraic_MOND_ratio"]
            <= float(gates["candidate_RMSE_to_algebraic_MOND_ratio_max"]),
            "weighted_RMSE_vs_MOND": comparisons[
                "weighted_RMSE_to_algebraic_MOND_ratio"
            ]
            <= float(gates["candidate_weighted_RMSE_to_algebraic_MOND_ratio_max"]),
            "RMSE_vs_3D_QUMOND": comparisons["RMSE_to_3D_QUMOND_ratio"]
            <= float(gates["candidate_RMSE_to_3D_QUMOND_ratio_max"]),
            "mean_bias": abs(float(score["mean_bias_km_s"]))
            <= float(gates["absolute_mean_bias_km_s_max"]),
        }
        rows.append(
            {
                "routing_fraction": float(fraction),
                "registered_P0693_marker": bool(
                    np.isclose(fraction, marker, rtol=0.0, atol=1e-15)
                ),
                "mixed_source_identity_relative_RMS": identity,
                "field_normalized_residual_RMS": solution.field.normalized_residual_rms,
                "boundary_maximum_relative_mismatch": boundary,
                **score,
                **comparisons,
                "passes_all_diagnostic_viability_gates": bool(all(row_gates.values())),
                "failed_diagnostic_gates": ",".join(
                    name for name, passed in row_gates.items() if not passed
                ),
            }
        )
    atlas = pd.DataFrame(rows)
    curves = pd.concat(curve_frames, ignore_index=True)
    viable = atlas.loc[atlas.passes_all_diagnostic_viability_gates.astype(bool)]
    retired = len(viable) == 0
    best_rmse_index = int(atlas.RMSE_km_s.astype(float).idxmin())
    best_weighted_index = int(atlas.weighted_RMSE_km_s.astype(float).idxmin())
    marker_row = atlas.loc[atlas.registered_P0693_marker.astype(bool)].iloc[0]
    reproduction = {
        "fraction": marker,
        "RMSE_absolute_difference_km_s": abs(
            float(marker_row.RMSE_km_s)
            - float(failure["spent_DDO154"]["candidate_score"]["RMSE_km_s"])
        ),
        "weighted_RMSE_absolute_difference_km_s": abs(
            float(marker_row.weighted_RMSE_km_s)
            - float(failure["spent_DDO154"]["candidate_score"]["weighted_RMSE_km_s"])
        ),
    }
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    atlas.to_csv(output / protocol["outputs"]["atlas_table"], index=False)
    curves.to_csv(output / protocol["outputs"]["curve_table"], index=False)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(atlas.routing_fraction, atlas.RMSE_to_algebraic_MOND_ratio, marker="o", label="ordinary RMSE / MOND")
    axes[0].plot(atlas.routing_fraction, atlas.weighted_RMSE_to_algebraic_MOND_ratio, marker="s", label="weighted RMSE / MOND")
    axes[0].axhline(float(gates["candidate_RMSE_to_algebraic_MOND_ratio_max"]), color="black", linestyle=":", label="frozen limit")
    axes[0].axvline(marker, color="C3", linestyle="--", label="P0693 marker")
    axes[0].set(title="Galaxy competitiveness", xlabel="routing fraction f", ylabel="error ratio")
    axes[0].legend(fontsize=8)
    axes[1].errorbar(
        sparc_curve.radius_kpc,
        sparc_curve.velocity_observed_kms,
        yerr=sparc_curve.velocity_error_kms,
        fmt="o",
        color="black",
        label="spent observations",
    )
    shown = {0.0, marker, 0.5, 1.0}
    for fraction, frame in curves.groupby("routing_fraction", sort=True):
        if any(np.isclose(float(fraction), value, rtol=0.0, atol=1e-15) for value in shown):
            axes[1].plot(frame.radius_kpc, frame.circular_speed_km_s, label=f"f={float(fraction):.3f}")
    axes[1].set(title="DDO154 field curves", xlabel="radius (kpc)", ylabel="circular speed (km/s)")
    axes[1].legend(fontsize=8)
    for axis_plot in axes:
        axis_plot.grid(alpha=0.2)
    figure.suptitle("P0694 spent DDO154 routing continuum (diagnostic only)")
    figure.tight_layout()
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)

    report = {
        "report_version": "P0694-SPENT-DDO154-ROUTING-CONTINUUM-RESULTS-1.0.0",
        "status": "complete",
        "evidence_class": protocol["evidence_class"],
        "diagnostic_only": True,
        "candidate_advanced": False,
        "shared_linear_endpoint_pair_retired": retired,
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__).resolve()),
        "operator_source_sha256": sha256(ROOT / "src/voidscreen/source_routing_qumond.py"),
        "map_sha256": sha256(map_path),
        "parent_sha256": {
            "P0693": sha256(failure_path),
            "P0635": sha256(galaxy_parent_path),
        },
        "integrity_gates": integrity,
        "routing_fraction_count": len(fractions),
        "viable_row_count": len(viable),
        "viable_routing_fractions": viable.routing_fraction.astype(float).tolist(),
        "best_RMSE_row": rows[best_rmse_index],
        "best_weighted_RMSE_row": rows[best_weighted_index],
        "algebraic_MOND_comparator": algebraic,
        "three_dimensional_QUMOND_comparator": qumond,
        "registered_marker_reproduction": reproduction,
        "atlas_rows": rows,
        "decision_rule": protocol["decision_rule"],
        "sealed_P0633_kinematics_opened": False,
        "sealed_P0640_lensing_constraints_opened": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2),
        encoding="utf-8",
    )
    best = rows[best_rmse_index]
    summary = f"""# P0694 spent DDO154 routing continuum

- Evidence class: **diagnostic only; no row may advance**.
- Frozen routing fractions: **{len(fractions)}**.
- Rows passing every viability gate: **{len(viable)}**.
- Best ordinary RMSE: **{best['RMSE_km_s']:.4g} km/s at f={best['routing_fraction']:.6g}**.
- Best ordinary RMSE / algebraic MOND: **{best['RMSE_to_algebraic_MOND_ratio']:.4g}**.
- Shared linear endpoint pair retired: **{'yes' if retired else 'no'}**.
- Candidate advanced: **no**.
- Sealed P0633/P0640 outcomes opened: **no**.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
