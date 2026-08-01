#!/usr/bin/env python3
"""Run the frozen full (radial plus angular) member-tidal metric test."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_member_tidal_metric import (
    build_contexts,
    json_safe,
    run_randomizations,
    run_selection_grid,
    run_validation,
)


def make_figure(report, grid, randomizations, output):
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    aggregate = grid[grid.row_type.eq("aggregate")].sort_values("tensor_t")
    good = aggregate[aggregate.all_training_roots.astype(bool)]
    bad = aggregate[~aggregate.all_training_roots.astype(bool)]
    axes[0].plot(good.tensor_t, good.training_exact_RMS_arcsec, "o-")
    axes[0].scatter(bad.tensor_t, bad.training_exact_RMS_arcsec, color="#bb3333", marker="x")
    axes[0].axvline(report["selection"]["selected_t"], color="black", linestyle="--")
    axes[0].set(xlabel="universal full-tensor coupling t", ylabel="selection training RMS (arcsec)", title="Frozen full-member tensor scan")

    axes[1].bar(
        ["scalar slip", "full tensor", "compact halo"],
        [
            report["validation"]["zero_tensor"]["equal_system_radial_RMS_arcsec"],
            report["validation"]["selected_tensor"]["equal_system_radial_RMS_arcsec"],
            report["comparators"]["compact_halo_validation_RMS_arcsec"],
        ],
        color=["#888888", "#2f78b7", "#d18f31"],
    )
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].set(ylabel="held-out validation RMS (arcsec)", title="Transfer to different clusters")

    if report["randomization_control"]["degenerate_because_selected_t_zero"]:
        axes[2].axis("off")
        axes[2].text(0.5, 0.5, "Degenerate randomization control\nselected t = 0", ha="center", va="center", fontsize=13)
    else:
        axes[2].hist(randomizations.aggregate_local_RMS_arcsec, bins=10, color="#bbbbbb")
        axes[2].axvline(report["randomization_control"]["actual_local_RMS_arcsec"], color="#2f78b7", linewidth=2)
        axes[2].set(xlabel="fixed-source local RMS (arcsec)", ylabel="randomized maps", title="Observed layout specificity")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main():
    protocol_path = ROOT / "configs/member_full_tidal_metric_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    output = ROOT / "results/member_full_tidal_metric"
    output.mkdir(parents=True, exist_ok=True)
    contexts, audits, hashes = build_contexts(
        protocol,
        softening_kpc=float(protocol["environment_tensor"]["primary_softening_kpc"]),
    )
    selected_t, grid, _, predictions, geometry = run_selection_grid(protocol, contexts)
    validation, validation_predictions, validation_geometry = run_validation(
        protocol, contexts, selected_t
    )
    predictions.extend(validation_predictions)
    geometry.extend(validation_geometry)
    randomizations, actual_by_system, actual_rms, p_value = run_randomizations(
        protocol, contexts, validation, selected_t
    )
    metric_report = json.loads(
        (ROOT / protocol["inputs"]["metric_slip_report"]).read_text(encoding="utf-8")
    )
    selected = validation[("aggregate", selected_t)]
    zero = validation[("aggregate", 0.0)]
    halo = float(
        metric_report["comparators"]["compact_halo_validation"]["equal_system_radial_RMS_arcsec"]
    )
    improvement = 1.0 - float(selected["equal_system_radial_RMS_arcsec"]) / float(
        zero["equal_system_radial_RMS_arcsec"]
    )
    halo_ratio = float(selected["equal_system_radial_RMS_arcsec"]) / halo
    primary_grid = list(map(float, protocol["tensor_coupling"]["primary_grid"]))
    gates = protocol["advance_gates"]
    maximum_solver_edge = max(item["maximum_solver_edge_Q_eigenvalue"] for item in audits)
    maximum_curl = max(item["normalized_curl_RMS"] for item in audits)
    gate_audit = {
        "selected_t_not_primary_grid_boundary_pass": selected_t
        not in {min(primary_grid), max(primary_grid)},
        "validation_all_roots_converged_pass": bool(selected["all_roots_converged"]),
        "validation_RMS_improvement_over_scalar_slip_fraction": improvement,
        "validation_RMS_improvement_pass": improvement
        >= float(gates["validation_RMS_improvement_over_scalar_slip_fraction_min"]),
        "validation_to_compact_halo_RMS_ratio": halo_ratio,
        "validation_to_compact_halo_pass": halo_ratio
        <= float(gates["validation_to_compact_halo_RMS_ratio_max"]),
        "member_randomization_p": p_value,
        "member_randomization_pass": selected_t != 0.0
        and p_value <= float(gates["actual_member_map_randomization_p_max"]),
        "maximum_solver_edge_Q_eigenvalue": maximum_solver_edge,
        "solver_edge_pass": maximum_solver_edge
        <= float(gates["maximum_solver_edge_Q_eigenvalue"]),
        "maximum_normalized_curl_RMS": maximum_curl,
        "curl_pass": maximum_curl <= float(gates["maximum_normalized_curl_RMS"]),
    }
    gate_audit["all_gates_pass"] = bool(
        all(value for key, value in gate_audit.items() if key.endswith("_pass"))
    )
    report = {
        "report_version": "MEMBER-FULL-TIDAL-METRIC-RESULTS-0.1.0",
        "status": "complete",
        "protocol": protocol["protocol_version"],
        "input_hashes": hashes,
        "equation": protocol["weak_field_equation"],
        "selection": {
            "selected_t": selected_t,
            "selection_training_equal_system_RMS_arcsec": float(
                grid[
                    grid.row_type.eq("aggregate")
                    & np.isclose(grid.tensor_t.astype(float), selected_t)
                ].iloc[0].training_exact_RMS_arcsec
            ),
        },
        "validation": {
            "selected_tensor": selected,
            "zero_tensor": zero,
            "per_system": {
                label: validation[(label, selected_t)]["heldout"]
                for label in protocol["cluster_split"]["validation_labels"]
            },
        },
        "comparators": {
            "compact_halo_validation_RMS_arcsec": halo,
            "fixed_RAR_galaxy_outer_RMSE_km_s": metric_report["matter_law"][
                "locked_galaxy_outer_RMSE_km_s"
            ],
        },
        "randomization_control": {
            "actual_local_RMS_arcsec": actual_rms,
            "actual_per_system_local_RMS_arcsec": actual_by_system,
            "one_sided_p_value": p_value,
            "degenerate_because_selected_t_zero": selected_t == 0.0,
        },
        "map_audit": {
            "maximum_solver_edge_Q_eigenvalue": maximum_solver_edge,
            "maximum_normalized_curl_RMS": maximum_curl,
        },
        "gate_audit": gate_audit,
        "verdict": {
            "full_member_tidal_metric_survives": gate_audit["all_gates_pass"],
            "gas_inclusive_tensor_test_completed": False,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    grid.to_csv(output / "grid_scores.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(geometry).to_csv(output / "geometry.csv", index=False)
    pd.DataFrame(audits).to_csv(output / "map_audit.csv", index=False)
    randomizations.to_csv(output / "randomizations.csv", index=False)
    (output / "report.json").write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    make_figure(report, grid, randomizations, output / "member_full_tidal_metric.png")
    (output / "SUMMARY.md").write_text(
        f"""# Full member-tidal metric result

The frozen grid selected `t={selected_t:g}`. Held-out cross-cluster RMS is
{selected['equal_system_radial_RMS_arcsec']:.3f} arcsec, versus
{zero['equal_system_radial_RMS_arcsec']:.3f} for scalar slip and {halo:.3f} for
the compact-halo comparator. The fractional improvement is {100*improvement:.2f}%.
The frozen gate result is {'PASS' if gate_audit['all_gates_pass'] else 'FAIL'}.
""",
        encoding="utf-8",
    )
    print(json.dumps(json_safe(report["gate_audit"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
