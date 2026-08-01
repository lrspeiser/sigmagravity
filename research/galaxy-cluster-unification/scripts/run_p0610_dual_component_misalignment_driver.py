#!/usr/bin/env python3
"""Quantify the posthoc dual-component misalignment clue from P0609."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]


def safe_number(value):
    value = float(value)
    return value if np.isfinite(value) else None


def main() -> None:
    config_path = ROOT / "configs/p0610_dual_component_misalignment_driver_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    directions = pd.read_csv(ROOT / protocol["inputs"]["direction_audits"])
    directions = directions[directions.direction_kind.eq("gas_masked")][
        ["system_label", "mean_alignment_with_member", "mean_alignment_with_star"]
    ].copy()
    directions["dual_misalignment"] = np.sqrt(
        np.maximum(0.0, 1.0 - directions.mean_alignment_with_member)
        * np.maximum(0.0, 1.0 - directions.mean_alignment_with_star)
    )
    amplitude = directions.dual_misalignment.to_numpy(float)
    directions["candidate_gate_H"] = amplitude**4 / (amplitude**4 + 0.3**4)
    directions["candidate_effective_strength"] = 0.0025 * directions.candidate_gate_H

    scores = pd.read_csv(ROOT / protocol["inputs"]["multicluster_scores"])
    pivot = scores.pivot(index="system_label", columns="variant_id", values="heldout_RMS_arcsec")
    pivot["raw_improvement_fraction"] = 1.0 - pivot.gas_route_gamma1 / pivot.P0599_no_route
    response = pivot[["raw_improvement_fraction"]].reset_index()
    rx = pd.read_csv(ROOT / protocol["inputs"]["RXJ2129_refits"]).set_index("role")
    rx_improvement = 1.0 - float(rx.loc["positive_route", "heldout_RMS_arcsec"]) / float(
        rx.loc["P0599_no_route_16_start_reference", "heldout_RMS_arcsec"]
    )
    response = pd.concat(
        [response, pd.DataFrame([{"system_label": "RXJ2129", "raw_improvement_fraction": rx_improvement}])],
        ignore_index=True,
    )
    table = directions.merge(response, on="system_label", how="left")
    finite = table[np.isfinite(table.raw_improvement_fraction)].copy()
    pearson = pearsonr(finite.dual_misalignment, finite.raw_improvement_fraction)
    spearman = spearmanr(finite.dual_misalignment, finite.raw_improvement_fraction)
    jackknife_rows = []
    for omitted in finite.system_label:
        block = finite[finite.system_label.ne(omitted)]
        jackknife_rows.append(
            {
                "omitted_system": omitted,
                "systems": len(block),
                "Pearson_r": float(pearsonr(block.dual_misalignment, block.raw_improvement_fraction).statistic),
                "Spearman_rho": float(spearmanr(block.dual_misalignment, block.raw_improvement_fraction).statistic),
            }
        )
    jackknife = pd.DataFrame(jackknife_rows)
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / protocol["outputs"]["driver_table"], index=False)
    jackknife.to_csv(output / protocol["outputs"]["jackknife"], index=False)
    report = {
        "report_version": "P0610-DUAL-COMPONENT-MISALIGNMENT-DRIVER-RESULTS-0.1.0",
        "status": "complete_posthoc_candidate_generator_only",
        "coverage": {
            "systems_with_direction_maps": len(table),
            "systems_with_finite_raw_response": len(finite),
        },
        "correlations": {
            "Pearson_r": float(pearson.statistic),
            "Pearson_p": float(pearson.pvalue),
            "Spearman_rho": float(spearman.statistic),
            "Spearman_p": float(spearman.pvalue),
            "minimum_leave_one_out_Pearson_r": float(jackknife.Pearson_r.min()),
            "minimum_leave_one_out_Spearman_rho": float(jackknife.Spearman_rho.min()),
        },
        "candidate_gate": {
            "formula": protocol["driver"]["candidate_future_gate"],
            "base_strength": 0.0025,
            "largest_activation_system": str(table.sort_values("candidate_gate_H").iloc[-1].system_label),
            "largest_activation": float(table.candidate_gate_H.max()),
            "next_largest_activation": float(table.candidate_gate_H.nlargest(2).iloc[-1]),
        },
        "driver_table": [
            {key: safe_number(value) if key != "system_label" else str(value) for key, value in row.items()}
            for row in table.to_dict("records")
        ],
        "interpretation": {
            "same_data_evidence": False,
            "candidate_for_fresh_predeclared_gate": True,
            "single_outlier_dominates": True,
        },
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    axes[0].scatter(finite.dual_misalignment, finite.raw_improvement_fraction, s=65)
    for row in finite.itertuples():
        axes[0].annotate(row.system_label, (row.dual_misalignment, row.raw_improvement_fraction), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[0].axhline(0.0, color="black", lw=0.8)
    axes[0].set(xlabel="dual gas/member/star misalignment", ylabel="raw held-out improvement", title="Posthoc driver (N=4)")
    display = table.sort_values("dual_misalignment")
    axes[1].barh(display.system_label, display.candidate_gate_H, color="#1261A0")
    axes[1].set(xlabel="candidate future gate H", title="A0=0.3, n=4 (not validated)")
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)
    (output / protocol["outputs"]["summary"]).write_text(
        "# P0610 dual-component misalignment driver\n\n"
        f"Across four finite raw responses, dual misalignment has Pearson r={pearson.statistic:.3f} (p={pearson.pvalue:.3f}) and Spearman rho={spearman.statistic:.3f} (p={spearman.pvalue:.3f}). "
        f"MACS0429 has gate H={table.loc[table.system_label.eq('MACS0429'), 'candidate_gate_H'].iloc[0]:.3f}; every other system is below {table.loc[table.system_label.ne('MACS0429'), 'candidate_gate_H'].max():.3f}.\n\n"
        "This is a posthoc candidate generator dominated by MACS0429, not evidence.\n",
        encoding="utf-8",
    )
    print(json.dumps({"correlations": report["correlations"], "candidate_gate": report["candidate_gate"], "interpretation": report["interpretation"]}, indent=2))


if __name__ == "__main__":
    main()
