#!/usr/bin/env python3
"""Posthoc whole-cluster stability test under strict galaxy budgets."""

from __future__ import annotations

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

from run_p0603_tensor_routing import json_safe  # noqa: E402
from run_p0604_cross_domain_routing_balance import (  # noqa: E402
    candidate_specs,
    cluster_scores,
    cross_validate,
    galaxy_scores,
    summarize_oof,
)


def main() -> None:
    protocol = json.loads(
        (ROOT / "configs/p0604b_strict_galaxy_budget_protocol.json").read_text()
    )
    parent = json.loads((ROOT / protocol["parent_protocol"]).read_text())
    parent["validation"]["galaxy_RMSE_ratio_budgets"] = protocol[
        "galaxy_RMSE_ratio_budgets"
    ]
    parent["validation"]["primary_galaxy_RMSE_ratio_budget"] = protocol[
        "primary_budget"
    ]
    specs = candidate_specs(parent)
    clusters = cluster_scores(parent, specs)
    galaxies, reference = galaxy_scores(parent, specs)
    folds, oof = cross_validate(parent, clusters, galaxies, specs)
    summary = summarize_oof(oof)
    primary = next(
        row
        for row in summary
        if row["galaxy_RMSE_ratio_budget"] == protocol["primary_budget"]
        and row["target_kind"] == "lenstool_best"
    )
    primary_glafic = next(
        row
        for row in summary
        if row["galaxy_RMSE_ratio_budget"] == protocol["primary_budget"]
        and row["target_kind"] == "glafic_best"
    )
    primary_folds = folds[
        folds.galaxy_RMSE_ratio_budget.eq(float(protocol["primary_budget"]))
    ]
    report = {
        "report_version": "P0604B-STRICT-GALAXY-BUDGET-RESULTS-0.1.0",
        "status": "complete_posthoc_strict_budget_whole_cluster_CV",
        "coverage": {
            "parent_candidates": len(specs),
            "strict_budgets": len(protocol["galaxy_RMSE_ratio_budgets"]),
            "cluster_folds_per_budget": 5,
            "clusters": clusters.system.nunique(),
            "galaxies": 131,
        },
        "fixed_RAR_reference": reference,
        "oof_summary": summary,
        "primary_budget": protocol["primary_budget"],
        "primary_lenstool": primary,
        "primary_glafic": primary_glafic,
        "primary_fold_selections": primary_folds.to_dict("records"),
        "primary_unique_selected_candidates": int(
            primary_folds.selected_candidate_id.nunique()
        ),
        "strict_interpretation": {
            "budgets_predeclared_before_parent_frontier": False,
            "whole_cluster_holdouts_used": True,
            "every_selected_candidate_beats_or_equals_fixed_RAR_on_all_galaxy_metric": bool(
                np.all(primary_folds.selected_galaxy_RMSE_ratio <= 1.0)
            ),
            "fresh_confirmation": False,
        },
        "status_warning": protocol["status_warning"],
    }
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output / protocol["outputs"]["fold_selections"], index=False)
    oof.to_csv(output / protocol["outputs"]["oof_scores"], index=False)
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n"
    )

    lenstool = [row for row in summary if row["target_kind"] == "lenstool_best"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    axes[0].plot(
        [row["galaxy_RMSE_ratio_budget"] for row in lenstool],
        [row["tensor_equal_JS"] for row in lenstool],
        marker="o",
        label="selected route",
    )
    axes[0].axhline(primary["LOCAL75_equal_JS"], color="gray", linestyle="--", label="LOCAL75")
    axes[0].axhline(primary["CENTRAL100_equal_JS"], color="black", linestyle="--", label="CENTRAL100")
    axes[0].axhline(primary["W060_equal_JS"], color="green", linestyle="--", label="W060")
    axes[0].set(xlabel="maximum galaxy RMSE / RAR", ylabel="OOF Lenstool JS", title="Strict cross-domain budget")
    axes[0].legend(fontsize=7)
    counts = primary_folds.selected_candidate_id.value_counts()
    axes[1].barh(counts.index, counts.values, color="#1261A0")
    axes[1].set(xlabel="fold selections", title="Strict-budget stability")
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)
    (output / protocol["outputs"]["summary"]).write_text(
        "# P0604B strict galaxy-budget diagnostic\n\n"
        f"At no galaxy degradation, OOF Lenstool JS is **{primary['tensor_equal_JS']:.5f}**. "
        f"Improvements are **{100 * primary['improvement_vs_LOCAL75']:.2f}%** versus LOCAL75 and "
        f"**{100 * primary['improvement_vs_CENTRAL100']:.2f}%** versus CENTRAL100; "
        f"the comparison versus W060 is **{100 * primary['improvement_vs_W060']:.2f}%**.\n\n"
        "This budget was motivated posthoc and is not fresh confirmation.\n"
    )
    print(json.dumps(json_safe(report), indent=2))


if __name__ == "__main__":
    main()
