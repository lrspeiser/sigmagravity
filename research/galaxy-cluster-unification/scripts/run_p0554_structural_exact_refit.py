#!/usr/bin/env python3
"""Refit ordinary lens geometry for selected P0554 structural controls."""

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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0554_compensated_interactions import (  # noqa: E402
    build_variants,
    evaluate_raw_context,
    evaluate_scalar_domains,
    matched_comparison,
)
from run_p0554_local_cross_domain_sensitivity import (  # noqa: E402
    json_safe,
    raw_contexts,
    sha256,
)


def comparison_table(raw: pd.DataFrame, variants, scopes: dict) -> pd.DataFrame:
    rows = []
    for scope, labels in scopes.items():
        for variant in variants:
            row = matched_comparison(raw, variant.variant_id, labels)
            row["scope"] = scope
            rows.append(row)
    return pd.DataFrame(rows)


def mapped_fixed_raw(protocol: dict, variants) -> pd.DataFrame:
    source = pd.read_csv(ROOT / protocol["inputs"]["structural_fixed_raw"])
    pieces = []
    for variant in variants:
        old_id = protocol["fixed_variant_mapping"][variant.variant_id]
        block = source[source.variant_id.eq(old_id)].copy()
        if len(block) != 5:
            raise RuntimeError(f"fixed raw coverage changed for {old_id}")
        block["variant_id"] = variant.variant_id
        pieces.append(block)
    return pd.concat(pieces, ignore_index=True)


def signed_direction(value: float, tolerance: float = 1.0e-8) -> str:
    if not np.isfinite(value):
        return "unavailable"
    if abs(float(value)) <= tolerance:
        return "neutral"
    return "improves" if float(value) > 0.0 else "worsens"


def fixed_vs_refit_table(fixed: pd.DataFrame, exact: pd.DataFrame) -> pd.DataFrame:
    keys = ["variant_id", "scope"]
    keep = [
        "matched_complete_systems",
        "matched_labels",
        "matched_improvement_fraction",
        "recovered_systems",
        "lost_systems",
        "candidate_complete_systems",
        "candidate_total_roots",
        "candidate_finite_only_RMS_arcsec",
    ]
    joined = fixed[keys + keep].merge(
        exact[keys + keep], on=keys, suffixes=("_fixed", "_refit"), validate="one_to_one"
    )
    joined["fixed_direction"] = joined.matched_improvement_fraction_fixed.map(
        signed_direction
    )
    joined["refit_direction"] = joined.matched_improvement_fraction_refit.map(
        signed_direction
    )
    joined["direction_survives"] = (
        joined.fixed_direction.eq(joined.refit_direction)
        & joined.fixed_direction.isin(["improves", "worsens"])
    )
    return joined


def make_figure(scores, raw, joined, output: Path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    axes[0, 0].scatter(
        scores.galaxy_outer_RMSE_km_s,
        scores.cluster_RMSE_dex,
        c=np.where(scores.all_solar_proxies_pass, "tab:blue", "crimson"),
        s=70,
    )
    for row in scores.itertuples(index=False):
        axes[0, 0].annotate(
            row.variant_id,
            (row.galaxy_outer_RMSE_km_s, row.cluster_RMSE_dex),
            fontsize=7,
        )
    axes[0, 0].set(
        xlabel="SPARC outer RMSE (km/s)",
        ylabel="CLASH RMSE (dex)",
        title="Structural shortlist (red fails Solar)",
    )

    rx = raw[raw.raw_group.eq("RXJ2129")].set_index("variant_id").loc[
        scores.variant_id
    ]
    value = pd.to_numeric(rx.heldout_RMS_arcsec, errors="coerce").replace(np.inf, np.nan)
    axes[0, 1].bar(
        np.arange(len(rx)), value, color=np.where(rx.heldout_all_roots, "tab:blue", "crimson")
    )
    axes[0, 1].set(
        xticks=np.arange(len(rx)),
        xticklabels=rx.index,
        ylabel="held-out RMS (arcsec)",
        title="RX J2129 exact geometry refits",
    )
    axes[0, 1].tick_params(axis="x", rotation=90, labelsize=7)

    all_five = joined[joined.scope.eq("all_five") & ~joined.variant_id.eq("baseline")]
    axes[1, 0].scatter(
        100.0 * all_five.matched_improvement_fraction_fixed,
        100.0 * all_five.matched_improvement_fraction_refit,
        c=np.where(all_five.direction_survives, "tab:green", "tab:orange"),
        s=80,
    )
    for row in all_five.itertuples(index=False):
        axes[1, 0].annotate(
            row.variant_id,
            (
                100.0 * row.matched_improvement_fraction_fixed,
                100.0 * row.matched_improvement_fraction_refit,
            ),
            fontsize=7,
        )
    axes[1, 0].axhline(0.0, color="black", ls="--")
    axes[1, 0].axvline(0.0, color="black", ls="--")
    axes[1, 0].set(
        xlabel="fixed-geometry matched change (%)",
        ylabel="exact-refit matched change (%)",
        title="Does the field-law direction survive geometry freedom?",
    )

    root = raw.groupby("variant_id").heldout_roots_converged.sum().loc[scores.variant_id]
    axes[1, 1].bar(np.arange(len(root)), root)
    axes[1, 1].axhline(18, color="black", ls="--")
    axes[1, 1].set(
        xticks=np.arange(len(root)),
        xticklabels=root.index,
        ylabel="converged held-out roots of 18",
        title="Exact-refit topology",
    )
    axes[1, 1].tick_params(axis="x", rotation=90, labelsize=7)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    config_path = ROOT / "configs/p0554_structural_exact_refit_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    variants = build_variants(protocol)

    scores, morphology, galaxy, clusters = evaluate_scalar_domains(protocol, variants)
    scores.to_csv(output / protocol["outputs"]["variant_scores"], index=False)
    morphology.to_csv(output / protocol["outputs"]["galaxy_morphology"], index=False)

    raw_rows, predictions, geometry = [], [], []
    contexts = raw_contexts(protocol)
    for system_index, context in enumerate(contexts):
        rows, pred, geom = evaluate_raw_context(
            context, variants, protocol, system_index
        )
        raw_rows.extend(rows)
        predictions.extend(pred)
        geometry.extend(geom)
    raw = pd.DataFrame(raw_rows)
    raw.to_csv(output / protocol["outputs"]["raw_scores"], index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        output / protocol["outputs"]["raw_predictions"], index=False
    )
    geometry_frame = pd.DataFrame(geometry)
    geometry_frame.to_csv(output / protocol["outputs"]["geometry"], index=False)

    scopes = protocol["comparison_scopes"]
    comparisons = comparison_table(raw, variants, scopes)
    comparisons.to_csv(output / protocol["outputs"]["matched_comparisons"], index=False)
    fixed_raw = mapped_fixed_raw(protocol, variants)
    fixed_comparisons = comparison_table(fixed_raw, variants, scopes)
    joined = fixed_vs_refit_table(fixed_comparisons, comparisons)
    joined.to_csv(output / protocol["outputs"]["fixed_vs_refit"], index=False)

    all_five = comparisons[comparisons.scope.eq("all_five")]
    combined = scores.merge(
        all_five[
            [
                "variant_id",
                "candidate_complete_systems",
                "candidate_total_roots",
                "candidate_finite_only_RMS_arcsec",
                "matched_improvement_fraction",
                "recovered_systems",
                "lost_systems",
            ]
        ],
        on="variant_id",
        how="left",
    )
    complete = combined[
        combined.candidate_complete_systems.eq(5)
        & combined.all_solar_proxies_pass.astype(bool)
    ].sort_values("candidate_finite_only_RMS_arcsec")
    compact = json.loads(
        (ROOT / protocol["inputs"]["compact_halo_report"]).read_text(encoding="utf-8")
    )["comparators"]["compact_halo_validation"]["equal_system_radial_RMS_arcsec"]
    direction_summary = {}
    for scope in scopes:
        block = joined[joined.scope.eq(scope) & ~joined.variant_id.eq("baseline")]
        comparable = np.isfinite(block.matched_improvement_fraction_fixed) & np.isfinite(
            block.matched_improvement_fraction_refit
        )
        direction_summary[scope] = {
            "comparable_variants": int(comparable.sum()),
            "surviving_directions": block[
                block.direction_survives.astype(bool)
            ].variant_id.tolist(),
            "reversed_or_unavailable": block[
                ~block.direction_survives.astype(bool)
            ].variant_id.tolist(),
        }
    report = {
        "report_version": "P0554-STRUCTURAL-EXACT-REFIT-RESULTS-0.1.0",
        "status": "complete",
        "protocol": {"path": str(config_path.relative_to(ROOT)), "sha256": sha256(config_path)},
        "coverage": {
            "variants": len(variants),
            "SPARC_galaxies": int(galaxy.galaxy.nunique()),
            "CLASH_systems": int(clusters.system.nunique()),
            "raw_clusters": len(contexts),
            "raw_heldout_images": int(raw[raw.variant_id.eq("baseline")].heldout_images.sum()),
            "geometry_refit_starts": int(protocol["evaluation"]["optimization_starts_per_variant_system"]),
            "exact_geometry_fits": len(geometry_frame),
        },
        "baseline": combined[combined.variant_id.eq("baseline")].iloc[0].to_dict(),
        "complete_solar_safe_ranked_by_all_five_RMS": complete.to_dict("records"),
        "direction_survival": direction_summary,
        "fixed_vs_refit": joined.to_dict("records"),
        "historical_validation_compact_halo_RMS_arcsec": float(compact),
        "geometry_boundary_fits": int(geometry_frame.geometry_at_boundary.astype(bool).sum()),
        "claim_limits": protocol["claim_limits"],
        "verdict": {
            "any_complete_solar_safe_variant": not complete.empty,
            "no_variant_promoted": True,
        },
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    make_figure(scores, raw, joined, output / protocol["outputs"]["figure"])
    best = None if complete.empty else complete.iloc[0]
    summary = f"""# P0554 structural exact refit

Eight frozen structural formulas were tested on 131 SPARC galaxies, 20 CLASH
systems, five raw clusters, and Solar proxies. Six ordinary geometry parameters
were refit per cluster with eight starts; no gravity parameter was fit.

The best complete Solar-safe descriptive formula is
`{best.variant_id if best is not None else 'none'}` with all-five raw RMS
{best.candidate_finite_only_RMS_arcsec if best is not None else float('nan'):.3f}
arcsec. No formula is promoted because the shortlist and systems are spent.
"""
    (output / protocol["outputs"]["summary"]).write_text(summary, encoding="utf-8")
    print(
        json.dumps(
            json_safe(
                {
                    "coverage": report["coverage"],
                    "complete_solar_safe": complete[
                        [
                            "variant_id",
                            "galaxy_outer_RMSE_km_s",
                            "cluster_RMSE_dex",
                            "candidate_finite_only_RMS_arcsec",
                            "candidate_total_roots",
                        ]
                    ].to_dict("records"),
                    "direction_survival": direction_summary,
                }
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
