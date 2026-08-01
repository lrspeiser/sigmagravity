#!/usr/bin/env python3
"""Test parent-preserving structural deformations of the P0554 response law."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0554_local_cross_domain_sensitivity import (  # noqa: E402
    evaluate_raw_context,
    json_safe,
    raw_contexts,
    sha256,
)
from run_p0554_multiscale_elasticity import (  # noqa: E402
    build_central_differences,
    build_variants,
    evaluate_scalar_domains,
    make_figure,
    summarize_parameters,
)


def top_stable_by_domain(summary: pd.DataFrame) -> dict:
    output = {}
    for domain in ("galaxy", "cluster", "RXJ2129", "four_cluster", "Mercury"):
        column = f"{domain}_median_abs_slope"
        available = summary[
            np.isfinite(pd.to_numeric(summary[column], errors="coerce"))
            & summary[f"{domain}_stable_material"].astype(bool)
        ]
        if available.empty:
            available = summary[
                np.isfinite(pd.to_numeric(summary[column], errors="coerce"))
            ]
        row = available.sort_values(column, ascending=False).iloc[0]
        output[domain] = {
            "parameter": row.parameter,
            "median_abs_normalized_slope": float(row[column]),
            "direction_consistency": float(row[f"{domain}_direction_consistency"]),
            "better_direction": row[f"{domain}_better_direction"],
            "stable_material": bool(row[f"{domain}_stable_material"]),
        }
    return output


def main():
    config_path = ROOT / "configs/p0554_structural_microvariations_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    variants = build_variants(protocol)

    scores, morphology, galaxy, clusters = evaluate_scalar_domains(protocol, variants)
    scores.to_csv(output / protocol["outputs"]["variant_scores"], index=False)
    morphology.to_csv(output / protocol["outputs"]["galaxy_morphology"], index=False)

    raw_rows, raw_predictions = [], []
    contexts = raw_contexts(protocol)
    for context in contexts:
        rows, predictions, _ = evaluate_raw_context(context, variants)
        raw_rows.extend(rows)
        raw_predictions.extend(predictions)
    raw = pd.DataFrame(raw_rows)
    raw.to_csv(output / protocol["outputs"]["raw_scores"], index=False)
    pd.concat(raw_predictions, ignore_index=True).to_csv(
        output / protocol["outputs"]["raw_predictions"], index=False
    )

    central = build_central_differences(protocol, scores, raw)
    summary = summarize_parameters(protocol, central)
    central.to_csv(output / protocol["outputs"]["central_differences"], index=False)
    summary.to_csv(output / protocol["outputs"]["parameter_summary"], index=False)
    top = top_stable_by_domain(summary)
    raw_roots = raw.groupby("variant_id").heldout_roots_converged.sum()
    baseline = scores[scores.variant_id.eq("baseline")].iloc[0]
    report = {
        "report_version": "P0554-STRUCTURAL-MICROVARIATIONS-RESULTS-0.1.0",
        "status": "complete",
        "protocol": {"path": str(config_path.relative_to(ROOT)), "sha256": sha256(config_path)},
        "coverage": {
            "variants": len(variants),
            "structural_parameters": len(protocol["parameter_coordinates"]),
            "coordinate_steps": len(protocol["coordinate_steps"]),
            "SPARC_galaxies": int(galaxy.galaxy.nunique()),
            "SPARC_outer_points": int(galaxy.split.eq("outer_holdout").sum()),
            "CLASH_systems": int(clusters.system.nunique()),
            "CLASH_points": len(clusters),
            "raw_clusters": len(contexts),
            "raw_heldout_images": int(raw[raw.variant_id.eq("baseline")].heldout_images.sum()),
        },
        "baseline": baseline.to_dict(),
        "parent_reproduction": {
            "galaxy_outer_RMSE_km_s": float(baseline.galaxy_outer_RMSE_km_s),
            "cluster_RMSE_dex": float(baseline.cluster_RMSE_dex),
            "Mercury_precession_mas_per_century": float(
                baseline.Mercury_precession_mas_per_century
            ),
            "raw_roots": int(raw_roots.loc["baseline"]),
        },
        "top_stable_structural_parameter_by_domain": top,
        "stable_same_direction_nonSolar": summary[
            summary.stable_nonSolar_directions_agree.astype(bool)
        ].parameter.tolist(),
        "root_topology": {
            "baseline_roots": int(raw_roots.loc["baseline"]),
            "minimum_roots": int(raw_roots.min()),
            "maximum_roots": int(raw_roots.max()),
            "parameters_bifurcating_at_smallest_step": summary[
                summary.root_bifurcation_at_smallest_step.astype(bool)
            ].parameter.tolist(),
            "parameters_never_changing_roots": summary[
                summary.smallest_root_bifurcation_u.isna()
            ].parameter.tolist(),
        },
        "solar_boundary_crossings": summary[
            summary.smallest_solar_boundary_crossing_u.notna()
        ][["parameter", "smallest_solar_boundary_crossing_u"]].to_dict("records"),
        "parameter_summary": summary.to_dict("records"),
        "claim_limits": protocol["claim_limits"],
        "verdict": {
            "candidate_selected": False,
            "purpose": "structural sensitivity and topology diagnostics only",
        },
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    make_figure(central, summary, output / protocol["outputs"]["figure"])
    summary_text = f"""# P0554 structural microvariations

The frozen experiment tested {len(variants)} parent-preserving structural
variants across galaxies, derived CLASH lensing, five raw clusters, and Solar
proxies. No gravity coefficient or lens geometry was fit.

The strongest stable structural levers are
`{top['galaxy']['parameter']}` for galaxies,
`{top['cluster']['parameter']}` for CLASH,
`{top['RXJ2129']['parameter']}` for RX J2129,
`{top['four_cluster']['parameter']}` for the four other raw clusters, and
`{top['Mercury']['parameter']}` for Mercury. Root topology spans
{report['root_topology']['minimum_roots']}--{report['root_topology']['maximum_roots']}
of 18 images around the 17-root parent.

No candidate is selected; these are algebraic sensitivity diagnostics on spent
systems, not a derived field theory.
"""
    (output / protocol["outputs"]["summary"]).write_text(summary_text, encoding="utf-8")
    print(
        json.dumps(
            json_safe(
                {
                    "coverage": report["coverage"],
                    "parent_reproduction": report["parent_reproduction"],
                    "top_stable_structural_parameter_by_domain": top,
                    "stable_same_direction_nonSolar": report[
                        "stable_same_direction_nonSolar"
                    ],
                    "root_topology": report["root_topology"],
                    "solar_boundary_crossings": report["solar_boundary_crossings"],
                }
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
