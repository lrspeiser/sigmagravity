#!/usr/bin/env python3
"""Profile local-neighbor route length, falloff, and source-weight sensitivity."""

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

from run_adaptive_route_multicluster_raw import route_fraction  # noqa: E402
from run_adaptive_route_raw_rxj2129 import baryon_field  # noqa: E402
from run_arc_invariant_absolute_lensing import raw_field  # noqa: E402
from run_clash_stellar_morphology_response import MorphologyLens  # noqa: E402
from run_p0554_local_cross_domain_sensitivity import A0, json_safe, raw_contexts, sha256  # noqa: E402
from run_p0554_route_localization_screen import (  # noqa: E402
    build_directional_field,
    geometry_for,
    linearized_residuals,
    rms,
)
from run_p0554_route_softness_interaction import build_variants, load_route_sources  # noqa: E402


def make_variants(protocol):
    parent = dict(protocol["parent"])
    variants = [{**parent, "coordinate": "parent", "coordinate_value": np.nan}]
    keys = ("softening_kpc", "distance_power", "neighbor_weight_power")
    prefixes = {"softening_kpc": "s", "distance_power": "p", "neighbor_weight_power": "w"}
    for coordinate in keys:
        for value in protocol["coordinates"][coordinate]:
            if np.isclose(float(value), float(parent[coordinate])):
                continue
            variant = dict(parent)
            variant[coordinate] = float(value)
            variant["variant_id"] = f"local_{prefixes[coordinate]}{int(round(100 * float(value))):04d}"
            variant["coordinate"] = coordinate
            variant["coordinate_value"] = float(value)
            variants.append(variant)
    return variants


def summarize(protocol, config_path, output):
    system_scores = pd.read_csv(output / protocol["outputs"]["system_scores"])
    audits = pd.read_csv(output / protocol["outputs"]["field_audits"])
    parent_id = protocol["parent"]["variant_id"]
    parent = system_scores[system_scores.variant_id.eq(parent_id)].set_index("system_label")
    primary_labels = [
        label
        for label in protocol["evaluation"]["systems"]
        if label != protocol["evaluation"]["spent_selection_system"]
    ]
    parent_primary = rms(parent.loc[primary_labels, "heldout_linearized_RMS_arcsec"])
    rows = []
    for variant_id, block in system_scores.groupby("variant_id", sort=False):
        indexed = block.set_index("system_label")
        primary_rms = rms(indexed.loc[primary_labels, "heldout_linearized_RMS_arcsec"])
        changes = 1.0 - (
            indexed.loc[primary_labels, "heldout_linearized_RMS_arcsec"]
            / parent.loc[primary_labels, "heldout_linearized_RMS_arcsec"]
        )
        first = block.iloc[0]
        rows.append(
            {
                "variant_id": variant_id,
                "coordinate": first.coordinate,
                "coordinate_value": first.coordinate_value,
                "softening_kpc": first.softening_kpc,
                "distance_power": first.distance_power,
                "neighbor_weight_power": first.neighbor_weight_power,
                "primary_equal_system_RMS_arcsec": primary_rms,
                "primary_improvement_fraction_vs_local_parent": 1.0 - primary_rms / parent_primary,
                "primary_systems_improved": int(changes.gt(0).sum()),
                "minimum_primary_system_improvement_fraction": float(changes.min()),
                "maximum_primary_system_improvement_fraction": float(changes.max()),
                "spent_MACS1931_improvement_fraction": 1.0
                - float(indexed.loc["MACS1931", "heldout_linearized_RMS_arcsec"])
                / float(parent.loc["MACS1931", "heldout_linearized_RMS_arcsec"]),
            }
        )
    scores = pd.DataFrame(rows).sort_values("primary_equal_system_RMS_arcsec")
    summaries = []
    parent_values = protocol["parent"]
    neighbor_pairs = {
        "softening_kpc": (150.0, 250.0),
        "distance_power": (1.5, 2.5),
        "neighbor_weight_power": (0.75, 1.25),
    }
    for coordinate, values in protocol["coordinates"].items():
        block_ids = system_scores[
            system_scores.coordinate.eq(coordinate)
        ].variant_id.unique().tolist() + [parent_id]
        block = scores[scores.variant_id.isin(block_ids)].copy()
        block["profile_value"] = block[coordinate]
        block = block.sort_values("profile_value")
        best = block.iloc[0] if len(block) == 1 else block.sort_values(
            "primary_equal_system_RMS_arcsec"
        ).iloc[0]
        low_value, high_value = neighbor_pairs[coordinate]
        low = block[np.isclose(block.profile_value, low_value)].iloc[0]
        high = block[np.isclose(block.profile_value, high_value)].iloc[0]
        elasticity = (
            np.log(high.primary_equal_system_RMS_arcsec / low.primary_equal_system_RMS_arcsec)
            / np.log(high_value / low_value)
        )
        per_system_best = {}
        for label in primary_labels:
            local = system_scores[
                system_scores.variant_id.isin(block.variant_id)
                & system_scores.system_label.eq(label)
            ].sort_values("heldout_linearized_RMS_arcsec")
            per_system_best[label] = float(local.iloc[0][coordinate])
        summaries.append(
            {
                "coordinate": coordinate,
                "parent_value": float(parent_values[coordinate]),
                "best_value": float(best[coordinate]),
                "best_variant_id": best.variant_id,
                "best_improvement_fraction_vs_parent": float(
                    best.primary_improvement_fraction_vs_local_parent
                ),
                "profile_RMS_span_arcsec": float(
                    block.primary_equal_system_RMS_arcsec.max()
                    - block.primary_equal_system_RMS_arcsec.min()
                ),
                "local_log_elasticity_at_parent": float(elasticity),
                "best_primary_systems_improved": int(best.primary_systems_improved),
                "best_minimum_system_improvement_fraction": float(
                    best.minimum_primary_system_improvement_fraction
                ),
                "best_at_tested_boundary": bool(
                    np.isclose(float(best[coordinate]), min(values))
                    or np.isclose(float(best[coordinate]), max(values))
                ),
                "per_system_best_values": per_system_best,
            }
        )
    coordinate_summary = pd.DataFrame(
        [{**row, "per_system_best_values": json.dumps(row["per_system_best_values"], sort_keys=True)} for row in summaries]
    ).sort_values("profile_RMS_span_arcsec", ascending=False)
    selection = protocol["evaluation"]
    shortlist = scores[
        ~scores.variant_id.eq(parent_id)
        & scores.primary_improvement_fraction_vs_local_parent.ge(
            float(selection["shortlist_minimum_improvement_fraction_vs_local_parent"])
        )
        & scores.primary_systems_improved.ge(
            int(selection["shortlist_minimum_primary_systems_improved"])
        )
        & scores.minimum_primary_system_improvement_fraction.ge(
            -float(selection["shortlist_maximum_single_system_worsening_fraction"])
        )
    ].copy()
    scores.to_csv(output / protocol["outputs"]["scores"], index=False)
    coordinate_summary.to_csv(
        output / protocol["outputs"]["coordinate_summary"], index=False
    )
    shortlist.to_csv(output / protocol["outputs"]["shortlist"], index=False)
    interaction = json.loads(
        (ROOT / protocol["inputs"]["interaction_protocol"]).read_text(encoding="utf-8")
    )
    cross_domain = pd.read_csv(
        ROOT
        / interaction["outputs"]["directory"]
        / interaction["outputs"]["variant_scores"]
    ).set_index("variant_id").loc["lensing_softness_098"]
    report = {
        "report_version": "P0554-LOCAL-NEIGHBOR-PARAMETER-SCREEN-RESULTS-0.2.0",
        "status": "complete",
        "protocol": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": sha256(config_path),
        },
        "coverage": {
            "variants": len(scores),
            "systems": int(system_scores.system_label.nunique()),
            "variant_system_scores": len(system_scores),
            "route_fields": len(audits),
            "coordinate_profiles": len(coordinate_summary),
        },
        "parent_primary_equal_system_RMS_arcsec": parent_primary,
        "coordinate_impact_ranked": [
            {
                **row,
                "per_system_best_values": json.loads(row["per_system_best_values"]),
            }
            for row in coordinate_summary.to_dict("records")
        ],
        "shortlist": shortlist.to_dict("records"),
        "best_overall": scores.iloc[0].to_dict(),
        "cross_domain_preservation": {
            "galaxy_outer_RMSE_km_s": float(cross_domain.galaxy_outer_RMSE_km_s),
            "CLASH_radial_RMSE_dex": float(cross_domain.cluster_RMSE_dex),
            "Mercury_precession_mas_per_century": float(
                cross_domain.Mercury_precession_mas_per_century
            ),
            "all_solar_proxies_pass": bool(cross_domain.all_solar_proxies_pass),
            "interpretation": "identical for every local-neighbor parameter because the angular correction has zero circular monopole and is absent in point-mass Solar controls",
        },
        "field_invariants": {
            "maximum_route_map_normalization_error": float(audits.route_map_normalization_error.max()),
            "maximum_annular_convergence_error": float(
                audits.maximum_annular_convergence_mean_fraction.max()
            ),
            "maximum_normalized_curl_RMS": float(audits.normalized_curl_RMS.max()),
        },
        "verdict": {
            "any_variant_meets_exact_followup_shortlist_rule": not shortlist.empty,
            "most_impactful_coordinate": coordinate_summary.iloc[0].coordinate,
            "no_formula_promoted": True,
        },
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    labels = {
        "softening_kpc": "softening (kpc)",
        "distance_power": "distance falloff power",
        "neighbor_weight_power": "neighbor light-weight power",
    }
    for ax, coordinate in zip(axes, labels):
        block_ids = system_scores[system_scores.coordinate.eq(coordinate)].variant_id.unique().tolist() + [parent_id]
        block = scores[scores.variant_id.isin(block_ids)].sort_values(coordinate)
        ax.plot(block[coordinate], 100.0 * block.primary_improvement_fraction_vs_local_parent, "o-")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.axvline(float(protocol["parent"][coordinate]), color="0.5", linestyle="--")
        ax.set(xlabel=labels[coordinate], ylabel="improvement vs local parent (%)", title=coordinate)
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)

    top = coordinate_summary.iloc[0]
    summary = f"""# P0554 local-neighbor parameter screen

The frozen one-at-a-time screen tested {len(scores)} local-neighbor formulas.
The most impactful coordinate is `{top.coordinate}`, spanning
{top.profile_RMS_span_arcsec:.4f} arcsec in the primary equal-system
local-response score. Its descriptive best tested value is {top.best_value:g},
with {100.0 * top.best_improvement_fraction_vs_parent:+.3f}% change versus the
200-kpc, inverse-square, linear-light parent.

{len(shortlist)} variants satisfy the frozen rule for an exact follow-up. This
is a spent-data sensitivity result; no formula is promoted.
"""
    (output / protocol["outputs"]["summary"]).write_text(summary, encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Rebuild summaries and figures from the archived field scores.",
    )
    args = parser.parse_args()
    config_path = ROOT / "configs/p0554_local_neighbor_parameter_screen_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if not protocol["status"].startswith("frozen_"):
        raise RuntimeError("local-neighbor parameter protocol is not frozen")
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    if args.postprocess_only:
        report = summarize(protocol, config_path, output)
        print(json.dumps(json_safe(report["verdict"]), indent=2), flush=True)
        return
    interaction = json.loads(
        (ROOT / protocol["inputs"]["interaction_protocol"]).read_text(encoding="utf-8")
    )
    parent = next(
        item for item in build_variants(interaction) if item.variant_id == "combined_parent"
    )
    contexts = raw_contexts(interaction)
    route_sources, route_protocols = load_route_sources(interaction, contexts)
    geometry = pd.read_csv(ROOT / protocol["inputs"]["fixed_geometry"])
    variants = make_variants(protocol)
    system_rows, audit_rows = [], []
    for context in contexts:
        parameters = geometry_for(geometry, context.label)
        radial, _ = raw_field(parent.spec, parent.q, context.anchors, context.local, A0)
        baryons = baryon_field(context.anchors, context.local)
        adaptive = route_fraction(parent.candidate, route_sources[context.label], context.local)
        strength = 0.30 * float(adaptive["routing_fraction"] ** parent.route_power)
        for variant in variants:
            print(f"{context.label}: {variant['variant_id']}", flush=True)
            morphology, audit = build_directional_field(
                interaction,
                route_protocols[context.label],
                context,
                route_sources[context.label],
                parent,
                radial,
                baryons,
                variant,
            )
            lens = MorphologyLens(
                context.local,
                {parent.variant_id: radial},
                parent=parent.variant_id,
                morphology=morphology,
                fraction=strength,
            )
            _, sources = lens.profiled_residuals(parent.variant_id, parameters, context.training)
            heldout = linearized_residuals(
                lens, parent.variant_id, parameters, sources, context.heldout, "heldout"
            )
            system_rows.append(
                {
                    "system_label": context.label,
                    "variant_id": variant["variant_id"],
                    "coordinate": variant["coordinate"],
                    "coordinate_value": variant["coordinate_value"],
                    "softening_kpc": variant["softening_kpc"],
                    "distance_power": variant["distance_power"],
                    "neighbor_weight_power": variant["neighbor_weight_power"],
                    "heldout_linearized_RMS_arcsec": rms(
                        heldout.linearized_radial_residual_arcsec
                    ),
                }
            )
            audit_rows.append(
                {
                    "system_label": context.label,
                    "variant_id": variant["variant_id"],
                    "coordinate": variant["coordinate"],
                    "coordinate_value": variant["coordinate_value"],
                    **audit,
                }
            )
    pd.DataFrame(system_rows).to_csv(
        output / protocol["outputs"]["system_scores"], index=False
    )
    pd.DataFrame(audit_rows).to_csv(
        output / protocol["outputs"]["field_audits"], index=False
    )
    report = summarize(protocol, config_path, output)
    print(
        json.dumps(
            json_safe(
                {
                    "coverage": report["coverage"],
                    "coordinate_impact_ranked": report["coordinate_impact_ranked"],
                    "shortlist": report["shortlist"],
                    "verdict": report["verdict"],
                }
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
