#!/usr/bin/env python3
"""Scan a one-parameter continuation below the P0554 angular-route caustic."""

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
from run_adaptive_route_raw_rxj2129 import baryon_field, build_route_field  # noqa: E402
from run_arc_invariant_absolute_lensing import raw_field  # noqa: E402
from run_clash_stellar_morphology_response import MorphologyLens  # noqa: E402
from run_p0554_local_cross_domain_sensitivity import A0, json_safe, raw_contexts, sha256  # noqa: E402
from run_p0554_multifamily_multiplicity import classify_family  # noqa: E402
from run_p0554_route_softness_interaction import (  # noqa: E402
    build_variants,
    fit_variant,
    load_route_sources,
)
from run_rxj2129_raw_theory_lensing import FIXED_LABELS, near_bound  # noqa: E402


def eta_id(value):
    return f"eta_{int(round(100 * float(value))):03d}"


def make_figure(eta_summary, family_summary, output):
    ordered = eta_summary.sort_values("eta")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    axes[0].plot(ordered.eta, ordered.equal_family_assignment_RMS_arcsec, "o-")
    axes[0].scatter(
        ordered.loc[ordered.subcritical.astype(bool), "eta"],
        ordered.loc[ordered.subcritical.astype(bool), "equal_family_assignment_RMS_arcsec"],
        color="tab:green",
        label="subcritical",
        zorder=3,
    )
    axes[0].set(
        xlabel="route continuation eta",
        ylabel="equal-family assigned RMS (arcsec)",
        title="Continuous position accuracy",
    )
    axes[0].legend()

    selected = family_summary[family_summary.source_family.isin([2, 3])]
    for family, block in selected.groupby("source_family"):
        axes[1].plot(block.eta, block.global_roots, "o-", label=f"family {int(family)}")
    axes[1].set(
        xlabel="route continuation eta",
        ylabel="global roots",
        title="MACS1931 caustic crossings",
    )
    axes[1].legend()

    axes[2].plot(ordered.eta, ordered.potentially_observable_surplus_roots, "o-", color="tab:purple")
    axes[2].set(
        xlabel="route continuation eta",
        ylabel="potentially observable surplus roots",
        title="Multiplicity liability across seven families",
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _first_changed_eta(eta_summary, column, baseline_value):
    changed = eta_summary[~eta_summary[column].eq(baseline_value)]
    return None if changed.empty else float(changed.eta.min())


def postprocess_existing(protocol, config_path, output, route_metadata=None):
    """Rebuild summaries from saved expensive-fit products without refitting."""
    eta_summary = pd.read_csv(output / protocol["outputs"]["eta_summary"])
    family_summary = pd.read_csv(output / protocol["outputs"]["family_summary"])
    roots = pd.read_csv(output / protocol["outputs"]["global_roots"])
    geometry = pd.read_csv(output / protocol["outputs"]["geometry"])
    heldout = pd.read_csv(output / protocol["outputs"]["heldout_predictions"])

    converged = heldout[
        heldout.root_converged.astype(bool) & heldout.radial_residual_arcsec.notna()
    ]
    converged_rms = converged.groupby("eta").radial_residual_arcsec.apply(
        lambda values: float(np.sqrt(np.mean(np.square(values))))
    )
    eta_summary["heldout_converged_only_RMS_arcsec"] = eta_summary.eta.map(converged_rms)

    eta_summary = eta_summary.sort_values("eta").reset_index(drop=True)
    baseline = eta_summary[eta_summary.eta.eq(0.0)].iloc[0]
    eta_summary["retains_eta0_root_count_vector"] = eta_summary.root_count_vector.eq(
        baseline.root_count_vector
    )
    eta_summary["does_not_increase_observable_surplus"] = (
        eta_summary.potentially_observable_surplus_roots
        <= baseline.potentially_observable_surplus_roots
    )
    eta_summary["subcritical"] = (
        eta_summary.retains_eta0_root_count_vector.astype(bool)
        & eta_summary.does_not_increase_observable_surplus.astype(bool)
    )
    eta_summary["assignment_improvement_fraction_vs_eta0"] = 1.0 - (
        eta_summary.equal_family_assignment_RMS_arcsec
        / baseline.equal_family_assignment_RMS_arcsec
    )
    survivors = eta_summary[eta_summary.subcritical.astype(bool)].sort_values(
        ["equal_family_assignment_RMS_arcsec", "eta"]
    )
    best = survivors.iloc[0]
    full_strength = eta_summary[eta_summary.eta.eq(1.0)].iloc[0]
    topology_changes = eta_summary[
        ~eta_summary.retains_eta0_root_count_vector.astype(bool)
    ]
    first_any = None if topology_changes.empty else float(topology_changes.eta.min())
    first_family_2 = _first_changed_eta(
        eta_summary, "family_2_roots", baseline.family_2_roots
    )
    first_family_3 = _first_changed_eta(
        eta_summary, "family_3_roots", baseline.family_3_roots
    )

    eta_summary.to_csv(output / protocol["outputs"]["eta_summary"], index=False)
    make_figure(eta_summary, family_summary, output / protocol["outputs"]["figure"])

    old_report_path = output / protocol["outputs"]["report"]
    old_report = (
        json.loads(old_report_path.read_text(encoding="utf-8"))
        if old_report_path.exists()
        else {}
    )
    route = route_metadata or old_report.get("route", {})
    report = {
        "report_version": "P0554-SUBCRITICAL-ROUTE-SCAN-RESULTS-0.2.0",
        "status": "complete",
        "protocol": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": sha256(config_path),
        },
        "coverage": {
            "eta_values": len(eta_summary),
            "geometry_fits": len(geometry),
            "optimization_starts": len(eta_summary)
            * int(protocol["evaluation"]["optimization_starts_per_eta"]),
            "formula_family_searches": len(family_summary),
            "accepted_global_roots": len(roots),
            "published_images": int(protocol["evaluation"]["published_images"]),
            "source_families": int(protocol["evaluation"]["source_families"]),
        },
        "route": route,
        "eta0": baseline.to_dict(),
        "descriptive_best_subcritical": best.to_dict(),
        "full_strength": full_strength.to_dict(),
        "maximum_subcritical_grid_eta": float(
            eta_summary[eta_summary.subcritical.astype(bool)].eta.max()
        ),
        "topology_change_eta_values": topology_changes.eta.tolist(),
        "first_any_topology_change_eta": first_any,
        "first_family_2_topology_change_eta": first_family_2,
        "first_family_3_topology_change_eta": first_family_3,
        "eta_summary": eta_summary.to_dict("records"),
        "verdict": {
            "any_positive_subcritical_eta": bool(
                eta_summary[eta_summary.eta.gt(0)].subcritical.astype(bool).any()
            ),
            "any_subcritical_eta_improves_assignment_RMS": bool(
                survivors.assignment_improvement_fraction_vs_eta0.gt(0).any()
            ),
            "topology_and_position_effects_partially_separable": bool(
                best.eta > 0 and best.assignment_improvement_fraction_vs_eta0 > 0
            ),
            "best_subcritical_assignment_improvement_fraction": float(
                best.assignment_improvement_fraction_vs_eta0
            ),
            "full_strength_assignment_improvement_fraction": float(
                full_strength.assignment_improvement_fraction_vs_eta0
            ),
            "most_full_strength_gain_occurs_after_topology_change": bool(
                full_strength.assignment_improvement_fraction_vs_eta0
                > 10.0 * max(best.assignment_improvement_fraction_vs_eta0, 0.0)
            ),
            "no_formula_promoted": True,
        },
        "claim_limits": protocol["claim_limits"],
    }
    old_report_path.write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    summary = f"""# P0554 subcritical route scan

The frozen eta continuation tested {len(eta_summary)} amplitudes with eight-start
geometry refits and seven-family global root searches. The eta=0 root vector is
`{baseline.root_count_vector}`. The best descriptive subcritical value is
**eta={best.eta:.2f}**, with equal-family assigned RMS
**{best.equal_family_assignment_RMS_arcsec:.3f} arcsec** versus
{baseline.equal_family_assignment_RMS_arcsec:.3f} at eta=0
({100.0 * best.assignment_improvement_fraction_vs_eta0:.3f}% improvement).

The first topology change is at eta={first_any:.2f}: family 3 changes first at
eta={first_family_3:.2f}, while family 2 does not change until eta={first_family_2:.2f}.
Full strength improves the assigned RMS by
{100.0 * full_strength.assignment_improvement_fraction_vs_eta0:.3f}%, but predicts
{int(full_strength.potentially_observable_surplus_roots)} potentially observable
surplus roots versus {int(baseline.potentially_observable_surplus_roots)} at eta=0.

The strict observed-seed held-out RMS remains infinite until every root converges.
The saved `heldout_converged_only_RMS_arcsec` is a transparent partial diagnostic,
not a replacement success score. No formula is promoted because eta was selected
on spent MACS1931.
"""
    (output / protocol["outputs"]["summary"]).write_text(summary, encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="rebuild summaries from saved scan products without expensive refits",
    )
    args = parser.parse_args()
    config_path = ROOT / "configs" / "p0554_subcritical_route_scan_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if not protocol["status"].startswith("frozen_"):
        raise RuntimeError("subcritical route protocol is not frozen")
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    if args.postprocess_only:
        report = postprocess_existing(protocol, config_path, output)
        print(
            json.dumps(
                json_safe(
                    {
                        "coverage": report["coverage"],
                        "eta0": report["eta0"],
                        "best": report["descriptive_best_subcritical"],
                        "verdict": report["verdict"],
                    }
                ),
                indent=2,
            ),
            flush=True,
        )
        return
    interaction = json.loads(
        (ROOT / protocol["inputs"]["interaction_protocol"]).read_text(encoding="utf-8")
    )
    parent = next(
        item for item in build_variants(interaction) if item.variant_id == "combined_parent"
    )
    contexts = raw_contexts(interaction)
    context = next(item for item in contexts if item.label == protocol["evaluation"]["system"])
    sources, route_protocols = load_route_sources(interaction, contexts)
    context.local["optimization"]["maximum_function_evaluations"] = int(
        protocol["evaluation"]["maximum_function_evaluations"]
    )
    baryons = baryon_field(context.anchors, context.local)
    radial, _ = raw_field(parent.spec, parent.q, context.anchors, context.local, A0)
    angular, route_audit = build_route_field(
        route_protocols[context.label],
        context.local,
        sources[context.label],
        parent.candidate,
        radial,
        baryons,
        contrast_cap=float(interaction["route_parent"]["contrast_cap"]),
        contrast_strength=1.0,
        centroid_mode=str(interaction["route_parent"]["centroid_mode"]),
    )
    adaptive = route_fraction(parent.candidate, sources[context.label], context.local)
    parent_strength = float(adaptive["routing_fraction"] ** parent.route_power)
    all_images = pd.concat([context.training, context.heldout], ignore_index=True)
    settings = {
        **protocol["evaluation"],
        "potentially_observable_surplus_fraction": protocol["evaluation"][
            "potentially_observable_surplus_fraction"
        ],
    }
    eta_rows, family_rows, root_rows, assignment_rows = [], [], [], []
    geometry_rows, heldout_frames = [], []
    for index, eta in enumerate(protocol["formula"]["eta_values"]):
        value = float(eta)
        variant_id = eta_id(value)
        print(f"{variant_id}: fit eight-start geometry", flush=True)
        lens = MorphologyLens(
            context.local,
            {parent.variant_id: radial},
            parent=parent.variant_id,
            morphology=angular,
            fraction=value * parent_strength,
        )
        fit, training, heldout, training_score, heldout_score = fit_variant(
            lens,
            parent.variant_id,
            context,
            starts=int(protocol["evaluation"]["optimization_starts_per_eta"]),
            seed=int(protocol["evaluation"]["random_seed"]) + index * 100,
        )
        local_heldout = heldout.copy()
        local_heldout.insert(0, "eta", value)
        local_heldout.insert(0, "variant_id", variant_id)
        heldout_frames.append(local_heldout)
        geometry_rows.append(
            {
                "variant_id": variant_id,
                "eta": value,
                **dict(zip(FIXED_LABELS, fit["result"].x)),
                "optimizer_cost": float(fit["result"].cost),
                "geometry_at_boundary": any(near_bound(parent.variant_id, fit["result"].x).values()),
            }
        )
        current_family_rows = []
        for family, family_images in all_images.groupby("source_family", sort=True):
            roots, assignments, summary = classify_family(
                lens,
                parent,
                fit["result"].x,
                fit["sources"][int(family)],
                family_images,
                settings,
                context.label,
            )
            for row in roots:
                row["variant_id"] = variant_id
                row["eta"] = value
            for row in assignments:
                row["variant_id"] = variant_id
                row["eta"] = value
            summary["variant_id"] = variant_id
            summary["eta"] = value
            root_rows.extend(roots)
            assignment_rows.extend(assignments)
            family_rows.append(summary)
            current_family_rows.append(summary)
        block = pd.DataFrame(current_family_rows)
        eta_rows.append(
            {
                "variant_id": variant_id,
                "eta": value,
                "applied_angular_strength": value * parent_strength,
                "training_RMS_arcsec": training_score["exact_radial_RMS_arcsec"],
                "heldout_observed_seed_RMS_arcsec": heldout_score["exact_radial_RMS_arcsec"],
                "heldout_observed_seed_roots": heldout_score["converged_roots"],
                "heldout_observed_seed_all_roots": heldout_score["all_roots_converged"],
                "equal_family_assignment_RMS_arcsec": float(block.assignment_RMS_arcsec.mean()),
                "families_all_observed_assigned": int(block.all_observed_images_assigned.astype(bool).sum()),
                "global_roots": int(block.global_roots.sum()),
                "potentially_observable_surplus_roots": int(
                    block.potentially_observable_surplus_roots.sum()
                ),
                "family_2_roots": int(block[block.source_family.eq(2)].global_roots.iloc[0]),
                "family_3_roots": int(block[block.source_family.eq(3)].global_roots.iloc[0]),
                "root_count_vector": ";".join(
                    str(int(value)) for value in block.sort_values("source_family").global_roots
                ),
                "optimizer_cost": float(fit["result"].cost),
                "geometry_at_boundary": geometry_rows[-1]["geometry_at_boundary"],
            }
        )
    eta_summary = pd.DataFrame(eta_rows).sort_values("eta")
    family_summary = pd.DataFrame(family_rows)
    baseline = eta_summary[eta_summary.eta.eq(0.0)].iloc[0]
    eta_summary["retains_eta0_root_count_vector"] = eta_summary.root_count_vector.eq(
        baseline.root_count_vector
    )
    eta_summary["does_not_increase_observable_surplus"] = (
        eta_summary.potentially_observable_surplus_roots
        <= baseline.potentially_observable_surplus_roots
    )
    eta_summary["subcritical"] = (
        eta_summary.retains_eta0_root_count_vector
        & eta_summary.does_not_increase_observable_surplus
    )
    eta_summary["assignment_improvement_fraction_vs_eta0"] = 1.0 - (
        eta_summary.equal_family_assignment_RMS_arcsec
        / baseline.equal_family_assignment_RMS_arcsec
    )
    survivors = eta_summary[eta_summary.subcritical.astype(bool)].sort_values(
        ["equal_family_assignment_RMS_arcsec", "eta"]
    )
    best = survivors.iloc[0]
    eta_summary.to_csv(output / protocol["outputs"]["eta_summary"], index=False)
    family_summary.to_csv(output / protocol["outputs"]["family_summary"], index=False)
    pd.DataFrame(root_rows).to_csv(output / protocol["outputs"]["global_roots"], index=False)
    pd.DataFrame(assignment_rows).to_csv(output / protocol["outputs"]["assignments"], index=False)
    pd.DataFrame(geometry_rows).to_csv(output / protocol["outputs"]["geometry"], index=False)
    pd.concat(heldout_frames, ignore_index=True).to_csv(
        output / protocol["outputs"]["heldout_predictions"], index=False
    )
    report = postprocess_existing(
        protocol,
        config_path,
        output,
        route_metadata={
            "parent_angular_strength": parent_strength,
            "routing_fraction": float(adaptive["routing_fraction"]),
            "route_power": parent.route_power,
            "maximum_route_curl_RMS": float(route_audit["normalized_curl_RMS"]),
        },
    )
    print(json.dumps(json_safe({"coverage": report["coverage"], "eta0": report["eta0"], "best": report["descriptive_best_subcritical"], "verdict": report["verdict"]}), indent=2), flush=True)


if __name__ == "__main__":
    main()
