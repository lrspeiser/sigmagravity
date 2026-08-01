#!/usr/bin/env python3
"""Transfer the MACS1931-selected subcritical route to four untouched clusters."""

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


VARIANTS = (("eta_000", 0.0), ("eta_030", 0.3))


def converged_only_rms(frame):
    valid = frame[frame.root_converged.astype(bool) & frame.radial_residual_arcsec.notna()]
    if valid.empty:
        return np.nan
    return float(np.sqrt(np.mean(np.square(valid.radial_residual_arcsec))))


def aggregate_variants(families, systems, excluded):
    rows = []
    for variant_id, block in families.groupby("variant_id", sort=False):
        primary = block[~block.system_label.eq(excluded)]
        rows.append(
            {
                "variant_id": variant_id,
                "families": len(block),
                "families_all_observed_assigned": int(
                    block.all_observed_images_assigned.astype(bool).sum()
                ),
                "families_missing_multiplicity": int(
                    block.multiplicity_classification.eq("missing_multiplicity").sum()
                ),
                "families_exact_multiplicity": int(
                    block.multiplicity_classification.eq("exact_multiplicity").sum()
                ),
                "families_demagnified_only_surplus": int(
                    block.multiplicity_classification.eq("demagnified_only_surplus").sum()
                ),
                "families_potentially_observable_surplus": int(
                    block.multiplicity_classification.eq("potentially_observable_surplus").sum()
                ),
                "potentially_observable_surplus_roots": int(
                    block.potentially_observable_surplus_roots.sum()
                ),
                "primary_other_four_missing_multiplicity_families": int(
                    primary.multiplicity_classification.eq("missing_multiplicity").sum()
                ),
                "primary_other_four_potentially_observable_surplus_roots": int(
                    primary.potentially_observable_surplus_roots.sum()
                ),
                "heldout_roots_converged": int(
                    systems[systems.variant_id.eq(variant_id)].heldout_roots_converged.sum()
                ),
                "primary_other_four_heldout_roots_converged": int(
                    systems[
                        systems.variant_id.eq(variant_id)
                        & ~systems.system_label.eq(excluded)
                    ].heldout_roots_converged.sum()
                ),
                "geometry_boundary_fits": int(
                    systems[
                        systems.variant_id.eq(variant_id)
                    ].geometry_at_boundary.astype(bool).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_position_comparison(families, excluded):
    pivot = families.pivot_table(
        index=["system_label", "source_family"],
        columns="variant_id",
        values="assignment_RMS_arcsec",
        aggfunc="first",
    ).dropna(subset=["eta_000", "eta_030"])
    pivot = pivot.reset_index()
    rows = []
    for label, block in pivot.groupby("system_label", sort=False):
        baseline = float(block.eta_000.mean())
        candidate = float(block.eta_030.mean())
        rows.append(
            {
                "system_label": label,
                "common_complete_families": len(block),
                "eta_000_equal_family_RMS_arcsec": baseline,
                "eta_030_equal_family_RMS_arcsec": candidate,
                "eta_030_improvement_fraction": 1.0 - candidate / baseline,
                "primary_transfer_system": label != excluded,
            }
        )
    comparisons = pd.DataFrame(rows)
    primary = pivot[~pivot.system_label.eq(excluded)]
    baseline = float(primary.eta_000.mean())
    candidate = float(primary.eta_030.mean())
    aggregate = {
        "common_complete_families": len(primary),
        "eta_000_equal_family_RMS_arcsec": baseline,
        "eta_030_equal_family_RMS_arcsec": candidate,
        "eta_030_improvement_fraction": 1.0 - candidate / baseline,
        "systems_improved": int(
            comparisons[
                comparisons.primary_transfer_system.astype(bool)
            ].eta_030_improvement_fraction.gt(0).sum()
        ),
    }
    return comparisons, aggregate


def make_figure(comparisons, family_summary, excluded, output):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    colors = [
        "0.65" if label == excluded else ("tab:green" if value > 0 else "crimson")
        for label, value in zip(
            comparisons.system_label, comparisons.eta_030_improvement_fraction
        )
    ]
    axes[0].bar(
        comparisons.system_label,
        100.0 * comparisons.eta_030_improvement_fraction,
        color=colors,
    )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set(
        ylabel="eta=0.30 position improvement (%)",
        title="Frozen route transfer by cluster",
    )

    roots = family_summary.pivot_table(
        index=["system_label", "source_family"],
        columns="variant_id",
        values="global_roots",
        aggfunc="first",
    ).dropna()
    roots["delta"] = roots.eta_030 - roots.eta_000
    changed = roots[~roots.delta.eq(0)].reset_index()
    if changed.empty:
        axes[1].text(0.5, 0.5, "No family root-count changes", ha="center", va="center")
        axes[1].set_xticks([])
        axes[1].set_yticks([])
    else:
        labels = [f"{row.system_label} F{int(row.source_family)}" for row in changed.itertuples()]
        axes[1].bar(labels, changed.delta, color="tab:purple")
        axes[1].axhline(0.0, color="black", linewidth=0.8)
        axes[1].tick_params(axis="x", rotation=35)
    axes[1].set(ylabel="eta=0.30 minus eta=0 roots", title="Global topology changes")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def postprocess(protocol, config_path, output):
    systems = pd.read_csv(output / protocol["outputs"]["system_scores"])
    families = pd.read_csv(output / protocol["outputs"]["family_summary"])
    roots = pd.read_csv(output / protocol["outputs"]["global_roots"])
    geometry = pd.read_csv(output / protocol["outputs"]["geometry"])
    excluded = protocol["evaluation"]["selection_system_excluded_from_primary_transfer"]
    specified_labels = list(protocol["evaluation"]["systems"])
    actual_labels = systems.system_label.drop_duplicates().tolist()
    variant_summary = aggregate_variants(families, systems, excluded)
    comparisons, aggregate = paired_position_comparison(families, excluded)
    baseline = variant_summary.set_index("variant_id").loc["eta_000"]
    candidate = variant_summary.set_index("variant_id").loc["eta_030"]
    gates = protocol["primary_transfer_gates"]
    topology = {
        "additional_potentially_observable_surplus_roots": int(
            candidate.primary_other_four_potentially_observable_surplus_roots
            - baseline.primary_other_four_potentially_observable_surplus_roots
        ),
        "additional_missing_multiplicity_families": int(
            candidate.primary_other_four_missing_multiplicity_families
            - baseline.primary_other_four_missing_multiplicity_families
        ),
        "lost_observed_seed_heldout_roots": int(
            baseline.primary_other_four_heldout_roots_converged
            - candidate.primary_other_four_heldout_roots_converged
        ),
    }
    gate_results = {
        "aggregate_position_improvement": aggregate["eta_030_improvement_fraction"]
        >= float(gates["aggregate_other_four_equal_family_assignment_RMS_improvement_fraction_minimum"]),
        "systems_improved": aggregate["systems_improved"]
        >= int(gates["other_four_systems_improved_minimum"]),
        "no_additional_observable_surplus": topology[
            "additional_potentially_observable_surplus_roots"
        ]
        <= int(gates["other_four_additional_potentially_observable_surplus_roots_maximum"]),
        "no_additional_missing_families": topology[
            "additional_missing_multiplicity_families"
        ]
        <= int(gates["other_four_additional_missing_multiplicity_families_maximum"]),
        "no_lost_heldout_roots": topology["lost_observed_seed_heldout_roots"]
        <= int(gates["other_four_lost_observed_seed_heldout_roots_maximum"]),
    }
    strong = all(gate_results.values())
    weak = (
        aggregate["eta_030_improvement_fraction"] > 0
        and gate_results["no_additional_observable_surplus"]
        and gate_results["no_additional_missing_families"]
        and gate_results["no_lost_heldout_roots"]
    )
    root_changes = families.pivot_table(
        index=["system_label", "source_family"],
        columns="variant_id",
        values="global_roots",
        aggfunc="first",
        fill_value=0,
    )
    root_changes["delta_eta030_minus_eta000"] = root_changes.eta_030 - root_changes.eta_000
    root_changes = root_changes[~root_changes.delta_eta030_minus_eta000.eq(0)].reset_index()

    variant_summary.to_csv(output / protocol["outputs"]["variant_summary"], index=False)
    comparisons.to_csv(output / "paired_position_comparisons.csv", index=False)
    make_figure(comparisons, families, excluded, output / protocol["outputs"]["figure"])
    report = {
        "report_version": "P0554-SUBCRITICAL-ROUTE-TRANSFER-RESULTS-0.1.1",
        "status": "complete",
        "protocol": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": sha256(config_path),
        },
        "coverage": {
            "variants": 2,
            "systems": int(systems.system_label.nunique()),
            "geometry_fits": len(geometry),
            "optimization_starts": len(geometry)
            * int(protocol["evaluation"]["optimization_starts_per_variant_cluster"]),
            "source_families": int(families[["system_label", "source_family"]].drop_duplicates().shape[0]),
            "formula_family_searches": len(families),
            "published_images": int(protocol["evaluation"]["published_images"]),
            "accepted_global_roots": len(roots),
        },
        "universal_eta": float(protocol["formula"]["universal_eta"]),
        "primary_transfer_excludes": excluded,
        "protocol_metadata_audit": {
            "specified_human_readable_system_labels": specified_labels,
            "actual_raw_context_labels": actual_labels,
            "labels_match": specified_labels == actual_labels,
            "disposition": "The frozen label list was stale metadata. The executable raw_contexts input selected the actual five datasets shown here; coverage remained the frozen 5 systems, 27 families, and 77 images. The frozen protocol was not rewritten after scoring.",
        },
        "primary_paired_position_comparison": aggregate,
        "primary_topology_changes": topology,
        "per_system_paired_position_comparison": comparisons.to_dict("records"),
        "variant_summary": variant_summary.to_dict("records"),
        "changed_family_root_counts": root_changes.to_dict("records"),
        "gate_results": gate_results,
        "verdict": {
            "strong_transfer": strong,
            "weak_topology_safe_transfer": weak,
            "eta030_universal_formula_promoted": False,
        },
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    summary = f"""# P0554 subcritical route transfer

Eta=0.30 was frozen from MACS1931 and transferred unchanged to the other four
raw clusters. Across {aggregate['common_complete_families']} common complete
primary-transfer families, equal-family assigned RMS changed from
{aggregate['eta_000_equal_family_RMS_arcsec']:.3f} to
{aggregate['eta_030_equal_family_RMS_arcsec']:.3f} arcsec
({100.0 * aggregate['eta_030_improvement_fraction']:+.3f}%).

It improved {aggregate['systems_improved']} of four transfer systems. Relative
to eta=0, it added {topology['additional_potentially_observable_surplus_roots']}
potentially observable surplus roots, added
{topology['additional_missing_multiplicity_families']} missing-multiplicity
families, and lost {topology['lost_observed_seed_heldout_roots']} observed-seed
held-out roots. Strong transfer: **{strong}**. Weak topology-safe transfer:
**{weak}**. No formula is promoted.

Protocol metadata note: the frozen human-readable system list contained stale
names. The executable loader actually evaluated {', '.join(actual_labels)}.
The frozen protocol was retained unchanged and this erratum is recorded in the
machine-readable report.
"""
    (output / protocol["outputs"]["summary"]).write_text(summary, encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--postprocess-only", action="store_true")
    args = parser.parse_args()
    config_path = ROOT / "configs" / "p0554_subcritical_route_transfer_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if not protocol["status"].startswith("frozen_"):
        raise RuntimeError("subcritical transfer protocol is not frozen")
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    if args.postprocess_only:
        report = postprocess(protocol, config_path, output)
        print(json.dumps(json_safe(report["verdict"]), indent=2), flush=True)
        return

    interaction = json.loads(
        (ROOT / protocol["inputs"]["interaction_protocol"]).read_text(encoding="utf-8")
    )
    parent = next(
        item for item in build_variants(interaction) if item.variant_id == "combined_parent"
    )
    contexts = raw_contexts(interaction)
    sources, route_protocols = load_route_sources(interaction, contexts)
    root_rows, assignment_rows, family_rows = [], [], []
    system_rows, geometry_rows, heldout_frames = [], [], []
    for system_index, context in enumerate(contexts):
        context.local["optimization"]["maximum_function_evaluations"] = int(
            protocol["evaluation"]["maximum_function_evaluations"]
        )
        images = pd.concat([context.training, context.heldout], ignore_index=True)
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
        for variant_id, eta in VARIANTS:
            print(f"{context.label}: {variant_id} exact refit", flush=True)
            lens = MorphologyLens(
                context.local,
                {parent.variant_id: radial},
                parent=parent.variant_id,
                morphology=angular,
                fraction=eta * parent_strength,
            )
            paired_seed = int(protocol["evaluation"]["random_seed"]) + system_index * 1000
            fit, training, heldout, training_score, heldout_score = fit_variant(
                lens,
                parent.variant_id,
                context,
                starts=int(protocol["evaluation"]["optimization_starts_per_variant_cluster"]),
                seed=paired_seed,
            )
            local_heldout = heldout.copy()
            local_heldout.insert(0, "eta", eta)
            local_heldout.insert(0, "variant_id", variant_id)
            local_heldout.insert(0, "system_label", context.label)
            heldout_frames.append(local_heldout)
            boundary = any(near_bound(parent.variant_id, fit["result"].x).values())
            geometry_rows.append(
                {
                    "system_label": context.label,
                    "variant_id": variant_id,
                    "eta": eta,
                    **dict(zip(FIXED_LABELS, fit["result"].x)),
                    "optimizer_cost": float(fit["result"].cost),
                    "geometry_at_boundary": boundary,
                }
            )
            system_rows.append(
                {
                    "system_label": context.label,
                    "variant_id": variant_id,
                    "eta": eta,
                    "published_images": len(images),
                    "training_RMS_arcsec": training_score["exact_radial_RMS_arcsec"],
                    "training_roots_converged": training_score["converged_roots"],
                    "heldout_RMS_arcsec": heldout_score["exact_radial_RMS_arcsec"],
                    "heldout_roots_converged": heldout_score["converged_roots"],
                    "heldout_images": len(context.heldout),
                    "heldout_converged_only_RMS_arcsec": converged_only_rms(local_heldout),
                    "applied_angular_strength": eta * parent_strength,
                    "route_curl_RMS": float(route_audit["normalized_curl_RMS"]),
                    "optimizer_cost": float(fit["result"].cost),
                    "geometry_at_boundary": boundary,
                }
            )
            settings = protocol["evaluation"]
            for family, family_images in images.groupby("source_family", sort=True):
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
                    row["eta"] = eta
                for row in assignments:
                    row["variant_id"] = variant_id
                    row["eta"] = eta
                summary["variant_id"] = variant_id
                summary["eta"] = eta
                root_rows.extend(roots)
                assignment_rows.extend(assignments)
                family_rows.append(summary)

    pd.DataFrame(system_rows).to_csv(output / protocol["outputs"]["system_scores"], index=False)
    pd.DataFrame(family_rows).to_csv(output / protocol["outputs"]["family_summary"], index=False)
    pd.DataFrame(root_rows).to_csv(output / protocol["outputs"]["global_roots"], index=False)
    pd.DataFrame(assignment_rows).to_csv(output / protocol["outputs"]["assignments"], index=False)
    pd.DataFrame(geometry_rows).to_csv(output / protocol["outputs"]["geometry"], index=False)
    pd.concat(heldout_frames, ignore_index=True).to_csv(
        output / protocol["outputs"]["heldout_predictions"], index=False
    )
    report = postprocess(protocol, config_path, output)
    print(json.dumps(json_safe({"coverage": report["coverage"], "verdict": report["verdict"]}), indent=2), flush=True)


if __name__ == "__main__":
    main()
