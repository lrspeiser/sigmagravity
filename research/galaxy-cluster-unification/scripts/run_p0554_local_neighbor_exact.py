#!/usr/bin/env python3
"""Exact-refit and global-root test of the selected local-neighbor route."""

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
from run_p0554_multifamily_multiplicity import classify_family  # noqa: E402
from run_p0554_route_localization_screen import build_directional_field  # noqa: E402
from run_p0554_route_softness_interaction import (  # noqa: E402
    build_variants,
    fit_variant,
    load_route_sources,
)
from run_p0554_subcritical_route_transfer import converged_only_rms  # noqa: E402
from run_rxj2129_raw_theory_lensing import FIXED_LABELS, near_bound  # noqa: E402


CANDIDATE = {
    "variant_id": "local_soft_200",
    "mode": "mixed_local",
    "local_mix": 1.0,
    "softening_kpc": 200.0,
    "symmetric_bend_degrees": 0.0,
    "center_mode": "light_centroid",
}


def reference_frames(protocol):
    systems = pd.read_csv(ROOT / protocol["inputs"]["reference_system_scores"])
    systems["variant_id"] = systems.variant_id.replace(
        {"eta_030": "global_centroid"}
    )
    families = pd.read_csv(ROOT / protocol["inputs"]["reference_family_summary"])
    families["variant_id"] = families.variant_id.replace(
        {"eta_030": "global_centroid"}
    )
    return systems, families


def variant_summary(families, systems, excluded):
    rows = []
    for variant_id, block in families.groupby("variant_id", sort=False):
        primary = block[~block.system_label.eq(excluded)]
        local_systems = systems[systems.variant_id.eq(variant_id)]
        rows.append(
            {
                "variant_id": variant_id,
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
                "primary_missing_multiplicity_families": int(
                    primary.multiplicity_classification.eq("missing_multiplicity").sum()
                ),
                "primary_potentially_observable_surplus_roots": int(
                    primary.potentially_observable_surplus_roots.sum()
                ),
                "heldout_roots_converged": int(local_systems.heldout_roots_converged.sum()),
                "primary_heldout_roots_converged": int(
                    local_systems[~local_systems.system_label.eq(excluded)]
                    .heldout_roots_converged.sum()
                ),
                "geometry_boundary_fits": int(
                    local_systems.geometry_at_boundary.astype(bool).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def common_family_comparisons(families, excluded):
    pivot = families.pivot_table(
        index=["system_label", "source_family"],
        columns="variant_id",
        values="assignment_RMS_arcsec",
        aggfunc="first",
    ).dropna(subset=["eta_000", "global_centroid", "local_soft_200"])
    rows = []
    for label, block in pivot.reset_index().groupby("system_label", sort=False):
        base = float(block.eta_000.mean())
        global_route = float(block.global_centroid.mean())
        local = float(block.local_soft_200.mean())
        rows.append(
            {
                "system_label": label,
                "primary_transfer_system": label != excluded,
                "common_complete_families": len(block),
                "eta_000_equal_family_RMS_arcsec": base,
                "global_centroid_equal_family_RMS_arcsec": global_route,
                "local_soft_200_equal_family_RMS_arcsec": local,
                "global_improvement_fraction_vs_eta0": 1.0 - global_route / base,
                "local_improvement_fraction_vs_eta0": 1.0 - local / base,
                "local_improvement_fraction_vs_global": 1.0 - local / global_route,
            }
        )
    comparisons = pd.DataFrame(rows)
    primary = pivot.reset_index()
    primary = primary[~primary.system_label.eq(excluded)]
    values = {
        name: float(primary[name].mean())
        for name in ("eta_000", "global_centroid", "local_soft_200")
    }
    aggregate = {
        "common_complete_families": len(primary),
        "eta_000_equal_family_RMS_arcsec": values["eta_000"],
        "global_centroid_equal_family_RMS_arcsec": values["global_centroid"],
        "local_soft_200_equal_family_RMS_arcsec": values["local_soft_200"],
        "global_improvement_fraction_vs_eta0": 1.0
        - values["global_centroid"] / values["eta_000"],
        "local_improvement_fraction_vs_eta0": 1.0
        - values["local_soft_200"] / values["eta_000"],
        "local_improvement_fraction_vs_global": 1.0
        - values["local_soft_200"] / values["global_centroid"],
    }
    return comparisons, aggregate


def postprocess(protocol, config_path, output):
    candidate_systems = pd.read_csv(output / protocol["outputs"]["candidate_system_scores"])
    candidate_families = pd.read_csv(output / protocol["outputs"]["candidate_family_summary"])
    candidate_roots = pd.read_csv(output / protocol["outputs"]["candidate_global_roots"])
    reference_systems, reference_families = reference_frames(protocol)
    systems = pd.concat([reference_systems, candidate_systems], ignore_index=True)
    families = pd.concat([reference_families, candidate_families], ignore_index=True)
    excluded = protocol["evaluation"]["selection_system_excluded_from_primary_transfer"]
    summary = variant_summary(families, systems, excluded)
    comparisons, aggregate = common_family_comparisons(families, excluded)
    indexed = summary.set_index("variant_id")
    baseline = indexed.loc["eta_000"]
    local = indexed.loc["local_soft_200"]

    primary_systems = systems[~systems.system_label.eq(excluded)]
    base_systems = primary_systems[primary_systems.variant_id.eq("eta_000")].set_index(
        "system_label"
    )
    local_systems = primary_systems[
        primary_systems.variant_id.eq("local_soft_200")
    ].set_index("system_label")
    comparable = [
        label
        for label in base_systems.index
        if int(base_systems.loc[label, "heldout_roots_converged"])
        == int(base_systems.loc[label, "heldout_images"])
        and int(local_systems.loc[label, "heldout_roots_converged"])
        == int(local_systems.loc[label, "heldout_images"])
        and np.isfinite(float(base_systems.loc[label, "heldout_RMS_arcsec"]))
        and np.isfinite(float(local_systems.loc[label, "heldout_RMS_arcsec"]))
    ]
    systems_improved = int(
        sum(
            float(local_systems.loc[label, "heldout_RMS_arcsec"])
            < float(base_systems.loc[label, "heldout_RMS_arcsec"])
            for label in comparable
        )
    )
    topology = {
        "additional_primary_observable_surplus_roots": int(
            local.primary_potentially_observable_surplus_roots
            - baseline.primary_potentially_observable_surplus_roots
        ),
        "additional_primary_missing_families": int(
            local.primary_missing_multiplicity_families
            - baseline.primary_missing_multiplicity_families
        ),
        "lost_primary_observed_seed_heldout_roots": int(
            baseline.primary_heldout_roots_converged
            - local.primary_heldout_roots_converged
        ),
    }
    gates = protocol["gates"]
    gate_results = {
        "assignment_improvement_vs_eta0": aggregate[
            "local_improvement_fraction_vs_eta0"
        ]
        >= float(gates["primary_common_family_assignment_RMS_improvement_vs_eta0_minimum"]),
        "heldout_systems_improved": systems_improved
        >= int(gates["primary_systems_heldout_RMS_improved_vs_eta0_minimum"]),
        "beats_global_centroid": aggregate["local_soft_200_equal_family_RMS_arcsec"]
        < aggregate["global_centroid_equal_family_RMS_arcsec"],
        "no_additional_observable_surplus": topology[
            "additional_primary_observable_surplus_roots"
        ]
        <= int(gates["additional_primary_observable_surplus_roots_maximum"]),
        "no_additional_missing_families": topology[
            "additional_primary_missing_families"
        ]
        <= int(gates["additional_primary_missing_families_maximum"]),
        "no_lost_heldout_roots": topology["lost_primary_observed_seed_heldout_roots"]
        <= int(gates["lost_primary_observed_seed_heldout_roots_maximum"]),
    }
    strong = all(gate_results.values())

    summary.to_csv(output / protocol["outputs"]["variant_summary"], index=False)
    comparisons.to_csv(
        output / protocol["outputs"]["paired_position_comparisons"], index=False
    )
    systems.to_csv(output / protocol["outputs"]["comparison_system_scores"], index=False)
    families.to_csv(output / protocol["outputs"]["comparison_family_summary"], index=False)
    root_changes = families.pivot_table(
        index=["system_label", "source_family"],
        columns="variant_id",
        values="global_roots",
        aggfunc="first",
    ).dropna(subset=["eta_000", "global_centroid", "local_soft_200"])
    root_changes["local_minus_eta0"] = root_changes.local_soft_200 - root_changes.eta_000
    root_changes["local_minus_global"] = (
        root_changes.local_soft_200 - root_changes.global_centroid
    )
    changed = root_changes[
        ~root_changes.local_minus_eta0.eq(0) | ~root_changes.local_minus_global.eq(0)
    ].reset_index()

    report = {
        "report_version": "P0554-LOCAL-NEIGHBOR-EXACT-RESULTS-0.1.0",
        "status": "complete",
        "protocol": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": sha256(config_path),
        },
        "coverage": {
            "candidate_geometry_fits": len(candidate_systems),
            "candidate_optimization_starts": len(candidate_systems)
            * int(protocol["evaluation"]["optimization_starts_per_cluster"]),
            "candidate_formula_family_searches": len(candidate_families),
            "candidate_accepted_global_roots": len(candidate_roots),
            "comparison_variants": 3,
            "systems": int(systems.system_label.nunique()),
            "source_families": int(protocol["evaluation"]["source_families"]),
            "published_images": int(protocol["evaluation"]["published_images"]),
        },
        "primary_common_family_position_comparison": aggregate,
        "per_system_common_family_position_comparison": comparisons.to_dict("records"),
        "primary_exact_heldout_systems_comparable": comparable,
        "primary_exact_heldout_systems_improved": systems_improved,
        "primary_topology_changes_vs_eta0": topology,
        "variant_summary": summary.to_dict("records"),
        "changed_family_root_counts": changed.to_dict("records"),
        "gate_results": gate_results,
        "verdict": {
            "strong_exact_survival": strong,
            "local_neighbor_beats_global_centroid_after_exact_refit": bool(
                gate_results["beats_global_centroid"]
            ),
            "no_formula_promoted": True,
        },
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    display = comparisons.sort_values("system_label")
    axis = np.arange(len(display))
    width = 0.36
    axes[0].bar(
        axis - width / 2,
        100.0 * display.global_improvement_fraction_vs_eta0,
        width,
        label="global centroid",
    )
    axes[0].bar(
        axis + width / 2,
        100.0 * display.local_improvement_fraction_vs_eta0,
        width,
        label="local neighbor",
    )
    axes[0].set(
        xticks=axis,
        xticklabels=display.system_label,
        ylabel="common-family improvement vs eta=0 (%)",
        title="Exact-refit position response",
    )
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].legend()
    if changed.empty:
        axes[1].text(0.5, 0.5, "No root-count differences", ha="center", va="center")
        axes[1].set_xticks([])
        axes[1].set_yticks([])
    else:
        labels = [f"{row.system_label} F{int(row.source_family)}" for row in changed.itertuples()]
        axes[1].bar(labels, changed.local_minus_eta0, label="local - eta0")
        axes[1].scatter(labels, changed.local_minus_global, color="black", label="local - global")
        axes[1].tick_params(axis="x", rotation=35)
        axes[1].legend()
    axes[1].set(ylabel="global root-count difference", title="Topology differences")
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)

    summary_text = f"""# P0554 local-neighbor exact follow-up

The screen-selected 200 kpc-softened local-neighbor route was refit from eight
starts in five clusters and searched globally across 27 source families. On
{aggregate['common_complete_families']} mutually complete primary-transfer
families it changes equal-family RMS from
{aggregate['eta_000_equal_family_RMS_arcsec']:.3f} arcsec at eta=0 and
{aggregate['global_centroid_equal_family_RMS_arcsec']:.3f} for the global route
to {aggregate['local_soft_200_equal_family_RMS_arcsec']:.3f} arcsec.

Local improvement versus eta=0:
{100.0 * aggregate['local_improvement_fraction_vs_eta0']:+.3f}%; versus the
global route: {100.0 * aggregate['local_improvement_fraction_vs_global']:+.3f}%.
It improves {systems_improved}/{len(comparable)} comparable exact held-out
systems. Strong frozen survival: **{strong}**. No formula is promoted.
"""
    (output / protocol["outputs"]["summary"]).write_text(summary_text, encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--postprocess-only", action="store_true")
    args = parser.parse_args()
    config_path = ROOT / "configs/p0554_local_neighbor_exact_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if not protocol["status"].startswith("frozen_"):
        raise RuntimeError("local-neighbor exact protocol is not frozen")
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
    route_sources, route_protocols = load_route_sources(interaction, contexts)
    system_rows, family_rows, root_rows, assignment_rows = [], [], [], []
    geometry_rows, heldout_frames = [], []
    for system_index, context in enumerate(contexts):
        context.local["optimization"]["maximum_function_evaluations"] = int(
            protocol["evaluation"]["maximum_function_evaluations"]
        )
        images = pd.concat([context.training, context.heldout], ignore_index=True)
        radial, _ = raw_field(parent.spec, parent.q, context.anchors, context.local, A0)
        baryons = baryon_field(context.anchors, context.local)
        morphology, audit = build_directional_field(
            interaction,
            route_protocols[context.label],
            context,
            route_sources[context.label],
            parent,
            radial,
            baryons,
            CANDIDATE,
        )
        adaptive = route_fraction(parent.candidate, route_sources[context.label], context.local)
        strength = 0.30 * float(adaptive["routing_fraction"] ** parent.route_power)
        lens = MorphologyLens(
            context.local,
            {parent.variant_id: radial},
            parent=parent.variant_id,
            morphology=morphology,
            fraction=strength,
        )
        print(f"{context.label}: local_soft_200 exact refit", flush=True)
        fit, training, heldout, training_score, heldout_score = fit_variant(
            lens,
            parent.variant_id,
            context,
            starts=int(protocol["evaluation"]["optimization_starts_per_cluster"]),
            seed=int(protocol["evaluation"]["random_seed"]) + system_index * 1000,
        )
        local_heldout = heldout.copy()
        local_heldout.insert(0, "variant_id", "local_soft_200")
        local_heldout.insert(0, "system_label", context.label)
        heldout_frames.append(local_heldout)
        boundary = any(near_bound(parent.variant_id, fit["result"].x).values())
        geometry_rows.append(
            {
                "system_label": context.label,
                "variant_id": "local_soft_200",
                **dict(zip(FIXED_LABELS, fit["result"].x)),
                "optimizer_cost": float(fit["result"].cost),
                "geometry_at_boundary": boundary,
            }
        )
        system_rows.append(
            {
                "system_label": context.label,
                "variant_id": "local_soft_200",
                "eta": 0.30,
                "published_images": len(images),
                "training_RMS_arcsec": training_score["exact_radial_RMS_arcsec"],
                "training_roots_converged": training_score["converged_roots"],
                "heldout_RMS_arcsec": heldout_score["exact_radial_RMS_arcsec"],
                "heldout_roots_converged": heldout_score["converged_roots"],
                "heldout_images": len(context.heldout),
                "heldout_converged_only_RMS_arcsec": converged_only_rms(local_heldout),
                "applied_angular_strength": strength,
                "route_curl_RMS": float(audit["normalized_curl_RMS"]),
                "optimizer_cost": float(fit["result"].cost),
                "geometry_at_boundary": boundary,
            }
        )
        for family, family_images in images.groupby("source_family", sort=True):
            roots, assignments, family_summary = classify_family(
                lens,
                parent,
                fit["result"].x,
                fit["sources"][int(family)],
                family_images,
                protocol["evaluation"],
                context.label,
            )
            for row in roots:
                row["variant_id"] = "local_soft_200"
            for row in assignments:
                row["variant_id"] = "local_soft_200"
            family_summary["variant_id"] = "local_soft_200"
            root_rows.extend(roots)
            assignment_rows.extend(assignments)
            family_rows.append(family_summary)

    pd.DataFrame(system_rows).to_csv(
        output / protocol["outputs"]["candidate_system_scores"], index=False
    )
    pd.DataFrame(family_rows).to_csv(
        output / protocol["outputs"]["candidate_family_summary"], index=False
    )
    pd.DataFrame(root_rows).to_csv(
        output / protocol["outputs"]["candidate_global_roots"], index=False
    )
    pd.DataFrame(assignment_rows).to_csv(
        output / protocol["outputs"]["candidate_assignments"], index=False
    )
    pd.DataFrame(geometry_rows).to_csv(
        output / protocol["outputs"]["candidate_geometry"], index=False
    )
    pd.concat(heldout_frames, ignore_index=True).to_csv(
        output / protocol["outputs"]["candidate_heldout_predictions"], index=False
    )
    report = postprocess(protocol, config_path, output)
    print(
        json.dumps(
            json_safe(
                {
                    "coverage": report["coverage"],
                    "position": report["primary_common_family_position_comparison"],
                    "topology": report["primary_topology_changes_vs_eta0"],
                    "verdict": report["verdict"],
                }
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
