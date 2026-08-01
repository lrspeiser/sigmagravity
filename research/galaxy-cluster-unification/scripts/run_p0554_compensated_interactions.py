#!/usr/bin/env python3
"""Refit ordinary lens geometry for selected compensated P0554 interactions."""

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

from run_arc_apogee_boundary_refinement import morphology_scores  # noqa: E402
from run_arc_apogee_cross_domain import score_predictions, velocity_prediction  # noqa: E402
from run_arc_invariant_absolute_lensing import (  # noqa: E402
    cluster_score,
    prepare_clusters,
    prepare_galaxies,
    raw_field,
    response_for_frame,
    response_parameters,
)
from run_p0554_local_cross_domain_sensitivity import (  # noqa: E402
    A0,
    RawContext,
    Variant,
    json_safe,
    raw_contexts,
    rms,
    sha256,
)
from run_rxj2129_raw_theory_lensing import FIXED_LABELS, RawLens, near_bound, score as raw_score  # noqa: E402
from voidscreen.arc_invariants import generalized_solar_diagnostics  # noqa: E402


def build_variants(protocol: dict) -> list[Variant]:
    baseline = dict(protocol["baseline"])
    q = float(baseline.pop("universal_q"))
    variants = []
    for item in protocol["variants"]:
        spec = dict(baseline)
        spec.update(item["changes"])
        spec["candidate_id"] = item["variant_id"]
        variants.append(
            Variant(
                item["variant_id"],
                "+".join(item["changes"].keys()) if item["changes"] else "none",
                item["role"],
                item["role"],
                spec,
                q,
                json.dumps(item["changes"], sort_keys=True),
            )
        )
    return variants


def evaluate_scalar_domains(protocol, variants):
    parent_protocol = json.loads(
        (ROOT / protocol["inputs"]["parent_protocol"]).read_text(encoding="utf-8")
    )
    galaxy, properties = prepare_galaxies(parent_protocol, A0)
    clusters, _ = prepare_clusters(parent_protocol)
    rows, morphology_rows = [], []
    for variant in variants:
        local = galaxy.copy()
        response = response_for_frame(
            local,
            variant.spec,
            q=variant.q,
            a0=A0,
            radius_column="radius_adjusted_kpc",
            gbar_column="g_bar_m_s2",
        )
        local["arc_coordinate"] = (
            response["fractional_dynamical_response"] / float(variant.q)
        )
        local["velocity_arc_km_s"] = velocity_prediction(local, variant.q)
        local["candidate_id"] = variant.variant_id
        outer = local[local.split.eq("outer_holdout")]
        galaxy_score = score_predictions(outer, outer.velocity_arc_km_s.to_numpy(float))
        response = response_for_frame(
            clusters,
            variant.spec,
            q=variant.q,
            a0=A0,
            radius_column="radius_kpc",
            gbar_column="gbar_m_s2",
        )
        cluster_metrics = cluster_score(
            clusters,
            clusters.gbar_m_s2.to_numpy(float) * response["lensing_enhancement"],
        )
        solar = generalized_solar_diagnostics(
            **response_parameters(variant.spec, q=variant.q, a0=A0)
        )
        rows.append(
            {
                "variant_id": variant.variant_id,
                "role": variant.direction,
                "changes": variant.changed_value,
                "galaxy_outer_RMSE_km_s": galaxy_score["RMSE_km_s"],
                "galaxy_equal_RMSE_km_s": galaxy_score["equal_galaxy_RMSE_km_s"],
                **cluster_metrics,
                **solar,
                "all_solar_proxies_pass": bool(
                    solar["Cassini_proxy_pass"]
                    and solar["Earth_proxy_pass"]
                    and solar["Mercury_proxy_pass"]
                ),
            }
        )
        morphology_rows.extend(morphology_scores(local, properties, variant.variant_id))
    return pd.DataFrame(rows), pd.DataFrame(morphology_rows), galaxy, clusters


def evaluate_raw_context(
    context: RawContext,
    variants: list[Variant],
    protocol: dict,
    system_index: int,
):
    context.local["optimization"]["maximum_function_evaluations"] = int(
        protocol["evaluation"]["maximum_function_evaluations"]
    )
    fields = {}
    for index, variant in enumerate(variants):
        print(f"{context.label}: field {variant.variant_id} ({index + 1}/{len(variants)})", flush=True)
        fields[variant.variant_id], _ = raw_field(
            variant.spec, variant.q, context.anchors, context.local, A0
        )
    lens = RawLens(context.local, fields)
    rows, predictions, geometry = [], [], []
    starts = int(protocol["evaluation"]["optimization_starts_per_variant_system"])
    seed = int(protocol["evaluation"]["random_seed"])
    for variant_index, variant in enumerate(variants):
        print(f"{context.label}: fit {variant.variant_id}", flush=True)
        fit = lens.fit(
            variant.variant_id,
            context.training,
            starts=starts,
            seed=seed + system_index * 10000 + variant_index * 100,
            initial_override=context.geometry,
        )
        train = lens.exact_predictions(
            variant.variant_id,
            fit["result"].x,
            fit["sources"],
            context.training,
            stage="training",
        )
        held = lens.exact_predictions(
            variant.variant_id,
            fit["result"].x,
            fit["sources"],
            context.heldout,
            stage="heldout",
        )
        train_score = raw_score(train, lens.sigma, free_parameters=6)
        held_score = raw_score(held, lens.sigma)
        rows.append(
            {
                "system": context.system,
                "system_label": context.label,
                "raw_group": context.group,
                "variant_id": variant.variant_id,
                "training_images": len(context.training),
                "training_RMS_arcsec": train_score["exact_radial_RMS_arcsec"],
                "training_roots_converged": train_score["converged_roots"],
                "training_all_roots": train_score["all_roots_converged"],
                "heldout_images": len(context.heldout),
                "heldout_RMS_arcsec": held_score["exact_radial_RMS_arcsec"],
                "heldout_roots_converged": held_score["converged_roots"],
                "heldout_all_roots": held_score["all_roots_converged"],
                "optimizer_cost": float(fit["result"].cost),
            }
        )
        for frame in (train, held):
            local = frame.copy()
            local.insert(0, "system", context.system)
            local.insert(1, "system_label", context.label)
            local.insert(2, "raw_group", context.group)
            local.insert(3, "variant_id", variant.variant_id)
            predictions.append(local)
        geometry.append(
            {
                "system": context.system,
                "system_label": context.label,
                "variant_id": variant.variant_id,
                **dict(zip(FIXED_LABELS, fit["result"].x)),
                "optimizer_cost": float(fit["result"].cost),
                "geometry_at_boundary": any(
                    near_bound(variant.variant_id, fit["result"].x).values()
                ),
            }
        )
    return rows, predictions, geometry


def matched_comparison(raw: pd.DataFrame, candidate: str, labels: list[str]) -> dict:
    block = raw[raw.system_label.isin(labels)]
    parent = block[block.variant_id.eq("baseline")].set_index("system_label")
    child = block[block.variant_id.eq(candidate)].set_index("system_label")
    requested = sorted(set(parent.index) & set(child.index))
    common = [
        label
        for label in requested
        if bool(parent.loc[label, "heldout_all_roots"])
        and bool(child.loc[label, "heldout_all_roots"])
        and np.isfinite(float(parent.loc[label, "heldout_RMS_arcsec"]))
        and np.isfinite(float(child.loc[label, "heldout_RMS_arcsec"]))
    ]
    recovered = [
        label
        for label in requested
        if not bool(parent.loc[label, "heldout_all_roots"])
        and bool(child.loc[label, "heldout_all_roots"])
    ]
    lost = [
        label
        for label in requested
        if bool(parent.loc[label, "heldout_all_roots"])
        and not bool(child.loc[label, "heldout_all_roots"])
    ]
    parent_rms = rms(parent.loc[common, "heldout_RMS_arcsec"]) if common else np.nan
    child_rms = rms(child.loc[common, "heldout_RMS_arcsec"]) if common else np.nan
    complete_child = child[
        child.heldout_all_roots.astype(bool)
        & np.isfinite(pd.to_numeric(child.heldout_RMS_arcsec, errors="coerce"))
    ]
    return {
        "variant_id": candidate,
        "requested_systems": len(requested),
        "matched_complete_systems": len(common),
        "matched_labels": "+".join(common),
        "parent_matched_RMS_arcsec": parent_rms,
        "candidate_matched_RMS_arcsec": child_rms,
        "matched_improvement_fraction": np.nan
        if not common
        else 1.0 - child_rms / parent_rms,
        "recovered_systems": "+".join(recovered),
        "lost_systems": "+".join(lost),
        "candidate_complete_systems": int(child.heldout_all_roots.astype(bool).sum()),
        "candidate_total_roots": int(child.heldout_roots_converged.sum()),
        "candidate_finite_only_RMS_arcsec": np.nan
        if complete_child.empty
        else rms(complete_child.heldout_RMS_arcsec),
    }


def make_figure(scores, raw, comparisons, output):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    baseline = scores[scores.variant_id.eq("baseline")].iloc[0]
    axes[0, 0].scatter(
        scores.galaxy_outer_RMSE_km_s,
        scores.cluster_RMSE_dex,
        c=scores.all_solar_proxies_pass.map({True: "tab:blue", False: "crimson"}),
        s=70,
    )
    for row in scores.itertuples(index=False):
        axes[0, 0].annotate(row.variant_id, (row.galaxy_outer_RMSE_km_s, row.cluster_RMSE_dex), fontsize=6)
    axes[0, 0].scatter([baseline.galaxy_outer_RMSE_km_s], [baseline.cluster_RMSE_dex], marker="*", s=200, color="black")
    axes[0, 0].set(xlabel="SPARC outer RMSE (km/s)", ylabel="CLASH RMSE (dex)", title="Scalar tradeoff (red fails Solar)")

    rx = raw[raw.raw_group.eq("RXJ2129")].set_index("variant_id").loc[scores.variant_id]
    values = pd.to_numeric(rx.heldout_RMS_arcsec, errors="coerce").replace(np.inf, np.nan)
    axes[0, 1].bar(np.arange(len(rx)), values, color=np.where(rx.heldout_all_roots, "tab:blue", "crimson"))
    axes[0, 1].set(xticks=np.arange(len(rx)), xticklabels=rx.index, ylabel="held-out RMS (arcsec)", title="RX J2129 exact geometry refits")
    axes[0, 1].tick_params(axis="x", rotation=90, labelsize=6)

    four = comparisons[comparisons.scope.eq("four_cluster")].set_index("variant_id").loc[scores.variant_id]
    axes[1, 0].bar(np.arange(len(four)), four.candidate_finite_only_RMS_arcsec, color=np.where(four.candidate_complete_systems.eq(4), "tab:green", "tab:orange"))
    axes[1, 0].set(xticks=np.arange(len(four)), xticklabels=four.index, ylabel="finite-root equal-system RMS (arcsec)", title="Four-cluster refits (green = 4/4 complete)")
    axes[1, 0].tick_params(axis="x", rotation=90, labelsize=6)

    axes[1, 1].barh(scores.variant_id, scores.Mercury_precession_mas_per_century.abs(), color=np.where(scores.all_solar_proxies_pass, "tab:blue", "crimson"))
    axes[1, 1].axvline(3.1, color="black", ls="--", label="Mercury margin")
    axes[1, 1].set(xlabel="absolute supplementary precession (mas/century)", title="Solar compensation")
    axes[1, 1].legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    config_path = ROOT / "configs/p0554_compensated_interactions_protocol.json"
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
    pd.DataFrame(geometry).to_csv(output / protocol["outputs"]["geometry"], index=False)

    comparison_rows = []
    scopes = {
        "RXJ2129": ["RXJ2129"],
        "four_cluster": ["MACS0329", "MACS0429", "MACS1115", "MACS1931"],
        "historical_validation": ["MACS1115", "MACS1931"],
        "all_five": ["RXJ2129", "MACS0329", "MACS0429", "MACS1115", "MACS1931"],
    }
    for scope, labels in scopes.items():
        for variant in variants:
            row = matched_comparison(raw, variant.variant_id, labels)
            row["scope"] = scope
            comparison_rows.append(row)
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(output / protocol["outputs"]["matched_comparisons"], index=False)

    combined = scores.merge(
        comparisons[comparisons.scope.eq("all_five")][
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
    alpha_unsafe = scores[scores.variant_id.eq("alpha_low_unsafe")].iloc[0]
    alpha_comp = scores[scores.variant_id.eq("alpha_screen_compensated")].iloc[0]
    report = {
        "report_version": "P0554-COMPENSATED-INTERACTIONS-RESULTS-0.1.0",
        "status": "complete",
        "protocol": {"path": str(config_path.relative_to(ROOT)), "sha256": sha256(config_path)},
        "coverage": {
            "variants": len(variants),
            "SPARC_galaxies": int(galaxy.galaxy.nunique()),
            "CLASH_systems": int(clusters.system.nunique()),
            "raw_clusters": len(contexts),
            "raw_heldout_images": int(raw[raw.variant_id.eq("baseline")].heldout_images.sum()),
            "geometry_refit_starts": int(protocol["evaluation"]["optimization_starts_per_variant_system"]),
        },
        "baseline": combined[combined.variant_id.eq("baseline")].iloc[0].to_dict(),
        "all_variant_scores": combined.to_dict("records"),
        "complete_solar_safe_variants_ranked_by_all_five_RMS": complete.to_dict("records"),
        "Solar_compensation_check": {
            "alpha_low_Mercury_mas_per_century": float(alpha_unsafe.Mercury_precession_mas_per_century),
            "alpha_low_pass": bool(alpha_unsafe.all_solar_proxies_pass),
            "compensated_Mercury_mas_per_century": float(alpha_comp.Mercury_precession_mas_per_century),
            "compensated_pass": bool(alpha_comp.all_solar_proxies_pass),
        },
        "historical_validation_compact_halo_RMS_arcsec": float(compact),
        "matched_comparisons": comparisons.to_dict("records"),
        "claim_limits": protocol["claim_limits"],
        "verdict": {
            "any_complete_solar_safe_variant": not complete.empty,
            "no_variant_promoted": True,
        },
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    make_figure(scores, raw, comparisons, output / protocol["outputs"]["figure"])
    best = None if complete.empty else complete.iloc[0]
    summary = f"""# P0554 compensated interaction results

The 12 frozen interactions were scored on 131 SPARC galaxies, 20 CLASH systems,
five raw clusters, and Solar proxies with no gravity-parameter fit. Ordinary
lens geometry was refit with eight starts per variant and cluster.

The alpha-low control is {'Solar safe' if alpha_unsafe.all_solar_proxies_pass else 'Solar unsafe'}
at {alpha_unsafe.Mercury_precession_mas_per_century:.3f} mas/century. Adding the
predeclared sharper screen gives {alpha_comp.Mercury_precession_mas_per_century:.3f}
mas/century and is {'safe' if alpha_comp.all_solar_proxies_pass else 'still unsafe'}.

The best all-five-root, Solar-safe descriptive variant is
`{best.variant_id if best is not None else 'none'}` with equal-system raw RMS
{best.candidate_finite_only_RMS_arcsec if best is not None else float('nan'):.3f}
arcsec. No variant is promoted because the systems and interaction choices are
spent and exploratory.
"""
    (output / protocol["outputs"]["summary"]).write_text(summary, encoding="utf-8")
    print(
        json.dumps(
            json_safe(
                {
                    "Solar_compensation": report["Solar_compensation_check"],
                    "complete_solar_safe": complete[
                        [
                            "variant_id",
                            "galaxy_outer_RMSE_km_s",
                            "cluster_RMSE_dex",
                            "candidate_finite_only_RMS_arcsec",
                            "candidate_total_roots",
                        ]
                    ].to_dict("records"),
                }
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
