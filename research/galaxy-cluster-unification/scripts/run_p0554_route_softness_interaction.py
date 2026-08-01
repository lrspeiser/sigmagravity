#!/usr/bin/env python3
"""Test small P0554 photon-law changes jointly with angular gravity routing."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_adaptive_route_multicluster_raw import (  # noqa: E402
    load_member_sources,
    route_fraction,
)
from run_adaptive_route_raw_rxj2129 import (  # noqa: E402
    baryon_field,
    build_route_field,
    load_sources as load_rx_sources,
)
from run_arc_invariant_absolute_lensing import raw_field  # noqa: E402
from run_clash_stellar_morphology_response import MorphologyLens  # noqa: E402
from run_p0554_compensated_interactions import evaluate_scalar_domains  # noqa: E402
from run_p0554_local_cross_domain_sensitivity import (  # noqa: E402
    A0,
    Variant,
    json_safe,
    raw_contexts,
    rms,
    sha256,
)
from run_rxj2129_raw_theory_lensing import (  # noqa: E402
    FIXED_LABELS,
    near_bound,
    score as raw_score,
)


@dataclass
class InteractionVariant:
    variant_id: str
    role: str
    spec: dict
    q: float
    route: bool
    route_power: float
    candidate: pd.Series


def numeric_candidate(row: pd.Series) -> pd.Series:
    result = row.copy()
    for key in (
        "base_fraction",
        "extent_slope",
        "base_length_kpc",
        "length_power",
        "base_width_kpc",
        "width_power",
        "gate_power",
        "source_weight_power",
    ):
        result[key] = float(result[key])
    return result


def changed_candidate(parent: pd.Series, changes: dict) -> pd.Series:
    result = parent.copy()
    for key, multiplier in changes.items():
        if not key.endswith("_multiplier"):
            raise ValueError(f"unsupported route change {key}")
        parameter = key[: -len("_multiplier")]
        if parameter not in result:
            raise ValueError(f"unknown route parameter {parameter}")
        result[parameter] = float(result[parameter]) * float(multiplier)
    return result


def build_variants(protocol: dict) -> list[InteractionVariant]:
    baseline = dict(protocol["baseline"])
    q = float(baseline.pop("universal_q"))
    candidates = pd.read_csv(ROOT / protocol["inputs"]["route_candidate_specs"])
    parent_id = protocol["route_parent"]["candidate_id"]
    parent = numeric_candidate(candidates[candidates.candidate_id.eq(parent_id)].iloc[0])
    variants = []
    for item in protocol["variants"]:
        spec = dict(baseline)
        spec["lensing_addition_softness"] = float(item["lensing_softness"])
        spec["candidate_id"] = item["variant_id"]
        candidate = changed_candidate(parent, item.get("route_changes", {}))
        variants.append(
            InteractionVariant(
                variant_id=item["variant_id"],
                role=item["role"],
                spec=spec,
                q=q,
                route=bool(item["route"]),
                route_power=float(item.get("route_power", protocol["route_parent"]["route_power"])),
                candidate=candidate,
            )
        )
    return variants


def scalar_variants(variants: list[InteractionVariant]) -> list[Variant]:
    return [
        Variant(
            item.variant_id,
            "route+lensing_softness" if item.route else "lensing_softness",
            item.role,
            item.role,
            item.spec,
            item.q,
            float(item.spec["lensing_addition_softness"]),
        )
        for item in variants
    ]


def load_route_sources(protocol: dict, raw):
    rx_route = json.loads(
        (ROOT / protocol["inputs"]["RXJ2129_route_protocol"]).read_text(encoding="utf-8")
    )
    rx_context = next(item for item in raw if item.label == "RXJ2129")
    sources = {"RXJ2129": load_rx_sources(rx_route, rx_context.local)}
    route_protocols = {"RXJ2129": rx_route}

    four_route = json.loads(
        (ROOT / protocol["inputs"]["four_cluster_route_protocol"]).read_text(
            encoding="utf-8"
        )
    )
    four_raw = json.loads(
        (ROOT / four_route["inputs"]["raw_cluster_protocol"]).read_text(
            encoding="utf-8"
        )
    )
    acquisition = json.loads(
        (ROOT / four_route["inputs"]["member_catalog_protocol"]).read_text(
            encoding="utf-8"
        )
    )
    acquired = {item["label"]: item for item in acquisition["systems"]}
    systems = {item["label"]: item for item in four_raw["systems"]}
    raw_by_label = {item.label: item for item in raw}
    for label in four_route["systems"]["labels"]:
        member_info = acquired[label]
        system = {**systems[label], **member_info}
        local = raw_by_label[label].local
        sources[label] = load_member_sources(
            ROOT / member_info["member_catalog"],
            system,
            local,
            four_route["member_sources"],
        )
        route_protocols[label] = four_route
    expected = {item.label for item in raw}
    if set(sources) != expected:
        raise RuntimeError("route-source coverage changed")
    return sources, route_protocols


def fit_variant(lens, model, context, *, starts: int, seed: int):
    fit = lens.fit(
        model,
        context.training,
        starts=int(starts),
        seed=int(seed),
        initial_override=context.geometry,
    )
    training = lens.exact_predictions(
        model, fit["result"].x, fit["sources"], context.training, stage="training"
    )
    heldout = lens.exact_predictions(
        model, fit["result"].x, fit["sources"], context.heldout, stage="heldout"
    )
    return fit, training, heldout, raw_score(
        training, lens.sigma, free_parameters=6
    ), raw_score(heldout, lens.sigma)


def evaluate_raw(protocol: dict, variants: list[InteractionVariant]):
    contexts = raw_contexts(protocol)
    sources, route_protocols = load_route_sources(protocol, contexts)
    rows, predictions, geometry, audits = [], [], [], []
    starts = int(protocol["evaluation"]["optimization_starts_per_variant_system"])
    seed = int(protocol["evaluation"]["random_seed"])
    maximum = int(protocol["evaluation"]["maximum_function_evaluations"])
    for system_index, context in enumerate(contexts):
        context.local["optimization"]["maximum_function_evaluations"] = maximum
        baryons = baryon_field(context.anchors, context.local)
        radial_cache = {}
        route_cache = {}
        for variant_index, variant in enumerate(variants):
            print(
                f"{context.label}: {variant.variant_id} "
                f"({variant_index + 1}/{len(variants)})",
                flush=True,
            )
            radial_key = float(variant.spec["lensing_addition_softness"])
            if radial_key not in radial_cache:
                radial_cache[radial_key], _ = raw_field(
                    variant.spec, variant.q, context.anchors, context.local, A0
                )
            radial = radial_cache[radial_key]
            angular = None
            angular_strength = 0.0
            if variant.route:
                adaptive = route_fraction(
                    variant.candidate, sources[context.label], context.local
                )
                angular_strength = float(
                    adaptive["routing_fraction"] ** variant.route_power
                )
                candidate_key = tuple(
                    str(variant.candidate[key])
                    for key in (
                        "feature",
                        "base_fraction",
                        "extent_slope",
                        "base_length_kpc",
                        "length_power",
                        "base_width_kpc",
                        "width_power",
                        "gate_power",
                        "source_weight_power",
                    )
                )
                route_key = (radial_key, candidate_key)
                if route_key not in route_cache:
                    route_cache[route_key] = build_route_field(
                        route_protocols[context.label],
                        context.local,
                        sources[context.label],
                        variant.candidate,
                        radial,
                        baryons,
                        contrast_cap=float(protocol["route_parent"]["contrast_cap"]),
                        contrast_strength=1.0,
                        centroid_mode=str(protocol["route_parent"]["centroid_mode"]),
                    )
                angular, cached_audit = route_cache[route_key]
                audit = dict(cached_audit)
                audit.pop("endpoints", None)
                audits.append(
                    {
                        "system_label": context.label,
                        "variant_id": variant.variant_id,
                        "route_power": variant.route_power,
                        "applied_angular_strength": angular_strength,
                        **audit,
                    }
                )
            lens = MorphologyLens(
                context.local,
                {variant.variant_id: radial},
                parent=variant.variant_id,
                morphology=angular,
                fraction=angular_strength,
            )
            fit, training, heldout, training_score, heldout_score = fit_variant(
                lens,
                variant.variant_id,
                context,
                starts=starts,
                seed=seed + system_index * 10000 + variant_index * 100,
            )
            rows.append(
                {
                    "system": context.system,
                    "system_label": context.label,
                    "raw_group": context.group,
                    "variant_id": variant.variant_id,
                    "training_images": len(context.training),
                    "training_RMS_arcsec": training_score["exact_radial_RMS_arcsec"],
                    "training_roots_converged": training_score["converged_roots"],
                    "training_all_roots": training_score["all_roots_converged"],
                    "heldout_images": len(context.heldout),
                    "heldout_RMS_arcsec": heldout_score["exact_radial_RMS_arcsec"],
                    "heldout_roots_converged": heldout_score["converged_roots"],
                    "heldout_all_roots": heldout_score["all_roots_converged"],
                    "optimizer_cost": float(fit["result"].cost),
                    "angular_strength": angular_strength,
                }
            )
            for frame in (training, heldout):
                local = frame.copy()
                local.insert(0, "system", context.system)
                local.insert(1, "system_label", context.label)
                local.insert(2, "variant_id", variant.variant_id)
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
    return (
        pd.DataFrame(rows),
        pd.concat(predictions, ignore_index=True),
        pd.DataFrame(geometry),
        pd.DataFrame(audits),
    )


def compare_variants(
    raw: pd.DataFrame, reference_id: str, candidate_id: str, labels: list[str]
) -> dict:
    block = raw[raw.system_label.isin(labels)]
    reference = block[block.variant_id.eq(reference_id)].set_index("system_label")
    candidate = block[block.variant_id.eq(candidate_id)].set_index("system_label")
    requested = sorted(set(reference.index) & set(candidate.index))
    common = [
        label
        for label in requested
        if bool(reference.loc[label, "heldout_all_roots"])
        and bool(candidate.loc[label, "heldout_all_roots"])
        and np.isfinite(float(reference.loc[label, "heldout_RMS_arcsec"]))
        and np.isfinite(float(candidate.loc[label, "heldout_RMS_arcsec"]))
    ]
    recovered = [
        label
        for label in requested
        if not bool(reference.loc[label, "heldout_all_roots"])
        and bool(candidate.loc[label, "heldout_all_roots"])
    ]
    lost = [
        label
        for label in requested
        if bool(reference.loc[label, "heldout_all_roots"])
        and not bool(candidate.loc[label, "heldout_all_roots"])
    ]
    reference_rms = rms(reference.loc[common, "heldout_RMS_arcsec"]) if common else np.nan
    candidate_rms = rms(candidate.loc[common, "heldout_RMS_arcsec"]) if common else np.nan
    finite_candidate = candidate[
        candidate.heldout_all_roots.astype(bool)
        & np.isfinite(pd.to_numeric(candidate.heldout_RMS_arcsec, errors="coerce"))
    ]
    return {
        "reference_id": reference_id,
        "variant_id": candidate_id,
        "requested_systems": len(requested),
        "matched_complete_systems": len(common),
        "matched_labels": "+".join(common),
        "reference_matched_RMS_arcsec": reference_rms,
        "candidate_matched_RMS_arcsec": candidate_rms,
        "matched_improvement_fraction": np.nan
        if not common
        else 1.0 - candidate_rms / reference_rms,
        "recovered_systems": "+".join(recovered),
        "lost_systems": "+".join(lost),
        "candidate_complete_systems": int(candidate.heldout_all_roots.astype(bool).sum()),
        "candidate_total_roots": int(candidate.heldout_roots_converged.sum()),
        "candidate_complete_RMS_arcsec": np.nan
        if finite_candidate.empty
        else rms(finite_candidate.heldout_RMS_arcsec),
    }


def comparison_table(raw: pd.DataFrame, variants, scopes: dict) -> pd.DataFrame:
    rows = []
    for scope, labels in scopes.items():
        for variant in variants:
            row = compare_variants(raw, "baseline", variant.variant_id, labels)
            row["scope"] = scope
            rows.append(row)
    return pd.DataFrame(rows)


def pair_impacts(protocol: dict, raw: pd.DataFrame, scalar: pd.DataFrame) -> pd.DataFrame:
    scalar = scalar.set_index("variant_id")
    rows = []
    labels = protocol["comparison_scopes"]["all_five"]
    for pair in protocol["impact_pairs"]:
        low_id, high_id = pair["low"], pair["high"]
        low = raw[raw.variant_id.eq(low_id)].set_index("system_label")
        high = raw[raw.variant_id.eq(high_id)].set_index("system_label")
        common = [
            label
            for label in labels
            if bool(low.loc[label, "heldout_all_roots"])
            and bool(high.loc[label, "heldout_all_roots"])
            and np.isfinite(float(low.loc[label, "heldout_RMS_arcsec"]))
            and np.isfinite(float(high.loc[label, "heldout_RMS_arcsec"]))
        ]
        low_rms = rms(low.loc[common, "heldout_RMS_arcsec"]) if common else np.nan
        high_rms = rms(high.loc[common, "heldout_RMS_arcsec"]) if common else np.nan
        low_value, high_value = float(pair["low_value"]), float(pair["high_value"])
        log_step = abs(math.log(high_value / low_value))
        raw_elasticity = (
            abs(math.log(high_rms / low_rms)) / log_step
            if common and low_rms > 0.0 and high_rms > 0.0 and log_step > 0.0
            else np.nan
        )
        galaxy_low = float(scalar.loc[low_id, "galaxy_outer_RMSE_km_s"])
        galaxy_high = float(scalar.loc[high_id, "galaxy_outer_RMSE_km_s"])
        cluster_low = float(scalar.loc[low_id, "cluster_RMSE_dex"])
        cluster_high = float(scalar.loc[high_id, "cluster_RMSE_dex"])
        rows.append(
            {
                "parameter": pair["parameter"],
                "low_variant": low_id,
                "high_variant": high_id,
                "low_value": low_value,
                "high_value": high_value,
                "common_complete_systems": len(common),
                "common_labels": "+".join(common),
                "low_equal_system_RMS_arcsec": low_rms,
                "high_equal_system_RMS_arcsec": high_rms,
                "raw_RMS_span_arcsec": abs(high_rms - low_rms)
                if common
                else np.nan,
                "raw_log_elasticity": raw_elasticity,
                "raw_preferred_direction": "low" if low_rms < high_rms else "high",
                "low_complete_systems": int(low.heldout_all_roots.astype(bool).sum()),
                "high_complete_systems": int(high.heldout_all_roots.astype(bool).sum()),
                "root_count_span": abs(
                    int(low.heldout_roots_converged.sum())
                    - int(high.heldout_roots_converged.sum())
                ),
                "galaxy_RMSE_span_km_s": abs(galaxy_high - galaxy_low),
                "CLASH_RMSE_span_dex": abs(cluster_high - cluster_low),
                "Mercury_span_mas_per_century": abs(
                    float(scalar.loc[high_id, "Mercury_precession_mas_per_century"])
                    - float(scalar.loc[low_id, "Mercury_precession_mas_per_century"])
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["raw_log_elasticity", "raw_RMS_span_arcsec"], ascending=False
    )


def four_way_interaction(raw: pd.DataFrame) -> dict:
    ids = ["baseline", "lensing_softness_098", "route_parent", "combined_parent"]
    blocks = {
        variant: raw[raw.variant_id.eq(variant)].set_index("system_label")
        for variant in ids
    }
    common = [
        label
        for label in blocks["baseline"].index
        if all(
            bool(blocks[variant].loc[label, "heldout_all_roots"])
            and np.isfinite(float(blocks[variant].loc[label, "heldout_RMS_arcsec"]))
            for variant in ids
        )
    ]
    values = {
        variant: rms(blocks[variant].loc[common, "heldout_RMS_arcsec"])
        if common
        else np.nan
        for variant in ids
    }
    interaction = (
        values["combined_parent"]
        - values["lensing_softness_098"]
        - values["route_parent"]
        + values["baseline"]
        if common
        else np.nan
    )
    return {
        "common_complete_systems": len(common),
        "common_labels": common,
        "equal_system_RMS_arcsec": values,
        "RMS_interaction_arcsec": interaction,
        "interpretation": "zero means additive at the aggregate-RMS level; this is a descriptive interaction diagnostic, not a physical superposition theorem",
    }


def make_figure(scalar, raw, impacts, output: Path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    axes[0, 0].scatter(
        scalar.galaxy_outer_RMSE_km_s,
        scalar.cluster_RMSE_dex,
        c=np.where(scalar.all_solar_proxies_pass, "tab:blue", "crimson"),
        s=55,
    )
    for row in scalar.itertuples(index=False):
        if row.variant_id in {
            "baseline",
            "lensing_softness_098",
            "route_parent",
            "combined_parent",
        }:
            offsets = {
                "baseline": (5, -15),
                "route_parent": (-72, -15),
                "lensing_softness_098": (5, -15),
                "combined_parent": (-88, -12),
            }
            axes[0, 0].annotate(
                row.variant_id,
                (row.galaxy_outer_RMSE_km_s, row.cluster_RMSE_dex),
                xytext=offsets[row.variant_id],
                textcoords="offset points",
                fontsize=7,
            )
    axes[0, 0].set(
        xlabel="SPARC outer RMSE (km/s)",
        ylabel="CLASH RMSE (dex)",
        title="Radial cross-domain controls",
    )

    total_roots = raw.groupby("variant_id").heldout_roots_converged.sum()
    axes[0, 1].bar(np.arange(len(total_roots)), total_roots)
    axes[0, 1].axhline(18, color="black", ls="--")
    axes[0, 1].set(
        xticks=np.arange(len(total_roots)),
        xticklabels=total_roots.index,
        ylabel="held-out roots",
        title="Exact topology across five clusters",
    )
    axes[0, 1].tick_params(axis="x", rotation=90, labelsize=6)

    display = impacts.sort_values("raw_log_elasticity")
    axes[1, 0].barh(display.parameter, display.raw_log_elasticity)
    axes[1, 0].set(
        xlabel="absolute raw-lensing log elasticity",
        title="Impact per fractional parameter change",
    )

    rx = raw[raw.raw_group.eq("RXJ2129")].set_index("variant_id")
    chosen = [
        item
        for item in (
            "baseline",
            "lensing_softness_098",
            "route_parent",
            "combined_parent",
            "combined_lens_097",
            "combined_lens_099",
            "combined_power_240",
            "combined_power_260",
        )
        if item in rx.index
    ]
    values = pd.to_numeric(rx.loc[chosen, "heldout_RMS_arcsec"], errors="coerce")
    axes[1, 1].bar(
        np.arange(len(chosen)),
        values,
        color=np.where(rx.loc[chosen, "heldout_all_roots"], "tab:blue", "crimson"),
    )
    axes[1, 1].set(
        xticks=np.arange(len(chosen)),
        xticklabels=chosen,
        ylabel="held-out RMS (arcsec)",
        title="RX J2129 response interaction",
    )
    axes[1, 1].tick_params(axis="x", rotation=90, labelsize=7)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    config_path = ROOT / "configs" / "p0554_route_softness_interaction_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if not protocol["status"].startswith("frozen_"):
        raise RuntimeError("interaction protocol is not frozen")
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    variants = build_variants(protocol)

    scalar, _, galaxy, clusters = evaluate_scalar_domains(
        protocol, scalar_variants(variants)
    )
    scalar.to_csv(output / protocol["outputs"]["variant_scores"], index=False)

    raw, predictions, geometry, audits = evaluate_raw(protocol, variants)
    raw.to_csv(output / protocol["outputs"]["raw_scores"], index=False)
    predictions.to_csv(output / protocol["outputs"]["raw_predictions"], index=False)
    geometry.to_csv(output / protocol["outputs"]["geometry"], index=False)
    audits.to_csv(output / protocol["outputs"]["field_audits"], index=False)

    comparisons = comparison_table(raw, variants, protocol["comparison_scopes"])
    comparisons.to_csv(output / protocol["outputs"]["matched_comparisons"], index=False)
    impacts = pair_impacts(protocol, raw, scalar)
    impacts.to_csv(output / protocol["outputs"]["parameter_impacts"], index=False)

    all_five = comparisons[comparisons.scope.eq("all_five")].merge(
        scalar[
            [
                "variant_id",
                "galaxy_outer_RMSE_km_s",
                "cluster_RMSE_dex",
                "all_solar_proxies_pass",
            ]
        ],
        on="variant_id",
        validate="one_to_one",
    )
    complete = all_five[
        all_five.candidate_complete_systems.eq(5)
        & all_five.all_solar_proxies_pass.astype(bool)
    ].sort_values("candidate_complete_RMS_arcsec")
    compact = json.loads(
        (ROOT / protocol["inputs"]["compact_halo_report"]).read_text(encoding="utf-8")
    )["comparators"]["compact_halo_validation"]["equal_system_radial_RMS_arcsec"]
    interaction = four_way_interaction(raw)
    report = {
        "report_version": "P0554-ROUTE-SOFTNESS-INTERACTION-RESULTS-0.1.0",
        "status": "complete",
        "protocol": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": sha256(config_path),
        },
        "coverage": {
            "variants": len(variants),
            "SPARC_galaxies": int(galaxy.galaxy.nunique()),
            "SPARC_outer_points": int((galaxy.split == "outer_holdout").sum()),
            "CLASH_systems": int(clusters.system.nunique()),
            "CLASH_points": len(clusters),
            "raw_clusters": int(raw.system_label.nunique()),
            "raw_heldout_images": int(
                raw[raw.variant_id.eq("baseline")].heldout_images.sum()
            ),
            "exact_geometry_fits": len(geometry),
            "route_fields": len(audits),
        },
        "baseline": all_five[all_five.variant_id.eq("baseline")].iloc[0].to_dict(),
        "key_variants": all_five[
            all_five.variant_id.isin(
                [
                    "baseline",
                    "lensing_softness_098",
                    "route_parent",
                    "combined_parent",
                ]
            )
        ].to_dict("records"),
        "complete_solar_safe_ranked": complete.to_dict("records"),
        "parameter_impacts": impacts.to_dict("records"),
        "radial_route_interaction": interaction,
        "geometry_boundary_fits": int(geometry.geometry_at_boundary.astype(bool).sum()),
        "maximum_route_curl_RMS": float(audits.normalized_curl_RMS.max()),
        "maximum_annular_convergence_error": float(
            audits.maximum_annular_convergence_mean_fraction.max()
        ),
        "historical_validation_compact_halo_RMS_arcsec": float(compact),
        "claim_limits": protocol["claim_limits"],
        "verdict": {
            "any_complete_solar_safe_formula": not complete.empty,
            "combined_parent_complete": bool(
                all_five[all_five.variant_id.eq("combined_parent")]
                .candidate_complete_systems.eq(5)
                .iloc[0]
            ),
            "no_formula_promoted": True,
        },
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    make_figure(scalar, raw, impacts, output / protocol["outputs"]["figure"])

    top = impacts.iloc[0]
    combined = all_five[all_five.variant_id.eq("combined_parent")].iloc[0]
    summary = f"""# P0554 route/softness interaction

The frozen experiment tested {len(variants)} formulas on 131 SPARC galaxies,
20 CLASH systems, five raw cluster lenses, and Solar proxies. The combined
parent has {int(combined.candidate_total_roots)}/18 held-out roots and
{int(combined.candidate_complete_systems)}/5 complete systems. Its matched
all-five change versus scalar P0554 is
{100.0 * float(combined.matched_improvement_fraction):+.3f}% on systems where
both are complete.

The largest geometry-refit elasticity is `{top.parameter}` at
{float(top.raw_log_elasticity):.4f}; its low/high raw RMS span is
{float(top.raw_RMS_span_arcsec):.4f} arcsec across
{int(top.common_complete_systems)} mutually complete systems. No formula is
promoted because every component and raw system is spent exploratory evidence.
"""
    (output / protocol["outputs"]["summary"]).write_text(summary, encoding="utf-8")
    print(
        json.dumps(
            json_safe(
                {
                    "coverage": report["coverage"],
                    "key_variants": report["key_variants"],
                    "top_impacts": impacts.head(7).to_dict("records"),
                    "verdict": report["verdict"],
                }
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
