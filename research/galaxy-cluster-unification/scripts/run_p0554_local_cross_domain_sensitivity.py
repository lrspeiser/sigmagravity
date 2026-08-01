#!/usr/bin/env python3
"""Local, fixed-parameter cross-domain sensitivity scan around P0554."""

from __future__ import annotations

import hashlib
import json
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
from run_rxj2129_raw_theory_lensing import (  # noqa: E402
    FIXED_LABELS,
    RawLens,
    load_baryonic_anchors,
    load_images,
    score as raw_score,
)
from run_unbounded_running_multicluster_raw import (  # noqa: E402
    load_anchors,
    load_system_images,
    predictive_split,
    system_protocol,
)
from voidscreen.arc_invariants import generalized_solar_diagnostics  # noqa: E402


A0 = 1.2e-10


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return value


@dataclass
class Variant:
    variant_id: str
    parameter: str
    direction: str
    concept: str
    spec: dict
    q: float
    changed_value: float | str


@dataclass
class RawContext:
    label: str
    system: str
    group: str
    local: dict
    training: pd.DataFrame
    heldout: pd.DataFrame
    anchors: pd.DataFrame
    geometry: np.ndarray


def build_variants(protocol: dict) -> list[Variant]:
    baseline = dict(protocol["baseline"])
    q0 = float(baseline.pop("universal_q"))
    baseline["candidate_id"] = "baseline"
    variants = [Variant("baseline", "none", "baseline", "P0554 parent", baseline, q0, "baseline")]
    for parameter, settings in protocol["perturbations"].items():
        for direction in ("low", "high"):
            raw_value = float(settings[direction])
            spec = dict(baseline)
            q = q0
            base_value = q0 if parameter == "universal_q" else float(spec[parameter])
            value = base_value * raw_value if settings["mode"] == "multiplicative" else raw_value
            if parameter == "universal_q":
                q = value
            else:
                spec[parameter] = value
            variant_id = f"{parameter}_{direction}"
            spec["candidate_id"] = variant_id
            variants.append(
                Variant(
                    variant_id,
                    parameter,
                    direction,
                    str(settings["concept"]),
                    spec,
                    q,
                    value,
                )
            )
    return variants


def load_fixed_geometry(path: Path, *, candidate: str, variant_column: str) -> np.ndarray:
    frame = pd.read_csv(path)
    block = frame[frame[variant_column].eq(candidate)].set_index("parameter")
    return block.loc[list(FIXED_LABELS), "value"].to_numpy(float)


def raw_contexts(protocol: dict) -> list[RawContext]:
    contexts = []
    rx_protocol = json.loads((ROOT / protocol["inputs"]["RXJ2129_protocol"]).read_text())
    rx_images = load_images(rx_protocol)
    heldout_ids = set(rx_protocol["predictive_split"]["heldout"])
    rx_training = rx_images[~rx_images.image_id.isin(heldout_ids)].copy()
    rx_heldout = rx_images[rx_images.image_id.isin(heldout_ids)].copy()
    rx_geometry = load_fixed_geometry(
        ROOT / protocol["inputs"]["RXJ2129_geometry"],
        candidate="P0554",
        variant_column="candidate_id",
    )
    contexts.append(
        RawContext(
            "RXJ2129",
            "RX J2129.7+0005",
            "RXJ2129",
            rx_protocol,
            rx_training,
            rx_heldout,
            load_baryonic_anchors(rx_protocol),
            rx_geometry,
        )
    )

    multi_protocol = json.loads(
        (ROOT / protocol["inputs"]["four_cluster_protocol"]).read_text(encoding="utf-8")
    )
    catalog = pd.read_csv(ROOT / protocol["inputs"]["image_catalog"])
    tian = pd.read_csv(
        ROOT / protocol["inputs"]["baryonic_profile"],
        sep=r"\s+",
        names=["system", "radius_kpc", "log_gbar", "log_gobs", "err_log_gbar", "err_log_gobs"],
    )
    geometry = pd.read_csv(ROOT / protocol["inputs"]["four_cluster_geometry"])
    labels = {"MACS0329", "MACS0429", "MACS1115", "MACS1931"}
    for system in multi_protocol["systems"]:
        if system["label"] not in labels:
            continue
        local = system_protocol(multi_protocol, system)
        images = load_system_images(catalog, system)
        training, heldout = predictive_split(images)
        block = geometry[
            geometry.system_label.eq(system["label"])
            & geometry.variant.eq("scalar_baseline")
        ].iloc[0]
        parameters = np.asarray([float(block[label]) for label in FIXED_LABELS])
        contexts.append(
            RawContext(
                system["label"],
                system["system"],
                "four_cluster",
                local,
                training,
                heldout,
                load_anchors(tian, system["label"]),
                parameters,
            )
        )
    if len(contexts) != 5:
        raise RuntimeError("raw system coverage changed")
    return contexts


def evaluate_raw_context(context: RawContext, variants: list[Variant]):
    fields = {}
    profiles = []
    for index, variant in enumerate(variants):
        print(f"{context.label}: build {variant.variant_id} ({index + 1}/{len(variants)})", flush=True)
        field, profile = raw_field(variant.spec, variant.q, context.anchors, context.local, A0)
        fields[variant.variant_id] = field
        profile.insert(0, "system_label", context.label)
        profiles.append(profile)
    lens = RawLens(context.local, fields)
    rows, predictions = [], []
    for variant in variants:
        _, sources = lens.profiled_residuals(
            variant.variant_id, context.geometry, context.training
        )
        train = lens.exact_predictions(
            variant.variant_id,
            context.geometry,
            sources,
            context.training,
            stage="training",
        )
        held = lens.exact_predictions(
            variant.variant_id,
            context.geometry,
            sources,
            context.heldout,
            stage="heldout",
        )
        train_score = raw_score(train, lens.sigma)
        held_score = raw_score(held, lens.sigma)
        rows.append(
            {
                "system": context.system,
                "system_label": context.label,
                "raw_group": context.group,
                "variant_id": variant.variant_id,
                "parameter": variant.parameter,
                "direction": variant.direction,
                "training_images": len(context.training),
                "training_RMS_arcsec": train_score["exact_radial_RMS_arcsec"],
                "training_roots_converged": train_score["converged_roots"],
                "training_all_roots": train_score["all_roots_converged"],
                "heldout_images": len(context.heldout),
                "heldout_RMS_arcsec": held_score["exact_radial_RMS_arcsec"],
                "heldout_roots_converged": held_score["converged_roots"],
                "heldout_all_roots": held_score["all_roots_converged"],
            }
        )
        for frame in (train, held):
            local = frame.copy()
            local.insert(0, "system", context.system)
            local.insert(1, "system_label", context.label)
            local.insert(2, "raw_group", context.group)
            local.insert(3, "variant_id", variant.variant_id)
            predictions.append(local)
    return rows, predictions, profiles


def rms(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(np.square(values))))


def direction_of_better(low: float, high: float, tolerance: float = 1.0e-12) -> str:
    if not np.isfinite(low) or not np.isfinite(high):
        return "unavailable"
    if abs(low - high) <= tolerance:
        return "neutral"
    return "low" if low < high else "high"


def pair_raw_impact(raw: pd.DataFrame, low_id: str, high_id: str, group: str) -> dict:
    block = raw[raw.raw_group.eq(group)]
    low = block[block.variant_id.eq(low_id)].set_index("system_label")
    high = block[block.variant_id.eq(high_id)].set_index("system_label")
    labels = sorted(set(low.index) & set(high.index))
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
    return {
        "requested_systems": len(labels),
        "common_complete_systems": len(common),
        "common_labels": "+".join(common),
        "low_RMS_arcsec": low_rms,
        "high_RMS_arcsec": high_rms,
        "span_arcsec": abs(low_rms - high_rms) if common else np.nan,
        "better_direction": direction_of_better(low_rms, high_rms),
        "low_heldout_roots": int(low.heldout_roots_converged.sum()),
        "high_heldout_roots": int(high.heldout_roots_converged.sum()),
        "total_heldout_images": int(low.heldout_images.sum()),
    }


def aggregate_raw_variant(raw: pd.DataFrame, variant_id: str, group: str) -> dict:
    block = raw[raw.raw_group.eq(group) & raw.variant_id.eq(variant_id)]
    finite = block[
        block.heldout_all_roots.astype(bool)
        & np.isfinite(pd.to_numeric(block.heldout_RMS_arcsec, errors="coerce"))
    ]
    return {
        "systems": len(block),
        "complete_systems": len(finite),
        "heldout_images": int(block.heldout_images.sum()),
        "converged_roots": int(block.heldout_roots_converged.sum()),
        "finite_only_equal_system_RMS_arcsec": None
        if finite.empty
        else rms(finite.heldout_RMS_arcsec),
    }


def build_impacts(protocol, scores, raw):
    baseline = scores[scores.variant_id.eq("baseline")].iloc[0]
    raw_baseline = raw[raw.variant_id.eq("baseline")]
    rx_base = float(raw_baseline[raw_baseline.raw_group.eq("RXJ2129")].heldout_RMS_arcsec.iloc[0])
    multi_base_finite = raw_baseline[
        raw_baseline.raw_group.eq("four_cluster")
        & raw_baseline.heldout_all_roots.astype(bool)
    ]
    multi_base = rms(multi_base_finite.heldout_RMS_arcsec)
    threshold = float(protocol["impact_rules"]["material_fraction_of_baseline"])
    rows = []
    for parameter, settings in protocol["perturbations"].items():
        low_id, high_id = f"{parameter}_low", f"{parameter}_high"
        low = scores[scores.variant_id.eq(low_id)].iloc[0]
        high = scores[scores.variant_id.eq(high_id)].iloc[0]
        rx = pair_raw_impact(raw, low_id, high_id, "RXJ2129")
        multi = pair_raw_impact(raw, low_id, high_id, "four_cluster")
        galaxy_span = abs(float(low.galaxy_outer_RMSE_km_s) - float(high.galaxy_outer_RMSE_km_s))
        cluster_span = abs(float(low.cluster_RMSE_dex) - float(high.cluster_RMSE_dex))
        mercury_span = abs(
            float(low.Mercury_precession_mas_per_century)
            - float(high.Mercury_precession_mas_per_century)
        )
        normalized = {
            "galaxy": galaxy_span / float(baseline.galaxy_outer_RMSE_km_s),
            "derived_cluster": cluster_span / float(baseline.cluster_RMSE_dex),
            "raw_RXJ2129": 0.0 if not np.isfinite(rx["span_arcsec"]) else rx["span_arcsec"] / rx_base,
            "raw_four_cluster": 0.0 if not np.isfinite(multi["span_arcsec"]) else multi["span_arcsec"] / multi_base,
            "solar_Mercury_margin": mercury_span
            / float(protocol["impact_rules"]["solar_mercury_margin_mas_per_century"]),
        }
        directions = {
            "galaxy": direction_of_better(low.galaxy_outer_RMSE_km_s, high.galaxy_outer_RMSE_km_s),
            "derived_cluster": direction_of_better(low.cluster_RMSE_dex, high.cluster_RMSE_dex),
            "raw_RXJ2129": rx["better_direction"],
            "raw_four_cluster": multi["better_direction"],
        }
        active_directions = [
            direction
            for domain, direction in directions.items()
            if normalized[domain] >= threshold and direction in {"low", "high"}
        ]
        rows.append(
            {
                "parameter": parameter,
                "concept": settings["concept"],
                "low_value": low.changed_value,
                "high_value": high.changed_value,
                "galaxy_span_km_s": galaxy_span,
                "galaxy_normalized_span": normalized["galaxy"],
                "galaxy_better_direction": directions["galaxy"],
                "cluster_span_dex": cluster_span,
                "cluster_normalized_span": normalized["derived_cluster"],
                "cluster_better_direction": directions["derived_cluster"],
                "RXJ2129_span_arcsec": rx["span_arcsec"],
                "RXJ2129_normalized_span": normalized["raw_RXJ2129"],
                "RXJ2129_better_direction": directions["raw_RXJ2129"],
                "RXJ2129_low_roots": rx["low_heldout_roots"],
                "RXJ2129_high_roots": rx["high_heldout_roots"],
                "four_cluster_common_complete_systems": multi["common_complete_systems"],
                "four_cluster_common_labels": multi["common_labels"],
                "four_cluster_span_arcsec": multi["span_arcsec"],
                "four_cluster_normalized_span": normalized["raw_four_cluster"],
                "four_cluster_better_direction": directions["raw_four_cluster"],
                "four_cluster_low_roots": multi["low_heldout_roots"],
                "four_cluster_high_roots": multi["high_heldout_roots"],
                "solar_Mercury_span_mas_per_century": mercury_span,
                "solar_margin_fraction_span": normalized["solar_Mercury_margin"],
                "both_solar_pass": bool(low.all_solar_proxies_pass and high.all_solar_proxies_pass),
                "material_domains": int(sum(value >= threshold for value in normalized.values())),
                "material_domain_directions_agree": bool(
                    len(active_directions) >= 2 and len(set(active_directions)) == 1
                ),
                "maximum_normalized_span": max(normalized.values()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["maximum_normalized_span", "material_domains"], ascending=[False, False]
    )


def make_figure(scores, raw, impacts, output):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    domains = [
        "galaxy_normalized_span",
        "cluster_normalized_span",
        "RXJ2129_normalized_span",
        "four_cluster_normalized_span",
        "solar_margin_fraction_span",
    ]
    matrix = impacts.set_index("parameter")[domains].to_numpy(float)
    shown = np.clip(matrix, 0.0, 1.0)
    image = axes[0, 0].imshow(shown, aspect="auto", cmap="magma", vmin=0.0, vmax=max(0.25, np.nanmax(shown)))
    axes[0, 0].set(
        yticks=np.arange(len(impacts)),
        yticklabels=impacts.parameter,
        xticks=np.arange(len(domains)),
        xticklabels=["galaxy", "CLASH", "RXJ2129", "4-cluster", "Solar"],
        title="Fractional impact of low-to-high perturbation",
    )
    axes[0, 0].tick_params(axis="x", rotation=30)
    fig.colorbar(image, ax=axes[0, 0], label="span / baseline or margin")

    for parameter, block in scores.groupby("parameter"):
        if parameter == "none":
            continue
        axes[0, 1].plot(
            block.galaxy_outer_RMSE_km_s,
            block.cluster_RMSE_dex,
            "o-",
            alpha=0.7,
            label=parameter,
        )
    baseline = scores[scores.variant_id.eq("baseline")].iloc[0]
    axes[0, 1].scatter(
        [baseline.galaxy_outer_RMSE_km_s],
        [baseline.cluster_RMSE_dex],
        marker="*",
        s=180,
        color="black",
        label="P0554",
    )
    axes[0, 1].set(
        xlabel="SPARC outer RMSE (km/s)",
        ylabel="CLASH RMSE (dex)",
        title="Local galaxy--cluster tradeoff",
    )
    axes[0, 1].legend(fontsize=6, ncol=2)

    root = raw.groupby("variant_id").agg(
        roots=("heldout_roots_converged", "sum"),
        images=("heldout_images", "sum"),
    )
    ordered_ids = scores.variant_id.tolist()
    axes[1, 0].bar(np.arange(len(ordered_ids)), root.loc[ordered_ids, "roots"])
    axes[1, 0].axhline(root.images.iloc[0], color="black", ls="--")
    axes[1, 0].set(
        xticks=np.arange(len(ordered_ids)),
        xticklabels=ordered_ids,
        ylabel="converged held-out roots across five clusters",
        title="Raw-lens topology sensitivity",
    )
    axes[1, 0].tick_params(axis="x", rotation=90, labelsize=6)

    axes[1, 1].barh(impacts.parameter, impacts.maximum_normalized_span, color="tab:purple")
    axes[1, 1].axvline(0.05, color="black", ls="--", label="5% material threshold")
    axes[1, 1].set(
        xlabel="largest normalized impact across domains",
        title="Overall local impact ranking",
    )
    axes[1, 1].legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    config_path = ROOT / "configs/p0554_local_cross_domain_sensitivity_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    variants = build_variants(protocol)

    parent_protocol = json.loads(
        (ROOT / protocol["inputs"]["parent_protocol"]).read_text(encoding="utf-8")
    )
    galaxy, properties = prepare_galaxies(parent_protocol, A0)
    clusters, _ = prepare_clusters(parent_protocol)
    outer = galaxy[galaxy.split.eq("outer_holdout")].copy()
    score_rows, morphology_rows = [], []
    for variant in variants:
        local = galaxy.copy()
        unit = response_for_frame(
            local,
            variant.spec,
            q=1.0,
            a0=A0,
            radius_column="radius_adjusted_kpc",
            gbar_column="g_bar_m_s2",
        )
        local["arc_coordinate"] = unit["unit_fractional_response"]
        local["velocity_arc_km_s"] = velocity_prediction(local, variant.q)
        local["candidate_id"] = variant.variant_id
        galaxy_score = score_predictions(
            local[local.split.eq("outer_holdout")],
            local[local.split.eq("outer_holdout")].velocity_arc_km_s.to_numpy(float),
        )
        cluster_response = response_for_frame(
            clusters,
            variant.spec,
            q=variant.q,
            a0=A0,
            radius_column="radius_kpc",
            gbar_column="gbar_m_s2",
        )
        cluster_prediction = (
            clusters.gbar_m_s2.to_numpy(float) * cluster_response["lensing_enhancement"]
        )
        cluster_metrics = cluster_score(clusters, cluster_prediction)
        solar = generalized_solar_diagnostics(
            **response_parameters(variant.spec, q=variant.q, a0=A0)
        )
        score_rows.append(
            {
                "variant_id": variant.variant_id,
                "parameter": variant.parameter,
                "direction": variant.direction,
                "concept": variant.concept,
                "changed_value": variant.changed_value,
                "universal_q": variant.q,
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
    scores = pd.DataFrame(score_rows)
    morphology = pd.DataFrame(morphology_rows)
    scores.to_csv(output / protocol["outputs"]["variant_scores"], index=False)
    morphology.to_csv(output / protocol["outputs"]["galaxy_morphology"], index=False)

    raw_rows, raw_predictions, raw_profiles = [], [], []
    for context in raw_contexts(protocol):
        rows, predictions, profiles = evaluate_raw_context(context, variants)
        raw_rows.extend(rows)
        raw_predictions.extend(predictions)
        raw_profiles.extend(profiles)
    raw = pd.DataFrame(raw_rows)
    raw.to_csv(output / protocol["outputs"]["raw_scores"], index=False)
    pd.concat(raw_predictions, ignore_index=True).to_csv(
        output / protocol["outputs"]["raw_predictions"], index=False
    )

    impacts = build_impacts(protocol, scores, raw)
    impacts.to_csv(output / protocol["outputs"]["parameter_impacts"], index=False)
    baseline_raw = {
        group: aggregate_raw_variant(raw, "baseline", group)
        for group in ("RXJ2129", "four_cluster")
    }
    top_by_domain = {}
    for domain, column in {
        "galaxy": "galaxy_normalized_span",
        "derived_cluster": "cluster_normalized_span",
        "raw_RXJ2129": "RXJ2129_normalized_span",
        "raw_four_cluster": "four_cluster_normalized_span",
        "solar": "solar_margin_fraction_span",
    }.items():
        row = impacts.sort_values(column, ascending=False).iloc[0]
        top_by_domain[domain] = {
            "parameter": row.parameter,
            "normalized_span": float(row[column]),
        }
    report = {
        "report_version": "P0554-LOCAL-CROSS-DOMAIN-SENSITIVITY-RESULTS-0.1.0",
        "status": "complete",
        "protocol": {"path": str(config_path.relative_to(ROOT)), "sha256": sha256(config_path)},
        "coverage": {
            "variants": len(variants),
            "parameters": len(protocol["perturbations"]),
            "SPARC_galaxies": int(outer.galaxy.nunique()),
            "SPARC_outer_points": len(outer),
            "CLASH_systems": int(clusters.system.nunique()),
            "CLASH_points": len(clusters),
            "raw_clusters": 5,
            "raw_heldout_images": int(raw[raw.variant_id.eq("baseline")].heldout_images.sum()),
        },
        "baseline": scores[scores.variant_id.eq("baseline")].iloc[0].to_dict(),
        "baseline_raw": baseline_raw,
        "top_parameter_by_domain": top_by_domain,
        "parameter_impacts": impacts.to_dict("records"),
        "all_variants_solar_safe": bool(scores.all_solar_proxies_pass.all()),
        "parameters_with_material_same_direction_across_domains": impacts[
            impacts.material_domain_directions_agree.astype(bool)
        ].parameter.tolist(),
        "root_topology": {
            "minimum_total_heldout_roots": int(raw.groupby("variant_id").heldout_roots_converged.sum().min()),
            "maximum_total_heldout_roots": int(raw.groupby("variant_id").heldout_roots_converged.sum().max()),
            "total_images": int(raw[raw.variant_id.eq("baseline")].heldout_images.sum()),
        },
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    make_figure(scores, raw, impacts, output / protocol["outputs"]["figure"])
    top = impacts.iloc[0]
    summary = f"""# P0554 local cross-domain sensitivity

The frozen scan changed 11 parameters in low/high pairs around P0554, with no
formula or lens-geometry refit. The largest normalized local effect is
`{top.parameter}` ({top.maximum_normalized_span:.3f} of its domain baseline or
Solar margin). The most influential parameter by domain is recorded in
`report.json`.

Across five raw clusters, the 23 variants produce between
{report['root_topology']['minimum_total_heldout_roots']} and
{report['root_topology']['maximum_total_heldout_roots']} of
{report['root_topology']['total_images']} held-out roots. RMS spans compare only
systems complete in both directions. All Solar proxies
{'pass' if report['all_variants_solar_safe'] else 'do not pass'} across the grid.
"""
    (output / protocol["outputs"]["summary"]).write_text(summary, encoding="utf-8")
    print(
        json.dumps(
            json_safe(
                {
                    "top_parameter_by_domain": top_by_domain,
                    "root_topology": report["root_topology"],
                    "same_direction": report[
                        "parameters_with_material_same_direction_across_domains"
                    ],
                }
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
