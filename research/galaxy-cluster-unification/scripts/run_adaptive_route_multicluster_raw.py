#!/usr/bin/env python3
"""Replay the locked adaptive gravity-route bridge on four raw cluster lenses."""

from __future__ import annotations

import hashlib
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

from run_adaptive_route_raw_rxj2129 import baryon_field, build_route_field  # noqa: E402
from run_arc_invariant_absolute_lensing import raw_field  # noqa: E402
from run_arc_invariant_pareto_refinement import build_specs  # noqa: E402
from run_clash_stellar_morphology_response import MorphologyLens  # noqa: E402
from run_rxj2129_raw_theory_lensing import FIXED_LABELS, near_bound, score, spec_for  # noqa: E402
from run_unbounded_running_multicluster_raw import (  # noqa: E402
    aggregate_system_scores,
    load_anchors,
    load_system_images,
    predictive_split,
    system_protocol,
)
from voidscreen.adaptive_route_kernel import (  # noqa: E402
    adaptive_route_parameters,
    transformed_source_weights,
)
from voidscreen.route_template import weighted_radius  # noqa: E402


MODEL = "P0554"


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
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


@dataclass
class Context:
    system: dict
    local: dict
    training: pd.DataFrame
    heldout: pd.DataFrame
    anchors: pd.DataFrame
    members: pd.DataFrame
    parent: object
    baryons: object


def load_member_sources(
    path: Path,
    system: dict,
    local: dict,
    settings: dict,
) -> pd.DataFrame:
    columns = settings["columns"]
    catalog = pd.read_csv(path, sep=r"\s+", comment="#", names=columns)
    ra = pd.to_numeric(catalog.RA_deg, errors="coerce").to_numpy(float)
    dec = pd.to_numeric(catalog.Dec_deg, errors="coerce").to_numpy(float)
    magnitude = pd.to_numeric(catalog.magnitude, errors="coerce").to_numpy(float)
    cosine = math.cos(math.radians(float(system["center_dec_deg"])))
    x = (ra - float(system["center_ra_deg"])) * 3600.0 * cosine
    y = (dec - float(system["center_dec_deg"])) * 3600.0
    scale = float(local["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
    radius = np.hypot(x, y) * scale
    keep = (
        np.isfinite(ra)
        & np.isfinite(dec)
        & np.isfinite(magnitude)
        & (magnitude > 0.0)
        & (radius <= float(settings["aperture_kpc"]))
    )
    result = pd.DataFrame(
        {
            "source_id": catalog.member_id.astype(str).to_numpy()[keep],
            "x_arcsec": x[keep],
            "y_arcsec": y[keep],
            "radius_kpc": radius[keep],
            "magnitude": magnitude[keep],
        }
    )
    result["base_weight"] = np.power(
        10.0, -0.4 * (result.magnitude - result.magnitude.min())
    )
    result["base_weight"] /= result.base_weight.sum()
    if len(result) < int(settings["minimum_members"]):
        raise RuntimeError(f"{system['label']} has too few member sources")
    return result.sort_values(["radius_kpc", "source_id"]).reset_index(drop=True)


def adjusted_candidate(candidate: pd.Series, variant: dict | None) -> pd.Series:
    result = candidate.copy()
    if variant is None:
        return result
    parameter = variant["parameter"]
    if "value" in variant:
        result[parameter] = float(variant["value"])
    else:
        result[parameter] = float(result[parameter]) * float(variant["multiplier"])
    return result


def route_fraction(candidate: pd.Series, sources: pd.DataFrame, local: dict) -> dict:
    weights = transformed_source_weights(
        sources.base_weight.to_numpy(float), float(candidate.source_weight_power)
    )
    scale = float(local["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
    radius = np.hypot(
        sources.x_arcsec.to_numpy(float), sources.y_arcsec.to_numpy(float)
    ) * scale
    r50 = weighted_radius(radius, weights, 0.5)
    r80 = weighted_radius(radius, weights, 0.8)
    adaptive = adaptive_route_parameters(
        r50_kpc=r50,
        concentration=r50 / r80,
        source_weights=weights,
        feature=str(candidate.feature),
        base_fraction=float(candidate.base_fraction),
        extent_slope=float(candidate.extent_slope),
        base_length_kpc=float(candidate.base_length_kpc),
        length_power=float(candidate.length_power),
        base_width_kpc=float(candidate.base_width_kpc),
        width_power=float(candidate.width_power),
        gate_power=float(candidate.gate_power),
    )
    return {**adaptive, "r50_kpc": r50, "r80_kpc": r80, "concentration": r50 / r80}


def make_route_field(protocol, context, candidate, power):
    adaptive = route_fraction(candidate, context.members, context.local)
    strength = float(adaptive["routing_fraction"] ** float(power))
    field, audit = build_route_field(
        protocol,
        context.local,
        context.members,
        candidate,
        context.parent,
        context.baryons,
        contrast_cap=float(protocol["route_to_deflection_translation"]["contrast_cap"]),
        contrast_strength=strength,
        centroid_mode="light_centroid",
    )
    return field, {**audit, "fraction_power": float(power), "angular_strength": strength}


def make_lens(context: Context, field=None) -> MorphologyLens:
    return MorphologyLens(
        context.local,
        {MODEL: context.parent},
        parent=MODEL,
        morphology=field,
        fraction=0.0 if field is None else 1.0,
    )


def fit_exact(lens, training, heldout, *, starts, seed, initial=None):
    fit = lens.fit(
        MODEL,
        training,
        starts=int(starts),
        seed=int(seed),
        initial_override=initial,
    )
    training_prediction = lens.exact_predictions(
        MODEL, fit["result"].x, fit["sources"], training, stage="training"
    )
    heldout_prediction = lens.exact_predictions(
        MODEL, fit["result"].x, fit["sources"], heldout, stage="heldout"
    )
    return {
        "parameters": fit["result"].x,
        "sources": fit["sources"],
        "optimizer_cost": float(fit["result"].cost),
        "training_prediction": training_prediction,
        "heldout_prediction": heldout_prediction,
        "training_score": score(training_prediction, lens.sigma, free_parameters=6),
        "heldout_score": score(heldout_prediction, lens.sigma),
    }


def decorate_predictions(frame, context, variant):
    result = frame.copy()
    result.insert(0, "system", context.system["system"])
    result.insert(1, "system_label", context.system["label"])
    result.insert(2, "variant", variant)
    return result


def score_row(context, variant, fit):
    train = fit["training_score"]
    held = fit["heldout_score"]
    return {
        "system": context.system["system"],
        "system_label": context.system["label"],
        "variant": variant,
        "training_images": len(context.training),
        "heldout_images": len(context.heldout),
        "training_RMS_arcsec": train["exact_radial_RMS_arcsec"],
        "training_converged_roots": train["converged_roots"],
        "training_all_roots": train["all_roots_converged"],
        "heldout_RMS_arcsec": held["exact_radial_RMS_arcsec"],
        "heldout_converged_roots": held["converged_roots"],
        "heldout_all_roots": held["all_roots_converged"],
        "optimizer_cost": fit["optimizer_cost"],
    }


def score_for_aggregate(row):
    return {
        "images": int(row["heldout_images"]),
        "all_roots_converged": bool(row["heldout_all_roots"]),
        "exact_radial_RMS_arcsec": float(row["heldout_RMS_arcsec"]),
        "coordinate_chi2": float("nan"),
        "degrees_of_freedom": 0,
    }


def aggregate_rows(rows: list[dict]) -> dict:
    finite = [row for row in rows if np.isfinite(float(row["heldout_RMS_arcsec"]))]
    if not finite:
        return {
            "systems": len(rows),
            "images": int(sum(row["heldout_images"] for row in rows)),
            "all_roots_converged": False,
            "equal_system_radial_RMS_arcsec": None,
        }
    values = np.asarray([row["heldout_RMS_arcsec"] for row in finite], dtype=float)
    return {
        "systems": len(rows),
        "complete_systems": int(sum(bool(row["heldout_all_roots"]) for row in rows)),
        "images": int(sum(row["heldout_images"] for row in rows)),
        "all_roots_converged": bool(
            len(finite) == len(rows) and all(row["heldout_all_roots"] for row in finite)
        ),
        "equal_system_radial_RMS_arcsec": float(np.sqrt(np.mean(np.square(values)))),
        "median_system_radial_RMS_arcsec": float(np.median(values)),
        "RMS_scope": "finite-root systems only when complete_systems < systems",
    }


def matched_comparison(
    exact: pd.DataFrame,
    reference: str,
    candidate: str,
    *,
    labels: set[str] | None = None,
) -> dict:
    block = exact if labels is None else exact[exact.system_label.isin(labels)]
    left = block[block.variant.eq(reference)].set_index("system_label")
    right = block[block.variant.eq(candidate)].set_index("system_label")
    requested = sorted(set(left.index) & set(right.index))
    matched = [
        label
        for label in requested
        if bool(left.loc[label, "heldout_all_roots"])
        and bool(right.loc[label, "heldout_all_roots"])
        and np.isfinite(float(left.loc[label, "heldout_RMS_arcsec"]))
        and np.isfinite(float(right.loc[label, "heldout_RMS_arcsec"]))
    ]
    reference_rms = float(
        np.sqrt(np.mean(np.square(left.loc[matched, "heldout_RMS_arcsec"].to_numpy(float))))
    )
    candidate_rms = float(
        np.sqrt(np.mean(np.square(right.loc[matched, "heldout_RMS_arcsec"].to_numpy(float))))
    )
    return {
        "requested_systems": len(requested),
        "matched_complete_systems": len(matched),
        "matched_labels": matched,
        "all_requested_systems_comparable": len(matched) == len(requested),
        "reference_RMS_arcsec": reference_rms,
        "candidate_RMS_arcsec": candidate_rms,
        "fractional_improvement": 1.0 - candidate_rms / reference_rms,
    }


def build_contexts(protocol):
    raw_protocol = json.loads(
        (ROOT / protocol["inputs"]["raw_cluster_protocol"]).read_text(encoding="utf-8")
    )
    acquisition = json.loads(
        (ROOT / protocol["inputs"]["member_catalog_protocol"]).read_text(encoding="utf-8")
    )
    catalog = pd.read_csv(ROOT / protocol["inputs"]["image_catalog"])
    tian = pd.read_csv(
        ROOT / protocol["inputs"]["baryonic_profile"],
        sep=r"\s+",
        names=["system", "radius_kpc", "log_gbar", "log_gobs", "err_log_gbar", "err_log_gobs"],
    )
    parent_protocol = json.loads(
        (ROOT / protocol["inputs"]["parent_protocol"]).read_text(encoding="utf-8")
    )
    parent_scores = pd.read_csv(ROOT / protocol["inputs"]["parent_scores"])
    parent_row = parent_scores[parent_scores.candidate_id.eq(MODEL)].iloc[0]
    specs = {item["candidate_id"]: item for item in build_specs(parent_protocol)}
    acquired = {item["label"]: item for item in acquisition["systems"]}
    selected = set(protocol["systems"]["labels"])
    contexts, member_rows = [], []
    for raw_system in raw_protocol["systems"]:
        if raw_system["label"] not in selected:
            continue
        member_info = acquired[raw_system["label"]]
        system = {**raw_system, **member_info}
        local = system_protocol(raw_protocol, system)
        local["optimization"]["maximum_function_evaluations"] = int(
            protocol["fit"]["maximum_function_evaluations"]
        )
        images = load_system_images(catalog, system)
        training, heldout = predictive_split(images)
        anchors = load_anchors(tian, system["label"])
        parent, _ = raw_field(
            specs[MODEL], float(parent_row.universal_q), anchors, local, 1.2e-10
        )
        baryons = baryon_field(anchors, local)
        members = load_member_sources(
            ROOT / member_info["member_catalog"],
            system,
            local,
            protocol["member_sources"],
        )
        saved = members.copy()
        saved.insert(0, "system", system["system"])
        saved.insert(1, "system_label", system["label"])
        member_rows.append(saved)
        contexts.append(
            Context(system, local, training, heldout, anchors, members, parent, baryons)
        )
    if {context.system["label"] for context in contexts} != selected:
        raise RuntimeError("raw context labels changed")
    return contexts, pd.concat(member_rows, ignore_index=True), raw_protocol


def make_figure(exact, impacts, report, output):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    aggregate = report["exact_aggregate_all_four"]
    variants = list(aggregate)
    labels = [
        f"{label} ({aggregate[label]['complete_systems']}/{aggregate[label]['systems']})"
        for label in variants
    ]
    values = [aggregate[label]["equal_system_radial_RMS_arcsec"] for label in variants]
    colors = ["tab:orange" if "primary" in label else "tab:blue" for label in variants]
    axes[0].barh(labels, values, color=colors)
    axes[0].set(xlabel="finite-root held-out RMS (arcsec)", title="Exact refits (complete systems shown)")

    pivot = exact.pivot(index="system_label", columns="variant", values="heldout_RMS_arcsec")
    x = np.arange(len(pivot))
    axes[1].bar(x - 0.18, pivot.scalar_baseline.replace(np.inf, np.nan), 0.36, label="scalar")
    axes[1].bar(x + 0.18, pivot.A0279_primary_power_2_5, 0.36, label="route")
    axes[1].set(xticks=x, xticklabels=pivot.index, ylabel="held-out RMS (arcsec)", title="Primary by cluster (missing = failed root)")
    axes[1].legend()

    display = impacts.sort_values("heldout_impact_span_arcsec")
    axes[2].barh(display.parameter, display.heldout_impact_span_arcsec, color="tab:green")
    axes[2].set(xlabel="RMS span (arcsec)", title="One-at-a-time impact")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    config_path = ROOT / "configs/adaptive_route_multicluster_raw_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)

    contexts, members, _ = build_contexts(protocol)
    members.to_csv(output / protocol["outputs"]["members"], index=False)
    candidates = pd.read_csv(ROOT / protocol["inputs"]["adaptive_candidate_specs"])
    candidate = candidates.set_index("candidate_id").loc["A0279"]
    bridge = json.loads(
        (ROOT / protocol["inputs"]["RXJ2129_bridge_report"]).read_text(encoding="utf-8")
    )
    if not any(
        row["variant"] == "power_2.5" and row["fractional_heldout_improvement_vs_scalar"] > 0
        for row in bridge["scores"]
    ):
        raise RuntimeError("the disclosed RX J2129 post-hoc bridge result changed")

    exact_rows, predictions, geometry, audit_rows = [], [], [], []
    primary_fits, primary_fields = {}, {}
    seed = int(protocol["fit"]["random_seed"])
    for system_index, context in enumerate(contexts):
        print(f"{context.system['label']}: scalar baseline", flush=True)
        baseline_lens = make_lens(context)
        baseline = fit_exact(
            baseline_lens,
            context.training,
            context.heldout,
            starts=protocol["fit"]["baseline_starts"],
            seed=seed + 1000 * system_index,
        )
        exact_rows.append(score_row(context, "scalar_baseline", baseline))
        for frame in (baseline["training_prediction"], baseline["heldout_prediction"]):
            predictions.append(decorate_predictions(frame, context, "scalar_baseline"))
        geometry.append(
            {
                "system_label": context.system["label"],
                "variant": "scalar_baseline",
                **dict(zip(FIXED_LABELS, baseline["parameters"])),
                "geometry_at_boundary": any(near_bound(MODEL, baseline["parameters"]).values()),
            }
        )

        for variant_index, variant in enumerate(protocol["exact_refit_variants"]):
            if variant["kind"] == "scalar":
                continue
            power = float(variant["fraction_power"])
            field, audit = make_route_field(protocol, context, candidate, power)
            label = variant["label"].replace(".", "_")
            audit_rows.append({"system_label": context.system["label"], "variant": label, **audit})
            lens = make_lens(context, field)
            starts = (
                protocol["fit"]["primary_starts"]
                if np.isclose(power, protocol["primary_hypothesis"]["locked_power"])
                else protocol["fit"]["amplitude_sensitivity_starts"]
            )
            print(f"{context.system['label']}: {label}", flush=True)
            fitted = fit_exact(
                lens,
                context.training,
                context.heldout,
                starts=starts,
                seed=seed + 1000 * system_index + 100 + variant_index,
                initial=baseline["parameters"],
            )
            exact_rows.append(score_row(context, label, fitted))
            for frame in (fitted["training_prediction"], fitted["heldout_prediction"]):
                predictions.append(decorate_predictions(frame, context, label))
            geometry.append(
                {
                    "system_label": context.system["label"],
                    "variant": label,
                    **dict(zip(FIXED_LABELS, fitted["parameters"])),
                    "geometry_at_boundary": any(near_bound(MODEL, fitted["parameters"]).values()),
                }
            )
            if np.isclose(power, protocol["primary_hypothesis"]["locked_power"]):
                primary_fits[context.system["label"]] = fitted
                primary_fields[context.system["label"]] = field

    exact = pd.DataFrame(exact_rows)
    exact.to_csv(output / protocol["outputs"]["exact_scores"], index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        output / protocol["outputs"]["predictions"], index=False
    )
    pd.DataFrame(geometry).to_csv(output / protocol["outputs"]["geometry"], index=False)

    sensitivity_rows = []
    for context in contexts:
        fit = primary_fits[context.system["label"]]
        for oat in protocol["one_at_a_time_variants"]:
            modified = adjusted_candidate(candidate, oat)
            field, audit = make_route_field(
                protocol,
                context,
                modified,
                protocol["primary_hypothesis"]["locked_power"],
            )
            label = oat["label"]
            audit_rows.append({"system_label": context.system["label"], "variant": label, **audit})
            lens = make_lens(context, field)
            train_prediction = lens.exact_predictions(
                MODEL, fit["parameters"], fit["sources"], context.training, stage="training"
            )
            hold_prediction = lens.exact_predictions(
                MODEL, fit["parameters"], fit["sources"], context.heldout, stage="heldout"
            )
            train_score = score(train_prediction, lens.sigma, free_parameters=0)
            hold_score = score(hold_prediction, lens.sigma)
            sensitivity_rows.append(
                {
                    "system_label": context.system["label"],
                    "variant": label,
                    "parameter": oat["parameter"],
                    "direction": "low" if ("low" in label or "shallow" in label) else "high",
                    "training_RMS_arcsec": train_score["exact_radial_RMS_arcsec"],
                    "training_all_roots": train_score["all_roots_converged"],
                    "heldout_RMS_arcsec": hold_score["exact_radial_RMS_arcsec"],
                    "heldout_images": len(context.heldout),
                    "heldout_all_roots": hold_score["all_roots_converged"],
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(output / protocol["outputs"]["sensitivity_scores"], index=False)
    pd.DataFrame(audit_rows).to_csv(output / protocol["outputs"]["field_audits"], index=False)

    impact_rows = []
    for parameter, block in sensitivity.groupby("parameter"):
        variants = list(block.variant.unique())
        if len(variants) != 2:
            raise RuntimeError(f"{parameter} does not have exactly two OAT directions")
        first = block[block.variant.eq(variants[0])].set_index("system_label")
        second = block[block.variant.eq(variants[1])].set_index("system_label")
        common = [
            label
            for label in first.index.intersection(second.index)
            if bool(first.loc[label, "heldout_all_roots"])
            and bool(second.loc[label, "heldout_all_roots"])
            and np.isfinite(float(first.loc[label, "heldout_RMS_arcsec"]))
            and np.isfinite(float(second.loc[label, "heldout_RMS_arcsec"]))
        ]
        values = {
            variants[0]: float(
                np.sqrt(np.mean(np.square(first.loc[common, "heldout_RMS_arcsec"].to_numpy(float))))
            ),
            variants[1]: float(
                np.sqrt(np.mean(np.square(second.loc[common, "heldout_RMS_arcsec"].to_numpy(float))))
            ),
        }
        impact_rows.append(
            {
                "parameter": parameter,
                "common_complete_systems": len(common),
                "common_system_labels": "+".join(common),
                "first_variant_complete_systems": int(first.heldout_all_roots.astype(bool).sum()),
                "second_variant_complete_systems": int(second.heldout_all_roots.astype(bool).sum()),
                "heldout_impact_span_arcsec": float(max(values.values()) - min(values.values())),
                "better_OAT_variant": min(values, key=values.get),
                "better_OAT_equal_system_RMS_arcsec": float(min(values.values())),
                "worse_OAT_variant": max(values, key=values.get),
                "worse_OAT_equal_system_RMS_arcsec": float(max(values.values())),
            }
        )
    impacts = pd.DataFrame(impact_rows).sort_values(
        "heldout_impact_span_arcsec", ascending=False
    )
    impacts.to_csv(output / protocol["outputs"]["parameter_impacts"], index=False)

    aggregate_all = {
        variant: aggregate_rows(group.to_dict("records"))
        for variant, group in exact.groupby("variant")
    }
    validation_labels = set(protocol["systems"]["historical_validation_labels"])
    aggregate_validation = {
        variant: aggregate_rows(group[group.system_label.isin(validation_labels)].to_dict("records"))
        for variant, group in exact.groupby("variant")
    }
    primary_label = "A0279_primary_power_2_5"
    scalar_all = aggregate_all["scalar_baseline"]
    primary_all = aggregate_all[primary_label]
    primary_validation = aggregate_validation[primary_label]
    matched_all = matched_comparison(exact, "scalar_baseline", primary_label)
    matched_validation = matched_comparison(
        exact,
        "scalar_baseline",
        primary_label,
        labels=validation_labels,
    )
    metric = json.loads(
        (ROOT / protocol["inputs"]["metric_slip_report"]).read_text(encoding="utf-8")
    )
    compact = float(
        metric["comparators"]["compact_halo_validation"]["equal_system_radial_RMS_arcsec"]
    )
    halo_ratio = float(primary_validation["equal_system_radial_RMS_arcsec"]) / compact
    gates = protocol["gates"]
    gate_audit = {
        "all_heldout_roots_pass": bool(primary_all["all_roots_converged"]),
        "all_four_comparison_available": matched_all["all_requested_systems_comparable"],
        "matched_complete_equal_system_improvement_fraction": matched_all["fractional_improvement"],
        "equal_system_improvement_pass": bool(
            matched_all["all_requested_systems_comparable"]
            and matched_all["fractional_improvement"]
            >= float(gates["equal_system_heldout_improvement_over_scalar_min"])
        ),
        "historical_validation_comparison_available": matched_validation[
            "all_requested_systems_comparable"
        ],
        "matched_historical_validation_improvement_fraction": matched_validation[
            "fractional_improvement"
        ],
        "historical_validation_improvement_pass": bool(
            matched_validation["all_requested_systems_comparable"]
            and matched_validation["fractional_improvement"]
            >= float(gates["historical_validation_improvement_over_scalar_min"])
        ),
        "absolute_equal_system_RMS_arcsec": primary_all["equal_system_radial_RMS_arcsec"],
        "absolute_equal_system_RMS_pass": float(primary_all["equal_system_radial_RMS_arcsec"])
        <= float(gates["absolute_equal_system_heldout_RMS_arcsec_max"]),
        "validation_to_compact_halo_RMS_ratio": halo_ratio,
        "validation_to_compact_halo_pass": halo_ratio
        <= float(gates["validation_to_compact_halo_RMS_ratio_max"]),
    }
    gate_audit["all_gates_pass"] = bool(
        all(value for key, value in gate_audit.items() if key.endswith("_pass"))
    )
    input_hashes = {
        key: sha256(ROOT / value)
        for key, value in protocol["inputs"].items()
        if (ROOT / value).is_file()
    }
    report = {
        "report_version": "ADAPTIVE-ROUTE-MULTICLUSTER-RAW-RESULTS-0.1.0",
        "status": "complete",
        "protocol": {"path": str(config_path.relative_to(ROOT)), "sha256": sha256(config_path)},
        "input_hashes": input_hashes,
        "formula": protocol["primary_hypothesis"],
        "coverage": {
            "clusters": len(contexts),
            "members": int(len(members)),
            "training_images": int(sum(len(context.training) for context in contexts)),
            "heldout_images": int(sum(len(context.heldout) for context in contexts)),
        },
        "exact_aggregate_all_four": aggregate_all,
        "exact_aggregate_historical_validation": aggregate_validation,
        "matched_primary_vs_scalar_all_four": matched_all,
        "matched_primary_vs_scalar_historical_validation": matched_validation,
        "primary_per_system": exact[exact.variant.eq(primary_label)].to_dict("records"),
        "scalar_per_system": exact[exact.variant.eq("scalar_baseline")].to_dict("records"),
        "parameter_impacts": impacts.to_dict("records"),
        "comparators": {
            "compact_halo_historical_validation_RMS_arcsec": compact,
            "primary_to_compact_halo_validation_ratio": halo_ratio,
        },
        "cross_scale_controls": protocol["cross_scale_controls"],
        "gate_audit": gate_audit,
        "verdict": {
            "posthoc_RXJ2129_bridge_transfers": gate_audit["all_gates_pass"],
            "route_map_remains_worth_mechanistic_study": bool(
                primary_all["all_roots_converged"]
                and not scalar_all["all_roots_converged"]
            ),
        },
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    make_figure(
        exact,
        impacts,
        report,
        output / protocol["outputs"]["figure"],
    )
    summary = f"""# Adaptive route multi-cluster raw replay

The frozen post-hoc `p=2.5` bridge has an all-four held-out equal-system RMS of
{primary_all['equal_system_radial_RMS_arcsec']:.3f} arcsec with 4/4 complete systems.
The scalar parent has only {scalar_all['complete_systems']}/4 complete systems, so
an all-four RMS improvement is not defined. On the {matched_all['matched_complete_systems']}
matched complete systems, the change is {100.0 * matched_all['fractional_improvement']:+.2f}%.
The route recovers the missing MACS1931 held-out root, but on the matched historical
validation subset the change is only {100.0 * matched_validation['fractional_improvement']:+.3f}%.
The comparable compact-halo validation RMS is {compact:.3f} arcsec. The frozen result is
**{'PASS' if gate_audit['all_gates_pass'] else 'FAIL'}**.

The largest local formula impact was `{impacts.iloc[0].parameter}`, spanning
{impacts.iloc[0].heldout_impact_span_arcsec:.3f} arcsec across its frozen low/high
perturbations. These sensitivity rows are descriptive and do not replace the primary.
"""
    (output / protocol["outputs"]["summary"]).write_text(summary, encoding="utf-8")
    print(json.dumps(json_safe({"gate_audit": gate_audit, "impacts": impacts.to_dict("records")}), indent=2), flush=True)


if __name__ == "__main__":
    main()
