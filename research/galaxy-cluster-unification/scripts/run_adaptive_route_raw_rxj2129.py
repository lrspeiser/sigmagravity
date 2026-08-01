#!/usr/bin/env python3
"""Translate the locked adaptive route kernel into RX J2129 raw deflections."""

from __future__ import annotations

import hashlib
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

from reconstruct_rxj2129_baryons import _read_molino_catalog  # noqa: E402
from run_arc_invariant_absolute_lensing import raw_field  # noqa: E402
from run_arc_invariant_pareto_refinement import build_specs  # noqa: E402
from run_clash_stellar_morphology_response import MorphologyLens  # noqa: E402
from run_rxj2129_member_geometry import split_images  # noqa: E402
from run_rxj2129_raw_theory_lensing import (  # noqa: E402
    FIXED_LABELS,
    load_baryonic_anchors,
    load_images,
    near_bound,
    score,
)
from voidscreen.adaptive_route_kernel import (  # noqa: E402
    adaptive_route_parameters,
    transformed_source_weights,
)
from voidscreen.raw_lensing import (  # noqa: E402
    RadialDeflectionField,
    loglog_interpolate_with_tails,
    spherical_deflection_radians,
)
from voidscreen.route_template import conservative_route_template, weighted_radius  # noqa: E402
from voidscreen.stellar_morphology_lensing import (  # noqa: E402
    build_stellar_morphology_deflection_field,
)


MODEL = "locked_universal_candidate"


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


def load_sources(protocol: dict, raw_protocol: dict) -> pd.DataFrame:
    settings = protocol["source_catalog"]
    catalog = _read_molino_catalog(ROOT / protocol["raw_inputs"]["molino_catalog"])
    geometry = raw_protocol["cosmology_and_coordinates"]
    ra = pd.to_numeric(catalog.RA, errors="coerce").to_numpy(float)
    dec = pd.to_numeric(catalog.Dec, errors="coerce").to_numpy(float)
    cosine = np.cos(np.deg2rad(float(geometry["center_dec_deg"])))
    x = (ra - float(geometry["center_ra_deg"])) * 3600.0 * cosine
    y = (dec - float(geometry["center_dec_deg"])) * 3600.0
    scale = float(geometry["angular_scale_kpc_per_arcsec"])
    radius_kpc = np.hypot(x, y) * scale
    point = pd.to_numeric(catalog.PointS, errors="coerce").to_numpy(float)
    magnitude = pd.to_numeric(catalog.F160W_WFC3_PHOTOZ, errors="coerce").to_numpy(float)
    low = pd.to_numeric(catalog.zb_Min_1, errors="coerce").to_numpy(float)
    high = pd.to_numeric(catalog.zb_Max_1, errors="coerce").to_numpy(float)
    odds = pd.to_numeric(catalog.Odds_1, errors="coerce").to_numpy(float)
    z_lens = float(geometry["lens_redshift"])
    selected = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(point)
        & (point < 0.8)
        & np.isfinite(magnitude)
        & (magnitude > 0.0)
        & (magnitude < 90.0)
        & np.isfinite(low)
        & np.isfinite(high)
        & (low <= z_lens)
        & (high >= z_lens)
        & np.isfinite(odds)
        & (odds >= 0.5)
        & (radius_kpc <= float(settings["aperture_kpc"]))
    )
    result = pd.DataFrame(
        {
            "source_id": catalog.CLASHID.astype(str).to_numpy()[selected],
            "x_arcsec": x[selected],
            "y_arcsec": y[selected],
            "radius_kpc": radius_kpc[selected],
            "F160W_magnitude": magnitude[selected],
            "photoz_low": low[selected],
            "photoz_high": high[selected],
            "photoz_odds": odds[selected],
        }
    )
    result["base_weight"] = np.power(10.0, -0.4 * (result.F160W_magnitude - result.F160W_magnitude.min()))
    result["base_weight"] /= result.base_weight.sum()
    if len(result) < 10:
        raise RuntimeError("too few hard photometric members for route translation")
    return result.sort_values("radius_kpc").reset_index(drop=True)


def baryon_field(anchors: pd.DataFrame, raw_protocol: dict) -> RadialDeflectionField:
    radius = np.geomspace(0.1, 1.0e6, 4096)
    gbar = loglog_interpolate_with_tails(
        radius,
        anchors.radius_kpc.to_numpy(float),
        np.power(10.0, anchors.log_gbar.to_numpy(float)),
        outer_slope=-2.0,
    )

    def lookup(target):
        return np.exp(np.interp(np.log(target), np.log(radius), np.log(gbar)))

    impact = np.geomspace(0.05, 500.0, 700)
    scale = float(raw_protocol["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
    physical = spherical_deflection_radians(
        impact * scale,
        lookup,
        maximum_radius_kpc=1.0e6,
        integration_points=800,
    )
    return RadialDeflectionField(impact, physical)


def parent_initial(parameters: pd.DataFrame, parent: str) -> np.ndarray:
    block = parameters[parameters.candidate_id.eq(parent)].set_index("parameter")
    return np.asarray([float(block.loc[label, "value"]) for label in FIXED_LABELS])


def build_route_field(
    protocol: dict,
    raw_protocol: dict,
    sources: pd.DataFrame,
    candidate: pd.Series,
    parent: RadialDeflectionField,
    baryons: RadialDeflectionField,
    *,
    contrast_cap: float,
    contrast_strength: float = 1.0,
    centroid_mode: str,
    randomized_xy: np.ndarray | None = None,
):
    scale = float(raw_protocol["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
    xy = (
        sources[["x_arcsec", "y_arcsec"]].to_numpy(float)
        if randomized_xy is None
        else np.asarray(randomized_xy, dtype=float)
    )
    weights = transformed_source_weights(
        sources.base_weight.to_numpy(float), float(candidate.source_weight_power)
    )
    radius_kpc = np.hypot(xy[:, 0], xy[:, 1]) * scale
    r50 = weighted_radius(radius_kpc, weights, 0.5)
    r80 = weighted_radius(radius_kpc, weights, 0.8)
    concentration = r50 / r80
    adaptive = adaptive_route_parameters(
        r50_kpc=r50,
        concentration=concentration,
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
    translation = protocol["route_to_deflection_translation"]
    axis = np.arange(-255.5, 256.0, float(translation["grid_spacing_arcsec"]))
    center = np.zeros(2) if centroid_mode == "cluster_origin" else None
    route_map, route_audit = conservative_route_template(
        axis,
        xy,
        weights,
        routing_fraction=adaptive["routing_fraction"],
        return_scale=adaptive["return_scale_kpc"] / scale,
        radius_exponent=float(translation["source_radius_exponent"]),
        reference_radius=float(translation["source_reference_radius_kpc"]) / scale,
        smoothing=adaptive["width_kpc"] / scale,
        center=center,
    )

    def carrier_alpha(radius_arcsec):
        return parent.reduced_alpha_arcsec(radius_arcsec, 1.0) - baryons.reduced_alpha_arcsec(radius_arcsec, 1.0)

    field = build_stellar_morphology_deflection_field(
        axis,
        route_map,
        carrier_alpha,
        contrast_cap=float(contrast_cap),
        contrast_strength=float(contrast_strength),
        annulus_width_arcsec=float(translation["annulus_width_arcsec"]),
        taper_inner_arcsec=float(translation["taper_inner_arcsec"]),
        support_radius_arcsec=float(translation["support_radius_arcsec"]),
        radial_samples=2048,
        circular_radii=512,
        circular_azimuths=720,
    )
    return field, {
        **adaptive,
        "r50_kpc": r50,
        "r80_kpc": r80,
        "concentration_r50_over_r80": concentration,
        "route_map_normalization_error": route_audit["normalization_error"],
        "route_centroid_x_arcsec": float(route_audit["centroid"][0]),
        "route_centroid_y_arcsec": float(route_audit["centroid"][1]),
        "endpoints": route_audit["endpoints"],
        **field.audit,
    }


def exact_fit(lens, training, heldout, *, initial, starts, seed):
    fit = lens.fit(MODEL, training, starts=starts, seed=seed, initial_override=initial)
    training_prediction = lens.exact_predictions(
        MODEL, fit["result"].x, fit["sources"], training, stage="training"
    )
    heldout_prediction = lens.exact_predictions(
        MODEL, fit["result"].x, fit["sources"], heldout, stage="heldout"
    )
    return {
        "parameters": fit["result"].x,
        "optimizer_cost": float(fit["result"].cost),
        "training_prediction": training_prediction,
        "heldout_prediction": heldout_prediction,
        "training_score": score(training_prediction, lens.sigma, free_parameters=20),
        "heldout_score": score(heldout_prediction, lens.sigma),
    }


def make_figure(scores, predictions, randomizations, sources, endpoints, output):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    ax = axes[0, 0]
    display = scores.sort_values("heldout_RMS_arcsec", na_position="last")
    ax.barh(display.variant, display.heldout_RMS_arcsec.fillna(0.0), color=np.where(display.variant.eq("A0279_primary"), "tab:orange", "tab:blue"))
    for index, row in enumerate(display.itertuples(index=False)):
        if not np.isfinite(row.heldout_RMS_arcsec):
            ax.text(0.03, index, f"{row.heldout_converged_roots}/7 roots", color="crimson", va="center")
    ax.set(xlabel="held-out RMS (arcsec)", title="Locked primary and descriptive translations")

    ax = axes[0, 1]
    block = predictions[predictions.stage.eq("heldout") & predictions.variant.isin(["scalar_baseline", "A0279_primary"])]
    pivot = block.pivot(index="image_id", columns="variant", values="radial_residual_arcsec")
    x = np.arange(len(pivot))
    ax.bar(x - 0.18, pivot.scalar_baseline, 0.36, label="scalar")
    ax.bar(x + 0.18, pivot.A0279_primary, 0.36, label="A0279")
    ax.set(xticks=x, xticklabels=pivot.index, ylabel="residual (arcsec)", title="Held-out images")
    ax.legend()

    ax = axes[1, 0]
    finite = randomizations[np.isfinite(randomizations.heldout_RMS_arcsec)]
    ax.hist(finite.heldout_RMS_arcsec, bins=12, alpha=0.7)
    actual = float(scores[scores.variant.eq("A0279_primary")].heldout_RMS_arcsec.iloc[0])
    if np.isfinite(actual):
        ax.axvline(actual, color="crimson", ls="--", label="measured layout")
    ax.set(xlabel="random-angle held-out RMS (arcsec)", title="Angular specificity control")
    ax.legend()

    ax = axes[1, 1]
    ax.scatter(sources.x_arcsec, sources.y_arcsec, s=25 + 100 * sources.base_weight / sources.base_weight.max(), label="baryonic sources", alpha=0.7)
    ax.scatter(endpoints[:, 0], endpoints[:, 1], marker="x", label="A0279 endpoints", alpha=0.7)
    for start, end in zip(sources[["x_arcsec", "y_arcsec"]].to_numpy(float), endpoints):
        ax.plot([start[0], end[0]], [start[1], end[1]], color="gray", alpha=0.12)
    ax.set(aspect="equal", xlabel="x (arcsec)", ylabel="y (arcsec)", title="Locked center-return routes")
    ax.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    config_path = ROOT / "configs/adaptive_route_raw_rxj2129_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    map_report_path = ROOT / protocol["locked_map_result"]["report"]
    map_report = json.loads(map_report_path.read_text(encoding="utf-8"))
    if not bool(map_report[protocol["locked_map_result"]["required_gate"]]):
        raise RuntimeError("adaptive map transfer gate did not pass")
    if map_report["all_cluster_selected_candidate"]["candidate_id"] != protocol["locked_map_result"]["candidate_id"]:
        raise RuntimeError("locked map candidate changed")

    raw_path = ROOT / protocol["raw_inputs"]["raw_protocol"]
    raw_protocol = json.loads(raw_path.read_text(encoding="utf-8"))
    images = load_images(raw_protocol)
    training, heldout = split_images(images, raw_protocol)
    sources = load_sources(protocol, raw_protocol)
    sources.to_csv(output / protocol["outputs"]["source_catalog"], index=False)
    anchors = load_baryonic_anchors(raw_protocol)
    parent_protocol_path = ROOT / protocol["raw_inputs"]["parent_protocol"]
    parent_protocol = json.loads(parent_protocol_path.read_text(encoding="utf-8"))
    parent_id = protocol["raw_inputs"]["parent_candidate"]
    parent_score_path = ROOT / protocol["raw_inputs"]["parent_scores"]
    parent_scores = pd.read_csv(parent_score_path)
    parent_row = parent_scores[parent_scores.candidate_id.eq(parent_id)].iloc[0]
    specs = {item["candidate_id"]: item for item in build_specs(parent_protocol)}
    parent, _ = raw_field(specs[parent_id], float(parent_row.universal_q), anchors, raw_protocol, 1.2e-10)
    baryons = baryon_field(anchors, raw_protocol)
    parameter_path = ROOT / protocol["raw_inputs"]["parent_parameters"]
    initial = parent_initial(pd.read_csv(parameter_path), parent_id)
    candidates = pd.read_csv(ROOT / protocol["locked_map_result"]["candidate_specs"]).set_index("candidate_id")

    variants = []
    fields = {}
    audits = []
    primary_endpoints = None
    for item in protocol["variants"]:
        candidate = candidates.loc[item["candidate_id"]]
        field, audit = build_route_field(
            protocol,
            raw_protocol,
            sources,
            candidate,
            parent,
            baryons,
            contrast_cap=float(item["contrast_cap"]),
            centroid_mode=str(item["centroid_mode"]),
        )
        fields[item["label"]] = field
        endpoints = audit.pop("endpoints")
        if item["label"] == "A0279_primary":
            primary_endpoints = endpoints
        audits.append({"variant": item["label"], **audit})
        variants.append(item)
    audit_frame = pd.DataFrame(audits)
    audit_frame.to_csv(output / protocol["outputs"]["field_audits"], index=False)

    fit_settings = protocol["fit"]
    score_rows, predictions, parameter_rows, fitted = [], [], [], {}
    all_variants = [{"label": "scalar_baseline", "field": None}] + [
        {"label": item["label"], "field": fields[item["label"]]} for item in variants
    ]
    for index, item in enumerate(all_variants):
        label = item["label"]
        lens = MorphologyLens(
            raw_protocol,
            {MODEL: parent},
            parent=MODEL,
            morphology=item["field"],
            fraction=0.0 if item["field"] is None else 1.0,
        )
        starts = (
            int(fit_settings["baseline_starts"])
            if label == "scalar_baseline"
            else int(fit_settings["primary_starts"])
            if label == "A0279_primary"
            else int(fit_settings["sensitivity_starts"])
        )
        fit = exact_fit(
            lens,
            training,
            heldout,
            initial=initial,
            starts=starts,
            seed=int(fit_settings["random_seed"]) + index,
        )
        fitted[label] = fit
        joined = pd.concat([fit["training_prediction"], fit["heldout_prediction"]], ignore_index=True)
        joined["variant"] = label
        predictions.append(joined)
        train_score, hold_score = fit["training_score"], fit["heldout_score"]
        score_rows.append(
            {
                "variant": label,
                "training_RMS_arcsec": train_score["exact_radial_RMS_arcsec"],
                "training_converged_roots": train_score["converged_roots"],
                "heldout_RMS_arcsec": hold_score["exact_radial_RMS_arcsec"],
                "heldout_converged_roots": hold_score["converged_roots"],
                "optimizer_cost": fit["optimizer_cost"],
            }
        )
        bounds = near_bound(MODEL, fit["parameters"])
        for name, value in zip(FIXED_LABELS, fit["parameters"]):
            parameter_rows.append(
                {"variant": label, "parameter": name, "value": value, "near_bound": bounds[name]}
            )
    scores = pd.DataFrame(score_rows)
    baseline_rms = float(scores[scores.variant.eq("scalar_baseline")].heldout_RMS_arcsec.iloc[0])
    scores["fractional_heldout_improvement_vs_scalar"] = (
        baseline_rms - scores.heldout_RMS_arcsec
    ) / baseline_rms
    scores.to_csv(output / protocol["outputs"]["variant_scores"], index=False)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    prediction_frame.to_csv(output / protocol["outputs"]["predictions"], index=False)
    pd.DataFrame(parameter_rows).to_csv(output / protocol["outputs"]["parameters"], index=False)

    rng = np.random.default_rng(int(fit_settings["random_seed"]) + 1000)
    primary_candidate = candidates.loc[protocol["locked_map_result"]["candidate_id"]]
    primary_parameters = fitted["A0279_primary"]["parameters"]
    random_rows = []
    radius = np.hypot(sources.x_arcsec, sources.y_arcsec).to_numpy(float)
    for trial in range(int(protocol["randomization"]["primary_radius_preserving_angle_trials"])):
        angle = rng.uniform(-np.pi, np.pi, len(sources))
        xy = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
        field, _ = build_route_field(
            protocol,
            raw_protocol,
            sources,
            primary_candidate,
            parent,
            baryons,
            contrast_cap=20.0,
            centroid_mode="light_centroid",
            randomized_xy=xy,
        )
        lens = MorphologyLens(raw_protocol, {MODEL: parent}, parent=MODEL, morphology=field, fraction=1.0)
        _, source_positions = lens.profiled_residuals(MODEL, primary_parameters, training)
        prediction = lens.exact_predictions(MODEL, primary_parameters, source_positions, heldout, stage="heldout")
        metrics = score(prediction, lens.sigma)
        random_rows.append(
            {"trial": trial, "heldout_RMS_arcsec": metrics["exact_radial_RMS_arcsec"], "heldout_converged_roots": metrics["converged_roots"]}
        )
    randomizations = pd.DataFrame(random_rows)
    randomizations.to_csv(output / protocol["outputs"]["randomizations"], index=False)

    primary = scores[scores.variant.eq("A0279_primary")].iloc[0]
    random_values = pd.to_numeric(randomizations.heldout_RMS_arcsec, errors="coerce").fillna(np.inf).to_numpy(float)
    empirical_p = float((1 + np.sum(random_values <= float(primary.heldout_RMS_arcsec))) / (1 + len(random_values))) if np.isfinite(primary.heldout_RMS_arcsec) else 1.0
    gates = protocol["gates"]
    report = {
        "protocol_version": protocol["protocol_version"],
        "coverage": {
            "hard_photometric_sources": len(sources),
            "training_images": len(training),
            "heldout_images": len(heldout),
            "locked_primary": "A0279",
            "descriptive_translations": len(variants) - 1,
            "random_angle_trials": len(randomizations),
        },
        "primary_realized_parameters": audit_frame[audit_frame.variant.eq("A0279_primary")].iloc[0].to_dict(),
        "scores": scores.to_dict("records"),
        "randomization": {
            "finite_trials": int(np.isfinite(random_values).sum()),
            "median_RMS_arcsec": float(np.median(random_values)),
            "empirical_p_random_as_good_or_better": empirical_p,
        },
        "gates": {
            "all_roots_pass": bool(int(primary.heldout_converged_roots) == len(heldout)),
            "improvement_pass": bool(float(primary.fractional_heldout_improvement_vs_scalar) >= float(gates["heldout_improvement_over_scalar_parent_fraction"])),
            "absolute_RMS_pass": bool(float(primary.heldout_RMS_arcsec) <= float(gates["absolute_heldout_RMS_arcsec"])),
            "random_angle_pass": bool(empirical_p <= float(gates["random_angle_empirical_p_max"])),
        },
        "inherited_cross_domain": {
            "scalar_parent": parent_id,
            "galaxy_outer_RMSE_km_s": float(parent_row.cross_galaxy_outer_RMSE_km_s),
            "CLASH_absolute_RMSE_dex": float(parent_row.cluster_RMSE_dex),
            "Solar_all_proxies_pass": bool(parent_row.all_solar_proxies_pass),
            "directional_single_centered_source_change": 0.0,
        },
        "claim_limits": protocol["claim_limits"],
        "hashes": {
            "protocol": sha256(config_path),
            "map_report": sha256(map_report_path),
            "raw_protocol": sha256(raw_path),
            "parent_protocol": sha256(parent_protocol_path),
            "parent_scores": sha256(parent_score_path),
            "parent_parameters": sha256(parameter_path),
            "molino_catalog": sha256(ROOT / protocol["raw_inputs"]["molino_catalog"]),
        },
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    make_figure(scores, prediction_frame, randomizations, sources, primary_endpoints, output / protocol["outputs"]["figure"])
    summary = f"""# Adaptive route raw RX J2129 result

The locked A0279 map law was translated without a new amplitude fit. Its held-out RMS is **{float(primary.heldout_RMS_arcsec):.4f} arcsec** versus **{baseline_rms:.4f} arcsec** for the same P0554 scalar parent, a **{100*float(primary.fractional_heldout_improvement_vs_scalar):.2f}%** change. The radius-preserving angle-control empirical p-value is **{empirical_p:.4f}**.
"""
    (output / protocol["outputs"]["summary"]).write_text(summary, encoding="utf-8")
    print(json.dumps(json_safe({"primary": report["scores"][1], "gates": report["gates"], "randomization": report["randomization"]}), indent=2))


if __name__ == "__main__":
    main()
