#!/usr/bin/env python3
"""Test the universal map-fraction to angular-potential bridge."""

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

from run_adaptive_route_raw_rxj2129 import (  # noqa: E402
    MODEL,
    baryon_field,
    build_route_field,
    exact_fit,
    load_sources,
)
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


def initial_geometry(path: Path) -> np.ndarray:
    parameters = pd.read_csv(path)
    block = parameters[parameters.variant.eq("scalar_baseline")].set_index("parameter")
    return np.asarray([float(block.loc[name, "value"]) for name in FIXED_LABELS])


def training_fit(lens, training, *, initial, starts, seed):
    fit = lens.fit(MODEL, training, starts=starts, seed=seed, initial_override=initial)
    prediction = lens.exact_predictions(
        MODEL, fit["result"].x, fit["sources"], training, stage="training"
    )
    return {
        "parameters": fit["result"].x,
        "optimizer_cost": float(fit["result"].cost),
        "prediction": prediction,
        "score": score(prediction, lens.sigma, free_parameters=20),
    }


def make_lens(raw_protocol, parent, field):
    return MorphologyLens(
        raw_protocol,
        {MODEL: parent},
        parent=MODEL,
        morphology=field,
        fraction=1.0,
    )


def make_figure(screen, refits, final, predictions, randomizations, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(screen.fraction_power, screen.screen_training_RMS_arcsec, marker="o", label="fixed geometry")
    ax.plot(refits.fraction_power, refits.training_RMS_arcsec, marker="s", label="training refit")
    ax.set(xlabel="bridge exponent p", ylabel="training RMS (arcsec)", title=r"Angular strength $s=f^p$")
    ax.legend()

    ax = axes[0, 1]
    route = final[final.variant.str.startswith("power_")]
    ax.plot(route.fraction_power, route.heldout_RMS_arcsec, marker="o")
    scalar = float(final[final.variant.eq("scalar_baseline")].heldout_RMS_arcsec.iloc[0])
    ax.axhline(scalar, color="black", ls="--", label="scalar parent")
    ax.set(xlabel="bridge exponent p", ylabel="held-out RMS (arcsec)", title="Post-selection response curve")
    ax.legend()

    ax = axes[1, 0]
    block = predictions[predictions.stage.eq("heldout") & predictions.variant.isin(["scalar_baseline", "selected_bridge"])]
    pivot = block.pivot(index="image_id", columns="variant", values="radial_residual_arcsec")
    x = np.arange(len(pivot))
    ax.bar(x - 0.18, pivot.scalar_baseline, 0.36, label="scalar")
    ax.bar(x + 0.18, pivot.selected_bridge, 0.36, label="selected bridge")
    ax.set(xticks=x, xticklabels=pivot.index, ylabel="residual (arcsec)", title="Held-out image residuals")
    ax.legend()

    ax = axes[1, 1]
    finite = randomizations[np.isfinite(randomizations.heldout_RMS_arcsec)]
    ax.hist(finite.heldout_RMS_arcsec, bins=14, alpha=0.7)
    actual = float(final[final.variant.eq("selected_bridge")].heldout_RMS_arcsec.iloc[0])
    ax.axvline(actual, color="crimson", ls="--", label="measured layout")
    ax.set(xlabel="random-angle held-out RMS (arcsec)", title="Selected bridge specificity")
    ax.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    config_path = ROOT / "configs/adaptive_route_amplitude_bridge_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    translation_path = ROOT / protocol["inputs"]["raw_translation_protocol"]
    translation = json.loads(translation_path.read_text(encoding="utf-8"))
    translation_report_path = ROOT / protocol["inputs"]["raw_translation_report"]
    translation_report = json.loads(translation_report_path.read_text(encoding="utf-8"))
    raw_path = ROOT / translation["raw_inputs"]["raw_protocol"]
    raw_protocol = json.loads(raw_path.read_text(encoding="utf-8"))
    images = load_images(raw_protocol)
    training, heldout = split_images(images, raw_protocol)
    sources = load_sources(translation, raw_protocol)
    anchors = load_baryonic_anchors(raw_protocol)
    parent_protocol_path = ROOT / translation["raw_inputs"]["parent_protocol"]
    parent_protocol = json.loads(parent_protocol_path.read_text(encoding="utf-8"))
    parent_score_path = ROOT / translation["raw_inputs"]["parent_scores"]
    parent_scores = pd.read_csv(parent_score_path)
    parent_id = translation["raw_inputs"]["parent_candidate"]
    parent_row = parent_scores[parent_scores.candidate_id.eq(parent_id)].iloc[0]
    parent_specs = {item["candidate_id"]: item for item in build_specs(parent_protocol)}
    parent, _ = raw_field(parent_specs[parent_id], float(parent_row.universal_q), anchors, raw_protocol, 1.2e-10)
    baryons = baryon_field(anchors, raw_protocol)
    candidate_path = ROOT / protocol["inputs"]["candidate_specs"]
    candidate = pd.read_csv(candidate_path).set_index("candidate_id").loc["A0279"]
    initial_path = ROOT / protocol["inputs"]["raw_translation_parameters"]
    initial = initial_geometry(initial_path)

    _, unit_audit = build_route_field(
        translation,
        raw_protocol,
        sources,
        candidate,
        parent,
        baryons,
        contrast_cap=20.0,
        contrast_strength=1.0,
        centroid_mode="light_centroid",
    )
    routed_fraction = float(unit_audit["routing_fraction"])
    fields, screen_rows = {}, []
    for power in protocol["grid"]["fraction_power"]:
        strength = float(routed_fraction**float(power))
        field, audit = build_route_field(
            translation,
            raw_protocol,
            sources,
            candidate,
            parent,
            baryons,
            contrast_cap=20.0,
            contrast_strength=strength,
            centroid_mode="light_centroid",
        )
        fields[float(power)] = field
        lens = make_lens(raw_protocol, parent, field)
        residual, _ = lens.profiled_residuals(MODEL, initial, training)
        xy = residual.reshape(-1, 2) * lens.sigma
        screen_rows.append(
            {
                "fraction_power": float(power),
                "routed_fraction": routed_fraction,
                "angular_strength": strength,
                "screen_training_RMS_arcsec": float(np.sqrt(np.mean(np.sum(xy**2, axis=1)))),
                "correction_RMS_arcsec": audit["raw_correction_RMS_arcsec"],
                "correction_maximum_arcsec": audit["raw_correction_maximum_arcsec"],
            }
        )
    screen = pd.DataFrame(screen_rows)
    screen.to_csv(output / protocol["outputs"]["screen"], index=False)

    refit_rows, refit_results = [], {}
    starts = int(protocol["selection"]["training_refit_starts"])
    for index, power in enumerate(protocol["grid"]["fraction_power"]):
        lens = make_lens(raw_protocol, parent, fields[float(power)])
        fit = training_fit(
            lens,
            training,
            initial=initial,
            starts=starts,
            seed=int(protocol["selection"]["random_seed"]) + index,
        )
        refit_results[float(power)] = fit
        metrics = fit["score"]
        refit_rows.append(
            {
                "fraction_power": float(power),
                "angular_strength": float(routed_fraction**float(power)),
                "training_RMS_arcsec": metrics["exact_radial_RMS_arcsec"],
                "training_converged_roots": metrics["converged_roots"],
                "training_all_roots": metrics["all_roots_converged"],
                "optimizer_cost": fit["optimizer_cost"],
            }
        )
    refits = pd.DataFrame(refit_rows)
    refits.to_csv(output / protocol["outputs"]["training_refits"], index=False)
    eligible = refits[refits.training_all_roots.astype(bool)].copy()
    eligible["selection_RMS"] = pd.to_numeric(eligible.training_RMS_arcsec, errors="coerce").fillna(np.inf)
    selected_power = float(eligible.sort_values(["selection_RMS", "fraction_power"], ascending=[True, False]).iloc[0].fraction_power)

    final_rows, predictions, parameter_rows = [], [], []
    scalar_lens = MorphologyLens(raw_protocol, {MODEL: parent}, parent=MODEL, morphology=None, fraction=0.0)
    scalar_fit = exact_fit(
        scalar_lens,
        training,
        heldout,
        initial=initial,
        starts=int(protocol["selection"]["final_selected_starts"]),
        seed=int(protocol["selection"]["random_seed"]) + 100,
    )
    final_fit_by_label = {"scalar_baseline": scalar_fit}
    for power in protocol["grid"]["fraction_power"]:
        label = f"power_{float(power):g}"
        lens = make_lens(raw_protocol, parent, fields[float(power)])
        seed_fit = refit_results[float(power)]
        _, sources_profiled = lens.profiled_residuals(MODEL, seed_fit["parameters"], training)
        held_prediction = lens.exact_predictions(
            MODEL, seed_fit["parameters"], sources_profiled, heldout, stage="heldout"
        )
        final_fit_by_label[label] = {
            "parameters": seed_fit["parameters"],
            "training_prediction": seed_fit["prediction"],
            "heldout_prediction": held_prediction,
            "training_score": seed_fit["score"],
            "heldout_score": score(held_prediction, lens.sigma),
            "optimizer_cost": seed_fit["optimizer_cost"],
        }
    selected_lens = make_lens(raw_protocol, parent, fields[selected_power])
    selected_seed = refit_results[selected_power]
    selected_fit = exact_fit(
        selected_lens,
        training,
        heldout,
        initial=selected_seed["parameters"],
        starts=int(protocol["selection"]["final_selected_starts"]),
        seed=int(protocol["selection"]["random_seed"]) + 200,
    )
    final_fit_by_label["selected_bridge"] = selected_fit

    for label, fit in final_fit_by_label.items():
        power = selected_power if label == "selected_bridge" else float(label.removeprefix("power_")) if label.startswith("power_") else np.nan
        train_score, hold_score = fit["training_score"], fit["heldout_score"]
        final_rows.append(
            {
                "variant": label,
                "fraction_power": power,
                "angular_strength": routed_fraction**power if np.isfinite(power) else 0.0,
                "training_RMS_arcsec": train_score["exact_radial_RMS_arcsec"],
                "training_converged_roots": train_score["converged_roots"],
                "heldout_RMS_arcsec": hold_score["exact_radial_RMS_arcsec"],
                "heldout_converged_roots": hold_score["converged_roots"],
                "optimizer_cost": fit["optimizer_cost"],
            }
        )
        joined = pd.concat([fit["training_prediction"], fit["heldout_prediction"]], ignore_index=True)
        joined["variant"] = label
        predictions.append(joined)
        bounds = near_bound(MODEL, fit["parameters"])
        for name, value in zip(FIXED_LABELS, fit["parameters"]):
            parameter_rows.append(
                {"variant": label, "parameter": name, "value": value, "near_bound": bounds[name]}
            )
    final = pd.DataFrame(final_rows)
    baseline_rms = float(final[final.variant.eq("scalar_baseline")].heldout_RMS_arcsec.iloc[0])
    final["fractional_heldout_improvement_vs_scalar"] = (
        baseline_rms - final.heldout_RMS_arcsec
    ) / baseline_rms
    final.to_csv(output / protocol["outputs"]["final_scores"], index=False)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    prediction_frame.to_csv(output / protocol["outputs"]["predictions"], index=False)
    pd.DataFrame(parameter_rows).to_csv(output / protocol["outputs"]["parameters"], index=False)

    rng = np.random.default_rng(int(protocol["selection"]["random_seed"]) + 1000)
    radius = np.hypot(sources.x_arcsec, sources.y_arcsec).to_numpy(float)
    selected_parameters = selected_fit["parameters"]
    strength = float(routed_fraction**selected_power)
    random_rows = []
    for trial in range(int(protocol["randomization"]["radius_preserving_angle_trials"])):
        angle = rng.uniform(-np.pi, np.pi, len(sources))
        xy = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
        field, _ = build_route_field(
            translation,
            raw_protocol,
            sources,
            candidate,
            parent,
            baryons,
            contrast_cap=20.0,
            contrast_strength=strength,
            centroid_mode="light_centroid",
            randomized_xy=xy,
        )
        lens = make_lens(raw_protocol, parent, field)
        _, source_positions = lens.profiled_residuals(MODEL, selected_parameters, training)
        held_prediction = lens.exact_predictions(
            MODEL, selected_parameters, source_positions, heldout, stage="heldout"
        )
        metrics = score(held_prediction, lens.sigma)
        random_rows.append(
            {"trial": trial, "heldout_RMS_arcsec": metrics["exact_radial_RMS_arcsec"], "heldout_converged_roots": metrics["converged_roots"]}
        )
    randomizations = pd.DataFrame(random_rows)
    randomizations.to_csv(output / protocol["outputs"]["randomizations"], index=False)

    selected = final[final.variant.eq("selected_bridge")].iloc[0]
    random_values = pd.to_numeric(randomizations.heldout_RMS_arcsec, errors="coerce").fillna(np.inf).to_numpy(float)
    empirical_p = float((1 + np.sum(random_values <= float(selected.heldout_RMS_arcsec))) / (1 + len(random_values))) if np.isfinite(selected.heldout_RMS_arcsec) else 1.0
    gates = protocol["gates"]
    report = {
        "protocol_version": protocol["protocol_version"],
        "routed_fraction": routed_fraction,
        "selected_training_only_fraction_power": selected_power,
        "selected_angular_strength": strength,
        "scores": final.to_dict("records"),
        "parameter_impact": {
            "heldout_RMS_span_arcsec": float(final[final.variant.str.startswith("power_")].heldout_RMS_arcsec.max() - final[final.variant.str.startswith("power_")].heldout_RMS_arcsec.min()),
            "training_RMS_span_arcsec": float(refits.training_RMS_arcsec.max() - refits.training_RMS_arcsec.min()),
        },
        "randomization": {
            "trials": len(randomizations),
            "finite_trials": int(np.isfinite(random_values).sum()),
            "median_RMS_arcsec": float(np.median(random_values)),
            "empirical_p_random_as_good_or_better": empirical_p,
        },
        "gates": {
            "all_roots_pass": bool(int(selected.heldout_converged_roots) == len(heldout)),
            "improvement_pass": bool(float(selected.fractional_heldout_improvement_vs_scalar) >= float(gates["heldout_improvement_over_scalar_fraction"])),
            "absolute_RMS_pass": bool(float(selected.heldout_RMS_arcsec) <= float(gates["absolute_heldout_RMS_arcsec"])),
            "random_angle_pass": bool(empirical_p <= float(gates["random_angle_empirical_p_max"])),
        },
        "inherited_cross_domain": translation_report["inherited_cross_domain"],
        "claim_limits": protocol["claim_limits"],
        "hashes": {
            "protocol": sha256(config_path),
            "translation_protocol": sha256(translation_path),
            "translation_report": sha256(translation_report_path),
            "candidate_specs": sha256(candidate_path),
            "raw_protocol": sha256(raw_path),
            "parent_protocol": sha256(parent_protocol_path),
            "parent_scores": sha256(parent_score_path),
            "initial_parameters": sha256(initial_path),
        },
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    make_figure(screen, refits, final, prediction_frame, randomizations, output / protocol["outputs"]["figure"])
    summary = f"""# Adaptive route amplitude bridge

Training-only selection chose **p={selected_power:g}**, so the 15.37% map-level route fraction becomes angular strength **{strength:.6f}**. Held-out RMS is **{float(selected.heldout_RMS_arcsec):.4f} arcsec** versus **{baseline_rms:.4f} arcsec** for the scalar parent. The empirical random-angle p-value is **{empirical_p:.4f}**.
"""
    (output / protocol["outputs"]["summary"]).write_text(summary, encoding="utf-8")
    print(json.dumps(json_safe({"selected_power": selected_power, "selected": final[final.variant.eq('selected_bridge')].iloc[0].to_dict(), "gates": report["gates"], "randomization": report["randomization"]}), indent=2))


if __name__ == "__main__":
    main()
