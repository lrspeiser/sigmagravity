#!/usr/bin/env python3
"""Search for an interior strength of the P0646 gas-minus-star flux closure."""

from __future__ import annotations

import argparse
import copy
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

from run_adaptive_route_raw_rxj2129 import baryon_field, exact_fit
from run_clash_stellar_morphology_response import MorphologyLens
from run_p0601_frozen_potential_raw_lensing import build_fields as build_p0599_fields
from run_p0607_component_direction_raw_lensing import fixed_geometry
from run_p0644_spent_rxj2129_accumulated_tensor import make_field, read_json
from run_p0645_fair_geometry_cv_accumulated_tensor import stratified_folds
from run_p0646_conservative_closure_atlas import evaluate_candidate
from run_rxj2129_member_geometry import split_images
from run_rxj2129_raw_theory_lensing import load_baryonic_anchors, load_images

DEFAULT_CONFIG = ROOT / "configs" / "p0647_component_flux_strength_expansion.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_minimum_audit(scores: pd.DataFrame, selected_lambda: float) -> dict:
    ordered = scores.sort_values("lambda").reset_index(drop=True)
    matches = ordered.index[np.isclose(ordered["lambda"], selected_lambda)].tolist()
    if len(matches) != 1:
        raise RuntimeError("selected lambda is not unique in the frozen grid")
    index = int(matches[0])
    interior = 0 < index < len(ordered) - 1
    if not interior:
        return {
            "interior": False,
            "left_lambda": None,
            "right_lambda": None,
            "root_complete_neighbors": False,
            "strict_local_minimum": False,
        }
    left, selected, right = ordered.iloc[index - 1], ordered.iloc[index], ordered.iloc[index + 1]
    neighbors_complete = bool(left.all_CV_roots and right.all_CV_roots)
    strict = bool(
        neighbors_complete
        and float(selected.pooled_CV_RMS_arcsec) < float(left.pooled_CV_RMS_arcsec)
        and float(selected.pooled_CV_RMS_arcsec) < float(right.pooled_CV_RMS_arcsec)
    )
    return {
        "interior": True,
        "left_lambda": float(left["lambda"]),
        "left_CV_RMS_arcsec": float(left.pooled_CV_RMS_arcsec),
        "right_lambda": float(right["lambda"]),
        "right_CV_RMS_arcsec": float(right.pooled_CV_RMS_arcsec),
        "root_complete_neighbors": neighbors_complete,
        "strict_local_minimum": strict,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    if protocol.get("status") != "frozen_before_any_P0647_score":
        raise RuntimeError("P0647 protocol is not frozen")

    p0644 = read_json(ROOT / protocol["inputs"]["P0644_protocol"])
    p0645_protocol = read_json(ROOT / protocol["inputs"]["P0645_protocol"])
    p0645_report = read_json(ROOT / protocol["inputs"]["P0645_report"])
    p0646_report = read_json(ROOT / protocol["inputs"]["P0646_report"])
    p0643_report = read_json(ROOT / protocol["inputs"]["P0643_report"])
    if p0646_report["selection"]["P0601_spent_heldout_used_for_selection"]:
        raise RuntimeError("P0646 selection boundary changed")

    raw_protocol = read_json(ROOT / p0644["inputs"]["raw_protocol"])
    p0601 = read_json(ROOT / p0644["inputs"]["P0601_protocol"])
    images = load_images(raw_protocol)
    training, spent_heldout = split_images(images, raw_protocol)
    anchors = load_baryonic_anchors(raw_protocol)
    radial_fields, _, _ = build_p0599_fields(anchors, raw_protocol, p0601["constants"])
    parent = radial_fields["P0599_potential_shape"]
    baryons = baryon_field(anchors, raw_protocol)
    initial = fixed_geometry(ROOT / p0644["inputs"]["P0601_parameters"])
    folds = list(stratified_folds(training, int(p0645_protocol["cross_validation"]["folds"])))

    field_protocol = copy.deepcopy(p0644)
    candidate = protocol["candidate"]
    field_protocol["candidate"].update(
        {
            "coherence_length_kpc": candidate["coherence_length_kpc"],
            "accumulation_power": candidate["accumulation_power"],
            "a0_m_s2": candidate["a0_m_s2"],
            "stellar_mass_fraction": candidate["stellar_mass_fraction"],
            "gas_mass_fraction": candidate["gas_mass_fraction"],
            "common_smoothing_kpc": candidate["common_physical_smoothing_kpc"],
            "closure": candidate["closure"],
        }
    )
    field, map_audits = make_field(field_protocol, raw_protocol, anchors, parent, baryons, images)

    cv = protocol["cross_validation"]
    summaries, fold_rows, prediction_frames = [], [], []
    for offset, strength in enumerate(candidate["lambda_grid"]):
        summary, rows, predictions = evaluate_candidate(
            candidate["closure"],
            float(strength),
            field,
            parent,
            raw_protocol,
            folds,
            initial,
            starts=int(cv["geometry_refit_starts_per_lambda_fold"]),
            seed=int(cv["random_seed"]) + 100 * offset,
            stage="exact_grid",
        )
        summaries.append(summary)
        fold_rows.extend(rows)
        prediction_frames.extend(predictions)
        print(
            f"lambda={float(strength):g}: CV={summary['pooled_CV_RMS_arcsec']:.6g} "
            f"roots={summary['CV_roots']}/{summary['CV_images']}",
            flush=True,
        )
    scores = pd.DataFrame(summaries)
    eligible = scores[
        scores.all_CV_roots & np.isfinite(scores.pooled_CV_RMS_arcsec)
    ].sort_values(["pooled_CV_RMS_arcsec", "lambda"])
    if eligible.empty:
        raise RuntimeError("no P0647 strength completed all CV roots")
    selected = eligible.iloc[0]
    selected_lambda = float(selected["lambda"])
    selected_cv = float(selected.pooled_CV_RMS_arcsec)
    local_audit = local_minimum_audit(scores, selected_lambda)

    baseline_cv = float(p0645_report["selection"]["lambda0_CV_RMS_arcsec"])
    isotropic_cv = float(p0646_report["selection"]["best_isotropic_CV_RMS_arcsec"])
    improvement = 1.0 - selected_cv / baseline_cv
    improvement_vs_isotropic = 1.0 - selected_cv / isotropic_cv

    lens = MorphologyLens(
        raw_protocol,
        {"P0599_potential_shape": parent},
        parent="P0599_potential_shape",
        morphology=field,
        fraction=selected_lambda,
    )
    full = exact_fit(
        lens,
        training,
        spent_heldout,
        initial=initial,
        starts=int(cv["full_training_refit_starts"]),
        seed=int(cv["random_seed"]) + 90000,
    )
    full_predictions = pd.concat(
        [full["training_prediction"], full["heldout_prediction"]], ignore_index=True
    )
    baseline_heldout = float(p0645_report["full_refit"]["spent_heldout_RMS_arcsec"])
    heldout_rms = float(full["heldout_score"]["exact_radial_RMS_arcsec"])
    heldout_worsening = heldout_rms / baseline_heldout - 1.0

    p0643_max_lambda = 20.0
    solar_at_max = float(
        p0643_report["primary_metrics"]["solar_1au_max_future_lambda_coefficient"]
    )
    solar_coefficient = solar_at_max * selected_lambda / p0643_max_lambda
    gates = protocol["predeclared_progression_gates"]
    audit = field.audit
    gate_results = {
        "selected_CV_roots": bool(selected.all_CV_roots)
        is bool(gates["selected_all_CV_roots"]),
        "CV_improvement": improvement >= gates["CV_improvement_fraction_vs_lambda0_min"],
        "beats_P0646_isotropic": improvement_vs_isotropic
        >= gates["CV_improvement_fraction_vs_P0646_isotropic_min"],
        "lambda_interior": bool(local_audit["interior"])
        is bool(gates["selected_lambda_strictly_inside_grid"]),
        "strict_local_minimum": bool(local_audit["strict_local_minimum"])
        is bool(gates["strict_local_CV_minimum_with_root_complete_neighbors"]),
        "full_training_roots": int(full["training_score"]["converged_roots"])
        == int(gates["full_training_roots"]),
        "spent_heldout_roots": int(full["heldout_score"]["converged_roots"])
        == int(gates["spent_heldout_roots"]),
        "spent_heldout_not_worse": heldout_worsening
        <= gates["spent_heldout_worsening_fraction_vs_P0599_max"],
        "field_curl": float(audit["normalized_curl_RMS"])
        <= gates["selected_field_normalized_curl_RMS_max"],
        "field_source_integral": float(audit["source_integral_fraction"])
        <= gates["selected_field_source_integral_fraction_max"],
        "solar_proxy": solar_coefficient <= gates["solar_1au_coefficient_max"],
        "sealed_targets_untouched": not bool(gates["sealed_target_outcomes_opened"]),
    }
    report = {
        "report_version": "P0647-COMPONENT-FLUX-STRENGTH-EXPANSION-RESULTS-1.0.0",
        "status": "pass" if all(gate_results.values()) else "fail",
        "all_progression_gates_pass": bool(all(gate_results.values())),
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__)),
        "coverage": {
            "lambda_rows": len(scores),
            "lambda_fold_refits": len(fold_rows),
            "CV_folds": len(folds),
            "ordinary_geometry_parameters_refit_per_run": 6,
            "per_object_spatial_gravity_parameters": 0,
        },
        "field_audit": audit,
        "map_audits": map_audits,
        "lambda_scores": scores.to_dict(orient="records"),
        "local_minimum_audit": local_audit,
        "selection": {
            "closure": candidate["closure"],
            "selected_lambda": selected_lambda,
            "lambda0_CV_RMS_arcsec": baseline_cv,
            "selected_CV_RMS_arcsec": selected_cv,
            "CV_improvement_fraction_vs_lambda0": improvement,
            "P0646_isotropic_CV_RMS_arcsec": isotropic_cv,
            "CV_improvement_fraction_vs_P0646_isotropic": improvement_vs_isotropic,
            "P0601_spent_heldout_used_for_selection": False,
        },
        "full_refit": {
            "training_RMS_arcsec": float(full["training_score"]["exact_radial_RMS_arcsec"]),
            "training_roots": int(full["training_score"]["converged_roots"]),
            "spent_heldout_RMS_arcsec": heldout_rms,
            "spent_heldout_roots": int(full["heldout_score"]["converged_roots"]),
            "spent_heldout_worsening_fraction_vs_P0599": heldout_worsening,
        },
        "solar_proxy": {
            "selected_lambda_1au_coefficient": solar_coefficient,
            "limit": gates["solar_1au_coefficient_max"],
            "is_a_full_PPN_or_Cassini_test": False,
        },
        "gate_results": gate_results,
        "sealed_P0633_kinematics_opened": False,
        "sealed_P0640_lensing_constraints_opened": False,
        "claim_boundary": protocol["claim_boundary"],
    }

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / "lambda_scores.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "fold_scores.csv", index=False)
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(
            output / "cv_predictions.csv", index=False
        )
    full_predictions.to_csv(output / "full_refit_predictions.csv", index=False)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(scores["lambda"], scores.pooled_CV_RMS_arcsec, "o-")
    axes[0].axhline(baseline_cv, color="black", linestyle="--", label="lambda 0")
    axes[0].axvline(selected_lambda, color="tab:red", linestyle=":", label="selected")
    axes[0].set(
        xlabel="universal lambda",
        ylabel="pooled CV RMS (arcsec)",
        title="Exact strength expansion",
    )
    axes[0].legend(fontsize=8)
    folds_frame = pd.DataFrame(fold_rows)
    for fold, block in folds_frame.groupby("fold"):
        axes[1].plot(block["lambda"], block.validation_RMS_arcsec, "o-", label=f"fold {fold}")
    axes[1].set(
        xlabel="universal lambda",
        ylabel="validation RMS (arcsec)",
        title="Source-family fold stability",
    )
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "strength_expansion.png", dpi=180)
    plt.close(figure)

    summary = f"""# P0647 component-flux strength expansion

- Status: **{report['status'].upper()}** ({sum(gate_results.values())}/{len(gate_results)} gates).
- Selected spent-data lambda: **{selected_lambda:g}**.
- Pooled CV RMS: lambda=0 **{baseline_cv:.6g} arcsec**, selected **{selected_cv:.6g} arcsec** ({100*improvement:+.3f}%).
- Strict root-complete local minimum: **{local_audit['strict_local_minimum']}**.
- Full-refit spent-heldout RMS: **{heldout_rms:.6g} arcsec** ({100*heldout_worsening:+.3f}% versus P0599).
- One-AU screening coefficient: **{solar_coefficient:.3e}** (proxy only).
- Sealed outcomes opened: **no**.

This is an identifiability test on spent RX J2129 images. It cannot validate
the closure, and a passing row still needs robustness and matched controls.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "selection": report["selection"],
                "local_minimum": local_audit,
                "gates": gate_results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
