#!/usr/bin/env python3
"""Compare conservative accumulated-response closures on spent raw lensing."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
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
from run_p0601_frozen_potential_raw_lensing import (
    build_fields as build_p0599_fields,
)
from run_p0607_component_direction_raw_lensing import fixed_geometry
from run_p0644_spent_rxj2129_accumulated_tensor import (
    make_field,
    read_json,
)
from run_p0645_fair_geometry_cv_accumulated_tensor import (
    pooled_rms,
    stratified_folds,
)
from run_rxj2129_member_geometry import split_images
from run_rxj2129_raw_theory_lensing import (
    load_baryonic_anchors,
    load_images,
)

DEFAULT_CONFIG = ROOT / "configs" / "p0646_conservative_closure_atlas.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_candidate(
    closure,
    strength,
    field,
    parent,
    raw_protocol,
    folds,
    initial,
    *,
    starts,
    seed,
    stage,
):
    rows, prediction_frames = [], []
    for fold, fit_images, validation_images in folds:
        lens = MorphologyLens(
            raw_protocol,
            {"P0599_potential_shape": parent},
            parent="P0599_potential_shape",
            morphology=field if float(strength) != 0.0 else None,
            fraction=float(strength),
        )
        try:
            fitted = exact_fit(
                lens,
                fit_images,
                validation_images,
                initial=initial,
                starts=int(starts),
                seed=int(seed) + fold,
            )
            validation_score = fitted["heldout_score"]
            row = {
                "stage": stage,
                "closure": closure,
                "lambda": float(strength),
                "fold": fold,
                "fit_images": len(fit_images),
                "validation_images": len(validation_images),
                "fit_RMS_arcsec": fitted["training_score"]["exact_radial_RMS_arcsec"],
                "fit_roots": fitted["training_score"]["converged_roots"],
                "validation_RMS_arcsec": validation_score["exact_radial_RMS_arcsec"],
                "validation_roots": validation_score["converged_roots"],
                "optimizer_cost": fitted["optimizer_cost"],
            }
            predictions = fitted["heldout_prediction"].copy()
            predictions["stage"] = stage
            predictions["closure"] = closure
            predictions["lambda"] = float(strength)
            predictions["fold"] = fold
            prediction_frames.append(predictions)
        except Exception as error:  # noqa: BLE001 - root loss is a scored result
            row = {
                "stage": stage,
                "closure": closure,
                "lambda": float(strength),
                "fold": fold,
                "fit_images": len(fit_images),
                "validation_images": len(validation_images),
                "fit_RMS_arcsec": math.inf,
                "fit_roots": 0,
                "validation_RMS_arcsec": math.inf,
                "validation_roots": 0,
                "optimizer_cost": math.inf,
                "failure": f"{type(error).__name__}: {error}",
            }
        rows.append(row)
    total_images = sum(int(row["validation_images"]) for row in rows)
    total_roots = sum(int(row["validation_roots"]) for row in rows)
    summary = {
        "stage": stage,
        "closure": closure,
        "lambda": float(strength),
        "pooled_CV_RMS_arcsec": pooled_rms(rows),
        "CV_roots": total_roots,
        "CV_images": total_images,
        "all_CV_roots": total_roots == total_images,
    }
    return summary, rows, prediction_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    if protocol.get("status") != "frozen_before_any_P0646_score":
        raise RuntimeError("P0646 protocol is not frozen")
    p0644 = read_json(ROOT / protocol["inputs"]["P0644_protocol"])
    p0645_report = read_json(ROOT / protocol["inputs"]["P0645_report"])
    if p0645_report["selection"]["P0601_spent_heldout_used_for_selection"]:
        raise RuntimeError("P0645 selection boundary changed")
    raw_protocol = read_json(ROOT / p0644["inputs"]["raw_protocol"])
    p0601 = read_json(ROOT / p0644["inputs"]["P0601_protocol"])
    images = load_images(raw_protocol)
    training, spent_heldout = split_images(images, raw_protocol)
    anchors = load_baryonic_anchors(raw_protocol)
    radial_fields, _, _ = build_p0599_fields(anchors, raw_protocol, p0601["constants"])
    parent = radial_fields["P0599_potential_shape"]
    baryons = baryon_field(anchors, raw_protocol)
    initial = fixed_geometry(ROOT / p0644["inputs"]["P0601_parameters"])
    folds = list(
        stratified_folds(training, int(read_json(ROOT / protocol["inputs"]["P0645_protocol"])["cross_validation"]["folds"]))
    )
    fields, field_audits = {}, {}
    for closure_row in protocol["closures"]:
        closure = closure_row["id"]
        field_protocol = copy.deepcopy(p0644)
        fixed = protocol["fixed_activation"]
        field_protocol["candidate"].update(
            {
                "coherence_length_kpc": fixed["coherence_length_kpc"],
                "accumulation_power": fixed["accumulation_power"],
                "a0_m_s2": fixed["a0_m_s2"],
                "stellar_mass_fraction": fixed["stellar_mass_fraction"],
                "gas_mass_fraction": fixed["gas_mass_fraction"],
                "common_smoothing_kpc": fixed["common_physical_smoothing_kpc"],
                "closure": closure,
            }
        )
        field, _ = make_field(field_protocol, raw_protocol, anchors, parent, baryons, images)
        fields[closure] = field
        field_audits[closure] = field.audit
        print(
            f"field {closure}: unit RMS={field.audit['unit_deflection_RMS_arcsec']:.5g}",
            flush=True,
        )
    cv = protocol["cross_validation"]
    summaries, fold_rows, prediction_frames = [], [], []
    baseline_summary, baseline_folds, baseline_predictions = evaluate_candidate(
        "lambda0_baseline",
        0.0,
        None,
        parent,
        raw_protocol,
        folds,
        initial,
        starts=int(cv["stage1_refit_starts"]),
        seed=int(cv["random_seed"]),
        stage="stage1",
    )
    summaries.append(baseline_summary)
    fold_rows.extend(baseline_folds)
    prediction_frames.extend(baseline_predictions)
    for closure_offset, closure_row in enumerate(protocol["closures"]):
        closure = closure_row["id"]
        for lambda_offset, strength in enumerate(protocol["lambda_grid_nonzero"]):
            summary, rows, predictions = evaluate_candidate(
                closure,
                strength,
                fields[closure],
                parent,
                raw_protocol,
                folds,
                initial,
                starts=int(cv["stage1_refit_starts"]),
                seed=int(cv["random_seed"]) + 1000 * (closure_offset + 1) + 100 * lambda_offset,
                stage="stage1",
            )
            summaries.append(summary)
            fold_rows.extend(rows)
            prediction_frames.extend(predictions)
            print(
                f"stage1 {closure} lambda={float(strength):g}: "
                f"CV={summary['pooled_CV_RMS_arcsec']:.6g} roots={summary['CV_roots']}/{summary['CV_images']}",
                flush=True,
            )
    stage1 = pd.DataFrame(summaries)
    eligible = stage1[
        stage1.all_CV_roots
        & stage1.closure.ne("lambda0_baseline")
        & np.isfinite(stage1.pooled_CV_RMS_arcsec)
    ].sort_values(["pooled_CV_RMS_arcsec", "lambda", "closure"])
    shortlist = eligible.head(int(cv["stage1_shortlist_rows"]))[["closure", "lambda"]].to_dict(
        orient="records"
    )
    isotropic = eligible[eligible.closure.eq("isotropic_control")]
    if not isotropic.empty:
        isotropic_choice = isotropic.iloc[0]
        if not any(
            row["closure"] == "isotropic_control"
            and np.isclose(row["lambda"], float(isotropic_choice["lambda"]))
            for row in shortlist
        ):
            shortlist.append(
                {"closure": "isotropic_control", "lambda": float(isotropic_choice["lambda"])}
            )
    stage2_summaries = []
    # Reuse the identical, already-audited P0645 lambda-zero fold result.  A
    # fresh three-start replay selected a lower profiled cost that subsequently
    # lost exact roots; treating that topology accident as an infinite physical
    # baseline would manufacture a false improvement.
    baseline2 = {
        "stage": "stage2",
        "closure": "lambda0_baseline",
        "lambda": 0.0,
        "pooled_CV_RMS_arcsec": float(
            p0645_report["selection"]["lambda0_CV_RMS_arcsec"]
        ),
        "CV_roots": 15,
        "CV_images": 15,
        "all_CV_roots": True,
        "provenance": "audited identical P0645 lambda-zero folds",
    }
    stage2_summaries.append(baseline2)
    for offset, row in enumerate(shortlist):
        summary, rows, predictions = evaluate_candidate(
            row["closure"],
            float(row["lambda"]),
            fields[row["closure"]],
            parent,
            raw_protocol,
            folds,
            initial,
            starts=int(cv["stage2_refit_starts"]),
            seed=int(cv["random_seed"]) + 51000 + 100 * offset,
            stage="stage2",
        )
        stage2_summaries.append(summary)
        fold_rows.extend(rows)
        prediction_frames.extend(predictions)
        print(
            f"stage2 {row['closure']} lambda={float(row['lambda']):g}: "
            f"CV={summary['pooled_CV_RMS_arcsec']:.6g} roots={summary['CV_roots']}/{summary['CV_images']}",
            flush=True,
        )
    stage2 = pd.DataFrame(stage2_summaries)
    selected_pool = stage2[
        stage2.all_CV_roots
        & stage2.closure.ne("lambda0_baseline")
        & np.isfinite(stage2.pooled_CV_RMS_arcsec)
    ].sort_values(["pooled_CV_RMS_arcsec", "lambda", "closure"])
    if selected_pool.empty:
        selected = stage2[stage2.closure.eq("lambda0_baseline")].iloc[0]
    else:
        selected = selected_pool.iloc[0]
    selected_closure = str(selected["closure"])
    selected_lambda = float(selected["lambda"])
    baseline_cv = float(
        stage2.loc[stage2.closure.eq("lambda0_baseline"), "pooled_CV_RMS_arcsec"].iloc[0]
    )
    selected_cv = float(selected["pooled_CV_RMS_arcsec"])
    improvement = 1.0 - selected_cv / baseline_cv
    isotropic2 = stage2[
        stage2.closure.eq("isotropic_control") & stage2.all_CV_roots
    ]
    best_isotropic = (
        float(isotropic2.pooled_CV_RMS_arcsec.min()) if not isotropic2.empty else math.inf
    )
    improvement_vs_isotropic = 1.0 - selected_cv / best_isotropic if np.isfinite(best_isotropic) else -math.inf
    selected_field = fields.get(selected_closure)
    selected_lens = MorphologyLens(
        raw_protocol,
        {"P0599_potential_shape": parent},
        parent="P0599_potential_shape",
        morphology=selected_field if selected_lambda != 0.0 else None,
        fraction=selected_lambda,
    )
    full = exact_fit(
        selected_lens,
        training,
        spent_heldout,
        initial=initial,
        starts=int(cv["full_refit_starts"]),
        seed=int(cv["random_seed"]) + 90000,
    )
    full_predictions = pd.concat(
        [full["training_prediction"], full["heldout_prediction"]], ignore_index=True
    )
    baseline_heldout = float(
        read_json(ROOT / protocol["inputs"]["P0645_report"])["full_refit"][
            "spent_heldout_RMS_arcsec"
        ]
    )
    heldout_rms = float(full["heldout_score"]["exact_radial_RMS_arcsec"])
    heldout_worsening = heldout_rms / baseline_heldout - 1.0
    gates = protocol["predeclared_progression_gates"]
    audit = selected_field.audit if selected_field is not None else {
        "normalized_curl_RMS": 0.0,
        "source_integral_fraction": 0.0,
    }
    gate_results = {
        "stage2_roots": bool(selected["all_CV_roots"])
        is bool(gates["selected_stage2_all_CV_roots"]),
        "CV_improvement": improvement
        >= gates["selected_stage2_CV_improvement_fraction_vs_lambda0_min"],
        "beats_isotropic": improvement_vs_isotropic
        >= gates["selected_beats_best_isotropic_control_fraction_min"],
        "lambda_not_endpoint": selected_lambda
        < max(float(value) for value in protocol["lambda_grid_nonzero"]),
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
        "sealed_targets_untouched": not bool(gates["sealed_target_outcomes_opened"]),
    }
    report = {
        "report_version": "P0646-CONSERVATIVE-CLOSURE-ATLAS-RESULTS-1.0.0",
        "status": "pass" if all(gate_results.values()) else "fail",
        "all_progression_gates_pass": bool(all(gate_results.values())),
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__)),
        "coverage": {
            "closures": len(protocol["closures"]),
            "stage1_nonzero_rows": len(protocol["closures"])
            * len(protocol["lambda_grid_nonzero"]),
            "stage1_fold_refits": (1 + len(protocol["closures"]) * len(protocol["lambda_grid_nonzero"]))
            * len(folds),
            "stage2_rows_including_baseline": len(stage2),
            "per_object_spatial_gravity_parameters": 0,
        },
        "field_audits": field_audits,
        "stage1_scores": stage1.to_dict(orient="records"),
        "stage1_shortlist": shortlist,
        "stage2_scores": stage2.to_dict(orient="records"),
        "selection": {
            "closure": selected_closure,
            "lambda": selected_lambda,
            "lambda0_CV_RMS_arcsec": baseline_cv,
            "selected_CV_RMS_arcsec": selected_cv,
            "improvement_fraction_vs_lambda0": improvement,
            "best_isotropic_CV_RMS_arcsec": best_isotropic,
            "improvement_fraction_vs_isotropic": improvement_vs_isotropic,
            "P0601_spent_heldout_used_for_selection": False,
        },
        "full_refit": {
            "training_RMS_arcsec": float(full["training_score"]["exact_radial_RMS_arcsec"]),
            "training_roots": int(full["training_score"]["converged_roots"]),
            "spent_heldout_RMS_arcsec": heldout_rms,
            "spent_heldout_roots": int(full["heldout_score"]["converged_roots"]),
            "spent_heldout_worsening_fraction_vs_P0599": heldout_worsening,
        },
        "gate_results": gate_results,
        "sealed_P0633_kinematics_opened": False,
        "sealed_P0640_lensing_constraints_opened": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    stage1.to_csv(output / "stage1_scores.csv", index=False)
    stage2.to_csv(output / "stage2_scores.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "fold_scores.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(output / "cv_predictions.csv", index=False)
    full_predictions.to_csv(output / "full_refit_predictions.csv", index=False)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for closure, block in stage1[stage1.closure.ne("lambda0_baseline")].groupby("closure"):
        axes[0].plot(block["lambda"], block.pooled_CV_RMS_arcsec, "o-", label=closure)
    axes[0].axhline(
        float(stage1.loc[stage1.closure.eq("lambda0_baseline"), "pooled_CV_RMS_arcsec"].iloc[0]),
        color="black",
        linestyle="--",
        label="lambda 0",
    )
    axes[0].set(xlabel="lambda", ylabel="stage-1 pooled CV RMS (arcsec)", title="Conservative closure atlas")
    axes[0].legend(fontsize=6, ncol=2)
    labels = [
        f"{closure}\nλ={strength:g}"
        for closure, strength in zip(stage2.closure, stage2["lambda"], strict=True)
    ]
    axes[1].bar(labels, stage2.pooled_CV_RMS_arcsec)
    axes[1].set(ylabel="stage-2 pooled CV RMS (arcsec)", title="Exact shortlist")
    axes[1].tick_params(axis="x", labelrotation=35)
    figure.tight_layout()
    figure.savefig(output / "closure_atlas.png", dpi=180)
    plt.close(figure)
    summary = f"""# P0646 conservative closure atlas

- Status: **{report['status'].upper()}** ({sum(gate_results.values())}/{len(gate_results)} gates).
- Selected spent-data closure: **{selected_closure}**, lambda **{selected_lambda:g}**.
- Stage-2 pooled CV RMS: baseline **{baseline_cv:.6g} arcsec**, selected **{selected_cv:.6g} arcsec** ({100*improvement:+.3f}%).
- Best isotropic control: **{best_isotropic:.6g} arcsec**; selected change **{100*improvement_vs_isotropic:+.3f}%**.
- Full-refit spent-heldout RMS: **{heldout_rms:.6g} arcsec** ({100*heldout_worsening:+.3f}% versus P0599).
- Sealed outcomes opened: **no**.

This is a closure-placement screen on spent RX J2129 data. It does not validate
new gravity or authorize unsealing P0640 by itself.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(json.dumps({"status": report["status"], "selection": report["selection"], "gates": gate_results}, indent=2))


if __name__ == "__main__":
    main()
