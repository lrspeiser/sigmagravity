#!/usr/bin/env python3
"""Fair geometry-refit cross-validation of the accumulated lens tensor."""

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

from run_adaptive_route_raw_rxj2129 import MODEL, baryon_field, exact_fit
from run_clash_stellar_morphology_response import MorphologyLens
from run_p0601_frozen_potential_raw_lensing import (
    build_fields as build_p0599_fields,
)
from run_p0607_component_direction_raw_lensing import fixed_geometry
from run_p0644_spent_rxj2129_accumulated_tensor import (
    make_field,
    read_json,
)
from run_rxj2129_member_geometry import split_images
from run_rxj2129_raw_theory_lensing import (
    load_baryonic_anchors,
    load_images,
    near_bound,
)

DEFAULT_CONFIG = ROOT / "configs" / "p0645_fair_geometry_cv_accumulated_tensor.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stratified_folds(training: pd.DataFrame, folds: int):
    assignment = {}
    families = sorted(training.source_family.unique())
    for family_offset, family in enumerate(families):
        block = training[training.source_family.eq(family)].sort_values("image_id")
        for within, row in enumerate(block.itertuples(index=False)):
            assignment[str(row.image_id)] = int((family_offset + within) % folds)
    frame = training.copy()
    frame["cv_fold"] = frame.image_id.map(assignment).astype(int)
    for fold in range(folds):
        heldout = frame[frame.cv_fold.eq(fold)].drop(columns="cv_fold")
        fit = frame[~frame.cv_fold.eq(fold)].drop(columns="cv_fold")
        if heldout.empty:
            raise RuntimeError(f"CV fold {fold} is empty")
        retained = fit.groupby("source_family").size()
        if set(retained.index) != set(families) or int(retained.min()) < 1:
            raise RuntimeError("a CV fold removed an entire source family")
        yield fold, fit, heldout


def pooled_rms(rows: list[dict]) -> float:
    finite = [row for row in rows if np.isfinite(row["validation_RMS_arcsec"])]
    if len(finite) != len(rows):
        return math.inf
    points = sum(int(row["validation_images"]) for row in finite)
    return math.sqrt(
        sum(
            int(row["validation_images"]) * float(row["validation_RMS_arcsec"]) ** 2
            for row in finite
        )
        / points
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    if protocol.get("status") != "frozen_before_any_P0645_score":
        raise RuntimeError("P0645 protocol is not frozen")
    p0644_protocol = read_json(ROOT / protocol["inputs"]["P0644_protocol"])
    p0644_report = read_json(ROOT / protocol["inputs"]["P0644_report"])
    if p0644_report["selection"]["spent_heldout_used_for_selection"]:
        raise RuntimeError("P0644 selection boundary changed")
    raw_protocol = read_json(ROOT / p0644_protocol["inputs"]["raw_protocol"])
    p0601_protocol = read_json(ROOT / p0644_protocol["inputs"]["P0601_protocol"])
    images = load_images(raw_protocol)
    training, spent_heldout = split_images(images, raw_protocol)
    anchors = load_baryonic_anchors(raw_protocol)
    radial_fields, _, _ = build_p0599_fields(
        anchors, raw_protocol, p0601_protocol["constants"]
    )
    parent = radial_fields["P0599_potential_shape"]
    initial = fixed_geometry(ROOT / p0644_protocol["inputs"]["P0601_parameters"])
    field_protocol = copy.deepcopy(p0644_protocol)
    field_protocol["candidate"].update(
        {
            "coherence_length_kpc": protocol["candidate"]["coherence_length_kpc"],
            "accumulation_power": protocol["candidate"]["accumulation_power"],
            "a0_m_s2": protocol["candidate"]["a0_m_s2"],
            "stellar_mass_fraction": protocol["candidate"]["stellar_mass_fraction"],
            "gas_mass_fraction": protocol["candidate"]["gas_mass_fraction"],
            "common_smoothing_kpc": protocol["candidate"]["common_physical_smoothing_kpc"],
        }
    )
    baryons = baryon_field(anchors, raw_protocol)
    field, map_audits = make_field(
        field_protocol, raw_protocol, anchors, parent, baryons, images
    )
    folds = list(stratified_folds(training, int(protocol["cross_validation"]["folds"])))
    fold_definition = [
        {
            "fold": fold,
            "fit_images": fit.image_id.tolist(),
            "validation_images": validation.image_id.tolist(),
        }
        for fold, fit, validation in folds
    ]
    fold_rows, prediction_frames, lambda_rows = [], [], []
    for lambda_offset, value in enumerate(protocol["candidate"]["lambda_grid"]):
        strength = float(value)
        local_rows = []
        for fold, fit_images, validation_images in folds:
            lens = MorphologyLens(
                raw_protocol,
                {"P0599_potential_shape": parent},
                parent="P0599_potential_shape",
                morphology=field if strength != 0.0 else None,
                fraction=strength,
            )
            try:
                fitted = exact_fit(
                    lens,
                    fit_images,
                    validation_images,
                    initial=initial,
                    starts=int(protocol["cross_validation"]["geometry_refit_starts_per_lambda_fold"]),
                    seed=int(protocol["cross_validation"]["random_seed"])
                    + 100 * lambda_offset
                    + fold,
                )
                score = fitted["heldout_score"]
                row = {
                    "lambda": strength,
                    "fold": fold,
                    "fit_images": len(fit_images),
                    "validation_images": len(validation_images),
                    "fit_RMS_arcsec": fitted["training_score"]["exact_radial_RMS_arcsec"],
                    "fit_roots": fitted["training_score"]["converged_roots"],
                    "validation_RMS_arcsec": score["exact_radial_RMS_arcsec"],
                    "validation_roots": score["converged_roots"],
                    "optimizer_cost": fitted["optimizer_cost"],
                }
                predictions = fitted["heldout_prediction"].copy()
                predictions["lambda"] = strength
                predictions["fold"] = fold
                prediction_frames.append(predictions)
            except Exception as error:  # noqa: BLE001 - topology failure is a scored result
                row = {
                    "lambda": strength,
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
            fold_rows.append(row)
            local_rows.append(row)
        total_validation = sum(row["validation_images"] for row in local_rows)
        total_roots = sum(row["validation_roots"] for row in local_rows)
        aggregate = {
            "lambda": strength,
            "pooled_CV_RMS_arcsec": pooled_rms(local_rows),
            "CV_roots": total_roots,
            "CV_images": total_validation,
            "all_CV_roots": total_roots == total_validation,
        }
        lambda_rows.append(aggregate)
        print(
            f"lambda={strength:g}: CV={aggregate['pooled_CV_RMS_arcsec']:.6g} "
            f"roots={total_roots}/{total_validation}",
            flush=True,
        )
    lambda_scores = pd.DataFrame(lambda_rows)
    eligible = lambda_scores[
        lambda_scores.all_CV_roots & np.isfinite(lambda_scores.pooled_CV_RMS_arcsec)
    ].sort_values(["pooled_CV_RMS_arcsec", "lambda"])
    if eligible.empty:
        raise RuntimeError("no lambda completed all CV roots")
    selected_lambda = float(eligible.iloc[0]["lambda"])
    selected_lens = MorphologyLens(
        raw_protocol,
        {"P0599_potential_shape": parent},
        parent="P0599_potential_shape",
        morphology=field if selected_lambda != 0.0 else None,
        fraction=selected_lambda,
    )
    full = exact_fit(
        selected_lens,
        training,
        spent_heldout,
        initial=initial,
        starts=int(protocol["cross_validation"]["full_training_refit_starts"]),
        seed=int(protocol["cross_validation"]["random_seed"]) + 10000,
    )
    full_predictions = pd.concat(
        [full["training_prediction"], full["heldout_prediction"]], ignore_index=True
    )
    lambda0 = float(lambda_scores.loc[lambda_scores["lambda"].eq(0.0), "pooled_CV_RMS_arcsec"].iloc[0])
    selected_cv = float(
        lambda_scores.loc[
            lambda_scores["lambda"].eq(selected_lambda), "pooled_CV_RMS_arcsec"
        ].iloc[0]
    )
    cv_improvement = 1.0 - selected_cv / lambda0
    baseline_heldout = float(p0644_report["comparators"]["P0599_spent_heldout_RMS_arcsec"])
    full_heldout = float(full["heldout_score"]["exact_radial_RMS_arcsec"])
    heldout_worsening = full_heldout / baseline_heldout - 1.0
    bound_flags = near_bound(MODEL, full["parameters"])
    gates = protocol["predeclared_progression_gates"]
    gate_results = {
        "CV_root_completion": bool(lambda_scores.all_CV_roots.all())
        is bool(gates["every_CV_root_converged"]),
        "CV_improvement": cv_improvement >= gates["CV_improvement_fraction_vs_lambda0_min"],
        "nonzero_lambda": (selected_lambda > 0.0) is bool(gates["selected_lambda_nonzero"]),
        "lambda_not_upper_endpoint": selected_lambda
        < max(float(value) for value in protocol["candidate"]["lambda_grid"]),
        "full_training_roots": int(full["training_score"]["converged_roots"])
        == int(gates["full_training_roots"]),
        "spent_heldout_roots": int(full["heldout_score"]["converged_roots"])
        == int(gates["spent_heldout_roots"]),
        "spent_heldout_not_worse": heldout_worsening
        <= gates["spent_heldout_worsening_fraction_vs_P0599_max"],
        "geometry_interior": (not any(bound_flags.values()))
        is (not bool(gates["ordinary_geometry_near_bound_allowed"])),
        "sealed_targets_untouched": not bool(gates["sealed_target_outcomes_opened"]),
    }
    report = {
        "report_version": "P0645-FAIR-GEOMETRY-CV-ACCUMULATED-TENSOR-RESULTS-1.0.0",
        "status": "pass" if all(gate_results.values()) else "fail",
        "all_progression_gates_pass": bool(all(gate_results.values())),
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__)),
        "coverage": {
            "CV_folds": len(folds),
            "lambda_rows": len(lambda_scores),
            "lambda_fold_refits": len(fold_rows),
            "CV_prediction_images": int(sum(row["validation_images"] for row in fold_rows)),
            "ordinary_geometry_parameters_refit_per_run": 6,
            "per_object_spatial_gravity_parameters": 0,
        },
        "fold_definition": fold_definition,
        "field_audit": field.audit,
        "map_audits": map_audits,
        "lambda_scores": lambda_scores.to_dict(orient="records"),
        "selection": {
            "selected_lambda": selected_lambda,
            "lambda0_CV_RMS_arcsec": lambda0,
            "selected_CV_RMS_arcsec": selected_cv,
            "CV_improvement_fraction_vs_lambda0": cv_improvement,
            "P0601_spent_heldout_used_for_selection": False,
        },
        "full_refit": {
            "training_RMS_arcsec": float(full["training_score"]["exact_radial_RMS_arcsec"]),
            "training_roots": int(full["training_score"]["converged_roots"]),
            "spent_heldout_RMS_arcsec": full_heldout,
            "spent_heldout_roots": int(full["heldout_score"]["converged_roots"]),
            "spent_heldout_worsening_fraction_vs_P0599": heldout_worsening,
            "near_bound": bound_flags,
        },
        "gate_results": gate_results,
        "sealed_P0633_kinematics_opened": False,
        "sealed_P0640_lensing_constraints_opened": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(output / "fold_scores.csv", index=False)
    lambda_scores.to_csv(output / "lambda_scores.csv", index=False)
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(
            output / "cv_predictions.csv", index=False
        )
    full_predictions.to_csv(output / "full_refit_predictions.csv", index=False)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(lambda_scores["lambda"], lambda_scores.pooled_CV_RMS_arcsec, "o-")
    axes[0].axvline(selected_lambda, color="black", linestyle="--", linewidth=1)
    axes[0].set(xlabel="universal lambda", ylabel="pooled CV RMS (arcsec)", title="Geometry refit in every fold")
    fold_frame = pd.DataFrame(fold_rows)
    for fold, block in fold_frame.groupby("fold"):
        axes[1].plot(block["lambda"], block.validation_RMS_arcsec, "o-", label=f"fold {fold}")
    axes[1].set(xlabel="universal lambda", ylabel="validation RMS (arcsec)", title="Fold stability")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "fair_geometry_cv.png", dpi=180)
    plt.close(figure)
    summary = f"""# P0645 fair geometry-refit CV

- Status: **{report['status'].upper()}** ({sum(gate_results.values())}/{len(gate_results)} gates).
- Training-internal CV selected lambda: **{selected_lambda:g}**.
- Pooled CV RMS: lambda=0 **{lambda0:.6g} arcsec**, selected **{selected_cv:.6g} arcsec** ({100*cv_improvement:+.3f}%).
- Full-refit spent-heldout RMS: **{full_heldout:.6g} arcsec** ({100*heldout_worsening:+.3f}% versus P0599).
- New validation outcomes opened: **no**.

All six conventional geometry/shear parameters were refit for every lambda and
fold. This remains development work on spent RX J2129 data.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(json.dumps({"status": report["status"], "selected_lambda": selected_lambda, "CV_improvement": cv_improvement, "gates": gate_results}, indent=2))


if __name__ == "__main__":
    main()
