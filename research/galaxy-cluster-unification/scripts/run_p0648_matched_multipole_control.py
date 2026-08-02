#!/usr/bin/env python3
"""Compare the rejected component flux with matched conventional multipoles."""

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
from run_p0601_frozen_potential_raw_lensing import build_fields as build_p0599_fields
from run_p0607_component_direction_raw_lensing import fixed_geometry
from run_p0644_spent_rxj2129_accumulated_tensor import make_field, read_json
from run_p0645_fair_geometry_cv_accumulated_tensor import stratified_folds
from run_p0646_conservative_closure_atlas import evaluate_candidate
from run_rxj2129_member_geometry import split_images
from run_rxj2129_raw_theory_lensing import load_baryonic_anchors, load_images

from voidscreen.multipole_lensing import build_matched_multipole_deflection_field

DEFAULT_CONFIG = ROOT / "configs" / "p0648_matched_multipole_control.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    if protocol.get("status") != "frozen_before_any_P0648_score":
        raise RuntimeError("P0648 protocol is not frozen")

    p0644 = read_json(ROOT / protocol["inputs"]["P0644_protocol"])
    p0645_protocol = read_json(ROOT / protocol["inputs"]["P0645_protocol"])
    p0645_report = read_json(ROOT / protocol["inputs"]["P0645_report"])
    p0647_report = read_json(ROOT / protocol["inputs"]["P0647_report"])
    p0607_report = read_json(ROOT / protocol["inputs"]["P0607_report"])
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

    star = p0607_report["direction_alignment"]["star_map_centroid_arcsec"]
    gas = p0607_report["direction_alignment"]["gas_map_centroid_arcsec"]
    measured_phase = math.atan2(float(gas[1]) - float(star[1]), float(gas[0]) - float(star[0]))
    controls = protocol["control_fields"]
    if not np.isclose(measured_phase, float(controls["phase_rad"]), atol=1e-12):
        raise RuntimeError("frozen control phase no longer matches the P0607 component centroids")

    candidate_protocol = copy.deepcopy(p0644)
    candidate_protocol["candidate"].update(
        {
            "coherence_length_kpc": 10.0,
            "accumulation_power": 1.0,
            "a0_m_s2": 1.2e-10,
            "stellar_mass_fraction": 0.1,
            "gas_mass_fraction": 0.9,
            "common_smoothing_kpc": 25.0,
            "closure": "gas_minus_star_flux",
        }
    )
    candidate_field, _ = make_field(
        candidate_protocol, raw_protocol, anchors, parent, baryons, images
    )
    target_rms = float(controls["target_unit_deflection_RMS_arcsec"])
    if not np.isclose(
        float(candidate_field.audit["unit_deflection_RMS_arcsec"]), target_rms, rtol=1e-12
    ):
        raise RuntimeError("matched control normalization no longer matches P0646")
    fields = {
        int(order): build_matched_multipole_deflection_field(
            candidate_field.axis_arcsec,
            order=int(order),
            phase_rad=float(controls["phase_rad"]),
            radial_scale_arcsec=float(controls["radial_scale_arcsec"]),
            taper_inner_arcsec=float(controls["taper_inner_arcsec"]),
            support_radius_arcsec=float(controls["support_radius_arcsec"]),
            target_deflection_rms_arcsec=target_rms,
        )
        for order in controls["orders"]
    }

    cv = protocol["cross_validation"]
    stage1_summaries, fold_rows, prediction_frames = [], [], []
    for order_offset, order in enumerate(controls["orders"]):
        for amplitude_offset, amplitude in enumerate(controls["signed_amplitude_grid"]):
            summary, rows, predictions = evaluate_candidate(
                f"multipole_m{int(order)}",
                float(amplitude),
                fields[int(order)],
                parent,
                raw_protocol,
                folds,
                initial,
                starts=int(cv["stage1_geometry_refit_starts"]),
                seed=int(cv["random_seed"]) + 1000 * order_offset + 100 * amplitude_offset,
                stage="stage1",
            )
            stage1_summaries.append({**summary, "order": int(order)})
            fold_rows.extend(rows)
            prediction_frames.extend(predictions)
            print(
                f"stage1 m={int(order)} amplitude={float(amplitude):g}: "
                f"CV={summary['pooled_CV_RMS_arcsec']:.6g} "
                f"roots={summary['CV_roots']}/{summary['CV_images']}",
                flush=True,
            )
    stage1 = pd.DataFrame(stage1_summaries)
    shortlist = []
    for order in controls["orders"]:
        eligible = stage1[
            stage1.order.eq(int(order))
            & stage1.all_CV_roots
            & np.isfinite(stage1.pooled_CV_RMS_arcsec)
        ].copy()
        if eligible.empty:
            continue
        eligible["absolute_amplitude"] = np.abs(eligible["lambda"])
        eligible = eligible.sort_values(
            ["pooled_CV_RMS_arcsec", "absolute_amplitude", "lambda"]
        )
        shortlist.extend(
            eligible.head(int(cv["stage2_rows_per_order"]))[["order", "lambda"]].to_dict(
                orient="records"
            )
        )

    stage2_summaries = []
    for offset, row in enumerate(shortlist):
        order = int(row["order"])
        amplitude = float(row["lambda"])
        summary, rows, predictions = evaluate_candidate(
            f"multipole_m{order}",
            amplitude,
            fields[order],
            parent,
            raw_protocol,
            folds,
            initial,
            starts=int(cv["stage2_geometry_refit_starts"]),
            seed=int(cv["random_seed"]) + 50000 + 100 * offset,
            stage="stage2",
        )
        stage2_summaries.append({**summary, "order": order})
        fold_rows.extend(rows)
        prediction_frames.extend(predictions)
        print(
            f"stage2 m={order} amplitude={amplitude:g}: "
            f"CV={summary['pooled_CV_RMS_arcsec']:.6g} "
            f"roots={summary['CV_roots']}/{summary['CV_images']}",
            flush=True,
        )
    stage2 = pd.DataFrame(stage2_summaries)
    eligible2 = stage2[
        stage2.all_CV_roots & np.isfinite(stage2.pooled_CV_RMS_arcsec)
    ].copy()
    if eligible2.empty:
        selected = None
        best_control_cv = math.inf
        selected_order = None
        selected_amplitude = None
        full = None
        full_predictions = pd.DataFrame()
    else:
        eligible2["absolute_amplitude"] = np.abs(eligible2["lambda"])
        selected = eligible2.sort_values(
            ["pooled_CV_RMS_arcsec", "absolute_amplitude", "lambda", "order"]
        ).iloc[0]
        best_control_cv = float(selected.pooled_CV_RMS_arcsec)
        selected_order = int(selected["order"])
        selected_amplitude = float(selected["lambda"])
        lens = MorphologyLens(
            raw_protocol,
            {"P0599_potential_shape": parent},
            parent="P0599_potential_shape",
            morphology=fields[selected_order],
            fraction=selected_amplitude,
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

    baseline_cv = float(p0645_report["selection"]["lambda0_CV_RMS_arcsec"])
    candidate_cv = float(p0647_report["selection"]["selected_CV_RMS_arcsec"])
    margin = float(protocol["predeclared_interpretation"]["candidate_specificity_margin_fraction"])
    specificity_survives = bool(candidate_cv <= best_control_cv * (1.0 - margin))
    generic_explains = bool(best_control_cv <= candidate_cv * (1.0 + margin))
    endpoints = {
        min(float(value) for value in controls["signed_amplitude_grid"]),
        max(float(value) for value in controls["signed_amplitude_grid"]),
    }
    amplitude_identified = bool(
        selected_amplitude is not None
        and not any(np.isclose(selected_amplitude, endpoint) for endpoint in endpoints)
    )
    field_gates = protocol["field_gates"]
    field_gate_results = {
        f"m{order}_curl": float(field.audit["normalized_curl_RMS"])
        <= field_gates["normalized_curl_RMS_max"]
        for order, field in fields.items()
    }
    field_gate_results.update(
        {
            f"m{order}_source": float(field.audit["source_integral_fraction"])
            <= field_gates["source_integral_fraction_max"]
            for order, field in fields.items()
        }
    )
    field_gate_results.update(
        {
            f"m{order}_normalization": abs(
                float(field.audit["unit_deflection_RMS_arcsec"]) / target_rms - 1.0
            )
            <= field_gates["unit_RMS_relative_match_error_max"]
            for order, field in fields.items()
        }
    )
    field_gate_results["sealed_targets_untouched"] = not bool(
        field_gates["sealed_target_outcomes_opened"]
    )

    full_report = None
    if full is not None:
        full_report = {
            "training_RMS_arcsec": float(full["training_score"]["exact_radial_RMS_arcsec"]),
            "training_roots": int(full["training_score"]["converged_roots"]),
            "spent_heldout_RMS_arcsec": float(full["heldout_score"]["exact_radial_RMS_arcsec"]),
            "spent_heldout_roots": int(full["heldout_score"]["converged_roots"]),
        }
    report = {
        "report_version": "P0648-MATCHED-MULTIPOLE-CONTROL-RESULTS-1.0.0",
        "status": "complete_spent_control",
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__)),
        "coverage": {
            "multipole_orders": len(fields),
            "stage1_rows": len(stage1),
            "stage1_fold_refits": len(stage1) * len(folds),
            "stage2_rows": len(stage2),
            "ordinary_geometry_parameters_refit_per_run": 6,
            "multipole_amplitude_parameters": 1,
            "per_object_spatial_gravity_parameters": 0,
        },
        "field_audits": {f"m{order}": field.audit for order, field in fields.items()},
        "field_gate_results": field_gate_results,
        "stage1_scores": stage1.to_dict(orient="records"),
        "stage1_shortlist": shortlist,
        "stage2_scores": stage2.to_dict(orient="records"),
        "comparison": {
            "lambda0_CV_RMS_arcsec": baseline_cv,
            "P0647_boundary_candidate_CV_RMS_arcsec": candidate_cv,
            "best_multipole_CV_RMS_arcsec": best_control_cv,
            "best_multipole_order": selected_order,
            "best_multipole_amplitude": selected_amplitude,
            "candidate_fractional_improvement_vs_best_multipole": (
                1.0 - candidate_cv / best_control_cv if np.isfinite(best_control_cv) else -math.inf
            ),
            "candidate_specificity_survives": specificity_survives,
            "generic_angular_control_explains_gain": generic_explains,
            "multipole_amplitude_identified": amplitude_identified,
            "P0647_candidate_was_already_rejected": True,
            "P0601_spent_heldout_used_for_selection": False,
        },
        "full_refit": full_report,
        "sealed_P0633_kinematics_opened": False,
        "sealed_P0640_lensing_constraints_opened": False,
        "claim_boundary": protocol["claim_boundary"],
    }

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    stage1.to_csv(output / "stage1_scores.csv", index=False)
    stage2.to_csv(output / "stage2_scores.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "fold_scores.csv", index=False)
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(
            output / "cv_predictions.csv", index=False
        )
    if not full_predictions.empty:
        full_predictions.to_csv(output / "full_refit_predictions.csv", index=False)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for order, block in stage1.groupby("order"):
        ordered = block.sort_values("lambda")
        finite = ordered[np.isfinite(ordered.pooled_CV_RMS_arcsec)]
        axes[0].plot(
            finite["lambda"],
            finite.pooled_CV_RMS_arcsec,
            "o-",
            label=f"m={int(order)}",
        )
    axes[0].axhline(baseline_cv, color="black", linestyle="--", label="lambda 0")
    axes[0].axhline(candidate_cv, color="tab:red", linestyle=":", label="P0647 boundary")
    axes[0].set(
        xlabel="signed matched amplitude",
        ylabel="stage-1 pooled CV RMS (arcsec)",
        title="Conventional multipole screen",
    )
    axes[0].legend(fontsize=8)
    labels = [f"m={int(row.order)}\na={float(row['lambda']):g}" for _, row in stage2.iterrows()]
    axes[1].bar(labels, stage2.pooled_CV_RMS_arcsec)
    axes[1].axhline(candidate_cv, color="tab:red", linestyle=":", label="P0647 boundary")
    axes[1].set(ylabel="stage-2 pooled CV RMS (arcsec)", title="Exact matched controls")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "matched_multipole_control.png", dpi=180)
    plt.close(figure)

    summary = f"""# P0648 matched multipole control

- Best exact control: **m={selected_order}**, amplitude **{selected_amplitude}**.
- CV RMS: lambda-zero **{baseline_cv:.6g} arcsec**, P0647 boundary candidate **{candidate_cv:.6g} arcsec**, multipole **{best_control_cv:.6g} arcsec**.
- Candidate specificity survives the frozen 1% margin: **{specificity_survives}**.
- Generic angular control explains the gain: **{generic_explains}**.
- Multipole amplitude is interior: **{amplitude_identified}**.
- Sealed outcomes opened: **no**.

P0647 remains rejected regardless of this control result. This experiment asks
whether its attractive spent-data score was specific to its baryonic geometry.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(json.dumps({"comparison": report["comparison"], "field_gates": field_gate_results}, indent=2))


if __name__ == "__main__":
    main()
