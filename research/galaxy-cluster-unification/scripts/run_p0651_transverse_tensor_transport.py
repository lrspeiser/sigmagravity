#!/usr/bin/env python3
"""Map-screen and spent-lens test a bounded transverse tensor invariant."""

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
from run_p0649_bounded_angle_transport_screen import evaluate, observed_suite, synthetic_suite
from run_rxj2129_member_geometry import split_images
from run_rxj2129_raw_theory_lensing import load_baryonic_anchors, load_images

DEFAULT_CONFIG = ROOT / "configs" / "p0651_transverse_tensor_transport.json"


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
    if protocol.get("status") != "frozen_before_any_P0651_score":
        raise RuntimeError("P0651 protocol is not frozen")

    synthetic = synthetic_suite(protocol)
    observed = observed_suite(protocol)
    map_gates, map_metrics, grouped = evaluate(protocol, synthetic, observed)
    if not all(map_gates.values()):
        report = {
            "report_version": "P0651-TRANSVERSE-TENSOR-TRANSPORT-RESULTS-1.0.0",
            "status": "fail_map_stage",
            "all_progression_gates_pass": False,
            "candidate_advanced_to_robustness": False,
            "protocol_sha256": sha256(config_path),
            "source_sha256": sha256(Path(__file__)),
            "map_gate_results": map_gates,
            "map_metrics": map_metrics,
            "spent_lens_stage_run": False,
            "sealed_P0633_kinematics_opened": False,
            "sealed_P0640_lensing_constraints_opened": False,
            "claim_boundary": protocol["claim_boundary"],
        }
        write_outputs(protocol, synthetic, observed, grouped, report)
        print(json.dumps({"status": report["status"], "map_gates": map_gates}, indent=2))
        return

    p0644 = read_json(ROOT / protocol["inputs"]["P0644_protocol"])
    p0645_protocol = read_json(ROOT / protocol["inputs"]["P0645_protocol"])
    p0645_report = read_json(ROOT / protocol["inputs"]["P0645_report"])
    p0648_report = read_json(ROOT / protocol["inputs"]["P0648_report"])
    p0650_report = read_json(ROOT / protocol["inputs"]["P0650_report"])
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

    fixed = protocol["fixed_field"]
    field_protocol = copy.deepcopy(p0644)
    field_protocol["candidate"].update(
        {
            "coherence_length_kpc": fixed["coherence_length_kpc"],
            "accumulation_power": fixed["accumulation_power"],
            "a0_m_s2": fixed["a0_m_s2"],
            "stellar_mass_fraction": fixed["stellar_mass_fraction"],
            "gas_mass_fraction": fixed["gas_mass_fraction"],
            "common_smoothing_kpc": fixed["common_physical_smoothing_kpc"],
            "mismatch_mode": fixed["primary_mode"],
            "closure": fixed["closure"],
        }
    )
    field, map_audits = make_field(
        field_protocol, raw_protocol, anchors, parent, baryons, images
    )
    amplitude = float(fixed["field_amplitude"])
    cv = protocol["cross_validation"]
    cv_summary, fold_rows, prediction_frames = evaluate_candidate(
        "bounded_transverse_tensor_flux",
        amplitude,
        field,
        parent,
        raw_protocol,
        folds,
        initial,
        starts=int(cv["geometry_refit_starts_per_fold"]),
        seed=int(cv["random_seed"]),
        stage="exact_amplitude_one",
    )
    candidate_cv = float(cv_summary["pooled_CV_RMS_arcsec"])
    baseline_cv = float(p0645_report["selection"]["lambda0_CV_RMS_arcsec"])
    multipole_cv = float(p0648_report["comparison"]["best_multipole_CV_RMS_arcsec"])
    improvement = 1.0 - candidate_cv / baseline_cv
    improvement_vs_multipole = 1.0 - candidate_cv / multipole_cv

    lens = MorphologyLens(
        raw_protocol,
        {"P0599_potential_shape": parent},
        parent="P0599_potential_shape",
        morphology=field,
        fraction=amplitude,
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
    limits = protocol["predeclared_spent_lens_gates"]
    audit = field.audit
    lens_gates = {
        "CV_roots": int(cv_summary["CV_roots"]) == int(limits["all_CV_roots"]),
        "CV_improvement": improvement >= limits["CV_improvement_fraction_vs_lambda0_min"],
        "beats_matched_multipole": improvement_vs_multipole
        >= limits["CV_improvement_fraction_vs_best_matched_multipole_min"],
        "full_training_roots": int(full["training_score"]["converged_roots"])
        == int(limits["full_training_roots"]),
        "spent_heldout_roots": int(full["heldout_score"]["converged_roots"])
        == int(limits["spent_heldout_roots"]),
        "spent_heldout_not_worse": heldout_worsening
        <= limits["spent_heldout_worsening_fraction_vs_P0599_max"],
        "field_curl": float(audit["normalized_curl_RMS"])
        <= limits["field_normalized_curl_RMS_max"],
        "field_source_integral": float(audit["source_integral_fraction"])
        <= limits["field_source_integral_fraction_max"],
        "amplitude_is_one": bool(np.isclose(amplitude, 1.0))
        is bool(limits["field_amplitude_exactly_one"]),
        "sealed_targets_untouched": not bool(limits["sealed_target_outcomes_opened"]),
    }
    all_pass = bool(all(map_gates.values()) and all(lens_gates.values()))
    report = {
        "report_version": "P0651-TRANSVERSE-TENSOR-TRANSPORT-RESULTS-1.0.0",
        "status": "pass" if all_pass else "fail_spent_lens_stage",
        "all_progression_gates_pass": all_pass,
        "candidate_advanced_to_robustness": all_pass,
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__)),
        "coverage": {
            "registered_galaxies": 13,
            "registered_clusters": 4,
            "candidate_fields": 1,
            "amplitude_rows": 1,
            "CV_folds": len(folds),
            "fitted_field_amplitude_parameters": 0,
            "per_object_spatial_gravity_parameters": 0,
        },
        "map_gate_results": map_gates,
        "map_metrics": map_metrics,
        "spent_lens_stage_run": True,
        "lens_gate_results": lens_gates,
        "field_audit": audit,
        "map_audits": map_audits,
        "CV_summary": cv_summary,
        "comparison": {
            "lambda0_CV_RMS_arcsec": baseline_cv,
            "candidate_CV_RMS_arcsec": candidate_cv,
            "CV_improvement_fraction_vs_lambda0": improvement,
            "best_matched_multipole_CV_RMS_arcsec": multipole_cv,
            "CV_improvement_fraction_vs_best_matched_multipole": improvement_vs_multipole,
            "P0650_linear_chord_CV_RMS_arcsec": float(
                p0650_report["comparison"]["candidate_CV_RMS_arcsec"]
            ),
            "P0601_spent_heldout_used_for_selection": False,
        },
        "full_refit": {
            "training_RMS_arcsec": float(full["training_score"]["exact_radial_RMS_arcsec"]),
            "training_roots": int(full["training_score"]["converged_roots"]),
            "spent_heldout_RMS_arcsec": heldout_rms,
            "spent_heldout_roots": int(full["heldout_score"]["converged_roots"]),
            "spent_heldout_worsening_fraction_vs_P0599": heldout_worsening,
        },
        "sealed_P0633_kinematics_opened": False,
        "sealed_P0640_lensing_constraints_opened": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    write_outputs(
        protocol,
        synthetic,
        observed,
        grouped,
        report,
        fold_rows=fold_rows,
        prediction_frames=prediction_frames,
        full_predictions=full_predictions,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "map_metrics": map_metrics,
                "comparison": report["comparison"],
                "lens_gates": lens_gates,
            },
            indent=2,
        )
    )


def write_outputs(
    protocol,
    synthetic,
    observed,
    grouped,
    report,
    *,
    fold_rows=None,
    prediction_frames=None,
    full_predictions=None,
):
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    synthetic.to_csv(output / "synthetic_scores.csv", index=False)
    observed.to_csv(output / "registered_map_scores.csv", index=False)
    grouped.to_csv(output / "map_domain_summary.csv", index=False)
    if fold_rows is not None:
        pd.DataFrame(fold_rows).to_csv(output / "fold_scores.csv", index=False)
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(
            output / "cv_predictions.csv", index=False
        )
    if full_predictions is not None:
        full_predictions.to_csv(output / "full_refit_predictions.csv", index=False)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    primary = observed[observed.scenario.eq("nominal")]
    primary.pivot(index="case", columns="domain", values="activation_weighted_mean").plot.bar(
        ax=axes[0], logy=True
    )
    axes[0].set(title="Bounded map activation", ylabel="activation")
    if fold_rows is not None:
        folds = pd.DataFrame(fold_rows)
        axes[1].bar(folds["fold"].astype(str), folds.validation_RMS_arcsec)
        axes[1].set(
            xlabel="source-family fold",
            ylabel="validation RMS (arcsec)",
            title="Amplitude-one exact CV",
        )
    else:
        axes[1].text(0.5, 0.5, "spent lens not run", ha="center", va="center")
        axes[1].set_axis_off()
    figure.tight_layout()
    figure.savefig(output / "transverse_tensor_transport.png", dpi=180)
    plt.close(figure)
    summary = f"""# P0651 transverse tensor transport

- Status: **{report['status']}**.
- Map gates passed: **{sum(report['map_gate_results'].values())}/{len(report['map_gate_results'])}**.
- Spent-lens stage run: **{report['spent_lens_stage_run']}**.
- Candidate advanced to robustness: **{report['candidate_advanced_to_robustness']}**.
- Sealed outcomes opened: **no**.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
