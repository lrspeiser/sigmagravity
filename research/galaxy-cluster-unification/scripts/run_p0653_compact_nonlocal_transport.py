#!/usr/bin/env python3
"""Test compactly supported nonlocal streamline transport on spent lensing."""

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

DEFAULT_CONFIG = ROOT / "configs" / "p0653_compact_nonlocal_transport.json"


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
    if protocol.get("status") != "frozen_before_any_P0653_score":
        raise RuntimeError("P0653 protocol is not frozen")

    p0644 = read_json(ROOT / protocol["inputs"]["P0644_protocol"])
    p0645_protocol = read_json(ROOT / protocol["inputs"]["P0645_protocol"])
    p0645_report = read_json(ROOT / protocol["inputs"]["P0645_report"])
    p0648_report = read_json(ROOT / protocol["inputs"]["P0648_report"])
    p0651_report = read_json(ROOT / protocol["inputs"]["P0651_report"])
    p0652_report = read_json(ROOT / protocol["inputs"]["P0652_report"])
    if not all(p0651_report["map_gate_results"].values()):
        raise RuntimeError("registered map gates no longer pass")
    if p0652_report["gate_results"]["field_source_integral"]:
        raise RuntimeError("P0652 no longer exhibits the source leakage P0653 corrects")

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

    candidate = protocol["candidate"]
    field_protocol = copy.deepcopy(p0644)
    field_protocol["candidate"].update(
        {
            "coherence_length_kpc": candidate["coherence_length_kpc"],
            "accumulation_power": candidate["accumulation_power"],
            "a0_m_s2": candidate["a0_m_s2"],
            "stellar_mass_fraction": candidate["stellar_mass_fraction"],
            "gas_mass_fraction": candidate["gas_mass_fraction"],
            "common_smoothing_kpc": candidate["common_physical_smoothing_kpc"],
            "mismatch_mode": candidate["mismatch_mode"],
            "closure": candidate["closure"],
            "transport_steps": candidate["transport_steps"],
            "taper_inner_arcsec": candidate["taper_inner_arcsec"],
            "support_radius_arcsec": candidate["support_radius_arcsec"],
        }
    )
    field, map_audits = make_field(
        field_protocol, raw_protocol, anchors, parent, baryons, images
    )
    amplitude = float(candidate["field_amplitude"])
    cv = protocol["cross_validation"]
    cv_summary, fold_rows, prediction_frames = evaluate_candidate(
        candidate["closure"],
        amplitude,
        field,
        parent,
        raw_protocol,
        folds,
        initial,
        starts=int(cv["geometry_refit_starts_per_fold"]),
        seed=int(cv["random_seed"]),
        stage="exact_compact_nonlocal",
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
    gates = protocol["predeclared_progression_gates"]
    audit = field.audit
    gate_results = {
        "inherited_map_gates": all(p0651_report["map_gate_results"].values())
        is bool(gates["P0651_map_gates_all_pass"]),
        "CV_roots": int(cv_summary["CV_roots"]) == int(gates["all_CV_roots"]),
        "CV_improvement": improvement >= gates["CV_improvement_fraction_vs_lambda0_min"],
        "beats_matched_multipole": improvement_vs_multipole
        >= gates["CV_improvement_fraction_vs_best_matched_multipole_min"],
        "full_training_roots": int(full["training_score"]["converged_roots"])
        == int(gates["full_training_roots"]),
        "spent_heldout_roots": int(full["heldout_score"]["converged_roots"])
        == int(gates["spent_heldout_roots"]),
        "spent_heldout_not_worse": heldout_worsening
        <= gates["spent_heldout_worsening_fraction_vs_P0599_max"],
        "field_curl": float(audit["normalized_curl_RMS"])
        <= gates["field_normalized_curl_RMS_max"],
        "field_source_integral": float(audit["source_integral_fraction"])
        <= gates["field_source_integral_fraction_max"],
        "edge_flux_closed": float(audit["maximum_flux_edge_fraction_of_RMS"])
        <= gates["maximum_flux_edge_fraction_of_RMS_max"],
        "transport_nontrivial": float(audit["transport_relative_change_RMS"])
        >= gates["transport_relative_change_RMS_min"],
        "compact_taper_applied": bool(audit.get("post_transport_compact_taper_applied", False))
        is bool(gates["post_transport_compact_taper_required"]),
        "amplitude_is_one": bool(np.isclose(amplitude, 1.0))
        is bool(gates["field_amplitude_exactly_one"]),
        "sealed_targets_untouched": not bool(gates["sealed_target_outcomes_opened"]),
    }
    all_pass = bool(all(gate_results.values()))
    report = {
        "report_version": "P0653-COMPACT-NONLOCAL-TRANSPORT-RESULTS-1.0.0",
        "status": "pass" if all_pass else "fail",
        "all_progression_gates_pass": all_pass,
        "candidate_advanced_to_robustness": all_pass,
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__)),
        "coverage": {
            "candidate_fields": 1,
            "amplitude_rows": 1,
            "CV_folds": len(folds),
            "fitted_field_amplitude_parameters": 0,
            "new_physical_length_constants": 0,
            "per_object_spatial_gravity_parameters": 0,
        },
        "field_audit": audit,
        "map_audits": map_audits,
        "CV_summary": cv_summary,
        "comparison": {
            "lambda0_CV_RMS_arcsec": baseline_cv,
            "candidate_CV_RMS_arcsec": candidate_cv,
            "CV_improvement_fraction_vs_lambda0": improvement,
            "best_matched_multipole_CV_RMS_arcsec": multipole_cv,
            "CV_improvement_fraction_vs_best_matched_multipole": improvement_vs_multipole,
            "P0652_open_transport_CV_RMS_arcsec": float(
                p0652_report["comparison"]["primary_CV_RMS_arcsec"]
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
        "gate_results": gate_results,
        "sealed_P0633_kinematics_opened": False,
        "sealed_P0640_lensing_constraints_opened": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(output / "fold_scores.csv", index=False)
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(
            output / "cv_predictions.csv", index=False
        )
    full_predictions.to_csv(output / "full_refit_predictions.csv", index=False)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    fold_frame = pd.DataFrame(fold_rows)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = ["lambda 0", "matched m=3", "P0652 open", "P0653 compact"]
    values = [
        baseline_cv,
        multipole_cv,
        p0652_report["comparison"]["primary_CV_RMS_arcsec"],
        candidate_cv,
    ]
    axes[0].bar(labels, values)
    axes[0].set(ylabel="pooled exact CV RMS (arcsec)", title="Boundary conservation repair")
    axes[0].tick_params(axis="x", labelrotation=25)
    axes[1].bar(fold_frame["fold"].astype(str), fold_frame.validation_RMS_arcsec)
    axes[1].set(
        xlabel="source-family fold",
        ylabel="validation RMS (arcsec)",
        title="Compact nonlocal fold stability",
    )
    figure.tight_layout()
    figure.savefig(output / "compact_nonlocal_transport.png", dpi=180)
    plt.close(figure)
    summary = f"""# P0653 compact nonlocal transport

- Status: **{report['status'].upper()}** ({sum(gate_results.values())}/{len(gate_results)} gates).
- CV RMS: **{candidate_cv:.6g} arcsec** ({100*improvement:+.3f}% versus lambda zero).
- Source integral fraction: **{audit['source_integral_fraction']:.3e}**.
- Edge flux fraction: **{audit['maximum_flux_edge_fraction_of_RMS']:.3e}**.
- Spent-heldout RMS: **{heldout_rms:.6g} arcsec** ({100*heldout_worsening:+.3f}% versus P0599).
- Candidate advanced to robustness: **{report['candidate_advanced_to_robustness']}**.
- Sealed outcomes opened: **no**.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(json.dumps({"status": report["status"], "comparison": report["comparison"], "gates": gate_results}, indent=2))


if __name__ == "__main__":
    main()
