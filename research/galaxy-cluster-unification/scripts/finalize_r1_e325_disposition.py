#!/usr/bin/env python3
"""Finalize E325 after its preregistered coordinate-map gate failure."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = ROOT / "data/derived/r1_new_rank3_candidate_queue.csv"
REPORT_PATH = ROOT / "results/r1_e325_final_disposition/report.json"
INPUTS = {
    "feasibility": ROOT / "results/r1_e325_feasibility/report.json",
    "acquisition": ROOT / "results/r1_e325_acquisition/report.json",
    "preprocessing": ROOT / "results/r1_e325_hst_preprocessing/report.json",
    "arc_mask": ROOT / "results/r1_e325_arc_mask/report.json",
    "coordinate_fit": ROOT / "data/derived/r1_e325_coordinate_lens_fit.json",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    reports = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in INPUTS.items()}
    upstream_passes = {
        "feasibility_pre_pixel_gate": reports["feasibility"]["gates"]["pre_pixel_acquisition_and_jacobian_protocol_authorized"],
        "complete_acquisition_gate": reports["acquisition"]["gates"]["complete_acquisition_gate_passed"],
        "complete_preprocessing_gate": reports["preprocessing"]["gates"]["complete_preprocessing_gate_passed"],
        "complete_arc_mask_gate": reports["arc_mask"]["gates"]["complete_arc_mask_gate_passed"],
    }
    coordinate = reports["coordinate_fit"]
    coordinate_pass = coordinate["gates"]["coordinate_map_engineering_gate_passed"]
    if not all(upstream_passes.values()) or coordinate_pass:
        raise RuntimeError("E325 final disposition preconditions changed")
    queue = pd.read_csv(QUEUE_PATH, keep_default_na=False)
    selected = queue["system"] == "ESO 325-G004"
    if int(selected.sum()) != 1:
        raise RuntimeError("Expected exactly one E325 candidate-queue row")
    updates = {
        "full_image_level_structural_rank": "not_established_coordinate_map_gate_failed",
        "counts_toward_ten_system_target": False,
        "disposition": "acquired_extended_arc_control_not_rank_three_promotion",
        "primary_blocker": "frozen semilinear coordinate map fails unchanged source-support closure: constrained edge flux 0.06794 > 0.05 and mapped support crosses the fixed source boundary",
        "next_authorized_stage": "none_for_E325_promotion_retain_hash_locked_data_as_control",
    }
    for column, value in updates.items():
        queue.loc[selected, column] = value
    queue = queue.sort_values("system", kind="stable").reset_index(drop=True)
    queue.to_csv(QUEUE_PATH, index=False, lineterminator="\n")
    report = {
        "report_version": "R1-E325-final-disposition-0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "system": "ESO 325-G004",
        "selection_blind": True,
        "gravity_residuals_inspected": False,
        "inputs": {
            name: {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(path),
            }
            for name, path in INPUTS.items()
        },
        "completed_gates": upstream_passes,
        "failed_gate": {
            "name": "coordinate_map_engineering_gate",
            "passed": coordinate_pass,
            "masked_chi_square_per_pixel": coordinate["fit"]["chi_square_per_masked_pixel"],
            "data_constrained_source_edge_absolute_flux_fraction": coordinate["metric_correction"]["corrected_source_edge_absolute_flux_fraction"],
            "maximum_allowed_source_edge_absolute_flux_fraction": 0.05,
            "image_mapped_source_boundary_reached": coordinate["metric_correction"]["image_mapped_source_boundary_reached"],
            "optimizer_or_threshold_rerun_after_failure": False,
        },
        "decision": "retain_as_hash_locked_extended_arc_control_not_a_rank_three_promotion",
        "ten_system_effect": {
            "previous_structural_ceiling": 3,
            "updated_structural_ceiling": 3,
            "minimum_new_rank_three_systems_still_required": 7,
            "strict_ready_systems": 0,
        },
        "new_source_search": {
            "external_candidates_completed": 2,
            "external_candidates_promoted": 0,
            "completed_candidates": ["SDSS J0946+1006", "ESO 325-G004"],
            "next_named_source_class": "the 14-lens SLACS-KCWI spatially resolved kinematics sample",
            "next_filter": "screen primary-source HST arc geometry, accepted KCWI dynamics support, public numerical kinematics/covariance, and rerunnable observable likelihood before downloading science arrays",
        },
        "outputs": {
            "candidate_queue": str(QUEUE_PATH.relative_to(ROOT)).replace("\\", "/"),
            "candidate_queue_sha256": sha256(QUEUE_PATH),
        },
        "authorization": {
            "continue_E325_promotion_work": False,
            "retain_E325_data_as_lower_rank_control": True,
            "screen_SLACS_KCWI_primary_source_and_archive_metadata": True,
            "count_E325_toward_ten_system_target": False,
            "freeze_ten_system_sample": False,
            "infer_gravity_response": False,
            "fit_new_force_or_action": False,
            "authorize_R2": False,
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
