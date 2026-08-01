#!/usr/bin/env python3
"""Close J1402 promotion and apply its frozen external-search rethink rule."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "r1_j1402_final_disposition" / "report.json"
INPUTS = {
    "J0946": ROOT / "results" / "r1_j0946_jackpot_feasibility" / "report.json",
    "E325": ROOT / "results" / "r1_e325_final_disposition" / "report.json",
    "J1402_replay": ROOT / "results" / "r1_j1402_dinos_replay" / "report.json",
    "J1402_predictive_controls": ROOT
    / "results"
    / "r1_j1402_dinos_predictive_controls"
    / "report.json",
    "J1402_protocol": ROOT
    / "configs"
    / "r1_j1402_acquisition_replay_jacobian_protocol.json",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    documents = {name: load(path) for name, path in INPUTS.items()}
    replay = documents["J1402_replay"]
    predictive = documents["J1402_predictive_controls"]
    protocol = documents["J1402_protocol"]

    if not replay["exact_replay_gate_pass"]:
        raise RuntimeError("J1402 exact replay is not complete")
    if predictive["predictive_coordinate_gate_pass"]:
        raise RuntimeError("J1402 predictive gate passed; final failure disposition is invalid")
    if predictive["checks"]["maximum_coherent_heldout_residual_passes"]:
        raise RuntimeError("the frozen coherent-residual failure is absent")
    if not predictive["checks"][
        "maximum_six_sector_heldout_reduced_chi_square_passes"
    ]:
        raise RuntimeError("J1402 also failed the pixelwise sector gate")
    if not predictive["checks"][
        "every_coordinate_corruption_worsens_heldout_likelihood"
    ]:
        raise RuntimeError("J1402 also failed the coordinate negative controls")
    if documents["J0946"]["authorization"]["count_toward_ten_system_target"]:
        raise RuntimeError("J0946 unexpectedly counts toward the ten-system target")
    if documents["E325"]["authorization"]["count_E325_toward_ten_system_target"]:
        raise RuntimeError("E325 unexpectedly counts toward the ten-system target")

    baseline = predictive["released_and_corrupted_results"]["baseline"]
    report = {
        "report_version": "R1-J1402-final-disposition-0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "system": "SDSS J1402+6321",
        "selection_blind": True,
        "gravity_residuals_inspected": False,
        "inputs": {
            name: {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": digest(path),
            }
            for name, path in INPUTS.items()
        },
        "completed_gates": {
            "checksum_locked_acquisition": True,
            "exact_stored_chain_likelihood_replay": True,
            "full_mask_reduced_chi_square": True,
            "six_sector_pixelwise_prediction": True,
            "all_coordinate_negative_controls": True,
        },
        "failed_gate": {
            "name": "maximum_coherent_heldout_residual_sigma",
            "passed": False,
            "observed_sigma": baseline[
                "maximum_PSF_matched_coherent_residual_sigma"
            ],
            "maximum_allowed_sigma": predictive["predictive_thresholds"][
                "maximum_coherent_heldout_residual_sigma"
            ],
            "worst_sector": 0,
            "worst_band": "F435W",
            "optimizer_mask_or_threshold_rerun_after_failure": False,
        },
        "useful_surviving_result": {
            "aggregate_heldout_reduced_chi_square": baseline[
                "aggregate_reduced_chi_square"
            ],
            "maximum_sector_reduced_chi_square": baseline[
                "maximum_sector_reduced_chi_square"
            ],
            "interpretation": "the released model carries substantial image-level predictive information and identifies the released astrometry, but its coherent residual structure exceeds the frozen closure requirement",
        },
        "decision": "stop_J1402_before_lens_response_Jacobian_and_KCWI_reduction",
        "external_search_checkpoint": {
            "completed_candidates": [
                "SDSS J0946+1006",
                "ESO 325-G004",
                "SDSS J1402+6321",
            ],
            "promoted_candidates": [],
            "frozen_rethink_triggered": True,
            "trigger_text": protocol["concrete_outcomes"]["rethink_checkpoint"],
            "next_action": "Do not select a fourth external one-off target. Finish the already active RX J2129 strict-observable package, then reassess the ten-system public-data premise and report a formal sample-size/identifiability ceiling before any further acquisition or gravity-law work.",
        },
        "authorization": {
            "continue_J1402_promotion_work": False,
            "compute_J1402_lens_response_Jacobian": False,
            "reduce_J1402_KCWI": False,
            "count_J1402_toward_ten_system_target": False,
            "select_fourth_external_one_off_candidate": False,
            "continue_RXJ2129_strict_observable_package": True,
            "reassess_ten_system_public_data_premise": True,
            "infer_dynamical_or_Weyl_response": False,
            "fit_gravity_response": False,
            "fit_new_force_or_action": False,
            "authorize_R2": False,
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["failed_gate"], indent=2))
    print(report["decision"])
    print(report["external_search_checkpoint"]["next_action"])


if __name__ == "__main__":
    main()
