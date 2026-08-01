#!/usr/bin/env python3
"""Audit frozen A2537 control calibrations before science processing."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import audit_r1_a2261_gmos_calibrations as shared


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/r1_a2537_gmos_reduction_covariance_protocol.json"
RAW = ROOT / "data/raw/r1_a2537_gemini"
CAL = ROOT / "data/derived/r1_a2537_gmos_control/calibrations"
REPORT = ROOT / "results/r1_a2537_gmos_calibrations/report.json"


def main() -> None:
    environment = json.loads((ROOT / "results/r1_a2537_dragons_environment/report.json").read_text())
    if not environment["authorization"]["execute_frozen_C2_calibration_reduction"]:
        raise RuntimeError("C1 did not authorize A2537 control calibrations")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    shared.RAW = RAW
    shared.CAL = CAL
    biases, bias_construction, bias_residuals = shared.audit_biases(config)
    flats, flat_gate = shared.audit_flats(config)
    arcs, arc_gate = shared.audit_arcs(config)
    gate = bool(bias_construction and bias_residuals and flat_gate and arc_gate)
    report = {
        "report_version": "R1B2-A2537-GMOS-control-calibrations-0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_blind": True,
        "disturbed_control": True,
        "counts_as_non_disturbed_pilot": False,
        "scope": "calibrations_only_before_any_science_frame_processing",
        "biases": biases,
        "flats": flats,
        "arcs": arcs,
        "gates": {
            "bias_construction_passed": bias_construction,
            "bias_overscan_residuals_passed": bias_residuals,
            "flat_normalization_passed": flat_gate,
            "arc_wavelength_solutions_passed": arc_gate,
            "C2a_calibration_products_gate_passed": gate,
            "C2_calibrated_2d_gate_passed": False,
            "C3_profile_covariance_gate_passed": False,
            "gravity_response_fit_authorized": False
        },
        "authorization": {
            "process_four_science_frames_with_frozen_mapping": gate,
            "fit_stellar_kinematics": False,
            "count_as_non_disturbed_pilot": False,
            "fit_new_force_or_action": False
        },
        "next_action": "Only if C2a passes, calibrate all four science frames independently with the frozen mapping and then audit exact history/provenance before any sky or pPXF operation." if gate else "Stop the A2537 control at the exact failed calibration threshold without tuning."
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
