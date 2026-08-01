#!/usr/bin/env python3
"""Audit the frozen MS2137 numerical protocol without opening science arrays."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import ppxf


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/r1_ms2137_ppxf_covariance_protocol.json"
REPORT_PATH = ROOT / "results/r1_ms2137_ppxf_protocol/report.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def build_report() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    feasibility = json.loads((ROOT / config["parent_gates"]["feasibility_report"]).read_text(encoding="utf-8"))
    acquisition = json.loads((ROOT / config["parent_gates"]["acquisition_report"]).read_text(encoding="utf-8"))
    input_cfg = config["input"]
    fit_cfg = config["spectral_fit"]
    cube_path = ROOT / input_cfg["cube_path"]
    xsl_path = ROOT / fit_cfg["template_path"]
    emiles_path = ROOT / fit_cfg["diagnostic_template_path"]
    edges = config["spatial_extraction"]["annulus_edges_arcsec"]
    sensitivities = config["covariance_protocol"]["sensitivity_protocols"]

    gates = {
        "status_frozen_before_array_read": config["status"] == "frozen_before_first_MS2137_DATA_or_STAT_array_read",
        "parent_feasibility_gate_passed": feasibility["decision"] == config["parent_gates"]["required_feasibility_decision"],
        "parent_acquisition_gate_passed": acquisition["decision"] == config["parent_gates"]["required_acquisition_decision"],
        "acquisition_confirms_no_pixel_array_read": acquisition["pixel_arrays_inspected"] is False,
        "cube_checksum_matches": sha256(cube_path) == input_cfg["cube_sha256"],
        "xsl_checksum_matches": sha256(xsl_path) == fit_cfg["template_sha256"],
        "emiles_checksum_matches": sha256(emiles_path) == fit_cfg["diagnostic_template_sha256"],
        "ppxf_version_matches": ppxf.__version__ == fit_cfg["software_version"],
        "nine_annuli_end_at_frozen_support": len(edges) - 1 == 9 and edges[0] == 0.0 and edges[-1] == 14.0,
        "published_ms2137_polynomial_orders_frozen": fit_cfg["additive_polynomial_degree"] == 5 and fit_cfg["multiplicative_polynomial_degree"] == 3,
        "resolution_valid_xsl_is_baseline": fit_cfg["template_family_baseline"] == "XSL",
        "sensitivity_grid_has_five_covariance_and_four_interaction_runs": sum(bool(item["covariance"]) for item in sensitivities) == 5 and sum(not bool(item["covariance"]) for item in sensitivities) == 4,
        "outer_support_cannot_be_shrunk": config["structural_target"]["no_support_shrink_after_pixel_inspection"] is True and config["authorization"]["change_support_or_thresholds_after_result"] is False,
        "theory_fits_unauthorized": not any(config["authorization"][key] for key in ("infer_dynamical_or_weyl_response", "fit_gravity_response", "fit_new_force_or_action")),
    }
    passed = all(gates.values())
    report = {
        "report_version": config["protocol_version"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "system": config["input"]["system_name"],
        "science_or_variance_arrays_read": False,
        "ppxf_version": ppxf.__version__,
        "annulus_edges_arcsec": edges,
        "sensitivity_protocol_count": len(sensitivities),
        "gates": {**gates, "protocol_freeze_gate_passed": passed},
        "decision": "authorize_P2_geometry_and_signal" if passed else "stop_MS2137_protocol_freeze_failure",
        "next_action": "Read DATA/STAT only to execute the frozen P2 centroid, mask, validity, opposite-half population, and S/N gate; do not run pPXF unless P2 passes." if passed else "Correct only source-traceable protocol serialization or environment errors; do not read science arrays.",
        "authorization": {
            "execute_P2_geometry_and_signal": passed,
            "execute_P3_ppxf": False,
            "execute_P4_covariance": False,
            "infer_dynamical_or_weyl_response": False,
            "fit_gravity_response": False,
            "fit_new_force_or_action": False
        }
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2))
