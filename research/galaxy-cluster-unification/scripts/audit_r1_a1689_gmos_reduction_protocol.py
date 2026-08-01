#!/usr/bin/env python3
"""Audit that A1689 reduction/covariance choices were frozen before products."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/r1_a1689_gmos_reduction_covariance_protocol.json"
REPORT = ROOT / "results/r1_a1689_gmos_reduction_protocol/report.json"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest().upper()


def build_audit() -> dict:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw = json.loads((ROOT / config["prerequisite_reports"]["raw_acquisition"]).read_text())
    lens = json.loads((ROOT / config["prerequisite_reports"]["lens_geometry"]).read_text())
    template = ROOT / config["stellar_kinematic_fit"]["template_path"]
    result_root = ROOT / "data/derived/r1_a1689_gmos_reconstruction"
    current_files = sorted(
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in result_root.rglob("*") if path.is_file()
    ) if result_root.exists() else []
    science_stems = {Path(name).stem for name in config["raw_inputs"]["science_to_flat_arc_mapping"]}
    science_product_files = [
        name for name in current_files
        if any(Path(name).name.startswith(stem) for stem in science_stems)
    ]
    calibration_product_files = [name for name in current_files if name not in science_product_files]
    if REPORT.exists():
        previous = json.loads(REPORT.read_text(encoding="utf-8"))
        science_products_at_freeze = previous.get(
            "science_products_present_at_freeze", science_product_files
        )
    else:
        science_products_at_freeze = science_product_files

    edges = config["spatial_extraction"]["signed_bin_edges_arcsec"]
    raw_gate = bool(raw["gates"]["raw_acquisition_gate_passed"])
    geometry_gate = bool(lens["gates"]["same_radius_bridge_passed"])
    template_gate = template.exists() and sha256(template) == config["stellar_kinematic_fit"]["template_sha256"]
    frozen_gate = bool(
        config["status"].endswith("before_any_science_reduction_or_kinematic_extraction")
        and len(edges) == 10
        and edges == sorted(edges)
        and config["covariance_protocol"]["replicates"] == 200
        and config["profile_acceptance"]["minimum_distinct_radial_bins_for_structural_promotion"] == 3
        and len(science_products_at_freeze) == 0
    )
    report = {
        "report_version": "R1B1-A1689-GMOS-reduction-protocol-0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_blind": True,
        "raw_acquisition_gate_passed": raw_gate,
        "lens_geometry_gate_passed": geometry_gate,
        "template_checksum_passed": template_gate,
        "science_products_present_at_freeze": science_products_at_freeze,
        "science_products_currently_present": science_product_files,
        "calibration_products_currently_present": calibration_product_files,
        "protocol_frozen_before_science_products": frozen_gate,
        "signed_bins_frozen": len(edges) - 1,
        "symmetrized_radial_bins_frozen": 5,
        "bootstrap_replicates_frozen": config["covariance_protocol"]["replicates"],
        "gates": {
            "protocol_freeze_gate_passed": raw_gate and geometry_gate and template_gate and frozen_gate,
            "P1_environment_and_bpm_gate_passed": False,
            "P2_calibrated_2d_gate_passed": False,
            "P3_profile_covariance_gate_passed": False,
            "gravity_response_fit_authorized": False
        },
        "next_action": "Acquire and checksum the archive-valid GMOS-N EEV BPM, install the isolated DRAGONS environment, and audit raw metadata recognition before executing preprocessing.",
        "authorization": {
            "acquire_bpm_and_install_environment": raw_gate and geometry_gate and template_gate and frozen_gate,
            "execute_science_reduction": False,
            "fit_stellar_kinematics": False,
            "infer_gravity_response": False,
            "fit_new_force_or_action": False
        }
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build_audit(), indent=2))
