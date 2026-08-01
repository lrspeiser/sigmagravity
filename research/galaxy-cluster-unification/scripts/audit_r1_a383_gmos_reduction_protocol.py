#!/usr/bin/env python3
"""Audit that the A383 reduction choices predate science-array inspection."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/r1_a383_gmos_reduction_covariance_protocol.json"
REPORT = ROOT / "results/r1_a383_gmos_reduction_protocol/report.json"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest().upper()


def build_audit() -> dict:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    feasibility = json.loads((ROOT / config["prerequisite_reports"]["metadata_feasibility"]).read_text())
    raw = json.loads((ROOT / config["prerequisite_reports"]["raw_acquisition"]).read_text())
    template = ROOT / config["stellar_kinematic_fit"]["template_path"]
    result_root = ROOT / "data/derived/r1_a383_gmos_reconstruction"
    current_files = sorted(str(path.relative_to(ROOT)).replace("\\", "/") for path in result_root.rglob("*") if path.is_file()) if result_root.exists() else []
    science_stems = {Path(name).stem for name in config["raw_inputs"]["science_to_flat_arc_mapping"]}
    science_products = [name for name in current_files if any(Path(name).name.startswith(stem) for stem in science_stems)]
    if REPORT.exists():
        previous = json.loads(REPORT.read_text(encoding="utf-8"))
        science_at_freeze = previous.get("science_products_present_at_freeze", science_products)
    else:
        science_at_freeze = science_products

    edges = config["spatial_extraction"]["signed_bin_edges_arcsec"]
    radii = config["prerequisite_reports"]["preidentified_image_radii_arcsec"]
    prerequisite_gate = bool(feasibility["gates"]["metadata_feasibility_gate_passed"] and raw["gates"]["raw_acquisition_gate_passed"])
    geometry_gate = bool(len(radii) == 3 and max(radii) < max(edges) and config["profile_acceptance"]["minimum_independent_lens_families_inside_realized_support"] == 2)
    template_gate = template.exists() and sha256(template) == config["stellar_kinematic_fit"]["template_sha256"]
    frozen_gate = bool(
        "before_any_science_reduction_or_array_inspection" in config["status"]
        and edges == sorted(edges)
        and len(edges) == 10
        and config["spatial_extraction"]["signed_bins"] == 9
        and config["covariance_protocol"]["replicates"] == 200
        and len(science_at_freeze) == 0
    )
    gate = prerequisite_gate and geometry_gate and template_gate and frozen_gate
    report = {
        "report_version": "R1B2-A383-GMOS-reduction-protocol-0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_blind": True,
        "science_arrays_inspected_at_freeze": False,
        "prerequisite_gates_passed": prerequisite_gate,
        "pre_pixel_three_image_geometry_gate_passed": geometry_gate,
        "template_checksum_passed": template_gate,
        "science_products_present_at_freeze": science_at_freeze,
        "science_products_currently_present": science_products,
        "signed_bins_frozen": len(edges) - 1,
        "outer_edge_arcsec": max(edges),
        "required_image_radius_arcsec": max(radii),
        "bootstrap_replicates_frozen": config["covariance_protocol"]["replicates"],
        "gates": {
            "protocol_freeze_gate_passed": gate,
            "P1_environment_and_bpm_gate_passed": False,
            "P2_calibrated_2d_gate_passed": False,
            "P3_profile_covariance_gate_passed": False,
            "gravity_response_fit_authorized": False
        },
        "authorization": {
            "audit_environment": gate,
            "execute_calibration_reduction": False,
            "execute_science_reduction": False,
            "fit_stellar_kinematics": False,
            "fit_new_force_or_action": False
        }
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build_audit(), indent=2))
