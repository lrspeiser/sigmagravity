#!/usr/bin/env python3
"""Audit the executable RX J2129 HST H2 freeze without reading arc pixels."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import run_r1_rxj2129_hst_h2 as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/r1_rxj2129_hst_h2_centroid_execution_protocol.json"
REPORT = ROOT / "results/r1_rxj2129_hst_h2_freeze/report.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def frozen_file(record: dict) -> bool:
    path = ROOT / record.get("path", record.get("report"))
    return bool(
        path.is_file()
        and path.stat().st_size == int(record["bytes"])
        and sha256(path) == record["sha256"]
    )


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    parent = json.loads((ROOT / config["parent_protocol"]["path"]).read_text())
    h1 = json.loads((ROOT / config["H1_gate"]["report"]).read_text())
    ledger_rule = config["inputs"]["coordinate_ledger"]
    ledger = pd.read_csv(ROOT / ledger_rule["path"], dtype={"image_id": str})
    selected = ledger[
        (ledger["likelihood_included"] == True)  # noqa: E712
        & (ledger["redshift_kind"] == "spectroscopic")
    ]
    selected_ids = set(selected["image_id"].astype(str))
    input_records = [config["parent_protocol"], config["H1_gate"]]
    input_records.extend(
        config["inputs"][name]
        for name in (
            "coordinate_ledger",
            "registration_draws",
            "union_segmentation",
            "PSF_field",
        )
    )
    self_test = runner.self_test(config)
    output_absent = all(
        not (ROOT / relative).exists()
        for relative in config["outputs"].values()
    )
    runner_hash = sha256(ROOT / config["implementation"]["runner"])
    gates = {
        "protocol_status_records_static_reaudit_pass_and_active_H2": config["status"]
        == "static_reaudit_pass_H2_execution_active_after_frame_correction",
        "all_frozen_input_hashes_match": all(frozen_file(record) for record in input_records),
        "parent_protocol_version_matches": parent["protocol_version"]
        == config["parent_protocol"]["required_version"],
        "H1_gate_passes_and_authorizes_H2": bool(
            h1["status"] == config["H1_gate"]["required_status"]
            and all(h1["gates"].values())
            and h1["authorization"]["execute_H2_arc_centroids"] is True
        ),
        "immutable_21_row_ledger_matches": len(selected) == int(ledger_rule["required_rows"]),
        "required_inner_images_are_present": set(ledger_rule["required_inner_images"]).issubset(selected_ids),
        "exact_37_pixel_stamp_geometry": bool(
            config["pixel_geometry"]["stamp_size_pixels"] == 37
            and config["pixel_geometry"]["stamp_width_arcsec"] == 2.4
            and config["pixel_geometry"]["pixel_scale_arcsec"] == 0.065
        ),
        "exact_three_component_twelve_start_model": bool(
            config["source_model"]["components"] == 3
            and config["source_model"]["optimizer_starts"] == 12
        ),
        "exact_500_draw_measurement_bootstrap": bool(
            config["bootstrap"]["draws"] == 500
            and config["bootstrap"]["minimum_successful_fraction"] == 0.95
        ),
        "unchanged_scientific_acceptance_gates": bool(
            config["image_acceptance"]["minimum_total_images_accepted"] == 18
            and config["image_acceptance"]["required_inner_images"] == ["5.2", "6.3", "8.2"]
            and config["image_acceptance"]["maximum_cross_band_centroid_difference_arcsec"] == 0.2
            and config["image_acceptance"]["minimum_source_flux_SNR_each_band"] == 10.0
            and config["image_acceptance"]["minimum_per_coordinate_standard_error_arcsec"] == 0.02
            and config["image_acceptance"]["maximum_per_coordinate_standard_error_arcsec"] == 0.3
        ),
        "runner_hash_matches_freeze": runner_hash == config["implementation"]["runner_sha256"],
        "pixel_free_synthetic_self_test_passes": self_test["status"] == "pass"
        and all(self_test["gates"].values())
        and self_test["HST_pixels_accessed"] is False,
        "no_H2_science_output_exists_before_authorization": output_absent,
        "lens_gravity_and_H3_remain_locked": bool(
            not config["authorization"]["assemble_H3_covariance"]
            and not config["authorization"]["use_lens_or_gravity_model"]
            and not config["authorization"]["infer_dynamical_or_Weyl_response"]
            and not config["authorization"]["fit_new_force_or_action"]
        ),
    }
    passed = bool(all(gates.values()))
    report = {
        "report_version": "R1B3-RXJ2129-HST-H2-freeze-0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if passed else "fail",
        "protocol_version": config["protocol_version"],
        "config_sha256": sha256(CONFIG),
        "runner_sha256": runner_hash,
        "HST_arc_pixels_accessed_before_original_runner_hash_freeze": False,
        "HST_arc_pixels_accessed_during_invalid_first_execution": True,
        "HST_arc_pixels_accessed_during_this_static_audit": False,
        "selected_ledger_rows": len(selected),
        "synthetic_self_test": self_test,
        "gates": gates,
        "authorization": {
            "execute_H2_arc_centroids": passed,
            "assemble_H3_covariance": False,
            "use_lens_or_gravity_model": False,
            "infer_dynamical_or_Weyl_response": False,
            "fit_new_force_or_action": False,
        },
        "next_action": (
            "Execute the exact hash-locked H2 runner on every immutable ledger row; preserve every failure and do not retune or move a stamp."
            if passed
            else "Correct only the failed pre-pixel implementation or provenance gate; do not read any arc stamp."
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
