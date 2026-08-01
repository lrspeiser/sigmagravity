#!/usr/bin/env python3
"""Audit the independent RX J2129 XMM/HST execution freezes without pixel access."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
XMM = ROOT / "configs/r1_rxj2129_xmm_event_processing_protocol.json"
HST = ROOT / "configs/r1_rxj2129_hst_centroid_covariance_protocol.json"
ENV = ROOT / "results/r1_rxj2129_xmm_environment/report.json"
REPORT = ROOT / "results/r1_rxj2129_xmm_hst_execution_protocols/report.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> None:
    xmm = json.loads(XMM.read_text(encoding="utf-8"))
    hst = json.loads(HST.read_text(encoding="utf-8"))
    env = json.loads(ENV.read_text(encoding="utf-8"))
    lens_rows = [
        line
        for line in (ROOT / hst["inputs"]["coordinate_ledger"]).read_text().splitlines()[1:]
        if line
    ]
    included = [line for line in lens_rows if ",spectroscopic,True," in line]
    hst_checksums = {}
    for band in ("F814W", "F125W"):
        item = hst["inputs"][band]
        hst_checksums[f"{band}_science"] = (
            sha256(ROOT / item["path"]) == item["sha256"]
        )
        hst_checksums[f"{band}_weight"] = (
            sha256(ROOT / item["weight_path"]) == item["weight_sha256"]
        )
    xmm_gate = bool(
        env["gates"]["R1B3_XMM_reduction_environment_gate_passed"]
        and (
            xmm["status"].endswith(
                "before_ODF_decompression_cifbuild_odfingest_or_event_array_access"
            )
            or xmm["status"]
            == "frozen_before_event_array_analysis_with_post_execution_SAS_OOT_semantics_correction_logged"
        )
        and xmm["inputs"]["primary_EPIC_exposures"]
        == {"MOS1": "M1S001", "MOS2": "M2S002", "pn": "PNS003"}
        and xmm["flare_filter"]["time_bin_seconds"] == 100
        and xmm["flare_filter"]["minimum_instruments_passing"] == 2
        and xmm["spectral_annuli"]["minimum_accepted_annuli"] == 5
        and xmm["spectral_annuli"]["posterior_draws"] == 2000
        and not xmm["authorization"]["inspect_or_tune_on_cluster_profile"]
        and not xmm["authorization"]["infer_gas_likelihood_before_X2"]
        and not xmm["authorization"]["fit_new_force_or_action"]
    )
    hst_gate = bool(
        all(hst_checksums.values())
        and hst["status"]
        in {
            "refrozen_with_exact_H1_detection_registration_segmentation_and_PSF_algorithms_before_H1_pixel_measurement",
            "refrozen_after_background_coverage_engineering_stop_and_before_H1_detection",
            "refrozen_after_v0_3_H1_engineering_audit_before_corrected_H1_rerun",
        }
        and len(included) == 21
        and hst["inputs"]["required_inner_images"] == ["5.2", "6.3", "8.2"]
        and hst["registration"]["bootstrap_draws"] == 500
        and hst["H1_execution_freeze"]["compact_source_detection"][
            "minimum_total_flux_SNR"
        ]
        == 20.0
        and hst["H1_execution_freeze"]["matching_and_affine_fit"][
            "bootstrap_draws"
        ]
        == 500
        and hst["H1_execution_freeze"]["spatial_PSF"]["bootstrap_draws"] == 500
        and hst["centroid_model"]["bootstrap_draws"] == 500
        and hst["covariance"]["required_shape"] == [42, 42]
        and hst["covariance"]["minimum_all_field_images_accepted"] == 18
        and hst["covariance"]["required_inner_image_acceptance"] == 3
        and not hst["authorization"]["use_lens_or_gravity_model"]
        and not hst["authorization"]["fit_new_force_or_action"]
    )
    report = {
        "report_version": "R1B3-RXJ2129-XMM-HST-execution-freezes-0.3",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_blind": True,
        "XMM_event_arrays_inspected_at_original_freeze": False,
        "XMM_event_arrays_processed_now": (
            ROOT / "results/r1_rxj2129_xmm_x3_annular_products/report.json"
        ).is_file(),
        "HST_arc_pixels_measured": False,
        "HST_H1_pixel_access_started_now": True,
        "HST_H1_source_detection_completed_now": True,
        "hst_checksums": hst_checksums,
        "spectroscopic_lens_images": len(included),
        "gates": {
            "XMM_event_processing_protocol_frozen": xmm_gate,
            "HST_centroid_covariance_protocol_frozen": hst_gate,
            "independent_execution_freezes_passed": xmm_gate and hst_gate,
        },
        "authorization": {
            "execute_declared_XMM_X1_calibration": xmm_gate,
            "execute_declared_HST_H1_registration_mask_and_PSF": hst_gate,
            "infer_gas_likelihood": False,
            "assemble_HST_covariance": False,
            "infer_dynamical_or_Weyl_response": False,
            "fit_new_force_or_action": False,
        },
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
