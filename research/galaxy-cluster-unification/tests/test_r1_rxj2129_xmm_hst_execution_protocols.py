from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_rxj2129_XMM_and_HST_execution_protocols_are_independently_frozen() -> None:
    report = json.loads(
        (ROOT / "results/r1_rxj2129_xmm_hst_execution_protocols/report.json").read_text()
    )
    assert report["XMM_event_arrays_inspected_at_original_freeze"] is False
    assert report["XMM_event_arrays_processed_now"] is True
    assert report["HST_arc_pixels_measured"] is False
    assert report["HST_H1_pixel_access_started_now"] is True
    assert report["HST_H1_source_detection_completed_now"] is True
    assert report["spectroscopic_lens_images"] == 21
    assert all(report["hst_checksums"].values())
    assert report["gates"]["XMM_event_processing_protocol_frozen"] is True
    assert report["gates"]["HST_centroid_covariance_protocol_frozen"] is True
    assert report["gates"]["independent_execution_freezes_passed"] is True
    assert report["authorization"]["execute_declared_XMM_X1_calibration"] is True
    assert report["authorization"]["execute_declared_HST_H1_registration_mask_and_PSF"] is True
    assert report["authorization"]["infer_gas_likelihood"] is False
    assert report["authorization"]["fit_new_force_or_action"] is False
