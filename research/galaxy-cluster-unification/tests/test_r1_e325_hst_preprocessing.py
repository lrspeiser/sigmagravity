from __future__ import annotations

import json
from pathlib import Path

from astropy.io import fits


ROOT = Path(__file__).resolve().parents[1]


def test_e325_hst_common_grid_is_blind_complete_and_not_a_rank_promotion() -> None:
    report = json.loads(
        (ROOT / "results/r1_e325_hst_preprocessing/report.json").read_text()
    )
    cutouts = ROOT / report["outputs"]["registered_cutouts"]

    assert report["selection_blind"] is True
    assert report["image_morphology_inspected"] is False
    assert report["gravity_residuals_inspected"] is False
    assert report["common_grid"]["shape_yx"] == [256, 256]
    assert report["common_grid"]["pixel_scale_arcsec"] == [0.05, 0.05]
    assert len(report["visit_metrics"]) == 4
    assert all(item["science_finite_fraction"] == 1.0 for item in report["visit_metrics"])
    assert all(item["positive_weight_fraction"] == 1.0 for item in report["visit_metrics"])
    assert report["coadd_metrics"]["F475W"]["exposure_seconds"] == 4800.0
    assert report["coadd_metrics"]["F814W"]["exposure_seconds"] == 18882.0
    assert report["psf_family"]["members"] == 36
    assert report["gates"]["complete_preprocessing_gate_passed"] is True
    assert report["gates"]["rank_three_candidate_admission_passed"] is False
    assert report["authorization"]["freeze_arc_and_negative_control_masks"] is True
    assert report["authorization"]["implement_frozen_image_level_jacobian"] is False
    with fits.open(cutouts) as hdul:
        assert len(hdul) == 13
        assert all(name in hdul for name in ("F475COA", "F475WHT", "F814COA", "F814WHT"))
