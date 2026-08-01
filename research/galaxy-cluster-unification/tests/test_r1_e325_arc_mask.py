from __future__ import annotations

import json
from pathlib import Path

from astropy.io import fits


ROOT = Path(__file__).resolve().parents[1]


def test_e325_arc_mask_passes_blind_support_and_visit_controls_only() -> None:
    report = json.loads(
        (ROOT / "results/r1_e325_arc_mask/report.json").read_text()
    )
    masks = ROOT / report["outputs"]["masks_and_residuals"]

    assert report["selection_blind"] is True
    assert report["gravity_residuals_inspected"] is False
    assert report["detection"]["final_arc_pixels"] >= 50
    assert report["detection"]["retained_connected_components"] >= 2
    assert report["detection"]["azimuthal_span_deg"] >= 90.0
    assert report["detection"]["knot_neighborhoods_intersected"] == 4
    assert report["detection"]["negative_control_pixels"] == report["detection"]["final_arc_pixels"]
    assert all(
        visit["integrated_arc_signal_to_noise"] >= 5.0
        for visit in report["visit_controls"]["visits"]
    )
    assert 0.8 <= report["visit_controls"]["difference_reduced_chi_square"] <= 1.2
    assert all(report["gates"].values()) is False  # rank admission remains deliberately false
    assert report["gates"]["complete_arc_mask_gate_passed"] is True
    assert report["gates"]["rank_three_candidate_admission_passed"] is False
    assert report["authorization"]["implement_frozen_image_level_jacobian"] is True
    assert report["authorization"]["count_toward_ten_system_target"] is False
    with fits.open(masks) as hdul:
        arc = hdul["ARCMASK"].data.astype(bool)
        control = hdul["NEGCTRL"].data.astype(bool)
        assert int(arc.sum()) == int(control.sum())
        assert not (arc & control).any()
