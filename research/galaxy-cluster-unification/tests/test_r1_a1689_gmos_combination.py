from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def test_a1689_p2_combination_coverage_gate() -> None:
    report = json.loads((ROOT / "results/r1_a1689_gmos_combination/report.json").read_text())
    product = np.load(ROOT / report["output"])
    assert product["exposure_science_electron"].shape[0] == 4
    assert np.all(product["exposure_coverage"][product["combined_dq"] == 0] >= 3)
    assert np.all(product["combined_dq"][product["exposure_coverage"] < 3] & 16)
    assert report["all_subthreshold_pixels_masked"] is True
    assert report["gates"]["P2_calibrated_2d_sky_centroid_coverage_gate_passed"] is True
    assert report["authorization"]["fit_frozen_nine_signed_stellar_kinematic_bins"] is True
    assert report["gates"]["gravity_response_fit_authorized"] is False
