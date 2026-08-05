import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_sigma_v19as_decam_forced_photometry_development.py"
SPEC = importlib.util.spec_from_file_location("sigma_v19as", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_protocol_keeps_validation_sealed_and_counts_exact():
    config = json.loads(
        (ROOT / "configs" / "sigma_v19as_decam_forced_photometry_development.json").read_text()
    )
    assert len(config["split"]["development_ids"]) == 10
    assert len(config["split"]["validation_ids"]) == 5
    assert set(config["split"]["development_ids"]).isdisjoint(config["split"]["validation_ids"])
    assert config["gates"]["exact_development_measurements"] == 670
    assert config["gates"]["exact_development_image_groups"] == 122
    assert config["gates"]["validation_anchors_measured"] == 0
    assert not config["authorization"]["open_or_measure_validation_anchor_pixels"]
    assert not config["authorization"]["read_lensing_or_halo_payload"]


def test_area_scaled_and_rotation_reconstruct_symmetric_contamination():
    yy, xx = np.indices((51, 51), dtype=float)
    center = (25.0, 25.0)
    target = 100.0 * np.exp(-0.5 * (((xx - 25.0) / 4.0) ** 2 + ((yy - 25.0) / 3.0) ** 2))
    contaminant_mask = ((xx - 29.0) ** 2 + (yy - 25.0) ** 2) <= 2.5**2
    measured = target.copy()
    measured[contaminant_mask] += 500.0
    result = MODULE.aperture_fluxes(measured, center, 10.0, contaminant_mask)
    truth = float(target[np.hypot(xx - 25.0, yy - 25.0) <= 10.0].sum())
    raw = result["raw"][0]
    area_scaled = result["area_scaled"][0]
    rotate180 = result["rotate180"][0]
    assert raw > truth
    assert abs(rotate180 - truth) / truth < 0.02
    assert abs(rotate180 - truth) < abs(area_scaled - truth)


def test_recommendation_prioritizes_completeness_then_repeatability():
    rows = [
        {
            "variant": "less_complete",
            "aperture_diameter_arcsec": 4.0,
            "valid_measurement_fraction": 0.99,
            "complete_griz_development_objects": 9,
            "median_repeatability_scatter_mag": 0.01,
            "leave_one_out_color_mae_mag": 0.01,
            "median_absolute_catalog_aper4_delta_mag": 0.01,
        },
        {
            "variant": "complete",
            "aperture_diameter_arcsec": 4.0,
            "valid_measurement_fraction": 0.95,
            "complete_griz_development_objects": 10,
            "median_repeatability_scatter_mag": 0.10,
            "leave_one_out_color_mae_mag": 0.10,
            "median_absolute_catalog_aper4_delta_mag": 0.10,
        },
    ]
    assert MODULE.choose_recommendation(rows, 10)["variant"] == "complete"
