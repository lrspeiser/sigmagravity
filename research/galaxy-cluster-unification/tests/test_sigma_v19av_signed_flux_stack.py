import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19av_signed_flux_stack.json"
SCRIPT = ROOT / "scripts" / "run_sigma_v19av_signed_flux_stack.py"
SPEC = importlib.util.spec_from_file_location("sigma_v19av", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_stack_includes_negative_flux_and_downweights_outlier():
    flux = np.asarray([10.0, 11.0, 9.0, -2.0, 1000.0])
    uncertainty = np.ones(5)
    result = MODULE.robust_stack(flux, uncertainty, 2.5, 10)
    assert result["finite_exposures"] == 5
    assert result["huber_downweighted_exposures"] >= 2
    assert 8.0 < result["stacked_flux"] < 13.0
    assert result["stacked_signal_to_noise"] > 0


def test_flux_normalization_preserves_characterization_magnitude():
    row = {
        "flux": "100",
        "flux_uncertainty": "2",
        "magzero": "25",
    }
    flux, sigma = MODULE.normalize_flux(row, 30.0)
    original_magnitude = 25.0 - 2.5 * np.log10(100.0)
    normalized_magnitude = 30.0 - 2.5 * np.log10(flux)
    assert np.isclose(original_magnitude, normalized_magnitude)
    assert np.isclose(sigma / flux, 0.02)


def test_v19av_is_not_a_candidate_association_protocol():
    config = json.loads(CONFIG.read_text())
    assert config["stack"]["candidate_detection_signal_to_noise"] == 3.0
    assert config["gates"]["minimum_candidate_complete_griz_fraction"] == 0.90
    assert not config["authorization"]["compare_candidate_colors_to_bri"]
    assert not config["authorization"]["combine_candidate_stack_with_positional_posterior"]
    assert not config["authorization"]["select_or_rank_counterparts"]
