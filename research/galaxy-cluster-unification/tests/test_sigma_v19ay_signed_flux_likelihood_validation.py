import importlib.util
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19ay_signed_flux_likelihood_validation.json"
SCRIPT = ROOT / "scripts" / "run_sigma_v19ay_signed_flux_likelihood_validation.py"
SPEC = importlib.util.spec_from_file_location("sigma_v19ay", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_profiled_amplitude_recovers_a_scaled_template():
    template = np.asarray([0.8, 1.0, 1.2, 1.3])
    flux = 4.0 * template
    uncertainty = np.ones(4)
    amplitude, chi2 = MODULE.profile_amplitude(flux, uncertainty, template)
    assert np.isclose(amplitude, 4.0)
    assert np.isclose(chi2, 0.0)


def test_negative_flux_is_retained_as_finite_evidence():
    colors = np.asarray([0.2, 0.1, 0.05])
    flux = np.asarray([2.0, -1.0, 1.5, 0.5])
    uncertainty = np.ones(4)
    score = MODULE.signed_flux_log_score(
        flux,
        uncertainty,
        colors,
        np.asarray([0.05, 0.05, 0.05]),
        5,
    )
    assert math.isfinite(score)
    assert score <= 0


def test_color_quadrature_weights_are_normalized():
    templates = MODULE.quadrature_templates(np.zeros(3), np.ones(3), 5)
    assert len(templates) == 125
    assert np.isclose(sum(math.exp(weight) for weight, _ in templates), 1.0)


def test_v19ay_cannot_open_ambiguous_candidates_or_mass():
    config = json.loads(CONFIG.read_text())
    assert config["status"] == "frozen_before_flux_space_validation_scoring"
    assert config["likelihood"]["gauss_hermite_order_per_color"] == 5
    assert config["validation_gates"]["minimum_top1_retrievals"] == 3
    assert config["validation_gates"]["minimum_mean_reciprocal_rank"] == 0.65
    assert not config["authorization"]["score_ambiguous_candidates"]
    assert not config["authorization"]["select_or_rank_counterparts"]
    assert not config["authorization"]["infer_mass_or_current"]
