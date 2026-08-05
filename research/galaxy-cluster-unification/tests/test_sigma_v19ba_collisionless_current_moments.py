import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19ba_collisionless_current_moments.json"
SCRIPT = ROOT / "scripts" / "run_sigma_v19ba_collisionless_current_moments.py"
SPEC = importlib.util.spec_from_file_location("sigma_v19ba", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_cloud_in_cell_is_positive_and_conservative():
    weights = MODULE.cic_weights(3.25, 4.75, 10, 12)
    assert len(weights) == 4
    assert all(weight >= 0.0 for _, _, weight in weights)
    assert np.isclose(sum(weight for _, _, weight in weights), 1.0)
    deposited = np.zeros((12, 10))
    for iy, ix, weight in weights:
        deposited[iy, ix] += 7.3 * weight
    assert np.isclose(np.sum(deposited), 7.3)


def test_positive_particle_moments_obey_cauchy_schwarz():
    luminosities = np.asarray([1.0, 2.0, 4.0])
    beta = np.asarray([-0.004, 0.001, 0.003])
    rho = float(np.sum(luminosities))
    current = float(np.sum(luminosities * beta))
    second = float(np.sum(luminosities * beta * beta))
    assert MODULE.normalized_cauchy_schwarz_margin(rho, current, second) >= 0.0
    coherent_beta = 0.002
    coherent_current = rho * coherent_beta
    coherent_second = rho * coherent_beta**2
    assert (
        abs(MODULE.normalized_cauchy_schwarz_margin(rho, coherent_current, coherent_second)) < 1e-15
    )


def test_protocol_constructs_source_moments_without_fitting_gravity():
    config = json.loads(CONFIG.read_text())
    assert config["status"].startswith("frozen_before")
    assert config["population"]["expected_ensemble_rows"] == 8192 * 72
    assert config["grid"]["expected_shape_yx"] == [626, 626]
    assert not config["authorization"]["apply_or_fit_long_wave_operator"]
    assert not config["authorization"]["infer_absolute_mass"]
    assert not config["authorization"]["impute_missing_luminosity_or_transverse_velocity"]
    assert not config["authorization"]["read_lensing_or_halo_payload"]
