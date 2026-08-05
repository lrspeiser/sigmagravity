import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19bb_abell2146_luminosity_current_ensemble.json"
SCRIPT = ROOT / "scripts" / "run_sigma_v19bb_abell2146_luminosity_current_ensemble.py"
SPEC = importlib.util.spec_from_file_location("sigma_v19bb", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_quantized_density_is_positive_and_symmetric():
    center = MODULE.quantized_position_pdf_arcsec2(
        0.0,
        0.0,
        east_half_width_arcsec=0.08,
        north_half_width_arcsec=0.18,
        sigma_arcsec=0.25,
    )
    positive = MODULE.quantized_position_pdf_arcsec2(
        0.3,
        -0.4,
        east_half_width_arcsec=0.08,
        north_half_width_arcsec=0.18,
        sigma_arcsec=0.25,
    )
    reflected = MODULE.quantized_position_pdf_arcsec2(
        -0.3,
        0.4,
        east_half_width_arcsec=0.08,
        north_half_width_arcsec=0.18,
        sigma_arcsec=0.25,
    )
    assert center > positive > 0.0
    assert np.isclose(positive, reflected)


def test_candidate_and_null_posteriors_normalize():
    candidate, null = MODULE.association_posterior([3.0, 0.5], 0.9)
    assert np.all(candidate >= 0.0)
    assert null >= 0.0
    assert np.isclose(float(np.sum(candidate)) + null, 1.0)
    assert candidate[0] > candidate[1] > null


def test_fold_assignment_is_deterministic_and_balanced():
    members = [str(value) for value in range(1, 64)]
    first = MODULE.fold_assignments(members, 7)
    second = MODULE.fold_assignments(list(reversed(members)), 7)
    assert first == second
    counts = np.bincount(list(first.values()), minlength=7)
    assert int(np.max(counts) - np.min(counts)) <= 1


def test_protocol_keeps_catalog_scale_separate_from_gravity():
    config = json.loads(CONFIG.read_text())
    assert config["status"].startswith("frozen_before")
    assert config["ensemble"]["draws"] == 8192
    assert config["population"]["expected_spectroscopic_members"] == 63
    assert config["catalog_level_astrometric_calibration"]["interpretation"].endswith(
        "Sigma wavelength"
    )
    assert not config["authorization"]["select_hard_counterparts"]
    assert not config["authorization"]["infer_missing_photometry_or_stellar_mass"]
    assert not config["authorization"]["impute_transverse_velocity"]
    assert not config["authorization"]["read_lensing_or_halo_payload"]
    assert not config["authorization"]["change_gravity_physics"]


def test_frozen_runner_hash_is_exact():
    config = json.loads(CONFIG.read_text())
    assert config["implementation"]["runner"] == str(SCRIPT.relative_to(ROOT)).replace("\\", "/")
    assert MODULE.sha256(SCRIPT) == config["implementation"]["runner_sha256"]
