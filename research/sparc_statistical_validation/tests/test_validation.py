import math
import importlib.util
from pathlib import Path
import sys

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "run_validation.py"
SPEC = importlib.util.spec_from_file_location("sparc_statistical_run_validation", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
VALIDATION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VALIDATION
SPEC.loader.exec_module(VALIDATION)

A0 = VALIDATION.A0
G_DAGGER = VALIDATION.G_DAGGER
evaluate_configuration = VALIDATION.evaluate_configuration
h_function = VALIDATION.h_function
load_raw_curves = VALIDATION.load_raw_curves
parse_table1 = VALIDATION.parse_table1
predict_acceleration_only = VALIDATION.predict_acceleration_only
predict_sigma = VALIDATION.predict_sigma
prepare_curve = VALIDATION.prepare_curve


def test_table1_fixed_width_parser() -> None:
    metadata = parse_table1()
    assert len(metadata) == 175
    assert metadata["CamB"].distance_mpc == 3.36
    assert metadata["CamB"].inclination_deg == 65.0
    assert metadata["CamB"].rdisk_kpc == 0.47
    assert metadata["CamB"].quality == 2


def test_h_at_transition_and_asymptotic_direction() -> None:
    assert math.isclose(float(h_function(G_DAGGER)), 0.5, rel_tol=1e-12)
    values = h_function(np.asarray([1e-13, G_DAGGER, 1e-8]))
    assert values[0] > values[1] > values[2]


def test_rdisk_l0_and_n_are_exactly_inert() -> None:
    radius = np.asarray([1.0, 2.0, 5.0])
    velocity = np.asarray([40.0, 60.0, 80.0])
    first = predict_sigma(radius, velocity, rdisk_kpc=0.2, l0_kpc=0.4, n_exponent=0.27)
    second = predict_sigma(radius, velocity, rdisk_kpc=20.0, l0_kpc=400.0, n_exponent=-9.0)
    assert np.array_equal(first, second)


def test_distance_rescaling_preserves_baryonic_acceleration() -> None:
    curve = next(item for item in load_raw_curves() if item.name == "NGC2403")
    central = prepare_curve(curve, 0.5, 0.7, 1.0, 0.0)
    shifted = prepare_curve(curve, 0.5, 0.7, 1.1, 0.0)
    assert central is not None and shifted is not None
    g0 = central["velocity_bar"] ** 2 / central["radius"]
    g1 = shifted["velocity_bar"] ** 2 / shifted["radius"]
    assert np.allclose(g0, g1, rtol=1e-12, atol=1e-12)


def test_zero_dispersion_limit_matches_acceleration_only() -> None:
    radius = np.asarray([1.0, 2.0, 5.0])
    velocity = np.asarray([40.0, 60.0, 80.0])
    fixed_point = predict_sigma(radius, velocity, sigma_kms=0.0)
    acceleration = predict_acceleration_only(radius, velocity)
    assert np.allclose(fixed_point, acceleration, rtol=0, atol=1e-6)


def test_central_dataset_is_nonempty_and_finite() -> None:
    frame, summary = evaluate_configuration(load_raw_curves())
    assert summary["n_galaxies"] > 150
    assert summary["n_disk_points"] > 2_000
    assert np.isfinite(frame.select_dtypes(include=[float, int]).to_numpy()).all()
    assert math.isclose(A0, math.exp(1.0 / (2.0 * math.pi)), rel_tol=1e-15)


def test_central_result_exactly_matches_production_regression() -> None:
    repo = Path(__file__).resolve().parents[3]
    script = repo / "scripts" / "run_regression_extended.py"
    spec = importlib.util.spec_from_file_location("production_regression", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    galaxies = module.load_sparc(repo / "data")
    production, _ = module.test_sparc_comparative(galaxies, verbose=False)
    _, audit = evaluate_configuration(load_raw_curves())
    assert production.n_objects == audit["n_galaxies"] == 164
    assert production.baseline.value == audit["mean_rms_sigma_kms"]
    assert production.mond.value == audit["mean_rms_mond_kms"]
    assert math.isclose(
        production.baseline.details["win_rate"] / 100.0,
        audit["sigma_win_fraction"],
        rel_tol=1e-15,
    )
