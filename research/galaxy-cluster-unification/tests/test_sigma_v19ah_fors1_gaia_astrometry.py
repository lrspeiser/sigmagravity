import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_sigma_v19ah_fors1_gaia_astrometry.py"
CONFIG = ROOT / "configs" / "sigma_v19ah_fors1_gaia_astrometry.json"
SPEC = importlib.util.spec_from_file_location("sigma_v19ah", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_frozen_runner_and_parent_hashes_match():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    hashes = MODULE.validate_config(CONFIG, config)
    assert hashes["runner"] == config["implementation"]["runner_sha256"]


def test_prepare_detection_image_only_fills_nonfinite_pixels():
    image = np.array([[1.0, np.nan], [3.0, 5.0]])
    prepared, metrics = MODULE.prepare_detection_image(image)
    assert prepared[0, 0] == 1.0
    assert prepared[1, 0] == 3.0
    assert prepared[1, 1] == 5.0
    assert prepared[0, 1] == 3.0
    assert metrics["finite_fraction"] == 0.75


def test_gaia_pmra_propagation_includes_cos_dec():
    catalog = pd.DataFrame(
        {
            "source_id": ["1"],
            "ref_epoch": [2016.0],
            "ra": [100.0],
            "dec": [60.0],
            "pmra": [100.0],
            "pmdec": [50.0],
            "phot_g_mean_mag": [18.0],
            "ruwe": [1.0],
            "visibility_periods_used": [12],
            "astrometric_params_solved": [31],
            "duplicated_source": [False],
        }
    )
    selection = json.loads(CONFIG.read_text(encoding="utf-8"))["gaia_selection"]
    result = MODULE.select_and_propagate_gaia(catalog, selection, 2017.0)
    np.testing.assert_allclose(result.loc[0, "ra_epoch"], 100.0 + 100.0 / 0.5 / 3_600_000.0)
    np.testing.assert_allclose(result.loc[0, "dec_epoch"], 60.0 + 50.0 / 3_600_000.0)


def test_config_has_no_member_or_physics_authorization():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    authorization = config["authorization"]
    assert not authorization["inspect_member_or_candidate_coordinates_or_cutouts"]
    assert not authorization["fit_source_photometry_or_deblending"]
    assert not authorization["infer_stellar_mass_or_current"]
    assert not authorization["change_gravity_physics_or_parameters"]
