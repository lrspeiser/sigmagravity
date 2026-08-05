import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19az_probabilistic_member_current_ensemble.json"
SCRIPT = ROOT / "scripts" / "run_sigma_v19az_probabilistic_member_current_ensemble.py"
REPORT = ROOT / "results" / "sigma_v19az_probabilistic_member_current_ensemble" / "report.json"
REPRODUCIBILITY = (
    ROOT
    / "results"
    / "sigma_v19az_probabilistic_member_current_ensemble"
    / "reproducibility_audit.json"
)
SPEC = importlib.util.spec_from_file_location("sigma_v19az", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_exact_component_conditions_away_a_double_assignment():
    states = {
        "01": [
            MODULE.State("01", "candidate", "shared", 1.0, 1.0, 0.8),
            MODULE.State("01", "null", "", 0.0, 0.0, 0.2),
        ],
        "02": [
            MODULE.State("02", "candidate", "shared", 1.0, 1.0, 0.5),
            MODULE.State("02", "null", "", 0.0, 0.0, 0.5),
        ],
    }
    component = MODULE.exact_component_posterior("C001", ("01", "02"), states)
    marginals = MODULE.component_marginals(component, states)
    assert component.cartesian_state_count == 4
    assert component.valid_state_count == 3
    assert np.isclose(marginals["01"][0], 2.0 / 3.0)
    assert np.isclose(marginals["02"][0], 1.0 / 6.0)
    assert np.isclose(marginals["01"][0] + marginals["02"][0], 5.0 / 6.0)


def test_null_positions_stay_inside_the_frozen_rounding_rectangle():
    rng = np.random.Generator(np.random.PCG64(17))
    ra0, dec0 = 104.6, -56.0
    east_half_width = 7.5 * np.cos(np.deg2rad(dec0))
    for _ in range(1000):
        ra, dec = MODULE.null_position(rng, ra0, dec0)
        east, north = MODULE.local_offsets_arcsec(ra, dec, ra0, dec0)
        assert abs(east) <= east_half_width + 1e-10
        assert abs(north) <= 0.5 + 1e-10


def test_protocol_forbids_hard_matches_mass_and_transverse_imputation():
    config = json.loads(CONFIG.read_text())
    assert config["status"] == "frozen_before_joint_assignment_or_ensemble_generation"
    assert config["joint_assignment_posterior"]["approximation"] == "none"
    assert config["ensemble"]["draws"] == 8192
    assert not config["authorization"]["select_hard_ambiguous_counterparts"]
    assert not config["authorization"]["infer_stellar_mass"]
    assert not config["authorization"]["impute_transverse_velocity"]
    assert not config["authorization"]["read_lensing_or_halo_payload"]


def test_completed_result_passes_exact_and_sampling_gates():
    report = json.loads(REPORT.read_text())
    assert report["decision"] == "passed"
    assert all(report["gate_results"].values())
    assert report["population"] == {
        "finite_bri_members": 72,
        "fixed_anchor_members": 15,
        "missing_bri_member_ids": ["01", "02", "03", "04", "05", "67"],
        "missing_bri_members": 6,
        "probabilistic_members": 57,
        "spectroscopic_members": 78,
    }
    assert report["exact_posterior"]["approximation"] == "none"
    assert report["ensemble"]["rows"] == 8192 * 72
    assert not report["lensing_or_halo_payload_opened"]
    assert not report["gravity_formula_or_parameter_changed"]


def test_scientific_outputs_are_byte_reproducible():
    audit = json.loads(REPRODUCIBILITY.read_text())
    assert audit["runs_compared"] == 2
    assert audit["all_scientific_outputs_byte_identical"]
    assert len(audit["outputs"]) == 7
    for output in audit["outputs"].values():
        assert output["sha256_run_1"] == output["sha256_run_2"]
