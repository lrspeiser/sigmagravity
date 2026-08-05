import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19bc_abell2146_collisionless_current_moments.json"
SCRIPT = ROOT / "scripts" / "run_sigma_v19bc_abell2146_collisionless_current_moments.py"
REPORT = ROOT / "results" / "sigma_v19bc_abell2146_collisionless_current_moments" / "report.json"
REPRODUCIBILITY = (
    ROOT
    / "results"
    / "sigma_v19bc_abell2146_collisionless_current_moments"
    / "reproducibility_audit.json"
)
SPEC = importlib.util.spec_from_file_location("sigma_v19bc", SCRIPT)
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
    assert (
        abs(
            MODULE.normalized_cauchy_schwarz_margin(
                rho, rho * coherent_beta, rho * coherent_beta**2
            )
        )
        < 1e-15
    )


def test_protocol_keeps_missing_light_explicit_without_fitting_gravity():
    config = json.loads(CONFIG.read_text())
    assert config["status"].startswith("frozen_before")
    assert config["population"]["expected_ensemble_rows"] == 8192 * 63
    assert config["population"]["expected_minimum_finite_luminosity_members_per_draw"] == 51
    assert config["population"]["expected_maximum_finite_luminosity_members_per_draw"] == 58
    assert config["grid"]["expected_shape_yx"] == [745, 745]
    assert not config["authorization"]["apply_or_fit_long_wave_operator"]
    assert not config["authorization"]["impute_missing_luminosity_or_transverse_velocity"]
    assert not config["authorization"]["infer_absolute_mass"]
    assert not config["authorization"]["read_lensing_or_halo_payload"]
    assert not config["authorization"]["change_gravity_physics"]


def test_frozen_runner_hash_is_exact():
    config = json.loads(CONFIG.read_text())
    assert config["implementation"]["runner"] == str(SCRIPT.relative_to(ROOT)).replace("\\", "/")
    assert MODULE.sha256(SCRIPT) == config["implementation"]["runner_sha256"]


def test_completed_result_passes_frozen_source_map_gates():
    report = json.loads(REPORT.read_text())
    assert report["decision"] == "passed"
    assert all(report["gate_results"].values())
    assert report["ensemble"]["draws"] == 8192
    assert report["ensemble"]["members_per_draw"] == 63
    assert report["ensemble"]["rows"] == 8192 * 63
    assert report["ensemble"]["finite_luminosity_members_per_draw_minimum"] == 51
    assert report["ensemble"]["finite_luminosity_members_per_draw_maximum"] == 58
    assert report["grid"]["outside_analysis_mask_member_ids"] == []
    assert not report["smoothing_length_or_response_amplitude_selected"]
    assert not report["absolute_mass_inferred"]
    assert not report["transverse_velocity_imputed"]
    assert not report["missing_luminosity_imputed"]
    assert not report["lensing_or_halo_payload_opened"]
    assert not report["gravity_formula_or_parameter_changed"]


def test_scientific_outputs_are_byte_reproducible():
    audit = json.loads(REPRODUCIBILITY.read_text())
    assert audit["runs_compared"] == 2
    assert audit["all_scientific_outputs_byte_identical"]
    assert len(audit["outputs"]) == 3
    for output in audit["outputs"].values():
        assert output["sha256_run_1"] == output["sha256_run_2"]
