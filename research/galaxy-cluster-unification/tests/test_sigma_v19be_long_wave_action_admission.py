import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19be_long_wave_action_admission.json"
SCRIPT = ROOT / "scripts" / "check_sigma_v19be_long_wave_action_admission.py"
REPORT = ROOT / "results" / "sigma_v19be_long_wave_action_admission" / "report.json"
SPEC = importlib.util.spec_from_file_location("sigma_v19be", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_scale_diagnostics_reproduce_the_long_wave_separation():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = MODULE.scale_diagnostics(config["illustrative_scale_check"])
    assert math.isclose(result["literal_wavelength_kpc"], 12.0 * math.pi)
    assert result["literal_wave_tidal_scale"] < 2e-16
    assert result["sourced_low_pass_small_baseline_scale"] < 4e-15
    assert 0.49 < result["sourced_low_pass_activation_at_galaxy_radius"] < 0.50
    assert result["light_crossing_time_years"] > 100_000.0


def test_protocol_requires_a_conserved_one_metric_action_without_selecting_it():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    requirements = config["action_admission_requirements"]
    assert requirements["one_physical_metric"]
    assert requirements["diffeomorphism_invariant_action"]
    assert requirements["total_metric_source_conservation"]
    assert requirements["matter_and_light_unified"]
    assert requirements["universal_wavelength_no_object_fit"]
    assert requirements["no_free_halo_equivalent_homogeneous_mode"]
    assert requirements["at_most_five_total_universal_physical_constants"]
    assert not config["authorization"]["select_candidate_action"]
    assert not config["authorization"]["select_long_wave_operator_or_constant"]
    assert not config["authorization"]["read_lensing_or_halo_payload"]


def test_closed_mechanisms_cannot_be_replayed_by_renaming_parameters():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    closed = " ".join(config["mechanism_boundary"]["closed_controls"]).lower()
    for required in ("linear isotropic", "positive-spectrum", "negative-residue", "yukawa"):
        assert required in closed
    assert "nonlinear" in config["mechanism_boundary"]["only_admissible_next_class"].lower()
    assert config["mechanism_boundary"]["does_not_select_a_member_of_that_class"]


def test_frozen_runner_hash_is_exact():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["implementation"]["runner"] == SCRIPT.relative_to(ROOT).as_posix()
    assert MODULE.sha256(SCRIPT) == config["implementation"]["runner_sha256"]


def test_completed_report_passes_only_the_action_admission_gate():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["decision"] == "passed_action_admission_requirements"
    assert all(report["gate_results"].values())
    assert report["theory_state"]["physical_postulate_recorded"]
    assert not report["theory_state"]["covariant_action_selected"]
    assert not report["theory_state"]["euler_lagrange_equations_derived"]
    assert not report["theory_state"]["weak_field_metric_derived"]
    assert not report["theory_state"]["universal_constants_selected"]


def test_report_is_byte_reproducible(tmp_path):
    first = MODULE.run(CONFIG)
    first_bytes = REPORT.read_bytes()
    second = MODULE.run(CONFIG)
    second_bytes = REPORT.read_bytes()
    assert first == second
    assert first_bytes == second_bytes
