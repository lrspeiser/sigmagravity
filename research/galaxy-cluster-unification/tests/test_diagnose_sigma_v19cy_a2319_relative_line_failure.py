import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import diagnose_sigma_v19cy_a2319_relative_line_failure as diagnosis


def test_frozen_diagnosis_reads_only_hashed_reports() -> None:
    config, parents = diagnosis.validate_inputs()
    assert set(parents) == {
        "relative_line_report",
        "readiness_report",
        "calibration_report",
    }
    assert not config["authorization"]["read_event_or_energy_value"]
    assert not config["authorization"]["refit_relative_template"]
    assert not config["authorization"]["access_validation_or_holdout_assets"]


def test_diagnosis_identifies_materially_different_next_closure() -> None:
    config, parents = diagnosis.validate_inputs()
    result = diagnosis.diagnose(config, parents)
    assert result["parent_relative_line_gate_failed"]
    assert result["calibration_execution_evidence_preserved"]
    assert result["region_geometry_evidence_preserved"]
    assert result["extreme_topology_evidence_preserved"]
    assert result["h_like_optimizer_failures"] == 3
    assert result["maximum_he_h_velocity_disagreement_kms"] > 100.0
    assert result["published_detector_temperature_span_keV"] > 1.5
    assert all(result["warnings"].values())
    assert result["supported_classification"] == (
        "response_free_shared_template_identifiability_failure_with_reduced_exposure_noise"
    )
    assert not result["calibration_failure_ruled_out"]
    assert result["calibration_failure_currently_disfavored"]
    assert result["authorize_response_aware_development_protocol"]
    assert config["allowed_next_model"]["model_class"].startswith("response-aware")
