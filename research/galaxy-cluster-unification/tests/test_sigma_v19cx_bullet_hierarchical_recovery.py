import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_sigma_v19cx_bullet_hierarchical_recovery as runner

CONFIG = ROOT / "configs" / "sigma_v19cx_bullet_hierarchical_recovery.json"
X2 = ROOT / "configs" / "sigma_v19x2_unified_spectral_combination_commissioning.json"


def test_v19cx_preserves_every_scientific_section() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    parent = json.loads(X2.read_text(encoding="utf-8"))
    for section in ("registered_workload", "combination", "fit_sequence", "gates", "runtime_authorization"):
        assert payload[section] == parent[section]


def test_v19cx_uses_outcome_blind_runtime_threshold() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    remediation = payload["runtime_remediation"]
    assert remediation["maximum_direct_stack_cells"] == 1270
    assert not remediation["threshold_uses_cluster_or_scientific_outcome"]
    assert remediation["abell_direct_reference_cells"] == 1270
    assert remediation["bullet_failed_direct_cells"] == 3812
    assert payload["hierarchy"]["partition_key"] == "obsid"
    assert payload["hierarchy"]["intermediate_rmf_threshold"] == 0.0
    assert payload["hierarchy"]["final_rmf_threshold"] == 1e-6


def test_v19cx_seals_downstream_science() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    authorization = payload["authorization"]
    assert authorization["run_v19cw_equivalent_hierarchy_only_above_frozen_cell_threshold"]
    assert not authorization["overwrite_v19x2_failure_report"]
    assert not authorization["run_v19bq_or_v19bs"]
    assert not authorization["derive_action"]
    assert not authorization["change_gravity_formula_parameter_source_state_or_lensing_target"]


def test_v19cx_runner_accepts_frozen_config() -> None:
    runner.validate_frozen(json.loads(CONFIG.read_text(encoding="utf-8")))


def test_v19cx_direct_reference_reuse_has_no_unfrozen_index_dependency(tmp_path: Path, monkeypatch) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    cells = [
        {
            "source_pha_total_counts": 259688,
            "source_band_events": 227561,
            "background_band_events": 81581,
        }
    ]

    def capture_snapshot(*args, **kwargs):
        return {"runtime_remediation": kwargs["remediation"]}

    monkeypatch.setattr(runner, "snapshot_combination", capture_snapshot)
    result = runner.reuse_direct("ABELL2146_integrated", cells, tmp_path, payload)
    assert result["runtime_remediation"]["mode"] == "hash_frozen_direct_reference"
    assert result["runtime_remediation"]["source_stack_sha256"] == payload["runtime_remediation"]["direct_references"]["ABELL2146_integrated"]["source_grouped"]["sha256"]
