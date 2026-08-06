import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19cy_direct_icm_velocity_evidence.json"


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v19cy_parent_evidence_is_exact() -> None:
    payload = load_config()
    parents = payload["parents"]
    for key in ("v19cx_config", "v19cx_report", "pre_deep_direction_checkpoint"):
        assert sha256(ROOT / parents[key]) == parents[f"{key}_sha256"]
    report = json.loads((ROOT / parents["v19cx_report"]).read_text(encoding="utf-8"))
    assert report["status"] == parents["required_v19cx_status"]


def test_v19cy_split_and_outcome_seals_are_frozen() -> None:
    split = load_config()["evidence_split"]
    assert [row["cluster"] for row in split.values()] == ["ABELL2319", "ABELL3667", "ABELL754"]
    assert split["development"]["outcome_known_before_freeze"]
    assert not split["validation"]["outcome_known_before_freeze"]
    assert not split["holdout"]["outcome_known_before_freeze"]
    assert [row["obsid"] for row in split["holdout"]["observations"]] == ["201015010", "201016010"]


def test_v19cy_archive_exposures_are_metadata_exact() -> None:
    split = load_config()["evidence_split"]
    validation_ks = sum(row["resolve_exposure_ks"] for row in split["validation"]["observations"])
    holdout_ks = sum(row["resolve_exposure_ks"] for row in split["holdout"]["observations"])
    assert validation_ks == 415.478
    assert holdout_ks == 320.097


def test_v19cy_separates_time_odd_current_from_time_even_stress() -> None:
    payload = load_config()
    candidates = payload["source_candidates"]
    assert "(v_los - v_systemic)" in candidates["J_LOS_SIGNED_GAS_CURRENT"]["source_equation"]
    assert candidates["J_LOS_SIGNED_GAS_CURRENT"]["parity"] == "time_odd"
    assert "(v_los - v_systemic)^2" in candidates["PI_LOS_KINETIC_STRESS"]["source_equation"]
    assert candidates["PI_LOS_KINETIC_STRESS"]["parity"] == "time_even"
    decision = payload["decision_rule"]
    assert decision["if_only_time_even_kinetic_stress_passes"].endswith("P2 remains unauthorized")
    assert decision["action_derivation_not_authorized_by_this_plan"]


def test_v19cy_keeps_gravity_targets_and_holdout_closed() -> None:
    payload = load_config()
    authorization = payload["authorization"]
    assert authorization["inventory_named_public_archives_now"]
    assert not authorization["open_validation_outcomes_before_development_freeze"]
    assert not authorization["open_holdout_outcomes_before_validation_pass"]
    assert not authorization["open_lensing_halo_or_gravity_targets"]
    assert not authorization["derive_or_select_action"]
    assert not authorization["change_gravity_formula_or_parameter"]
