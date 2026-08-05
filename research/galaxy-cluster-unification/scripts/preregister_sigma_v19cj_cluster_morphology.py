from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "configs" / "sigma_v19cj_cluster_morphology_preregistration.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_json(config_path)
    parent_path = ROOT / config["parent"]["path"]
    parent = load_json(parent_path)
    requirements = config["diversity_and_admission_requirements"]
    authorization = config["authorization"]
    gates = {
        "v19ch_parent_hash_decision_and_162_clean_candidates_exact": (
            sha256(parent_path) == config["parent"]["sha256"]
            and parent["decision"] == config["parent"]["required_decision"]
            and parent["summary"]["zero_hit_unexposed_candidates"]
            == config["parent"]["required_clean_candidates"]
        ),
        "primary_and_secondary_sources_registered_before_value_access": (
            config["registered_sources"]["primary"]["catalog_rows"] == 964
            and config["registered_sources"]["secondary"]["catalog_rows"] == 150
            and not authorization["select_or_admit_cluster_here"]
        ),
        "signed_uncertainty_aware_delta_states_frozen": (
            config["primary_state_rule"]["secure_relaxed"]
            == "delta+e_delta < 0"
            and config["primary_state_rule"]["secure_disturbed"]
            == "delta-e_delta > 0"
            and config["primary_state_rule"]["boundary_intermediate"]
            == "delta-e_delta <= 0 <= delta+e_delta"
            and config["primary_state_rule"][
                "no_threshold_may_be_moved_after_values_are_read"
            ]
        ),
        "multimetric_confirmation_directions_and_medians_frozen": (
            "higher logc" in config["multimetric_consistency_rule"]["relaxed_direction"]
            and "lower logw" in config["multimetric_consistency_rule"]["relaxed_direction"]
            and "lower logP3/P0" in config["multimetric_consistency_rule"]["relaxed_direction"]
            and config["multimetric_consistency_rule"][
                "statistics_computed_only_within_clean_finite_primary_crossmatches"
            ]
        ),
        "minimum_three_state_diversity_and_failure_fallbacks_frozen": (
            requirements["minimum_metadata_shortlist"] >= 8
            and requirements["minimum_secure_relaxed"] >= 3
            and requirements["minimum_secure_disturbed"] >= 3
            and requirements["minimum_boundary_or_discordant"] >= 2
            and config["failure_and_fallback_rules"][
                "no_cluster_is_admitted_by_this_protocol"
            ]
        ),
        "lensing_and_sigma_outcomes_forbidden_from_selection": (
            len(requirements["forbidden_selection_axes"]) == 4
            and not authorization["open_raw_lensing_coordinate_map_or_halo_target"]
        ),
        "no_cluster_target_action_constant_or_solar_setting_selected": (
            not authorization["select_or_admit_cluster_here"]
            and not authorization["open_raw_lensing_coordinate_map_or_halo_target"]
            and not authorization["select_or_change_action_or_gravity_formula"]
            and not authorization["fit_universal_constants"]
            and not authorization["perform_detailed_solar_optimization"]
        ),
    }
    if set(gates) != set(config["required_gates"]):
        raise ValueError("implemented and declared V19CJ gate names differ")
    if not all(config["required_gates"].values()):
        raise ValueError("every V19CJ gate must be mandatory")
    return {
        "protocol_version": config["protocol_version"],
        "status": "cluster_morphology_rule_preregistered_before_value_access",
        "decision": (
            "registered_morphology_tables_may_be_acquired_under_frozen_rule"
            if all(gates.values())
            else "cluster_morphology_preregistration_failed_closed"
        ),
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "parent": {
            "path": config["parent"]["path"],
            "expected_sha256": config["parent"]["sha256"],
            "actual_sha256": sha256(parent_path),
            "decision": parent["decision"],
            "clean_candidates": parent["summary"][
                "zero_hit_unexposed_candidates"
            ],
        },
        "registered_sources": config["registered_sources"],
        "crossmatch_rule": config["crossmatch_rule"],
        "primary_state_rule": config["primary_state_rule"],
        "multimetric_consistency_rule": config["multimetric_consistency_rule"],
        "diversity_and_admission_requirements": requirements,
        "failure_and_fallback_rules": config["failure_and_fallback_rules"],
        "gate_results": gates,
        "access_boundary_audit": {
            "downloaded_registered_table_before_freeze": False,
            "read_candidate_morphology_value_before_freeze": False,
            "selected_cluster": False,
            "opened_lensing_halo_or_gravity_target": False,
            "changed_action_formula_or_constants": False,
            "performed_detailed_solar_optimization": False,
        },
        "claim_boundary": config["claim_boundary"],
        "implementation": config["implementation"],
    }


def main() -> None:
    report = build_report()
    config = load_json(DEFAULT_CONFIG)
    output = ROOT / config["outputs"]["report"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"decision": report["decision"], "gate_results": report["gate_results"], "output": output.relative_to(ROOT).as_posix()}, indent=2, sort_keys=True))
    if report["decision"] != "registered_morphology_tables_may_be_acquired_under_frozen_rule":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
