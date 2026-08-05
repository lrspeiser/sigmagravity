from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19cf_cross_scale_data_readiness.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def build_report(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_json(config_path)
    parents: dict[str, dict[str, Any]] = {}
    for name, spec in config["parents"].items():
        path = ROOT / spec["path"]
        actual = sha256(path)
        parents[name] = {
            "path": spec["path"],
            "expected_sha256": spec["sha256"],
            "actual_sha256": actual,
            "exact": actual == spec["sha256"],
        }

    cross_scale = load_json(
        ROOT / config["parents"]["cross_scale_prediction_gates"]["path"]
    )
    cluster_admission = load_json(
        ROOT / config["parents"]["blind_cluster_admission"]["path"]
    )
    galaxy_admission = load_json(
        ROOT / config["parents"]["blind_galaxy_admission"]["path"]
    )
    mixture = load_json(
        ROOT / config["parents"]["wallaby_counterpart_mixture"]["path"]
    )
    spent_clusters = load_json(
        ROOT / config["parents"]["spent_cluster_evidence_registry"]["path"]
    )
    wallaby_rows = load_csv(
        ROOT / config["parents"]["wallaby_source_only_variety_frame"]["path"]
    )

    expected = config["expected_inventory"]
    galaxy_expected = expected["galaxy"]
    cluster_expected = expected["cluster"]
    availability = Counter(
        row["kinematic_availability_across_policies"] for row in wallaby_rows
    )
    kinematic_any = availability["all_policies"] + availability["some_policies"]
    unique_wallaby_ids = {row["canonical_id"] for row in wallaby_rows}
    variety_cells = {row["variety_cell"] for row in wallaby_rows}

    cluster_state = cluster_admission["admission_state"]
    spent_sample = spent_clusters["sample"]
    galaxy_state = galaxy_admission["admission_state"]
    core = cross_scale["core_breadth"]

    galaxy_inventory = {
        "canonical_source_rows": len(wallaby_rows),
        "unique_canonical_ids": len(unique_wallaby_ids),
        "source_variety_cells": len(variety_cells),
        "kinematic_availability": dict(sorted(availability.items())),
        "kinematic_available_any_policy": kinematic_any,
        "release_rows_with_counterpart_mixtures": mixture["release_rows"],
        "candidate_rows_in_mixture": mixture["candidate_rows"],
        "selected_prospective_galaxies": galaxy_state["selected_galaxies"],
        "opened_new_kinematic_targets": galaxy_state["new_kinematic_targets_opened"],
        "future_holdout_minimum": galaxy_admission["final_holdout_requirements"][
            "minimum_unique_galaxies"
        ],
        "future_wallaby_minimum": galaxy_admission["final_holdout_requirements"][
            "minimum_primary_WALLABY_galaxies"
        ],
        "source_universe_ready": kinematic_any
        >= galaxy_admission["final_holdout_requirements"][
            "minimum_primary_WALLABY_galaxies"
        ],
        "prospective_sample_admitted": False,
        "execution_ready": False,
    }
    cluster_inventory = {
        "metadata_shortlist": cluster_state["metadata_shortlist_count"],
        "relaxed_side": cluster_state["relaxed_side_count"],
        "disturbed_side": cluster_state["disturbed_side_count"],
        "admitted_prospective_holdouts": cluster_state["admitted_holdouts"],
        "selected_final_clusters": 6 if cluster_state["final_six_selected"] else 0,
        "opened_new_raw_lensing_targets": 1
        if cluster_state["raw_target_payload_opened"]
        else 0,
        "spent_registry_clusters": spent_sample["systemCount"],
        "spent_raw_forward_score_ready": spent_sample[
            "rawForwardScoreReadySystems"
        ],
        "future_holdout_minimum": cluster_admission["final_sample_requirements"][
            "clusters"
        ],
        "source_universe_ready": (
            cluster_state["metadata_shortlist_count"]
            >= cluster_admission["final_sample_requirements"]["clusters"]
            and cluster_state["relaxed_side_count"]
            >= cluster_admission["final_sample_requirements"][
                "relaxed_side_minimum"
            ]
            and cluster_state["disturbed_side_count"]
            >= cluster_admission["final_sample_requirements"][
                "disturbed_side_minimum"
            ]
        ),
        "prospective_sample_admitted": cluster_state["admitted_holdouts"] >= 6,
        "execution_ready": False,
    }

    gates = {
        "all_parent_hashes_exact": all(row["exact"] for row in parents.values()),
        "wallaby_source_inventory_is_large_and_kinematically_stratifiable": (
            galaxy_inventory["canonical_source_rows"]
            == galaxy_expected["wallaby_canonical_source_rows"]
            and galaxy_inventory["unique_canonical_ids"]
            == galaxy_expected["wallaby_canonical_source_rows"]
            and availability["all_policies"]
            == galaxy_expected["wallaby_kinematic_available_all_policies"]
            and availability["some_policies"]
            == galaxy_expected["wallaby_kinematic_available_some_policies"]
            and kinematic_any
            == galaxy_expected["wallaby_kinematic_available_any_policy"]
            and galaxy_inventory["source_universe_ready"]
        ),
        "wallaby_identity_uncertainty_is_preserved": (
            mixture["decision"]
            == "counterpart_uncertainty_ready_for_target_blind_marginalization"
            and mixture["release_rows"]
            == galaxy_expected["wallaby_release_rows_with_counterpart_mixtures"]
            and mixture["candidate_rows"]
            == galaxy_expected["wallaby_candidates_in_mixture"]
            and not mixture["access_boundary_audit"][
                "counterpart_treatment_or_kernel_selected"
            ]
            and not mixture["access_boundary_audit"][
                "wallaby_kinematic_table_row_read"
            ]
        ),
        "cluster_shortlist_is_state_balanced_but_not_misrepresented_as_admitted": (
            cluster_inventory["metadata_shortlist"]
            == cluster_expected["metadata_shortlist"]
            and cluster_inventory["relaxed_side"]
            == cluster_expected["relaxed_side"]
            and cluster_inventory["disturbed_side"]
            == cluster_expected["disturbed_side"]
            and cluster_inventory["admitted_prospective_holdouts"]
            == cluster_expected["admitted_prospective_holdouts"]
            and not cluster_inventory["prospective_sample_admitted"]
        ),
        "spent_cluster_products_are_not_misrepresented_as_blind_or_score_ready": (
            spent_sample["sampleState"] == "spent"
            and spent_sample["prospectiveHoldoutSystems"] == 0
            and cluster_inventory["spent_registry_clusters"]
            == cluster_expected["spent_registry_clusters"]
            and cluster_inventory["spent_raw_forward_score_ready"]
            == cluster_expected["spent_raw_forward_score_ready"]
        ),
        "prospective_galaxy_and_cluster_execution_is_truthfully_not_ready": (
            galaxy_inventory["selected_prospective_galaxies"]
            == galaxy_expected["selected_prospective_galaxies"]
            and galaxy_inventory["opened_new_kinematic_targets"]
            == galaxy_expected["opened_new_kinematic_targets"]
            and not galaxy_inventory["prospective_sample_admitted"]
            and not galaxy_inventory["execution_ready"]
            and cluster_inventory["selected_final_clusters"]
            == cluster_expected["selected_final_clusters"]
            and cluster_inventory["opened_new_raw_lensing_targets"]
            == cluster_expected["opened_new_raw_lensing_targets"]
            and not cluster_inventory["execution_ready"]
        ),
        "broad_core_thresholds_and_later_solar_veto_are_preserved": (
            cross_scale["decision"] == "passed_cross_scale_prediction_gate_freeze"
            and core["galaxy_holdout_minimum"] >= 48
            and core["galaxy_morphology_strata"] == 8
            and core["cluster_holdout_minimum"] >= 6
            and core["cluster_relaxed_minimum"] >= 2
            and core["cluster_disturbed_minimum"] >= 2
            and cross_scale["priority_order"][0]
            == "broad_galaxy_and_raw_cluster_core"
            and cross_scale["priority_order"][-1]
            == "local_relativistic_and_mathematical_veto"
            and not cross_scale["theory_state"][
                "detailed_solar_optimization_performed"
            ]
        ),
        "no_target_action_constant_formula_or_sample_selected": (
            not galaxy_state["action_selected"]
            and not galaxy_state["universal_constants_selected"]
            and not cluster_admission["authorization_audit"][
                "read_raw_target_coordinates"
            ]
            and not cluster_admission["authorization_audit"]["select_final_six"]
            and not config["authorization"]["read_new_kinematic_values"]
            and not config["authorization"]["read_new_lensing_coordinates_or_maps"]
            and not config["authorization"][
                "select_final_galaxy_or_cluster_sample"
            ]
            and not config["authorization"][
                "select_or_change_action_or_gravity_formula"
            ]
            and not config["authorization"]["fit_universal_constants"]
            and not config["authorization"]["perform_detailed_solar_optimization"]
        ),
    }
    if set(gates) != set(config["required_gates"]):
        raise ValueError("implemented and declared V19CF gate names differ")
    if not all(config["required_gates"].values()):
        raise ValueError("every declared V19CF gate must be mandatory")

    return {
        "protocol_version": config["protocol_version"],
        "status": "completed_cross_scale_data_readiness_audit",
        "decision": (
            "source_universes_ready_but_prospective_core_not_execution_ready"
            if all(gates.values())
            else "cross_scale_data_readiness_audit_failed_closed"
        ),
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "parent_audit": parents,
        "galaxy_inventory": galaxy_inventory,
        "cluster_inventory": cluster_inventory,
        "next_data_actions": config["next_data_actions"],
        "access_boundary_audit": {
            "read_source_and_metadata_reports": True,
            "read_wallaby_source_only_variety_rows": True,
            "read_new_kinematic_values": False,
            "read_new_lensing_coordinates_or_maps": False,
            "selected_final_sample": False,
            "changed_action_formula_or_constants": False,
            "performed_detailed_solar_optimization": False,
        },
        "gate_results": gates,
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
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "galaxy_inventory": report["galaxy_inventory"],
                "cluster_inventory": report["cluster_inventory"],
                "gate_results": report["gate_results"],
                "output": output.relative_to(ROOT).as_posix(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if report["decision"] != (
        "source_universes_ready_but_prospective_core_not_execution_ready"
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
