from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19cc_cross_scale_prediction_gates.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
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
            "payload": load_json(path),
        }

    broad = parents["broad_phenomenology_contract"]["payload"]
    source = parents["source_invariant_action_preselection"]["payload"]
    cluster_admission = parents["blind_cluster_admission"]["payload"]
    galaxy_admission = parents["blind_galaxy_admission"]["payload"]

    galaxy = config["core_breadth_gates"]["galaxy"]
    cluster = config["core_breadth_gates"]["cluster"]
    broad_core_preserved = (
        broad["decision"] == "passed_broad_phenomenology_contract"
        and cluster_admission["decision"]
        == "passed_metadata_only_holdout_admission_protocol"
        and galaxy_admission["decision"]
        == "passed_survey_level_blind_galaxy_admission_protocol"
        and galaxy["minimum_unique_holdout_galaxies"] >= 48
        and galaxy["minimum_wallaby_galaxies"] >= 32
        and galaxy["minimum_members_per_morphology_stratum"] >= 6
        and len(galaxy["required_morphology_strata"]) == 8
        and galaxy["minimum_raw_cube_or_velocity_field_systems"] >= 12
        and galaxy["minimum_high_resolution_inner_curve_systems"] >= 8
        and galaxy["minimum_joint_radial_vertical_systems"] >= 8
        and galaxy["overall_RMSE_maximum_relative_to_fixed_MOND_RAR"] <= 1.05
        and galaxy["each_stratum_RMSE_maximum_relative_to_fixed_MOND_RAR"] <= 1.25
        and cluster["minimum_new_clusters"] >= 6
        and cluster["minimum_relaxed_clusters"] >= 2
        and cluster["minimum_disturbed_or_merging_clusters"] >= 2
        and cluster["minimum_secure_families_per_cluster"] >= 3
        and cluster["minimum_spectroscopic_families_per_cluster"] >= 1
        and cluster["minimum_images_per_cluster"] >= 8
        and cluster["complete_baryon_model_required"]
        and cluster["raw_target_likelihood_required"]
        and cluster["inferred_halo_map_as_validation_truth_forbidden"]
        and cluster["image_root_recovery_fraction_minimum"] == 1.0
        and cluster["image_RMS_maximum_relative_to_same_catalog_halo_comparator"]
        <= 1.25
        and cluster["baryon_to_halo_gap_closure_fraction_minimum"] >= 0.75
    )

    near = {row["id"]: row for row in config["near_term_quantitative_gates"]}
    required_near = {
        "N1_RESOLVED_CLUSTER_WEAK_LENSING",
        "N2_GALAXY_GALAXY_WEAK_LENSING",
        "N3_JOINT_DYNAMICS_AND_LENSING",
        "N4_COLLIDING_CLUSTER_DIRECTION_AND_OFFSETS",
    }
    near_term_quantitative = (
        set(near) == required_near
        and near["N1_RESOLVED_CLUSTER_WEAK_LENSING"]["minimum_systems"] >= 6
        and near["N1_RESOLVED_CLUSTER_WEAK_LENSING"][
            "aggregate_deviance_maximum_relative_to_halo"
        ]
        <= 1.25
        and near["N1_RESOLVED_CLUSTER_WEAK_LENSING"][
            "baryon_to_halo_gap_closure_fraction_minimum"
        ]
        >= 0.75
        and near["N2_GALAXY_GALAXY_WEAK_LENSING"][
            "minimum_baryonic_mass_bins"
        ]
        >= 3
        and near["N3_JOINT_DYNAMICS_AND_LENSING"]["minimum_systems"] >= 8
        and near["N3_JOINT_DYNAMICS_AND_LENSING"][
            "matter_and_light_use_identical_metric_and_constants"
        ]
        and near["N4_COLLIDING_CLUSTER_DIRECTION_AND_OFFSETS"][
            "minimum_untouched_mergers"
        ]
        >= 2
        and near["N4_COLLIDING_CLUSTER_DIRECTION_AND_OFFSETS"][
            "median_axial_angle_error_deg_maximum"
        ]
        <= 30.0
        and all(
            row.get("new_gravity_parameters_allowed", 0) == 0
            and row.get(
                "new_amplitude_orientation_scale_or_lag_parameters_allowed", 0
            )
            == 0
            and row.get("failure_meaning")
            for row in near.values()
        )
    )

    raw_roles = (
        near["N4_COLLIDING_CLUSTER_DIRECTION_AND_OFFSETS"]["primary_truth"]
        == "raw strong_and_weak_lensing_catalog_likelihood"
        and near["N4_COLLIDING_CLUSTER_DIRECTION_AND_OFFSETS"][
            "inferred_halo_peak_secondary_diagnostic_only"
        ]
        and config["authorization"]["use_inferred_halo_map_as_validation_truth"]
        is False
        and cluster["inferred_halo_map_as_validation_truth_forbidden"]
    )

    broader = config["broader_prediction_dispositions"]
    required_broader = {
        "B1_DWARF_SATELLITES",
        "B2_STREAMS_AND_COMPACT_SUBSTRUCTURE",
        "B3_DYNAMICAL_FRICTION",
        "B4_GROWTH_COSMIC_SHEAR_AND_CLUSTER_ABUNDANCE",
        "B5_PRIMARY_CMB_AND_CMB_LENSING",
    }
    broader_unearned = (
        {row["id"] for row in broader} == required_broader
        and all(
            row["present_claim"] == "not_earned"
            and row["required_theory"]
            and row["must_predict"]
            and row["decisive_risk"]
            for row in broader
        )
    )

    priority = config["priority_order"]
    solar_priority_safe = (
        [row["rank"] for row in priority] == [1, 2, 3, 4, 5]
        and priority[0]["stage"] == "broad_galaxy_and_raw_cluster_core"
        and priority[-1]["stage"] == "local_relativistic_and_mathematical_veto"
        and config["stop_and_promotion_rules"][
            "solar_work_is_later_but_remains_mandatory"
        ]
        and not config["stop_and_promotion_rules"][
            "detailed_solar_optimization_authorized_now"
        ]
    )

    lock = config["theory_lock"]
    authorization = config["authorization"]
    no_selection = (
        source["decision"] == "passed_target_blind_source_preselection_freeze"
        and lock["one_physical_metric_for_matter_and_light"]
        and lock["maximum_universal_physical_constants"] <= 5
        and lock["per_object_gravity_parameters"] == 0
        and lock["object_type_switches"] == 0
        and lock["lensing_only_multipliers"] == 0
        and lock["equation_and_constants_frozen_before_targets"]
        and lock["same_constants_across_all_registered_tests"]
        and not lock["action_selected_here"]
        and not lock["constant_selected_here"]
        and not lock["gravity_formula_changed_here"]
        and not authorization["open_new_holdout"]
        and not authorization["inspect_sealed_target_values"]
        and not authorization["select_source_invariant"]
        and not authorization["derive_or_select_action"]
        and not authorization["fit_universal_constants"]
        and not authorization["change_gravity_formula"]
        and not authorization["perform_detailed_solar_optimization"]
    )

    gates = {
        "all_parent_hashes_exact": all(row["exact"] for row in parents.values()),
        "broad_core_gates_preserved": broad_core_preserved,
        "near_term_tests_are_quantitative_and_no_retuning": near_term_quantitative,
        "raw_observation_roles_are_preserved": raw_roles,
        "broader_claims_are_explicitly_unearned": broader_unearned,
        "priority_keeps_solar_as_later_hard_veto": solar_priority_safe,
        "no_action_constant_formula_or_target_selected": no_selection,
    }
    if set(gates) != set(config["required_gates"]):
        raise ValueError("implemented and declared gate names differ")
    if not all(config["required_gates"].values()):
        raise ValueError("every declared gate must be mandatory")

    return {
        "protocol_version": config["protocol_version"],
        "status": "completed_cross_scale_prediction_gate_freeze",
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "parent_audit": {
            name: {key: value for key, value in row.items() if key != "payload"}
            for name, row in parents.items()
        },
        "priority_order": [row["stage"] for row in priority],
        "core_breadth": {
            "galaxy_holdout_minimum": galaxy["minimum_unique_holdout_galaxies"],
            "galaxy_morphology_strata": len(galaxy["required_morphology_strata"]),
            "cluster_holdout_minimum": cluster["minimum_new_clusters"],
            "cluster_relaxed_minimum": cluster["minimum_relaxed_clusters"],
            "cluster_disturbed_minimum": cluster[
                "minimum_disturbed_or_merging_clusters"
            ],
        },
        "near_term_gate_ids": list(near),
        "broader_prediction_states": {
            row["id"]: row["present_claim"] for row in broader
        },
        "theory_state": {
            "source_invariant_selected": False,
            "action_selected": False,
            "universal_constants_selected": False,
            "gravity_formula_changed": False,
            "new_holdout_opened": False,
            "detailed_solar_optimization_performed": False,
        },
        "gate_results": gates,
        "decision": (
            "passed_cross_scale_prediction_gate_freeze"
            if all(gates.values())
            else "failed_cross_scale_prediction_gate_freeze"
        ),
        "claim_boundary": config["claim_boundary"],
        "implementation": config["implementation"],
    }


def main() -> None:
    report = build_report()
    config = load_json(DEFAULT_CONFIG)
    output = ROOT / config["outputs"]["report"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "gate_results": report["gate_results"],
                "output": output.relative_to(ROOT).as_posix(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if report["decision"] != "passed_cross_scale_prediction_gate_freeze":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
