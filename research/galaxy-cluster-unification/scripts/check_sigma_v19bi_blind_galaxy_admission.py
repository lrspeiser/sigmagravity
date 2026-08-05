from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19bi_blind_galaxy_admission.json"


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

    broad = parents["broad_phenomenology"]["payload"]
    cluster = parents["blind_cluster_admission"]["payload"]
    spent = {row["id"]: row for row in config["spent_galaxy_evidence"]}
    universes = {row["id"]: row for row in config["public_candidate_universes"]}

    spent_segregated = (
        spent["SPARC_DEVELOPMENT"]["systems"] == broad["current_coverage"]["SPARC_full_systems"]
        and spent["SPARC_DEVELOPMENT"]["observations"]
        == broad["current_coverage"]["SPARC_full_points"]
        and spent["LITTLE_THINGS_RESOLVED"]["systems"] >= 13
        and spent["SPIDERS_MANGA_BCG"]["systems"] == broad["current_coverage"]["BCG_systems"]
        and all("never a new holdout" in row["role"] for row in spent.values())
    )

    candidate_breadth = (
        universes["WALLABY_PDR1_KINEMATICS"]["published_systems"] >= 109
        and universes["PHANGS_CO_KINEMATICS"]["published_systems"] >= 67
        and universes["DISKMASS_RADIAL_VERTICAL"]["published_systems"] >= 30
        and len({row["url"] for row in universes.values()}) == 3
        and all(row["strength"] and row["limitation"] for row in universes.values())
    )

    required_strata = set(config["required_galaxy_strata"])
    strata_retained = required_strata == set(broad["registered_galaxy_strata"])

    final = config["final_holdout_requirements"]
    observable_breadth = (
        final["minimum_unique_galaxies"] >= 48
        and final["minimum_primary_WALLABY_galaxies"] >= 32
        and final["minimum_independent_non_WALLABY_galaxies"] >= 8
        and final["minimum_per_required_stratum"] >= 6
        and final["minimum_raw_cube_or_velocity_field_systems"] >= 12
        and final["minimum_high_resolution_inner_curve_systems"] >= 8
        and final["minimum_radial_plus_vertical_systems"] >= 8
        and final["minimum_group_or_cluster_environment_systems"] >= 8
        and final["minimum_low_density_field_environment_systems"] >= 8
        and final["strata_may_overlap"]
    )

    per_galaxy = config["per_galaxy_admission_requirements"]
    source_target_separated = (
        all(per_galaxy.values())
        and [row["stage"] for row in config["admission_sequence"]] == [1, 2, 3, 4, 5]
        and per_galaxy["source_and_target_files_separated"]
    )

    nuisance = config["nuisance_boundary"]
    nuisance_separation = (
        len(nuisance["measurement_nuisances"]) >= 9
        and len(nuisance["gravity_parameters"]) <= 5
        and not set(nuisance["measurement_nuisances"]).intersection(nuisance["gravity_parameters"])
        and "No nuisance" in nuisance["rule"]
    )

    comparator = config["fair_comparator_contract"]
    comparator_fair = (
        comparator["fixed_MOND_RAR_parameters_frozen_before_targets"]
        and comparator["same_baryon_draws_and_measurement_nuisances_for_all_gravity_models"]
        and comparator["no_per_galaxy_gravity_parameter_for_sigma"]
        and comparator["halo_comparator_may_fit_per_galaxy_but_parameter_count_reported"]
        and comparator["primary_aggregate_is_object_balanced"]
        and comparator["point_weighted_secondary_score_reported"]
        and comparator["overall_RMSE_maximum_relative_to_fixed_MOND_RAR"] <= 1.05
        and comparator["each_stratum_RMSE_maximum_relative_to_fixed_MOND_RAR"] <= 1.25
        and comparator["raw_field_residual_structure_reported"]
    )

    controls = {row["id"] for row in config["systematic_controls"]}
    control_breadth = controls == {
        "WALLABY_FORWARD_CUBE_CONTROL",
        "PHANGS_INNER_GEOMETRY_CONTROL",
        "DISKMASS_VERTICAL_CONTROL",
    }

    authorization = config["authorization"]
    sealed = (
        not final["sample_selected_here"]
        and not authorization["download_or_open_new_kinematic_targets"]
        and not authorization["select_final_galaxy_sample"]
    )
    priority = config["priority"]
    no_selection = (
        cluster["admission_state"]["admitted_holdouts"] == 0
        and not authorization["change_or_select_gravity_formula"]
        and not authorization["fit_universal_constants"]
        and not authorization["perform_detailed_solar_optimization"]
    )

    gates = {
        "all_parent_hashes_exact": all(row["exact"] for row in parents.values()),
        "spent_galaxy_evidence_is_segregated": spent_segregated,
        "independent_candidate_universes_are_broad": candidate_breadth,
        "all_frozen_galaxy_strata_are_retained": strata_retained,
        "sample_size_and_observable_breadth_are_required": observable_breadth and control_breadth,
        "source_and_target_payloads_remain_separated": source_target_separated,
        "measurement_nuisances_are_not_gravity_parameters": nuisance_separation,
        "comparators_and_scores_are_fair": comparator_fair,
        "final_sample_and_targets_remain_sealed": sealed,
        "solar_is_retained_as_later_veto": (
            not priority["detailed_solar_optimization_now"]
            and "Solar-System" in priority["later_hard_veto"]
        ),
        "no_theory_or_constant_selected": no_selection,
    }
    if set(gates) != set(config["required_gates"]):
        raise ValueError("implemented and declared gate names differ")
    if not all(config["required_gates"].values()):
        raise ValueError("every declared gate must be mandatory")

    return {
        "protocol_version": config["protocol_version"],
        "status": "completed_survey_level_blind_galaxy_admission_protocol",
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "parent_audit": {
            name: {key: value for key, value in row.items() if key != "payload"}
            for name, row in parents.items()
        },
        "spent_evidence": config["spent_galaxy_evidence"],
        "candidate_universes": config["public_candidate_universes"],
        "registered_strata": config["required_galaxy_strata"],
        "final_holdout_requirements": final,
        "systematic_controls": config["systematic_controls"],
        "fair_comparator_contract": comparator,
        "nuisance_boundary": nuisance,
        "admission_state": {
            "selected_galaxies": 0,
            "new_kinematic_targets_opened": 0,
            "survey_level_protocol_frozen": True,
            "action_selected": False,
            "universal_constants_selected": False,
        },
        "priority": priority,
        "authorization_audit": authorization,
        "gate_results": gates,
        "decision": (
            "passed_survey_level_blind_galaxy_admission_protocol"
            if all(gates.values())
            else "failed_survey_level_blind_galaxy_admission_protocol"
        ),
        "claim_boundary": config["claim_boundary"],
        "implementation": config["implementation"],
    }


def main() -> None:
    report = build_report()
    output = ROOT / load_json(DEFAULT_CONFIG)["outputs"]["report"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "output": output.relative_to(ROOT).as_posix(),
                "admission_state": report["admission_state"],
                "gate_results": report["gate_results"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if report["decision"] != "passed_survey_level_blind_galaxy_admission_protocol":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
