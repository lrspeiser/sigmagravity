from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
DEFAULT_CONFIG = (
    ROOT / "configs" / "sigma_v19cg_whole_repo_cluster_holdout_contamination.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(path: str) -> Path:
    return (ROOT / path).resolve()


def build_report(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_json(config_path)
    parents: dict[str, dict[str, Any]] = {}
    for name, spec in config["parents"].items():
        path = resolve(spec["path"])
        actual = sha256(path)
        parents[name] = {
            "path": spec["path"],
            "resolved_path": path.relative_to(REPO).as_posix(),
            "expected_sha256": spec["sha256"],
            "actual_sha256": actual,
            "exact": actual == spec["sha256"],
        }

    v19cf = load_json(resolve(config["parents"]["cross_scale_data_readiness"]["path"]))
    v19bt = load_json(
        resolve(config["parents"]["blind_cluster_source_readiness"]["path"])
    )
    v19bh = load_json(resolve(config["parents"]["blind_cluster_admission"]["path"]))
    evidence_text = {
        name: resolve(spec["path"]).read_text(encoding="utf-8", errors="replace")
        for name, spec in config["parents"].items()
        if name.startswith("legacy_")
    }

    dispositions = config["original_shortlist_disposition"]
    shortlist_ids = {row["id"] for row in dispositions}
    expected_ids = {row["id"] for row in v19bh["metadata_shortlist"]}
    prior_hits: dict[str, list[str]] = {}
    for row in dispositions:
        alias = row["id"].removeprefix("SDSS_")
        prior_hits[row["id"]] = sorted(
            name for name, text in evidence_text.items() if alias in text
        )

    incident = config["coordinate_exposure_incident"]
    exposed = set(incident["systems_whose_raw_multiple_image_rows_were_exposed"])
    shortlist_exposed = set(incident["original_v19bh_shortlist_systems_exposed"])
    prior_used = {row["id"] for row in dispositions if prior_hits[row["id"]]}
    disqualified = prior_used | shortlist_exposed
    prospective = [
        row for row in dispositions if row["id"] not in disqualified
    ]
    admitted = [row for row in prospective if row["future_role"] == "admitted_holdout"]

    six_preflight_ids = {
        row["id"]
        for row in v19bt["systems"]
        if row["source_imaging_preflight"] == "passed_not_admitted"
    }
    preflight_disqualified = six_preflight_ids & disqualified
    supersession = config["supersession"]
    authorization = config["authorization"]

    gates = {
        "all_parent_and_evidence_hashes_exact": all(
            row["exact"] for row in parents.values()
        ),
        "whole_repository_audit_finds_all_six_prior_used_shortlist_systems": (
            len(prior_used) == 6
            and prior_used
            == {
                "SDSS_J0851+3331",
                "SDSS_J0952+3434",
                "SDSS_J1038+4849",
                "SDSS_J1050+0017",
                "SDSS_J1207+5254",
                "SDSS_J1209+2640",
            }
            and all(
                row["prior_sigma_gravity_information_used"]
                == (row["id"] in prior_used)
                for row in dispositions
            )
        ),
        "coordinate_exposure_incident_is_complete_and_fail_closed": (
            len(exposed) == 10
            and shortlist_exposed
            == {
                "SDSS_J0851+3331",
                "SDSS_J0952+3434",
                "SDSS_J1002+2031",
            }
            and shortlist_exposed <= exposed
            and not incident["coordinate_values_copied_into_repository"]
            and not incident["coordinate_values_used_for_a_score_or_selection"]
            and all(
                row["raw_multiple_image_coordinates_exposed"]
                == (row["id"] in shortlist_exposed)
                for row in dispositions
            )
        ),
        "every_prior_used_or_coordinate_exposed_shortlist_system_is_removed_from_holdout_role": all(
            row["future_role"] not in {"prospective_holdout", "admitted_holdout"}
            for row in dispositions
            if row["id"] in disqualified
        ),
        "only_one_original_shortlist_system_remains_prospective_and_it_is_not_admitted": (
            len(prospective) == 1
            and prospective[0]["id"] == "SDSS_J1226+2149"
            and prospective[0]["future_role"]
            == "source_incomplete_reserve_not_admitted"
            and len(admitted) == 0
        ),
        "v19bh_v19bt_and_v19cf_cluster_holdout_conclusions_are_superseded": (
            shortlist_ids == expected_ids
            and v19cf["cluster_inventory"]["source_universe_ready"]
            and len(six_preflight_ids) == 6
            and preflight_disqualified == six_preflight_ids
            and not supersession[
                "v19bh_shortlist_may_supply_final_whole_object_holdouts"
            ]
            and not supersession[
                "v19bt_six_source_imaging_preflights_may_supply_final_whole_object_holdouts"
            ]
            and not supersession[
                "v19cf_cluster_source_universe_ready_for_prospective_core"
            ]
            and not supersession["replacement_cluster_universe_selected_here"]
        ),
        "galaxy_readiness_is_unchanged": (
            supersession["v19cf_galaxy_readiness_conclusion_unchanged"]
            and v19cf["galaxy_inventory"]["source_universe_ready"]
        ),
        "no_replacement_target_action_constant_formula_or_solar_setting_selected": (
            not authorization["open_another_raw_lensing_coordinate_or_map"]
            and not authorization["select_replacement_clusters"]
            and not authorization["select_or_change_action_or_gravity_formula"]
            and not authorization["fit_universal_constants"]
            and not authorization["perform_detailed_solar_optimization"]
        ),
    }
    if set(gates) != set(config["required_gates"]):
        raise ValueError("implemented and declared V19CG gate names differ")
    if not all(config["required_gates"].values()):
        raise ValueError("every V19CG gate must be mandatory")

    return {
        "protocol_version": config["protocol_version"],
        "status": "completed_whole_repo_cluster_holdout_contamination_audit",
        "decision": (
            "original_cluster_shortlist_retired_from_prospective_holdout_role"
            if all(gates.values())
            else "cluster_holdout_contamination_audit_failed_closed"
        ),
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "parent_and_evidence_audit": parents,
        "whole_repository_prior_hits": prior_hits,
        "coordinate_exposure_incident": incident,
        "original_shortlist_disposition": dispositions,
        "summary": {
            "original_shortlist_systems": len(dispositions),
            "prior_sigma_used_systems": len(prior_used),
            "raw_coordinate_exposed_shortlist_systems": len(shortlist_exposed),
            "disqualified_unique_systems": len(disqualified),
            "remaining_source_incomplete_reserves": len(prospective),
            "admitted_prospective_holdouts": len(admitted),
            "v19bt_direct_preflight_systems_disqualified_as_whole_object_holdouts": len(
                preflight_disqualified
            ),
        },
        "supersession": supersession,
        "required_next_cluster_work": config["required_next_cluster_work"],
        "access_boundary_audit": {
            "recorded_incident_without_coordinate_values": True,
            "opened_another_raw_coordinate_or_map_after_incident": False,
            "selected_replacement_cluster": False,
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
                "summary": report["summary"],
                "gate_results": report["gate_results"],
                "output": output.relative_to(ROOT).as_posix(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if report["decision"] != (
        "original_cluster_shortlist_retired_from_prospective_holdout_role"
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
