from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19bh_blind_cluster_admission.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def tracked_alias_hits(alias: str, commit: str) -> list[str]:
    command = [
        "git",
        "grep",
        "-l",
        "-i",
        "-F",
        "--",
        alias,
        commit,
        "--",
        "research/galaxy-cluster-unification/**",
        ":(exclude)research/galaxy-cluster-unification/data/raw/**",
        ":(exclude)research/galaxy-cluster-unification/data/derived/r1_rxj2129_xmm_conda_meta_lock.json",
    ]
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=False)
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr.strip() or "git grep alias audit failed")
    return sorted(line for line in result.stdout.splitlines() if line)


def build_report(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_json(config_path)
    parent_path = ROOT / config["parent"]["path"]
    parent_actual = sha256(parent_path)
    parent = load_json(parent_path)
    parent_hash_exact = parent_actual == config["parent"]["sha256"]

    baseline = config["alias_audit"]["repository_commit_before_this_protocol"]
    resolved = subprocess.run(
        ["git", "rev-parse", baseline],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    shortlist = config["metadata_only_shortlist"]
    alias_audit = {
        row["id"]: {
            "alias": row["alias"],
            "hits": tracked_alias_hits(row["alias"], resolved),
        }
        for row in shortlist
    }

    universes = {row["id"]: row for row in config["public_source_universes"]}
    public_universes_broad = (
        universes["SGAS_HLSP"]["systems"] >= 37
        and universes["CHANDRA_STRONG_LENS_SAMPLE"]["systems"] >= 28
        and universes["RELICS_HLSP_RESERVE"]["systems"] >= 41
        and len({row["url"] for row in universes.values()}) == 3
        and all(row["available_metadata"] for row in universes.values())
    )

    states = [row["state_side"] for row in shortlist]
    shortlist_balanced = (
        len(shortlist) >= 8
        and states.count("relaxed") >= 4
        and states.count("disturbed") >= 4
        and all(row["status"] == "metadata_only_not_admitted" for row in shortlist)
    )

    per_cluster = config["per_cluster_admission_requirements"]
    final = config["final_sample_requirements"]
    availability_not_admission = (
        per_cluster["secure_families_minimum"] >= 3
        and per_cluster["spectroscopic_families_minimum"] >= 1
        and per_cluster["images_minimum"] >= 8
        and per_cluster["per_image_position_uncertainties_required"]
        and len(per_cluster["complete_baryons_required"]) == 5
        and per_cluster["source_and_target_coordinate_frames_registered"]
        and per_cluster["same_catalog_halo_comparator_required"]
        and per_cluster["independent_baryon_uncertainty_ensemble_required"]
        and per_cluster["no_prior_project_formula_exposure"]
        and any("PLCK G004.5-19.5" in row["systems"] for row in config["known_non_holdout_examples"])
    )

    final_sample_stratified = (
        final["clusters"] >= 6
        and final["relaxed_side_minimum"] >= 2
        and final["disturbed_side_minimum"] >= 2
        and final["cool_core_relaxed_minimum"] >= 1
        and final["non_cool_core_relaxed_minimum"] >= 1
        and final["plane_of_sky_merger_minimum"] >= 1
        and final["projection_challenging_or_line_of_sight_merger_minimum"] >= 1
        and final["lower_mass_half_minimum"] >= 2
        and final["higher_mass_half_minimum"] >= 2
        and not final["final_six_selected_here"]
    )

    stages = config["admission_sequence"]
    authorization = config["authorization"]
    outcome_sealed = (
        [row["stage"] for row in stages] == [1, 2, 3, 4, 5]
        and not authorization["read_raw_target_coordinates"]
        and not authorization["read_lens_maps_for_candidate_selection"]
        and not authorization["select_final_six"]
        and all(row["status"] == "metadata_only_not_admitted" for row in shortlist)
    )

    expected_phenomena = {row["id"] for row in parent["other_phenomena"]}
    ladder = config["cross_domain_prediction_ladder"]
    ladder_phenomena = {
        phenomenon
        for tier in ladder
        for phenomenon in tier["phenomena"]
    }
    phenomena_ordered = (
        [tier["tier"] for tier in ladder] == [1, 2, 3]
        and ladder_phenomena == expected_phenomena
        and all(tier["why_now"] and tier["distinctive_failure"] for tier in ladder)
    )

    priority = config["priority"]
    no_selection = (
        not authorization["change_or_select_gravity_formula"]
        and not authorization["fit_universal_constants"]
        and not authorization["perform_detailed_solar_optimization"]
        and not priority["detailed_solar_optimization_now"]
    )

    gates = {
        "parent_hash_exact": parent_hash_exact,
        "public_candidate_universes_are_broad": public_universes_broad,
        "shortlist_aliases_were_absent_at_cutoff": all(
            not row["hits"] for row in alias_audit.values()
        ),
        "metadata_shortlist_is_state_balanced": shortlist_balanced,
        "availability_is_not_admission": availability_not_admission,
        "outcome_payload_remains_sealed": outcome_sealed,
        "final_sample_not_selected": final_sample_stratified,
        "all_other_dark_matter_phenomena_are_ordered": phenomena_ordered,
        "solar_is_retained_as_later_veto": (
            not priority["detailed_solar_optimization_now"]
            and "Solar-System" in priority["retain_as_later_hard_veto"]
        ),
        "no_theory_or_constant_selected": no_selection,
    }
    if set(gates) != set(config["required_gates"]):
        raise ValueError("implemented and declared gate names differ")
    if not all(config["required_gates"].values()):
        raise ValueError("every declared gate must be mandatory")

    return {
        "protocol_version": config["protocol_version"],
        "status": "completed_metadata_only_holdout_admission_protocol",
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "parent_audit": {
            "path": config["parent"]["path"],
            "expected_sha256": config["parent"]["sha256"],
            "actual_sha256": parent_actual,
            "exact": parent_hash_exact,
        },
        "alias_audit": {
            "cutoff_commit_expected": baseline,
            "cutoff_commit_resolved": resolved,
            "systems": alias_audit,
        },
        "source_universes": [
            {
                "id": row["id"],
                "systems": row["systems"],
                "scientific_role": row["scientific_role"],
            }
            for row in config["public_source_universes"]
        ],
        "metadata_shortlist": shortlist,
        "admission_state": {
            "metadata_shortlist_count": len(shortlist),
            "relaxed_side_count": states.count("relaxed"),
            "disturbed_side_count": states.count("disturbed"),
            "admitted_holdouts": 0,
            "raw_target_payload_opened": False,
            "final_six_selected": False,
        },
        "final_sample_requirements": final,
        "cross_domain_prediction_ladder": ladder,
        "priority": priority,
        "authorization_audit": authorization,
        "gate_results": gates,
        "decision": (
            "passed_metadata_only_holdout_admission_protocol"
            if all(gates.values())
            else "failed_metadata_only_holdout_admission_protocol"
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
    if report["decision"] != "passed_metadata_only_holdout_admission_protocol":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
