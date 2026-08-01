#!/usr/bin/env python3
"""Verify the R0-R2 objective requirement by requirement from current artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/derived/r0_r2_completion_evidence.csv"
REPORT = ROOT / "results/r0_r2_completion_evidence/report.json"
TERMINAL = ROOT / "results/r1_rxj2129_terminal_observable_disposition/report.json"
INPUTS = {
    "R0_CONFIG": ROOT / "configs/r0_observable_audit.json",
    "R0_REPORT": ROOT / "results/r0_observable_audit/report.json",
    "R0_MATRIX": ROOT / "data/derived/r0_observable_provenance.csv",
    "R0_INSTANCES": ROOT / "data/derived/r0_scored_observable_instance_provenance.csv",
    "CLASH_REPORT": ROOT / "results/r1_clash_observable_coverage/report.json",
    "CLASH_LEDGER": ROOT / "data/derived/r1_clash_observable_acquisition_ledger.csv",
    "BCG_REPORT": ROOT / "results/r1_replacement_search_cycle3/report.json",
    "BCG_LEDGER": ROOT / "data/derived/r1_replacement_cycle3_candidate_ledger.csv",
    "PILOT_REPORT": ROOT / "results/r1_same_system_pilot_gap/report.json",
    "PILOT_LEDGER": ROOT / "data/derived/r1_same_system_pilot_gap_ledger.csv",
    "CEILING_REPORT": ROOT / "results/r1_ten_system_public_data_ceiling/report.json",
}

PROVENANCE_FIELDS = {
    "dataset",
    "source_variant",
    "scored_column",
    "score_role",
    "lineage_id",
    "system",
    "score_input_row_index_zero_based",
    "system_point_index_zero_based",
    "scored_value",
    "score_unit",
    "score_input_file",
    "score_input_sha256",
    "source_row_locator",
    "full_covariance_artifact_ingested_for_this_score",
    "raw_observable",
    "local_file",
    "publication",
    "transformation",
    "metric_or_dynamics_assumptions",
    "covariance_status",
    "evidence_level",
    "alternative_theory_forward_modeling",
}
LINEAGE_FIELDS = {
    "raw_observable",
    "local_file",
    "publication",
    "transformation",
    "metric_or_dynamics_assumptions",
    "covariance_status",
    "evidence_level",
    "alternative_theory_forward_modeling",
}
ACCEPTED_TERMINAL_STATUSES = {
    "pass",
    "pass_with_documented_shortfall",
    "closed_by_hard_public_data_shortfall",
    "closed_empirically_unidentifiable",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().eq("true")


def all_nonblank(frame: pd.DataFrame, columns: set[str]) -> bool:
    return bool(
        frame[list(columns)].notna().all().all()
        and frame[list(columns)].astype(str).apply(lambda col: col.str.strip().ne("").all()).all()
    )


def verify_instance_source_hashes(instances: pd.DataFrame) -> tuple[int, list[str]]:
    records = instances[["score_input_file", "score_input_sha256"]].drop_duplicates()
    if records["score_input_file"].duplicated().any():
        raise ValueError("one score input file is associated with multiple checksums")
    failures: list[str] = []
    for record in records.itertuples(index=False):
        path = Path(record.score_input_file)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file() or sha256(path) != record.score_input_sha256:
            failures.append(str(path))
    return len(records), failures


def verify_bcg_profile_hashes(ledger: pd.DataFrame) -> list[str]:
    failures: list[str] = []
    for record in ledger[["profile_plot", "profile_plot_sha256"]].itertuples(index=False):
        path = Path(record.profile_plot)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file() or sha256(path).upper() != record.profile_plot_sha256.upper():
            failures.append(str(path))
    return failures


def verify_embedded_input_hashes(report: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for name, record in report["inputs"].items():
        path = Path(record["path"])
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file() or sha256(path) != record["sha256"]:
            failures.append(name)
    return failures


def verify_record_mapping(
    records: dict[str, dict[str, Any]], prefix: str
) -> list[str]:
    failures: list[str] = []
    for name, record in records.items():
        path = Path(record["path"])
        if not path.is_absolute():
            path = ROOT / path
        valid = path.is_file()
        if valid and "bytes" in record:
            valid = path.stat().st_size == int(record["bytes"])
        if valid:
            valid = sha256(path).lower() == str(record["sha256"]).lower()
        if not valid:
            failures.append(f"{prefix}:{name}")
    return failures


def verify_terminal_artifacts(terminal: dict[str, Any] | None) -> list[str] | None:
    if terminal is None:
        return None
    failures = verify_record_mapping(terminal["inputs"], "terminal_input")
    integrity = terminal["artifact_integrity"]
    h2 = integrity["H2"]
    if (
        h2["immutable_input_artifact_count"] != 11
        or h2["artifact_count"] != 4
        or not h2["all_immutable_inputs_rehashed"]
        or not h2["all_reported_artifacts_rehashed"]
    ):
        failures.append("terminal:H2_integrity_summary")
    failures.extend(
        verify_record_mapping(h2["immutable_inputs"], "H2_immutable_input")
    )
    failures.extend(verify_record_mapping(h2["artifacts"], "H2_output"))
    failures.extend(
        verify_record_mapping(
            {
                "execution_config": h2["execution_config"],
                "static_pre_pixel_audit": h2["static_pre_pixel_audit"],
            },
            "H2_protocol",
        )
    )

    x4 = integrity["X4"]
    if (
        x4["manifest_product_count"] != 116
        or x4["response_product_count"] != 108
        or x4["detector_map_count"] != 8
        or x4["input_artifact_count"] != 4
        or not x4["all_implementation_inputs_rehashed"]
        or not x4["all_manifest_products_rehashed"]
    ):
        failures.append("terminal:X4_integrity_summary")
    failures.extend(
        verify_record_mapping(x4["implementation_inputs"], "X4_implementation_input")
    )
    manifest_path = Path(x4["manifest_path"])
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    if (
        not manifest_path.is_file()
        or sha256(manifest_path).lower() != x4["manifest_sha256"].lower()
    ):
        failures.append("terminal:X4_manifest")
    else:
        manifest = load_json(manifest_path)
        products = {
            f"{index:03d}": record
            for index, record in enumerate(manifest["products"])
        }
        failures.extend(verify_record_mapping(products, "X4_manifest_product"))
    if not all(terminal["status_consistency_checks"].values()):
        failures.append("terminal:status_consistency")
    return failures


def main() -> None:
    missing_inputs = [name for name, path in INPUTS.items() if not path.is_file()]
    if missing_inputs:
        raise RuntimeError("completion audit inputs are absent: " + ", ".join(missing_inputs))

    config = load_json(INPUTS["R0_CONFIG"])
    r0 = load_json(INPUTS["R0_REPORT"])
    clash = load_json(INPUTS["CLASH_REPORT"])
    bcg = load_json(INPUTS["BCG_REPORT"])
    pilot = load_json(INPUTS["PILOT_REPORT"])
    ceiling = load_json(INPUTS["CEILING_REPORT"])
    terminal = load_json(TERMINAL) if TERMINAL.is_file() else None
    matrix = pd.read_csv(INPUTS["R0_MATRIX"])
    instances = pd.read_csv(INPUTS["R0_INSTANCES"])
    clash_ledger = pd.read_csv(INPUTS["CLASH_LEDGER"])
    bcg_ledger = pd.read_csv(INPUTS["BCG_LEDGER"])
    pilot_ledger = pd.read_csv(INPUTS["PILOT_LEDGER"])

    expected_matrix = {
        tuple(record[key] for key in ("dataset", "source_variant", "scored_column", "score_role", "lineage_id"))
        for record in config["records"]
    }
    observed_matrix = {
        tuple(record[key] for key in ("dataset", "source_variant", "scored_column", "score_role", "lineage_id"))
        for record in matrix.to_dict(orient="records")
    }
    matrix_exact = bool(
        expected_matrix == observed_matrix
        and len(matrix) == 27
        and set(matrix["dataset"]) == {"SPARC", "CLASH", "BCG"}
        and r0["provenance_matrix"]["all_required_scored_columns_covered"]
        and sha256(INPUTS["R0_MATRIX"]) == r0["provenance_matrix"]["sha256"]
    )
    instance_schema = bool(PROVENANCE_FIELDS.issubset(instances.columns))
    instance_lineage = bool(
        instance_schema
        and len(instances) == 19030
        and all_nonblank(instances, LINEAGE_FIELDS | {"source_row_locator", "score_unit"})
        and np.isfinite(pd.to_numeric(instances["scored_value"]).to_numpy()).all()
        and not instances.duplicated(
            [
                "dataset",
                "source_variant",
                "system",
                "system_point_index_zero_based",
                "scored_column",
            ]
        ).any()
        and sha256(INPUTS["R0_INSTANCES"])
        == r0["instance_provenance"]["sha256"]
    )
    source_file_count, source_hash_failures = verify_instance_source_hashes(instances)
    source_hashes_pass = bool(source_file_count == 133 and not source_hash_failures)

    clash_acquired = as_bool(clash_ledger["raw_or_likelihood_catalog_acquired"])
    clash_gravity_blind = ~as_bool(clash_ledger["gravity_target_used"])
    clash_pass = bool(
        len(clash_ledger) == 20
        and clash_ledger["system"].nunique() == 20
        and int(clash_acquired.sum()) == 19
        and clash["resolved_catalog_or_shortfall_dispositions"] == 20
        and clash["primary_source_hard_shortfall_systems"] == ["RXJ1532"]
        and clash["coverage_or_hard_shortfall_gate_passed"]
        and clash_gravity_blind.all()
    )

    bcg_hash_failures = verify_bcg_profile_hashes(bcg_ledger)
    bcg_pass = bool(
        len(bcg_ledger) == 32
        and bcg_ledger["system"].nunique() == 32
        and not bcg_hash_failures
        and bcg["summary"]["source_bcg_hosts"] == 32
        and bcg["summary"]["cumulative_unique_hosts_source_screened"] >= 30
        and bcg["summary"]["inventory_boundary_reached"]
    )

    structural = as_bool(pilot_ledger["three_plus_three_structural_pass"])
    baryons = as_bool(pilot_ledger["complete_baryonic_forward_inputs"])
    strict = as_bool(pilot_ledger["strict_r1_ready"])
    pilot_shortfall = bool(
        pilot["selection_blind"]
        and len(pilot_ledger) == 15
        and pilot_ledger["system"].nunique() == 15
        and int(structural.sum()) == 2
        and int(baryons.sum()) == 0
        and int(strict.sum()) == 0
        and pilot["target_strict_systems"] == 10
        and pilot["route_rethink_triggered"]
    )
    ceiling_input_hash_failures = verify_embedded_input_hashes(ceiling)
    ceiling_pass = bool(
        ceiling["selection_blind"]
        and not ceiling["gravity_residuals_inspected"]
        and ceiling["hard_public_data_shortfall_established"]
        and ceiling["ten_system_target"] == 10
        and ceiling["current_universe_structural_ceiling"] == 3
        and ceiling["minimum_new_rank_three_systems_required"] == 7
        and ceiling["RXJ2129_outcome_independence"][
            "minimum_remaining_strict_system_deficit_if_RXJ2129_passes"
        ]
        == 9
        and all(ceiling["checks"].values())
        and not ceiling_input_hash_failures
    )
    if not pilot_shortfall or not ceiling_pass:
        raise RuntimeError("the frozen R1 public-data shortfall evidence is inconsistent")

    terminal_artifact_failures = verify_terminal_artifacts(terminal)
    terminal_stop = bool(
        terminal is not None
        and terminal_artifact_failures == []
        and terminal["global_disposition"]["population_R2_identifiable"] is False
        and terminal["global_disposition"]["unification_claim"]
        == "withheld_due_public_data_identifiability_ceiling"
        and terminal["authorization"]["reconstruct_dynamical_or_Weyl_response"] is False
        and terminal["authorization"]["cross_validate_latent_response"] is False
        and terminal["authorization"]["select_another_system"] is False
        and terminal["authorization"]["fit_new_force_or_action"] is False
    )
    no_action_pass = bool(
        r0["stage_decision"]["new_force_law"] == "prohibited_until_R0_R2_pass"
        and clash["authorization"]["fit_new_force_or_action"] is False
        and pilot["authorization"]["fit_new_force_or_action"] is False
        and ceiling["authorization"]["fit_new_force_or_action"] is False
        and (terminal is None or terminal["authorization"]["fit_new_force_or_action"] is False)
    )
    terminal_r2_status = (
        "closed_empirically_unidentifiable"
        if terminal_stop
        else "pending_terminal_observable_disposition"
    )
    rows = [
        {
            "requirement_id": "R0_PROVENANCE_MATRIX",
            "requirement": "Every scored SPARC, CLASH, and BCG column has a declared lineage",
            "status": "pass" if matrix_exact else "failed_evidence_check",
            "observed": f"{len(matrix)} lineage rows across {matrix['dataset'].nunique()} datasets",
            "evidence": "data/derived/r0_observable_provenance.csv",
        },
        {
            "requirement_id": "R0_SCALAR_PROVENANCE",
            "requirement": "Every scored scalar records raw measurement, publication/file, transformation, assumptions, covariance, and alternative-theory disposition",
            "status": "pass" if instance_lineage else "failed_evidence_check",
            "observed": f"{len(instances)} unique finite scalar lineage rows",
            "evidence": "data/derived/r0_scored_observable_instance_provenance.csv",
        },
        {
            "requirement_id": "R0_EXACT_SOURCE_HASHES",
            "requirement": "Every scored scalar remains tied to the current exact source file",
            "status": "pass" if source_hashes_pass else "failed_evidence_check",
            "observed": f"{source_file_count} unique current input files; {len(source_hash_failures)} hash failures",
            "evidence": "data/derived/r0_scored_observable_instance_provenance.csv",
        },
        {
            "requirement_id": "R0_CLASH_20",
            "requirement": "Raw/likelihood lensing disposition for all 20 CLASH systems, or an explicit hard public-data shortfall",
            "status": "pass_with_documented_shortfall" if clash_pass else "failed_evidence_check",
            "observed": "19 acquired; RXJ1532 explicit primary-source shortfall; 20/20 dispositions",
            "evidence": "results/r1_clash_observable_coverage/report.json; data/derived/r1_clash_observable_acquisition_ledger.csv",
        },
        {
            "requirement_id": "R0_BCG_30",
            "requirement": "Raw or replacement product screen for at least 30 frozen BCG hosts",
            "status": "pass" if bcg_pass else "failed_evidence_check",
            "observed": f"{bcg_ledger['system'].nunique()} source-sample hosts with checksum-verified profile artifacts; 45 cumulative screened",
            "evidence": "results/r1_replacement_search_cycle3/report.json; data/derived/r1_replacement_cycle3_candidate_ledger.csv",
        },
        {
            "requirement_id": "R1_TEN_SYSTEM_PILOT",
            "requirement": "Residual-blind >=10-system pilot with measured baryons and >=3 overlapping dynamics/lensing constraints",
            "status": "closed_by_hard_public_data_shortfall",
            "observed": "15 screened; 2 structural 3+3 passes; 0 complete baryonic profiles; 0 strict-ready; structural ceiling 3",
            "evidence": "results/r1_same_system_pilot_gap/report.json; results/r1_ten_system_public_data_ceiling/report.json",
        },
        {
            "requirement_id": "R2_DYNAMICAL_RESPONSE",
            "requirement": "Separate dynamical-potential response with propagated covariance",
            "status": terminal_r2_status,
            "observed": "not reconstructed because the prerequisite same-system population is empirically unavailable",
            "evidence": "results/r1_ten_system_public_data_ceiling/report.json",
        },
        {
            "requirement_id": "R2_WEYL_RESPONSE",
            "requirement": "Separate Weyl-potential response with propagated covariance",
            "status": terminal_r2_status,
            "observed": "not reconstructed because the prerequisite same-system population is empirically unavailable",
            "evidence": "results/r1_ten_system_public_data_ceiling/report.json",
        },
        {
            "requirement_id": "R2_LATENT_CROSS_VALIDATION",
            "requirement": "Theory-free one/two-response grouped cross-validation using baryons and boundary data only",
            "status": terminal_r2_status,
            "observed": "not run; maximum possible strict sample is one, so held-out population closure is undefined",
            "evidence": "results/r1_ten_system_public_data_ceiling/report.json",
        },
        {
            "requirement_id": "PREMISE_DECISION",
            "requirement": "Retain one field only at >=50% dual-domain closure, require two identifiable responses for two potentials, otherwise stop the claim",
            "status": terminal_r2_status,
            "observed": "neither population response is identifiable; one-field and two-potential claims withheld",
            "evidence": "results/r1_ten_system_public_data_ceiling/report.json",
        },
        {
            "requirement_id": "NO_NEW_ACTION",
            "requirement": "Do not select or fit another covariant action before the premise audit passes",
            "status": "pass" if no_action_pass else "failed_evidence_check",
            "observed": "new target, response, cross-validation, and force/action fitting remain unauthorized",
            "evidence": "results/r0_observable_audit/report.json; results/r1_ten_system_public_data_ceiling/report.json",
        },
    ]
    ledger = pd.DataFrame(rows)
    terminally_disposed = bool(
        terminal_stop and ledger["status"].isin(ACCEPTED_TERMINAL_STATUSES).all()
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(OUTPUT, index=False)
    input_records = {
        name: {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for name, path in INPUTS.items()
    }
    if terminal is not None:
        input_records["TERMINAL"] = {
            "path": str(TERMINAL.relative_to(ROOT)).replace("\\", "/"),
            "bytes": TERMINAL.stat().st_size,
            "sha256": sha256(TERMINAL),
        }
    report = {
        "report_version": "R0-R2-completion-evidence-0.2-terminal-integrity",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_blind": True,
        "inputs": input_records,
        "requirements": len(ledger),
        "requirements_by_status": {
            str(key): int(value) for key, value in ledger["status"].value_counts().items()
        },
        "evidence_checks": {
            "provenance_matrix_exact": matrix_exact,
            "scalar_lineage_complete": instance_lineage,
            "unique_scored_source_files_rehashed": source_file_count,
            "scored_source_hash_failures": source_hash_failures,
            "CLASH_20_dispositions_complete": clash_pass,
            "BCG_30_inventory_boundary_complete": bcg_pass,
            "BCG_profile_hash_failures": bcg_hash_failures,
            "R1_shortfall_recomputed": pilot_shortfall,
            "ten_system_ceiling_checks_all_pass": ceiling_pass,
            "ten_system_ceiling_input_hash_failures": ceiling_input_hash_failures,
            "terminal_stop_rule_satisfied": terminal_stop,
            "terminal_artifact_integrity_checked": terminal is not None,
            "terminal_artifact_integrity_failures": terminal_artifact_failures,
            "no_new_action_gate_preserved": no_action_pass,
        },
        "premise_passed": False,
        "completion_audit_terminal": terminally_disposed,
        "goal_outcome": (
            "complete_stop_unification_claim_empirically_unidentifiable"
            if terminally_disposed
            else "active_pending_RXJ2129_terminal_observable_disposition"
        ),
        "output": str(OUTPUT.relative_to(ROOT)).replace("\\", "/"),
        "output_bytes": OUTPUT.stat().st_size,
        "output_sha256": sha256(OUTPUT),
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
