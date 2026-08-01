#!/usr/bin/env python3
"""Audit the full premise-level R0-R2 goal without treating inventory as identifiability."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/derived/r0_r2_goal_progress.csv"
REPORT = ROOT / "results/r0_r2_goal_progress/report.json"
INPUTS = {
    "R0": ROOT / "results/r0_observable_audit/report.json",
    "CLASH": ROOT / "results/r1_clash_observable_coverage/report.json",
    "PILOT": ROOT / "results/r1_same_system_pilot_gap/report.json",
    "CEILING": ROOT / "results/r1_ten_system_public_data_ceiling/report.json",
    "EXECUTION": ROOT / "configs/r1_execution_targets.json",
    "COMPLETION": ROOT / "results/r0_r2_completion_evidence/report.json",
}
TERMINAL = ROOT / "results/r1_rxj2129_terminal_observable_disposition/report.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in INPUTS.items()}
    terminal = json.loads(TERMINAL.read_text(encoding="utf-8")) if TERMINAL.is_file() else None
    r0, clash, pilot, ceiling, execution, completion = (
        source["R0"],
        source["CLASH"],
        source["PILOT"],
        source["CEILING"],
        source["EXECUTION"],
        source["COMPLETION"],
    )
    baseline = execution["baseline"]
    cycles = execution["cycle_progress_rule"]
    rows = [
        {
            "requirement_id": "R0_PROVENANCE",
            "requirement": "Machine-readable lineage for every scored SPARC, CLASH, and BCG scalar",
            "target": "all currently scored scalars",
            "current": str(r0["instance_provenance"]["rows"]),
            "status": "pass",
            "evidence": "results/r0_observable_audit/report.json; data/derived/r0_scored_observable_instance_provenance.csv",
            "shortfall": "",
        },
        {
            "requirement_id": "R0_CLASH_ACQUISITION",
            "requirement": "Raw/likelihood-level lensing disposition for all 20 CLASH systems",
            "target": "20 ingested systems or explicit hard public-data shortfall",
            "current": f"{clash['raw_or_likelihood_catalogs_acquired']} acquired; {clash['resolved_catalog_or_shortfall_dispositions']} dispositions",
            "status": "pass_with_documented_shortfall",
            "evidence": "results/r1_clash_observable_coverage/report.json; data/derived/r1_clash_observable_acquisition_ledger.csv",
            "shortfall": ",".join(clash["primary_source_hard_shortfall_systems"]),
        },
        {
            "requirement_id": "R0_BCG_ACQUISITION",
            "requirement": "Raw or replacement lensing/dynamical product inventory for at least 30 frozen BCG hosts",
            "target": str(cycles["replacement_cycle_definition"]),
            "current": str(cycles["current_unique_hosts_source_screened"]),
            "status": "pass_inventory_boundary_not_strict_readiness",
            "evidence": "configs/r1_execution_targets.json; data/derived/r1_replacement_source_inventory.csv",
            "shortfall": "No published numerical BCG likelihood in the audited replacement sources; most hosts lack same-object lens covariance and overlapping baryonic profiles.",
        },
        {
            "requirement_id": "R1_TEN_SYSTEM_FREEZE",
            "requirement": "Residual-blind pilot of at least 10 same systems with measured baryons and >=3 overlapping dynamics and lensing constraints",
            "target": str(pilot["target_strict_systems"]),
            "current": str(pilot["strict_r1_ready_systems"]),
            "status": "hard_public_data_shortfall",
            "evidence": "results/r1_same_system_pilot_gap/report.json; results/r1_ten_system_public_data_ceiling/report.json; data/derived/r1_same_system_pilot_gap_ledger.csv",
            "shortfall": f"{pilot['strict_ready_system_gap']} strict systems; audited structural ceiling {ceiling['current_universe_structural_ceiling']}/10 and at least {ceiling['minimum_new_rank_three_systems_required']} genuinely new rank-three systems required. The frozen public-data ceiling is binding.",
        },
        {
            "requirement_id": "R2_DYNAMICAL_RESPONSE",
            "requirement": "Reconstruct the dynamical-potential response with propagated covariance",
            "target": "frozen >=10-system same-object sample",
            "current": str(baseline["full_marginalized_jacobians_completed"]),
            "status": "empirically_unidentifiable_with_audited_public_data",
            "evidence": "configs/r1_execution_targets.json; results/r1_ten_system_public_data_ceiling/report.json",
            "shortfall": "The required ten-system sample is unattainable in the audited public-data universe. RX J2129 may only support a one-system method demonstration if its independent gates pass.",
        },
        {
            "requirement_id": "R2_WEYL_RESPONSE",
            "requirement": "Reconstruct the Weyl-potential response separately with propagated covariance",
            "target": "frozen >=10-system same-object sample",
            "current": str(baseline["full_marginalized_jacobians_completed"]),
            "status": "empirically_unidentifiable_with_audited_public_data",
            "evidence": "configs/r1_execution_targets.json; configs/r1_identifiability_targets.json; results/r1_ten_system_public_data_ceiling/report.json",
            "shortfall": "No ten-system same-object sample and no authorized population-level marginalized Weyl Jacobian. A successful RX J2129 result would remain one-system only.",
        },
        {
            "requirement_id": "R2_LATENT_CROSS_VALIDATION",
            "requirement": "Cross-validate the smallest theory-free one- versus two-potential latent response",
            "target": ">=50% held-out benchmark-gap closure in each domain without an object-class rule",
            "current": "not run",
            "status": "empirically_unidentifiable_with_audited_public_data",
            "evidence": "configs/r1_execution_targets.json; results/r1_ten_system_public_data_ceiling/report.json",
            "shortfall": "Grouped population cross-validation cannot be defined with a maximum of one possible strict-ready system after RX J2129; no one- or two-potential population claim is identifiable.",
        },
        {
            "requirement_id": "THEORY_STOP_RULE",
            "requirement": "Do not select, add, or fit another covariant action before R0-R2 pass",
            "target": "no new action fit",
            "current": "prohibited by active stage gates",
            "status": "pass",
            "evidence": "results/r0_observable_audit/report.json; configs/r1_execution_targets.json",
            "shortfall": "",
        },
    ]
    ledger = pd.DataFrame(rows)
    premise_passed = bool((ledger["status"] == "pass").all())
    terminal_stop_complete = bool(
        terminal is not None
        and completion["completion_audit_terminal"] is True
        and terminal["global_disposition"]["population_R2_identifiable"] is False
        and terminal["global_disposition"]["unification_claim"]
        == "withheld_due_public_data_identifiability_ceiling"
        and terminal["authorization"]["select_another_system"] is False
        and terminal["authorization"]["fit_new_force_or_action"] is False
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(OUTPUT, index=False)
    report = {
        "report_version": "R0-R2-goal-progress-0.4-full-completion-evidence",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_blind": True,
        "inputs": {
            name: {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(path)}
            for name, path in INPUTS.items()
        }
        | (
            {
                "RXJ2129_TERMINAL": {
                    "path": str(TERMINAL.relative_to(ROOT)).replace("\\", "/"),
                    "sha256": sha256(TERMINAL),
                }
            }
            if terminal is not None
            else {}
        ),
        "requirements": len(ledger),
        "requirements_passed_without_shortfall": int((ledger["status"] == "pass").sum()),
        "requirements_with_documented_acquisition_boundary": int(ledger["status"].str.startswith("pass_").sum()),
        "requirements_closed_by_hard_public_data_ceiling": int(
            ledger["status"].isin(
                [
                    "hard_public_data_shortfall",
                    "empirically_unidentifiable_with_audited_public_data",
                ]
            ).sum()
        ),
        "requirements_incomplete_or_not_authorized": int(
            (~ledger["status"].isin(["pass", "pass_with_documented_shortfall", "pass_inventory_boundary_not_strict_readiness"])).sum()
        ),
        "full_goal_complete": terminal_stop_complete,
        "premise_passed": premise_passed,
        "terminal_stop_rule_satisfied": terminal_stop_complete,
        "completion_evidence_requirements": completion["requirements"],
        "completion_evidence_checks": completion["evidence_checks"],
        "goal_outcome": (
            "complete_stop_unification_claim_empirically_unidentifiable"
            if terminal_stop_complete
            else "active_pending_RXJ2129_terminal_observable_disposition"
        ),
        "strict_ready_systems": pilot["strict_r1_ready_systems"],
        "target_strict_systems": pilot["target_strict_systems"],
        "hard_public_data_shortfall_established": ceiling[
            "hard_public_data_shortfall_established"
        ],
        "current_universe_structural_ceiling": ceiling[
            "current_universe_structural_ceiling"
        ],
        "minimum_strict_system_deficit_even_if_RXJ2129_passes": ceiling[
            "RXJ2129_outcome_independence"
        ]["minimum_remaining_strict_system_deficit_if_RXJ2129_passes"],
        "current_active_system": None if terminal_stop_complete else "RX J2129",
        "current_active_stage": (
            terminal["decision"]
            if terminal_stop_complete
            else execution["active_branch_decisions"]["rxj2129_strict_readiness"]["status"]
        ),
        "output": str(OUTPUT.relative_to(ROOT)).replace("\\", "/"),
        "output_bytes": OUTPUT.stat().st_size,
        "output_sha256": sha256(OUTPUT),
        "authorization": {
            "finish_already_running_RXJ2129_observable_gates": not terminal_stop_complete,
            "select_another_system": False,
            "reconstruct_dynamical_or_Weyl_response": False,
            "cross_validate_latent_response": False,
            "claim_one_or_two_potential_identification": False,
            "fit_new_force_or_action": False,
        },
        "next_action": (
            "Preserve the audited products and terminal stop disposition. Do not select "
            "another system, reconstruct a population response, or fit a new force/action "
            "without a genuinely new data release that changes the documented ceiling."
            if terminal_stop_complete
            else ceiling["decision"]["next_action"]
        ),
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
