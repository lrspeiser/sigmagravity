#!/usr/bin/env python3
"""Audit the frozen ten-system public-data ceiling after three nonpromotions."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "r1_ten_system_public_data_ceiling_protocol.json"
OUTPUT = ROOT / "results" / "r1_ten_system_public_data_ceiling" / "report.json"
INPUTS = {
    "protocol": PROTOCOL,
    "replacement_cycle3": ROOT / "results" / "r1_replacement_search_cycle3" / "report.json",
    "attainability": ROOT / "results" / "r1_ten_system_attainability" / "report.json",
    "same_system_gap": ROOT / "results" / "r1_same_system_pilot_gap" / "report.json",
    "SLACS_KCWI": ROOT / "results" / "r1_slacs_kcwi_sample_feasibility" / "report.json",
    "J0946": ROOT / "results" / "r1_j0946_jackpot_feasibility" / "report.json",
    "E325": ROOT / "results" / "r1_e325_final_disposition" / "report.json",
    "J1402": ROOT / "results" / "r1_j1402_final_disposition" / "report.json",
}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    docs = {name: read(path) for name, path in INPUTS.items()}
    protocol = docs["protocol"]
    cycle3 = docs["replacement_cycle3"]
    attainability = docs["attainability"]
    gap = docs["same_system_gap"]
    slacs = docs["SLACS_KCWI"]
    j1402 = docs["J1402"]

    external = {
        "SDSS J0946+1006": not docs["J0946"]["authorization"][
            "count_toward_ten_system_target"
        ],
        "ESO 325-G004": not docs["E325"]["authorization"][
            "count_E325_toward_ten_system_target"
        ],
        "SDSS J1402+6321": not j1402["authorization"][
            "count_J1402_toward_ten_system_target"
        ],
    }
    checks = {
        "inventory_boundary_reached": bool(
            cycle3["summary"]["cumulative_unique_hosts_source_screened"]
            >= protocol["scope"]["source_screened_BCG_hosts_minimum"]
        ),
        "candidate_universe_structural_ceiling_below_ten": bool(
            attainability["current_candidate_universe_structural_ceiling"]
            < protocol["scope"]["strict_same_system_target"]
        ),
        "three_external_candidates_completed_without_promotion": bool(
            len(external) == 3 and all(external.values())
        ),
        "all_fourteen_SLACS_KCWI_systems_screened": bool(
            slacs["sample_summary"]["candidates_audited"] == 14
        ),
        "zero_public_SLACS_KCWI_numerical_kinematic_maps": bool(
            slacs["sample_summary"]["numerical_kinematic_maps_public"] == 0
        ),
        "at_least_seven_new_rank_three_systems_still_required": bool(
            attainability[
                "minimum_new_rank_three_systems_required_even_if_every_ceiling_system_is_repaired"
            ]
            >= 7
        ),
        "zero_current_strict_ready_systems": bool(
            gap["strict_r1_ready_systems"] == 0
        ),
        "J1402_frozen_rethink_triggered": bool(
            j1402["external_search_checkpoint"]["frozen_rethink_triggered"]
        ),
        "selection_and_gravity_residual_blind": bool(
            protocol["selection_blind"] and not protocol["gravity_residuals_seen"]
        ),
    }
    hard_shortfall = all(checks.values())
    if not hard_shortfall:
        raise RuntimeError("the frozen hard-public-data-shortfall gate did not pass")

    structural_ceiling = attainability[
        "current_candidate_universe_structural_ceiling"
    ]
    target = protocol["scope"]["strict_same_system_target"]
    report = {
        "report_version": "R1C-public-data-ceiling-0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_blind": True,
        "gravity_residuals_inspected": False,
        "inputs": {
            name: {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(path),
            }
            for name, path in INPUTS.items()
        },
        "audited_public_data_universe": {
            "unique_BCG_hosts_source_screened": cycle3["summary"][
                "cumulative_unique_hosts_source_screened"
            ],
            "same_system_candidates_in_structural_ledger": gap[
                "candidate_systems_evaluated"
            ],
            "SLACS_KCWI_systems_screened": slacs["sample_summary"][
                "candidates_audited"
            ],
            "external_one_off_candidates_completed": len(external),
            "external_one_off_candidates_promoted": 0,
            "current_strict_ready_systems": gap["strict_r1_ready_systems"],
        },
        "checks": checks,
        "hard_public_data_shortfall_established": hard_shortfall,
        "ten_system_target": target,
        "current_universe_structural_ceiling": structural_ceiling,
        "structural_deficit_even_if_every_ceiling_system_is_repaired": target
        - structural_ceiling,
        "minimum_new_rank_three_systems_required": attainability[
            "minimum_new_rank_three_systems_required_even_if_every_ceiling_system_is_repaired"
        ],
        "strict_ready_deficit_before_RXJ2129_outcome": target
        - gap["strict_r1_ready_systems"],
        "RXJ2129_outcome_independence": {
            "maximum_strict_ready_if_RXJ2129_passes": 1,
            "minimum_remaining_strict_system_deficit_if_RXJ2129_passes": 9,
            "ten_system_shortfall_changes_if_RXJ2129_passes": False,
            "allowed_success_role": protocol["RXJ2129_branches"]["if_strict_ready"],
            "allowed_failure_role": protocol["RXJ2129_branches"]["if_not_strict_ready"],
        },
        "decision": {
            "R1C_ten_system_freeze": "unattainable_with_audited_public_data",
            "R2_population_response_reconstruction": "empirically_unidentifiable_without_ten_system_sample",
            "R2_latent_response_cross_validation": "not_authorized",
            "unification_claim": "withheld_due_public_data_identifiability_ceiling",
            "next_action": "Finish RX J2129 only because its immutable H2/X4 gates are already running. Then issue its branch-specific disposition; do not select another object or fit a force law.",
        },
        "authorization": {
            "finish_RXJ2129_H2_and_X4": True,
            "freeze_ten_system_sample": False,
            "select_fourth_external_one_off_target": False,
            "run_population_R2_cross_validation": False,
            "claim_one_or_two_potential_population_identification": False,
            "fit_new_force_or_action": False,
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
