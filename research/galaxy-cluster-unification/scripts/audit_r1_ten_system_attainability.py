#!/usr/bin/env python3
"""Compute the residual-blind upper bound on the current ten-system R1 pilot."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "data/derived/r1_same_system_pilot_gap_ledger.csv"
TARGETS = ROOT / "configs/r1_execution_targets.json"
OUTPUT = ROOT / "data/derived/r1_ten_system_attainability.csv"
REPORT = ROOT / "results/r1_ten_system_attainability/report.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ledger = pd.read_csv(LEDGER)
    targets = json.loads(TARGETS.read_text(encoding="utf-8"))
    strict = targets["strict_system_definition"]
    required_rank = int(strict["resolved_lensing_radial_points_minimum"])
    target_systems = 10
    ranked = ledger.copy()
    ranked["structural_rank_three_ceiling"] = (
        ranked["structural_radial_rank_upper_bound"] >= required_rank
    )
    ranked["accepted_dynamics_and_rank_three"] = (
        ranked["structural_rank_three_ceiling"]
        & ranked["dynamics_internal_consistency_pass"]
        & (ranked["reported_or_numerical_dynamics_points"] >= int(strict["resolved_dynamics_radial_points_minimum"]))
    )
    ranked["remaining_to_strict_readiness"] = ranked["primary_obstruction"]
    columns = [
        "system",
        "candidate_class",
        "reported_or_numerical_dynamics_points",
        "lensing_points_on_dynamics_support",
        "structural_radial_rank_upper_bound",
        "structural_rank_three_ceiling",
        "dynamics_internal_consistency_pass",
        "accepted_dynamics_and_rank_three",
        "complete_baryonic_forward_inputs",
        "coordinate_covariance_independent_of_fitted_gr_residuals",
        "strict_r1_ready",
        "remaining_to_strict_readiness",
    ]
    output = ranked[columns].sort_values(
        ["structural_rank_three_ceiling", "structural_radial_rank_upper_bound", "system"],
        ascending=[False, False, True],
        kind="stable",
    )
    ceiling_systems = output.loc[output["structural_rank_three_ceiling"], "system"].tolist()
    accepted_geometry_dynamics = output.loc[
        output["accepted_dynamics_and_rank_three"], "system"
    ].tolist()
    strict_ready = output.loc[output["strict_r1_ready"], "system"].tolist()
    structural_ceiling = len(ceiling_systems)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT, index=False)
    report = {
        "report_version": "R1-ten-system-attainability-0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_blind": True,
        "selection_inputs": "published/pre-fit radial coverage and previously frozen science gates only; no gravity residual or model score",
        "inputs": {
            "pilot_ledger": {"path": str(LEDGER.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(LEDGER)},
            "execution_targets": {"path": str(TARGETS.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(TARGETS)},
        },
        "candidate_systems_evaluated": len(output),
        "target_strict_systems": target_systems,
        "current_strict_ready_systems": len(strict_ready),
        "current_strict_ready_names": strict_ready,
        "current_candidate_universe_structural_ceiling": structural_ceiling,
        "structural_ceiling_systems": ceiling_systems,
        "accepted_dynamics_and_rank_three_systems": accepted_geometry_dynamics,
        "minimum_new_rank_three_systems_required_even_if_every_ceiling_system_is_repaired": max(0, target_systems - structural_ceiling),
        "minimum_new_strict_systems_required_at_current_state": max(0, target_systems - len(strict_ready)),
        "ten_system_freeze_attainable_from_current_candidate_universe": structural_ceiling >= target_systems,
        "decision": "new_structurally_qualified_source_class_required",
        "forbidden_shortcut": "Do not count source-screened hosts, outer strong-lens images, repeated images at one radius, GR-fit covariance, or failed dynamics reductions as independent same-support response modes.",
        "next_acquisition_requirement": "Add at least seven new systems that already have a pre-fit structural radial-rank ceiling >=3, >=3 accepted dynamics bins on the same support, and observable-level lens inputs before investing in baryonic/covariance completion.",
        "output": str(OUTPUT.relative_to(ROOT)).replace("\\", "/"),
        "output_bytes": OUTPUT.stat().st_size,
        "output_sha256": sha256(OUTPUT),
        "authorization": {
            "continue_RXJ2129_strict_readiness": True,
            "reuse_current_rank_zero_candidates_for_ten_system_claim": False,
            "freeze_ten_system_sample": False,
            "reconstruct_dynamical_or_Weyl_response": False,
            "fit_new_force_or_action": False,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
