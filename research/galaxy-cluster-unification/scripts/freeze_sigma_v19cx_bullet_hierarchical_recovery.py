#!/usr/bin/env python3
"""Freeze V19CX after V19CW proves observation-hierarchy equivalence."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_sigma_v19cx_bullet_hierarchical_recovery as successor

ROOT = Path(__file__).resolve().parents[1]
X2_CONFIG = ROOT / "configs" / "sigma_v19x2_unified_spectral_combination_commissioning.json"
X2_RUNNER = ROOT / "scripts" / "run_sigma_v19x2_unified_spectral_combination_commissioning.py"
X2_REPORT = ROOT / "results" / "sigma_v19x2_unified_spectral_combination_commissioning" / "report.json"
CW_CONFIG = ROOT / "configs" / "sigma_v19cw_observation_hierarchy_equivalence.json"
CW_RUNNER = ROOT / "scripts" / "run_sigma_v19cw_observation_hierarchy_equivalence.py"
CW_REPORT = ROOT / "results" / "sigma_v19cw_observation_hierarchy_equivalence" / "report.json"
SUCCESSOR = ROOT / "scripts" / "run_sigma_v19cx_bullet_hierarchical_recovery.py"
OUTPUT = ROOT / "configs" / "sigma_v19cx_bullet_hierarchical_recovery.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def validate_cw() -> tuple[dict[str, Any], dict[str, Any]]:
    for path in (CW_CONFIG, CW_RUNNER, CW_REPORT):
        if not path.is_file():
            raise RuntimeError(f"V19CX equivalence parent is absent: {path}")
    config = load_json(CW_CONFIG)
    report = load_json(CW_REPORT)
    if config.get("freeze_state") != "frozen_after_v19x2_large_stack_header_merge_failure_before_hierarchical_execution":
        raise RuntimeError("V19CW config is not frozen at the required boundary")
    if config["implementation"]["runner_sha256"] != sha256(CW_RUNNER):
        raise RuntimeError("V19CW runner changed")
    if report.get("config_sha256") != sha256(CW_CONFIG) or report.get("runner_sha256") != sha256(CW_RUNNER):
        raise RuntimeError("V19CW report names another implementation")
    if report.get("status") != "observation_hierarchy_equivalent_and_bullet_recovery_may_be_frozen":
        raise RuntimeError("V19CW did not pass equivalence")
    if not report.get("gates") or not all(report["gates"].values()):
        raise RuntimeError("V19CW report contains a failed gate")
    if report.get("bullet_hierarchical_execution_authorized") is not True:
        raise RuntimeError("V19CW did not authorize a Bullet recovery freeze")
    for key in ("gravity_formula_or_parameter_changed", "source_state_or_lensing_target_opened", "v19bq_or_v19bs_run", "action_derived"):
        if report.get(key) is not False:
            raise RuntimeError(f"V19CW crossed sealed boundary: {key}")
    return config, report


def freeze() -> dict[str, Any]:
    for path in (X2_CONFIG, X2_RUNNER, X2_REPORT, Path(__file__).resolve(), SUCCESSOR):
        if not path.is_file():
            raise RuntimeError(f"V19CX parent is absent: {path}")
    x2 = load_json(X2_CONFIG)
    x2_report = load_json(X2_REPORT)
    cw, _cw_report = validate_cw()
    if x2.get("freeze_state") != "frozen_after_terminal_v19w5_pass":
        raise RuntimeError("V19X2 config is not frozen")
    if x2_report.get("status") != "unified_spectral_combination_commissioning_execution_failed":
        raise RuntimeError("V19X2 no longer records the direct-stack failure")
    config = copy.deepcopy(x2)
    config.update(
        {
            "protocol_version": "SIGMA-V19CX-BULLET-HIERARCHICAL-RECOVERY-1.0.0",
            "freeze_state": successor.FROZEN_STATE,
            "status": "frozen after V19CW passed every Abell direct-versus-hierarchical product and forward-fold gate; before hierarchical Bullet execution or any temperature fit, source-state construction, lensing, gravity, V19BQ, V19BS or action access",
            "purpose": "Recover the mechanically failed large Bullet integrated response with the V19CW-equivalent observation hierarchy, then run the otherwise unchanged V19X2 commissioning fits and gates.",
        }
    )
    config["parents"].update(
        {
            "v19x2_failed_report": relative(X2_REPORT),
            "v19x2_failed_report_sha256": sha256(X2_REPORT),
            "v19cw_config": relative(CW_CONFIG),
            "v19cw_config_sha256": sha256(CW_CONFIG),
            "v19cw_runner": relative(CW_RUNNER),
            "v19cw_runner_sha256": sha256(CW_RUNNER),
            "v19cw_report": relative(CW_REPORT),
            "v19cw_report_sha256": sha256(CW_REPORT),
        }
    )
    config["hierarchy"] = copy.deepcopy(cw["hierarchy"])
    config["runtime_remediation"] = {
        "maximum_direct_stack_cells": 1270,
        "rule": "Use the already hash-frozen direct product when available; otherwise apply the V19CW observation hierarchy iff aperture cells exceed 1270; retain the frozen direct V19X2 path at or below 1270.",
        "threshold_uses_cluster_or_scientific_outcome": False,
        "abell_direct_reference_cells": 1270,
        "bullet_failed_direct_cells": 3812,
        "v19cw_equivalence_report_sha256": sha256(CW_REPORT),
        "direct_references": {
            "ABELL2146_integrated": copy.deepcopy(cw["direct_reference"]["products"]),
            "ABELL2146_bin62": {
                "source_grouped": {
                    "path": "results/sigma_v19x2_unified_spectral_combination_commissioning/frozen_products/ABELL2146_bin62/ABELL2146_bin62_src_grp.pi",
                    "bytes": 89280,
                    "sha256": "4b3f76b6fc531b9d61c127a3628d457ab3e7f24044d1778f285b8bd8388801ed"
                },
                "background": {
                    "path": "results/sigma_v19x2_unified_spectral_combination_commissioning/frozen_products/ABELL2146_bin62/ABELL2146_bin62_bkg.pi",
                    "bytes": 57600,
                    "sha256": "8f8c2ca6c74ba69f45bedac990f881a023a2a9411a7401d66a0d028fee050260"
                },
                "arf": {
                    "path": "results/sigma_v19x2_unified_spectral_combination_commissioning/frozen_products/ABELL2146_bin62/ABELL2146_bin62_src.arf",
                    "bytes": 46080,
                    "sha256": "39ae36f33893cc8ac8d20b99cea72e12c091bb7e7e152c8b7bd8e1ac00ae5dfb"
                },
                "rmf": {
                    "path": "results/sigma_v19x2_unified_spectral_combination_commissioning/frozen_products/ABELL2146_bin62/ABELL2146_bin62_src.rmf",
                    "bytes": 2208960,
                    "sha256": "564e09c0f684e9bdfe74a4e63f09fd19b927ef6a0325c899568bd245d99c9e43"
                }
            }
        }
    }
    config["execution"].update(
        {
            "runner": relative(SUCCESSOR),
            "scratch_root": "/home/henry/sigma-v19cx-bullet-hierarchical-recovery/v100",
            "result_root": "results/sigma_v19cx_bullet_hierarchical_recovery",
            "reason_for_serial_combination": "V19CW commissioned serial observation groups; V19CX preserves that exact execution topology.",
        }
    )
    config["authorization"].update(
        {
            "run_v19cw_equivalent_hierarchy_only_above_frozen_cell_threshold": True,
            "overwrite_v19x2_failure_report": False,
            "run_v19bq_or_v19bs": False,
            "derive_action": False,
            "change_gravity_formula_parameter_source_state_or_lensing_target": False,
        }
    )
    config["advance"] = {
        "if_all_gates_pass": "freeze V19X3B with V19CX passed report/config/runner supplied as the commissioning authority",
        "if_any_gate_fails": "retain the terminal failure; do not relax hierarchy equivalence, spectral, fit or source gates",
        "gravity_theory_tested_here": False,
    }
    config["claim_boundary"] = [
        "V19CX changes only the numerically commissioned response-reduction topology for stacks larger than the largest successful direct stack.",
        "Registered cells, source and background PHA rules, ASCA scaling, exposure origin, RMF threshold, grouping, spectral model, fit order and gates are inherited unchanged from V19X2.",
        "A pass authorizes full regional spectral production; it is not evidence for Sigma Gravity.",
        "No source state, lensing, halo, gravity parameter, V19BQ, V19BS or action is opened here.",
    ]
    config["implementation"].update(
        {
            "freezer": relative(Path(__file__).resolve()),
            "freezer_sha256": sha256(Path(__file__).resolve()),
            "runner": relative(SUCCESSOR),
            "runner_sha256": sha256(SUCCESSOR),
        }
    )
    successor.validate_frozen(config)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config


def main() -> None:
    config = freeze()
    print(OUTPUT)
    print(config["status"])


if __name__ == "__main__":
    main()
