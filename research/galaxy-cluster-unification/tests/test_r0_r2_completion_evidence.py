from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import audit_r0_r2_completion_evidence as completion_audit


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results/r0_r2_completion_evidence/report.json"
TERMINAL = ROOT / "results/r1_rxj2129_terminal_observable_disposition/report.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_completion_evidence_covers_the_full_premise_objective() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    ledger_path = ROOT / report["output"]
    ledger = pd.read_csv(ledger_path).set_index("requirement_id")
    terminal = TERMINAL.is_file()

    assert report["report_version"] == "R0-R2-completion-evidence-0.2-terminal-integrity"
    assert report["requirements"] == 11
    assert report["output_bytes"] == ledger_path.stat().st_size
    assert report["output_sha256"] == _sha256(ledger_path)
    assert set(ledger.index) == {
        "R0_PROVENANCE_MATRIX",
        "R0_SCALAR_PROVENANCE",
        "R0_EXACT_SOURCE_HASHES",
        "R0_CLASH_20",
        "R0_BCG_30",
        "R1_TEN_SYSTEM_PILOT",
        "R2_DYNAMICAL_RESPONSE",
        "R2_WEYL_RESPONSE",
        "R2_LATENT_CROSS_VALIDATION",
        "PREMISE_DECISION",
        "NO_NEW_ACTION",
    }
    checks = report["evidence_checks"]
    assert checks["provenance_matrix_exact"]
    assert checks["scalar_lineage_complete"]
    assert checks["unique_scored_source_files_rehashed"] == 133
    assert checks["scored_source_hash_failures"] == []
    assert checks["CLASH_20_dispositions_complete"]
    assert checks["BCG_30_inventory_boundary_complete"]
    assert checks["BCG_profile_hash_failures"] == []
    assert checks["R1_shortfall_recomputed"]
    assert checks["ten_system_ceiling_checks_all_pass"]
    assert checks["ten_system_ceiling_input_hash_failures"] == []
    assert checks["no_new_action_gate_preserved"]
    assert checks["terminal_stop_rule_satisfied"] is terminal
    assert checks["terminal_artifact_integrity_checked"] is terminal
    assert checks["terminal_artifact_integrity_failures"] == ([] if terminal else None)

    assert ledger.loc["R0_PROVENANCE_MATRIX", "status"] == "pass"
    assert ledger.loc["R0_SCALAR_PROVENANCE", "status"] == "pass"
    assert ledger.loc["R0_EXACT_SOURCE_HASHES", "status"] == "pass"
    assert ledger.loc["R0_CLASH_20", "status"] == "pass_with_documented_shortfall"
    assert ledger.loc["R0_BCG_30", "status"] == "pass"
    assert ledger.loc["R1_TEN_SYSTEM_PILOT", "status"] == (
        "closed_by_hard_public_data_shortfall"
    )
    r2_status = (
        "closed_empirically_unidentifiable"
        if terminal
        else "pending_terminal_observable_disposition"
    )
    for requirement in (
        "R2_DYNAMICAL_RESPONSE",
        "R2_WEYL_RESPONSE",
        "R2_LATENT_CROSS_VALIDATION",
        "PREMISE_DECISION",
    ):
        assert ledger.loc[requirement, "status"] == r2_status
    assert ledger.loc["NO_NEW_ACTION", "status"] == "pass"
    assert report["premise_passed"] is False
    assert report["completion_audit_terminal"] is terminal


def test_terminal_completion_rehashes_every_synthetic_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def record(relative: str, content: bytes) -> dict:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return {
            "path": relative,
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "integrity_passed": True,
        }

    terminal_inputs = {"protocol": record("terminal/protocol.json", b"protocol")}
    h2_inputs = {
        f"input_{index:02d}": record(f"h2/input_{index:02d}.bin", bytes([index]))
        for index in range(11)
    }
    h2_outputs = {
        f"output_{index:02d}": record(f"h2/output_{index:02d}.bin", bytes([20 + index]))
        for index in range(4)
    }
    h2_config = record("h2/config.json", b"config")
    h2_static = record("h2/static.json", b"static")
    x4_inputs = {
        f"input_{index:02d}": record(f"x4/input_{index:02d}.bin", bytes([40 + index]))
        for index in range(4)
    }
    products = [
        {
            "kind": "detector_map" if index >= 108 else "response",
            **record(f"x4/product_{index:03d}.bin", bytes([index % 251])),
        }
        for index in range(116)
    ]
    manifest_path = tmp_path / "x4/manifest.json"
    manifest_path.write_text(json.dumps({"products": products}), encoding="utf-8")
    terminal = {
        "inputs": terminal_inputs,
        "status_consistency_checks": {"H2": True, "X4": True},
        "artifact_integrity": {
            "H2": {
                "execution_config": h2_config,
                "static_pre_pixel_audit": h2_static,
                "immutable_input_artifact_count": 11,
                "all_immutable_inputs_rehashed": True,
                "immutable_inputs": h2_inputs,
                "artifact_count": 4,
                "all_reported_artifacts_rehashed": True,
                "artifacts": h2_outputs,
            },
            "X4": {
                "manifest_path": str(manifest_path),
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "manifest_product_count": 116,
                "response_product_count": 108,
                "detector_map_count": 8,
                "input_artifact_count": 4,
                "all_implementation_inputs_rehashed": True,
                "implementation_inputs": x4_inputs,
                "all_manifest_products_rehashed": True,
            },
        },
    }
    monkeypatch.setattr(completion_audit, "ROOT", tmp_path)
    assert completion_audit.verify_terminal_artifacts(terminal) == []
    (tmp_path / products[0]["path"]).write_bytes(b"mutated")
    assert "X4_manifest_product:000" in completion_audit.verify_terminal_artifacts(
        terminal
    )
