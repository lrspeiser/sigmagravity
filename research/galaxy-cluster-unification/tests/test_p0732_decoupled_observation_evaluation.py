from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "p0732_decoupled_observation_evaluation.json"
REPORT = ROOT / "results" / "p0732_decoupled_observation_evaluation" / "report.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_p0732_frozen_acceptance_passed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert config["status"] == "frozen_before_P0732_implementation_acceptance"
    assert report["status"] == "pass"
    assert report["configSha256"] == sha256(CONFIG)
    assert report["fixtures"]["dimensions"] == [2, 3]
    assert set(report["fixtures"]["targetKinds"]) == {
        "circular_speed_curve",
        "line_of_sight_velocity_field",
    }
    assert report["fixtures"]["scoreArtifactsByteExact"] is True
    assert report["fixtures"]["predictionArtifactsByteExact"] is True
    assert report["failedGates"] == []
    assert all(report["gateResults"].values())
    assert report["httpAcceptance"]["fieldSolverInvocationsDuringEvaluation"] == 0
    assert report["httpAcceptance"]["fieldJobCount"] == 1
    assert report["httpAcceptance"]["evaluationAddedGravityParameters"] == 0


def test_p0732_source_hashes_remain_reproducible() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    for relative, expected in report["sourceSha256"].items():
        assert sha256(ROOT / relative) == expected
