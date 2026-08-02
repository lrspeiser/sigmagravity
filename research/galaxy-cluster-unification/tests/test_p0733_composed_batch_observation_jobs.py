from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parents[1]
IMPLEMENTATION_COMMIT = "b0f5d482"
CONFIG = ROOT / "configs" / "p0733_composed_batch_observation_jobs.json"
REPORT = ROOT / "results" / "p0733_composed_batch_observation_jobs" / "report.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def historical_sha256(relative: str) -> str:
    content = subprocess.check_output(
        [
            "git",
            "show",
            f"{IMPLEMENTATION_COMMIT}:research/galaxy-cluster-unification/{relative}",
        ],
        cwd=REPOSITORY,
    )
    return hashlib.sha256(content).hexdigest()


def test_p0733_frozen_acceptance_passed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    smoke = report["httpAcceptance"]
    assert config["status"] == "frozen_before_P0733_implementation_acceptance"
    assert report["status"] == "pass"
    assert report["configSha256"] == sha256(CONFIG)
    assert report["failedGates"] == []
    assert all(report["gateResults"].values())
    assert report["hostedPassCount"] == report["hostedTestCount"]
    assert smoke["fieldChildObservationTargetCount"] == 0
    assert smoke["fieldOnlyObservationChildren"] == 0
    assert smoke["changedObservationPreservedFieldJobId"] is True
    assert smoke["changedObservationChangedEvaluationJobId"] is True
    assert smoke["duplicateComposedBatchReused"] is True
    assert smoke["perObjectGravityParameters"] == 0
    assert smoke["observationAddedGravityParameters"] == 0
    assert smoke["allDownloadedArtifactHashesValid"] is True


def test_p0733_source_hashes_match_the_immutable_implementation_commit() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    for relative, expected in report["sourceSha256"].items():
        assert historical_sha256(relative) == expected
