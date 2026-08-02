from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "p0734_typed_photon_lensing_adapter.json"
REPORT = ROOT / "results" / "p0734_typed_photon_lensing_adapter" / "report.json"
IMPLEMENTATION_COMMIT = "f579bfbf"


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
        cwd=ROOT.parents[1],
    )
    return hashlib.sha256(content).hexdigest()


def test_p0734_frozen_acceptance_passed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert config["status"] == "frozen_before_P0734_implementation_acceptance"
    assert report["status"] == "pass"
    assert report["configSha256"] == sha256(CONFIG)
    assert report["failedGates"] == []
    assert all(report["gateResults"].values())
    assert report["executionAcceptance"]["hostedTestCount"] >= 68
    assert report["executionAcceptance"]["integratedAndDecoupledMapParity"] is True
    analytic = report["analyticAcceptance"]
    assert analytic["pointMass"]["medianRelativeError"] < 0.02
    assert analytic["pointMass"]["p95RelativeError"] < 0.04
    assert analytic["syntheticScoring"]["legacyVelocityRmse"] is None
    assert analytic["syntheticScoring"]["channelNames"] == [
        "deflection_arcsec",
        "reduced_shear_dimensionless",
    ]


def test_p0734_source_hashes_match_the_immutable_implementation_commit() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    for relative, expected in report["sourceSha256"].items():
        assert historical_sha256(relative) == expected
