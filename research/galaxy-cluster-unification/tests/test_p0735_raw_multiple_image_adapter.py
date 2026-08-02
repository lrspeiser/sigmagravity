from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p0735_raw_multiple_image_adapter.json"
REPORT = ROOT / "results/p0735_raw_multiple_image_adapter/report.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def historical_sha256(commit: str, relative: str) -> str:
    content = subprocess.check_output(
        [
            "git",
            "show",
            f"{commit}:research/galaxy-cluster-unification/{relative}",
        ],
        cwd=ROOT.parents[1],
    )
    return hashlib.sha256(content).hexdigest()


def test_p0735_frozen_acceptance_passed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert config["status"] == "frozen_before_P0735_implementation_acceptance"
    assert report["status"] == "pass"
    assert report["configSha256"] == sha256(CONFIG)
    assert report["failedGates"] == []
    assert all(report["gateResults"].values())
    assert report["syntheticAcceptance"]["matchedMultiplicityFraction"] == 1.0
    assert report["syntheticAcceptance"]["gravityParametersAdded"] == 0
    assert report["syntheticAcceptance"]["incompleteFixture"]["aggregateRmseArcsec"] is None
    assert report["executionAcceptance"]["integratedAndDecoupledArtifactsByteIdentical"] is True
    assert report["executionAcceptance"]["hostedTestCount"] >= 70
    catalog = report["catalogImportAudit"]
    assert catalog["secureImages"] == 65
    assert catalog["secureFamilies"] == 18
    assert catalog["coordinateRoundTripMaximumArcsec"] <= 1e-12
    assert catalog["inventedUncertainties"] is False
    assert catalog["targetSerializationState"] == "blocked_missing_published_position_uncertainties"


def test_p0735_source_hashes_match_immutable_implementation_commit() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    commit = report.get("implementationCommit")
    if commit is None:
        # The implementation commit is injected immediately after the first
        # commit that contains all source files. Until then, the working-tree
        # hashes still protect the acceptance run.
        for relative, expected in report["sourceSha256"].items():
            assert sha256(ROOT / relative) == expected
        return
    for relative, expected in report["sourceSha256"].items():
        assert historical_sha256(commit, relative) == expected

