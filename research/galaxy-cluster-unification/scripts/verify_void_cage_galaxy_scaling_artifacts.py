"""Verify the galaxy-scaling and transition-isolation result packages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(value: str) -> Path:
    return ROOT / Path(value.replace("\\", "/"))


def main() -> None:
    scaling_protocol_path = (
        ROOT / "configs" / "void_cage_galaxy_scaling_protocol.json"
    )
    transition_protocol_path = (
        ROOT / "configs" / "void_cage_transition_isolation_protocol.json"
    )
    scaling_report_path = (
        ROOT / "results" / "void_cage_galaxy_scaling_test" / "report.json"
    )
    transition_report_path = (
        ROOT / "results" / "void_cage_transition_isolation" / "report.json"
    )
    scaling = json.loads(scaling_report_path.read_text(encoding="utf-8"))
    transition = json.loads(transition_report_path.read_text(encoding="utf-8"))

    scaling_artifacts = {
        name: sha256(resolve(record["path"])) == record["sha256"]
        for name, record in scaling["artifacts"].items()
    }
    transition_artifacts = {
        name: sha256(resolve(record["path"])) == record["sha256"]
        for name, record in transition["artifacts"].items()
    }
    checks = {
        "scaling_protocol_hash_matches": scaling["protocol"]["sha256"]
        == sha256(scaling_protocol_path),
        "transition_protocol_hash_matches": transition["protocol"]["sha256"]
        == sha256(transition_protocol_path),
        "transition_parent_hash_matches": transition["parent_protocol_sha256"]
        == sha256(scaling_protocol_path),
        "scaling_sample_size_matches": scaling["design"]["galaxies"] == 131
        and scaling["design"]["points"] == 3034,
        "transition_sample_size_matches": transition["design"]["galaxies"] == 131
        and transition["design"]["points"] == 3034,
        "all_scaling_artifact_hashes_match": all(scaling_artifacts.values()),
        "all_transition_artifact_hashes_match": all(
            transition_artifacts.values()
        ),
        "no_velocity_predictor_used": not scaling["design"][
            "forbidden_velocity_predictors_used"
        ],
        "scientific_decisions_are_boolean": all(
            isinstance(value, bool)
            for value in [
                scaling["any_internal_galaxy_scaling_pass"],
                scaling["primary_incremental_void_pass"],
                transition["any_transition_driver_pass"],
            ]
        ),
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "report_version": "void-cage-galaxy-scaling-verification-0.1",
        "checks": checks,
        "scaling_artifact_checks": scaling_artifacts,
        "transition_artifact_checks": transition_artifacts,
        "scientific_outcomes_reproduced": {
            "internal_galaxy_scaling_pass": scaling[
                "any_internal_galaxy_scaling_pass"
            ],
            "incremental_void_pass": scaling[
                "primary_incremental_void_pass"
            ],
            "galaxy_dependent_transition_pass": transition[
                "any_transition_driver_pass"
            ],
        },
    }
    output = (
        ROOT
        / "results"
        / "void_cage_galaxy_scaling_verification"
        / "report.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
