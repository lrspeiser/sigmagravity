"""Verify frozen inputs and outputs from the void-cage experiment."""

from __future__ import annotations

import argparse
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


def relative_path(value: str) -> Path:
    return ROOT / Path(value.replace("\\", "/"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "void_cage_verification" / "report.json",
    )
    args = parser.parse_args()

    protocol_path = ROOT / "configs" / "void_cage_test_protocol.json"
    geometry_report_path = ROOT / "results" / "void_cage_geometry" / "report.json"
    galaxy_report_path = ROOT / "results" / "void_cage_test" / "report.json"
    lensing_report_path = ROOT / "results" / "void_cage_lensing_gate" / "report.json"

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    geometry = json.loads(geometry_report_path.read_text(encoding="utf-8"))
    galaxy = json.loads(galaxy_report_path.read_text(encoding="utf-8"))
    lensing = json.loads(lensing_report_path.read_text(encoding="utf-8"))
    protocol_hash = sha256(protocol_path)

    checks: dict[str, bool] = {
        "geometry_protocol_hash_matches": geometry["protocol"]["sha256"]
        == protocol_hash,
        "galaxy_protocol_hash_matches": galaxy["protocol"]["sha256"]
        == protocol_hash,
        "lensing_protocol_hash_matches": lensing["inputs"]["protocol"]["sha256"]
        == protocol_hash,
        "geometry_output_hash_matches": sha256(
            relative_path(geometry["output"]["path"])
        )
        == geometry["output"]["sha256"],
        "galaxy_geometry_hash_matches": galaxy["inputs"]["geometry"]["sha256"]
        == geometry["output"]["sha256"],
        "lensing_galaxy_report_hash_matches": lensing["inputs"]["galaxy_test"][
            "sha256"
        ]
        == sha256(galaxy_report_path),
        "expected_sample_size": galaxy["design"]["galaxies"]
        == protocol["rotation_sample"]["expected_galaxies"]
        and galaxy["design"]["points"]
        == protocol["rotation_sample"]["expected_points"],
        "primary_gate_is_boolean": isinstance(
            galaxy["primary_screened_cage_pass"], bool
        ),
        "literal_gate_is_boolean": isinstance(
            galaxy["literal_external_harmonic_cage_pass"], bool
        ),
        "lensing_authorization_is_boolean": isinstance(
            lensing["lensing_replay_authorized"], bool
        ),
    }

    artifact_checks: dict[str, bool] = {}
    for name, artifact in galaxy["artifacts"].items():
        artifact_checks[name] = sha256(relative_path(artifact["path"])) == artifact[
            "sha256"
        ]
    checks["all_galaxy_artifact_hashes_match"] = all(artifact_checks.values())

    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "report_version": "void-cage-verification-0.1",
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": protocol_hash,
        "checks": checks,
        "artifact_checks": artifact_checks,
        "scientific_outcomes_reproduced": {
            "primary_screened_cage_pass": galaxy["primary_screened_cage_pass"],
            "literal_external_harmonic_cage_pass": galaxy[
                "literal_external_harmonic_cage_pass"
            ],
            "lensing_replay_authorized": lensing["lensing_replay_authorized"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
