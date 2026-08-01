#!/usr/bin/env python3
"""Verify and record the P0633 external-validation freeze."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.preregistration import (
    contamination_ledger,
    protocol_sha256,
    target_directories,
    validate_p0633_protocol,
)

DEFAULT_PROTOCOL = ROOT / "configs" / "p0633_external_validation_preregistration.json"
DEFAULT_OUTPUT = ROOT / "results" / "p0633_external_validation_preregistration"


def git_repository_root() -> Path:
    completed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return Path(completed.stdout.strip()).resolve()


def verify_commit(repository_root: Path, commit: str) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--verify", f"{commit}^{{commit}}"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def build_report(protocol_path: Path) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_p0633_protocol(protocol)
    repository_root = git_repository_root()
    project_relative = ROOT.relative_to(repository_root).as_posix()
    verified_commit = verify_commit(repository_root, protocol["baseline_commit"])
    aliases = contamination_ledger(protocol, repository_root, project_relative)
    contaminated = [row for row in aliases if row["matches"]]
    directories = target_directories(protocol, ROOT)
    present = [path.relative_to(ROOT).as_posix() for path in directories if path.exists()]
    if contaminated:
        raise RuntimeError(f"P0633 target aliases occurred at baseline: {contaminated}")
    if present:
        raise RuntimeError(f"P0633 target directories already exist: {present}")

    return {
        "protocol_version": protocol["protocol_version"],
        "status": "frozen_targets_verified_and_unopened",
        "generated_utc": datetime.now(UTC).isoformat(),
        "protocol_path": protocol_path.relative_to(ROOT).as_posix(),
        "protocol_sha256": protocol_sha256(protocol),
        "baseline_commit": verified_commit,
        "targets": {
            "galaxies": len(protocol["galaxy_validation"]["systems"]),
            "clusters": len(protocol["cluster_validation"]["systems"]),
            "aliases_checked": len(aliases),
            "baseline_alias_matches": 0,
        },
        "target_directories_present": present,
        "target_products_opened": False,
        "contamination_scan": aliases,
        "rejection_thresholds": protocol["rejection_thresholds"],
    }


def write_outputs(report: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "ledger.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    targets = report["targets"]
    summary = f"""# P0633 external-validation freeze

- Status: `{report['status']}`
- Baseline: `{report['baseline_commit']}`
- Protocol SHA-256: `{report['protocol_sha256']}`
- Locked targets: {targets['galaxies']} galaxies and {targets['clusters']} clusters
- Historical aliases checked: {targets['aliases_checked']}
- Historical matches: {targets['baseline_alias_matches']}
- Target products opened: `{str(report['target_products_opened']).lower()}`

The rejection gates and sealed-data boundary are copied verbatim into
`ledger.json`. This result records a preregistration, not a physics result.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(args.protocol.resolve())
    write_outputs(report, args.output.resolve())
    print(json.dumps({key: report[key] for key in ("status", "protocol_sha256", "targets")}, indent=2))


if __name__ == "__main__":
    main()
