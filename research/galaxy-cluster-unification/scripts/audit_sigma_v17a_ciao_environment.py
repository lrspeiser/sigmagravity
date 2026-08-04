#!/usr/bin/env python3
"""Audit the isolated CIAO runtime frozen for the Sigma v17A reduction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v17a_chandra_reduction.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v17a_ciao_environment"
SMOKE_RE = re.compile(
    r"(?P<run>\d+) smoke tests run: (?P<passed>\d+) PASSED, "
    r"(?P<failed>\d+) FAILED, (?P<skipped>\d+) SKIPPED"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout + result.stderr


def package_map(rows: list[dict]) -> dict[str, dict]:
    return {str(row["name"]): row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = config["runtime"]
    if config["status"] != (
        "frozen before reprocessing, event-image inspection, temperature-region "
        "construction, spectral fitting, or reading a v17 dynamical-feature score"
    ):
        raise RuntimeError("the v17A Chandra reduction protocol is not frozen")

    install = Path(os.environ["ASCDS_INSTALL"]).resolve()
    caldb = Path(os.environ["CALDB"]).resolve()
    smoke_runner = install / "test" / "smoke" / "bin" / "run_smoke_tests.sh"
    if not smoke_runner.is_file():
        raise FileNotFoundError(smoke_runner)

    ciaover = run(["ciaover", "-v"])
    packages = json.loads(run(["conda", "list", "--json"]))
    packages_by_name = package_map(packages)
    smoke_log = run(["bash", str(smoke_runner)], cwd=install)
    match = SMOKE_RE.search(smoke_log)
    if match is None:
        raise RuntimeError("could not parse the official CIAO smoke-test summary")
    smoke = {key: int(value) for key, value in match.groupdict().items()}

    required_packages = {
        "ciao": expected["ciao"],
        "ciao-contrib": expected["ciao_contrib"],
        "caldb_main": expected["caldb_main"],
        "acis_bkg_evt": expected["acis_bkg_evt"],
        "sherpa": expected["sherpa"],
        "xspec-modelsonly": expected["xspec_modelsonly"],
    }
    package_checks = {
        name: {
            "expected": version,
            "actual": packages_by_name.get(name, {}).get("version"),
            "passed": packages_by_name.get(name, {}).get("version") == version,
        }
        for name, version in required_packages.items()
    }

    background_dir = caldb / "data" / "chandra" / "acis" / "bkgrnd"
    background_files = sorted(path for path in background_dir.glob("*.fits") if path.is_file())
    background_inventory = {
        "directory": str(background_dir),
        "files": len(background_files),
        "bytes": sum(path.stat().st_size for path in background_files),
        "filenames": [path.name for path in background_files],
    }

    gates = {
        "all_required_package_versions_match": all(
            value["passed"] for value in package_checks.values()
        ),
        "official_smoke_count_matches": smoke["run"] == int(expected["required_smoke_tests"]),
        "official_smoke_failures_within_gate": smoke["failed"]
        <= int(expected["maximum_failed_smoke_tests"]),
        "official_smoke_has_no_skips": smoke["skipped"] == 0,
        "acis_blank_sky_files_installed": bool(background_files),
    }
    gates["runtime_gate_passed"] = all(gates.values())

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    smoke_path = output / "ciao_smoke.log"
    smoke_path.write_text(smoke_log, encoding="utf-8")
    report = {
        "status": "ciao_runtime_audited",
        "generated_utc": datetime.now(UTC).isoformat(),
        "protocol_version": config["protocol_version"],
        "config_sha256": sha256(config_path),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "wsl_interop_present": "WSL_INTEROP" in os.environ,
            "formal_support_caveat": expected["support_caveat"],
        },
        "ciaover": ciaover,
        "packages": packages,
        "required_package_checks": package_checks,
        "background_inventory": background_inventory,
        "smoke": {
            **smoke,
            "runner": str(smoke_runner),
            "log": smoke_path.relative_to(ROOT).as_posix(),
            "log_sha256": sha256(smoke_path),
        },
        "gates": gates,
        "lensing_target_opened": False,
        "temperature_map_constructed": False,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"smoke": report["smoke"], "gates": gates}, indent=2))
    if not gates["runtime_gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
