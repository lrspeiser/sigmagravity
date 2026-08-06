#!/usr/bin/env python3
"""Install and audit the frozen V19CY A2319 HEASoft/XRISM environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19cy_a2319_environment.json"
BLOCK_BYTES = 4 * 1024 * 1024
USER_AGENT = "SigmaGravity-V19CY-A2319-Environment/1.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_wsl(distribution: str, command: str, timeout: int = 120) -> dict[str, Any]:
    result = subprocess.run(
        ["wsl.exe", "-d", distribution, "--", "bash", "-lc", command],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    return {
        "command": command,
        "exit_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def to_wsl_path(path: Path, distribution: str) -> str:
    result = subprocess.run(
        ["wsl.exe", "-d", distribution, "--", "wslpath", "-a", str(path.resolve())],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def validate_member_names(names: list[str], required_prefix: str) -> None:
    if not names:
        raise RuntimeError("CALDB archive is empty")
    for name in names:
        member = PurePosixPath(name)
        if member.is_absolute() or ".." in member.parts:
            raise RuntimeError(f"unsafe CALDB archive member: {name}")
        if not name.startswith(required_prefix):
            raise RuntimeError(f"CALDB archive member is outside {required_prefix}: {name}")


def validate_inputs(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_json(config_path)
    if config.get("protocol_version") != "SIGMA-V19CY-A2319-ENVIRONMENT-1.0.0":
        raise RuntimeError("unexpected A2319 environment protocol")
    if not config.get("status", "").startswith("frozen after the complete A2319 acquisition"):
        raise RuntimeError("A2319 environment protocol is not frozen")
    parent = config["parent"]
    acquisition_config = ROOT / parent["acquisition_config"]
    provenance_path = ROOT / parent["download_provenance"]
    if sha256(acquisition_config) != parent["acquisition_config_sha256"]:
        raise RuntimeError("A2319 acquisition config changed")
    if sha256(provenance_path) != parent["download_provenance_sha256"]:
        raise RuntimeError("A2319 download provenance changed")
    provenance = load_json(provenance_path)
    if provenance.get("status") != parent["required_download_status"]:
        raise RuntimeError("A2319 acquisition did not pass")
    if provenance.get("validation_or_holdout_asset_accessed"):
        raise RuntimeError("A2319 acquisition seal was violated")
    authorization = config["authorization"]
    for key in (
        "inspect_A2319_event_or_gain_arrays",
        "fit_A2319_spectra_or_velocities",
        "access_validation_or_holdout_assets",
        "open_lensing_halo_or_gravity_targets",
        "change_gravity_formula_or_parameters",
        "derive_or_select_action",
    ):
        if authorization[key]:
            raise RuntimeError(f"sealed environment boundary is open: {key}")
    return config, provenance


def validate_archives(config: dict[str, Any], provenance: dict[str, Any]) -> list[dict[str, Any]]:
    raw_root = (ROOT / config["paths"]["acquisition_raw_root"]).resolve()
    provenance_by_path = {record["download_path"]: record for record in provenance["records"]}
    archives: list[dict[str, Any]] = []
    for item in config["archives"]:
        path = (raw_root / item["download_path"]).resolve()
        if not path.is_relative_to(raw_root):
            raise RuntimeError(f"archive path escapes raw root: {item['download_path']}")
        record = provenance_by_path.get(item["download_path"])
        if record is None or record["bytes"] != item["bytes"] or record["sha256"] != item["sha256"]:
            raise RuntimeError(f"archive disagrees with terminal provenance: {item['role']}")
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise RuntimeError(f"archive failed local verification: {item['role']}")
        archives.append({**item, "path": path})
    return archives


def response_etag(response: requests.Response) -> str:
    return response.headers.get("ETag", "")


def acquire_setup_files(config: dict[str, Any]) -> list[dict[str, Any]]:
    raw_root = (ROOT / config["paths"]["setup_raw_root"]).resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for item in config["official_setup_files"]:
        destination = (raw_root / item["name"]).resolve()
        if not destination.is_relative_to(raw_root):
            raise RuntimeError(f"setup destination escapes raw root: {item['name']}")
        with requests.get(
            item["url"],
            headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"},
            stream=True,
            timeout=(30, 120),
        ) as response:
            response.raise_for_status()
            if int(response.headers.get("Content-Length", -1)) != item["bytes"]:
                raise RuntimeError(f"setup-file size metadata changed: {item['name']}")
            if response_etag(response) != item["etag"]:
                raise RuntimeError(f"setup-file ETag changed: {item['name']}")
            if response.headers.get("Last-Modified", "") != item["last_modified"]:
                raise RuntimeError(f"setup-file modification time changed: {item['name']}")
            partial = destination.with_suffix(destination.suffix + ".part")
            with partial.open("wb") as stream:
                response.raw.decode_content = False
                shutil.copyfileobj(response.raw, stream, length=BLOCK_BYTES)
            if partial.stat().st_size != item["bytes"]:
                raise RuntimeError(f"setup-file body has wrong size: {item['name']}")
            os.replace(partial, destination)
        records.append(
            {
                **item,
                "path": destination,
                "sha256": sha256(destination),
            }
        )
    return records


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def audit_runtime_hashes(config: dict[str, Any]) -> dict[str, Any]:
    distribution = config["runtime"]["wsl_distribution"]
    prefix = config["runtime"]["heasoft_prefix"]
    binaries = config["runtime"]["executables"]
    results: dict[str, Any] = {}
    for name, expected_hash in binaries.items():
        path = f"{prefix}/heasoft/bin/{name}"
        result = run_wsl(distribution, f"sha256sum {shell_quote(path)}")
        actual = result["stdout"].split()[0] if result["exit_code"] == 0 else ""
        results[name] = {
            "path": path,
            "expected_sha256": expected_hash,
            "actual_sha256": actual,
            "passed": result["exit_code"] == 0 and actual == expected_hash,
        }
    return results


def install_caldb(
    config: dict[str, Any], archives: list[dict[str, Any]], setup_files: list[dict[str, Any]]
) -> dict[str, Any]:
    distribution = config["runtime"]["wsl_distribution"]
    target = config["paths"]["caldb_install_root_wsl"]
    parent = str(PurePosixPath(target).parent)
    existence = run_wsl(distribution, f"test ! -e {shell_quote(target)}")
    if existence["exit_code"] != 0:
        raise RuntimeError(f"CALDB target already exists; refusing overwrite: {target}")
    created = run_wsl(
        distribution,
        f"mkdir -p {shell_quote(parent)} && mktemp -d {shell_quote(parent + '/caldb.installing.XXXXXX')}",
    )
    if created["exit_code"] != 0 or not created["stdout"]:
        raise RuntimeError(f"failed to create CALDB staging directory: {created['stderr']}")
    staging = created["stdout"].splitlines()[-1]
    try:
        by_role = {item["role"]: item for item in archives}
        member_audits: dict[str, Any] = {}
        for role in config["installation"]["extract_order"]:
            archive = by_role[role]
            archive_wsl = to_wsl_path(archive["path"], distribution)
            listing = run_wsl(distribution, f"tar -tf {shell_quote(archive_wsl)}", timeout=180)
            if listing["exit_code"] != 0:
                raise RuntimeError(f"failed to list {role}: {listing['stderr']}")
            names = [line for line in listing["stdout"].splitlines() if line]
            validate_member_names(names, config["installation"]["allowed_archive_member_prefix"])
            extraction = run_wsl(
                distribution,
                f"tar -xf {shell_quote(archive_wsl)} -C {shell_quote(staging)}",
                timeout=900,
            )
            if extraction["exit_code"] != 0:
                raise RuntimeError(f"failed to extract {role}: {extraction['stderr']}")
            member_audits[role] = {
                "members": len(names),
                "first": names[0],
                "last": names[-1],
            }
        tools_dir = f"{staging}/software/tools"
        made_tools = run_wsl(distribution, f"mkdir -p {shell_quote(tools_dir)}")
        if made_tools["exit_code"] != 0:
            raise RuntimeError(f"failed to create CALDB tools directory: {made_tools['stderr']}")
        for item in setup_files:
            source = to_wsl_path(item["path"], distribution)
            copied = run_wsl(
                distribution,
                f"cp {shell_quote(source)} {shell_quote(tools_dir + '/' + item['name'])}",
            )
            if copied["exit_code"] != 0:
                raise RuntimeError(f"failed to install {item['name']}: {copied['stderr']}")
        required = " && ".join(
            f"test -f {shell_quote(staging + '/' + relative)}"
            for relative in config["installation"]["required_paths"]
        )
        tokens = " && ".join(
            f"grep -q {shell_quote(token)} {shell_quote(tools_dir + '/caldb.config')}"
            for token in config["installation"]["required_caldb_config_tokens"]
        )
        staged_gate = run_wsl(distribution, f"{required} && {tokens}")
        if staged_gate["exit_code"] != 0:
            raise RuntimeError("staged CALDB failed required-path or configuration-token gate")
        moved = run_wsl(distribution, f"mv {shell_quote(staging)} {shell_quote(target)}")
        if moved["exit_code"] != 0:
            raise RuntimeError(f"failed to publish CALDB target: {moved['stderr']}")
        staging = ""
        return {"target": target, "member_audits": member_audits}
    except Exception as error:
        if staging:
            raise RuntimeError(f"CALDB staging failed closed at {staging}: {error}") from error
        raise


def audit_installed_environment(
    config: dict[str, Any], setup_files: list[dict[str, Any]], installation: dict[str, Any]
) -> dict[str, Any]:
    distribution = config["runtime"]["wsl_distribution"]
    prefix = config["runtime"]["heasoft_prefix"]
    target = config["paths"]["caldb_install_root_wsl"]
    environment = (
        f"export CONDA_PREFIX={shell_quote(prefix)}; "
        f"source {shell_quote(prefix + '/bin/heainit.sh')} >/dev/null 2>&1; "
        f"export CALDB={shell_quote(target)}; "
        f"export CALDBCONFIG={shell_quote(target + '/software/tools/caldb.config')}; "
        f"export CALDBALIAS={shell_quote(target + '/software/tools/alias_config.fits')}; "
    )
    commands = {
        "ftversion": "ftversion",
        "xspec_version": "xspec --version",
        "caldbinfo_xrism_gen": "caldbinfo infomode=INST mission=XRISM instrument=GEN chatter=20",
        "caldbinfo_xrism_resolve": (
            "caldbinfo infomode=INST mission=XRISM instrument=RESOLVE chatter=20"
        ),
        "inventory": (
            f"find {shell_quote(target)} -type f | wc -l; "
            f"du -sb {shell_quote(target)} | cut -f1"
        ),
    }
    runtime = {
        name: run_wsl(distribution, environment + command, timeout=120)
        for name, command in commands.items()
    }
    index_hashes = run_wsl(
        distribution,
        "sha256sum "
        + " ".join(
            shell_quote(target + "/" + relative)
            for relative in config["installation"]["required_paths"]
        ),
    )
    runtime_hashes = audit_runtime_hashes(config)
    inventory_lines = runtime["inventory"]["stdout"].splitlines()
    setup_hashes = {item["name"]: item["sha256"] for item in setup_files}
    installed_setup_hashes: dict[str, str] = {}
    for line in index_hashes["stdout"].splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) == 2:
            installed_setup_hashes[PurePosixPath(fields[1]).name] = fields[0]
    command_gate = all(item["exit_code"] == 0 for item in runtime.values())
    version_gate = (
        config["runtime"]["heasoft_version_token"] in runtime["ftversion"]["stdout"]
        and config["runtime"]["xspec_version_token"] in runtime["xspec_version"]["stdout"]
    )
    caldb_gate = all(
        runtime[key]["exit_code"] == 0
        and "ERROR" not in (runtime[key]["stdout"] + runtime[key]["stderr"]).upper()
        for key in ("caldbinfo_xrism_gen", "caldbinfo_xrism_resolve")
    )
    setup_gate = all(installed_setup_hashes.get(name) == digest for name, digest in setup_hashes.items())
    runtime_hash_gate = all(item["passed"] for item in runtime_hashes.values())
    report = {
        "protocol_version": config["protocol_version"],
        "status": (
            "a2319_frozen_heasoft_xrism_caldb_environment_installed_and_audited"
            if command_gate and version_gate and caldb_gate and setup_gate and runtime_hash_gate
            else "a2319_environment_audit_failed_closed"
        ),
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(DEFAULT_CONFIG),
        "download_provenance_sha256": config["parent"]["download_provenance_sha256"],
        "installation": installation,
        "setup_files": [
            {
                key: item[key]
                for key in ("name", "url", "bytes", "last_modified", "etag", "sha256")
            }
            for item in setup_files
        ],
        "runtime": runtime,
        "runtime_hashes": runtime_hashes,
        "required_path_hashes": index_hashes,
        "inventory": {
            "files": int(inventory_lines[0]) if inventory_lines and inventory_lines[0].isdigit() else 0,
            "bytes": (
                int(inventory_lines[1])
                if len(inventory_lines) > 1 and inventory_lines[1].isdigit()
                else 0
            ),
        },
        "gates": {
            "all_runtime_commands_exited_zero": command_gate,
            "heasoft_and_xspec_versions_exact": version_gate,
            "xrism_gen_and_resolve_caldbinfo_queries_passed": caldb_gate,
            "official_setup_files_installed_hash_exact": setup_gate,
            "runtime_executable_hashes_exact": runtime_hash_gate,
            "validation_or_holdout_accessed": False,
            "scientific_fit_performed": False,
        },
        "authorization": {
            "freeze_gain_reconstruction_protocol": (
                command_gate and version_gate and caldb_gate and setup_gate and runtime_hash_gate
            ),
            "inspect_A2319_event_or_gain_arrays": False,
            "fit_A2319_spectra_or_velocities": False,
            "access_validation_or_holdout_assets": False,
            "open_lensing_halo_or_gravity_targets": False,
            "derive_or_select_action": False,
        },
    }
    output = ROOT / config["paths"]["report"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config, provenance = validate_inputs(config_path)
    archives = validate_archives(config, provenance)
    runtime_hashes = audit_runtime_hashes(config)
    if not all(item["passed"] for item in runtime_hashes.values()):
        raise RuntimeError("HEASoft runtime hashes changed before CALDB installation")
    setup_files = acquire_setup_files(config)
    installation = install_caldb(config, archives, setup_files)
    report = audit_installed_environment(config, setup_files, installation)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"].endswith("installed_and_audited") else 1


if __name__ == "__main__":
    raise SystemExit(main())
