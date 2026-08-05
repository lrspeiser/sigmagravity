#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19cv_bullet_nofile_runtime_remediation.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def cmdline(pid: int) -> list[str]:
    raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    return [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]


def children(pid: int) -> list[int]:
    path = Path("/proc") / str(pid) / "task" / str(pid) / "children"
    return [int(value) for value in path.read_text(encoding="utf-8").split()]


def find_live_v19x2(config: dict[str, Any]) -> int:
    runner = str((ROOT / config["live_precondition"]["v19x2_runner"]).resolve())
    scratch = config["live_precondition"]["scratch_root"]
    matches: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = cmdline(int(entry.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if len(args) < 2 or args[1] != runner:
            continue
        if "--scratch" in args and args[args.index("--scratch") + 1] == scratch:
            matches.append(int(entry.name))
    if len(matches) != 1:
        raise RuntimeError(f"expected one live V19X2 Python process, found {matches}")
    return matches[0]


def active_abell_evidence(v19x2_pid: int, config: dict[str, Any]) -> dict[str, Any]:
    pre = config["live_precondition"]
    direct = children(v19x2_pid)
    if len(direct) != 1:
        raise RuntimeError(f"expected one V19X2 child, found {direct}")
    combine_pid = direct[0]
    combine_args = cmdline(combine_pid)
    if not any("combine_spectra" in value for value in combine_args) or not any(
        "ABELL2146_integrated" in value for value in combine_args
    ):
        raise RuntimeError(f"active V19X2 child is not the Abell integrated combination: {combine_args}")
    addresp_children = children(combine_pid)
    if len(addresp_children) != 1:
        raise RuntimeError(f"expected one combine_spectra child, found {addresp_children}")
    addresp_pid = addresp_children[0]
    addresp_args = cmdline(addresp_pid)
    if not addresp_args or "addresp" not in Path(addresp_args[0]).name:
        raise RuntimeError(f"active response child is not addresp: {addresp_args}")
    descriptor_count = len(list((Path("/proc") / str(addresp_pid) / "fd").iterdir()))
    expected = pre["active_addresp_open_descriptors"]
    if descriptor_count != expected:
        raise RuntimeError(f"live Abell descriptor count changed: expected {expected}, got {descriptor_count}")
    if Path(pre["bullet_directory_must_be_absent"]).exists():
        raise RuntimeError("Bullet integrated combination already started")
    projected = pre["bullet_integrated_cells"] * pre["descriptors_per_spectrum"] + pre["fixed_process_descriptors"]
    if projected != pre["projected_bullet_open_descriptors"]:
        raise RuntimeError("projected Bullet descriptor arithmetic changed")
    return {
        "v19x2_pid": v19x2_pid,
        "combine_spectra_pid": combine_pid,
        "addresp_pid": addresp_pid,
        "active_addresp_open_descriptors": descriptor_count,
        "projected_bullet_open_descriptors": projected,
        "combine_command": combine_args,
        "addresp_command": addresp_args,
    }


def execute(config: dict[str, Any]) -> dict[str, Any]:
    if os.name != "posix" or not Path("/proc").is_dir():
        raise RuntimeError("V19CV must execute inside the active WSL/Linux runtime")
    parent_checks = {
        name: sha256(ROOT / spec["path"]) == spec["sha256"]
        for name, spec in config["parents"].items()
    }
    auth = config["authorization"]
    authorization_exact = (
        auth["change_only_live_v19x2_parent_nofile_soft_limit"]
        and not auth["change_hard_limit"]
        and not auth["change_spectrum_response_weight_grouping_fit_or_gate"]
        and not auth["change_gravity_formula_or_parameter"]
        and not auth["open_lensing_halo_action_holdout_or_solar_payload"]
    )
    if not all(parent_checks.values()) or not authorization_exact:
        raise RuntimeError(f"V19CV static preflight failed: parents={parent_checks}, auth={authorization_exact}")

    v19x2_pid = find_live_v19x2(config)
    evidence = active_abell_evidence(v19x2_pid, config)
    change = config["runtime_change"]
    import resource

    before = resource.prlimit(v19x2_pid, resource.RLIMIT_NOFILE)
    required_before = (change["required_soft_before"], change["required_hard_before"])
    if before != required_before:
        raise RuntimeError(f"unexpected V19X2 RLIMIT_NOFILE before change: {before}")
    if evidence["projected_bullet_open_descriptors"] <= before[0]:
        raise RuntimeError("live descriptor evidence does not require remediation")
    after_requested = (change["soft_after"], before[1])
    resource.prlimit(v19x2_pid, resource.RLIMIT_NOFILE, after_requested)
    after = resource.prlimit(v19x2_pid, resource.RLIMIT_NOFILE)
    if after != after_requested or after[1] != before[1]:
        raise RuntimeError(f"V19X2 RLIMIT_NOFILE postcondition failed: {after}")
    return {
        "protocol_version": config["protocol_version"],
        "status": "bullet_nofile_runtime_remediation_applied",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(DEFAULT_CONFIG),
        "parent_checks": parent_checks,
        "authorization_exact": authorization_exact,
        "live_evidence": evidence,
        "rlimit_nofile_before": list(before),
        "rlimit_nofile_after": list(after),
        "hard_limit_changed": after[1] != before[1],
        "scientific_payload_changed": False,
        "claim_boundary": config["claim_boundary"],
    }


def main() -> None:
    config = load_json(DEFAULT_CONFIG)
    output = ROOT / config["outputs"]["report"]
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = execute(config)
    except Exception as exc:  # noqa: BLE001 - preserve a fail-closed runtime record
        report = {
            "protocol_version": config["protocol_version"],
            "status": "bullet_nofile_runtime_remediation_failed_closed",
            "generated_utc": datetime.now(UTC).isoformat(),
            "exception": f"{type(exc).__name__}: {exc}",
            "scientific_payload_changed": False,
            "claim_boundary": config["claim_boundary"],
        }
    temp = output.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temp.replace(output)
    print(json.dumps(report, indent=2))
    if report["status"] != "bullet_nofile_runtime_remediation_applied":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
