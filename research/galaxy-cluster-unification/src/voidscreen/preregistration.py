"""Validation helpers for frozen external-test protocols."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

P0633_STATUS = "frozen_before_any_P0633_target_product_download_or_score"
P0633_VERSION = "P0633-EXTERNAL-VALIDATION-PREREGISTRATION-1.0.0"


def canonical_protocol_bytes(protocol: dict) -> bytes:
    """Return stable bytes for a protocol independent of file formatting."""

    return json.dumps(
        protocol,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def protocol_sha256(protocol: dict) -> str:
    return hashlib.sha256(canonical_protocol_bytes(protocol)).hexdigest()


def _target_ids(protocol: dict, domain: str) -> list[str]:
    return [str(row["id"]) for row in protocol[f"{domain}_validation"]["systems"]]


def validate_p0633_protocol(protocol: dict) -> None:
    """Reject a weakened or internally inconsistent P0633 protocol."""

    if protocol.get("protocol_version") != P0633_VERSION:
        raise ValueError("unexpected P0633 protocol version")
    if protocol.get("status") != P0633_STATUS:
        raise ValueError("P0633 protocol is not frozen")
    if not re.fullmatch(r"[0-9a-f]{40}", str(protocol.get("baseline_commit", ""))):
        raise ValueError("baseline_commit must be a full Git object id")

    galaxies = _target_ids(protocol, "galaxy")
    clusters = _target_ids(protocol, "cluster")
    if len(galaxies) != 13 or len(set(galaxies)) != 13:
        raise ValueError("P0633 must retain 13 unique galaxy targets")
    if len(clusters) != 4 or len(set(clusters)) != 4:
        raise ValueError("P0633 must retain four unique cluster targets")

    all_aliases: list[str] = []
    for domain in ("galaxy", "cluster"):
        for row in protocol[f"{domain}_validation"]["systems"]:
            aliases = [str(alias) for alias in row.get("aliases", [])]
            if not aliases:
                raise ValueError(f"{row['id']} has no contamination aliases")
            all_aliases.extend(alias.casefold() for alias in aliases)
    if len(all_aliases) != len(set(all_aliases)):
        raise ValueError("target aliases must be globally unique")

    boundary = protocol["selection_boundary"]
    if boundary.get("maximum_per_object_gravity_parameters") != 0:
        raise ValueError("per-object gravity parameters are forbidden")

    gates = protocol["rejection_thresholds"]
    if gates["galaxy"]["equal_galaxy_RMSE_ratio_to_best_frozen_MOND_max"] > 1.05:
        raise ValueError("galaxy gate was weakened")
    if gates["cluster"]["heldout_image_RMS_ratio_to_compact_halo_max"] > 1.25:
        raise ValueError("cluster image gate was weakened")
    if gates["cluster"]["heldout_root_convergence_fraction_min"] != 1.0:
        raise ValueError("all heldout lens roots must converge")
    if not gates["cluster"]["all_heldout_family_topologies_correct"]:
        raise ValueError("exact heldout topology is mandatory")
    if not gates["solar_system"]["metric_PPN_quantities_must_be_derived_not_assumed"]:
        raise ValueError("PPN quantities may not be assumed")


def git_alias_matches(
    repository_root: Path,
    baseline_commit: str,
    project_relative_path: str,
    alias: str,
) -> list[str]:
    """Return exact fixed-string alias matches in a historical Git tree."""

    command = [
        "git",
        "grep",
        "-n",
        "-i",
        "-F",
        "-e",
        alias,
        baseline_commit,
        "--",
        project_relative_path,
    ]
    completed = subprocess.run(
        command,
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode == 1:
        return []
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "git grep failed")
    return [line for line in completed.stdout.splitlines() if line.strip()]


def contamination_ledger(
    protocol: dict,
    repository_root: Path,
    project_relative_path: str,
) -> list[dict]:
    """Prove that every locked alias was absent at the preserved baseline."""

    rows: list[dict] = []
    baseline = str(protocol["baseline_commit"])
    for domain in ("galaxy", "cluster"):
        for target in protocol[f"{domain}_validation"]["systems"]:
            for alias in target["aliases"]:
                matches = git_alias_matches(
                    repository_root,
                    baseline,
                    project_relative_path,
                    str(alias),
                )
                rows.append(
                    {
                        "domain": domain,
                        "target_id": target["id"],
                        "alias": alias,
                        "matches": len(matches),
                        "first_match": matches[0] if matches else None,
                    }
                )
    return rows


def target_directories(protocol: dict, project_root: Path) -> list[Path]:
    acquisition = protocol["acquisition"]
    keys = (
        "galaxy_input_directory",
        "galaxy_sealed_directory",
        "cluster_input_directory",
        "cluster_sealed_directory",
    )
    return [project_root / acquisition[key] for key in keys]
