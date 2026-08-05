#!/usr/bin/env python3
"""Shared fail-closed helpers for the frozen Sigma v19F Chandra reduction."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19f_chandra_source_reduction.json"
FROZEN_STATUS = (
    "frozen before reprocessing, flare cleaning, source-image inspection, "
    "shock-front fitting, source construction, or opening any replacement-cluster "
    "lensing target"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_parent_hashes(config: dict) -> None:
    parents = config["parents"]
    path_keys = [key for key in parents if not key.endswith("_sha256")]
    for key in path_keys:
        hash_key = f"{key}_sha256"
        if hash_key not in parents:
            raise RuntimeError(f"v19F parent lacks a hash: {key}")
        path = ROOT / parents[key]
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256(path) != parents[hash_key]:
            raise RuntimeError(f"v19F frozen parent changed: {key}")


def validate_protocol(config_path: Path = DEFAULT_CONFIG) -> tuple[dict, dict, dict]:
    config_path = config_path.resolve()
    config = load_json(config_path)
    if config["status"] != FROZEN_STATUS:
        raise RuntimeError("v19F Chandra source-reduction protocol is not frozen")
    validate_parent_hashes(config)

    clusters = config["clusters"]
    if set(clusters) != {"BULLET", "ABELL2146"}:
        raise RuntimeError("v19F changed the preregistered development pair")
    obsids = [
        int(obsid)
        for cluster in clusters.values()
        for obsid in cluster["obsids"]
    ]
    if len(obsids) != config["gates"]["required_observations"]:
        raise RuntimeError("v19F observation count changed")
    if len(obsids) != len(set(obsids)):
        raise RuntimeError("v19F contains a duplicate ObsID")
    for cluster in clusters.values():
        declared_modes = cluster["expected_archive_datamode"]
        if set(declared_modes) != {str(obsid) for obsid in cluster["obsids"]}:
            raise RuntimeError("v19F DATAMODE declarations do not match the ObsIDs")
        if any(mode not in {"FAINT", "VFAINT"} for mode in declared_modes.values()):
            raise RuntimeError("v19F contains an unsupported DATAMODE")

    parents = config["parents"]
    acquisition = load_json(ROOT / parents["acquisition_report"])
    member_report = load_json(ROOT / parents["member_report"])
    runtime = load_json(ROOT / parents["runtime_audit"])
    if acquisition["config_sha256"] != parents["acquisition_config_sha256"]:
        raise RuntimeError("v19F acquisition ancestry is inconsistent")
    if acquisition["lensing_target_opened"] is not False:
        raise RuntimeError("v19F acquisition opened a lensing target")
    acquired = {
        (row["cluster"], int(row["obsid"])) for row in acquisition["per_obsid"]
    }
    requested = {
        (cluster, int(obsid))
        for cluster, values in clusters.items()
        for obsid in values["obsids"]
    }
    if acquired != requested:
        raise RuntimeError("v19F requested observations differ from v19E acquisition")
    if member_report["lensing_or_halo_payload_used"] is not False:
        raise RuntimeError("v19F member ancestry used lensing or halo data")
    if runtime["gates"]["runtime_gate_passed"] is not True:
        raise RuntimeError("the inherited CIAO runtime did not pass its audit")
    if runtime["lensing_target_opened"] is not False:
        raise RuntimeError("the inherited CIAO runtime audit opened a lensing target")
    for name, version in (
        ("ciao", config["runtime"]["ciao"]),
        ("ciao-contrib", config["runtime"]["ciao_contrib"]),
        ("caldb_main", config["runtime"]["caldb_main"]),
        ("acis_bkg_evt", config["runtime"]["acis_bkg_evt"]),
        ("sherpa", config["runtime"]["sherpa"]),
        ("xspec-modelsonly", config["runtime"]["xspec_modelsonly"]),
    ):
        check = runtime["required_package_checks"][name]
        if check["passed"] is not True or check["actual"] != version:
            raise RuntimeError(f"v19F inherited runtime changed: {name}")
    return config, acquisition, runtime


def resolved_shared_config(config: dict) -> dict:
    """Resolve hashed v17A detector choices without inheriting its science targets."""
    shared_path = ROOT / config["parents"]["shared_reduction_config"]
    shared = deepcopy(load_json(shared_path))
    shared["protocol_version"] = config["protocol_version"]
    shared["status"] = config["status"]
    shared["runtime"] = deepcopy(config["runtime"])
    shared["clusters"] = deepcopy(config["clusters"])
    shared["event_reprocessing"]["parallel_observations"] = int(
        config["event_reprocessing"]["parallel_observations"]
    )
    shared["event_reprocessing"]["pix_adj"] = config["event_reprocessing"][
        "pix_adj"
    ]
    return shared


def declared_mode(config: dict, cluster: str, obsid: int) -> str:
    return str(config["clusters"][cluster]["expected_archive_datamode"][str(obsid)])


def requested_observations(config: dict) -> list[tuple[str, int]]:
    return [
        (cluster, int(obsid))
        for cluster, values in config["clusters"].items()
        for obsid in values["obsids"]
    ]
