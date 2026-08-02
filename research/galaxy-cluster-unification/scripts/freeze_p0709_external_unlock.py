#!/usr/bin/env python3
"""Freeze the one-time P0633 external-data unlock without reading outcomes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "p0709_external_unlock_manifest.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    if protocol["status"] != "frozen_before_one_time_external_target_parse":
        raise RuntimeError("P0709 protocol is not frozen")

    prereg_path = ROOT / protocol["preregistration"]
    lock_config_path = ROOT / protocol["candidate_lock"]
    lock_report_path = ROOT / protocol["candidate_lock_report"]
    prediction_manifest_path = ROOT / protocol["candidate_prediction_manifest"]
    prereg = read_json(prereg_path)
    lock_config = read_json(lock_config_path)
    lock_report = read_json(lock_report_path)
    if not lock_report.get("all_prediction_lock_gates_pass"):
        raise RuntimeError("P0708 prediction lock did not pass")
    if not lock_report.get("candidate_authorized_for_one_external_unlock"):
        raise RuntimeError("P0708 did not authorize the unlock")
    if lock_report["universal_parameter_sha256"] != protocol["required_universal_parameter_sha256"]:
        raise RuntimeError("universal parameter hash changed")
    if lock_config["universal_parameters"]["per_object_gravity_parameters"] != 0:
        raise RuntimeError("candidate has per-object gravity parameters")
    if lock_report["sealed_P0633_kinematics_opened"] or lock_report["sealed_P0640_lensing_constraints_opened"]:
        raise RuntimeError("a sealed outcome was already marked open")

    with prediction_manifest_path.open(newline="", encoding="utf-8") as handle:
        predictions = list(csv.DictReader(handle))
    if len(predictions) != 17 or len({row["system"] for row in predictions}) != 17:
        raise RuntimeError("candidate prediction manifest is incomplete")
    for row in predictions:
        prediction_path = ROOT / row["prediction_path"]
        if sha256(prediction_path) != row["prediction_sha256"]:
            raise RuntimeError(f"prediction hash changed: {row['system']}")

    constraint_hashes = []
    for item in protocol["cluster_target_sources"]["constraint_containers"]:
        path = ROOT / item["path"]
        actual = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        if actual["bytes"] != item["expected_bytes"] or actual["sha256"] != item["expected_sha256"]:
            raise RuntimeError(f"sealed constraint container changed: {item['id']}")
        constraint_hashes.append({"id": item["id"], "path": item["path"], **actual})

    with (ROOT / protocol["registered_galaxy_audit"]).open(
        newline="", encoding="utf-8"
    ) as handle:
        audit = list(csv.DictReader(handle))
    coordinates = protocol["frozen_scoring"]["morphology"]["coordinates"]
    morphology_columns = {
        "concentration_5log_r80_r20": "concentration_5log_r80_r20",
        "lopsidedness_180": "lopsidedness_180",
        "clumpiness_positive_highpass": "clumpiness_positive_highpass",
        "inclination_deg": "inclination_deg",
    }
    medians = {
        coordinate: median([float(row[morphology_columns[coordinate]]) for row in audit])
        for coordinate in coordinates
    }

    moment_products = []
    galaxy_sources = protocol["galaxy_target_sources"]
    for target in galaxy_sources["targets"]:
        for product in galaxy_sources["products"]:
            filename = f"{target['id']}_{target['cube_weight']}_{product}.FITS"
            url = galaxy_sources["url_template"].format(
                archive_directory=target["archive_directory"],
                id=target["id"],
                cube_weight=target["cube_weight"],
                product=product,
            )
            moment_products.append(
                {
                    "system": target["id"],
                    "product": product,
                    "url": url,
                    "filename": filename,
                    "expected_bytes": target["moment1_bytes" if product == "XMOM1" else "moment2_bytes"],
                }
            )

    commit = git("rev-parse", "HEAD")
    if not git("merge-base", "--is-ancestor", protocol["required_candidate_commit"], commit) == "":
        raise RuntimeError("required candidate commit is not an ancestor")
    tracked_protocol = git("ls-files", "--error-unmatch", str(config_path.relative_to(ROOT)))
    tracked_source = git("ls-files", "--error-unmatch", str(Path(__file__).resolve().relative_to(ROOT)))
    if not tracked_protocol or not tracked_source:
        raise RuntimeError("unlock protocol and source must be committed before freezing")

    manifest = {
        "manifest_version": "P0709-EXTERNAL-UNLOCK-MANIFEST-RESULT-1.0.0",
        "status": "authorized_for_exactly_one_external_parse",
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__).resolve()),
        "unlock_protocol_commit": commit,
        "candidate_lock_commit": protocol["required_candidate_commit"],
        "preregistration_sha256": sha256(prereg_path),
        "candidate_lock_config_sha256": sha256(lock_config_path),
        "candidate_lock_report_sha256": sha256(lock_report_path),
        "prediction_manifest_sha256": sha256(prediction_manifest_path),
        "universal_parameter_sha256": lock_report["universal_parameter_sha256"],
        "prediction_systems": len(predictions),
        "prediction_hashes": [
            {"domain": row["domain"], "system": row["system"], "sha256": row["prediction_sha256"]}
            for row in predictions
        ],
        "sealed_constraint_containers": constraint_hashes,
        "galaxy_moment_products": moment_products,
        "published_circular_speed_source": galaxy_sources["published_circular_speed_source"],
        "compact_halo_comparator": protocol["cluster_target_sources"]["compact_halo_comparator"],
        "morphology_median_splits": medians,
        "family_split": prereg["cluster_validation"]["family_split"],
        "rejection_thresholds": prereg["rejection_thresholds"],
        "frozen_scoring": protocol["frozen_scoring"],
        "authorization": protocol["authorization"],
        "outcomes_opened_at_manifest_creation": False,
        "formula_change_after_this_manifest_invalidates_P0633_validation": True,
    }
    output = ROOT / protocol["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT)}")
    print(f"Protocol commit: {commit}")
    print(f"Universal parameter SHA-256: {manifest['universal_parameter_sha256']}")
    print("External parses authorized: 1")


if __name__ == "__main__":
    main()
