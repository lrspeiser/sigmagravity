#!/usr/bin/env python3
"""Acquire every frozen group from the current Archive processing product."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
AQ_PATH = SCRIPT_DIR / "acquire_sigma_v19aq_all_archive_decam_cutouts.py"
AQ_SPEC = importlib.util.spec_from_file_location("sigma_v19aq_base", AQ_PATH)
AQ = importlib.util.module_from_spec(AQ_SPEC)
assert AQ_SPEC.loader is not None
AQ_SPEC.loader.exec_module(AQ)

ROOT = AQ.ROOT
BASE = AQ.BASE
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19ar_current_archive_decam_cutouts.json"
PLAN_COLUMNS = BASE.PLAN_COLUMNS + [
    "retrieval_method",
    "identity_selection_rule",
    "identity_query_url",
    "identity_payload_path",
    "identity_payload_sha256",
    "sia_assoc_id",
    "source_archive_filename",
    "source_original_filename",
    "source_md5",
    "source_updated_utc",
    "archive_header_url",
    "archive_header_payload_path",
    "archive_header_payload_sha256",
    "fits_hdu_index",
    "header_extname",
    "header_ccdnum",
    "retrieval_url",
]
DOWNLOAD_COLUMNS = PLAN_COLUMNS + AQ.DOWNLOAD_COLUMNS[len(AQ.PLAN_COLUMNS) :]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def validate_config(config: dict[str, Any], require_frozen: bool) -> dict[str, str]:
    allowed = {
        "planning_current_archive_identities_and_headers_without_new_pixels",
        "frozen_before_v19ar_pixel_retrieval",
    }
    if config["status"] not in allowed:
        raise RuntimeError("V19AR status is invalid")
    if require_frozen and config["status"] != "frozen_before_v19ar_pixel_retrieval":
        raise RuntimeError("V19AR pixel retrieval is not frozen")
    hashes: dict[str, str] = {}
    for artifact in config["parent_artifacts"]:
        path = ROOT / artifact["path"]
        digest = sha256(path)
        if digest != artifact["sha256"]:
            raise RuntimeError(f"V19AR parent hash mismatch: {path}")
        hashes[artifact["path"]] = digest
    runner = ROOT / config["implementation"]["runner"]
    digest = sha256(runner)
    if digest != config["implementation"]["runner_sha256"]:
        raise RuntimeError("V19AR runner hash mismatch")
    hashes["runner"] = digest
    if sha256(AQ_PATH) != config["implementation"]["frozen_v19aq_runner_sha256"]:
        raise RuntimeError("V19AR V19AQ-runner hash mismatch")
    hashes["frozen_v19aq_runner"] = sha256(AQ_PATH)
    prohibited = (
        "rank_or_select_exposures_by_science_values",
        "fit_or_compare_photometry",
        "choose_psf_or_deblend_model",
        "query_ambiguous_candidates",
        "infer_mass_or_current",
        "read_lensing_or_halo_payload",
        "change_gravity_physics_or_parameters",
        "open_holdout",
    )
    if any(config["authorization"][name] for name in prohibited):
        raise RuntimeError("V19AR authorizes a prohibited action")
    if require_frozen and not config["authorization"]["download_every_frozen_group"]:
        raise RuntimeError("V19AR does not authorize complete frozen retrieval")
    return hashes


def identity_query_name(exposure: str, assoc_id: str) -> tuple[str, str]:
    if exposure.startswith("c4d_"):
        return (
            exposure.rsplit("_", 1)[0] + "_",
            "unique_latest_same_observation_c4d_ooi_instcal",
        )
    return (
        AQ.assoc_archive_stem(assoc_id),
        "unique_latest_same_observation_assoc_ooi_instcal",
    )


def identity_matches(
    exposure: str,
    query_name: str,
    expected_filter: str,
    row: dict[str, Any],
) -> bool:
    basename = Path(str(row["archive_filename"])).name
    if exposure.startswith("c4d_"):
        return basename.startswith(query_name)
    return AQ.assoc_fallback_matches(query_name, expected_filter, row)


def select_current_identity(
    exposure: str,
    assoc_id: str,
    expected_filter: str,
    coordinate: tuple[float, float],
    config: dict[str, Any],
    raw_root: Path,
) -> dict[str, Any]:
    query_name, rule = identity_query_name(exposure, assoc_id)
    url = AQ.vohdu_url(
        config["archive_identity"]["vohdu_endpoint"], coordinate, query_name
    )
    status, payload = AQ.fetch_bytes(
        url, float(config["archive_identity"]["timeout_seconds"])
    )
    if status != 200:
        raise RuntimeError(f"current Archive identity status {status}")
    path = raw_root / f"{exposure}.json"
    path.write_bytes(payload)
    candidates = [
        row
        for row in AQ.vohdu_rows(payload)
        if identity_matches(exposure, query_name, expected_filter, row)
    ]
    for row in candidates:
        AQ.validate_candidate(row, expected_filter)
    if not candidates:
        raise RuntimeError(f"no current Archive candidate for {exposure}")
    original_names = {str(row["original_filename"]) for row in candidates}
    if len(original_names) != 1:
        raise RuntimeError(f"Archive candidates do not identify one observation: {exposure}")
    newest = max(str(row["file_updated"]) for row in candidates)
    winners = [row for row in candidates if str(row["file_updated"]) == newest]
    if len(winners) != 1:
        raise RuntimeError(f"current Archive candidate is not unique: {exposure}")
    return {
        "identity": winners[0],
        "identity_selection_rule": rule,
        "identity_query_url": url,
        "identity_payload_path": path.relative_to(ROOT).as_posix(),
        "identity_payload_sha256": sha256(path),
    }


def build_current_plan(config: dict[str, Any]) -> list[dict[str, Any]]:
    base_rows = read_csv(ROOT / config["inputs"]["v19an_group_plan"])
    groups = AQ.AP.AO.parent_groups(config)
    metadata = AQ.exposure_metadata(config)
    identity_root = ROOT / config["outputs"]["archive_identity_directory"]
    header_root = ROOT / config["outputs"]["archive_header_directory"]
    output_root = ROOT / config["outputs"]["cutout_directory"]
    identity_root.mkdir(parents=True, exist_ok=True)
    header_root.mkdir(parents=True, exist_ok=True)

    identities: dict[str, dict[str, Any]] = {}
    header_cache: dict[str, tuple[bytes, Path]] = {}
    for exposure in sorted({row["exposure"] for row in base_rows}):
        meta = metadata[exposure]
        selected = select_current_identity(
            exposure,
            meta["sia_assoc_id"],
            meta["filter"],
            (float(meta["ra_deg"]), float(meta["dec_deg"])),
            config,
            identity_root,
        )
        identities[exposure] = selected
        md5 = str(selected["identity"]["md5sum"])
        if md5 not in header_cache:
            url = config["archive_header_resolution"]["header_endpoint"].format(md5=md5)
            status, payload = AQ.fetch_bytes(
                url, float(config["archive_header_resolution"]["timeout_seconds"])
            )
            if status != 200:
                raise RuntimeError(f"current Archive header status {status}")
            path = header_root / f"{md5}.json"
            path.write_bytes(payload)
            header_cache[md5] = (payload, path)

    plan: list[dict[str, Any]] = []
    for base in base_rows:
        exposure = base["exposure"]
        selected = identities[exposure]
        identity = selected["identity"]
        md5 = str(identity["md5sum"])
        header_payload, header_path = header_cache[md5]
        resolved = AQ.AP.resolve_unique_header(
            header_payload,
            groups[(exposure, base["sia_extension"])],
            float(config["archive_header_resolution"]["wcs_containment_tolerance_pixel"]),
        )
        fits_index = int(resolved["fits_hdu_index"])
        row: dict[str, Any] = dict(base)
        row.update(
            {
                "output_path": (
                    output_root / f"{base['group_id']}.fits"
                ).relative_to(ROOT).as_posix(),
                "retrieval_method": "current_archive_header_wcs_selected_hdu",
                "identity_selection_rule": selected["identity_selection_rule"],
                "identity_query_url": selected["identity_query_url"],
                "identity_payload_path": selected["identity_payload_path"],
                "identity_payload_sha256": selected["identity_payload_sha256"],
                "sia_assoc_id": metadata[exposure]["sia_assoc_id"],
                "source_archive_filename": Path(
                    str(identity["archive_filename"])
                ).name,
                "source_original_filename": str(identity["original_filename"]),
                "source_md5": md5,
                "source_updated_utc": str(identity["file_updated"]),
                "archive_header_url": config["archive_header_resolution"][
                    "header_endpoint"
                ].format(md5=md5),
                "archive_header_payload_path": header_path.relative_to(ROOT).as_posix(),
                "archive_header_payload_sha256": sha256(header_path),
                "fits_hdu_index": fits_index,
                "header_extname": resolved["header_extname"],
                "header_ccdnum": resolved["header_ccdnum"],
                "retrieval_url": config["archive_header_resolution"][
                    "retrieval_endpoint"
                ].format(md5=md5, fits_hdu_index=fits_index),
            }
        )
        plan.append(row)
    return plan


def plan_only(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    hashes = validate_config(config, require_frozen=False)
    plan = build_current_plan(config)
    plan_path = ROOT / config["outputs"]["resolved_plan"]
    write_csv(plan_path, plan, PLAN_COLUMNS)
    rules = Counter(row["identity_selection_rule"] for row in plan)
    return {
        "status": "current_archive_identities_and_headers_resolved_without_new_pixels",
        "config_sha256": sha256(config_path),
        "parent_hashes": hashes,
        "groups": len(plan),
        "measurement_memberships": sum(int(row["measurement_rows"]) for row in plan),
        "unique_exposures": len({row["exposure"] for row in plan}),
        "unique_archive_files": len({row["source_md5"] for row in plan}),
        "identity_selection_rules": dict(sorted(rules.items())),
        "unique_retrieval_urls": len({row["retrieval_url"] for row in plan}),
        "resolved_plan": plan_path.relative_to(ROOT).as_posix(),
        "resolved_plan_sha256": sha256(plan_path),
        "new_image_pixels_downloaded": False,
    }


def acquire(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    hashes = validate_config(config, require_frozen=True)
    plan_path = ROOT / config["frozen_resolved_plan"]["path"]
    if sha256(plan_path) != config["frozen_resolved_plan"]["sha256"]:
        raise RuntimeError("V19AR frozen resolved-plan hash mismatch")
    plan = read_csv(plan_path)
    groups = AQ.AP.AO.parent_groups(config)
    if len(plan) != int(config["gates"]["exact_groups"]):
        raise RuntimeError("V19AR frozen group count changed")

    completed: list[dict[str, Any]] = []
    for number, row in enumerate(plan, start=1):
        output = ROOT / row["output_path"]
        output.parent.mkdir(parents=True, exist_ok=True)
        group_rows = groups[(row["exposure"], row["sia_extension"])]
        if output.is_file():
            payload = output.read_bytes()
            inspection = AQ.inspect_payload(
                payload,
                group_rows,
                row,
                float(config["retrieval"]["wcs_containment_tolerance_pixel"]),
            )
            status = 200
            attempt = 0
        else:
            status, payload, inspection, attempt = AQ.fetch_validate(
                row, group_rows, config
            )
            output.write_bytes(payload)
        completed.append(
            {
                **row,
                "http_status": status,
                "download_attempt": attempt,
                "download_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                **inspection,
            }
        )
        print(
            f"V19AR {number}/{len(plan)} {row['group_id']} "
            f"HDU {row['fits_hdu_index']} {len(payload)} bytes",
            flush=True,
        )

    manifest_path = ROOT / config["outputs"]["download_manifest"]
    write_csv(manifest_path, completed, DOWNLOAD_COLUMNS)
    checksum_rows = [row for row in completed if row["checksum_keyword_present"]]
    datasum_rows = [row for row in completed if row["datasum_keyword_present"]]
    gates = {
        "exact_groups": len(completed) == int(config["gates"]["exact_groups"]),
        "exact_measurement_memberships": sum(
            int(row["measurement_rows"]) for row in completed
        )
        == int(config["gates"]["exact_measurement_memberships"]),
        "exact_unique_exposures": len({row["exposure"] for row in completed})
        == int(config["gates"]["exact_unique_exposures"]),
        "http_200_every_group": all(int(row["http_status"]) == 200 for row in completed),
        "fits_structure_every_group": all(
            bool(row["fits_structure_passed"]) for row in completed
        ),
        "celestial_wcs_every_group": all(bool(row["wcs_celestial"]) for row in completed),
        "all_frozen_anchors_contained": all(
            bool(row["anchors_contained"]) for row in completed
        ),
        "archive_detector_identity_every_group": all(
            row["returned_extname"] == row["header_extname"]
            and row["returned_ccdnum"] == row["header_ccdnum"]
            for row in completed
        ),
        "all_present_checksum_keywords_valid": all(
            bool(row["checksum_keyword_valid"]) for row in checksum_rows
        )
        and all(bool(row["datasum_keyword_valid"]) for row in datasum_rows),
        "raw_sha256_recorded_every_group": all(len(row["sha256"]) == 64 for row in completed),
        "at_least_one_finite_pixel_every_group": all(
            float(row["finite_pixel_fraction"]) > 0.0 for row in completed
        ),
        "all_groups_retained_without_science_selection": True,
        "no_photometric_model_or_prohibited_payload_opened": True,
    }
    gates["all_v19ar_acquisition_gates_pass"] = all(gates.values())
    report_path = ROOT / config["outputs"]["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "report_version": "SIGMA-V19AR-CURRENT-ARCHIVE-DECAM-CUTOUTS-1.0.0",
        "status": "completed_current_archive_group_acquisition",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "parent_hashes": hashes,
        "frozen_resolved_plan": config["frozen_resolved_plan"],
        "counts": {
            "groups": len(completed),
            "measurement_memberships": sum(
                int(row["measurement_rows"]) for row in completed
            ),
            "unique_exposures": len({row["exposure"] for row in completed}),
            "unique_archive_files": len({row["source_md5"] for row in completed}),
            "download_bytes": sum(int(row["download_bytes"]) for row in completed),
            "checksum_keyword_present": len(checksum_rows),
            "datasum_keyword_present": len(datasum_rows),
        },
        "minimum_finite_pixel_fraction": min(
            float(row["finite_pixel_fraction"]) for row in completed
        ),
        "minimum_anchor_edge_margin_pixel": min(
            float(row["minimum_anchor_edge_margin_pixel"]) for row in completed
        ),
        "download_manifest": manifest_path.relative_to(ROOT).as_posix(),
        "download_manifest_sha256": sha256(manifest_path),
        "gates": gates,
        "claim_boundary": config["claim_boundary"],
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not gates["all_v19ar_acquisition_gates_pass"]:
        raise RuntimeError("V19AR acquisition gates failed")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    result = plan_only(args.config) if args.plan_only else acquire(args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
