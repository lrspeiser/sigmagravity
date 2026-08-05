#!/usr/bin/env python3
"""Resolve every frozen DECam group through the Archive and acquire it."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import re
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


SCRIPT_DIR = Path(__file__).resolve().parent
AP_PATH = SCRIPT_DIR / "acquire_sigma_v19ap_header_wcs_resolved_decam_cutouts.py"
AP_SPEC = importlib.util.spec_from_file_location("sigma_v19ap_base", AP_PATH)
AP = importlib.util.module_from_spec(AP_SPEC)
assert AP_SPEC.loader is not None
AP_SPEC.loader.exec_module(AP)

ROOT = AP.ROOT
BASE = AP.AO.BASE
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19aq_all_archive_decam_cutouts.json"
USER_AGENT = "SigmaGravity-V19AQ-all-Archive-DECam-cutouts/1.0"
MD5 = re.compile(r"^[0-9a-f]{32}$")
ASSOC = re.compile(r"^ct4m(\d{4})(\d{2})(\d{2})t(\d{2})(\d{2})(\d{2})$")
PLAN_COLUMNS = BASE.PLAN_COLUMNS + [
    "retrieval_method",
    "identity_selection_rule",
    "exact_identity_query_url",
    "exact_identity_payload_path",
    "exact_identity_payload_sha256",
    "fallback_identity_query_url",
    "fallback_identity_payload_path",
    "fallback_identity_payload_sha256",
    "sia_assoc_id",
    "source_archive_filename",
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
DOWNLOAD_COLUMNS = PLAN_COLUMNS + [
    "http_status",
    "download_attempt",
    "download_bytes",
    "sha256",
    "image_hdu_index",
    "image_naxis1",
    "image_naxis2",
    "finite_pixel_fraction",
    "wcs_celestial",
    "anchors_contained",
    "minimum_anchor_edge_margin_pixel",
    "header_filter",
    "header_expnum",
    "header_proctype",
    "returned_extname",
    "returned_ccdnum",
    "checksum_keyword_present",
    "checksum_keyword_valid",
    "datasum_keyword_present",
    "datasum_keyword_valid",
    "fits_structure_passed",
]


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
        "planning_all_archive_identities_and_headers_without_new_pixels",
        "frozen_before_v19aq_pixel_retrieval",
    }
    if config["status"] not in allowed:
        raise RuntimeError("V19AQ status is invalid")
    if require_frozen and config["status"] != "frozen_before_v19aq_pixel_retrieval":
        raise RuntimeError("V19AQ pixel retrieval is not frozen")
    hashes: dict[str, str] = {}
    for artifact in config["parent_artifacts"]:
        path = ROOT / artifact["path"]
        digest = sha256(path)
        if digest != artifact["sha256"]:
            raise RuntimeError(f"V19AQ parent hash mismatch: {path}")
        hashes[artifact["path"]] = digest
    runner = ROOT / config["implementation"]["runner"]
    digest = sha256(runner)
    if digest != config["implementation"]["runner_sha256"]:
        raise RuntimeError("V19AQ runner hash mismatch")
    hashes["runner"] = digest
    if sha256(AP_PATH) != config["implementation"]["frozen_v19ap_runner_sha256"]:
        raise RuntimeError("V19AQ V19AP-runner hash mismatch")
    hashes["frozen_v19ap_runner"] = sha256(AP_PATH)
    prohibited = (
        "rank_or_select_exposures",
        "fit_or_compare_photometry",
        "choose_psf_or_deblend_model",
        "query_ambiguous_candidates",
        "infer_mass_or_current",
        "read_lensing_or_halo_payload",
        "change_gravity_physics_or_parameters",
        "open_holdout",
    )
    if any(config["authorization"][name] for name in prohibited):
        raise RuntimeError("V19AQ authorizes a prohibited action")
    if require_frozen and not config["authorization"]["download_every_frozen_group"]:
        raise RuntimeError("V19AQ does not authorize complete frozen retrieval")
    return hashes


def fetch_bytes(url: str, timeout: float) -> tuple[int, bytes]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return int(response.status), response.read()


def vohdu_url(endpoint: str, coordinate: tuple[float, float], filename: str) -> str:
    params = [
        ("POS", f"{coordinate[0]:.12f},{coordinate[1]:.12f}"),
        ("SIZE", "0.01"),
        ("archive_filename", filename),
        ("VERB", "3"),
        ("format", "json"),
        ("limit", "100"),
    ]
    return endpoint + "?" + urllib.parse.urlencode(params, safe=",")


def vohdu_rows(payload: bytes) -> list[dict[str, Any]]:
    decoded = json.loads(payload)
    if not isinstance(decoded, list):
        raise RuntimeError("Archive vohdu response is not a list")
    return [row for row in decoded if isinstance(row, dict) and "md5sum" in row]


def validate_candidate(row: dict[str, Any], expected_filter: str) -> None:
    if str(row.get("instrument", "")).lower() != "decam":
        raise RuntimeError("Archive identity is not DECam")
    if str(row.get("proc_type", "")).lower() != "instcal":
        raise RuntimeError("Archive identity is not InstCal")
    if str(row.get("prod_type", "")).lower() != "image":
        raise RuntimeError("Archive identity is not an image")
    if str(row.get("obs_type", "")).lower() != "object":
        raise RuntimeError("Archive identity is not an object exposure")
    if not str(row.get("filter", "")).startswith(expected_filter):
        raise RuntimeError("Archive identity filter does not match")
    if not MD5.fullmatch(str(row.get("md5sum", ""))):
        raise RuntimeError("Archive identity MD5 is invalid")


def assoc_archive_stem(assoc_id: str) -> str:
    match = ASSOC.fullmatch(assoc_id)
    if match is None:
        raise RuntimeError(f"unsupported frozen SIA association ID: {assoc_id}")
    year, month, day, hour, minute, second = match.groups()
    return f"c4d_{year[2:]}{month}{day}_{hour}{minute}{second}"


def exact_basename_matches(exposure: str, row: dict[str, Any]) -> bool:
    basename = Path(str(row["archive_filename"])).name
    return basename in {f"{exposure}.fits", f"{exposure}.fits.fz"}


def assoc_fallback_matches(stem: str, expected_filter: str, row: dict[str, Any]) -> bool:
    basename = Path(str(row["archive_filename"])).name
    return basename.startswith(stem + "_ooi_") and f"_ooi_{expected_filter}_" in basename


def fallback_query_and_match(
    exposure: str,
    assoc_id: str,
    expected_filter: str,
) -> tuple[str, str, Any]:
    if exposure.startswith("c4d_") and "_" in exposure:
        prefix = exposure.rsplit("_", 1)[0] + "_"
        return (
            prefix,
            "stale_c4d_prefix_unique_latest_instcal",
            lambda row: Path(str(row["archive_filename"])).name.startswith(prefix),
        )
    stem = assoc_archive_stem(assoc_id)
    return (
        stem,
        "frozen_assoc_id_unique_latest_ooi_instcal",
        lambda row: assoc_fallback_matches(stem, expected_filter, row),
    )


def select_identity(
    exposure: str,
    expected_filter: str,
    assoc_id: str,
    coordinate: tuple[float, float],
    config: dict[str, Any],
    raw_root: Path,
) -> dict[str, Any]:
    endpoint = config["archive_identity"]["vohdu_endpoint"]
    timeout = float(config["archive_identity"]["timeout_seconds"])
    exact_url = vohdu_url(endpoint, coordinate, exposure)
    status, exact_payload = fetch_bytes(exact_url, timeout)
    if status != 200:
        raise RuntimeError(f"exact Archive identity status {status}")
    exact_path = raw_root / f"{exposure}__exact.json"
    exact_path.write_bytes(exact_payload)
    exact = [row for row in vohdu_rows(exact_payload) if exact_basename_matches(exposure, row)]
    for row in exact:
        validate_candidate(row, expected_filter)
    fallback_url = ""
    fallback_path: Path | None = None
    fallback_payload = b""
    if len(exact) == 1:
        selected = exact[0]
        rule = "exact_frozen_archive_basename"
    elif len(exact) > 1:
        raise RuntimeError(f"multiple exact Archive basenames for {exposure}")
    else:
        query_name, rule, matches = fallback_query_and_match(
            exposure, assoc_id, expected_filter
        )
        fallback_url = vohdu_url(endpoint, coordinate, query_name)
        fallback_status, fallback_payload = fetch_bytes(fallback_url, timeout)
        if fallback_status != 200:
            raise RuntimeError(f"association fallback status {fallback_status}")
        fallback_path = raw_root / f"{exposure}__association.json"
        fallback_path.write_bytes(fallback_payload)
        candidates = [
            row
            for row in vohdu_rows(fallback_payload)
            if matches(row)
        ]
        for row in candidates:
            validate_candidate(row, expected_filter)
        if not candidates:
            raise RuntimeError(f"no association-derived Archive identity for {exposure}")
        newest = max(str(row["file_updated"]) for row in candidates)
        winners = [row for row in candidates if str(row["file_updated"]) == newest]
        if len(winners) != 1:
            raise RuntimeError(f"association-derived Archive identity is not unique for {exposure}")
        selected = winners[0]
    return {
        "identity": selected,
        "identity_selection_rule": rule,
        "exact_identity_query_url": exact_url,
        "exact_identity_payload_path": exact_path.relative_to(ROOT).as_posix(),
        "exact_identity_payload_sha256": sha256(exact_path),
        "fallback_identity_query_url": fallback_url,
        "fallback_identity_payload_path": (
            fallback_path.relative_to(ROOT).as_posix() if fallback_path else ""
        ),
        "fallback_identity_payload_sha256": sha256(fallback_path) if fallback_path else "",
    }


def exposure_metadata(config: dict[str, Any]) -> dict[str, dict[str, str]]:
    rows = read_csv(ROOT / config["inputs"]["v19am_manifest"])
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["exposure"]].append(row)
    result: dict[str, dict[str, str]] = {}
    for exposure, members in grouped.items():
        assoc_ids = {row["sia_assoc_id"] for row in members}
        filters = {row["filter"] for row in members}
        if len(assoc_ids) != 1 or len(filters) != 1:
            raise RuntimeError(f"non-unique association or filter for {exposure}")
        representative = min(members, key=lambda row: row["nsc_id"])
        result[exposure] = {
            "sia_assoc_id": next(iter(assoc_ids)),
            "filter": next(iter(filters)),
            "ra_deg": representative["ra_deg"],
            "dec_deg": representative["dec_deg"],
        }
    return result


def build_all_archive_plan(config: dict[str, Any]) -> list[dict[str, Any]]:
    base_rows = read_csv(ROOT / config["inputs"]["v19an_group_plan"])
    groups = AP.AO.parent_groups(config)
    metadata = exposure_metadata(config)
    identity_root = ROOT / config["outputs"]["archive_identity_directory"]
    header_root = ROOT / config["outputs"]["archive_header_directory"]
    output_root = ROOT / config["outputs"]["cutout_directory"]
    identity_root.mkdir(parents=True, exist_ok=True)
    header_root.mkdir(parents=True, exist_ok=True)

    identities: dict[str, dict[str, Any]] = {}
    header_cache: dict[str, tuple[bytes, Path]] = {}
    for exposure in sorted({row["exposure"] for row in base_rows}):
        meta = metadata[exposure]
        identities[exposure] = select_identity(
            exposure,
            meta["filter"],
            meta["sia_assoc_id"],
            (float(meta["ra_deg"]), float(meta["dec_deg"])),
            config,
            identity_root,
        )
        identity = identities[exposure]["identity"]
        md5 = str(identity["md5sum"])
        if md5 not in header_cache:
            url = config["archive_header_resolution"]["header_endpoint"].format(md5=md5)
            status, payload = fetch_bytes(
                url, float(config["archive_header_resolution"]["timeout_seconds"])
            )
            if status != 200:
                raise RuntimeError(f"Archive header status {status}")
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
        resolved = AP.resolve_unique_header(
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
                "retrieval_method": "archive_header_wcs_selected_hdu",
                "identity_selection_rule": selected["identity_selection_rule"],
                "exact_identity_query_url": selected["exact_identity_query_url"],
                "exact_identity_payload_path": selected["exact_identity_payload_path"],
                "exact_identity_payload_sha256": selected[
                    "exact_identity_payload_sha256"
                ],
                "fallback_identity_query_url": selected[
                    "fallback_identity_query_url"
                ],
                "fallback_identity_payload_path": selected[
                    "fallback_identity_payload_path"
                ],
                "fallback_identity_payload_sha256": selected[
                    "fallback_identity_payload_sha256"
                ],
                "sia_assoc_id": metadata[exposure]["sia_assoc_id"],
                "source_archive_filename": Path(
                    str(identity["archive_filename"])
                ).name,
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
    plan = build_all_archive_plan(config)
    plan_path = ROOT / config["outputs"]["resolved_plan"]
    write_csv(plan_path, plan, PLAN_COLUMNS)
    rules = Counter(row["identity_selection_rule"] for row in plan)
    return {
        "status": "all_archive_identities_and_headers_resolved_without_new_pixels",
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


def image_hdu(payload: bytes) -> tuple[int, fits.Header, np.ndarray, dict[str, Any]]:
    with fits.open(io.BytesIO(payload), memmap=False, checksum=False) as hdul:
        hdul.verify("exception")
        for index, hdu in enumerate(hdul):
            if hdu.data is None or np.asarray(hdu.data).ndim != 2:
                continue
            checksum_present = "CHECKSUM" in hdu.header
            datasum_present = "DATASUM" in hdu.header
            checksum_valid = hdu.verify_checksum() == 1 if checksum_present else None
            datasum_valid = hdu.verify_datasum() == 1 if datasum_present else None
            return (
                index,
                hdu.header.copy(),
                np.asarray(hdu.data, dtype=float).copy(),
                {
                    "checksum_keyword_present": checksum_present,
                    "checksum_keyword_valid": checksum_valid,
                    "datasum_keyword_present": datasum_present,
                    "datasum_keyword_valid": datasum_valid,
                },
            )
    raise RuntimeError("downloaded FITS has no two-dimensional image HDU")


def inspect_payload(
    payload: bytes,
    group_rows: list[dict[str, str]],
    row: dict[str, str],
    tolerance: float,
) -> dict[str, Any]:
    index, header, data, checksums = image_hdu(payload)
    if data.shape[0] < 16 or data.shape[1] < 16:
        raise RuntimeError(f"Archive subset is unexpectedly small: {data.shape}")
    wcs = WCS(header)
    if not wcs.has_celestial:
        raise RuntimeError("Archive subset lacks celestial WCS")
    anchors: dict[str, tuple[float, float]] = {}
    for group_row in group_rows:
        anchors[group_row["nsc_id"]] = (
            float(group_row["ra_deg"]),
            float(group_row["dec_deg"]),
        )
    world = np.asarray(list(anchors.values()), dtype=float)
    pixel = wcs.celestial.all_world2pix(world, 0)
    nx, ny = data.shape[1], data.shape[0]
    contained = (
        (pixel[:, 0] >= -tolerance)
        & (pixel[:, 0] <= nx - 1 + tolerance)
        & (pixel[:, 1] >= -tolerance)
        & (pixel[:, 1] <= ny - 1 + tolerance)
    )
    if not bool(np.all(contained)):
        raise RuntimeError("one or more frozen anchors fall outside the Archive subset")
    margins = np.minimum.reduce(
        [pixel[:, 0], nx - 1 - pixel[:, 0], pixel[:, 1], ny - 1 - pixel[:, 1]]
    )
    finite_fraction = float(np.isfinite(data).mean())
    if finite_fraction <= 0.0:
        raise RuntimeError("Archive subset contains no finite pixels")
    extname = str(header.get("EXTNAME", ""))
    ccdnum = str(header.get("CCDNUM", ""))
    if extname != row["header_extname"] or ccdnum != row["header_ccdnum"]:
        raise RuntimeError("returned detector does not match frozen header-WCS identity")
    if checksums["checksum_keyword_present"] and not checksums["checksum_keyword_valid"]:
        raise RuntimeError("present FITS CHECKSUM keyword is invalid")
    if checksums["datasum_keyword_present"] and not checksums["datasum_keyword_valid"]:
        raise RuntimeError("present FITS DATASUM keyword is invalid")
    return {
        "image_hdu_index": index,
        "image_naxis1": nx,
        "image_naxis2": ny,
        "finite_pixel_fraction": f"{finite_fraction:.9f}",
        "wcs_celestial": True,
        "anchors_contained": True,
        "minimum_anchor_edge_margin_pixel": f"{float(np.min(margins)):.6f}",
        "header_filter": str(header.get("FILTER", "")),
        "header_expnum": str(header.get("EXPNUM", "")),
        "header_proctype": str(header.get("PROCTYPE", "")),
        "returned_extname": extname,
        "returned_ccdnum": ccdnum,
        **checksums,
        "fits_structure_passed": True,
    }


def fetch_validate(
    row: dict[str, str],
    group_rows: list[dict[str, str]],
    config: dict[str, Any],
) -> tuple[int, bytes, dict[str, Any], int]:
    retrieval = config["retrieval"]
    last_error: Exception | None = None
    for attempt in range(1, int(retrieval["maximum_attempts"]) + 1):
        try:
            status, payload = fetch_bytes(
                row["retrieval_url"], float(retrieval["timeout_seconds"])
            )
            if status != 200:
                raise RuntimeError(f"Archive retrieval status {status}")
            inspection = inspect_payload(
                payload,
                group_rows,
                row,
                float(retrieval["wcs_containment_tolerance_pixel"]),
            )
            return status, payload, inspection, attempt
        except Exception as error:  # transport and structural retries share one budget
            last_error = error
            if attempt < int(retrieval["maximum_attempts"]):
                time.sleep(float(attempt))
    assert last_error is not None
    raise last_error


def acquire(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    hashes = validate_config(config, require_frozen=True)
    plan_path = ROOT / config["frozen_resolved_plan"]["path"]
    if sha256(plan_path) != config["frozen_resolved_plan"]["sha256"]:
        raise RuntimeError("V19AQ frozen resolved-plan hash mismatch")
    plan = read_csv(plan_path)
    groups = AP.AO.parent_groups(config)
    if len(plan) != int(config["gates"]["exact_groups"]):
        raise RuntimeError("V19AQ frozen group count changed")

    completed: list[dict[str, Any]] = []
    for number, row in enumerate(plan, start=1):
        output = ROOT / row["output_path"]
        output.parent.mkdir(parents=True, exist_ok=True)
        group_rows = groups[(row["exposure"], row["sia_extension"])]
        if output.is_file():
            payload = output.read_bytes()
            inspection = inspect_payload(
                payload,
                group_rows,
                row,
                float(config["retrieval"]["wcs_containment_tolerance_pixel"]),
            )
            status = 200
            attempt = 0
        else:
            status, payload, inspection, attempt = fetch_validate(
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
            f"V19AQ {number}/{len(plan)} {row['group_id']} "
            f"HDU {row['fits_hdu_index']} {len(payload)} bytes",
            flush=True,
        )

    manifest_path = ROOT / config["outputs"]["download_manifest"]
    write_csv(manifest_path, completed, DOWNLOAD_COLUMNS)
    present_checksum_rows = [row for row in completed if row["checksum_keyword_present"]]
    present_datasum_rows = [row for row in completed if row["datasum_keyword_present"]]
    gates = {
        "exact_groups": len(completed) == int(config["gates"]["exact_groups"]),
        "exact_measurement_memberships": sum(
            int(row["measurement_rows"]) for row in completed
        )
        == int(config["gates"]["exact_measurement_memberships"]),
        "exact_archive_groups": len(completed)
        == int(config["gates"]["exact_archive_groups"]),
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
            bool(row["checksum_keyword_valid"]) for row in present_checksum_rows
        )
        and all(bool(row["datasum_keyword_valid"]) for row in present_datasum_rows),
        "raw_sha256_recorded_every_group": all(len(row["sha256"]) == 64 for row in completed),
        "at_least_one_finite_pixel_every_group": all(
            float(row["finite_pixel_fraction"]) > 0.0 for row in completed
        ),
        "all_groups_retained_without_selection": True,
        "no_photometric_model_or_prohibited_payload_opened": True,
    }
    gates["all_v19aq_acquisition_gates_pass"] = all(gates.values())
    report_path = ROOT / config["outputs"]["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "report_version": "SIGMA-V19AQ-ALL-ARCHIVE-DECAM-CUTOUTS-1.0.0",
        "status": "completed_all_archive_group_acquisition",
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
            "checksum_keyword_present": len(present_checksum_rows),
            "datasum_keyword_present": len(present_datasum_rows),
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
    if not gates["all_v19aq_acquisition_gates_pass"]:
        raise RuntimeError("V19AQ acquisition gates failed")
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
