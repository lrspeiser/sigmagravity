#!/usr/bin/env python3
"""Resolve Archive FITS HDUs by header WCS, then acquire every frozen group."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


SCRIPT_DIR = Path(__file__).resolve().parent
AO_PATH = SCRIPT_DIR / "acquire_sigma_v19ao_resilient_decam_cutouts.py"
AO_SPEC = importlib.util.spec_from_file_location("sigma_v19ao_base", AO_PATH)
AO = importlib.util.module_from_spec(AO_SPEC)
assert AO_SPEC.loader is not None
AO_SPEC.loader.exec_module(AO)

ROOT = AO.ROOT
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19ap_header_wcs_resolved_decam_cutouts.json"
USER_AGENT = "SigmaGravity-V19AP-header-WCS-DECam-cutouts/1.0"
PLAN_COLUMNS = AO.PLAN_COLUMNS + [
    "archive_header_url",
    "archive_header_payload_path",
    "archive_header_payload_sha256",
    "vohdu_hdu_index",
    "fits_hdu_index",
    "header_extname",
    "header_ccdnum",
]
DOWNLOAD_COLUMNS = PLAN_COLUMNS + [
    "http_status",
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
    "fits_integrity_passed",
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
        "planning_archive_header_wcs_without_new_pixels",
        "frozen_before_v19ap_pixel_retrieval",
    }
    if config["status"] not in allowed:
        raise RuntimeError("V19AP status is invalid")
    if require_frozen and config["status"] != "frozen_before_v19ap_pixel_retrieval":
        raise RuntimeError("V19AP pixel retrieval is not frozen")
    hashes: dict[str, str] = {}
    for artifact in config["parent_artifacts"]:
        path = ROOT / artifact["path"]
        digest = sha256(path)
        if digest != artifact["sha256"]:
            raise RuntimeError(f"V19AP parent hash mismatch: {path}")
        hashes[artifact["path"]] = digest
    runner = ROOT / config["implementation"]["runner"]
    digest = sha256(runner)
    if digest != config["implementation"]["runner_sha256"]:
        raise RuntimeError("V19AP runner hash mismatch")
    hashes["runner"] = digest
    if sha256(AO_PATH) != config["implementation"]["frozen_v19ao_runner_sha256"]:
        raise RuntimeError("V19AP V19AO-runner hash mismatch")
    hashes["frozen_v19ao_runner"] = sha256(AO_PATH)
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
        raise RuntimeError("V19AP authorizes a prohibited action")
    if require_frozen and not config["authorization"]["download_every_frozen_group"]:
        raise RuntimeError("V19AP does not authorize complete frozen retrieval")
    return hashes


def fetch_json(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if int(response.status) != 200:
            raise RuntimeError(f"Archive header status {response.status}")
        return response.read()


def header_url(template: str, md5: str) -> str:
    return template.format(md5=md5)


def header_wcs_candidates(
    payload: bytes,
    group_rows: list[dict[str, str]],
    tolerance: float,
) -> list[dict[str, Any]]:
    decoded = json.loads(payload)
    if not isinstance(decoded, list):
        raise RuntimeError("Archive header response is not a header list")
    anchors: dict[str, tuple[float, float]] = {}
    for row in group_rows:
        anchors[row["nsc_id"]] = (float(row["ra_deg"]), float(row["dec_deg"]))
    world = np.asarray(list(anchors.values()), dtype=float)
    candidates: list[dict[str, Any]] = []
    for index, mapping in enumerate(decoded):
        if not isinstance(mapping, dict):
            continue
        try:
            nx = int(mapping["NAXIS1"])
            ny = int(mapping["NAXIS2"])
            header = fits.Header(mapping)
            wcs = WCS(header)
        except (KeyError, TypeError, ValueError):
            continue
        if nx < 16 or ny < 16 or not wcs.has_celestial:
            continue
        pixel = wcs.celestial.all_world2pix(world, 0)
        contained = (
            (pixel[:, 0] >= -tolerance)
            & (pixel[:, 0] <= nx - 1 + tolerance)
            & (pixel[:, 1] >= -tolerance)
            & (pixel[:, 1] <= ny - 1 + tolerance)
        )
        if bool(np.all(contained)):
            margins = np.minimum.reduce(
                [
                    pixel[:, 0],
                    nx - 1 - pixel[:, 0],
                    pixel[:, 1],
                    ny - 1 - pixel[:, 1],
                ]
            )
            candidates.append(
                {
                    "fits_hdu_index": index,
                    "header_extname": str(mapping.get("EXTNAME", "")),
                    "header_ccdnum": str(mapping.get("CCDNUM", "")),
                    "minimum_anchor_edge_margin_pixel": float(np.min(margins)),
                    "header_count": len(decoded),
                }
            )
    return candidates


def resolve_unique_header(
    payload: bytes,
    group_rows: list[dict[str, str]],
    tolerance: float,
) -> dict[str, Any]:
    candidates = header_wcs_candidates(payload, group_rows, tolerance)
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one full-header WCS containing all anchors, found {len(candidates)}"
        )
    return candidates[0]


def build_resolved_plan(config: dict[str, Any]) -> list[dict[str, Any]]:
    base_rows = read_csv(ROOT / config["inputs"]["v19ao_hybrid_plan"])
    groups = AO.parent_groups(config)
    raw_root = ROOT / config["outputs"]["archive_header_directory"]
    raw_root.mkdir(parents=True, exist_ok=True)
    output_root = ROOT / config["outputs"]["cutout_directory"]
    cache: dict[str, tuple[bytes, Path]] = {}
    plan: list[dict[str, Any]] = []
    for base in base_rows:
        row: dict[str, Any] = dict(base)
        row["output_path"] = (
            output_root / f"{base['group_id']}.fits"
        ).relative_to(ROOT).as_posix()
        if base["retrieval_method"] == "archive_selected_hdu":
            md5 = base["source_md5"]
            url = header_url(config["archive_header_resolution"]["header_endpoint"], md5)
            if md5 not in cache:
                raw_path = raw_root / f"{md5}.json"
                payload = fetch_json(
                    url, float(config["archive_header_resolution"]["timeout_seconds"])
                )
                raw_path.write_bytes(payload)
                cache[md5] = (payload, raw_path)
            payload, raw_path = cache[md5]
            group_rows = groups[(base["exposure"], base["sia_extension"])]
            resolved = resolve_unique_header(
                payload,
                group_rows,
                float(config["archive_header_resolution"]["wcs_containment_tolerance_pixel"]),
            )
            fits_index = int(resolved["fits_hdu_index"])
            row.update(
                {
                    "retrieval_method": "archive_header_wcs_selected_hdu",
                    "retrieval_url": config["archive_header_resolution"][
                        "retrieval_endpoint"
                    ].format(md5=md5, fits_hdu_index=fits_index),
                    "archive_header_url": url,
                    "archive_header_payload_path": raw_path.relative_to(ROOT).as_posix(),
                    "archive_header_payload_sha256": sha256(raw_path),
                    "vohdu_hdu_index": base["source_hdu_index"],
                    "fits_hdu_index": fits_index,
                    "header_extname": resolved["header_extname"],
                    "header_ccdnum": resolved["header_ccdnum"],
                }
            )
        elif base["retrieval_method"] == "nsc_sia_group_cutout":
            row.update(
                {
                    "archive_header_url": "",
                    "archive_header_payload_path": "",
                    "archive_header_payload_sha256": "",
                    "vohdu_hdu_index": "",
                    "fits_hdu_index": "",
                    "header_extname": "",
                    "header_ccdnum": "",
                }
            )
        else:
            raise RuntimeError(f"unexpected V19AO retrieval method: {base['retrieval_method']}")
        plan.append(row)
    return plan


def plan_only(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    hashes = validate_config(config, require_frozen=False)
    plan = build_resolved_plan(config)
    plan_path = ROOT / config["outputs"]["resolved_plan"]
    write_csv(plan_path, plan, PLAN_COLUMNS)
    methods = Counter(row["retrieval_method"] for row in plan)
    archive_rows = [
        row for row in plan if row["retrieval_method"] == "archive_header_wcs_selected_hdu"
    ]
    return {
        "status": "archive_headers_resolved_without_new_image_pixel_access",
        "config_sha256": sha256(config_path),
        "parent_hashes": hashes,
        "groups": len(plan),
        "measurement_memberships": sum(int(row["measurement_rows"]) for row in plan),
        "retrieval_methods": dict(sorted(methods.items())),
        "unique_archive_header_payloads": len(
            {row["archive_header_payload_sha256"] for row in archive_rows}
        ),
        "unique_archive_retrieval_urls": len(
            {row["retrieval_url"] for row in archive_rows}
        ),
        "resolved_plan": plan_path.relative_to(ROOT).as_posix(),
        "resolved_plan_sha256": sha256(plan_path),
        "new_image_pixels_downloaded": False,
    }


def verify_archive_identity(payload: bytes, row: dict[str, str]) -> dict[str, str]:
    _index, header, _data = AO.BASE.image_hdu(payload)
    extname = str(header.get("EXTNAME", ""))
    ccdnum = str(header.get("CCDNUM", ""))
    if extname != row["header_extname"] or ccdnum != row["header_ccdnum"]:
        raise RuntimeError(
            "returned Archive detector does not match the frozen header-WCS identity"
        )
    return {"returned_extname": extname, "returned_ccdnum": ccdnum}


def acquire(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    hashes = validate_config(config, require_frozen=True)
    plan_path = ROOT / config["frozen_resolved_plan"]["path"]
    if sha256(plan_path) != config["frozen_resolved_plan"]["sha256"]:
        raise RuntimeError("V19AP frozen resolved-plan hash mismatch")
    plan = read_csv(plan_path)
    groups = AO.parent_groups(config)
    if len(plan) != int(config["gates"]["exact_groups"]):
        raise RuntimeError("V19AP frozen group count changed")

    completed: list[dict[str, Any]] = []
    retrieval = config["retrieval"]
    for number, row in enumerate(plan, start=1):
        output = ROOT / row["output_path"]
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.is_file():
            payload = output.read_bytes()
            status = 200
        else:
            status, payload = AO.BASE.fetch(
                row["retrieval_url"],
                float(retrieval["timeout_seconds"]),
                int(retrieval["maximum_attempts"]),
            )
        inspection = AO.BASE.inspect_payload(
            payload,
            groups[(row["exposure"], row["sia_extension"])],
            float(retrieval["wcs_containment_tolerance_pixel"]),
        )
        detector = {"returned_extname": "", "returned_ccdnum": ""}
        if row["retrieval_method"] == "archive_header_wcs_selected_hdu":
            detector = verify_archive_identity(payload, row)
        if not output.is_file():
            output.write_bytes(payload)
        completed.append(
            {
                **row,
                "http_status": status,
                "download_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                **inspection,
                **detector,
            }
        )
        print(
            f"V19AP {number}/{len(plan)} {row['group_id']} "
            f"{row['retrieval_method']} {len(payload)} bytes",
            flush=True,
        )

    manifest_path = ROOT / config["outputs"]["download_manifest"]
    write_csv(manifest_path, completed, DOWNLOAD_COLUMNS)
    methods = Counter(row["retrieval_method"] for row in completed)
    gates = {
        "exact_groups": len(completed) == int(config["gates"]["exact_groups"]),
        "exact_measurement_memberships": sum(
            int(row["measurement_rows"]) for row in completed
        )
        == int(config["gates"]["exact_measurement_memberships"]),
        "exact_sia_groups": methods["nsc_sia_group_cutout"]
        == int(config["gates"]["exact_sia_groups"]),
        "exact_archive_groups": methods["archive_header_wcs_selected_hdu"]
        == int(config["gates"]["exact_archive_groups"]),
        "http_200_every_group": all(int(row["http_status"]) == 200 for row in completed),
        "fits_integrity_every_group": all(
            bool(row["fits_integrity_passed"]) for row in completed
        ),
        "celestial_wcs_every_group": all(bool(row["wcs_celestial"]) for row in completed),
        "all_frozen_anchors_contained": all(
            bool(row["anchors_contained"]) for row in completed
        ),
        "at_least_one_finite_pixel_every_group": all(
            float(row["finite_pixel_fraction"]) > 0.0 for row in completed
        ),
        "archive_detector_identity_every_group": all(
            row["retrieval_method"] != "archive_header_wcs_selected_hdu"
            or (
                row["returned_extname"] == row["header_extname"]
                and row["returned_ccdnum"] == row["header_ccdnum"]
            )
            for row in completed
        ),
        "all_groups_retained_without_selection": True,
        "no_photometric_model_or_prohibited_payload_opened": True,
    }
    gates["all_v19ap_acquisition_gates_pass"] = all(gates.values())
    report_path = ROOT / config["outputs"]["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "report_version": "SIGMA-V19AP-HEADER-WCS-RESOLVED-DECAM-CUTOUTS-1.0.0",
        "status": "completed_header_wcs_resolved_group_acquisition",
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
            "retrieval_methods": dict(sorted(methods.items())),
            "download_bytes": sum(int(row["download_bytes"]) for row in completed),
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
    if not gates["all_v19ap_acquisition_gates_pass"]:
        raise RuntimeError("V19AP acquisition gates failed")
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
