#!/usr/bin/env python3
"""Plan, acquire, and structurally validate grouped V19AN DECam cutouts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19an_nsc_decam_group_cutouts.json"
USER_AGENT = "SigmaGravity-V19AN-NSC-DECam-group-cutouts/1.0"
PLAN_COLUMNS = [
    "group_id",
    "exposure",
    "filter",
    "sia_extension",
    "split_scope",
    "anchor_count",
    "measurement_rows",
    "nsc_ids",
    "member_ids",
    "ra_min_deg",
    "ra_max_deg",
    "dec_min_deg",
    "dec_max_deg",
    "center_ra_deg",
    "center_dec_deg",
    "size_ra_deg",
    "size_dec_deg",
    "access_url",
    "output_path",
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


def validate_config(config_path: Path, config: dict[str, Any], require_frozen: bool) -> dict[str, str]:
    allowed = {"planning_group_manifest_no_pixels", "frozen_before_v19an_pixel_retrieval"}
    if config["status"] not in allowed:
        raise RuntimeError("V19AN status is invalid")
    if require_frozen and config["status"] != "frozen_before_v19an_pixel_retrieval":
        raise RuntimeError("V19AN pixel retrieval is not frozen")
    hashes: dict[str, str] = {}
    for artifact in config["parent_artifacts"]:
        path = ROOT / artifact["path"]
        digest = sha256(path)
        if digest != artifact["sha256"]:
            raise RuntimeError(f"parent artifact hash mismatch: {path}")
        hashes[artifact["path"]] = digest
    runner = ROOT / config["implementation"]["runner"]
    digest = sha256(runner)
    if digest != config["implementation"]["runner_sha256"]:
        raise RuntimeError("V19AN runner hash mismatch")
    hashes["runner"] = digest
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
        raise RuntimeError("V19AN authorizes a prohibited downstream action")
    if require_frozen and not config["authorization"]["download_every_group_cutout"]:
        raise RuntimeError("frozen V19AN does not authorize complete group retrieval")
    return hashes


def replace_cutout_region(base_url: str, ra: float, dec: float, size_ra: float, size_dec: float) -> str:
    parsed = urllib.parse.urlparse(base_url)
    query = urllib.parse.parse_qs(parsed.query)
    required = {}
    for key in ("col", "siaRef", "extn"):
        values = query.get(key, [])
        if len(values) != 1:
            raise RuntimeError(f"access URL lacks unique {key}: {base_url}")
        required[key] = values[0]
    ordered = [
        ("col", required["col"]),
        ("siaRef", required["siaRef"]),
        ("extn", required["extn"]),
        ("POS", f"{ra:.12f},{dec:.12f}"),
        ("SIZE", f"{size_ra:.8f},{size_dec:.8f}"),
    ]
    return urllib.parse.urlunparse(
        (parsed.scheme, parsed.netloc, parsed.path, "", urllib.parse.urlencode(ordered), "")
    )


def build_plan(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[dict[str, str]]]]:
    manifest = read_csv(ROOT / config["inputs"]["v19am_manifest"])
    expected_rows = int(config["gates"]["exact_measurement_rows"])
    if len(manifest) != expected_rows:
        raise RuntimeError(f"V19AN parent manifest row count changed: {len(manifest)}")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in manifest:
        grouped[(row["exposure"], row["sia_extension"])].append(row)
    if len(grouped) != int(config["gates"]["exact_exposure_extension_groups"]):
        raise RuntimeError(f"V19AN group count changed: {len(grouped)}")

    settings = config["group_cutout"]
    margin = float(settings["fixed_sky_margin_degrees"])
    minimum = float(settings["minimum_coordinate_size_degrees"])
    maximum_ra = float(settings["maximum_ra_coordinate_size_degrees"])
    maximum_dec = float(settings["maximum_dec_coordinate_size_degrees"])
    output_root = ROOT / config["outputs"]["cutout_directory"]
    plan: list[dict[str, Any]] = []
    for (exposure, extension), rows in sorted(grouped.items()):
        filters = {row["filter"] for row in rows}
        refs = {row["sia_ref"] for row in rows}
        urls = {row["sia_access_url"] for row in rows}
        if len(filters) != 1 or refs != {exposure}:
            raise RuntimeError(f"inconsistent V19AN group {exposure}/{extension}")
        anchors: dict[str, dict[str, str]] = {}
        for row in rows:
            anchors[row["nsc_id"]] = row
        ra_values = [float(row["ra_deg"]) for row in anchors.values()]
        dec_values = [float(row["dec_deg"]) for row in anchors.values()]
        ra_min, ra_max = min(ra_values), max(ra_values)
        dec_min, dec_max = min(dec_values), max(dec_values)
        center_ra = 0.5 * (ra_min + ra_max)
        center_dec = 0.5 * (dec_min + dec_max)
        cos_dec = max(math.cos(math.radians(center_dec)), 0.1)
        ra_margin_coordinate = margin / cos_dec
        size_ra = max(minimum, ra_max - ra_min + 2.0 * ra_margin_coordinate)
        size_dec = max(minimum, dec_max - dec_min + 2.0 * margin)
        if size_ra > maximum_ra or size_dec > maximum_dec:
            raise RuntimeError(
                f"group request exceeds frozen ceiling: {exposure}/{extension} "
                f"({size_ra:.5f},{size_dec:.5f})"
            )
        split_values = {row["split"] for row in anchors.values()}
        split_scope = next(iter(split_values)) if len(split_values) == 1 else "mixed"
        base_url = sorted(urls)[0]
        access_url = replace_cutout_region(
            base_url, center_ra, center_dec, size_ra, size_dec
        )
        group_id = f"{exposure}_ext{int(extension):02d}"
        output_path = output_root / f"{group_id}.fits"
        plan.append(
            {
                "group_id": group_id,
                "exposure": exposure,
                "filter": next(iter(filters)),
                "sia_extension": extension,
                "split_scope": split_scope,
                "anchor_count": len(anchors),
                "measurement_rows": len(rows),
                "nsc_ids": ";".join(sorted(anchors)),
                "member_ids": ";".join(
                    sorted({row["member_id"] for row in anchors.values()})
                ),
                "ra_min_deg": f"{ra_min:.12f}",
                "ra_max_deg": f"{ra_max:.12f}",
                "dec_min_deg": f"{dec_min:.12f}",
                "dec_max_deg": f"{dec_max:.12f}",
                "center_ra_deg": f"{center_ra:.12f}",
                "center_dec_deg": f"{center_dec:.12f}",
                "size_ra_deg": f"{size_ra:.8f}",
                "size_dec_deg": f"{size_dec:.8f}",
                "access_url": access_url,
                "output_path": output_path.relative_to(ROOT).as_posix(),
            }
        )
    return plan, grouped


def assert_plan_matches(path: Path, expected: list[dict[str, Any]]) -> None:
    actual = read_csv(path)
    normalized = [{key: str(value) for key, value in row.items()} for row in expected]
    if actual != normalized:
        raise RuntimeError("frozen V19AN group plan does not match deterministic rebuild")


def fetch(url: str, timeout: float, attempts: int) -> tuple[int, bytes]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = int(response.status)
                payload = response.read()
            if status == 200:
                return status, payload
            raise RuntimeError(f"HTTP status {status}")
        except (urllib.error.URLError, TimeoutError, RuntimeError):
            if attempt == attempts:
                raise
            time.sleep(float(attempt))
    raise AssertionError("unreachable")


def image_hdu(payload: bytes) -> tuple[int, fits.Header, np.ndarray]:
    with fits.open(io.BytesIO(payload), memmap=False, checksum=True) as hdul:
        hdul.verify("exception")
        for index, hdu in enumerate(hdul):
            if hdu.data is not None and np.asarray(hdu.data).ndim == 2:
                return index, hdu.header.copy(), np.asarray(hdu.data, dtype=float).copy()
    raise RuntimeError("downloaded FITS has no two-dimensional image HDU")


def inspect_payload(payload: bytes, group_rows: list[dict[str, str]], tolerance: float) -> dict[str, Any]:
    index, header, data = image_hdu(payload)
    if data.shape[0] < 16 or data.shape[1] < 16:
        raise RuntimeError(f"cutout is unexpectedly small: {data.shape}")
    wcs = WCS(header)
    if not wcs.has_celestial:
        raise RuntimeError("cutout lacks celestial WCS")
    anchors: dict[str, tuple[float, float]] = {}
    for row in group_rows:
        anchors[row["nsc_id"]] = (float(row["ra_deg"]), float(row["dec_deg"]))
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
        raise RuntimeError("one or more frozen anchors fall outside the returned cutout")
    margins = np.minimum.reduce(
        [pixel[:, 0], nx - 1 - pixel[:, 0], pixel[:, 1], ny - 1 - pixel[:, 1]]
    )
    finite_fraction = float(np.isfinite(data).mean())
    if finite_fraction <= 0.0:
        raise RuntimeError("cutout contains no finite pixels")
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
        "fits_integrity_passed": True,
    }


def build_plan_only(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    hashes = validate_config(config_path, config, require_frozen=False)
    plan, _groups = build_plan(config)
    path = ROOT / config["outputs"]["group_plan"]
    write_csv(path, plan, PLAN_COLUMNS)
    return {
        "status": "group_plan_built_without_pixel_access",
        "config_sha256": sha256(config_path),
        "parent_hashes": hashes,
        "groups": len(plan),
        "unique_exposures": len({row["exposure"] for row in plan}),
        "maximum_size_ra_deg": max(float(row["size_ra_deg"]) for row in plan),
        "maximum_size_dec_deg": max(float(row["size_dec_deg"]) for row in plan),
        "group_plan": path.relative_to(ROOT).as_posix(),
        "group_plan_sha256": sha256(path),
        "image_pixels_downloaded": False,
    }


def acquire(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parent_hashes = validate_config(config_path, config, require_frozen=True)
    plan, groups = build_plan(config)
    plan_path = ROOT / config["outputs"]["group_plan"]
    if sha256(plan_path) != config["frozen_group_plan"]["sha256"]:
        raise RuntimeError("V19AN frozen group-plan hash mismatch")
    assert_plan_matches(plan_path, plan)

    download_rows: list[dict[str, Any]] = []
    source = config["retrieval"]
    total = len(plan)
    for number, row in enumerate(plan, start=1):
        output = ROOT / row["output_path"]
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.is_file():
            payload = output.read_bytes()
            status = 200
        else:
            status, payload = fetch(
                row["access_url"],
                float(source["timeout_seconds"]),
                int(source["maximum_attempts"]),
            )
            output.write_bytes(payload)
        inspection = inspect_payload(
            payload,
            groups[(row["exposure"], row["sia_extension"])],
            float(source["wcs_containment_tolerance_pixel"]),
        )
        download_rows.append(
            {
                **row,
                "http_status": status,
                "download_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                **inspection,
            }
        )
        print(f"V19AN {number}/{total} {row['group_id']} {len(payload)} bytes", flush=True)

    download_manifest = ROOT / config["outputs"]["download_manifest"]
    write_csv(download_manifest, download_rows, DOWNLOAD_COLUMNS)
    gates = {
        "exact_group_cutouts": len(download_rows)
        == int(config["gates"]["exact_exposure_extension_groups"]),
        "exact_unique_exposures": len({row["exposure"] for row in download_rows})
        == int(config["gates"]["exact_unique_exposures"]),
        "exact_measurement_membership_preserved": sum(
            int(row["measurement_rows"]) for row in download_rows
        )
        == int(config["gates"]["exact_measurement_rows"]),
        "http_200_every_group": all(int(row["http_status"]) == 200 for row in download_rows),
        "fits_integrity_every_group": all(
            bool(row["fits_integrity_passed"]) for row in download_rows
        ),
        "celestial_wcs_every_group": all(bool(row["wcs_celestial"]) for row in download_rows),
        "all_frozen_anchors_contained": all(
            bool(row["anchors_contained"]) for row in download_rows
        ),
        "at_least_one_finite_pixel_every_group": all(
            float(row["finite_pixel_fraction"]) > 0.0 for row in download_rows
        ),
        "all_groups_retained_without_exposure_selection": True,
        "no_photometric_model_or_prohibited_payload_opened": True,
    }
    gates["all_acquisition_integrity_gates_pass"] = all(gates.values())
    report_path = ROOT / config["outputs"]["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "report_version": "SIGMA-V19AN-NSC-DECAM-GROUP-CUTOUTS-1.0.0",
        "status": "completed_group_cutout_acquisition",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "parent_hashes": parent_hashes,
        "frozen_group_plan": config["frozen_group_plan"],
        "counts": {
            "group_cutouts": len(download_rows),
            "unique_exposures": len({row["exposure"] for row in download_rows}),
            "measurement_memberships": sum(int(row["measurement_rows"]) for row in download_rows),
            "download_bytes": sum(int(row["download_bytes"]) for row in download_rows),
            "development_only_groups": sum(row["split_scope"] == "development" for row in download_rows),
            "validation_only_groups": sum(row["split_scope"] == "validation" for row in download_rows),
            "mixed_groups": sum(row["split_scope"] == "mixed" for row in download_rows),
        },
        "minimum_finite_pixel_fraction": min(
            float(row["finite_pixel_fraction"]) for row in download_rows
        ),
        "minimum_anchor_edge_margin_pixel": min(
            float(row["minimum_anchor_edge_margin_pixel"]) for row in download_rows
        ),
        "gates": gates,
        "outputs": {
            "download_manifest": download_manifest.relative_to(ROOT).as_posix(),
            "download_manifest_sha256": sha256(download_manifest),
        },
        "claim_boundary": config["claim_boundary"],
        "exposures_ranked_or_selected": False,
        "photometry_fitted_or_compared": False,
        "psf_or_deblend_model_selected": False,
        "ambiguous_candidates_queried": False,
        "mass_or_current_inferred": False,
        "lensing_or_halo_payload_opened": False,
        "gravity_formula_or_parameter_changed": False,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    result = build_plan_only(args.config) if args.plan_only else acquire(args.config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
