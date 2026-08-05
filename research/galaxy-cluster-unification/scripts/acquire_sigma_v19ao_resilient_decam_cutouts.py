#!/usr/bin/env python3
"""Resolve stale NSC descriptors and acquire every V19AO DECam image group."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_PATH = SCRIPT_DIR / "acquire_sigma_v19an_nsc_decam_group_cutouts.py"
BASE_SPEC = importlib.util.spec_from_file_location("sigma_v19an_base", BASE_PATH)
BASE = importlib.util.module_from_spec(BASE_SPEC)
assert BASE_SPEC.loader is not None
BASE_SPEC.loader.exec_module(BASE)

ROOT = BASE.ROOT
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19ao_resilient_decam_cutouts.json"
USER_AGENT = "SigmaGravity-V19AO-resilient-DECam-cutouts/1.0"
MD5 = re.compile(r"^[0-9a-f]{32}$")
PLAN_COLUMNS = BASE.PLAN_COLUMNS + [
    "retrieval_method",
    "identity_query_url",
    "identity_payload_path",
    "identity_payload_sha256",
    "source_archive_filename",
    "source_md5",
    "source_hdu_index",
    "source_updated_utc",
    "retrieval_url",
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


def validate_config(config: dict[str, Any], require_frozen: bool) -> dict[str, str]:
    allowed = {"planning_archive_identity_no_pixels", "frozen_before_v19ao_pixel_retrieval"}
    if config["status"] not in allowed:
        raise RuntimeError("V19AO status is invalid")
    if require_frozen and config["status"] != "frozen_before_v19ao_pixel_retrieval":
        raise RuntimeError("V19AO pixel retrieval is not frozen")
    hashes: dict[str, str] = {}
    for artifact in config["parent_artifacts"]:
        path = ROOT / artifact["path"]
        digest = sha256(path)
        if digest != artifact["sha256"]:
            raise RuntimeError(f"V19AO parent hash mismatch: {path}")
        hashes[artifact["path"]] = digest
    runner = ROOT / config["implementation"]["runner"]
    digest = sha256(runner)
    if digest != config["implementation"]["runner_sha256"]:
        raise RuntimeError("V19AO runner hash mismatch")
    hashes["runner"] = digest
    if sha256(BASE_PATH) != config["implementation"]["frozen_base_runner_sha256"]:
        raise RuntimeError("V19AO base-runner hash mismatch")
    hashes["frozen_base_runner"] = sha256(BASE_PATH)
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
        raise RuntimeError("V19AO authorizes a prohibited action")
    if require_frozen and not config["authorization"]["download_every_frozen_group"]:
        raise RuntimeError("V19AO does not authorize complete frozen retrieval")
    return hashes


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def parent_groups(config: dict[str, Any]) -> dict[tuple[str, str], list[dict[str, str]]]:
    rows = read_csv(ROOT / config["inputs"]["v19am_manifest"])
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["exposure"], row["sia_extension"])].append(row)
    return groups


def archive_identity_url(endpoint: str, exposure: str, anchor: dict[str, str]) -> str:
    if not exposure.endswith("_d2"):
        raise ValueError("archive fallback is only defined for stale _d2 descriptors")
    prefix = exposure[:-2]
    params = [
        ("POS", f"{float(anchor['ra_deg']):.12f},{float(anchor['dec_deg']):.12f}"),
        ("SIZE", "0.01"),
        ("archive_filename", prefix),
        ("VERB", "3"),
        ("format", "json"),
        ("limit", "100"),
    ]
    return endpoint + "?" + urllib.parse.urlencode(params, safe=",")


def query_archive_identity(url: str, timeout: float) -> tuple[bytes, dict[str, Any]]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if int(response.status) != 200:
            raise RuntimeError(f"archive identity status {response.status}")
        payload = response.read()
    decoded = json.loads(payload)
    rows = [row for row in decoded if isinstance(row, dict) and "md5sum" in row]
    if len(rows) != 1:
        raise RuntimeError(f"expected one archive identity row, found {len(rows)}")
    return payload, rows[0]


def validate_archive_identity(row: dict[str, Any], exposure: str, expected_filter: str) -> None:
    archive_name = Path(str(row["archive_filename"])).name
    prefix = exposure[:-2]
    if not archive_name.startswith(prefix):
        raise RuntimeError(f"archive identity does not match exposure prefix: {archive_name}")
    if str(row["instrument"]).lower() != "decam":
        raise RuntimeError("archive fallback is not DECam")
    if str(row["proc_type"]).lower() != "instcal" or str(row["prod_type"]).lower() != "image":
        raise RuntimeError("archive fallback is not an InstCal image")
    if not str(row["filter"]).startswith(expected_filter):
        raise RuntimeError("archive fallback filter does not match")
    if not MD5.fullmatch(str(row["md5sum"])):
        raise RuntimeError("archive fallback MD5 is invalid")
    expected_suffix = f"?hdus=0,{int(row['hdu_idx'])}"
    if not str(row["url"]).endswith(expected_suffix):
        raise RuntimeError("archive retrieval URL is not an exact primary-plus-HDU subset")


def build_hybrid_plan(config: dict[str, Any]) -> list[dict[str, Any]]:
    base_rows = read_csv(ROOT / config["inputs"]["v19an_group_plan"])
    groups = parent_groups(config)
    raw_root = ROOT / config["outputs"]["identity_payload_directory"]
    raw_root.mkdir(parents=True, exist_ok=True)
    output_root = ROOT / config["outputs"]["cutout_directory"]
    plan: list[dict[str, Any]] = []
    for base in base_rows:
        exposure = base["exposure"]
        extension = base["sia_extension"]
        group_rows = groups[(exposure, extension)]
        row: dict[str, Any] = dict(base)
        row["output_path"] = (
            output_root / f"{base['group_id']}.fits"
        ).relative_to(ROOT).as_posix()
        if exposure.endswith("_d2"):
            representative = min(group_rows, key=lambda item: item["nsc_id"])
            identity_url = archive_identity_url(
                config["archive_fallback"]["vohdu_endpoint"], exposure, representative
            )
            payload, identity = query_archive_identity(
                identity_url, float(config["archive_fallback"]["timeout_seconds"])
            )
            validate_archive_identity(identity, exposure, base["filter"])
            raw_path = raw_root / f"{base['group_id']}.json"
            raw_path.write_bytes(payload)
            row.update(
                {
                    "retrieval_method": "archive_selected_hdu",
                    "identity_query_url": identity_url,
                    "identity_payload_path": raw_path.relative_to(ROOT).as_posix(),
                    "identity_payload_sha256": sha256(raw_path),
                    "source_archive_filename": Path(str(identity["archive_filename"])).name,
                    "source_md5": str(identity["md5sum"]),
                    "source_hdu_index": int(identity["hdu_idx"]),
                    "source_updated_utc": str(identity["file_updated"]),
                    "retrieval_url": str(identity["url"]),
                }
            )
        else:
            row.update(
                {
                    "retrieval_method": "nsc_sia_group_cutout",
                    "identity_query_url": "",
                    "identity_payload_path": "",
                    "identity_payload_sha256": "",
                    "source_archive_filename": "",
                    "source_md5": "",
                    "source_hdu_index": "",
                    "source_updated_utc": "",
                    "retrieval_url": base["access_url"],
                }
            )
        plan.append(row)
    return plan


def plan_only(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    hashes = validate_config(config, require_frozen=False)
    plan = build_hybrid_plan(config)
    plan_path = ROOT / config["outputs"]["hybrid_plan"]
    write_csv(plan_path, plan, PLAN_COLUMNS)
    methods = Counter(row["retrieval_method"] for row in plan)
    return {
        "status": "hybrid_identity_plan_built_without_pixel_access",
        "config_sha256": sha256(config_path),
        "parent_hashes": hashes,
        "groups": len(plan),
        "measurement_memberships": sum(int(row["measurement_rows"]) for row in plan),
        "retrieval_methods": dict(sorted(methods.items())),
        "hybrid_plan": plan_path.relative_to(ROOT).as_posix(),
        "hybrid_plan_sha256": sha256(plan_path),
        "identity_payloads": sum(bool(row["identity_payload_path"]) for row in plan),
        "image_pixels_downloaded": False,
    }


def acquire(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    hashes = validate_config(config, require_frozen=True)
    plan_path = ROOT / config["frozen_hybrid_plan"]["path"]
    if sha256(plan_path) != config["frozen_hybrid_plan"]["sha256"]:
        raise RuntimeError("V19AO frozen hybrid-plan hash mismatch")
    plan = read_csv(plan_path)
    groups = parent_groups(config)
    if len(plan) != int(config["gates"]["exact_groups"]):
        raise RuntimeError("V19AO frozen group count changed")

    completed: list[dict[str, Any]] = []
    retrieval = config["retrieval"]
    for number, row in enumerate(plan, start=1):
        output = ROOT / row["output_path"]
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.is_file():
            payload = output.read_bytes()
            status = 200
        else:
            status, payload = BASE.fetch(
                row["retrieval_url"],
                float(retrieval["timeout_seconds"]),
                int(retrieval["maximum_attempts"]),
            )
            output.write_bytes(payload)
        inspection = BASE.inspect_payload(
            payload,
            groups[(row["exposure"], row["sia_extension"])],
            float(retrieval["wcs_containment_tolerance_pixel"]),
        )
        completed.append(
            {
                **row,
                "http_status": status,
                "download_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                **inspection,
            }
        )
        print(
            f"V19AO {number}/{len(plan)} {row['group_id']} "
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
        "exact_archive_groups": methods["archive_selected_hdu"]
        == int(config["gates"]["exact_archive_groups"]),
        "http_200_every_group": all(int(row["http_status"]) == 200 for row in completed),
        "fits_integrity_every_group": all(bool(row["fits_integrity_passed"]) for row in completed),
        "celestial_wcs_every_group": all(bool(row["wcs_celestial"]) for row in completed),
        "all_frozen_anchors_contained": all(bool(row["anchors_contained"]) for row in completed),
        "at_least_one_finite_pixel_every_group": all(
            float(row["finite_pixel_fraction"]) > 0.0 for row in completed
        ),
        "all_groups_retained_without_selection": True,
        "no_photometric_model_or_prohibited_payload_opened": True,
    }
    gates["all_resilient_acquisition_gates_pass"] = all(gates.values())
    report_path = ROOT / config["outputs"]["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "report_version": "SIGMA-V19AO-RESILIENT-DECAM-CUTOUTS-1.0.0",
        "status": "completed_resilient_group_acquisition",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "parent_hashes": hashes,
        "frozen_hybrid_plan": config["frozen_hybrid_plan"],
        "counts": {
            "groups": len(completed),
            "measurement_memberships": sum(int(row["measurement_rows"]) for row in completed),
            "retrieval_methods": dict(sorted(methods.items())),
            "download_bytes": sum(int(row["download_bytes"]) for row in completed),
        },
        "minimum_finite_pixel_fraction": min(float(row["finite_pixel_fraction"]) for row in completed),
        "minimum_anchor_edge_margin_pixel": min(
            float(row["minimum_anchor_edge_margin_pixel"]) for row in completed
        ),
        "gates": gates,
        "outputs": {
            "download_manifest": manifest_path.relative_to(ROOT).as_posix(),
            "download_manifest_sha256": sha256(manifest_path),
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
    result = plan_only(args.config) if args.plan_only else acquire(args.config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
