#!/usr/bin/env python3
"""Audit NSC SIA image coverage for the fifteen already-open V19AB anchors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io.votable import parse_single_table


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19am_nsc_sia_anchor_coverage.json"
USER_AGENT = "SigmaGravity-V19AM-NSC-SIA-anchor-coverage/1.0"
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
MANIFEST_COLUMNS = [
    "cluster",
    "member_id",
    "split",
    "nsc_id",
    "ra_deg",
    "dec_deg",
    "measid",
    "exposure",
    "filter",
    "mjd",
    "measurement_flags",
    "measurement_exposure_fwhm",
    "sia_assoc_id",
    "sia_ref",
    "sia_extension",
    "sia_access_url",
    "sia_access_format",
    "sia_access_estsize",
    "sia_calib_level",
    "sia_instrument_name",
    "sia_obs_bandpass",
    "sia_prodtype",
    "sia_proctype",
    "sia_date_obs",
    "sia_mjd_obs",
    "sia_seeing",
    "sia_exptime",
    "sia_magzero",
    "raw_metadata_path",
    "raw_metadata_sha256",
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


def validate_config(config_path: Path, config: dict[str, Any]) -> dict[str, str]:
    if config["status"] != "frozen_before_v19am_metadata_audit":
        raise RuntimeError("V19AM protocol is not frozen")
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
        raise RuntimeError("V19AM runner hash mismatch")
    hashes["runner"] = digest
    if config["authorization"]["download_image_pixels"]:
        raise RuntimeError("V19AM cannot download image pixels")
    prohibited = (
        "rank_or_select_exposures",
        "inspect_image_pixels",
        "query_ambiguous_candidates",
        "infer_photometry_mass_or_current",
        "read_lensing_or_halo_payload",
        "change_gravity_physics_or_parameters",
        "open_holdout",
    )
    if any(config["authorization"][name] for name in prohibited):
        raise RuntimeError("V19AM authorizes a prohibited action")
    return hashes


def scalar(value: Any) -> str:
    if np.ma.is_masked(value):
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def parse_access_descriptor(url: str) -> tuple[str, str]:
    query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    refs = query.get("siaRef", [])
    extensions = query.get("extn", [])
    if len(refs) != 1 or len(extensions) != 1:
        raise RuntimeError(f"SIA access URL lacks a unique siaRef/extn: {url}")
    ref = refs[0]
    if ref.endswith(".fits.fz"):
        ref = ref[: -len(".fits.fz")]
    return ref, extensions[0]


def build_query_url(endpoint: str, ra_deg: float, dec_deg: float, size_deg: float) -> str:
    return endpoint + "?" + urllib.parse.urlencode(
        {"POS": f"{ra_deg:.12f},{dec_deg:.12f}", "SIZE": f"{size_deg:.8f}"}
    )


def fetch_metadata(url: str, timeout_seconds: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        if int(response.status) != 200:
            raise RuntimeError(f"SIA metadata HTTP status {response.status}")
        return response.read()


def parse_metadata(payload: bytes) -> list[dict[str, str]]:
    table = parse_single_table(io.BytesIO(payload)).to_table(use_names_over_ids=True)
    return [
        {column: scalar(row[column]) for column in table.colnames}
        for row in table
    ]


def load_anchors(config: dict[str, Any]) -> list[dict[str, str]]:
    sample = read_csv(ROOT / config["inputs"]["commissioning_sample"])
    expected = int(config["gates"]["exact_anchor_count"])
    if len(sample) != expected:
        raise RuntimeError(f"V19AM anchor count changed: {len(sample)}")
    if Counter(row["split"] for row in sample) != Counter(
        {"development": 10, "validation": 5}
    ):
        raise RuntimeError("V19AM development/validation split changed")
    ids = [row["nsc_id"] for row in sample]
    if len(set(ids)) != expected or any(not SAFE_ID.fullmatch(item) for item in ids):
        raise RuntimeError("V19AM anchor IDs are not unique and safe")
    return sample


def run(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parent_hashes = validate_config(config_path, config)
    anchors = load_anchors(config)
    anchor_ids = {row["nsc_id"] for row in anchors}
    candidates = [
        row
        for row in read_csv(ROOT / config["inputs"]["unified_candidates"])
        if row["nsc_id"] in anchor_ids
    ]
    by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        by_id[row["nsc_id"]].append(row)
    if set(by_id) != anchor_ids or any(len(rows) != 1 for rows in by_id.values()):
        raise RuntimeError("each V19AM anchor must have exactly one frozen coordinate row")

    measurements = [
        row
        for row in read_csv(ROOT / config["inputs"]["all_measurements"])
        if row["objectid"] in anchor_ids
    ]
    expected_measurements = int(config["gates"]["exact_anchor_measurement_rows"])
    if len(measurements) != expected_measurements:
        raise RuntimeError(f"V19AM measurement count changed: {len(measurements)}")
    measurement_keys = [(row["objectid"], row["exposure"]) for row in measurements]
    if len(set(measurement_keys)) != len(measurement_keys):
        raise RuntimeError("duplicate object/exposure measurement in V19AM inputs")

    measurements_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in measurements:
        measurements_by_id[row["objectid"]].append(row)

    source = config["source"]
    raw_root = ROOT / config["outputs"]["raw_metadata_directory"]
    raw_root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    query_records: list[dict[str, Any]] = []
    anchor_lookup = {row["nsc_id"]: row for row in anchors}

    for nsc_id in [row["nsc_id"] for row in anchors]:
        candidate = by_id[nsc_id][0]
        ra_deg = float(candidate["ra_deg"])
        dec_deg = float(candidate["dec_deg"])
        url = build_query_url(
            source["endpoint"], ra_deg, dec_deg, float(source["query_size_degrees"])
        )
        payload = fetch_metadata(url, float(source["timeout_seconds"]))
        raw_path = raw_root / f"nsc_{nsc_id}.vot.xml"
        raw_path.write_bytes(payload)
        raw_rel = raw_path.relative_to(ROOT).as_posix()
        raw_digest = sha256(raw_path)
        metadata = parse_metadata(payload)
        rows_by_ref: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in metadata:
            ref, _extension = parse_access_descriptor(row["access_url"])
            rows_by_ref[ref].append(row)

        exact_rows = 0
        for measurement in measurements_by_id[nsc_id]:
            exposure = measurement["exposure"]
            matches = rows_by_ref.get(exposure, [])
            if len(matches) != 1:
                raise RuntimeError(
                    f"expected one SIA row for {nsc_id}/{exposure}, found {len(matches)}"
                )
            sia = matches[0]
            ref, extension = parse_access_descriptor(sia["access_url"])
            if ref != exposure:
                raise AssertionError("SIA reference changed during parsing")
            anchor = anchor_lookup[nsc_id]
            manifest.append(
                {
                    "cluster": anchor["cluster"],
                    "member_id": anchor["object_id"],
                    "split": anchor["split"],
                    "nsc_id": nsc_id,
                    "ra_deg": candidate["ra_deg"],
                    "dec_deg": candidate["dec_deg"],
                    "measid": measurement["measid"],
                    "exposure": exposure,
                    "filter": measurement["filter"],
                    "mjd": measurement["mjd"],
                    "measurement_flags": measurement["flags"],
                    "measurement_exposure_fwhm": measurement["exposure_fwhm"],
                    "sia_assoc_id": sia["assoc_id"],
                    "sia_ref": ref,
                    "sia_extension": extension,
                    "sia_access_url": sia["access_url"],
                    "sia_access_format": sia["access_format"],
                    "sia_access_estsize": sia["access_estsize"],
                    "sia_calib_level": sia["calib_level"],
                    "sia_instrument_name": sia["instrument_name"],
                    "sia_obs_bandpass": sia["obs_bandpass"],
                    "sia_prodtype": sia["prodtype"],
                    "sia_proctype": sia["proctype"],
                    "sia_date_obs": sia["date_obs"],
                    "sia_mjd_obs": sia["mjd_obs"],
                    "sia_seeing": sia["seeing"],
                    "sia_exptime": sia["exptime"],
                    "sia_magzero": sia["magzero"],
                    "raw_metadata_path": raw_rel,
                    "raw_metadata_sha256": raw_digest,
                }
            )
            exact_rows += 1
        query_records.append(
            {
                "nsc_id": nsc_id,
                "query_url": url,
                "returned_metadata_rows": len(metadata),
                "exact_measurement_matches": exact_rows,
                "raw_metadata_path": raw_rel,
                "raw_metadata_sha256": raw_digest,
            }
        )

    manifest.sort(key=lambda row: (row["nsc_id"], float(row["mjd"]), row["measid"]))
    manifest_path = ROOT / config["outputs"]["manifest"]
    write_csv(manifest_path, manifest, MANIFEST_COLUMNS)

    exact_instrument = all(
        row["sia_instrument_name"] == source["required_instrument_name"]
        and row["sia_proctype"] == source["required_proctype"]
        and row["sia_prodtype"] == source["required_prodtype"]
        and row["sia_access_format"] == source["required_access_format"]
        and row["sia_calib_level"] == str(source["required_calib_level"])
        for row in manifest
    )
    band_matches = all(row["filter"] == row["sia_obs_bandpass"] for row in manifest)
    unique_exposures = {row["exposure"] for row in manifest}
    unique_exposure_extensions = {
        (row["exposure"], row["sia_extension"]) for row in manifest
    }
    group_counts = Counter(
        (row["exposure"], row["sia_extension"]) for row in manifest
    )
    gates = {
        "exact_anchor_count": len(anchor_ids) == int(config["gates"]["exact_anchor_count"]),
        "exact_development_anchor_count": sum(
            row["split"] == "development" for row in anchors
        )
        == int(config["gates"]["exact_development_anchor_count"]),
        "exact_validation_anchor_count": sum(
            row["split"] == "validation" for row in anchors
        )
        == int(config["gates"]["exact_validation_anchor_count"]),
        "exact_anchor_measurement_rows": len(manifest) == expected_measurements,
        "every_measurement_has_exactly_one_sia_descriptor": len(manifest)
        == len(measurements),
        "measurement_and_sia_band_match": band_matches,
        "every_descriptor_is_required_calibrated_decam_image": exact_instrument,
        "all_measurements_retained_without_quality_or_exposure_selection": True,
        "raw_metadata_payload_and_query_hashed_for_every_anchor": len(query_records)
        == len(anchor_ids)
        and all(record["raw_metadata_sha256"] for record in query_records),
        "no_image_pixels_or_prohibited_payload_opened": True,
    }
    gates["all_metadata_coverage_gates_pass"] = all(gates.values())
    report_path = ROOT / config["outputs"]["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "report_version": "SIGMA-V19AM-NSC-SIA-ANCHOR-COVERAGE-1.0.0",
        "status": "completed_metadata_coverage_audit",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "parent_hashes": parent_hashes,
        "counts": {
            "anchors": len(anchor_ids),
            "development_anchors": sum(row["split"] == "development" for row in anchors),
            "validation_anchors": sum(row["split"] == "validation" for row in anchors),
            "measurement_descriptor_pairs": len(manifest),
            "unique_exposures": len(unique_exposures),
            "unique_exposure_extensions": len(unique_exposure_extensions),
            "minimum_anchors_per_exposure_extension": min(group_counts.values()),
            "maximum_anchors_per_exposure_extension": max(group_counts.values()),
        },
        "filter_rows": dict(sorted(Counter(row["filter"] for row in manifest).items())),
        "split_rows": dict(sorted(Counter(row["split"] for row in manifest).items())),
        "query_records": query_records,
        "gates": gates,
        "outputs": {
            "manifest": manifest_path.relative_to(ROOT).as_posix(),
            "manifest_sha256": sha256(manifest_path),
        },
        "claim_boundary": config["claim_boundary"],
        "image_pixels_downloaded": False,
        "exposures_ranked_or_selected": False,
        "ambiguous_candidates_queried": False,
        "photometry_mass_or_current_inferred": False,
        "lensing_or_halo_payload_opened": False,
        "gravity_formula_or_parameter_changed": False,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
