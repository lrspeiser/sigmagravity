#!/usr/bin/env python3
"""Audit whether public AS295 spectroscopy satisfies the frozen stage-B gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v18a_collisionless_stress_readiness.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v18a_collisionless_stress_readiness"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "SigmaGravity/18A public-catalog readiness audit"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
        if response.status != 200 or not payload:
            raise RuntimeError(f"catalog request failed with HTTP {response.status}")
    return payload


def acquire_inputs(config: dict[str, Any], refresh: bool) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    sources = {
        **config["public_inputs"],
        **config["coverage_inputs"],
        **config.get("photometric_only_inputs", {}),
    }
    for name, source in sources.items():
        path = ROOT / source["raw_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if refresh or not path.is_file():
            payload = fetch(source["url"])
            if b"VizieR Astronomical Server" not in payload:
                raise RuntimeError(f"unexpected VizieR response for {name}")
            path.write_bytes(payload)
        paths[name] = path
    return paths


def noncomment_lines(path: Path) -> list[str]:
    return [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]


def parse_vizier_tsv(path: Path) -> list[dict[str, str]]:
    lines = noncomment_lines(path)
    if len(lines) < 4:
        raise RuntimeError(f"no tabular data in {path}")
    header = lines[0]
    rows = list(csv.DictReader(io.StringIO("\n".join([header, *lines[3:]])), delimiter="\t"))
    return [{key: value.strip() for key, value in row.items()} for row in rows]


def vizier_row_count(path: Path) -> int:
    lines = noncomment_lines(path)
    if not lines:
        return 0
    if len(lines) < 3:
        raise RuntimeError(f"malformed VizieR table in {path}")
    return len(lines) - 3


def vizier_schema_columns(path: Path) -> list[dict[str, str]]:
    columns = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("#Column\t"):
            continue
        fields = line.split("\t")
        if len(fields) < 4:
            raise RuntimeError(f"malformed VizieR column declaration in {path}")
        description = fields[3]
        metadata = "\t".join(fields[3:])
        ucd = ""
        if "[ucd=" in metadata:
            ucd = metadata.split("[ucd=", 1)[1].split("]", 1)[0]
        columns.append({"name": fields[1], "description": description, "ucd": ucd})
    if not columns:
        raise RuntimeError(f"no VizieR schema columns in {path}")
    return columns


def audit_photometric_catalog(
    path: Path, source: dict[str, Any]
) -> dict[str, Any]:
    columns = vizier_schema_columns(path)
    redshift_columns = [
        column
        for column in columns
        if "redshift" in column["description"].lower()
        or "redshift" in column["ucd"].lower()
    ]
    required_ucd = source["required_redshift_ucd"]
    if [column["ucd"] for column in redshift_columns] != [required_ucd]:
        raise RuntimeError("MGCLS redshift schema changed; re-audit before use")
    if any("spect" in column["description"].lower() for column in redshift_columns):
        raise RuntimeError("MGCLS now declares a spectroscopic redshift column")
    return {
        "catalog": source["catalog"],
        "rows": vizier_row_count(path),
        "schema_column_count": len(columns),
        "redshift_columns": redshift_columns,
        "has_spectroscopic_redshift": False,
        "usable_as_velocity_measurement": False,
    }


def angular_separation_arcsec(a: dict[str, str], b: dict[str, str]) -> float:
    ra1 = math.radians(float(a["RAJ2000"]))
    dec1 = math.radians(float(a["DEJ2000"]))
    ra2 = math.radians(float(b["RAJ2000"]))
    dec2 = math.radians(float(b["DEJ2000"]))
    cosine = (
        math.sin(dec1) * math.sin(dec2)
        + math.cos(dec1) * math.cos(dec2) * math.cos(ra1 - ra2)
    )
    angle = math.acos(max(-1.0, min(1.0, cosine)))
    return math.degrees(angle) * 3600.0


def one_to_one_matches(
    left: list[dict[str, str]],
    right: list[dict[str, str]],
    radius_arcsec: float,
) -> list[dict[str, Any]]:
    candidates = []
    for left_index, left_row in enumerate(left):
        for right_index, right_row in enumerate(right):
            separation = angular_separation_arcsec(left_row, right_row)
            if separation <= radius_arcsec:
                candidates.append((separation, left_index, right_index))
    used_left: set[int] = set()
    used_right: set[int] = set()
    matches = []
    for separation, left_index, right_index in sorted(candidates):
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        matches.append(
            {
                "ruel_index": left_index,
                "bayliss_index": right_index,
                "ruel_galaxy": left[left_index].get("Galaxy", ""),
                "bayliss_galaxy": right[right_index].get("Gal", ""),
                "separation_arcsec": separation,
                "absolute_redshift_difference": abs(
                    float(left[left_index]["z"]) - float(right[right_index]["z"])
                ),
            }
        )
    return matches


def rest_frame_velocity_km_s(z: float, cluster_z: float, c_km_s: float) -> float:
    return c_km_s * (z - cluster_z) / (1.0 + cluster_z)


def member_indices(
    rows: list[dict[str, str]],
    cluster_z: float,
    c_km_s: float,
    window_km_s: float,
    center: dict[str, str],
    kpc_per_arcsec: float,
    aperture_kpc: float,
) -> set[int]:
    return {
        index
        for index, row in enumerate(rows)
        if abs(rest_frame_velocity_km_s(float(row["z"]), cluster_z, c_km_s))
        <= window_km_s
        and angular_separation_arcsec(row, center) * kpc_per_arcsec
        <= aperture_kpc
    }


def require_one(rows: list[dict[str, str]], label: str) -> dict[str, str]:
    if len(rows) != 1:
        raise RuntimeError(f"expected one {label} row, found {len(rows)}")
    return rows[0]


def run(config_path: Path, output: Path, refresh: bool) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not config["audit_rules"]["minimum_is_inherited_unchanged"]:
        raise RuntimeError("the inherited member threshold is not locked")
    paths = acquire_inputs(config, refresh)
    ruel = parse_vizier_tsv(paths["ruel_2014_galaxies"])
    bayliss = parse_vizier_tsv(paths["bayliss_2016_galaxies"])
    ruel_summary = require_one(
        parse_vizier_tsv(paths["ruel_2014_summary"]), "Ruel summary"
    )
    bayliss_summary = require_one(
        parse_vizier_tsv(paths["bayliss_2016_summary"]), "Bayliss summary"
    )
    expected_name = config["audit_rules"]["cluster_name_must_equal"]
    for row in [*ruel, *bayliss, ruel_summary, bayliss_summary]:
        if row["SPT-CL"] != expected_name:
            raise RuntimeError(f"unexpected cluster row {row['SPT-CL']}")

    radius = float(config["audit_rules"]["catalog_crossmatch_radius_arcsec"])
    matches = one_to_one_matches(ruel, bayliss, radius)
    cluster_z = float(config["cluster_identity"]["redshift"])
    c_km_s = float(config["audit_rules"]["speed_of_light_km_s"])
    window = float(config["audit_rules"]["member_velocity_window_km_s"])
    center = {
        "RAJ2000": str(config["cluster_identity"]["center_ra_deg"]),
        "DEJ2000": str(config["cluster_identity"]["center_dec_deg"]),
    }
    kpc_per_arcsec = float(config["cluster_identity"]["kpc_per_arcsec_Planck18"])
    aperture_kpc = float(config["audit_rules"]["member_projected_aperture_kpc"])
    ruel_members = member_indices(
        ruel,
        cluster_z,
        c_km_s,
        window,
        center,
        kpc_per_arcsec,
        aperture_kpc,
    )
    bayliss_members = member_indices(
        bayliss,
        cluster_z,
        c_km_s,
        window,
        center,
        kpc_per_arcsec,
        aperture_kpc,
    )
    matched_ruel_members = {
        row["ruel_index"]
        for row in matches
        if row["ruel_index"] in ruel_members
        and row["bayliss_index"] in bayliss_members
    }
    unique_member_count = len(bayliss_members) + len(ruel_members - matched_ruel_members)
    minimum = int(config["audit_rules"]["minimum_unique_secure_members"])

    prerequisites = {
        "secure_member_positions": unique_member_count > 0,
        "secure_member_spectroscopic_redshifts": unique_member_count > 0,
        "photometry_tied_stellar_mass_weights": False,
        "minimum_secure_members_inside_1_8_Mpc": unique_member_count >= minimum,
        "declared_redshift_quality_selection": True,
        "universal_spatial_kernel_and_membership_rule": False,
    }
    stage_b_authorized = all(prerequisites.values())
    coverage = {
        name: {
            "catalog": config["coverage_inputs"][name]["catalog"],
            "matching_rows": vizier_row_count(paths[name]),
        }
        for name in config["coverage_inputs"]
    }
    if any(item["matching_rows"] != 0 for item in coverage.values()):
        raise RuntimeError("an ACT coverage query now contains rows; re-audit before use")
    photometric_only = {
        name: audit_photometric_catalog(paths[name], source)
        for name, source in config.get("photometric_only_inputs", {}).items()
    }
    if any(item["rows"] == 0 for item in photometric_only.values()):
        raise RuntimeError("a declared photometric coverage catalog is empty")
    if stage_b_authorized:
        decision = (
            "AS295 spectroscopy is source-ready; stage B still requires the unchanged "
            "matched two-cluster authorization and target-blind source construction"
        )
    else:
        decision = (
            "AS295 remains below the frozen matched collisionless-stress gate; do not "
            "double-count catalogs, lower the member threshold, or run PLCKG287 alone"
        )

    report = {
        "status": "completed Sigma v18A AS295 collisionless-stress readiness audit",
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "input_hashes": {
            "config": sha256(config_path),
            "parent_gate": sha256(ROOT / config["parent_gate"]),
            **{name: sha256(path) for name, path in paths.items()},
        },
        "catalogs": {
            "ruel_2014": {
                "spectra_rows": len(ruel),
                "published_member_count": int(ruel_summary["N"]),
                "fixed_velocity_window_member_count": len(ruel_members),
            },
            "bayliss_2016": {
                "spectra_rows": len(bayliss),
                "published_spectra_count": int(bayliss_summary["Nsp"]),
                "published_member_count": int(bayliss_summary["NMm"]),
                "fixed_velocity_window_member_count": len(bayliss_members),
            },
        },
        "independent_act_catalog_coverage": coverage,
        "photometric_only_catalog_coverage": photometric_only,
        "deduplication": {
            "radius_arcsec": radius,
            "member_projected_aperture_kpc": aperture_kpc,
            "matched_spectra": len(matches),
            "unmatched_ruel_spectra": len(ruel) - len(matches),
            "unmatched_bayliss_spectra": len(bayliss) - len(matches),
            "combined_unique_spectra": len(ruel) + len(bayliss) - len(matches),
            "combined_unique_fixed_window_members": unique_member_count,
            "matches": matches,
        },
        "frozen_member_requirement": minimum,
        "member_shortfall": max(0, minimum - unique_member_count),
        "stage_b_prerequisites": prerequisites,
        "stage_b_source_construction_authorized": stage_b_authorized,
        "formula_or_kernel_selection_authorized": False,
        "lensing_target_opened": False,
        "decision": decision,
        "claim_boundary": [
            "The fixed velocity window is a transparent readiness count, not a replacement for the papers' membership algorithms.",
            "The same galaxies appearing in two releases do not provide independent phase-space information.",
            "A source-readiness pass would not be evidence for a modified-gravity formula.",
            "MGCLS supplies optical/radio morphology and photometric redshifts, not the line-of-sight velocities required by the frozen collisionless-stress source.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="redownload all public VizieR query products before auditing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args.config.resolve(), args.output.resolve(), args.refresh)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
