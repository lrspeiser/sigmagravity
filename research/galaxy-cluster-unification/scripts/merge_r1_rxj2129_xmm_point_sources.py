#!/usr/bin/env python3
"""Filter and merge the frozen RX J2129 three-band emldetect catalogs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u


PROJECT = Path(__file__).resolve().parents[1]
X2B = Path("/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/x2b")
DERIVED = PROJECT / "data/derived/r1_rxj2129_xmm_x2"
BANDS = [
    (1, 500, 1200, "detect_band1_500_1200eV"),
    (2, 1200, 2000, "detect_band2_1200_2000eV"),
    (3, 2000, 7000, "detect_band3_2000_7000eV"),
]
ML_MIN = 10.0
EXTENT_MAX_ARCSEC = 6.0
MERGE_RADIUS_ARCSEC = 6.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left != root_right:
            self.parent[max(root_left, root_right)] = min(root_left, root_right)


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, object]] = []
    band_audit: dict[str, object] = {}
    for band_index, elow, ehigh, dirname in BANDS:
        root = X2B / dirname
        catalog = root / "emllist.fits"
        log_path = root / "cheese.log"
        log_text = log_path.read_text(errors="replace")
        emldetect_end = bool(
            re.search(r"emldetect \(emldetect-[^)]+\).* ended:", log_text)
        )
        emldetect_errors = len(
            re.findall(r"^\*\* emldetect: error \(", log_text, flags=re.MULTILINE)
        )
        downstream_unused_mask_error = "** makemask: error (NoInfile)" in log_text
        with fits.open(catalog, memmap=True) as hdus:
            table = hdus["SRCLIST"]
            data = table.data
            summary = data[(data["ID_INST"] == 0) & (data["ID_BAND"] == 0)]
            unique_ids = np.unique(summary["ML_ID_SRC"])
            schema_gate = (
                summary.size > 0
                and unique_ids.size == summary.size
                and all(
                    name in table.columns.names
                    for name in (
                        "ML_ID_SRC",
                        "RA",
                        "DEC",
                        "RADEC_ERR",
                        "EXT",
                        "DET_ML",
                    )
                )
            )
            for row in summary:
                ra = float(row["RA"])
                dec = float(row["DEC"])
                position_error = float(row["RADEC_ERR"])
                extent = float(row["EXT"])
                likelihood = float(row["DET_ML"])
                finite = all(
                    math.isfinite(value)
                    for value in (ra, dec, position_error, extent, likelihood)
                )
                retained = (
                    finite
                    and likelihood >= ML_MIN
                    and 0.0 <= extent <= EXTENT_MAX_ARCSEC
                )
                candidates.append(
                    {
                        "band_index": band_index,
                        "elow_eV": elow,
                        "ehigh_eV": ehigh,
                        "ML_ID_SRC": int(row["ML_ID_SRC"]),
                        "ra_deg": ra,
                        "dec_deg": dec,
                        "position_error_arcsec": position_error,
                        "extent_arcsec": extent,
                        "detection_likelihood": likelihood,
                        "finite": finite,
                        "retained": retained,
                        "source_catalog": str(catalog),
                    }
                )
        band_audit[f"band{band_index}"] = {
            "energy_eV": [elow, ehigh],
            "catalog": str(catalog),
            "catalog_sha256": sha256(catalog),
            "summary_rows": int(summary.size),
            "retained_rows": sum(
                bool(item["retained"])
                for item in candidates
                if item["band_index"] == band_index
            ),
            "emldetect_end_record_present": emldetect_end,
            "emldetect_error_records": emldetect_errors,
            "catalog_schema_gate_passed": schema_gate,
            "downstream_unused_cheese_makemask_NoInfile_recorded": downstream_unused_mask_error,
            "gate_passed": emldetect_end and emldetect_errors == 0 and schema_gate,
        }

    candidate_path = DERIVED / "point_source_candidates.csv"
    fields = list(candidates[0])
    with candidate_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidates)

    retained = [item for item in candidates if item["retained"]]
    retained.sort(
        key=lambda item: (
            float(item["ra_deg"]),
            float(item["dec_deg"]),
            -float(item["detection_likelihood"]),
            int(item["band_index"]),
        )
    )
    coordinates = SkyCoord(
        ra=[float(item["ra_deg"]) for item in retained] * u.deg,
        dec=[float(item["dec_deg"]) for item in retained] * u.deg,
    )
    union = UnionFind(len(retained))
    for left in range(len(retained)):
        separations = coordinates[left].separation(coordinates[left + 1 :]).arcsec
        for offset in np.flatnonzero(separations <= MERGE_RADIUS_ARCSEC):
            union.union(left, left + 1 + int(offset))

    components: dict[int, list[int]] = {}
    for index in range(len(retained)):
        components.setdefault(union.find(index), []).append(index)

    merged: list[dict[str, object]] = []
    for indices in components.values():
        members = [retained[index] for index in indices]
        valid_weight = np.asarray(
            [
                1.0 / float(item["position_error_arcsec"]) ** 2
                if float(item["position_error_arcsec"]) > 0
                and math.isfinite(float(item["position_error_arcsec"]))
                else 0.0
                for item in members
            ]
        )
        if valid_weight.sum() > 0:
            ra = float(
                np.average([float(item["ra_deg"]) for item in members], weights=valid_weight)
            )
            dec = float(
                np.average([float(item["dec_deg"]) for item in members], weights=valid_weight)
            )
            error = float(math.sqrt(1.0 / valid_weight.sum()))
        else:
            best = max(members, key=lambda item: float(item["detection_likelihood"]))
            ra, dec = float(best["ra_deg"]), float(best["dec_deg"])
            error = float("nan")
        merged.append(
            {
                "source_id": 0,
                "ra_deg": ra,
                "dec_deg": dec,
                "position_error_arcsec": error,
                "maximum_detection_likelihood": max(
                    float(item["detection_likelihood"]) for item in members
                ),
                "maximum_fitted_extent_arcsec": max(
                    float(item["extent_arcsec"]) for item in members
                ),
                "detected_band_indices": ";".join(
                    str(value)
                    for value in sorted({int(item["band_index"]) for item in members})
                ),
                "detection_members": len(members),
                "PSF_mask_status": "pending",
            }
        )
    merged.sort(key=lambda item: (float(item["ra_deg"]), float(item["dec_deg"])))
    for source_id, item in enumerate(merged, start=1):
        item["source_id"] = source_id

    pre_psf_path = DERIVED / "point_source_catalog_pre_psf.csv"
    with pre_psf_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(merged[0]))
        writer.writeheader()
        writer.writerows(merged)

    manifest = {
        "version": "R1B3-RXJ2129-XMM-X2b1-catalog-0.2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "configs/r1_rxj2129_xmm_background_mask_protocol.json",
        "band_audit": band_audit,
        "candidate_summary_rows": len(candidates),
        "candidate_rows_passing_frozen_likelihood_extent_filter": len(retained),
        "merged_point_sources_pending_PSF": len(merged),
        "merge_radius_arcsec": MERGE_RADIUS_ARCSEC,
        "candidate_catalog": str(candidate_path.relative_to(PROJECT)),
        "pre_PSF_catalog": str(pre_psf_path.relative_to(PROJECT)),
        "manual_edits": False,
        "gates": {
            "all_three_emldetect_catalog_gates_passed": all(
                item["gate_passed"] for item in band_audit.values()
            ),
            "frozen_catalog_filter_and_merge_completed": bool(merged),
            "all_PSF_radii_completed": False,
            "immutable_point_source_mask_frozen": False,
            "X2b1_gate_passed": False,
            "full_X2_gate_passed": False,
        },
        "authorization": {
            "run_frozen_PSF_radius_stage": bool(merged),
            "run_background_before_mask_gate": False,
            "fit_temperature_or_density": False,
            "fit_new_force_or_action": False,
        },
    }
    manifest_path = DERIVED / "point_source_mask_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    if not all(item["gate_passed"] for item in band_audit.values()) or not merged:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
