#!/usr/bin/env python3
"""Build the metadata-only V19AU ambiguous-candidate image plan."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parents[1]
ANCHOR_IDS = {"06", "07", "14", "16", "21", "22", "23", "24", "26", "29", "37", "57", "66", "71", "78"}
UNIFIED = ROOT / "data" / "derived" / "sigma_v19aa_member_counterpart_association" / "unified_candidates.csv"
POSTERIORS = ROOT / "data" / "derived" / "sigma_v19aa_member_counterpart_association" / "candidate_posteriors.csv"
BRI = ROOT / "data" / "derived" / "sigma_v19z_member_photometry" / "bullet_published_bri.csv"
IMAGES = ROOT / "data" / "derived" / "sigma_v19ar_current_archive_decam_cutouts" / "download_manifest.csv"
OUT = ROOT / "data" / "derived" / "sigma_v19au_ambiguous_candidate_image_measurement"
REPORT = ROOT / "results" / "sigma_v19au_ambiguous_candidate_image_measurement" / "metadata_plan.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    bri_rows = read_csv(BRI)
    eligible_members = {
        row["object_id"]
        for row in bri_rows
        if row["published_bri_available"] == "True" and row["object_id"] not in ANCHOR_IDS
    }
    posterior_rows = [
        row
        for row in read_csv(POSTERIORS)
        if row["cluster"] == "BULLET" and row["object_id"] in eligible_members
    ]
    candidates = {row["candidate_id"]: row for row in read_csv(UNIFIED)}
    candidate_ids = sorted({row["candidate_id"] for row in posterior_rows})
    if any(candidate_id not in candidates for candidate_id in candidate_ids):
        raise RuntimeError("posterior references a missing unified candidate")

    bri_by_member = {row["object_id"]: row for row in bri_rows}
    hypotheses: list[dict[str, Any]] = []
    for row in sorted(posterior_rows, key=lambda item: (item["object_id"], item["candidate_id"])):
        candidate = candidates[row["candidate_id"]]
        bri = bri_by_member[row["object_id"]]
        hypotheses.append(
            {
                "member_id": row["object_id"],
                "candidate_id": row["candidate_id"],
                "candidate_ra_deg": candidate["ra_deg"],
                "candidate_dec_deg": candidate["dec_deg"],
                "b_bessel_mag": bri["b_bessel_mag"],
                "r_bessel_mag": bri["r_bessel_mag"],
                "i_bessel_mag": bri["i_bessel_mag"],
                "angular_separation_arcsec": row["angular_separation_arcsec"],
                "positional_likelihood_ratio": row["likelihood_ratio"],
                "positional_posterior_q_0_90": row["posterior_q_0.90"],
                "dual_survey": row["dual_survey"],
                "repeated_detection_support": row["repeated_detection_support"],
                "probable_foreground_star_diagnostic": row["probable_foreground_star_diagnostic"],
            }
        )

    candidate_sky = SkyCoord(
        [float(candidates[candidate_id]["ra_deg"]) for candidate_id in candidate_ids],
        [float(candidates[candidate_id]["dec_deg"]) for candidate_id in candidate_ids],
        unit="deg",
    )
    plan: list[dict[str, Any]] = []
    coverage: dict[str, Counter[str]] = defaultdict(Counter)
    image_manifest = read_csv(IMAGES)
    for image_row in image_manifest:
        image_path = ROOT / image_row["output_path"]
        if sha256(image_path) != image_row["sha256"]:
            raise RuntimeError(f"image hash changed: {image_row['output_path']}")
        with fits.open(image_path, memmap=True, do_not_scale_image_data=True) as hdul:
            image_hdu = next(
                hdu
                for hdu in hdul
                if hdu.header.get("NAXIS") == 2 and int(hdu.header.get("NAXIS1", 0)) > 0
            )
            wcs = WCS(image_hdu.header).celestial
            xx, yy = wcs.world_to_pixel(candidate_sky)
            inside = (
                (xx >= 0)
                & (xx < int(image_hdu.header["NAXIS1"]))
                & (yy >= 0)
                & (yy < int(image_hdu.header["NAXIS2"]))
            )
        for candidate_id, x_pixel, y_pixel in zip(
            np.asarray(candidate_ids)[inside], xx[inside], yy[inside]
        ):
            candidate = candidates[str(candidate_id)]
            coverage[str(candidate_id)][image_row["filter"]] += 1
            plan.append(
                {
                    "group_id": image_row["group_id"],
                    "exposure": image_row["exposure"],
                    "sia_extension": image_row["sia_extension"],
                    "filter": image_row["filter"],
                    "candidate_id": candidate_id,
                    "candidate_ra_deg": candidate["ra_deg"],
                    "candidate_dec_deg": candidate["dec_deg"],
                    "predicted_x_pixel": f"{x_pixel:.9f}",
                    "predicted_y_pixel": f"{y_pixel:.9f}",
                    "image_path": image_row["output_path"],
                    "image_sha256": image_row["sha256"],
                }
            )

    plan.sort(key=lambda row: (row["group_id"], row["candidate_id"]))
    required_bands = ("g", "r", "i", "z", "Y")
    incomplete = [
        candidate_id
        for candidate_id in candidate_ids
        if not all(coverage[candidate_id][band] > 0 for band in required_bands)
    ]
    if incomplete:
        raise RuntimeError(f"candidate plan lacks grizY coverage: {incomplete[:5]}")

    plan_path = OUT / "candidate_measurement_plan.csv"
    hypothesis_path = OUT / "candidate_hypotheses.csv"
    write_csv(plan_path, plan, list(plan[0]))
    write_csv(hypothesis_path, hypotheses, list(hypotheses[0]))
    report = {
        "protocol_stage": "SIGMA-V19AU-metadata-only-plan",
        "members": len(eligible_members),
        "member_candidate_hypotheses": len(hypotheses),
        "unique_candidates": len(candidate_ids),
        "image_groups_with_candidates": len({row["group_id"] for row in plan}),
        "candidate_exposure_measurements": len(plan),
        "measurements_by_filter": dict(Counter(row["filter"] for row in plan)),
        "all_candidates_complete_grizY": not incomplete,
        "science_pixels_opened_or_interpreted": False,
        "bri_or_positional_values_used_for_image_coverage_selection": False,
        "outputs": {
            "candidate_measurement_plan": plan_path.relative_to(ROOT).as_posix(),
            "candidate_measurement_plan_sha256": sha256(plan_path),
            "candidate_hypotheses": hypothesis_path.relative_to(ROOT).as_posix(),
            "candidate_hypotheses_sha256": sha256(hypothesis_path),
        },
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
