#!/usr/bin/env python3
"""Summarize the frozen V19AX acquisition after its positive-weight gate failed."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

SCRIPT_DIR = Path(__file__).resolve().parent
ACQUISITION_PATH = SCRIPT_DIR / "acquire_sigma_v19ax_delve_dr3_coadds.py"
SPEC = importlib.util.spec_from_file_location("sigma_v19ax_acquisition", ACQUISITION_PATH)
ACQ = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ACQ)

ROOT = ACQ.ROOT
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19ax_delve_dr3_coadd_acquisition.json"
V19AV_STACKS = ROOT / "data" / "derived" / "sigma_v19av_signed_flux_stack" / "candidate_stacks.csv"
V19AV_STACKS_SHA256 = "271772b5a68652bf5ffec0c4ffefbed37d3f40bbadc35125c5ba0ae5a156712e"
MANIFEST_COLUMNS = ACQ.PRODUCT_COLUMNS + [
    "candidate_positive_weight_centers",
    "global_positive_weight_gate",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hypotheses(config: dict[str, Any]) -> tuple[list[str], SkyCoord, dict[str, set[str]]]:
    path = ROOT / config["inputs"]["candidate_hypotheses"]
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    positions: dict[str, tuple[str, str]] = {}
    member_candidates: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        position = (row["candidate_ra_deg"], row["candidate_dec_deg"])
        previous = positions.setdefault(row["candidate_id"], position)
        if previous != position:
            raise RuntimeError(f"candidate coordinates disagree: {row['candidate_id']}")
        member_candidates[row["member_id"]].add(row["candidate_id"])
    candidate_ids = list(positions)
    sky = SkyCoord(
        [float(positions[key][0]) for key in candidate_ids],
        [float(positions[key][1]) for key in candidate_ids],
        unit="deg",
    )
    return candidate_ids, sky, member_candidates


def inspect(
    selected: list[dict[str, Any]],
    config: dict[str, Any],
    candidate_ids: list[str],
    candidates: SkyCoord,
) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    output_root = ROOT / config["outputs"]["coadd_directory"]
    rows: list[dict[str, Any]] = []
    support_by_band: dict[str, set[str]] = {}
    for selected_row in selected:
        path = output_root / f"{selected_row['band']}_{selected_row['product']}.fits"
        with fits.open(path, memmap=True) as hdul:
            data = np.asarray(hdul[0].data)
            wcs = WCS(hdul[0].header).celestial
            height, width = data.shape
            x, y = wcs.world_to_pixel(candidates)
            inside = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
            xi = np.clip(np.rint(x).astype(int), 0, width - 1)
            yi = np.clip(np.rint(y).astype(int), 0, height - 1)
            finite = np.isfinite(data)
            positive = (data > 0) & finite
            scales = np.abs(proj_plane_pixel_scales(wcs) * 3600.0)
            candidate_positive = ""
            positive_gate: bool | str = ""
            if selected_row["product"] == "weight":
                center_support = inside & positive[yi, xi]
                support_by_band[selected_row["band"]] = {
                    key for key, supported in zip(candidate_ids, center_support) if supported
                }
                candidate_positive = int(np.sum(center_support))
                positive_gate = float(np.mean(positive)) >= float(
                    config["gates"]["minimum_positive_weight_fraction"]
                )
            rows.append(
                {
                    **selected_row,
                    "output_path": path.relative_to(ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "height_pixels": height,
                    "width_pixels": width,
                    "pixel_scale_arcsec": float(np.mean(scales)),
                    "finite_fraction": float(np.mean(finite)),
                    "positive_fraction": float(np.mean(positive)),
                    "candidate_positions_inside": int(np.sum(inside)),
                    "candidate_positive_weight_centers": candidate_positive,
                    "global_positive_weight_gate": positive_gate,
                }
            )
    return rows, support_by_band


def prior_stack_support() -> set[str]:
    if sha256(V19AV_STACKS) != V19AV_STACKS_SHA256:
        raise RuntimeError("V19AV candidate stacks changed")
    support: dict[str, set[str]] = defaultdict(set)
    with V19AV_STACKS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if float(row["stacked_signal_to_noise"]) >= 3.0:
                support[row["candidate_id"]].add(row["filter"])
    return {key for key, bands in support.items() if set("griz") <= bands}


def run(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runner = ROOT / config["implementation"]["runner"]
    if sha256(runner) != config["implementation"]["runner_sha256"]:
        raise RuntimeError("frozen V19AX acquisition runner changed")
    metadata_path = ROOT / config["outputs"]["sia_response"]
    query_path = ROOT / config["outputs"]["sia_query_url"]
    if query_path.read_text(encoding="utf-8").strip() != ACQ.sia_query_url(config):
        raise RuntimeError("frozen SIA query changed")
    payload = metadata_path.read_bytes()
    selected = ACQ.select_products(payload, config)
    candidate_ids, candidates, member_candidates = hypotheses(config)
    rows, support_by_band = inspect(selected, config, candidate_ids, candidates)

    complete_support = set(candidate_ids)
    for band in config["selection"]["bands"]:
        complete_support &= support_by_band[band]
    members_supported = sum(
        bool(candidate_set & complete_support) for candidate_set in member_candidates.values()
    )
    single_exposure_support = prior_stack_support()
    optimistic_union = complete_support | single_exposure_support
    union_members_supported = sum(
        bool(candidate_set & optimistic_union)
        for candidate_set in member_candidates.values()
    )
    shapes = {(row["height_pixels"], row["width_pixels"]) for row in rows}
    weight_rows = [row for row in rows if row["product"] == "weight"]
    gate_results = {
        "exact_sia_rows": True,
        "exact_products": len(rows) == int(config["gates"]["exact_products"]),
        "common_shape": len(shapes) == 1,
        "all_candidates_inside_every_product": all(
            row["candidate_positions_inside"] == int(config["gates"]["exact_candidates"])
            for row in rows
        ),
        "minimum_positive_weight_fraction_every_band": all(
            row["global_positive_weight_gate"] for row in weight_rows
        ),
        "no_source_photometry_or_association": True,
    }
    manifest_path = ROOT / config["outputs"]["product_manifest"]
    ACQ.write_csv(manifest_path, rows, MANIFEST_COLUMNS)
    report = {
        "protocol_version": config["protocol_version"],
        "decision": "failed_closed",
        "failure": "The frozen acquisition runner stopped when the g-band weight plane had less than 99% positive support.",
        "sia": {
            "rows": int(config["gates"]["exact_sia_rows"]),
            "response_sha256": sha256(metadata_path),
            "query_url_sha256": sha256(query_path),
        },
        "products": {
            "count": len(rows),
            "shapes": [list(shape) for shape in sorted(shapes)],
            "bytes": sum(row["bytes"] for row in rows),
            "manifest": manifest_path.relative_to(ROOT).as_posix(),
            "manifest_sha256": sha256(manifest_path),
        },
        "weight_support": {
            "global_positive_fraction_by_band": {
                row["band"]: row["positive_fraction"] for row in weight_rows
            },
            "candidate_centers_with_positive_weight_by_band": {
                band: len(support_by_band[band]) for band in config["selection"]["bands"]
            },
            "candidate_centers_with_positive_weight_all_griz": len(complete_support),
            "maximum_possible_complete_candidate_fraction": len(complete_support)
            / len(candidate_ids),
            "members_with_at_least_one_all_griz_supported_candidate": members_supported,
        },
        "post_failure_optimistic_union": {
            "definition": "Candidate-center positive coadd weight in all griz OR prior V19AV signed-stack SNR >= 3 in all griz. This is an upper-bound support diagnostic, not a combined flux measurement.",
            "v19av_candidate_stacks_sha256": V19AV_STACKS_SHA256,
            "prior_signed_stack_complete_candidates": len(single_exposure_support),
            "coadd_and_prior_stack_intersection": len(
                complete_support & single_exposure_support
            ),
            "union_candidates": len(optimistic_union),
            "union_fraction": len(optimistic_union) / len(candidate_ids),
            "members_with_at_least_one_union_candidate": union_members_supported,
        },
        "gate_results": gate_results,
        "source_photometry_or_candidate_association_computed": False,
        "post_failure_summarizer": Path(__file__).relative_to(ROOT).as_posix(),
        "post_failure_summarizer_sha256": sha256(Path(__file__)),
        "claim_boundary": config["claim_boundary"],
    }
    report_path = ROOT / config["outputs"]["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
