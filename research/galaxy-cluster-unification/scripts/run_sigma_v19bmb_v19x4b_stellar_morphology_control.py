#!/usr/bin/env python3
"""Build the unchanged V19BM stellar control on terminal V19X4B grids."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_sigma_v19bm_stellar_morphology_control as inherited_v19bm

ROOT = Path(__file__).resolve().parents[1]
FROZEN_STATE = "frozen_after_terminal_v19x4b_pass"
AUTHORIZED_X4B_STATUS = (
    "gas_state_posterior_and_common_grids_passed_source_invariant_scoring_authorized"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return inherited_v19bm.sha256(path)


def validate_static(config: dict[str, Any]) -> None:
    if config.get("freeze_state") != FROZEN_STATE:
        raise RuntimeError("V19BMB is not frozen after terminal V19X4B")
    for name, spec in config["parents"].items():
        path = ROOT / spec["path"]
        if not path.is_file() or sha256(path) != spec["sha256"]:
            raise RuntimeError(f"V19BMB parent changed: {name}")
    implementation = config["implementation"]
    runner = ROOT / implementation["runner"]
    if runner.resolve() != Path(__file__).resolve():
        raise RuntimeError("V19BMB configuration names another runner")
    for name in ("runner", "inherited_v19bm_runner", "stellar_module"):
        path = ROOT / implementation[name]
        if not path.is_file() or sha256(path) != implementation[f"{name}_sha256"]:
            raise RuntimeError(f"V19BMB implementation changed: {name}")


def validate_x4b_report(
    config: dict[str, Any], report_path: Path
) -> dict[str, Any]:
    report = load_json(report_path)
    if (
        report.get("status") != AUTHORIZED_X4B_STATUS
        or report.get("config_sha256")
        != config["parents"]["v19x4b_config"]["sha256"]
        or report.get("runner_sha256")
        != config["parents"]["v19x4b_runner"]["sha256"]
        or report.get("source_invariant_scoring_authorized") is not True
        or not report.get("gates")
        or not all(report["gates"].values())
        or report.get("lensing_or_halo_payload_opened") is not False
        or len(report.get("products", [])) != 12
    ):
        raise RuntimeError("V19BMB requires a passing target-sealed V19X4B report")
    for product in report["products"]:
        path = ROOT / product["relative_path"]
        if (
            not path.is_file()
            or path.stat().st_size != int(product["bytes"])
            or sha256(path) != product["sha256"]
        ):
            raise RuntimeError(f"V19BMB changed V19X4B product: {path}")
    return report


def execute(config: dict[str, Any], x4b_report_path: Path) -> dict[str, Any]:
    validate_static(config)
    x4b_report = validate_x4b_report(config, x4b_report_path)
    grids = inherited_v19bm.common_grid_products(config, x4b_report)
    source_report = load_json(ROOT / config["parents"]["source_map_report"]["path"])
    centers = {row["cluster"]: row["final_center"] for row in source_report["clusters"]}
    construction = config["construction"]
    draws = int(construction["draws"])
    batch_size = int(construction["batch_size"])
    output_root = ROOT / config["outputs"]["root"]
    products: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for cluster, spec in config["clusters"].items():
        east_axis, north_axis, labels = grids[cluster]
        if not np.array_equal(east_axis, north_axis):
            raise RuntimeError(f"{cluster} common axes differ")
        region_ids = np.unique(labels[labels >= 0])
        if region_ids.size != int(spec["expected_regions"]):
            raise RuntimeError(f"{cluster} common-grid region count changed")
        with fits.open(ROOT / spec["analysis_grid"], memmap=False) as handle:
            wcs = WCS(handle[0].header)
        accumulated: dict[str, list[np.ndarray]] = {
            f"light_mean_{fwhm:g}kpc": []
            for fwhm in construction["smoothing_fwhm_kpc"]
        }
        accumulated.update(
            {
                f"light_percentile_rank_{fwhm:g}kpc": []
                for fwhm in construction["smoothing_fwhm_kpc"]
            }
        )
        ids: list[np.ndarray] = []
        for sample_ids, maps in inherited_v19bm.member_map_batches(
            ROOT / spec["ensemble"],
            cluster=cluster,
            spec=spec,
            wcs=wcs,
            center=centers[cluster],
            common_axis=east_axis,
            draws=draws,
            batch_size=batch_size,
            output_pixel_arcsec=float(construction["output_pixel_arcsec"]),
        ):
            ids.append(sample_ids)
            for fwhm in construction["smoothing_fwhm_kpc"]:
                sigma_pixels = (
                    float(fwhm)
                    / (2.0 * math.sqrt(2.0 * math.log(2.0)))
                    / float(construction["common_axis_kpc"]["spacing"])
                )
                smoothed = inherited_v19bm.smooth_light_draws(
                    maps, sigma_pixels=sigma_pixels
                )
                means, ranks = inherited_v19bm.region_light_percentile_ranks(
                    smoothed, labels, region_ids
                )
                accumulated[f"light_mean_{fwhm:g}kpc"].append(
                    means.astype(np.float32)
                )
                accumulated[f"light_percentile_rank_{fwhm:g}kpc"].append(ranks)
        arrays: dict[str, Any] = {
            "sample_id": np.concatenate(ids),
            "bin_id": region_ids,
        }
        arrays.update({key: np.concatenate(value) for key, value in accumulated.items()})
        output = output_root / cluster / "stellar_morphology_control.npz"
        inherited_v19bm.atomic_npz(output, arrays)
        rank_keys = [key for key in arrays if key.startswith("light_percentile_rank_")]
        summaries.append(
            {
                "cluster": cluster,
                "draws": int(arrays["sample_id"].size),
                "regions": int(region_ids.size),
                "rank_minimum": min(float(np.min(arrays[key])) for key in rank_keys),
                "rank_maximum": max(float(np.max(arrays[key])) for key in rank_keys),
                "maximum_mean_rank_error_from_half": max(
                    float(np.max(np.abs(np.mean(arrays[key], axis=1) - 0.5)))
                    for key in rank_keys
                ),
            }
        )
        products.append(
            {
                "cluster": cluster,
                "role": "stellar_morphology_control",
                "relative_path": output.relative_to(ROOT).as_posix(),
                "bytes": output.stat().st_size,
                "sha256": sha256(output),
            }
        )
    gates = {
        "both_clusters_exact_draw_and_region_counts": all(
            row["draws"] == draws
            and row["regions"]
            == int(config["clusters"][row["cluster"]]["expected_regions"])
            for row in summaries
        ),
        "all_region_ranks_strictly_between_zero_and_one": all(
            row["rank_minimum"] > 0.0 and row["rank_maximum"] < 1.0
            for row in summaries
        ),
        "mean_region_rank_each_draw_equals_half_to_1e_12": all(
            row["maximum_mean_rank_error_from_half"] <= 1.0e-12
            for row in summaries
        ),
        "two_hash_bound_products": len(products) == 2,
        "cross_filter_amplitudes_not_compared": True,
        "lensing_halo_action_and_gravity_payload_not_opened": True,
    }
    return {
        "status": (
            "stellar_morphology_control_passed_invariant_scoring_ready"
            if all(gates.values())
            else "stellar_morphology_control_failed_closed"
        ),
        "x4b_report_sha256": sha256(x4b_report_path),
        "cluster_summaries": summaries,
        "products": products,
        "gates": gates,
        "invariant_scoring_ready": all(gates.values()),
        "cross_filter_luminosity_amplitudes_compared": False,
        "stellar_mass_inferred": False,
        "lensing_halo_action_or_gravity_payload_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--x4b-report", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    try:
        result = execute(config, args.x4b_report.resolve())
    except Exception as exc:  # noqa: BLE001 - preserve terminal failure evidence
        result = {
            "status": "v19bmb_stellar_morphology_execution_failed_closed",
            "execution_exception": f"{type(exc).__name__}: {exc}",
            "cluster_summaries": [],
            "products": [],
            "gates": {"execution_completed": False},
            "invariant_scoring_ready": False,
            "cross_filter_luminosity_amplitudes_compared": False,
            "stellar_mass_inferred": False,
            "lensing_halo_action_or_gravity_payload_opened": False,
        }
    report = {
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        **result,
        "claim_boundary": config["claim_boundary"],
    }
    output = ROOT / config["outputs"]["terminal_report"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(report["status"])
    if report["status"] == "v19bmb_stellar_morphology_execution_failed_closed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
