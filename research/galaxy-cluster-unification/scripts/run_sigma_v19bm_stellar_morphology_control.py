#!/usr/bin/env python3
"""Build the V19BM filter-invariant stellar morphology nuisance control."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from voidscreen.sigma_stellar_control import (
    cloud_in_cell_light_map,
    logical_pixels_to_common_kpc,
    region_light_percentile_ranks,
    smooth_light_draws,
)

DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19bm_stellar_morphology_control.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_npz(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def validate_static(config: dict[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, spec in config["parents"].items():
        path = ROOT / spec["path"]
        actual = sha256(path)
        if actual != spec["sha256"]:
            raise RuntimeError(f"V19BM parent changed: {name}")
        hashes[name] = actual
    implementation = config["implementation"]
    runner = ROOT / implementation["runner"]
    module = ROOT / implementation["module"]
    if runner.resolve() != Path(__file__).resolve():
        raise RuntimeError("V19BM configuration names another runner")
    if sha256(runner) != implementation["runner_sha256"]:
        raise RuntimeError("V19BM runner changed after freeze")
    if sha256(module) != implementation["module_sha256"]:
        raise RuntimeError("V19BM module changed after freeze")
    for cluster, spec in config["clusters"].items():
        for role, hash_role in (
            ("ensemble", "ensemble_sha256"),
            ("analysis_grid", "analysis_grid_sha256"),
        ):
            path = ROOT / spec[role]
            actual = sha256(path)
            if actual != spec[hash_role]:
                raise RuntimeError(f"V19BM {cluster} {role} changed")
            hashes[f"{cluster}_{role}"] = actual
    return hashes


def build_preflight_report(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_json(config_path)
    hashes = validate_static(config)
    construction = config["construction"]
    axis = construction["common_axis_kpc"]
    v19bl = load_json(ROOT / config["parents"]["v19bl_config"]["path"])
    checks = {
        "all_static_parent_input_and_implementation_hashes_exact": bool(hashes),
        "exact_4096_draw_pairing": construction["draws"] == 4096,
        "common_grid_matches_v19x4": (
            axis == {"minimum": -1200.0, "maximum": 1200.0, "spacing": 10.0, "cells": 241}
            and construction["smoothing_fwhm_kpc"] == [50.0, 100.0]
        ),
        "rank_control_matches_v19bl": (
            "within-cluster stellar-light percentile rank"
            in v19bl["density_novelty_control"]["five_base_predictors"]
        ),
        "filter_amplitudes_and_stellar_mass_forbidden": (
            not config["authorization"]["compare_cross_filter_amplitudes"]
            and not config["authorization"]["infer_stellar_mass"]
        ),
        "terminal_run_waits_for_v19x4": (
            config["authorization"]["run_terminal_after_v19x4"]
            and not config["authorization"]["run_terminal_before_v19x4"]
        ),
        "lensing_halo_action_and_holdout_sealed": (
            not config["authorization"]["read_lensing_or_halo_payload"]
            and not config["authorization"]["select_action_or_gravity_parameter"]
            and not config["authorization"]["open_holdout"]
        ),
    }
    return {
        "protocol_version": config["protocol_version"],
        "decision": (
            "passed_stellar_control_preflight_awaiting_terminal_v19x4"
            if all(checks.values())
            else "failed_closed"
        ),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "module_sha256": sha256(ROOT / config["implementation"]["module"]),
        "input_hashes": hashes,
        "gates": checks,
        "observed_v19x4_gas_posterior_opened": False,
        "stellar_control_computed": False,
        "cross_filter_luminosity_amplitudes_compared": False,
        "stellar_mass_inferred": False,
        "lensing_halo_action_or_gravity_payload_opened": False,
        "claim_boundary": config["claim_boundary"],
    }


def common_grid_products(
    config: dict[str, Any], x4_report: dict[str, Any]
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    expected_status = config["future_runtime_gates"]["terminal_v19x4_status"]
    if x4_report.get("status") != expected_status or not x4_report.get(
        "source_invariant_scoring_authorized"
    ):
        raise RuntimeError("V19BM requires a passing terminal V19X4 report")
    if not x4_report.get("gates") or not all(x4_report["gates"].values()):
        raise RuntimeError("V19X4 has a failed terminal gate")
    if x4_report.get("lensing_or_halo_payload_opened") is not False:
        raise RuntimeError("V19X4 opened a prohibited target")
    result: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for cluster in config["clusters"]:
        rows = [
            row
            for row in x4_report["products"]
            if row["cluster"] == cluster and row["role"] == "common_grid_summary"
        ]
        if len(rows) != 3:
            raise RuntimeError(f"V19BM expected three {cluster} common-grid branches")
        reference: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        for row in rows:
            path = ROOT / row["relative_path"]
            if sha256(path) != row["sha256"]:
                raise RuntimeError(f"V19BM changed V19X4 product: {path}")
            with np.load(path) as payload:
                current = (
                    np.asarray(payload["axis_east_kpc"], dtype=float),
                    np.asarray(payload["axis_north_kpc"], dtype=float),
                    np.asarray(payload["bin_id"], dtype=np.int64),
                )
            if reference is None:
                reference = current
            elif not all(np.array_equal(left, right) for left, right in zip(reference, current, strict=True)):
                raise RuntimeError(f"{cluster} V19X4 branch grids differ")
        if reference is None:
            raise AssertionError("unreachable empty V19X4 branch inventory")
        result[cluster] = reference
    return result


def member_map_batches(
    path: Path,
    *,
    cluster: str,
    spec: dict[str, Any],
    wcs: WCS,
    center: dict[str, Any],
    common_axis: np.ndarray,
    draws: int,
    batch_size: int,
    output_pixel_arcsec: float,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    active_id: int | None = None
    active_rows: list[dict[str, str]] = []
    sample_ids: list[int] = []
    maps: list[np.ndarray] = []
    reference_members: set[str] | None = None
    completed = 0

    def consume(sample_id: int, rows: list[dict[str, str]]) -> None:
        nonlocal reference_members, completed
        if sample_id != completed:
            raise RuntimeError(f"{cluster} member sample sequence changed at {sample_id}")
        if len(rows) != int(spec["expected_members_per_draw"]):
            raise RuntimeError(f"{cluster} sample {sample_id} member count changed")
        members = {row["member_id"] for row in rows}
        if len(members) != len(rows):
            raise RuntimeError(f"{cluster} sample {sample_id} has duplicate members")
        if reference_members is None:
            reference_members = members
        elif members != reference_members:
            raise RuntimeError(f"{cluster} member inventory changed")
        finite = [row for row in rows if row[spec["luminosity_field"]]]
        light = np.asarray([float(row[spec["luminosity_field"]]) for row in finite])
        if np.any(~np.isfinite(light)) or np.any(light < 0.0) or np.sum(light) <= 0.0:
            raise RuntimeError(f"{cluster} sample {sample_id} has invalid light")
        ra = np.asarray([float(row["ra_deg"]) for row in finite])
        dec = np.asarray([float(row["dec_deg"]) for row in finite])
        x, y = wcs.world_to_pixel_values(ra, dec)
        east, north = logical_pixels_to_common_kpc(
            x,
            y,
            center_logical_x=float(center["logicalx"]),
            center_logical_y=float(center["logicaly"]),
            native_pixel_kpc=output_pixel_arcsec * float(spec["kpc_per_arcsec"]),
        )
        maps.append(cloud_in_cell_light_map(east, north, light, common_axis))
        sample_ids.append(sample_id)
        completed += 1

    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample_id = int(row["sample_id"])
            if sample_id >= draws:
                break
            if active_id is None:
                active_id = sample_id
            if sample_id != active_id:
                consume(active_id, active_rows)
                active_id = sample_id
                active_rows = []
                if len(maps) == batch_size:
                    yield np.asarray(sample_ids, dtype=np.int64), np.stack(maps)
                    sample_ids, maps = [], []
            active_rows.append(row)
        if active_id is not None and active_id < draws:
            consume(active_id, active_rows)
    if maps:
        yield np.asarray(sample_ids, dtype=np.int64), np.stack(maps)
    if completed != draws:
        raise RuntimeError(f"{cluster} produced {completed} rather than {draws} draws")


def execute(config: dict[str, Any], x4_report_path: Path) -> dict[str, Any]:
    validate_static(config)
    x4_report = load_json(x4_report_path)
    if x4_report.get("config_sha256") != config["parents"]["v19x4_config"]["sha256"]:
        raise RuntimeError("terminal V19X4 report names another configuration")
    grids = common_grid_products(config, x4_report)
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
            f"light_mean_{fwhm:g}kpc": [] for fwhm in construction["smoothing_fwhm_kpc"]
        }
        accumulated.update(
            {f"light_percentile_rank_{fwhm:g}kpc": [] for fwhm in construction["smoothing_fwhm_kpc"]}
        )
        ids: list[np.ndarray] = []
        for sample_ids, maps in member_map_batches(
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
                sigma_pixels = float(fwhm) / (2.0 * math.sqrt(2.0 * math.log(2.0))) / float(
                    construction["common_axis_kpc"]["spacing"]
                )
                smoothed = smooth_light_draws(maps, sigma_pixels=sigma_pixels)
                means, ranks = region_light_percentile_ranks(smoothed, labels, region_ids)
                accumulated[f"light_mean_{fwhm:g}kpc"].append(means.astype(np.float32))
                accumulated[f"light_percentile_rank_{fwhm:g}kpc"].append(ranks)
        arrays: dict[str, Any] = {
            "sample_id": np.concatenate(ids),
            "bin_id": region_ids,
        }
        arrays.update({key: np.concatenate(value) for key, value in accumulated.items()})
        output = output_root / cluster / "stellar_morphology_control.npz"
        atomic_npz(output, arrays)
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
            and row["regions"] == int(config["clusters"][row["cluster"]]["expected_regions"])
            for row in summaries
        ),
        "all_region_ranks_strictly_between_zero_and_one": all(
            row["rank_minimum"] > 0.0 and row["rank_maximum"] < 1.0 for row in summaries
        ),
        "mean_region_rank_each_draw_equals_half_to_1e_12": all(
            row["maximum_mean_rank_error_from_half"] <= 1.0e-12 for row in summaries
        ),
        "two_hash_bound_products": len(products) == 2,
        "cross_filter_amplitudes_not_compared": True,
        "lensing_halo_action_and_gravity_payload_not_opened": True,
    }
    return {
        "status": "stellar_morphology_control_passed_invariant_scoring_ready" if all(gates.values()) else "stellar_morphology_control_failed_closed",
        "x4_report_sha256": sha256(x4_report_path),
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
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--x4-report", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    if args.preflight_only:
        report = build_preflight_report(config_path)
        output = ROOT / config["outputs"]["preflight_report"]
    else:
        x4_report = args.x4_report or (ROOT / config["future_v19x4_report"])
        result = execute(config, x4_report.resolve())
        report = {
            "protocol_version": config["protocol_version"],
            "generated_utc": datetime.now(UTC).isoformat(),
            "config_sha256": sha256(config_path),
            "runner_sha256": sha256(Path(__file__).resolve()),
            **result,
        }
        output = ROOT / config["outputs"]["terminal_report"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(report.get("decision", report.get("status")))
    if report.get("decision") == "failed_closed" or report.get("status") == "stellar_morphology_control_failed_closed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
