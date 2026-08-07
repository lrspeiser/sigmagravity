#!/usr/bin/env python3
"""Audit per-observation soft/background heterogeneity behind V19DN."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import run_sigma_v19dn_integrated_residual_localization as v19dn
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def validate_frozen(
    config: dict[str, Any], runner: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256(runner) != config["implementation"]["runner_sha256"]:
        raise RuntimeError("V19DO runner changed after freeze")
    for name, item in config["parents"].items():
        path = ROOT / item["path"]
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"V19DO parent changed: {name}")
    parent_config_path = ROOT / config["parents"]["v19dn_config"]["path"]
    parent_runner = ROOT / config["parents"]["v19dn_runner"]["path"]
    _, v19dl_report, _ = v19dn.validate_frozen(
        load_json(parent_config_path), parent_runner
    )
    parent_report = load_json(ROOT / config["parents"]["v19dn_report"]["path"])
    if parent_report["status"] != "integrated_residual_localization_completed":
        raise RuntimeError("V19DN no longer supplies the registered soft failure")
    if not parent_report["aggregate_pass"]:
        raise RuntimeError("V19DN execution did not complete")
    if parent_report["full_regional_successor_authorized"]:
        raise RuntimeError("V19DN unexpectedly authorized regional production")
    if parent_report["lensing_halo_action_gravity_or_holdout_payload_opened"]:
        raise RuntimeError("V19DN unexpectedly opened a sealed payload")
    index = ROOT / config["inputs"]["validated_cell_index"]["path"]
    if sha256(index) != config["inputs"]["validated_cell_index"]["sha256"]:
        raise RuntimeError("V19DO validated-cell index changed")
    products = ROOT / config["inputs"]["unified_product_index"]["path"]
    if sha256(products) != config["inputs"]["unified_product_index"]["sha256"]:
        raise RuntimeError("V19DO unified-product index changed")
    return v19dl_report, parent_report


def product_path(row: dict[str, str], role: str) -> Path:
    return Path(row["cell_directory"]) / "products" / row[role]


def read_ebounds(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with fits.open(path, memmap=True, lazy_load_hdus=True) as hdus:
        table = hdus["EBOUNDS"].data
        return (
            np.asarray(table["CHANNEL"], dtype=np.int32).copy(),
            np.asarray(table["E_MIN"], dtype=np.float64).copy(),
            np.asarray(table["E_MAX"], dtype=np.float64).copy(),
        )


def band_masks(
    channels: np.ndarray,
    energy_lo: np.ndarray,
    energy_hi: np.ndarray,
    bands: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    order = np.argsort(channels)
    if not np.array_equal(channels[order], np.arange(1, channels.size + 1)):
        raise RuntimeError("V19DO EBOUNDS channels are not contiguous from one")
    lo = energy_lo[order]
    hi = energy_hi[order]
    return {
        band["id"]: (hi > float(band["minimum_keV"]))
        & (lo < float(band["maximum_keV"]))
        for band in bands
    }


def numeric(header: fits.Header, key: str) -> float:
    value = float(header[key])
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"V19DO invalid positive header {key}={value}")
    return value


def read_spectrum(
    path: Path, expected_hash: str, masks: dict[str, np.ndarray]
) -> tuple[dict[str, float], dict[str, Any]]:
    if sha256(path) != expected_hash:
        raise RuntimeError(f"V19DO spectrum changed: {path}")
    with fits.open(path, memmap=True, lazy_load_hdus=True) as hdus:
        spectrum = hdus["SPECTRUM"]
        counts = np.asarray(spectrum.data["COUNTS"], dtype=np.float64)
        channels = np.asarray(spectrum.data["CHANNEL"], dtype=np.int32)
        if counts.size != next(iter(masks.values())).size:
            raise RuntimeError(f"V19DO unexpected channel count: {path}")
        if not np.array_equal(channels, np.arange(1, counts.size + 1)):
            raise RuntimeError(f"V19DO PHA channels are not canonical: {path}")
        if np.any(~np.isfinite(counts)) or np.any(counts < 0):
            raise RuntimeError(f"V19DO invalid counts: {path}")
        header = spectrum.header
        totals = {name: float(np.sum(counts[mask])) for name, mask in masks.items()}
        metadata = {
            "exposure": numeric(header, "EXPOSURE"),
            "backscal": numeric(header, "BACKSCAL"),
            "areascal": numeric(header, "AREASCAL"),
            "obs_id": str(header["OBS_ID"]),
            "date_obs": str(header["DATE-OBS"]),
            "detnam": str(header["DETNAM"]),
            "backfile": str(header.get("BACKFILE", "")),
            "respfile": str(header.get("RESPFILE", "")),
            "ancrfile": str(header.get("ANCRFILE", "")),
        }
    return totals, metadata


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else math.nan


def aggregate(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    source = sum(float(row[f"source_{key}"]) for row in rows)
    background = sum(float(row[f"scaled_background_{key}"]) for row in rows)
    variance = sum(float(row[f"variance_{key}"]) for row in rows)
    net = source - background
    return {
        "source_counts": source,
        "scaled_background_counts": background,
        "net_counts": net,
        "background_fraction_of_source": ratio(background, source),
        "net_signal_to_noise": ratio(net, math.sqrt(variance)),
    }


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        "cells": len(rows),
        "date_obs_min": min(row["date_obs"] for row in rows),
        "date_obs_max": max(row["date_obs"] for row in rows),
        "detnam": sorted({row["detnam"] for row in rows}),
    }
    for key in ("full_0p5_7", "soft_0p5_2", "transition_1_2", "hard_2_7"):
        result[key] = aggregate(rows, key)
    result["net_soft_to_hard_ratio"] = ratio(
        result["soft_0p5_2"]["net_counts"], result["hard_2_7"]["net_counts"]
    )
    return result


def write_cell_audit(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def execute(
    config: dict[str, Any], v19dl_report: dict[str, Any], output: Path
) -> dict[str, Any]:
    validated_path = ROOT / config["inputs"]["validated_cell_index"]["path"]
    products_path = ROOT / config["inputs"]["unified_product_index"]["path"]
    validated = read_csv(validated_path)
    products = read_csv(products_path)
    by_cell = {row["cell_name"]: row for row in products}
    if len(validated) != int(config["gates"]["expected_cells"]):
        raise RuntimeError("V19DO validated-cell count changed")
    if len(by_cell) != len(products) or {
        row["cell_name"] for row in validated
    } != set(by_cell):
        raise RuntimeError("V19DO validated and product indexes do not match exactly")

    representative: dict[tuple[str, str], dict[str, str]] = {}
    for row in validated:
        representative.setdefault((row["obsid"], row["ccd_id"]), by_cell[row["cell_name"]])
    ebounds_reference = None
    ebounds_hashes = []
    for key, row in sorted(representative.items()):
        rmf = product_path(row, "rmf_name")
        if sha256(rmf) != row["rmf_sha256"]:
            raise RuntimeError(f"V19DO representative RMF changed: {rmf}")
        arrays = read_ebounds(rmf)
        array_hash = hashlib.sha256(
            b"".join(np.asarray(value).tobytes() for value in arrays)
        ).hexdigest()
        ebounds_hashes.append({"obsid": key[0], "ccd_id": key[1], "sha256": array_hash})
        if ebounds_reference is None:
            ebounds_reference = arrays
        elif not all(
            np.array_equal(actual, expected)
            for actual, expected in zip(arrays, ebounds_reference, strict=True)
        ):
            raise RuntimeError("V19DO EBOUNDS differ across observation/CCD controls")
    if ebounds_reference is None:
        raise RuntimeError("V19DO found no response controls")
    masks = band_masks(*ebounds_reference, config["bands"])

    cell_rows = []
    scaling_tolerance = float(config["gates"]["scaling_relative_tolerance"])
    for index_row in validated:
        product = by_cell[index_row["cell_name"]]
        source_path = product_path(product, "source_pha_name")
        background_path = product_path(product, "background_pha_name")
        if str(source_path) != index_row["source_pha"]:
            raise RuntimeError(f"V19DO source path mismatch: {index_row['cell_name']}")
        if product["source_pha_sha256"] != index_row["source_pha_sha256"]:
            raise RuntimeError(f"V19DO source index hash mismatch: {index_row['cell_name']}")
        source_counts, source = read_spectrum(
            source_path, product["source_pha_sha256"], masks
        )
        background_counts, background = read_spectrum(
            background_path, product["background_pha_sha256"], masks
        )
        if source["obs_id"] != index_row["obsid"]:
            raise RuntimeError(f"V19DO source ObsID mismatch: {index_row['cell_name']}")
        if Path(source["backfile"]).name != product["background_pha_name"]:
            raise RuntimeError(f"V19DO BACKFILE mismatch: {index_row['cell_name']}")
        if Path(source["respfile"]).name != product["rmf_name"]:
            raise RuntimeError(f"V19DO RESPFILE mismatch: {index_row['cell_name']}")
        if Path(source["ancrfile"]).name != product["arf_name"]:
            raise RuntimeError(f"V19DO ANCRFILE mismatch: {index_row['cell_name']}")
        scale = (
            source["exposure"]
            / background["exposure"]
            * source["backscal"]
            / background["backscal"]
            * source["areascal"]
            / background["areascal"]
        )
        if not math.isfinite(scale) or scale <= 0:
            raise RuntimeError(f"V19DO invalid background scale: {index_row['cell_name']}")
        if abs(source["backscal"] / background["backscal"] - 1.0) > scaling_tolerance:
            raise RuntimeError(f"V19DO BACKSCAL mismatch: {index_row['cell_name']}")
        row: dict[str, Any] = {
            "cluster": index_row["cluster"],
            "bin_id": int(index_row["bin_id"]),
            "obsid": int(index_row["obsid"]),
            "ccd_id": int(index_row["ccd_id"]),
            "cell_name": index_row["cell_name"],
            "date_obs": source["date_obs"],
            "detnam": source["detnam"],
            "background_scale": scale,
        }
        for key in masks:
            source_value = source_counts[key]
            background_value = scale * background_counts[key]
            row[f"source_{key}"] = source_value
            row[f"raw_background_{key}"] = background_counts[key]
            row[f"scaled_background_{key}"] = background_value
            row[f"net_{key}"] = source_value - background_value
            row[f"variance_{key}"] = source_value + scale * scale * background_counts[key]
        cell_rows.append(row)

    cell_path = output / "cell_soft_background_audit.csv"
    write_cell_audit(cell_path, cell_rows)
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cell_rows:
        grouped[(row["cluster"], row["obsid"], row["ccd_id"])].append(row)
        clusters[row["cluster"]].append(row)
    observation_rows = []
    for (cluster, obsid, ccd_id), rows in sorted(grouped.items()):
        observation_rows.append(
            {
                "cluster": cluster,
                "obsid": obsid,
                "ccd_id": ccd_id,
                **summarize_group(rows),
            }
        )
    cluster_summary = {cluster: summarize_group(rows) for cluster, rows in clusters.items()}

    parent_integrated = {row["cluster"]: row for row in v19dl_report["integrated_fits"]}
    count_checks = {}
    for cluster, summary in cluster_summary.items():
        observed = summary["full_0p5_7"]["source_counts"]
        expected = float(
            parent_integrated[cluster][
                "sherpa_response_energy_fit_band_source_counts"
            ]
        )
        count_checks[cluster] = {
            "cell_sum": observed,
            "v19dl_sherpa_integrated": expected,
            "exact": observed == expected,
        }

    thresholds = config["interpretation_thresholds"]
    interpretations = {}
    for cluster, summary in cluster_summary.items():
        fraction = summary["soft_0p5_2"]["background_fraction_of_source"]
        observation_fractions = [
            row["soft_0p5_2"]["background_fraction_of_source"]
            for row in observation_rows
            if row["cluster"] == cluster
        ]
        span = max(observation_fractions) - min(observation_fractions)
        if fraction >= float(thresholds["background_dominated_fraction"]):
            regime = "background_dominated"
        elif fraction <= float(thresholds["source_dominated_fraction"]):
            regime = "source_dominated"
        else:
            regime = "mixed"
        interpretations[cluster] = {
            "aggregate_soft_background_regime": regime,
            "aggregate_soft_background_fraction": fraction,
            "observation_soft_background_fraction_min": min(observation_fractions),
            "observation_soft_background_fraction_max": max(observation_fractions),
            "observation_soft_background_fraction_span": span,
            "strong_observation_heterogeneity": span
            >= float(thresholds["strong_observation_fraction_span"]),
        }
    bullet = interpretations["BULLET"]
    if bullet["aggregate_soft_background_regime"] == "background_dominated":
        next_test = "observation_resolved_blank_sky_and_soft_calibration_likelihood"
    elif bullet["strong_observation_heterogeneity"]:
        next_test = "observation_resolved_joint_likelihood_with_detector_epoch_controls"
    else:
        next_test = "spatially_resolved_joint_plasma_likelihood_with_unmerged_responses"

    gates = {
        "all_expected_cells_audited": len(cell_rows)
        == int(config["gates"]["expected_cells"]),
        "every_observation_ccd_ebounds_exact": len({row["sha256"] for row in ebounds_hashes})
        == 1,
        "cell_sums_reproduce_v19dl_integrated_source_counts_exactly": all(
            row["exact"] for row in count_checks.values()
        ),
    }
    return {
        "status": (
            "observation_resolved_soft_background_audit_completed"
            if all(gates.values())
            else "observation_resolved_soft_background_audit_gate_failed"
        ),
        "aggregate_pass": all(gates.values()),
        "cell_audit": {
            "path": str(cell_path.relative_to(ROOT)),
            "rows": len(cell_rows),
            "bytes": cell_path.stat().st_size,
            "sha256": sha256(cell_path),
        },
        "response_ebounds_controls": ebounds_hashes,
        "observation_ccd_summary": observation_rows,
        "cluster_summary": cluster_summary,
        "integrated_count_checks": count_checks,
        "interpretations": interpretations,
        "next_test": next_test,
        "gates": gates,
        "joint_likelihood_or_full_regional_successor_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        config = load_json(config_path)
        v19dl_report, parent_report = validate_frozen(
            config, Path(__file__).resolve()
        )
        result = execute(config, v19dl_report, output)
        result["v19dn_report_sha256"] = sha256(
            ROOT / config["parents"]["v19dn_report"]["path"]
        )
        result["v19dn_status"] = parent_report["status"]
    except Exception as exc:  # noqa: BLE001 - preserve terminal audit evidence
        result = {
            "status": "observation_resolved_soft_background_audit_execution_failed",
            "execution_exception": f"{type(exc).__name__}: {exc}",
            "aggregate_pass": False,
            "joint_likelihood_or_full_regional_successor_authorized": False,
        }
    report = {
        "protocol_version": "SIGMA-V19DO-OBSERVATION-SOFT-BACKGROUND-1.0.0",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        **result,
        "additional_plasma_component_admitted": False,
        "thermal_stress_or_baroclinicity_constructed": False,
        "lensing_halo_action_gravity_or_holdout_payload_opened": False,
        "gravity_formula_or_parameter_changed": False,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report_path)
    print(f"status: {report['status']}")
    if not report["aggregate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
