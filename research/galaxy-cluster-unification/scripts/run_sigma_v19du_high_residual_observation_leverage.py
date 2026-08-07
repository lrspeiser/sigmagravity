#!/usr/bin/env python3
"""Test whether one observation drives each high-residual V19DS region."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import run_sigma_v19dp_unmerged_regional_joint_likelihood_preflight as v19dp

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19du_high_residual_observation_leverage.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19du_high_residual_observation_leverage"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate(config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    if config["freeze_state"] != "frozen_after_v19ds_and_v19dt_before_any_omission_fit":
        raise RuntimeError("V19DU configuration is not frozen before omission fitting")
    if sha256(Path(__file__).resolve()) != config["implementation"]["runner_sha256"]:
        raise RuntimeError("V19DU runner changed after freeze")
    for name, item in config["parents"].items():
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"V19DU parent changed: {name}")
    report = load_json(ROOT / config["parents"]["v19ds_report"]["path"])
    if (
        report["status"] != "full_494_region_unmerged_joint_likelihood_failed_closed"
        or report["aggregate_pass"]
        or len(report["regions"]) != 494
        or report["i4_i5_source_only_successor_authorized"]
        or report["thermal_stress_or_baroclinicity_constructed"]
        or report["lensing_halo_action_gravity_or_holdout_payload_opened"]
        or report["gravity_formula_or_parameter_changed"]
    ):
        raise RuntimeError("V19DU parent is not the sealed V19DS failure")
    dp_config = load_json(ROOT / config["parents"]["v19dp_config"]["path"])
    for key in ("model", "clusters"):
        if config[key] != dp_config[key]:
            raise RuntimeError(f"V19DU changed the V19DP {key}")
    products = read_csv(ROOT / config["parents"]["v19dr_unified_index"]["path"])
    return report, products


def build_plan(
    config: dict[str, Any], report: dict[str, Any], products: list[dict[str, str]]
) -> list[tuple[str, int, float, list[dict[str, str]]]]:
    threshold = float(config["selection"]["minimum_reduced_statistic_exclusive"])
    selected = [
        row
        for row in report["regions"]
        if float(row["fit"]["fit"]["reduced_statistic"]) > threshold
    ]
    product_groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in products:
        product_groups[(row["cluster"], int(row["bin_id"]))].append(row)
    plan = []
    omissions = Counter()
    for row in selected:
        cluster = row["cluster"]
        bin_id = int(row["bin_id"])
        cells = product_groups[(cluster, bin_id)]
        cells.sort(key=lambda item: (int(item["obsid"]), int(item["ccd_id"])))
        unique_obsids = {int(item["obsid"]) for item in cells}
        omissions[cluster] += len(unique_obsids)
        plan.append(
            (
                cluster,
                bin_id,
                float(row["fit"]["fit"]["reduced_statistic"]),
                cells,
            )
        )
    plan.sort(key=lambda item: (item[0], item[1]))
    expected = config["selection"]["expected"]
    if len(plan) != int(expected["regions_total"]):
        raise RuntimeError("V19DU selected region count changed")
    counts = Counter(row[0] for row in plan)
    for cluster, values in expected["clusters"].items():
        if counts[cluster] != int(values["regions"]):
            raise RuntimeError(f"V19DU selected region count changed for {cluster}")
        if omissions[cluster] != int(values["observation_omission_fits"]):
            raise RuntimeError(f"V19DU omission count changed for {cluster}")
    if sum(omissions.values()) != int(expected["observation_omission_fits_total"]):
        raise RuntimeError("V19DU total omission count changed")
    return plan


def input_digest(
    config: dict[str, Any], cluster: str, bin_id: int, rows: list[dict[str, str]]
) -> str:
    return canonical_sha256(
        {
            "protocol_version": config["protocol_version"],
            "cluster": cluster,
            "bin_id": bin_id,
            "model": config["model"],
            "cells": [
                {
                    "cell_name": row["cell_name"],
                    "source": row["source_pha_sha256"],
                    "background": row["background_pha_sha256"],
                    "arf": row["arf_sha256"],
                    "rmf": row["rmf_sha256"],
                }
                for row in rows
            ],
        }
    )


def checkpoint_path(output: Path, cluster: str, bin_id: int) -> Path:
    return output / "checkpoints" / cluster / f"bin{bin_id}.json"


def compact_fit(fit: dict[str, Any]) -> dict[str, Any]:
    return {
        "cells": int(fit["cells"]),
        "obsids": sorted({int(row["obsid"]) for row in fit["datasets"]}),
        "statistic": float(fit["fit"]["statistic"]),
        "dof": int(fit["fit"]["dof"]),
        "reduced_statistic": float(fit["fit"]["reduced_statistic"]),
        "temperature_keV": float(fit["parameters"]["temperature_keV"]),
        "abundance_solar": float(fit["parameters"]["abundance_solar"]),
        "normalization": float(fit["parameters"]["normalization"]),
        "all_free_parameters_strictly_inside_bounds": bool(
            fit["all_free_parameters_strictly_inside_bounds"]
        ),
    }


def fit_region_omissions(
    config: dict[str, Any],
    cluster: str,
    bin_id: int,
    primary_reduced: float,
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    started = time.monotonic()
    digest = input_digest(config, cluster, bin_id, rows)
    results = []
    for obsid in sorted({int(row["obsid"]) for row in rows}):
        subset = [row for row in rows if int(row["obsid"]) != obsid]
        try:
            fit = v19dp.fit_joint(config, cluster, subset, confidence=False)
            compact = compact_fit(fit)
            results.append(
                {
                    "omitted_obsid": obsid,
                    "omitted_cells": sum(int(row["obsid"]) == obsid for row in rows),
                    "status": "fit_completed",
                    **compact,
                    "reduced_statistic_ratio_to_primary": compact["reduced_statistic"]
                    / primary_reduced,
                }
            )
        except Exception as exc:  # noqa: BLE001 - preserve every omission failure
            results.append(
                {
                    "omitted_obsid": obsid,
                    "omitted_cells": sum(int(row["obsid"]) == obsid for row in rows),
                    "status": "fit_failed",
                    "exception": f"{type(exc).__name__}: {exc}",
                }
            )
    completed = [row for row in results if row["status"] == "fit_completed"]
    best = min(completed, key=lambda row: row["reduced_statistic"]) if completed else None
    return {
        "cluster": cluster,
        "bin_id": bin_id,
        "cells": len(rows),
        "observations": len(results),
        "input_digest": digest,
        "primary_reduced_statistic": primary_reduced,
        "omissions": results,
        "all_omission_fits_completed": len(completed) == len(results),
        "best_omission": best,
        "rescued_to_reduced_statistic_at_most_1_5": bool(
            best is not None
            and float(best["reduced_statistic"])
            <= float(config["decision_gates"]["rescue_maximum_reduced_statistic"])
        ),
        "best_omission_reduces_reduced_statistic_by_at_least_half": bool(
            best is not None
            and float(best["reduced_statistic_ratio_to_primary"])
            <= float(config["decision_gates"]["large_improvement_maximum_ratio"])
        ),
        "elapsed_seconds": time.monotonic() - started,
    }


def validate_checkpoint(
    path: Path, digest: str, cluster: str, bin_id: int
) -> dict[str, Any]:
    row = load_json(path)
    if (
        row.get("input_digest") != digest
        or row.get("cluster") != cluster
        or int(row.get("bin_id", -1)) != bin_id
    ):
        raise RuntimeError(f"V19DU checkpoint changed: {path}")
    return row


def observation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_observation: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    best_counts: Counter[tuple[str, int]] = Counter()
    for row in rows:
        if row["best_omission"] is not None:
            best_counts[(row["cluster"], int(row["best_omission"]["omitted_obsid"]))] += 1
        for omission in row["omissions"]:
            if omission["status"] == "fit_completed":
                by_observation[(row["cluster"], int(omission["omitted_obsid"]))].append(
                    omission
                )
    result = {}
    for (cluster, obsid), values in sorted(by_observation.items()):
        ratios = np.asarray(
            [float(row["reduced_statistic_ratio_to_primary"]) for row in values]
        )
        key = f"{cluster}_obs{obsid}"
        result[key] = {
            "cluster": cluster,
            "obsid": obsid,
            "eligible_regions": len(values),
            "best_omission_regions": best_counts[(cluster, obsid)],
            "best_omission_fraction": best_counts[(cluster, obsid)] / len(values),
            "median_reduced_statistic_ratio": float(np.median(ratios)),
            "minimum_reduced_statistic_ratio": float(np.min(ratios)),
            "regions_rescued_to_1_5": sum(
                float(row["reduced_statistic"]) <= 1.5 for row in values
            ),
        }
    return result


def summarize(config: dict[str, Any], rows: list[dict[str, Any]], reused: int) -> dict[str, Any]:
    rows.sort(key=lambda row: (row["cluster"], int(row["bin_id"])))
    observation = observation_summary(rows)
    expected = config["selection"]["expected"]
    omission_count = sum(len(row["omissions"]) for row in rows)
    failed_omissions = sum(
        omission["status"] != "fit_completed"
        for row in rows
        for omission in row["omissions"]
    )
    rescued = sum(row["rescued_to_reduced_statistic_at_most_1_5"] for row in rows)
    large = sum(
        row["best_omission_reduces_reduced_statistic_by_at_least_half"] for row in rows
    )
    culprit_threshold = float(
        config["decision_gates"]["consistent_observation_best_omission_fraction"]
    )
    ratio_threshold = float(
        config["decision_gates"]["consistent_observation_median_maximum_ratio"]
    )
    culprits = [
        value
        for value in observation.values()
        if value["best_omission_fraction"] >= culprit_threshold
        and value["median_reduced_statistic_ratio"] <= ratio_threshold
    ]
    maximum_rescue_fraction = float(
        config["decision_gates"]["maximum_region_rescue_fraction_for_model_dominated"]
    )
    model_dominated = rescued / len(rows) <= maximum_rescue_fraction and not culprits
    gates = {
        "all_127_high_residual_regions_audited": len(rows)
        == int(expected["regions_total"]),
        "all_1250_observation_omission_fits_attempted": omission_count
        == int(expected["observation_omission_fits_total"]),
        "every_observation_omission_fit_completed": failed_omissions == 0,
        "no_single_observation_is_a_consistent_cross_region_culprit": not culprits,
        "at_most_25_percent_of_regions_are_rescued_by_one_observation_omission": rescued
        / len(rows)
        <= maximum_rescue_fraction,
        "i4_i5_lensing_halo_gravity_and_holdout_remain_sealed": True,
    }
    return {
        "status": "high_residual_observation_leverage_audit_completed",
        "aggregate_pass": all(gates.values()),
        "regions": rows,
        "regions_audited": len(rows),
        "observation_omission_fits": omission_count,
        "failed_observation_omission_fits": failed_omissions,
        "regions_rescued_to_reduced_statistic_at_most_1_5": rescued,
        "region_rescue_fraction": rescued / len(rows),
        "regions_with_at_least_half_reduced_statistic_reduction": large,
        "observation_summaries": observation,
        "consistent_observation_culprits": culprits,
        "model_inadequacy_dominates_observation_leverage": model_dominated,
        "gates": gates,
        "next_stage": (
            "freeze_complexity_penalized_one_temperature_vs_multitemperature_plasma_comparison"
            if model_dominated and failed_omissions == 0
            else "retain_measurement_failure_and_diagnose_observation_or_implementation_systematics"
        ),
        "checkpoint_reused_regions": reused,
        "i4_i5_source_only_successor_authorized": False,
    }


def run(
    config: dict[str, Any], report: dict[str, Any], products: list[dict[str, str]], output: Path
) -> dict[str, Any]:
    plan = build_plan(config, report, products)
    results = []
    pending = []
    reused = 0
    for cluster, bin_id, primary, rows in plan:
        digest = input_digest(config, cluster, bin_id, rows)
        checkpoint = checkpoint_path(output, cluster, bin_id)
        if checkpoint.is_file():
            results.append(validate_checkpoint(checkpoint, digest, cluster, bin_id))
            reused += 1
        else:
            pending.append((cluster, bin_id, primary, rows))
    started = time.monotonic()
    workers = int(config["implementation"]["maximum_concurrent_regions"])
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fit_region_omissions, config, *item): (item[0], item[1])
            for item in pending
        }
        for index, future in enumerate(as_completed(futures), start=1):
            cluster, bin_id = futures[future]
            result = future.result()
            write_json(checkpoint_path(output, cluster, bin_id), result)
            results.append(result)
            print(
                f"V19DU {reused + index}/{len(plan)}: {cluster} bin {bin_id}; "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
    return summarize(config, results, reused)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_json(config_path)
    try:
        report, products = validate(config)
        result = run(config, report, products, output)
    except Exception as exc:  # noqa: BLE001 - retain terminal diagnostic failure
        result = {
            "status": "high_residual_observation_leverage_audit_execution_failed",
            "aggregate_pass": False,
            "execution_exception": f"{type(exc).__name__}: {exc}",
            "gates": {"execution_completed": False},
            "i4_i5_source_only_successor_authorized": False,
        }
    payload = {
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        **result,
        "thermal_stress_or_baroclinicity_constructed": False,
        "lensing_halo_action_gravity_or_holdout_payload_opened": False,
        "gravity_formula_or_parameter_changed": False,
    }
    write_json(output / "report.json", payload)
    print(json.dumps({key: value for key, value in payload.items() if key != "regions"}, indent=2))
    if not payload["aggregate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
