#!/usr/bin/env python3
"""Fit registered regions jointly without merging observation responses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import run_sigma_v19dj_direct_response_commissioning as v19dj
import run_sigma_v19do2_backscal_ratio_remediation as v19do2
from sherpa.astro import ui
from sherpa.utils.err import SherpaErr

ROOT = Path(__file__).resolve().parents[1]
INHERITED_FIT = v19dj.v19x2.inherited_v19x.inherited_fit


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
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    if sha256(runner) != config["implementation"]["runner_sha256"]:
        raise RuntimeError("V19DP runner changed after freeze")
    for name, item in config["parents"].items():
        path = ROOT / item["path"]
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"V19DP parent changed: {name}")
    parent_config_path = ROOT / config["parents"]["v19do2_config"]["path"]
    parent_runner = ROOT / config["parents"]["v19do2_runner"]["path"]
    v19do_config, v19dl_report, _ = v19do2.validate_frozen(
        load_json(parent_config_path), parent_runner
    )
    parent_report = load_json(ROOT / config["parents"]["v19do2_report"]["path"])
    if parent_report["status"] != (
        "backscal_ratio_observation_soft_background_audit_completed"
    ):
        raise RuntimeError("V19DO2 no longer supplies the completed source audit")
    if not parent_report["aggregate_pass"]:
        raise RuntimeError("V19DO2 did not pass")
    if parent_report["joint_likelihood_or_full_regional_successor_authorized"]:
        raise RuntimeError("V19DO2 unexpectedly authorized a successor")
    if parent_report["lensing_halo_action_gravity_or_holdout_payload_opened"]:
        raise RuntimeError("V19DO2 unexpectedly opened a sealed payload")
    validated_path = ROOT / v19do_config["inputs"]["validated_cell_index"]["path"]
    products_path = ROOT / v19do_config["inputs"]["unified_product_index"]["path"]
    if sha256(validated_path) != v19do_config["inputs"]["validated_cell_index"][
        "sha256"
    ]:
        raise RuntimeError("V19DP validated-cell index changed")
    if sha256(products_path) != v19do_config["inputs"]["unified_product_index"][
        "sha256"
    ]:
        raise RuntimeError("V19DP product index changed")
    products = read_csv(products_path)
    return v19dl_report, parent_report, products


def product_path(row: dict[str, str], role: str) -> Path:
    return Path(row["cell_directory"]) / "products" / row[role]


def selected_cells(
    config: dict[str, Any], products: list[dict[str, str]]
) -> dict[str, list[dict[str, str]]]:
    result = {}
    for cluster, selection in config["registered_regions"].items():
        rows = [
            row
            for row in products
            if row["cluster"] == cluster
            and int(row["bin_id"]) == int(selection["bin_id"])
        ]
        rows.sort(key=lambda row: (int(row["obsid"]), int(row["ccd_id"])))
        if len(rows) != int(selection["expected_cells"]):
            raise RuntimeError(f"V19DP cell count changed for {cluster}")
        if [row["cell_name"] for row in rows] != selection["cell_names"]:
            raise RuntimeError(f"V19DP cell membership changed for {cluster}")
        for row in rows:
            for name, hash_name in (
                ("source_pha_name", "source_pha_sha256"),
                ("background_pha_name", "background_pha_sha256"),
                ("arf_name", "arf_sha256"),
                ("rmf_name", "rmf_sha256"),
            ):
                path = product_path(row, name)
                if sha256(path) != row[hash_name]:
                    raise RuntimeError(f"V19DP product changed: {path}")
        result[cluster] = rows
    return result


def configure_parameter(
    parameter: Any, value: float, bounds: dict[str, Any]
) -> None:
    parameter.min = float(bounds["minimum"])
    parameter.max = float(bounds["maximum"])
    parameter.val = value
    ui.thaw(parameter)


def optimize(ids: list[int]) -> tuple[Any, list[str]]:
    attempts = []
    try:
        ui.set_method("levmar")
        ui.set_method_opt("maxfev", 10000)
        ui.fit(*ids)
        fit = ui.get_fit_results()
        attempts.append("levmar")
        if math.isfinite(float(fit.statval)):
            return fit, attempts
    except (SherpaErr, RuntimeError, TypeError, ValueError):
        attempts.append("levmar_failed")
    ui.set_method("neldermead")
    ui.set_method_opt("maxfev", 30000)
    ui.fit(*ids)
    attempts.append("neldermead")
    ui.set_method("levmar")
    ui.set_method_opt("maxfev", 10000)
    ui.fit(*ids)
    attempts.append("levmar_polish")
    return ui.get_fit_results(), attempts


def fit_joint(
    config: dict[str, Any],
    cluster: str,
    rows: list[dict[str, str]],
    confidence: bool,
) -> dict[str, Any]:
    ui.clean()
    atomic_data = INHERITED_FIT.configure_apec_data(ui)
    model = config["model"]
    ui.set_xsabund(str(model["abundance_table"]).split(maxsplit=1)[0])
    ids = list(range(1, len(rows) + 1))
    dataset_rows = []
    initial_norms = []
    for dataset_id, row in zip(ids, rows, strict=True):
        source = product_path(row, "source_pha_name")
        ui.load_pha(dataset_id, str(source))
        ui.ungroup(dataset_id)
        ui.set_analysis(dataset_id, "energy", "counts")
        data = ui.get_data(dataset_id)
        if not getattr(data, "background_ids", []):
            raise RuntimeError(f"V19DP missing background: {source}")
        if bool(getattr(data, "subtracted", False)):
            raise RuntimeError(f"V19DP source already subtracted: {source}")
        ui.group_counts(dataset_id, int(model["minimum_grouped_source_counts"]))
        fit_lo, fit_hi = map(float, model["fit_energy_keV"])
        ui.notice_id(dataset_id, fit_lo, fit_hi)
        grouped_counts = np.asarray(data.get_dep(filter=True), dtype=float)
        counts = float(np.sum(grouped_counts))
        exposure = float(data.exposure)
        initial_norms.append(max(1e-8, 0.01 * counts / exposure))
        dataset_rows.append(
            {
                "dataset_id": dataset_id,
                "cell_name": row["cell_name"],
                "obsid": int(row["obsid"]),
                "ccd_id": int(row["ccd_id"]),
                "grouped_bins": int(grouped_counts.size),
                "source_counts_in_fit_band": counts,
                "source_exposure_s": exposure,
                "background_scale": float(ui.get_bkg_scale(dataset_id)),
            }
        )
        ui.subtract(dataset_id)

    suffix = cluster.lower()
    thermal = ui.create_model_component("xsapec", f"apec_v19dp_{suffix}")
    temperature = model["temperature_keV"]
    abundance = model["abundance_solar"]
    normalization = model["normalization"]
    configure_parameter(thermal.kT, float(temperature["initial"]), temperature)
    configure_parameter(thermal.Abundanc, float(abundance["initial"]), abundance)
    configure_parameter(thermal.norm, float(np.median(initial_norms)), normalization)
    thermal.Redshift = float(config["clusters"][cluster]["redshift"])
    ui.freeze(thermal.Redshift)
    absorptions = []
    for dataset_id in ids:
        absorption = ui.create_model_component(
            "xstbabs", f"tbabs_v19dp_{suffix}_{dataset_id}"
        )
        absorption.nH = float(config["clusters"][cluster]["galactic_nh_cm2"]) / 1e22
        ui.freeze(absorption.nH)
        ui.set_source(dataset_id, absorption * thermal)
        absorptions.append(absorption)
    ui.set_stat(model["statistic"])

    starts = []
    for index, start in enumerate(model["starts"]):
        thermal.kT = float(start["temperature_keV"])
        thermal.Abundanc = float(start["abundance_solar"])
        thermal.norm = float(np.median(initial_norms)) * float(start["norm_multiplier"])
        try:
            fit, attempts = optimize(ids)
            starts.append(
                {
                    "index": index,
                    "fit_completed": True,
                    "attempts": attempts,
                    "statistic": float(fit.statval),
                    "dof": int(fit.dof),
                    "temperature_keV": float(thermal.kT.val),
                    "abundance_solar": float(thermal.Abundanc.val),
                    "normalization": float(thermal.norm.val),
                }
            )
        except Exception as exc:  # noqa: BLE001 - retain every frozen start
            starts.append(
                {
                    "index": index,
                    "fit_completed": False,
                    "exception": f"{type(exc).__name__}: {exc}",
                }
            )
    successful = [row for row in starts if row["fit_completed"]]
    if not successful:
        raise RuntimeError(f"V19DP every start failed for {cluster}")
    best = min(successful, key=lambda row: row["statistic"])
    thermal.kT = best["temperature_keV"]
    thermal.Abundanc = best["abundance_solar"]
    thermal.norm = best["normalization"]
    final_fit, final_attempts = optimize(ids)
    best_temperature = float(thermal.kT.val)
    best_abundance = float(thermal.Abundanc.val)
    best_norm = float(thermal.norm.val)
    statval = float(final_fit.statval)
    dof = int(final_fit.dof)
    reduced = statval / dof if dof > 0 else math.nan
    lower = upper = math.nan
    conf_error = "not_requested"
    if confidence:
        conf_error = ""
        try:
            ui.set_conf_opt("sigma", 1.0)
            ui.conf(thermal.kT)
            conf = ui.get_conf_results()
            position = list(conf.parnames).index(thermal.kT.fullname)
            lower = best_temperature + float(conf.parmins[position])
            upper = best_temperature + float(conf.parmaxes[position])
        except (SherpaErr, RuntimeError, TypeError, ValueError) as exc:
            conf_error = f"{type(exc).__name__}: {exc}"
    ordered = all(math.isfinite(value) for value in (lower, upper)) and (
        lower < best_temperature < upper
    )
    half_width = (
        (upper - lower) / (2.0 * best_temperature)
        if ordered and best_temperature > 0
        else math.nan
    )
    inside_bounds = (
        float(temperature["minimum"])
        < best_temperature
        < float(temperature["maximum"])
        and float(abundance["minimum"])
        < best_abundance
        < float(abundance["maximum"])
        and float(normalization["minimum"])
        < best_norm
        < float(normalization["maximum"])
    )
    ui.clean()
    return {
        "cluster": cluster,
        "cells": len(rows),
        "cell_names": [row["cell_name"] for row in rows],
        "datasets": dataset_rows,
        "atomic_data": atomic_data,
        "model": model["expression"],
        "statistic_name": model["statistic"],
        "minimum_grouped_source_counts": int(model["minimum_grouped_source_counts"]),
        "free_parameter_count": 3,
        "shared_parameters": ["temperature_keV", "abundance_solar", "normalization"],
        "starts": starts,
        "best_start_index": int(best["index"]),
        "final_optimization_attempts": final_attempts,
        "fit": {
            "statistic": statval,
            "dof": dof,
            "reduced_statistic": reduced,
        },
        "parameters": {
            "temperature_keV": best_temperature,
            "abundance_solar": best_abundance,
            "normalization": best_norm,
        },
        "temperature_confidence_68_percent": {
            "lower_keV": lower,
            "upper_keV": upper,
            "fractional_half_width": half_width,
            "ordered": ordered,
            "error": conf_error,
        },
        "all_free_parameters_strictly_inside_bounds": inside_bounds,
    }


def intervals_overlap(first: tuple[float, float], second: tuple[float, float]) -> bool:
    return max(first[0], second[0]) <= min(first[1], second[1])


def execute(
    config: dict[str, Any],
    v19dl_report: dict[str, Any],
    products: list[dict[str, str]],
) -> dict[str, Any]:
    cells = selected_cells(config, products)
    parent_regions = {row["cluster"]: row for row in v19dl_report["regional_fits"]}
    results = []
    for cluster, rows in cells.items():
        repeats = [fit_joint(config, cluster, rows, True) for _ in range(2)]
        primary = repeats[0]
        leave_one_out = []
        for omitted in rows:
            subset = [row for row in rows if row["cell_name"] != omitted["cell_name"]]
            fit = fit_joint(config, cluster, subset, False)
            leave_one_out.append(
                {
                    "omitted_cell": omitted["cell_name"],
                    "omitted_obsid": int(omitted["obsid"]),
                    "temperature_keV": fit["parameters"]["temperature_keV"],
                    "abundance_solar": fit["parameters"]["abundance_solar"],
                    "normalization": fit["parameters"]["normalization"],
                    "reduced_statistic": fit["fit"]["reduced_statistic"],
                }
            )
        parent = parent_regions[cluster]
        parent_interval = (
            float(parent["temperature_confidence_68_percent"]["lower_keV"]),
            float(parent["temperature_confidence_68_percent"]["upper_keV"]),
        )
        joint_interval = (
            float(primary["temperature_confidence_68_percent"]["lower_keV"]),
            float(primary["temperature_confidence_68_percent"]["upper_keV"]),
        )
        temperature = float(primary["parameters"]["temperature_keV"])
        leave_shifts = [
            abs(float(row["temperature_keV"]) / temperature - 1.0)
            for row in leave_one_out
        ]
        repeat_tolerance = float(config["gates"]["repeat_relative_tolerance"])
        gates = {
            "both_independent_builds_exact_within_tolerance": all(
                abs(
                    float(repeats[1]["parameters"][key])
                    / float(repeats[0]["parameters"][key])
                    - 1.0
                )
                <= repeat_tolerance
                for key in ("temperature_keV", "abundance_solar", "normalization")
            )
            and abs(
                float(repeats[1]["fit"]["statistic"])
                / float(repeats[0]["fit"]["statistic"])
                - 1.0
            )
            <= repeat_tolerance,
            "reduced_statistic_at_most_1_5": float(
                primary["fit"]["reduced_statistic"]
            )
            <= float(config["gates"]["maximum_reduced_statistic"]),
            "temperature_interval_ordered_and_precise": bool(
                primary["temperature_confidence_68_percent"]["ordered"]
            )
            and float(
                primary["temperature_confidence_68_percent"][
                    "fractional_half_width"
                ]
            )
            <= float(config["gates"]["maximum_fractional_temperature_half_width"]),
            "all_free_parameters_strictly_inside_bounds": bool(
                primary["all_free_parameters_strictly_inside_bounds"]
            ),
            "merged_and_unmerged_temperature_intervals_overlap": intervals_overlap(
                parent_interval, joint_interval
            ),
            "merged_and_unmerged_best_temperatures_within_25_percent": abs(
                temperature / float(parent["parameters"]["temperature_keV"]) - 1.0
            )
            <= float(config["gates"]["maximum_merged_temperature_relative_shift"]),
            "leave_one_observation_out_temperature_shift_at_most_25_percent": max(
                leave_shifts
            )
            <= float(config["gates"]["maximum_leave_one_out_temperature_shift"]),
            "no_dataset_exceeds_30_percent_of_source_counts": max(
                float(row["source_counts_in_fit_band"])
                for row in primary["datasets"]
            )
            / sum(
                float(row["source_counts_in_fit_band"])
                for row in primary["datasets"]
            )
            <= float(config["gates"]["maximum_single_dataset_count_fraction"]),
        }
        results.append(
            {
                "cluster": cluster,
                "primary": primary,
                "repeat": repeats[1],
                "parent_merged_region_fit": parent,
                "leave_one_observation_out": leave_one_out,
                "maximum_leave_one_out_temperature_relative_shift": max(leave_shifts),
                "gates": gates,
                "passed": all(gates.values()),
            }
        )
    aggregate_pass = all(row["passed"] for row in results)
    return {
        "status": (
            "unmerged_regional_joint_likelihood_preflight_passed"
            if aggregate_pass
            else "unmerged_regional_joint_likelihood_preflight_failed"
        ),
        "aggregate_pass": aggregate_pass,
        "regions": results,
        "full_regional_joint_likelihood_successor_authorized": aggregate_pass,
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
        v19dl_report, parent_report, products = validate_frozen(
            config, Path(__file__).resolve()
        )
        result = execute(config, v19dl_report, products)
        result["v19do2_report_sha256"] = sha256(
            ROOT / config["parents"]["v19do2_report"]["path"]
        )
        result["v19do2_status"] = parent_report["status"]
    except Exception as exc:  # noqa: BLE001 - preserve terminal preflight evidence
        result = {
            "status": "unmerged_regional_joint_likelihood_preflight_execution_failed",
            "execution_exception": f"{type(exc).__name__}: {exc}",
            "aggregate_pass": False,
            "full_regional_joint_likelihood_successor_authorized": False,
        }
    report = {
        "protocol_version": "SIGMA-V19DP-UNMERGED-REGIONAL-JOINT-1.0.0",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        **result,
        "all_494_regions_run": False,
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
