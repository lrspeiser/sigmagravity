#!/usr/bin/env python3
"""Commission V19X integrated and selected-region spectral combination and fits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fit_sigma_v17c_integrated_temperatures as inherited_fit
import run_sigma_v17c_integrated_spectra as inherited_spectra

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19x_spectral_combination_commissioning.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19x_spectral_combination_commissioning"
DEFAULT_RESPONSE_SCRATCH = Path("/home/henry/sigma-v19w-response-production/v100")
DEFAULT_SCRATCH = Path("/home/henry/sigma-v19x-spectral-combination/v101")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_parent_hashes(config: dict[str, Any]) -> None:
    for key, value in config["parents"].items():
        if key.endswith("_sha256"):
            continue
        expected = config["parents"].get(f"{key}_sha256")
        if expected is not None and sha256(ROOT / value) != expected:
            raise RuntimeError(f"V19X parent hash mismatch: {value}")
    execution = config["execution"]
    declared_runner = ROOT / execution["runner"]
    if declared_runner.resolve() != Path(__file__).resolve():
        raise RuntimeError("V19X config names another execution runner")
    if sha256(declared_runner) != execution["runner_sha256"]:
        raise RuntimeError("V19X execution runner changed after freeze")


def load_manifest(config: dict[str, Any]) -> list[dict[str, str]]:
    path = ROOT / config["parents"]["v19u_manifest"]
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = int(config["runtime_authorization"]["required_completed_cells"])
    keys = [task_key(row) for row in rows]
    if len(rows) != expected or len(keys) != len(set(keys)):
        raise RuntimeError("V19X manifest is not the frozen 5082-task workload")
    return rows


def task_key(row: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(row["cluster"]),
        int(row["bin_id"]),
        int(row["obsid"]),
        int(row["ccd_id"]),
    )


def cell_name(row: dict[str, Any]) -> str:
    cluster, bin_id, obsid, ccd_id = task_key(row)
    return f"{cluster}_bin{bin_id}_obs{obsid}_ccd{ccd_id}"


def build_aperture_plan(
    config: dict[str, Any], manifest: list[dict[str, str]]
) -> dict[str, dict[str, list[dict[str, str]]]]:
    plan: dict[str, dict[str, list[dict[str, str]]]] = {}
    for cluster, registered in config["registered_workload"]["clusters"].items():
        integrated = [row for row in manifest if row["cluster"] == cluster]
        selected_bin = int(registered["commissioning_region"]["bin_id"])
        regional = [
            row
            for row in integrated
            if int(row["bin_id"]) == selected_bin
        ]
        if len(integrated) != int(registered["total_cells"]):
            raise RuntimeError(f"{cluster} integrated cell count changed")
        if len(regional) != int(registered["commissioning_region"]["cells"]):
            raise RuntimeError(f"{cluster} commissioning-region cell count changed")
        if sum(int(row["source_band_events"]) for row in integrated) != int(
            registered["total_source_events_0p5_7_keV"]
        ):
            raise RuntimeError(f"{cluster} integrated source count changed")
        if sum(int(row["source_band_events"]) for row in regional) != int(
            registered["commissioning_region"]["source_events_0p5_7_keV"]
        ):
            raise RuntimeError(f"{cluster} selected-region source count changed")
        plan[cluster] = {"integrated": integrated, "regional": regional}
    return plan


def validate_runtime_authorization(
    config: dict[str, Any], report_path: Path | None = None
) -> tuple[dict[str, Any], Path]:
    runtime = config["runtime_authorization"]
    if report_path is None:
        report_path = ROOT / runtime["required_v19w_report"]
    if not report_path.is_file():
        raise RuntimeError("V19W has not produced its final authorization report")
    report = load_json(report_path)
    if report.get("status") != runtime["required_status"]:
        raise RuntimeError(f"V19W status does not authorize V19X: {report.get('status')}")
    if int(report.get("completed_cells", -1)) != int(runtime["required_completed_cells"]):
        raise RuntimeError("V19W completed-cell count does not authorize V19X")
    if int(report.get("product_index", {}).get("rows", -1)) != int(
        runtime["required_product_index_rows"]
    ):
        raise RuntimeError("V19W product-index row count does not authorize V19X")
    if report.get("regional_spectral_fitting_authorized") is not True:
        raise RuntimeError("V19W explicitly withheld regional fitting authorization")
    v19w_config = ROOT / config["parents"]["v19w_config"]
    v19w_runner = ROOT / config["parents"]["v19w_runner"]
    if report.get("config_sha256") != sha256(v19w_config):
        raise RuntimeError("V19W final report names another config")
    if report.get("runner_sha256") != sha256(v19w_runner):
        raise RuntimeError("V19W final report names another runner")
    product_index = ROOT / report["product_index"]["path"]
    if not product_index.is_file():
        raise RuntimeError("V19W product index is absent")
    if sha256(product_index) != report["product_index"]["sha256"]:
        raise RuntimeError("V19W product index hash changed")
    return report, product_index


def load_product_index(path: Path, expected_rows: int) -> dict[tuple[str, int, int, int], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {task_key(row): row for row in rows}
    if len(rows) != expected_rows or len(indexed) != expected_rows:
        raise RuntimeError("V19W product index is incomplete or has duplicate task keys")
    return indexed


def validate_cell(
    manifest_row: dict[str, str],
    index_row: dict[str, str],
    response_scratch: Path,
) -> dict[str, Any]:
    name = cell_name(manifest_row)
    completed = response_scratch / "completed" / name
    report_path = completed / "cell_report.json"
    if not report_path.is_file():
        raise RuntimeError(f"V19X missing completed cell report: {name}")
    report = load_json(report_path)
    if task_key(report) != task_key(manifest_row) or report.get("cell_name") != name:
        raise RuntimeError(f"V19X cell identity mismatch: {name}")
    if not report.get("gates") or not all(report["gates"].values()):
        raise RuntimeError(f"V19X cell has a failed response gate: {name}")
    preflight = report["preflight"]
    if int(preflight["source_band_events"]) != int(manifest_row["source_band_events"]):
        raise RuntimeError(f"V19X source event-energy count mismatch: {name}")
    if int(preflight["background_band_events"]) != int(
        manifest_row["background_band_events"]
    ):
        raise RuntimeError(f"V19X background event-energy count mismatch: {name}")
    products = report["products"]
    index_hash_keys = {
        "source_pha": "source_pha_sha256",
        "background_pha": "background_pha_sha256",
        "arf": "arf_sha256",
        "rmf": "rmf_sha256",
    }
    paths: dict[str, Path] = {}
    for role, index_key in index_hash_keys.items():
        item = products[role]
        path = completed / "products" / item["name"]
        if not path.is_file() or path.stat().st_size != int(item["bytes"]):
            raise RuntimeError(f"V19X missing or resized {role}: {name}")
        digest = sha256(path)
        if digest != item["sha256"] or digest != index_row[index_key]:
            raise RuntimeError(f"V19X changed {role} hash: {name}")
        paths[role] = path
    pha_total = int(report["source_pha_channel_audit"]["pha_total_counts"])
    if not report["source_pha_channel_audit"]["exact"] or pha_total <= 0:
        raise RuntimeError(f"V19X source PHA audit is not exact: {name}")
    return {
        "cluster": manifest_row["cluster"],
        "bin_id": int(manifest_row["bin_id"]),
        "obsid": int(manifest_row["obsid"]),
        "ccd_id": int(manifest_row["ccd_id"]),
        "cell_name": name,
        "source_band_events": int(manifest_row["source_band_events"]),
        "background_band_events": int(manifest_row["background_band_events"]),
        "source_pha_total_counts": pha_total,
        "source_pha": paths["source_pha"],
        "source_pha_sha256": products["source_pha"]["sha256"],
    }


def validate_archive(
    config: dict[str, Any],
    manifest: list[dict[str, str]],
    product_index_path: Path,
    response_scratch: Path,
) -> dict[tuple[str, int, int, int], dict[str, Any]]:
    expected = int(config["runtime_authorization"]["required_completed_cells"])
    index = load_product_index(product_index_path, expected)
    validated = {}
    for row in manifest:
        key = task_key(row)
        if key not in index:
            raise RuntimeError(f"V19X product index lacks task {key}")
        validated[key] = validate_cell(row, index[key], response_scratch)
    if len(validated) != expected:
        raise RuntimeError("V19X did not validate all 5082 response cells")
    return validated


def write_validated_index(records: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    fields = [
        "cluster",
        "bin_id",
        "obsid",
        "ccd_id",
        "cell_name",
        "source_band_events",
        "background_band_events",
        "source_pha_total_counts",
        "source_pha",
        "source_pha_sha256",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: str(record[key]) for key in fields})
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "rows": len(records),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def pha_total_counts(path: Path) -> int:
    from astropy.io import fits
    import numpy as np

    with fits.open(path, memmap=False) as hdus:
        values = np.asarray(hdus["SPECTRUM"].data["COUNTS"], dtype=np.int64)
    return int(values.sum())


def pha_links(path: Path, env: dict[str, str]) -> dict[str, str]:
    return {
        key: inherited_spectra.command_text(
            ["dmkeypar", str(path), key, "echo+"], env
        )
        for key in ("BACKFILE", "ANCRFILE", "RESPFILE")
    }


def combine_aperture(
    label: str,
    cells: list[dict[str, Any]],
    scratch: Path,
    output: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    work = scratch / label
    logs = work / "logs"
    env = inherited_spectra.isolated_environment(
        os.environ, work / "pfiles", work / "tmp"
    )
    source_paths = [Path(row["source_pha"]) for row in cells]
    stack = work / "source_spectra.lis"
    stack.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(str(path) for path in source_paths) + "\n"
    if stack.exists() and stack.read_text(encoding="utf-8") != content:
        raise RuntimeError(f"V19X existing stack changed: {stack}")
    stack.write_text(content, encoding="utf-8")
    outroot = work / label
    combined_source = outroot.with_name(outroot.name + "_src.pi")
    combined_background = outroot.with_name(outroot.name + "_bkg.pi")
    combined_arf = outroot.with_name(outroot.name + "_src.arf")
    combined_rmf = outroot.with_name(outroot.name + "_src.rmf")
    combination = config["combination"]
    combine_command = [
        "combine_spectra",
        f"src_spectra=@{stack}",
        f"outroot={outroot}",
        f"method={combination['method']}",
        f"bscale_method={combination['bscale_method']}",
        f"exp_origin={combination['exp_origin']}",
        "clobber=no",
        "verbose=1",
        "mode=h",
    ]
    combine_step = inherited_spectra.run_step(
        combine_command,
        logs / "combine_spectra.log",
        [combined_source, combined_background, combined_arf, combined_rmf],
        env,
    )
    expected_total = sum(int(row["source_pha_total_counts"]) for row in cells)
    combined_total = pha_total_counts(combined_source)
    if combined_total != expected_total:
        raise RuntimeError(
            f"V19X {label} combine_spectra did not conserve PHA counts: "
            f"{combined_total} != {expected_total}"
        )
    grouped = work / f"{label}_src_grp.pi"
    grouping = combination["group_after_combination"]
    group_command = [
        "dmgroup",
        f"infile={combined_source}",
        f"outfile={grouped}",
        f"grouptype={grouping['grouptype']}",
        f"grouptypeval={int(grouping['minimum_counts'])}",
        "binspec=",
        f"xcolumn={grouping['xcolumn']}",
        f"ycolumn={grouping['ycolumn']}",
        "tabspec=",
        "tabcolumn=",
        "stopspec=",
        "stopcolumn=",
        "clobber=no",
        "verbose=1",
        "mode=h",
    ]
    group_step = inherited_spectra.run_step(
        group_command, logs / "dmgroup.log", [grouped], env
    )
    links = pha_links(grouped, env)
    expected_links = {
        "BACKFILE": combined_background.name,
        "ANCRFILE": combined_arf.name,
        "RESPFILE": combined_rmf.name,
    }
    if links != expected_links:
        raise RuntimeError(f"V19X {label} grouped PHA links changed: {links}")
    snapshot_root = output / "frozen_products" / label
    snapshots = []
    for role, source in (
        ("grouped_source_spectrum", grouped),
        ("background_spectrum", combined_background),
        ("source_arf", combined_arf),
        ("source_rmf", combined_rmf),
    ):
        item = inherited_spectra.copy_snapshot(source, snapshot_root / source.name)
        item["role"] = role
        snapshots.append(item)
    return {
        "label": label,
        "cells": len(cells),
        "event_energy_source_counts_0p5_7_keV": sum(
            int(row["source_band_events"]) for row in cells
        ),
        "event_energy_background_counts_0p5_7_keV": sum(
            int(row["background_band_events"]) for row in cells
        ),
        "expected_full_pha_source_counts": expected_total,
        "combined_full_pha_source_counts": combined_total,
        "full_pha_count_conservation_exact": combined_total == expected_total,
        "source_stack_sha256": sha256(stack),
        "combine_step": combine_step,
        "group_step": group_step,
        "grouped_pha_links": links,
        "expected_grouped_pha_links": expected_links,
        "frozen_snapshot": {
            "files": len(snapshots),
            "bytes": sum(int(item["bytes"]) for item in snapshots),
            "products": snapshots,
        },
    }


def snapshot_path(combination: dict[str, Any], role: str) -> Path:
    matches = [
        item
        for item in combination["frozen_snapshot"]["products"]
        if item["role"] == role
    ]
    if len(matches) != 1:
        raise RuntimeError(f"V19X {combination['label']} has {len(matches)} {role} files")
    item = matches[0]
    path = ROOT / item["relative_path"]
    if not path.is_file() or sha256(path) != item["sha256"]:
        raise RuntimeError(f"V19X frozen {role} changed: {path}")
    return path


def fit_spectrum(
    config: dict[str, Any],
    cluster: str,
    combination: dict[str, Any],
    abundance_fixed: float | None,
) -> dict[str, Any]:
    import numpy as np
    from sherpa.astro import ui
    from sherpa.utils.err import SherpaErr

    fit_config = config["fit_sequence"]
    cluster_config = config["registered_workload"]["clusters"][cluster]
    source = snapshot_path(combination, "grouped_source_spectrum")
    background = snapshot_path(combination, "background_spectrum")
    arf = snapshot_path(combination, "source_arf")
    rmf = snapshot_path(combination, "source_rmf")
    ui.clean()
    atomic_data = inherited_fit.configure_apec_data(ui)
    abundance_token = fit_config["abundance_table"].split(maxsplit=1)[0]
    ui.set_xsabund(abundance_token)
    ui.load_pha(1, str(source))
    ui.set_analysis(1, "energy", "counts")
    fit_lo, fit_hi = map(float, fit_config["fit_energy_keV"])
    ui.notice_id(1, fit_lo, fit_hi)
    data = ui.get_data(1)
    if not getattr(data, "background_ids", []):
        raise RuntimeError(f"V19X grouped PHA has no background: {source}")
    filtered_counts = float(np.sum(np.asarray(data.get_dep(filter=True), dtype=float)))
    filtered_bins = int(np.asarray(data.get_dep(filter=True)).size)
    exposure = float(data.exposure)
    if filtered_counts <= 0 or exposure <= 0:
        raise RuntimeError(f"V19X invalid fit-band counts or exposure: {source}")
    norm_initial = max(1e-8, 0.01 * filtered_counts / exposure)
    ui.subtract(1)
    suffix = combination["label"].lower().replace("-", "_")
    absorption = ui.create_model_component("xstbabs", f"tbabs_{suffix}")
    thermal = ui.create_model_component("xsapec", f"apec_{suffix}")
    ui.set_source(1, absorption * thermal)
    absorption.nH = float(cluster_config["galactic_nh_cm2"]) / 1e22
    ui.freeze(absorption.nH)
    temperature = fit_config["temperature_keV"]
    thermal.kT = float(temperature["initial"])
    thermal.kT.min = float(temperature["minimum"])
    thermal.kT.max = float(temperature["maximum"])
    ui.thaw(thermal.kT)
    abundance = fit_config["integrated_abundance_solar"]
    if abundance_fixed is None:
        thermal.Abundanc = float(abundance["initial"])
        thermal.Abundanc.min = float(abundance["minimum"])
        thermal.Abundanc.max = float(abundance["maximum"])
        ui.thaw(thermal.Abundanc)
        abundance_mode = "free_integrated"
    else:
        thermal.Abundanc = float(abundance_fixed)
        ui.freeze(thermal.Abundanc)
        abundance_mode = "fixed_to_integrated"
    thermal.Redshift = float(cluster_config["redshift"])
    ui.freeze(thermal.Redshift)
    normalization = fit_config["normalization"]
    thermal.norm = norm_initial
    thermal.norm.min = float(normalization["minimum"])
    thermal.norm.max = float(normalization["maximum"])
    ui.thaw(thermal.norm)
    apec_probe = inherited_fit.evaluate_apec_probe(thermal, fit_lo, fit_hi)
    ui.set_stat(fit_config["statistic"])
    fit_result, attempts = inherited_fit.run_fit(
        ui,
        thermal,
        inherited_fit.primary_optimization_method(fit_config["optimization"]),
        SherpaErr,
    )
    best_temperature = float(thermal.kT.val)
    best_abundance = float(thermal.Abundanc.val)
    best_normalization = float(thermal.norm.val)
    statval = float(fit_result.statval)
    dof = int(fit_result.dof)
    reduced = statval / dof if dof > 0 else math.nan
    conf_error = ""
    conf_result = None
    lower = math.nan
    upper = math.nan
    try:
        ui.set_conf_opt("sigma", 1.0)
        ui.conf(thermal.kT)
        conf_result = ui.get_conf_results()
        index = list(conf_result.parnames).index(thermal.kT.fullname)
        lower = best_temperature + float(conf_result.parmins[index])
        upper = best_temperature + float(conf_result.parmaxes[index])
    except (SherpaErr, RuntimeError, TypeError, ValueError) as exc:
        conf_error = f"{type(exc).__name__}: {exc}"
    interval_ordered = all(
        inherited_fit.finite_number(value) for value in (lower, best_temperature, upper)
    ) and lower < best_temperature < upper
    fractional_half_width = (
        (upper - lower) / (2.0 * best_temperature)
        if interval_ordered and best_temperature > 0
        else math.nan
    )
    bounds_passed = (
        float(temperature["minimum"]) < best_temperature < float(temperature["maximum"])
        and float(normalization["minimum"])
        < best_normalization
        < float(normalization["maximum"])
    )
    if abundance_fixed is None:
        bounds_passed = bounds_passed and (
            float(abundance["minimum"]) < best_abundance < float(abundance["maximum"])
        )
    regional = abundance_fixed is not None
    gates = {
        "finite_parameters_and_ordered_temperature_interval": interval_ordered
        and all(
            inherited_fit.finite_number(value)
            for value in (best_abundance, best_normalization, statval, reduced)
        ),
        "reduced_statistic_at_most_1_5": inherited_fit.finite_number(reduced)
        and reduced <= float(config["gates"]["both_integrated_reduced_statistics_at_most"]),
        "all_free_parameters_strictly_inside_bounds": bounds_passed,
    }
    if regional:
        gates["fractional_68_percent_temperature_half_width_at_most_0_5"] = (
            inherited_fit.finite_number(fractional_half_width)
            and fractional_half_width
            <= float(
                config["gates"][
                    "both_regional_fractional_68_percent_temperature_half_widths_at_most"
                ]
            )
        )
    gates["all_passed"] = all(gates.values())
    return {
        "cluster": cluster,
        "label": combination["label"],
        "fit_completed": True,
        "abundance_mode": abundance_mode,
        "source_spectrum": str(source),
        "source_spectrum_sha256": sha256(source),
        "background_spectrum_sha256": sha256(background),
        "arf_sha256": sha256(arf),
        "rmf_sha256": sha256(rmf),
        "fit_band_keV": [fit_lo, fit_hi],
        "sherpa_response_energy_fit_band_source_counts": filtered_counts,
        "sherpa_response_energy_fit_band_grouped_bins": filtered_bins,
        "event_energy_manifest_count_is_not_equated_to_sherpa_count": True,
        "source_exposure_s": exposure,
        "normalization_initial": norm_initial,
        "model": fit_config["model"],
        "statistic": fit_config["statistic"],
        "abundance_table": fit_config["abundance_table"],
        "xspec_atomic_data": atomic_data,
        "apec_model_probe": apec_probe,
        "optimization_attempts": attempts,
        "parameters": {
            "nH_1e22_cm2_fixed": float(absorption.nH.val),
            "redshift_fixed": float(thermal.Redshift.val),
            "temperature_keV": best_temperature,
            "abundance_solar": best_abundance,
            "normalization": best_normalization,
        },
        "temperature_confidence_68_percent": {
            "lower_keV": inherited_fit.json_value(lower),
            "upper_keV": inherited_fit.json_value(upper),
            "fractional_half_width": inherited_fit.json_value(fractional_half_width),
            "error": conf_error,
            "raw": inherited_fit.result_attributes(
                conf_result,
                (
                    "datasets",
                    "methodname",
                    "fitname",
                    "statname",
                    "sigma",
                    "percent",
                    "parnames",
                    "parvals",
                    "parmins",
                    "parmaxes",
                    "nfits",
                ),
            )
            if conf_result is not None
            else None,
        },
        "fit": {
            "statval": inherited_fit.json_value(statval),
            "dof": dof,
            "reduced_statistic": inherited_fit.json_value(reduced),
            "raw": inherited_fit.result_attributes(
                fit_result,
                (
                    "datasets",
                    "methodname",
                    "statname",
                    "succeeded",
                    "message",
                    "nfev",
                    "istatval",
                    "statval",
                    "dstatval",
                    "numpoints",
                    "dof",
                    "parnames",
                    "parvals",
                ),
            ),
        },
        "gates": gates,
    }


def failed_fit(cluster: str, label: str, exc: Exception) -> dict[str, Any]:
    return {
        "cluster": cluster,
        "label": label,
        "fit_completed": False,
        "fit_exception": f"{type(exc).__name__}: {exc}",
        "parameters": {
            "temperature_keV": None,
            "abundance_solar": None,
            "normalization": None,
        },
        "gates": {"all_passed": False},
    }


def execute(
    config: dict[str, Any],
    output: Path,
    scratch: Path,
    response_scratch: Path,
    v19w_report_path: Path | None,
) -> dict[str, Any]:
    v19w_report, product_index = validate_runtime_authorization(
        config, v19w_report_path
    )
    manifest = load_manifest(config)
    plan = build_aperture_plan(config, manifest)
    validated = validate_archive(config, manifest, product_index, response_scratch)
    ordered_records = [validated[task_key(row)] for row in manifest]
    validated_index = write_validated_index(
        ordered_records, output / "validated_cell_index.csv"
    )
    combinations: dict[str, dict[str, dict[str, Any]]] = {}
    for cluster in plan:
        combinations[cluster] = {}
        for kind, rows in plan[cluster].items():
            label = (
                f"{cluster}_integrated"
                if kind == "integrated"
                else f"{cluster}_bin{int(rows[0]['bin_id'])}"
            )
            cells = [validated[task_key(row)] for row in rows]
            combinations[cluster][kind] = combine_aperture(
                label, cells, scratch, output, config
            )
    integrated_fits = []
    for cluster in plan:
        try:
            integrated_fits.append(
                fit_spectrum(
                    config, cluster, combinations[cluster]["integrated"], None
                )
            )
        except Exception as exc:  # retain the attempted commissioning fit
            integrated_fits.append(
                failed_fit(cluster, combinations[cluster]["integrated"]["label"], exc)
            )
    integrated_by_cluster = {row["cluster"]: row for row in integrated_fits}
    regional_fits = []
    for cluster in plan:
        integrated = integrated_by_cluster[cluster]
        if not integrated["fit_completed"]:
            regional_fits.append(
                failed_fit(
                    cluster,
                    combinations[cluster]["regional"]["label"],
                    RuntimeError("integrated abundance fit failed; regional fit not run"),
                )
            )
            continue
        try:
            regional_fits.append(
                fit_spectrum(
                    config,
                    cluster,
                    combinations[cluster]["regional"],
                    float(integrated["parameters"]["abundance_solar"]),
                )
            )
        except Exception as exc:  # retain the attempted commissioning fit
            regional_fits.append(
                failed_fit(cluster, combinations[cluster]["regional"]["label"], exc)
            )
    combination_rows = [
        item for cluster in combinations.values() for item in cluster.values()
    ]
    gates = {
        "v19w_complete_and_every_product_hash_exact": len(validated) == 5082,
        "combination_uses_every_registered_cell_exactly_once": all(
            combinations[cluster]["integrated"]["cells"]
            == int(config["registered_workload"]["clusters"][cluster]["total_cells"])
            and combinations[cluster]["regional"]["cells"]
            == int(
                config["registered_workload"]["clusters"][cluster][
                    "commissioning_region"
                ]["cells"]
            )
            for cluster in combinations
        ),
        "combined_source_background_arf_and_rmf_exist_and_links_are_exact": all(
            row["grouped_pha_links"] == row["expected_grouped_pha_links"]
            and row["frozen_snapshot"]["files"] == 4
            for row in combination_rows
        ),
        "every_cell_event_energy_counts_equal_manifest": True,
        "combined_full_pha_source_counts_conserved_exactly": all(
            row["full_pha_count_conservation_exact"] for row in combination_rows
        ),
        "both_integrated_fits_pass": all(
            row["fit_completed"] and row["gates"]["all_passed"]
            for row in integrated_fits
        ),
        "both_regional_fits_pass": all(
            row["fit_completed"] and row["gates"]["all_passed"]
            for row in regional_fits
        ),
    }
    all_passed = all(gates.values())
    return {
        "status": (
            "spectral_combination_commissioning_passed_and_full_regional_fits_authorized"
            if all_passed
            else "spectral_combination_commissioning_gate_failed"
        ),
        "v19w_report_sha256": sha256(
            ROOT / config["runtime_authorization"]["required_v19w_report"]
            if v19w_report_path is None
            else v19w_report_path
        ),
        "v19w_product_index_sha256": v19w_report["product_index"]["sha256"],
        "validated_cell_index": validated_index,
        "validated_response_cells": len(validated),
        "combinations": combinations,
        "integrated_fits": integrated_fits,
        "regional_fits": regional_fits,
        "gates": gates,
        "full_494_region_combination_and_fit_authorized": all_passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument(
        "--response-scratch", type=Path, default=DEFAULT_RESPONSE_SCRATCH
    )
    parser.add_argument("--v19w-report", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    output = args.output.resolve()
    config = load_json(config_path)
    validate_parent_hashes(config)
    output.mkdir(parents=True, exist_ok=True)
    try:
        result = execute(
            config,
            output,
            args.scratch.resolve(),
            args.response_scratch.resolve(),
            args.v19w_report.resolve() if args.v19w_report else None,
        )
    except Exception as exc:
        result = {
            "status": "spectral_combination_commissioning_execution_failed",
            "execution_exception": f"{type(exc).__name__}: {exc}",
            "gates": {"execution_completed": False},
            "full_494_region_combination_and_fit_authorized": False,
        }
    report = {
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        **result,
        "scientific_temperature_map_claimed": False,
        "thermal_stress_constructed": False,
        "replacement_cluster_lensing_target_opened": False,
        "gravity_formula_or_parameter_changed": False,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report_path)
    print(f"status: {report['status']}")
    if not report["full_494_region_combination_and_fit_authorized"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
