#!/usr/bin/env python3
"""Commission the frozen V19T Sherpa temperature fit on one response cell."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fit_sigma_v17c_integrated_temperatures as inherited_fit
import run_sigma_v17c_integrated_spectra as inherited_spectra
import run_sigma_v19p_exact_flux_obs_support as v19p

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19t_temperature_fit_commissioning.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19t_temperature_fit_commissioning"
DEFAULT_SCRATCH = Path("/home/henry/sigma-v19t-temperature-commissioning/v100")


def validate_parent_hashes(config: dict[str, Any]) -> None:
    for key, value in config["parents"].items():
        if key.endswith("_sha256"):
            continue
        expected = config["parents"].get(f"{key}_sha256")
        if expected is not None and v19p.sha256(ROOT / value) != expected:
            raise RuntimeError(f"V19T parent hash mismatch: {value}")


def v19r_products(config: dict[str, Any]) -> dict[str, Path]:
    report = v19p.load_json(ROOT / config["parents"]["v19r_report"])
    if not report["full_response_production_authorized"]:
        raise RuntimeError("V19R response commissioning did not pass")
    paths = {}
    for role, item in report["frozen_snapshot"].items():
        if role == "specextract_log":
            continue
        path = ROOT / item["relative_path"]
        if path.stat().st_size != item["bytes"] or v19p.sha256(path) != item["sha256"]:
            raise RuntimeError(f"V19R frozen product changed: {path}")
        paths[role] = path
    required = {"source_pha", "background_pha", "arf", "rmf"}
    if set(paths) != required:
        raise RuntimeError(f"V19T V19R product set mismatch: {set(paths)}")
    return paths


def copy_inputs(paths: dict[str, Path], destination: Path) -> dict[str, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    copied = {}
    for role, source in paths.items():
        target = destination / source.name
        if target.exists():
            if v19p.sha256(target) != v19p.sha256(source):
                raise RuntimeError(f"V19T scratch input changed: {target}")
        else:
            shutil.copy2(source, target)
        copied[role] = target
    return copied


def group_source(
    source: Path,
    grouped: Path,
    minimum_counts: int,
    env: dict[str, str],
    log: Path,
) -> dict[str, Any]:
    command = [
        "dmgroup",
        f"infile={source}",
        f"outfile={grouped}",
        "grouptype=NUM_CTS",
        f"grouptypeval={minimum_counts}",
        "binspec=",
        "xcolumn=CHANNEL",
        "ycolumn=COUNTS",
        "tabspec=",
        "tabcolumn=",
        "stopspec=",
        "stopcolumn=",
        "clobber=no",
        "verbose=1",
        "mode=h",
    ]
    return inherited_spectra.run_step(command, log, [grouped], env)


def pha_links(path: Path, env: dict[str, str]) -> dict[str, str]:
    return {
        key: inherited_spectra.command_text(
            ["dmkeypar", str(path), key, "echo+"], env
        )
        for key in ("BACKFILE", "ANCRFILE", "RESPFILE")
    }


def fit_once(
    config: dict[str, Any],
    grouped: Path,
    initial_temperature: float,
    compute_confidence: bool,
) -> dict[str, Any]:
    import numpy as np
    from sherpa.astro import ui
    from sherpa.utils.err import SherpaErr

    model = config["model"]
    ui.clean()
    atomic_data = inherited_fit.configure_apec_data(ui)
    abundance_token = model["abundance_table"].split(maxsplit=1)[0]
    ui.set_xsabund(abundance_token)
    ui.load_pha(1, str(grouped))
    ui.set_analysis(1, "energy", "counts")
    fit_lo, fit_hi = map(float, model["fit_energy_keV"])
    ui.notice_id(1, fit_lo, fit_hi)
    data = ui.get_data(1)
    if not getattr(data, "background_ids", []):
        raise RuntimeError("V19T grouped PHA has no background reference")
    filtered_counts = float(np.sum(np.asarray(data.get_dep(filter=True), dtype=float)))
    filtered_bins = int(np.asarray(data.get_dep(filter=True)).size)
    exposure = float(data.exposure)
    if filtered_counts <= 0.0 or exposure <= 0.0:
        raise RuntimeError("V19T grouped PHA has invalid counts or exposure")
    normalization_initial = max(1e-8, 0.01 * filtered_counts / exposure)
    ui.subtract(1)
    token = str(initial_temperature).replace(".", "p")
    absorption = ui.create_model_component("xstbabs", f"tbabs_v19t_{token}")
    thermal = ui.create_model_component("xsapec", f"apec_v19t_{token}")
    ui.set_source(1, absorption * thermal)
    absorption.nH = float(model["galactic_nh_cm2_fixed"]) / 1e22
    ui.freeze(absorption.nH)
    temperature_config = model["temperature_keV"]
    thermal.kT = initial_temperature
    thermal.kT.min = float(temperature_config["minimum"])
    thermal.kT.max = float(temperature_config["maximum"])
    ui.thaw(thermal.kT)
    thermal.Abundanc = float(model["abundance_solar_fixed"])
    ui.freeze(thermal.Abundanc)
    thermal.Redshift = float(model["redshift_fixed"])
    ui.freeze(thermal.Redshift)
    thermal.norm = normalization_initial
    thermal.norm.min = float(model["normalization"]["minimum"])
    thermal.norm.max = float(model["normalization"]["maximum"])
    ui.thaw(thermal.norm)
    apec_probe = inherited_fit.evaluate_apec_probe(thermal, fit_lo, fit_hi)
    ui.set_stat(model["statistic"])
    fit_result, attempts = inherited_fit.run_fit(
        ui,
        thermal,
        inherited_fit.primary_optimization_method(model["optimization"]),
        SherpaErr,
    )
    temperature = float(thermal.kT.val)
    normalization = float(thermal.norm.val)
    statval = float(fit_result.statval)
    dof = int(fit_result.dof)
    reduced = statval / dof if dof > 0 else math.nan
    confidence = None
    if compute_confidence:
        ui.set_conf_opt("sigma", 1.0)
        ui.conf(thermal.kT)
        conf_result = ui.get_conf_results()
        index = list(conf_result.parnames).index(thermal.kT.fullname)
        lower_delta = float(conf_result.parmins[index])
        upper_delta = float(conf_result.parmaxes[index])
        confidence = {
            "lower_delta_keV": lower_delta,
            "upper_delta_keV": upper_delta,
            "lower_keV": temperature + lower_delta,
            "upper_keV": temperature + upper_delta,
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
            ),
        }
    return {
        "initial_temperature_keV": initial_temperature,
        "temperature_keV": temperature,
        "normalization": normalization,
        "normalization_initial": normalization_initial,
        "fixed_nh_1e22_cm2": float(absorption.nH.val),
        "fixed_redshift": float(thermal.Redshift.val),
        "fixed_abundance_solar": float(thermal.Abundanc.val),
        "filtered_source_counts": filtered_counts,
        "filtered_grouped_bins": filtered_bins,
        "source_exposure_s": exposure,
        "statval": statval,
        "dof": dof,
        "reduced_statistic": inherited_fit.json_value(reduced),
        "optimization_attempts": attempts,
        "fit_raw": inherited_fit.result_attributes(
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
        "confidence_68_percent": confidence,
        "xspec_atomic_data": atomic_data,
        "apec_model_probe": apec_probe,
    }


def execute(config: dict[str, Any], scratch: Path, output: Path) -> dict[str, Any]:
    products = copy_inputs(v19r_products(config), scratch / "inputs")
    env = inherited_spectra.isolated_environment(
        os.environ, scratch / "pfiles", scratch / "tmp"
    )
    grouped = scratch / "inputs" / "BULLET_bin390_obs5356_ccd2_grp.pi"
    grouping_step = group_source(
        products["source_pha"],
        grouped,
        int(config["grouping"]["minimum_counts"]),
        env,
        scratch / "logs" / "dmgroup.log",
    )
    links = pha_links(grouped, env)
    expected_links = {
        "BACKFILE": products["background_pha"].name,
        "ANCRFILE": products["arf"].name,
        "RESPFILE": products["rmf"].name,
    }
    link_gate = links == expected_links
    starts = [
        float(config["model"]["temperature_keV"]["primary_initial"]),
        *map(float, config["model"]["temperature_keV"]["alternate_initials"]),
    ]
    fits = [
        fit_once(config, grouped, value, compute_confidence=index == 0)
        for index, value in enumerate(starts)
    ]
    temperatures = [row["temperature_keV"] for row in fits]
    fractional_spread = (max(temperatures) - min(temperatures)) / median(temperatures)
    primary = fits[0]
    confidence = primary["confidence_68_percent"]
    if confidence is None:
        raise RuntimeError("V19T primary fit did not compute confidence")
    lower = float(confidence["lower_keV"])
    upper = float(confidence["upper_keV"])
    temperature = float(primary["temperature_keV"])
    fractional_half_width = (upper - lower) / (2.0 * temperature)
    temperature_config = config["model"]["temperature_keV"]
    normalization_config = config["model"]["normalization"]
    finite_fits = all(
        all(
            inherited_fit.finite_number(row[key])
            for key in ("temperature_keV", "normalization", "statval", "reduced_statistic")
        )
        for row in fits
    )
    interval_gate = all(
        inherited_fit.finite_number(value) for value in (lower, temperature, upper)
    ) and lower < temperature < upper
    bounds_gate = all(
        float(temperature_config["minimum"])
        < float(row["temperature_keV"])
        < float(temperature_config["maximum"])
        and float(normalization_config["minimum"])
        < float(row["normalization"])
        < float(normalization_config["maximum"])
        for row in fits
    )
    gates = {
        "grouped_pha_retains_v19r_background_arf_and_rmf_links": link_gate,
        "apec_probe_is_finite_positive": all(
            row["apec_model_probe"]["integrated_flux"] > 0.0 for row in fits
        ),
        "all_three_initial_temperatures_fit_successfully": finite_fits
        and all(bool(row["fit_raw"].get("succeeded")) for row in fits),
        "multistart_fractional_temperature_spread_at_most_0_05": fractional_spread
        <= float(config["gates"]["maximum_multistart_fractional_temperature_spread"]),
        "finite_ordered_68_percent_temperature_interval": interval_gate,
        "fractional_68_percent_half_width_at_most_0_5": fractional_half_width
        <= float(config["gates"]["maximum_fractional_68_percent_half_width"]),
        "reduced_statistic_at_most_1_5": inherited_fit.finite_number(
            primary["reduced_statistic"]
        )
        and float(primary["reduced_statistic"])
        <= float(config["gates"]["maximum_reduced_statistic"]),
        "temperature_and_normalization_not_on_bounds": bounds_gate,
    }
    snapshot = inherited_spectra.copy_snapshot(
        grouped,
        output / "frozen_products" / grouped.name,
    )
    return {
        "status": (
            "temperature_fit_commissioning_passed_and_full_fit_pipeline_authorized"
            if all(gates.values())
            else "temperature_fit_commissioning_gate_failed"
        ),
        "grouping_step": grouping_step,
        "grouped_pha_links": links,
        "expected_grouped_pha_links": expected_links,
        "fits": fits,
        "multistart_fractional_temperature_spread": fractional_spread,
        "primary_fractional_68_percent_half_width": fractional_half_width,
        "frozen_grouped_pha": snapshot,
        "gates": gates,
        "full_response_and_fit_production_authorized": all(gates.values()),
    }


def main() -> None:
    from sherpa.utils.err import SherpaErr

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    args = parser.parse_args()
    config_path = args.config.resolve()
    output = args.output.resolve()
    config = v19p.load_json(config_path)
    validate_parent_hashes(config)
    output.mkdir(parents=True, exist_ok=True)
    try:
        result = execute(config, args.scratch.resolve(), output)
    except (SherpaErr, RuntimeError, OSError, TypeError, ValueError, KeyError) as exc:
        result = {
            "status": "temperature_fit_commissioning_execution_failed",
            "fit_exception": f"{type(exc).__name__}: {exc}",
            "gates": {"execution_completed": False},
            "full_response_and_fit_production_authorized": False,
        }
    report = {
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": v19p.sha256(config_path),
        "runner_sha256": v19p.sha256(Path(__file__).resolve()),
        **result,
        "scientific_temperature_map_claimed": False,
        "thermal_stress_constructed": False,
        "lensing_target_opened": False,
        "gravity_formula_or_parameter_changed": False,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report_path)
    print(f"status: {report['status']}")
    if "fits" in report:
        primary = report["fits"][0]
        print(
            f"kT={primary['temperature_keV']:.6g} keV; "
            f"reduced={primary['reduced_statistic']}; "
            f"spread={report['multistart_fractional_temperature_spread']:.3g}"
        )
    if not report["full_response_and_fit_production_authorized"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
