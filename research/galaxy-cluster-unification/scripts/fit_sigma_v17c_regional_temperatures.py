#!/usr/bin/env python3
"""Fit every frozen v17C regional Chandra spectrum without selection."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fit_sigma_v17c_integrated_temperatures import (
    DEFAULT_CONFIG,
    ROOT,
    configure_apec_data,
    evaluate_apec_probe,
    finite_number,
    json_value,
    primary_optimization_method,
    result_attributes,
    run_fit,
    sha256,
)

DEFAULT_SPECTRA = ROOT / "results" / "sigma_v17c_regional_spectra" / "report.json"
DEFAULT_INTEGRATED_TEMPERATURES = (
    ROOT / "results" / "sigma_v17c_integrated_temperatures" / "report.json"
)
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v17c_regional_temperatures"


def product_path(region: dict, role: str) -> Path:
    matches = [
        ROOT / item["relative_path"]
        for item in region["frozen_snapshot"]["products"]
        if item["role"] == role
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"region {region['region_id']} expected one {role}, found {matches}"
        )
    path = matches[0]
    expected = next(
        item["sha256"]
        for item in region["frozen_snapshot"]["products"]
        if item["role"] == role
    )
    if sha256(path) != expected:
        raise RuntimeError(f"frozen regional spectrum product changed: {path}")
    return path


def fit_region(
    cluster_name: str,
    region: dict,
    config: dict,
    integrated_fit: dict,
) -> dict[str, Any]:
    import numpy as np
    from sherpa.astro import ui
    from sherpa.utils.err import SherpaErr

    rid = int(region["region_id"])
    cluster_config = config["clusters"][cluster_name]
    model_config = config["model"]
    source_pha = product_path(region, "grouped_source_spectrum")
    background_pha = product_path(region, "background_spectrum")
    arf = product_path(region, "source_arf")
    rmf = product_path(region, "source_rmf")

    ui.clean()
    apec_data = configure_apec_data(ui)
    abundance_table_token = model_config["abundance_table"].split(maxsplit=1)[0]
    ui.set_xsabund(abundance_table_token)
    ui.load_pha(1, str(source_pha))
    ui.set_analysis(1, "energy", "counts")
    fit_lo, fit_hi = map(float, model_config["fit_energy_keV"])
    ui.notice_id(1, fit_lo, fit_hi)
    data = ui.get_data(1)
    if not getattr(data, "background_ids", []):
        raise RuntimeError(f"{source_pha} does not reference a background spectrum")
    filtered_counts = float(np.sum(np.asarray(data.get_dep(filter=True), dtype=float)))
    exposure = float(data.exposure)
    if not finite_number(filtered_counts) or filtered_counts <= 0 or exposure <= 0:
        raise RuntimeError(f"invalid filtered counts or exposure in {source_pha}")
    count_rate = filtered_counts / exposure
    norm_initial = max(1e-8, 0.01 * count_rate)

    ui.subtract(1)
    suffix = f"{cluster_name.lower()}_r{rid:03d}"
    absorption = ui.create_model_component("xstbabs", f"tbabs_{suffix}")
    thermal = ui.create_model_component("xsapec", f"apec_{suffix}")
    ui.set_source(1, absorption * thermal)

    absorption.nH = float(cluster_config["weighted_HI4PI_nH_cm2"]) / 1e22
    ui.freeze(absorption.nH)
    thermal.kT = float(model_config["temperature_keV"]["initial"])
    thermal.kT.min = float(model_config["temperature_keV"]["minimum"])
    thermal.kT.max = float(model_config["temperature_keV"]["maximum"])
    ui.thaw(thermal.kT)
    integrated_abundance = float(integrated_fit["parameters"]["abundance_solar"])
    thermal.Abundanc = integrated_abundance
    ui.freeze(thermal.Abundanc)
    thermal.Redshift = float(cluster_config["redshift"])
    ui.freeze(thermal.Redshift)
    thermal.norm = norm_initial
    thermal.norm.min = float(model_config["normalization"]["minimum"])
    thermal.norm.max = float(model_config["normalization"]["maximum"])
    ui.thaw(thermal.norm)
    apec_probe = evaluate_apec_probe(thermal, fit_lo, fit_hi)

    ui.set_stat(model_config["statistic"])
    fit_result, attempts = run_fit(
        ui,
        thermal,
        primary_optimization_method(model_config["optimization"]),
        SherpaErr,
    )
    temperature = float(thermal.kT.val)
    normalization = float(thermal.norm.val)
    statval = float(fit_result.statval)
    dof = int(fit_result.dof)
    reduced_statistic = statval / dof if dof > 0 else math.nan

    conf_error = ""
    conf_result = None
    lower_delta = math.nan
    upper_delta = math.nan
    try:
        ui.set_conf_opt("sigma", 1.0)
        ui.conf(thermal.kT)
        conf_result = ui.get_conf_results()
        temperature_index = list(conf_result.parnames).index(thermal.kT.fullname)
        lower_delta = float(conf_result.parmins[temperature_index])
        upper_delta = float(conf_result.parmaxes[temperature_index])
    except (SherpaErr, RuntimeError, TypeError, ValueError) as exc:
        conf_error = f"{type(exc).__name__}: {exc}"
    lower = temperature + lower_delta
    upper = temperature + upper_delta
    finite_parameters = all(
        finite_number(value)
        for value in (temperature, normalization, lower, upper, integrated_abundance)
    )
    interval_ordered = finite_parameters and lower < temperature < upper
    fractional_half_width = (
        (upper - lower) / (2.0 * temperature)
        if interval_ordered and temperature > 0
        else math.nan
    )
    regional_gates = config["gates"]["regional"]
    gates = {
        "finite_temperature_and_interval": finite_parameters and interval_ordered,
        "fractional_68_percent_half_width_at_most_0_5": finite_number(
            fractional_half_width
        )
        and fractional_half_width
        <= float(regional_gates["maximum_fractional_68_percent_half_width"]),
        "reduced_statistic_at_most_1_5": finite_number(reduced_statistic)
        and reduced_statistic <= float(regional_gates["maximum_reduced_statistic"]),
    }
    gates["all_passed"] = all(gates.values())
    return {
        "cluster": cluster_name,
        "region_id": rid,
        "fit_completed": True,
        "fit_exception": "",
        "source_region": region["source_region"],
        "source_region_sha256": region["source_region_sha256"],
        "source_spectrum": str(source_pha),
        "source_spectrum_sha256": sha256(source_pha),
        "background_spectrum": str(background_pha),
        "background_spectrum_sha256": sha256(background_pha),
        "arf_sha256": sha256(arf),
        "rmf_sha256": sha256(rmf),
        "fit_band_keV": [fit_lo, fit_hi],
        "background_subtracted": True,
        "grouped_source_counts_in_fit_band": filtered_counts,
        "source_exposure_s": exposure,
        "background_unsubtracted_count_rate_s": count_rate,
        "normalization_initial": norm_initial,
        "model": model_config["expression"],
        "xspec_atomic_data": apec_data,
        "apec_model_probe": apec_probe,
        "abundance_table": model_config["abundance_table"],
        "xspec_abundance_table_token": abundance_table_token,
        "statistic": model_config["statistic"],
        "optimization_attempts": attempts,
        "parameters": {
            "nH_1e22_cm2_fixed": float(absorption.nH.val),
            "redshift_fixed": float(thermal.Redshift.val),
            "temperature_keV": temperature,
            "abundance_solar_fixed_to_integrated": integrated_abundance,
            "normalization": normalization,
        },
        "temperature_confidence_68_percent": {
            "lower_delta_keV": json_value(lower_delta),
            "upper_delta_keV": json_value(upper_delta),
            "lower_keV": json_value(lower),
            "upper_keV": json_value(upper),
            "fractional_half_width_definition": "(upper-lower)/(2*best_fit_temperature)",
            "fractional_half_width": json_value(fractional_half_width),
            "error": conf_error,
            "raw": result_attributes(
                conf_result,
                (
                    "datasets",
                    "methodname",
                    "iterfitname",
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
            "statval": statval,
            "dof": dof,
            "reduced_statistic": json_value(reduced_statistic),
            "raw": result_attributes(
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
                    "qval",
                    "rstat",
                    "parnames",
                    "parvals",
                ),
            ),
        },
        "gates": gates,
    }


def failed_region_result(
    cluster_name: str,
    region: dict,
    exc: Exception,
) -> dict[str, Any]:
    """Retain an attempted region as an explicit failed row without refitting it."""

    return {
        "cluster": cluster_name,
        "region_id": int(region["region_id"]),
        "fit_completed": False,
        "fit_exception": f"{type(exc).__name__}: {exc}",
        "source_region": region.get("source_region"),
        "source_region_sha256": region.get("source_region_sha256"),
        "parameters": {
            "nH_1e22_cm2_fixed": None,
            "redshift_fixed": None,
            "temperature_keV": None,
            "abundance_solar_fixed_to_integrated": None,
            "normalization": None,
        },
        "temperature_confidence_68_percent": {
            "lower_delta_keV": None,
            "upper_delta_keV": None,
            "lower_keV": None,
            "upper_keV": None,
            "fractional_half_width_definition": "(upper-lower)/(2*best_fit_temperature)",
            "fractional_half_width": None,
            "error": f"fit execution failed: {type(exc).__name__}: {exc}",
            "raw": None,
        },
        "fit": {
            "statval": None,
            "dof": None,
            "reduced_statistic": None,
            "raw": None,
        },
        "gates": {
            "finite_temperature_and_interval": False,
            "fractional_68_percent_half_width_at_most_0_5": False,
            "reduced_statistic_at_most_1_5": False,
            "all_passed": False,
        },
    }


def validate_inputs(
    config_path: Path,
    config: dict,
    spectra_path: Path,
    spectra: dict,
    integrated_path: Path,
    integrated: dict,
) -> None:
    config_hash = sha256(config_path)
    if spectra["status"] != "both_frozen_regional_spectra_extracted_combined_and_grouped":
        raise RuntimeError("regional spectral extraction is incomplete")
    if spectra.get("regional_temperature_fit_authorized") is not True:
        raise RuntimeError("regional temperature fitting is not authorized")
    if integrated["status"] != "both_integrated_temperature_gates_passed":
        raise RuntimeError("integrated temperature gate has not passed")
    if integrated.get("regional_fit_authorized") is not True:
        raise RuntimeError("integrated fit did not authorize regional fitting")
    if spectra["protocol_version"] != config["protocol_version"]:
        raise RuntimeError("regional spectra and fit protocols differ")
    if integrated["protocol_version"] != config["protocol_version"]:
        raise RuntimeError("integrated temperatures and regional protocols differ")
    if spectra["config_sha256"] != config_hash or integrated["config_sha256"] != config_hash:
        raise RuntimeError("frozen spectral config changed")
    if spectra["integrated_temperatures_report_sha256"] != sha256(integrated_path):
        raise RuntimeError("regional spectra used another integrated-temperature report")
    if not spectra_path.is_file() or not integrated_path.is_file():
        raise RuntimeError("missing regional fit input report")


def main() -> None:
    from sherpa.utils.err import SherpaErr

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--spectra", type=Path, default=DEFAULT_SPECTRA)
    parser.add_argument(
        "--integrated-temperatures", type=Path, default=DEFAULT_INTEGRATED_TEMPERATURES
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    spectra_path = args.spectra.resolve()
    integrated_path = args.integrated_temperatures.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    spectra = json.loads(spectra_path.read_text(encoding="utf-8"))
    integrated = json.loads(integrated_path.read_text(encoding="utf-8"))
    validate_inputs(
        config_path,
        config,
        spectra_path,
        spectra,
        integrated_path,
        integrated,
    )
    integrated_by_cluster = {row["cluster"]: row for row in integrated["clusters"]}
    cluster_results = []
    minimum_passing = int(config["gates"]["regional"]["minimum_passing_regions_per_cluster"])
    for cluster in spectra["clusters"]:
        cluster_name = cluster["cluster"]
        fits = []
        for region in cluster["regions"]:
            try:
                fits.append(
                    fit_region(
                        cluster_name,
                        region,
                        config,
                        integrated_by_cluster[cluster_name],
                    )
                )
            except (SherpaErr, RuntimeError, TypeError, ValueError, OSError) as exc:
                fits.append(failed_region_result(cluster_name, region, exc))
        passing = sum(int(row["gates"]["all_passed"]) for row in fits)
        minimum_passed = passing >= minimum_passing
        all_regions_fit_completed = all(row["fit_completed"] for row in fits)
        all_regions_have_finite_best_fit = all(
            finite_number(row["parameters"]["temperature_keV"]) for row in fits
        )
        cluster_passed = bool(
            minimum_passed
            and all_regions_fit_completed
            and all_regions_have_finite_best_fit
        )
        cluster_results.append(
            {
                "cluster": cluster_name,
                "regions_fitted": len(fits),
                "regions_passing_all_individual_gates": passing,
                "minimum_passing_regions_required": minimum_passing,
                "minimum_passing_regions_gate": minimum_passed,
                "all_regions_fit_completed": all_regions_fit_completed,
                "all_regions_have_finite_best_fit": all_regions_have_finite_best_fit,
                "regional_temperature_gate": cluster_passed,
                "regions": fits,
            }
        )
        print(
            f"{cluster_name}: {passing}/{len(fits)} regions pass; "
            f"minimum required={minimum_passing}",
            flush=True,
        )
    all_passed = all(row["regional_temperature_gate"] for row in cluster_results)
    report = {
        "status": "both_regional_temperature_gates_passed"
        if all_passed
        else "regional_temperature_gate_failed",
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "regional_spectra_report_sha256": sha256(spectra_path),
        "integrated_temperatures_report_sha256": sha256(integrated_path),
        "fractional_half_width_definition": "(upper-lower)/(2*best_fit_temperature)",
        "clusters": cluster_results,
        "all_regional_gates_passed": all_passed,
        "thermal_stress_construction_authorized": all_passed,
        "thermal_stress_constructed": False,
        "lensing_target_opened": False,
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(report_path)
    if not all_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
