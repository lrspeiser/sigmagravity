#!/usr/bin/env python3
"""Diagnose V19DL with one universal minimal-adequate thermal-mixture rule."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import run_sigma_v19dj_direct_response_commissioning as v19dj
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


def validate_frozen(
    config: dict[str, Any], runner: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256(runner) != config["implementation"]["runner_sha256"]:
        raise RuntimeError("V19DM runner changed after freeze")
    for name, item in config["parents"].items():
        path = ROOT / item["path"]
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"V19DM parent changed: {name}")
    v19dj_config_path = ROOT / config["parents"]["v19dj_config"]["path"]
    v19dj_runner = ROOT / config["parents"]["v19dj_runner"]["path"]
    science = v19dj.validate_frozen(load_json(v19dj_config_path), v19dj_runner)
    parent = load_json(ROOT / config["parents"]["v19dl_report"]["path"])
    if parent["status"] != "canonicalized_direct_response_commissioning_gate_failed":
        raise RuntimeError("V19DL terminal status changed")
    required_true = (
        "v19w5_unified_archive_and_every_product_hash_exact",
        "combination_uses_every_registered_cell_exactly_once",
        "combined_source_background_arf_and_rmf_exist_and_links_are_exact",
        "every_cell_event_energy_counts_equal_manifest",
        "combined_full_pha_source_counts_conserved_exactly",
        "both_regional_fits_pass",
        "every_snapshot_canonicalized_with_valid_checksums",
    )
    if not all(parent["gates"][key] for key in required_true):
        raise RuntimeError("V19DL failed outside the registered integrated-fit gate")
    if parent["gates"]["both_integrated_fits_pass"]:
        raise RuntimeError("V19DL no longer supplies the registered 1T failure")
    if parent["thermal_stress_constructed"]:
        raise RuntimeError("V19DL unexpectedly constructed thermal stress")
    if parent["lensing_halo_action_gravity_or_holdout_payload_opened"]:
        raise RuntimeError("V19DL unexpectedly opened a sealed payload")
    return science, parent


def configure_parameter(parameter: Any, value: float, bounds: dict[str, Any]) -> None:
    parameter.val = value
    parameter.min = float(bounds["minimum"])
    parameter.max = float(bounds["maximum"])
    ui.thaw(parameter)


def fit_two_temperature(
    config: dict[str, Any],
    science: dict[str, Any],
    cluster: str,
    parent_fit: dict[str, Any],
) -> dict[str, Any]:
    source = Path(parent_fit["source_spectrum"])
    if sha256(source) != parent_fit["source_spectrum_sha256"]:
        raise RuntimeError(f"V19DM source changed: {source}")
    cluster_config = science["registered_workload"]["clusters"][cluster]
    fit_config = science["fit_sequence"]
    mixture = config["mixture"]
    ui.clean()
    atomic_data = INHERITED_FIT.configure_apec_data(ui)
    ui.set_xsabund(fit_config["abundance_table"].split(maxsplit=1)[0])
    ui.load_pha(1, str(source))
    ui.set_analysis(1, "energy", "counts")
    fit_lo, fit_hi = map(float, fit_config["fit_energy_keV"])
    ui.notice_id(1, fit_lo, fit_hi)
    data = ui.get_data(1)
    counts = float(np.sum(np.asarray(data.get_dep(filter=True), dtype=float)))
    bins = int(np.asarray(data.get_dep(filter=True)).size)
    exposure = float(data.exposure)
    ui.subtract(1)

    suffix = cluster.lower()
    absorption = ui.create_model_component("xstbabs", f"tbabs_v19dm_{suffix}")
    cool = ui.create_model_component("xsapec", f"cool_v19dm_{suffix}")
    hot = ui.create_model_component("xsapec", f"hot_v19dm_{suffix}")
    ui.set_source(1, absorption * (cool + hot))
    absorption.nH = float(cluster_config["galactic_nh_cm2"]) / 1.0e22
    ui.freeze(absorption.nH)
    redshift = float(cluster_config["redshift"])
    for component in (cool, hot):
        component.Redshift = redshift
        ui.freeze(component.Redshift)
    abundance = fit_config["integrated_abundance_solar"]
    configure_parameter(cool.Abundanc, float(abundance["initial"]), abundance)
    hot.Abundanc.link = cool.Abundanc
    temperature = fit_config["temperature_keV"]
    normalization = fit_config["normalization"]
    norm_total = float(parent_fit["parameters"]["normalization"])
    starts = []
    for index, start in enumerate(mixture["starts"]):
        configure_parameter(cool.kT, float(start["cool_keV"]), temperature)
        configure_parameter(hot.kT, float(start["hot_keV"]), temperature)
        configure_parameter(
            cool.norm, norm_total * float(start["cool_norm_fraction"]), normalization
        )
        configure_parameter(
            hot.norm,
            norm_total * (1.0 - float(start["cool_norm_fraction"])),
            normalization,
        )
        cool.Abundanc.val = float(abundance["initial"])
        try:
            result, attempts = INHERITED_FIT.run_fit(
                ui,
                cool,
                INHERITED_FIT.primary_optimization_method(
                    fit_config["optimization"]
                ),
                SherpaErr,
            )
            starts.append(
                {
                    "index": index,
                    "fit_completed": True,
                    "attempts": attempts,
                    "statistic": float(result.statval),
                    "dof": int(result.dof),
                    "cool_temperature_keV": float(cool.kT.val),
                    "hot_temperature_keV": float(hot.kT.val),
                    "abundance_solar": float(cool.Abundanc.val),
                    "cool_normalization": float(cool.norm.val),
                    "hot_normalization": float(hot.norm.val),
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
        raise RuntimeError(f"V19DM all two-temperature starts failed for {cluster}")
    best = min(successful, key=lambda row: row["statistic"])
    temperatures_and_norms = sorted(
        (
            (best["cool_temperature_keV"], best["cool_normalization"]),
            (best["hot_temperature_keV"], best["hot_normalization"]),
        )
    )
    cool_temperature, cool_norm = temperatures_and_norms[0]
    hot_temperature, hot_norm = temperatures_and_norms[1]
    total_norm = cool_norm + hot_norm
    norm_fractions = [cool_norm / total_norm, hot_norm / total_norm]
    reduced = best["statistic"] / best["dof"]
    one_statistic = float(parent_fit["fit"]["statval"])
    one_parameters = int(mixture["one_temperature_free_parameters"])
    two_parameters = int(mixture["two_temperature_free_parameters"])
    delta_aic = (best["statistic"] + 2 * two_parameters) - (
        one_statistic + 2 * one_parameters
    )
    delta_bic = (
        best["statistic"] + two_parameters * math.log(bins)
    ) - (one_statistic + one_parameters * math.log(bins))
    inside_bounds = (
        float(temperature["minimum"])
        < cool_temperature
        < float(temperature["maximum"])
        and float(temperature["minimum"])
        < hot_temperature
        < float(temperature["maximum"])
        and float(abundance["minimum"])
        < best["abundance_solar"]
        < float(abundance["maximum"])
        and all(
            float(normalization["minimum"])
            < value
            < float(normalization["maximum"])
            for value in (cool_norm, hot_norm)
        )
    )
    gates = {
        "finite_successful_fit": all(
            math.isfinite(value)
            for value in (
                best["statistic"],
                reduced,
                cool_temperature,
                hot_temperature,
                best["abundance_solar"],
                cool_norm,
                hot_norm,
                delta_aic,
                delta_bic,
            )
        ),
        "reduced_statistic_at_most_1_5": reduced
        <= float(config["decision"]["maximum_reduced_statistic"]),
        "all_free_parameters_strictly_inside_bounds": inside_bounds,
        "temperatures_separated": hot_temperature / cool_temperature
        >= float(mixture["minimum_temperature_ratio"]),
        "both_components_non_negligible": min(norm_fractions)
        >= float(mixture["minimum_normalization_fraction"]),
        "strong_bic_preference": delta_bic
        <= float(mixture["maximum_delta_bic_for_admission"]),
    }
    one_temperature_passed = bool(parent_fit["gates"]["all_passed"])
    two_temperature_admitted = all(gates.values())
    if one_temperature_passed:
        selected_model = "one_temperature"
        selected_abundance = float(parent_fit["parameters"]["abundance_solar"])
        minimal_adequate_passed = True
    elif two_temperature_admitted:
        selected_model = "two_temperature"
        selected_abundance = float(best["abundance_solar"])
        minimal_adequate_passed = True
    else:
        selected_model = "none"
        selected_abundance = None
        minimal_adequate_passed = False
    ui.clean()
    return {
        "cluster": cluster,
        "source_spectrum": str(source),
        "source_counts_0p5_7_keV": counts,
        "grouped_bins": bins,
        "source_exposure_s": exposure,
        "atomic_data": atomic_data,
        "one_temperature": {
            "passed": one_temperature_passed,
            "statistic": one_statistic,
            "dof": int(parent_fit["fit"]["dof"]),
            "reduced_statistic": float(parent_fit["fit"]["reduced_statistic"]),
            "abundance_solar": float(parent_fit["parameters"]["abundance_solar"]),
        },
        "two_temperature": {
            "starts": starts,
            "best_start_index": int(best["index"]),
            "statistic": float(best["statistic"]),
            "dof": int(best["dof"]),
            "reduced_statistic": reduced,
            "cool_temperature_keV": cool_temperature,
            "hot_temperature_keV": hot_temperature,
            "temperature_ratio": hot_temperature / cool_temperature,
            "abundance_solar": float(best["abundance_solar"]),
            "cool_normalization": cool_norm,
            "hot_normalization": hot_norm,
            "normalization_fractions": norm_fractions,
            "delta_aic_vs_one_temperature": delta_aic,
            "delta_bic_vs_one_temperature": delta_bic,
            "gates": gates,
            "admitted_when_one_temperature_fails": two_temperature_admitted,
        },
        "selection": {
            "model": selected_model,
            "abundance_solar": selected_abundance,
            "minimal_adequate_rule_passed": minimal_adequate_passed,
        },
    }


def execute(
    config: dict[str, Any], science: dict[str, Any], parent: dict[str, Any]
) -> dict[str, Any]:
    parent_integrated = {row["cluster"]: row for row in parent["integrated_fits"]}
    integrated = [
        fit_two_temperature(config, science, cluster, parent_integrated[cluster])
        for cluster in science["registered_workload"]["clusters"]
    ]
    regional = []
    for result in integrated:
        cluster = result["cluster"]
        abundance = result["selection"]["abundance_solar"]
        if abundance is None:
            regional.append(
                {
                    "cluster": cluster,
                    "fit_completed": False,
                    "reason": "no minimal adequate integrated abundance",
                    "gates": {"all_passed": False},
                }
            )
            continue
        regional.append(
            v19dj.v19x2.inherited_v19x.fit_spectrum(
                science,
                cluster,
                parent["combinations"][cluster]["regional"],
                float(abundance),
            )
        )
    gates = {
        "both_clusters_have_minimal_adequate_integrated_model": all(
            row["selection"]["minimal_adequate_rule_passed"] for row in integrated
        ),
        "both_registered_regions_pass_with_selected_abundance": all(
            row["fit_completed"] and row["gates"]["all_passed"] for row in regional
        ),
    }
    passed = all(gates.values())
    return {
        "status": (
            "minimal_thermal_mixture_diagnostic_passed_successor_may_be_frozen"
            if passed
            else "minimal_thermal_mixture_diagnostic_failed_no_successor_authorized"
        ),
        "aggregate_pass": passed,
        "integrated_model_selection": integrated,
        "regional_refits": regional,
        "gates": gates,
        "minimal_adequate_full_regional_successor_authorized": passed,
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
        science, parent = validate_frozen(config, Path(__file__).resolve())
        result = execute(config, science, parent)
    except Exception as exc:  # noqa: BLE001 - retain diagnostic terminal evidence
        result = {
            "status": "minimal_thermal_mixture_diagnostic_execution_failed",
            "execution_exception": f"{type(exc).__name__}: {exc}",
            "aggregate_pass": False,
            "minimal_adequate_full_regional_successor_authorized": False,
        }
    report = {
        "protocol_version": "SIGMA-V19DM-MINIMAL-THERMAL-MIXTURE-1.0.0",
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
