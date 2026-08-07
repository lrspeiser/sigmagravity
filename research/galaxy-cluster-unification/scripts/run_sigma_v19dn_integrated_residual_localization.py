#!/usr/bin/env python3
"""Localize the V19DM2 Bullet integrated-spectrum failure by frozen bands."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import run_sigma_v19dm2_statistic_parity_remediation as v19dm2

ROOT = Path(__file__).resolve().parents[1]
FIT = v19dm2.v19dm.v19dj.v19x2.inherited_v19x.fit_spectrum


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
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if sha256(runner) != config["implementation"]["runner_sha256"]:
        raise RuntimeError("V19DN runner changed after freeze")
    for name, item in config["parents"].items():
        path = ROOT / item["path"]
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"V19DN parent changed: {name}")
    parent_config_path = ROOT / config["parents"]["v19dm2_config"]["path"]
    parent_runner = ROOT / config["parents"]["v19dm2_runner"]["path"]
    science, v19dl_report, _ = v19dm2.validate_frozen(
        load_json(parent_config_path), parent_runner
    )
    parent_report = load_json(ROOT / config["parents"]["v19dm2_report"]["path"])
    if parent_report["status"] != (
        "statistic_parity_minimal_thermal_mixture_failed_no_successor_authorized"
    ):
        raise RuntimeError("V19DM2 no longer supplies the registered thermal failure")
    if parent_report["aggregate_pass"]:
        raise RuntimeError("V19DM2 unexpectedly passed")
    if parent_report["all_494_regions_run"]:
        raise RuntimeError("V19DM2 unexpectedly ran all regions")
    if parent_report["lensing_halo_action_gravity_or_holdout_payload_opened"]:
        raise RuntimeError("V19DM2 unexpectedly opened a sealed payload")
    if science["fit_sequence"]["statistic"] != "chi2xspecvar":
        raise RuntimeError("V19DN inherited statistic changed")
    return science, v19dl_report, parent_report


def execute(
    config: dict[str, Any], science: dict[str, Any], v19dl_report: dict[str, Any]
) -> dict[str, Any]:
    rows = []
    for cluster in science["registered_workload"]["clusters"]:
        combination = v19dl_report["combinations"][cluster]["integrated"]
        for band in config["diagnostic_bands"]:
            band_science = copy.deepcopy(science)
            band_science["fit_sequence"]["fit_energy_keV"] = [
                float(band["minimum_keV"]),
                float(band["maximum_keV"]),
            ]
            fit = FIT(band_science, cluster, combination, None)
            rows.append(
                {
                    "cluster": cluster,
                    "band_id": band["id"],
                    "interpretation": band["interpretation"],
                    "fit": fit,
                }
            )
    lookup = {(row["cluster"], row["band_id"]): row["fit"] for row in rows}

    def reduced(cluster: str, band: str) -> float:
        return float(lookup[(cluster, band)]["fit"]["reduced_statistic"])

    limit = float(config["interpretation_rules"]["adequate_reduced_statistic"])
    bullet_soft = reduced("BULLET", "soft_only_0p5_2")
    bullet_hard = reduced("BULLET", "hard_only_2_7")
    bullet_one_up = reduced("BULLET", "soft_edge_removed_1_7")
    classification = {
        "soft_only_failure": bullet_soft > limit,
        "hard_only_failure": bullet_hard > limit,
        "soft_edge_removal_recovers_adequacy": bullet_one_up <= limit,
        "broad_band_failure": bullet_soft > limit and bullet_hard > limit,
    }
    if classification["broad_band_failure"]:
        next_test = "observation_resolved_joint_fit_before_any_further_plasma_component"
    elif classification["soft_edge_removal_recovers_adequacy"]:
        next_test = "soft_background_and_calibration_audit"
    elif classification["hard_only_failure"]:
        next_test = "spatial_temperature_distribution_and_response_averaging_audit"
    else:
        next_test = "inspect_bandwise_residual_pattern_without_authorizing_production"
    return {
        "status": "integrated_residual_localization_completed",
        "aggregate_pass": True,
        "band_fits": rows,
        "bullet_classification": classification,
        "next_test": next_test,
        "full_regional_successor_authorized": False,
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
        science, v19dl_report, parent_report = validate_frozen(
            config, Path(__file__).resolve()
        )
        result = execute(config, science, v19dl_report)
        result["v19dm2_report_sha256"] = sha256(
            ROOT / config["parents"]["v19dm2_report"]["path"]
        )
        result["v19dm2_status"] = parent_report["status"]
    except Exception as exc:  # noqa: BLE001 - preserve terminal diagnostic evidence
        result = {
            "status": "integrated_residual_localization_execution_failed",
            "execution_exception": f"{type(exc).__name__}: {exc}",
            "aggregate_pass": False,
            "full_regional_successor_authorized": False,
        }
    report = {
        "protocol_version": "SIGMA-V19DN-INTEGRATED-RESIDUAL-LOCALIZATION-1.0.0",
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
