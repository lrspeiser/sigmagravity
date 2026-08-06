#!/usr/bin/env python3
"""Diagnose the frozen A2319 relative-line failure without reopening events."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "configs" / "sigma_v19cy_a2319_relative_line_failure_diagnosis.json"
)
BLOCK_BYTES = 4 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_inputs(
    config_path: Path = DEFAULT_CONFIG,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    config = load_json(config_path)
    if config.get("protocol_version") != (
        "SIGMA-V19CY-A2319-RELATIVE-LINE-FAILURE-DIAGNOSIS-1.0.0"
    ):
        raise RuntimeError("unexpected relative-line diagnosis protocol")
    if config.get("status") != (
        "frozen after the relative-line terminal gate failed and its report was "
        "hashed, before calculating a root-cause classification or authorizing a "
        "different spectral model"
    ):
        raise RuntimeError("relative-line diagnosis is not frozen")
    parents = {}
    for name in ("relative_line_report", "readiness_report", "calibration_report"):
        path = ROOT / config["parents"][name]
        if not path.is_file() or sha256(path) != config["parents"][f"{name}_sha256"]:
            raise RuntimeError(f"diagnosis parent changed: {path}")
        parents[name] = load_json(path)
    if parents["relative_line_report"].get("validation_or_holdout_accessed"):
        raise RuntimeError("relative-line parent opened sealed data")
    for key in (
        "read_event_or_energy_value",
        "refit_relative_template",
        "access_validation_or_holdout_assets",
        "open_lensing_halo_or_gravity_targets",
        "change_gravity_formula_or_parameters",
        "derive_or_select_action",
    ):
        if config["authorization"][key]:
            raise RuntimeError(f"sealed diagnosis boundary is open: {key}")
    return config, parents


def diagnose(
    config: dict[str, Any], parents: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    relative = parents["relative_line_report"]
    readiness = parents["readiness_report"]
    calibration = parents["calibration_report"]
    rules = config["classification_rules"]
    names = relative["primary_comparison"]["region_order"]

    calibration_preserved = (
        calibration.get("terminal_gate_passed")
        and len(calibration.get("applications", [])) == 3
        and all(item.get("passed") for item in calibration["applications"])
    )
    geometry_preserved = (
        readiness.get("terminal_gate_passed")
        and all(item.get("partition_exact") for item in readiness["branches"])
    )
    extrema_preserved = (
        relative["primary_comparison"]["most_blueshifted_region"] == "b_prime"
        and relative["primary_comparison"]["most_redshifted_region"] == "e_prime"
    )
    h_failures = sum(
        not item["optimizer_success"]
        for item in relative["windows"]["h_like"]["fits"].values()
    )
    he_h_differences = {
        name: abs(
            relative["windows"]["he_like"]["fits"][name][
                "velocity_relative_unweighted_mean_kms"
            ]
            - relative["windows"]["h_like"]["fits"][name][
                "velocity_relative_unweighted_mean_kms"
            ]
        )
        for name in names
    }
    maximum_he_h_disagreement = max(he_h_differences.values())
    temperatures = np.asarray(
        [config["published_detector_temperatures_keV"][name] for name in names],
        dtype=float,
    )
    temperature_span = float(np.ptp(temperatures))
    primary_uncertainties = {
        name: float(
            relative["primary_bootstrap"]["regions"][name]["total_uncertainty_kms"]
        )
        for name in names
    }
    warnings = {
        "sparse_h_like_optimizer_failures": h_failures
        >= int(rules["minimum_h_like_optimizer_failures_for_sparse_band_warning"]),
        "he_h_shape_disagreement": maximum_he_h_disagreement
        >= float(rules["minimum_maximum_he_h_velocity_disagreement_kms_for_shape_warning"]),
        "published_temperature_span": temperature_span
        >= float(rules["minimum_published_temperature_span_keV_for_line_ratio_warning"]),
        "reduced_exposure_uncertainty": max(primary_uncertainties.values())
        >= float(rules["minimum_primary_region_uncertainty_kms_for_reduced_exposure_warning"]),
    }
    authorization = (
        not relative.get("terminal_gate_passed")
        and calibration_preserved
        and geometry_preserved
        and extrema_preserved
        and sum(warnings.values()) >= 2
    )
    return {
        "parent_relative_line_gate_failed": not relative.get("terminal_gate_passed"),
        "calibration_execution_evidence_preserved": calibration_preserved,
        "region_geometry_evidence_preserved": geometry_preserved,
        "extreme_topology_evidence_preserved": extrema_preserved,
        "h_like_optimizer_failures": h_failures,
        "he_h_absolute_velocity_differences_kms": he_h_differences,
        "maximum_he_h_velocity_disagreement_kms": maximum_he_h_disagreement,
        "published_detector_temperature_span_keV": temperature_span,
        "primary_total_uncertainties_kms": primary_uncertainties,
        "warnings": warnings,
        "supported_classification": (
            "response_free_shared_template_identifiability_failure_with_reduced_exposure_noise"
            if authorization
            else "failure_not_sufficiently_diagnosed"
        ),
        "calibration_failure_ruled_out": False,
        "calibration_failure_currently_disfavored": bool(
            calibration_preserved and extrema_preserved
        ),
        "authorize_response_aware_development_protocol": authorization,
    }


def build_report(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config, parents = validate_inputs(config_path)
    diagnosis = diagnose(config, parents)
    report = {
        "protocol_version": config["protocol_version"],
        "status": "a2319_relative_line_failure_diagnosed_from_frozen_reports",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "parent_hashes": {
            name: sha256(ROOT / config["parents"][name])
            for name in ("relative_line_report", "readiness_report", "calibration_report")
        },
        "diagnosis": diagnosis,
        "event_or_energy_value_read": False,
        "relative_template_refit": False,
        "validation_or_holdout_accessed": False,
        "decision": (
            "authorize_separately_frozen_response_aware_bapec_development_protocol"
            if diagnosis["authorize_response_aware_development_protocol"]
            else "do_not_advance_spectral_model"
        ),
        "allowed_next_model": config["allowed_next_model"],
        "claim_boundary": [
            "The failed empirical relative-line model remains failed and is not retuned.",
            "The diagnosis does not prove the gain is correct; it makes calibration failure less favored than template identifiability for the observed pattern.",
            "Only a response-aware development fit is authorized; validation and holdout data remain sealed."
        ],
    }
    path = ROOT / config["paths"]["report"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = build_report()
    print(json.dumps(result, indent=2, sort_keys=True))
