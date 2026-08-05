#!/usr/bin/env python3
"""Run the target-blind Sigma v19B replacement-cluster source screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19b_replacement_cluster_screen.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19b_replacement_cluster_screen"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def history_significance_sigma(statistic: dict[str, Any]) -> float | None:
    """Return the conservative one-sided significance against no shock."""
    basis = statistic["basis"]
    value = statistic["measured_value"]
    if basis == "mach_excess_above_unity":
        uncertainty = statistic["uncertainty_toward_no_shock"]
        if value is None or uncertainty is None or float(uncertainty) <= 0.0:
            raise ValueError("Mach significance requires a value and positive uncertainty")
        return (float(value) - 1.0) / float(uncertainty)
    if basis == "published_detection_confidence":
        if value is None or not 0.5 < float(value) < 1.0:
            raise ValueError("detection confidence must be strictly between 0.5 and 1")
        return NormalDist().inv_cdf(float(value))
    if basis == "none":
        return None
    raise ValueError(f"unknown history statistic basis: {basis}")


def at_least(value: int | None, minimum: int) -> bool:
    return value is not None and int(value) >= minimum


def candidate_result(candidate: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    significance = history_significance_sigma(candidate["history_statistic"])
    members = candidate["member_sample"]["secure_spectroscopic_members"]
    source_archives = candidate["source_archives"]
    geometry = candidate["assembly_geometry"]
    lensing = candidate["later_lensing_suitability_metadata"]
    local = candidate["local_source_products"]

    source_gates = {
        "direct_primary_merger_front": bool(candidate["direct_primary_merger_front"]),
        "unique_merger_attribution": bool(candidate["unique_merger_attribution"]),
        "five_sigma_history_statistic": significance is not None
        and significance >= float(gates["minimum_history_statistic_significance_sigma"]),
        "minimum_secure_members": at_least(
            members, int(gates["minimum_secure_spectroscopic_members"])
        ),
        "mach_or_speed_uncertainty": bool(
            candidate["mach_or_speed_uncertainty_available"]
        ),
        "published_projection_constraint": bool(
            geometry["published_projection_constraint_available"]
        ),
        "time_since_passage_constraint": bool(
            geometry["time_since_passage_constraint_available"]
        ),
        "public_source_archives": bool(
            source_archives["public_source_archives_identified"]
            and source_archives["member_table_source_archive_identified"]
            and source_archives["xray_archive_identified"]
        ),
    }
    acquisition_eligible = all(source_gates.values())

    final_sample_metadata_gates = {
        "minimum_image_families": at_least(
            lensing["reported_image_families"],
            int(gates["final_sample_minimum_image_families"]),
        ),
        "minimum_spectroscopic_families": at_least(
            lensing["reported_spectroscopic_families"],
            int(gates["final_sample_minimum_spectroscopic_families"]),
        ),
        "minimum_images": at_least(
            lensing["minimum_reported_image_instances"],
            int(gates["final_sample_minimum_images"]),
        ),
        "target_payload_remained_sealed": not bool(
            lensing["coordinates_or_model_read_during_screen"]
        ),
    }

    construction_gates = {
        "acquisition_eligible": acquisition_eligible,
        "local_member_table_with_uncertainties": bool(
            local["member_table_with_uncertainties"]
        ),
        "local_resolved_gas_uncertainty_product": bool(
            local["resolved_gas_uncertainty_product"]
        ),
        "local_projection_uncertainty_ensemble": bool(
            local["projection_uncertainty_ensemble"]
        ),
        "local_front_position_with_uncertainty": bool(
            local["front_position_with_uncertainty"]
        ),
    }

    return {
        "name": candidate["name"],
        "history_statistic_basis": candidate["history_statistic"]["basis"],
        "history_statistic_sigma": significance,
        "secure_spectroscopic_members": members,
        "source_acquisition_gates": source_gates,
        "source_acquisition_eligible": acquisition_eligible,
        "final_sample_lensing_metadata_gates": final_sample_metadata_gates,
        "final_sample_lensing_metadata_ready": all(
            final_sample_metadata_gates.values()
        ),
        "source_construction_gates": construction_gates,
        "source_construction_ready": all(construction_gates.values()),
        "time_constraint_is_assumption_independent": bool(
            geometry["time_constraint_is_assumption_independent"]
        ),
        "declared_clock_assumption": geometry["declared_clock_assumption"],
    }


def validate_inputs(config_path: Path, config: dict[str, Any]) -> dict[str, str]:
    hashes = {"config": sha256(config_path)}
    for key in ("assembly_readiness_config", "assembly_readiness_report"):
        path = ROOT / config["parents"][key]
        actual = sha256(path)
        if actual != config["parents"][f"{key}_sha256"]:
            raise RuntimeError(f"frozen {key} changed")
        hashes[key] = actual
    return hashes


def run(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not config["status"].startswith("frozen after a source-only"):
        raise RuntimeError("v19B candidate screen is not frozen")
    if not config["blindness"]["all_replacement_lensing_targets_remain_sealed"]:
        raise RuntimeError("replacement lensing targets must remain sealed")
    authorization = config["authorization"]
    if authorization["formula_selection_authorized"]:
        raise RuntimeError("candidate screening cannot select a gravity formula")
    if authorization["lensing_or_halo_payload_access_authorized"]:
        raise RuntimeError("candidate screening cannot access lensing or halo payloads")

    hashes = validate_inputs(config_path, config)
    results = {
        key: candidate_result(candidate, config["gates"])
        for key, candidate in config["candidates"].items()
    }
    selected = sorted(
        key for key, result in results.items() if result["source_acquisition_eligible"]
    )
    declared = sorted(config["preferred_development_pair"])
    if selected != declared:
        raise RuntimeError(
            "preferred development pair does not equal the source-gate survivors"
        )

    minimum_selected = int(config["gates"]["minimum_selected_clusters"])
    selected_pair_identified = len(selected) >= minimum_selected
    source_construction_ready = selected_pair_identified and all(
        results[key]["source_construction_ready"] for key in selected
    )
    final_sample_metadata_ready = selected_pair_identified and all(
        results[key]["final_sample_lensing_metadata_ready"] for key in selected
    )
    assumption_independent_clock_pair = selected_pair_identified and all(
        results[key]["time_constraint_is_assumption_independent"] for key in selected
    )

    return {
        "status": "completed Sigma v19B target-blind replacement-cluster screen",
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "input_hashes": hashes,
        "candidate_results": results,
        "selected_development_pair": selected,
        "gate_results": {
            "source_gate_survivors_equal_preregistered_pair": selected == declared,
            "replacement_development_pair_identified": selected_pair_identified,
            "selected_pair_has_assumption_independent_clocks": assumption_independent_clock_pair,
            "selected_pair_final_lensing_sample_metadata_ready": final_sample_metadata_ready,
            "selected_pair_local_source_products_ready": source_construction_ready,
            "source_archive_acquisition_authorized": selected_pair_identified,
            "source_construction_authorized": source_construction_ready,
            "all_replacement_lensing_targets_remained_sealed": True,
        },
        "decision": (
            "construct the frozen causal source"
            if source_construction_ready
            else "acquire and audit source-only Bullet Cluster and Abell 2146 products; do not construct a causal source or open their lensing targets yet"
        ),
        "scientific_interpretation": (
            "Two clusters pass the source-side acquisition screen, but their local uncertainty products and assumption-aware projection ensembles are not yet assembled. Abell 2146 also lacks a reported spectroscopic lensed family, so this pair is for mechanism development rather than the final four-cluster validation sample."
        ),
        "failure_classification": "data acquisition and uncertainty-identification work remains; no causal-history physics test has yet occurred",
        "formula_selected": False,
        "gravity_parameters_fit": 0,
        "lensing_or_halo_payload_used": False,
        "new_lensing_target_opened": False,
        "holdout_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(args.config)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
