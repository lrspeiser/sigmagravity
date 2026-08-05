#!/usr/bin/env python3
"""Audit whether a spent cluster pair can identify a causal assembly coordinate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_sigma_v18c_collisionless_stress_maps import member_arrays

DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19a_assembly_history_readiness.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19a_assembly_history_readiness"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def one_sided_sigma(p_value: float) -> float:
    clipped = min(max(float(p_value), np.finfo(float).eps), 1.0 - np.finfo(float).eps)
    return float(norm.isf(clipped))


def dressler_shectman(
    x_kpc: np.ndarray,
    y_kpc: np.ndarray,
    velocity_km_s: np.ndarray,
    neighbor_count: int,
) -> tuple[float, np.ndarray]:
    positions = np.column_stack([x_kpc, y_kpc])
    separation_squared = np.sum(
        (positions[:, None, :] - positions[None, :, :]) ** 2,
        axis=2,
    )
    local_indices = np.argsort(separation_squared, axis=1, kind="stable")[
        :, : neighbor_count + 1
    ]
    global_mean = float(np.mean(velocity_km_s))
    global_dispersion = float(np.std(velocity_km_s, ddof=1))
    if global_dispersion <= 0.0:
        raise RuntimeError("global velocity dispersion is not positive")
    local_velocity = velocity_km_s[local_indices]
    local_mean = np.mean(local_velocity, axis=1)
    local_dispersion = np.std(local_velocity, axis=1, ddof=1)
    delta = np.sqrt(
        (neighbor_count + 1)
        / global_dispersion**2
        * ((local_mean - global_mean) ** 2 + (local_dispersion - global_dispersion) ** 2)
    )
    return float(np.sum(delta)), local_indices


def velocity_gradient(
    x_kpc: np.ndarray,
    y_kpc: np.ndarray,
    velocity_km_s: np.ndarray,
) -> dict[str, float | np.ndarray]:
    design = np.column_stack(
        [
            np.ones_like(x_kpc),
            (x_kpc - np.mean(x_kpc)) / 1000.0,
            (y_kpc - np.mean(y_kpc)) / 1000.0,
        ]
    )
    projection = np.linalg.pinv(design)
    coefficient = projection @ velocity_km_s
    prediction = design @ coefficient
    residual = velocity_km_s - prediction
    total = float(np.sum((velocity_km_s - np.mean(velocity_km_s)) ** 2))
    r_squared = 1.0 - float(np.sum(residual**2)) / total
    east, north = float(coefficient[1]), float(coefficient[2])
    return {
        "east_km_s_per_mpc": east,
        "north_km_s_per_mpc": north,
        "magnitude_km_s_per_mpc": math.hypot(east, north),
        "angle_east_of_north_deg": math.degrees(math.atan2(east, north)) % 360.0,
        "r_squared": r_squared,
        "projection": projection,
    }


def phase_space_control(
    members: dict[str, np.ndarray],
    protocol: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    x_kpc = np.asarray(members["x_kpc"], dtype=float)
    y_kpc = np.asarray(members["y_kpc"], dtype=float)
    velocity = np.asarray(members["velocity_km_s"], dtype=float)
    count = int(velocity.size)
    neighbor_count = max(2, round(math.sqrt(count)))
    observed_ds, local_indices = dressler_shectman(
        x_kpc, y_kpc, velocity, neighbor_count
    )
    observed_gradient = velocity_gradient(x_kpc, y_kpc, velocity)
    gradient_projection = np.asarray(observed_gradient.pop("projection"), dtype=float)

    rng = np.random.default_rng(seed)
    permutations = int(protocol["permutations"])
    ds_exceedances = 0
    gradient_exceedances = 0
    global_mean = float(np.mean(velocity))
    global_dispersion = float(np.std(velocity, ddof=1))
    for _ in range(permutations):
        shuffled = rng.permutation(velocity)
        local_velocity = shuffled[local_indices]
        local_mean = np.mean(local_velocity, axis=1)
        local_dispersion = np.std(local_velocity, axis=1, ddof=1)
        delta = np.sqrt(
            (neighbor_count + 1)
            / global_dispersion**2
            * (
                (local_mean - global_mean) ** 2
                + (local_dispersion - global_dispersion) ** 2
            )
        )
        ds_exceedances += int(float(np.sum(delta)) >= observed_ds)
        shuffled_gradient = gradient_projection @ shuffled
        shuffled_amplitude = math.hypot(
            float(shuffled_gradient[1]), float(shuffled_gradient[2])
        )
        gradient_exceedances += int(
            shuffled_amplitude >= float(observed_gradient["magnitude_km_s_per_mpc"])
        )

    p_ds = (ds_exceedances + 1) / (permutations + 1)
    p_gradient = (gradient_exceedances + 1) / (permutations + 1)
    minimum_p = 1.0 / (permutations + 1)

    angles = []
    bootstrap_resamples = int(protocol["bootstrap_resamples"])
    for _ in range(bootstrap_resamples):
        indices = rng.integers(0, count, size=count)
        boot = velocity_gradient(x_kpc[indices], y_kpc[indices], velocity[indices])
        angles.append(float(boot["angle_east_of_north_deg"]))
    angle = float(observed_gradient["angle_east_of_north_deg"])
    angular_difference = (
        np.asarray(angles, dtype=float) - angle + 180.0
    ) % 360.0 - 180.0
    half_angle = float(protocol["direction_stability_half_angle_deg"])

    return {
        "selected_members": count,
        "median_redshift": float(members["median_redshift"]),
        "global_velocity_mean_km_s": global_mean,
        "global_velocity_dispersion_km_s": global_dispersion,
        "dressler_shectman": {
            "neighbor_count_excluding_self": neighbor_count,
            "delta_sum": observed_ds,
            "delta_sum_per_member": observed_ds / count,
            "permutation_exceedances": ds_exceedances,
            "permutation_p_value": p_ds,
            "one_sided_sigma_equivalent": one_sided_sigma(p_ds),
        },
        "velocity_gradient": {
            **observed_gradient,
            "permutation_exceedances": gradient_exceedances,
            "permutation_p_value": p_gradient,
            "one_sided_sigma_equivalent": one_sided_sigma(p_gradient),
            "bootstrap_absolute_angle_error_deg": {
                "median": float(np.median(np.abs(angular_difference))),
                "p68": float(np.percentile(np.abs(angular_difference), 68.0)),
                "p95": float(np.percentile(np.abs(angular_difference), 95.0)),
            },
            "bootstrap_fraction_within_declared_half_angle": float(
                np.mean(np.abs(angular_difference) <= half_angle)
            ),
        },
        "permutation_resolution": {
            "permutations": permutations,
            "minimum_resolvable_p_value": minimum_p,
            "maximum_claimable_one_sided_sigma": one_sided_sigma(minimum_p),
            "meets_five_sigma_resolution": one_sided_sigma(minimum_p)
            >= float(protocol["minimum_claimable_significance_sigma"]),
        },
        "is_genuinely_time_ordered": False,
        "contains_transverse_velocity_or_depth_information": False,
    }


def validate_inputs(config_path: Path, config: dict[str, Any]) -> dict[str, str]:
    hashes = {"config": sha256(config_path)}
    for key in ("observability_gate", "member_readiness_config", "member_readiness_report"):
        path = ROOT / config["parents"][key]
        actual = sha256(path)
        if actual != config["parents"][f"{key}_sha256"]:
            raise RuntimeError(f"frozen {key} changed")
        hashes[key] = actual
    parent = json.loads(
        (ROOT / config["parents"]["member_readiness_config"]).read_text(encoding="utf-8")
    )
    for name, cluster in parent["clusters"].items():
        path = ROOT / cluster["member_catalog"]
        actual = sha256(path)
        if actual != cluster["member_catalog_sha256"]:
            raise RuntimeError(f"frozen {name} member catalog changed")
        hashes[f"{name}_member_catalog"] = actual
    return hashes


def run(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not config["status"].startswith("frozen after the v19 observability gate"):
        raise RuntimeError("v19A readiness protocol is not frozen")
    authorization = config["authorization"]
    if authorization["formula_selection_authorized"]:
        raise RuntimeError("readiness audit cannot select a formula")
    if authorization["lensing_or_halo_input_authorized"]:
        raise RuntimeError("readiness audit cannot use lensing or halo inputs")

    hashes = validate_inputs(config_path, config)
    readiness_config = json.loads(
        (ROOT / config["parents"]["member_readiness_config"]).read_text(encoding="utf-8")
    )
    controls: dict[str, Any] = {}
    for index, (name, cluster) in enumerate(readiness_config["clusters"].items()):
        members = member_arrays(
            cluster,
            readiness_config["universal_selection"],
            readiness_config["universal_stellar_weight"],
        )
        controls[name] = phase_space_control(
            members,
            config["instantaneous_phase_space_controls"],
            int(config["instantaneous_phase_space_controls"]["permutation_seed"])
            + index,
        )

    evidence = config["clusters"]
    minimum_members = int(config["gates"]["minimum_secure_members_each"])
    gate_results = {
        "member_phase_space_controls_available": all(
            row["selected_members"] >= minimum_members for row in controls.values()
        ),
        "one_common_time_ordered_observable": all(
            bool(row["direct_time_ordered_clock_available"])
            for row in evidence.values()
        ),
        "unique_causal_origin_both_clusters": all(
            bool(row["unique_causal_origin"]) for row in evidence.values()
        ),
        "five_sigma_history_statistic_both_clusters": all(
            row["accepted_history_statistic_sigma"] is not None
            and float(row["accepted_history_statistic_sigma"])
            >= float(config["gates"]["minimum_history_statistic_significance_sigma_each"])
            for row in evidence.values()
        ),
        "projection_uncertainty_ensemble_both_clusters": all(
            bool(row["projection_uncertainty_ensemble_available"])
            for row in evidence.values()
        ),
        "member_and_temperature_uncertainty_products_both_clusters": all(
            bool(row["member_redshift_uncertainty_column_available"])
            and bool(row["resolved_temperature_uncertainty_map_available"])
            for row in evidence.values()
        ),
        "equal_snapshot_different_history_sentinel_distinguished": False,
    }
    required = (
        "one_common_time_ordered_observable",
        "unique_causal_origin_both_clusters",
        "five_sigma_history_statistic_both_clusters",
        "projection_uncertainty_ensemble_both_clusters",
        "member_and_temperature_uncertainty_products_both_clusters",
        "equal_snapshot_different_history_sentinel_distinguished",
    )
    gate_results["history_source_construction_authorized"] = all(
        gate_results[key] for key in required
    )

    return {
        "status": "completed Sigma v19A causal-assembly readiness audit",
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "input_hashes": hashes,
        "instantaneous_phase_space_controls": controls,
        "external_history_evidence": evidence,
        "gate_results": gate_results,
        "decision": (
            "freeze and construct one universal causal-history source"
            if gate_results["history_source_construction_authorized"]
            else "do not construct a causal-history source from this spent pair; acquire a matched pair with identifiable clocks and projection ensembles"
        ),
        "failure_classification": (
            None
            if gate_results["history_source_construction_authorized"]
            else "data insufficiency and causal non-identifiability; not a falsification of causal-history physics"
        ),
        "instantaneous_controls_count_as_history": False,
        "formula_selected": False,
        "lensing_or_halo_input_used": False,
        "new_lensing_target_opened": False,
        "holdout_opened": False,
        "gravity_parameters_fit": 0,
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
