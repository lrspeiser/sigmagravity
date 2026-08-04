"""Audit the flat and standard-PPN skeleton of Sigma v17K before data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def carrier_coefficients(epsilon: float) -> dict[str, float]:
    value = float(epsilon)
    if not math.isfinite(value) or value <= 0.0 or value >= 0.5:
        raise ValueError("epsilon must lie in 0<epsilon<1/2")
    c_1 = value
    c_2 = value / (1.0 - 2.0 * value)
    c_3 = -value
    c_4 = 0.0
    return {
        "c_1": c_1,
        "c_2": c_2,
        "c_3": c_3,
        "c_4": c_4,
        "c_13": c_1 + c_3,
        "c_14": c_1 + c_4,
        "c_123": c_1 + c_2 + c_3,
    }


def mode_speeds(coefficients: dict[str, float]) -> dict[str, float]:
    c_1 = coefficients["c_1"]
    c_2 = coefficients["c_2"]
    c_3 = coefficients["c_3"]
    c_13 = coefficients["c_13"]
    c_14 = coefficients["c_14"]
    c_123 = coefficients["c_123"]
    return {
        "spin_2_squared": 1.0 / (1.0 - c_13),
        "spin_1_squared": (2.0 * c_1 - c_1**2 + c_3**2) / (2.0 * c_14 * (1.0 - c_13)),
        "spin_0_squared": c_123 * (2.0 - c_14) / (c_14 * (1.0 - c_13) * (2.0 + c_13 + 3.0 * c_2)),
    }


def standard_ppn(coefficients: dict[str, float]) -> dict[str, float]:
    c_1 = coefficients["c_1"]
    c_2 = coefficients["c_2"]
    c_3 = coefficients["c_3"]
    c_4 = coefficients["c_4"]
    c_14 = coefficients["c_14"]
    c_123 = coefficients["c_123"]
    alpha_1 = -8.0 * (c_3**2 + c_1 * c_4) / (2.0 * c_1 - c_1**2 + c_3**2)
    alpha_2 = alpha_1 / 2.0 - (c_1 + 2.0 * c_3 - c_4) * (2.0 * c_1 + 3.0 * c_2 + c_3 + c_4) / (
        c_123 * (2.0 - c_14)
    )
    return {"gamma": 1.0, "beta": 1.0, "alpha_1": alpha_1, "alpha_2": alpha_2}


def diagnostics(epsilon: float, bounds: dict[str, Any]) -> dict[str, Any]:
    coefficients = carrier_coefficients(epsilon)
    speeds = mode_speeds(coefficients)
    ppn = standard_ppn(coefficients)
    spin_1_energy = (
        2.0 * coefficients["c_1"] - coefficients["c_1"] ** 2 + coefficients["c_3"] ** 2
    ) / (1.0 - coefficients["c_13"])
    spin_0_energy = coefficients["c_14"] * (2.0 - coefficients["c_14"])
    newtonian_denominator = 1.0 - coefficients["c_14"] / 2.0
    newtonian_ratio = 1.0 / newtonian_denominator
    maximum_speed_deviation = max(abs(value - 1.0) for value in speeds.values())
    gates = {
        "c_13_pass": abs(coefficients["c_13"]) <= bounds["maximum_abs_c_13"],
        "all_modes_luminal_pass": maximum_speed_deviation
        <= bounds["maximum_mode_speed_squared_deviation_from_one"],
        "spin_1_energy_pass": spin_1_energy > bounds["minimum_spin_1_energy_sign_proxy"],
        "spin_0_energy_pass": spin_0_energy > bounds["minimum_spin_0_energy_sign_proxy"],
        "alpha_1_pass": abs(ppn["alpha_1"]) <= bounds["maximum_abs_alpha_1"],
        "alpha_2_pass": abs(ppn["alpha_2"]) <= bounds["maximum_abs_alpha_2"],
        "c_14_pass": coefficients["c_14"] <= bounds["maximum_c_14"],
        "newtonian_shift_pass": abs(newtonian_ratio - 1.0)
        <= bounds["maximum_newtonian_coupling_fractional_shift"],
    }
    return {
        "epsilon": float(epsilon),
        "coefficients": coefficients,
        "mode_speeds": speeds,
        "maximum_mode_speed_squared_deviation_from_one": maximum_speed_deviation,
        "standard_aether_ppn_at_X_zero": ppn,
        "energy_sign_proxies": {"spin_1": spin_1_energy, "spin_0": spin_0_energy},
        "newtonian_coupling": {
            "denominator_1_minus_c14_over_2": newtonian_denominator,
            "G_N_over_G": newtonian_ratio,
            "fractional_shift": newtonian_ratio - 1.0,
        },
        "gates": {**gates, "all_selection_gates_pass": all(gates.values())},
    }


def family_identity_scan(config: dict[str, Any]) -> dict[str, Any]:
    audit = config["executable_audit"]
    bounds = config["selection_bounds"]
    values = np.geomspace(
        float(audit["epsilon_log_minimum"]),
        float(audit["epsilon_log_maximum"]),
        int(audit["epsilon_samples"]),
    )
    rows = [diagnostics(float(value), bounds) for value in values]
    identity_errors = {
        name: max(abs(row["mode_speeds"][name] - 1.0) for row in rows)
        for name in ("spin_2_squared", "spin_1_squared", "spin_0_squared")
    }
    alpha_2_error = max(abs(row["standard_aether_ppn_at_X_zero"]["alpha_2"]) for row in rows)
    alpha_1_identity_error = max(
        abs(row["standard_aether_ppn_at_X_zero"]["alpha_1"] + 4.0 * row["epsilon"]) for row in rows
    )
    tolerance = float(audit["identity_tolerance"])
    return {
        "epsilon_minimum": float(values[0]),
        "epsilon_maximum": float(values[-1]),
        "sample_count": len(rows),
        "maximum_mode_identity_errors": identity_errors,
        "maximum_alpha_2_absolute_error": alpha_2_error,
        "maximum_alpha_1_plus_4epsilon_error": alpha_1_identity_error,
        "symbolic_relations_verified_numerically": max(
            *identity_errors.values(), alpha_2_error, alpha_1_identity_error
        )
        <= tolerance,
        "selection_pass_count": sum(row["gates"]["all_selection_gates_pass"] for row in rows),
    }


def build_report(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    epsilon = float(config["fixed_coefficients"]["epsilon_aether"])
    fixed = diagnostics(epsilon, config["selection_bounds"])
    family = family_identity_scan(config)
    no_data = not any(
        config["authorization"][key]
        for key in ("observational_data_opened", "empirical_fit_authorized", "holdout_authorized")
    )
    gate_results = {
        **fixed["gates"],
        "family_identities_pass": family["symbolic_relations_verified_numerically"],
        "no_observational_data_used": no_data,
        "one_physical_metric": config["complexity"]["one_physical_metric"],
        "constant_budget_pass": config["complexity"]["maximum_candidate_constants"] <= 5,
        "holdout_authorized": False,
    }
    selection_pass = all(
        value for key, value in gate_results.items() if key != "holdout_authorized"
    )
    return {
        "report_version": config["protocol_version"],
        "status": "passed_carrier_selection" if selection_pass else "failed_carrier_selection",
        "generated_utc": datetime.now(UTC).isoformat(),
        "protocol": config_path.relative_to(ROOT).as_posix(),
        "protocol_sha256": sha256(config_path),
        "observational_data_opened": False,
        "empirical_fit_performed": False,
        "fixed_carrier": fixed,
        "family_identity_scan": family,
        "gates": {**gate_results, "carrier_selection_pass": selection_pass},
        "limitations": {
            "full_derivative_metric_constraint_matrix_computed": False,
            "tilted_or_matter_background_characteristics_computed": False,
            "full_Sigma_PPN_solution_computed": False,
            "strong_field_sensitivities_computed": False,
            "target_blind_aether_acceleration_map_computed": False,
            "halo_radius_or_lensing_prediction_made": False,
        },
        "decision": {
            "outcome": (
                "advance_to_localized_tilted_constraint_gate_only"
                if selection_pass
                else "retire_v17K_carrier"
            ),
            "scope": (
                "A pass selects a known healthy flat-vacuum aether skeleton for the "
                "Sigma pressure metric. It does not validate the derivative-dependent "
                "matter coupling, Solar solution, strong-field behavior, or halo data."
            ),
            "novelty": (
                "The coefficient family is prior art. Only a successful universal "
                "pressure/susceptibility-to-halo prediction could be project-specific."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v17k_luminal_aether_pressure_carrier.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "sigma_v17k_luminal_aether_pressure_carrier",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = build_report(args.config, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
