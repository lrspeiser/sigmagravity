"""Audit the flat-vacuum kinetic and characteristic sector of Sigma v17H/v17I."""

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


def aether_coefficients(c_u: float, acceleration_action_sign: float = -1.0) -> dict[str, float]:
    """Map the quadratic action to the standard Einstein-aether invariant basis."""

    values = (float(c_u), float(acceleration_action_sign))
    if any(not math.isfinite(value) for value in values):
        raise ValueError("aether coefficients must be finite")
    c_1 = values[0]
    c_2 = 0.0
    c_3 = -values[0]
    c_4 = values[1]
    return {
        "c_1": c_1,
        "c_2": c_2,
        "c_3": c_3,
        "c_4": c_4,
        "c_13": c_1 + c_3,
        "c_14": c_1 + c_4,
        "c_123": c_1 + c_2 + c_3,
    }


def quadratic_coefficients(c_u: float, acceleration_action_sign: float = -1.0) -> dict[str, float]:
    coefficients = aether_coefficients(c_u, acceleration_action_sign)
    return {
        "transverse_kinetic": coefficients["c_14"],
        "transverse_gradient": float(c_u),
        "spin_1_energy_numerator": 2.0 * float(c_u),
    }


def mode_speeds(
    c_u: float,
    acceleration_action_sign: float = -1.0,
    *,
    zero_tolerance: float = 1e-12,
) -> dict[str, float | None]:
    coefficients = aether_coefficients(c_u, acceleration_action_sign)
    c_1 = coefficients["c_1"]
    c_2 = coefficients["c_2"]
    c_3 = coefficients["c_3"]
    c_13 = coefficients["c_13"]
    c_14 = coefficients["c_14"]
    c_123 = coefficients["c_123"]
    tensor_denominator = 1.0 - c_13
    vector_denominator = 2.0 * c_14 * tensor_denominator
    scalar_denominator = c_14 * tensor_denominator * (2.0 + c_13 + 3.0 * c_2)
    tensor = None if abs(tensor_denominator) <= zero_tolerance else 1.0 / tensor_denominator
    vector = (
        None
        if abs(vector_denominator) <= zero_tolerance
        else (2.0 * c_1 - c_1**2 + c_3**2) / vector_denominator
    )
    scalar = (
        None
        if abs(scalar_denominator) <= zero_tolerance
        else c_123 * (2.0 - c_14) / scalar_denominator
    )
    return {"spin_2_squared": tensor, "spin_1_squared": vector, "spin_0_squared": scalar}


def exact_time_lagrangian(velocity: float, c_u: float, acceleration_action_sign: float) -> float:
    """Dimensionless transverse time Lagrangian with the square root rationalized."""

    velocity_squared = float(velocity) ** 2
    born_infeld = (
        float(acceleration_action_sign)
        * velocity_squared
        / (math.sqrt(1.0 + velocity_squared) + 1.0)
    )
    return 0.5 * float(c_u) * velocity_squared + born_infeld


def five_point_second_derivative(function: Any, step: float) -> float:
    return float(
        (
            -function(2.0 * step)
            + 16.0 * function(step)
            - 30.0 * function(0.0)
            + 16.0 * function(-step)
            - function(-2.0 * step)
        )
        / (12.0 * step**2)
    )


def classify_point(
    c_u: float,
    acceleration_action_sign: float,
    *,
    zero_tolerance: float,
    cone_tolerance: float,
) -> dict[str, Any]:
    coefficients = aether_coefficients(c_u, acceleration_action_sign)
    quadratic = quadratic_coefficients(c_u, acceleration_action_sign)
    speeds = mode_speeds(
        c_u,
        acceleration_action_sign,
        zero_tolerance=zero_tolerance,
    )
    vector = speeds["spin_1_squared"]
    scalar = speeds["spin_0_squared"]
    tensor = speeds["spin_2_squared"]
    gates = {
        "transverse_kinetic_positive": quadratic["transverse_kinetic"] > zero_tolerance,
        "transverse_gradient_positive": quadratic["transverse_gradient"] > zero_tolerance,
        "spin_1_energy_numerator_positive": quadratic["spin_1_energy_numerator"] > zero_tolerance,
        "spin_1_finite_nonnegative": vector is not None and vector >= 0.0,
        "spin_1_within_physical_cone": vector is not None and vector <= 1.0 + cone_tolerance,
        "spin_0_finite_positive": scalar is not None and scalar > zero_tolerance,
        "spin_2_luminal": tensor is not None and abs(tensor - 1.0) <= 1e-15,
    }
    return {
        "c_U": float(c_u),
        "coefficients": coefficients,
        "quadratic": quadratic,
        "mode_speeds": speeds,
        "newtonian_coupling_denominator": 1.0 - coefficients["c_14"] / 2.0,
        "gates": {**gates, "all_mode_gates_pass": all(gates.values())},
    }


def scan_values(minimum: float, maximum: float, samples_per_sign: int) -> np.ndarray:
    if minimum <= 0.0 or maximum <= minimum or samples_per_sign < 3:
        raise ValueError("invalid c_U scan")
    positive = np.geomspace(minimum, maximum, samples_per_sign)
    values = np.unique(np.r_[-positive[::-1], 0.0, positive, 1.0])
    return values


def audit_sign(config: dict[str, Any], acceleration_action_sign: float) -> dict[str, Any]:
    audit = config["executable_audit"]
    values = scan_values(
        float(audit["c_U_log_absolute_minimum"]),
        float(audit["c_U_log_absolute_maximum"]),
        int(audit["c_U_samples_per_sign"]),
    )
    rows = [
        classify_point(
            float(value),
            acceleration_action_sign,
            zero_tolerance=float(audit["characteristic_zero_tolerance"]),
            cone_tolerance=float(audit["physical_cone_tolerance"]),
        )
        for value in values
    ]
    vector_passes = [
        row
        for row in rows
        if all(
            row["gates"][name]
            for name in (
                "transverse_kinetic_positive",
                "transverse_gradient_positive",
                "spin_1_energy_numerator_positive",
                "spin_1_finite_nonnegative",
                "spin_1_within_physical_cone",
                "spin_2_luminal",
            )
        )
    ]
    full_passes = [row for row in rows if row["gates"]["all_mode_gates_pass"]]
    return {
        "acceleration_action_sign_sigma_A": float(acceleration_action_sign),
        "sample_count": len(rows),
        "vector_gate_pass_count": len(vector_passes),
        "full_mode_gate_pass_count": len(full_passes),
        "vector_gate_pass_c_U_range": (
            [vector_passes[0]["c_U"], vector_passes[-1]["c_U"]] if vector_passes else None
        ),
        "representative_rows": [
            classify_point(
                value,
                acceleration_action_sign,
                zero_tolerance=float(audit["characteristic_zero_tolerance"]),
                cone_tolerance=float(audit["physical_cone_tolerance"]),
            )
            for value in (-10.0, -1.0, -0.1, 0.0, 0.1, 0.5, 0.9, 1.0, 1.1, 2.0, 3.0, 10.0)
        ],
    }


def build_report(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    audit = config["executable_audit"]
    frozen = config["frozen_action"]
    c_u = float(frozen["fixed_maxwell_aether_coefficient_c_U"])
    sign = float(frozen["acceleration_action_sign_sigma_A"])
    step = float(audit["finite_difference_step"])
    numerical_time_hessian = five_point_second_derivative(
        lambda velocity: exact_time_lagrangian(velocity, c_u, sign), step
    )
    analytic_time_hessian = quadratic_coefficients(c_u, sign)["transverse_kinetic"]
    quadratic_error = abs(numerical_time_hessian - analytic_time_hessian)
    frozen_point = classify_point(
        c_u,
        sign,
        zero_tolerance=float(audit["characteristic_zero_tolerance"]),
        cone_tolerance=float(audit["physical_cone_tolerance"]),
    )
    frozen_scan = audit_sign(config, sign)
    flipped_scan = audit_sign(config, 1.0)
    gate_results = {
        "quadratic_expansion_verified": quadratic_error
        <= float(audit["maximum_quadratic_coefficient_error"]),
        "frozen_transverse_kinetic_pass": frozen_point["gates"]["transverse_kinetic_positive"],
        "frozen_transverse_gradient_pass": frozen_point["gates"]["transverse_gradient_positive"],
        "frozen_spin_1_characteristic_pass": frozen_point["gates"]["spin_1_finite_nonnegative"]
        and frozen_point["gates"]["spin_1_within_physical_cone"],
        "frozen_spin_0_characteristic_pass": frozen_point["gates"]["spin_0_finite_positive"],
        "frozen_tensor_speed_pass": frozen_point["gates"]["spin_2_luminal"],
        "any_c_U_passes_frozen_sign_vector_gate": frozen_scan["vector_gate_pass_count"] > 0,
        "any_c_U_passes_frozen_sign_full_mode_gate": frozen_scan["full_mode_gate_pass_count"] > 0,
        "holdout_authorized": False,
    }
    theory_pass = all(
        gate_results[name]
        for name in (
            "quadratic_expansion_verified",
            "frozen_transverse_kinetic_pass",
            "frozen_transverse_gradient_pass",
            "frozen_spin_1_characteristic_pass",
            "frozen_spin_0_characteristic_pass",
            "frozen_tensor_speed_pass",
        )
    )
    return {
        "report_version": config["protocol_version"],
        "status": "passed_flat_kinetic_gate" if theory_pass else "failed_flat_kinetic_gate",
        "generated_utc": datetime.now(UTC).isoformat(),
        "protocol": config_path.relative_to(ROOT).as_posix(),
        "protocol_sha256": sha256(config_path),
        "observational_data_opened": False,
        "empirical_fit_performed": False,
        "quadratic_expansion": {
            "finite_difference_step": step,
            "numerical_time_hessian": numerical_time_hessian,
            "analytic_time_hessian_c_U_plus_sigma_A": analytic_time_hessian,
            "absolute_error": quadratic_error,
            "verified": gate_results["quadratic_expansion_verified"],
        },
        "frozen_point": frozen_point,
        "frozen_sign_scan": frozen_scan,
        "sign_flipped_control_scan": flipped_scan,
        "analytic_no_go": {
            "frozen_sign_vector_partition_is_exhaustive": True,
            "real_c_U_vector_pass_exists": False,
            "reason": (
                "c_U<0 has wrong-sign gradient/energy numerator; c_U=0 is degenerate; "
                "0<c_U<1 has K_T<0 and s_1^2<0; c_U=1 has K_T=0 and a singular "
                "spin-1 characteristic; c_U>1 has s_1^2=1+1/(c_U-1)>1."
            ),
            "independent_scalar_obstruction": (
                "c_123=0 for every c_U, so the aether spin-0 characteristic is zero "
                "or singular independently of the vector-cone rule."
            ),
        },
        "gates": {**gate_results, "flat_kinetic_gate_pass": theory_pass},
        "decision": {
            "outcome": (
                "advance_v17H_v17I_to_tilted_backgrounds"
                if theory_pass
                else "retire_frozen_v17H_v17I_susceptibility_action"
            ),
            "supersedes": (
                "The v17H reduced static acceleration-Hessian statement that the Maxwell "
                "term supplied a positive floor. It did not retain the Lorentzian action "
                "sign in the time-dependent quadratic block."
            ),
            "scope": (
                "This rejects the frozen Maxwell plus negative Born-Infeld acceleration "
                "completion before data. It does not reject a pressure source, a baryon-derived "
                "halo scale, or every possible susceptibility carrier."
            ),
            "sign_flip_is_not_a_rescue": (
                "The displayed sign flip is a distinct control. Although it repairs the vector "
                "quadratic sign for c_U>0, c_123 remains zero; it is not advanced or fit."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v17j_flat_kinetic_gate.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "sigma_v17j_flat_kinetic_gate",
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
