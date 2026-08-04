"""Verify the algebraic matter variation in the localized v17H action."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chi(z: float) -> float:
    if not np.isfinite(z) or z < 0.0:
        raise ValueError("Z must be finite and non-negative")
    return float((1.0 + z) ** -0.25)


def chi_derivative(z: float) -> float:
    if not np.isfinite(z) or z < 0.0:
        raise ValueError("Z must be finite and non-negative")
    return float(-0.25 * (1.0 + z) ** -1.25)


def physical_metric(
    metric: np.ndarray,
    u_covector: np.ndarray,
    acceleration_covector: np.ndarray,
    scalar_x: float,
    alpha: float,
    a_sigma: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    g = np.asarray(metric, dtype=float)
    u = np.asarray(u_covector, dtype=float)
    acceleration = np.asarray(acceleration_covector, dtype=float)
    if g.shape != (4, 4) or u.shape != (4,) or acceleration.shape != (4,):
        raise ValueError("localized metric inputs have incompatible shapes")
    inverse = np.linalg.inv(g)
    acceleration_up = inverse @ acceleration
    z = float(acceleration @ acceleration_up / a_sigma**2)
    susceptibility = chi(z)
    q = float(alpha * susceptibility * scalar_x)
    exponential = float(np.exp(2.0 * q))
    physical = exponential * (g + 2.0 * q * np.outer(u, u))
    d_metric_dq = 2.0 * exponential * (
        g + (2.0 * q + 1.0) * np.outer(u, u)
    )
    return physical, {
        "inverse_metric": inverse,
        "acceleration_up": acceleration_up,
        "Z": z,
        "chi": susceptibility,
        "chi_Z": chi_derivative(z),
        "q": q,
        "exp_2q": exponential,
        "D_q": d_metric_dq,
    }


def contracted_matter_variation(
    metric: np.ndarray,
    u_covector: np.ndarray,
    acceleration_covector: np.ndarray,
    scalar_x: float,
    alpha: float,
    a_sigma: float,
    densitized_stress: np.ndarray,
) -> float:
    physical, _ = physical_metric(
        metric,
        u_covector,
        acceleration_covector,
        scalar_x,
        alpha,
        a_sigma,
    )
    return float(0.5 * np.sum(np.asarray(densitized_stress, dtype=float) * physical))


def analytic_directional_derivatives(
    metric: np.ndarray,
    u_covector: np.ndarray,
    acceleration_covector: np.ndarray,
    scalar_x: float,
    alpha: float,
    a_sigma: float,
    densitized_stress: np.ndarray,
    *,
    metric_direction: np.ndarray,
    u_direction: np.ndarray,
    acceleration_direction: np.ndarray,
    scalar_direction: float,
) -> dict[str, float]:
    _, state = physical_metric(
        metric,
        u_covector,
        acceleration_covector,
        scalar_x,
        alpha,
        a_sigma,
    )
    h = np.asarray(densitized_stress, dtype=float)
    reciprocal = float(0.5 * np.sum(h * state["D_q"]))
    acceleration_up = state["acceleration_up"]
    direct_metric = 0.5 * state["exp_2q"] * float(np.sum(h * metric_direction))
    delta_z_metric = -float(
        np.einsum("i,j,ij->", acceleration_up, acceleration_up, metric_direction)
    ) / a_sigma**2
    metric_value = direct_metric + reciprocal * alpha * scalar_x * state["chi_Z"] * (
        delta_z_metric
    )
    u_value = 2.0 * state["q"] * state["exp_2q"] * float(
        np.einsum("ij,j,i->", h, u_covector, u_direction)
    )
    delta_z_acceleration = (
        2.0 * float(acceleration_up @ acceleration_direction) / a_sigma**2
    )
    acceleration_value = (
        reciprocal * alpha * scalar_x * state["chi_Z"] * delta_z_acceleration
    )
    scalar_value = reciprocal * alpha * state["chi"] * float(scalar_direction)
    metric_gradient = (
        0.5 * state["exp_2q"] * h
        - reciprocal
        * alpha
        * scalar_x
        * state["chi_Z"]
        * np.outer(acceleration_up, acceleration_up)
        / a_sigma**2
    )
    u_gradient = (
        2.0 * state["q"] * state["exp_2q"] * (h @ u_covector)
    )
    acceleration_gradient = (
        2.0
        * reciprocal
        * alpha
        * scalar_x
        * state["chi_Z"]
        * acceleration_up
        / a_sigma**2
    )
    return {
        "metric": metric_value,
        "U": u_value,
        "A": acceleration_value,
        "X": scalar_value,
        "J": reciprocal,
        "metric_scale": float(
            np.linalg.norm(metric_gradient) * np.linalg.norm(metric_direction)
        ),
        "U_scale": float(np.linalg.norm(u_gradient) * np.linalg.norm(u_direction)),
        "A_scale": float(
            np.linalg.norm(acceleration_gradient) * np.linalg.norm(acceleration_direction)
        ),
        "X_scale": abs(reciprocal * alpha * state["chi"] * scalar_direction),
    }


def five_point_derivative(function: Callable[[float], float], step: float) -> float:
    return float(
        (
            function(-2.0 * step)
            - 8.0 * function(-step)
            + 8.0 * function(step)
            - function(2.0 * step)
        )
        / (12.0 * step)
    )


def normalized_error(observed: float, expected: float, chain_scale: float) -> float:
    return abs(observed - expected) / max(
        abs(observed), abs(expected), abs(chain_scale), 1e-12
    )


def random_lorentz_state(rng: np.random.Generator) -> dict[str, Any]:
    lapse = float(rng.uniform(0.8, 1.3))
    spatial = rng.uniform(0.7, 1.4, size=3)
    metric = np.diag(np.r_[-lapse**2, spatial**2])
    u_covector = np.array([-lapse, 0.0, 0.0, 0.0])
    acceleration = np.r_[0.0, rng.normal(0.0, 1.5, size=3)]
    scalar_x = float(rng.uniform(-1e-4, 1e-4))
    alpha = float(rng.uniform(0.2, 3.0))
    raw_stress = rng.normal(size=(4, 4))
    stress = 0.5 * (raw_stress + raw_stress.T)

    raw_metric_direction = rng.normal(size=(4, 4))
    metric_direction = 0.5 * (raw_metric_direction + raw_metric_direction.T)
    metric_direction /= np.linalg.norm(metric_direction)
    u_direction = rng.normal(size=4)
    u_direction /= np.linalg.norm(u_direction)
    acceleration_direction = rng.normal(size=4)
    acceleration_direction /= np.linalg.norm(acceleration_direction)
    return {
        "metric": metric,
        "U": u_covector,
        "A": acceleration,
        "X": scalar_x,
        "alpha": alpha,
        "stress": stress,
        "metric_direction": metric_direction,
        "U_direction": u_direction,
        "A_direction": acceleration_direction,
    }


def numerical_directional_derivatives(
    state: dict[str, Any],
    steps: dict[str, float],
) -> dict[str, float]:
    def evaluate(
        metric: np.ndarray,
        u_covector: np.ndarray,
        acceleration: np.ndarray,
        scalar_x: float,
    ) -> float:
        return contracted_matter_variation(
            metric,
            u_covector,
            acceleration,
            scalar_x,
            state["alpha"],
            1.0,
            state["stress"],
        )

    return {
        "metric": five_point_derivative(
            lambda value: evaluate(
                state["metric"] + value * state["metric_direction"],
                state["U"],
                state["A"],
                state["X"],
            ),
            steps["metric"],
        ),
        "U": five_point_derivative(
            lambda value: evaluate(
                state["metric"],
                state["U"] + value * state["U_direction"],
                state["A"],
                state["X"],
            ),
            steps["U"],
        ),
        "A": five_point_derivative(
            lambda value: evaluate(
                state["metric"],
                state["U"],
                state["A"] + value * state["A_direction"],
                state["X"],
            ),
            steps["A"],
        ),
        "X": five_point_derivative(
            lambda value: evaluate(
                state["metric"],
                state["U"],
                state["A"],
                state["X"] + value,
            ),
            steps["X"],
        ),
    }


def audit_random_variations(samples: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    errors: dict[str, list[float]] = {name: [] for name in ("metric", "U", "A", "X")}
    steps = {"metric": 1e-2, "U": 1e-2, "A": 1e-2, "X": 1e-3}
    for _ in range(samples):
        state = random_lorentz_state(rng)
        analytic = analytic_directional_derivatives(
            state["metric"],
            state["U"],
            state["A"],
            state["X"],
            state["alpha"],
            1.0,
            state["stress"],
            metric_direction=state["metric_direction"],
            u_direction=state["U_direction"],
            acceleration_direction=state["A_direction"],
            scalar_direction=1.0,
        )

        numerical = numerical_directional_derivatives(state, steps)
        for name, values in errors.items():
            values.append(
                normalized_error(numerical[name], analytic[name], analytic[f"{name}_scale"])
            )
    return {
        name: {
            "maximum_normalized_error": float(np.max(values)),
            "median_normalized_error": float(np.median(values)),
        }
        for name, values in errors.items()
    }


def perfect_fluid_source(energy_density: float, pressure: float) -> float:
    metric = np.diag([-1.0, 1.0, 1.0, 1.0])
    inverse = metric.copy()
    u_covector = np.array([-1.0, 0.0, 0.0, 0.0])
    u_vector = inverse @ u_covector
    stress = (energy_density + pressure) * np.outer(u_vector, u_vector) + pressure * inverse
    _, state = physical_metric(metric, u_covector, np.zeros(4), 0.0, 1.0, 1.0)
    return float(0.5 * np.sum(stress * state["D_q"]))


def build_report(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    audit = config["executable_variation_audit"]
    variations = audit_random_variations(int(audit["samples"]), int(audit["random_seed"]))
    maximum_variation_error = max(
        row["maximum_normalized_error"] for row in variations.values()
    )
    pressure_values = np.asarray([0.0, 1e-8, 0.25, 3.0, 20.0])
    pressure_rows = []
    for pressure in pressure_values:
        energy = 100.0 + pressure
        observed = perfect_fluid_source(float(energy), float(pressure))
        expected = 3.0 * float(pressure)
        pressure_rows.append(
            {
                "energy_density": float(energy),
                "pressure": float(pressure),
                "J": observed,
                "expected_3p": expected,
                "absolute_error": abs(observed - expected),
            }
        )
    maximum_pressure_error = max(row["absolute_error"] for row in pressure_rows)
    derivative_orders = config["derivative_order_claim"]
    maximum_order = max(
        int(derivative_orders[name])
        for name in ("metric", "scalar_X", "aether_U", "acceleration_A", "multiplier_B")
    )
    gates = config["gates"]
    gate_results = {
        "matter_variation_finite_difference_pass": maximum_variation_error
        <= gates["maximum_normalized_variation_error"],
        "perfect_fluid_pressure_source_pass": maximum_pressure_error
        <= gates["maximum_perfect_fluid_source_error"],
        "localized_derivative_order_pass": maximum_order
        <= gates["maximum_gravitational_field_equation_order"],
        "cold_dust_source_cancels": abs(perfect_fluid_source(100.0, 0.0))
        <= gates["maximum_perfect_fluid_source_error"],
    }
    localized_variation_pass = all(gate_results.values())
    return {
        "report_version": config["protocol_version"],
        "status": "passed_localized_variation" if localized_variation_pass else "failed_localized_variation",
        "generated_utc": datetime.now(UTC).isoformat(),
        "protocol": config_path.relative_to(ROOT).as_posix(),
        "protocol_sha256": sha256(config_path),
        "observational_data_opened": False,
        "empirical_fit_performed": False,
        "variation_finite_difference": {
            "samples": int(audit["samples"]),
            "seed": int(audit["random_seed"]),
            "fields": variations,
            "maximum_normalized_error": maximum_variation_error,
        },
        "perfect_fluid_source": {
            "rows": pressure_rows,
            "maximum_absolute_error": maximum_pressure_error,
            "dust_value": perfect_fluid_source(100.0, 0.0),
            "identity": "J=T+E=3p",
        },
        "localized_equation_system": {
            "classically_equivalent_to_v17H": True,
            "new_physical_constants": 0,
            "new_propagating_fields_claimed": 0,
            "maximum_Euler_differential_order": maximum_order,
            "stress_tensors_explicitly_identified": True,
            "off_shell_diffeomorphism_identity_derived": True,
            "full_Dirac_constraint_matrix_computed": False,
            "full_characteristic_matrix_computed": False,
        },
        "gates": {
            **gate_results,
            "localized_variation_pass": localized_variation_pass,
            "full_Hamiltonian_health_pass": False,
            "full_causality_pass": False,
            "holdout_authorized": False,
        },
        "decision": {
            "outcome": (
                "advance_to_Dirac_and_principal_symbol_only"
                if localized_variation_pass
                else "retire_v17H_localization"
            ),
            "scope": (
                "This establishes the reciprocal localized Euler system and its stress "
                "bookkeeping. It does not establish the constrained degree-of-freedom "
                "count, energy sign, causal cones, PPN limit, or data transfer."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v17i_localized_variation.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "sigma_v17i_localized_variation",
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
