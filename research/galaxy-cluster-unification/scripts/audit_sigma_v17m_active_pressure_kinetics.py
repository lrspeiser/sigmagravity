"""Audit v17L's exact transverse kinetic matrix with canonical matter."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mpmath as mp

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def logspace(start_exponent: float, stop_exponent: float, samples: int) -> list[float]:
    if samples < 2:
        raise ValueError("samples must be at least two")
    step = (stop_exponent - start_exponent) / (samples - 1)
    return [10.0 ** (start_exponent + index * step) for index in range(samples)]


def matter_lagrangian_hat(q: float, rho_hat: float, p_hat: float) -> float:
    if not q < 0.5:
        raise ValueError("the physical metric is non-Lorentzian for q>=1/2")
    if rho_hat < 0.0 or abs(p_hat) > rho_hat:
        raise ValueError("canonical matter requires rho>=0 and |p|<=rho")
    root = math.sqrt(1.0 - 2.0 * q)
    kinetic = 0.5 * (rho_hat + p_hat) * math.exp(2.0 * q) / root
    potential = 0.5 * (rho_hat - p_hat) * math.exp(4.0 * q) * root
    return kinetic - potential


def reciprocal_source_hat(q: float, rho_hat: float, p_hat: float) -> float:
    if not q < 0.5:
        raise ValueError("the physical metric is non-Lorentzian for q>=1/2")
    root = math.sqrt(1.0 - 2.0 * q)
    kinetic = (
        0.5 * (rho_hat + p_hat) * math.exp(2.0 * q) * (3.0 - 4.0 * q) / ((1.0 - 2.0 * q) * root)
    )
    potential = 0.5 * (rho_hat - p_hat) * math.exp(4.0 * q) * (3.0 - 8.0 * q) / root
    return kinetic - potential


def transverse_sector(q: float, rho_hat: float, p_hat: float, epsilon: float) -> dict[str, float]:
    j_hat = reciprocal_source_hat(q, rho_hat, p_hat)
    matter_hessian = -0.5 * q * j_hat
    c14_effective = epsilon + matter_hessian
    speed_squared = math.inf if c14_effective == 0.0 else epsilon / c14_effective
    return {
        "q": q,
        "rho_hat": rho_hat,
        "p_hat": p_hat,
        "w": p_hat / rho_hat if rho_hat else math.nan,
        "J_hat": j_hat,
        "matter_hessian": matter_hessian,
        "c_14_effective": c14_effective,
        "spin_1_speed_squared": speed_squared,
        "matter_light_speed_squared": 1.0 - 2.0 * q,
        "matter_scalar_kinetic": math.exp(2.0 * q) / math.sqrt(1.0 - 2.0 * q),
    }


def finite_difference_hessian(
    q_0: float, rho_hat: float, p_hat: float, step: float = 0.01
) -> float:
    # The dust-like witness subtracts O(10^3) kinetic and potential pieces to
    # recover an O(10^-2) pressure Lagrangian and then differentiates an
    # O(10^-7) correction. Extra precision prevents that legitimate
    # cancellation from masquerading as a failed chain-rule identity.
    with mp.workdps(50):
        q_0_mp = mp.mpf(str(q_0))
        rho_mp = mp.mpf(str(rho_hat))
        pressure_mp = mp.mpf(str(p_hat))
        step_mp = mp.mpf(str(step))

        def lagrangian_at_velocity(velocity: mp.mpf) -> mp.mpf:
            q = q_0_mp * (1 + velocity * velocity) ** mp.mpf("-0.25")
            root = mp.sqrt(1 - 2 * q)
            kinetic = mp.mpf("0.5") * (rho_mp + pressure_mp) * mp.exp(2 * q) / root
            potential = mp.mpf("0.5") * (rho_mp - pressure_mp) * mp.exp(4 * q) * root
            return kinetic - potential

        values = [
            -lagrangian_at_velocity(2 * step_mp),
            16 * lagrangian_at_velocity(step_mp),
            -30 * lagrangian_at_velocity(mp.mpf("0")),
            16 * lagrangian_at_velocity(-step_mp),
            -lagrangian_at_velocity(-2 * step_mp),
        ]
        return float(mp.fsum(values) / (12 * step_mp * step_mp))


def critical_density(q: float, w: float, epsilon: float) -> float:
    response_per_density = reciprocal_source_hat(q, 1.0, w)
    if response_per_density <= 0.0:
        return math.inf
    return 2.0 * epsilon / (q * response_per_density)


def build_report(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    parent = config["parent"]
    parent_protocol_path = ROOT / parent["protocol"]
    parent_report_path = ROOT / parent["report"]
    parent_protocol_hash_ok = sha256(parent_protocol_path) == parent["sha256"]
    parent_report_hash_ok = sha256(parent_report_path) == parent["report_sha256"]
    if not parent_protocol_hash_ok or not parent_report_hash_ok:
        raise RuntimeError("frozen v17L parent protocol or report hash changed")

    normalization = config["dimensionless_normalization"]
    epsilon = float(normalization["epsilon_aether"])
    audit = config["executable_audit"]
    zero_q_rows = []
    for rho_hat, p_hat in ((1.0, 0.0), (1.0, 1e-5), (2.0, 0.4), (3.0, 3.0)):
        observed = reciprocal_source_hat(0.0, rho_hat, p_hat)
        zero_q_rows.append(
            {
                "rho_hat": rho_hat,
                "p_hat": p_hat,
                "J_hat": observed,
                "expected_3p_hat": 3.0 * p_hat,
                "absolute_error": abs(observed - 3.0 * p_hat),
            }
        )
    maximum_zero_q_error = max(row["absolute_error"] for row in zero_q_rows)

    finite_difference_rows = []
    for witness in audit["finite_difference_witnesses"]:
        q = float(witness["q"])
        rho_hat = float(witness["rho_hat"])
        p_hat = rho_hat * float(witness["w"])
        analytic = -0.5 * q * reciprocal_source_hat(q, rho_hat, p_hat)
        observed = finite_difference_hessian(q, rho_hat, p_hat)
        error = abs(observed - analytic) / max(abs(analytic), 1e-14)
        finite_difference_rows.append(
            {
                **transverse_sector(q, rho_hat, p_hat, epsilon),
                "analytic_matter_hessian": analytic,
                "finite_difference_matter_hessian": observed,
                "normalized_error": error,
            }
        )
    maximum_hessian_error = max(row["normalized_error"] for row in finite_difference_rows)

    q_values = logspace(
        math.log10(float(audit["q_log_minimum"])),
        math.log10(float(audit["q_log_maximum"])),
        int(audit["q_samples"]),
    )
    rho_values = logspace(
        math.log10(float(audit["rho_hat_log_minimum"])),
        math.log10(float(audit["rho_hat_log_maximum"])),
        int(audit["rho_hat_samples"]),
    )
    w_values = [float(value) for value in audit["equation_of_state_w"]]
    stable_count = 0
    unstable_count = 0
    minimum_c14 = math.inf
    first_unstable = None
    for q in q_values:
        for rho_hat in rho_values:
            for w in w_values:
                row = transverse_sector(q, rho_hat, w * rho_hat, epsilon)
                minimum_c14 = min(minimum_c14, row["c_14_effective"])
                if row["c_14_effective"] > 0.0 and row["spin_1_speed_squared"] > 0.0:
                    stable_count += 1
                else:
                    unstable_count += 1
                    if first_unstable is None:
                        first_unstable = row

    threshold_rows = []
    for q in (1e-8, 1e-6, 1e-5, 1e-4, 1e-2):
        for w in w_values:
            rho_critical = critical_density(q, w, epsilon)
            pressure_critical_hat = w * rho_critical
            pressure_scale_pa = float(normalization["a_sigma_m_s2"]) ** 2 / (
                8.0 * math.pi * 6.67430e-11
            )
            threshold_rows.append(
                {
                    "q": q,
                    "w": w,
                    "rho_hat_critical": rho_critical,
                    "p_hat_critical": pressure_critical_hat,
                    "pressure_critical_Pa": pressure_critical_hat * pressure_scale_pa,
                }
            )

    finite_positive_thresholds = [
        row
        for row in threshold_rows
        if math.isfinite(row["rho_hat_critical"]) and row["rho_hat_critical"] > 0.0
    ]
    gates = config["gates"]
    gate_results = {
        "parent_protocol_hash_pass": parent_protocol_hash_ok,
        "parent_report_hash_pass": parent_report_hash_ok,
        "zero_q_pressure_identity_pass": maximum_zero_q_error <= audit["zero_q_identity_tolerance"],
        "matter_hessian_identity_pass": maximum_hessian_error
        <= audit["maximum_normalized_hessian_error"],
        "positive_matter_kinetic_pass": all(
            row["matter_scalar_kinetic"] > 0.0 for row in finite_difference_rows
        ),
        "positive_c14_everywhere_pass": unstable_count == 0,
        "no_finite_positive_density_zero_surface_pass": not finite_positive_thresholds,
        "finite_positive_spin_1_speed_everywhere_pass": unstable_count == 0,
    }
    active_pressure_pass = all(gate_results.values())
    return {
        "report_version": config["protocol_version"],
        "status": (
            "passed_active_pressure_kinetic_gate"
            if active_pressure_pass
            else "failed_active_pressure_kinetic_gate"
        ),
        "generated_utc": datetime.now(UTC).isoformat(),
        "protocol": config_path.relative_to(ROOT).as_posix(),
        "protocol_sha256": sha256(config_path),
        "observational_data_opened": False,
        "empirical_fit_performed": False,
        "parent": {
            "protocol": parent["protocol"],
            "protocol_sha256": parent["sha256"],
            "protocol_hash_verified": parent_protocol_hash_ok,
            "report": parent["report"],
            "report_sha256": parent["report_sha256"],
            "report_hash_verified": parent_report_hash_ok,
        },
        "exact_matter_action": {
            "zero_q_rows": zero_q_rows,
            "maximum_zero_q_identity_error": maximum_zero_q_error,
            "finite_difference_rows": finite_difference_rows,
            "maximum_normalized_hessian_error": maximum_hessian_error,
            "transverse_matter_mixing_at_A_zero": 0.0,
        },
        "grid": {
            "q_samples": len(q_values),
            "rho_hat_samples": len(rho_values),
            "w_values": w_values,
            "total_backgrounds": len(q_values) * len(rho_values) * len(w_values),
            "stable_backgrounds": stable_count,
            "unstable_backgrounds": unstable_count,
            "minimum_c_14_effective": minimum_c14,
            "first_unstable_background": first_unstable,
        },
        "zero_surfaces": {
            "analytic_formula": "rho_hat_crit=2 epsilon/[q F(q,w)]",
            "rows": threshold_rows,
            "finite_positive_threshold_count": len(finite_positive_thresholds),
        },
        "gates": {**gate_results, "active_pressure_kinetic_pass": active_pressure_pass},
        "decision": {
            "outcome": (
                "retain_derivative_susceptibility"
                if active_pressure_pass
                else "retire_acceleration_susceptibility_from_physical_metric"
            ),
            "scope": (
                "This gate tests the exact transverse principal direction of the explicit "
                "canonical matter system. Failure retires chi(A^2) inside g_tilde, not the "
                "luminal aether carrier or the baryon-derived halo-scale objective."
            ),
            "holdout_authorized": False,
        },
        "unused_gate_declaration": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v17m_active_pressure_kinetic_gate.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "sigma_v17m_active_pressure_kinetic_gate",
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
