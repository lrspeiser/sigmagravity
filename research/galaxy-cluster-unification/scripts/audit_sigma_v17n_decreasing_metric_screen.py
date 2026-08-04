"""Audit the no-go theorem for decreasing acceleration screens in g_tilde."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mpmath as mp

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_shared_kernel(path: Path):
    spec = importlib.util.spec_from_file_location("sigma_v17m_shared_matter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load the hash-locked v17M matter kernel")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def original(z: float) -> tuple[float, float]:
    return (1.0 + z) ** (-0.25), -0.25 * (1.0 + z) ** (-1.25)


def quartic_soft_start(z: float) -> tuple[float, float]:
    return (1.0 + z * z) ** (-0.125), -0.25 * z * (1.0 + z * z) ** (-1.125)


def exponential(z: float) -> tuple[float, float]:
    value = math.exp(-0.25 * z)
    return value, -0.25 * value


def rational(z: float) -> tuple[float, float]:
    denominator = 1.0 + 0.25 * z
    return 1.0 / denominator, -0.25 / denominator**2


def saturating_half(z: float) -> tuple[float, float]:
    return 1.0 - 0.5 * z / (1.0 + z), -0.5 / (1.0 + z) ** 2


CURVES: dict[str, Callable[[float], tuple[float, float]]] = {
    "original": original,
    "quartic_soft_start": quartic_soft_start,
    "exponential": exponential,
    "rational": rational,
    "saturating_half": saturating_half,
}


def finite_difference_matter_hessian(
    curve_id: str,
    z_0: float,
    q_base: float,
    rho_hat: float,
    p_hat: float,
    step: float,
) -> float:
    with mp.workdps(50):
        z_mp = mp.mpf(str(z_0))
        q_base_mp = mp.mpf(str(q_base))
        rho_mp = mp.mpf(str(rho_hat))
        pressure_mp = mp.mpf(str(p_hat))
        step_mp = mp.mpf(str(step))

        def chi(value: mp.mpf) -> mp.mpf:
            if curve_id == "original":
                return (1 + value) ** mp.mpf("-0.25")
            if curve_id == "quartic_soft_start":
                return (1 + value * value) ** mp.mpf("-0.125")
            if curve_id == "exponential":
                return mp.exp(-value / 4)
            if curve_id == "rational":
                return 1 / (1 + value / 4)
            if curve_id == "saturating_half":
                return 1 - value / (2 * (1 + value))
            raise KeyError(curve_id)

        def lagrangian(delta: mp.mpf) -> mp.mpf:
            q = q_base_mp * chi(z_mp + delta * delta)
            root = mp.sqrt(1 - 2 * q)
            kinetic = mp.mpf("0.5") * (rho_mp + pressure_mp) * mp.exp(2 * q) / root
            potential = mp.mpf("0.5") * (rho_mp - pressure_mp) * mp.exp(4 * q) * root
            return kinetic - potential

        values = [
            -lagrangian(2 * step_mp),
            16 * lagrangian(step_mp),
            -30 * lagrangian(mp.mpf("0")),
            16 * lagrangian(-step_mp),
            -lagrangian(-2 * step_mp),
        ]
        return float(mp.fsum(values) / (12 * step_mp * step_mp))


def build_report(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    parent = config["parent"]
    parent_protocol_ok = sha256(ROOT / parent["protocol"]) == parent["sha256"]
    parent_report_ok = sha256(ROOT / parent["report"]) == parent["report_sha256"]
    shared_config = config["shared_exact_matter_kernel"]
    shared_path = ROOT / shared_config["path"]
    shared_hash_ok = sha256(shared_path) == shared_config["sha256"]
    if not parent_protocol_ok or not parent_report_ok or not shared_hash_ok:
        raise RuntimeError("v17M parent or exact matter kernel hash changed")
    shared = load_shared_kernel(shared_path)

    audit = config["executable_audit"]
    q_base = float(audit["q_base"])
    bare_k = float(audit["epsilon_aether_as_bare_K"])
    rho_hat = float(audit["canonical_witness_rho_hat"])
    p_hat = rho_hat * float(audit["canonical_witness_w"])
    rows = []
    for declaration in config["representative_curves"]:
        curve_id = declaration["id"]
        z = float(declaration["audit_z"])
        chi, derivative = CURVES[curve_id](z)
        q = q_base * chi
        j_hat = shared.reciprocal_source_hat(q, rho_hat, p_hat)
        analytic_hessian = 2.0 * q_base * j_hat * derivative
        finite_difference = finite_difference_matter_hessian(
            curve_id,
            z,
            q_base,
            rho_hat,
            p_hat,
            float(audit["transverse_finite_difference_step"]),
        )
        error = abs(finite_difference - analytic_hessian) / max(abs(analytic_hessian), 1e-14)
        j_critical = math.inf if derivative >= 0.0 else bare_k / (-2.0 * q_base * derivative)
        response_per_density = shared.reciprocal_source_hat(q, 1.0, 1.0)
        rho_critical = j_critical / response_per_density
        rows.append(
            {
                "id": curve_id,
                "formula": declaration["formula"],
                "z": z,
                "chi": chi,
                "chi_prime": derivative,
                "q": q,
                "J_hat_at_unit_density": j_hat,
                "analytic_matter_hessian": analytic_hessian,
                "finite_difference_matter_hessian": finite_difference,
                "normalized_hessian_error": error,
                "J_hat_critical": j_critical,
                "rho_hat_critical_for_w_1": rho_critical,
                "finite_positive_zero_surface": math.isfinite(j_critical) and j_critical > 0.0,
            }
        )

    maximum_error = max(row["normalized_hessian_error"] for row in rows)
    every_decreasing_has_zero = all(
        row["chi_prime"] < -float(audit["derivative_sign_tolerance"])
        and row["finite_positive_zero_surface"]
        for row in rows
    )
    soft = next(row for row in rows if row["id"] == "quartic_soft_start")
    soft_derivative_at_zero = quartic_soft_start(0.0)[1]
    soft_start_relocates = (
        soft_derivative_at_zero == 0.0
        and soft["chi_prime"] < 0.0
        and soft["finite_positive_zero_surface"]
    )
    gate_results = {
        "parent_protocol_hash_pass": parent_protocol_ok,
        "parent_report_hash_pass": parent_report_ok,
        "shared_exact_matter_kernel_hash_pass": shared_hash_ok,
        "exact_transverse_hessian_identity_pass": maximum_error
        <= audit["maximum_normalized_hessian_error"],
        "every_decreasing_curve_has_no_finite_zero_surface_pass": not every_decreasing_has_zero,
        "quartic_soft_start_removes_failure_pass": not soft_start_relocates,
    }
    class_survives = all(gate_results.values())
    return {
        "report_version": config["protocol_version"],
        "status": (
            "decreasing_metric_screen_class_survives"
            if class_survives
            else "decreasing_metric_screen_class_retired"
        ),
        "generated_utc": datetime.now(UTC).isoformat(),
        "protocol": config_path.relative_to(ROOT).as_posix(),
        "protocol_sha256": sha256(config_path),
        "observational_data_opened": False,
        "empirical_fit_performed": False,
        "parent_hashes_verified": parent_protocol_ok and parent_report_ok,
        "shared_exact_matter_kernel_hash_verified": shared_hash_ok,
        "theorem": {
            "transverse_identity": "Delta K_T=2 q_base J_hat chi_prime(z_0)",
            "finite_zero_surface": "J_hat_crit=K_b/[-2 q_base chi_prime(z_0)]",
            "mean_value_consequence": (
                "Every differentiable susceptibility that decreases on an interval has "
                "chi_prime<0 somewhere and therefore a finite positive-density zero surface "
                "for finite matter-independent K_b."
            ),
            "maximum_normalized_hessian_error": maximum_error,
        },
        "curve_rows": rows,
        "gates": {**gate_results, "decreasing_metric_screen_class_survives": class_survives},
        "decision": {
            "outcome": (
                "continue_curve_search"
                if class_survives
                else "retire_all_decreasing_acceleration_screens_inside_physical_metric"
            ),
            "next_mechanism_requirement": (
                "Move environmental susceptibility outside derivative dependence of the "
                "physical matter metric; retain one metric for matter and light."
            ),
            "holdout_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v17n_decreasing_metric_screen_no_go.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "sigma_v17n_decreasing_metric_screen_no_go",
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
