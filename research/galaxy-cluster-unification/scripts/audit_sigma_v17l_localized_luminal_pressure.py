"""Audit the localized reciprocal equations of the v17K pressure metric."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "scripts" / "audit_sigma_v17i_localized_variation.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_shared_kernel():
    spec = importlib.util.spec_from_file_location("sigma_v17i_shared_variation", SHARED)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load hash-locked v17I variation kernel")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def carrier_coefficients(epsilon: float) -> dict[str, float]:
    value = float(epsilon)
    if not 0.0 < value < 0.5:
        raise ValueError("epsilon must lie in 0<epsilon<1/2")
    c_2 = value / (1.0 - 2.0 * value)
    return {
        "c_1": value,
        "c_2": c_2,
        "c_3": -value,
        "c_4": 0.0,
        "c_13": 0.0,
        "c_14": value,
        "c_123": c_2,
    }


def auxiliary_acceleration_source(
    scalar_x: float,
    reciprocal_source: float,
    acceleration_up: list[float],
    *,
    alpha: float,
    chi_z: float,
    a_sigma: float,
) -> list[float]:
    coefficient = 2.0 * alpha * float(scalar_x) * chi_z * float(reciprocal_source) / a_sigma**2
    return [coefficient * float(value) for value in acceleration_up]


def build_report(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    parent = config["parent"]
    parent_protocol_path = ROOT / parent["protocol"]
    parent_report_path = ROOT / parent["report"]
    parent_protocol_hash_ok = sha256(parent_protocol_path) == parent["sha256"]
    parent_report_hash_ok = sha256(parent_report_path) == parent["report_sha256"]
    if not parent_protocol_hash_ok or not parent_report_hash_ok:
        raise RuntimeError("frozen v17K parent protocol or report hash changed")
    parent_config = json.loads(parent_protocol_path.read_text(encoding="utf-8"))
    epsilon = float(parent_config["fixed_coefficients"]["epsilon_aether"])

    shared_config = config["shared_matter_variation_kernel"]
    shared_hash_ok = sha256(ROOT / shared_config["path"]) == shared_config["sha256"]
    if not shared_hash_ok:
        raise RuntimeError("shared v17I variation kernel hash changed")
    shared = load_shared_kernel()
    audit = config["executable_variation_audit"]
    variations = shared.audit_random_variations(int(audit["samples"]), int(audit["random_seed"]))
    maximum_variation_error = max(row["maximum_normalized_error"] for row in variations.values())
    pressure_rows = []
    for pressure in (0.0, 1e-8, 0.25, 3.0, 20.0):
        energy = 100.0 + pressure
        observed = shared.perfect_fluid_source(energy, pressure)
        expected = 3.0 * pressure
        pressure_rows.append(
            {
                "energy_density": energy,
                "pressure": pressure,
                "J": observed,
                "expected_3p": expected,
                "absolute_error": abs(observed - expected),
            }
        )
    maximum_pressure_error = max(row["absolute_error"] for row in pressure_rows)
    orders = config["derivative_order_claim"]
    maximum_order = max(
        int(orders[name])
        for name in ("metric", "scalar_X", "aether_U", "acceleration_A", "multiplier_B")
    )
    coefficients = carrier_coefficients(epsilon)
    vacuum_q_a = auxiliary_acceleration_source(
        0.0,
        3.0,
        [0.0, 0.1, -0.2, 0.3],
        alpha=2.0,
        chi_z=-0.25,
        a_sigma=1.0,
    )
    dust_q_a = auxiliary_acceleration_source(
        0.1,
        0.0,
        [0.0, 0.1, -0.2, 0.3],
        alpha=2.0,
        chi_z=-0.25,
        a_sigma=1.0,
    )
    gates = config["gates"]
    gate_results = {
        "parent_protocol_hash_pass": parent_protocol_hash_ok,
        "parent_report_hash_pass": parent_report_hash_ok,
        "shared_kernel_hash_pass": shared_hash_ok,
        "matter_variation_pass": maximum_variation_error
        <= gates["maximum_normalized_variation_error"],
        "pressure_source_pass": maximum_pressure_error
        <= gates["maximum_perfect_fluid_source_error"],
        "cold_dust_source_cancels": abs(shared.perfect_fluid_source(100.0, 0.0))
        <= gates["maximum_perfect_fluid_source_error"],
        "vacuum_auxiliary_B_zero": max(abs(value) for value in vacuum_q_a) == 0.0,
        "dust_auxiliary_B_zero": max(abs(value) for value in dust_q_a) == 0.0,
        "derivative_order_pass": maximum_order
        <= gates["maximum_gravitational_field_equation_order"],
        "luminal_carrier_coefficients_preserved": coefficients["c_13"] == 0.0
        and coefficients["c_4"] == 0.0
        and coefficients["c_123"] > 0.0,
    }
    localization_pass = all(gate_results.values())
    return {
        "report_version": config["protocol_version"],
        "status": "passed_localized_variation"
        if localization_pass
        else "failed_localized_variation",
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
            "epsilon_aether_loaded": epsilon,
        },
        "shared_kernel": {
            "path": shared_config["path"],
            "sha256": shared_config["sha256"],
            "hash_verified": shared_hash_ok,
        },
        "matter_variation": {
            "samples": int(audit["samples"]),
            "seed": int(audit["random_seed"]),
            "fields": variations,
            "maximum_normalized_error": maximum_variation_error,
        },
        "perfect_fluid_source": {
            "rows": pressure_rows,
            "maximum_absolute_error": maximum_pressure_error,
            "dust_value": shared.perfect_fluid_source(100.0, 0.0),
            "identity": "J=T+E=3p",
        },
        "localized_equations": {
            "acceleration_auxiliary": "B^m=-Q_A^m",
            "born_infeld_force_removed": True,
            "vacuum_Q_A": vacuum_q_a,
            "dust_Q_A": dust_q_a,
            "maximum_Euler_differential_order": maximum_order,
            "stress_tensors_identified": True,
            "off_shell_diffeomorphism_identity_derived": True,
            "new_physical_constants": 0,
            "new_propagating_fields_claimed": 0,
        },
        "carrier_coefficients": coefficients,
        "gates": {
            **gate_results,
            "localized_variation_pass": localization_pass,
            "active_pressure_kinetic_pass": False,
            "full_Sigma_PPN_pass": False,
            "holdout_authorized": False,
        },
        "decision": {
            "outcome": (
                "advance_to_active_pressure_kinetic_gate_only"
                if localization_pass
                else "retire_v17K_localization"
            ),
            "scope": (
                "This establishes the exact localized reciprocal equation and stress system. "
                "It does not establish the sign or rank of the active matter-background "
                "kinetic matrix, a complete Solar solution, or any halo prediction."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v17l_localized_luminal_pressure.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "sigma_v17l_localized_luminal_pressure",
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
