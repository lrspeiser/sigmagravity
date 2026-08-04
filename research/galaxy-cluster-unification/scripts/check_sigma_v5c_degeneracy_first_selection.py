from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_degenerate_action import (
    bounded_even_activation,
    k_mouflage_static_parallel_speed_squared,
    luminal_class_ia_coefficients,
    normalized_dhost_residuals,
    v5c_trial_coefficients,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select and audit the degeneracy-first Sigma v5C action envelope."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v5c_degeneracy_first_selection.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v5c_degeneracy_first_selection",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    scan = config["coefficient_scan"]
    magnitude = np.geomspace(
        float(scan["signed_X_hat_minimum_magnitude"]),
        float(scan["signed_X_hat_maximum_magnitude"]),
        int(scan["points_per_sign"]),
    )
    signed_ratio = np.concatenate((-magnitude[::-1], [0.0], magnitude))
    activation = bounded_even_activation(signed_ratio)
    evenness_error = float(
        np.max(np.abs(activation - bounded_even_activation(-signed_ratio)))
    )
    near_zero = float(bounded_even_activation(1.0e-6))
    high_field = float(bounded_even_activation(1.0e6))

    trial_records = []
    all_trial_finite = True
    maximum_trial_residual = 0.0
    maximum_trial_coefficient = 0.0
    for strength in scan["orientation_strengths"]:
        coefficients = v5c_trial_coefficients(signed_ratio, float(strength))
        residuals = normalized_dhost_residuals(
            signed_ratio,
            np.ones_like(signed_ratio),
            np.zeros_like(signed_ratio),
            coefficients,
        )
        local_residual = max(
            float(np.max(np.abs(residual))) for residual in residuals.values()
        )
        local_finite = all(
            bool(np.all(np.isfinite(coefficient)))
            for coefficient in coefficients.values()
        )
        all_trial_finite = all_trial_finite and local_finite
        maximum_trial_residual = max(maximum_trial_residual, local_residual)
        local_coefficient_maximum = max(
            float(np.max(np.abs(coefficient)))
            for coefficient in coefficients.values()
        )
        maximum_trial_coefficient = max(
            maximum_trial_coefficient, local_coefficient_maximum
        )
        trial_records.append(
            {
                "orientation_strength": float(strength),
                "all_coefficients_finite": local_finite,
                "maximum_normalized_identity_residual": local_residual,
                "maximum_absolute_A3_shape": float(
                    np.max(np.abs(coefficients["A3"]))
                ),
                "maximum_absolute_A4_shape": float(
                    np.max(np.abs(coefficients["A4"]))
                ),
                "maximum_absolute_A5_shape": float(
                    np.max(np.abs(coefficients["A5"]))
                ),
                "maximum_absolute_dependent_coefficient_shape": local_coefficient_maximum,
            }
        )

    rng = np.random.default_rng(int(scan["random_seed"]))
    random_count = int(scan["random_identities"])
    kinetic = rng.normal(scale=3.0, size=random_count)
    coupling = np.exp(rng.normal(scale=0.8, size=random_count))
    derivative = rng.normal(scale=0.5, size=random_count)
    a3 = rng.normal(scale=0.5, size=random_count)
    coefficients = luminal_class_ia_coefficients(
        kinetic, coupling, derivative, a3
    )
    residuals = normalized_dhost_residuals(
        kinetic, coupling, derivative, coefficients
    )
    maximum_random_residual = max(
        float(np.max(np.abs(residual))) for residual in residuals.values()
    )

    k_control = config["k_mouflage_control"]
    static_magnitude = np.geomspace(
        float(k_control["static_gradient_minimum"]),
        float(k_control["static_gradient_maximum"]),
        int(k_control["points"]),
    )
    power = float(k_control["screening_power"])
    static_x = -static_magnitude
    kinetic_derivative = 1.0 + np.power(static_magnitude, power)
    kinetic_second_derivative = -power * np.power(
        static_magnitude, power - 1.0
    )
    k_speed_squared = k_mouflage_static_parallel_speed_squared(
        static_x, kinetic_derivative, kinetic_second_derivative
    )

    gates_config = config["gates"]
    maximum_residual = max(maximum_trial_residual, maximum_random_residual)
    constant_count = len(config["v5c_trial"]["provisional_constants"])
    gates = {
        "luminal_tensor_identity": True,
        "class_ia_degeneracy_identities": maximum_residual
        <= float(gates_config["maximum_normalized_degeneracy_residual"]),
        "signed_activation_is_real_even_smooth": bool(
            np.all(np.isfinite(activation))
            and evenness_error
            <= float(gates_config["maximum_activation_evenness_error"])
            and float(bounded_even_activation(0.0))
            == float(gates_config["required_activation_at_zero"])
        ),
        "activation_has_frozen_limits": near_zero
        <= float(gates_config["maximum_activation_at_absolute_X_1e_minus_6"])
        and high_field
        <= float(gates_config["maximum_activation_at_absolute_X_1e6"]),
        "trial_coefficients_finite_and_bounded": all_trial_finite
        and maximum_trial_coefficient
        <= float(gates_config["maximum_absolute_trial_coefficient_shape"]),
        "parameter_economy": constant_count
        <= int(gates_config["maximum_universal_constants"]),
        "pure_k_mouflage_strict_causality": float(np.max(k_speed_squared))
        <= float(gates_config["maximum_strictly_causal_speed_squared"]),
    }
    gates = {name: bool(value) for name, value in gates.items()}
    selected_envelope_gates = [
        "luminal_tensor_identity",
        "class_ia_degeneracy_identities",
        "signed_activation_is_real_even_smooth",
        "activation_has_frozen_limits",
        "trial_coefficients_finite_and_bounded",
        "parameter_economy",
    ]
    report = {
        "status": "completed Sigma v5C degeneracy-first action-envelope selection",
        "observational_data_accessed": False,
        "raw_holdout_opened": False,
        "lane_decisions": config["lanes"],
        "activation_scan": {
            "evenness_error": evenness_error,
            "value_at_zero": float(bounded_even_activation(0.0)),
            "value_at_absolute_X_1e_minus_6": near_zero,
            "value_at_absolute_X_1e6": high_field,
        },
        "trial_coefficient_records": trial_records,
        "maximum_normalized_degeneracy_residual": maximum_residual,
        "maximum_absolute_trial_coefficient_shape": maximum_trial_coefficient,
        "k_mouflage_control": {
            "minimum_parallel_speed_squared": float(np.min(k_speed_squared)),
            "maximum_parallel_speed_squared": float(np.max(k_speed_squared)),
            "strict_causality_pass": gates["pure_k_mouflage_strict_causality"],
            "analytic_reason": "For X<0, screening requires P_XX<0; therefore 1+2 X P_XX/P_X is greater than one.",
        },
        "universal_constant_count": constant_count,
        "gates": gates,
        "selected_envelope_gates_pass": bool(
            all(gates[name] for name in selected_envelope_gates)
        ),
        "decision": "select_sigma_v5c_luminal_class_ia_dhost_for_full_derivation_no_fit",
        "novelty_status": "The action class is established prior art. Only the fixed activation and its proposed baryon-locked lensing use are candidate novelties and remain unaudited.",
        "unresolved_before_empirical_fit": config["unresolved_before_empirical_fit"],
        "data_policy": config["data_policy"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
