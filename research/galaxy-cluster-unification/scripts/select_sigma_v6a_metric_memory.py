from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v6_metric_memory import (
    metric_memory_activation,
    nonlinear_superposition_residual,
    retarded_convolution,
    rotation_covariance_residual,
    time_symmetric_convolution,
)


def random_rotation(generator: np.random.Generator) -> np.ndarray:
    trial = generator.normal(size=(3, 3))
    orthogonal, _ = np.linalg.qr(trial)
    if np.linalg.det(orthogonal) < 0.0:
        orthogonal[:, 0] *= -1.0
    return orthogonal


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the Sigma v6A action envelope.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v6a_metric_memory_selection.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v6a_metric_memory_selection",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))

    causal = config["causality_audit"]
    sample_count = int(causal["samples"])
    impulse_index = int(causal["impulse_index"])
    source = np.zeros(sample_count)
    source[impulse_index] = 1.0
    lag = np.arange(int(causal["kernel_samples"]), dtype=float)
    kernel = np.exp(-lag / float(causal["kernel_decay_samples"]))
    retarded = retarded_convolution(source, kernel)
    symmetric = time_symmetric_convolution(source, kernel)
    zero_response = retarded_convolution(np.zeros_like(source), kernel)

    generator = np.random.default_rng(7601)
    covariance_residuals = []
    nonlinear_residuals = []
    for _ in range(1000):
        first = generator.normal(size=(3, 3))
        second = generator.normal(size=(3, 3))
        rotation = random_rotation(generator)
        covariance_residuals.append(rotation_covariance_residual(first, rotation))
        nonlinear_residuals.append(nonlinear_superposition_residual(first, second))

    construction = config["construction_gates"]
    tiny_chi = 1.0e-12
    deep_ratio = float(metric_memory_activation(tiny_chi) / tiny_chi)
    deep_error = abs(deep_ratio - 1.0)
    high_ratio = float(construction["high_field_acceleration_ratio"])
    high_chi = high_ratio**2
    high_fraction = float(metric_memory_activation(high_chi) / high_chi)
    maximum_covariance = float(np.max(covariance_residuals))
    median_nonlinearity = float(np.median(nonlinear_residuals))
    maximum_retarded_pre_response = float(np.max(np.abs(retarded[:impulse_index])))
    maximum_symmetric_pre_response = float(np.max(np.abs(symmetric[:impulse_index])))
    maximum_zero_source_response = float(np.max(np.abs(zero_response)))

    gates = {
        "retarded_support": maximum_retarded_pre_response
        <= float(causal["maximum_retarded_pre_response"]),
        "traditional_symmetric_variation_exposes_advanced_support": maximum_symmetric_pre_response
        >= float(causal["minimum_symmetric_pre_response"]),
        "zero_source_zero_fixed_state": maximum_zero_source_response == 0.0,
        "rotational_covariance": maximum_covariance
        <= float(construction["maximum_rotation_covariance_residual"]),
        "nonlinear_component_order_is_material": median_nonlinearity
        >= float(construction["minimum_nonlinear_superposition_residual"]),
        "deep_field_quadratic_cancellation_envelope": deep_error
        <= float(construction["maximum_deep_field_quadratic_cancellation_error"]),
        "high_field_suppression_envelope": high_fraction
        <= float(construction["maximum_high_field_fractional_action_correction"]),
        "parameter_count": int(config["physical_parameters"]["count"])
        <= int(config["physical_parameters"]["maximum_allowed"]),
    }
    gates = {name: bool(value) for name, value in gates.items()}

    report = {
        "status": "completed Sigma v6A pre-data action-envelope selection",
        "candidate": config["candidate"],
        "physical_parameter_count": int(config["physical_parameters"]["count"]),
        "causality_audit": {
            "maximum_retarded_pre_impulse_response": maximum_retarded_pre_response,
            "maximum_time_symmetric_pre_impulse_response": maximum_symmetric_pre_response,
            "maximum_zero_source_response": maximum_zero_source_response,
            "interpretation": "A traditional symmetric nonlocal variation is advanced as well as retarded; v6A must use a closed-time-path/in-in effective action or an equivalent derived causal prescription.",
        },
        "orientation_audit": {
            "maximum_rotation_covariance_residual_1000_trials": maximum_covariance,
            "median_nonlinear_superposition_residual_1000_trials": median_nonlinearity,
            "interpretation": "The trace-free Hessian response is coordinate-rotation covariant and responds differently when separated components are transformed before versus after summation.",
        },
        "activation_audit": {
            "f_chi": "chi exp(-sqrt(chi))",
            "deep_field_chi": tiny_chi,
            "deep_field_relative_quadratic_coefficient_error": deep_error,
            "high_field_acceleration_ratio": high_ratio,
            "high_field_fractional_correction": high_fraction,
        },
        "gates": gates,
        "all_construction_gates_pass": bool(all(gates.values())),
        "decision": "advance_v6A_to_full_CTP_variation_and_spectrum_only",
        "not_yet_demonstrated": [
            "a complete closed-time-path metric variation",
            "covariant conservation of the resulting physical equation",
            "positive scalar/vector/tensor spectrum on Minkowski and FLRW backgrounds",
            "luminal tensor propagation on inhomogeneous backgrounds",
            "the MOND/BTFR coefficient from the full spherical equation",
            "cluster lensing amplitude, topology, or held-out performance"
        ],
        "prior_art_boundary": "The scalar nonlocal MOND+lensing activation is published prior art. Only a healthy baryon-forced trace-free-Hessian orientation closure could become project-specific, and no novelty is claimed at this stage.",
        "data_policy": config["data_policy"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
