from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v6_metric_memory import (
    bounded_tensor_coherence,
    hessian_power_regularities,
    metric_memory_activation,
    static_trace_free_projector,
    v6a_perturbation_action,
    v6b_metric_memory_chi,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retire the v6A cusp and select a differentiable v6B orientation closure."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v6b_differentiable_orientation_selection.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v6b_differentiable_orientation_selection",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    audit = config["perturbation_audit"]
    gradient = float(audit["gradient_coefficient"])
    hessian = float(audit["hessian_coefficient"])
    coupling = float(audit["orientation_coupling"])

    derivatives = []
    for step_value in audit["finite_difference_steps"]:
        step = float(step_value)
        center = float(v6a_perturbation_action(0.0, gradient, hessian, coupling))
        positive = float(v6a_perturbation_action(step, gradient, hessian, coupling))
        negative = float(v6a_perturbation_action(-step, gradient, hessian, coupling))
        right = (positive - center) / step
        left = (center - negative) / step
        derivatives.append(
            {
                "step": step,
                "right_first_variation": right,
                "left_first_variation": left,
                "jump": right - left,
            }
        )

    power_records = [hessian_power_regularities(float(power)) for power in audit["hessian_powers"]]

    amplitudes = np.geomspace(1.0e-6, 1.0e-3, 500)
    scalar_x = amplitudes**2
    tensor_z = amplitudes**2
    selected_chi = v6b_metric_memory_chi(scalar_x, tensor_z, 2.0, 1.0)
    chi_correction = selected_chi - scalar_x
    action_correction = metric_memory_activation(selected_chi) - metric_memory_activation(scalar_x)
    chi_order = float(np.polyfit(np.log(amplitudes), np.log(chi_correction), 1)[0])
    action_order = float(np.polyfit(np.log(amplitudes), np.log(action_correction), 1)[0])

    direction = np.array([1.0, -2.0, 4.0])
    projector_norms = [
        float(np.linalg.norm(static_trace_free_projector(scale * direction)))
        for scale in np.geomspace(1.0e-12, 1.0e12, 101)
    ]
    projector_variation = float(max(projector_norms) - min(projector_norms))
    coherence = bounded_tensor_coherence(np.geomspace(1.0e-18, 1.0e18, 1000), 1.0)

    thresholds = config["gates"]
    final_jump = float(derivatives[-1]["jump"])
    gates = {
        "v6A_unique_first_variation": False,
        "v6A_cusp_detected": final_jump >= float(thresholds["minimum_v6a_derivative_jump"]),
        "v6B_orientation_begins_at_fourth_order": chi_order
        >= float(thresholds["minimum_v6b_orientation_perturbation_order"]),
        "v6B_action_orientation_begins_at_fourth_order": abs(action_order - 4.0)
        <= float(thresholds["maximum_v6b_orientation_perturbation_order_error"]),
        "v6B_static_transfer_has_no_wavenumber_growth": projector_variation
        <= float(thresholds["maximum_static_projector_norm_variation"]),
        "v6B_coherence_is_bounded": float(np.min(coherence))
        >= float(thresholds["minimum_coherence"])
        and float(np.max(coherence)) <= float(thresholds["maximum_coherence"]),
        "parameter_count": int(config["physical_parameters"]["count"])
        <= int(config["physical_parameters"]["maximum_allowed"]),
    }
    gates = {name: bool(value) for name, value in gates.items()}
    v6b_gates = {name: value for name, value in gates.items() if not name.startswith("v6A_")}

    report = {
        "status": "completed Sigma v6A differentiability rejection and v6B pre-data selection",
        "observational_data_accessed": False,
        "raw_holdout_opened": False,
        "v6A": {
            "decision": "retire_exact_sqrt_Z_orientation_closure",
            "reason": "Z is second order in a metric perturbation, so sqrt(Z) is proportional to the absolute perturbation amplitude and the first variation is not unique at a zero-Hessian GR background.",
            "finite_difference_first_variations": derivatives,
            "analytic_derivative_jump": 2.0 * coupling * np.sqrt(hessian),
        },
        "power_classification": power_records,
        "v6B": {
            "decision": "advance_twice_retarded_bounded_coherence_to_full_CTP_variation",
            "envelope": config["v6b_envelope"],
            "physical_parameters": config["physical_parameters"],
            "orientation_chi_perturbation_order": chi_order,
            "orientation_action_perturbation_order": action_order,
            "static_projector_expected_norm": float(np.sqrt(2.0 / 3.0)),
            "static_projector_norm_variation_24_decades": projector_variation,
            "coherence_minimum": float(np.min(coherence)),
            "coherence_maximum": float(np.max(coherence)),
            "why_it_evades_v6A": "The bounded Q-tensor coherence is analytic at Q=0 and is multiplied by X, so it first changes the action at fourth perturbative order and does not alter the zero-background quadratic propagator.",
        },
        "gates": gates,
        "all_v6B_selection_gates_pass": bool(all(v6b_gates.values())),
        "not_yet_demonstrated": [
            "complete closed-time-path variation",
            "diffeomorphism Ward identity after the retarded physical limit",
            "health about nonzero galactic, cluster, and FLRW memory backgrounds",
            "absence of nonlinear secular growth from the second retarded inverse",
            "static spherical MOND coefficient",
            "any galaxy or cluster performance"
        ],
        "data_policy": config["data_policy"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
