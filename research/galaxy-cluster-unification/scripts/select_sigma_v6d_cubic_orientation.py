from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v6_metric_memory import (
    v6c_total_constitutive_coefficient,
    v6d_cubic_orientation_correction,
    v6d_deep_acceleration_enhancement,
    v6d_parallel_ellipticity_coefficient,
    v6d_total_constitutive_coefficient,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reject v6C's quadratic placement and select cubic-only v6D orientation."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v6d_cubic_orientation_selection.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v6d_cubic_orientation_selection",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    surrogate = config["weak_scalar_surrogate"]
    x_min, x_max = (float(value) for value in surrogate["X_scan"])
    invariant = np.geomspace(x_min, x_max, int(surrogate["X_scan_points"]))
    strengths = [0.0, 0.1, 0.5, 0.9, 0.99, float(surrogate["allowed_q_interval"][1])]

    v6c_records = []
    v6d_records = []
    minimum_mu = np.inf
    minimum_parallel = np.inf
    maximum_v6c_limit_error = 0.0
    for strength in strengths:
        v6c_mu = v6c_total_constitutive_coefficient(invariant, strength)
        v6c_limit_error = abs(float(v6c_mu[0]) + strength)
        maximum_v6c_limit_error = max(maximum_v6c_limit_error, v6c_limit_error)
        v6c_records.append(
            {
                "q": strength,
                "smallest_X_coefficient": float(v6c_mu[0]),
                "analytic_X_to_zero_limit": -strength,
                "minimum_coefficient": float(np.min(v6c_mu)),
            }
        )
        v6d_mu = v6d_total_constitutive_coefficient(invariant, strength)
        v6d_parallel = v6d_parallel_ellipticity_coefficient(invariant, strength)
        minimum_mu = min(minimum_mu, float(np.min(v6d_mu)))
        minimum_parallel = min(minimum_parallel, float(np.min(v6d_parallel)))
        v6d_records.append(
            {
                "q": strength,
                "minimum_constitutive_coefficient": float(np.min(v6d_mu)),
                "minimum_parallel_ellipticity": float(np.min(v6d_parallel)),
                "deep_mu_over_sqrt_X": float(v6d_mu[0] / np.sqrt(invariant[0])),
                "analytic_deep_mu_over_sqrt_X": 1.5 * (1.0 - strength),
                "deep_acceleration_enhancement": (
                    v6d_deep_acceleration_enhancement(strength)
                    if strength < 1.0
                    else float("inf")
                ),
            }
        )

    scaling_x = np.geomspace(1.0e-10, 1.0e-5, 1000)
    base = v6d_cubic_orientation_correction(scaling_x, 0.0)
    oriented = v6d_cubic_orientation_correction(scaling_x, 0.8)
    cubic_power = float(np.polyfit(np.log(scaling_x), np.log(oriented - base), 1)[0])
    enhancement_099 = v6d_deep_acceleration_enhancement(0.99)
    thresholds = config["gates"]
    gates = {
        "v6C_negative_deep_response_detected": all(
            record["minimum_coefficient"] < 0.0 for record in v6c_records if record["q"] > 0.0
        ),
        "v6C_deep_limit_matches_minus_q": maximum_v6c_limit_error
        <= float(thresholds["maximum_v6c_deep_coefficient_error"]),
        "v6D_constitutive_response_positive": minimum_mu
        > float(thresholds["minimum_v6d_constitutive_coefficient"]),
        "v6D_parallel_ellipticity_positive": minimum_parallel
        > float(thresholds["minimum_v6d_parallel_ellipticity"]),
        "v6D_orientation_is_cubic_only": abs(cubic_power - 1.5)
        <= float(thresholds["maximum_cubic_power_error"]),
        "v6D_cluster_sized_deep_enhancement_is_available": enhancement_099
        >= float(thresholds["minimum_deep_enhancement_at_q_099"])
        and enhancement_099 <= float(thresholds["maximum_deep_enhancement_at_q_099"]),
        "parameter_count": int(config["physical_parameters"]["count"])
        <= int(config["physical_parameters"]["maximum_allowed"]),
    }
    gates = {name: bool(value) for name, value in gates.items()}
    report = {
        "status": "completed Sigma v6C nonzero-background rejection and v6D selection",
        "observational_data_accessed": False,
        "raw_holdout_opened": False,
        "v6C": {
            "decision": "retire_orientation_inside_quadratic_cancellation",
            "records": v6c_records,
            "maximum_deep_limit_error": maximum_v6c_limit_error,
            "reason": "For fixed nonzero C_Q, chi=(1+q)X changes the quadratic cancellation and the total deep constitutive coefficient tends to -q, producing a negative-response interval for every q>0.",
        },
        "v6D": {
            "decision": "advance_cubic_only_orientation_to_complete_CTP_variation",
            "weak_scalar_surrogate": surrogate,
            "tensor_memory": config["tensor_memory"],
            "physical_parameters": config["physical_parameters"],
            "records": v6d_records,
            "global_minimum_constitutive_coefficient": minimum_mu,
            "global_minimum_parallel_ellipticity": minimum_parallel,
            "measured_orientation_power_in_X": cubic_power,
            "deep_acceleration_enhancement_at_q_0.99": enhancement_099,
            "why_it_evades_v6C": "The orientation factor multiplies only X^(3/2), leaving the X term available to cancel the Einstein quadratic action for every C_Q. The remaining deep coefficient is 1-q and stays positive for q<1.",
        },
        "gates": gates,
        "all_v6D_selection_gates_pass": bool(all(gates.values())),
        "not_yet_demonstrated": [
            "complete covariant CTP influence action and metric variation",
            "Ward identity with metric-built u_mu and projected tensor memory",
            "full scalar-vector-tensor kinetic matrix on nonzero backgrounds",
            "spherical coefficient including both metric potentials",
            "whether C_Q is small in coherent disks and large in real clusters",
            "any observational performance"
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
