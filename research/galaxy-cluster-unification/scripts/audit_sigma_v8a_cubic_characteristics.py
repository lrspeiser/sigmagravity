from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v8_aest_galileon import (
    negative_cubic_branch_limit,
    positive_cubic_causality_limit,
    spherical_positive_cubic_characteristics,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the v8A cubic spherical characteristic tradeoff."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v8a_cubic_characteristic_gate.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v8a_cubic_characteristic_gate",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fixed = config["fixed_values"]
    gates_config = config["gates"]
    base = float(fixed["base_scalar_speed_squared"])
    deep_u = float(fixed["deep_branch_probe"])

    limit = positive_cubic_causality_limit(base_speed_squared=base)
    threshold = spherical_positive_cubic_characteristics(
        float(limit["dimensionless_tangential_hessian"]),
        base_speed_squared=base,
    )
    deep = spherical_positive_cubic_characteristics(
        deep_u,
        base_speed_squared=base,
    )
    opposite_sign = negative_cubic_branch_limit(base_speed_squared=base)

    scan_u = np.geomspace(1.0e-10, deep_u, 20000)
    radial_speed_squared = np.array(
        [
            spherical_positive_cubic_characteristics(
                float(value),
                base_speed_squared=base,
            )["radial_speed_squared"]
            for value in scan_u
        ],
        dtype=float,
    )
    first_superluminal = int(np.flatnonzero(radial_speed_squared > 1.0)[0])
    order_unity_minimum = float(
        gates_config["minimum_nonlinear_fraction_for_order_unity_geometry"]
    )
    maximum_causal_fraction = float(
        limit["maximum_nonlinear_fraction_of_total_flux"]
    )
    gates = {
        "flat_base_is_causal": base <= float(gates_config["maximum_speed_squared"]),
        "positive_branch_remains_hyperbolic_in_deep_probe": bool(deep["positive"]),
        "positive_branch_remains_causal_in_deep_probe": bool(deep["causal"]),
        "order_unity_geometry_available_before_superluminality": (
            maximum_causal_fraction >= order_unity_minimum
        ),
        "negative_sign_has_global_positive_source_branch": (
            float(opposite_sign["radial_spatial_coefficient_at_endpoint"])
            > float(gates_config["minimum_spatial_eigenvalue"])
        ),
    }
    report = {
        "status": "completed Sigma v8A cubic spherical characteristic gate",
        "candidate": config["candidate"],
        "fixed_values": fixed,
        "analytic_positive_sign_light_crossing": {
            **limit,
            "radial_speed_squared": float(threshold["radial_speed_squared"]),
        },
        "positive_sign_deep_branch": deep,
        "opposite_sign_branch_endpoint": opposite_sign,
        "scan_cross_check": {
            "samples": int(scan_u.size),
            "first_superluminal_dimensionless_hessian": float(
                scan_u[first_superluminal]
            ),
            "maximum_radial_speed_squared": float(np.max(radial_speed_squared)),
        },
        "gates": {name: bool(value) for name, value in gates.items()},
        "all_required_gates_pass": bool(all(gates.values())),
        "decision": "retire_v8A_cubic_geometry_interaction_retain_one_metric_AeST_base",
        "reason": "For the selected positive sign, radial scalar characteristics cross the physical light cone when the cubic term supplies only 17.7% of the conserved spherical flux, and approach c_r^2=4/3 in the nonlinear limit. Reversing the sign makes the positive-source exterior branch terminate where its radial ellipticity vanishes. Zero coupling is healthy but supplies no Hessian geometry response.",
        "scope": "This is an analytic and numerical necessary-condition failure of the exact cubic geometry interaction in its scalar decoupling regime. It is not a full characteristic determinant for every AeST-Horndeski theory and does not reject the one-metric AeST base or other degenerate geometry interactions.",
        "observational_data_accessed": False,
        "raw_holdout_opened": False,
        "data_policy": config["data_policy"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
