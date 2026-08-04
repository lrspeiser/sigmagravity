from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_degenerate_action import (
    newton_yukawa_circular_speed_ratio,
    newton_yukawa_log_acceleration_slope,
)


def longest_true_log_span(radius: np.ndarray, mask: np.ndarray) -> float:
    best = 0.0
    start: int | None = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        if start is not None and (not value or index == mask.size - 1):
            end = index if value and index == mask.size - 1 else index - 1
            best = max(best, float(np.log10(radius[end] / radius[start])))
            start = None
    return best


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the exterior acceleration law of the fixed Sigma v5C row."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v5c_exterior_law_audit.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v5c_exterior_law_audit",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    linear = config["linear_exterior_limit"]
    requirement = config["flat_curve_requirement"]
    radius = np.geomspace(
        float(linear["radius_over_range_min"]),
        float(linear["radius_over_range_max"]),
        int(linear["points"]),
    )
    records = []
    best_slope_span = 0.0
    best_speed_ratio = 0.0
    shallowest_slope = -np.inf
    for strength in linear["allowed_scalar_strengths"]:
        scalar_strength = float(strength)
        slope = newton_yukawa_log_acceleration_slope(radius, scalar_strength)
        flat_mask = (slope >= float(requirement["minimum_log_acceleration_slope"])) & (
            slope <= float(requirement["maximum_log_acceleration_slope"])
        )
        span = longest_true_log_span(radius, flat_mask)
        speed_ratio = newton_yukawa_circular_speed_ratio(
            10.0, radius, scalar_strength
        )
        local_best_speed = float(np.max(speed_ratio))
        best_slope_span = max(best_slope_span, span)
        best_speed_ratio = max(best_speed_ratio, local_best_speed)
        shallowest_slope = max(shallowest_slope, float(np.max(slope)))
        records.append(
            {
                "scalar_strength": scalar_strength,
                "shallowest_log_acceleration_slope": float(np.max(slope)),
                "steepest_log_acceleration_slope": float(np.min(slope)),
                "longest_flat_slope_span_dex": span,
                "maximum_speed_ratio_over_decade": local_best_speed,
            }
        )

    screened_vacuum_modification = False
    slope_pass = best_slope_span >= float(requirement["minimum_radial_span_dex"])
    speed_pass = (
        best_speed_ratio >= float(requirement["minimum_speed_ratio_over_decade"])
        and best_speed_ratio
        <= float(requirement["maximum_speed_ratio_over_decade"])
    )
    gates = {
        "screened_vacuum_differs_from_GR": screened_vacuum_modification,
        "linear_exterior_reaches_flat_acceleration_slope": slope_pass,
        "linear_exterior_holds_speed_over_one_decade": speed_pass,
    }
    gates = {name: bool(value) for name, value in gates.items()}
    report = {
        "status": "completed Sigma v5C theory-only exterior-law audit",
        "observational_data_accessed": False,
        "raw_holdout_opened": False,
        "screened_limit": {
            "Phi_prime_exterior": "G_N M/r^2",
            "Psi_prime_exterior": "G_N M/r^2",
            "M_prime": 0.0,
            "M_second": 0.0,
            "differs_from_GR": screened_vacuum_modification,
        },
        "linear_newton_yukawa_scan": records,
        "best_case": {
            "shallowest_log_acceleration_slope": shallowest_slope,
            "longest_flat_slope_span_dex": best_slope_span,
            "maximum_speed_ratio_over_decade": best_speed_ratio,
            "flat_curve_required_log_acceleration_slope": [-1.1, -0.9],
            "flat_curve_required_speed_ratio_over_decade": [0.9, 1.1],
        },
        "analytic_identity": "d log(g)/d log(r)=-2-alpha x^2 exp(-x)/[1+alpha(1+x)exp(-x)] <= -2",
        "gates": gates,
        "all_exterior_gates_pass": bool(all(gates.values())),
        "decision": "retire_fixed_sigma_v5c_row_before_full_variation_or_data",
        "scope": "This rejects the fixed canonical massive-scalar v5C row, not every luminal DHOST theory.",
        "reason": "Its screened exterior is exactly GR and its unscreened attractive scalar is no shallower than inverse-square, so it cannot sustain flat outer rotation curves.",
        "data_policy": config["data_policy"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
