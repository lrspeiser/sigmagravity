from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v10a_spatial_polarization import (
    constant_mixing_ellipticity_thresholds,
    static_high_k_mixed_spectrum,
)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Sigma v10A static ellipticity.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v10a_quasistatic_ellipticity_gate.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v10a_quasistatic_ellipticity_gate",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fixed = config["fixed_values"]
    scan = config["scan"]
    carrier_speed = float(fixed["carrier_speed_squared"])
    beta = float(fixed["mixing_beta"])
    positive_ratios = np.geomspace(
        float(scan["acceleration_ratio_min_positive"]),
        float(scan["acceleration_ratio_max"]),
        int(scan["acceleration_ratio_points"]),
    )
    ratios = np.concatenate(([0.0], positive_ratios))
    cosines = np.asarray(scan["propagation_cosines"], dtype=float)
    rows = [
        static_high_k_mixed_spectrum(
            float(ratio),
            propagation_cosine=float(cosine),
            carrier_speed_squared=carrier_speed,
            mixing_beta=beta,
        )
        for ratio in ratios
        for cosine in cosines
    ]
    minimum = min(rows, key=lambda row: float(np.min(row["gradient_eigenvalues"])))
    negative_count = sum(not row["elliptic"] for row in rows)
    angle_summaries = {}
    for cosine in cosines:
        subset = [row for row in rows if row["propagation_cosine"] == float(cosine)]
        first_positive = next((row for row in subset if row["elliptic"]), None)
        angle_summaries[str(float(cosine))] = {
            "minimum_gradient_eigenvalue": float(
                min(np.min(row["gradient_eigenvalues"]) for row in subset)
            ),
            "first_scanned_elliptic_acceleration_ratio": None
            if first_positive is None
            else float(first_positive["acceleration_ratio"]),
            "negative_points": int(sum(not row["elliptic"] for row in subset)),
        }
    thresholds = constant_mixing_ellipticity_thresholds(
        carrier_speed_squared=carrier_speed,
        mixing_beta=beta,
    )
    zero_field = static_high_k_mixed_spectrum(
        0.0,
        propagation_cosine=0.0,
        carrier_speed_squared=carrier_speed,
        mixing_beta=beta,
    )
    exact_no_go = bool(
        beta != 0.0
        and np.min(zero_field["gradient_eigenvalues"]) < 0.0
        and not thresholds["globally_elliptic_for_all_nonnegative_accelerations"]
    )
    report = {
        "status": "completed Sigma v10A quasistatic ellipticity falsification",
        "parent_protocol": config["parent_protocol"],
        "fixed_values": fixed,
        "analytic_thresholds": thresholds,
        "zero_field_block": zero_field,
        "scan_points": len(rows),
        "negative_or_zero_ellipticity_points": int(negative_count),
        "minimum_scan_row": minimum,
        "angle_summaries": angle_summaries,
        "mass_or_quartic_can_repair_high_k_sign": False,
        "exact_constant_mixing_no_go": exact_no_go,
        "gate_passed": not exact_no_go and negative_count == 0,
        "decision": "retire_exact_v10A_constant_derivative_mixing",
        "reason": "On the frozen simple-mu quasistatic branch, K_T and K_L both vanish with acceleration. The nonzero constant beta leaves the high-k 2x2 determinant K(theta)c_P^2-beta^2 negative in a finite low-field interval. At the selected row beta^2/c_P^2=0.5625: transverse perturbations require x>1.285714 and longitudinal perturbations require x>0.511858. The zero-field block has a negative eigenvalue. Carrier mass and quartic convexity are k^0 terms and cannot change this k^2 result.",
        "scope": "This falsifies the exact v10A constant derivative coupling on its intended AeST simple-mu quasistatic branch. It does not falsify every spatial tensor carrier. A successor must derive a positive global principal matrix rather than multiply beta by a fitted galaxy/cluster or acceleration gate.",
        "next_mechanism_requirement": "Use a manifestly positive/degenerate coupling or source the tensor from a sector with nonvanishing principal stiffness, while retaining trace, STF orientation, one metric, five constants, and unique baryon-forced boundary data.",
        "observational_data_accessed": False,
        "new_observational_product_accessed": False,
        "raw_holdout_opened": False,
        "data_policy": config["data_policy"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, cls=NumpyEncoder) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, cls=NumpyEncoder))


if __name__ == "__main__":
    main()
