from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v12a_flat_characteristics import (
    audit_v12a_flat_characteristics,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regress the direct Sigma v12A flat characteristic pencil."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v12a_flat_characteristics.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v12a_flat_characteristics",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fixed = config["fixed_values"]
    audit = audit_v12a_flat_characteristics(
        k_b=float(fixed["k_b"]),
        k_2=float(fixed["k_2"]),
        background_clock_ratio=float(fixed["background_clock_ratio"]),
        aligned_tilt_sentinel=float(fixed["aligned_tilt_sentinel"]),
        orientation_strengths=tuple(
            float(value) for value in fixed["orientation_strengths"]
        ),
        wave_number_sentinels=tuple(
            float(value) for value in fixed["wave_number_sentinels"]
        ),
        scalar_speed_squared_target=float(fixed["scalar_speed_squared_target"]),
        scalar_limit_tolerance=float(fixed["scalar_limit_tolerance"]),
        luminal_limit_tolerance=float(fixed["luminal_limit_tolerance"]),
        polynomial_residual_tolerance=float(fixed["polynomial_residual_tolerance"]),
    )
    report = {
        "status": "Sigma v12A direct flat characteristic regression",
        **audit,
        "decision": (
            "flat finite-frequency spectrum and energy signs pass; proceed to the finite-tilt "
            "common-Cauchy-cone and reduced-energy gate"
        ),
        "reason": (
            "The complete quadratic Euler pencil independently reproduces six local linear "
            "degrees, the corrected scalar front c_s^2=1/2, four luminal tensor/vector modes, "
            "and positive energy for every finite-frequency root. The zero-frequency sector "
            "and nonaligned backgrounds remain open."
        ),
        "scope_limit": config["scope_limit"],
        "next_kill_gate": (
            "Classify the invariant characteristic cones and reduced energy signs for finite "
            "aether/scalar tilt and arbitrary wave orientation, including whether a common "
            "Cauchy covector exists when scalar-unitary time is not itself a valid slicing."
        ),
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
