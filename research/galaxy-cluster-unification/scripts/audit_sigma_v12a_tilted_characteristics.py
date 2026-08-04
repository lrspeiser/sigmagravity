from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v12a_tilted_characteristics import (
    audit_v12a_tilted_characteristics,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit finite-tilt v12A characteristics in scalar-unitary time."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v12a_tilted_characteristics.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v12a_tilted_characteristics",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fixed = config["fixed_values"]
    report = audit_v12a_tilted_characteristics(
        k_b=float(fixed["k_b"]),
        k_2=float(fixed["k_2"]),
        background_clock_ratio=float(fixed["background_clock_ratio"]),
        orientation_strengths=tuple(
            float(value) for value in fixed["orientation_strengths"]
        ),
        scalar_clock_ratios=tuple(
            float(value) for value in fixed["scalar_clock_ratios"]
        ),
        tilt_magnitudes=tuple(float(value) for value in fixed["tilt_magnitudes"]),
        relative_angles_degrees=tuple(
            float(value) for value in fixed["relative_angles_degrees"]
        ),
        grid_wave_number=float(fixed["grid_wave_number"]),
        convergence_wave_numbers=tuple(
            float(value) for value in fixed["convergence_wave_numbers"]
        ),
        convergence_sentinels=tuple(fixed["convergence_sentinels"]),
        principal_growth_threshold=float(fixed["principal_growth_threshold"]),
        metric_cone_frequency_tolerance=float(
            fixed["metric_cone_frequency_tolerance"]
        ),
        polynomial_residual_tolerance=float(fixed["polynomial_residual_tolerance"]),
    )
    output = {
        "status": "Sigma v12A finite-tilt scalar-unitary characteristic gate",
        **report,
        "decision": (
            "scalar-unitary metric time fails on finite backgrounds; do not open data and do "
            "not yet retire the covariant action; derive the general-time common-cone test"
        ),
        "reason": (
            "The finite physical root count survives, but the frozen grid contains persistent "
            "principal exponential roots, frequencies outside the metric light scale, and "
            "negative coordinate-time quadratic energies. A different common metric-timelike "
            "Cauchy covector may still exist, so this subgate is diagnostic rather than a final "
            "covariant falsification."
        ),
        "scope_limit": config["scope_limit"],
        "next_kill_gate": (
            "Generalize the principal symbol to a scalar background with spatial gradient, "
            "scan general metric-timelike time covectors, and either exhibit one common "
            "hyperbolic positive-energy cone or retire exact v12A before observations."
        ),
        "data_policy": config["data_policy"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
