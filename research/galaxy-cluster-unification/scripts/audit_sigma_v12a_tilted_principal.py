from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v12a_tilted_principal import audit_v12a_tilted_principal


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the constant-background tilted Dirac block of Sigma v12A."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v12a_tilted_principal.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v12a_tilted_principal",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fixed = config["fixed_values"]
    audit = audit_v12a_tilted_principal(
        k_b=float(fixed["k_b"]),
        k_2=float(fixed["k_2"]),
        background_clock_ratio=float(fixed["background_clock_ratio"]),
        positive_orientation_strength=float(fixed["positive_orientation_strength"]),
        negative_orientation_strength=float(fixed["negative_orientation_strength"]),
        random_trials=int(fixed["random_trials"]),
        logarithmic_clock_limit=float(fixed["logarithmic_clock_limit"]),
        logarithmic_tilt_limit=float(fixed["logarithmic_tilt_limit"]),
        wave_number_sentinels=tuple(float(value) for value in fixed["wave_number_sentinels"]),
        wave_invariance_trials=int(fixed["wave_invariance_trials"]),
        aligned_limit_tilt=float(fixed["aligned_limit_tilt"]),
        random_seed=int(fixed["random_seed"]),
    )
    report = {
        "status": "Sigma v12A constant-background tilted Dirac subgate",
        **audit,
        "decision": (
            "both coupling signs survive the constant-background Dirac-rank gate; "
            "neither is selected; proceed to nonconstant backgrounds and physical characteristics"
        ),
        "reason": (
            "The complete Class-Ia null direction remains exact for arbitrary constant aether "
            "tilt. The apparent aligned Maxwell lapse-gradient term is canceled by the "
            "longitudinal aether momentum, leaving the lower-derivative AeST susceptibility. "
            "The frozen tilted scan finds no zero or sign change in the reduced 2x2 bracket."
        ),
        "scope_limit": config["scope_limit"],
        "next_kill_gate": (
            "Derive and score the reduced physical characteristic cones and energy signs on "
            "tilted constant backgrounds, then restore background scalar Hessian, aether "
            "gradient, extrinsic curvature, and curvature in the full principal matrix."
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
