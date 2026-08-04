from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v12a_aligned_finite_k import (
    audit_v12a_aligned_finite_k,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the aligned finite-k Dirac symbol of Sigma v12A."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v12a_aligned_finite_k.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v12a_aligned_finite_k",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fixed = config["fixed_values"]
    audit = audit_v12a_aligned_finite_k(
        k_2=float(fixed["k_2"]),
        background_clock_ratio=float(fixed["background_clock_ratio"]),
        selected_positive_strength=float(fixed["selected_positive_strength"]),
        surviving_negative_strength=float(fixed["surviving_negative_strength"]),
        counterexample_clock_ratio=float(fixed["counterexample_clock_ratio"]),
        random_trials=int(fixed["random_trials"]),
        logarithmic_clock_limit=float(fixed["logarithmic_clock_limit"]),
        logarithmic_wave_limit=float(fixed["logarithmic_wave_limit"]),
        random_seed=int(fixed["random_seed"]),
    )
    report = {
        "status": "completed Sigma v12A aligned finite-k sign subgate",
        **audit,
        "decision": "retire_positive_lambda_D_and_advance_negative_branch_to_tilted_anisotropic_symbol",
        "reason": "The frozen lambda_D=+1 row has A4<0 and an exact finite-wave-vector zero in the primary-secondary bracket. The same covariant action has an analytically safe aligned branch for -8/sqrt(1+x0^2)<=lambda_D<0; lambda_D=-1 is retained as a theory-only sentinel, not an observational refit.",
        "scope_limit": config["scope_limit"],
        "next_kill_gate": "Derive the full negative-branch Delta_eff principal matrix with nonzero scalar spatial gradient, finite aether tilt, arbitrary wave-vector orientation, and anisotropic metric/aether perturbations.",
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
