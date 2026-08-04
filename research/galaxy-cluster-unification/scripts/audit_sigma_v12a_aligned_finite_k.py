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
        k_b=float(fixed["k_b"]),
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
        "status": "corrected Sigma v12A aligned finite-k dynamical-aether subgate",
        **audit,
        "decision": "withdraw_positive_sign_falsification_and_advance_both_signs_to_full_tilted_anisotropic_symbol",
        "reason": "The first clock-only block omitted the conformal metric component of the Class-Ia null direction. The second correction included that metric component but held the longitudinal aether velocity fixed. Class-Ia cancels the DHOST/metric gradients, and the dynamical aether Schur term then cancels the apparent Maxwell lapse gradient. The fully reduced aligned bracket is the nonzero AeST clock susceptibility -4 K2 for both signs.",
        "scope_limit": config["scope_limit"],
        "next_kill_gate": "Derive the full Delta_eff principal matrix for both orientation signs with nonzero scalar spatial gradient, finite aether tilt, arbitrary wave-vector orientation, and anisotropic metric/aether perturbations.",
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
