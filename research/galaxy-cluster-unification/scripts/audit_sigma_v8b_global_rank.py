from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v8b_global_rank import audit_global_rank_falsification


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the Sigma v8B global Legendre-rank falsification."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v8b_global_rank_falsification.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v8b_global_rank_falsification",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    selected = config["selected_values"]
    robustness = config["robustness_values"]
    audit = audit_global_rank_falsification(
        ell_h_q0_values=robustness["L_H_Q0"],
        a_sigma_over_q0_values=robustness["a_sigma_over_Q0"],
        k_b_escape_values=robustness["K_B_escape_scan"],
        selected_aether_velocity=float(selected["selected_aether_velocity"]),
        k_b=float(selected["K_B"]),
        alpha=float(selected["alpha"]),
    )
    report = {
        "status": "completed Sigma v8B global Legendre-rank falsification",
        "candidate": config["candidate"],
        "protocol_version": config["protocol_version"],
        "selected_values": selected,
        **audit,
        "decision": "retire_exact_v8B_before_data",
        "reason": (
            "For the frozen K_B=1, the leading large-Q kinetic mixing has a "
            "positive Schur coefficient above finite aether tilt 0.902088. Every "
            "nonzero tested completion length therefore reaches a finite rank-zero "
            "surface. The crossing has finite canonical energy, mixes metric and "
            "aether velocities, and adds a second raw negative direction. "
            "Raising K_B removes that asymptotic sign but cannot rescue the action: "
            "its determinant is affine in isotropic extrinsic curvature and crosses "
            "zero at finite K for every tested K_B in the healthy open interval."
        ),
        "zero_coupling_limit": (
            "L_H=0 removes both the cubic geometry response and its causal partner, "
            "leaving the AeST base without the proposed cluster-Hessian mechanism. "
            "Alpha=1 retains the v8A cubic but removes its causal partner and restores "
            "the already-demonstrated superluminal nonlinear scalar cone."
        ),
        "scope": (
            "This is a necessary global Legendre-regularity failure, not a complete "
            "Dirac constraint solution. A full Dirac calculation cannot make the "
            "already demonstrated global velocity Hessian regular; avoiding the "
            "surface would require a separately derived invariant domain restriction."
        ),
        "decision_rule": config["decision_rule"],
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
