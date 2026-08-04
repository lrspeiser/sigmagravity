from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v12a_joint_adm_rank import audit_v12a_joint_adm_rank


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the Sigma v12A joint AeST--DHOST ADM rank.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v12a_joint_adm_rank.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v12a_joint_adm_rank",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fixed = config["fixed_values"]
    audit = audit_v12a_joint_adm_rank(
        k_b=float(fixed["k_b"]),
        random_trials=int(fixed["random_trials"]),
        random_seed=int(fixed["random_seed"]),
        finite_difference_step=float(fixed["finite_difference_step"]),
    )
    report = {
        "status": "completed Sigma v12A unreduced joint ADM kinetic-rank subgate",
        "candidate": config["candidate"],
        **audit,
        "decision": "advance_to_secondary_constraint_and_complete_dirac_chain",
        "reason": "After introducing B_mu=nabla_mu phi, all second-scalar-derivative velocities reside in the DHOST block. The AeST Maxwell term is connection-free and supplies a separate positive vector block; the remaining AeST scalar/aether interaction is affine in ADM velocities. It shifts momenta but cannot lift the exact Class-Ia null direction.",
        "scope_limit": config["scope_limit"],
        "next_kill_gate": "Derive the primary constraint explicitly in canonical variables, preserve it in time, and show that its Poisson bracket with the AeST unit-vector constraints generates one regular secondary rather than fixing a forbidden multiplier or changing rank.",
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
