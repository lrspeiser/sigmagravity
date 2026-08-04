from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v12a_primary_dirac import audit_v12a_primary_dirac


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the Sigma v12A canonical primary Dirac subgate."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v12a_primary_dirac.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v12a_primary_dirac",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fixed = config["fixed_values"]
    audit = audit_v12a_primary_dirac(
        random_trials=int(fixed["random_trials"]),
        random_seed=int(fixed["random_seed"]),
    )
    report = {
        "status": "completed Sigma v12A canonical-primary Dirac subgate",
        "candidate": config["candidate"],
        **audit,
        "decision": "advance_to_explicit_secondary_density_and_effective_bracket",
        "reason": "The published AeST reduced metric momentum is exactly the GR momentum, so adding AeST does not alter the Class-Ia canonical primary. The primary is independent of the AeST auxiliary momenta, forcing one secondary. Regularity is controlled by the Schur-complement bracket Delta_eff after eliminating the mu/nu auxiliary pairs.",
        "scope_limit": config["scope_limit"],
        "next_kill_gate": "Derive Omega_Sigma locally for the fixed A3(X), compute Delta_eff=Delta-E C^-1 D including spatial derivative operators, and prove it remains a regular invertible operator on all admitted timelike-gradient and aether-tilt backgrounds.",
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
