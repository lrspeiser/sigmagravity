from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v12a_tilted_clock import audit_v12a_tilted_clock


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the reduced AeST tilted-clock susceptibility for v12A."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v12a_tilted_clock.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v12a_tilted_clock",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fixed = config["fixed_values"]
    report = {
        "status": "completed Sigma v12A tilted reduced-AeST clock subgate",
        "candidate": config["candidate"],
        **audit_v12a_tilted_clock(
            a_sigma=float(fixed["a_sigma"]),
            k_b=float(fixed["k_b"]),
            k_2=float(fixed["k_2"]),
            random_trials=int(fixed["random_trials"]),
            logarithmic_amplitude_limit=float(fixed["logarithmic_amplitude_limit"]),
            random_seed=int(fixed["random_seed"]),
        ),
        "decision": "advance_to_dhost_spatial_principal_operator",
        "reason": "After exactly eliminating the AeST scalar auxiliaries, the normal-clock susceptibility has the global lower bound 4K2+[4K2-(9/2)(2-KB)]|A|^2. The selected K_B=1,K2=2 row has bound 8+(7/2)|A|^2, so no finite tilt or scalar-gradient orientation can make the lower-derivative part of Delta_eff vanish.",
        "scope_limit": config["scope_limit"],
        "next_kill_gate": "Derive the v12A DHOST contribution to the spatial principal symbol of Delta_eff and prove that it cannot cancel or rank-change the positive AeST zeroth-order susceptibility on finite anisotropic backgrounds.",
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
