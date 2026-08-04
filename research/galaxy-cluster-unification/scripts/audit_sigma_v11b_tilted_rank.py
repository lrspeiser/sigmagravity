from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v11b_tilted_rank import audit_v11b_tilted_rank


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Sigma v11B tilted rank.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "sigma_v11b_tilted_rank.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "sigma_v11b_tilted_rank")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fixed = config["fixed_values"]
    audit = audit_v11b_tilted_rank(
        aether_tilt=float(fixed["aether_tilt"]),
        shear_speed_squared=float(fixed["shear_speed_squared"]),
        bulk_weight=float(fixed["bulk_weight"]),
    )
    report = {
        "status": "completed Sigma v11B tilted-flow kinetic falsification",
        "candidate": config["candidate"],
        **audit,
        "decision": "retire_exact_v11b_before_observational_data",
        "reason": "The strain square becomes quartic in a material-coordinate velocity on a tilted slice. Its negative large-velocity curvature drives the physical phonon Legendre Hessian through zero while the material flow is still timelike.",
        "scope": "The failed direction is a physical material-coordinate velocity at fixed metric and aether. A negative Rayleigh quotient cannot be repaired by omitted off-diagonal mixing.",
        "data_policy": config["data_policy"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
