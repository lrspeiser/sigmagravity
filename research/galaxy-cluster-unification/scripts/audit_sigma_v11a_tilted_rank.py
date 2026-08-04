from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v11a_tilted_rank import audit_v11a_tilted_rank


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the Sigma v11A tilted nonlinear kinetic rank."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v11a_tilted_rank.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v11a_tilted_rank",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fixed = config["fixed_values"]
    audit = audit_v11a_tilted_rank(
        aether_tilt=float(fixed["aether_tilt"]),
        acceleration_scale=float(fixed["acceleration_scale"]),
        memory_speed_squared=float(fixed["memory_speed_squared"]),
        anisotropy_fraction=float(fixed["anisotropy_fraction"]),
        base_scalar_velocity_hessian=float(
            fixed["base_scalar_velocity_hessian"]
        ),
        finite_difference_step=float(fixed["finite_difference_step"]),
    )
    report = {
        "status": "completed Sigma v11A tilted nonlinear kinetic falsification",
        "candidate": config["candidate"],
        **audit,
        "decision": "retire_exact_v11a_before_observational_data",
        "reason": "On a finite tilted-aether background, the bounded S-alignment is a concave function of the coordinate scalar velocity. Its coefficient is multiplied by the unbounded but finite memory spatial-gradient energy. At the reported finite configuration the scalar Legendre Hessian vanishes and immediately beyond it is negative.",
        "no_patch_rule": config["no_patch_rule"],
        "scope": "A negative diagonal Rayleigh direction of the fixed-metric scalar subblock is sufficient to retire the exact action; omitted velocity mixing cannot make that quadratic form positive. The failure persists for any finite positive base scalar Hessian, with the critical memory gradient shifted but finite.",
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
