from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v11c_biot_stretch_rank import (
    audit_v11c_biot_stretch_rank,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Sigma v11C Biot-stretch tilted kinetic rank."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v11c_biot_stretch_rank.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v11c_biot_stretch_rank",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fixed = config["fixed_values"]
    audit = audit_v11c_biot_stretch_rank(
        aether_tilt=float(fixed["aether_tilt"]),
        shear_speed_squared=float(fixed["shear_speed_squared"]),
        bulk_weight=float(fixed["bulk_weight"]),
        transverse_stretch=float(fixed["transverse_stretch"]),
        counterexample_axial_stretch=float(fixed["counterexample_axial_stretch"]),
        finite_difference_step=float(fixed["finite_difference_step"]),
    )
    report = {
        "status": "completed Sigma v11C Biot-stretch kinetic falsification",
        "candidate": config["candidate"],
        **audit,
        "decision": "retire_exact_v11c_and_reset_material_memory_mechanism",
        "reason": "The Biot strain removes v11B's one-dimensional negative quartic, but its finite anisotropic bulk/shear curvature is too large on an exact rank-one material-shear direction. A tilted foliation converts that spatial curvature into a negative physical coordinate-velocity Hessian while Q=0 material flow remains timelike.",
        "scope": "The counterexample lies in GL+(3), has finite action, and is a physical material-coordinate Rayleigh direction at fixed metric and aether. Omitted mixing cannot make a quadratic form positive when it is already negative on this direction.",
        "stopping_rule": config["stopping_rule"],
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
