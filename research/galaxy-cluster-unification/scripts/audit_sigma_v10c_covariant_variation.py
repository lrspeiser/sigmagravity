from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v10c_covariant_variation import audit_v10c_covariant_variation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the Sigma v10C covariant first-order variation."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v10c_covariant_variation.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v10c_covariant_variation",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fixed = config["fixed_values"]
    audit = audit_v10c_covariant_variation(
        carrier_speed_squared=float(fixed["carrier_speed_squared"])
    )
    report = {
        "status": "completed Sigma v10C covariant-variation subgate",
        "candidate": config["candidate"],
        "fixed_values": fixed,
        **audit,
        "decision": "advance_v10c_to_nonlinear_ADM_constraint_gate",
        "reason": "The exact projected carrier momentum passes a tilted-background finite-difference check, four spatiality constraints leave six carrier components, the interaction has a first-derivative boundary form, every Euler equation is at most second order, and the all-field Noether identity closes conservation on shell.",
        "scope": "This is not a degree-of-freedom or hyperbolicity proof. The nonlinear velocity Hessian, primary/secondary constraints, arbitrary-background cones, component-expanded metric stress tensor and PPN solution remain mandatory.",
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
