from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v12a_same_clock_dhost import (
    audit_v12a_same_clock_dhost,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the Sigma v12A same-clock DHOST selection.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v12a_same_clock_dhost_selection.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v12a_same_clock_dhost_selection",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fixed = config["fixed_values"]
    audit = audit_v12a_same_clock_dhost(
        k_b=float(fixed["k_b"]),
        k_2=float(fixed["k_2"]),
        lambda_s=float(fixed["lambda_s"]),
        orientation_strength=float(fixed["orientation_strength"]),
        background_kinetic_ratios=[float(value) for value in fixed["background_kinetic_ratios"]],
        signed_scan_limit=float(fixed["signed_scan_limit"]),
        signed_scan_points=int(fixed["signed_scan_points"]),
        high_acceleration_ratio=float(fixed["high_acceleration_ratio"]),
        random_rotation_trials=int(fixed["random_rotation_trials"]),
        random_seed=int(fixed["random_seed"]),
    )
    report = {
        "status": "completed Sigma v12A same-clock DHOST selection",
        **audit,
        "decision": "advance_only_to_complete_joint_adm_and_covariant_variation_gate",
        "reason": "The construction replaces the retired independent-memory family with exact higher-derivative degeneracy of the already baryon-forced AeST scalar. It adds no field or object state, preserves the flat AeST spectrum and one metric tensor cone, and carries trace-free directional information. These are selection conditions only; the shared metric can couple the AeST and DHOST constraint blocks, so the combined arbitrary-background Hamiltonian must be derived next.",
        "selection_rule": config["selection_rule"],
        "failure_rule_next": config["failure_rule_next"],
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
