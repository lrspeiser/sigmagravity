from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v12a_homogeneous_dirac import audit_v12a_homogeneous_dirac


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the Sigma v12A homogeneous aligned Dirac branch."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v12a_homogeneous_dirac.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v12a_homogeneous_dirac",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fixed = config["fixed_values"]
    report = {
        "status": "completed Sigma v12A homogeneous aligned Dirac subgate",
        "candidate": config["candidate"],
        **audit_v12a_homogeneous_dirac(
            f0=float(fixed["f0"]),
            k_2=float(fixed["k_2"]),
            orientation_strength=float(fixed["orientation_strength"]),
            background_clock=float(fixed["background_clock"]),
            clock_scan_minimum=float(fixed["clock_scan_minimum"]),
            clock_scan_maximum=float(fixed["clock_scan_maximum"]),
            clock_scan_points=int(fixed["clock_scan_points"]),
            random_velocity_trials=int(fixed["random_velocity_trials"]),
            random_seed=int(fixed["random_seed"]),
        ),
        "decision": "advance_to_spatial_gradient_and_aether_tilt_effective_bracket",
        "reason": "The Class-Ia terms form an exact degenerate square on the homogeneous branch and disappear from the reduced Hamiltonian. At the intended flat clock A3 and its mixing vanish, but the AeST clock susceptibility leaves the primary-secondary bracket equal to -4 K2, so the auxiliary pair is regular rather than strongly coupled.",
        "scope_limit": config["scope_limit"],
        "next_kill_gate": "Evaluate the full Delta_eff principal symbol with nonzero spatial B_i, finite aether tilt, and anisotropic K_ij; retire v12A if it vanishes or changes differential rank on any finite admitted background.",
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
