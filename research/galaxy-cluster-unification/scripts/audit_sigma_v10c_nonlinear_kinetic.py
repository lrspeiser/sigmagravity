from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v10c_nonlinear_kinetic import audit_v10c_nonlinear_kinetic


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the Sigma v10C finite-amplitude kinetic matrix."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v10c_nonlinear_kinetic.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v10c_nonlinear_kinetic",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    audit = audit_v10c_nonlinear_kinetic(k_b=float(config["fixed_values"]["K_B"]))
    report = {
        "status": "completed Sigma v10C nonlinear kinetic falsification",
        "candidate": config["candidate"],
        **audit,
        "decision": "retire_exact_v10c_before_observational_data",
        "reason": "The spatial constraint converts the first-order carrier/aether interaction into an amplitude-dependent physical vector kinetic matrix K_B I-beta P. At the finite isotropic amplitude P=sqrt(11 K_B/2) I an eigenvalue vanishes, and immediately above it three physical vector directions are ghosts. The convex quartic potential is finite there and the spatiality constraint imposes no amplitude bound.",
        "no_patch_rule": "Do not impose a data- or object-selected P cutoff. Removing beta removes the mechanism; bounding the kinetic mixing or completing it as a positive square defines a materially new action.",
        "scope": "The failure occurs in a local homogeneous decoupling subblock containing the physical AeST vector modes, so lapse/shift constraints cannot restore the negative kinetic directions. A full ADM count is unnecessary for retiring this exact action.",
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
