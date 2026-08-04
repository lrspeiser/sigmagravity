from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v11b_elastic_triad import audit_v11b_elastic_triad


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the Sigma v11B stress-free elastic-triad selection."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v11b_elastic_triad_selection.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v11b_elastic_triad_selection",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fixed = config["fixed_values"]
    audit = audit_v11b_elastic_triad(
        shear_speed_squared=float(fixed["shear_speed_squared"]),
        longitudinal_speed_squared=float(fixed["longitudinal_speed_squared"]),
        physical_parameter_count=int(fixed["physical_parameter_count"]),
        maximum_physical_parameters=int(fixed["maximum_physical_parameters"]),
        random_directions=int(fixed["random_directions"]),
    )
    report = {
        "status": "completed Sigma v11B theory-only architecture selection",
        "candidate": config["candidate"],
        **audit,
        "decision": "advance_v11b_only_to_nonlinear_tilted_rank_and_metric_constraint_gates",
        "reason": "The aether-provided time square and strain-square potential vanish with their first variations in the unstrained vacuum, while the three connection-free scalar coordinates carry two shear phonons at 3/11 and one longitudinal phonon at 3/4 in every direction. The metric TT front remains Einstein-Hilbert and the single new modulus length respects the five-constant cap.",
        "scope": "This selects a new elastic-spacetime architecture. It does not yet prove nonlinear rank, the full metric constraint algebra, a useful weak gravitational response, Solar viability, or observational adequacy.",
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
