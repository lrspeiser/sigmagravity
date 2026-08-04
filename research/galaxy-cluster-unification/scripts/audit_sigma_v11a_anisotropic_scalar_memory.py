from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v11a_anisotropic_scalar_memory import (
    audit_v11a_anisotropic_scalar_memory,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the Sigma v11A anisotropic scalar-memory selection."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs"
        / "sigma_v11a_anisotropic_scalar_memory_selection.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v11a_anisotropic_scalar_memory_selection",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fixed = config["fixed_values"]
    audit = audit_v11a_anisotropic_scalar_memory(
        k_b=float(fixed["K_B"]),
        aether_speed_squared=float(fixed["aether_speed_squared_u"]),
        maximum_memory_speed_squared=float(fixed["maximum_memory_speed_squared_s"]),
        normalized_mixing_squared=float(fixed["normalized_mixing_squared_q"]),
        anisotropy_fraction=float(fixed["anisotropy_fraction"]),
        physical_parameter_count=int(fixed["physical_parameter_count"]),
        maximum_physical_parameters=int(fixed["maximum_physical_parameters"]),
        ratio_scan_maximum=float(fixed["ratio_scan_maximum"]),
        ratio_scan_samples=int(fixed["ratio_scan_samples"]),
        angle_scan_samples=int(fixed["angle_scan_samples"]),
    )
    report = {
        "status": "completed Sigma v11A theory-only mechanism selection",
        "candidate": config["candidate"],
        **audit,
        "decision": "advance_v11a_only_to_covariant_variation_and_global_rank_gates",
        "reason": "The bounded scalar transport tensor stays positive, the aether-memory static margin is at least 1/44, all fixed-background mixed roots are positive and no greater than one, the aether-rest TT metric symbol remains Einstein-Hilbert, and the massive retarded scalar has a universal source-selected state. Full nonlinear and weak-metric gates remain unresolved.",
        "scope": "This selects a materially different post-reset field architecture. It is not an observational result and does not establish the nonlinear action.",
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
