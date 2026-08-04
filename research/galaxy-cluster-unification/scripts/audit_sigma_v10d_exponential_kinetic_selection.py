from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v10d_exponential_kinetic import (
    audit_v10d_exponential_kinetic,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the Sigma v10D exponential kinetic selection."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v10d_exponential_kinetic_selection.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v10d_exponential_kinetic_selection",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fixed = config["fixed_values"]
    parameters = config["physical_parameters"]
    audit = audit_v10d_exponential_kinetic(
        k_b=float(fixed["K_B"]),
        u=float(fixed["u"]),
        carrier_speed_squared=float(fixed["carrier_speed_squared"]),
        normalized_mixing_squared=float(fixed["normalized_mixing_squared"]),
        physical_parameter_count=int(parameters["count"]),
        maximum_physical_parameters=int(parameters["maximum_allowed"]),
    )
    report = {
        "status": "completed Sigma v10D exponential kinetic selection",
        "candidate": config["candidate"],
        **audit,
        "decision": "advance_v10d_to_full_nonlinear_ADM_and_characteristic_gate",
        "reason": "The fixed function exp(x)-x has global minimum one, so the completed physical aether kinetic matrix is at least K_B for every real carrier eigenvalue. Zero-background response and cones are unchanged, and amplitude scans keep the local static and mixed hyperbolic blocks positive and causal without a new constant.",
        "scope": "This is a narrow necessary selection pass. Tilted metric/aether/carrier velocity mixing, nonzero-J characteristics, the complete constraint chain, PPN/Solar limits and numerical well-posedness remain mandatory before data.",
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
