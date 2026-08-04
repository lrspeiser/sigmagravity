from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v10b_auxiliary_aether_tidal import (
    audit_v10b_constraint_causality,
)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit v10B constraint causality.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v10b_constraint_causality_gate.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v10b_constraint_causality_gate",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fixed = config["fixed_values"]
    scan = config["scan"]
    audit = audit_v10b_constraint_causality(
        k_b=float(fixed["K_B"]),
        inverse_length=float(fixed["inverse_carrier_length_units"]),
        wave_numbers=np.asarray(scan["wave_numbers"], dtype=float),
        radii=np.asarray(scan["radii_in_carrier_length_units"], dtype=float),
    )
    report = {
        "status": "completed Sigma v10B auxiliary-constraint causality falsification",
        "parent_protocol": config["parent_protocol"],
        "fixed_values": fixed,
        "canonical_channel": config["canonical_channel"],
        "equal_time_kernel_derivation": config["equal_time_kernel"],
        **audit,
        "decision": "retire_exact_v10B_finite_range_auxiliary_carrier",
        "reason": "The canonical constraint result is healthy but physical causality is not. Each of the six pi_P=0 primaries pairs with a positive secondary elliptic equation, so P adds no degree of freedom and the reduced Hamiltonian is positive. For every finite L_P and nonzero beta, eliminating P leaves an equal-time Yukawa tail. In the transverse channel at K_B=1 the local coefficient is 3/4, the inverse range is sqrt(3/4)/L_P, and the tail coefficient is 3/(16 L_P^2). The transverse aether vector is physical, not a lapse gauge variable, so a localized source changes its acceleration at all radii on the same preferred-time slice.",
        "scope": "This retires the exact finite-range nondynamical v10B carrier. It does not retire the positive aether-tidal static block or a causal hyperbolic carrier. The massless limit removes the Yukawa tail only by removing the finite transition scale and nonlinear finite-range screening; beta zero removes the mechanism.",
        "next_mechanism_requirement": "Give the aether-tidal tensor a hyperbolic causal completion whose full mixed characteristic cone remains inside the metric cone, while preserving the exact positive static block, one metric, five constants, and no homogeneous object state.",
        "observational_data_accessed": False,
        "new_observational_product_accessed": False,
        "raw_holdout_opened": False,
        "data_policy": config["data_policy"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, cls=NumpyEncoder) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, cls=NumpyEncoder))


if __name__ == "__main__":
    main()
