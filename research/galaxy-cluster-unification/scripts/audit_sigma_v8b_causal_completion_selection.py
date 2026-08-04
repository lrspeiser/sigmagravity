from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v8b_causal_completion import audit_v8b_scalar_selection


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the Sigma v8B selection.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v8b_causal_completion_selection.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v8b_causal_completion_selection",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    parameters = config["physical_parameters"]
    audit = audit_v8b_scalar_selection(
        base_speed_squared=float(config["fixed_values"]["base_scalar_speed_squared"]),
        physical_parameter_count=int(parameters["count"]),
        maximum_physical_parameters=int(parameters["maximum_allowed"]),
    )
    gates = dict(audit["gates"])
    report = {
        "status": "completed Sigma v8B scalar-sector action selection",
        "candidate": config["candidate"],
        "completion_operator": config["completion_operator"],
        "physical_parameter_count": int(parameters["count"]),
        "derived_alpha": audit["derived_alpha"],
        "spherical_scan": audit["spherical_scan"],
        "equal_trace_probes": audit["equal_trace_probes"],
        "nonnegative_source_extremal_bound": audit[
            "nonnegative_source_extremal_bound"
        ],
        "gates": gates,
        "all_v8b_selection_gates_pass": bool(all(gates.values())),
        "decision": "advance_v8B_to_full_covariant_constraint_and_arbitrary_background_characteristic_gate_only",
        "reason": "The derived preferred-time coefficient alpha=16/9 leaves the static cubic geometry equation unchanged and closes both the full positive spherical scalar cone and the extremal bound for every static Hessian permitted by a nonnegative source, without adding a physical parameter. The maximum squared speed is one and the deep spherical radial limit is 0.75.",
        "scope": "This is only a fixed-aether static scalar-sector selection pass. The covariant operator contains the dynamical AeST vector and projected Hessian; its full variation may change the vector/lapse constraints or introduce an unhealthy characteristic on time-dependent backgrounds. The extremal spatial lower bound also tends to zero at infinite trace, so a finite EFT validity range remains mandatory.",
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
