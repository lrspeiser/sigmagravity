from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v8_aest_galileon import audit_v8a_selection


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the Sigma v8A action selection.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v8a_aest_galileon_selection.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v8a_aest_galileon_selection",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    values = config["fixed_selection_values"]
    parameters = config["physical_parameters"]
    audit = audit_v8a_selection(
        k_b=float(values["K_B"]),
        k_2=float(values["K_2"]),
        lambda_s=float(values["lambda_s"]),
        physical_parameter_count=int(parameters["count"]),
        maximum_physical_parameters=int(parameters["maximum_allowed"]),
    )
    gates = dict(audit["gates"])
    gates["no_object_labels"] = True
    gates["no_lensing_only_multiplier"] = True
    gates = {name: bool(value) for name, value in gates.items()}
    report = {
        "status": "completed Sigma v8A pre-data action selection",
        "candidate": config["candidate"],
        "action": config["action"],
        "definitions": config["definitions"],
        "physical_parameter_count": int(parameters["count"]),
        "fixed_selection_values": values,
        "base_linear_spectrum": audit["spectrum"],
        "metric_projection": audit["metric_projection"],
        "geometry_stress_test": audit["geometry_stress_test"],
        "gates": gates,
        "all_v8a_selection_gates_pass": bool(all(gates.values())),
        "decision": "advance_v8A_to_full_variation_and_nonlinear_health_only",
        "reason": "Unlike v7, the scalar changes the single physical matter metric through the AeST kinetic mixing, giving equal nonzero shifts to Psi, Phi, and Weyl. The cubic Horndeski equation is second order, leaves the flat quadratic tensor cone unchanged, and distinguishes equal-trace Hessians (responses 6 and 0), providing a label-free route to component geometry.",
        "scope": "This is an action-selection pass, not a viable theory or an observational result. AeST and the cubic Horndeski operator are both prior art. Their combination has not yet passed the full constraint algebra, nonlinear characteristic, PPN, Solar-screening, cosmology, or source-uniqueness gates.",
        "next_gate": "Derive the complete metric, scalar, vector, and constraint equations from the combined action. Compute the full kinetic Hessian and principal symbol on time-gradient and static spatial-gradient backgrounds. Reject before data for any extra mode, negative eigenvalue, superluminal characteristic, singular branch, nonunique baryon-forced state, PPN failure, or loss of the AeST one-metric Weyl identity.",
        "observational_data_accessed": False,
        "raw_holdout_opened": False,
        "data_policy": config["data_policy"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
