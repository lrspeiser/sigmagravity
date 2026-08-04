from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v7_metric_projection import audit_v7c_metric_projection


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the v7C physical metric projection.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v7c_metric_projection_gate.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v7c_metric_projection_gate",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    samples = config["audit_samples"]
    frozen = config["frozen_v7c_content"]
    thresholds = config["gates"]
    audit = audit_v7c_metric_projection(
        scalar_samples=np.linspace(
            float(samples["scalar_minimum"]),
            float(samples["scalar_maximum"]),
            int(samples["points"]),
        ),
        conformal_cancellation_tolerance=float(
            thresholds["maximum_absolute_conformal_weyl_response"]
        ),
        minimum_nonzero_null_response=float(
            thresholds["minimum_nonzero_null_response"]
        ),
        disformal_mapping_frozen=bool(frozen["disformal_mapping_frozen"]),
        complete_scalar_equation_frozen=bool(
            frozen["complete_dRGT_scalar_equation_frozen"]
        ),
        coupled_tensor_equation_frozen=bool(
            frozen["coupled_tensor_X3_equation_frozen"]
        ),
    )
    gates = dict(audit["gates"])
    gates["parameter_count"] = (
        int(config["physical_parameters"]["count"])
        <= int(config["physical_parameters"]["maximum_allowed"])
    )
    gates["no_lensing_only_multiplier"] = True
    gates = {name: bool(value) for name, value in gates.items()}
    report = {
        "status": "completed Sigma v7C physical-metric projection gate",
        "candidate": config["candidate"],
        "weak_field_convention": config["weak_field_convention"],
        "field_redefinition": config["decoupling_limit_field_redefinition"],
        "physical_parameter_count": int(config["physical_parameters"]["count"]),
        "observational_data_accessed": False,
        "raw_holdout_opened": False,
        "projection": {
            "leading_conformal_shifts": "delta Psi=-pi/2; delta Phi=+pi/2; delta Weyl=0",
            "maximum_absolute_conformal_weyl_response": float(
                audit["maximum_absolute_conformal_weyl_response"]
            ),
            "massive_spin2_decomposition": audit["massive_spin2_decomposition"],
            "static_disformal_diagnostic": audit["static_disformal_diagnostic"],
            "frozen_content": audit["frozen_content"],
        },
        "gates": gates,
        "all_v7c_physical_projection_gates_pass": bool(all(gates.values())),
        "decision": "retire_scalar_only_v7C_as_lensing_carrier_retain_solver_as_dynamics_control",
        "reason": "The leading helicity-zero metric perturbation is conformal and cancels exactly from the Weyl potential. A static disformal term can affect null rays and a general dRGT branch can source the tensor through X^(3), but neither the disformal physical-metric mapping, the complete dRGT scalar equation, nor the coupled tensor equation is contained in the frozen v7C scalar PDE. Its 7.22% source nonadditivity therefore cannot be counted as a lensing prediction.",
        "scope": "This retires the frozen scalar-only v7C equation as a lensing carrier. It does not claim that every Galileon, disformal scalar-tensor model, or fully coupled ghost-free bimetric solution has zero lensing response.",
        "next_mechanism_requirement": "Do not open spent or held-out maps. Under the three-formulation reset rule, synthesize v7A, v7B, and v7C as a failed positive-spin2 carrier sequence before selecting a physically distinct mechanism. Any later return to bimetric gravity must first freeze the full coupled physical metric and all action-linked interaction coefficients.",
        "data_policy": config["data_policy"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
