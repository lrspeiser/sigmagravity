from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v7_positive_carrier import audit_linear_positive_carrier


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the Sigma v7A positive local carrier.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v7a_positive_carrier_gate.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v7a_positive_carrier_gate",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    scan = config["scan"]
    gates = config["gates"]
    radius_over_range = np.geomspace(
        float(scan["radius_over_range_min"]),
        float(scan["radius_over_range_max"]),
        int(scan["points"]),
    )
    audit = audit_linear_positive_carrier(
        ppn_bound=float(gates["maximum_ppn_gamma_minus_one"]),
        high_field_force_bound=float(
            gates["maximum_high_field_fractional_extra_force"]
        ),
        required_lensing_enhancement=float(
            gates["minimum_useful_large_scale_lensing_enhancement"]
        ),
        radius_over_range=radius_over_range,
    )
    spectrum = audit["spectrum"]
    construction_gates = dict(audit["gates"])
    construction_gates["parameter_count"] = (
        int(config["physical_parameters"]["count"])
        <= int(config["physical_parameters"]["maximum_allowed"])
    )
    construction_gates["declared_kinetic_signature"] = (
        int(spectrum["negative_kinetic_directions"])
        <= int(gates["maximum_negative_kinetic_directions"])
    )
    construction_gates["declared_spin2_constraint_count"] = (
        int(spectrum["total_degrees_of_freedom"])
        == int(gates["required_total_spin2_degrees_of_freedom"])
    )
    construction_gates = {
        name: bool(value) for name, value in construction_gates.items()
    }
    report = {
        "status": "completed Sigma v7A positive-local-carrier gate",
        "observational_data_accessed": False,
        "raw_holdout_opened": False,
        "candidate": config["candidate"],
        "quadratic_action": config["quadratic_action"],
        "nonlinear_completion_boundary": config["nonlinear_completion_boundary"],
        "physical_parameter_count": int(config["physical_parameters"]["count"]),
        "spectrum": {
            "massless_spin2_degrees_of_freedom": int(
                spectrum["massless_spin2_degrees_of_freedom"]
            ),
            "massive_spin2_degrees_of_freedom": int(
                spectrum["massive_spin2_degrees_of_freedom"]
            ),
            "total_degrees_of_freedom": int(spectrum["total_degrees_of_freedom"]),
            "kinetic_eigenvalues": [
                float(value) for value in spectrum["kinetic_eigenvalues"]
            ],
            "negative_kinetic_directions": int(
                spectrum["negative_kinetic_directions"]
            ),
        },
        "analytic_response": {
            "dynamical_force": "1+(4/3) alpha^2 (1+r/L) exp(-r/L)",
            "spatial_force": "1+(2/3) alpha^2 (1+r/L) exp(-r/L)",
            "lensing_force": "1+alpha^2 (1+r/L) exp(-r/L)",
            "ppn_gamma": "[1+(2/3)alpha^2]/[1+(4/3)alpha^2]",
            "kernel_derivative": "d[(1+x)exp(-x)]/dx=-x exp(-x)<=0",
        },
        "bounds": {
            "maximum_residue_from_ppn": float(audit["maximum_residue_from_ppn"]),
            "maximum_residue_from_high_field_force": float(
                audit["maximum_residue_from_high_field_force"]
            ),
            "maximum_jointly_allowed_residue": float(
                audit["maximum_jointly_allowed_residue"]
            ),
            "maximum_lensing_enhancement": float(
                audit["maximum_lensing_enhancement"]
            ),
            "minimum_locally_calibrated_far_force_ratio": float(
                audit["minimum_locally_calibrated_far_force_ratio"]
            ),
        },
        "gates": construction_gates,
        "all_v7a_gates_pass": bool(all(construction_gates.values())),
        "decision": "retire_unscreened_v7A_retain_positive_spin2_as_screened_carrier_option",
        "reason": "The Fierz-Pauli carrier is locally healthy, but an unscreened long-range mode is limited by the high-field gate to alpha^2<=7.5e-6, so its maximum lensing enhancement is below 1.000008. Its positive Yukawa kernel also decreases with distance and cannot turn on outside the Solar-calibrated regime.",
        "scope": "This rejects the unscreened linear carrier, not a nonlinear ghost-free Vainshtein-screened Hassan-Rosen/dRGT completion.",
        "next_mechanism_requirement": "Test whether a ghost-free nonlinear screening radius can suppress the same positive spin-2 mode locally while activating a useful, universal dynamics-and-lensing window across dwarfs, disks, and clusters.",
        "data_policy": config["data_policy"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
