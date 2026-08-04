from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v8b_source_constraint import audit_v8b_source_constraint_gate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the Sigma v8B inherited constraints and source uniqueness."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v8b_source_constraint_gate.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v8b_source_constraint_gate",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    values = config["selected_values"]
    parameters = config["physical_parameters"]
    audit = audit_v8b_source_constraint_gate(
        k_2=float(values["K_2"]),
        alpha=float(values["alpha"]),
        horndeski_length=float(values["L_H_audit_units"]),
        hubble_inverse_length=float(values["H_audit_units"]),
        q_0=float(values["Q0_audit_units"]),
        frozen_cosmological_charge=float(values["I0"]),
        physical_parameter_count=int(parameters["count"]),
        maximum_physical_parameters=int(parameters["maximum_allowed"]),
    )
    report = {
        "status": "completed v8B inherited-constraint and homogeneous-source audit",
        "candidate": config["candidate"],
        "primary_sources": config["primary_sources"],
        "boundary_policy": config["boundary_policy"],
        "physical_parameter_count": int(parameters["count"]),
        **audit,
        "decision": "hold_v8B_before_data_for_combined_Hamiltonian_and_inhomogeneous_uniqueness",
        "reason": (
            "The published AeST base has six nonlinear physical degrees of freedom, but "
            "that count does not cover the added v8B projected-Hessian operator. A "
            "nonzero homogeneous shift charge supplies an arbitrary leading a^-3 density, "
            "so I0 is frozen to zero and may not be used as missing gravity. At I0=0, "
            "Q=Q0 is the only positive-clock homogeneous v8B branch; the other algebraic "
            "root has negative current slope. Full inhomogeneous uniqueness remains unproved."
        ),
        "scope": (
            "This closes only the published-base count reproduction and homogeneous "
            "source subgate. It does not prove the AeST-plus-completion constraint count, "
            "Hamiltonian boundedness, arbitrary-background characteristics, or uniqueness "
            "for a three-dimensional baryonic source."
        ),
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
