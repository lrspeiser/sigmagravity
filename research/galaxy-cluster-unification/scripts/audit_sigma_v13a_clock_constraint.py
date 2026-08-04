from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v13a_clock_constraint import (
    ClockConstraintParameters,
    clock_constraint_no_go_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the exact Sigma v13A clock-current constraint."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v13a_clock_constraint.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v13a_clock_constraint",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fixed = config["fixed_values"]
    tolerance = float(fixed["verification_tolerance"])
    audit = clock_constraint_no_go_audit(
        scale_factors=tuple(float(value) for value in fixed["scale_factors"]),
        signed_charges=tuple(
            float(value) for value in fixed["signed_comoving_charges"]
        ),
        positive_source_uniqueness_charges=tuple(
            float(value) for value in fixed["positive_source_uniqueness_charges"]
        ),
        auxiliary_curvatures=tuple(
            float(value) for value in fixed["auxiliary_curvatures"]
        ),
        delta_q=float(fixed["regularization_delta_q"]),
        parameters=ClockConstraintParameters(
            q0=float(fixed["q0"]),
            k2=float(fixed["k2"]),
        ),
    )
    verification_gates = {
        "exact_constraint_current_conserved": float(
            audit["maximum_conserved_current_residual"]
        )
        <= tolerance,
        "exact_constraint_redshifts_as_dust": float(
            audit["maximum_dust_redshift_residual"]
        )
        <= tolerance,
        "signed_charge_exposes_negative_hamiltonian": bool(
            audit["hamiltonian_unbounded_for_unrestricted_signed_charge"]
            and int(audit["negative_hamiltonian_row_count"]) > 0
        ),
        "same_baryons_allow_distinct_nonnegative_charge_states": bool(
            audit["source_uniqueness_violated_even_for_nonnegative_charge"]
        ),
        "finite_auxiliary_reduction_identity_verified": float(
            audit["maximum_regularization_identity_residual"]
        )
        <= tolerance,
        "finite_auxiliary_stationarity_verified_independently": float(
            audit["maximum_finite_difference_stationarity_residual"]
        )
        <= tolerance,
        "finite_auxiliary_only_renormalizes_k2": bool(
            audit["finite_regularization_is_only_k2_renormalization"]
        ),
    }
    physical_gates = {
        "bounded_constraint_sector_hamiltonian": False,
        "no_freely_specified_dust_like_state": False,
        "unique_baryon_forced_branch": False,
    }
    report = {
        "status": "Sigma v13A exact clock-current constraint falsification",
        "protocol_status": config["protocol_status"],
        "candidate": config["candidate"],
        "audit": audit,
        "verification_gates": verification_gates,
        "all_verification_gates_pass": bool(all(verification_gates.values())),
        "physical_gates": physical_gates,
        "all_physical_gates_pass": bool(all(physical_gates.values())),
        "v13a_formulation_rejected_before_data": bool(
            all(verification_gates.values()) and not all(physical_gates.values())
        ),
        "post_v12_reset_total_material_formulation_failure_count": 2,
        "bounded_hamiltonian_same_gate_failure_count": 2,
        "source_uniqueness_same_gate_failure_count": 1,
        "three_failure_mechanism_reset_triggered": False,
        "observational_data_accessed": False,
        "raw_holdout_opened": False,
        "theory_viable": False,
        "decision": (
            "Reject exact v13A before observations. The covariant multiplier does "
            "enforce Q=Q0, but the scalar equation leaves a signed conserved charge "
            "I with H=Q0 I and rho proportional to a^-3. Restricting I to be positive "
            "still leaves an invisible dust-like state not fixed by baryons. Giving the "
            "multiplier finite curvature integrates it out and only shifts K2, returning "
            "to the soft AeST clock sector already unable to pass the energy screen."
        ),
        "next_formulation_requirement": (
            "The next materially distinct carrier must have a convex reduced Hamiltonian "
            "without a conserved object-level clock charge. Do not add another linear "
            "clock multiplier or tune K2. Preserve one physical metric, luminal tensors, "
            "a baryon-forced spatial response, and at most five universal constants."
        ),
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
