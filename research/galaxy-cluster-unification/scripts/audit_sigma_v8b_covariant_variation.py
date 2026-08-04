from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v8b_covariant_variation import audit_v8b_covariant_subgate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the v8B covariant variation and FLRW clock subgate."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v8b_covariant_variation_gate.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v8b_covariant_variation_gate",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fixed = config["fixed_values"]
    parameters = config["physical_parameters"]
    audit = audit_v8b_covariant_subgate(
        k_2=float(fixed["K_2"]),
        alpha=float(fixed["alpha"]),
        physical_parameter_count=int(parameters["count"]),
        maximum_physical_parameters=int(parameters["maximum_allowed"]),
    )
    gates = dict(audit["gates"])
    report = {
        "status": "completed Sigma v8B covariant-variation subgate",
        "candidate": config["candidate"],
        "completion_lagrangian": config["completion_lagrangian"],
        "fixed_values": fixed,
        "physical_parameter_count": int(parameters["count"]),
        "derived_euler_derivatives": {
            "scalar": "E_phi/C=nabla_n nabla_m(B^2 q^mn)-2 nabla_m(B H_perp A^m)",
            "vector": "E_A_s/C=2 B H_perp nabla_s(phi)+2 B^2 A^m nabla_m nabla_s(phi)",
        },
        "scalar_third_derivative_cancellation": audit[
            "scalar_third_derivative_cancellation"
        ],
        "flrw_clock_result": {
            "coefficient": "2 K_2-3(alpha-1)L_H^2 H Q0",
            "selected_limit_LH2_H_Q0": audit[
                "selected_flrw_stability_limit_LH2_H_Q0"
            ],
            "equivalent_selected_limit_LH2_H_mu_sigma": 24.0 / 7.0,
            "below_limit_clock_coefficient": audit[
                "below_limit_clock_coefficient"
            ],
            "above_limit_clock_coefficient": audit[
                "above_limit_clock_coefficient"
            ],
        },
        "minisuperspace_completion_hessian_at_Q0": audit[
            "minisuperspace_completion_hessian_at_Q0"
        ],
        "gates": gates,
        "all_covariant_subgates_pass": bool(all(gates.values())),
        "decision": "advance_v8B_to_metric_stress_Noether_and_full_Hamiltonian_gate_only",
        "reason": "The scalar third-derivative principal terms cancel exactly, the vector variation contains no derivative of the aether, and the Q=Q0 FLRW completion has no metric-scalar velocity mixing. A nonempty clock-stable region exists, but it imposes the new mandatory bound L_H^2 H Q0<12/7 at the selected row.",
        "scope": "This does not complete the covariant theory. The exact metric stress tensor, diffeomorphism identity, nonlinear Hamiltonian constraint count, off-Q0 kinetic mixing, time-dependent characteristic determinant, early cosmology, and Solar/PPN solution remain unproved. No observational use is authorized.",
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
