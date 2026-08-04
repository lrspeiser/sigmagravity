from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v8b_tilted_adm import audit_tilted_adm_kinetic_gate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the necessary Sigma v8B tilted-ADM kinetic gate."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v8b_tilted_adm_gate.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v8b_tilted_adm_gate",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    envelope = config["declared_local_envelope"]
    audit = audit_tilted_adm_kinetic_gate(
        deterministic_velocities=envelope["aether_velocities"],
        deterministic_q_ratios=envelope["Q_over_Q0"],
        a_sigma_ratios=envelope["a_sigma_over_Q0"],
        maximum_ell_h_q0=float(envelope["maximum_L_H_Q0"]),
        maximum_background_kinematic=float(
            envelope["maximum_absolute_K_and_E_over_Q0"]
        ),
        random_samples=int(envelope["random_samples"]),
        random_seed=int(envelope["random_seed"]),
    )
    report = {
        "status": "completed necessary tilted-ADM kinetic subgate",
        "candidate": config["candidate"],
        "protocol_version": config["protocol_version"],
        "physical_parameters": config["physical_parameters"],
        "selected_values": config["selected_values"],
        **audit,
        "decision": "retain_v8B_for_full_Dirac_and_reachability_audit_only",
        "reason": (
            "The completion's scalar normal acceleration is removable by an exact "
            "boundary subtraction. The resulting local ten-velocity Legendre map is "
            "full rank with unchanged inertia throughout the declared bounded envelope. "
            "A finite rank-changing surface nevertheless exists outside that envelope, "
            "so global Hamiltonian regularity is not established."
        ),
        "scope": (
            "This is a homogeneous local necessary condition. It is neither the full "
            "inhomogeneous Dirac constraint algebra nor a proof that the Hamiltonian is "
            "bounded, that the extra negative direction is removed, or that all coupled "
            "characteristics are causal."
        ),
        "decision_rule": config["decision_rule"],
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
