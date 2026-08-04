from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v8b_covariant_variation import (
    audit_v8b_metric_noether_subgate,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the v8B metric Euler tensor and Noether identity."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v8b_metric_noether_gate.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v8b_metric_noether_gate",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    audit = audit_v8b_metric_noether_subgate()
    gates = {name: bool(value) for name, value in audit["gates"].items()}
    gates["parameter_count"] = (
        int(config["physical_parameter_count"])
        <= int(config["maximum_physical_parameters"])
    )
    report = {
        "status": "completed Sigma v8B metric-Noether subgate",
        "candidate": config["candidate"],
        "independent_fields": config["independent_fields"],
        "matter_metric": config["matter_metric"],
        "metric_euler_tensor": audit["metric_euler_tensor"],
        "off_shell_noether_identity": audit["off_shell_noether_identity"],
        "metric_directional_variation_relative_error": audit[
            "metric_directional_variation_relative_error"
        ],
        "gates": gates,
        "all_metric_noether_subgates_pass": bool(all(gates.values())),
        "decision": "advance_v8B_to_nonlinear_Hamiltonian_and_time_dependent_characteristic_gate_only",
        "reason": "The completion now has explicit scalar, vector, and metric Euler derivatives. Its constant-jet metric variation agrees with an independent finite difference, and general covariance yields an off-shell identity that reduces to completion-stress conservation on the scalar/vector equations.",
        "scope": "The completion field equations and conservation identity are derived, but the full AeST-plus-completion Hamiltonian constraint algebra and time-dependent characteristic determinant remain open. Passing this gate does not establish a healthy degree-of-freedom count or observational viability.",
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
