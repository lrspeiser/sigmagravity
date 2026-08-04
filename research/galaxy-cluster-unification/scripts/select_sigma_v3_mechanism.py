from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Sigma v3 mechanism pre-action audit.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v3_mechanism_selection.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v3_mechanism_selection",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    reports = {
        relative: json.loads((ROOT / relative).read_text(encoding="utf-8"))
        for relative in config["input_reports"]
    }

    requirements = config["hard_requirements"]
    rows: list[dict[str, object]] = []
    for mechanism in config["mechanisms"]:
        checks: dict[str, bool] = {}
        for requirement in requirements:
            if requirement == "at_most_five_universal_constants":
                checks[requirement] = int(mechanism["universal_constants"]) <= 5
            else:
                checks[requirement] = bool(mechanism[requirement])
        failed = [name for name, passed in checks.items() if not passed]
        rows.append(
            {
                "mechanism": mechanism["id"],
                **checks,
                "hard_gate_pass": len(failed) == 0,
                "failed_requirement_count": len(failed),
                "failed_requirements": ";".join(failed),
                "universal_constants": int(mechanism["universal_constants"]),
                "prior_art_boundary": mechanism["prior_art_boundary"],
                "disposition": mechanism["disposition"],
            }
        )
    frame = pd.DataFrame.from_records(rows)

    # A common localization of U Box^{-1} S introduces a cross kinetic term
    # grad(U).grad(S).  In the (U,S) basis its normalized kinetic matrix has
    # eigenvalues -1 and +1, so the naive two-field localization is not a
    # positive-energy fundamental theory.
    localized_pair_kinetic = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    localized_pair_eigenvalues = np.linalg.eigvalsh(localized_pair_kinetic)

    constants = config["diagnostic_constants"]
    velocity_suppression = (
        float(constants["generous_cluster_baryon_speed_km_s"])
        / float(constants["speed_of_light_km_s"])
    )
    minimum_linear_coupling = (
        float(constants["minimum_order_unity_response"]) / velocity_suppression
    )
    minimum_quadratic_coupling = (
        float(constants["minimum_order_unity_response"]) / velocity_suppression**2
    )

    v1 = reports["results/sigma_v1_nonmetricity_cycle/report.json"]
    v2 = reports["results/sigma_v2_trace_nonmetricity_cycle/report.json"]
    action_failures = [
        {
            "model": v1["model_id"],
            "weak_reduction": v1["weak_field_derivation"]["reduced_theory"],
            "raw_cluster_root_fraction": v1["spent_observation_mapping"][
                "raw_cluster_minimum_root_convergence_fraction"
            ],
            "raw_cluster_pass": v1["gate_results"]["raw_cluster_lensing"],
        },
        {
            "model": v2["model_id"],
            "weak_reduction": v2["weak_field_derivation"]["known_reduction"],
            "raw_cluster_root_fraction": v2["spent_observation_mapping"][
                "raw_cluster_minimum_root_convergence_fraction"
            ],
            "raw_cluster_pass": v2["gate_results"]["raw_cluster_lensing"],
        },
    ]
    completed_same_gate_failures = sum(not row["raw_cluster_pass"] for row in action_failures)

    preferred = frame[frame.mechanism == "degenerate_baryon_forced_tidal_geometry"].iloc[0]
    fallback = frame[
        frame.mechanism == "causal_pure_metric_nonlocal_tidal_response"
    ].iloc[0]
    report = {
        "status": "completed Sigma v3 pre-action mechanism selection",
        "input_hashes": {
            "config": sha256(args.config),
            **{relative: sha256(ROOT / relative) for relative in config["input_reports"]},
        },
        "hard_requirements": requirements,
        "ready_to_freeze_mechanisms": frame[frame.hard_gate_pass].mechanism.tolist(),
        "selection": {
            "preferred_derivation_target": preferred.mechanism,
            "preferred_unresolved_requirements": preferred.failed_requirements.split(";"),
            "fallback_derivation_target": fallback.mechanism,
            "fallback_unresolved_requirements": fallback.failed_requirements.split(";"),
            "action_frozen": False,
            "reason": "no candidate yet proves every source, orientation, variational, health, wave, and Solar gate",
        },
        "naive_localization_audit": {
            "cross_kinetic_matrix": localized_pair_kinetic.tolist(),
            "eigenvalues": localized_pair_eigenvalues.tolist(),
            "positive_definite": bool(np.all(localized_pair_eigenvalues > 0.0)),
        },
        "baryon_current_scaling_audit": {
            "generous_cluster_speed_over_c": velocity_suppression,
            "minimum_linear_coupling_for_order_unity_response": minimum_linear_coupling,
            "minimum_quadratic_coupling_for_order_unity_response": minimum_quadratic_coupling,
            "interpretation": "a current-only vector is a low-priority null control unless a covariant nonlinear mechanism removes velocity suppression while passing preferred-frame bounds",
        },
        "completed_action_level_raw_topology_failures": action_failures,
        "completed_same_gate_failure_count": completed_same_gate_failures,
        "remaining_failures_before_mandatory_rethink": max(0, 3 - completed_same_gate_failures),
        "decision": (
            "do not freeze a Sigma v3 equation yet; derive a degenerate baryon-forced tidal action first, with the causal pure-metric nonlocal tidal route as fallback"
        ),
        "selection_rule": config["selection_rule"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "mechanism_matrix.csv", index=False)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["selection"], indent=2, sort_keys=True))
    print(
        f"same raw-topology gate failures: {completed_same_gate_failures}; "
        f"remaining before mandatory rethink: {report['remaining_failures_before_mandatory_rethink']}"
    )


if __name__ == "__main__":
    main()
