from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    paths = {
        "protocol": ROOT / "configs" / "void_cage_test_protocol.json",
        "galaxy_test": ROOT / "results" / "void_cage_test" / "report.json",
        "observable_audit": ROOT / "results" / "r0_observable_audit" / "report.json",
        "same_system_audit": ROOT / "results" / "r1_same_system_pilot_gap" / "report.json",
        "public_data_ceiling": ROOT / "results" / "r1_ten_system_public_data_ceiling" / "report.json",
    }
    documents = {name: read(path) for name, path in paths.items()}
    galaxy = documents["galaxy_test"]
    observables = documents["observable_audit"]
    same_system = documents["same_system_audit"]
    ceiling = documents["public_data_ceiling"]

    galaxy_gate_pass = bool(galaxy["primary_screened_cage_pass"])
    theory_neutral_clash_ready = (
        int(observables["clash"]["raw_or_likelihood_level_systems_ingested"]) > 0
        and int(observables["clash"]["alternative_metric_forward_model_ready_systems"]) > 0
    )
    same_system_population_ready = (
        int(same_system["strict_r1_ready_systems"])
        >= int(same_system["target_strict_systems"])
    )
    lensing_replay_authorized = (
        galaxy_gate_pass and theory_neutral_clash_ready and same_system_population_ready
    )

    report = {
        "status": "completed void-cage lensing and same-system authorization gate",
        "report_version": "void-cage-lensing-gate-0.1",
        "inputs": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
            }
            for name, path in paths.items()
        },
        "checks": {
            "galaxy_primary_gate_pass": galaxy_gate_pass,
            "theory_neutral_CLASH_likelihood_ready": theory_neutral_clash_ready,
            "ten_same_system_population_ready": same_system_population_ready,
            "hard_public_data_ceiling_established": bool(
                ceiling["hard_public_data_shortfall_established"]
            ),
        },
        "available_summary_data": {
            "CLASH_systems": int(observables["clash"]["systems"]),
            "CLASH_summary_points": int(observables["clash"]["scored_summary_points"]),
            "CLASH_raw_or_likelihood_systems": int(
                observables["clash"]["raw_or_likelihood_level_systems_ingested"]
            ),
            "CLASH_alternative_metric_forward_ready": int(
                observables["clash"]["alternative_metric_forward_model_ready_systems"]
            ),
            "BCG_systems": int(observables["bcg"]["frozen_systems"]),
            "BCG_resolved_dynamics_likelihood_systems": int(
                observables["bcg"]["resolved_dynamics_likelihood_systems_ingested"]
            ),
            "same_system_candidates": int(same_system["candidate_systems_evaluated"]),
            "same_system_structural_passes": int(
                same_system["systems_passing_three_plus_three_structural_overlap"]
            ),
            "same_system_strict_ready": int(same_system["strict_r1_ready_systems"]),
            "same_system_target": int(same_system["target_strict_systems"]),
        },
        "lensing_replay_authorized": lensing_replay_authorized,
        "action_taken": (
            "fit no lensing response and introduce no lensing-only amplitude"
            if not lensing_replay_authorized
            else "forward-predict raw lensing observables with galaxy-frozen constants"
        ),
        "decision": {
            "CLASH_summary_replay": "not_scientifically_authorized",
            "reason": [
                "The preregistered galaxy cage failed before lensing.",
                "The 84 CLASH summary accelerations are NFW-deprojected GR reconstructions, not raw alternative-metric likelihoods.",
                "There are zero strict-ready same systems, so Phi and Phi+Psi are not population-identifiable from the current public package.",
            ],
            "what_a_future_pass_would_test": (
                "Constants frozen on dynamics must predict raw shear or image coordinates on the same systems "
                "with Phi=Psi and no lensing normalization."
            ),
        },
    }
    output = ROOT / "results" / "void_cage_lensing_gate" / "report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
