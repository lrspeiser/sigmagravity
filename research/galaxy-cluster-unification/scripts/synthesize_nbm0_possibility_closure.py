from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs" / "nbm0_possibility_closure_protocol.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "nbm0_possibility_closure",
    )
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    reports = {}
    report_hashes = {}
    for relative in protocol["input_reports"]:
        path = ROOT / relative
        reports[relative] = json.loads(path.read_text(encoding="utf-8"))
        report_hashes[relative] = sha256(path)

    action = reports["results/nbm0_action_space/report.json"]
    survivors = reports["results/nbm0_survivor_derivation/report.json"]
    constitutive = reports["results/nbm0_constitutive_basin/report.json"]
    cage = reports["results/void_cage_test/report.json"]
    scaling = reports["results/void_cage_galaxy_scaling_test/report.json"]
    transition = reports["results/void_cage_transition_isolation/report.json"]

    rows = [
        {
            "mechanism_class": "healthy additive linear field",
            "decision": "reject",
            "evidence": "Positive Yukawa/spectral response has dln(v)/dln(r)<=-1/2 outside finite baryons; pure conformal response has no added lensing.",
            "premise_needed_to_reopen": "negative spectral weight/ghost or nonlinear screening",
        },
        {
            "mechanism_class": "local nonlinear isotropic flux",
            "decision": "excluded by user direction",
            "evidence": "Flat speed plus v_flat^4 proportional to M uniquely selects the deep AQUAL/MOND response power m=2.",
            "premise_needed_to_reopen": "allow a MOND/AQUAL galaxy limit",
        },
        {
            "mechanism_class": "single-field nonlinear/nonlocal response",
            "decision": "reject within one-state scope",
            "evidence": "One inverse field cannot supply flat radial and square-root mass scaling algebraically; nonlinear charge is nonadditive; memory adds a second state variable.",
            "premise_needed_to_reopen": "allow multiple state variables with cosmologically fixed initial data",
        },
        {
            "mechanism_class": "direct external void force or regular tide",
            "decision": "reject",
            "evidence": "Regular isotropic tide gives g proportional to r; direct CF4 cage failed held-out rotation and permutation gates.",
            "premise_needed_to_reopen": "new independently measured non-smooth boundary observable",
        },
        {
            "mechanism_class": "self-gravitating canonical basin energy",
            "decision": "reject",
            "evidence": "Exterior energy fraction is d^2 times compactness; five baryonic masses require direct-force amplitudes 1e7 (galaxy) or 1e6 (cluster).",
            "premise_needed_to_reopen": "independent condensate energy reservoir (dark component)",
        },
        {
            "mechanism_class": "constitutive/boundary flux refraction",
            "decision": "reject as project-specific void mechanism; retain known prior-art benchmark",
            "evidence": "Structure-only slab inversion is worse than mass-only BTFR control by 52.8%; CF4 void term worsens RMSE and has p=0.564.",
            "premise_needed_to_reopen": "directly measured boundary/permittivity map plus reciprocal covariant action",
        },
        {
            "mechanism_class": "linear logarithmic/fractional response",
            "decision": "reject",
            "evidence": "p=3/2 makes speed flat but linear sourcing predicts v_flat^4 proportional to M_b^2.",
            "premise_needed_to_reopen": "nonlinear mass charge, returning to AQUAL or nonadditivity",
        },
        {
            "mechanism_class": "modified inertia",
            "decision": "reject for unification target",
            "evidence": "Changing massive-particle trajectories alone does not predict the missing cluster Weyl potential/lensing.",
            "premise_needed_to_reopen": "derive a universal relativistic metric completion",
        },
        {
            "mechanism_class": "multi-field or independent-state hybrid",
            "decision": "outside fixed scope",
            "evidence": "Can be made flexible enough to supply galaxy dynamics and cluster lensing, but exceeds the one-response/four-global premise before same-system data identify that complexity.",
            "premise_needed_to_reopen": "same-system evidence for more than one latent response",
        },
    ]
    frame = pd.DataFrame(rows)

    report = {
        "report_version": "NBM0-A4-possibility-closure-0.1",
        "status": "declared weak-field possibility space exhausted",
        "protocol": str(args.protocol.relative_to(ROOT)).replace("\\", "/"),
        "protocol_sha256": sha256(args.protocol),
        "input_report_sha256": report_hashes,
        "closure_claim": protocol["closure_claim"],
        "fixed_premises": protocol["fixed_premises"],
        "mechanism_decisions": rows,
        "quantitative_anchors": {
            "canonical_maximum_speed_log_slope": action["radial_shape_theorems"][
                "canonical_Yukawa_speed_slope_maximum"
            ],
            "positive_spectral_maximum_speed_log_slope": action[
                "radial_shape_theorems"
            ]["positive_spectral_speed_slope_maximum"],
            "synthetic_three_parameter_identifiability_pass": action[
                "synthetic_identifiability"
            ]["pass"],
            "A8_A9_fixed_premise_survivor": survivors["fixed_premise_result"],
            "canonical_galaxy_field_energy_direct_force_requirement": survivors[
                "canonical_field_energy_budget"
            ]["galaxy"]["direct_force_amplitude_for_target_field_energy"],
            "slab_structure_log_velocity_rmse_dex": constitutive[
                "SPARC_boundary_inversion"
            ]["cross_validated_metrics"]["structure_only"]["log10_velocity_rmse"],
            "mass_only_log_velocity_rmse_dex": constitutive[
                "SPARC_boundary_inversion"
            ]["cross_validated_metrics"]["mass_only_BTFR_control"][
                "log10_velocity_rmse"
            ],
            "void_boundary_relative_improvement": constitutive[
                "SPARC_boundary_inversion"
            ]["void_relative_height_rmse_improvement"],
            "void_boundary_permutation_p": constitutive["SPARC_boundary_inversion"][
                "void_permutation_p"
            ],
            "direct_void_primary_decision": cage.get("decision", cage.get("status")),
            "galaxy_scaling_primary_status": scaling.get("status"),
            "transition_isolation_status": transition.get("status"),
        },
        "surviving_under_all_fixed_premises": [],
        "allowed_future_relaxations": protocol[
            "allowed_premise_relaxations_for_future_work"
        ],
        "recommended_next_scientific_action": "Complete theory-neutral same-system dynamics+lensing measurements before selecting a multi-field or dark-medium relaxation.",
        "reopen_rule": protocol["reopen_rule"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "mechanism_decisions.csv", index=False)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
