#!/usr/bin/env python3
"""Freeze and audit the post-gas source-invariant/action preselection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "configs" / "sigma_v19bj_source_invariant_action_preselection.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_parent_hashes(
    config: dict[str, Any],
) -> tuple[dict[str, str], dict[str, Path]]:
    hashes: dict[str, str] = {}
    paths: dict[str, Path] = {}
    for name, spec in config["parents"].items():
        path = ROOT / spec["path"]
        actual = sha256(path)
        if actual != spec["sha256"]:
            raise ValueError(
                f"parent hash mismatch for {name}: {actual} != {spec['sha256']}"
            )
        hashes[name] = actual
        paths[name] = path
    return hashes, paths


def build_report(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    implementation = config["implementation"]
    runner_path = (ROOT / implementation["runner"]).resolve()
    if runner_path != Path(__file__).resolve():
        raise ValueError("frozen implementation path does not identify this runner")
    runner_hash = sha256(runner_path)
    if runner_hash != implementation["runner_sha256"]:
        raise ValueError("frozen implementation hash mismatch")

    parent_hashes, parent_paths = verify_parent_hashes(config)
    v19be = json.loads(
        parent_paths["long_wave_action_admission_report"].read_text(encoding="utf-8")
    )
    v19bd = json.loads(
        parent_paths["collisionless_source_uncertainty"].read_text(encoding="utf-8")
    )
    v19bg = json.loads(
        parent_paths["broad_phenomenology_contract"].read_text(encoding="utf-8")
    )

    library = config["invariant_library"]
    eligible = [row for row in library if row["eligible_as_new_source_state"]]
    directional = [
        row
        for row in eligible
        if "vector" in row["tensor_character"]
        or "tensor" in row["tensor_character"]
    ]
    controls = [row for row in library if not row["eligible_as_new_source_state"]]
    thresholds = config["identifiability_gates"]
    authorization = config["authorization"]
    placement_ids = {row["id"] for row in config["action_placement_classes"]}

    gate_results = {
        "all_parent_hashes_exact": True,
        "parent_action_gate_passed_without_action_selection": (
            v19be["decision"] == "passed_action_admission_requirements"
            and not v19be["theory_state"]["covariant_action_selected"]
            and not v19be["theory_state"]["universal_constants_selected"]
        ),
        "collisionless_parent_is_source_only": (
            v19bd["decision"] == "passed"
            and not v19bd["lensing_halo_gas_response_or_gravity_payload_opened"]
            and not v19bd["long_wave_operator_or_parameter_selected"]
        ),
        "broad_contract_retains_galaxy_cluster_first_priority": (
            v19bg["decision"] == "passed_broad_phenomenology_contract"
            and v19bg["gate_results"][
                "solar_gate_retained_but_not_optimized_first"
            ]
        ),
        "covariant_baryon_frame_declared": (
            "J_b^mu" in config["covariant_source_frame"]["definition"]
            and not config["covariant_source_frame"]["object_labels_used_in_equations"]
        ),
        "density_only_is_an_ineligible_control": any(
            row["id"] == "D0_DENSITY_ONLY_CONTROL"
            and not row["eligible_as_new_source_state"]
            for row in controls
        ),
        "at_least_four_eligible_source_features": len(eligible) >= 4,
        "eligible_directional_features_exist": len(directional) >= 2,
        "scalar_activation_and_direction_both_required": (
            thresholds["minimum_eligible_scalar_activation_features"] >= 1
            and thresholds[
                "minimum_eligible_directional_vector_or_tensor_features"
            ]
            >= 1
        ),
        "same_definition_must_pass_both_clusters": (
            thresholds["same_dimensionless_definition_and_thresholds_for_both_clusters"]
            and thresholds["minimum_clusters_passing_each_advanced_feature"] == 2
        ),
        "projection_resolution_and_density_novelty_are_quantified": (
            thresholds["minimum_projection_draw_pass_fraction"] >= 0.90
            and thresholds["maximum_resolution_change_in_normalized_amplitude_fraction"]
            <= 0.10
            and thresholds[
                "minimum_cross_validated_variance_not_predicted_by_total_density_fraction"
            ]
            >= 0.20
        ),
        "lensing_and_halo_cannot_select_the_source": (
            not thresholds["lensing_or_halo_agreement_used_for_selection"]
            and not authorization["read_lensing_or_halo_payload"]
        ),
        "three_materially_distinct_placements_and_stop_rule": (
            placement_ids
            == {
                "P1_CONSTRAINED_COMPOSITE_RESPONSE",
                "P2_CAUSAL_DYNAMIC_RESPONSE",
                "P3_DEGENERATE_PURE_METRIC_NONLINEAR_VERTEX",
            }
            and config["selection_order_and_stop_rule"][
                "maximum_materially_distinct_action_derivations_before_mechanism_reconsideration"
            ]
            == 3
        ),
        "execution_remains_pending": (
            config["execution_prerequisites"]["current_state"].startswith("pending")
            and not authorization["read_live_or_terminal_v19w_products"]
            and not authorization["read_v19x_or_regional_temperature_results"]
            and not authorization["compute_source_invariant_scores_now"]
        ),
        "nothing_physical_selected_or_changed": not any(
            authorization[key]
            for key in (
                "select_source_invariant_now",
                "select_action_placement_now",
                "select_field_content_operator_or_constant_now",
                "change_gravity_formula",
                "open_holdout",
            )
        ),
    }
    gate_results = {key: bool(value) for key, value in gate_results.items()}
    decision = (
        "passed_target_blind_source_preselection_freeze"
        if all(gate_results.values())
        else "failed_closed"
    )

    return {
        "protocol_version": config["protocol_version"],
        "decision": decision,
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "implementation": {
            "runner": implementation["runner"],
            "runner_sha256": runner_hash,
        },
        "input_hashes": parent_hashes,
        "source_library_summary": {
            "total_registered": len(library),
            "eligible_after_gas": [row["id"] for row in eligible],
            "directional_after_gas": [row["id"] for row in directional],
            "ineligible_controls_or_future_only": [row["id"] for row in controls],
        },
        "identifiability_gates": thresholds,
        "action_placement_classes": config["action_placement_classes"],
        "execution_prerequisites": config["execution_prerequisites"],
        "gate_results": gate_results,
        "theory_state": {
            "source_invariant_selected": False,
            "action_placement_selected": False,
            "covariant_action_written": False,
            "field_content_selected": False,
            "universal_constants_selected": False,
            "lensing_or_halo_target_opened": False,
            "gravity_formula_changed": False,
            "scientific_source_test_executed": False,
        },
        "next_decision": {
            "when_authorized": (
                "After V19W4, V19X2, all 494 regional fits, and the common-grid gas "
                "posterior pass, score only the registered source features on Bullet and "
                "Abell 2146 without opening lensing."
            ),
            "advance": (
                "Require at least one scalar activation and one direction to pass every "
                "frozen gate in both clusters, then derive the least-field-content healthy "
                "compatible action placement."
            ),
            "stop": (
                "If no registered feature passes, obtain direct gas velocity or another "
                "independent merger sample; do not tune an action. If three materially "
                "different action derivations fail the same gate, reconsider the mechanism."
            ),
        },
        "dark_matter_phenomenology_implications": config[
            "dark_matter_phenomenology_implications"
        ],
        "claim_boundary": config["claim_boundary"],
    }


def run(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    report = build_report(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / config["outputs"]["report"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if report["decision"] == "failed_closed":
        raise RuntimeError(f"V19BJ failed closed: {report['gate_results']}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    report = run(args.config)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "source_library_summary": report["source_library_summary"],
                "theory_state": report["theory_state"],
                "next_decision": report["next_decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
