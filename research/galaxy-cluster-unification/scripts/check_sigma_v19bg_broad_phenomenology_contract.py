from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19bg_broad_phenomenology_contract.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_json(config_path)
    parents: dict[str, dict[str, Any]] = {}
    parent_hashes_exact = True
    for name, spec in config["parents"].items():
        path = ROOT / spec["path"]
        actual = sha256(path)
        parents[name] = {
            "path": spec["path"],
            "expected_sha256": spec["sha256"],
            "actual_sha256": actual,
            "exact": actual == spec["sha256"],
            "payload": load_json(path),
        }
        parent_hashes_exact &= parents[name]["exact"]

    mog = parents["identity_long_wave_control"]["payload"]
    regimes = parents["cross_domain_regime_diagnostics"]["payload"]
    relics = parents["resolved_cluster_evidence_registry"]["payload"]
    mergers = parents["two_merger_directional_source_state"]["payload"]
    scale = parents["long_wave_scale_window"]["payload"]

    identity_closed = (
        mog["inputs"]["sparc_systems"] == 131
        and mog["inputs"]["sparc_points"] == 3034
        and mog["inputs"]["clash_systems"] == 20
        and mog["inputs"]["clash_points"] == 84
        and mog["inputs"]["bcg_systems"] == 34
        and not mog["constant_field_control"]["passes_5_percent"]
        and mog["decision"]["emog_q0_retired"]
        and not mog["decision"]["astrophysical_fit_performed"]
    )

    scale_compatibility_only = (
        scale["decision"] == "passed_dimensionless_scale_window"
        and scale["derived_correlation_length_window_kpc"]["lower"]
        < scale["derived_correlation_length_window_kpc"]["upper"]
        and not scale["theory_state"]["covariant_action_selected"]
        and not scale["theory_state"]["universal_constants_selected"]
        and not scale["theory_state"]["gas_source_state_available"]
    )

    core = config["current_core_evidence"]
    core_ids = {row["id"] for row in core}
    required_core_ids = {
        "G1_SPARC_FULL",
        "G2_SPARC_STRATIFIED",
        "G3_BCG_PRESSURE_SUPPORTED",
        "C1_CLASH_RADIAL",
        "C2_RAW_LENSING_SPENT",
        "C3_RELICS_RESOLVED_BARYONS",
        "C4_MERGER_SOURCE_STATE",
    }
    coverage_exact = (
        regimes["coverage"]["SPARC_galaxies"] == 131
        and regimes["coverage"]["galaxy_regime_rows"] == 32
        and regimes["coverage"]["galaxy_continuous_correlation_tests"] == 22
        and regimes["coverage"]["fixed_geometry_phase_systems"] == 5
        and relics["sample"]["systemCount"] == 4
        and relics["sample"]["registeredBaryonMaps"] == 4
        and relics["sample"]["rawForwardScoreReadySystems"] == 0
        and set(mergers["cluster_summaries"]) == {"ABELL2146", "BULLET"}
        and mergers["decision"] == "passed"
        and not mergers["lensing_halo_gas_response_or_gravity_payload_opened"]
        and core_ids == required_core_ids
    )

    required_galaxy_strata = {
        "low_baryonic_mass_dwarf",
        "high_baryonic_mass_giant",
        "gas_rich",
        "gas_poor",
        "low_surface_brightness",
        "high_surface_brightness",
        "bulgeless_disk",
        "bulge_dominated_or_pressure_supported",
    }
    required_cluster_strata = {
        "relaxed_cool_core",
        "relaxed_non_cool_core",
        "plane_of_sky_merger",
        "line_of_sight_or_projection_challenging_merger",
        "lower_mass_strong_lens",
        "high_mass_strong_lens",
    }
    galaxy_diversity = set(config["required_galaxy_strata"]) == required_galaxy_strata
    cluster_diversity = set(config["required_cluster_strata"]) == required_cluster_strata

    blind = config["future_blind_core_gate"]
    blind_cluster_stratified = (
        blind["minimum_new_clusters"] >= 6
        and blind["minimum_relaxed_clusters"] >= 2
        and blind["minimum_disturbed_or_merging_clusters"] >= 2
        and blind["minimum_secure_families_per_cluster"] >= 3
        and blind["minimum_spectroscopic_families_per_cluster"] >= 1
        and blind["minimum_images_per_cluster"] >= 8
        and blind["per_image_position_uncertainties_required"]
        and blind["complete_stars_gas_BCG_ICL_and_member_baryon_model_required"]
        and blind["equation_and_constants_frozen_before_opening"]
    )

    phenomena = config["other_dark_matter_attributed_phenomena"]
    phenomenon_ids = {row["id"] for row in phenomena}
    required_phenomena = {
        "P1_GALAXY_GALAXY_AND_CLUSTER_WEAK_LENSING",
        "P2_COLLIDING_CLUSTER_OFFSETS",
        "P3_DWARF_SPHEROIDAL_AND_SATELLITE_DYNAMICS",
        "P4_STELLAR_STREAMS_AND_SUBSTRUCTURE_LENSING",
        "P5_DYNAMICAL_FRICTION_AND_MERGER_TIMES",
        "P6_COSMIC_STRUCTURE_GROWTH",
        "P7_CMB_PRIMARY_AND_LENSING",
    }
    forward_contracts_complete = (
        phenomenon_ids == required_phenomena
        and all(
            all(
                row.get(field)
                for field in (
                    "prediction_required",
                    "long_wave_signature",
                    "minimum_data",
                    "theory_prerequisite",
                    "current_state",
                )
            )
            for row in phenomena
        )
    )

    roles_safe = (
        relics["roleContract"]["baryonic_input"]
        != relics["roleContract"]["model_derived_discovery_target"]
        and relics["roleContract"]["model_derived_discovery_target"]
        != relics["roleContract"]["raw_observation"]
        and relics["sample"]["prospectiveHoldoutSystems"] == 0
        and not config["authorization"]["use_halo_maps_as_validation_truth"]
    )

    priority_safe = (
        not config["priority"]["detailed_solar_parameter_optimization_authorized_now"]
        and config["priority"]["solar_exclusion_gate_retained"]
        and not config["authorization"]["perform_detailed_solar_optimization"]
    )

    boundary = config["mechanism_boundary"]
    authorization = config["authorization"]
    no_selection_or_payload = (
        boundary["one_physical_metric_for_matter_and_light"]
        and boundary["maximum_universal_physical_constants"] <= 5
        and boundary["per_object_gravity_parameters"] == 0
        and boundary["object_type_switches"] == 0
        and boundary["lensing_only_multipliers"] == 0
        and not boundary["action_selected_here"]
        and not boundary["constant_selected_here"]
        and not authorization["open_new_holdout"]
        and not authorization["fit_or_select_action"]
        and not authorization["fit_or_select_universal_constant"]
        and not authorization["change_gravity_formula"]
    )

    gates = {
        "all_parent_hashes_exact": parent_hashes_exact,
        "identity_long_wave_control_confirmed_closed": identity_closed,
        "scale_window_confirmed_as_compatibility_only": scale_compatibility_only,
        "galaxy_and_cluster_diversity_registered": coverage_exact
        and galaxy_diversity
        and cluster_diversity,
        "future_blind_cluster_sample_is_state_stratified": blind_cluster_stratified,
        "other_dark_matter_phenomena_have_forward_prediction_contracts": forward_contracts_complete,
        "data_roles_do_not_confuse_inferred_halo_maps_with_raw_observations": roles_safe,
        "solar_gate_retained_but_not_optimized_first": priority_safe,
        "no_action_constant_holdout_or_formula_change": no_selection_or_payload,
    }
    declared_gates = config["required_gates"]
    if set(gates) != set(declared_gates):
        raise ValueError("implemented and declared gate names differ")
    if not all(declared_gates.values()):
        raise ValueError("every declared gate must be mandatory")

    parent_audit = {
        name: {
            key: value
            for key, value in audit.items()
            if key != "payload"
        }
        for name, audit in parents.items()
    }
    report = {
        "protocol_version": config["protocol_version"],
        "status": "completed_broad_phenomenology_contract",
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "parent_audit": parent_audit,
        "mechanism_decision": {
            "linear_isotropic_identity_long_wave_control": "closed",
            "active_class": "nonlinear_baryon_forced_source_state_sensitive_one_metric_long_wave_response",
            "action_selected": False,
            "constant_selected": False,
        },
        "current_coverage": {
            "SPARC_full_systems": mog["inputs"]["sparc_systems"],
            "SPARC_full_points": mog["inputs"]["sparc_points"],
            "SPARC_morphology_regime_rows": regimes["coverage"]["galaxy_regime_rows"],
            "SPARC_continuous_correlations": regimes["coverage"]["galaxy_continuous_correlation_tests"],
            "BCG_systems": mog["inputs"]["bcg_systems"],
            "CLASH_model_derived_systems": mog["inputs"]["clash_systems"],
            "CLASH_model_derived_points": mog["inputs"]["clash_points"],
            "spent_raw_lensing_systems": regimes["coverage"]["fixed_geometry_phase_systems"],
            "RELICS_resolved_baryon_maps": relics["sample"]["registeredBaryonMaps"],
            "RELICS_raw_score_ready_systems": relics["sample"]["rawForwardScoreReadySystems"],
            "merging_cluster_directional_source_systems": len(mergers["cluster_summaries"]),
        },
        "long_wave_scale_state": {
            "correlation_length_compatibility_window_kpc": [
                scale["derived_correlation_length_window_kpc"]["lower"],
                scale["derived_correlation_length_window_kpc"]["upper"],
            ],
            "measured_interval": False,
            "selected_constant": False,
        },
        "future_blind_core_gate": blind,
        "registered_galaxy_strata": config["required_galaxy_strata"],
        "registered_cluster_strata": config["required_cluster_strata"],
        "other_phenomena": [
            {
                "id": row["id"],
                "current_state": row["current_state"],
                "theory_prerequisite": row["theory_prerequisite"],
            }
            for row in phenomena
        ],
        "gate_results": gates,
        "decision": (
            "passed_broad_phenomenology_contract"
            if all(gates.values())
            else "failed_broad_phenomenology_contract"
        ),
        "authorization_audit": authorization,
        "claim_boundary": config["claim_boundary"],
        "implementation": config["implementation"],
    }
    return report


def main() -> None:
    report = build_report()
    output = ROOT / load_json(DEFAULT_CONFIG)["outputs"]["report"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": report["decision"],
        "output": output.relative_to(ROOT).as_posix(),
        "gate_results": report["gate_results"],
    }, indent=2, sort_keys=True))
    if report["decision"] != "passed_broad_phenomenology_contract":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
