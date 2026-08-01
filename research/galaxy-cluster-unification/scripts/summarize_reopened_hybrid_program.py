#!/usr/bin/env python3
"""Consolidate the reopened hybrid sensitivity stages into one evidence table."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_reopened_hybrid_sensitivity import expand_variants, json_safe  # noqa: E402


STAGES = [
    (
        "initial",
        "configs/reopened_hybrid_sensitivity_protocol.json",
        "results/reopened_hybrid_sensitivity/report.json",
        "results/reopened_hybrid_sensitivity/scores.csv",
    ),
    (
        "solar_safe_followup",
        "configs/reopened_hybrid_followup_protocol.json",
        "results/reopened_hybrid_followup/report.json",
        "results/reopened_hybrid_followup/scores.csv",
    ),
    (
        "screened_structure",
        "configs/reopened_hybrid_screened_structure_protocol.json",
        "results/reopened_hybrid_screened_structure/report.json",
        "results/reopened_hybrid_screened_structure/scores.csv",
    ),
    (
        "threshold_saturation_cross",
        "configs/reopened_hybrid_threshold_saturation_protocol.json",
        "results/reopened_hybrid_threshold_saturation/report.json",
        "results/reopened_hybrid_threshold_saturation/scores.csv",
    ),
    (
        "channel_saturation",
        "configs/reopened_hybrid_channel_saturation_protocol.json",
        "results/reopened_hybrid_channel_saturation/report.json",
        "results/reopened_hybrid_channel_saturation/scores.csv",
    ),
    (
        "channel_saturation_fine",
        "configs/reopened_hybrid_channel_saturation_fine_protocol.json",
        "results/reopened_hybrid_channel_saturation_fine/report.json",
        "results/reopened_hybrid_channel_saturation_fine/scores.csv",
    ),
    (
        "geometry_gate",
        "configs/reopened_hybrid_geometry_gate_protocol.json",
        "results/reopened_hybrid_geometry_gate/report.json",
        "results/reopened_hybrid_geometry_gate/scores.csv",
    ),
    (
        "geometry_gate_topology",
        "configs/reopened_hybrid_geometry_gate_topology_protocol.json",
        "results/reopened_hybrid_geometry_gate_topology/report.json",
        "results/reopened_hybrid_geometry_gate_topology/scores.csv",
    ),
    (
        "tidal_shape_gate",
        "configs/reopened_hybrid_tidal_shape_gate_protocol.json",
        "results/reopened_hybrid_tidal_shape_gate/report.json",
        "results/reopened_hybrid_tidal_shape_gate/scores.csv",
    ),
    (
        "tidal_shape_common_spherical",
        "configs/reopened_hybrid_tidal_shape_common_spherical_protocol.json",
        "results/reopened_hybrid_tidal_shape_common_spherical/report.json",
        "results/reopened_hybrid_tidal_shape_common_spherical/scores.csv",
    ),
    (
        "tidal_shape_common_spherical_adaptive",
        "configs/reopened_hybrid_tidal_shape_common_spherical_adaptive_protocol.json",
        "results/reopened_hybrid_tidal_shape_common_spherical_adaptive/report.json",
        "results/reopened_hybrid_tidal_shape_common_spherical_adaptive/scores.csv",
    ),
    (
        "radial_memory",
        "configs/reopened_hybrid_radial_memory_protocol.json",
        "results/reopened_hybrid_radial_memory/report.json",
        "results/reopened_hybrid_radial_memory/scores.csv",
    ),
    (
        "memory_carrier",
        "configs/reopened_hybrid_memory_carrier_protocol.json",
        "results/reopened_hybrid_memory_carrier/report.json",
        "results/reopened_hybrid_memory_carrier/scores.csv",
    ),
    (
        "slope_adaptive_carrier",
        "configs/reopened_hybrid_slope_adaptive_carrier_protocol.json",
        "results/reopened_hybrid_slope_adaptive_carrier/report.json",
        "results/reopened_hybrid_slope_adaptive_carrier/scores.csv",
    ),
    (
        "slope_response_modes",
        "configs/reopened_hybrid_slope_response_modes_protocol.json",
        "results/reopened_hybrid_slope_response_modes/report.json",
        "results/reopened_hybrid_slope_response_modes/scores.csv",
    ),
    (
        "slope_response_fine",
        "configs/reopened_hybrid_slope_response_fine_protocol.json",
        "results/reopened_hybrid_slope_response_fine/report.json",
        "results/reopened_hybrid_slope_response_fine/scores.csv",
    ),
    (
        "slope_response_pivot_extension",
        "configs/reopened_hybrid_slope_response_pivot_extension_protocol.json",
        "results/reopened_hybrid_slope_response_pivot_extension/report.json",
        "results/reopened_hybrid_slope_response_pivot_extension/scores.csv",
    ),
    (
        "slope_response_repeatability",
        "configs/reopened_hybrid_slope_response_best_repeatability_protocol.json",
        "results/reopened_hybrid_slope_response_best_repeatability/report.json",
        "results/reopened_hybrid_slope_response_best_repeatability/scores.csv",
    ),
    (
        "smoothed_local_slope",
        "configs/reopened_hybrid_smoothed_local_slope_protocol.json",
        "results/reopened_hybrid_smoothed_local_slope/report.json",
        "results/reopened_hybrid_smoothed_local_slope/scores.csv",
    ),
    (
        "smoothed_local_pivot_extension",
        "configs/reopened_hybrid_smoothed_local_pivot_extension_protocol.json",
        "results/reopened_hybrid_smoothed_local_pivot_extension/report.json",
        "results/reopened_hybrid_smoothed_local_pivot_extension/scores.csv",
    ),
    (
        "endpoint_power_memory",
        "configs/reopened_hybrid_endpoint_power_memory_protocol.json",
        "results/reopened_hybrid_endpoint_power_memory/report.json",
        "results/reopened_hybrid_endpoint_power_memory/scores.csv",
    ),
    (
        "endpoint_boundary_refinement",
        "configs/reopened_hybrid_endpoint_boundary_refinement_protocol.json",
        "results/reopened_hybrid_endpoint_boundary_refinement/report.json",
        "results/reopened_hybrid_endpoint_boundary_refinement/scores.csv",
    ),
    (
        "endpoint_high_q_ridge",
        "configs/reopened_hybrid_endpoint_high_q_ridge_protocol.json",
        "results/reopened_hybrid_endpoint_high_q_ridge/report.json",
        "results/reopened_hybrid_endpoint_high_q_ridge/scores.csv",
    ),
    (
        "endpoint_interaction_factorial",
        "configs/reopened_hybrid_endpoint_interaction_factorial_protocol.json",
        "results/reopened_hybrid_endpoint_interaction_factorial/report.json",
        "results/reopened_hybrid_endpoint_interaction_factorial/scores.csv",
    ),
    (
        "tidal_gate_memory",
        "configs/reopened_hybrid_tidal_gate_memory_protocol.json",
        "results/reopened_hybrid_tidal_gate_memory/report.json",
        "results/reopened_hybrid_tidal_gate_memory/scores.csv",
    ),
    (
        "tidal_gate_topology",
        "configs/reopened_hybrid_tidal_gate_topology_protocol.json",
        "results/reopened_hybrid_tidal_gate_topology/report.json",
        "results/reopened_hybrid_tidal_gate_topology/scores.csv",
    ),
    (
        "tidal_memory_placement",
        "configs/reopened_hybrid_tidal_memory_placement_protocol.json",
        "results/reopened_hybrid_tidal_memory_placement/report.json",
        "results/reopened_hybrid_tidal_memory_placement/scores.csv",
    ),
    (
        "profile_diffusion",
        "configs/reopened_hybrid_profile_diffusion_protocol.json",
        "results/reopened_hybrid_profile_diffusion/report.json",
        "results/reopened_hybrid_profile_diffusion/scores.csv",
    ),
]
ROBUSTNESS_REPORTS = [
    (
        "solar_safe_followup",
        "initial_robustness",
        "results/reopened_hybrid_raw_robustness/report.json",
    ),
    (
        "channel_saturation",
        "channel_robustness",
        "results/reopened_hybrid_channel_raw_robustness/report.json",
    ),
    (
        "channel_saturation_fine",
        "channel_fine_robustness",
        "results/reopened_hybrid_channel_fine_raw_robustness/report.json",
    ),
    (
        "channel_saturation_fine",
        "low_rg_robustness",
        "results/reopened_hybrid_channel_low_rg_raw_robustness/report.json",
    ),
    (
        "geometry_gate",
        "geometry_gate_robustness",
        "results/reopened_hybrid_geometry_gate_raw_robustness/report.json",
    ),
    (
        "geometry_gate_topology",
        "geometry_gate_topology_robustness",
        "results/reopened_hybrid_geometry_gate_topology_raw_robustness/report.json",
    ),
    (
        "tidal_shape_gate",
        "tidal_shape_gate_robustness",
        "results/reopened_hybrid_tidal_shape_gate_raw_robustness/report.json",
    ),
    (
        "tidal_shape_common_spherical",
        "tidal_shape_common_spherical_robustness",
        "results/reopened_hybrid_tidal_shape_common_spherical_raw_robustness/report.json",
    ),
    (
        "tidal_shape_common_spherical_adaptive",
        "tidal_shape_common_spherical_adaptive_robustness",
        "results/reopened_hybrid_tidal_shape_common_spherical_adaptive_raw_robustness/report.json",
    ),
    (
        "radial_memory",
        "radial_memory_robustness",
        "results/reopened_hybrid_radial_memory_raw_robustness/report.json",
    ),
    (
        "memory_carrier",
        "memory_carrier_robustness",
        "results/reopened_hybrid_memory_carrier_raw_robustness/report.json",
    ),
    (
        "memory_carrier",
        "memory_carrier_slope_neutral_robustness",
        "results/reopened_hybrid_memory_carrier_slope_neutral_raw_robustness/report.json",
    ),
    (
        "slope_adaptive_carrier",
        "slope_adaptive_carrier_robustness",
        "results/reopened_hybrid_slope_adaptive_carrier_raw_robustness/report.json",
    ),
    (
        "slope_response_modes",
        "slope_response_modes_robustness",
        "results/reopened_hybrid_slope_response_modes_raw_robustness/report.json",
    ),
    (
        "slope_response_fine",
        "slope_response_fine_robustness",
        "results/reopened_hybrid_slope_response_fine_raw_robustness/report.json",
    ),
    (
        "slope_response_pivot_extension",
        "slope_response_pivot_extension_robustness",
        "results/reopened_hybrid_slope_response_pivot_extension_raw_robustness/report.json",
    ),
    (
        "slope_response_repeatability",
        "slope_response_repeatability_robustness",
        "results/reopened_hybrid_slope_response_best_repeatability_raw/report.json",
    ),
    (
        "smoothed_local_slope",
        "smoothed_local_slope_robustness",
        "results/reopened_hybrid_smoothed_local_slope_raw_robustness/report.json",
    ),
    (
        "smoothed_local_pivot_extension",
        "smoothed_local_pivot_extension_robustness",
        "results/reopened_hybrid_smoothed_local_pivot_extension_raw_robustness/report.json",
    ),
    (
        "endpoint_power_memory",
        "endpoint_power_memory_robustness",
        "results/reopened_hybrid_endpoint_power_memory_raw_robustness/report.json",
    ),
    (
        "endpoint_boundary_refinement",
        "endpoint_boundary_refinement_robustness",
        "results/reopened_hybrid_endpoint_boundary_refinement_raw_robustness/report.json",
    ),
    (
        "endpoint_high_q_ridge",
        "endpoint_high_q_ridge_robustness",
        "results/reopened_hybrid_endpoint_high_q_ridge_raw_robustness/report.json",
    ),
    (
        "endpoint_interaction_factorial",
        "endpoint_interaction_factorial_robustness",
        "results/reopened_hybrid_endpoint_interaction_factorial_raw_robustness/report.json",
    ),
    (
        "tidal_gate_memory",
        "tidal_gate_memory_robustness",
        "results/reopened_hybrid_tidal_gate_memory_raw_robustness/report.json",
    ),
    (
        "tidal_gate_topology",
        "tidal_gate_topology_robustness",
        "results/reopened_hybrid_tidal_gate_topology_raw_robustness/report.json",
    ),
    (
        "tidal_memory_placement",
        "tidal_memory_placement_robustness",
        "results/reopened_hybrid_tidal_memory_placement_raw_robustness/report.json",
    ),
    (
        "profile_diffusion",
        "profile_diffusion_robustness",
        "results/reopened_hybrid_profile_diffusion_raw_robustness/report.json",
    ),
]
GEOMETRY_AUDIT_REPORT = (
    "results/reopened_geometry_indicator_audit/report.json"
)
TIDAL_SHAPE_AUDIT_REPORT = (
    "results/reopened_tidal_shape_indicator_audit/report.json"
)
TIDAL_COMMON_SPHERICAL_AUDIT_REPORT = (
    "results/reopened_tidal_shape_common_spherical_audit/report.json"
)
SPHERICAL_TIDAL_IDENTITY_REPORT = (
    "results/reopened_spherical_tidal_identity/report.json"
)
RADIAL_MEMORY_AUDIT_REPORT = (
    "results/reopened_radial_memory_transfer_audit/report.json"
)
MEMORY_CARRIER_FIXED_AUDIT_REPORT = (
    "results/reopened_hybrid_memory_carrier_audit/report.json"
)
MEMORY_CARRIER_SLOPE_AUDIT_REPORT = (
    "results/reopened_profile_slope_audit/report.json"
)
MEMORY_CARRIER_ANALYSIS_REPORT = (
    "results/reopened_hybrid_memory_carrier_analysis/report.json"
)
SLOPE_ADAPTIVE_CARRIER_ANALYSIS_REPORT = (
    "results/reopened_hybrid_slope_adaptive_carrier_analysis/report.json"
)
SLOPE_RESPONSE_MODES_ANALYSIS_REPORT = (
    "results/reopened_hybrid_slope_response_modes_analysis/report.json"
)
SMOOTHED_LOCAL_SLOPE_ANALYSIS_REPORT = (
    "results/reopened_hybrid_smoothed_local_slope_analysis/report.json"
)
SMOOTHED_LOCAL_PIVOT_EXTENSION_ANALYSIS_REPORT = (
    "results/reopened_hybrid_smoothed_local_pivot_extension_analysis/report.json"
)
ENDPOINT_POWER_MEMORY_ANALYSIS_REPORT = (
    "results/reopened_hybrid_endpoint_power_memory_analysis/report.json"
)
ENDPOINT_BOUNDARY_REFINEMENT_ANALYSIS_REPORT = (
    "results/reopened_hybrid_endpoint_boundary_refinement_analysis/report.json"
)
ENDPOINT_HIGH_Q_RIDGE_ANALYSIS_REPORT = (
    "results/reopened_hybrid_endpoint_high_q_ridge_analysis/report.json"
)
ENDPOINT_INTERACTION_FACTORIAL_ANALYSIS_REPORT = (
    "results/reopened_hybrid_endpoint_interaction_factorial_analysis/report.json"
)
TIDAL_GATE_MEMORY_ANALYSIS_REPORT = (
    "results/reopened_hybrid_tidal_gate_memory_analysis/report.json"
)
TIDAL_GATE_TOPOLOGY_ANALYSIS_REPORT = (
    "results/reopened_hybrid_tidal_gate_topology_analysis/report.json"
)
TIDAL_MEMORY_PLACEMENT_ANALYSIS_REPORT = (
    "results/reopened_hybrid_tidal_memory_placement_analysis/report.json"
)
PROFILE_DIFFUSION_ANALYSIS_REPORT = (
    "results/reopened_hybrid_profile_diffusion_analysis/report.json"
)
OUTPUT = ROOT / "results/reopened_hybrid_program"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get_row(frame: pd.DataFrame, stage: str, variant: str) -> pd.Series:
    selected = frame[(frame.stage == stage) & (frame.variant == variant)]
    if len(selected) != 1:
        raise RuntimeError(f"expected one row for {stage}/{variant}")
    return selected.iloc[0]


def main() -> None:
    tables = []
    input_hashes = {}
    for stage, config_name, report_name, scores_name in STAGES:
        config_path = ROOT / config_name
        report_path = ROOT / report_name
        scores_path = ROOT / scores_name
        protocol = json.loads(config_path.read_text(encoding="utf-8"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        scores = pd.read_csv(scores_path)
        settings = {
            row["name"]: json.dumps(row["settings"], sort_keys=True)
            for row in expand_variants(protocol)
        }
        scores.insert(0, "stage", stage)
        scores.insert(1, "protocol_version", protocol["protocol_version"])
        scores["settings_json"] = scores.variant.map(settings)
        scores["tidal_geometry_json"] = json.dumps(
            protocol.get("tidal_geometry", {}),
            sort_keys=True,
        )
        scores["evaluation_signature_json"] = scores.apply(
            lambda row: json.dumps(
                {
                    "settings": json.loads(row.settings_json),
                    "tidal_geometry": protocol.get("tidal_geometry", {}),
                },
                sort_keys=True,
            ),
            axis=1,
        )
        scores["parameter_json"] = scores.variant.map(
            lambda name: json.dumps(
                report["results"][name]["full_fit_parameters"],
                sort_keys=True,
            )
        )
        scores["boundary_parameters"] = scores.variant.map(
            lambda name: ",".join(
                parameter
                for parameter, hit in report["results"][name][
                    "full_fit_at_boundary"
                ].items()
                if hit
            )
        )
        tables.append(scores)
        input_hashes[stage] = {
            "protocol": sha256(config_path),
            "report": sha256(report_path),
            "scores": sha256(scores_path),
        }
    combined = pd.concat(tables, ignore_index=True, sort=False)
    combined["solar_all_pass"] = (
        combined.Cassini_proxy_pass.astype(bool)
        & combined.Earth_pass.astype(bool)
        & combined.Mercury_pass.astype(bool)
    )
    combined["fully_evaluable"] = (
        combined.solar_all_pass
        & combined.raw_all_roots_converged.astype(bool)
    )
    references = {
        "SPARC_fixed_RAR_outer_RMSE_km_s": 10.681519066526649,
        "raw_baryons_RMS_arcsec": 27.43864684589079,
        "raw_simple_MOND_RMS_arcsec": 28.188493432162694,
        "raw_compact_halo_RMS_arcsec": 9.048410306058654,
        "raw_compact_halo_pooled_reduced_chi2": 142.61809057489606,
    }
    combined["SPARC_RMSE_ratio_to_RAR"] = (
        combined.SPARC_outer_RMSE_km_s
        / references["SPARC_fixed_RAR_outer_RMSE_km_s"]
    )
    combined["raw_RMS_ratio_to_compact_halo"] = (
        combined.raw_lensing_RMS_arcsec
        / references["raw_compact_halo_RMS_arcsec"]
    )
    combined["raw_improvement_vs_baryons_percent"] = 100.0 * (
        1.0
        - combined.raw_lensing_RMS_arcsec
        / references["raw_baryons_RMS_arcsec"]
    )
    combined["raw_improvement_vs_simple_MOND_percent"] = 100.0 * (
        1.0
        - combined.raw_lensing_RMS_arcsec
        / references["raw_simple_MOND_RMS_arcsec"]
    )
    combined["repeat_count_for_same_formula_settings"] = combined.groupby(
        "settings_json"
    )["settings_json"].transform("size")

    combined["raw_eight_start_RMS_arcsec"] = np.nan
    combined["raw_eight_start_all_roots_converged"] = pd.Series(
        pd.NA, index=combined.index, dtype="boolean"
    )
    combined["raw_eight_start_fractional_change"] = np.nan
    combined["raw_eight_start_pooled_reduced_chi2"] = np.nan
    robustness_hashes = {}
    robustness_rows = 0
    for stage, label, report_name in ROBUSTNESS_REPORTS:
        robust_path = ROOT / report_name
        robustness = json.loads(robust_path.read_text(encoding="utf-8"))
        robustness_hashes[label] = sha256(robust_path)
        robustness_rows += len(robustness["selected_variants"])
        for variant, comparison in robustness["comparisons"].items():
            mask = combined.stage.eq(stage) & combined.variant.eq(variant)
            if int(mask.sum()) != 1:
                raise RuntimeError(
                    f"expected one robustness target for {stage}/{variant}"
                )
            robust = comparison["eight_start"]
            combined.loc[mask, "raw_eight_start_RMS_arcsec"] = robust[
                "equal_system_radial_RMS_arcsec"
            ]
            combined.loc[
                mask, "raw_eight_start_all_roots_converged"
            ] = robust["all_roots_converged"]
            combined.loc[
                mask, "raw_eight_start_fractional_change"
            ] = comparison["RMS_fractional_change"]
            combined.loc[
                mask, "raw_eight_start_pooled_reduced_chi2"
            ] = robust["pooled_reduced_chi2"]

    initial_additive = get_row(
        combined, "initial", "interaction:interaction_eta=0"
    )
    initial_product = get_row(
        combined, "initial", "interaction:interaction_eta=1"
    )
    screen_low = get_row(combined, "initial", "screen:screen_power=0.7")
    screen_safe = get_row(
        combined, "solar_safe_followup", "fine_screen:screen_power=1.5"
    )
    saturation_low = get_row(
        combined,
        "solar_safe_followup",
        "solar_safe_saturation:saturation_ceiling=3",
    )
    saturation_high = get_row(
        combined,
        "solar_safe_followup",
        "solar_safe_saturation:saturation_ceiling=12",
    )
    threshold_zero = get_row(
        combined,
        "screened_structure",
        "screened_moving_threshold:threshold_acceleration_power=0",
    )
    threshold_high = get_row(
        combined,
        "screened_structure",
        "screened_moving_threshold:threshold_acceleration_power=0.75",
    )
    cross_base = get_row(
        combined,
        "threshold_saturation_cross",
        "ceiling_8:threshold_acceleration_power=0",
    )
    cross_shift = get_row(
        combined,
        "threshold_saturation_cross",
        "ceiling_8:threshold_acceleration_power=1.5",
    )
    channel_baseline = get_row(
        combined,
        "channel_saturation",
        "baseline_unsaturated:screen_power=1.5",
    )
    sigma_channel = get_row(
        combined,
        "channel_saturation_fine",
        "sigma_fine:sigma_saturation_ceiling=6.5",
    )
    rg_low_incomplete = get_row(
        combined,
        "channel_saturation_fine",
        "rg_fine:rg_saturation_ceiling=2.5",
    )
    rg_boundary = get_row(
        combined,
        "channel_saturation_fine",
        "rg_fine:rg_saturation_ceiling=2.75",
    )
    dual_channel = get_row(
        combined,
        "channel_saturation_fine",
        "rg_2_sigma_fine:sigma_saturation_ceiling=1.5",
    )
    density_gate_lens = get_row(
        combined,
        "geometry_gate",
        "density_ratio_orientation_control:channel_gate_cluster_high=0",
    )
    density_gate_reversed = get_row(
        combined,
        "geometry_gate",
        "density_ratio_orientation_control:channel_gate_cluster_high=1",
    )
    mass_gate_soft = get_row(
        combined,
        "geometry_gate",
        "mass_sharpness:channel_gate_sharpness=0.5",
    )
    rg_cluster_soft = get_row(
        combined,
        "geometry_gate_topology",
        "rg_only_cluster:channel_gate_sharpness=0.5",
    )
    rg_cluster_sharp = get_row(
        combined,
        "geometry_gate_topology",
        "rg_only_cluster:channel_gate_sharpness=4",
    )
    rg_galaxy_soft = get_row(
        combined,
        "geometry_gate_topology",
        "rg_only_galaxy:channel_gate_sharpness=0.5",
    )
    rg_galaxy_sharp = get_row(
        combined,
        "geometry_gate_topology",
        "rg_only_galaxy:channel_gate_sharpness=8",
    )
    sigma_cluster_soft = get_row(
        combined,
        "geometry_gate_topology",
        "sigma_only_cluster:channel_gate_sharpness=0.5",
    )
    sigma_galaxy_soft = get_row(
        combined,
        "geometry_gate_topology",
        "sigma_only_galaxy:channel_gate_sharpness=0.5",
    )
    tidal_middle_reversed = get_row(
        combined,
        "tidal_shape_gate",
        "middle_orientation_control:channel_gate_cluster_high=0",
    )
    tidal_middle_expected = get_row(
        combined,
        "tidal_shape_gate",
        "middle_orientation_control:channel_gate_cluster_high=1",
    )
    tidal_lensing_focus = get_row(
        combined,
        "tidal_shape_gate",
        "middle_ratio_pivot:channel_gate_pivot=0.85",
    )
    tidal_lensing_incomplete = get_row(
        combined,
        "tidal_shape_gate",
        "middle_ratio_sharpness:channel_gate_sharpness=5",
    )
    tidal_middle_reversed_common = get_row(
        combined,
        "tidal_shape_common_spherical",
        "middle_orientation_control:channel_gate_cluster_high=0",
    )
    common_adaptive_compromise = get_row(
        combined,
        "tidal_shape_common_spherical_adaptive",
        "third_axis_common_orientation:channel_gate_cluster_high=0",
    )
    common_adaptive_raw = get_row(
        combined,
        "tidal_shape_common_spherical_adaptive",
        "positive_fraction_pivot:channel_gate_pivot=0.82",
    )
    common_adaptive_stable_raw = get_row(
        combined,
        "tidal_shape_common_spherical_adaptive",
        "traceless_fraction_sharpness:channel_gate_sharpness=40",
    )
    common_adaptive_galaxy_incomplete = get_row(
        combined,
        "tidal_shape_common_spherical_adaptive",
        "determinant_orientation:channel_gate_cluster_high=1",
    )
    radial_dual_local = get_row(
        combined,
        "radial_memory",
        "dual_strength:radial_memory_strength=0",
    )
    radial_dual_full = get_row(
        combined,
        "radial_memory",
        "dual_strength:radial_memory_strength=1",
    )
    radial_dual_short = get_row(
        combined,
        "radial_memory",
        "dual_scale:radial_memory_log_scale=0.25",
    )
    radial_dual_scale_2 = get_row(
        combined,
        "radial_memory",
        "dual_scale:radial_memory_log_scale=2",
    )
    radial_dual_scale_8 = get_row(
        combined,
        "radial_memory",
        "dual_scale:radial_memory_log_scale=8",
    )
    radial_dual_reverse = get_row(
        combined,
        "radial_memory",
        "dual_direction:radial_memory_outer_to_inner=1",
    )
    radial_dual_post_screen = get_row(
        combined,
        "radial_memory",
        "dual_placement:radial_memory_pre_screen=0",
    )
    radial_dual_pre_screen = get_row(
        combined,
        "radial_memory",
        "dual_placement:radial_memory_pre_screen=1",
    )
    radial_unsaturated_local = get_row(
        combined,
        "radial_memory",
        "unsaturated_strength:radial_memory_strength=0",
    )
    radial_unsaturated_full = get_row(
        combined,
        "radial_memory",
        "unsaturated_strength:radial_memory_strength=1",
    )
    mixed_closure_rows = combined[
        combined.stage.eq("tidal_shape_gate")
    ].set_index("variant")
    common_closure_rows = combined[
        combined.stage.eq("tidal_shape_common_spherical")
    ].set_index("variant")
    shared_closure_variants = mixed_closure_rows.index.intersection(
        common_closure_rows.index
    )
    maximum_closure_bridge_delta = float(
        np.max(
            np.abs(
                mixed_closure_rows.loc[
                    shared_closure_variants, "bridge_RMSE_dex"
                ].to_numpy(float)
                - common_closure_rows.loc[
                    shared_closure_variants, "bridge_RMSE_dex"
                ].to_numpy(float)
            )
        )
    )
    maximum_closure_raw_delta = float(
        np.max(
            np.abs(
                mixed_closure_rows.loc[
                    shared_closure_variants, "raw_lensing_RMS_arcsec"
                ].to_numpy(float)
                - common_closure_rows.loc[
                    shared_closure_variants, "raw_lensing_RMS_arcsec"
                ].to_numpy(float)
            )
        )
    )
    robust_mask = (
        combined.raw_eight_start_RMS_arcsec.notna()
        & combined.raw_eight_start_all_roots_converged.fillna(False).astype(bool)
    )
    robust_candidates = combined[robust_mask].copy()
    best_verified_raw = robust_candidates.sort_values(
        "raw_eight_start_RMS_arcsec"
    ).iloc[0]
    robust_candidates["eight_start_cross_domain_ratio"] = np.maximum(
        robust_candidates.SPARC_RMSE_ratio_to_RAR,
        robust_candidates.raw_eight_start_RMS_arcsec
        / references["raw_compact_halo_RMS_arcsec"],
    )
    best_verified_compromise = robust_candidates.sort_values(
        ["eight_start_cross_domain_ratio", "bridge_RMSE_dex"]
    ).iloc[0]

    unique_settings = int(combined.settings_json.nunique())
    unique_evaluations = int(combined.evaluation_signature_json.nunique())
    geometry_audit_path = ROOT / GEOMETRY_AUDIT_REPORT
    geometry_audit = json.loads(
        geometry_audit_path.read_text(encoding="utf-8")
    )
    geometry_ranking = geometry_audit[
        "system_level_SPARC_vs_CLASH_ranking"
    ]
    geometry_by_name = {
        row["indicator"]: row for row in geometry_ranking
    }
    tidal_audit_path = ROOT / TIDAL_SHAPE_AUDIT_REPORT
    tidal_audit = json.loads(
        tidal_audit_path.read_text(encoding="utf-8")
    )
    tidal_ranking = tidal_audit[
        "system_level_SPARC_vs_CLASH_ranking"
    ]
    tidal_by_name = {
        row["indicator"]: row for row in tidal_ranking
    }
    common_tidal_audit_path = ROOT / TIDAL_COMMON_SPHERICAL_AUDIT_REPORT
    common_tidal_audit = json.loads(
        common_tidal_audit_path.read_text(encoding="utf-8")
    )
    common_tidal_by_name = {
        row["indicator"]: row
        for row in common_tidal_audit[
            "system_level_SPARC_vs_CLASH_ranking"
        ]
    }
    spherical_identity_path = ROOT / SPHERICAL_TIDAL_IDENTITY_REPORT
    spherical_identity = json.loads(
        spherical_identity_path.read_text(encoding="utf-8")
    )
    radial_memory_audit_path = ROOT / RADIAL_MEMORY_AUDIT_REPORT
    radial_memory_audit = json.loads(
        radial_memory_audit_path.read_text(encoding="utf-8")
    )
    memory_carrier_fixed_path = ROOT / MEMORY_CARRIER_FIXED_AUDIT_REPORT
    memory_carrier_fixed = json.loads(
        memory_carrier_fixed_path.read_text(encoding="utf-8")
    )
    memory_carrier_slope_path = ROOT / MEMORY_CARRIER_SLOPE_AUDIT_REPORT
    memory_carrier_slope = json.loads(
        memory_carrier_slope_path.read_text(encoding="utf-8")
    )
    memory_carrier_analysis_path = ROOT / MEMORY_CARRIER_ANALYSIS_REPORT
    memory_carrier_analysis = json.loads(
        memory_carrier_analysis_path.read_text(encoding="utf-8")
    )
    slope_adaptive_analysis_path = ROOT / SLOPE_ADAPTIVE_CARRIER_ANALYSIS_REPORT
    slope_adaptive_analysis = json.loads(
        slope_adaptive_analysis_path.read_text(encoding="utf-8")
    )
    slope_response_analysis_path = ROOT / SLOPE_RESPONSE_MODES_ANALYSIS_REPORT
    slope_response_analysis = json.loads(
        slope_response_analysis_path.read_text(encoding="utf-8")
    )
    smoothed_local_analysis_path = ROOT / SMOOTHED_LOCAL_SLOPE_ANALYSIS_REPORT
    smoothed_local_analysis = json.loads(
        smoothed_local_analysis_path.read_text(encoding="utf-8")
    )
    endpoint_analysis_path = (
        ROOT / SMOOTHED_LOCAL_PIVOT_EXTENSION_ANALYSIS_REPORT
    )
    endpoint_analysis = json.loads(
        endpoint_analysis_path.read_text(encoding="utf-8")
    )
    power_memory_analysis_path = ROOT / ENDPOINT_POWER_MEMORY_ANALYSIS_REPORT
    power_memory_analysis = json.loads(
        power_memory_analysis_path.read_text(encoding="utf-8")
    )
    boundary_analysis_path = ROOT / ENDPOINT_BOUNDARY_REFINEMENT_ANALYSIS_REPORT
    boundary_analysis = json.loads(
        boundary_analysis_path.read_text(encoding="utf-8")
    )
    high_q_analysis_path = ROOT / ENDPOINT_HIGH_Q_RIDGE_ANALYSIS_REPORT
    high_q_analysis = json.loads(
        high_q_analysis_path.read_text(encoding="utf-8")
    )
    factorial_analysis_path = ROOT / ENDPOINT_INTERACTION_FACTORIAL_ANALYSIS_REPORT
    factorial_analysis = json.loads(
        factorial_analysis_path.read_text(encoding="utf-8")
    )
    tidal_gate_memory_analysis_path = ROOT / TIDAL_GATE_MEMORY_ANALYSIS_REPORT
    tidal_gate_memory_analysis = json.loads(
        tidal_gate_memory_analysis_path.read_text(encoding="utf-8")
    )
    tidal_gate_topology_analysis_path = ROOT / TIDAL_GATE_TOPOLOGY_ANALYSIS_REPORT
    tidal_gate_topology_analysis = json.loads(
        tidal_gate_topology_analysis_path.read_text(encoding="utf-8")
    )
    tidal_memory_placement_analysis_path = (
        ROOT / TIDAL_MEMORY_PLACEMENT_ANALYSIS_REPORT
    )
    tidal_memory_placement_analysis = json.loads(
        tidal_memory_placement_analysis_path.read_text(encoding="utf-8")
    )
    profile_diffusion_analysis_path = ROOT / PROFILE_DIFFUSION_ANALYSIS_REPORT
    profile_diffusion_analysis = json.loads(
        profile_diffusion_analysis_path.read_text(encoding="utf-8")
    )
    summary = {
        "status": "completed consolidated reopened-hybrid sensitivity program",
        "coverage": {
            "scored_rows": len(combined),
            "unique_formula_settings": unique_settings,
            "unique_formula_evaluation_contexts": unique_evaluations,
            "bridge_systems_per_row": 64,
            "SPARC_galaxies_per_row": 131,
            "SPARC_outer_points_per_row": 968,
            "raw_clusters_per_row": 4,
            "raw_heldout_images_per_row": 11,
            "eight_start_raw_robustness_rows": robustness_rows,
        },
        "references": references,
        "input_hashes": input_hashes,
        "robustness_report_hashes": robustness_hashes,
        "geometry_indicator_audit": {
            "report_sha256": sha256(geometry_audit_path),
            "systems": geometry_audit["coverage"]["systems"],
            "equivalent_mass_system_AUC": geometry_by_name[
                "log10_equivalent_enclosed_mass_msun"
            ]["separation_auc"],
            "local_to_mean_density_system_AUC": geometry_by_name[
                "log10_local_to_mean_density_ratio"
            ]["separation_auc"],
        },
        "tidal_shape_indicator_audit": {
            "report_sha256": sha256(tidal_audit_path),
            "systems": tidal_audit["coverage"]["systems"],
            "points": tidal_audit["coverage"]["points"],
            "l1_dominance_system_AUC": tidal_by_name[
                "tidal_l1_dominance"
            ]["separation_auc"],
            "middle_to_max_system_AUC": tidal_by_name[
                "tidal_middle_to_max"
            ]["separation_auc"],
            "third_axis_fraction_system_AUC": tidal_by_name[
                "tidal_third_axis_abs_fraction"
            ]["separation_auc"],
            "solar_point_mass_invariants": tidal_audit[
                "solar_point_mass_invariants"
            ],
            "geometry_methods": tidal_audit["coverage"][
                "geometry_methods"
            ],
        },
        "common_spherical_tidal_audit": {
            "report_sha256": sha256(common_tidal_audit_path),
            "systems": common_tidal_audit["coverage"]["systems"],
            "points": common_tidal_audit["coverage"]["points"],
            "sparc_method": common_tidal_audit["coverage"]["sparc_method"],
            "l1_dominance_system_AUC": common_tidal_by_name[
                "tidal_l1_dominance"
            ]["separation_auc"],
            "middle_to_max_system_AUC": common_tidal_by_name[
                "tidal_middle_to_max"
            ]["separation_auc"],
            "signed_determinant_system_AUC": common_tidal_by_name[
                "tidal_signed_determinant_shape"
            ]["separation_auc"],
        },
        "spherical_tidal_identity": {
            "report_sha256": sha256(spherical_identity_path),
            "status": spherical_identity["status"],
            "points": spherical_identity["points"],
            "systems": spherical_identity["systems"],
            "identity": spherical_identity["identity"],
            "maximum_error_over_all_invariants": spherical_identity[
                "maximum_error_over_all_invariants"
            ],
            "implication": spherical_identity["implication"],
        },
        "radial_memory_fixed_parameter_audit": {
            "report_sha256": sha256(radial_memory_audit_path),
            "rows": radial_memory_audit["rows"],
            "best_fixed_parameter_galaxy_setting_by_base": (
                radial_memory_audit[
                    "best_fixed_parameter_galaxy_setting_by_base"
                ]
            ),
        },
        "memory_carrier_audits": {
            "fixed_parameter_report_sha256": sha256(
                memory_carrier_fixed_path
            ),
            "fixed_parameter_rows": memory_carrier_fixed["rows"],
            "fixed_parameter_solar_valid_rows": memory_carrier_fixed[
                "solar_valid_rows"
            ],
            "profile_slope_report_sha256": sha256(
                memory_carrier_slope_path
            ),
            "SPARC_profile_slope_systems": memory_carrier_slope["SPARC"][
                "systems"
            ],
            "CLASH_profile_slope_systems": memory_carrier_slope["CLASH"][
                "systems"
            ],
            "analysis_report_sha256": sha256(memory_carrier_analysis_path),
            "transport_power_surface_points": memory_carrier_analysis[
                "coverage"
            ]["transport_power_surface_points"],
            "slope_neutral_fixed_parameter_rows": memory_carrier_analysis[
                "coverage"
            ]["slope_neutral_fixed_parameter_variants"],
        },
        "slope_adaptive_carrier_audits": {
            "analysis_report_sha256": sha256(
                slope_adaptive_analysis_path
            ),
            "fixed_parameter_rows": slope_adaptive_analysis["coverage"][
                "fixed_parameter_variants"
            ],
            "full_variants": slope_adaptive_analysis["coverage"][
                "full_variants"
            ],
            "eight_start_raw_replays": slope_adaptive_analysis[
                "coverage"
            ]["eight_start_raw_replays"],
            "SPARC_gate_geometry_systems": slope_adaptive_analysis[
                "coverage"
            ]["SPARC_gate_geometry_systems"],
            "CLASH_gate_geometry_systems": slope_adaptive_analysis[
                "coverage"
            ]["CLASH_gate_geometry_systems"],
        },
        "observed_sensitivities": {
            "interaction_refit_degeneracy": {
                "eta_0_to_1_bridge_change_dex": float(
                    initial_product.bridge_RMSE_dex
                    - initial_additive.bridge_RMSE_dex
                ),
                "eta_0_to_1_SPARC_change_km_s": float(
                    initial_product.SPARC_outer_RMSE_km_s
                    - initial_additive.SPARC_outer_RMSE_km_s
                ),
                "eta_0_to_1_raw_change_arcsec": float(
                    initial_product.raw_lensing_RMS_arcsec
                    - initial_additive.raw_lensing_RMS_arcsec
                ),
            },
            "screen_exponent_controls_solar_limit": {
                "n_0p7_maximum_solar_fractional_change": float(
                    screen_low.solar_maximum_fractional_change
                ),
                "n_1p5_maximum_solar_fractional_change": float(
                    screen_safe.solar_maximum_fractional_change
                ),
                "n_0p7_Mercury_precession_mas_per_century": float(
                    screen_low.Mercury_precession_mas_per_century
                ),
                "n_1p5_Mercury_precession_mas_per_century": float(
                    screen_safe.Mercury_precession_mas_per_century
                ),
            },
            "saturation_trades_galaxies_against_clusters": {
                "ceiling_3_SPARC_RMSE_km_s": float(
                    saturation_low.SPARC_outer_RMSE_km_s
                ),
                "ceiling_12_SPARC_RMSE_km_s": float(
                    saturation_high.SPARC_outer_RMSE_km_s
                ),
                "ceiling_3_bridge_RMSE_dex": float(
                    saturation_low.bridge_RMSE_dex
                ),
                "ceiling_12_bridge_RMSE_dex": float(
                    saturation_high.bridge_RMSE_dex
                ),
                "ceiling_3_raw_RMS_arcsec_two_start": float(
                    saturation_low.raw_lensing_RMS_arcsec
                ),
                "ceiling_12_raw_RMS_arcsec_two_start": float(
                    saturation_high.raw_lensing_RMS_arcsec
                ),
            },
            "positive_threshold_shift_has_correct_but_small_direction": {
                "alpha_0_SPARC_RMSE_km_s": float(
                    threshold_zero.SPARC_outer_RMSE_km_s
                ),
                "alpha_0p75_SPARC_RMSE_km_s": float(
                    threshold_high.SPARC_outer_RMSE_km_s
                ),
                "alpha_0_bridge_RMSE_dex": float(
                    threshold_zero.bridge_RMSE_dex
                ),
                "alpha_0p75_bridge_RMSE_dex": float(
                    threshold_high.bridge_RMSE_dex
                ),
                "alpha_0_raw_RMS_arcsec": float(
                    threshold_zero.raw_lensing_RMS_arcsec
                ),
                "alpha_0p75_raw_RMS_arcsec": float(
                    threshold_high.raw_lensing_RMS_arcsec
                ),
            },
            "threshold_saturation_nonadditivity": {
                "ceiling_8_alpha_0_SPARC_RMSE_km_s": float(
                    cross_base.SPARC_outer_RMSE_km_s
                ),
                "ceiling_8_alpha_1p5_SPARC_RMSE_km_s": float(
                    cross_shift.SPARC_outer_RMSE_km_s
                ),
                "ceiling_8_alpha_0_bridge_RMSE_dex": float(
                    cross_base.bridge_RMSE_dex
                ),
                "ceiling_8_alpha_1p5_bridge_RMSE_dex": float(
                    cross_shift.bridge_RMSE_dex
                ),
            },
            "sigma_channel_cap_modestly_separates_galaxies": {
                "unsaturated_SPARC_RMSE_km_s": float(
                    channel_baseline.SPARC_outer_RMSE_km_s
                ),
                "sigma_ceiling_6p5_SPARC_RMSE_km_s": float(
                    sigma_channel.SPARC_outer_RMSE_km_s
                ),
                "SPARC_improvement_percent": float(
                    100.0
                    * (
                        1.0
                        - sigma_channel.SPARC_outer_RMSE_km_s
                        / channel_baseline.SPARC_outer_RMSE_km_s
                    )
                ),
                "unsaturated_raw_eight_start_RMS_arcsec": float(
                    channel_baseline.raw_eight_start_RMS_arcsec
                ),
                "sigma_ceiling_6p5_raw_eight_start_RMS_arcsec": float(
                    sigma_channel.raw_eight_start_RMS_arcsec
                ),
            },
            "rg_channel_cap_is_lensing_anti_galaxy_lever": {
                "unsaturated_SPARC_RMSE_km_s": float(
                    channel_baseline.SPARC_outer_RMSE_km_s
                ),
                "rg_ceiling_2p75_SPARC_RMSE_km_s": float(
                    rg_boundary.SPARC_outer_RMSE_km_s
                ),
                "SPARC_worsening_percent": float(
                    100.0
                    * (
                        rg_boundary.SPARC_outer_RMSE_km_s
                        / channel_baseline.SPARC_outer_RMSE_km_s
                        - 1.0
                    )
                ),
                "unsaturated_raw_eight_start_RMS_arcsec": float(
                    channel_baseline.raw_eight_start_RMS_arcsec
                ),
                "rg_ceiling_2p75_raw_eight_start_RMS_arcsec": float(
                    rg_boundary.raw_eight_start_RMS_arcsec
                ),
                "raw_improvement_percent": float(
                    100.0
                    * (
                        1.0
                        - rg_boundary.raw_eight_start_RMS_arcsec
                        / channel_baseline.raw_eight_start_RMS_arcsec
                    )
                ),
            },
            "raw_root_boundary": {
                "rg_ceiling_2p5_two_start_RMS_arcsec": float(
                    rg_low_incomplete.raw_lensing_RMS_arcsec
                ),
                "rg_ceiling_2p5_eight_start_RMS_arcsec": float(
                    rg_low_incomplete.raw_eight_start_RMS_arcsec
                ),
                "rg_ceiling_2p5_eight_start_all_roots": bool(
                    rg_low_incomplete.raw_eight_start_all_roots_converged
                ),
                "rg_ceiling_2p75_two_start_RMS_arcsec": float(
                    rg_boundary.raw_lensing_RMS_arcsec
                ),
                "rg_ceiling_2p75_eight_start_RMS_arcsec": float(
                    rg_boundary.raw_eight_start_RMS_arcsec
                ),
                "rg_ceiling_2p75_eight_start_all_roots": bool(
                    rg_boundary.raw_eight_start_all_roots_converged
                ),
            },
            "dual_channel_best_local_compromise": {
                "rg_ceiling": 2.0,
                "sigma_ceiling": 1.5,
                "SPARC_RMSE_km_s": float(
                    dual_channel.SPARC_outer_RMSE_km_s
                ),
                "bridge_RMSE_dex": float(dual_channel.bridge_RMSE_dex),
                "raw_eight_start_RMS_arcsec": float(
                    dual_channel.raw_eight_start_RMS_arcsec
                ),
                "raw_eight_start_all_roots": bool(
                    dual_channel.raw_eight_start_all_roots_converged
                ),
                "boundary_parameters": dual_channel.boundary_parameters,
            },
            "classifier_quality_does_not_imply_gravity_gate_quality": {
                "equivalent_mass_system_AUC": geometry_by_name[
                    "log10_equivalent_enclosed_mass_msun"
                ]["separation_auc"],
                "soft_mass_gate_SPARC_RMSE_km_s": float(
                    mass_gate_soft.SPARC_outer_RMSE_km_s
                ),
                "soft_mass_gate_bridge_RMSE_dex": float(
                    mass_gate_soft.bridge_RMSE_dex
                ),
                "soft_mass_gate_raw_eight_start_RMS_arcsec": float(
                    mass_gate_soft.raw_eight_start_RMS_arcsec
                ),
                "local_to_mean_density_system_AUC": geometry_by_name[
                    "log10_local_to_mean_density_ratio"
                ]["separation_auc"],
            },
            "density_ratio_gate_orientation_tradeoff": {
                "cluster_low_orientation_SPARC_RMSE_km_s": float(
                    density_gate_lens.SPARC_outer_RMSE_km_s
                ),
                "cluster_low_orientation_raw_eight_start_RMS_arcsec": float(
                    density_gate_lens.raw_eight_start_RMS_arcsec
                ),
                "cluster_low_orientation_all_roots": bool(
                    density_gate_lens.raw_eight_start_all_roots_converged
                ),
                "reversed_orientation_SPARC_RMSE_km_s": float(
                    density_gate_reversed.SPARC_outer_RMSE_km_s
                ),
                "reversed_orientation_raw_eight_start_RMS_arcsec": float(
                    density_gate_reversed.raw_eight_start_RMS_arcsec
                ),
                "reversed_orientation_all_roots": bool(
                    density_gate_reversed.raw_eight_start_all_roots_converged
                ),
            },
            "independent_rg_gate_topology": {
                "cluster_side_soft_SPARC_RMSE_km_s": float(
                    rg_cluster_soft.SPARC_outer_RMSE_km_s
                ),
                "cluster_side_soft_raw_eight_start_RMS_arcsec": float(
                    rg_cluster_soft.raw_eight_start_RMS_arcsec
                ),
                "cluster_side_sharp_SPARC_RMSE_km_s": float(
                    rg_cluster_sharp.SPARC_outer_RMSE_km_s
                ),
                "cluster_side_sharp_raw_eight_start_RMS_arcsec": float(
                    rg_cluster_sharp.raw_eight_start_RMS_arcsec
                ),
                "galaxy_side_soft_SPARC_RMSE_km_s": float(
                    rg_galaxy_soft.SPARC_outer_RMSE_km_s
                ),
                "galaxy_side_soft_raw_eight_start_RMS_arcsec": float(
                    rg_galaxy_soft.raw_eight_start_RMS_arcsec
                ),
                "galaxy_side_soft_all_roots": bool(
                    rg_galaxy_soft.raw_eight_start_all_roots_converged
                ),
                "galaxy_side_sharp_SPARC_RMSE_km_s": float(
                    rg_galaxy_sharp.SPARC_outer_RMSE_km_s
                ),
                "galaxy_side_sharp_raw_eight_start_RMS_arcsec": float(
                    rg_galaxy_sharp.raw_eight_start_RMS_arcsec
                ),
                "galaxy_side_sharp_all_roots": bool(
                    rg_galaxy_sharp.raw_eight_start_all_roots_converged
                ),
            },
            "sigma_gate_topology_is_negligible_after_refit": {
                "cluster_side_SPARC_RMSE_km_s": float(
                    sigma_cluster_soft.SPARC_outer_RMSE_km_s
                ),
                "galaxy_side_SPARC_RMSE_km_s": float(
                    sigma_galaxy_soft.SPARC_outer_RMSE_km_s
                ),
                "absolute_SPARC_difference_km_s": float(
                    abs(
                        sigma_cluster_soft.SPARC_outer_RMSE_km_s
                        - sigma_galaxy_soft.SPARC_outer_RMSE_km_s
                    )
                ),
                "cluster_side_raw_RMS_arcsec": float(
                    sigma_cluster_soft.raw_lensing_RMS_arcsec
                ),
                "galaxy_side_raw_RMS_arcsec": float(
                    sigma_galaxy_soft.raw_lensing_RMS_arcsec
                ),
            },
            "tidal_shape_gate_orientation_tradeoff": {
                "middle_ratio_system_AUC": tidal_by_name[
                    "tidal_middle_to_max"
                ]["separation_auc"],
                "expected_cluster_high_SPARC_RMSE_km_s": float(
                    tidal_middle_expected.SPARC_outer_RMSE_km_s
                ),
                "expected_cluster_high_raw_two_start_RMS_arcsec": float(
                    tidal_middle_expected.raw_lensing_RMS_arcsec
                ),
                "reversed_cluster_low_SPARC_RMSE_km_s": float(
                    tidal_middle_reversed.SPARC_outer_RMSE_km_s
                ),
                "reversed_cluster_low_raw_eight_start_RMS_arcsec": float(
                    tidal_middle_reversed.raw_eight_start_RMS_arcsec
                ),
                "reversed_cluster_low_all_roots": bool(
                    tidal_middle_reversed.raw_eight_start_all_roots_converged
                ),
            },
            "tidal_shape_lensing_branch": {
                "pivot_0p85_SPARC_RMSE_km_s": float(
                    tidal_lensing_focus.SPARC_outer_RMSE_km_s
                ),
                "pivot_0p85_raw_two_start_RMS_arcsec": float(
                    tidal_lensing_focus.raw_lensing_RMS_arcsec
                ),
                "pivot_0p85_raw_eight_start_RMS_arcsec": float(
                    tidal_lensing_focus.raw_eight_start_RMS_arcsec
                ),
                "pivot_0p85_eight_start_all_roots": bool(
                    tidal_lensing_focus.raw_eight_start_all_roots_converged
                ),
                "sharpness_5_two_start_RMS_arcsec": float(
                    tidal_lensing_incomplete.raw_lensing_RMS_arcsec
                ),
                "sharpness_5_two_start_all_roots": bool(
                    tidal_lensing_incomplete.raw_all_roots_converged
                ),
                "sharpness_5_eight_start_RMS_arcsec": float(
                    tidal_lensing_incomplete.raw_eight_start_RMS_arcsec
                ),
                "sharpness_5_eight_start_all_roots": bool(
                    tidal_lensing_incomplete.raw_eight_start_all_roots_converged
                ),
                "sharpness_5_eight_start_pooled_reduced_chi2": float(
                    tidal_lensing_incomplete.raw_eight_start_pooled_reduced_chi2
                ),
                "compact_halo_pooled_reduced_chi2": references[
                    "raw_compact_halo_pooled_reduced_chi2"
            ],
        },
        "slope_response_mode_audits": {
            "analysis_report_sha256": sha256(slope_response_analysis_path),
            "fixed_parameter_rows": slope_response_analysis["coverage"][
                "fixed_parameter_variants"
            ],
            "full_variants": slope_response_analysis["coverage"][
                "full_universal_refits"
            ],
            "eight_start_raw_replays": slope_response_analysis["coverage"][
                "eight_start_raw_replays"
            ],
            "SPARC_profile_slopes": slope_response_analysis["coverage"][
                "SPARC_profile_slopes"
            ],
            "CLASH_profile_slopes": slope_response_analysis["coverage"][
                "CLASH_profile_slopes"
            ],
        },
        "smoothed_local_slope_audits": {
            "analysis_report_sha256": sha256(smoothed_local_analysis_path),
            "full_variants": smoothed_local_analysis["coverage"][
                "full_universal_refits"
            ],
            "eight_start_raw_replays": smoothed_local_analysis["coverage"][
                "eight_start_raw_replays"
            ],
            "stable_root_complete_replays": smoothed_local_analysis[
                "coverage"
            ]["stable_root_complete_replays"],
        },
        "smoothed_local_endpoint_audits": {
            "analysis_report_sha256": sha256(endpoint_analysis_path),
            "full_variants": endpoint_analysis["coverage"][
                "universal_refits"
            ],
            "eight_start_raw_replays": endpoint_analysis["coverage"][
                "eight_start_raw_replays"
            ],
            "exact_endpoint_independent_refits": endpoint_analysis[
                "coverage"
            ]["exact_endpoint_independent_refits"],
        },
        "endpoint_power_memory_audits": {
            "analysis_report_sha256": sha256(power_memory_analysis_path),
            "fixed_parameter_rows": power_memory_analysis["coverage"][
                "fixed_parameter_audit_rows"
            ],
            "full_variants": power_memory_analysis["coverage"][
                "universal_refits"
            ],
            "eight_start_raw_replays": power_memory_analysis["coverage"][
                "eight_start_raw_replays"
            ],
            "stable_root_complete_replays": power_memory_analysis["coverage"][
                "stable_root_complete_rows"
            ],
        },
        "endpoint_boundary_refinement_audits": {
            "analysis_report_sha256": sha256(boundary_analysis_path),
            "fixed_parameter_rows": boundary_analysis["coverage"][
                "fixed_parameter_refinement_rows"
            ],
            "full_variants": boundary_analysis["coverage"][
                "universal_refits"
            ],
            "eight_start_raw_replays": boundary_analysis["coverage"][
                "eight_start_raw_replays"
            ],
            "stable_root_complete_replays": boundary_analysis["coverage"][
                "stable_root_complete_rows"
            ],
            "exact_repeat_refits": boundary_analysis["coverage"][
                "exact_repeat_refits"
            ],
        },
        "endpoint_high_q_ridge_audits": {
            "analysis_report_sha256": sha256(high_q_analysis_path),
            "fixed_parameter_rows": high_q_analysis["coverage"][
                "fixed_parameter_rows"
            ],
            "full_variants": high_q_analysis["coverage"][
                "universal_refits"
            ],
            "eight_start_raw_replays": high_q_analysis["coverage"][
                "eight_start_raw_replays"
            ],
            "stable_root_complete_replays": high_q_analysis["coverage"][
                "stable_root_complete_rows"
            ],
            "exact_repeat_refits": high_q_analysis["coverage"][
                "exact_repeat_refits"
            ],
        },
        "endpoint_interaction_factorial_audits": {
            "analysis_report_sha256": sha256(factorial_analysis_path),
            "factorial_cells": factorial_analysis["coverage"][
                "factorial_cells"
            ],
            "full_variants": factorial_analysis["coverage"][
                "universal_refits"
            ],
            "eight_start_raw_replays": factorial_analysis["coverage"][
                "eight_start_raw_replays"
            ],
            "stable_root_complete_replays": factorial_analysis["coverage"][
                "stable_root_complete_rows"
            ],
            "exact_repeat_refits": factorial_analysis["coverage"][
                "exact_repeat_refits"
            ],
        },
        "tidal_gate_memory_audits": {
            "analysis_report_sha256": sha256(tidal_gate_memory_analysis_path),
            "full_variants": tidal_gate_memory_analysis["coverage"][
                "universal_refits"
            ],
            "eight_start_raw_replays": tidal_gate_memory_analysis["coverage"][
                "eight_start_raw_replays"
            ],
            "stable_root_complete_replays": tidal_gate_memory_analysis[
                "coverage"
            ]["stable_root_complete_replays"],
        },
        "tidal_gate_topology_audits": {
            "analysis_report_sha256": sha256(tidal_gate_topology_analysis_path),
            "full_variants": tidal_gate_topology_analysis["coverage"][
                "universal_refits"
            ],
            "nonmonotonic_factorial_cells": tidal_gate_topology_analysis[
                "coverage"
            ]["nonmonotonic_factorial_cells"],
            "eight_start_raw_replays": tidal_gate_topology_analysis[
                "coverage"
            ]["eight_start_raw_replays"],
            "stable_root_complete_replays": tidal_gate_topology_analysis[
                "coverage"
            ]["stable_root_complete_replays"],
        },
        "tidal_memory_placement_audits": {
            "analysis_report_sha256": sha256(
                tidal_memory_placement_analysis_path
            ),
            "full_variants": tidal_memory_placement_analysis["coverage"][
                "universal_refits"
            ],
            "both_placement_factorial_cells": tidal_memory_placement_analysis[
                "coverage"
            ]["both_placement_factorial_cells"],
            "eight_start_raw_replays": tidal_memory_placement_analysis[
                "coverage"
            ]["eight_start_raw_replays"],
            "stable_root_complete_replays": tidal_memory_placement_analysis[
                "coverage"
            ]["stable_root_complete_replays"],
        },
        "profile_diffusion_audits": {
            "analysis_report_sha256": sha256(profile_diffusion_analysis_path),
            "full_variants": profile_diffusion_analysis["coverage"]["rows"],
            "diffusion_factorial_cells": profile_diffusion_analysis["coverage"][
                "diffusion_factorial_rows"
            ],
            "memory_plus_diffusion_cells": profile_diffusion_analysis[
                "coverage"
            ]["memory_plus_diffusion_rows"],
            "eight_start_raw_replays": profile_diffusion_analysis["coverage"][
                "rows"
            ],
            "stable_root_complete_replays": profile_diffusion_analysis[
                "coverage"
            ]["eight_start_complete_root_rows"],
            "universal_parameter_boundary_rows": profile_diffusion_analysis[
                "coverage"
            ]["universal_parameter_boundary_rows"],
        },
            "tidal_closure_control": {
                "identical_formula_variants": len(shared_closure_variants),
                "mixed_middle_ratio_system_AUC": tidal_by_name[
                    "tidal_middle_to_max"
                ]["separation_auc"],
                "common_spherical_middle_ratio_system_AUC": (
                    common_tidal_by_name["tidal_middle_to_max"][
                        "separation_auc"
                    ]
                ),
                "mixed_best_compromise_SPARC_RMSE_km_s": float(
                    tidal_middle_reversed.SPARC_outer_RMSE_km_s
                ),
                "common_spherical_same_formula_SPARC_RMSE_km_s": float(
                    tidal_middle_reversed_common.SPARC_outer_RMSE_km_s
                ),
                "same_formula_raw_RMS_arcsec": float(
                    tidal_middle_reversed_common.raw_lensing_RMS_arcsec
                ),
                "maximum_bridge_RMSE_delta_over_control": (
                    maximum_closure_bridge_delta
                ),
                "maximum_raw_RMS_delta_over_control": maximum_closure_raw_delta,
            },
            "common_spherical_tidal_identity": {
                "verified_points": spherical_identity["points"],
                "verified_systems": spherical_identity["systems"],
                "maximum_numerical_error": spherical_identity[
                    "maximum_error_over_all_invariants"
                ],
                "normalized_eigenvalue_identity": spherical_identity[
                    "identity"
                ]["normalized_tidal_eigenvalues"],
                "implication": spherical_identity["implication"],
            },
            "adaptive_common_spherical_reparameterization": {
                "best_compromise_variant": common_adaptive_compromise.variant,
                "best_compromise_SPARC_RMSE_km_s": float(
                    common_adaptive_compromise.SPARC_outer_RMSE_km_s
                ),
                "best_compromise_raw_eight_start_RMS_arcsec": float(
                    common_adaptive_compromise.raw_eight_start_RMS_arcsec
                ),
                "best_compromise_eight_start_all_roots": bool(
                    common_adaptive_compromise.raw_eight_start_all_roots_converged
                ),
                "best_raw_variant": common_adaptive_raw.variant,
                "best_raw_SPARC_RMSE_km_s": float(
                    common_adaptive_raw.SPARC_outer_RMSE_km_s
                ),
                "best_raw_eight_start_RMS_arcsec": float(
                    common_adaptive_raw.raw_eight_start_RMS_arcsec
                ),
                "best_raw_eight_start_all_roots": bool(
                    common_adaptive_raw.raw_eight_start_all_roots_converged
                ),
                "best_stable_raw_variant": common_adaptive_stable_raw.variant,
                "best_stable_raw_SPARC_RMSE_km_s": float(
                    common_adaptive_stable_raw.SPARC_outer_RMSE_km_s
                ),
                "best_stable_raw_eight_start_RMS_arcsec": float(
                    common_adaptive_stable_raw.raw_eight_start_RMS_arcsec
                ),
                "best_stable_raw_eight_start_all_roots": bool(
                    common_adaptive_stable_raw.raw_eight_start_all_roots_converged
                ),
                "best_stable_raw_pooled_reduced_chi2": float(
                    common_adaptive_stable_raw.raw_eight_start_pooled_reduced_chi2
                ),
                "best_galaxy_incomplete_variant": (
                    common_adaptive_galaxy_incomplete.variant
                ),
                "best_galaxy_incomplete_SPARC_RMSE_km_s": float(
                    common_adaptive_galaxy_incomplete.SPARC_outer_RMSE_km_s
                ),
                "best_galaxy_incomplete_raw_eight_start_RMS_arcsec": float(
                    common_adaptive_galaxy_incomplete.raw_eight_start_RMS_arcsec
                ),
                "best_galaxy_incomplete_eight_start_all_roots": bool(
                    common_adaptive_galaxy_incomplete.raw_eight_start_all_roots_converged
                ),
            },
            "radial_memory_profile_history": {
                "definition": (
                    "exponential running memory of excess force in log radius"
                ),
                "dual_local_SPARC_RMSE_km_s": float(
                    radial_dual_local.SPARC_outer_RMSE_km_s
                ),
                "dual_full_memory_SPARC_RMSE_km_s": float(
                    radial_dual_full.SPARC_outer_RMSE_km_s
                ),
                "dual_scale_8_SPARC_RMSE_km_s": float(
                    radial_dual_scale_8.SPARC_outer_RMSE_km_s
                ),
                "dual_scale_8_bridge_RMSE_dex": float(
                    radial_dual_scale_8.bridge_RMSE_dex
                ),
                "dual_short_scale_SPARC_RMSE_km_s": float(
                    radial_dual_short.SPARC_outer_RMSE_km_s
                ),
                "dual_short_scale_raw_eight_start_RMS_arcsec": float(
                    radial_dual_short.raw_eight_start_RMS_arcsec
                ),
                "dual_short_scale_all_roots": bool(
                    radial_dual_short.raw_eight_start_all_roots_converged
                ),
                "dual_scale_2_SPARC_RMSE_km_s": float(
                    radial_dual_scale_2.SPARC_outer_RMSE_km_s
                ),
                "dual_scale_2_raw_eight_start_RMS_arcsec": float(
                    radial_dual_scale_2.raw_eight_start_RMS_arcsec
                ),
                "dual_scale_2_all_roots": bool(
                    radial_dual_scale_2.raw_eight_start_all_roots_converged
                ),
                "dual_scale_2_cross_domain_reference_ratio": float(
                    max(
                        radial_dual_scale_2.SPARC_RMSE_ratio_to_RAR,
                        radial_dual_scale_2.raw_eight_start_RMS_arcsec
                        / references["raw_compact_halo_RMS_arcsec"],
                    )
                ),
                "reverse_direction_SPARC_RMSE_km_s": float(
                    radial_dual_reverse.SPARC_outer_RMSE_km_s
                ),
                "reverse_direction_raw_eight_start_RMS_arcsec": float(
                    radial_dual_reverse.raw_eight_start_RMS_arcsec
                ),
                "post_screen_SPARC_RMSE_km_s": float(
                    radial_dual_post_screen.SPARC_outer_RMSE_km_s
                ),
                "post_screen_raw_eight_start_RMS_arcsec": float(
                    radial_dual_post_screen.raw_eight_start_RMS_arcsec
                ),
                "post_screen_all_roots": bool(
                    radial_dual_post_screen.raw_eight_start_all_roots_converged
                ),
                "pre_screen_SPARC_RMSE_km_s": float(
                    radial_dual_pre_screen.SPARC_outer_RMSE_km_s
                ),
                "unsaturated_local_SPARC_RMSE_km_s": float(
                    radial_unsaturated_local.SPARC_outer_RMSE_km_s
                ),
                "unsaturated_full_memory_SPARC_RMSE_km_s": float(
                    radial_unsaturated_full.SPARC_outer_RMSE_km_s
                ),
                "unsaturated_full_memory_raw_eight_start_RMS_arcsec": float(
                    radial_unsaturated_full.raw_eight_start_RMS_arcsec
                ),
            },
            "memory_carrier_effective_radial_power": {
                "transported_quantity": memory_carrier_analysis["formula"][
                    "transported_quantity"
                ],
                "effective_radial_power": memory_carrier_analysis["formula"][
                    "effective_radial_power"
                ],
                "SPARC_median_dln_gbar_dln_r": memory_carrier_analysis[
                    "measured_profile_slopes"
                ]["SPARC_median_dln_gbar_dln_r"],
                "CLASH_median_dln_gbar_dln_r": memory_carrier_analysis[
                    "measured_profile_slopes"
                ]["CLASH_median_dln_gbar_dln_r"],
                "SPARC_effective_power_vs_SPARC_RMSE_spearman": (
                    memory_carrier_analysis[
                        "transport_power_surface_spearman"
                    ]["SPARC_effective_power_vs_SPARC_RMSE"]
                ),
                "CLASH_effective_power_vs_bridge_RMSE_spearman": (
                    memory_carrier_analysis[
                        "transport_power_surface_spearman"
                    ]["CLASH_effective_power_vs_bridge_RMSE"]
                ),
                "best_stable": memory_carrier_analysis[
                    "best_eight_start_stable_compromise"
                ],
                "slope_neutral_balanced_candidate": (
                    memory_carrier_analysis[
                        "slope_neutral_balanced_candidate"
                    ]
                ),
                "solar_failure_count": len(
                    memory_carrier_analysis["solar_failures"]
                ),
                "lens_root_reversal_count": len(
                    memory_carrier_analysis["lens_root_reversals"]
                ),
            },
            "local_slope_adaptive_carrier": {
                "formula": slope_adaptive_analysis["formula"],
                "best_stable_stage_compromise": slope_adaptive_analysis[
                    "best_stable_stage_compromise"
                ],
                "best_stable_raw_stage_case": slope_adaptive_analysis[
                    "best_stable_raw_stage_case"
                ],
                "geometry_mechanism": slope_adaptive_analysis[
                    "geometry_mechanism"
                ],
                "failure_mode_correlations": slope_adaptive_analysis[
                    "failure_mode_correlations"
                ],
                "duplicate_setting_groups": slope_adaptive_analysis[
                    "duplicate_setting_groups"
                ],
                "lens_root_reversal_count": len(
                    slope_adaptive_analysis["lens_root_reversals"]
                ),
                "solar_failure_count": len(
                    slope_adaptive_analysis["solar_failures"]
                ),
            },
            "derivative_safe_slope_responses": {
                "formula": slope_response_analysis["formula"],
                "fixed_parameter_mode_comparison": slope_response_analysis[
                    "mode_comparison_fixed_parameters"
                ],
                "pointwise_exponent_failure_removed": slope_response_analysis[
                    "pointwise_exponent_failure_removed"
                ],
                "best_stable_derivative_safe_compromise": (
                    slope_response_analysis[
                        "best_stable_derivative_safe_compromise"
                    ]
                ),
                "best_stable_raw_stage_case": slope_response_analysis[
                    "best_stable_raw_stage_case"
                ],
                "radial_range_sensitivity": slope_response_analysis[
                    "radial_range_sensitivity"
                ],
                "lens_root_reversal_count": len(
                    slope_response_analysis["lens_root_reversals"]
                ),
                "solar_failure_count": len(
                    slope_response_analysis["solar_failures"]
                ),
            },
            "smoothed_local_slope_response": {
                "formula": smoothed_local_analysis["formula"],
                "best_stable_compromise": smoothed_local_analysis[
                    "best_stable_compromise"
                ],
                "best_stable_raw_case": smoothed_local_analysis[
                    "best_stable_raw_case"
                ],
                "parameter_impact_ranking": smoothed_local_analysis[
                    "parameter_impact_ranking"
                ],
                "root_reversal_count": len(
                    smoothed_local_analysis["root_reversals"]
                ),
            },
            "finite_pivot_versus_exact_endpoint": {
                "formula_tested": endpoint_analysis["formula_tested"],
                "best_observed_stable_compromise": endpoint_analysis[
                    "best_stable_compromise"
                ],
                "best_stable_raw_case": endpoint_analysis[
                    "best_stable_raw_case"
                ],
                "exact_endpoint_repeatability": endpoint_analysis[
                    "exact_endpoint_repeatability"
                ],
                "pivot6_vs_exact_endpoint_median": endpoint_analysis[
                    "pivot6_vs_exact_endpoint_median"
                ],
                "root_reversal_count": len(endpoint_analysis["root_reversals"]),
            },
            "endpoint_source_power_and_memory": {
                "formula": power_memory_analysis["formula"],
                "best_stable_observed": power_memory_analysis[
                    "best_stable_observed"
                ],
                "best_stable_raw_case": power_memory_analysis[
                    "best_stable_raw_case"
                ],
                "coordinate_impact_ranking": power_memory_analysis[
                    "coordinate_impact_ranking"
                ],
                "root_reversal_count": len(
                    power_memory_analysis["root_reversals"]
                ),
            },
            "bracketed_endpoint_boundary": {
                "formula": boundary_analysis["formula"],
                "best_stable_observed": boundary_analysis[
                    "best_stable_observed"
                ],
                "best_stable_raw_case": boundary_analysis[
                    "best_stable_raw_case"
                ],
                "exact_formula_repeatability": boundary_analysis[
                    "exact_formula_repeatability"
                ],
                "surface_minima_by_q": boundary_analysis[
                    "surface_minima_by_q"
                ],
                "root_reversal_count": len(boundary_analysis["root_reversals"]),
            },
            "high_q_endpoint_ridge": {
                "formula": high_q_analysis["formula"],
                "best_stable_observed": high_q_analysis[
                    "best_stable_observed"
                ],
                "best_stable_raw_case": high_q_analysis[
                    "best_stable_raw_case"
                ],
                "ridge_minima_by_q": high_q_analysis["ridge_minima_by_q"],
                "matched_effective_power_paths": high_q_analysis[
                    "matched_effective_power_paths"
                ],
                "memory_response": high_q_analysis["memory_response"],
                "exact_formula_repeatability": high_q_analysis[
                    "exact_formula_repeatability"
                ],
                "root_reversal_count": len(high_q_analysis["root_reversals"]),
            },
            "endpoint_exponent_memory_interactions": {
                "formula": factorial_analysis["formula"],
                "factor_levels": factorial_analysis["factor_levels"],
                "best_stable_observed": factorial_analysis[
                    "best_stable_observed"
                ],
                "best_stable_raw_case": factorial_analysis[
                    "best_stable_raw_case"
                ],
                "balanced_effect_decompositions": factorial_analysis[
                    "balanced_effect_decompositions"
                ],
                "root_completion": factorial_analysis["root_completion"],
                "near_optimal_plateau": factorial_analysis[
                    "near_optimal_plateau"
                ],
                "exact_formula_repeatability": factorial_analysis[
                    "exact_formula_repeatability"
                ],
                "root_reversal_count": len(
                    factorial_analysis["root_reversals"]
                ),
            },
            "tidal_gate_memory": {
                "formula": tidal_gate_memory_analysis["formula"],
                "best_stable_observed": tidal_gate_memory_analysis[
                    "best_stable_observed"
                ],
                "best_stable_raw_case": tidal_gate_memory_analysis[
                    "best_stable_raw_case"
                ],
                "inner_to_outer_balanced_effects": tidal_gate_memory_analysis[
                    "inner_to_outer_balanced_effects"
                ],
                "direction_balanced_effects_at_log_scale_0p35": (
                    tidal_gate_memory_analysis[
                        "direction_balanced_effects_at_log_scale_0p35"
                    ]
                ),
                "exact_local_repeatability": tidal_gate_memory_analysis[
                    "exact_local_repeatability"
                ],
                "root_reversal_count": len(
                    tidal_gate_memory_analysis["root_reversals"]
                ),
            },
            "tidal_gate_topology": {
                "formula": tidal_gate_topology_analysis["formula"],
                "best_stable_observed": tidal_gate_topology_analysis[
                    "best_stable_observed"
                ],
                "best_stable_raw_case": tidal_gate_topology_analysis[
                    "best_stable_raw_case"
                ],
                "balanced_effects": tidal_gate_topology_analysis[
                    "balanced_effects"
                ],
                "exact_complement_pairs": tidal_gate_topology_analysis[
                    "exact_complement_pairs"
                ],
                "reconciliation_count": tidal_gate_topology_analysis[
                    "reconciliation_count"
                ],
                "topology_parameter_summaries": tidal_gate_topology_analysis[
                    "topology_parameter_summaries"
                ],
                "root_reversal_count": len(
                    tidal_gate_topology_analysis["root_reversals"]
                ),
            },
            "tidal_memory_placement": {
                "formula": tidal_memory_placement_analysis["formula"],
                "best_stable_observed": tidal_memory_placement_analysis[
                    "best_stable_observed"
                ],
                "best_stable_raw_case": tidal_memory_placement_analysis[
                    "best_stable_raw_case"
                ],
                "balanced_effects": tidal_memory_placement_analysis[
                    "balanced_effects"
                ],
                "matched_control_contrasts": tidal_memory_placement_analysis[
                    "matched_control_contrasts"
                ],
                "rows_improving_both_global_control_domains": (
                    tidal_memory_placement_analysis[
                        "rows_improving_both_global_control_domains"
                    ]
                ),
                "rows_beating_prior_global_ratio": (
                    tidal_memory_placement_analysis[
                        "rows_beating_prior_global_ratio"
                    ]
                ),
                "rows_meeting_both_external_references": (
                    tidal_memory_placement_analysis[
                        "rows_meeting_both_external_references"
                    ]
                ),
                "placement_parameter_summaries": (
                    tidal_memory_placement_analysis[
                        "placement_parameter_summaries"
                    ]
                ),
                "root_reversal_count": len(
                    tidal_memory_placement_analysis["root_reversals"]
                ),
            },
            "profile_diffusion": {
                "controls": profile_diffusion_analysis["controls"],
                "best_complete_cross_domain": profile_diffusion_analysis[
                    "best_complete_cross_domain"
                ],
                "best_complete_raw_diffusion": profile_diffusion_analysis[
                    "best_complete_raw_diffusion"
                ],
                "smallest_galaxy_cost_raw_improvement": (
                    profile_diffusion_analysis[
                        "smallest_galaxy_cost_raw_improvement"
                    ]
                ),
                "best_memory_plus_diffusion_galaxy_row": (
                    profile_diffusion_analysis[
                        "best_memory_plus_diffusion_galaxy_row"
                    ]
                ),
                "diffusion_factorial_effects": profile_diffusion_analysis[
                    "diffusion_factorial_effects"
                ],
                "memory_plus_diffusion_effects": profile_diffusion_analysis[
                    "memory_plus_diffusion_effects"
                ],
                "conclusions": profile_diffusion_analysis["conclusions"],
            },
        },
        "best_eight_start_verified_raw_case": {
            "variant": best_verified_raw.variant,
            "stage": best_verified_raw.stage,
            "SPARC_outer_RMSE_km_s": float(
                best_verified_raw.SPARC_outer_RMSE_km_s
            ),
            "bridge_RMSE_dex": float(best_verified_raw.bridge_RMSE_dex),
            "raw_eight_start_RMS_arcsec": float(
                best_verified_raw.raw_eight_start_RMS_arcsec
            ),
            "raw_eight_start_all_roots_converged": bool(
                best_verified_raw.raw_eight_start_all_roots_converged
            ),
            "raw_eight_start_pooled_reduced_chi2": float(
                best_verified_raw.raw_eight_start_pooled_reduced_chi2
            ),
            "Mercury_precession_mas_per_century": float(
                best_verified_raw.Mercury_precession_mas_per_century
            ),
        },
        "best_eight_start_verified_cross_domain_compromise": {
            "variant": best_verified_compromise.variant,
            "stage": best_verified_compromise.stage,
            "SPARC_outer_RMSE_km_s": float(
                best_verified_compromise.SPARC_outer_RMSE_km_s
            ),
            "bridge_RMSE_dex": float(
                best_verified_compromise.bridge_RMSE_dex
            ),
            "raw_eight_start_RMS_arcsec": float(
                best_verified_compromise.raw_eight_start_RMS_arcsec
            ),
            "cross_domain_reference_ratio": float(
                best_verified_compromise.eight_start_cross_domain_ratio
            ),
        },
        "claim_boundary": [
            "This table combines exploratory stages selected sequentially after inspecting earlier stages.",
            "Repeated settings with different optimization seeds are retained as a reproducibility diagnostic.",
            "Two-start raw scores with incomplete roots are never treated as successful lensing fits.",
            f"Only {robustness_rows} selected rows received an eight-start raw-geometry replay.",
            "Directional stages record either the mixed axisymmetric-SPARC/spherical-cluster closure or the explicit common-spherical control.",
            "Under the common-spherical closure, all tested scale-free tidal invariants reduce exactly to local-to-mean density ratio.",
            "Radial memory is a phenomenological profile-history test, not a derived claim that gravity literally propagates along radius.",
            "Generalized memory carriers are phenomenological source-weighting tests; a failed p,q,channel setting does not reject nonlocal or history-dependent gravity as a parent idea.",
            "The local-slope stage tests pointwise exponent interpolation only; its failure does not reject profile-level, bounded-output, or independently smoothed slope responses.",
            "The derivative-safe slope-response stage tests profile-constant and bounded completed-response blends; the global profile slope remains sensitive to the raw-lensing extrapolation range.",
            "The smoothed-local stage and its exact-endpoint control show that a saturated slope gate is empirically interchangeable with selecting the endpoint response in the tested formula; this does not reject other slope-dependent mechanisms.",
            "Exact endpoint duplicates retain bridge-parameter non-identification, so the typical repeated prediction is more defensible than the single best optimizer branch.",
            "The endpoint source-power response is a broad p,q ridge. Only explicit exponent pairs and ranges are disfavored; the parent nonlocal or history mechanism is not rejected.",
            "The short-memory refinement brackets memory length, while the later high-q stage shows that a fixed-p exponent turnover is not a bracket on the joint p,q ridge.",
            "The high-q stage corrects the fixed-p turnover by following the moving p,q ridge, then brackets that ridge using matched effective-power stress tests before full refitting.",
            "The local endpoint factorial measures q/effective-power, memory-length, and memory-strength interactions over a complete balanced grid; its variance fractions apply only to that sampled neighborhood.",
            "Tidal-gate memory remembers a bounded channel classification rather than force; it is a mixed-closure proxy, not a common two-dimensional or three-dimensional tensor-map calculation.",
            "The tidal-gate-memory factorial shows that gate orientation dominates the sampled response; under spherical closure its cluster-side coordinate remains a density-ratio reparameterization.",
            "The nonmonotonic tidal-gate stage tests exact complementary middle-band and two-tail placements; a failure applies to those frozen threshold and sharpness ranges, not every nonmonotonic field response.",
            "The tidal memory-placement stage separates gating a channel ceiling from gating radial-memory strength; its variance fractions apply only to the frozen endpoint carrier and mixed tidal closure.",
            "The profile-diffusion stage conserves the transported carrier integral on each measured radial profile; its failure applies to symmetric no-flux radial redistribution, not a registered two-dimensional or three-dimensional tensor field.",
            "Forty-six of 64 bridge systems contain one radial point and therefore cannot constrain profile memory internally.",
            "The experiment measures parameter leverage; it does not validate a relativistic theory.",
        ],
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT / "combined_scores.csv", index=False)
    (OUTPUT / "program_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Reopened hybrid program summary",
        "",
        f"- Scored rows: **{len(combined)}**",
        f"- Unique formula settings: **{unique_settings}**",
        f"- Unique formula/evaluation contexts: **{unique_evaluations}**",
        "- Every row: 64 BCG+CLASH bridge systems, 131 SPARC galaxies, "
        "four raw strong-lensing clusters, and Solar-System proxies",
        f"- Eight-start raw robustness replays: **{robustness_rows}**",
        "",
        "The strongest repeatable effects are:",
        "",
        "1. The acceleration-screen exponent controls Solar survival.",
        "2. Saturation controls galaxy overprediction but removes cluster response.",
        "3. Interaction strength is almost fully absorbed by the fitted Sigma amplitude.",
        "4. A positive acceleration-moving density threshold moves galaxies in the "
        "desired direction, but the effect is modest and becomes degenerate under saturation.",
        "5. Sigma-only saturation modestly improves galaxies without moving raw lensing; "
        "RG-only saturation improves raw lensing while worsening galaxies.",
        "6. The first stable RG ceiling is a lens-root boundary, not a smooth joint optimum.",
        "7. High galaxy/cluster classification AUC does not predict a useful gravity gate.",
        "8. Cluster-side RG gating is a stable lensing lever; galaxy-side RG gating "
        "improves galaxies but consistently loses held-out lens roots.",
        "9. Sigma gate placement is negligible after the bridge refit drives its amplitude down.",
        "10. Tidal-shape gates have strong leverage, but their galaxy and lensing gains "
        "still occupy opposing orientations of the same selector.",
        "11. Forcing one spherical closure worsens the best tidal compromise and proves "
        "that every scale-free spherical tidal invariant is only a density-ratio reparameterization.",
        "12. Nonlinear spherical reparameterizations can approach halo-like lensing but "
        "do not recover the galaxy fit; the best attractive raw setting also loses a root.",
        "13. Inner-to-outer radial memory improves galaxy transfer monotonically over the "
        "useful range, while reverse memory worsens it; screen placement and memory scale "
        "strongly control whether raw lens roots survive.",
        "14. A two-log-radius memory scale is the new best complete-root cross-domain "
        "compromise within the local fractional-memory stage.",
        "15. The quantity transported is a higher-leverage knob than memory amplitude: "
        "the effective power q+p*s orders both domains, but in opposing useful directions.",
        "16. A low-acceleration, outer-weighted carrier is the new best stable compromise "
        "at 4.11 times the worse reference error, a 24.3% gain over the preceding result.",
        "17. Neutralizing the carrier for the measured CLASH slope gives a more balanced "
        "alternative: it improves raw lensing over baryons and preserves most of the galaxy gain.",
        "18. Smooth local-slope interpolation behaves mostly like another fixed carrier and "
        "does not beat the prior 4.11 cross-domain ratio.",
        "19. Hard pointwise exponent switching is a strong anti-galaxy/lensing lever: it "
        "improves stable raw lensing to 22.89 arcsec but raises galaxy error to 319.90 km/s.",
        "20. Exact formula duplicates expose bridge-fit non-identifiability: nearly identical "
        "bridge scores can hide large universal-parameter and lens-branch differences.",
        "21. Universally smoothing the local profile slope removes extrapolation dependence, "
        "but the best pivot runs to an always-on endpoint rather than identifying a slope transition.",
        "22. Five exact endpoint refits reproduce the saturated-pivot prediction; the typical "
        "repeat scores about 3.93 times the worse reference, so the slope coordinate is not required.",
        "23. Extending the endpoint source to X=F (g_N/g_ref)^p (r/kpc)^q identifies a "
        "cross-domain effective-power coordinate; q is the largest raw-lensing lever while "
        "memory strength is the largest galaxy lever.",
        "24. The first short-memory/higher-power refinement improved the stable cross-domain "
        "ratio to about 3.58 while leaving raw lensing near baryons/MOND.",
        "25. The apparent fixed-p turnover near q=5 is a coordinate artifact; following p "
        "with q moves the bracket toward q=9--10 and reveals strong coupling to memory length.",
        "26. Matched galaxy- and cluster-effective-power paths test whether the exponents "
        "supply a real differential lever: they preserve their matched domain closely while "
        "moving the other domain more strongly.",
        "27. Refitting high-q memory length improves the best stable ratio again to about "
        "3.49; memory strength remains the largest galaxy/raw-lensing tradeoff lever.",
        "28. A balanced local factorial separates exponent, memory-length, and memory-strength "
        "main effects from their two-way interactions instead of attributing a coupled ridge to one knob.",
        "29. No tested formula approaches both the fixed-RAR galaxy error and compact-halo "
        "raw-lensing error with one universal setting.",
        "30. Remembering the bounded tidal channel gate does not improve the global "
        "compromise: orientation explains 80--99.8% of sampled variation, the best "
        "stable gate-memory ratio is 6.44, and the best raw branch keeps memory off.",
        "31. Exact complementary middle-band and two-tail gates do not combine the "
        "galaxy-favored and lensing-favored orientations: zero of 16 nonmonotonic "
        "cells beats both matched monotonic endpoints, so monotonic ordering is not "
        "the sole source of the cross-domain conflict.",
        "32. Gating radial-memory strength does not improve the global compromise. "
        "Zero of 27 placement rows improves both galaxy and robust raw-lensing error "
        "over global full memory, while cap orientation remains the dominant galaxy "
        "and bridge control and memory strength is the largest raw-lensing main effect.",
    ]
    (OUTPUT / "SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["coverage"], indent=2))


if __name__ == "__main__":
    main()
