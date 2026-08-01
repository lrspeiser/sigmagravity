#!/usr/bin/env python3
"""Build a machine-readable atlas of parameter leverage and transfer outcomes."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


FAMILY = {
    "target_weight_power": "destination_weighting",
    "top_k": "destination_topology",
    "external_direction_mix": "destination_topology",
    "carrier": "destination_topology",
    "hop_fraction": "path_length",
    "link_scale_kpc": "path_length",
    "length_over_R80": "path_length",
    "return_length_over_R80": "path_length",
    "softening_kpc": "spatial_width",
    "arrival_smoothing_kpc": "spatial_width",
    "arrival_width": "spatial_width",
    "width_over_R80": "spatial_width",
    "smoothing_r80_fraction": "spatial_width",
    "aperture_r80_fraction": "spatial_width",
    "distance_power": "distance_falloff",
    "radius_exponent": "distance_falloff",
    "route_fraction_f": "routed_fraction",
    "route_fraction": "routed_fraction",
    "route_fraction_multiplier": "routed_fraction",
    "fraction_max": "routed_fraction",
    "f": "routed_fraction",
    "gate_mode": "environmental_gate",
    "shape_gate": "environmental_gate",
    "gate_power": "environmental_gate",
    "gate_sharpness_n": "environmental_gate",
    "Q0": "environmental_gate",
    "tidal_power": "environmental_gate",
    "cancellation_power": "environmental_gate",
    "route_mode": "residence_rule",
    "contrast_cap": "response_saturation",
    "nominal_cap": "response_saturation",
    "contrast_mode": "saturation_shape",
    "anisotropy_tau": "anisotropy_strength",
    "tensor_axis_ratio": "anisotropy_shape",
    "tensor_orientation": "anisotropy_orientation",
    "mode": "operator_shape",
    "minimum_permittivity": "constitutive_strength",
    "a0_m_s2": "acceleration_scale",
    "removal_fraction": "affine_removal",
}


OUTCOME = {
    "P0554": "failed_RXJ1347_network_holdout",
    "P0572": "selected_formula_failed_heldout_maps",
    "P0574": "passed_fresh_normalized_maps_then_failed_raw_positions",
    "P0579_calibration": "calibration_choice_failed_heldout_and_mass_sheet_gate",
    "P0579_posthoc": "posthoc_only_and_mass_sheet_degenerate",
    "P0580": "mass_conserving_return_far_worse_than_RAR",
    "P0581": "topology_sensitive_then_failed_RXJ2129",
    "P0582": "saturation_restored_roots_then_failed_RXJ2129",
    "P0586": "selection_gain_reversed_on_exact_validation",
    "P0586C": "common_small_gain_but_cluster_preferred_signs_conflict",
    "P0587": "small_exact_gain_not_compact_halo_competitive",
    "P0603": "normalized_map_sensitivity_not_raw_validation",
    "P0604": "cross_domain_sensitivity_not_formula_success",
    "P0606": "zero_route_selected_and_many_nonzero_variants_lost_roots",
}


def read_protocol() -> dict:
    path = ROOT / "configs/p0612_cross_stage_parameter_impact_protocol.json"
    return json.loads(path.read_text(encoding="utf-8"))


def add_rows(
    target: list[dict],
    frame: pd.DataFrame,
    *,
    stage: str,
    domain: str,
    observable: str,
    metric: str,
    metric_type: str,
    coordinate: str,
    span: str,
    unit: str,
    best: str | None = None,
    worst: str | None = None,
    evidence: str,
) -> None:
    for row in frame.itertuples(index=False):
        name = str(getattr(row, coordinate))
        value = float(getattr(row, span))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"invalid impact span for {stage}:{name}")
        target.append(
            {
                "stage_id": stage,
                "domain": domain,
                "observable": observable,
                "metric": metric,
                "metric_type": metric_type,
                "coordinate": name,
                "coordinate_family": FAMILY.get(name, name),
                "raw_span": value,
                "raw_unit": unit,
                "best_level": None if best is None else str(getattr(row, best)),
                "worst_level": None if worst is None else str(getattr(row, worst)),
                "evidence_level": evidence,
                "transfer_outcome": OUTCOME[stage],
            }
        )


def main() -> None:
    protocol = read_protocol()
    inputs = {key: ROOT / value for key, value in protocol["inputs"].items()}
    missing = [str(path) for path in inputs.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing P0612 inputs:\n" + "\n".join(missing))
    rows: list[dict] = []

    add_rows(
        rows,
        pd.read_csv(inputs["P0554_network"]),
        stage="P0554",
        domain="cluster_raw_lensing",
        observable="profiled_multiple_image_positions",
        metric="profile_RMS_span",
        metric_type="performance",
        coordinate="coordinate",
        span="profile_RMS_span_arcsec",
        unit="arcsec",
        best="best_value",
        evidence="spent_multicluster_screen_plus_formula_holdout",
    )
    add_rows(
        rows,
        pd.read_csv(inputs["P0572_arrival"]),
        stage="P0572",
        domain="cluster_reconstructed_lensing",
        observable="normalized_convergence_maps",
        metric="mean_JS_span",
        metric_type="performance",
        coordinate="coordinate",
        span="main_effect_span_JS",
        unit="JS",
        best="minimum_level",
        evidence="development_factorial",
    )
    p0574 = json.loads(inputs["P0574_symmetry"].read_text(encoding="utf-8"))
    add_rows(
        rows,
        pd.DataFrame(p0574["parameter_impacts"]),
        stage="P0574",
        domain="cluster_reconstructed_lensing",
        observable="fresh_normalized_convergence_maps",
        metric="mean_JS_span",
        metric_type="performance",
        coordinate="coordinate",
        span="JS_span",
        unit="JS",
        best="best_candidate",
        evidence="fresh_three_cluster_formula_transfer",
    )
    p0579 = pd.read_csv(inputs["P0579_raw_return"])
    add_rows(
        rows,
        p0579,
        stage="P0579_calibration",
        domain="cluster_raw_lensing",
        observable="linearized_multiple_image_positions",
        metric="calibration_RMS_span",
        metric_type="performance",
        coordinate="parameter",
        span="calibration_RMS_span_arcsec",
        unit="arcsec",
        best="calibration_best_level",
        worst="calibration_worst_level",
        evidence="two_cluster_calibration",
    )
    add_rows(
        rows,
        p0579,
        stage="P0579_posthoc",
        domain="cluster_raw_lensing",
        observable="heldout_multiple_image_positions",
        metric="posthoc_heldout_RMS_span",
        metric_type="performance",
        coordinate="parameter",
        span="heldout_RMS_span_arcsec_posthoc",
        unit="arcsec",
        best="heldout_best_level_posthoc",
        worst="heldout_worst_level_posthoc",
        evidence="spent_posthoc_diagnostic",
    )
    add_rows(
        rows,
        pd.read_csv(inputs["P0580_galaxy_return"]),
        stage="P0580",
        domain="galaxy_rotation",
        observable="SPARC_outer_rotation_speed",
        metric="outer_RMSE_span",
        metric_type="performance",
        coordinate="parameter",
        span="outer_RMSE_impact_span_km_s",
        unit="km/s",
        best="best_level",
        worst="worst_level",
        evidence="131_galaxy_spent_factorial",
    )
    p0581 = pd.read_csv(inputs["P0581_exact_root"])
    add_rows(
        rows,
        p0581,
        stage="P0581",
        domain="cluster_raw_lensing",
        observable="exact_multiple_image_roots",
        metric="heldout_RMS_span",
        metric_type="performance",
        coordinate="parameter",
        span="heldout_impact_span_arcsec",
        unit="arcsec",
        best="best_level",
        worst="worst_level",
        evidence="four_cluster_exact_root_sensitivity",
    )
    add_rows(
        rows,
        p0581,
        stage="P0581",
        domain="cluster_raw_lensing",
        observable="exact_multiple_image_roots",
        metric="converged_root_count_span",
        metric_type="topology",
        coordinate="parameter",
        span="converged_root_span",
        unit="roots",
        evidence="four_cluster_exact_root_sensitivity",
    )

    p0582 = pd.read_csv(inputs["P0582_saturation"])
    saturation_rows = []
    for coordinate in ["contrast_mode", "nominal_cap"]:
        means = p0582.groupby(coordinate, dropna=False).heldout_converged_roots.mean()
        saturation_rows.append(
            {
                "coordinate": coordinate,
                "root_main_effect_span": float(means.max() - means.min()),
                "best_level": str(means.idxmax()),
                "worst_level": str(means.idxmin()),
            }
        )
    add_rows(
        rows,
        pd.DataFrame(saturation_rows),
        stage="P0582",
        domain="cluster_raw_lensing",
        observable="exact_multiple_image_roots",
        metric="mean_converged_root_count_span",
        metric_type="topology",
        coordinate="coordinate",
        span="root_main_effect_span",
        unit="roots",
        best="best_level",
        worst="worst_level",
        evidence="four_cluster_saturation_diagnostic",
    )
    add_rows(
        rows,
        pd.read_csv(inputs["P0586_metric"]),
        stage="P0586",
        domain="cluster_raw_lensing",
        observable="fixed_geometry_source_plane_response",
        metric="source_plane_RMS_span",
        metric_type="performance",
        coordinate="coordinate",
        span="main_effect_span_arcsec",
        unit="arcsec",
        best="best_main_effect_level",
        evidence="four_cluster_metric_screen_and_exact_validation",
    )
    p0586c = pd.read_csv(inputs["P0586C_signed_metric"])
    p0586c_mean = (
        p0586c.groupby("coordinate", as_index=False)
        .agg(
            main_effect_span_arcsec=("main_effect_span_arcsec", "mean"),
            best_main_effect_level=("best_main_effect_level", lambda x: "+".join(map(str, x))),
        )
    )
    add_rows(
        rows,
        p0586c_mean,
        stage="P0586C",
        domain="cluster_raw_lensing",
        observable="per_cluster_signed_source_plane_response",
        metric="mean_system_RMS_span",
        metric_type="performance",
        coordinate="coordinate",
        span="main_effect_span_arcsec",
        unit="arcsec",
        best="best_main_effect_level",
        evidence="spent_per_system_sign_diagnostic",
    )
    add_rows(
        rows,
        pd.read_csv(inputs["P0587_highpass"]),
        stage="P0587",
        domain="cluster_raw_lensing",
        observable="fixed_geometry_source_plane_response",
        metric="source_plane_RMS_span",
        metric_type="performance",
        coordinate="coordinate",
        span="main_effect_span_arcsec",
        unit="arcsec",
        best="best_main_effect_level",
        evidence="four_cluster_highpass_diagnostic",
    )
    add_rows(
        rows,
        pd.read_csv(inputs["P0603_tensor_route"]),
        stage="P0603",
        domain="cluster_reconstructed_lensing",
        observable="normalized_convergence_maps",
        metric="equal_system_JS_span",
        metric_type="performance",
        coordinate="parameter",
        span="equal_JS_span",
        unit="JS",
        best="best_level",
        worst="worst_level",
        evidence="thirty_tensor_plus_720_route_screen",
    )
    p0604 = pd.read_csv(inputs["P0604_cross_domain"])
    add_rows(
        rows,
        p0604,
        stage="P0604",
        domain="cluster_reconstructed_lensing",
        observable="normalized_convergence_maps",
        metric="median_JS_span",
        metric_type="performance",
        coordinate="parameter",
        span="cluster_median_JS_span",
        unit="JS",
        best="best_cluster_level",
        worst="worst_cluster_level",
        evidence="same_formula_cross_domain_factorial",
    )
    add_rows(
        rows,
        p0604,
        stage="P0604",
        domain="galaxy_rotation",
        observable="SPARC_outer_rotation_speed",
        metric="median_RMSE_span",
        metric_type="performance",
        coordinate="parameter",
        span="galaxy_median_RMSE_span_km_s",
        unit="km/s",
        evidence="same_formula_cross_domain_factorial",
    )
    p0606 = pd.read_csv(inputs["P0606_raw_sensitivity"])
    add_rows(
        rows,
        p0606,
        stage="P0606",
        domain="cluster_raw_lensing",
        observable="exact_multiple_image_roots",
        metric="training_RMS_span",
        metric_type="performance",
        coordinate="parameter",
        span="training_RMS_span_arcsec",
        unit="arcsec",
        best="best_training_variant",
        evidence="raw_exact_one_coordinate_sensitivity",
    )
    add_rows(
        rows,
        p0606,
        stage="P0606",
        domain="cluster_raw_lensing",
        observable="exact_multiple_image_roots",
        metric="failed_variant_count",
        metric_type="topology_risk",
        coordinate="parameter",
        span="failed_exact_root_variants",
        unit="failed variants",
        evidence="raw_exact_one_coordinate_sensitivity",
    )

    observations = pd.DataFrame(rows)
    keys = ["stage_id", "domain", "observable", "metric"]
    maxima = observations.groupby(keys).raw_span.transform("max")
    observations["normalized_leverage"] = np.where(maxima > 0.0, observations.raw_span / maxima, 0.0)
    observations["within_metric_rank"] = observations.groupby(keys).raw_span.rank(
        method="min", ascending=False
    ).astype(int)
    observations = observations.sort_values(keys + ["within_metric_rank", "coordinate"]).reset_index(drop=True)

    family = (
        observations.groupby("coordinate_family", as_index=False)
        .agg(
            observations=("normalized_leverage", "size"),
            independent_stages=("stage_id", "nunique"),
            domain_count=("domain", "nunique"),
            domains=("domain", lambda x: "+".join(sorted(set(x)))),
            median_normalized_leverage=("normalized_leverage", "median"),
            mean_normalized_leverage=("normalized_leverage", "mean"),
            maximum_normalized_leverage=("normalized_leverage", "max"),
            rank_one_count=("within_metric_rank", lambda x: int((x == 1).sum())),
            high_leverage_count=("normalized_leverage", lambda x: int((x >= 0.5).sum())),
        )
    )
    family["impact_priority_score"] = (
        family.median_normalized_leverage
        * np.log2(1.0 + family.independent_stages)
        * (1.0 + 0.25 * (family.domain_count - 1.0))
    )
    family = family.sort_values(
        ["impact_priority_score", "independent_stages", "mean_normalized_leverage"],
        ascending=False,
    ).reset_index(drop=True)
    family.insert(0, "priority_rank", np.arange(1, len(family) + 1))

    stage_winners = observations[observations.within_metric_rank.eq(1)].copy()
    stage_winners = stage_winners[
        [
            "stage_id",
            "domain",
            "metric",
            "metric_type",
            "coordinate",
            "coordinate_family",
            "raw_span",
            "raw_unit",
            "transfer_outcome",
        ]
    ]

    p0610 = json.loads(inputs["P0610_gate_driver"].read_text(encoding="utf-8"))
    p0611 = json.loads(inputs["P0611_gate_transfer"].read_text(encoding="utf-8"))
    transfer = pd.DataFrame(
        [
            {
                "stage_id": "P0610",
                "coordinate_family": "environmental_gate",
                "candidate": "dual_component_misalignment",
                "systems": p0610["coverage"]["systems_with_finite_raw_response"],
                "status": "posthoc_outlier_dominated_candidate",
                "diagnostic_value": p0610["correlations"]["minimum_leave_one_out_Pearson_r"],
                "diagnostic_name": "minimum_leave_one_out_Pearson_r",
                "passes": False,
            },
            {
                "stage_id": "P0611",
                "coordinate_family": "environmental_gate",
                "candidate": "dual_component_misalignment",
                "systems": p0611["coverage"]["systems"],
                "status": "frozen_transfer_failed_all_advance_gates",
                "diagnostic_value": p0611["responses"][1]["heldout_only_diagnostic_improvement_fraction"],
                "diagnostic_name": "high_activation_system_heldout_improvement_fraction",
                "passes": bool(p0611["gate_audit"]["all_gates_pass"]),
            },
        ]
    )

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    observations.to_csv(output / protocol["outputs"]["observations"], index=False)
    family.to_csv(output / protocol["outputs"]["family_summary"], index=False)
    stage_winners.to_csv(output / protocol["outputs"]["stage_winners"], index=False)
    transfer.to_csv(output / protocol["outputs"]["transfer_evidence"], index=False)

    cross_domain = family[family.domain_count.ge(2)]
    top = family.head(8)
    report = {
        "report_version": "P0612-CROSS-STAGE-PARAMETER-IMPACT-RESULTS-0.1.0",
        "status": "complete_descriptive_cross_stage_synthesis",
        "coverage": {
            "impact_observations": int(len(observations)),
            "stages": int(observations.stage_id.nunique()),
            "coordinate_families": int(observations.coordinate_family.nunique()),
            "domains": sorted(observations.domain.unique().tolist()),
            "cross_domain_families": int(len(cross_domain)),
        },
        "normalization": protocol["normalization"],
        "highest_priority_families": top.to_dict("records"),
        "cross_domain_families": cross_domain.to_dict("records"),
        "transfer_evidence": transfer.to_dict("records"),
        "universal_truths": [
            "Routed fraction is the most explosive raw-position coordinate, but its largest responses are destructive: the P0606 training optimum is zero route and five of eight fraction variants lose at least one exact root.",
            "Spatial width or support recurs across reconstructed clusters, raw lenses, and galaxy rotation. It is a robust sensitivity coordinate, not a demonstrated universal correction.",
            "Endpoint residence is repeatedly the least-bad return rule in raw heldout diagnostics and SPARC, while conservative return remains about 6.85 times the fixed-RAR galaxy RMSE.",
            "Anisotropy strength can be locally large, but preferred signs conflict by cluster; tensor orientation itself is consistently low leverage.",
            "Environmental gates and response saturation influence image-root topology more than their smooth-score spans suggest, yet the frozen dual-misalignment gate failed transfer.",
            "Distance falloff and detailed tensor orientation are low-leverage compared with fraction, width, path length, residence, and saturation coordinates in the tested ranges.",
            "Solar compatibility has generally been supplied by an exact symmetry null or high-acceleration screen; that protects local tests but is not evidence for the cluster mechanism.",
        ],
        "next_test": {
            "formula_family": "bounded endpoint residence with baryonic-size width and no object-specific gravity parameter",
            "freeze_coordinates": [
                "endpoint residence rule",
                "width_over_R80",
                "smooth response saturation",
                "one universal routed fraction or strength",
            ],
            "required_scores": [
                "SPARC outer velocity RMSE",
                "raw cluster heldout RMS",
                "exact image-root count",
                "Solar maximum fractional change and Mercury precession proxy",
            ],
            "reason": "These coordinates combine recurrence, topology control, and direct cross-domain observability; destination networks, tensor orientation, and the dual-misalignment gate have already failed transfer.",
        },
        "claim_limits": [
            "Every input experiment is already project-spent; this is a synthesis, not a new holdout.",
            "Within-experiment normalization preserves ranks but discards absolute scale, so normalized leverage must not be read as percent improvement.",
            "High impact includes harmful responses and topology failures.",
            "Reconstructed convergence-map response is not equivalent to raw multiple-image prediction.",
        ],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    matrix = observations.pivot_table(
        index="coordinate_family", columns="stage_id", values="normalized_leverage", aggfunc="max"
    ).reindex(family.coordinate_family)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.2), constrained_layout=True)
    shown = family.head(12).sort_values("impact_priority_score")
    axes[0].barh(shown.coordinate_family, shown.impact_priority_score, color="#1261A0")
    axes[0].set(xlabel="recurrence-weighted sensitivity priority", title="Impact is not improvement")
    display = matrix.head(12)
    image = axes[1].imshow(display.fillna(0.0), aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    axes[1].set_xticks(np.arange(len(display.columns)), display.columns, rotation=55, ha="right")
    axes[1].set_yticks(np.arange(len(display.index)), display.index)
    axes[1].set_title("Maximum normalized leverage within each stage")
    fig.colorbar(image, ax=axes[1], label="within-metric normalized leverage", shrink=0.85)
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)

    top_lines = "\n".join(
        f"- {row.coordinate_family}: priority {row.impact_priority_score:.3f}, "
        f"{row.independent_stages} stages, {row.domain_count} domains"
        for row in family.head(6).itertuples()
    )
    (output / protocol["outputs"]["summary"]).write_text(
        "# P0612 cross-stage parameter impact\n\n"
        f"The atlas contains {len(observations)} impact observations from "
        f"{observations.stage_id.nunique()} stages. Scores are normalized only within a metric; "
        "large values can be harmful.\n\n"
        "Highest recurrence-weighted sensitivities:\n\n"
        f"{top_lines}\n\n"
        "The next bounded test should hold endpoint residence fixed and vary only baryonic-size "
        "width, smooth saturation, and one universal strength, while scoring exact image roots, "
        "SPARC velocities, and Solar nulls separately.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "coverage": report["coverage"],
                "highest_priority_families": top[
                    ["priority_rank", "coordinate_family", "impact_priority_score", "independent_stages", "domains"]
                ].to_dict("records"),
                "next_test": report["next_test"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
