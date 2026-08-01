#!/usr/bin/env python3
"""Build the P0612-P0619 cross-domain parameter-impact synthesis."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def main() -> None:
    p0612 = load("results/p0612_cross_stage_parameter_impact/report.json")
    p0613 = load("results/p0613_bounded_endpoint_cross_domain/report.json")
    p0614 = load("results/p0614_composite_formula_audit/report.json")
    p0615 = load("results/p0615_self_coupled_quadrupole_route/report.json")
    p0616 = load("results/p0616_frozen_self_coupled_transfer/report.json")
    p0617 = load("results/p0617_self_coupled_support_phase_atlas/report.json")
    p0618 = load("results/p0618_universal_route_phase/report.json")
    p0619 = load("results/p0619_frozen_tangential_transfer/report.json")

    atlas = {row["coordinate_family"]: row for row in p0612["highest_priority_families"]}
    factorial = {row["parameter"]: row for row in p0613["parameter_impacts"]}
    family = {row["family"]: row for row in p0617["family_impacts"]}
    radial_a383 = next(row for row in p0616["responses"] if row["system_label"] == "A383")
    tangent_a383 = next(row for row in p0619["responses"] if row["system_label"] == "A383")
    q2 = next(
        row
        for row in p0615["combined_diagnostic"]
        if row["law"] == "quadratic_Q2_over_total"
    )
    parameter_rows = [
        {
            "coordinate": "angular_phase",
            "impact_label": "largest new raw-lens response",
            "quantitative_evidence": f"+90 deg: {100*p0618['selected_universal_phase']['mean_system_improvement']:.3f}% mean; {100*p0618['selected_universal_phase']['RXJ2129_improvement']:.3f}% RXJ2129",
            "transfer_evidence": f"A383 {100*tangent_a383['tangential_improvement_fraction']:+.3f}% vs radial {100*radial_a383['heldout_improvement_fraction']:+.3f}%",
            "main_limit": "only 3/5 opened-data systems improve; A383 remains 9.081 arcsec RMS",
        },
        {
            "coordinate": "spatial_width",
            "impact_label": "most recurrent cross-stage coordinate",
            "quantitative_evidence": f"P0612 priority {atlas['spatial_width']['impact_priority_score']:.3f}; local response span {100*family['width']['mean_system_improvement_span']:.3f} percentage points",
            "transfer_evidence": "mild widening participates in the A383 tangential transfer",
            "main_limit": "aggressive linear widening loses a raw image root",
        },
        {
            "coordinate": "width_return_coupling",
            "impact_label": "largest support-law interaction",
            "quantitative_evidence": f"local mean-response span {100*family['joint']['mean_system_improvement_span']:.3f} percentage points",
            "transfer_evidence": "not independently transferred apart from the frozen widened +90 law",
            "main_limit": "changes magnitude but not the cluster-dependent sign",
        },
        {
            "coordinate": "routed_fraction_or_strength",
            "impact_label": "most explosive and destructive coordinate",
            "quantitative_evidence": f"root-count span {factorial['route_fraction_multiplier']['mean_root_count_span']:.3f}; SPARC span {factorial['route_fraction_multiplier']['SPARC_RMSE_span_km_s']:.3f} km/s",
            "transfer_evidence": "large independent route strengths failed RXJ2129 transfer",
            "main_limit": f"route-only galaxy RMSE is {p0613['comparators']['winner_to_fixed_RAR_ratio']:.3f}x fixed RAR",
        },
        {
            "coordinate": "amplitude_functional_form",
            "impact_label": "root-safety regulator",
            "quantitative_evidence": f"Q^2 law: {100*q2['four_cluster_improvement']:.3f}% four-cluster and {100*q2['RXJ2129_improvement']:.3f}% RXJ2129 at fixed geometry",
            "transfer_evidence": f"radial A383 transfer {100*radial_a383['heldout_improvement_fraction']:+.3f}%",
            "main_limit": "linear Q loses a root; Q^2 radial gain did not transfer",
        },
        {
            "coordinate": "return_length",
            "impact_label": "moderate support coordinate",
            "quantitative_evidence": f"local mean-response span {100*family['return_length']['mean_system_improvement_span']:.3f} percentage points; RX span {100*family['return_length']['RXJ2129_improvement_span']:.3f}",
            "transfer_evidence": "fixed at 0.36 R80 in the A383 transfer",
            "main_limit": "no tested law resolves all five response signs",
        },
        {
            "coordinate": "center_crossing_rule",
            "impact_label": "low local leverage",
            "quantitative_evidence": f"local mean-response span {100*family['center_crossing']['mean_system_improvement_span']:.3f} percentage points",
            "transfer_evidence": "not selected for transfer",
            "main_limit": "smaller effect than width, return length, or phase",
        },
        {
            "coordinate": "contrast_cap",
            "impact_label": "interaction-only topology control",
            "quantitative_evidence": f"main root span {factorial['contrast_cap']['mean_root_count_span']:.3f}; SPARC span {factorial['contrast_cap']['SPARC_RMSE_span_km_s']:.3f} km/s",
            "transfer_evidence": "tanh cap 20 retained for numerical safety",
            "main_limit": "near-zero marginal effect in the bounded factorial",
        },
    ]
    parameter_findings = pd.DataFrame(parameter_rows)

    scorecard = {row["domain"] + " / " + row["comparator"]: row for row in p0614["scorecard"]}
    mercury = scorecard["Solar Mercury / published project margin absolute 3.1"]
    cross_domain_rows = [
        {
            "domain": "galaxy rotation",
            "current_value": p0619["inherited_cross_domain"]["SPARC_outer_RMSE_km_s"],
            "unit": "km/s RMSE",
            "comparator": "fixed RAR 10.348 km/s",
            "ratio_or_change": p0619["inherited_cross_domain"]["SPARC_to_RAR_ratio"],
            "outcome": "near but worse; carried entirely by P0554 scalar parent",
        },
        {
            "domain": "Solar Mercury",
            "current_value": mercury["composite_value"],
            "unit": "mas/century",
            "comparator": "absolute margin 3.1 mas/century",
            "ratio_or_change": mercury["ratio_to_comparator"],
            "outcome": "pass; angular route is an exact point-source null",
        },
        {
            "domain": "five-cluster frozen-geometry phase diagnostic",
            "current_value": 100 * p0618["selected_universal_phase"]["mean_system_improvement"],
            "unit": "% mean improvement",
            "comparator": "P0554 scalar at same geometry",
            "ratio_or_change": p0618["selected_universal_phase"]["systems_not_worse"] / 5,
            "outcome": "18/18 roots, but only 3/5 systems improve",
        },
        {
            "domain": "A383 chronological formula transfer",
            "current_value": next(row["heldout_RMS_arcsec"] for row in p0619["scores"] if row["system_label"] == "A383" and row["variant_id"] == "P0619_tangential_self_route"),
            "unit": "arcsec RMS",
            "comparator": "P0554 scalar 9.097 arcsec",
            "ratio_or_change": tangent_a383["tangential_improvement_fraction"],
            "outcome": "directional transfer positive but absolute 2 arcsec gate fails",
        },
        {
            "domain": "raw validation clusters",
            "current_value": scorecard["raw validation clusters / compact halo"]["composite_value"],
            "unit": "arcsec RMS",
            "comparator": "compact halo 9.989 arcsec",
            "ratio_or_change": scorecard["raw validation clusters / compact halo"]["ratio_to_comparator"],
            "outcome": "1.91x compact-halo error",
        },
        {
            "domain": "MS2137 transfer",
            "current_value": None,
            "unit": "arcsec RMS",
            "comparator": "P0554 scalar",
            "ratio_or_change": None,
            "outcome": "inconclusive: control and candidate both root-incomplete",
        },
    ]
    cross_domain = pd.DataFrame(cross_domain_rows)

    report = {
        "report_version": "P0620-PARAMETER-IMPACT-SYNTHESIS-0.1.0",
        "status": "complete_P0612_through_P0619_stage_synthesis",
        "current_formula": {
            "lensing": "alpha_test(x)=alpha_P0554(r)+epsilon*R_90[delta_alpha_route(x)]",
            "Delta80": "alpha_P0554(R80)/alpha_b(R80)-1",
            "self_routed_fraction": "Delta80/(1+Delta80)",
            "epsilon": "Q^2/(1+Delta80)",
            "width": "0.23 R80 sqrt(1+Q^2)",
            "return_length": "0.36 R80",
            "phase": "+90 degrees shared by all systems",
            "galaxy_and_solar_route_limit": "zero under the defined axisymmetric-disk and point-source symmetries",
        },
        "most_impactful_findings": {
            "most_recurrent": "spatial width/support",
            "largest_new_lens_response": "universal angular phase",
            "most_explosive_but_destructive": "routed fraction/strength",
            "most_important_unresolved_variable": "a baryon-predicted angular direction that explains why two clusters resist the shared tangential phase",
        },
        "parameter_findings": parameter_rows,
        "cross_domain_scorecard": cross_domain_rows,
        "universal_truths": [
            "The scalar P0554 parent, not the route layer, carries galaxy-rotation accuracy.",
            "Exact symmetry nulls protect Solar tests but do not validate cluster physics.",
            "Increasing route strength changes roots quickly and does not repair galaxy amplitudes.",
            "Width and path length regulate how strongly a conservative route is expressed, but usually do not select its beneficial sign.",
            "Angular phase produced the largest robust local lens response and converted A383 from a radial loss to a tangential gain under a frozen full refit.",
            "No tested universal phase improves all clusters, and absolute cluster error remains far above the frozen target and compact-halo comparison.",
        ],
        "decision": {
            "formula_promoted": False,
            "stage_objective_met": True,
            "next_required_observation": "independent baryonic direction data such as gas-versus-stellar centroids, external tidal axes, or resolved multipole orientation on new complete-baseline clusters",
            "next_required_test": "freeze a baryon-only phase predictor before raw image scoring and require full-root, full-refit, multi-cluster transfer",
        },
    }
    output = ROOT / "results/p0620_parameter_impact_synthesis"
    output.mkdir(parents=True, exist_ok=True)
    parameter_findings.to_csv(output / "parameter_findings.csv", index=False)
    cross_domain.to_csv(output / "cross_domain_scorecard.csv", index=False)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / "SUMMARY.md").write_text(
        "# P0620 parameter-impact synthesis\n\n"
        "- Most recurrent: **spatial width/support**.\n"
        "- Largest new raw-lens response: **universal angular phase**.\n"
        "- Most explosive but destructive: **routed fraction/strength**.\n"
        "- Best frozen phase transfer: **A383 +0.174%**, but **9.081 arcsec RMS**.\n"
        "- Galaxy: **12.592 km/s** versus fixed RAR **10.348 km/s**.\n"
        "- Solar proxies: **pass**, mostly through exact symmetry nulls.\n"
        "- Formula promoted: **False**.\n",
        encoding="utf-8",
    )

    plot = pd.DataFrame(
        [
            {"test": "Q2 fixed support", "percent": 100 * p0617["fixed_support_response"]["mean_system_improvement"]},
            {"test": "mild width law", "percent": 100 * p0617["selected_diagnostic"]["mean_system_improvement"]},
            {"test": "+90 phase", "percent": 100 * p0618["selected_universal_phase"]["mean_system_improvement"]},
            {"test": "A383 full-refit +90", "percent": 100 * tangent_a383["tangential_improvement_fraction"]},
        ]
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    axes[0].barh(plot.test, plot.percent, color=["#7570b3", "#1b9e77", "#d95f02", "#e7298a"])
    axes[0].axvline(0.0, color="black", lw=0.8)
    axes[0].set(xlabel="improvement vs matched scalar (%)", title="Phase dominates recent local changes")
    axes[1].bar(
        ["P0554", "fixed RAR", "simple MOND"],
        [
            p0619["inherited_cross_domain"]["SPARC_outer_RMSE_km_s"],
            scorecard["SPARC outer rotation / fixed RAR"]["comparator_value"],
            scorecard["SPARC outer rotation / simple MOND"]["comparator_value"],
        ],
        color=["#d95f02", "#1b9e77", "#66a61e"],
    )
    axes[1].set(ylabel="outer SPARC RMSE (km/s)", title="Galaxy result is inherited from scalar parent")
    figure.savefig(output / "p0620_parameter_impact_synthesis.png", dpi=180)
    plt.close(figure)
    print(json.dumps(report["most_impactful_findings"], indent=2))


if __name__ == "__main__":
    main()
