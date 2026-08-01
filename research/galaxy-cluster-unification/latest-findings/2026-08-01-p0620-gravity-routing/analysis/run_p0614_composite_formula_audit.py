#!/usr/bin/env python3
"""Audit the actual scalar-plus-endpoint composite without mixing equations."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    protocol_path = ROOT / "configs/p0614_composite_formula_audit_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_accounting_audit_after_P0613":
        raise RuntimeError("P0614 protocol is not frozen")
    inputs = {key: ROOT / value for key, value in protocol["inputs"].items()}
    p0554_report = json.loads(inputs["P0554_report"].read_text(encoding="utf-8"))
    p0554 = next(
        row
        for row in p0554_report["shortlist_selected_without_raw_scores"]
        if row["candidate_id"] == "P0554"
    )
    sparc = json.loads(inputs["SPARC_report"].read_text(encoding="utf-8"))
    rar = float(sparc["scores"]["fixed_RAR:invariant"]["outer_holdout"]["RMSE_km_s"])
    mond = float(sparc["scores"]["simple_MOND:invariant"]["outer_holdout"]["RMSE_km_s"])
    p0613 = json.loads(inputs["P0613_report"].read_text(encoding="utf-8"))
    winner = p0613["diagnostic_winner"]
    cluster = pd.read_csv(inputs["P0613_cluster_scores"])
    selected = cluster[cluster.variant.eq(winner["variant"])].copy()
    p0581 = json.loads(inputs["P0581_protocol"].read_text(encoding="utf-8"))
    validation_labels = set(p0581["systems"]["historical_validation_labels"])
    validation = selected[selected.system_label.isin(validation_labels)]
    validation_rms = float(
        np.sqrt(np.mean(np.square(validation.heldout_RMS_arcsec.to_numpy(float))))
    )
    compact = 9.989136027113078
    p0583 = json.loads(inputs["P0583_report"].read_text(encoding="utf-8"))
    rx = {row["variant"]: row for row in p0583["scores"]}
    rx_scalar = rx["scalar_baseline"]
    rx_route = rx["K0338_tanh20_candidate"]

    scorecard = pd.DataFrame(
        [
            {
                "domain": "SPARC outer rotation",
                "composite_value": float(p0554["cross_galaxy_outer_RMSE_km_s"]),
                "unit": "km/s RMSE",
                "comparator": "fixed RAR",
                "comparator_value": rar,
                "ratio_to_comparator": float(p0554["cross_galaxy_outer_RMSE_km_s"]) / rar,
                "status": "within_50_percent_not_better",
            },
            {
                "domain": "SPARC outer rotation",
                "composite_value": float(p0554["cross_galaxy_outer_RMSE_km_s"]),
                "unit": "km/s RMSE",
                "comparator": "simple MOND",
                "comparator_value": mond,
                "ratio_to_comparator": float(p0554["cross_galaxy_outer_RMSE_km_s"]) / mond,
                "status": "within_50_percent_not_better",
            },
            {
                "domain": "four raw clusters",
                "composite_value": float(winner["cluster_equal_complete_RMS_arcsec"]),
                "unit": "arcsec RMS",
                "comparator": "matched P0554 scalar parent",
                "comparator_value": float(
                    p0613["matched_winner_vs_scalar"]["scalar_RMS_arcsec"]
                ),
                "ratio_to_comparator": float(
                    p0613["matched_winner_vs_scalar"]["candidate_RMS_arcsec"]
                    / p0613["matched_winner_vs_scalar"]["scalar_RMS_arcsec"]
                ),
                "status": "root_complete_but_subpercent_matched_gain",
            },
            {
                "domain": "raw validation clusters",
                "composite_value": validation_rms,
                "unit": "arcsec RMS",
                "comparator": "compact halo",
                "comparator_value": compact,
                "ratio_to_comparator": validation_rms / compact,
                "status": "worse_than_compact_halo",
            },
            {
                "domain": "RXJ2129 route transfer",
                "composite_value": float(rx_route["heldout_RMS_arcsec"]),
                "unit": "arcsec RMS",
                "comparator": "P0554 scalar parent",
                "comparator_value": float(rx_scalar["heldout_RMS_arcsec"]),
                "ratio_to_comparator": float(rx_route["heldout_RMS_arcsec"])
                / float(rx_scalar["heldout_RMS_arcsec"]),
                "status": "formula_transfer_failed",
            },
            {
                "domain": "Solar Mercury",
                "composite_value": float(p0554["Mercury_precession_mas_per_century"]),
                "unit": "mas/century",
                "comparator": "published project margin absolute 3.1",
                "comparator_value": 3.1,
                "ratio_to_comparator": abs(
                    float(p0554["Mercury_precession_mas_per_century"])
                )
                / 3.1,
                "status": "pass",
            },
        ]
    )
    cfg = protocol["gates"]
    gates = {
        "galaxy_near_fixed_RAR_pass": bool(
            float(p0554["cross_galaxy_outer_RMSE_km_s"])
            <= float(cfg["galaxy_RMSE_to_fixed_RAR_max"]) * rar
        ),
        "raw_four_cluster_roots_pass": bool(
            int(winner["heldout_converged_roots"]) == 11
            and int(winner["complete_systems"]) == 4
        ),
        "raw_validation_near_compact_halo_pass": bool(
            validation_rms
            <= float(cfg["raw_validation_RMS_to_compact_halo_max"]) * compact
        ),
        "RXJ2129_route_transfer_improvement_pass": bool(
            float(rx_route["heldout_RMS_arcsec"])
            < float(rx_scalar["heldout_RMS_arcsec"])
        ),
        "RXJ2129_all_roots_pass": bool(rx_route["heldout_all_roots"]),
        "Solar_all_proxies_pass": bool(p0554["all_solar_proxies_pass"]),
        "zero_per_object_gravity_parameters_pass": True,
    }
    gates["composite_unification_pass"] = bool(all(gates.values()))
    report = {
        "report_version": "P0614-COMPOSITE-FORMULA-AUDIT-RESULTS-0.1.0",
        "status": "complete_composite_accounting_audit",
        "composite_equation": protocol["composite_equation"],
        "P0554_scalar_parameters": {
            key: p0554[key]
            for key in [
                "alpha",
                "apogee_ratio",
                "screen_exponent",
                "screen_scale",
                "mass_radius_delta",
                "extent_leak",
                "invariant_mode",
                "invariant_power",
                "invariant_scale",
                "secondary_path_ratio_power",
                "photon_extra_multiplier",
                "universal_q",
            ]
        },
        "endpoint_parameters": {
            "route_fraction_multiplier": winner["route_fraction_multiplier"],
            "width_over_R80": winner["width_over_R80"],
            "contrast_cap": winner["contrast_cap"],
            "return_length_over_R80": 0.36,
            "gate": "cluster_logistic(C=R50/R80)",
        },
        "parameter_accounting": {
            "per_object_gravity_parameters": 0,
            "explicit_global_scalar_constants_listed": 11,
            "additional_selected_endpoint_numbers": 4,
            "ordinary_nuisance_parameters_still_present": True,
            "one_parameter_theory": False,
        },
        "coverage": {
            "SPARC_galaxies": 131,
            "SPARC_outer_points": 968,
            "raw_factorial_clusters": 4,
            "raw_factorial_heldout_images": 11,
            "route_formula_transfer_clusters": 1,
            "route_formula_transfer_heldout_images": 7,
        },
        "scorecard": scorecard.to_dict("records"),
        "gates": gates,
        "interpretation": {
            "formula_promoted": False,
            "galaxy_success_carried_by_scalar_parent": True,
            "endpoint_layer_explains_galaxy_rotation": False,
            "endpoint_layer_main_observed_effect": "raw lens caustic topology",
            "distinct_next_test": "Derive route fraction from the scalar excess itself, f_self=Delta_0554(R80)/(1+Delta_0554(R80)), eliminating the independent extent-gate strength before scoring raw roots.",
        },
        "claim_limits": protocol["claim_limits"],
    }
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    scorecard.to_csv(output / protocol["outputs"]["scorecard"], index=False)
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    plot = scorecard[scorecard.unit.str.contains("RMSE|RMS", regex=True)].copy()
    figure, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    colors = ["#1b9e77" if ratio <= 1.25 else "#d95f02" for ratio in plot.ratio_to_comparator]
    ax.barh(
        plot.domain + " / " + plot.comparator,
        plot.ratio_to_comparator,
        color=colors,
    )
    ax.axvline(1.0, color="black", lw=1.0, label="comparator")
    ax.axvline(1.25, color="gray", lw=1.0, ls="--", label="25% margin")
    ax.set(xlabel="error ratio to stated comparator", title="Same composite equation across domains")
    ax.legend()
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)
    (output / protocol["outputs"]["summary"]).write_text(
        "# P0614 composite formula audit\n\n"
        f"P0554 plus the endpoint operator is **{float(p0554['cross_galaxy_outer_RMSE_km_s'])/rar:.3f}x** fixed RAR on SPARC, "
        f"**{validation_rms/compact:.3f}x** the compact-halo validation RMS, and "
        f"**{float(rx_route['heldout_RMS_arcsec'])/float(rx_scalar['heldout_RMS_arcsec']):.3f}x** its scalar parent on RXJ2129.\n\n"
        f"Solar proxies pass: **{p0554['all_solar_proxies_pass']}**. Composite unification passes: **{gates['composite_unification_pass']}**.\n",
        encoding="utf-8",
    )
    print(json.dumps({"scorecard": report["scorecard"], "gates": gates, "interpretation": report["interpretation"]}, indent=2))


if __name__ == "__main__":
    main()
