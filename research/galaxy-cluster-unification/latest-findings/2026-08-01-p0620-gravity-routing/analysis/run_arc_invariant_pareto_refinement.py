#!/usr/bin/env python3
"""Refine the potential-depth/path-ratio absolute-lensing Pareto boundary."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_arc_apogee_cross_domain import fit_q, score_predictions, velocity_prediction  # noqa: E402
from run_arc_apogee_boundary_refinement import cross_galaxy_score, morphology_scores  # noqa: E402
from run_arc_invariant_absolute_lensing import (  # noqa: E402
    cluster_score,
    json_safe,
    pareto_ids,
    prepare_clusters,
    prepare_galaxies,
    response_for_frame,
    response_parameters,
    run_raw_shortlist,
    sha256,
)
from voidscreen.arc_invariants import generalized_solar_diagnostics  # noqa: E402
from voidscreen.unified import rar_acceleration  # noqa: E402


def build_specs(protocol: dict) -> list[dict]:
    fixed = protocol["law"]["fixed_terms"]
    grid = protocol["grid"]
    specs = []
    for values in itertools.product(
        grid["potential_power"],
        grid["potential_scale"],
        grid["path_ratio_power"],
        grid["photon_extra_multiplier"],
    ):
        potential_power, potential_scale, path_power, photon = values
        specs.append(
            {
                **fixed,
                "invariant_mode": "potential_depth",
                "invariant_power": potential_power,
                "invariant_scale": potential_scale,
                "secondary_path_ratio_power": path_power,
                "photon_extra_multiplier": photon,
                "candidate_id": f"P{len(specs):04d}",
            }
        )
    return specs


def select_shortlist(scores: pd.DataFrame, rar_rmse: float, protocol: dict) -> pd.DataFrame:
    policy = protocol["selection_without_raw_scores"]
    eligible = scores[scores.all_solar_proxies_pass]
    galaxy_limit = float(policy["galaxy_limit_relative_to_RAR"]) * rar_rmse
    constrained = eligible[eligible.cross_galaxy_outer_RMSE_km_s <= galaxy_limit]
    zero_slip = eligible[eligible.photon_extra_multiplier.eq(1.0)]
    zero_constrained = constrained[constrained.photon_extra_multiplier.eq(1.0)]
    good_cluster = eligible[
        eligible.cluster_RMSE_dex <= float(policy["cluster_good_threshold_dex"])
    ]
    chosen = [
        ("best_cluster_within_galaxy_limit", constrained.sort_values("cluster_RMSE_dex").iloc[0]),
        ("best_zero_slip_within_galaxy_limit", zero_constrained.sort_values("cluster_RMSE_dex").iloc[0]),
        ("best_galaxy_below_cluster_threshold", good_cluster.sort_values("cross_galaxy_outer_RMSE_km_s").iloc[0]),
        ("best_zero_slip_cluster_overall", zero_slip.sort_values("cluster_RMSE_dex").iloc[0]),
    ]
    unique = {}
    for role, row in chosen:
        if row.candidate_id not in unique:
            unique[row.candidate_id] = {**row.to_dict(), "selection_role": role}
        else:
            unique[row.candidate_id]["selection_role"] += "+" + role
    return pd.DataFrame(unique.values())


def parameter_impacts(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for parameter in (
        "invariant_power",
        "invariant_scale",
        "secondary_path_ratio_power",
        "photon_extra_multiplier",
    ):
        grouped = scores.groupby(parameter).agg(
            median_galaxy_RMSE=("cross_galaxy_outer_RMSE_km_s", "median"),
            median_cluster_RMSE=("cluster_RMSE_dex", "median"),
        )
        rows.append(
            {
                "parameter": parameter,
                "galaxy_impact_span_km_s": float(grouped.median_galaxy_RMSE.max() - grouped.median_galaxy_RMSE.min()),
                "cluster_impact_span_dex": float(grouped.median_cluster_RMSE.max() - grouped.median_cluster_RMSE.min()),
                "best_galaxy_level": str(grouped.median_galaxy_RMSE.idxmin()),
                "best_cluster_level": str(grouped.median_cluster_RMSE.idxmin()),
            }
        )
    return pd.DataFrame(rows).sort_values("cluster_impact_span_dex", ascending=False)


def make_figure(scores, shortlist, impacts, raw_scores, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax = axes[0, 0]
    scatter = ax.scatter(
        scores.cross_galaxy_outer_RMSE_km_s,
        scores.cluster_RMSE_dex,
        c=scores.photon_extra_multiplier,
        s=20,
        cmap="viridis",
        alpha=0.7,
    )
    ax.scatter(shortlist.cross_galaxy_outer_RMSE_km_s, shortlist.cluster_RMSE_dex, marker="*", s=180, edgecolor="black", facecolor="orange")
    ax.set(xlabel="held-out galaxy RMSE (km/s)", ylabel="CLASH absolute RMSE (dex)", title="Fine potential/path Pareto surface")
    fig.colorbar(scatter, ax=ax, label="photon extra multiplier")

    ax = axes[0, 1]
    zero = scores[scores.photon_extra_multiplier.eq(1.0)]
    for power, block in zero.groupby("invariant_power"):
        ax.scatter(block.cross_galaxy_outer_RMSE_km_s, block.cluster_RMSE_dex, label=f"p={power:g}", s=25)
    ax.set(xlabel="galaxy RMSE (km/s)", ylabel="CLASH RMSE (dex)", title="Zero-slip boundary")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    ordered = impacts.sort_values("cluster_impact_span_dex")
    ax.barh(ordered.parameter, ordered.cluster_impact_span_dex, color="tab:purple")
    ax.set(xlabel="median CLASH-RMSE span (dex)", title="Marginal parameter impact")

    ax = axes[1, 1]
    display = raw_scores.copy()
    values = pd.to_numeric(display.heldout_RMS_arcsec, errors="coerce").fillna(0.0)
    ax.barh(display.selection_role.str.replace("_", " "), values, color="tab:blue")
    for index, row in enumerate(display.itertuples(index=False)):
        if not bool(row.heldout_all_roots_converged):
            ax.text(0.05, index, f"{row.heldout_roots_converged}/7 roots", va="center", color="crimson")
    ax.axvline(0.5, color="crimson", ls="--")
    ax.set(xlabel="held-out RX J2129 RMS (arcsec)", title="Raw-image falsification")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    config_path = ROOT / "configs" / "arc_invariant_pareto_refinement_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    a0 = 1.2e-10
    galaxy, properties = prepare_galaxies(protocol, a0)
    clusters, _ = prepare_clusters(protocol)
    specs = build_specs(protocol)
    inner_mask = galaxy.split.eq("inner_train")
    outer_mask = galaxy.split.eq("outer_holdout")
    rows = []
    selected_cluster_cache = {}
    for index, spec in enumerate(specs):
        local = galaxy.copy()
        unit = response_for_frame(
            local,
            spec,
            q=1.0,
            a0=a0,
            radius_column="radius_adjusted_kpc",
            gbar_column="g_bar_m_s2",
        )
        local["arc_coordinate"] = unit["unit_fractional_response"]
        q = fit_q(local[inner_mask], protocol["grid"]["universal_q_bounds"])
        cross, fold_q = cross_galaxy_score(
            local[inner_mask], local[outer_mask], properties, protocol["grid"]["universal_q_bounds"]
        )
        response = response_for_frame(
            clusters,
            spec,
            q=q,
            a0=a0,
            radius_column="radius_kpc",
            gbar_column="gbar_m_s2",
        )
        prediction = clusters.gbar_m_s2.to_numpy(float) * response["lensing_enhancement"]
        cluster_metrics = cluster_score(clusters, prediction)
        solar = generalized_solar_diagnostics(**response_parameters(spec, q=q, a0=a0))
        rows.append(
            {
                **spec,
                "universal_q": q,
                "fold_q_min": float(np.min(fold_q)),
                "fold_q_max": float(np.max(fold_q)),
                "cross_galaxy_outer_RMSE_km_s": cross["RMSE_km_s"],
                "cross_galaxy_equal_RMSE_km_s": cross["equal_galaxy_RMSE_km_s"],
                **cluster_metrics,
                **solar,
                "all_solar_proxies_pass": bool(solar["Cassini_proxy_pass"] and solar["Earth_proxy_pass"] and solar["Mercury_proxy_pass"]),
            }
        )
        selected_cluster_cache[spec["candidate_id"]] = prediction
        if index % 96 == 0:
            print(f"Pareto refinement {index + 1}/{len(specs)}", flush=True)
    scores = pd.DataFrame(rows)
    scores["pareto"] = scores.candidate_id.isin(pareto_ids(scores))
    outer = galaxy[outer_mask]
    rar = score_predictions(outer, outer.velocity_RAR_same_nuisance_km_s.to_numpy(float))
    shortlist = select_shortlist(scores, rar["RMSE_km_s"], protocol)
    specs_by_id = {spec["candidate_id"]: spec for spec in specs}
    raw_scores, raw_predictions, raw_parameters, raw_profiles = run_raw_shortlist(
        protocol, shortlist, specs_by_id, a0
    )
    impacts = parameter_impacts(scores)

    cluster_selected = []
    morphology_rows = []
    role_map = shortlist.set_index("candidate_id").selection_role
    for candidate_id in shortlist.candidate_id:
        local = clusters.copy()
        local["candidate_id"] = candidate_id
        local["selection_role"] = role_map[candidate_id]
        local["predicted_lensing_m_s2"] = selected_cluster_cache[candidate_id]
        local["residual_dex"] = np.log10(local.predicted_lensing_m_s2) - local.log_gtot
        cluster_selected.append(local)
        galaxy_local = galaxy.copy()
        unit = response_for_frame(
            galaxy_local,
            specs_by_id[candidate_id],
            q=1.0,
            a0=a0,
            radius_column="radius_adjusted_kpc",
            gbar_column="g_bar_m_s2",
        )
        galaxy_local["arc_coordinate"] = unit["unit_fractional_response"]
        galaxy_local["velocity_arc_km_s"] = velocity_prediction(
            galaxy_local, float(shortlist.set_index("candidate_id").loc[candidate_id, "universal_q"])
        )
        galaxy_local["candidate_id"] = candidate_id
        morphology_rows.extend(morphology_scores(galaxy_local, properties, candidate_id))
    cluster_selected = pd.concat(cluster_selected, ignore_index=True)
    morphology = pd.DataFrame(morphology_rows).merge(
        role_map.rename("selection_role"), left_on="candidate_id", right_index=True
    )
    scores.to_csv(output / protocol["outputs"]["scores"], index=False)
    morphology.to_csv(output / protocol["outputs"]["morphology_scores"], index=False)
    cluster_selected.to_csv(output / protocol["outputs"]["cluster_predictions"], index=False)
    raw_predictions.to_csv(output / protocol["outputs"]["raw_predictions"], index=False)
    raw_parameters.to_csv(output / protocol["outputs"]["raw_parameters"], index=False)
    raw_profiles.to_csv(output / "raw_RXJ2129_profiles.csv", index=False)
    impacts.to_csv(output / protocol["outputs"]["parameter_impacts"], index=False)

    best_constrained = shortlist[shortlist.selection_role.str.contains("best_cluster_within")].iloc[0]
    best_zero_constrained = shortlist[shortlist.selection_role.str.contains("best_zero_slip_within")].iloc[0]
    best_raw_finite = raw_scores[
        raw_scores.heldout_all_roots_converged.astype(bool)
    ].sort_values("heldout_RMS_arcsec")
    report = {
        "report_version": protocol["protocol_version"],
        "status": "completed potential-path Pareto refinement with raw-image shortlist",
        "protocol_sha256": sha256(config_path),
        "coverage": {
            "variants": int(len(scores)),
            "SPARC_galaxies": int(galaxy.galaxy.nunique()),
            "SPARC_outer_points": int(len(outer)),
            "CLASH_systems": int(clusters.system.nunique()),
            "CLASH_points": int(len(clusters)),
            "raw_shortlist": int(len(shortlist)),
            "raw_optimization_starts_each": int(protocol["raw_lensing"]["optimization_starts"]),
        },
        "references": {
            "RAR_galaxy_RMSE_km_s": rar["RMSE_km_s"],
            "fixed_RAR_cluster_RMSE_dex": cluster_score(clusters, rar_acceleration(clusters.gbar_m_s2, a0))["cluster_RMSE_dex"],
            "previous_locked_raw_candidate_heldout_RMS_arcsec": 1.0642772678285497,
            "compact_halo_raw_heldout_RMS_arcsec": 2.5361068843508456,
        },
        "best_cluster_within_galaxy_limit": best_constrained.to_dict(),
        "best_zero_slip_within_galaxy_limit": best_zero_constrained.to_dict(),
        "shortlist_selected_without_raw_scores": shortlist.to_dict("records"),
        "raw_RXJ2129_scores": raw_scores.to_dict("records"),
        "best_finite_raw_score": None if best_raw_finite.empty else best_raw_finite.iloc[0].to_dict(),
        "selected_morphology_scores": morphology.to_dict("records"),
        "pareto_candidates": scores[scores.pareto].sort_values("cross_galaxy_outer_RMSE_km_s").to_dict("records"),
        "parameter_impacts": impacts.to_dict("records"),
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    summary = [
        "# Arc-invariant Pareto refinement",
        "",
        f"Fine-grid variants: {len(scores)}.",
        f"Best CLASH score within 1.5x RAR galaxy error: {best_constrained.cluster_RMSE_dex:.4f} dex at {best_constrained.cross_galaxy_outer_RMSE_km_s:.3f} km/s.",
        f"Best zero-slip score in the same galaxy range: {best_zero_constrained.cluster_RMSE_dex:.4f} dex at {best_zero_constrained.cross_galaxy_outer_RMSE_km_s:.3f} km/s.",
        "Raw RX J2129 scores are in report.json and were not used to select the shortlist.",
    ]
    (output / protocol["outputs"]["summary"]).write_text("\n".join(summary) + "\n", encoding="utf-8")
    make_figure(scores, shortlist, impacts, raw_scores, output / protocol["outputs"]["figure"])
    print(json.dumps(json_safe(report), indent=2))


if __name__ == "__main__":
    main()
