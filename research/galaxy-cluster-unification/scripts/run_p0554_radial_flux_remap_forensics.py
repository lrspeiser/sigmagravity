#!/usr/bin/env python3
"""Relate inward/outward radial-remap preference to measured object properties."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_arc_invariant_absolute_lensing import prepare_clusters, prepare_galaxies  # noqa: E402
from run_p0554_local_cross_domain_sensitivity import json_safe, sha256  # noqa: E402


def symmetric_sensitivity(inward: float, outward: float) -> float:
    mean = 0.5 * (float(inward) + float(outward))
    return float((float(inward) - float(outward)) / mean)


def profile_features(frame, *, object_column, radius_column, gbar_column):
    rows = []
    for name, block in frame.groupby(object_column, sort=True):
        radius = block[radius_column].to_numpy(float)
        gbar = block[gbar_column].to_numpy(float)
        rows.append(
            {
                "object": name,
                "log10_baryonic_mass": float(np.log10(block.force_equivalent_mass_solar.iloc[0])),
                "concentration_r50_over_r80": float(block.force_equivalent_concentration_r50_over_r80.iloc[0]),
                "median_log10_gbar": float(np.median(np.log10(gbar))),
                "median_potential_depth": float(np.median(block.potential_depth)),
                "maximum_potential_depth": float(np.max(block.potential_depth)),
                "median_path_ratio": float(np.median(block.potential_path_ratio)),
                "maximum_path_ratio": float(np.max(block.potential_path_ratio)),
                "radial_log_span": float(np.log10(np.max(radius) / np.min(radius))),
            }
        )
    return pd.DataFrame(rows)


def per_object_rmse(predictions, *, object_column, observed_column, predicted_column, log_residual):
    local = predictions.copy()
    if log_residual:
        local["residual"] = np.log10(local[predicted_column]) - np.log10(local[observed_column])
    else:
        local["residual"] = local[predicted_column] - local[observed_column]
    return (
        local.assign(residual2=np.square(local.residual))
        .groupby(["candidate_id", object_column], sort=True)
        .residual2.mean().pow(0.5)
        .rename("RMSE")
        .reset_index()
    )


def add_sensitivity(rmse, *, object_column, inward_id, outward_id, parent_id="parent"):
    pivot = rmse.pivot(index=object_column, columns="candidate_id", values="RMSE")
    output = pd.DataFrame({"object": pivot.index})
    output["parent_RMSE"] = pivot[parent_id].to_numpy(float)
    output["inward_RMSE"] = pivot[inward_id].to_numpy(float)
    output["outward_RMSE"] = pivot[outward_id].to_numpy(float)
    output["outward_preference"] = [
        symmetric_sensitivity(i, o)
        for i, o in zip(output.inward_RMSE, output.outward_RMSE, strict=True)
    ]
    output["preferred_direction"] = np.where(output.outward_preference > 0.0, "outward", "inward")
    output["parent_fractional_RMSE"] = output.parent_RMSE
    return output


def bh_qvalues(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    q = np.empty_like(p)
    running = 1.0
    for rank_index in range(len(p) - 1, -1, -1):
        original = order[rank_index]
        rank = rank_index + 1
        running = min(running, p[original] * len(p) / rank)
        q[original] = running
    return np.clip(q, 0.0, 1.0)


def correlations(frame, domain, features, minimum):
    rows = []
    for feature in features:
        local = frame[[feature, "outward_preference"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(local) < minimum or local[feature].nunique() < 3:
            continue
        result = spearmanr(local[feature], local.outward_preference)
        rows.append({"domain": domain, "feature": feature, "objects": len(local), "spearman_rho": float(result.statistic), "p_value": float(result.pvalue)})
    output = pd.DataFrame(rows)
    if not output.empty:
        output["fdr_q_value"] = bh_qvalues(output.p_value)
        output["absolute_rho"] = output.spearman_rho.abs()
    return output


def main():
    config_path = ROOT / "configs/p0554_radial_flux_remap_forensics_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    inward_id = protocol["symmetric_probe"]["inward_candidate"]
    outward_id = protocol["symmetric_probe"]["outward_candidate"]
    parent_protocol = json.loads((ROOT / protocol["inputs"]["parent_protocol"]).read_text(encoding="utf-8"))
    galaxies, _ = prepare_galaxies(parent_protocol, 1.2e-10)
    clusters, _ = prepare_clusters(parent_protocol)

    galaxy_predictions = pd.read_csv(ROOT / protocol["inputs"]["galaxy_predictions"])
    galaxy_predictions = galaxy_predictions[galaxy_predictions.candidate_id.isin(["parent", inward_id, outward_id])]
    galaxy_rmse = per_object_rmse(galaxy_predictions, object_column="galaxy", observed_column="velocity_observed_adjusted_km_s", predicted_column="predicted_km_s", log_residual=False)
    galaxy_sensitivity = add_sensitivity(galaxy_rmse, object_column="galaxy", inward_id=inward_id, outward_id=outward_id)
    galaxy_observed_scale = (
        galaxy_predictions[galaxy_predictions.candidate_id.eq("parent")]
        .groupby("galaxy")
        .velocity_observed_adjusted_km_s.median()
    )
    galaxy_sensitivity["parent_fractional_RMSE"] = [
        row.parent_RMSE / galaxy_observed_scale.loc[row.object]
        for row in galaxy_sensitivity.itertuples()
    ]
    galaxy_features = profile_features(galaxies, object_column="galaxy", radius_column="radius_adjusted_kpc", gbar_column="g_bar_m_s2")
    galaxy_properties = galaxies.groupby("galaxy", sort=True).first().reset_index()
    galaxy_extra = pd.DataFrame({
        "object": galaxy_properties.galaxy,
        "gas_fraction": galaxy_properties.gas_fraction,
        "stellar_bulge_fraction": galaxy_properties.stellar_bulge_fraction,
        "hubble_type": galaxy_properties.hubble_type,
        "log10_effective_radius": np.log10(galaxy_properties.effective_radius_kpc),
        "log10_disk_scale": np.log10(galaxy_properties.disk_scale_kpc),
        "log10_surface_density": np.log10(galaxy_properties.baryonic_mass_solar / (2.0 * np.pi * np.square(galaxy_properties.effective_radius_kpc))),
    })
    galaxy_sensitivity = galaxy_sensitivity.merge(galaxy_features, on="object", validate="one_to_one").merge(galaxy_extra, on="object", validate="one_to_one")
    galaxy_sensitivity.insert(0, "domain", "galaxy")

    cluster_predictions = pd.read_csv(ROOT / protocol["inputs"]["cluster_predictions"])
    cluster_predictions = cluster_predictions[cluster_predictions.candidate_id.isin(["parent", inward_id, outward_id])]
    cluster_rmse = per_object_rmse(cluster_predictions, object_column="system", observed_column="observed_g_m_s2", predicted_column="predicted_g_m_s2", log_residual=True)
    cluster_sensitivity = add_sensitivity(cluster_rmse, object_column="system", inward_id=inward_id, outward_id=outward_id)
    cluster_features = profile_features(clusters, object_column="system", radius_column="radius_kpc", gbar_column="gbar_m_s2")
    cluster_sensitivity = cluster_sensitivity.merge(cluster_features, on="object", validate="one_to_one")
    cluster_sensitivity.insert(0, "domain", "derived_cluster")

    multi = pd.read_csv(ROOT / protocol["inputs"]["multicluster_raw_scores"])
    multi = multi[multi.candidate_id.isin(["parent", inward_id, outward_id])].copy()
    rx = pd.read_csv(ROOT / protocol["inputs"]["RXJ1347_raw_scores"])
    rx = rx[rx.candidate_id.isin(["parent", inward_id, outward_id])].copy()
    rx["system_label"] = "RXJ1347"
    rx = rx.rename(columns={"heldout_exact_RMS_arcsec": "heldout_RMS_arcsec"})
    raw = pd.concat([multi, rx], ignore_index=True, sort=False)
    raw_rows = []
    for label, block in raw.groupby("system_label", sort=True):
        indexed = block.set_index("candidate_id")
        inward = indexed.loc[inward_id]
        outward = indexed.loc[outward_id]
        parent = indexed.loc["parent"]
        finite_pair = bool(inward.heldout_all_roots and outward.heldout_all_roots and np.isfinite(inward.heldout_RMS_arcsec) and np.isfinite(outward.heldout_RMS_arcsec))
        raw_rows.append({
            "domain": "raw_cluster",
            "object": label,
            "parent_RMSE": float(parent.heldout_RMS_arcsec) if np.isfinite(parent.heldout_RMS_arcsec) else np.nan,
            "inward_RMSE": float(inward.heldout_RMS_arcsec) if np.isfinite(inward.heldout_RMS_arcsec) else np.nan,
            "outward_RMSE": float(outward.heldout_RMS_arcsec) if np.isfinite(outward.heldout_RMS_arcsec) else np.nan,
            "outward_preference": symmetric_sensitivity(inward.heldout_RMS_arcsec, outward.heldout_RMS_arcsec) if finite_pair else np.nan,
            "preferred_direction": "unavailable" if not finite_pair else ("outward" if inward.heldout_RMS_arcsec > outward.heldout_RMS_arcsec else "inward"),
            "parent_fractional_RMSE": float(parent.heldout_RMS_arcsec) if np.isfinite(parent.heldout_RMS_arcsec) else np.nan,
            "heldout_images": int(parent.heldout_images),
        })
    raw_sensitivity = pd.DataFrame(raw_rows).merge(cluster_features, on="object", how="left", validate="one_to_one")

    common_features = protocol["features"]["common"]
    galaxy_feature_names = common_features + protocol["features"]["galaxy_only"]
    cluster_feature_names = common_features
    minimum = int(protocol["statistics"]["minimum_objects"])
    correlation_frame = pd.concat([
        correlations(galaxy_sensitivity, "galaxy", galaxy_feature_names, minimum),
        correlations(cluster_sensitivity, "derived_cluster", cluster_feature_names, minimum),
    ], ignore_index=True)
    objects = pd.concat([galaxy_sensitivity, cluster_sensitivity, raw_sensitivity], ignore_index=True, sort=False)
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    objects.to_csv(output / protocol["outputs"]["object_sensitivities"], index=False)
    correlation_frame.to_csv(output / protocol["outputs"]["correlations"], index=False)

    significant = correlation_frame[correlation_frame.fdr_q_value < 0.1]
    same_feature = []
    for feature, block in significant.groupby("feature"):
        if set(block.domain) == {"galaxy", "derived_cluster"} and len(set(np.sign(block.spearman_rho))) == 1:
            same_feature.append(feature)
    top_by_domain = {
        domain: block.sort_values(["fdr_q_value", "absolute_rho"]).head(8).to_dict("records")
        for domain, block in correlation_frame.groupby("domain")
    }
    raw_table = raw_sensitivity[["object", "outward_preference", "preferred_direction", "log10_baryonic_mass", "concentration_r50_over_r80", "median_potential_depth", "median_path_ratio", "heldout_images"]].to_dict("records")
    report = {
        "report_version": "P0554-RADIAL-FLUX-REMAP-FORENSICS-RESULTS-0.1.0",
        "status": "complete",
        "protocol": {"path": str(config_path.relative_to(ROOT)), "sha256": sha256(config_path)},
        "coverage": {"galaxies": len(galaxy_sensitivity), "derived_clusters": len(cluster_sensitivity), "raw_clusters": len(raw_sensitivity), "tested_correlations": len(correlation_frame)},
        "top_correlations": top_by_domain,
        "FDR_significant_features": significant.to_dict("records"),
        "same_direction_FDR_features_in_both_large_domains": same_feature,
        "raw_cluster_sign_table": raw_table,
        "verdict": {"predeclared_invariant_promotion_gate_passed": bool(same_feature), "no_invariant_promoted": not bool(same_feature)},
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, (domain, frame) in zip(axes[:2], [("galaxies", galaxy_sensitivity), ("derived clusters", cluster_sensitivity)]):
        ax.scatter(frame.median_potential_depth, frame.outward_preference, alpha=0.65)
        ax.axhline(0.0, color="black", ls="--")
        ax.set(xscale="log", xlabel="median baryonic potential depth", ylabel="outward preference", title=domain)
    axes[2].bar(raw_sensitivity.object, raw_sensitivity.outward_preference, color=np.where(raw_sensitivity.outward_preference > 0, "tab:blue", "tab:orange"))
    axes[2].axhline(0.0, color="black", ls="--")
    axes[2].set(ylabel="outward preference", title="six raw clusters")
    axes[2].tick_params(axis="x", rotation=45)
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)
    summary = (
        "# Radial-remap sign forensics\n\n"
        f"Tested {len(correlation_frame)} preregistered univariate relationships. "
        f"The cross-domain invariant promotion gate returned {bool(same_feature)}; shared features: {', '.join(same_feature) if same_feature else 'none'}.\n"
    )
    (output / protocol["outputs"]["summary"]).write_text(summary, encoding="utf-8")
    print(json.dumps(json_safe(report), indent=2), flush=True)


if __name__ == "__main__":
    main()
