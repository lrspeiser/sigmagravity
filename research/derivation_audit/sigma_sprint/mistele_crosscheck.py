"""Covariance-aware reconstruction check against Mistele et al. profiles."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .datasets import normalize_cluster_name
from .model import G_SI

KPC_M = 3.085677581491367e19
MSUN_KG = 1.98847e30


def _interpolation_matrix(source_log_r, target_log_r):
    source = np.asarray(source_log_r, dtype=float)
    target = np.asarray(target_log_r, dtype=float)
    matrix = np.zeros((len(target), len(source)))
    for row, value in enumerate(target):
        if value <= source[0]:
            matrix[row, 0] = 1.0
        elif value >= source[-1]:
            matrix[row, -1] = 1.0
        else:
            upper = int(np.searchsorted(source, value))
            lower = upper - 1
            fraction = (value - source[lower]) / (source[upper] - source[lower])
            matrix[row, lower] = 1.0 - fraction
            matrix[row, upper] = fraction
    return matrix


def crosscheck_mistele(tian: pd.DataFrame, profiles: dict[str, dict]):
    rows = []
    cluster_stats = []
    for profile_name, item in sorted(profiles.items()):
        normalized = normalize_cluster_name(profile_name)
        target = tian[tian["cluster"].map(normalize_cluster_name) == normalized].sort_values(
            "radius_kpc"
        )
        if target.empty:
            continue
        profile_all = item["data"].sort_values("r_Mpc").reset_index(drop=True)
        valid_profile = (
            np.isfinite(profile_all["r_Mpc"])
            & np.isfinite(profile_all["M_Msun"])
            & (profile_all["r_Mpc"] > 0)
            & (profile_all["M_Msun"] > 0)
        )
        valid_indices = np.flatnonzero(valid_profile.to_numpy())
        profile = profile_all.loc[valid_profile].reset_index(drop=True)
        if len(profile) < 2:
            continue
        radius_kpc = profile["r_Mpc"].to_numpy(dtype=float) * 1000.0
        mass = profile["M_Msun"].to_numpy(dtype=float)
        inside = target[
            (target["radius_kpc"] >= radius_kpc.min())
            & (target["radius_kpc"] <= radius_kpc.max())
        ].copy()
        if inside.empty:
            continue
        weights = _interpolation_matrix(
            np.log(radius_kpc), np.log(inside["radius_kpc"].to_numpy())
        )
        log_mass = np.log(mass)
        interpolated_mass = np.exp(weights @ log_mass)
        radius_m = inside["radius_kpc"].to_numpy() * KPC_M
        acceleration = G_SI * interpolated_mass * MSUN_KG / radius_m**2
        log_acceleration = np.log10(acceleration)
        difference = log_acceleration - inside["log_gtot"].to_numpy()

        stat_error = profile["M_stat_err_Msun"].to_numpy(dtype=float)
        correlation = item.get("mass_correlation")
        covariance_used = correlation is not None and correlation.shape == (
            len(profile_all), len(profile_all)
        )
        if covariance_used:
            correlation = correlation[np.ix_(valid_indices, valid_indices)]
            mass_covariance = correlation * np.outer(stat_error, stat_error)
        else:
            mass_covariance = np.diag(stat_error**2)
        jacobian_log10_mass = np.diag(1.0 / (np.log(10.0) * mass))
        profile_log_covariance = jacobian_log10_mass @ mass_covariance @ jacobian_log10_mass
        interpolated_covariance = weights @ profile_log_covariance @ weights.T
        total_covariance = interpolated_covariance + np.diag(
            inside["err_log_gtot"].to_numpy() ** 2
        )
        inverse = np.linalg.pinv(total_covariance)
        chi2 = float(difference @ inverse @ difference)
        cluster_stats.append(
            {
                "cluster": inside["cluster"].iloc[0],
                "n_points": int(len(inside)),
                "covariance_used": bool(covariance_used),
                "chi2": chi2,
                "mean_difference_dex": float(np.mean(difference)),
                "rms_difference_dex": float(np.sqrt(np.mean(difference**2))),
            }
        )
        for index, (_, tian_row) in enumerate(inside.iterrows()):
            rows.append(
                {
                    "cluster": tian_row["cluster"],
                    "radius_kpc": float(tian_row["radius_kpc"]),
                    "log_gtot_tian": float(tian_row["log_gtot"]),
                    "log_gtot_mistele": float(log_acceleration[index]),
                    "difference_dex": float(difference[index]),
                    "mistele_stat_sigma_dex": float(
                        np.sqrt(max(interpolated_covariance[index, index], 0.0))
                    ),
                    "covariance_used": bool(covariance_used),
                }
            )
    residuals = pd.DataFrame(rows)
    if residuals.empty:
        return {
            "matched_clusters": 0,
            "matched_points": 0,
            "warning": "No overlapping radial ranges were found.",
        }, residuals, pd.DataFrame(cluster_stats)
    difference = residuals["difference_dex"].to_numpy()
    summary = {
        "matched_clusters": int(residuals["cluster"].nunique()),
        "matched_points": int(len(residuals)),
        "clusters_with_mass_covariance": int(
            sum(row["covariance_used"] for row in cluster_stats)
        ),
        "mean_difference_dex": float(np.mean(difference)),
        "rms_difference_dex": float(np.sqrt(np.mean(difference**2))),
        "median_abs_difference_dex": float(np.median(np.abs(difference))),
        "interpretation": (
            "This checks reconstruction consistency. The catalogs share CLASH lensing "
            "inputs and therefore do not constitute independent physical validation."
        ),
    }
    return summary, residuals, pd.DataFrame(cluster_stats)
