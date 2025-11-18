"""
Fit a single cluster amplitude scaling factor A_cluster for the theory kernel
to match observed Einstein radius masses across all clusters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from theory_metric_resonance import compute_theory_kernel
from run_theory_kernel_clusters import (
    ClusterDataLoader,
    load_master_catalog,
    parse_cluster_list,
    sigma_crit,
    angular_diameter_distance,
    G_KPC_KM2_S2_MSUN,
)


def compute_cluster_mass_with_scaling(
    cluster_name: str,
    loader: ClusterDataLoader,
    master_catalog: pd.DataFrame,
    theory_params: dict,
    A_cluster: float,
    sigma_v_default: float,
) -> dict:
    """Compute theory mass at Einstein radius with A_cluster scaling."""
    cluster_dir = cluster_name.strip().upper()
    catalog_candidates = [cluster_dir]
    if cluster_dir.startswith("ABELL_"):
        catalog_candidates.append("A" + cluster_dir.split("_", 1)[1])
    if "MACSJ" in cluster_dir:
        catalog_candidates.append(cluster_dir.replace("MACSJ", "MACS"))
    elif "MACS" in cluster_dir:
        catalog_candidates.append(cluster_dir.replace("MACS", "MACSJ"))

    catalog_key = next(
        (c for c in catalog_candidates if c in master_catalog.index), None
    )
    if catalog_key is None:
        raise KeyError(f"{cluster_name} not found in master_catalog")

    meta = master_catalog.loc[catalog_key]
    z_l = float(meta["z_lens"])
    z_s = float(meta.get("z_source", 2.0))
    theta_e_obs = float(meta["theta_E_obs_arcsec"])

    try:
        data = loader.load_cluster(cluster_dir, validate=False)
    except FileNotFoundError:
        alt_name = (
            cluster_dir.replace("MACSJ", "MACS")
            if "MACSJ" in cluster_dir
            else cluster_dir.replace("MACS", "MACSJ")
        )
        data = loader.load_cluster(alt_name, validate=False)

    R = data.r_kpc
    M_enc, g_bar = loader.compute_baryonic_mass(data)

    th = theory_params["theory_fit_params"]
    phase_sign = float(th.get("phase_sign", 1.0))

    # Compute base theory kernel
    K_th_base = compute_theory_kernel(
        R_kpc=R,
        sigma_v_kms=sigma_v_default,
        alpha=th["alpha"],
        lam_coh_kpc=th["lam_coh_kpc"],
        lam_cut_kpc=th["lam_cut_kpc"],
        A_global=th["A_global"],
        burr_ell0_kpc=th.get("burr_ell0_kpc"),
        burr_p=th.get("burr_p", 1.0),
        burr_n=th.get("burr_n", 0.5),
        Q_ref=th.get("Q_ref", 1.0),
    )

    # Scale by A_cluster: M_theory = M_baryon * (1 + A_cluster * phase_sign * K_th_base)
    K_th_scaled = A_cluster * phase_sign * K_th_base
    g_eff = g_bar * (1.0 + K_th_scaled)
    M_eff = g_eff * R * R / G_KPC_KM2_S2_MSUN

    sigma_c = sigma_crit(z_l, z_s)
    D_l = angular_diameter_distance(z_l) * 1e3  # kpc
    theta_rad = theta_e_obs * np.pi / (180.0 * 3600.0)
    R_e_kpc = theta_rad * D_l
    M_required = np.pi * R_e_kpc**2 * sigma_c

    M_pred = float(np.interp(R_e_kpc, R, M_eff, left=np.nan, right=np.nan))
    M_baryon = float(np.interp(R_e_kpc, R, M_enc, left=np.nan, right=np.nan))

    return {
        "cluster": cluster_name,
        "M_required": M_required,
        "M_baryon": M_baryon,
        "M_theory": M_pred,
        "R_E_kpc": R_e_kpc,
    }


def objective(
    A_cluster: float,
    cluster_names: list[str],
    loader: ClusterDataLoader,
    master_catalog: pd.DataFrame,
    theory_params: dict,
    sigma_v_default: float,
) -> float:
    """
    Compute normalized chi-squared for A_cluster across all clusters.
    """
    if A_cluster <= 0:
        return 1e10  # Penalty for negative amplitudes

    residuals_sq = []
    weights = []

    for cluster_name in cluster_names:
        try:
            result = compute_cluster_mass_with_scaling(
                cluster_name,
                loader,
                master_catalog,
                theory_params,
                A_cluster,
                sigma_v_default,
            )
            M_req = result["M_required"]
            M_th = result["M_theory"]

            if np.isfinite(M_req) and np.isfinite(M_th) and M_req > 0:
                residual = (M_th - M_req) / M_req
                residuals_sq.append(residual**2)
                weights.append(1.0 / M_req**2)  # Weight by inverse variance
        except Exception:
            continue

    if not residuals_sq:
        return 1e10

    # Weighted chi-squared
    weighted_chi2 = np.sum(np.array(residuals_sq) * np.array(weights)) / np.sum(
        weights
    )
    return float(weighted_chi2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit A_cluster scaling factor for theory kernel on clusters."
    )
    parser.add_argument(
        "--cluster-summary-csv",
        default="gravitywavebaseline/theory_kernel_cluster_summary.csv",
        help="CSV from run_theory_kernel_clusters.py (optional, for comparison)",
    )
    parser.add_argument(
        "--clusters",
        default="MACSJ0416,MACSJ0717,ABELL_1689",
        help="Comma-separated list of cluster names",
    )
    parser.add_argument(
        "--cluster-list",
        default=None,
        help="Optional text file listing cluster names",
    )
    parser.add_argument(
        "--theory-fit-json",
        default="gravitywavebaseline/theory_metric_resonance_mw_fit.json",
    )
    parser.add_argument(
        "--sigma-v-default",
        type=float,
        default=1000.0,
        help="Default velocity dispersion for clusters (km/s)",
    )
    parser.add_argument(
        "--A-cluster-min",
        type=float,
        default=0.01,
        help="Minimum A_cluster to search",
    )
    parser.add_argument(
        "--A-cluster-max",
        type=float,
        default=1000.0,
        help="Maximum A_cluster to search",
    )
    parser.add_argument(
        "--out-json",
        default="gravitywavebaseline/theory_kernel_cluster_amp_fit.json",
    )
    args = parser.parse_args()

    # Load theory parameters
    theory_params = json.loads(Path(args.theory_fit_json).read_text())
    print(f"[info] Loaded theory parameters from {args.theory_fit_json}")

    # Load cluster data infrastructure
    loader = ClusterDataLoader(data_dir="data/clusters")
    master_catalog = load_master_catalog()
    cluster_names = list(parse_cluster_list(args.clusters, args.cluster_list))
    print(f"[info] Fitting A_cluster for {len(cluster_names)} clusters: {cluster_names}")

    # Optimize A_cluster
    print(f"[info] Searching A_cluster in [{args.A_cluster_min}, {args.A_cluster_max}]")
    
    result = minimize_scalar(
        objective,
        args=(cluster_names, loader, master_catalog, theory_params, args.sigma_v_default),
        method="bounded",
        bounds=(args.A_cluster_min, args.A_cluster_max),
        options={"xatol": 0.01, "maxiter": 200},
    )

    A_cluster_best = float(result.x)
    chi2_best = float(result.fun)
    print(f"[info] Best A_cluster = {A_cluster_best:.6g}")
    print(f"[info] Best normalized chi² = {chi2_best:.6g}")

    # Evaluate at best-fit A_cluster
    cluster_results = []
    for cluster_name in cluster_names:
        try:
            result_dict = compute_cluster_mass_with_scaling(
                cluster_name,
                loader,
                master_catalog,
                theory_params,
                A_cluster_best,
                args.sigma_v_default,
            )
            result_dict["A_cluster"] = A_cluster_best
            result_dict["mass_ratio"] = (
                result_dict["M_theory"] / result_dict["M_required"]
                if result_dict["M_required"] > 0
                else np.nan
            )
            result_dict["mass_deficit"] = (
                result_dict["M_required"] - result_dict["M_theory"]
            )
            cluster_results.append(result_dict)
        except Exception as exc:
            print(f"[warn] Failed to process {cluster_name}: {exc}")

    # Compare to empirical A_cluster if available
    A_gal = float(theory_params["theory_fit_params"].get("A_global", 1.0))
    A_ratio = A_cluster_best / A_gal if A_gal != 0 else np.inf
    print(f"[info] A_cluster / A_galaxy = {A_ratio:.3f}")

    # Load empirical cluster summary if available for comparison
    empirical_A_cluster = None
    if Path(args.cluster_summary_csv).exists():
        try:
            emp_df = pd.read_csv(args.cluster_summary_csv)
            # Check if there's an empirical A_cluster value stored somewhere
            # (This would come from Σ-Gravity fits, not this script)
            print(f"[info] Found empirical cluster summary at {args.cluster_summary_csv}")
        except Exception:
            pass

    out = {
        "A_cluster_best": A_cluster_best,
        "A_galaxy": A_gal,
        "A_cluster_A_gal_ratio": A_ratio,
        "chi2_norm": chi2_best,
        "optimization_success": bool(result.success),
        "optimization_message": str(result.message),
        "optimization_nfev": int(result.nfev),
        "sigma_v_default": args.sigma_v_default,
        "cluster_names": cluster_names,
        "cluster_results": cluster_results,
        "theory_params_source": str(args.theory_fit_json),
    }

    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"[info] Saved cluster amplitude fit to {args.out_json}")

    # Print summary table
    print("\n[summary] Cluster mass predictions at best-fit A_cluster:")
    print(f"{'Cluster':<20} {'M_req (Msun)':<15} {'M_theory (Msun)':<15} {'Ratio':<10}")
    print("-" * 65)
    for cr in cluster_results:
        print(
            f"{cr['cluster']:<20} "
            f"{cr['M_required']:>14.2e} "
            f"{cr['M_theory']:>14.2e} "
            f"{cr['mass_ratio']:>9.3f}"
        )


if __name__ == "__main__":
    main()


