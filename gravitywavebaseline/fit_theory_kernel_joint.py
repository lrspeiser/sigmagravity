"""
Joint fit of theory kernel parameters across MW, SPARC galaxies, and clusters.

This script fits a single parameter set θ* that minimizes a combined objective:
  L_total = w_MW * L_MW + w_SPARC * L_SPARC + w_cluster * L_cluster

where:
  - L_MW: chi-squared between K_theory and K_empirical on MW
  - L_SPARC: mean RMS improvement across SPARC galaxies
  - L_cluster: normalized chi-squared for cluster Einstein masses (with A_cluster scaling)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from glob import glob
import os

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from metric_resonance_multiplier import metric_resonance_multiplier
from theory_metric_resonance import compute_theory_kernel
from run_theory_kernel_clusters import (
    ClusterDataLoader,
    load_master_catalog,
    parse_cluster_list,
    sigma_crit,
    angular_diameter_distance,
    G_KPC_KM2_S2_MSUN,
)


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values * values)))


def load_mw_data(parquet_path: str, r_min: float, r_max: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load MW baseline and empirical kernel."""
    df = pd.read_parquet(parquet_path)
    mask = (
        df["R"].between(r_min, r_max)
        & np.isfinite(df["v_phi"])
        & np.isfinite(df["v_phi_GR"])
    )
    subset = df.loc[mask, ["R", "v_phi", "v_phi_GR"]].copy()
    if subset.empty:
        raise RuntimeError(f"No MW data in [{r_min}, {r_max}] kpc")
    
    R = subset["R"].to_numpy(float)
    
    # Load empirical kernel parameters
    mw_fit_path = Path("gravitywavebaseline/metric_resonance_mw_fit.json")
    mw_fit = json.loads(mw_fit_path.read_text())
    
    lam_orb = 2.0 * np.pi * R
    f_emp = metric_resonance_multiplier(
        R_kpc=R,
        lambda_orb_kpc=lam_orb,
        A=mw_fit["A"],
        ell0_kpc=mw_fit["ell0_kpc"],
        p=mw_fit["p"],
        n_coh=mw_fit["n_coh"],
        lambda_peak_kpc=mw_fit["lambda_peak_kpc"],
        sigma_ln_lambda=mw_fit["sigma_ln_lambda"],
    )
    K_emp = f_emp - 1.0
    
    return R, K_emp, mw_fit


def load_sparc_sample(
    rotmod_dir: str,
    sparc_summary: str,
    galaxy_col: str,
    sigma_col: str,
    max_galaxies: int | None = None,
) -> list[dict]:
    """Load a sample of SPARC galaxies for joint fitting."""
    summary_df = pd.read_csv(sparc_summary)
    sigma_map = dict(
        zip(
            summary_df[galaxy_col].astype(str),
            summary_df[sigma_col].astype(float),
        )
    )
    
    rotmod_paths = sorted(glob.glob(os.path.join(rotmod_dir, "*_rotmod.dat")))
    if max_galaxies:
        rotmod_paths = rotmod_paths[:max_galaxies]
    
    galaxies = []
    for path in rotmod_paths:
        galaxy_name = Path(path).name.replace("_rotmod.dat", "")
        try:
            df = pd.read_csv(
                path,
                sep=r"\s+",
                comment="#",
                header=None,
                usecols=[0, 1, 3, 4, 5],
                names=["R_kpc", "V_obs", "V_gas", "V_disk", "V_bul"],
                engine="python",
            )
            if df.empty or len(df) < 4:
                continue
            
            v_gr = np.sqrt(
                np.clip(
                    df["V_gas"].to_numpy() ** 2
                    + df["V_disk"].to_numpy() ** 2
                    + df["V_bul"].to_numpy() ** 2,
                    0.0,
                    None,
                )
            )
            df["V_gr"] = v_gr
            
            sigma_v = sigma_map.get(galaxy_name, 25.0)
            
            galaxies.append({
                "name": galaxy_name,
                "R": df["R_kpc"].to_numpy(float),
                "V_obs": df["V_obs"].to_numpy(float),
                "V_gr": v_gr,
                "sigma_v": sigma_v,
            })
        except Exception:
            continue
    
    return galaxies


def compute_mw_loss(
    R_mw: np.ndarray,
    K_emp: np.ndarray,
    sigma_v_mw: float,
    theta: np.ndarray,
    burr_p: float,
    burr_n: float,
) -> float:
    """Compute MW kernel shape loss."""
    A_global, alpha, lam_coh_kpc, lam_cut_kpc, burr_ell0_kpc, Q_ref = theta
    
    try:
        K_th = compute_theory_kernel(
            R_kpc=R_mw,
            sigma_v_kms=sigma_v_mw,
            alpha=alpha,
            lam_coh_kpc=lam_coh_kpc,
            lam_cut_kpc=lam_cut_kpc,
            A_global=A_global,
            burr_ell0_kpc=burr_ell0_kpc,
            burr_p=burr_p,
            burr_n=burr_n,
            Q_ref=Q_ref,
        )
        
        corr = float(np.corrcoef(K_th, K_emp)[0, 1])
        if not np.isfinite(corr):
            corr = 0.0
        
        # Chi-squared with correlation penalty
        chi2 = rms(K_th - K_emp) ** 2
        if corr < 0:
            chi2 += 1e6 * (1.0 - corr) ** 2
        
        return chi2
    except Exception:
        return 1e10


def compute_sparc_loss(
    galaxies: list[dict],
    theta: np.ndarray,
    burr_p: float,
    burr_n: float,
    use_sigma_gating: bool = True,
    sigma_ref: float = 25.0,
    beta_sigma: float = 1.0,
) -> float:
    """Compute SPARC RMS loss."""
    A_global, alpha, lam_coh_kpc, lam_cut_kpc, burr_ell0_kpc, Q_ref = theta
    
    delta_rms_list = []
    
    for gal in galaxies:
        try:
            R = gal["R"]
            V_obs = gal["V_obs"]
            V_gr = gal["V_gr"]
            sigma_v = gal["sigma_v"]
            
            # Apply sigma gating if requested
            if use_sigma_gating:
                v_flat = np.nanmedian(V_gr[-min(len(V_gr), 5):]) or 200.0
                Q = v_flat / max(sigma_v, 1e-3)
                G_sigma = (Q**beta_sigma) / (1.0 + Q**beta_sigma)
                A_eff = A_global * G_sigma
            else:
                A_eff = A_global
            
            K_th = compute_theory_kernel(
                R_kpc=R,
                sigma_v_kms=sigma_v,
                alpha=alpha,
                lam_coh_kpc=lam_coh_kpc,
                lam_cut_kpc=lam_cut_kpc,
                A_global=A_eff,
                burr_ell0_kpc=burr_ell0_kpc,
                burr_p=burr_p,
                burr_n=burr_n,
                Q_ref=Q_ref,
            )
            
            f_th = 1.0 + K_th
            V_model = V_gr * np.sqrt(np.clip(f_th, 0.0, None))
            
            rms_gr = rms(V_obs - V_gr)
            rms_th = rms(V_obs - V_model)
            delta_rms_list.append(rms_th - rms_gr)
        except Exception:
            continue
    
    if not delta_rms_list:
        return 1e10
    
    # Mean squared delta RMS (penalize increases)
    mean_delta_rms = np.mean(delta_rms_list)
    return max(0.0, mean_delta_rms) ** 2  # Only penalize if RMS increases


def compute_cluster_loss(
    cluster_names: list[str],
    loader: ClusterDataLoader,
    master_catalog: pd.DataFrame,
    theta: np.ndarray,
    A_cluster: float,
    sigma_v_default: float,
    burr_p: float,
    burr_n: float,
) -> float:
    """Compute cluster mass loss."""
    A_global, alpha, lam_coh_kpc, lam_cut_kpc, burr_ell0_kpc, Q_ref = theta
    
    residuals_sq = []
    
    for cluster_name in cluster_names:
        try:
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
                continue
            
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
            
            K_th_base = compute_theory_kernel(
                R_kpc=R,
                sigma_v_kms=sigma_v_default,
                alpha=alpha,
                lam_coh_kpc=lam_coh_kpc,
                lam_cut_kpc=lam_cut_kpc,
                A_global=A_global,
                burr_ell0_kpc=burr_ell0_kpc,
                burr_p=burr_p,
                burr_n=burr_n,
                Q_ref=Q_ref,
            )
            
            # Scale by A_cluster
            K_th_scaled = A_cluster * K_th_base
            g_eff = g_bar * (1.0 + K_th_scaled)
            M_eff = g_eff * R * R / G_KPC_KM2_S2_MSUN
            
            sigma_c = sigma_crit(z_l, z_s)
            D_l = angular_diameter_distance(z_l) * 1e3
            theta_rad = theta_e_obs * np.pi / (180.0 * 3600.0)
            R_e_kpc = theta_rad * D_l
            M_required = np.pi * R_e_kpc**2 * sigma_c
            
            M_pred = float(np.interp(R_e_kpc, R, M_eff, left=np.nan, right=np.nan))
            
            if np.isfinite(M_required) and np.isfinite(M_pred) and M_required > 0:
                residual = (M_pred - M_required) / M_required
                residuals_sq.append(residual**2)
        except Exception:
            continue
    
    if not residuals_sq:
        return 1e10
    
    return float(np.mean(residuals_sq))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Joint fit of theory kernel across MW, SPARC, and clusters."
    )
    parser.add_argument(
        "--baseline-parquet",
        default="gravitywavebaseline/gaia_with_gr_baseline.parquet",
    )
    parser.add_argument("--r-min", type=float, default=12.0)
    parser.add_argument("--r-max", type=float, default=16.0)
    parser.add_argument("--sigma-v-mw", type=float, default=30.0)
    parser.add_argument(
        "--rotmod-dir",
        default="data/Rotmod_LTG",
    )
    parser.add_argument(
        "--sparc-summary",
        default="data/sparc/sparc_combined.csv",
    )
    parser.add_argument(
        "--summary-galaxy-col",
        default="galaxy_name",
    )
    parser.add_argument(
        "--summary-sigma-col",
        default="sigma_velocity",
    )
    parser.add_argument(
        "--max-sparc-galaxies",
        type=int,
        default=50,
        help="Maximum number of SPARC galaxies to use in joint fit",
    )
    parser.add_argument(
        "--clusters",
        default="MACSJ0416,MACSJ0717,ABELL_1689",
    )
    parser.add_argument(
        "--sigma-v-cluster",
        type=float,
        default=1000.0,
    )
    parser.add_argument(
        "--weight-mw",
        type=float,
        default=1.0,
        help="Weight for MW kernel shape loss",
    )
    parser.add_argument(
        "--weight-sparc",
        type=float,
        default=1.0,
        help="Weight for SPARC RMS loss",
    )
    parser.add_argument(
        "--weight-cluster",
        type=float,
        default=1.0,
        help="Weight for cluster mass loss",
    )
    parser.add_argument(
        "--use-sigma-gating",
        action="store_true",
        help="Apply sigma gating for SPARC galaxies",
    )
    parser.add_argument(
        "--sigma-ref",
        type=float,
        default=25.0,
    )
    parser.add_argument(
        "--beta-sigma",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--out-json",
        default="gravitywavebaseline/theory_kernel_joint_fit.json",
    )
    args = parser.parse_args()

    print("[info] Loading MW data...")
    R_mw, K_emp, mw_fit = load_mw_data(
        args.baseline_parquet, args.r_min, args.r_max
    )
    burr_p = float(mw_fit.get("p", 1.0))
    burr_n = float(mw_fit.get("n_coh", 0.5))
    print(f"[info] MW: {len(R_mw)} points, K_emp range [{K_emp.min():.4f}, {K_emp.max():.4f}]")

    print(f"[info] Loading SPARC sample (max {args.max_sparc_galaxies})...")
    galaxies = load_sparc_sample(
        args.rotmod_dir,
        args.sparc_summary,
        args.summary_galaxy_col,
        args.summary_sigma_col,
        max_galaxies=args.max_sparc_galaxies,
    )
    print(f"[info] SPARC: {len(galaxies)} galaxies loaded")

    print("[info] Loading cluster infrastructure...")
    loader = ClusterDataLoader(data_dir="data/clusters")
    master_catalog = load_master_catalog()
    cluster_names = list(parse_cluster_list(args.clusters, None))
    print(f"[info] Clusters: {len(cluster_names)} clusters")

    # Parameter bounds: [A_global, alpha, lam_coh_kpc, lam_cut_kpc, burr_ell0_kpc, Q_ref, A_cluster]
    bounds = [
        (-50.0, 50.0),   # A_global
        (-2.0, 8.0),     # alpha
        (0.5, 150.0),    # lam_coh_kpc
        (20.0, 3000.0),  # lam_cut_kpc
        (2.0, 80.0),     # burr_ell0_kpc
        (0.1, 10.0),     # Q_ref
        (0.1, 1000.0),   # A_cluster
    ]

    def objective(theta: np.ndarray) -> float:
        A_global, alpha, lam_coh_kpc, lam_cut_kpc, burr_ell0_kpc, Q_ref, A_cluster = theta
        
        theta_base = np.array([A_global, alpha, lam_coh_kpc, lam_cut_kpc, burr_ell0_kpc, Q_ref])
        
        # MW loss
        L_mw = compute_mw_loss(R_mw, K_emp, args.sigma_v_mw, theta_base, burr_p, burr_n)
        
        # SPARC loss
        L_sparc = compute_sparc_loss(
            galaxies,
            theta_base,
            burr_p,
            burr_n,
            use_sigma_gating=args.use_sigma_gating,
            sigma_ref=args.sigma_ref,
            beta_sigma=args.beta_sigma,
        )
        
        # Cluster loss
        L_cluster = compute_cluster_loss(
            cluster_names,
            loader,
            master_catalog,
            theta_base,
            A_cluster,
            args.sigma_v_cluster,
            burr_p,
            burr_n,
        )
        
        # Weighted sum
        L_total = (
            args.weight_mw * L_mw
            + args.weight_sparc * L_sparc
            + args.weight_cluster * L_cluster
        )
        
        return L_total

    print("[info] Starting joint optimization...")
    print(f"[info] Weights: MW={args.weight_mw}, SPARC={args.weight_sparc}, Cluster={args.weight_cluster}")
    
    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=300,
        polish=True,
        seed=123,
        atol=1e-4,
        tol=1e-3,
    )
    
    A_global, alpha, lam_coh_kpc, lam_cut_kpc, burr_ell0_kpc, Q_ref, A_cluster = result.x
    
    print("[info] Optimization complete!")
    print(f"[info] Best-fit parameters:")
    print(f"  A_global: {A_global:.6g}")
    print(f"  alpha: {alpha:.6g}")
    print(f"  lam_coh_kpc: {lam_coh_kpc:.6g}")
    print(f"  lam_cut_kpc: {lam_cut_kpc:.6g}")
    print(f"  burr_ell0_kpc: {burr_ell0_kpc:.6g}")
    print(f"  Q_ref: {Q_ref:.6g}")
    print(f"  A_cluster: {A_cluster:.6g}")
    print(f"  A_cluster/A_galaxy: {A_cluster/A_global:.3f}")
    
    # Evaluate final losses
    theta_base = np.array([A_global, alpha, lam_coh_kpc, lam_cut_kpc, burr_ell0_kpc, Q_ref])
    L_mw_final = compute_mw_loss(R_mw, K_emp, args.sigma_v_mw, theta_base, burr_p, burr_n)
    L_sparc_final = compute_sparc_loss(
        galaxies,
        theta_base,
        burr_p,
        burr_n,
        use_sigma_gating=args.use_sigma_gating,
        sigma_ref=args.sigma_ref,
        beta_sigma=args.beta_sigma,
    )
    L_cluster_final = compute_cluster_loss(
        cluster_names,
        loader,
        master_catalog,
        theta_base,
        A_cluster,
        args.sigma_v_cluster,
        burr_p,
        burr_n,
    )
    
    print(f"[info] Final losses:")
    print(f"  L_MW: {L_mw_final:.6e}")
    print(f"  L_SPARC: {L_sparc_final:.6e}")
    print(f"  L_cluster: {L_cluster_final:.6e}")
    
    # Compute final correlation
    K_th_final = compute_theory_kernel(
        R_kpc=R_mw,
        sigma_v_kms=args.sigma_v_mw,
        alpha=alpha,
        lam_coh_kpc=lam_coh_kpc,
        lam_cut_kpc=lam_cut_kpc,
        A_global=A_global,
        burr_ell0_kpc=burr_ell0_kpc,
        burr_p=burr_p,
        burr_n=burr_n,
        Q_ref=Q_ref,
    )
    corr_final = float(np.corrcoef(K_th_final, K_emp)[0, 1])
    print(f"[info] Final MW correlation: {corr_final:.6f}")
    
    out = {
        "parameters": {
            "A_global": float(A_global),
            "alpha": float(alpha),
            "lam_coh_kpc": float(lam_coh_kpc),
            "lam_cut_kpc": float(lam_cut_kpc),
            "burr_ell0_kpc": float(burr_ell0_kpc),
            "burr_p": float(burr_p),
            "burr_n": float(burr_n),
            "Q_ref": float(Q_ref),
            "A_cluster": float(A_cluster),
            "A_cluster_A_galaxy_ratio": float(A_cluster / A_global),
        },
        "losses": {
            "L_MW": float(L_mw_final),
            "L_SPARC": float(L_sparc_final),
            "L_cluster": float(L_cluster_final),
            "L_total": float(result.fun),
        },
        "metrics": {
            "MW_correlation": float(corr_final),
            "MW_chi2": float(L_mw_final),
        },
        "optimization": {
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(result.nfev),
        },
        "weights": {
            "weight_mw": args.weight_mw,
            "weight_sparc": args.weight_sparc,
            "weight_cluster": args.weight_cluster,
        },
        "data": {
            "n_mw_points": len(R_mw),
            "n_sparc_galaxies": len(galaxies),
            "n_clusters": len(cluster_names),
            "cluster_names": cluster_names,
        },
    }
    
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"[info] Saved joint fit to {args.out_json}")


if __name__ == "__main__":
    main()

