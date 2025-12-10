"""
SPARC: Test if "extra time in the field" numerically matches required boost.

Compares K_required (from V_obs²/V_bar² - 1) vs K_rough (from time-coherence kernel)
radius-by-radius for each galaxy.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from coherence_time_kernel import (
    compute_coherence_kernel,
    compute_tau_geom,
    compute_tau_noise,
    compute_tau_coh,
    compute_exposure_factor,
)
from test_sparc_coherence import load_rotmod

G_MSUN_KPC_KM2_S2 = 4.302e-6


def compute_required_boost(df):
    """
    Compute required boost K_req = V_obs²/V_bar² - 1.
    
    Returns: V_bar, V_obs, K_req, mask
    """
    # Baryonic GR speed
    V_bar = np.sqrt(
        np.clip(
            df["V_gas"].to_numpy() ** 2
            + df["V_disk"].to_numpy() ** 2
            + df["V_bul"].to_numpy() ** 2,
            0.0,
            None,
        )
    )
    V_obs = df["V_obs"].to_numpy()
    mask = (V_bar > 1e-6) & np.isfinite(V_obs) & (V_obs > 0)
    K_req = np.zeros_like(V_obs)
    K_req[mask] = (V_obs[mask] ** 2 / V_bar[mask] ** 2) - 1.0
    return V_bar, V_obs, K_req, mask


def analyze_one_galaxy(rotmod_path, params, sigma_v_default=None):
    """
    Analyze a single galaxy: compare K_required vs K_rough.
    
    Returns dict with correlation, RMS difference, etc.
    """
    try:
        df = load_rotmod(str(rotmod_path))
    except Exception:
        return None

    R = df["R_kpc"].to_numpy(dtype=float)
    V_bar, V_obs, K_req, mask = compute_required_boost(df)

    if np.sum(mask) < 4:
        return None

    # Circular speed for coherence should be near the baryonic GR speed
    v_circ_kms = V_bar.copy()
    
    # Compute g_bar from V_bar
    g_bar_kms2 = v_circ_kms**2 / (R * 1e3)  # km/s^2
    
    # Estimate density
    rho_bar_msun_pc3 = g_bar_kms2 / (G_MSUN_KPC_KM2_S2 * R * 1e3) * 1e-9

    # sigma_v: use galaxy-level value
    sigma_v_kms = sigma_v_default if sigma_v_default is not None else 25.0
    sigma_v_array = np.full_like(R, sigma_v_kms, dtype=float)

    # Compute K_rough and Xi for each radius
    K_rough = np.zeros_like(R)
    Xi = np.zeros_like(R)

    for i in range(len(R)):
        if not mask[i]:
            continue
            
        # Compute timescales
        tau_geom = compute_tau_geom(
            np.array([R[i]]),
            np.array([g_bar_kms2[i]]),
            np.array([rho_bar_msun_pc3[i]]),
            method=params.get("tau_geom_method", "tidal"),
            alpha_geom=params.get("alpha_geom", 1.0),
        )[0]
        
        tau_noise = compute_tau_noise(
            np.array([R[i]]),
            sigma_v_array[i],
            method="galaxy",
            beta_sigma=params.get("beta_sigma", 1.5),
        )[0]
        
        tau_coh = compute_tau_coh(
            np.array([tau_geom]),
            np.array([tau_noise])
        )[0]

        # Compute kernel
        K_rough[i] = compute_coherence_kernel(
            R_kpc=np.array([R[i]]),
            g_bar_kms2=np.array([g_bar_kms2[i]]),
            sigma_v_kms=sigma_v_kms,  # Use scalar
            A_global=params["A_global"],
            p=params["p"],
            n_coh=params["n_coh"],
            method="galaxy",
            rho_bar_msun_pc3=np.array([rho_bar_msun_pc3[i]]),
            tau_geom_method=params.get("tau_geom_method", "tidal"),
            alpha_length=params["alpha_length"],
            beta_sigma=params["beta_sigma"],
            alpha_geom=params.get("alpha_geom", 1.0),
            backreaction_cap=params.get("backreaction_cap", 10.0),
        )[0]

        # Compute exposure factor
        Xi[i] = compute_exposure_factor(
            np.array([R[i]]),
            np.array([g_bar_kms2[i]]),
            np.array([tau_coh])
        )[0]

    # Compare where both are defined
    good = mask & np.isfinite(K_rough) & np.isfinite(K_req) & (K_req >= -0.5)  # Allow some negative
    
    if not np.any(good) or np.sum(good) < 4:
        return None

    # Compute statistics
    K_req_good = K_req[good]
    K_rough_good = K_rough[good]
    Xi_good = Xi[good]
    
    # Check for sufficient variation
    std_K_req = np.std(K_req_good)
    std_K_rough = np.std(K_rough_good)
    
    if std_K_req < 1e-6 or std_K_rough < 1e-6:
        corr = 0.0
    else:
        corr_matrix = np.corrcoef(K_req_good, K_rough_good)
        corr = float(corr_matrix[0, 1]) if corr_matrix.size > 1 else 0.0
        # Handle NaN
        if np.isnan(corr):
            corr = 0.0
    
    rms_diff = float(np.sqrt(np.mean((K_req_good - K_rough_good) ** 2)))
    mean_abs_diff = float(np.mean(np.abs(K_req_good - K_rough_good)))
    
    # Relative error
    mean_K_req = float(np.mean(np.abs(K_req_good)))
    rel_error = mean_abs_diff / max(mean_K_req, 1e-6)

    return {
        "galaxy": Path(rotmod_path).stem.replace("_rotmod", ""),
        "n_points": int(good.sum()),
        "corr_Kreq_Krough": corr,
        "rms_diff_K": rms_diff,
        "mean_abs_diff_K": mean_abs_diff,
        "rel_error": rel_error,
        "Kreq_mean": float(np.mean(K_req_good)),
        "Kreq_median": float(np.median(K_req_good)),
        "Kreq_max": float(np.max(K_req_good)),
        "Krough_mean": float(np.mean(K_rough_good)),
        "Krough_median": float(np.median(K_rough_good)),
        "Krough_max": float(np.max(K_rough_good)),
        "Xi_mean": float(np.mean(Xi_good)),
        "Xi_median": float(np.median(Xi_good)),
        "Xi_max": float(np.max(Xi_good)),
        "sigma_v": sigma_v_kms,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test roughness vs required boost for SPARC galaxies"
    )
    parser.add_argument(
        "--rotmod-dir",
        type=str,
        default="data/Rotmod_LTG",
        help="Directory containing SPARC rotmod files",
    )
    parser.add_argument(
        "--fiducial-json",
        type=str,
        default="time-coherence/time_coherence_fiducial.json",
        help="Path to fiducial parameters JSON",
    )
    parser.add_argument(
        "--sigma-map",
        type=str,
        default="data/sparc/sparc_combined.csv",
        help="CSV with sigma_v per galaxy",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="time-coherence/roughness_vs_required_sparc.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    # Load parameters
    fiducial_path = Path(args.fiducial_json)
    if fiducial_path.exists():
        with open(fiducial_path, "r") as f:
            params = json.load(f)
    else:
        print(f"Warning: {fiducial_path} not found, using defaults")
        params = {
            "A_global": 1.0,
            "p": 0.757,
            "n_coh": 0.5,
            "alpha_length": 0.037,
            "beta_sigma": 1.5,
            "alpha_geom": 1.0,
            "backreaction_cap": 10.0,
            "tau_geom_method": "tidal",
        }

    # Load sigma_v map
    sigma_map = {}
    sigma_path = Path(args.sigma_map)
    if sigma_path.exists():
        summary = pd.read_csv(sigma_path)
        if "galaxy_name" in summary.columns and "sigma_velocity" in summary.columns:
            sigma_map = dict(
                zip(
                    summary["galaxy_name"].astype(str),
                    summary["sigma_velocity"].astype(float),
                )
            )

    # Process galaxies
    rotmod_dir = Path(args.rotmod_dir)
    if not rotmod_dir.exists():
        print(f"Error: {rotmod_dir} not found")
        return

    rotmod_paths = sorted(rotmod_dir.glob("*_rotmod.dat"))
    print(f"Processing {len(rotmod_paths)} SPARC galaxies...")
    print(f"Using parameters from: {fiducial_path}")

    rows = []
    for i, path in enumerate(rotmod_paths):
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(rotmod_paths)} galaxies...")

        galaxy = path.stem.replace("_rotmod", "")
        sigma_v = sigma_map.get(galaxy, 25.0)

        res = analyze_one_galaxy(str(path), params, sigma_v_default=sigma_v)
        if res is not None:
            rows.append(res)

    df = pd.DataFrame(rows)
    outpath = Path(args.out_csv)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outpath, index=False)

    print(f"\n{'=' * 80}")
    print("SPARC ROUGHNESS VS REQUIRED BOOST")
    print(f"{'=' * 80}")
    print(f"\nProcessed {len(df)} galaxies")
    print(f"Results saved to {outpath}")

    if len(df) > 0:
        print("\nGlobal statistics:")
        print(f"  Mean correlation: {df['corr_Kreq_Krough'].mean():.3f}")
        print(f"  Median correlation: {df['corr_Kreq_Krough'].median():.3f}")
        print(f"  Fraction with corr > 0.7: {(df['corr_Kreq_Krough'] > 0.7).sum()}/{len(df)} ({(df['corr_Kreq_Krough'] > 0.7).sum()/len(df)*100:.1f}%)")
        print(f"  Mean RMS difference: {df['rms_diff_K'].mean():.3f}")
        print(f"  Mean relative error: {df['rel_error'].mean():.2%}")
        print(f"\n  Mean K_req: {df['Kreq_mean'].mean():.3f}")
        print(f"  Mean K_rough: {df['Krough_mean'].mean():.3f}")
        print(f"  Mean Xi: {df['Xi_mean'].mean():.3e}")

        # Identify outliers
        low_corr = df[df["corr_Kreq_Krough"] < 0.3]
        if len(low_corr) > 0:
            print(f"\n  Galaxies with low correlation (< 0.3): {len(low_corr)}")
            print("    These may be outliers (bars, warps, bad data)")


if __name__ == "__main__":
    main()

