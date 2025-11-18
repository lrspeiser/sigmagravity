"""
Sweep unified kernel parameters to find optimal performance.

Explores f_amp, extra_amp, and F_max to match empirical Σ-Gravity.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from unified_kernel import UnifiedKernelParams, MassCoherenceParams, compute_unified_kernel
from test_sparc_coherence import load_rotmod

G_MSUN_KPC_KM2_S2 = 4.302e-6


def process_galaxy_with_params(
    rotmod_path: str | Path,
    summary_df: pd.DataFrame,
    params: UnifiedKernelParams,
) -> dict | None:
    """Process one galaxy with given parameters."""
    rotmod_path = Path(rotmod_path)
    galaxy_name = rotmod_path.stem.replace("_rotmod", "")
    
    try:
        df = load_rotmod(str(rotmod_path))
    except Exception:
        return None

    R = df["R_kpc"].to_numpy(float)
    V_obs = df["V_obs"].to_numpy(float)
    V_gr = df["V_gr"].to_numpy(float)

    if len(R) < 4:
        return None

    # Find galaxy in summary
    galaxy_row = summary_df[summary_df["galaxy_name"].astype(str) == galaxy_name]
    if len(galaxy_row) == 0:
        return None

    galaxy_row = galaxy_row.iloc[0]
    M_baryon = float(galaxy_row.get("M_baryon", np.nan))
    R_disk = float(galaxy_row.get("R_disk", np.nan))
    sigma_v = float(galaxy_row.get("sigma_velocity", 25.0))

    if np.isnan(M_baryon) or np.isnan(R_disk) or M_baryon <= 0 or R_disk <= 0:
        return None

    # Compute g_bar and density
    g_bar_kms2 = (V_gr**2) / (R * 1e3)
    rho_bar_msun_pc3 = g_bar_kms2 / (G_MSUN_KPC_KM2_S2 * R * 1e3) * 1e-9

    # Compute unified kernel
    try:
        galaxy_props = {
            "sigma_v": sigma_v,
            "R_disk": R_disk,
            "M_baryon": M_baryon,
        }
        
        K_total, info = compute_unified_kernel(
            R_kpc=R,
            g_bar_kms2=g_bar_kms2,
            sigma_v_kms=sigma_v,
            rho_bar_msun_pc3=rho_bar_msun_pc3,
            galaxy_props=galaxy_props,
            params=params,
        )
    except Exception:
        return None

    # Compute enhanced velocity
    g_eff = g_bar_kms2 * (1.0 + K_total)
    V_model = np.sqrt(R * 1e3 * g_eff)

    # Compute RMS
    mask = (V_gr > 1e-3) & (V_obs > 0) & np.isfinite(V_model)
    if np.sum(mask) < 4:
        return None

    rms_gr = float(np.sqrt(np.mean((V_obs[mask] - V_gr[mask]) ** 2)))
    rms_model = float(np.sqrt(np.mean((V_obs[mask] - V_model[mask]) ** 2)))
    delta_rms = rms_model - rms_gr

    return {
        "galaxy": galaxy_name,
        "rms_gr": rms_gr,
        "rms_model": rms_model,
        "delta_rms": delta_rms,
        "K_total_mean": info["K_total_mean"],
        "F_missing": info["F_missing"],
    }


def sweep_parameters():
    """Sweep parameter space and find optimal values."""
    print("=" * 80)
    print("UNIFIED KERNEL PARAMETER SWEEP")
    print("=" * 80)
    
    # Load data
    rotmod_dir = Path("data/Rotmod_LTG")
    summary_csv = Path("data/sparc/sparc_combined.csv")
    summary_df = pd.read_csv(summary_csv)
    
    rotmod_files = sorted(rotmod_dir.glob("*_rotmod.dat"))
    print(f"\nProcessing {len(rotmod_files)} galaxies...")
    
    # Parameter ranges to explore
    extra_amp_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    F_max_values = [3.0, 4.0, 5.0, 6.0, 7.0]
    
    results = []
    
    for extra_amp in extra_amp_values:
        for F_max in F_max_values:
            print(f"\nTesting: extra_amp={extra_amp:.2f}, F_max={F_max:.1f}")
            
            params = UnifiedKernelParams(
                ell0_kpc=5.0,
                f_amp=1.0,
                extra_amp=extra_amp,
                mass_params=MassCoherenceParams(F_max=F_max),
            )
            
            galaxy_results = []
            for rotmod_path in rotmod_files:
                result = process_galaxy_with_params(
                    str(rotmod_path),
                    summary_df,
                    params,
                )
                if result is not None:
                    galaxy_results.append(result)
            
            if len(galaxy_results) == 0:
                continue
            
            df_gal = pd.DataFrame(galaxy_results)
            
            n_improved = int(np.sum(df_gal["delta_rms"] < 0))
            n_worsened = int(np.sum(df_gal["delta_rms"] > 0))
            mean_delta_rms = float(df_gal["delta_rms"].mean())
            median_delta_rms = float(df_gal["delta_rms"].median())
            mean_K_total = float(df_gal["K_total_mean"].mean())
            mean_F_missing = float(df_gal["F_missing"].mean())
            
            results.append({
                "extra_amp": extra_amp,
                "F_max": F_max,
                "n_galaxies": len(df_gal),
                "n_improved": n_improved,
                "n_worsened": n_worsened,
                "frac_improved": n_improved / len(df_gal),
                "mean_delta_rms": mean_delta_rms,
                "median_delta_rms": median_delta_rms,
                "mean_K_total": mean_K_total,
                "mean_F_missing": mean_F_missing,
            })
            
            print(f"  Improved: {n_improved}/{len(df_gal)} ({100*n_improved/len(df_gal):.1f}%)")
            print(f"  Mean delta_RMS: {mean_delta_rms:.2f} km/s")
            print(f"  Mean K_total: {mean_K_total:.3f}")
    
    # Save results
    df_results = pd.DataFrame(results)
    out_csv = Path("time-coherence/unified_kernel_param_sweep.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"\n\nResults saved to {out_csv}")
    
    # Find best parameters
    print("\n" + "=" * 80)
    print("BEST PARAMETERS")
    print("=" * 80)
    
    # Best by fraction improved
    best_frac = df_results.loc[df_results["frac_improved"].idxmax()]
    print(f"\nBest by fraction improved:")
    print(f"  extra_amp: {best_frac['extra_amp']:.2f}")
    print(f"  F_max: {best_frac['F_max']:.1f}")
    print(f"  Fraction improved: {best_frac['frac_improved']:.1%}")
    print(f"  Mean delta_RMS: {best_frac['mean_delta_rms']:.2f} km/s")
    print(f"  Mean K_total: {best_frac['mean_K_total']:.3f}")
    
    # Best by mean delta_RMS (most negative = best)
    best_rms = df_results.loc[df_results["mean_delta_rms"].idxmin()]
    print(f"\nBest by mean delta_RMS:")
    print(f"  extra_amp: {best_rms['extra_amp']:.2f}")
    print(f"  F_max: {best_rms['F_max']:.1f}")
    print(f"  Mean delta_RMS: {best_rms['mean_delta_rms']:.2f} km/s")
    print(f"  Fraction improved: {best_rms['frac_improved']:.1%}")
    print(f"  Mean K_total: {best_rms['mean_K_total']:.3f}")
    
    # Best by median delta_RMS
    best_median = df_results.loc[df_results["median_delta_rms"].idxmin()]
    print(f"\nBest by median delta_RMS:")
    print(f"  extra_amp: {best_median['extra_amp']:.2f}")
    print(f"  F_max: {best_median['F_max']:.1f}")
    print(f"  Median delta_RMS: {best_median['median_delta_rms']:.2f} km/s")
    print(f"  Fraction improved: {best_median['frac_improved']:.1%}")
    print(f"  Mean K_total: {best_median['mean_K_total']:.3f}")
    
    # Save best parameters
    best_params = {
        "by_fraction_improved": {
            "extra_amp": float(best_frac["extra_amp"]),
            "F_max": float(best_frac["F_max"]),
            "frac_improved": float(best_frac["frac_improved"]),
            "mean_delta_rms": float(best_frac["mean_delta_rms"]),
            "mean_K_total": float(best_frac["mean_K_total"]),
        },
        "by_mean_delta_rms": {
            "extra_amp": float(best_rms["extra_amp"]),
            "F_max": float(best_rms["F_max"]),
            "mean_delta_rms": float(best_rms["mean_delta_rms"]),
            "frac_improved": float(best_rms["frac_improved"]),
            "mean_K_total": float(best_rms["mean_K_total"]),
        },
        "by_median_delta_rms": {
            "extra_amp": float(best_median["extra_amp"]),
            "F_max": float(best_median["F_max"]),
            "median_delta_rms": float(best_median["median_delta_rms"]),
            "frac_improved": float(best_median["frac_improved"]),
            "mean_K_total": float(best_median["mean_K_total"]),
        },
    }
    
    best_path = Path("time-coherence/unified_kernel_best_params.json")
    with open(best_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nBest parameters saved to {best_path}")
    
    return df_results, best_params


if __name__ == "__main__":
    sweep_parameters()

