"""
Simplified unified kernel test using existing working code structure.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Import from existing working modules
import sys
sys.path.insert(0, str(Path(__file__).parent))

from unified_kernel import compute_unified_kernel
from test_sparc_coherence import load_rotmod

G_MSUN_KPC_KM2_S2 = 4.302e-6


def main():
    print("=" * 80)
    print("UNIFIED KERNEL TEST ON SPARC (SIMPLIFIED)")
    print("=" * 80)

    # Load data
    rotmod_dir = Path("data/Rotmod_LTG")
    summary_csv = Path("data/sparc/sparc_combined.csv")

    summary_df = pd.read_csv(summary_csv)
    print(f"\nLoaded summary: {len(summary_df)} galaxies")

    # Parameters are now handled by UnifiedKernelParams
    print("\nUsing unified kernel with multiplicative F_missing scaling")

    # Process a few galaxies to test
    rotmod_files = sorted(rotmod_dir.glob("*_rotmod.dat"))  # All galaxies
    print(f"\nTesting on {len(rotmod_files)} galaxies...")

    results = []
    for rotmod_path in rotmod_files:
        galaxy_name = rotmod_path.stem.replace("_rotmod", "")
        
        try:
            df = load_rotmod(str(rotmod_path))
        except Exception as e:
            print(f"  {galaxy_name}: Failed to load rotmod - {e}")
            continue

        R = df["R_kpc"].to_numpy(float)
        V_obs = df["V_obs"].to_numpy(float)
        V_gr = df["V_gr"].to_numpy(float)

        if len(R) < 4:
            continue

        # Find galaxy in summary
        galaxy_row = summary_df[summary_df["galaxy_name"].astype(str) == galaxy_name]
        if len(galaxy_row) == 0:
            print(f"  {galaxy_name}: Not found in summary")
            continue

        galaxy_row = galaxy_row.iloc[0]
        M_baryon = float(galaxy_row.get("M_baryon", np.nan))
        R_disk = float(galaxy_row.get("R_disk", np.nan))
        sigma_v = float(galaxy_row.get("sigma_velocity", 25.0))

        if np.isnan(M_baryon) or np.isnan(R_disk) or M_baryon <= 0 or R_disk <= 0:
            print(f"  {galaxy_name}: Missing M_baryon or R_disk")
            continue

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
            
            from unified_kernel import UnifiedKernelParams, MassCoherenceParams
            
            params = UnifiedKernelParams(
                ell0_kpc=5.0,
                f_amp=1.0,
                extra_amp=1.0,
                mass_params=MassCoherenceParams(F_max=5.0),
            )
            
            K_total, info = compute_unified_kernel(
                R_kpc=R,
                g_bar_kms2=g_bar_kms2,
                sigma_v_kms=sigma_v,
                rho_bar_msun_pc3=rho_bar_msun_pc3,
                galaxy_props=galaxy_props,
                params=params,
            )
        except Exception as e:
            print(f"  {galaxy_name}: Kernel computation failed - {e}")
            continue

        # Compute enhanced velocity
        g_eff = g_bar_kms2 * (1.0 + K_total)
        V_model = np.sqrt(R * 1e3 * g_eff)

        # Compute RMS
        mask = (V_gr > 1e-3) & (V_obs > 0) & np.isfinite(V_model)
        if np.sum(mask) < 4:
            continue

        rms_gr = float(np.sqrt(np.mean((V_obs[mask] - V_gr[mask]) ** 2)))
        rms_model = float(np.sqrt(np.mean((V_obs[mask] - V_model[mask]) ** 2)))
        delta_rms = rms_model - rms_gr

        results.append({
            "galaxy": galaxy_name,
            "rms_gr": rms_gr,
            "rms_model": rms_model,
            "delta_rms": delta_rms,
            "K_rough": info["K_rough"],
            "F_missing": info["F_missing"],
            "scale": info["scale"],
            "K_total_mean": info["K_total_mean"],
            "Xi_mean": info["Xi_mean"],
            "M_baryon": M_baryon,
            "R_disk": R_disk,
            "sigma_v": sigma_v,
        })
        if len(results) % 20 == 0:
            print(f"  Processed {len(results)} galaxies...")

    if len(results) > 0:
        df_results = pd.DataFrame(results)
        
        # Save results
        out_csv = Path("time-coherence/unified_kernel_sparc_results.csv")
        df_results.to_csv(out_csv, index=False)
        print(f"\nResults saved to {out_csv}")
        
        # Save summary
        summary = {
            "n_galaxies": int(len(df_results)),
            "n_improved": int(np.sum(df_results["delta_rms"] < 0)),
            "n_worsened": int(np.sum(df_results["delta_rms"] > 0)),
            "mean_rms_gr": float(df_results["rms_gr"].mean()),
            "mean_rms_model": float(df_results["rms_model"].mean()),
            "mean_delta_rms": float(df_results["delta_rms"].mean()),
            "median_delta_rms": float(df_results["delta_rms"].median()),
            "mean_K_rough": float(df_results["K_rough"].mean()),
            "mean_F_missing": float(df_results["F_missing"].mean()),
            "mean_scale": float(df_results["scale"].mean()),
            "mean_K_total": float(df_results["K_total_mean"].mean()),
            "mean_Xi": float(df_results["Xi_mean"].mean()),
        }
        summary_path = Path("time-coherence/unified_kernel_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total galaxies: {len(results)}")
        print(f"Galaxies improved: {np.sum(df_results['delta_rms'] < 0)}")
        print(f"Galaxies worsened: {np.sum(df_results['delta_rms'] > 0)}")
        print(f"\nRMS Statistics:")
        print(f"  Mean GR RMS: {df_results['rms_gr'].mean():.2f} km/s")
        print(f"  Mean Model RMS: {df_results['rms_model'].mean():.2f} km/s")
        print(f"  Mean delta_RMS: {df_results['delta_rms'].mean():.2f} km/s")
        print(f"  Median delta_RMS: {df_results['delta_rms'].median():.2f} km/s")
        print(f"\nKernel Statistics:")
        print(f"  Mean K_rough: {df_results['K_rough'].mean():.3f}")
        print(f"  Mean F_missing: {df_results['F_missing'].mean():.3f}")
        print(f"  Mean scale: {df_results['scale'].mean():.3f}")
        print(f"  Mean K_total: {df_results['K_total_mean'].mean():.3f}")
        print(f"  Mean Xi: {df_results['Xi_mean'].mean():.3f}")
        print(f"\nGalaxy Properties:")
        print(f"  Mean M_baryon: {df_results['M_baryon'].mean():.2e} Msun")
        print(f"  Mean R_disk: {df_results['R_disk'].mean():.2f} kpc")
        print(f"  Mean sigma_v: {df_results['sigma_v'].mean():.2f} km/s")
        print(f"\nSummary saved to {summary_path}")
    else:
        print("\nNo valid results")


if __name__ == "__main__":
    main()

