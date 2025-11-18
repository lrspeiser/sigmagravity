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

    # Load fitted parameters
    mass_coherence_fit = Path("time-coherence/mass_coherence_fit.json")
    if mass_coherence_fit.exists():
        with open(mass_coherence_fit, "r") as f:
            fit_params = json.load(f)
        K_max = fit_params["params"]["K_max"]
        psi0 = fit_params["params"]["psi0"]
        gamma_mass = fit_params["params"]["gamma"]
        R_eff_factor = fit_params["params"]["R_eff_factor"]
    else:
        K_max = 19.58
        psi0 = 7.34e-8
        gamma_mass = 0.136
        R_eff_factor = 1.33

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
            K_total, info = compute_unified_kernel(
                R_kpc=R,
                g_bar_kms2=g_bar_kms2,
                sigma_v_kms=sigma_v,
                rho_bar_msun_pc3=rho_bar_msun_pc3,
                M_baryon_msun=M_baryon,
                R_disk_kpc=R_disk,
                ell0_kpc=5.0,  # Default
                K_max=K_max,
                psi0=psi0,
                gamma_mass=gamma_mass,
                R_eff_factor=R_eff_factor,
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
            "K_missing": info["K_missing"],
            "K_total_mean": info["K_total_mean"],
        })
        print(f"  {galaxy_name}: delta_RMS = {delta_rms:.2f} km/s, K_total = {info['K_total_mean']:.3f}")

    if len(results) > 0:
        df_results = pd.DataFrame(results)
        print(f"\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Galaxies processed: {len(results)}")
        print(f"Mean delta_RMS: {df_results['delta_rms'].mean():.2f} km/s")
        print(f"Mean K_total: {df_results['K_total_mean'].mean():.3f}")
        print(f"Mean K_rough: {df_results['K_rough'].mean():.3f}")
        print(f"Mean K_missing: {df_results['K_missing'].mean():.3f}")
    else:
        print("\nNo valid results")


if __name__ == "__main__":
    main()

