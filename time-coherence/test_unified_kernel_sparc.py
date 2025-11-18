"""
Test unified kernel on SPARC galaxies with real data.

Compares unified kernel (K_rough + K_missing) to:
- GR baseline
- Empirical Σ-Gravity
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
from pathlib import Path

# Add time-coherence to path
sys.path.insert(0, str(Path(__file__).parent))

from unified_kernel import compute_unified_kernel

# Load rotmod function (exact copy from test_sparc_coherence.py)
def load_rotmod(path: str) -> pd.DataFrame:
    """Load SPARC rotmod file."""
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        usecols=[0, 1, 3, 4, 5],
        names=["R_kpc", "V_obs", "V_gas", "V_disk", "V_bul"],
        engine="python",
    )
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
    return df[["R_kpc", "V_obs", "V_gr", "V_gas", "V_disk", "V_bul"]]

G_MSUN_KPC_KM2_S2 = 4.302e-6


def process_galaxy(
    rotmod_path: str,
    summary_df: pd.DataFrame,
    *,
    # Roughness parameters
    A0: float = 0.774,
    gamma_rough: float = 0.1,
    p: float = 0.757,
    n_coh: float = 0.5,
    ell0_kpc: float = 5.0,
    # Mass-coherence parameters
    K_max: float = 19.58,
    psi0: float = 7.34e-8,
    gamma_mass: float = 0.136,
    R_eff_factor: float = 1.33,
):
    """Process one galaxy with unified kernel."""
    galaxy_name = Path(rotmod_path).stem.replace("_rotmod", "")

    try:
        df = load_rotmod(rotmod_path)
    except Exception as e:
        return None

    R = df["R_kpc"].to_numpy(float)
    V_obs = df["V_obs"].to_numpy(float)
    V_gr = df["V_gr"].to_numpy(float)

    if len(R) < 4:
        return None

    # Get galaxy properties from summary
    galaxy_col = None
    for col in ["galaxy_name", "Galaxy", "name", "gal"]:
        if col in summary_df.columns:
            galaxy_col = col
            break

    if galaxy_col is None:
        return None

    galaxy_row = summary_df[summary_df[galaxy_col].astype(str) == galaxy_name]
    if len(galaxy_row) == 0:
        # Try alternative name matching
        galaxy_row = summary_df[summary_df[galaxy_col].astype(str).str.upper() == galaxy_name.upper()]
        if len(galaxy_row) == 0:
            return None

    galaxy_row = galaxy_row.iloc[0]

    # Extract properties
    M_baryon = float(galaxy_row.get("M_baryon", np.nan))
    R_disk = float(galaxy_row.get("R_disk", np.nan))
    sigma_v = float(galaxy_row.get("sigma_velocity", 25.0))

    # If M_baryon missing, try to estimate from V_flat
    if np.isnan(M_baryon) or M_baryon <= 0:
        v_flat = galaxy_row.get("v_flat", np.nan)
        if not np.isnan(v_flat) and v_flat > 0:
            # Rough estimate: M ~ V^2 R / G
            M_baryon = (v_flat ** 2) * R_disk / G_MSUN_KPC_KM2_S2 if not np.isnan(R_disk) else np.nan
        else:
            return None

    if np.isnan(R_disk) or R_disk <= 0:
        return None

    if np.isnan(M_baryon) or M_baryon <= 0:
        return None

    # Compute g_bar
    g_bar_kms2 = (V_gr**2) / (R * 1e3)

    # Estimate density
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
            ell0_kpc=ell0_kpc,
            A0=A0,
            gamma_rough=gamma_rough,
            p=p,
            n_coh=n_coh,
            K_max=K_max,
            psi0=psi0,
            gamma_mass=gamma_mass,
            R_eff_factor=R_eff_factor,
        )
    except Exception as e:
        return None

    # Compute enhanced velocity
    g_eff = g_bar_kms2 * (1.0 + K_total)
    V_model = np.sqrt(R * 1e3 * g_eff)  # Convert R to km

    # Compute RMS
    mask = (V_gr > 1e-3) & (V_obs > 0) & np.isfinite(V_model)
    if np.sum(mask) < 4:
        return None

    rms_gr = float(np.sqrt(np.mean((V_obs[mask] - V_gr[mask]) ** 2)))
    rms_model = float(np.sqrt(np.mean((V_obs[mask] - V_model[mask]) ** 2)))
    delta_rms = rms_model - rms_gr

    return {
        "galaxy": galaxy_name,
        "n_points": int(np.sum(mask)),
        "rms_gr": float(rms_gr),
        "rms_model": float(rms_model),
        "delta_rms": float(delta_rms),
        "K_rough": info["K_rough"],
        "K_missing": info["K_missing"],
        "K_total_mean": info["K_total_mean"],
        "Xi_mean": info["Xi_mean"],
        "M_baryon": float(M_baryon),
        "R_disk": float(R_disk),
        "sigma_v": float(sigma_v),
    }


def main():
    print("=" * 80)
    print("UNIFIED KERNEL TEST ON SPARC GALAXIES")
    print("=" * 80)

    # Load data
    rotmod_dir = Path("data/Rotmod_LTG")
    summary_csv = Path("data/sparc/sparc_combined.csv")

    if not rotmod_dir.exists():
        print(f"Error: {rotmod_dir} not found")
        return

    if not summary_csv.exists():
        print(f"Error: {summary_csv} not found")
        return

    summary_df = pd.read_csv(summary_csv)

    # Load fitted parameters
    mass_coherence_fit = Path("time-coherence/mass_coherence_fit.json")
    if mass_coherence_fit.exists():
        with open(mass_coherence_fit, "r") as f:
            fit_params = json.load(f)
        K_max = fit_params["params"]["K_max"]
        psi0 = fit_params["params"]["psi0"]
        gamma_mass = fit_params["params"]["gamma"]
        R_eff_factor = fit_params["params"]["R_eff_factor"]
        print(f"\nUsing fitted mass-coherence parameters:")
        print(f"  K_max: {K_max:.3f}")
        print(f"  psi0: {psi0:.6e}")
        print(f"  gamma: {gamma_mass:.3f}")
        print(f"  R_eff_factor: {R_eff_factor:.3f}")
    else:
        # Use defaults
        K_max = 19.58
        psi0 = 7.34e-8
        gamma_mass = 0.136
        R_eff_factor = 1.33
        print("\nUsing default mass-coherence parameters")

    # Process galaxies
    rotmod_files = sorted(rotmod_dir.glob("*_rotmod.dat"))
    print(f"\nProcessing {len(rotmod_files)} galaxies...")

    results = []
    errors = []
    for i, rotmod_path in enumerate(rotmod_files):
        try:
            result = process_galaxy(
                str(rotmod_path),
                summary_df,
                K_max=K_max,
                psi0=psi0,
                gamma_mass=gamma_mass,
                R_eff_factor=R_eff_factor,
            )
            if result is not None:
                results.append(result)
            else:
                errors.append(f"{rotmod_path.name}: returned None")
        except Exception as e:
            errors.append(f"{rotmod_path.name}: {str(e)}")
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(rotmod_files)} galaxies... ({len(results)} valid)")
    
    if len(errors) > 0 and len(results) == 0:
        print(f"\nErrors encountered (showing first 5):")
        for err in errors[:5]:
            print(f"  {err}")

    if len(results) == 0:
        print("\nError: No valid results")
        return

    df_results = pd.DataFrame(results)

    # Save results
    out_csv = Path("time-coherence/unified_kernel_sparc_results.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"\nResults saved to {out_csv}")

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    print(f"\nTotal galaxies: {len(df_results)}")
    print(f"Galaxies improved: {np.sum(df_results['delta_rms'] < 0)}")
    print(f"Galaxies worsened: {np.sum(df_results['delta_rms'] > 0)}")

    print(f"\nRMS Statistics:")
    print(f"  Mean GR RMS: {df_results['rms_gr'].mean():.2f} km/s")
    print(f"  Mean Model RMS: {df_results['rms_model'].mean():.2f} km/s")
    print(f"  Mean ΔRMS: {df_results['delta_rms'].mean():.2f} km/s")
    print(f"  Median ΔRMS: {df_results['delta_rms'].median():.2f} km/s")

    print(f"\nKernel Statistics:")
    print(f"  Mean K_rough: {df_results['K_rough'].mean():.3f}")
    print(f"  Mean K_missing: {df_results['K_missing'].mean():.3f}")
    print(f"  Mean K_total: {df_results['K_total_mean'].mean():.3f}")
    print(f"  Mean Xi: {df_results['Xi_mean'].mean():.3f}")

    print(f"\nGalaxy Properties:")
    print(f"  Mean M_baryon: {df_results['M_baryon'].mean():.2e} Msun")
    print(f"  Mean R_disk: {df_results['R_disk'].mean():.2f} kpc")
    print(f"  Mean sigma_v: {df_results['sigma_v'].mean():.2f} km/s")

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
        "mean_K_missing": float(df_results["K_missing"].mean()),
        "mean_K_total": float(df_results["K_total_mean"].mean()),
        "mean_Xi": float(df_results["Xi_mean"].mean()),
    }

    summary_path = Path("time-coherence/unified_kernel_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()

