"""
Test SPARC galaxies with roughness amplitude + Burr-XII shape.

This refactors the Σ-Gravity kernel to use:
- K_rough(Ξ) as system-level amplitude (from time-coherence)
- Burr-XII C(R/ℓ₀) as radial shape (unit amplitude)

Then computes F_missing = A_empirical / K_rough for analysis.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from coherence_time_kernel import (
    compute_tau_geom,
    compute_tau_noise,
    compute_tau_coh,
    compute_exposure_factor,
)
from burr_xii_shape import burr_xii_shape, compute_total_kernel
from system_level_k import system_level_K, compute_Xi_mean
from test_sparc_coherence import load_rotmod

G_MSUN_KPC_KM2_S2 = 4.302e-6


def process_galaxy(
    rotmod_path: str,
    sigma_v_kms: float,
    ell0_kpc: float = 5.0,  # Default from empirical fits
    p: float = 0.757,
    n_coh: float = 0.5,
    A0: float = 0.774,
    gamma: float = 0.1,
    tau_geom_method: str = "tidal",
) -> dict | None:
    """
    Process a single galaxy with roughness amplitude + Burr-XII shape.
    
    Returns:
    - K_rough: System-level roughness enhancement
    - K_total: Total enhancement profile
    - F_missing: Missing factor = A_empirical / K_rough
    - RMS improvement
    """
    galaxy = Path(rotmod_path).stem.replace("_rotmod", "")
    
    try:
        df = load_rotmod(rotmod_path)
    except Exception:
        return None
    
    R = df["R_kpc"].to_numpy(float)
    V_obs = df["V_obs"].to_numpy(float)
    V_gr = df["V_gr"].to_numpy(float)
    
    if len(R) < 4:
        return None
    
    # Compute g_bar
    g_bar_kms2 = (V_gr**2) / (R * 1e3)
    
    # Estimate density
    rho_bar_msun_pc3 = g_bar_kms2 / (G_MSUN_KPC_KM2_S2 * R * 1e3) * 1e-9
    
    # Compute timescales
    tau_geom = compute_tau_geom(
        R,
        g_bar_kms2,
        rho_bar_msun_pc3,
        method=tau_geom_method,
        alpha_geom=1.0,
    )
    
    tau_noise = compute_tau_noise(
        R,
        sigma_v_kms,
        method="galaxy",
        beta_sigma=1.5,
    )
    
    tau_coh = compute_tau_coh(tau_geom, tau_noise)
    
    # Compute mean exposure factor
    Xi_mean = compute_Xi_mean(R, g_bar_kms2, tau_coh)
    
    # Get system-level roughness enhancement
    K_rough = system_level_K(Xi_mean, A0=A0, gamma=gamma)
    
    # Compute radial shape (unit amplitude)
    C_R = burr_xii_shape(R, ell0_kpc, p=p, n_coh=n_coh)
    
    # Total enhancement
    K_total = K_rough * C_R
    
    # Apply enhancement
    f_enh = 1.0 + K_total
    V_model = V_gr * np.sqrt(np.clip(f_enh, 0.0, None))
    
    # Compute RMS
    rms_gr = float(np.sqrt(np.mean((V_obs - V_gr) ** 2)))
    rms_model = float(np.sqrt(np.mean((V_obs - V_model) ** 2)))
    
    # Estimate empirical amplitude needed
    # This is what Σ-Gravity would fit per galaxy
    # Approximate as: A_empirical ≈ mean(K_total) / mean(C_R)
    # Or use the actual enhancement needed: A_empirical ≈ mean((V_obs/V_gr)^2 - 1) / mean(C_R)
    V_ratio_sq = (V_obs / np.clip(V_gr, 1e-6, None)) ** 2
    K_req = np.clip(V_ratio_sq - 1.0, 0, None)
    K_req_mean = float(np.mean(K_req[K_req > 0]))
    
    # Empirical amplitude (what Σ-Gravity would fit)
    C_mean = float(np.mean(C_R))
    if C_mean > 1e-6:
        A_empirical = K_req_mean / C_mean
    else:
        A_empirical = K_req_mean
    
    # Missing factor
    if K_rough > 1e-6:
        F_missing = A_empirical / K_rough
    else:
        F_missing = np.nan
    
    return {
        "galaxy": galaxy,
        "sigma_v_kms": float(sigma_v_kms),
        "Xi_mean": float(Xi_mean),
        "K_rough": float(K_rough),
        "A_empirical": float(A_empirical),
        "F_missing": float(F_missing),
        "ell0_kpc": float(ell0_kpc),
        "C_mean": float(C_mean),
        "K_total_mean": float(np.mean(K_total)),
        "rms_gr": float(rms_gr),
        "rms_model": float(rms_model),
        "delta_rms": float(rms_gr - rms_model),
        "n_points": int(len(R)),
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test SPARC with roughness amplitude + Burr-XII shape"
    )
    parser.add_argument(
        "--rotmod-dir",
        type=str,
        default="data/Rotmod_LTG",
        help="Directory containing rotmod files",
    )
    parser.add_argument(
        "--sparc-summary-csv",
        type=str,
        default="data/sparc/sparc_combined.csv",
        help="SPARC summary CSV with sigma_v",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="time-coherence/sparc_roughness_amplitude.csv",
        help="Output CSV",
    )
    parser.add_argument(
        "--ell0-kpc",
        type=float,
        default=5.0,
        help="Characteristic coherence length (default 5.0 kpc)",
    )
    parser.add_argument(
        "--A0",
        type=float,
        default=0.774,
        help="K(Ξ) amplitude (default 0.774)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="K(Ξ) power-law index (default 0.1)",
    )
    args = parser.parse_args()
    
    # Load SPARC summary
    summary_path = Path(args.sparc_summary_csv)
    if not summary_path.exists():
        print(f"Error: {summary_path} not found")
        return
    
    summary = pd.read_csv(summary_path)
    
    # Get sigma_v map
    galaxy_col = None
    for col in ["galaxy_name", "Galaxy", "name", "gal"]:
        if col in summary.columns:
            galaxy_col = col
            break
    
    if galaxy_col is None:
        print("Error: Could not find galaxy name column")
        return
    
    sigma_map = dict(
        zip(
            summary[galaxy_col].astype(str),
            summary["sigma_velocity"].astype(float),
        )
    )
    
    # Process galaxies
    rotmod_dir = Path(args.rotmod_dir)
    if not rotmod_dir.exists():
        print(f"Error: {rotmod_dir} not found")
        return
    
    results = []
    
    print("=" * 80)
    print("SPARC ROUGHNESS AMPLITUDE TEST")
    print("=" * 80)
    print(f"Processing galaxies from {rotmod_dir}...")
    print(f"Parameters: ell0={args.ell0_kpc} kpc, A0={args.A0}, gamma={args.gamma}")
    print()
    
    rotmod_files = sorted(rotmod_dir.glob("*_rotmod.dat"))
    
    for i, rotmod_path in enumerate(rotmod_files):
        galaxy = rotmod_path.stem.replace("_rotmod", "")
        sigma_v = sigma_map.get(galaxy, 25.0)
        
        result = process_galaxy(
            str(rotmod_path),
            sigma_v,
            ell0_kpc=args.ell0_kpc,
            A0=args.A0,
            gamma=args.gamma,
        )
        
        if result:
            results.append(result)
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(rotmod_files)} galaxies...")
    
    if not results:
        print("No valid results")
        return
    
    df = pd.DataFrame(results)
    df.to_csv(args.out_csv, index=False)
    
    print(f"\nProcessed {len(df)} galaxies")
    print(f"Results saved to {args.out_csv}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"\nRoughness enhancement:")
    print(f"  Mean K_rough: {df['K_rough'].mean():.3f}")
    print(f"  Median K_rough: {df['K_rough'].median():.3f}")
    print(f"  Range: {df['K_rough'].min():.3f} - {df['K_rough'].max():.3f}")
    
    print(f"\nEmpirical amplitude:")
    print(f"  Mean A_empirical: {df['A_empirical'].mean():.3f}")
    print(f"  Median A_empirical: {df['A_empirical'].median():.3f}")
    
    print(f"\nMissing factor:")
    valid_F = df["F_missing"].dropna()
    if len(valid_F) > 0:
        print(f"  Mean F_missing: {valid_F.mean():.3f}")
        print(f"  Median F_missing: {valid_F.median():.3f}")
        print(f"  Range: {valid_F.min():.3f} - {valid_F.max():.3f}")
        print(f"\n  Roughness explains ~{100/valid_F.mean():.1f}% of enhancement")
    
    print(f"\nPerformance:")
    print(f"  Mean ΔRMS: {df['delta_rms'].mean():.2f} km/s")
    print(f"  Median ΔRMS: {df['delta_rms'].median():.2f} km/s")
    print(f"  Galaxies improved: {(df['delta_rms'] > 0).sum()}/{len(df)}")


if __name__ == "__main__":
    main()

