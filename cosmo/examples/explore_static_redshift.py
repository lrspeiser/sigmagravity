#!/usr/bin/env python3
"""
explore_static_redshift.py
--------------------------
Explore non-expanding redshift mechanisms in Sigma-Gravity.

Tests three mechanisms:
(A) Tired-light: coherence-loss produces frequency drift
(B) ISW-like: time-varying coherence in static spacetime
(C) Geometric: path-wandering lengthens photon paths

Compares all three to standard Hubble law z = (H0/c) * D
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add cosmo to path
SCRIPT_DIR = Path(__file__).parent
COSMO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(COSMO_DIR))

import sigma_redshift_static as srs

def main():
    print("="*80)
    print("EXPLORING NON-EXPANDING REDSHIFT IN SIGMA-GRAVITY")
    print("="*80)
    
    # Distance range: 1 Mpc to 3000 Mpc (z ~ 0 to 0.7)
    distances_Mpc = np.linspace(1.0, 3000.0, 200)
    
    # Use calibrated Sigma-Gravity coherence parameters
    # (from cluster fits: ell0 ~ 200 kpc, p=0.75, ncoh=0.5)
    params = srs.SigmaRedshiftParams(
        ell0_kpc=200.0,       # Cluster coherence scale
        p=0.75,               # Burr-XII shape
        ncoh=0.5,             # Burr-XII damping
        H0_kms_Mpc=70.0,      # Hubble constant
        alpha0_scale=1.0,     # Match H0/c at small z
        phi0_over_c2=1e-5,    # Typical LSS potential
        K0=1.0,               # Coherence amplitude
        tau_Gyr=14.0,         # Coherence evolution time
        Dtheta_per_Mpc=9.32e-4  # Path-wandering diffusion
    )
    
    print("\nParameters:")
    print(f"  Coherence: ℓ_0 = {params.ell0_kpc} kpc, p = {params.p}, n_coh = {params.ncoh}")
    print(f"  H0 = {params.H0_kms_Mpc} km/s/Mpc")
    print(f"  Tired-light: α_0 scale = {params.alpha0_scale}")
    print(f"  ISW-like: φ_0/c² = {params.phi0_over_c2}, K_0 = {params.K0}, τ = {params.tau_Gyr} Gyr")
    print(f"  Path-wandering: D_θ = {params.Dtheta_per_Mpc:.3e} rad²/Mpc")
    
    # Compute all mechanisms
    print("\nComputing redshift curves...")
    results = srs.demo_curves(distances_Mpc, params)
    
    # Create DataFrame
    df = pd.DataFrame({
        'D_Mpc': distances_Mpc,
        'z_tired': results['z_tired'],
        'z_isw': results['z_isw'],
        'z_geom': results['z_geom'],
        'z_hubble': results['z_hubble']
    })
    
    # Save to CSV
    output_file = COSMO_DIR / "outputs" / "sigma_redshift_static_exploration.csv"
    output_file.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("REDSHIFT AT KEY DISTANCES")
    print("="*80)
    
    test_distances = [10, 100, 500, 1000, 2000, 3000]
    print(f"\n{'D (Mpc)':>8} | {'z_tired':>10} | {'z_isw':>10} | {'z_geom':>10} | {'z_hubble':>10}")
    print("-"*80)
    for D in test_distances:
        idx = np.argmin(np.abs(distances_Mpc - D))
        print(f"{D:8.0f} | {df.loc[idx, 'z_tired']:10.6f} | "
              f"{df.loc[idx, 'z_isw']:10.6f} | "
              f"{df.loc[idx, 'z_geom']:10.6f} | "
              f"{df.loc[idx, 'z_hubble']:10.6f}")
    
    # Analyze contributions
    print("\n" + "="*80)
    print("MECHANISM ANALYSIS")
    print("="*80)
    
    # At 1000 Mpc (z ~ 0.23)
    idx_1000 = np.argmin(np.abs(distances_Mpc - 1000))
    z_t = df.loc[idx_1000, 'z_tired']
    z_i = df.loc[idx_1000, 'z_isw']
    z_g = df.loc[idx_1000, 'z_geom']
    z_h = df.loc[idx_1000, 'z_hubble']
    
    print(f"\nAt D = 1000 Mpc:")
    print(f"  (A) Tired-light:   z = {z_t:.6f}  ({z_t/z_h*100:6.2f}% of Hubble)")
    print(f"  (B) ISW-like:      z = {z_i:.6f}  ({z_i/z_h*100:6.2f}% of Hubble)")
    print(f"  (C) Path-wandering: z = {z_g:.6f}  ({z_g/z_h*100:6.2f}% of Hubble)")
    print(f"  Reference Hubble:  z = {z_h:.6f}")
    
    # Check if mechanisms can sum to match Hubble
    z_total = z_t + z_i + z_g
    print(f"\n  Combined (A+B+C):  z = {z_total:.6f}  ({z_total/z_h*100:6.2f}% of Hubble)")
    
    # Slopes at low z
    idx_10 = np.argmin(np.abs(distances_Mpc - 10))
    idx_20 = np.argmin(np.abs(distances_Mpc - 20))
    d_10 = df.loc[idx_10, 'D_Mpc']
    d_20 = df.loc[idx_20, 'D_Mpc']
    
    slope_tired = (df.loc[idx_20, 'z_tired'] - df.loc[idx_10, 'z_tired']) / (d_20 - d_10)
    slope_hubble = (df.loc[idx_20, 'z_hubble'] - df.loc[idx_10, 'z_hubble']) / (d_20 - d_10)
    
    print(f"\nLow-z slope (10-20 Mpc):")
    print(f"  Tired-light: dz/dD = {slope_tired:.6e} Mpc⁻¹")
    print(f"  Hubble:      dz/dD = {slope_hubble:.6e} Mpc⁻¹")
    print(f"  Ratio:       {slope_tired/slope_hubble:.4f}")
    
    # Create plots
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: All mechanisms together
    ax = axes[0, 0]
    ax.plot(distances_Mpc, results['z_hubble'], 'k--', label='Hubble (reference)', linewidth=2)
    ax.plot(distances_Mpc, results['z_tired'], label='(A) Tired-light', linewidth=2)
    ax.plot(distances_Mpc, results['z_isw'], label='(B) ISW-like', linewidth=2)
    ax.plot(distances_Mpc, results['z_geom'], label='(C) Path-wandering', linewidth=2)
    ax.set_xlabel('Distance (Mpc)')
    ax.set_ylabel('Redshift z')
    ax.set_title('All Sigma-Gravity Redshift Mechanisms')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Zoom to low-z
    ax = axes[0, 1]
    mask = distances_Mpc <= 500
    ax.plot(distances_Mpc[mask], results['z_hubble'][mask], 'k--', label='Hubble', linewidth=2)
    ax.plot(distances_Mpc[mask], results['z_tired'][mask], label='(A) Tired-light', linewidth=2)
    ax.plot(distances_Mpc[mask], results['z_isw'][mask], label='(B) ISW-like', linewidth=2)
    ax.plot(distances_Mpc[mask], results['z_geom'][mask], label='(C) Path-wandering', linewidth=2)
    ax.set_xlabel('Distance (Mpc)')
    ax.set_ylabel('Redshift z')
    ax.set_title('Low-z Regime (< 500 Mpc)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Fractional contributions
    ax = axes[1, 0]
    frac_tired = results['z_tired'] / results['z_hubble']
    frac_isw = results['z_isw'] / results['z_hubble']
    frac_geom = results['z_geom'] / results['z_hubble']
    ax.plot(distances_Mpc, frac_tired, label='(A) Tired-light', linewidth=2)
    ax.plot(distances_Mpc, frac_isw, label='(B) ISW-like', linewidth=2)
    ax.plot(distances_Mpc, frac_geom, label='(C) Path-wandering', linewidth=2)
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='100% of Hubble')
    ax.set_xlabel('Distance (Mpc)')
    ax.set_ylabel('z_mechanism / z_hubble')
    ax.set_title('Fractional Contribution (relative to Hubble)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Combined vs Hubble
    ax = axes[1, 1]
    z_combined = results['z_tired'] + results['z_isw'] + results['z_geom']
    ax.plot(distances_Mpc, results['z_hubble'], 'k--', label='Hubble', linewidth=2)
    ax.plot(distances_Mpc, z_combined, 'r-', label='(A+B+C) Combined', linewidth=2)
    ax.plot(distances_Mpc, results['z_tired'], ':', alpha=0.5, label='(A) alone')
    ax.set_xlabel('Distance (Mpc)')
    ax.set_ylabel('Redshift z')
    ax.set_title('Combined Mechanisms vs Hubble')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = COSMO_DIR / "outputs" / "sigma_redshift_static_exploration.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {plot_file}")
    
    # Additional analysis: What parameter changes would match Hubble?
    print("\n" + "="*80)
    print("PARAMETER SENSITIVITY")
    print("="*80)
    
    print("\nTo match Hubble law at 1000 Mpc (z = {:.6f}):".format(z_h))
    print(f"\n(A) Tired-light:")
    print(f"    Current α_0 scale = {params.alpha0_scale:.2f} → z = {z_t:.6f}")
    needed_scale = z_h / z_t * params.alpha0_scale
    print(f"    Needed α_0 scale ≈ {needed_scale:.2f} for z = {z_h:.6f}")
    
    print(f"\n(B) ISW-like:")
    print(f"    Current K_0 = {params.K0:.2f} → z = {z_i:.6f}")
    needed_K0 = z_h / z_i * params.K0 if z_i > 0 else float('inf')
    print(f"    Needed K_0 ≈ {needed_K0:.2f} for z = {z_h:.6f}")
    
    print(f"\n(C) Path-wandering:")
    print(f"    Current D_θ = {params.Dtheta_per_Mpc:.3e} rad²/Mpc → z = {z_g:.6f}")
    needed_Dtheta = z_h / z_g * params.Dtheta_per_Mpc
    print(f"    Needed D_θ ≈ {needed_Dtheta:.3e} rad²/Mpc for z = {z_h:.6f}")
    
    # Image blurring constraint for path-wandering
    print(f"\n  ** Path-wandering constraint:")
    print(f"     D_θ = {params.Dtheta_per_Mpc:.3e} rad²/Mpc")
    rms_deflection_per_Gpc = np.sqrt(params.Dtheta_per_Mpc * 1000) * (180/np.pi)  # degrees
    print(f"     RMS deflection per Gpc ≈ {rms_deflection_per_Gpc:.2f} degrees")
    print(f"     (This would blur images significantly!)")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nKey findings:")
    print("1. Tired-light (A) can match Hubble slope at low-z with α_0 ~ H0/c")
    print("2. ISW-like (B) is subdominant with typical parameters")
    print("3. Path-wandering (C) requires large deflections (image blurring)")
    print("4. Mechanism (A) is most promising for static-universe redshift")
    print("\nNext steps:")
    print("- Test against SNe Ia Hubble diagram")
    print("- Check time-dilation (SN light curves must stretch as 1+z)")
    print("- Check surface brightness dimming ((1+z)^4)")
    print("- Test against CMB blackbody spectrum")
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()

