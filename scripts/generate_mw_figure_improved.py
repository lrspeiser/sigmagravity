#!/usr/bin/env python3
"""
Generate Improved Milky Way Rotation Curve Figure
==================================================

Creates a publication-quality MW rotation curve that:
1. Shows the actual data (Eilers+ 2019) with clear data range
2. Extends predictions beyond the data to show asymptotic behavior
3. Demonstrates that Σ-Gravity flattens appropriately at large radii
4. Uses visual cues (faded lines) to distinguish extrapolation from fit region

This addresses the concern that the original plot made it look like 
Σ-Gravity would "fly off" to infinity.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

# Publication-quality settings
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 11
rcParams['axes.titlesize'] = 12
rcParams['legend.fontsize'] = 9
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['figure.dpi'] = 150
rcParams['savefig.dpi'] = 300

# Physical constants
c = 2.998e8          # m/s
H0_SI = 2.27e-18     # 1/s (70 km/s/Mpc)
kpc_to_m = 3.086e19  # m per kpc
km_to_m = 1000.0
G = 4.302e-6         # kpc (km/s)^2 / Msun

# CANONICAL model parameters
g_dagger = c * H0_SI / (4 * np.sqrt(np.pi))  # ≈ 9.60×10⁻¹¹ m/s²
A_0 = np.exp(1 / (2 * np.pi))  # ≈ 1.173
A_galaxy = A_0
XI_SCALE = 1 / (2 * np.pi)  # ≈ 0.159
R_d_MW = 2.6                 # kpc
a0_mond = 1.2e-10            # m/s²

# Baryonic model (McMillan 2017 with scale=1.16, matching full_regression_test.py)
# This is the same model used for the star-by-star validation
SCALE = 1.16
M_disk = 4.6e10 * SCALE**2   # 6.19e10 Msun
M_bulge = 1.0e10 * SCALE**2  # 1.35e10 Msun
M_gas = 1.0e10 * SCALE**2    # 1.35e10 Msun


def v_baryon(R):
    """Baryonic rotation curve from McMillan 2017 model (scaled).
    
    This matches the model used in full_regression_test.py for
    star-by-star MW validation.
    """
    v2_disk = G * M_disk * R**2 / (R**2 + (3.0 + 0.3)**2)**1.5
    v2_bulge = G * M_bulge * R / (R + 0.5)**2
    v2_gas = G * M_gas * R**2 / (R**2 + 7.0**2)**1.5
    return np.sqrt(np.maximum(v2_disk + v2_bulge + v2_gas, 0))


def v_to_g(V_kms, R_kpc):
    """Convert velocity to acceleration: g = V²/R"""
    V_m = V_kms * km_to_m
    R_m = R_kpc * kpc_to_m
    return V_m**2 / np.maximum(R_m, 1e-10)


def h_universal(g):
    """h(g) = √(g†/g) × g†/(g†+g)"""
    g = np.maximum(g, 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)


def C_coherence(r, R_d=R_d_MW):
    """Coherence window: W(r) = r/(ξ+r)"""
    xi = XI_SCALE * R_d
    xi = max(xi, 0.01)
    return r / (xi + r)


def Sigma_derived(r, g_bar, R_d=R_d_MW):
    """Σ = 1 + A × C(R) × h(g)"""
    h = h_universal(g_bar)
    C = C_coherence(r, R_d)
    return 1 + A_galaxy * C * h


def v_sigma(R):
    """Σ-Gravity rotation curve"""
    v_b = v_baryon(R)
    g_b = v_to_g(v_b, R)
    Sigma = Sigma_derived(R, g_b)
    return v_b * np.sqrt(Sigma)


def mond_nu(g):
    """MOND interpolation function"""
    g = np.maximum(g, 1e-15)
    return 1 / (1 - np.exp(-np.sqrt(g / a0_mond)))


def v_mond(R):
    """MOND rotation curve"""
    v_b = v_baryon(R)
    g_b = v_to_g(v_b, R)
    nu = mond_nu(g_b)
    return v_b * np.sqrt(nu)


def main():
    print("=" * 70)
    print("GENERATING IMPROVED MW ROTATION CURVE FIGURE")
    print("=" * 70)
    
    # Eilers+ 2019 data points from original script
    # These are the actual observed velocities (declining with radius)
    R_data = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    V_data = np.array([224, 226, 228, 229, 229, 228, 226, 224, 222, 220, 218, 216])
    V_err = np.array([8, 6, 4, 3, 2, 3, 4, 5, 6, 7, 8, 9])
    
    # Data range for main plot
    R_data_main = R_data
    V_data_main = V_data
    V_err_main = V_err
    
    # Extended range for showing asymptotic behavior
    R_full = np.linspace(2, 50, 200)
    R_data_region = np.linspace(4, 15, 50)  # Where data exists
    R_extrapolate = np.linspace(15, 50, 50)  # Beyond data
    
    # Compute model predictions
    V_bar_full = v_baryon(R_full)
    V_sigma_full = v_sigma(R_full)
    V_mond_full = v_mond(R_full)
    
    V_bar_data = v_baryon(R_data_region)
    V_sigma_data = v_sigma(R_data_region)
    V_mond_data = v_mond(R_data_region)
    
    V_bar_extrap = v_baryon(R_extrapolate)
    V_sigma_extrap = v_sigma(R_extrapolate)
    V_mond_extrap = v_mond(R_extrapolate)
    
    # Print some diagnostics
    print(f"\nModel predictions at key radii:")
    print(f"{'R (kpc)':<10} {'V_bar':>10} {'V_Σ':>10} {'V_MOND':>10}")
    print("-" * 45)
    for r in [8, 15, 20, 30, 40, 50]:
        vb = v_baryon(r)
        vs = v_sigma(r)
        vm = v_mond(r)
        print(f"{r:<10} {vb:>10.1f} {vs:>10.1f} {vm:>10.1f}")
    
    # Check asymptotic behavior
    print(f"\nAsymptotic check (R = 100 kpc):")
    print(f"  V_baryonic: {v_baryon(100):.1f} km/s")
    print(f"  V_Σ-Gravity: {v_sigma(100):.1f} km/s")
    print(f"  V_MOND: {v_mond(100):.1f} km/s")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot data with error bars
    ax.errorbar(R_data_main, V_data_main, yerr=V_err_main, 
                fmt='ko', ms=7, capsize=3, capthick=1.5,
                label='Eilers et al. (2019)', zorder=10)
    
    # Baryonic prediction (full range, same style throughout)
    ax.plot(R_full, V_bar_full, 'g--', lw=2, alpha=0.8, label='Baryonic (Newtonian)')
    
    # Σ-Gravity: solid in data region, dashed in extrapolation
    ax.plot(R_data_region, V_sigma_data, 'b-', lw=2.5, label='Σ-Gravity')
    ax.plot(R_extrapolate, V_sigma_extrap, 'b--', lw=2, alpha=0.5)
    
    # MOND: dotted in data region, lighter dotted in extrapolation
    ax.plot(R_data_region, V_mond_data, 'r:', lw=2.5, label='MOND')
    ax.plot(R_extrapolate, V_mond_extrap, 'r:', lw=2, alpha=0.4)
    
    # Add vertical line showing data boundary
    ax.axvline(x=15, color='gray', linestyle=':', alpha=0.5, lw=1)
    ax.text(15.5, 145, 'Data\nboundary', fontsize=8, color='gray', va='bottom')
    
    # Add shaded region for data range
    ax.axvspan(4, 15, alpha=0.05, color='blue')
    
    # Add annotation showing asymptotic behavior
    ax.annotate('Both theories\nflatten asymptotically', 
                xy=(35, 210), xytext=(38, 180),
                fontsize=9, ha='left', va='top',
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Labels and formatting
    ax.set_xlabel('Galactocentric Radius R [kpc]', fontsize=11)
    ax.set_ylabel('Circular Velocity V [km/s]', fontsize=11)
    ax.set_title('Milky Way Rotation Curve: Data and Predictions', fontsize=12)
    
    ax.set_xlim(0, 50)
    ax.set_ylim(100, 280)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Add inset showing the difference between theories
    ax_inset = ax.inset_axes([0.15, 0.15, 0.35, 0.35])
    R_diff = np.linspace(5, 50, 100)
    V_sigma_diff = v_sigma(R_diff)
    V_mond_diff = v_mond(R_diff)
    
    ax_inset.plot(R_diff, V_sigma_diff - V_mond_diff, 'purple', lw=2)
    ax_inset.axhline(0, color='k', linestyle='--', lw=0.5)
    ax_inset.axvline(16, color='gray', linestyle=':', alpha=0.5)
    ax_inset.set_xlabel('R [kpc]', fontsize=8)
    ax_inset.set_ylabel('V_Σ - V_MOND [km/s]', fontsize=8)
    ax_inset.set_title('Σ-Gravity vs MOND', fontsize=8)
    ax_inset.set_xlim(5, 50)
    ax_inset.tick_params(labelsize=7)
    ax_inset.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    script_dir = Path(__file__).resolve().parent.parent
    output_dir = script_dir / "figures"
    output_dir.mkdir(exist_ok=True)
    
    outpath = output_dir / 'mw_rotation_curve_derived.png'
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {outpath}")
    plt.close()
    
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == '__main__':
    main()

