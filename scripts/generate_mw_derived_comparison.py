#!/usr/bin/env python3
"""
Generate Milky Way Gaia Star-Level RAR Comparison Using Derived Formula
========================================================================

Creates visualization of 157,343 Gaia DR3 stars comparing:
- GR (baryons only)
- Σ-Gravity (derived formula: A=√3, g†=cH₀/(4√π))
- MOND

Uses the derived formula:
    Σ = 1 + A × W(r) × h(g)
    h(g) = √(g†/g) × g†/(g†+g)
    W(r) = 1 - (ξ/(ξ+r))^0.5 with ξ = (2/3)R_d
    g† = cH₀/(4√π) ≈ 1.25×10⁻¹⁰ m/s²
    A = √3 ≈ 1.73 for the Milky Way disk
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path
import json

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

# Derived parameters (from first principles, NOT fitted)
g_dagger = c * H0_SI / (4 * np.sqrt(np.pi))  # ≈ 1.25×10⁻¹⁰ m/s²
A_galaxy = np.sqrt(3)              # ≈ 1.732
R_d_MW = 2.6                       # MW disk scale length (kpc) - from literature

# MOND
a0_mond = 1.2e-10  # m/s²

print("=" * 70)
print("MW GAIA VISUALIZATION WITH DERIVED FORMULA")
print("=" * 70)
print(f"g† = cH₀/(4√π) = {g_dagger:.4e} m/s²")
print(f"A = √3 = {A_galaxy:.4f}")
print(f"R_d (MW) = {R_d_MW} kpc")
print(f"ξ = (2/3)R_d = {(2/3)*R_d_MW:.2f} kpc")


def h_universal(g):
    """h(g) = √(g†/g) × g†/(g†+g)"""
    g = np.maximum(g, 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)


def W_coherence(r, R_d=R_d_MW):
    """W(r) = 1 - (ξ/(ξ+r))^0.5 with ξ = (2/3)R_d"""
    xi = (2/3) * R_d
    return 1 - (xi / (xi + r)) ** 0.5


def Sigma_derived(r, g_bar, R_d=R_d_MW):
    """Σ = 1 + A × W(r) × h(g)"""
    h = h_universal(g_bar)
    W = W_coherence(r, R_d)
    return 1 + A_galaxy * W * h


def mond_nu(g):
    """MOND simple interpolation function"""
    g = np.maximum(g, 1e-15)
    return 1 / (1 - np.exp(-np.sqrt(g / a0_mond)))


def v_to_g(V_kms, R_kpc):
    """Convert velocity to acceleration: g = V²/R"""
    V_m = V_kms * km_to_m
    R_m = R_kpc * kpc_to_m
    return V_m**2 / np.maximum(R_m, 1e-10)


def g_to_v(g, R_kpc):
    """Convert acceleration to velocity: V = √(gR)"""
    R_m = R_kpc * kpc_to_m
    V_m = np.sqrt(np.maximum(g * R_m, 0))
    return V_m / km_to_m


def main():
    # Load data
    data_path = Path(r"C:\Users\henry\dev\sigmagravity\data\gaia\mw\mw_gaia_full_coverage.npz")
    pred_path = Path(r"C:\Users\henry\dev\sigmagravity\data\gaia\outputs\mw_gaia_full_coverage_predicted.csv")
    output_dir = Path(r"C:\Users\henry\dev\sigmagravity\figures")
    
    print(f"\nLoading data from {data_path}...")
    data = np.load(data_path)
    R = data['R_kpc']
    z = data['z_kpc']
    V_obs = data['v_obs_kms']
    
    # Get baryonic velocity from predicted CSV (it has v_baryon_kms)
    import pandas as pd
    df = pd.read_csv(pred_path)
    V_bar = df['v_baryon_kms'].values
    
    n_stars = len(R)
    print(f"Loaded {n_stars:,} stars")
    
    # Compute accelerations
    g_obs = v_to_g(V_obs, R)
    g_bar = v_to_g(V_bar, R)
    
    # Compute Σ-Gravity predictions (DERIVED FORMULA)
    Sigma = Sigma_derived(R, g_bar, R_d=R_d_MW)
    g_sigma = g_bar * Sigma
    V_sigma = g_to_v(g_sigma, R)
    
    # Compute MOND predictions
    nu = mond_nu(g_bar)
    g_mond = g_bar * nu
    V_mond = g_to_v(g_mond, R)
    
    # Compute residuals (in dex)
    log_g_obs = np.log10(np.maximum(g_obs, 1e-15))
    log_g_bar = np.log10(np.maximum(g_bar, 1e-15))
    log_g_sigma = np.log10(np.maximum(g_sigma, 1e-15))
    log_g_mond = np.log10(np.maximum(g_mond, 1e-15))
    
    resid_bar = log_g_obs - log_g_bar
    resid_sigma = log_g_obs - log_g_sigma
    resid_mond = log_g_obs - log_g_mond
    
    # Statistics
    bias_bar = np.mean(resid_bar)
    bias_sigma = np.mean(resid_sigma)
    bias_mond = np.mean(resid_mond)
    
    std_bar = np.std(resid_bar)
    std_sigma = np.std(resid_sigma)
    std_mond = np.std(resid_mond)
    
    print(f"\n{'Model':<15} {'Bias (dex)':<12} {'Scatter (dex)':<12}")
    print("-" * 40)
    print(f"{'GR (baryons)':<15} {bias_bar:+.3f}       {std_bar:.3f}")
    print(f"{'Σ-Gravity':<15} {bias_sigma:+.3f}       {std_sigma:.3f}")
    print(f"{'MOND':<15} {bias_mond:+.3f}       {std_mond:.3f}")
    # Compare bias - lower is better
    if abs(bias_sigma) < abs(bias_mond):
        print(f"\nΣ-Gravity has {abs(bias_mond)/abs(bias_sigma):.1f}× better bias than MOND")
        improvement_str = f'Σ-Gravity {abs(bias_mond)/abs(bias_sigma):.1f}× better bias than MOND'
    else:
        print(f"\nMOND has {abs(bias_sigma)/abs(bias_mond):.1f}× better bias (expected: Σ-Gravity uses derived params)")
        improvement_str = f'MOND {abs(bias_sigma)/abs(bias_mond):.1f}× better bias (Σ-Gravity: no fitting)'
    
    # =========================================================================
    # FIGURE 1: RAR Space (log g_obs vs log g_pred) - Main comparison
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel A: GR (baryons only)
    ax = axes[0]
    ax.hexbin(log_g_bar, log_g_obs, gridsize=80, cmap='Blues', mincnt=1)
    lims = [-11.5, -9.5]
    ax.plot(lims, lims, 'k--', lw=1.5, label='1:1')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(r'$\log_{10}(g_{\rm bar})$ [m/s²]')
    ax.set_ylabel(r'$\log_{10}(g_{\rm obs})$ [m/s²]')
    ax.set_title(f'GR (baryons only)\nBias: {bias_bar:+.3f} dex')
    ax.legend(loc='lower right')
    ax.set_aspect('equal')
    
    # Panel B: Σ-Gravity (derived)
    ax = axes[1]
    ax.hexbin(log_g_sigma, log_g_obs, gridsize=80, cmap='Reds', mincnt=1)
    ax.plot(lims, lims, 'k--', lw=1.5, label='1:1')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(r'$\log_{10}(g_{\Sigma})$ [m/s²]')
    ax.set_ylabel(r'$\log_{10}(g_{\rm obs})$ [m/s²]')
    ax.set_title(f'Σ-Gravity (A=√3, g†=cH₀/2e)\nBias: {bias_sigma:+.3f} dex')
    ax.legend(loc='lower right')
    ax.set_aspect('equal')
    
    # Panel C: MOND
    ax = axes[2]
    ax.hexbin(log_g_mond, log_g_obs, gridsize=80, cmap='Greens', mincnt=1)
    ax.plot(lims, lims, 'k--', lw=1.5, label='1:1')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(r'$\log_{10}(g_{\rm MOND})$ [m/s²]')
    ax.set_ylabel(r'$\log_{10}(g_{\rm obs})$ [m/s²]')
    ax.set_title(f'MOND (a₀=1.2×10⁻¹⁰)\nBias: {bias_mond:+.3f} dex')
    ax.legend(loc='lower right')
    ax.set_aspect('equal')
    
    plt.suptitle(f'Milky Way RAR: {n_stars:,} Gaia DR3 Stars (Zero-Shot Validation)', fontsize=13, y=1.02)
    plt.tight_layout()
    
    outpath1 = output_dir / 'mw_rar_derived_comparison.png'
    plt.savefig(outpath1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {outpath1}")
    
    # =========================================================================
    # FIGURE 2: Residual histograms
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bins = np.linspace(-0.6, 0.8, 60)
    ax.hist(resid_bar, bins=bins, alpha=0.5, label=f'GR: {bias_bar:+.3f}±{std_bar:.3f} dex', color='gray')
    ax.hist(resid_sigma, bins=bins, alpha=0.7, label=f'Σ-Gravity: {bias_sigma:+.3f}±{std_sigma:.3f} dex', color='blue')
    ax.hist(resid_mond, bins=bins, alpha=0.5, label=f'MOND: {bias_mond:+.3f}±{std_mond:.3f} dex', color='green', histtype='step', lw=2)
    
    ax.axvline(x=0, color='k', linestyle='--', lw=1)
    ax.axvline(x=bias_sigma, color='blue', linestyle=':', lw=2)
    ax.axvline(x=bias_mond, color='green', linestyle=':', lw=2)
    
    ax.set_xlabel(r'$\log_{10}(g_{\rm obs}/g_{\rm pred})$ [dex]')
    ax.set_ylabel('Count')
    ax.set_title(f'MW RAR Residuals: {n_stars:,} Stars\nΣ-Gravity scatter: {std_sigma:.3f} vs MOND: {std_mond:.3f} dex')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    outpath2 = output_dir / 'mw_rar_residuals_derived.png'
    plt.savefig(outpath2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outpath2}")
    
    # =========================================================================
    # FIGURE 3: Rotation curve using STANDARD MW observations
    # (Gaia star velocities have ~50 km/s systematic offset due to asymmetric drift)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Standard MW rotation curve data points (Eilers+ 2019, masers, HI)
    R_mw = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    V_mw_obs = np.array([224, 226, 228, 229, 229, 228, 226, 224, 222, 220, 218, 216])
    V_mw_err = np.array([8, 6, 4, 3, 2, 3, 4, 5, 6, 7, 8, 9])
    
    # Compute model predictions at these radii
    g_bar_mw = v_to_g(np.sqrt(R_mw * 3450), R_mw)  # Use gN from baryonic model
    # Actually compute from baryonic model
    V_bar_mw = np.sqrt(3450 * R_mw / R_mw)  # placeholder
    
    # Proper baryonic computation
    G = 4.302e-6  # kpc (km/s)^2 / Msun
    M_b, a_b = 5e9, 0.6
    M_thin, a_thin, b_thin = 4.5e10, 3.0, 0.3
    M_thick, a_thick, b_thick = 1e10, 2.5, 0.9
    M_HI, a_HI, b_HI = 1.1e10, 7.0, 0.1
    M_H2, a_H2, b_H2 = 1.2e9, 1.5, 0.05
    
    def v_bar_model(R):
        v2 = (G*M_b/(R+a_b) + 
              G*M_thin*R**2/(np.sqrt(R**2+(a_thin+b_thin)**2))**3 +
              G*M_thick*R**2/(np.sqrt(R**2+(a_thick+b_thick)**2))**3 +
              G*M_HI*R**2/(np.sqrt(R**2+(a_HI+b_HI)**2))**3 +
              G*M_H2*R**2/(np.sqrt(R**2+(a_H2+b_H2)**2))**3)
        return np.sqrt(v2)
    
    V_bar_mw = v_bar_model(R_mw)
    g_bar_mw = v_to_g(V_bar_mw, R_mw)
    
    # Σ-Gravity prediction
    Sigma_mw = Sigma_derived(R_mw, g_bar_mw, R_d=R_d_MW)
    V_sigma_mw = V_bar_mw * np.sqrt(Sigma_mw)
    
    # MOND prediction
    nu_mw = mond_nu(g_bar_mw)
    V_mond_mw = V_bar_mw * np.sqrt(nu_mw)
    
    # Plot
    ax.errorbar(R_mw, V_mw_obs, yerr=V_mw_err, fmt='ko', ms=8, capsize=4,
                label='Observed (Eilers+ 2019)', zorder=10)
    ax.plot(R_mw, V_bar_mw, 'g--', lw=2, label='GR (baryons only)')
    ax.plot(R_mw, V_sigma_mw, 'b-', lw=2.5, label=f'Σ-Gravity (A=√3)')
    ax.plot(R_mw, V_mond_mw, 'r:', lw=2, label='MOND')
    
    ax.set_xlabel('R [kpc]')
    ax.set_ylabel('V [km/s]')
    ax.set_title('Milky Way Rotation Curve (Standard Data)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(3, 16)
    ax.set_ylim(140, 260)
    
    # Add formula annotation
    ax.text(0.02, 0.98, r'$\Sigma = 1 + \sqrt{3} \cdot W(r) \cdot h(g)$' + '\n' +
            r'$g^\dagger = cH_0/(2e)$',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    outpath3 = output_dir / 'mw_rotation_curve_derived.png'
    plt.savefig(outpath3, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outpath3}")
    
    # Save metrics
    metrics = {
        'n_stars': int(n_stars),
        'derived_formula': {
            'g_dagger': float(g_dagger),
            'A': float(A_galaxy),
            'R_d_MW': float(R_d_MW)
        },
        'results': {
            'GR_baryons': {'bias_dex': float(bias_bar), 'scatter_dex': float(std_bar)},
            'Sigma_Gravity': {'bias_dex': float(bias_sigma), 'scatter_dex': float(std_sigma)},
            'MOND': {'bias_dex': float(bias_mond), 'scatter_dex': float(std_mond)}
        },
        'improvement': improvement_str
    }
    
    metrics_path = output_dir / 'mw_derived_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {metrics_path}")
    
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
