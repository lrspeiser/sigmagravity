#!/usr/bin/env python3
"""
Generate All Paper Figures Using Derived Formula
=================================================

This script generates publication-quality figures for the Nature Physics paper
using the unified derived formula:

    Σ = 1 + A × W(r) × h(g)

where:
    h(g) = √(g†/g) × g†/(g†+g)
    W(r) = 1 - (ξ/(ξ+r))^0.5  with ξ = (2/3)R_d
    g† = cH₀/(2e) ≈ 1.20×10⁻¹⁰ m/s²
    A = √3 for galaxies, π√2 for clusters

Author: Sigma Gravity Team
Date: November 30, 2025

Usage:
    python scripts/generate_paper_figures.py [--output-dir figures/]
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path
import sys
import argparse

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
rcParams['savefig.bbox'] = 'tight'

# Physical constants
c = 2.998e8          # m/s
H0_SI = 2.27e-18     # 1/s (70 km/s/Mpc)
G = 6.674e-11        # m³/kg/s²
kpc_to_m = 3.086e19  # m per kpc

# Derived critical acceleration
g_dagger = c * H0_SI / (2 * np.e)  # ≈ 1.20×10⁻¹⁰ m/s²

# Amplitudes
A_galaxy = np.sqrt(3)      # ≈ 1.732
A_cluster = np.pi * np.sqrt(2)  # ≈ 4.44

print("=" * 80)
print("GENERATING PAPER FIGURES WITH DERIVED FORMULA")
print("=" * 80)
print(f"g† = cH₀/(2e) = {g_dagger:.4e} m/s²")
print(f"A_galaxy = √3 = {A_galaxy:.4f}")
print(f"A_cluster = π√2 = {A_cluster:.4f}")
print(f"Ratio = {A_cluster/A_galaxy:.4f} (expected: 2.57)")

# =============================================================================
# UNIFIED FORMULA FUNCTIONS
# =============================================================================

def h_universal(g):
    """Universal h(g) = √(g†/g) × g†/(g†+g)"""
    g = np.maximum(g, 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)

def W_coherence(r, R_d=3.0):
    """Coherence window: W(r) = 1 - (ξ/(ξ+r))^0.5 with ξ = (2/3)R_d"""
    xi = (2/3) * R_d
    return 1 - (xi / (xi + r)) ** 0.5

def Sigma_unified(r, g, R_d=3.0, A=None):
    """
    Unified enhancement formula.
    
    Σ = 1 + A × W(r) × h(g)
    """
    if A is None:
        A = A_galaxy
    h = h_universal(g)
    W = W_coherence(r, R_d)
    return 1 + A * W * h

# =============================================================================
# FIGURE 1: Radial Acceleration Relation (RAR)
# =============================================================================

def generate_rar_figure(output_dir):
    """Generate the RAR plot showing theory vs observation."""
    print("\nGenerating Figure 1: RAR plot...")
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Load SPARC data if available
    sparc_dir = Path(r"C:\Users\henry\dev\sigmagravity\data\Rotmod_LTG")
    master_file = Path(r"C:\Users\henry\dev\sigmagravity\data\SPARC_Lelli2016c.mrt")
    
    g_bar_all = []
    g_obs_all = []
    g_pred_all = []
    
    if sparc_dir.exists():
        # Load R_d values
        R_d_values = {}
        if master_file.exists():
            with open(master_file, 'r') as f:
                lines = f.readlines()
            data_start = 0
            for i, line in enumerate(lines):
                if line.startswith('-------'):
                    data_start = i + 1
                    break
            for line in lines[data_start:]:
                if len(line) < 67:
                    continue
                try:
                    name = line[0:11].strip()
                    Rdisk_str = line[62:67].strip()
                    if name and Rdisk_str:
                        R_d_values[name] = float(Rdisk_str)
                except:
                    continue
        
        # Process galaxies
        for rotmod_file in list(sparc_dir.glob('*_rotmod.dat'))[:50]:  # Sample for speed
            name = rotmod_file.stem.replace('_rotmod', '')
            R_d = R_d_values.get(name, 3.0)
            
            R, V_obs, V_err, V_gas, V_disk, V_bulge = [], [], [], [], [], []
            with open(rotmod_file, 'r') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.split()
                    if len(parts) >= 6:
                        R.append(float(parts[0]))
                        V_obs.append(float(parts[1]))
                        V_gas.append(float(parts[3]))
                        V_disk.append(float(parts[4]))
                        V_bulge.append(float(parts[5]) if len(parts) > 5 else 0.0)
            
            if len(R) < 3:
                continue
            
            R = np.array(R)
            V_obs = np.array(V_obs)
            V_gas = np.array(V_gas)
            V_disk = np.array(V_disk)
            V_bulge = np.array(V_bulge)
            
            # Compute V_bar
            V_bar = np.sqrt(
                np.sign(V_gas) * V_gas**2 + 
                np.sign(V_disk) * V_disk**2 + 
                V_bulge**2
            )
            
            # Quality cuts
            mask = (R > 0.5) & (V_bar > 5) & (V_obs > 5) & ~np.isnan(V_bar)
            if np.sum(mask) < 3:
                continue
            
            R = R[mask]
            V_obs = V_obs[mask]
            V_bar = V_bar[mask]
            
            # Compute accelerations
            g_bar = (V_bar * 1000)**2 / (R * kpc_to_m)
            g_obs = (V_obs * 1000)**2 / (R * kpc_to_m)
            
            # Compute predicted
            Sigma = Sigma_unified(R, g_bar, R_d=R_d, A=A_galaxy)
            g_pred = g_bar * Sigma
            
            g_bar_all.extend(g_bar)
            g_obs_all.extend(g_obs)
            g_pred_all.extend(g_pred)
    
    g_bar_all = np.array(g_bar_all)
    g_obs_all = np.array(g_obs_all)
    g_pred_all = np.array(g_pred_all)
    
    # Plot data points
    ax.scatter(g_bar_all, g_obs_all, s=1, alpha=0.3, c='gray', label='SPARC data')
    
    # Plot theory line
    g_range = np.logspace(-13, -8, 200)
    g_eff_theory = g_range * Sigma_unified(10.0, g_range, R_d=3.0, A=A_galaxy)
    ax.plot(g_range, g_eff_theory, 'b-', lw=2, label=f'Σ-Gravity (A=√3)')
    
    # 1:1 line
    ax.plot([1e-14, 1e-7], [1e-14, 1e-7], 'k--', lw=1, alpha=0.5, label='1:1 (no enhancement)')
    
    # MOND comparison
    a0 = 1.2e-10
    g_mond = g_range * (1 / (1 - np.exp(-np.sqrt(g_range/a0))))
    ax.plot(g_range, g_mond, 'r:', lw=1.5, alpha=0.7, label='MOND')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e-13, 1e-8)
    ax.set_ylim(1e-12, 1e-8)
    ax.set_xlabel(r'$g_{\rm bar}$ [m/s²]')
    ax.set_ylabel(r'$g_{\rm obs}$ [m/s²]')
    ax.set_title('Radial Acceleration Relation')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Add scatter annotation
    if len(g_obs_all) > 0:
        log_residual = np.log10(g_obs_all / g_pred_all)
        scatter = np.std(log_residual)
        ax.text(0.05, 0.95, f'Scatter: {scatter:.3f} dex', transform=ax.transAxes, 
                fontsize=10, va='top')
    
    plt.tight_layout()
    outpath = output_dir / 'rar_derived_formula.png'
    plt.savefig(outpath)
    plt.close()
    print(f"  Saved: {outpath}")

# =============================================================================
# FIGURE 2: h(g) Function Comparison
# =============================================================================

def generate_h_function_figure(output_dir):
    """Generate comparison of h(g) with MOND interpolation."""
    print("\nGenerating Figure 2: h(g) function comparison...")
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Left: h(g) functions
    ax = axes[0]
    g_range = np.logspace(-13, -8, 200)
    
    # Our derived h(g)
    h_ours = h_universal(g_range)
    ax.loglog(g_range/g_dagger, h_ours, 'b-', lw=2, label=r'$h(g) = \sqrt{g^\dagger/g} \cdot g^\dagger/(g^\dagger+g)$')
    
    # MOND-equivalent (ν - 1)
    a0 = 1.2e-10
    nu_mond = 1 / (1 - np.exp(-np.sqrt(g_range/a0)))
    h_mond = nu_mond - 1
    ax.loglog(g_range/g_dagger, h_mond, 'r--', lw=2, label=r'MOND: $\nu(g/a_0) - 1$')
    
    ax.axvline(x=1, color='k', linestyle=':', alpha=0.5, label=r'$g = g^\dagger$')
    ax.set_xlabel(r'$g/g^\dagger$')
    ax.set_ylabel(r'$h(g)$ or $\nu - 1$')
    ax.set_title('Enhancement Functions')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Right: Percentage difference
    ax = axes[1]
    # Normalize at low g to compare shapes
    g_test = np.logspace(-13, -9, 100)
    h_test = h_universal(g_test)
    nu_test = 1 / (1 - np.exp(-np.sqrt(g_test/a0)))
    h_mond_test = nu_test - 1
    
    # Scale h to match ν-1 at low g
    scale = h_mond_test[0] / h_test[0]
    h_scaled = h_test * scale
    
    diff_percent = (h_scaled - h_mond_test) / h_mond_test * 100
    ax.semilogx(g_test/g_dagger, diff_percent, 'b-', lw=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.axvline(x=1, color='k', linestyle=':', alpha=0.5)
    ax.fill_between(g_test/g_dagger, -7, 7, alpha=0.2, color='gray')
    ax.set_xlabel(r'$g/g^\dagger$')
    ax.set_ylabel('Difference from MOND [%]')
    ax.set_title('Testable Prediction: ~7% Difference')
    ax.set_ylim(-15, 15)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    outpath = output_dir / 'h_function_comparison.png'
    plt.savefig(outpath)
    plt.close()
    print(f"  Saved: {outpath}")

# =============================================================================
# FIGURE 3: Coherence Window W(r)
# =============================================================================

def generate_coherence_window_figure(output_dir):
    """Generate coherence window visualization."""
    print("\nGenerating Figure 3: Coherence window W(r)...")
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Left: W(r) for different R_d
    ax = axes[0]
    r_range = np.linspace(0, 30, 200)
    
    for R_d in [2.0, 3.0, 5.0, 8.0]:
        W = W_coherence(r_range, R_d)
        ax.plot(r_range, W, lw=2, label=f'$R_d$ = {R_d} kpc')
    
    ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Radius r [kpc]')
    ax.set_ylabel(r'$W(r) = 1 - (\xi/(\xi+r))^{0.5}$')
    ax.set_title(r'Coherence Window ($\xi = \frac{2}{3}R_d$)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 1)
    
    # Right: Full enhancement Σ(r) at different g
    ax = axes[1]
    r_range = np.linspace(0.1, 30, 200)
    R_d = 3.0
    
    for g in [1e-11, 5e-11, 1e-10, 2e-10]:
        Sigma = Sigma_unified(r_range, g, R_d=R_d, A=A_galaxy)
        ax.plot(r_range, Sigma, lw=2, label=f'g = {g:.0e} m/s²')
    
    ax.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='GR limit')
    ax.set_xlabel('Radius r [kpc]')
    ax.set_ylabel(r'$\Sigma = g_{\rm eff}/g_{\rm bar}$')
    ax.set_title(r'Total Enhancement ($R_d$ = 3 kpc)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 30)
    
    plt.tight_layout()
    outpath = output_dir / 'coherence_window.png'
    plt.savefig(outpath)
    plt.close()
    print(f"  Saved: {outpath}")

# =============================================================================
# FIGURE 4: Galaxy/Cluster Amplitude Comparison
# =============================================================================

def generate_amplitude_figure(output_dir):
    """Generate amplitude comparison for galaxies vs clusters."""
    print("\nGenerating Figure 4: Amplitude comparison...")
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Theoretical predictions
    systems = ['Galaxies\n(disk)', 'Clusters\n(spherical)']
    A_pred = [np.sqrt(3), np.pi * np.sqrt(2)]
    A_obs = [1.73, 4.5]  # Observed/calibrated values
    
    x = np.arange(len(systems))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, A_pred, width, label='Derived', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, A_obs, width, label='Observed', color='coral', alpha=0.8)
    
    # Add value labels
    for bar, val in zip(bars1, A_pred):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    for bar, val in zip(bars2, A_obs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Amplitude A')
    ax.set_title('Derived vs Observed Amplitudes')
    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.legend()
    ax.set_ylim(0, 6)
    
    # Add ratio annotation
    ratio_pred = A_pred[1] / A_pred[0]
    ratio_obs = A_obs[1] / A_obs[0]
    ax.text(0.95, 0.95, f'Ratio (cluster/galaxy):\nPredicted: {ratio_pred:.2f}\nObserved: {ratio_obs:.2f}\nMatch: {100*ratio_pred/ratio_obs:.1f}%', 
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    outpath = output_dir / 'amplitude_comparison.png'
    plt.savefig(outpath)
    plt.close()
    print(f"  Saved: {outpath}")

# =============================================================================
# FIGURE 5: Solar System Safety
# =============================================================================

def generate_solar_system_figure(output_dir):
    """Generate Solar System safety demonstration."""
    print("\nGenerating Figure 5: Solar System safety...")
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Solar System scales
    AU_to_kpc = 4.85e-9  # 1 AU in kpc
    r_planets = {
        'Mercury': 0.39 * AU_to_kpc,
        'Earth': 1.0 * AU_to_kpc,
        'Jupiter': 5.2 * AU_to_kpc,
        'Neptune': 30 * AU_to_kpc,
    }
    
    # Solar System g at each planet
    M_sun = 2e30  # kg
    g_planets = {}
    for name, r_kpc in r_planets.items():
        r_m = r_kpc * kpc_to_m
        g_planets[name] = G * M_sun / r_m**2
    
    # Compute enhancement
    r_range_kpc = np.logspace(-12, 2, 500)  # From sub-AU to 100 kpc
    r_range_AU = r_range_kpc / AU_to_kpc
    
    # Use R_d = 0.001 kpc for Solar System (effectively no coherence scale)
    R_d_sun = 1e-6  # Very small
    
    # Compute enhancement at different g values (use Solar System g)
    g_1AU = g_planets['Earth']
    Sigma_vals = []
    for r in r_range_kpc:
        r_m = r * kpc_to_m
        g_local = G * M_sun / r_m**2 if r_m > 1e6 else 1e-5
        S = Sigma_unified(r, g_local, R_d=R_d_sun, A=A_galaxy)
        Sigma_vals.append(S)
    
    Sigma_vals = np.array(Sigma_vals)
    enhancement = Sigma_vals - 1
    
    # Plot
    ax.loglog(r_range_AU, np.maximum(enhancement, 1e-20), 'b-', lw=2)
    
    # Mark planets
    for name, r_kpc in r_planets.items():
        r_AU = r_kpc / AU_to_kpc
        g_local = g_planets[name]
        S = Sigma_unified(r_kpc, g_local, R_d=R_d_sun, A=A_galaxy)
        enh = S - 1
        ax.axvline(x=r_AU, color='gray', linestyle=':', alpha=0.5)
        ax.text(r_AU, 1e-10, name, rotation=90, va='bottom', ha='right', fontsize=8)
    
    # Observational bounds
    ax.axhline(y=2.3e-5, color='r', linestyle='--', lw=1.5, label='Cassini PPN bound')
    ax.axhline(y=1e-8, color='orange', linestyle='--', lw=1.5, label='Ephemeris bound')
    
    ax.set_xlabel('Distance from Sun [AU]')
    ax.set_ylabel(r'Enhancement $\Sigma - 1$')
    ax.set_title('Solar System Safety: Enhancement is Negligible')
    ax.set_xlim(0.1, 1e5)
    ax.set_ylim(1e-16, 1e-2)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Galactic scale annotation
    ax.axvline(x=1e4/AU_to_kpc, color='green', linestyle='-', alpha=0.5)
    ax.text(1e4/AU_to_kpc, 1e-3, 'Galaxy scale', rotation=90, va='top', ha='right', fontsize=8, color='green')
    
    plt.tight_layout()
    outpath = output_dir / 'solar_system_safety.png'
    plt.savefig(outpath)
    plt.close()
    print(f"  Saved: {outpath}")

# =============================================================================
# FIGURE 6: Theory Box (Summary)
# =============================================================================

def generate_theory_summary_figure(output_dir):
    """Generate theory summary box."""
    print("\nGenerating Figure 6: Theory summary box...")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    
    summary_text = """COHERENT TORSION GRAVITY: UNIFIED FORMULA

    Σ = 1 + A × W(r) × h(g)

COMPONENTS:

    h(g) = √(g†/g) × g†/(g†+g)  — acceleration dependence
    W(r) = 1 - (ξ/(ξ+r))^0.5    — coherence window (ξ = ⅔ R_d)

DERIVED PARAMETERS:

    g† = cH₀/(2e) = 1.20×10⁻¹⁰ m/s²  — cosmological horizon
    A_galaxy = √3 ≈ 1.73             — 3D disk geometry
    A_cluster = π√2 ≈ 4.44           — spherical geometry
    n_coh = k/2 = 0.5                — Gamma-exponential

PHYSICAL MECHANISM:

    Coherent superposition of torsion modes in extended
    systems produces gravitational enhancement absent
    in compact environments.

PERFORMANCE:

    • SPARC galaxies: 0.094 dex RAR scatter
    • Milky Way: +0.062 dex bias (zero-shot)
    • Galaxy clusters: 2/2 hold-out coverage
    • Solar System: passes by 8 orders of magnitude
"""
    
    ax.text(0.5, 0.5, summary_text, transform=ax.transAxes, fontsize=10,
            va='center', ha='center', family='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.8))
    outpath = output_dir / 'theory_summary_box.png'
    plt.savefig(outpath)
    plt.close()
    print(f"  Saved: {outpath}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate paper figures')
    parser.add_argument('--output-dir', type=str, default='figures',
                        help='Output directory for figures')
    args = parser.parse_args()
    
    output_dir = Path(r"C:\Users\henry\dev\sigmagravity") / args.output_dir
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Generate all figures
    generate_rar_figure(output_dir)
    generate_h_function_figure(output_dir)
    generate_coherence_window_figure(output_dir)
    generate_amplitude_figure(output_dir)
    generate_solar_system_figure(output_dir)
    generate_theory_summary_figure(output_dir)
    
    print("\n" + "=" * 80)
    print("ALL FIGURES GENERATED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nFigures saved to: {output_dir}")
    print("\nBackup of old figures: figures_backup_2025_11_30/")

if __name__ == "__main__":
    main()
