#!/usr/bin/env python3
"""
Generate Model Comparison Plots: GR vs Σ-Gravity vs MOND vs Dark Matter
========================================================================

This script generates 7 individual galaxy plots comparing four models:
1. GR (baryonic only) - green dashed
2. Σ-Gravity (derived formula) - blue solid  
3. MOND - red dotted
4. NFW Dark Matter halo - purple dash-dot

Uses the CANONICAL formula (from run_regression.py):
    Σ = 1 + A(D,L) × W(r) × h(g)
    h(g) = √(g†/g) × g†/(g†+g)
    W(r) = r/(ξ+r)  [k=1 for 2D coherence]
    ξ = R_d/(2π)    [one azimuthal wavelength]
    g† = cH₀/(4√π) ≈ 9.60×10⁻¹¹ m/s²
    A₀ = exp(1/2π) ≈ 1.173 for galaxies

Author: Sigma Gravity Team
Date: December 2025 (Updated to canonical formula)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path
import sys

# Publication-quality settings
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 13
rcParams['legend.fontsize'] = 9
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['figure.dpi'] = 150
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'

# Physical constants
c = 2.998e8          # m/s
H0_SI = 2.27e-18     # 1/s (70 km/s/Mpc)
G = 6.674e-11        # m³/kg/s²
kpc_to_m = 3.086e19  # m per kpc

# Derived critical acceleration (canonical formula)
# g† = cH₀/(4√π) - purely geometric derivation
g_dagger = c * H0_SI / (4 * np.sqrt(np.pi))  # ≈ 9.60×10⁻¹¹ m/s²

# Canonical amplitude (from unified formula with D=0)
A_0 = np.exp(1 / (2 * np.pi))  # ≈ 1.173
A_galaxy = A_0  # For 2D disk galaxies

# Coherence scale factor (canonical: ξ = R_d/(2π))
XI_SCALE = 1 / (2 * np.pi)  # ≈ 0.159

# MOND acceleration scale
a0_mond = 1.2e-10

print("=" * 80)
print("GENERATING MODEL COMPARISON PLOTS (Canonical Formula)")
print("=" * 80)
print(f"g† = cH₀/(4√π) = {g_dagger:.4e} m/s²")
print(f"A₀ = exp(1/2π) = {A_0:.4f}")
print(f"ξ = R_d/(2π) = {XI_SCALE:.4f} × R_d")
print(f"a0_MOND = {a0_mond:.4e} m/s²")

# =============================================================================
# UNIFIED FORMULA FUNCTIONS
# =============================================================================

def h_universal(g):
    """Universal h(g) = √(g†/g) × g†/(g†+g)"""
    g = np.maximum(g, 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)

def W_coherence(r, R_d=3.0):
    """Coherence window: W(r) = r/(ξ+r) with ξ = R_d/(2π) [canonical formula]"""
    xi = XI_SCALE * R_d
    xi = max(xi, 0.01)  # Avoid division by zero
    return r / (xi + r)

def Sigma_unified(r, g, R_d=3.0, A=None):
    """Unified enhancement formula: Σ = 1 + A × W(r) × h(g)"""
    if A is None:
        A = A_galaxy
    h = h_universal(g)
    W = W_coherence(r, R_d)
    return 1 + A * W * h

def mond_nu(g):
    """MOND interpolation function (simple form)"""
    g = np.maximum(g, 1e-15)
    return 1 / (1 - np.exp(-np.sqrt(g / a0_mond)))

def nfw_velocity(r, V200=150, c_nfw=10):
    """
    NFW halo velocity contribution.
    
    V200: circular velocity at r200 in km/s
    c_nfw: concentration parameter
    r: radius in kpc
    
    Returns V_halo in km/s
    """
    # Scale radius in kpc (typical for V200=150 km/s)
    r200 = 200  # kpc (approximate)
    rs = r200 / c_nfw
    
    x = r / rs
    
    # NFW enclosed mass profile: M(<r) ∝ [ln(1+x) - x/(1+x)]
    # V² = GM(<r)/r
    f_x = np.log(1 + x) - x / (1 + x)
    f_c = np.log(1 + c_nfw) - c_nfw / (1 + c_nfw)
    
    # Normalize to V200 at r200
    V_halo = V200 * np.sqrt(f_x / x / (f_c / c_nfw))
    
    return V_halo

# =============================================================================
# LOAD SPARC DATA
# =============================================================================

def load_sparc_data():
    """Load SPARC galaxy data and R_d values."""
    script_dir = Path(__file__).resolve().parent.parent
    sparc_dir = script_dir / "data" / "Rotmod_LTG"
    master_file = script_dir / "data" / "SPARC_Lelli2016c.mrt"
    
    if not sparc_dir.exists():
        print(f"ERROR: SPARC data not found at {sparc_dir}")
        sys.exit(1)
    
    # Load R_d values from master file
    # Rdisk (disk scale length) is the 12th whitespace-separated field (index 11)
    R_d_values = {}
    if master_file.exists():
        with open(master_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.split()
            # Data lines have at least 15 fields and start with a galaxy name
            if len(parts) < 15:
                continue
            name = parts[0]
            # Skip header/note lines
            if name.startswith('-') or name.startswith('=') or name.startswith('Note'):
                continue
            if name.startswith('Byte') or name.startswith('Title') or name.startswith('Table'):
                continue
            try:
                # Rdisk is the 12th field (index 11)
                R_d = float(parts[11])
                R_d_values[name] = R_d
            except (ValueError, IndexError):
                continue
        print(f"  Loaded R_d for {len(R_d_values)} galaxies from master file")
        # Debug: print a few
        debug_galaxies = ['NGC2403', 'NGC3198', 'DDO154']
        for g in debug_galaxies:
            if g in R_d_values:
                print(f"    {g}: R_d = {R_d_values[g]:.2f} kpc")
    else:
        print(f"  WARNING: Master file not found at {master_file}")
    
    return sparc_dir, R_d_values

def load_galaxy_data(rotmod_file, R_d_values):
    """Load rotation curve data for a single galaxy."""
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
                V_err.append(float(parts[2]))
                V_gas.append(float(parts[3]))
                V_disk.append(float(parts[4]))
                V_bulge.append(float(parts[5]) if len(parts) > 5 else 0.0)
    
    if len(R) < 5:
        return None
    
    R = np.array(R)
    V_obs = np.array(V_obs)
    V_err = np.array(V_err)
    V_gas = np.array(V_gas)
    V_disk = np.array(V_disk)
    V_bulge = np.array(V_bulge)
    
    # Compute V_bar with mass-to-light scaling (matching regression test)
    # M/L_disk = 0.5, M/L_bulge = 0.7 (Lelli+ 2016 standard)
    ML_DISK = 0.5
    ML_BULGE = 0.7
    V_bar_sq = np.abs(V_gas)**2 + ML_DISK * np.abs(V_disk)**2 + ML_BULGE * np.abs(V_bulge)**2
    V_bar = np.sqrt(V_bar_sq)
    V_bar = np.where(np.isnan(V_bar), 0.1, V_bar)
    V_bar = np.maximum(V_bar, 0.1)  # Avoid division by zero
    
    return {
        'name': name,
        'R': R,
        'V_obs': V_obs,
        'V_err': V_err,
        'V_bar': V_bar,
        'R_d': R_d,
        'V_gas': V_gas,
        'V_disk': V_disk,
        'V_bulge': V_bulge
    }

# =============================================================================
# COMPUTE MODEL PREDICTIONS
# =============================================================================

def compute_all_predictions(galaxy):
    """Compute predictions for all four models."""
    R = galaxy['R']
    V_bar = galaxy['V_bar']
    R_d = galaxy['R_d']
    
    # Baryonic acceleration
    g_bar = (V_bar * 1000)**2 / (R * kpc_to_m)
    g_bar = np.maximum(g_bar, 1e-15)
    
    # 1. GR (baryonic only)
    V_gr = V_bar.copy()
    
    # 2. Σ-Gravity
    K = A_galaxy * W_coherence(R, R_d) * h_universal(g_bar)
    V_sigma = V_bar * np.sqrt(1 + K)
    
    # 3. MOND
    nu = mond_nu(g_bar)
    V_mond = V_bar * np.sqrt(nu)
    
    # 4. NFW Dark Matter
    # Fit V200 to roughly match the outer observed velocity
    V_outer = np.median(galaxy['V_obs'][-3:]) if len(galaxy['V_obs']) >= 3 else galaxy['V_obs'][-1]
    V200_fit = V_outer * 0.9  # Rough scaling
    V_halo = nfw_velocity(R, V200=V200_fit, c_nfw=10)
    V_dm = np.sqrt(V_bar**2 + V_halo**2)
    
    return {
        'V_gr': V_gr,
        'V_sigma': V_sigma,
        'V_mond': V_mond,
        'V_dm': V_dm,
        'V_halo': V_halo
    }

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_single_galaxy(galaxy, predictions, output_dir, idx):
    """Generate a single galaxy comparison plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    R = galaxy['R']
    V_obs = galaxy['V_obs']
    V_err = galaxy['V_err']
    name = galaxy['name']
    
    # Data points
    ax.errorbar(R, V_obs, yerr=V_err, fmt='ko', ms=5, capsize=2, 
                label='Observed', zorder=5, alpha=0.8)
    
    # Model predictions
    ax.plot(R, predictions['V_gr'], 'g--', lw=2, label='GR (baryons only)', alpha=0.9)
    ax.plot(R, predictions['V_sigma'], 'b-', lw=2.5, label='Σ-Gravity', zorder=4)
    ax.plot(R, predictions['V_mond'], 'r:', lw=2, label='MOND', alpha=0.9)
    ax.plot(R, predictions['V_dm'], 'm-.', lw=2, label='NFW Dark Matter', alpha=0.9)
    
    # Compute residuals for annotation
    V_sigma = predictions['V_sigma']
    V_mond = predictions['V_mond']
    V_dm = predictions['V_dm']
    
    # RMS residuals (only where V_obs > 10 to avoid low-V noise)
    mask = V_obs > 10
    if np.sum(mask) > 3:
        rms_sigma = np.sqrt(np.mean((V_obs[mask] - V_sigma[mask])**2))
        rms_mond = np.sqrt(np.mean((V_obs[mask] - V_mond[mask])**2))
        rms_dm = np.sqrt(np.mean((V_obs[mask] - V_dm[mask])**2))
        
        textstr = f'RMS residuals:\n  Σ-Gravity: {rms_sigma:.1f} km/s\n  MOND: {rms_mond:.1f} km/s\n  NFW: {rms_dm:.1f} km/s'
        ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    ax.set_xlabel('Radius [kpc]')
    ax.set_ylabel('Rotation Velocity [km/s]')
    ax.set_title(f'{name}: Model Comparison')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    
    # Add R_d annotation
    ax.axvline(x=galaxy['R_d'], color='gray', linestyle=':', alpha=0.5)
    ax.text(galaxy['R_d'], ax.get_ylim()[1]*0.95, f'$R_d$={galaxy["R_d"]:.1f}', 
            fontsize=8, ha='center', va='top', alpha=0.7)
    
    plt.tight_layout()
    outpath = output_dir / f'comparison_{idx:02d}_{name}.png'
    plt.savefig(outpath)
    plt.close()
    print(f"  Saved: {outpath}")
    return outpath

def plot_summary_grid(galaxies, predictions_list, output_dir):
    """Generate a 2x4 grid showing all 7 galaxies + summary stats."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, (galaxy, predictions) in enumerate(zip(galaxies, predictions_list)):
        if idx >= 7:
            break
        ax = axes[idx]
        
        R = galaxy['R']
        V_obs = galaxy['V_obs']
        
        # Simplified plot for grid
        ax.plot(R, V_obs, 'ko', ms=3, alpha=0.6)
        ax.plot(R, predictions['V_gr'], 'g--', lw=1.5, alpha=0.8)
        ax.plot(R, predictions['V_sigma'], 'b-', lw=2)
        ax.plot(R, predictions['V_mond'], 'r:', lw=1.5, alpha=0.8)
        ax.plot(R, predictions['V_dm'], 'm-.', lw=1.5, alpha=0.8)
        
        ax.set_title(galaxy['name'], fontsize=11)
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.2)
        
        if idx >= 4:
            ax.set_xlabel('R [kpc]', fontsize=9)
        if idx % 4 == 0:
            ax.set_ylabel('V [km/s]', fontsize=9)
    
    # 8th panel: legend and summary
    ax = axes[7]
    ax.axis('off')
    
    # Create legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=6, label='Observed'),
        Line2D([0], [0], color='g', linestyle='--', lw=2, label='GR (baryons)'),
        Line2D([0], [0], color='b', linestyle='-', lw=2, label='Σ-Gravity'),
        Line2D([0], [0], color='r', linestyle=':', lw=2, label='MOND'),
        Line2D([0], [0], color='m', linestyle='-.', lw=2, label='NFW Dark Matter'),
    ]
    ax.legend(handles=legend_elements, loc='center', fontsize=11, frameon=True)
    
    # Add formula
    ax.text(0.5, 0.25, r'$\Sigma = 1 + A_0 \cdot W(r) \cdot h(g)$', 
            transform=ax.transAxes, fontsize=12, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax.text(0.5, 0.1, r'$g^\dagger = cH_0/(4\sqrt{\pi}) \approx 9.6 \times 10^{-11}$ m/s²', 
            transform=ax.transAxes, fontsize=10, ha='center')
    
    plt.suptitle('SPARC Galaxy Rotation Curves: Four-Model Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    outpath = output_dir / 'comparison_grid_all.png'
    plt.savefig(outpath)
    plt.close()
    print(f"  Saved: {outpath}")

# =============================================================================
# MAIN
# =============================================================================

def compute_galaxy_stats(galaxy, predictions):
    """Compute RMS residuals and RAR scatter for each model."""
    R = galaxy['R']
    V_obs = galaxy['V_obs']
    V_bar = galaxy['V_bar']
    mask = (V_obs > 10) & (V_bar > 5)  # Quality cut
    
    if np.sum(mask) < 5:
        return None
    
    V_obs_m = V_obs[mask]
    V_bar_m = V_bar[mask]
    R_m = R[mask]
    
    # Velocity RMS (km/s)
    rms_sigma = np.sqrt(np.mean((V_obs_m - predictions['V_sigma'][mask])**2))
    rms_mond = np.sqrt(np.mean((V_obs_m - predictions['V_mond'][mask])**2))
    rms_gr = np.sqrt(np.mean((V_obs_m - predictions['V_gr'][mask])**2))
    rms_dm = np.sqrt(np.mean((V_obs_m - predictions['V_dm'][mask])**2))
    
    # RAR scatter (dex) - this is what we claim in the paper
    # g = V^2/R, so log(g_obs/g_pred) = 2*log(V_obs/V_pred)
    g_obs = (V_obs_m * 1000)**2 / (R_m * kpc_to_m)
    g_sigma = (predictions['V_sigma'][mask] * 1000)**2 / (R_m * kpc_to_m)
    g_mond = (predictions['V_mond'][mask] * 1000)**2 / (R_m * kpc_to_m)
    
    # Avoid log of zero/negative
    valid = (g_obs > 0) & (g_sigma > 0) & (g_mond > 0)
    if np.sum(valid) < 3:
        rar_scatter_sigma = 999
        rar_scatter_mond = 999
    else:
        log_resid_sigma = np.log10(g_obs[valid] / g_sigma[valid])
        log_resid_mond = np.log10(g_obs[valid] / g_mond[valid])
        rar_scatter_sigma = np.std(log_resid_sigma)
        rar_scatter_mond = np.std(log_resid_mond)
    
    # Fractional RMS (normalized by mean V_obs)
    V_mean = np.mean(V_obs_m)
    
    return {
        'rms_sigma': rms_sigma,
        'rms_mond': rms_mond,
        'rms_gr': rms_gr,
        'rms_dm': rms_dm,
        'frac_rms_sigma': rms_sigma / V_mean,
        'frac_rms_mond': rms_mond / V_mean,
        'rar_scatter_sigma': rar_scatter_sigma,
        'rar_scatter_mond': rar_scatter_mond,
        'n_points': np.sum(mask),
        'V_max': np.max(V_obs)
    }


def main():
    script_dir = Path(__file__).resolve().parent.parent
    output_dir = script_dir / "figures" / "model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    sparc_dir, R_d_values = load_sparc_data()
    
    # Process ALL galaxies
    all_galaxies = []
    all_predictions = []
    all_stats = []
    
    print(f"\nProcessing all SPARC galaxies...")
    
    for rotmod_file in sorted(sparc_dir.glob('*_rotmod.dat')):
        name = rotmod_file.stem.replace('_rotmod', '')
        
        galaxy = load_galaxy_data(rotmod_file, R_d_values)
        if galaxy is None:
            continue
        
        predictions = compute_all_predictions(galaxy)
        stats = compute_galaxy_stats(galaxy, predictions)
        
        if stats is None:
            continue
        
        all_galaxies.append(galaxy)
        all_predictions.append(predictions)
        all_stats.append(stats)
    
    print(f"\n  Successfully processed {len(all_galaxies)} galaxies")
    
    # Compute overall statistics
    rms_sigma_all = [s['rms_sigma'] for s in all_stats]
    rms_mond_all = [s['rms_mond'] for s in all_stats]
    frac_sigma_all = [s['frac_rms_sigma'] for s in all_stats]
    frac_mond_all = [s['frac_rms_mond'] for s in all_stats]
    rar_sigma_all = [s['rar_scatter_sigma'] for s in all_stats if s['rar_scatter_sigma'] < 900]
    rar_mond_all = [s['rar_scatter_mond'] for s in all_stats if s['rar_scatter_mond'] < 900]
    
    print(f"\n" + "=" * 80)
    print(f"OVERALL STATISTICS (all {len(all_galaxies)} galaxies)")
    print("=" * 80)
    print(f"  Velocity RMS:")
    print(f"    Σ-Gravity: {np.mean(rms_sigma_all):.1f} km/s (median: {np.median(rms_sigma_all):.1f})")
    print(f"    MOND:       {np.mean(rms_mond_all):.1f} km/s (median: {np.median(rms_mond_all):.1f})")
    print(f"  Fractional RMS:")
    print(f"    Σ-Gravity: {np.mean(frac_sigma_all)*100:.1f}%")
    print(f"    MOND:       {np.mean(frac_mond_all)*100:.1f}%")
    print(f"  RAR Scatter (dex) - paper metric:")
    print(f"    Σ-Gravity: {np.mean(rar_sigma_all):.3f} dex (median: {np.median(rar_sigma_all):.3f})")
    print(f"    MOND:       {np.mean(rar_mond_all):.3f} dex (median: {np.median(rar_mond_all):.3f})")
    
    # Count wins (by RAR scatter)
    sigma_wins_rar = sum(1 for s in all_stats if s['rar_scatter_sigma'] < s['rar_scatter_mond'])
    mond_wins_rar = sum(1 for s in all_stats if s['rar_scatter_mond'] < s['rar_scatter_sigma'])
    print(f"\n  Head-to-head (by RMS): Σ-Gravity wins {sum(1 for s in all_stats if s['rms_sigma'] < s['rms_mond'])}, MOND wins {sum(1 for s in all_stats if s['rms_mond'] < s['rms_sigma'])}")
    print(f"  Head-to-head (by RAR): Σ-Gravity wins {sigma_wins_rar}, MOND wins {mond_wins_rar}")
    
    # Rank galaxies by Σ-Gravity performance (lowest fractional RMS = best)
    ranked = sorted(zip(all_galaxies, all_predictions, all_stats), 
                   key=lambda x: x[2]['frac_rms_sigma'])
    
    print(f"\n" + "=" * 80)
    print("TOP 20 GALAXIES (best Σ-Gravity fit)")
    print("=" * 80)
    print(f"{'Rank':<5} {'Galaxy':<12} {'R_d':<6} {'V_max':<7} {'RMS_Σ':<8} {'RMS_M':<8} {'Frac_Σ':<8} {'Winner':<8}")
    print("-" * 70)
    for i, (gal, pred, stat) in enumerate(ranked[:20]):
        winner = 'Σ' if stat['rms_sigma'] < stat['rms_mond'] else 'MOND'
        print(f"{i+1:<5} {gal['name']:<12} {gal['R_d']:<6.2f} {stat['V_max']:<7.0f} {stat['rms_sigma']:<8.1f} {stat['rms_mond']:<8.1f} {stat['frac_rms_sigma']*100:<7.1f}% {winner:<8}")
    
    print(f"\n" + "=" * 80)
    print("BOTTOM 10 GALAXIES (worst Σ-Gravity fit)")
    print("=" * 80)
    for i, (gal, pred, stat) in enumerate(ranked[-10:]):
        winner = 'Σ' if stat['rms_sigma'] < stat['rms_mond'] else 'MOND'
        idx = len(ranked) - 10 + i + 1
        print(f"{idx:<5} {gal['name']:<12} {gal['R_d']:<6.2f} {stat['V_max']:<7.0f} {stat['rms_sigma']:<8.1f} {stat['rms_mond']:<8.1f} {stat['frac_rms_sigma']*100:<7.1f}% {winner:<8}")
    
    # Select representative galaxies for plotting:
    # - 3 from top performers (Σ-Gravity best)
    # - 2 from middle (average)
    # - 2 where MOND does better (to be fair)
    
    n_total = len(ranked)
    selected_indices = [
        0, 2, 4,  # Top 3 (best Σ fits)
        n_total // 3, n_total // 2,  # Middle
        n_total - 3, n_total - 1  # Where MOND wins
    ]
    
    # Also ensure variety in V_max (different galaxy masses)
    selected = [ranked[i] for i in selected_indices if i < n_total]
    
    print(f"\n" + "=" * 80)
    print("SELECTED 7 REPRESENTATIVE GALAXIES")
    print("=" * 80)
    for i, (gal, pred, stat) in enumerate(selected[:7]):
        winner = 'Σ' if stat['rms_sigma'] < stat['rms_mond'] else 'MOND'
        print(f"  {i+1}. {gal['name']:<12} R_d={gal['R_d']:.2f} V_max={stat['V_max']:.0f} RMS_Σ={stat['rms_sigma']:.1f} RMS_M={stat['rms_mond']:.1f} ({winner} wins)")
    
    # Generate individual plots for selected galaxies
    print(f"\nGenerating plots for {len(selected)} selected galaxies...")
    for idx, (galaxy, predictions, stats) in enumerate(selected[:7]):
        plot_single_galaxy(galaxy, predictions, output_dir, idx + 1)
    
    # Generate summary grid
    print("\nGenerating summary grid...")
    galaxies_sel = [s[0] for s in selected[:7]]
    preds_sel = [s[1] for s in selected[:7]]
    if len(galaxies_sel) >= 7:
        plot_summary_grid(galaxies_sel, preds_sel, output_dir)
    
    # Also generate ALL individual plots in a subdirectory
    all_plots_dir = output_dir / "all_galaxies"
    all_plots_dir.mkdir(exist_ok=True)
    print(f"\nGenerating plots for ALL {len(all_galaxies)} galaxies in {all_plots_dir}...")
    for idx, (galaxy, predictions) in enumerate(zip(all_galaxies, all_predictions)):
        plot_single_galaxy(galaxy, predictions, all_plots_dir, idx + 1)
    
    # Save statistics to CSV
    csv_path = output_dir / "galaxy_statistics.csv"
    with open(csv_path, 'w') as f:
        f.write("Galaxy,R_d,V_max,N_points,RMS_Sigma,RMS_MOND,RMS_GR,RMS_DM,RAR_Sigma,RAR_MOND,Winner_RMS,Winner_RAR\n")
        for gal, pred, stat in zip(all_galaxies, all_predictions, all_stats):
            winner_rms = 'Sigma' if stat['rms_sigma'] < stat['rms_mond'] else 'MOND'
            winner_rar = 'Sigma' if stat['rar_scatter_sigma'] < stat['rar_scatter_mond'] else 'MOND'
            f.write(f"{gal['name']},{gal['R_d']:.3f},{stat['V_max']:.1f},{stat['n_points']},")
            f.write(f"{stat['rms_sigma']:.2f},{stat['rms_mond']:.2f},{stat['rms_gr']:.2f},{stat['rms_dm']:.2f},")
            f.write(f"{stat['rar_scatter_sigma']:.4f},{stat['rar_scatter_mond']:.4f},{winner_rms},{winner_rar}\n")
    print(f"\nSaved statistics to: {csv_path}")
    
    print("\n" + "=" * 80)
    print("DONE!")
    print(f"  - {len(selected)} selected plots in: {output_dir}")
    print(f"  - {len(all_galaxies)} complete plots in: {all_plots_dir}")
    print(f"  - Statistics CSV: {csv_path}")
    print("=" * 80)

if __name__ == '__main__':
    main()
