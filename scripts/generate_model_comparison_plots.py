#!/usr/bin/env python3
"""
Generate Model Comparison Plots: GR vs Σ-Gravity vs MOND vs Dark Matter
========================================================================

This script generates 7 individual galaxy plots comparing four models:
1. GR (baryonic only) - green dashed
2. Σ-Gravity (derived formula) - blue solid  
3. MOND - red dotted
4. NFW Dark Matter halo - purple dash-dot

Uses the derived formula:
    Σ = 1 + A × W(r) × h(g)
    h(g) = √(g†/g) × g†/(g†+g)
    W(r) = 1 - (ξ/(ξ+r))^0.5 with ξ = (2/3)R_d
    g† = cH₀/(2e) ≈ 1.20×10⁻¹⁰ m/s²
    A = √3 for galaxies

Author: Sigma Gravity Team
Date: November 30, 2025
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

# Derived critical acceleration
g_dagger = c * H0_SI / (2 * np.e)  # ≈ 1.20×10⁻¹⁰ m/s²

# Amplitudes
A_galaxy = np.sqrt(3)  # ≈ 1.732

# MOND acceleration scale
a0_mond = 1.2e-10

print("=" * 80)
print("GENERATING MODEL COMPARISON PLOTS")
print("=" * 80)
print(f"g† = cH₀/(2e) = {g_dagger:.4e} m/s²")
print(f"A_galaxy = √3 = {A_galaxy:.4f}")
print(f"a0_MOND = {a0_mond:.4e} m/s²")

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
    sparc_dir = Path(r"C:\Users\henry\dev\sigmagravity\data\Rotmod_LTG")
    master_file = Path(r"C:\Users\henry\dev\sigmagravity\data\Rotmod_LTG\MasterSheet_SPARC.mrt")
    
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
    
    # Compute V_bar
    V_bar_sq = np.abs(V_gas)**2 + np.abs(V_disk)**2 + np.abs(V_bulge)**2
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
    ax.text(0.5, 0.25, r'$\Sigma = 1 + \sqrt{3} \cdot W(r) \cdot h(g)$', 
            transform=ax.transAxes, fontsize=12, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax.text(0.5, 0.1, r'$g^\dagger = cH_0/(2e) \approx 1.2 \times 10^{-10}$ m/s²', 
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

def main():
    output_dir = Path(r"C:\Users\henry\dev\sigmagravity\figures\model_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    sparc_dir, R_d_values = load_sparc_data()
    
    # Select 7 diverse galaxies (mix of masses, sizes, quality)
    target_galaxies = [
        'NGC2403',   # Well-studied spiral, R_d=1.39
        'NGC3198',   # Classic example, R_d=3.14
        'NGC6946',   # Nearby spiral, R_d=2.44
        'DDO154',    # Dwarf irregular, R_d=0.37
        'UGC00128',  # Low surface brightness, R_d=5.95
        'NGC2841',   # Massive spiral, R_d=3.64
        'NGC7331',   # Sb galaxy, R_d=5.02
    ]
    
    galaxies = []
    predictions_list = []
    
    print(f"\nProcessing {len(target_galaxies)} galaxies...")
    
    for rotmod_file in sparc_dir.glob('*_rotmod.dat'):
        name = rotmod_file.stem.replace('_rotmod', '')
        if name not in target_galaxies:
            continue
        
        galaxy = load_galaxy_data(rotmod_file, R_d_values)
        if galaxy is None:
            print(f"  Skipping {name}: insufficient data")
            continue
        
        predictions = compute_all_predictions(galaxy)
        galaxies.append(galaxy)
        predictions_list.append(predictions)
        
        print(f"\n  Processing {name} (R_d = {galaxy['R_d']:.2f} kpc)")
    
    # Sort to match target order
    galaxy_order = {name: i for i, name in enumerate(target_galaxies)}
    sorted_pairs = sorted(zip(galaxies, predictions_list), 
                         key=lambda x: galaxy_order.get(x[0]['name'], 99))
    galaxies = [p[0] for p in sorted_pairs]
    predictions_list = [p[1] for p in sorted_pairs]
    
    # Generate individual plots
    print(f"\nGenerating {len(galaxies)} individual comparison plots...")
    for idx, (galaxy, predictions) in enumerate(zip(galaxies, predictions_list)):
        plot_single_galaxy(galaxy, predictions, output_dir, idx + 1)
    
    # Generate summary grid
    print("\nGenerating summary grid...")
    if len(galaxies) >= 7:
        plot_summary_grid(galaxies[:7], predictions_list[:7], output_dir)
    else:
        print(f"  Only {len(galaxies)} galaxies available, skipping grid")
    
    print("\n" + "=" * 80)
    print("DONE! Generated files in:", output_dir)
    print("=" * 80)

if __name__ == '__main__':
    main()
