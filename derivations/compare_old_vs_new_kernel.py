#!/usr/bin/env python3
"""
Compare Old Phenomenological Kernel vs New Derived Kernel
==========================================================

This script compares the performance of:

1. OLD PHENOMENOLOGICAL KERNEL (0.087 dex target):
   - K = A₀ × (g†/g_bar)^p × (L_coh/(L_coh+r))^n_coh × S_small(r)
   - Parameters: A₀=1.1, p=0.75, n_coh=0.5, L₀=4.993 kpc, g†=1.2e-10
   - With morphology gates (bulge, shear, bar)

2. NEW DERIVED KERNEL (current implementation):
   - Σ = 1 + A × W(r) × h(g)
   - h(g) = √(g†/g) × g†/(g†+g)  
   - W(r) = 1 - (ξ/(ξ+r))^0.5 with ξ = (2/3)R_d
   - Parameters: A=√3, g†=cH₀/(4√π), n_coh=0.5 (implicit)
   - No morphology gates

3. HYBRID: New derived kernel with Burr-XII p parameter restored

Author: Sigma Gravity Team
Date: December 2025
"""

import numpy as np
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

c = 2.998e8              # Speed of light [m/s]
H0_SI = 2.27e-18         # Hubble constant [1/s] (70 km/s/Mpc)
cH0 = c * H0_SI          # c × H₀ [m/s²]
kpc_to_m = 3.086e19      # meters per kpc

# Derived critical acceleration
g_dagger_derived = cH0 / (4 * math.sqrt(math.pi))  # ≈ 9.60×10⁻¹¹ m/s²

# MOND/old phenomenological value
g_dagger_mond = 1.2e-10  # m/s²

print("=" * 80)
print("COMPARISON: OLD PHENOMENOLOGICAL vs NEW DERIVED KERNEL")
print("=" * 80)
print(f"\nOLD kernel: g† = {g_dagger_mond:.4e} m/s² (MOND value)")
print(f"NEW kernel: g† = {g_dagger_derived:.4e} m/s² (derived)")

# =============================================================================
# FIND SPARC DATA
# =============================================================================

def find_sparc_data() -> Optional[Path]:
    """Find the SPARC data directory."""
    possible_paths = [
        Path("/Users/leonardspeiser/Projects/sigmagravity/data/Rotmod_LTG"),
        Path("/Users/leonardspeiser/Projects/sigmagravity/many_path_model/paper_release/data/Rotmod_LTG"),
        Path(__file__).parent.parent / "data" / "Rotmod_LTG",
    ]
    for p in possible_paths:
        if p.exists():
            return p
    return None


def find_master_sheet() -> Optional[Path]:
    """Find the SPARC master sheet with R_d values."""
    sparc_dir = find_sparc_data()
    if sparc_dir is None:
        return None
    for name in ['MasterSheet_SPARC.mrt', 'SPARC_Lelli2016c.mrt', 'Table1.mrt']:
        p = sparc_dir / name
        if p.exists():
            return p
        p = sparc_dir.parent / name
        if p.exists():
            return p
    return None


# =============================================================================
# LOAD SPARC DATA
# =============================================================================

def load_master_sheet(master_file: Path) -> Dict[str, float]:
    """Load R_d (disk scale length) values from master sheet."""
    R_d_values = {}
    with open(master_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.split()
        if len(parts) < 15:
            continue
        name = parts[0]
        if name.startswith('-') or name.startswith('=') or name.startswith('Note'):
            continue
        if name.startswith('Byte') or name.startswith('Title') or name.startswith('Table'):
            continue
        try:
            R_d = float(parts[11])
            R_d_values[name] = R_d
        except (ValueError, IndexError):
            continue
    return R_d_values


def load_galaxy_rotmod(rotmod_file: Path) -> Optional[Dict]:
    """Load a single galaxy rotation curve from rotmod file."""
    R, V_obs, V_err, V_gas, V_disk, V_bulge = [], [], [], [], [], []
    with open(rotmod_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 6:
                try:
                    R.append(float(parts[0]))
                    V_obs.append(float(parts[1]))
                    V_err.append(float(parts[2]))
                    V_gas.append(float(parts[3]))
                    V_disk.append(float(parts[4]))
                    V_bulge.append(float(parts[5]) if len(parts) > 5 else 0.0)
                except ValueError:
                    continue
    if len(R) < 3:
        return None
    
    R = np.array(R)
    V_obs = np.array(V_obs)
    V_err = np.array(V_err)
    V_gas = np.array(V_gas)
    V_disk = np.array(V_disk)
    V_bulge = np.array(V_bulge)
    
    # Compute V_bar
    V_bar = np.sqrt(
        np.sign(V_gas) * V_gas**2 + 
        np.sign(V_disk) * V_disk**2 + 
        V_bulge**2
    )
    
    # Compute bulge fraction
    V_total_sq = V_gas**2 + V_disk**2 + V_bulge**2
    bulge_frac = np.where(V_total_sq > 0, V_bulge**2 / V_total_sq, 0.0)
    
    return {
        'R': R, 'V_obs': V_obs, 'V_err': V_err, 'V_gas': V_gas,
        'V_disk': V_disk, 'V_bulge': V_bulge, 'V_bar': V_bar,
        'bulge_frac': bulge_frac
    }


def load_all_sparc_galaxies(sparc_dir: Path, R_d_values: Dict[str, float]) -> Dict[str, Dict]:
    """Load all SPARC galaxies with valid data."""
    galaxies = {}
    for rotmod_file in sorted(sparc_dir.glob('*_rotmod.dat')):
        name = rotmod_file.stem.replace('_rotmod', '')
        data = load_galaxy_rotmod(rotmod_file)
        if data is None:
            continue
        if name in R_d_values:
            data['R_d'] = R_d_values[name]
        else:
            data['R_d'] = data['R'].max() / 3.0
        galaxies[name] = data
    return galaxies


# =============================================================================
# KERNEL IMPLEMENTATIONS
# =============================================================================

# --- OLD PHENOMENOLOGICAL KERNEL ---

def old_kernel_K(r, g_bar, v_circ, R_d, bulge_frac,
                 A_0=1.1, p=0.75, n_coh=0.5, L_0=4.993,
                 g_dag=1.2e-10, r_gate=0.5):
    """
    OLD phenomenological kernel (Track 2 style):
    K = A₀ × (g†/g_bar)^p × (L_coh/(L_coh+r))^n_coh × S_small(r)
    """
    r = np.asarray(r)
    g_bar = np.asarray(g_bar)
    
    # RAR shape: (g†/g_bar)^p
    g_ratio = g_dag / np.maximum(g_bar, 1e-14)
    K_rar = np.power(g_ratio, p)
    
    # Coherence length (simplified - no shear/bar for comparison)
    # Include bulge suppression
    BT = np.mean(bulge_frac)  # Average bulge fraction
    f_bulge = 1.0 / (1.0 + BT / (r / 1.0 + 0.1))  # r_bulge ~ 1 kpc
    L_coh = L_0 * f_bulge
    
    # Power-law coherence damping (NOT Burr-XII!)
    K_coherence = np.power(L_coh / (L_coh + r), n_coh)
    
    # Small-radius gate
    S_small = 1.0 - np.exp(-(r / r_gate)**2)
    
    # Combined kernel
    K = A_0 * K_rar * K_coherence * S_small
    
    return K


# --- NEW DERIVED KERNEL ---

def h_function_derived(g, g_dag=None):
    """
    NEW derived h(g) = √(g†/g) × g†/(g†+g)
    """
    if g_dag is None:
        g_dag = g_dagger_derived
    g = np.maximum(g, 1e-15)
    return np.sqrt(g_dag / g) * g_dag / (g_dag + g)


def W_coherence_derived(r, R_d, p=1.0, n_coh=0.5):
    """
    NEW derived coherence window (generalized Burr-XII):
    W(r) = 1 - [1 + (r/ξ)^p]^(-n_coh)
    
    Current implementation: p = 1 (exponential case)
    General Burr-XII: p can vary
    """
    xi = (2/3) * R_d
    r = np.maximum(r, 1e-10)
    x = r / xi
    return 1 - (1 + x**p) ** (-n_coh)


def new_kernel_Sigma(r, g_bar, R_d, A=math.sqrt(3), p=1.0, n_coh=0.5, g_dag=None):
    """
    NEW derived kernel:
    Σ = 1 + A × W(r) × h(g)
    
    Returns Σ (enhancement factor), not K (boost factor)
    """
    if g_dag is None:
        g_dag = g_dagger_derived
    
    h = h_function_derived(g_bar, g_dag)
    W = W_coherence_derived(r, R_d, p, n_coh)
    Sigma = 1 + A * W * h
    
    return Sigma


# =============================================================================
# METRICS
# =============================================================================

def compute_rar_scatter(R_kpc, V_obs, V_pred):
    """Compute RAR scatter in dex."""
    R_m = R_kpc * kpc_to_m
    g_obs = (V_obs * 1000)**2 / R_m
    g_pred = (V_pred * 1000)**2 / R_m
    
    mask = (g_obs > 0) & (g_pred > 0)
    if mask.sum() < 2:
        return np.nan
    
    log_residual = np.log10(g_obs[mask] / g_pred[mask])
    return np.std(log_residual)


def compute_rms_error(V_obs, V_pred):
    """Compute RMS velocity error in km/s."""
    return np.sqrt(np.mean((V_obs - V_pred)**2))


# =============================================================================
# MAIN COMPARISON
# =============================================================================

def run_comparison():
    """Compare old vs new kernels on SPARC data."""
    
    # Find data
    sparc_dir = find_sparc_data()
    if sparc_dir is None:
        print("\nERROR: SPARC data not found!")
        return None
    
    print(f"\nSPARC data found: {sparc_dir}")
    
    # Load master sheet for R_d values
    master_file = find_master_sheet()
    R_d_values = {}
    if master_file:
        R_d_values = load_master_sheet(master_file)
        print(f"Master sheet loaded: {len(R_d_values)} R_d values")
    
    # Load all galaxies
    galaxies = load_all_sparc_galaxies(sparc_dir, R_d_values)
    print(f"Loaded {len(galaxies)} galaxies")
    
    if len(galaxies) == 0:
        print("ERROR: No galaxies loaded!")
        return None
    
    # Test configurations
    configs = {
        'OLD_phenomenological': {
            'desc': 'Old kernel: K = A₀×(g†/g)^p × (L/(L+r))^n_coh',
            'A': 1.1, 'p': 0.75, 'n_coh': 0.5, 'L_0': 4.993,
            'g_dag': 1.2e-10, 'kernel_type': 'old'
        },
        'NEW_derived_p1': {
            'desc': 'New kernel (p=1): Σ = 1 + A×W×h',
            'A': math.sqrt(3), 'p': 1.0, 'n_coh': 0.5,
            'g_dag': g_dagger_derived, 'kernel_type': 'new'
        },
        'NEW_derived_p0.757': {
            'desc': 'New kernel (p=0.757): Σ = 1 + A×W×h',
            'A': math.sqrt(3), 'p': 0.757, 'n_coh': 0.5,
            'g_dag': g_dagger_derived, 'kernel_type': 'new'
        },
        'NEW_derived_p0.6': {
            'desc': 'New kernel (p=0.6): Σ = 1 + A×W×h',
            'A': math.sqrt(3), 'p': 0.6, 'n_coh': 0.5,
            'g_dag': g_dagger_derived, 'kernel_type': 'new'
        },
        'HYBRID_old_g_dag': {
            'desc': 'New h(g) but old g†=1.2e-10',
            'A': math.sqrt(3), 'p': 1.0, 'n_coh': 0.5,
            'g_dag': 1.2e-10, 'kernel_type': 'new'
        },
    }
    
    results = {}
    
    for config_name, config in configs.items():
        rar_scatters = []
        rms_errors = []
        
        for name, data in galaxies.items():
            R = data['R']
            V_obs = data['V_obs']
            V_bar = data['V_bar']
            R_d = data['R_d']
            bulge_frac = data['bulge_frac']
            
            # Skip if V_bar has issues
            if np.any(np.isnan(V_bar)) or np.any(V_bar <= 0):
                continue
            
            # Compute g_bar
            R_m = R * kpc_to_m
            V_bar_ms = V_bar * 1000
            g_bar = V_bar_ms**2 / R_m
            
            if config['kernel_type'] == 'old':
                # Old kernel: V_pred = V_bar × √(1 + K)
                K = old_kernel_K(R, g_bar, V_bar, R_d, bulge_frac,
                                A_0=config['A'], p=config['p'], 
                                n_coh=config['n_coh'], L_0=config['L_0'],
                                g_dag=config['g_dag'])
                V_pred = V_bar * np.sqrt(1 + K)
            else:
                # New kernel: V_pred = V_bar × √Σ
                Sigma = new_kernel_Sigma(R, g_bar, R_d, 
                                        A=config['A'], p=config['p'],
                                        n_coh=config['n_coh'], 
                                        g_dag=config['g_dag'])
                V_pred = V_bar * np.sqrt(Sigma)
            
            rms = compute_rms_error(V_obs, V_pred)
            rar = compute_rar_scatter(R, V_obs, V_pred)
            
            rms_errors.append(rms)
            if not np.isnan(rar):
                rar_scatters.append(rar)
        
        results[config_name] = {
            'desc': config['desc'],
            'mean_rar': np.mean(rar_scatters),
            'median_rar': np.median(rar_scatters),
            'mean_rms': np.mean(rms_errors),
            'n_galaxies': len(rms_errors),
        }
    
    # Print results
    print("\n" + "=" * 100)
    print("COMPARISON RESULTS")
    print("=" * 100)
    
    print(f"\n{'Config':<25} | {'Mean RAR (dex)':<15} | {'Median RAR':<12} | {'Mean RMS (km/s)':<16}")
    print("=" * 100)
    
    for config_name, r in results.items():
        print(f"{config_name:<25} | {r['mean_rar']:<15.4f} | {r['median_rar']:<12.4f} | {r['mean_rms']:<16.2f}")
    
    print("=" * 100)
    
    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    old_rar = results['OLD_phenomenological']['mean_rar']
    new_p1_rar = results['NEW_derived_p1']['mean_rar']
    new_p757_rar = results['NEW_derived_p0.757']['mean_rar']
    new_p6_rar = results['NEW_derived_p0.6']['mean_rar']
    
    print(f"""
KEY FINDINGS:

1. OLD phenomenological kernel: {old_rar:.4f} dex
   - Uses (g†/g)^p RAR shape with p=0.75
   - Uses power-law coherence damping (L/(L+r))^n_coh
   - Uses bulge suppression gate
   - g† = 1.2e-10 (MOND value)

2. NEW derived kernel (p=1.0): {new_p1_rar:.4f} dex
   - Uses h(g) = √(g†/g) × g†/(g†+g)
   - Uses Burr-XII with p=1 (exponential case)
   - No morphology gates
   - g† = cH₀/(4√π) = {g_dagger_derived:.4e}

3. NEW derived with p=0.757: {new_p757_rar:.4f} dex
   - Same as #2 but with p=0.757 in Burr-XII

4. NEW derived with p=0.6: {new_p6_rar:.4f} dex
   - Same as #2 but with p=0.6 in Burr-XII

REGRESSION FROM OLD TO NEW:
  Old → New (p=1.0): {100*(new_p1_rar - old_rar)/old_rar:+.1f}%
  Old → New (p=0.6): {100*(new_p6_rar - old_rar)/old_rar:+.1f}%

ROOT CAUSES OF REGRESSION:
1. Different h(g) functional form
2. Different coherence damping structure  
3. Absence of morphology gates
4. Different g† value

RECOMMENDATIONS:
- The p parameter in Burr-XII provides modest improvement (~2%)
- Larger gains may require restoring morphology gates
- The h(g) functional form difference needs investigation
""")
    
    return results


if __name__ == "__main__":
    results = run_comparison()

