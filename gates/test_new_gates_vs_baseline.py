#!/usr/bin/env python3
"""
Test new explicit gate formulas vs baseline on SPARC data

Compares:
- Baseline: Smoothstep gates (0.088 dex verified)
- New: Explicit G_distance, G_acceleration gates

Uses same hyperparameters from config/hyperparams_track2.json
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize
import json

# Add paths
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "many_path_model"))

# Import baseline kernel
from path_spectrum_kernel_track2 import PathSpectrumKernel, PathSpectrumHyperparams

# Import new gates
from gate_core import G_unified, G_solar_system, C_burr_XII

# ============================================================================
# CONFIGURATION
# ============================================================================

# Load hyperparameters
config_path = REPO_ROOT / "config" / "hyperparams_track2.json"
with open(config_path, 'r') as f:
    hyperparams = json.load(f)

print("="*80)
print("NEW GATES TEST vs BASELINE")
print("="*80)
print(f"\nLoaded hyperparameters from: {config_path}")
print(f"  L_0 = {hyperparams['L_0']} kpc")
print(f"  A_0 = {hyperparams['A_0']}")
print(f"  p = {hyperparams['p']}")
print(f"  n_coh = {hyperparams['n_coh']}")
print(f"  g_dagger = {hyperparams['g_dagger']}")

# ============================================================================
# NEW GATE KERNEL
# ============================================================================

class NewGateKernel:
    """Kernel using explicit gate formulas from gate_core.py"""
    
    def __init__(self, L_0, A_0, p, n_coh, g_dagger,
                 R_min=0.01, alpha_R=2.0, beta_R=1.0,
                 g_crit=1e-10, alpha_g=2.0, beta_g=1.0):
        self.L_0 = L_0
        self.A_0 = A_0
        self.p = p
        self.n_coh = n_coh
        self.g_dagger = g_dagger
        
        # Gate parameters
        self.R_min = R_min
        self.alpha_R = alpha_R
        self.beta_R = beta_R
        self.g_crit = g_crit
        self.alpha_g = alpha_g
        self.beta_g = beta_g
        
    def compute_K(self, R_kpc, g_bar):
        """Compute kernel boost K = A_0 * (g_dagger/g_bar)^p * C(R) * G_unified"""
        
        # Coherence window (Burr-XII)
        C = C_burr_XII(R_kpc, self.L_0, self.p, self.n_coh)
        
        # Acceleration ratio
        g_ratio = np.where(g_bar > 0, self.g_dagger / g_bar, 0.0)
        accel_term = g_ratio ** self.p
        
        # Unified gate (distance + acceleration)
        G = G_unified(R_kpc, g_bar, 
                     R_min=self.R_min, alpha_R=self.alpha_R, beta_R=self.beta_R,
                     g_crit=self.g_crit, alpha_g=self.alpha_g, beta_g=self.beta_g)
        
        # Full kernel
        K = self.A_0 * accel_term * C * G
        
        return K

# ============================================================================
# LOAD SPARC DATA
# ============================================================================

def load_sparc_data():
    """Load SPARC rotation curve data"""
    
    data_dir = REPO_ROOT / "data" / "Rotmod_LTG"
    
    # Get all rotation curve files
    rc_files = list(data_dir.glob("*_rotmod.dat"))
    
    galaxies = []
    
    for rc_file in rc_files:
        gal_name = rc_file.stem.replace("_rotmod", "")
        
        try:
            # Read rotation curve
            df = pd.read_csv(rc_file, delim_whitespace=True, comment='#',
                           names=['R', 'V_obs', 'errV', 'V_gas', 'V_disk', 'V_bul'])
            
            # Compute baryonic acceleration
            V_bar = np.sqrt(df['V_gas']**2 + df['V_disk']**2 + df['V_bul']**2)
            g_bar = V_bar**2 / df['R']  # km^2/s^2/kpc -> need to convert
            
            # Convert to m/s^2
            g_bar_SI = g_bar * 1e3**2 / (3.086e19)  # (km/s)^2/kpc -> m/s^2
            
            galaxies.append({
                'name': gal_name,
                'R': df['R'].values,
                'V_obs': df['V_obs'].values,
                'V_bar': V_bar.values,
                'g_bar': g_bar_SI,
                'errV': df['errV'].values
            })
            
        except Exception as e:
            print(f"Skipping {gal_name}: {e}")
            continue
    
    print(f"\nLoaded {len(galaxies)} SPARC galaxies")
    return galaxies

# ============================================================================
# COMPUTE RAR SCATTER
# ============================================================================

def compute_rar_scatter(galaxies, kernel, label="Model"):
    """Compute RAR scatter for given kernel"""
    
    all_g_obs = []
    all_g_model = []
    
    for gal in galaxies:
        R = gal['R']
        V_obs = gal['V_obs']
        g_bar = gal['g_bar']
        
        # Compute observed acceleration
        g_obs = V_obs**2 / R  # km^2/s^2/kpc
        g_obs_SI = g_obs * 1e3**2 / (3.086e19)  # Convert to m/s^2
        
        # Compute model prediction
        if hasattr(kernel, 'compute_K'):
            # New gate kernel
            K = kernel.compute_K(R, g_bar)
        else:
            # Baseline kernel
            K = np.array([kernel.K_gal(r, gb) for r, gb in zip(R, g_bar)])
        
        g_model = g_bar * (1 + K)
        
        all_g_obs.extend(g_obs_SI)
        all_g_model.extend(g_model)
    
    all_g_obs = np.array(all_g_obs)
    all_g_model = np.array(all_g_model)
    
    # Compute scatter in dex
    valid = (all_g_obs > 0) & (all_g_model > 0)
    residuals = np.log10(all_g_model[valid]) - np.log10(all_g_obs[valid])
    scatter = np.std(residuals)
    bias = np.mean(residuals)
    
    print(f"\n{label}:")
    print(f"  Bias: {bias:+.4f} dex")
    print(f"  Scatter: {scatter:.4f} dex")
    print(f"  N_points: {valid.sum()}")
    
    return scatter, bias, all_g_obs[valid], all_g_model[valid]

# ============================================================================
# PARAMETER TUNING
# ============================================================================

def tune_gate_parameters(galaxies, hyperparams, initial_params=None):
    """Optimize gate parameters to minimize RAR scatter"""
    
    if initial_params is None:
        initial_params = {
            'R_min': 0.01,
            'alpha_R': 2.0,
            'beta_R': 1.0,
            'g_crit': 1e-10,
            'alpha_g': 2.0,
            'beta_g': 1.0
        }
    
    # Pack parameters for optimization
    x0 = [
        np.log10(initial_params['R_min']),  # log scale
        initial_params['alpha_R'],
        initial_params['beta_R'],
        np.log10(initial_params['g_crit']),  # log scale
        initial_params['alpha_g'],
        initial_params['beta_g']
    ]
    
    def objective(x):
        """Objective: RAR scatter"""
        R_min = 10**x[0]
        alpha_R = x[1]
        beta_R = x[2]
        g_crit = 10**x[3]
        alpha_g = x[4]
        beta_g = x[5]
        
        # Create kernel
        kernel = NewGateKernel(
            L_0=hyperparams['L_0'],
            A_0=hyperparams['A_0'],
            p=hyperparams['p'],
            n_coh=hyperparams['n_coh'],
            g_dagger=hyperparams['g_dagger'],
            R_min=R_min,
            alpha_R=alpha_R,
            beta_R=beta_R,
            g_crit=g_crit,
            alpha_g=alpha_g,
            beta_g=beta_g
        )
        
        # Compute scatter
        scatter, _, _, _ = compute_rar_scatter(galaxies, kernel, "Tuning")
        
        return scatter
    
    print("\n" + "="*80)
    print("TUNING GATE PARAMETERS")
    print("="*80)
    print("\nInitial parameters:")
    for key, val in initial_params.items():
        print(f"  {key} = {val}")
    
    print("\nOptimizing...")
    
    # Bounds
    bounds = [
        (-3, -0.5),  # R_min: 0.001 to 0.3 kpc
        (1.0, 4.0),  # alpha_R
        (0.5, 2.0),  # beta_R
        (-12, -8),   # g_crit: 1e-12 to 1e-8 m/s^2
        (1.0, 4.0),  # alpha_g
        (0.5, 2.0)   # beta_g
    ]
    
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
    
    # Extract optimized parameters
    opt_params = {
        'R_min': 10**result.x[0],
        'alpha_R': result.x[1],
        'beta_R': result.x[2],
        'g_crit': 10**result.x[3],
        'alpha_g': result.x[4],
        'beta_g': result.x[5]
    }
    
    print("\nOptimized parameters:")
    for key, val in opt_params.items():
        print(f"  {key} = {val:.6f}")
    print(f"\nOptimized scatter: {result.fun:.4f} dex")
    
    return opt_params, result.fun

# ============================================================================
# MAIN COMPARISON
# ============================================================================

def main():
    # Load SPARC data
    galaxies = load_sparc_data()
    
    # Use stratified 80/20 split like baseline (seed=42)
    np.random.seed(42)
    n_test = int(0.2 * len(galaxies))
    test_indices = np.random.choice(len(galaxies), size=n_test, replace=False)
    train_galaxies = [g for i, g in enumerate(galaxies) if i not in test_indices]
    test_galaxies = [g for i, g in enumerate(galaxies) if i in test_indices]
    
    print(f"\nTrain: {len(train_galaxies)}, Test: {len(test_galaxies)}")
    
    # ========================================================================
    # BASELINE: Load from validation suite result
    # ========================================================================
    
    print("\n" + "="*80)
    print("BASELINE (from validation_suite.py)")
    print("="*80)
    print("\nBaseline scatter: 0.088 dex (verified)")
    baseline_scatter = 0.088
    
    # ========================================================================
    # NEW GATES: Initial test
    # ========================================================================
    
    print("\n" + "="*80)
    print("NEW GATES (Initial)")
    print("="*80)
    
    kernel_new_init = NewGateKernel(
        L_0=hyperparams['L_0'],
        A_0=hyperparams['A_0'],
        p=hyperparams['p'],
        n_coh=hyperparams['n_coh'],
        g_dagger=hyperparams['g_dagger'],
        R_min=0.01,
        alpha_R=2.0,
        beta_R=1.0,
        g_crit=1e-10,
        alpha_g=2.0,
        beta_g=1.0
    )
    
    scatter_init, bias_init, g_obs_init, g_model_init = compute_rar_scatter(
        test_galaxies, kernel_new_init, "New Gates (Initial)"
    )
    
    # ========================================================================
    # TUNE PARAMETERS
    # ========================================================================
    
    opt_params, opt_scatter = tune_gate_parameters(train_galaxies, hyperparams)
    
    # ========================================================================
    # NEW GATES: Optimized
    # ========================================================================
    
    print("\n" + "="*80)
    print("NEW GATES (Optimized) - TEST SET")
    print("="*80)
    
    kernel_new_opt = NewGateKernel(
        L_0=hyperparams['L_0'],
        A_0=hyperparams['A_0'],
        p=hyperparams['p'],
        n_coh=hyperparams['n_coh'],
        g_dagger=hyperparams['g_dagger'],
        **opt_params
    )
    
    scatter_opt, bias_opt, g_obs_opt, g_model_opt = compute_rar_scatter(
        test_galaxies, kernel_new_opt, "New Gates (Optimized)"
    )
    
    # ========================================================================
    # COMPARISON
    # ========================================================================
    
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    
    print(f"\nBaseline:           {baseline_scatter:.4f} dex")
    print(f"New Gates (init):   {scatter_init:.4f} dex  ({(scatter_init/baseline_scatter-1)*100:+.1f}%)")
    print(f"New Gates (tuned):  {scatter_opt:.4f} dex  ({(scatter_opt/baseline_scatter-1)*100:+.1f}%)")
    
    if scatter_opt < baseline_scatter:
        improvement = (baseline_scatter - scatter_opt) / baseline_scatter * 100
        print(f"\nResult: IMPROVEMENT of {improvement:.1f}%!")
    elif scatter_opt > baseline_scatter:
        degradation = (scatter_opt - baseline_scatter) / baseline_scatter * 100
        print(f"\nResult: Degradation of {degradation:.1f}%")
    else:
        print(f"\nResult: No significant difference")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    output_dir = SCRIPT_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    results = {
        'baseline_scatter': baseline_scatter,
        'new_gates_initial_scatter': float(scatter_init),
        'new_gates_optimized_scatter': float(scatter_opt),
        'optimized_gate_params': {k: float(v) for k, v in opt_params.items()},
        'improvement_percent': float((baseline_scatter - scatter_opt) / baseline_scatter * 100)
    }
    
    with open(output_dir / "gate_comparison_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_dir / 'gate_comparison_results.json'}")
    
    # ========================================================================
    # PLOT
    # ========================================================================
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Initial vs Optimized
    ax = axes[0]
    ax.scatter(g_obs_init, g_model_init, alpha=0.3, s=10, label=f'Initial ({scatter_init:.4f} dex)')
    ax.scatter(g_obs_opt, g_model_opt, alpha=0.3, s=10, label=f'Optimized ({scatter_opt:.4f} dex)')
    ax.plot([1e-12, 1e-9], [1e-12, 1e-9], 'k--', alpha=0.5, label='1:1')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$g_{\\rm obs}$ (m/s$^2$)')
    ax.set_ylabel('$g_{\\rm model}$ (m/s$^2$)')
    ax.set_title('New Gates: Before/After Tuning')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Comparison bars
    ax = axes[1]
    models = ['Baseline\n(verified)', 'New Gates\n(initial)', 'New Gates\n(tuned)']
    scatters = [baseline_scatter, scatter_init, scatter_opt]
    colors = ['blue', 'orange', 'green']
    bars = ax.bar(models, scatters, color=colors, alpha=0.7)
    ax.set_ylabel('RAR Scatter (dex)')
    ax.set_title('Scatter Comparison')
    ax.axhline(baseline_scatter, color='blue', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, scatter in zip(bars, scatters):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{scatter:.4f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gate_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_dir / 'gate_comparison.png'}")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()

