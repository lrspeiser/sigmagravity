#!/usr/bin/env python3
"""
Test alternative formula components against baseline

Tests individual components while keeping rest of kernel fixed:
1. Coherence damping: power-law vs exponential vs Burr-XII vs stretched-exp
2. Small-radius gate: exponential vs smoothstep vs tanh
3. Bulge suppression: baseline vs alternatives
4. RAR shape: (g†/g)^p vs alternatives

Baseline (0.088 dex):
K = A_0 * (g†/g_bar)^p * (L_coh/(L_coh+r))^n_coh * S_small(r)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Callable, Dict, List, Tuple

# Add paths
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "many_path_model"))

from path_spectrum_kernel_track2 import PathSpectrumKernel, PathSpectrumHyperparams

# ============================================================================
# ALTERNATIVE COHERENCE DAMPING FUNCTIONS
# ============================================================================

def coherence_powerlaw(r, L_coh, n_coh):
    """Baseline: Power-law damping"""
    return (L_coh / (L_coh + r))**n_coh

def coherence_exponential(r, L_coh, n_coh):
    """Alternative: Exponential damping"""
    return np.exp(-n_coh * r / L_coh)

def coherence_burr_xii(r, L_coh, n_coh):
    """Alternative: Burr-XII (grows to 1 - inverted behavior!)"""
    p = 2.0  # Fixed shape parameter
    return 1.0 - (1.0 + (r / L_coh)**p)**(-n_coh)

def coherence_stretched_exp(r, L_coh, n_coh):
    """Alternative: Stretched exponential"""
    beta = 0.5  # Sub-exponential stretching
    return np.exp(-(n_coh * r / L_coh)**beta)

def coherence_gaussian(r, L_coh, n_coh):
    """Alternative: Gaussian damping"""
    return np.exp(-n_coh * (r / L_coh)**2)

# ============================================================================
# ALTERNATIVE SMALL-RADIUS GATES
# ============================================================================

def gate_exponential(r, r_gate=0.5, p=2.0):
    """Baseline: Exponential turn-on"""
    return 1.0 - np.exp(-(r / r_gate)**p)

def gate_smoothstep(r, r_gate=0.5):
    """Alternative: Smoothstep (C2 continuous)"""
    x = np.clip(r / (2 * r_gate), 0, 1)
    return x**2 * (3 - 2*x)

def gate_tanh(r, r_gate=0.5):
    """Alternative: Hyperbolic tangent"""
    return 0.5 * (1 + np.tanh(4 * (r - r_gate) / r_gate))

def gate_logistic(r, r_gate=0.5):
    """Alternative: Logistic function"""
    k = 10.0  # Steepness
    return 1.0 / (1.0 + np.exp(-k * (r - r_gate) / r_gate))

# ============================================================================
# ALTERNATIVE RAR SHAPES
# ============================================================================

def rar_powerlaw(g_bar, g_dagger, p):
    """Baseline: Power-law (g†/g)^p"""
    return (g_dagger / np.maximum(g_bar, 1e-14))**p

def rar_log(g_bar, g_dagger, p):
    """Alternative: Logarithmic enhancement"""
    return p * np.log(1 + g_dagger / np.maximum(g_bar, 1e-14))

def rar_tanh(g_bar, g_dagger, p):
    """Alternative: Hyperbolic tangent transition"""
    return np.tanh(p * g_dagger / np.maximum(g_bar, 1e-14))

def rar_exponential(g_bar, g_dagger, p):
    """Alternative: Exponential approach"""
    return 1.0 - np.exp(-p * g_dagger / np.maximum(g_bar, 1e-14))

# ============================================================================
# KERNEL BUILDER
# ============================================================================

class FlexibleKernel:
    """Kernel with swappable components for testing alternatives"""
    
    def __init__(self, hyperparams: Dict,
                 coherence_func: Callable = coherence_powerlaw,
                 gate_func: Callable = gate_exponential,
                 rar_func: Callable = rar_powerlaw):
        """
        Parameters
        ----------
        hyperparams : dict
            Must contain: L_0, A_0, p, n_coh, g_dagger, beta_bulge, alpha_shear, gamma_bar
        coherence_func : callable
            Function(r, L_coh, n_coh) -> damping factor
        gate_func : callable
            Function(r, r_gate) -> small-radius gate
        rar_func : callable
            Function(g_bar, g_dagger, p) -> RAR shape
        """
        self.hp = hyperparams
        self.coherence_func = coherence_func
        self.gate_func = gate_func
        self.rar_func = rar_func
        
        # Use baseline physics gates from PathSpectrumKernel
        self.baseline_kernel = PathSpectrumKernel(PathSpectrumHyperparams.from_dict(hyperparams))
    
    def compute_K(self, r, v_circ, g_bar, BT=0.0, bar_strength=0.0, 
                  r_bulge=1.0, r_bar=3.0, r_gate=0.5):
        """
        Compute kernel with alternative components
        
        Structure:
        K = A_0 * RAR_shape * coherence_damping * small_gate
        
        Where:
        - RAR_shape uses rar_func (default: power-law)
        - coherence_damping uses coherence_func (default: power-law)
        - small_gate uses gate_func (default: exponential)
        - L_coh = L_0 * f_bulge * f_shear * f_bar (always baseline physics)
        """
        # Simplified: just use L_0 (no morphology gates for now - need galaxy-specific data)
        L_coh = self.hp['L_0']
        
        # Small-radius gate (SWAPPABLE)
        S_small = self.gate_func(r, r_gate)
        
        # RAR shape (SWAPPABLE)
        K_rar = self.rar_func(g_bar, self.hp['g_dagger'], self.hp['p'])
        
        # Coherence damping (SWAPPABLE)
        K_coh = self.coherence_func(r, L_coh, self.hp['n_coh'])
        
        # Final kernel
        K = self.hp['A_0'] * K_rar * K_coh * S_small
        
        return K

# ============================================================================
# EVALUATION ON SPARC
# ============================================================================

def load_sparc_data():
    """Load SPARC rotation curve data (same as baseline test)"""
    data_dir = REPO_ROOT / "data" / "Rotmod_LTG"
    rc_files = list(data_dir.glob("*_rotmod.dat"))
    
    galaxies = []
    for rc_file in rc_files:
        gal_name = rc_file.stem.replace("_rotmod", "")
        try:
            df = pd.read_csv(rc_file, sep=r'\s+', comment='#',
                           names=['R', 'V_obs', 'errV', 'V_gas', 'V_disk', 'V_bul'])
            
            V_bar = np.sqrt(df['V_gas']**2 + df['V_disk']**2 + df['V_bul']**2)
            g_bar = V_bar**2 / df['R'] * 1e3**2 / (3.086e19)  # Convert to m/s^2
            
            galaxies.append({
                'name': gal_name,
                'R': df['R'].values,
                'V_obs': df['V_obs'].values,
                'V_bar': V_bar.values,
                'g_bar': g_bar,
                'errV': df['errV'].values
            })
        except Exception as e:
            continue
    
    print(f"Loaded {len(galaxies)} SPARC galaxies")
    return galaxies

def compute_rar_scatter(galaxies, kernel, label="Model"):
    """Compute RAR scatter for given kernel"""
    all_g_obs = []
    all_g_model = []
    
    for gal in galaxies:
        R = gal['R']
        V_obs = gal['V_obs']
        V_bar = gal['V_bar']
        g_bar = gal['g_bar']
        
        # Observed acceleration
        g_obs = V_obs**2 / R * 1e3**2 / (3.086e19)
        
        # Model prediction
        try:
            K = kernel.compute_K(R, V_bar, g_bar)
        except Exception as e:
            print(f"Error computing K for {gal['name']}: {e}")
            continue
        g_model = g_bar * (1 + K)
        
        all_g_obs.extend(g_obs)
        all_g_model.extend(g_model)
    
    all_g_obs = np.array(all_g_obs)
    all_g_model = np.array(all_g_model)
    
    # Scatter in dex
    valid = (all_g_obs > 0) & (all_g_model > 0)
    residuals = np.log10(all_g_model[valid]) - np.log10(all_g_obs[valid])
    scatter = np.std(residuals)
    bias = np.mean(residuals)
    
    print(f"{label:30s}: scatter = {scatter:.4f} dex, bias = {bias:+.4f} dex")
    
    return scatter, bias

# ============================================================================
# COMPARISON TESTS
# ============================================================================

def test_coherence_functions(galaxies, hyperparams):
    """Test alternative coherence damping functions"""
    print("\n" + "="*80)
    print("TEST 1: COHERENCE DAMPING FUNCTIONS")
    print("="*80)
    print("\nBaseline: (L_coh/(L_coh+r))^n_coh")
    print("Testing alternatives...\n")
    
    results = {}
    
    # Baseline
    kernel_base = FlexibleKernel(hyperparams, coherence_powerlaw, gate_exponential, rar_powerlaw)
    scatter_base, bias_base = compute_rar_scatter(galaxies, kernel_base, "Baseline (power-law)")
    results['baseline'] = scatter_base
    
    # Exponential
    kernel_exp = FlexibleKernel(hyperparams, coherence_exponential, gate_exponential, rar_powerlaw)
    scatter_exp, bias_exp = compute_rar_scatter(galaxies, kernel_exp, "Exponential")
    results['exponential'] = scatter_exp
    
    # Burr-XII
    kernel_burr = FlexibleKernel(hyperparams, coherence_burr_xii, gate_exponential, rar_powerlaw)
    scatter_burr, bias_burr = compute_rar_scatter(galaxies, kernel_burr, "Burr-XII")
    results['burr_xii'] = scatter_burr
    
    # Stretched exponential
    kernel_stretch = FlexibleKernel(hyperparams, coherence_stretched_exp, gate_exponential, rar_powerlaw)
    scatter_stretch, bias_stretch = compute_rar_scatter(galaxies, kernel_stretch, "Stretched exponential")
    results['stretched_exp'] = scatter_stretch
    
    # Gaussian
    kernel_gauss = FlexibleKernel(hyperparams, coherence_gaussian, gate_exponential, rar_powerlaw)
    scatter_gauss, bias_gauss = compute_rar_scatter(galaxies, kernel_gauss, "Gaussian")
    results['gaussian'] = scatter_gauss
    
    print(f"\nBest: {min(results, key=results.get)} ({results[min(results, key=results.get)]:.4f} dex)")
    
    return results

def test_gate_functions(galaxies, hyperparams):
    """Test alternative small-radius gate functions"""
    print("\n" + "="*80)
    print("TEST 2: SMALL-RADIUS GATE FUNCTIONS")
    print("="*80)
    print("\nBaseline: 1 - exp(-(r/r_gate)^2)")
    print("Testing alternatives...\n")
    
    results = {}
    
    # Baseline
    kernel_base = FlexibleKernel(hyperparams, coherence_powerlaw, gate_exponential, rar_powerlaw)
    scatter_base, _ = compute_rar_scatter(galaxies, kernel_base, "Baseline (exponential)")
    results['baseline'] = scatter_base
    
    # Smoothstep
    kernel_smooth = FlexibleKernel(hyperparams, coherence_powerlaw, gate_smoothstep, rar_powerlaw)
    scatter_smooth, _ = compute_rar_scatter(galaxies, kernel_smooth, "Smoothstep")
    results['smoothstep'] = scatter_smooth
    
    # Tanh
    kernel_tanh = FlexibleKernel(hyperparams, coherence_powerlaw, gate_tanh, rar_powerlaw)
    scatter_tanh, _ = compute_rar_scatter(galaxies, kernel_tanh, "Tanh")
    results['tanh'] = scatter_tanh
    
    # Logistic
    kernel_logistic = FlexibleKernel(hyperparams, coherence_powerlaw, gate_logistic, rar_powerlaw)
    scatter_logistic, _ = compute_rar_scatter(galaxies, kernel_logistic, "Logistic")
    results['logistic'] = scatter_logistic
    
    print(f"\nBest: {min(results, key=results.get)} ({results[min(results, key=results.get)]:.4f} dex)")
    
    return results

def test_rar_shapes(galaxies, hyperparams):
    """Test alternative RAR shape functions"""
    print("\n" + "="*80)
    print("TEST 3: RAR SHAPE FUNCTIONS")
    print("="*80)
    print("\nBaseline: (g†/g_bar)^p")
    print("Testing alternatives...\n")
    
    results = {}
    
    # Baseline
    kernel_base = FlexibleKernel(hyperparams, coherence_powerlaw, gate_exponential, rar_powerlaw)
    scatter_base, _ = compute_rar_scatter(galaxies, kernel_base, "Baseline (power-law)")
    results['baseline'] = scatter_base
    
    # Logarithmic
    kernel_log = FlexibleKernel(hyperparams, coherence_powerlaw, gate_exponential, rar_log)
    scatter_log, _ = compute_rar_scatter(galaxies, kernel_log, "Logarithmic")
    results['logarithmic'] = scatter_log
    
    # Tanh
    kernel_tanh = FlexibleKernel(hyperparams, coherence_powerlaw, gate_exponential, rar_tanh)
    scatter_tanh, _ = compute_rar_scatter(galaxies, kernel_tanh, "Tanh")
    results['tanh'] = scatter_tanh
    
    # Exponential
    kernel_exp = FlexibleKernel(hyperparams, coherence_powerlaw, gate_exponential, rar_exponential)
    scatter_exp, _ = compute_rar_scatter(galaxies, kernel_exp, "Exponential approach")
    results['exponential'] = scatter_exp
    
    print(f"\nBest: {min(results, key=results.get)} ({results[min(results, key=results.get)]:.4f} dex)")
    
    return results

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_function_comparison(output_dir):
    """Plot all alternative functions for visual comparison"""
    r = np.linspace(0, 20, 200)
    L_coh = 5.0
    n_coh = 0.5
    r_gate = 0.5
    g_bar = np.logspace(-12, -9, 200)
    g_dagger = 1.2e-10
    p = 0.75
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Coherence damping
    ax = axes[0, 0]
    ax.plot(r, coherence_powerlaw(r, L_coh, n_coh), label='Power-law (baseline)', linewidth=2)
    ax.plot(r, coherence_exponential(r, L_coh, n_coh), '--', label='Exponential')
    ax.plot(r, coherence_stretched_exp(r, L_coh, n_coh), '-.', label='Stretched exp')
    ax.plot(r, coherence_gaussian(r, L_coh, n_coh), ':', label='Gaussian')
    ax.set_xlabel('Radius (kpc)')
    ax.set_ylabel('Coherence Factor')
    ax.set_title('Coherence Damping Functions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Small-radius gates
    ax = axes[0, 1]
    ax.plot(r, gate_exponential(r, r_gate), label='Exponential (baseline)', linewidth=2)
    ax.plot(r, gate_smoothstep(r, r_gate), '--', label='Smoothstep')
    ax.plot(r, gate_tanh(r, r_gate), '-.', label='Tanh')
    ax.plot(r, gate_logistic(r, r_gate), ':', label='Logistic')
    ax.set_xlabel('Radius (kpc)')
    ax.set_ylabel('Gate Factor')
    ax.set_title('Small-Radius Gate Functions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # RAR shapes
    ax = axes[0, 2]
    ax.plot(g_bar, rar_powerlaw(g_bar, g_dagger, p), label='Power-law (baseline)', linewidth=2)
    ax.plot(g_bar, rar_log(g_bar, g_dagger, p), '--', label='Logarithmic')
    ax.plot(g_bar, rar_tanh(g_bar, g_dagger, p), '-.', label='Tanh')
    ax.plot(g_bar, rar_exponential(g_bar, g_dagger, p), ':', label='Exponential')
    ax.set_xscale('log')
    ax.set_xlabel('$g_{\\rm bar}$ (m/s$^2$)')
    ax.set_ylabel('RAR Factor')
    ax.set_title('RAR Shape Functions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Coherence damping (log scale)
    ax = axes[1, 0]
    ax.semilogy(r, coherence_powerlaw(r, L_coh, n_coh), label='Power-law (baseline)', linewidth=2)
    ax.semilogy(r, coherence_exponential(r, L_coh, n_coh), '--', label='Exponential')
    ax.semilogy(r, coherence_stretched_exp(r, L_coh, n_coh), '-.', label='Stretched exp')
    ax.semilogy(r, coherence_gaussian(r, L_coh, n_coh), ':', label='Gaussian')
    ax.set_xlabel('Radius (kpc)')
    ax.set_ylabel('Coherence Factor (log scale)')
    ax.set_title('Coherence Damping (Log Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Small-radius gates (zoomed)
    ax = axes[1, 1]
    r_zoom = np.linspace(0, 2, 200)
    ax.plot(r_zoom, gate_exponential(r_zoom, r_gate), label='Exponential (baseline)', linewidth=2)
    ax.plot(r_zoom, gate_smoothstep(r_zoom, r_gate), '--', label='Smoothstep')
    ax.plot(r_zoom, gate_tanh(r_zoom, r_gate), '-.', label='Tanh')
    ax.plot(r_zoom, gate_logistic(r_zoom, r_gate), ':', label='Logistic')
    ax.axvline(r_gate, color='gray', linestyle=':', alpha=0.5, label=f'r_gate={r_gate}')
    ax.set_xlabel('Radius (kpc)')
    ax.set_ylabel('Gate Factor')
    ax.set_title('Small-Radius Gates (Zoomed)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # RAR shapes (log-log)
    ax = axes[1, 2]
    ax.loglog(g_bar, rar_powerlaw(g_bar, g_dagger, p), label='Power-law (baseline)', linewidth=2)
    ax.loglog(g_bar, rar_log(g_bar, g_dagger, p), '--', label='Logarithmic')
    ax.loglog(g_bar, rar_tanh(g_bar, g_dagger, p), '-.', label='Tanh')
    ax.loglog(g_bar, rar_exponential(g_bar, g_dagger, p), ':', label='Exponential')
    ax.set_xlabel('$g_{\\rm bar}$ (m/s$^2$)')
    ax.set_ylabel('RAR Factor (log scale)')
    ax.set_title('RAR Shapes (Log-Log)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'alternative_functions.png', dpi=150, bbox_inches='tight')
    print(f"\nFunction comparison plot saved: {output_dir / 'alternative_functions.png'}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("TESTING ALTERNATIVE FORMULA COMPONENTS")
    print("="*80)
    
    # Load hyperparameters
    config_path = REPO_ROOT / "config" / "hyperparams_track2.json"
    with open(config_path, 'r') as f:
        hyperparams = json.load(f)
    
    print(f"\nUsing baseline hyperparameters from: {config_path}")
    for key, val in hyperparams.items():
        print(f"  {key} = {val}")
    
    # Load SPARC data
    galaxies = load_sparc_data()
    
    # Use test split (seed=42)
    np.random.seed(42)
    n_test = int(0.2 * len(galaxies))
    test_indices = np.random.choice(len(galaxies), size=n_test, replace=False)
    test_galaxies = [g for i, g in enumerate(galaxies) if i in test_indices]
    print(f"Using test set: {len(test_galaxies)} galaxies")
    
    # Create output directory
    output_dir = SCRIPT_DIR / "outputs" / "alternative_tests"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Run tests
    results_coh = test_coherence_functions(test_galaxies, hyperparams)
    results_gate = test_gate_functions(test_galaxies, hyperparams)
    results_rar = test_rar_shapes(test_galaxies, hyperparams)
    
    # Plot function shapes
    plot_function_comparison(output_dir)
    
    # Save results
    all_results = {
        'coherence_functions': results_coh,
        'gate_functions': results_gate,
        'rar_shapes': results_rar,
        'baseline_scatter': results_coh['baseline'],
        'hyperparams': hyperparams
    }
    
    with open(output_dir / 'alternative_test_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nBaseline scatter: {results_coh['baseline']:.4f} dex")
    print("\nBest alternatives:")
    print(f"  Coherence: {min(results_coh, key=results_coh.get)} ({results_coh[min(results_coh, key=results_coh.get)]:.4f} dex)")
    print(f"  Gate:      {min(results_gate, key=results_gate.get)} ({results_gate[min(results_gate, key=results_gate.get)]:.4f} dex)")
    print(f"  RAR shape: {min(results_rar, key=results_rar.get)} ({results_rar[min(results_rar, key=results_rar.get)]:.4f} dex)")
    
    print(f"\nResults saved: {output_dir / 'alternative_test_results.json'}")
    print("="*80)

if __name__ == "__main__":
    main()

