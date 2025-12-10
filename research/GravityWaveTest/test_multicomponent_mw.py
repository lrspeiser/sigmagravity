"""
Comprehensive multi-component MW rotation curve test.
Tests disk + bulge with different λ hypotheses and component treatments.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✓ CuPy available - using GPU acceleration")
except ImportError:
    print("⚠ CuPy not available - using NumPy (slower)")
    cp = np
    GPU_AVAILABLE = False

import time
import json
import os

# Import from existing test
import sys
sys.path.insert(0, 'GravityWaveTest')
from test_star_by_star_mw import StarByStarCalculator, burr_xii_window, G_KPC

def compute_bulge_contribution(xp, R_obs, M_bulge=2.0e10, a_bulge=0.7):
    """
    Compute bulge contribution using Hernquist profile.
    
    v²_bulge = G M_bulge R² / (R + a)²
    
    Parameters:
    -----------
    R_obs : array
        Observation radii (kpc)
    M_bulge : float
        Bulge mass (M_☉), default 2×10^10
    a_bulge : float
        Hernquist scale radius (kpc), default 0.7
    
    Returns:
    --------
    v_bulge : array
        Circular velocity from bulge (km/s)
    """
    R_obs = xp.asarray(R_obs, dtype=xp.float32)
    
    # Hernquist circular velocity
    # v² = G M R² / (R + a)²
    v_squared = G_KPC * M_bulge * R_obs**2 / (R_obs + a_bulge)**2
    v_bulge = xp.sqrt(v_squared)
    
    return v_bulge

def select_component_stars(stars_df, component='disk'):
    """
    Select stars appropriate for each component.
    
    Parameters:
    -----------
    component : str
        'disk', 'bulge', 'all'
    
    Returns:
    --------
    mask : boolean array
    """
    R = stars_df['R_cyl'].values
    z = stars_df['z'].values
    
    if component == 'disk':
        # Thin + thick disk: |z/R| < 0.3 (flared disk)
        h_scale = 0.3
        mask = np.abs(z) < h_scale * R
        
    elif component == 'bulge':
        # Central region: R < 3 kpc, all z
        mask = R < 3.0
        
    elif component == 'outer':
        # Outer disk: R > 10 kpc (disk-dominated)
        mask = R > 10.0
        
    elif component == 'all':
        mask = np.ones(len(stars_df), dtype=bool)
    
    else:
        raise ValueError(f"Unknown component: {component}")
    
    return mask

class MultiComponentCalculator:
    """
    Calculate rotation curve with multiple components.
    
    Components:
    - Disk (with Σ-Gravity enhancement)
    - Bulge (Newtonian or with Σ-Gravity)
    """
    
    def __init__(self, stars_df, use_gpu=True):
        self.stars_df = stars_df
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np
        
        # Pre-compute component masks
        self.mask_disk = select_component_stars(stars_df, 'disk')
        self.mask_bulge = select_component_stars(stars_df, 'bulge')
        self.mask_all = select_component_stars(stars_df, 'all')
        
        print(f"\nComponent selection:")
        print(f"  Disk stars: {self.mask_disk.sum():,} (|z/R| < 0.3)")
        print(f"  Bulge stars: {self.mask_bulge.sum():,} (R < 3 kpc)")
        print(f"  Total stars: {len(stars_df):,}")
    
    def test_configuration(self, R_obs, config):
        """
        Test a specific component configuration.
        
        config = {
            'disk': {'enabled': True, 'lambda': 'universal', 'M': 5e10},
            'bulge': {'enabled': True, 'treatment': 'hernquist', 'M': 2e10},
            'A': 0.591,
            'name': 'Disk (Σ-G) + Bulge (Hernquist)'
        }
        """
        xp = self.xp
        v_squared_total = xp.zeros(len(R_obs), dtype=xp.float32)
        
        # Component 1: Disk (with Σ-Gravity)
        if config['disk']['enabled']:
            disk_stars = self.stars_df[self.mask_disk]
            
            if len(disk_stars) > 0:
                calc_disk = StarByStarCalculator(disk_stars, use_gpu=self.use_gpu)
                
                # Get lambda hypothesis
                lambda_hyp = config['disk']['lambda']
                if lambda_hyp == 'universal':
                    lambda_func = lambda: calc_disk.compute_lambda_universal(4.993)
                    kwargs = {}
                elif lambda_hyp == 'h_R':
                    lambda_func = lambda: calc_disk.compute_lambda_local_disk(calc_disk.R_stars)
                    kwargs = {}
                elif lambda_hyp == 'hybrid':
                    lambda_func = lambda M_weights: calc_disk.compute_lambda_hybrid(M_weights, calc_disk.R_stars)
                    kwargs = {'M_weights': None}
                else:
                    raise ValueError(f"Unknown lambda: {lambda_hyp}")
                
                print(f"\nComputing disk component (λ={lambda_hyp}, {len(disk_stars):,} stars)...")
                v_disk, _ = calc_disk.test_hypothesis(
                    R_obs, lambda_func, 
                    A=config['A'], 
                    M_disk=config['disk']['M'],
                    **kwargs
                )
                
                v_squared_total += xp.asarray(v_disk, dtype=xp.float32)**2
        
        # Component 2: Bulge
        if config['bulge']['enabled']:
            treatment = config['bulge']['treatment']
            
            if treatment == 'hernquist':
                # Analytical Hernquist (fast)
                print(f"\nComputing bulge component (Hernquist, M={config['bulge']['M']:.2e} M_☉)...")
                v_bulge = compute_bulge_contribution(
                    xp, R_obs, 
                    M_bulge=config['bulge']['M']
                )
                v_squared_total += v_bulge**2
                
            elif treatment == 'sigma_gravity':
                # Bulge stars with Σ-Gravity (slower)
                bulge_stars = self.stars_df[self.mask_bulge]
                
                if len(bulge_stars) > 0:
                    calc_bulge = StarByStarCalculator(bulge_stars, use_gpu=self.use_gpu)
                    
                    lambda_hyp = config['bulge'].get('lambda', 'universal')
                    if lambda_hyp == 'universal':
                        lambda_func = lambda: calc_bulge.compute_lambda_universal(4.993)
                        kwargs = {}
                    else:
                        lambda_func = lambda: calc_bulge.compute_lambda_local_disk(calc_bulge.R_stars)
                        kwargs = {}
                    
                    print(f"\nComputing bulge component (Σ-Gravity, {len(bulge_stars):,} stars)...")
                    v_bulge, _ = calc_bulge.test_hypothesis(
                        R_obs, lambda_func,
                        A=config['A'],
                        M_disk=config['bulge']['M'],
                        **kwargs
                    )
                    
                    v_squared_total += xp.asarray(v_bulge, dtype=xp.float32)**2
        
        # Total velocity
        v_total = xp.sqrt(v_squared_total)
        
        if self.use_gpu:
            v_total = cp.asnumpy(v_total)
        
        return v_total

def run_comprehensive_test(stars_csv='data/gaia/gaia_processed.csv'):
    """
    Run comprehensive multi-component test.
    """
    
    print("="*80)
    print("COMPREHENSIVE MULTI-COMPONENT MW TEST")
    print("="*80)
    
    # Load data
    print(f"\nLoading stellar data from {stars_csv}...")
    stars = pd.read_csv(stars_csv)
    print(f"Loaded {len(stars):,} stars")
    
    # Initialize calculator
    calc = MultiComponentCalculator(stars, use_gpu=GPU_AVAILABLE)
    
    # Observation radii
    R_obs = np.linspace(0.5, 15.0, 30)
    
    # MW observed rotation curve
    v_obs_MW = 220 * np.ones_like(R_obs)
    
    # Define configurations to test
    configurations = [
        {
            'name': '1. Disk only (universal λ)',
            'disk': {'enabled': True, 'lambda': 'universal', 'M': 5e10},
            'bulge': {'enabled': False},
            'A': 0.591,
            'color': 'blue'
        },
        {
            'name': '2. Disk only (λ=h(R))',
            'disk': {'enabled': True, 'lambda': 'h_R', 'M': 5e10},
            'bulge': {'enabled': False},
            'A': 0.591,
            'color': 'cyan'
        },
        {
            'name': '3. Disk + Bulge (M_b=1e10)',
            'disk': {'enabled': True, 'lambda': 'universal', 'M': 5e10},
            'bulge': {'enabled': True, 'treatment': 'hernquist', 'M': 1e10},
            'A': 0.591,
            'color': 'lightgreen'
        },
        {
            'name': '4. Disk + Bulge (M_b=2e10)',
            'disk': {'enabled': True, 'lambda': 'universal', 'M': 5e10},
            'bulge': {'enabled': True, 'treatment': 'hernquist', 'M': 2e10},
            'A': 0.591,
            'color': 'green'
        },
        {
            'name': '5. Disk (λ=h(R)) + Bulge (M_b=1e10)',
            'disk': {'enabled': True, 'lambda': 'h_R', 'M': 5e10},
            'bulge': {'enabled': True, 'treatment': 'hernquist', 'M': 1e10},
            'A': 0.591,
            'color': 'orange'
        },
        {
            'name': '6. Disk (λ=h(R)) + Bulge (M_b=2e10)',
            'disk': {'enabled': True, 'lambda': 'h_R', 'M': 5e10},
            'bulge': {'enabled': True, 'treatment': 'hernquist', 'M': 2e10},
            'A': 0.591,
            'color': 'red'
        },
    ]
    
    # Run tests
    print("\n" + "="*80)
    print("TESTING CONFIGURATIONS")
    print("="*80)
    
    results = {}
    
    for i, config in enumerate(configurations):
        print(f"\n[{i+1}/{len(configurations)}] {config['name']}")
        
        start = time.time()
        v_circ = calc.test_configuration(R_obs, config)
        elapsed = time.time() - start
        
        # Compute metrics
        chi2 = np.sum((v_circ - v_obs_MW)**2) / len(R_obs)
        rms = np.sqrt(chi2)
        v_at_solar = np.interp(8.2, R_obs, v_circ)
        
        results[config['name']] = {
            'R': R_obs,
            'v_circ': v_circ,
            'chi2': chi2,
            'rms': rms,
            'v_solar': v_at_solar,
            'time': elapsed,
            'color': config['color']
        }
        
        print(f"  v @ R=8.2 kpc: {v_at_solar:.1f} km/s (obs: 220 km/s)")
        print(f"  RMS: {rms:.1f} km/s")
        print(f"  χ²/dof: {chi2:.2f}")
        print(f"  Time: {elapsed:.2f}s")
    
    # Generate plots
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Multi-Component MW Test ({len(stars):,} stars, GPU-accelerated)', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Rotation curves
    ax = axes[0, 0]
    for name, res in results.items():
        ax.plot(res['R'], res['v_circ'], label=name, color=res['color'], linewidth=2)
    
    ax.plot(R_obs, v_obs_MW, 'k--', linewidth=2, label='Observed (220 km/s)')
    ax.axvline(8.2, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('R [kpc]', fontsize=12)
    ax.set_ylabel('v_circ [km/s]', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title('Rotation Curves')
    
    # Plot 2: Residuals
    ax = axes[0, 1]
    for name, res in results.items():
        residuals = res['v_circ'] - v_obs_MW
        ax.plot(res['R'], residuals, label=name, color=res['color'], linewidth=2)
    
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.set_xlabel('R [kpc]', fontsize=12)
    ax.set_ylabel('Δv [km/s]', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title('Residuals')
    
    # Plot 3: χ² comparison
    ax = axes[1, 0]
    names = list(results.keys())
    chi2_values = [results[n]['chi2'] for n in names]
    colors = [results[n]['color'] for n in names]
    
    bars = ax.bar(range(len(names)), chi2_values, color=colors, alpha=0.7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.split('. ')[1] if '. ' in n else n for n in names],
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('χ² / dof', fontsize=12)
    ax.set_title('Goodness of Fit')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: v at Solar radius
    ax = axes[1, 1]
    v_solar_values = [results[n]['v_solar'] for n in names]
    
    bars = ax.bar(range(len(names)), v_solar_values, color=colors, alpha=0.7)
    ax.axhline(220, color='r', linestyle='--', linewidth=2, label='Observed (220 km/s)')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.split('. ')[1] if '. ' in n else n for n in names],
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('v [km/s]', fontsize=12)
    ax.set_title('Velocity at R=8.2 kpc')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_dir = "GravityWaveTest/mw_multicomponent"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/mw_multicomponent_comparison.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot to {output_dir}/mw_multicomponent_comparison.png")
    plt.close()
    
    # Save results
    results_export = {
        'n_stars': int(len(stars)),
        'n_disk_stars': int(calc.mask_disk.sum()),
        'n_bulge_stars': int(calc.mask_bulge.sum()),
        'R_obs': R_obs.tolist(),
        'v_obs': v_obs_MW.tolist(),
        'configurations': {}
    }
    
    for name, res in results.items():
        results_export['configurations'][name] = {
            'v_pred': res['v_circ'].tolist(),
            'v_solar': float(res['v_solar']),
            'chi2': float(res['chi2']),
            'rms': float(res['rms']),
            'time_sec': float(res['time'])
        }
    
    with open(f"{output_dir}/multicomponent_results.json", 'w') as f:
        json.dump(results_export, f, indent=2)
    
    print(f"✓ Saved results to {output_dir}/multicomponent_results.json")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    best_config = min(results.keys(), key=lambda k: results[k]['chi2'])
    
    print(f"\nBest configuration: {best_config}")
    print(f"  v @ R=8.2: {results[best_config]['v_solar']:.1f} km/s (obs: 220 km/s)")
    print(f"  RMS: {results[best_config]['rms']:.1f} km/s")
    print(f"  χ²/dof: {results[best_config]['chi2']:.2f}")
    
    print("\nRankings:")
    for i, name in enumerate(sorted(results.keys(), key=lambda k: results[k]['chi2']), 1):
        print(f"  {i}. {name}")
        print(f"     v={results[name]['v_solar']:.1f} km/s, RMS={results[name]['rms']:.1f} km/s, χ²={results[name]['chi2']:.2f}")
    
    return results

if __name__ == "__main__":
    # Check which data to use
    if os.path.exists('data/gaia/gaia_large_sample.csv'):
        print("\nFound large Gaia sample!")
        choice = input("Use large sample (L) or standard 144k (S)? [l/S]: ").strip().lower()
        if choice == 'l':
            stars_csv = 'data/gaia/gaia_large_sample.csv'
        else:
            stars_csv = 'data/gaia/gaia_processed.csv'
    else:
        stars_csv = 'data/gaia/gaia_processed.csv'
    
    results = run_comprehensive_test(stars_csv)

