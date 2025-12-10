"""
ROBUST IMPROVED MULTIPLIER CALCULATION

Uses sampling to avoid GPU memory issues.
Can run in background with progress logging.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import json
from datetime import datetime

try:
    import cupy as cp
    GPU_AVAILABLE = True
    print(f"[{datetime.now().strftime('%H:%M:%S')}] GPU available")
except ImportError:
    cp = np
    GPU_AVAILABLE = False
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Using CPU")

# Constants
G_KPC = 4.498e-12

# ============================================================================
# ANALYTICAL COMPONENTS
# ============================================================================

def hernquist_bulge(R, M_bulge=1.5e10, a_bulge=0.7):
    """Hernquist bulge model."""
    v_squared = G_KPC * M_bulge * R / (R + a_bulge)**2
    return np.sqrt(v_squared)

def exponential_gas(R, M_gas=1e10, R_gas=7.0):
    """Exponential gas disk."""
    from scipy.special import i0, i1, k0, k1
    y = R / (2 * R_gas)
    v_squared = 4 * np.pi * G_KPC * M_gas * R**2 / (2 * R_gas)**2 * \
                (i0(y) * k0(y) - i1(y) * k1(y))
    return np.sqrt(np.maximum(v_squared, 0))

# ============================================================================
# MULTIPLIER FUNCTIONS
# ============================================================================

def multiplier_power_law(lam, r, params, xp=np):
    """f = 1 + A(lambda/lambda_0)^alpha"""
    A, lambda_0, alpha = params
    return 1.0 + A * (lam / lambda_0)**alpha

# ============================================================================
# SAMPLED CALCULATOR (AVOIDS MEMORY ISSUES)
# ============================================================================

def compute_velocity_sampled(gaia, obs_indices, period_name, multiplier_func, params,
                             M_disk=5e10, n_source_samples=50000,
                             use_bulge=False, use_gas=False):
    """
    Compute velocity using sampled source stars.
    
    This is the KEY fix: We sample 50k source stars instead of using all 1.8M.
    """
    
    N_obs = len(obs_indices)
    
    # Sample source stars (stratified by radius)
    R_all = gaia['R'].values
    R_bins = np.percentile(R_all, np.linspace(0, 100, 21))
    source_indices = []
    
    for i in range(len(R_bins) - 1):
        in_bin = np.where((R_all >= R_bins[i]) & (R_all < R_bins[i+1]))[0]
        if len(in_bin) > 0:
            n_sample = min(len(in_bin), n_source_samples // 20)
            sampled = np.random.choice(in_bin, size=n_sample, replace=False)
            source_indices.extend(sampled)
    
    source_indices = np.array(source_indices)
    
    # Get source star data
    x_src = gaia['x'].values[source_indices].astype(np.float32)
    y_src = gaia['y'].values[source_indices].astype(np.float32)
    z_src = gaia['z'].values[source_indices].astype(np.float32)
    M_src = gaia['M_star'].values[source_indices].astype(np.float32)
    lambda_src = gaia[f'lambda_{period_name}'].values[source_indices].astype(np.float32)
    
    # Scale masses
    M_scale_factor = M_disk / M_src.sum()
    M_scaled = M_src * M_scale_factor
    
    # Get observation points
    x_obs = gaia['x'].values[obs_indices].astype(np.float32)
    y_obs = gaia['y'].values[obs_indices].astype(np.float32)
    z_obs = gaia['z'].values[obs_indices].astype(np.float32)
    R_obs = gaia['R'].values[obs_indices].astype(np.float32)
    
    # Transfer to GPU if available
    if GPU_AVAILABLE:
        x_src_gpu = cp.array(x_src)
        y_src_gpu = cp.array(y_src)
        z_src_gpu = cp.array(z_src)
        M_scaled_gpu = cp.array(M_scaled)
        lambda_src_gpu = cp.array(lambda_src)
        
        x_obs_gpu = cp.array(x_obs)
        y_obs_gpu = cp.array(y_obs)
        z_obs_gpu = cp.array(z_obs)
        R_obs_gpu = cp.array(R_obs)
        
        # Compute on GPU
        dx = x_obs_gpu[:, None] - x_src_gpu[None, :]
        dy = y_obs_gpu[:, None] - y_src_gpu[None, :]
        dz = z_obs_gpu[:, None] - z_src_gpu[None, :]
        r = cp.sqrt(dx**2 + dy**2 + dz**2 + 0.01**2)
        
        g_base = G_KPC * M_scaled_gpu[None, :] / r**2
        multiplier = multiplier_func(lambda_src_gpu[None, :], r, params, cp)
        g_enhanced = g_base * multiplier
        
        cos_theta = dx / r
        g_radial = g_enhanced * cos_theta
        g_total = cp.sum(g_radial, axis=1)
        
        v_squared = R_obs_gpu * g_total
        v_stars = cp.asnumpy(cp.sqrt(cp.maximum(v_squared, 0)))
    else:
        # CPU computation
        dx = x_obs[:, None] - x_src[None, :]
        dy = y_obs[:, None] - y_src[None, :]
        dz = z_obs[:, None] - z_src[None, :]
        r = np.sqrt(dx**2 + dy**2 + dz**2 + 0.01**2)
        
        g_base = G_KPC * M_scaled[None, :] / r**2
        multiplier = multiplier_func(lambda_src[None, :], r, params, np)
        g_enhanced = g_base * multiplier
        
        cos_theta = dx / r
        g_radial = g_enhanced * cos_theta
        g_total = np.sum(g_radial, axis=1)
        
        v_squared = R_obs * g_total
        v_stars = np.sqrt(np.maximum(v_squared, 0))
    
    # Add analytical components
    v_bulge = hernquist_bulge(R_obs) if use_bulge else np.zeros_like(R_obs)
    v_gas = exponential_gas(R_obs) if use_gas else np.zeros_like(R_obs)
    
    # Total
    v_total = np.sqrt(v_stars**2 + v_bulge**2 + v_gas**2)
    
    return v_total, v_stars, v_bulge, v_gas

# ============================================================================
# SIMPLE OPTIMIZER
# ============================================================================

def optimize_simple(gaia, obs_indices, v_observed, period_name, params_init,
                   use_bulge=False, use_gas=False):
    """
    Use differential_evolution but with proper memory management.
    """
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Testing {period_name}, bulge={use_bulge}, gas={use_gas}")
    
    from scipy.optimize import differential_evolution
    
    n_calls = [0]
    best_rms = [np.inf]
    
    def objective(params):
        n_calls[0] += 1
        try:
            v_model, v_stars, v_bulge, v_gas = compute_velocity_sampled(
                gaia, obs_indices, period_name, multiplier_power_law, params,
                use_bulge=use_bulge, use_gas=use_gas,
                n_source_samples=30000  # Use fewer samples for speed
            )
            
            rms = np.sqrt(np.mean((v_model - v_observed)**2))
            chi_sq = float(np.sum((v_model - v_observed)**2))
            
            if rms < best_rms[0]:
                best_rms[0] = rms
                if n_calls[0] % 10 == 1:  # Print every 10 calls
                    print(f"      Call {n_calls[0]}: RMS={rms:.1f} km/s")
            
            # Free GPU memory each call
            if GPU_AVAILABLE:
                cp.get_default_memory_pool().free_all_blocks()
            
            return chi_sq
        
        except Exception as e:
            print(f"      Error in call {n_calls[0]}: {e}")
            if GPU_AVAILABLE:
                cp.get_default_memory_pool().free_all_blocks()
            return 1e10
    
    # Optimize
    result = differential_evolution(
        objective,
        bounds=[(0, 3), (1, 20), (1, 4)],
        maxiter=15,
        popsize=5,
        seed=42,
        workers=1,
        polish=False,
        updating='immediate'
    )
    
    # Final evaluation with more samples
    v_model, v_stars, v_bulge, v_gas = compute_velocity_sampled(
        gaia, obs_indices, period_name, multiplier_power_law, result.x,
        use_bulge=use_bulge, use_gas=use_gas,
        n_source_samples=100000  # More samples for final result
    )
    
    rms = np.sqrt(np.mean((v_model - v_observed)**2))
    
    return {
        'period': period_name,
        'params': tuple(result.x),
        'rms': rms,
        'v_model': v_model,
        'v_stars': v_stars,
        'v_bulge': v_bulge,
        'v_gas': v_gas,
        'n_calls': n_calls[0]
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run robust analysis."""
    
    print("="*80)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ROBUST IMPROVED ANALYSIS")
    print("="*80)
    
    # Load data
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Loading data...")
    gaia = pd.read_parquet('gravitywavebaseline/gaia_with_periods.parquet')
    print(f"  {len(gaia):,} stars loaded")
    
    # Setup observations
    N_obs = 1000
    obs_indices = np.linspace(0, len(gaia)-1, N_obs, dtype=int)
    v_observed = np.ones(N_obs, dtype=np.float32) * 220.0
    
    # Test configurations
    tests = [
        ('Baseline (disk only)', 'jeans', False, False),
        ('With bulge', 'jeans', True, False),
        ('With bulge + gas', 'jeans', True, True),
    ]
    
    results = []
    
    for name, period, bulge, gas in tests:
        print(f"\n{'='*80}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {name}")
        print("="*80)
        
        try:
            result = optimize_simple(
                gaia, obs_indices, v_observed, period,
                params_init=(2.08, 9.94, 2.78),
                use_bulge=bulge, use_gas=gas
            )
            result['config'] = name
            results.append(result)
            
            # Component breakdown
            v_stars_contrib = np.mean(result['v_stars']**2) / np.mean(result['v_model']**2) * 100
            v_bulge_contrib = np.mean(result['v_bulge']**2) / np.mean(result['v_model']**2) * 100
            v_gas_contrib = np.mean(result['v_gas']**2) / np.mean(result['v_model']**2) * 100
            
            print(f"\n  Results:")
            print(f"    RMS: {result['rms']:.1f} km/s")
            print(f"    Params: A={result['params'][0]:.2f}, lambda_0={result['params'][1]:.2f}, alpha={result['params'][2]:.2f}")
            print(f"    Components: stars={v_stars_contrib:.1f}%, bulge={v_bulge_contrib:.1f}%, gas={v_gas_contrib:.1f}%")
            
            # Free GPU memory
            if GPU_AVAILABLE:
                cp.get_default_memory_pool().free_all_blocks()
                print(f"    [GPU memory freed]")
        
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*80}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] RESULTS SUMMARY")
    print("="*80)
    
    for i, res in enumerate(results, 1):
        print(f"\n{i}. {res['config']}")
        print(f"   RMS: {res['rms']:.1f} km/s")
        print(f"   Params: {res['params']}")
    
    # Save results
    output_file = 'gravitywavebaseline/robust_improvement_results.json'
    with open(output_file, 'w') as f:
        # Remove v_model arrays for JSON
        results_save = []
        for r in results:
            r_save = {k: v for k, v in r.items() if not k.startswith('v_')}
            r_save['rms'] = float(r['rms'])
            results_save.append(r_save)
        json.dump(results_save, f, indent=2)
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Results saved to: {output_file}")
    
    # Calculate improvements
    if len(results) > 1:
        baseline = results[0]
        best = min(results, key=lambda x: x['rms'])
        
        print(f"\n{'='*80}")
        print("IMPROVEMENT ANALYSIS")
        print("="*80)
        print(f"\nBaseline: {baseline['rms']:.1f} km/s")
        print(f"Best: {best['rms']:.1f} km/s ({best['config']})")
        print(f"Improvement: {baseline['rms'] - best['rms']:.1f} km/s ({(baseline['rms'] - best['rms'])/baseline['rms']*100:.1f}%)")
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Analysis complete!")
    
    return results

if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Analysis interrupted by user")
    except Exception as e:
        print(f"\n\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()

