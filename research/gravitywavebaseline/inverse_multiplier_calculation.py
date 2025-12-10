"""
STEP 2: Inverse Multiplier Calculation (GPU-Accelerated)

This script works BACKWARDS from observed velocities to find what multiplier
function on gravitational wave periods is needed to reproduce observations.

For each star:
1. Calculate gravity contribution from all other stars
2. Apply multiplier function f(lambda_i, r_ij) based on their periods
3. Optimize f to match observed velocity

This is computationally intensive: O(N²) operations!
GPU acceleration is ESSENTIAL for 1.8M stars.

Input: gaia_with_periods.parquet (from calculate_periods.py)
Output: optimized_multipliers.json + diagnostic plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import json
from scipy.optimize import differential_evolution, minimize
from pathlib import Path

try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("[OK] GPU (CuPy) available - enabling GPU acceleration")
    
    # Print GPU info
    device = cp.cuda.Device(0)
    print(f"  GPU: {device.compute_capability}")
    meminfo = device.mem_info
    print(f"  Memory: {meminfo[1] / 1e9:.1f} GB total, {meminfo[0] / 1e9:.1f} GB free")
    
except ImportError:
    cp = np
    GPU_AVAILABLE = False
    print("[!] GPU not available - will use CPU (SLOW for large datasets!)")

# Constants
G_KPC = 4.498e-12  # kpc^3 / M_sun / (km/s)^2

# ============================================================================
# MULTIPLIER FUNCTIONS TO TEST
# ============================================================================

def multiplier_linear(lambda_vals, r, params, xp=np):
    """f(lambda, r) = 1 + A * (lambda / r)"""
    A = params[0]
    return 1.0 + A * (lambda_vals / (r + 0.1))

def multiplier_power_law(lambda_vals, r, params, xp=np):
    """f(lambda, r) = 1 + A * (lambda / lambda_0)^alpha"""
    A, lambda_0, alpha = params
    return 1.0 + A * (lambda_vals / lambda_0) ** alpha

def multiplier_saturating(lambda_vals, r, params, xp=np):
    """f(lambda, r) = 1 + A * [1 - 1/(1 + (lambda/lambda_0)^p)]"""
    A, lambda_0, p = params
    return 1.0 + A * (1.0 - 1.0 / (1.0 + (lambda_vals / lambda_0) ** p))

def multiplier_distance_modulated(lambda_vals, r, params, xp=np):
    """f(lambda, r) = 1 + A * lambda / (r + lambda)"""
    A = params[0]
    return 1.0 + A * lambda_vals / (r + lambda_vals)

def multiplier_exponential(lambda_vals, r, params, xp=np):
    """f(lambda, r) = 1 + A * exp(-r/lambda)"""
    A = params[0]
    return 1.0 + A * xp.exp(-r / (lambda_vals + 0.1))

def multiplier_resonant(lambda_vals, r, params, xp=np):
    """f(lambda, r) = 1 + A * exp(-(r - lambda)^2 / sigma^2)"""
    A, sigma = params
    return 1.0 + A * xp.exp(-(r - lambda_vals)**2 / (sigma**2 + 0.1))

def multiplier_inverse_square(lambda_vals, r, params, xp=np):
    """f(lambda, r) = 1 + A * lambda^2 / (r^2 + lambda^2)"""
    A = params[0]
    return 1.0 + A * lambda_vals**2 / (r**2 + lambda_vals**2)

# ============================================================================
# GPU-ACCELERATED GRAVITY CALCULATOR
# ============================================================================

class GPUGravityCalculator:
    """
    Calculate gravitational acceleration at multiple points from stellar distribution.
    
    Uses GPU for massive speedup on N×N calculations.
    
    For 1.8M stars, full N×N is not feasible (3.2 trillion pairs!).
    We use spatial chunking and batch processing.
    """
    
    def __init__(self, stars_data, use_gpu=True, max_source_batch=10000):
        """
        Parameters:
        -----------
        stars_data : DataFrame
            Must have: x, y, z, M_star, lambda_* columns
        use_gpu : bool
            Use GPU if available
        max_source_batch : int
            Maximum number of source stars to process at once
        """
        
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np
        
        print("\n" + "="*80)
        print("INITIALIZING GPU GRAVITY CALCULATOR")
        print("="*80)
        
        self.N_stars = len(stars_data)
        print(f"\n  Loading {self.N_stars:,} stars...")
        
        # For 1.8M stars, we need to be memory-conscious
        # Store on CPU, transfer to GPU in batches
        self.x = stars_data['x'].values.astype(np.float32)
        self.y = stars_data['y'].values.astype(np.float32)
        self.z = stars_data['z'].values.astype(np.float32)
        self.M = stars_data['M_star'].values.astype(np.float32)
        self.R = np.sqrt(self.x**2 + self.y**2)
        
        # Store all period hypotheses (on CPU)
        self.periods = {}
        for col in stars_data.columns:
            if col.startswith('lambda_'):
                period_name = col.replace('lambda_', '')
                self.periods[period_name] = stars_data[col].values.astype(np.float32)
        
        print(f"  [OK] Loaded {len(self.periods)} period hypotheses: {list(self.periods.keys())}")
        
        # Memory estimates
        mem_per_star = 4 * 4  # 4 arrays × 4 bytes (float32)
        total_mem_mb = (mem_per_star * self.N_stars) / 1e6
        print(f"  CPU memory: ~{total_mem_mb:.1f} MB")
        
        self.max_source_batch = max_source_batch
        
        print(f"\n[OK] Calculator initialized (batch size: {max_source_batch:,} stars)")
    
    def compute_gravity_sampled(self, obs_indices, period_name, multiplier_func, params, 
                                M_scale=5e10, n_source_samples=50000):
        """
        Compute circular velocity using SAMPLED stellar distribution.
        
        Instead of summing over ALL 1.8M stars for each observation point,
        we sample a representative subset for computational efficiency.
        
        This is justified because:
        1. Distant stars contribute little (1/r² falloff)
        2. Random sampling preserves statistical properties
        3. We can increase n_source_samples to test convergence
        
        Parameters:
        -----------
        obs_indices : array
            Indices of stars to use as observation points
        period_name : str
            Which period hypothesis to use
        multiplier_func : callable
            Function f(lambda, r, params) returning multiplier
        params : tuple
            Parameters for multiplier function
        M_scale : float
            Scale stellar masses to this total (M_☉)
        n_source_samples : int
            Number of source stars to sample for each observation
        
        Returns:
        --------
        v_circ : array
            Circular velocity at each observation point (km/s)
        """
        
        N_obs = len(obs_indices)
        v_circ = np.zeros(N_obs, dtype=np.float32)
        
        # Scale masses to match total disk
        M_scale_factor = M_scale / float(np.sum(self.M))
        M_scaled = self.M * M_scale_factor
        
        # Get periods for all stars
        lambda_vals = self.periods[period_name]
        
        # Sample source stars (stratified by radius for better coverage)
        print(f"    Sampling {n_source_samples:,} source stars (stratified by radius)...")
        
        # Bin by radius
        R_bins = np.percentile(self.R, np.linspace(0, 100, 21))
        source_indices = []
        
        for i in range(len(R_bins) - 1):
            in_bin = np.where((self.R >= R_bins[i]) & (self.R < R_bins[i+1]))[0]
            if len(in_bin) > 0:
                n_sample = min(len(in_bin), n_source_samples // 20)
                sampled = np.random.choice(in_bin, size=n_sample, replace=False)
                source_indices.extend(sampled)
        
        source_indices = np.array(source_indices)
        print(f"    Selected {len(source_indices):,} source stars")
        
        # Transfer source stars to GPU
        if self.use_gpu:
            x_src = cp.array(self.x[source_indices])
            y_src = cp.array(self.y[source_indices])
            z_src = cp.array(self.z[source_indices])
            M_src = cp.array(M_scaled[source_indices])
            lambda_src = cp.array(lambda_vals[source_indices])
        else:
            x_src = self.x[source_indices]
            y_src = self.y[source_indices]
            z_src = self.z[source_indices]
            M_src = M_scaled[source_indices]
            lambda_src = lambda_vals[source_indices]
        
        # Process observation points in batches
        obs_batch_size = 1000 if self.use_gpu else 100
        n_batches = (N_obs + obs_batch_size - 1) // obs_batch_size
        
        print(f"    Computing {N_obs:,} observations in {n_batches} batches...")
        
        for i in range(n_batches):
            start_idx = i * obs_batch_size
            end_idx = min((i + 1) * obs_batch_size, N_obs)
            batch_obs_idx = obs_indices[start_idx:end_idx]
            
            # Observation points
            if self.use_gpu:
                x_obs = cp.array(self.x[batch_obs_idx])
                y_obs = cp.array(self.y[batch_obs_idx])
                z_obs = cp.array(self.z[batch_obs_idx])
                R_obs = cp.array(self.R[batch_obs_idx])
            else:
                x_obs = self.x[batch_obs_idx]
                y_obs = self.y[batch_obs_idx]
                z_obs = self.z[batch_obs_idx]
                R_obs = self.R[batch_obs_idx]
            
            # Distance from each obs point to all source stars
            # Shape: (n_obs_batch, n_sources)
            dx = x_obs[:, None] - x_src[None, :]
            dy = y_obs[:, None] - y_src[None, :]
            dz = z_obs[:, None] - z_src[None, :]
            
            r = self.xp.sqrt(dx**2 + dy**2 + dz**2 + 0.01**2)  # Softening
            
            # Base Newtonian gravity from each source star
            g_base = G_KPC * M_src[None, :] / r**2
            
            # Apply multiplier based on source star's period and distance
            multiplier = multiplier_func(lambda_src[None, :], r, params, self.xp)
            
            g_enhanced = g_base * multiplier
            
            # Project to radial direction (x-component of radial gravity)
            cos_theta = dx / r
            g_radial = g_enhanced * cos_theta
            
            # Sum contributions from all source stars
            g_total = self.xp.sum(g_radial, axis=1)
            
            # Circular velocity: v² = R × g
            v_squared = R_obs * g_total
            v_batch = self.xp.sqrt(self.xp.maximum(v_squared, 0))
            
            # Transfer back to CPU if using GPU
            if self.use_gpu:
                v_circ[start_idx:end_idx] = cp.asnumpy(v_batch)
            else:
                v_circ[start_idx:end_idx] = v_batch
            
            if (i + 1) % max(1, n_batches // 5) == 0:
                print(f"      Batch {i+1}/{n_batches} complete")
        
        return v_circ

# ============================================================================
# INVERSE OPTIMIZATION
# ============================================================================

def optimize_multiplier_for_observations(calculator, obs_indices, v_observed, 
                                        period_name, multiplier_func, param_bounds,
                                        M_scale=5e10):
    """
    Find optimal multiplier parameters to match observed velocities.
    
    This solves the INVERSE problem:
    Given: observed velocities v_obs
    Find: parameters of f(lambda, r) such that model matches observations
    """
    
    print(f"\n  Optimizing: {period_name} + {multiplier_func.__name__}")
    t0 = time.time()
    
    # Track best result
    best_chi_sq = np.inf
    best_params = None
    n_calls = [0]
    
    def objective(params):
        """Compute chi-squared between model and observations"""
        n_calls[0] += 1
        
        v_model = calculator.compute_gravity_sampled(
            obs_indices, period_name, multiplier_func, params, M_scale,
            n_source_samples=30000  # Balance speed vs accuracy
        )
        
        # Chi-squared (equal weights)
        chi_sq = np.sum((v_model - v_observed)**2)
        
        # Track progress
        nonlocal best_chi_sq, best_params
        if chi_sq < best_chi_sq:
            best_chi_sq = chi_sq
            best_params = params
            rms = np.sqrt(np.mean((v_model - v_observed)**2))
            print(f"      Call {n_calls[0]}: RMS={rms:.1f} km/s, params={params}")
        
        return chi_sq
    
    # Global optimization with limited iterations (each is expensive!)
    result = differential_evolution(
        objective,
        bounds=param_bounds,
        maxiter=20,  # Limited for 1.8M star dataset
        popsize=6,
        seed=42,
        workers=1,  # GPU doesn't parallelize well across processes
        polish=False,
        updating='deferred',  # Evaluate generation in parallel (CPU only)
    )
    
    t1 = time.time()
    
    # Compute final model with more samples for accuracy
    print(f"    Computing final model with high accuracy...")
    v_model = calculator.compute_gravity_sampled(
        obs_indices, period_name, multiplier_func, result.x, M_scale,
        n_source_samples=100000  # Higher accuracy for final result
    )
    
    # Statistics
    rms = np.sqrt(np.mean((v_model - v_observed)**2))
    chi_sq = result.fun
    
    print(f"    [OK] Complete in {t1-t0:.1f}s ({n_calls[0]} function calls)")
    print(f"    Final params: {result.x}")
    print(f"    Final RMS: {rms:.1f} km/s")
    print(f"    Final chi^2: {chi_sq:.1f}")
    
    return {
        'period_name': period_name,
        'multiplier_func': multiplier_func.__name__,
        'params': result.x.tolist(),
        'rms': float(rms),
        'chi_squared': float(chi_sq),
        'v_model': v_model,
        'time': t1 - t0,
        'n_calls': n_calls[0]
    }

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def load_data_with_observations():
    """
    Load stellar data with periods AND observed velocities.
    """
    
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    # Load periods file
    possible_paths = [
        'gravitywavebaseline/gaia_with_periods.parquet',
        'gaia_with_periods.parquet',
    ]
    
    stars = None
    for path in possible_paths:
        try:
            stars = pd.read_parquet(path)
            print(f"[OK] Loaded periods from: {path}")
            break
        except:
            continue
    
    if stars is None:
        raise FileNotFoundError("Could not find gaia_with_periods.parquet. Run calculate_periods.py first!")
    
    print(f"  Total stars: {len(stars):,}")
    
    # Check if we have observed velocities
    has_vphi = 'v_phi' in stars.columns
    
    if has_vphi:
        # Use stars with non-zero v_phi as observations
        valid_vphi = (stars['v_phi'] != 0) & (stars['v_phi'].notna())
        n_with_vphi = valid_vphi.sum()
        
        print(f"  Stars with v_phi: {n_with_vphi:,}")
        
        if n_with_vphi > 1000:
            # Sample observation points across radii
            stars_with_vphi = stars[valid_vphi]
            
            # Stratified sampling by radius
            N_obs = min(5000, n_with_vphi)  # Use up to 5000 observation points
            R_bins = np.percentile(stars_with_vphi['R'], np.linspace(0, 100, 21))
            
            obs_indices_list = []
            for i in range(len(R_bins) - 1):
                in_bin = stars_with_vphi[(stars_with_vphi['R'] >= R_bins[i]) & 
                                         (stars_with_vphi['R'] < R_bins[i+1])].index
                if len(in_bin) > 0:
                    n_sample = min(len(in_bin), N_obs // 20)
                    sampled = np.random.choice(in_bin, size=n_sample, replace=False)
                    obs_indices_list.extend(sampled)
            
            obs_indices = np.array(obs_indices_list)
            v_observed = stars.loc[obs_indices, 'v_phi'].values
            
            print(f"\n[OK] Using {len(obs_indices):,} observation points (stratified by radius)")
            print(f"  v_phi range: {v_observed.min():.1f} - {v_observed.max():.1f} km/s")
            print(f"  v_phi median: {np.median(v_observed):.1f} km/s")
        else:
            # Not enough real observations, use synthetic target
            print(f"\n[!] Only {n_with_vphi:,} stars with v_phi, using synthetic target")
            N_obs = min(3000, len(stars))
            obs_indices = np.linspace(0, len(stars)-1, N_obs, dtype=int)
            v_observed = np.ones(N_obs) * 220.0
            print(f"  Using {N_obs:,} observation points with target v=220 km/s")
    else:
        print("\n[!] No v_phi column found, using synthetic target")
        N_obs = min(3000, len(stars))
        obs_indices = np.linspace(0, len(stars)-1, N_obs, dtype=int)
        v_observed = np.ones(N_obs) * 220.0
        print(f"  Using {N_obs:,} observation points with target v=220 km/s")
    
    return stars, obs_indices, v_observed

def run_inverse_calculation():
    """
    Main inverse calculation pipeline.
    """
    
    print("="*80)
    print("INVERSE MULTIPLIER CALCULATION - 1.8M GAIA STARS")
    print("="*80)
    print("\nThis finds what multiplier function on GW periods is needed")
    print("to reproduce the observed Milky Way rotation curve.")
    print("\nNote: With 1.8M stars, we use statistical sampling methods")
    print("to make the O(N²) calculation tractable.")
    
    t_start = time.time()
    
    # Load data
    stars, obs_indices, v_observed = load_data_with_observations()
    
    # Initialize GPU calculator
    calculator = GPUGravityCalculator(stars, use_gpu=GPU_AVAILABLE)
    
    # Test cases: period hypotheses × multiplier functions
    print("\n" + "="*80)
    print("RUNNING OPTIMIZATION")
    print("="*80)
    
    # Select subset of hypotheses for initial tests
    period_tests = ['orbital', 'dynamical', 'jeans', 'hybrid', 'gw']
    
    multiplier_tests = [
        ('linear', multiplier_linear, [(0, 5)]),
        ('power_law', multiplier_power_law, [(0, 3), (0.1, 50), (0.5, 3)]),
        ('saturating', multiplier_saturating, [(0, 3), (0.1, 50), (0.5, 3)]),
        ('distance_modulated', multiplier_distance_modulated, [(0, 5)]),
        ('inverse_square', multiplier_inverse_square, [(0, 5)]),
    ]
    
    results = []
    
    for period_name in period_tests:
        if period_name not in calculator.periods:
            print(f"\n[!] Period '{period_name}' not found, skipping")
            continue
        
        print(f"\n{'='*80}")
        print(f"PERIOD HYPOTHESIS: {period_name}")
        print(f"{'='*80}")
        
        for func_name, func, bounds in multiplier_tests:
            try:
                result = optimize_multiplier_for_observations(
                    calculator, obs_indices, v_observed,
                    period_name, func, bounds, M_scale=5e10
                )
                
                results.append(result)
                
            except Exception as e:
                print(f"    [X] Failed: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Rank results
    print("\n" + "="*80)
    print("RESULTS RANKING")
    print("="*80)
    
    results_sorted = sorted(results, key=lambda x: x['rms'])
    
    print(f"\n{'Rank':<5} {'Period':<15} {'Multiplier':<20} {'RMS (km/s)':<12} {'chi^2':<15}")
    print("-"*75)
    
    for i, res in enumerate(results_sorted, 1):
        print(f"{i:<5} {res['period_name']:<15} {res['multiplier_func']:<20} "
              f"{res['rms']:<12.1f} {res['chi_squared']:<15.1e}")
    
    # Save results
    output_file = 'gravitywavebaseline/inverse_multiplier_results.json'
    
    with open(output_file, 'w') as f:
        # Remove v_model arrays (too large for JSON)
        results_save = []
        for r in results_sorted:
            r_save = r.copy()
            r_save.pop('v_model', None)
            results_save.append(r_save)
        
        json.dump(results_save, f, indent=2)
    
    print(f"\n[OK] Results saved to: {output_file}")
    
    # Create visualization
    create_visualization(results_sorted, obs_indices, v_observed, stars)
    
    t_end = time.time()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\n[OK] Complete in {(t_end - t_start)/60:.1f} minutes")
    print(f"[OK] Tested {len(results)} combinations")
    print(f"[OK] Best RMS: {results_sorted[0]['rms']:.1f} km/s")
    
    best = results_sorted[0]
    print(f"\nBest result:")
    print(f"  Period: {best['period_name']}")
    print(f"  Multiplier: {best['multiplier_func']}")
    print(f"  Parameters: {best['params']}")
    print(f"  RMS: {best['rms']:.1f} km/s")
    
    return results_sorted

def create_visualization(results, obs_indices, v_observed, stars):
    """Create diagnostic plots."""
    
    print("\nGenerating visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Inverse Multiplier Optimization Results (1.8M Stars)', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: RMS ranking
    ax = axes[0, 0]
    names = [f"{r['period_name'][:8]}+{r['multiplier_func'][:10]}" for r in results[:10]]
    rms_vals = [r['rms'] for r in results[:10]]
    colors = ['green' if rms < 20 else 'orange' if rms < 50 else 'red' for rms in rms_vals]
    
    ax.barh(range(len(names)), rms_vals, color=colors, alpha=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('RMS (km/s)')
    ax.set_title('Top 10 Results (lower is better)')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    # Plot 2: Best result rotation curve
    ax = axes[0, 1]
    best = results[0]
    R_obs = stars.iloc[obs_indices]['R'].values
    
    # Bin by radius for clarity
    R_bins = np.linspace(R_obs.min(), R_obs.max(), 20)
    R_bin_centers = (R_bins[:-1] + R_bins[1:]) / 2
    
    v_obs_binned = []
    v_model_binned = []
    for i in range(len(R_bins) - 1):
        in_bin = (R_obs >= R_bins[i]) & (R_obs < R_bins[i+1])
        if np.sum(in_bin) > 0:
            v_obs_binned.append(np.median(v_observed[in_bin]))
            v_model_binned.append(np.median(best['v_model'][in_bin]))
        else:
            v_obs_binned.append(np.nan)
            v_model_binned.append(np.nan)
    
    ax.plot(R_bin_centers, v_obs_binned, 'ko-', markersize=5, alpha=0.6, 
            label='Target (binned)', linewidth=2)
    ax.plot(R_bin_centers, v_model_binned, 'r.-', markersize=4, alpha=0.7,
           label=f"{best['period_name']}+{best['multiplier_func']}", linewidth=1.5)
    ax.axhline(220, color='gray', linestyle=':', alpha=0.5, label='220 km/s')
    ax.set_xlabel('R [kpc]')
    ax.set_ylabel('v [km/s]')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Best: RMS={best['rms']:.1f} km/s")
    
    # Plot 3: Residuals for best
    ax = axes[1, 0]
    residuals = best['v_model'] - v_observed
    ax.scatter(R_obs, residuals, s=5, alpha=0.3, c='blue')
    ax.axhline(0, color='k', linestyle='--')
    ax.axhline(np.median(residuals), color='r', linestyle=':', label=f'Median: {np.median(residuals):.1f}')
    ax.set_xlabel('R [kpc]')
    ax.set_ylabel('v_model - v_obs [km/s]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Residuals (Best Result)')
    
    # Plot 4: Parameter values for top results
    ax = axes[1, 1]
    ax.axis('off')
    
    text = "Top 5 Results:\n\n"
    for i, res in enumerate(results[:5], 1):
        text += f"{i}. {res['period_name']} + {res['multiplier_func']}\n"
        text += f"   RMS: {res['rms']:.1f} km/s\n"
        text += f"   Params: {[f'{p:.3f}' for p in res['params']]}\n"
        text += f"   Time: {res['time']:.1f}s ({res['n_calls']} calls)\n\n"
    
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', family='monospace')
    
    plt.tight_layout()
    output_path = 'gravitywavebaseline/inverse_multiplier_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    results = run_inverse_calculation()

