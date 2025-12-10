"""
FIXED CALCULATION WITH SAMPLING

Uses sampling strategy like original (which worked) to avoid GPU OOM.
Key differences from broken version:
1. Sample source stars (not all 1.8M at once)
2. Proper array handling for CuPy/NumPy
3. Capped selection weights
"""

import numpy as np
import pandas as pd
import time
import json
from scipy.optimize import differential_evolution

try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("[OK] GPU available")
except ImportError:
    cp = np
    GPU_AVAILABLE = False
    print("[!] GPU not available")

G_KPC = 4.498e-12

# ============================================================================
# ANALYTICAL COMPONENTS
# ============================================================================

def hernquist_bulge(R, M_bulge=1.5e10, a_bulge=0.7):
    """Hernquist bulge."""
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
# SELECTION WEIGHTS (CAPPED)
# ============================================================================

def estimate_selection_weights_capped(R, z, M_star):
    """Capped selection weights."""
    R_disk = 3.0
    spatial_expected = np.exp(-R / R_disk)
    R_bins = np.linspace(0, 25, 50)
    hist, _ = np.histogram(R, bins=R_bins)
    R_centers = 0.5 * (R_bins[:-1] + R_bins[1:])
    spatial_observed = hist / (hist.sum() + 1e-10)
    spatial_obs_interp = np.interp(R, R_centers, spatial_observed, left=1e-10, right=1e-10)
    spatial_expected_interp = np.exp(-R / R_disk)
    spatial_weight = spatial_expected_interp / (spatial_obs_interp + 1e-10)
    
    distance_pc = R * 1000
    M_G_approx = 5 - 2.5 * np.log10(M_star + 0.1)
    G_approx = M_G_approx + 5 * np.log10(distance_pc + 1) - 5
    mag_weight = np.ones_like(G_approx)
    mag_weight[G_approx > 16] = np.exp(-(G_approx[G_approx > 16] - 16) / 2)
    
    total_weight = spatial_weight * mag_weight
    total_weight /= (total_weight.mean() + 1e-10)
    total_weight = np.clip(total_weight, 0.2, 5.0)
    total_weight /= (total_weight.mean() + 1e-10)
    
    return total_weight

# ============================================================================
# MULTIPLIERS
# ============================================================================

def multiplier_power_law(lam, r, params, xp=np):
    A, lambda_0, alpha = params
    return 1.0 + A * (lam / lambda_0)**alpha

def multiplier_distance_dependent(lam, r, params, xp=np):
    A, lambda_0, alpha, r_0 = params
    return 1.0 + A * (lam / lambda_0)**alpha * xp.exp(-r / (r_0 + 0.1))

# ============================================================================
# SAMPLED CALCULATOR (BASED ON WORKING ORIGINAL)
# ============================================================================

class SampledGravityCalculator:
    """
    Uses sampling to avoid GPU OOM with 1.8M stars.
    Based on inverse_multiplier_calculation.py which worked.
    """
    
    def __init__(self, stars_data, use_gpu=True, use_bulge=True, 
                 use_gas=True, use_selection_weights=True):
        
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np
        
        print(f"\n{'='*80}")
        print("SAMPLED GRAVITY CALCULATOR (GPU Memory Safe)")
        print("="*80)
        
        self.N_stars = len(stars_data)
        print(f"  Stars: {self.N_stars:,}")
        
        # Keep data on CPU, transfer to GPU in batches
        self.x = stars_data['x'].values.astype(np.float32)
        self.y = stars_data['y'].values.astype(np.float32)
        self.z = stars_data['z'].values.astype(np.float32)
        self.M = stars_data['M_star'].values.astype(np.float32)
        self.R = np.sqrt(self.x**2 + self.y**2)
        
        # Periods
        self.periods = {}
        for col in stars_data.columns:
            if col.startswith('lambda_'):
                period_name = col.replace('lambda_', '')
                self.periods[period_name] = stars_data[col].values.astype(np.float32)
        
        print(f"  Periods: {list(self.periods.keys())}")
        
        # Selection weights (capped)
        if use_selection_weights:
            print("  Calculating capped selection weights...")
            weights = estimate_selection_weights_capped(self.R, self.z, self.M)
            self.weights = weights.astype(np.float32)
            print(f"    Range: {weights.min():.2f} - {weights.max():.2f}")
        else:
            self.weights = np.ones(self.N_stars, dtype=np.float32)
        
        self.use_bulge = use_bulge
        self.use_gas = use_gas
        
        print(f"  Components: Stars={'ON'}, Bulge={'ON' if use_bulge else 'OFF'}, Gas={'ON' if use_gas else 'OFF'}")
        print(f"[OK] Calculator initialized\n")
    
    def compute_total_velocity(self, obs_indices, period_name, multiplier_func, params,
                               M_disk=5e10, n_source_samples=50000):
        """
        Compute velocity using SAMPLED source stars (GPU memory safe).
        """
        
        N_obs = len(obs_indices)
        
        # STELLAR COMPONENT with sampling
        v_stars = self._compute_stellar_sampled(
            obs_indices, period_name, multiplier_func, params, M_disk, n_source_samples
        )
        
        # ANALYTICAL COMPONENTS (NumPy only)
        R_obs = self.R[obs_indices]
        v_bulge = hernquist_bulge(R_obs) if self.use_bulge else np.zeros_like(R_obs)
        v_gas = exponential_gas(R_obs) if self.use_gas else np.zeros_like(R_obs)
        
        # Total
        v_total = np.sqrt(v_stars**2 + v_bulge**2 + v_gas**2)
        
        return v_total, {'stars': v_stars, 'bulge': v_bulge, 'gas': v_gas}
    
    def _compute_stellar_sampled(self, obs_indices, period_name, multiplier_func, params,
                                 M_disk, n_source_samples):
        """Sample source stars to avoid GPU OOM."""
        
        N_obs = len(obs_indices)
        v_circ = np.zeros(N_obs, dtype=np.float32)
        
        # Scale masses
        M_total_weighted = np.sum(self.M * self.weights)
        M_scale_factor = M_disk / M_total_weighted
        M_scaled = self.M * M_scale_factor * self.weights
        
        # Sample source stars (stratified by radius)
        R_bins = np.percentile(self.R, np.linspace(0, 100, 21))
        source_indices = []
        
        for i in range(len(R_bins) - 1):
            in_bin = np.where((self.R >= R_bins[i]) & (self.R < R_bins[i+1]))[0]
            if len(in_bin) > 0:
                n_sample = min(len(in_bin), n_source_samples // 20)
                sampled = np.random.choice(in_bin, size=n_sample, replace=False)
                source_indices.extend(sampled)
        
        source_indices = np.array(source_indices)
        
        # Transfer sampled sources to GPU
        if self.use_gpu:
            x_src = cp.array(self.x[source_indices])
            y_src = cp.array(self.y[source_indices])
            z_src = cp.array(self.z[source_indices])
            M_src = cp.array(M_scaled[source_indices])
            lambda_src = cp.array(self.periods[period_name][source_indices])
        else:
            x_src = self.x[source_indices]
            y_src = self.y[source_indices]
            z_src = self.z[source_indices]
            M_src = M_scaled[source_indices]
            lambda_src = self.periods[period_name][source_indices]
        
        # Process observation points in batches
        obs_batch_size = 1000 if self.use_gpu else 100
        n_batches = (N_obs + obs_batch_size - 1) // obs_batch_size
        
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
            
            # Distances
            dx = x_obs[:, None] - x_src[None, :]
            dy = y_obs[:, None] - y_src[None, :]
            dz = z_obs[:, None] - z_src[None, :]
            r = self.xp.sqrt(dx**2 + dy**2 + dz**2 + 0.01**2)
            
            # Gravity
            g_base = G_KPC * M_src[None, :] / r**2
            multiplier = multiplier_func(lambda_src[None, :], r, params, self.xp)
            g_enhanced = g_base * multiplier
            
            # Radial component
            cos_theta = dx / r
            g_radial = g_enhanced * cos_theta
            g_total = self.xp.sum(g_radial, axis=1)
            
            # Velocity
            v_squared = R_obs * g_total
            v_batch = self.xp.sqrt(self.xp.maximum(v_squared, 0))
            
            if self.use_gpu:
                v_circ[start_idx:end_idx] = cp.asnumpy(v_batch)
            else:
                v_circ[start_idx:end_idx] = v_batch
        
        return v_circ

# ============================================================================
# OPTIMIZATION
# ============================================================================

def optimize_config(calculator, obs_indices, v_observed, period_name,
                    multiplier_func, param_bounds):
    """Optimize one configuration."""
    
    print(f"\n  Testing: {period_name} + {multiplier_func.__name__}")
    t0 = time.time()
    
    v_obs = np.asarray(v_observed, dtype=np.float32)
    n_calls = [0]
    best_rms = [np.inf]
    
    def objective(params):
        n_calls[0] += 1
        v_model, _ = calculator.compute_total_velocity(
            obs_indices, period_name, multiplier_func, params
        )
        chi_sq = float(np.sum((v_model - v_obs)**2))
        
        rms = np.sqrt(np.mean((v_model - v_obs)**2))
        if rms < best_rms[0]:
            best_rms[0] = rms
            if n_calls[0] % 10 == 0:
                print(f"      Call {n_calls[0]}: RMS={rms:.1f} km/s")
        
        return chi_sq
    
    result = differential_evolution(
        objective,
        bounds=param_bounds,
        maxiter=15,  # Reduce iterations for speed
        popsize=5,
        seed=42,
        workers=1,
        polish=False,
        updating='immediate'
    )
    
    t1 = time.time()
    
    # Final model
    v_model, v_components = calculator.compute_total_velocity(
        obs_indices, period_name, multiplier_func, result.x
    )
    
    rms = np.sqrt(np.mean((v_model - v_obs)**2))
    
    print(f"    [OK] RMS={rms:.1f} km/s (time: {t1-t0:.1f}s, calls: {n_calls[0]})")
    
    # Component breakdown
    for comp, v in v_components.items():
        contrib = np.mean(v**2) / (np.mean(v_model**2) + 1e-10) * 100
        avg_v = np.mean(v)
        print(f"      {comp}: avg={avg_v:.1f} km/s, contrib={contrib:.1f}%")
    
    return {
        'period_name': period_name,
        'multiplier_func': multiplier_func.__name__,
        'params': result.x.tolist(),
        'rms': float(rms),
        'time': t1 - t0,
        'n_calls': n_calls[0]
    }

# ============================================================================
# MAIN
# ============================================================================

def run_sampled_analysis():
    """Run analysis with sampling."""
    
    print("="*80)
    print("IMPROVED ANALYSIS (WITH SAMPLING)")
    print("="*80)
    print("\nGoal: Reduce RMS from 74.5 km/s using improvements")
    print("Strategy: Sample source stars to avoid GPU OOM\n")
    
    # Load
    gaia = pd.read_parquet('gravitywavebaseline/gaia_with_periods.parquet')
    print(f"Loaded {len(gaia):,} stars\n")
    
    # Observations
    N_obs = min(5000, len(gaia))
    obs_indices = np.linspace(0, len(gaia)-1, N_obs, dtype=int)
    v_observed = np.ones(N_obs) * 220.0
    
    # Test matrix - progressive improvements
    configs = [
        # 1. BASELINE (should reproduce ~74.5 km/s)
        {
            'name': '1_BASELINE',
            'use_bulge': False,
            'use_gas': False,
            'use_weights': False,
            'tests': [('jeans', multiplier_power_law, [(0, 3), (1, 20), (1, 4)])]
        },
        
        # 2. ADD BULGE
        {
            'name': '2_WITH_BULGE',
            'use_bulge': True,
            'use_gas': False,
            'use_weights': False,
            'tests': [('jeans', multiplier_power_law, [(0, 3), (1, 20), (1, 4)])]
        },
        
        # 3. ADD GAS
        {
            'name': '3_WITH_BULGE_GAS',
            'use_bulge': True,
            'use_gas': True,
            'use_weights': False,
            'tests': [('jeans', multiplier_power_law, [(0, 3), (1, 20), (1, 4)])]
        },
        
        # 4. ADD WEIGHTS (capped)
        {
            'name': '4_WITH_ALL_WEIGHTS',
            'use_bulge': True,
            'use_gas': True,
            'use_weights': True,
            'tests': [('jeans', multiplier_power_law, [(0, 3), (1, 20), (1, 4)])]
        },
        
        # 5. TRY DISTANCE-DEPENDENT MULTIPLIER
        {
            'name': '5_DISTANCE_DEPENDENT',
            'use_bulge': True,
            'use_gas': True,
            'use_weights': True,
            'tests': [('jeans', multiplier_distance_dependent, [(0, 3), (1, 20), (1, 4), (5, 30)])]
        },
    ]
    
    results = []
    
    for config in configs:
        print(f"{'='*80}")
        print(f"CONFIG: {config['name']}")
        print(f"{'='*80}")
        
        calc = SampledGravityCalculator(
            gaia,
            use_gpu=GPU_AVAILABLE,
            use_bulge=config['use_bulge'],
            use_gas=config['use_gas'],
            use_selection_weights=config['use_weights']
        )
        
        for period, func, bounds in config['tests']:
            try:
                result = optimize_config(calc, obs_indices, v_observed, period, func, bounds)
                result['config'] = config['name']
                results.append(result)
            except Exception as e:
                print(f"  [!] Failed: {e}")
                import traceback
                traceback.print_exc()
    
    # Summary
    print(f"\n{'='*80}")
    print("RESULTS")
    print("="*80)
    
    if len(results) == 0:
        print("\n[ERROR] All tests failed!")
        return []
    
    results.sort(key=lambda x: x['rms'])
    
    print(f"\n{'#':<3} {'Config':<25} {'Period':<10} {'Mult':<20} {'RMS':<10} {'Time':<8}")
    print("-"*85)
    for i, r in enumerate(results, 1):
        print(f"{i:<3} {r['config']:<25} {r['period_name']:<10} {r['multiplier_func']:<20} "
              f"{r['rms']:<10.1f} {r['time']:<8.1f}")
    
    # Save
    with open('gravitywavebaseline/sampled_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Saved: gravitywavebaseline/sampled_results.json")
    
    # Analysis
    baseline = next((r for r in results if r['config'] == '1_BASELINE'), None)
    best = results[0]
    
    if baseline:
        print(f"\n{'='*80}")
        print("IMPROVEMENT ANALYSIS")
        print("="*80)
        print(f"\nBaseline: {baseline['rms']:.1f} km/s")
        print(f"Best:     {best['rms']:.1f} km/s")
        print(f"Improvement: {baseline['rms'] - best['rms']:.1f} km/s ({(baseline['rms']-best['rms'])/baseline['rms']*100:.1f}%)")
        print(f"\nBest config: {best['config']}")
        
        if best['rms'] < 20:
            print("\n[SUCCESS] RMS < 20 km/s achieved!")
        elif best['rms'] < 40:
            print("\n[GOOD] RMS < 40 km/s - publishable result!")
        elif best['rms'] < 60:
            print("\n[OK] Progress made, more work needed")
        else:
            print("\n[WARN] Limited improvement")
    
    return results

if __name__ == "__main__":
    import sys
    sys.stdout.flush()
    results = run_sampled_analysis()
    sys.stdout.flush()

