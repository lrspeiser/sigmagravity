"""
FIXED IMPROVED MULTIPLIER CALCULATION

Fixes the selection weight issue that caused stellar component to vanish.

Key fix: Cap selection weights to reasonable range (0.2 to 5.0)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import json
from scipy.optimize import differential_evolution

try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("[OK] GPU (CuPy) available")
except ImportError:
    cp = np
    GPU_AVAILABLE = False
    print("[!] GPU not available, using CPU")

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
# FIXED SELECTION WEIGHTS
# ============================================================================

def estimate_selection_weights_capped(R, z, M_star):
    """
    FIXED version with capped weights.
    
    Key fix: Weights capped to [0.2, 5.0] range
    This prevents extreme values from breaking the calculation.
    """
    
    # Spatial completeness
    R_disk = 3.0
    spatial_expected = np.exp(-R / R_disk)
    
    # Observed distribution
    R_bins = np.linspace(0, 25, 50)
    hist, _ = np.histogram(R, bins=R_bins, weights=np.ones_like(R))
    R_centers = 0.5 * (R_bins[:-1] + R_bins[1:])
    spatial_observed = hist / (hist.sum() + 1e-10)
    
    # Interpolate
    spatial_obs_interp = np.interp(R, R_centers, spatial_observed, left=1e-10, right=1e-10)
    spatial_expected_interp = np.exp(-R / R_disk)
    
    # Weight = expected / observed
    spatial_weight = spatial_expected_interp / (spatial_obs_interp + 1e-10)
    
    # Magnitude completeness (simplified)
    distance_pc = R * 1000
    M_G_approx = 5 - 2.5 * np.log10(M_star + 0.1)
    G_approx = M_G_approx + 5 * np.log10(distance_pc + 1) - 5
    
    mag_weight = np.ones_like(G_approx)
    mag_weight[G_approx > 16] = np.exp(-(G_approx[G_approx > 16] - 16) / 2)
    
    # Combined weight
    total_weight = spatial_weight * mag_weight
    
    # Normalize so mean = 1 FIRST
    total_weight /= (total_weight.mean() + 1e-10)
    
    # KEY FIX: Cap to reasonable range AFTER normalization
    total_weight = np.clip(total_weight, 0.2, 5.0)
    
    # Re-normalize after clipping
    total_weight /= (total_weight.mean() + 1e-10)
    
    return total_weight

# ============================================================================
# MULTIPLIER FUNCTIONS
# ============================================================================

def multiplier_power_law(lam, r, params, xp=np):
    """Original best: f = 1 + A(lambda/lambda_0)^alpha"""
    A, lambda_0, alpha = params
    return 1.0 + A * (lam / lambda_0)**alpha

def multiplier_distance_dependent(lam, r, params, xp=np):
    """Distance-modulated: f = 1 + A(lambda/lambda_0)^alpha × exp(-r/r_0)"""
    A, lambda_0, alpha, r_0 = params
    return 1.0 + A * (lam / lambda_0)**alpha * xp.exp(-r / (r_0 + 0.1))

def multiplier_saturating(lam, r, params, xp=np):
    """Saturating: f = 1 + A[1 - 1/(1 + (lambda/lambda_0)^p)]"""
    A, lambda_0, p = params
    return 1.0 + A * (1.0 - 1.0 / (1.0 + (lam / lambda_0)**p))

# ============================================================================
# IMPROVED CALCULATOR (FIXED)
# ============================================================================

class FixedGravityCalculator:
    """Fixed calculator with proper selection weights."""
    
    def __init__(self, stars_data, use_gpu=True, use_bulge=True, 
                 use_gas=True, use_selection_weights=True):
        
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np
        
        print("\n" + "="*80)
        print("FIXED GRAVITY CALCULATOR")
        print("="*80)
        
        self.N_stars = len(stars_data)
        print(f"\n  Stars: {self.N_stars:,}")
        
        # Load data
        if self.use_gpu:
            self.x = cp.array(stars_data['x'].values, dtype=cp.float32)
            self.y = cp.array(stars_data['y'].values, dtype=cp.float32)
            self.z = cp.array(stars_data['z'].values, dtype=cp.float32)
            self.M = cp.array(stars_data['M_star'].values, dtype=cp.float32)
        else:
            self.x = stars_data['x'].values.astype(np.float32)
            self.y = stars_data['y'].values.astype(np.float32)
            self.z = stars_data['z'].values.astype(np.float32)
            self.M = stars_data['M_star'].values.astype(np.float32)
        
        self.R = self.xp.sqrt(self.x**2 + self.y**2)
        
        # Load periods
        self.periods = {}
        for col in stars_data.columns:
            if col.startswith('lambda_'):
                period_name = col.replace('lambda_', '')
                if self.use_gpu:
                    self.periods[period_name] = cp.array(stars_data[col].values, dtype=cp.float32)
                else:
                    self.periods[period_name] = stars_data[col].values.astype(np.float32)
        
        print(f"  Periods: {list(self.periods.keys())}")
        
        # FIXED selection weights
        self.use_selection_weights = use_selection_weights
        if use_selection_weights:
            print("\n  Calculating CAPPED selection weights...")
            R_cpu = cp.asnumpy(self.R) if self.use_gpu else self.R
            z_cpu = cp.asnumpy(self.z) if self.use_gpu else self.z
            M_cpu = cp.asnumpy(self.M) if self.use_gpu else self.M
            
            weights = estimate_selection_weights_capped(R_cpu, z_cpu, M_cpu)
            
            if self.use_gpu:
                self.weights = cp.array(weights, dtype=cp.float32)
            else:
                self.weights = weights.astype(np.float32)
            
            print(f"    Weight range: {weights.min():.2f} - {weights.max():.2f} (CAPPED)")
            print(f"    Mean: {weights.mean():.2f}")
        else:
            self.weights = self.xp.ones(self.N_stars, dtype=self.xp.float32)
            print("\n  Selection weights: DISABLED")
        
        # Components
        self.use_bulge = use_bulge
        self.use_gas = use_gas
        
        print(f"\n  Components:")
        print(f"    Stars: ON")
        print(f"    Bulge: {'ON' if use_bulge else 'OFF'}")
        print(f"    Gas: {'ON' if use_gas else 'OFF'}")
        
        print(f"\n[OK] Calculator initialized")
    
    def compute_total_velocity(self, R_obs, period_name, multiplier_func, params,
                               M_disk=5e10, batch_size=5000):
        """Compute total velocity with all components."""
        
        # Stellar component
        v_stars = self._compute_stellar_component(
            R_obs, period_name, multiplier_func, params, M_disk, batch_size
        )
        
        # Bulge
        v_bulge = hernquist_bulge(R_obs) if self.use_bulge else np.zeros_like(R_obs)
        
        # Gas
        v_gas = exponential_gas(R_obs) if self.use_gas else np.zeros_like(R_obs)
        
        # Total (quadrature)
        v_total = np.sqrt(v_stars**2 + v_bulge**2 + v_gas**2)
        
        return v_total, {
            'stars': v_stars,
            'bulge': v_bulge,
            'gas': v_gas
        }
    
    def _compute_stellar_component(self, R_obs, period_name, multiplier_func, params,
                                   M_disk, batch_size):
        """Compute stellar contribution."""
        
        N_obs = len(R_obs)
        v_stars = np.zeros(N_obs, dtype=np.float32)
        
        # Scale masses with weights
        M_total_weighted = float(self.xp.sum(self.M * self.weights))
        M_scale_factor = M_disk / M_total_weighted
        M_scaled = self.M * M_scale_factor * self.weights
        
        # Diagnostic
        M_total_scaled = float(self.xp.sum(M_scaled))
        print(f"    Stellar mass scaling: {M_total_weighted:.2e} -> {M_total_scaled:.2e} M_sun")
        
        # Get periods
        lambda_vals = self.periods[period_name]
        
        # Process in batches
        n_batches = (N_obs + batch_size - 1) // batch_size
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, N_obs)
            
            R_batch = R_obs[start_idx:end_idx]
            
            if self.use_gpu:
                x_obs = cp.array(R_batch)
                y_obs = cp.zeros_like(x_obs)
                z_obs = cp.zeros_like(x_obs)
            else:
                x_obs = R_batch
                y_obs = np.zeros_like(x_obs)
                z_obs = np.zeros_like(x_obs)
            
            # Distances
            dx = x_obs[:, None] - self.x[None, :]
            dy = y_obs[:, None] - self.y[None, :]
            dz = z_obs[:, None] - self.z[None, :]
            r = self.xp.sqrt(dx**2 + dy**2 + dz**2 + 0.01**2)
            
            # Gravity
            g_base = G_KPC * M_scaled[None, :] / r**2
            
            # Multiplier
            multiplier = multiplier_func(lambda_vals[None, :], r, params, self.xp)
            g_enhanced = g_base * multiplier
            
            # Radial component
            cos_theta = dx / r
            g_radial = g_enhanced * cos_theta
            g_total = self.xp.sum(g_radial, axis=1)
            
            # Velocity
            v_squared = R_batch * g_total
            v_batch = self.xp.sqrt(self.xp.maximum(v_squared, 0))
            
            if self.use_gpu:
                v_stars[start_idx:end_idx] = cp.asnumpy(v_batch)
            else:
                v_stars[start_idx:end_idx] = v_batch
        
        return v_stars

# ============================================================================
# OPTIMIZATION
# ============================================================================

def optimize_fixed(calculator, obs_indices, v_observed, period_name,
                  multiplier_func, param_bounds):
    """Optimize with fixed calculator."""
    
    print(f"\n  Testing: {period_name} + {multiplier_func.__name__}")
    t0 = time.time()
    
    R_obs = calculator.R.get() if calculator.use_gpu else calculator.R
    R_obs = R_obs[obs_indices]
    
    def objective(params):
        try:
            v_model, _ = calculator.compute_total_velocity(
                R_obs, period_name, multiplier_func, params
            )
            chi_sq = float(np.sum((v_model - v_observed)**2))
            return chi_sq
        except Exception as e:
            print(f"      Error in objective: {e}")
            return 1e10  # Return large value on error
    
    result = differential_evolution(
        objective,
        bounds=param_bounds,
        maxiter=20,
        popsize=6,
        seed=42,
        workers=1,
        polish=False,
        updating='immediate',  # Process one at a time
        atol=0,
        tol=0.01
    )
    
    t1 = time.time()
    
    # Final model
    v_model, v_components = calculator.compute_total_velocity(
        R_obs, period_name, multiplier_func, result.x
    )
    
    rms = np.sqrt(np.mean((v_model - v_observed)**2))
    
    print(f"    RMS: {rms:.1f} km/s (time: {t1-t0:.1f}s)")
    print(f"    Params: {result.x}")
    
    # Component breakdown
    for comp, v in v_components.items():
        contrib = np.mean(v**2) / np.mean(v_model**2) * 100
        print(f"      {comp}: {np.mean(v):.1f} km/s ({contrib:.1f}%)")
    
    return {
        'period_name': period_name,
        'multiplier_func': multiplier_func.__name__,
        'params': result.x.tolist(),
        'rms': float(rms),
        'chi_squared': float(result.fun),
        'time': t1 - t0
    }

# ============================================================================
# MAIN
# ============================================================================

def run_fixed_analysis():
    """Run analysis with fixed weights."""
    
    print("="*80)
    print("FIXED IMPROVED ANALYSIS")
    print("="*80)
    print("\nFix: Capped selection weights to [0.2, 5.0]")
    print("Goal: Reduce RMS from 74.5 km/s")
    
    # Load
    print("\nLoading data...")
    gaia = pd.read_parquet('gravitywavebaseline/gaia_with_periods.parquet')
    print(f"  Loaded {len(gaia):,} stars")
    
    # Observations
    N_obs = min(1000, len(gaia))
    obs_indices = np.linspace(0, len(gaia)-1, N_obs, dtype=int)
    v_observed = np.ones(N_obs) * 220.0
    
    # Test configurations
    print("\n" + "="*80)
    print("TEST MATRIX")
    print("="*80)
    
    configs = [
        # 1. Baseline (verify we reproduce 74.5 km/s)
        {
            'name': 'BASELINE',
            'use_bulge': False,
            'use_gas': False,
            'use_weights': False,
            'tests': [('jeans', multiplier_power_law, [(0, 3), (1, 20), (1, 4)])]
        },
        
        # 2. Add bulge only
        {
            'name': 'WITH BULGE',
            'use_bulge': True,
            'use_gas': False,
            'use_weights': False,
            'tests': [('jeans', multiplier_power_law, [(0, 3), (1, 20), (1, 4)])]
        },
        
        # 3. Add gas
        {
            'name': 'WITH BULGE + GAS',
            'use_bulge': True,
            'use_gas': True,
            'use_weights': False,
            'tests': [('jeans', multiplier_power_law, [(0, 3), (1, 20), (1, 4)])]
        },
        
        # 4. Add selection weights (fixed)
        {
            'name': 'WITH ALL + WEIGHTS',
            'use_bulge': True,
            'use_gas': True,
            'use_weights': True,
            'tests': [('jeans', multiplier_power_law, [(0, 3), (1, 20), (1, 4)])]
        },
        
        # 5. Try distance-dependent multiplier
        {
            'name': 'DISTANCE-DEPENDENT',
            'use_bulge': True,
            'use_gas': True,
            'use_weights': True,
            'tests': [('jeans', multiplier_distance_dependent, [(0, 3), (1, 20), (1, 4), (1, 20)])]
        },
        
        # 6. Try other periods
        {
            'name': 'OTHER PERIODS',
            'use_bulge': True,
            'use_gas': True,
            'use_weights': True,
            'tests': [
                ('orbital', multiplier_power_law, [(0, 3), (1, 20), (1, 4)]),
                ('dynamical', multiplier_power_law, [(0, 3), (1, 20), (1, 4)])
            ]
        }
    ]
    
    all_results = []
    
    for config in configs:
        print(f"\n{'='*80}")
        print(f"CONFIG: {config['name']}")
        print(f"{'='*80}")
        
        calc = FixedGravityCalculator(
            gaia,
            use_gpu=GPU_AVAILABLE,
            use_bulge=config['use_bulge'],
            use_gas=config['use_gas'],
            use_selection_weights=config['use_weights']
        )
        
        for period, func, bounds in config['tests']:
            try:
                result = optimize_fixed(calc, obs_indices, v_observed, period, func, bounds)
                result['config'] = config['name']
                all_results.append(result)
            except Exception as e:
                print(f"    [!] Failed: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    results_sorted = sorted(all_results, key=lambda x: x['rms'])
    
    print(f"\n{'Rank':<5} {'Config':<25} {'Period':<12} {'Multiplier':<20} {'RMS':<12}")
    print("-"*80)
    for i, res in enumerate(results_sorted, 1):
        print(f"{i:<5} {res['config']:<25} {res['period_name']:<12} "
              f"{res['multiplier_func']:<20} {res['rms']:<12.1f}")
    
    # Save
    output = 'gravitywavebaseline/fixed_improved_results.json'
    with open(output, 'w') as f:
        json.dump(results_sorted, f, indent=2)
    
    print(f"\n[OK] Saved: {output}")
    
    # Progress report
    best = results_sorted[0]
    baseline = next((r for r in all_results if r['config'] == 'BASELINE'), None)
    
    if baseline:
        improvement = baseline['rms'] - best['rms']
        pct = improvement / baseline['rms'] * 100
        
        print(f"\n{'='*80}")
        print("IMPROVEMENT SUMMARY")
        print("="*80)
        print(f"\nBaseline RMS: {baseline['rms']:.1f} km/s")
        print(f"Best RMS: {best['rms']:.1f} km/s")
        print(f"Improvement: {improvement:.1f} km/s ({pct:.1f}%)")
        print(f"\nBest config: {best['config']}")
        print(f"Period: {best['period_name']}")
        print(f"Multiplier: {best['multiplier_func']}")
        
        if best['rms'] < 20:
            print("\n[SUCCESS] Target achieved!")
        elif best['rms'] < 40:
            print("\n[OK] Significant progress toward target")
        else:
            print("\n[!] More optimization needed")
    
    return results_sorted

if __name__ == "__main__":
    results = run_fixed_analysis()

