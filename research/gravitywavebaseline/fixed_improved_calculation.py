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
# Gravitational constant (km/s)^2 kpc / Msun
G_KPC = 4.30091e-6

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

def multiplier_constant_scale(lam, r, params, xp=np):
    """Constant Richter-style boost independent of lambda."""
    (scale,) = params
    return xp.ones_like(lam) * scale

def multiplier_log_richter(lam, r, params, xp=np):
    """
    Richter-style logarithmic multiplier:
        g = 1 + A * log10(lambda / lambda_0)
    """
    A, lambda_0 = params
    ratio = xp.maximum(lam / lambda_0, 1e-4)
    return 1.0 + A * xp.log10(ratio)

def multiplier_piecewise_steps(lam, r, params, xp=np):
    """
    Piecewise step multiplier with 3 levels:
        g = level1       if lambda < lambda_1
          = level2       if lambda_1 <= lambda < lambda_2
          = level3       otherwise
    params = (lambda_1, lambda_2, level1, level2, level3)
    """
    lambda_1, lambda_2, level1, level2, level3 = params
    out = xp.full_like(lam, level3)
    out = xp.where(lam < lambda_1, level1, out)
    out = xp.where((lam >= lambda_1) & (lam < lambda_2), level2, out)
    return out

def multiplier_power_ramp(lam, r, params, xp=np):
    """
    g = (lambda/lambda_0)^alpha
    Pure power ramp (no +1 offset).
    """
    lambda_0, alpha = params
    ratio = xp.maximum(lam / lambda_0, 1e-4)
    return ratio**alpha

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
                 use_gas=True, use_selection_weights=True, mass_boost=1.0):
        
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np
        
        print("\n" + "="*80)
        print("FIXED GRAVITY CALCULATOR")
        print("="*80)
        
        self.N_stars = len(stars_data)
        print(f"\n  Stars: {self.N_stars:,}")
        
        # Load CPU copies (always)
        self.x_cpu = stars_data['x'].values.astype(np.float32)
        self.y_cpu = stars_data['y'].values.astype(np.float32)
        self.z_cpu = stars_data['z'].values.astype(np.float32)
        self.M_cpu = stars_data['M_star'].values.astype(np.float32)
        
        if self.use_gpu:
            self.x = cp.array(self.x_cpu)
            self.y = cp.array(self.y_cpu)
            self.z = cp.array(self.z_cpu)
            self.M = cp.array(self.M_cpu)
        else:
            self.x = self.x_cpu
            self.y = self.y_cpu
            self.z = self.z_cpu
            self.M = self.M_cpu
        
        self.R_cpu = np.sqrt(self.x_cpu**2 + self.y_cpu**2)
        self.R = cp.array(self.R_cpu) if self.use_gpu else self.R_cpu
        
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
            weights = estimate_selection_weights_capped(self.R_cpu, self.z_cpu, self.M_cpu)
            self.weights_cpu = weights.astype(np.float32)
            self.weights = cp.array(self.weights_cpu) if self.use_gpu else self.weights_cpu
            
            print(f"    Weight range: {weights.min():.2f} - {weights.max():.2f} (CAPPED)")
            print(f"    Mean: {weights.mean():.2f}")
        else:
            self.weights_cpu = np.ones(self.N_stars, dtype=np.float32)
            self.weights = cp.array(self.weights_cpu) if self.use_gpu else self.weights_cpu
            print("\n  Selection weights: DISABLED")
        
        # Components
        self.use_bulge = use_bulge
        self.use_gas = use_gas
        self.v_phi_cpu = stars_data['v_phi'].values.astype(np.float32) if 'v_phi' in stars_data.columns else np.full(self.N_stars, np.nan, dtype=np.float32)
        self.v_phi = cp.array(self.v_phi_cpu) if self.use_gpu else self.v_phi_cpu

        self.mass_boost = float(mass_boost)
        print(f"    Mass boost factor: {self.mass_boost:.2f}x")
        
        print(f"\n  Components:")
        print(f"    Stars: ON")
        print(f"    Bulge: {'ON' if use_bulge else 'OFF'}")
        print(f"    Gas: {'ON' if use_gas else 'OFF'}")
        
        # Batching parameters (observations vs. source chunks)
        self.batch_size = 200 if self.use_gpu else 500
        self.source_chunk_size = 50000 if self.use_gpu else 10000
        print(f"    Observation batch size: {self.batch_size}")
        print(f"    Source chunk size: {self.source_chunk_size}")
        
        print(f"\n[OK] Calculator initialized")
    
    def compute_total_velocity(self, obs_indices, period_name, multiplier_func, params,
                               M_disk=5e10, batch_size=None):
        """Compute total velocity with all components."""
        
        batch_size = batch_size or self.batch_size
        
        # Stellar component
        v_stars = self._compute_stellar_component(
            obs_indices, period_name, multiplier_func, params, batch_size
        )
        
        # Analytical components use CPU arrays
        R_obs = self.R_cpu[obs_indices]
        v_bulge = hernquist_bulge(R_obs) if self.use_bulge else np.zeros_like(R_obs)
        v_gas = exponential_gas(R_obs) if self.use_gas else np.zeros_like(R_obs)
        
        v_total = np.sqrt(v_stars**2 + v_bulge**2 + v_gas**2)
        
        return v_total, {
            'stars': v_stars,
            'bulge': v_bulge,
            'gas': v_gas
        }
    
    def _compute_stellar_component(self, obs_indices, period_name, multiplier_func, params,
                                   batch_size):
        """Compute stellar contribution using the full Gaia catalogue."""

        N_obs = len(obs_indices)
        v_stars = np.zeros(N_obs, dtype=np.float32)

        # Pre-compute effective masses for ALL stars
        lambda_vals = self.periods[period_name]
        M_effective = self.M * self.weights * self.mass_boost
        total_mass = float(self.xp.sum(M_effective))
        print(f"    Effective stellar mass in sum: {total_mass:.2e} M_sun")
        print(f"    Obs batch size: {batch_size}, Source chunk: {self.source_chunk_size}")

        n_obs_batches = (N_obs + batch_size - 1) // batch_size

        for i in range(n_obs_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, N_obs)
            batch_obs_idx = obs_indices[start_idx:end_idx]

            if self.use_gpu:
                obs_idx_xp = cp.array(batch_obs_idx, dtype=cp.int32)
                x_obs = self.x[obs_idx_xp]
                y_obs = self.y[obs_idx_xp]
                z_obs = self.z[obs_idx_xp]
                R_obs_batch = self.R[obs_idx_xp]
            else:
                x_obs = self.x[batch_obs_idx]
                y_obs = self.y[batch_obs_idx]
                z_obs = self.z[batch_obs_idx]
                R_obs_batch = self.R[batch_obs_idx]

            g_total_batch = self.xp.zeros_like(R_obs_batch)

            for src_start in range(0, self.N_stars, self.source_chunk_size):
                src_end = min(src_start + self.source_chunk_size, self.N_stars)
                src_slice = slice(src_start, src_end)

                x_src = self.x[src_slice]
                y_src = self.y[src_slice]
                z_src = self.z[src_slice]
                m_src = M_effective[src_slice]
                lambda_src = lambda_vals[src_slice]

                dx = x_obs[:, None] - x_src[None, :]
                dy = y_obs[:, None] - y_src[None, :]
                dz = z_obs[:, None] - z_src[None, :]
                r = self.xp.sqrt(dx**2 + dy**2 + dz**2 + 0.01**2)

            g_base = G_KPC * m_src[None, :] / r**2
            multiplier = multiplier_func(lambda_src[None, :], r, params, self.xp)
            g_enhanced = g_base * multiplier

            # Unit radial vector at observation position
            radial_norm = R_obs_batch + 1e-6
            ux = x_obs / radial_norm
            uy = y_obs / radial_norm

            force_x = g_enhanced * (dx / r)
            force_y = g_enhanced * (dy / r)
            g_radial = force_x * ux[:, None] + force_y * uy[:, None]

            g_total_batch += self.xp.sum(g_radial, axis=1)

            v_squared = R_obs_batch * self.xp.abs(g_total_batch)
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
    
    # FIX: Pass indices instead of R values
    v_observed_np = np.asarray(v_observed, dtype=np.float32)
    
    n_calls = [0]
    best_chi = [np.inf]
    
    def objective(params):
        try:
            n_calls[0] += 1
            v_model, _ = calculator.compute_total_velocity(
                obs_indices, period_name, multiplier_func, params
            )
            # v_model is already NumPy
            chi_sq = float(np.sum((v_model - v_observed_np)**2))
            
            # Track best
            if chi_sq < best_chi[0]:
                best_chi[0] = chi_sq
                rms = np.sqrt(np.mean((v_model - v_observed_np)**2))
                if n_calls[0] % 5 == 0:  # Print every 5 calls
                    print(f"      Call {n_calls[0]}: RMS={rms:.1f} km/s")
            
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
        obs_indices, period_name, multiplier_func, result.x
    )
    
    rms = np.sqrt(np.mean((v_model - v_observed_np)**2))
    
    print(f"    RMS: {rms:.1f} km/s (time: {t1-t0:.1f}s)")
    print(f"    Params: {result.x}")
    
    # Component breakdown
    for comp, v in v_components.items():
        v_np = np.asarray(v, dtype=np.float32)
        v_model_np = np.asarray(v_model, dtype=np.float32)
        contrib = np.mean(v_np**2) / (np.mean(v_model_np**2) + 1e-10) * 100
        print(f"      {comp}: {np.mean(v_np):.1f} km/s ({contrib:.1f}%)")
    
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
    
    # Observations: use real Gaia v_phi where available
    vphi_values = gaia['v_phi'].values.astype(np.float32) if 'v_phi' in gaia.columns else None
    if vphi_values is None:
        raise ValueError("v_phi column missing in Gaia dataset; cannot fit to real velocities.")
    
    valid_mask = np.isfinite(vphi_values) & (vphi_values != 0.0)
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        raise ValueError("No stars with valid v_phi values found.")
    
    N_obs = min(1000, len(valid_indices))
    print(f"\nObservation set: {N_obs} stars sampled from {len(valid_indices):,} with measured v_phi")
    obs_indices = np.random.choice(valid_indices, size=N_obs, replace=False)
    v_observed = vphi_values[obs_indices]
    
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
                ('dynamical', multiplier_power_law, [(0, 3), (1, 20), (1, 4)]),
                ('jeans', multiplier_log_richter, [(0, 10), (0.5, 50)]),
                ('jeans', multiplier_piecewise_steps, [(1, 10), (10, 100), (0.5, 5.0), (1.0, 10.0), (5.0, 50.0)]),
                ('jeans', multiplier_power_ramp, [(0.5, 50), (0.5, 5)])
            ]
        },

        # 7. Demonstrate explicit mass boost + constant Richter scale
        {
            'name': 'MASS BOOST x200',
            'use_bulge': True,
            'use_gas': True,
            'use_weights': True,
            'mass_boost': 200.0,
            'tests': [
                ('jeans', multiplier_constant_scale, [(0.1, 500)])
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
            use_selection_weights=config['use_weights'],
            mass_boost=config.get('mass_boost', 1.0)
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
    if len(results_sorted) == 0:
        print("\n[ERROR] No results generated - all tests failed!")
        return []
    
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

