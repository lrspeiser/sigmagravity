"""
IMPROVED INVERSE MULTIPLIER CALCULATION

Implements improvements to reduce RMS from 74.5 km/s to <20 km/s:

1. Add missing components (analytical bulge + gas)
2. Distance-dependent multipliers
3. Hybrid period combinations
4. Simplified selection bias correction

Run this after calculate_periods.py
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
G_KPC = 4.498e-12  # kpc^3 / M_sun / (km/s)^2

# ============================================================================
# ANALYTICAL MASS COMPONENTS (NO EXTRA DATA NEEDED)
# ============================================================================

def hernquist_potential(R, M_bulge=1.5e10, a_bulge=0.7):
    """
    Hernquist bulge model (no data needed - analytical).
    
    Parameters:
    -----------
    R : array
        Galactocentric radius (kpc)
    M_bulge : float
        Bulge mass (M_sun)
    a_bulge : float
        Scale length (kpc)
    
    Returns:
    --------
    v_bulge : array
        Contribution to circular velocity from bulge (km/s)
    """
    
    # Circular velocity from Hernquist profile
    # v^2 = GM / (r + a)
    v_squared = G_KPC * M_bulge * R / (R + a_bulge)**2
    return np.sqrt(v_squared)

def exponential_gas_disk(R, M_gas=1e10, R_gas=7.0):
    """
    Exponential gas disk (analytical approximation).
    
    Real MW: ~10^10 M_sun in HI + H2
    Without actual HI maps, we use exponential profile
    """
    
    y = R / (2 * R_gas)
    
    # Exact solution for exponential disk
    from scipy.special import i0, i1, k0, k1
    
    v_squared = 4 * np.pi * G_KPC * M_gas * R**2 / (2 * R_gas)**2 * \
                (i0(y) * k0(y) - i1(y) * k1(y))
    
    return np.sqrt(np.maximum(v_squared, 0))

# ============================================================================
# SELECTION BIAS CORRECTION (SIMPLIFIED)
# ============================================================================

def estimate_selection_weights(R, z, M_star, G_mag_limit=18.0):
    """
    Estimate Gaia selection completeness weights.
    
    Without full selection function, use simple model:
    - Spatial: exponential with radius
    - Magnitude: completeness vs apparent magnitude
    
    Returns weight: higher weight = under-represented region
    """
    
    # Spatial completeness (Gaia over-samples solar neighborhood)
    # Expected: exponential disk ~ exp(-R/R_disk)
    # Observed: concentrated at R~8 kpc
    R_disk = 3.0  # kpc scale length
    spatial_expected = np.exp(-R / R_disk)
    spatial_expected /= spatial_expected.sum()
    
    # Observed distribution
    R_bins = np.linspace(0, 25, 50)
    hist, _ = np.histogram(R, bins=R_bins)
    R_centers = 0.5 * (R_bins[:-1] + R_bins[1:])
    spatial_observed = hist / hist.sum()
    
    # Interpolate to get observed probability at each star's R
    spatial_obs_interp = np.interp(R, R_centers, spatial_observed)
    
    # Weight = expected / observed (upweight under-represented regions)
    spatial_expected_interp = np.exp(-R / R_disk)
    spatial_weight = spatial_expected_interp / (spatial_obs_interp + 1e-10)
    
    # Magnitude completeness (brighter stars over-represented)
    # Approximate G from M_star (very rough!)
    # M_G ~ 5 - 2.5 log10(M_star)  (approximate main sequence)
    # G = M_G + 5 log10(distance) - 5
    distance_pc = R * 1000  # kpc to pc
    M_G_approx = 5 - 2.5 * np.log10(M_star + 0.1)
    G_approx = M_G_approx + 5 * np.log10(distance_pc + 1) - 5
    
    # Completeness drops for G > 16
    mag_weight = np.ones_like(G_approx)
    mag_weight[G_approx > 16] = np.exp(-(G_approx[G_approx > 16] - 16) / 2)
    
    # Combined weight
    total_weight = spatial_weight * mag_weight
    
    # Normalize so mean weight = 1
    total_weight /= total_weight.mean()
    
    return total_weight

# ============================================================================
# IMPROVED MULTIPLIER FUNCTIONS
# ============================================================================

def multiplier_distance_dependent(lam, r, params, xp=np):
    """
    NEW: Distance-modulated multiplier
    
    f(lambda, r) = 1 + A × (lambda/lambda_0)^alpha × exp(-r/r_0)
    
    Enhancement decays with distance from source
    """
    A, lambda_0, alpha, r_0 = params
    return 1.0 + A * (lam / lambda_0)**alpha * xp.exp(-r / r_0)

def multiplier_hybrid_saturating(lam, r, params, xp=np):
    """
    NEW: Hybrid between power-law and saturating
    
    f(lambda, r) = 1 + A × [tanh((lambda/lambda_0)^alpha) + B(r/r_0)^beta]
    
    Combines local (lambda) and global (r) scales
    """
    A, lambda_0, alpha, B, r_0, beta = params
    return 1.0 + A * (xp.tanh((lam / lambda_0)**alpha) + B * (r / r_0)**beta)

def multiplier_resonant_enhanced(lam, r, params, xp=np):
    """
    NEW: Resonance with distance decay
    
    f(lambda, r) = 1 + A × exp(-(r - lambda)^2/sigma^2) × (lambda/lambda_0)^alpha
    
    Strong when r ~ lambda (resonance), modulated by lambda scale
    """
    A, sigma, lambda_0, alpha = params
    return 1.0 + A * xp.exp(-(r - lam)**2 / (sigma**2)) * (lam / lambda_0)**alpha

# Original simple functions for comparison
def multiplier_power_law(lam, r, params, xp=np):
    """Original power law: f = 1 + A(lambda/lambda_0)^alpha"""
    A, lambda_0, alpha = params
    return 1.0 + A * (lam / lambda_0)**alpha

def multiplier_linear(lam, r, params, xp=np):
    """Original linear: f = 1 + A(lambda/r)"""
    A = params[0]
    return 1.0 + A * (lam / (r + 0.1))

# ============================================================================
# HYBRID PERIOD COMBINATIONS
# ============================================================================

def create_hybrid_periods(gaia, combination='jeans+orbital'):
    """
    Combine multiple period hypotheses.
    
    Options:
    - 'jeans+orbital': sqrt(lambda_jeans^2 + lambda_orbital^2)
    - 'jeans*mass': lambda_jeans × lambda_mass^0.5
    - 'weighted_avg': weighted average of multiple periods
    """
    
    if combination == 'jeans+orbital':
        lam = np.sqrt(gaia['lambda_jeans']**2 + gaia['lambda_orbital']**2)
        name = 'hybrid_jeans_orbital'
    
    elif combination == 'jeans*mass':
        lam = gaia['lambda_jeans'] * gaia['lambda_mass']**0.5
        name = 'hybrid_jeans_mass'
    
    elif combination == 'geometric_mean':
        lam = (gaia['lambda_jeans'] * gaia['lambda_orbital'] * 
               gaia['lambda_dynamical'])**(1/3)
        name = 'hybrid_geometric'
    
    elif combination == 'weighted_avg':
        # Weight by inverse scatter from first run
        w_jeans = 1.0
        w_orbital = 0.98
        w_dynamical = 0.97
        w_total = w_jeans + w_orbital + w_dynamical
        
        lam = (w_jeans * gaia['lambda_jeans'] + 
               w_orbital * gaia['lambda_orbital'] +
               w_dynamical * gaia['lambda_dynamical']) / w_total
        name = 'hybrid_weighted'
    
    else:
        raise ValueError(f"Unknown combination: {combination}")
    
    gaia[name] = lam
    print(f"  Created {name}: median={np.median(lam):.2f} kpc")
    
    return name

# ============================================================================
# IMPROVED GPU CALCULATOR
# ============================================================================

class ImprovedGravityCalculator:
    """
    Enhanced calculator with multiple improvements.
    """
    
    def __init__(self, stars_data, use_gpu=True, use_bulge=True,
                 use_gas=True, use_selection_weights=True):
        """
        Parameters:
        -----------
        use_bulge : bool
            Add analytical bulge component
        use_gas : bool
            Add exponential gas disk
        use_selection_weights : bool
            Apply Gaia selection bias correction
        """
        
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np
        
        print("\n" + "="*80)
        print("INITIALIZING IMPROVED GRAVITY CALCULATOR")
        print("="*80)
        
        self.N_stars = len(stars_data)
        print(f"\n  Stars: {self.N_stars:,}")
        
        # Convert to arrays
        self.x = stars_data['x'].values.astype(np.float32)
        self.y = stars_data['y'].values.astype(np.float32)
        self.z = stars_data['z'].values.astype(np.float32)
        self.M = stars_data['M_star'].values.astype(np.float32)
        self.R = np.sqrt(self.x**2 + self.y**2)
        
        # Load all periods
        self.periods = {}
        for col in stars_data.columns:
            if col.startswith('lambda_'):
                period_name = col.replace('lambda_', '')
                self.periods[period_name] = stars_data[col].values.astype(np.float32)
            elif col.startswith('hybrid_'):
                # Keep full hybrid name
                self.periods[col] = stars_data[col].values.astype(np.float32)
        
        print(f"  Periods: {list(self.periods.keys())[:5]}... ({len(self.periods)} total)")
        
        # Selection weights
        self.use_selection_weights = use_selection_weights
        if use_selection_weights:
            print("\n  Calculating selection bias weights...")
            weights = estimate_selection_weights(self.R, self.z, self.M)
            self.weights = weights.astype(np.float32)
            print(f"    Weight range: {weights.min():.2f} - {weights.max():.2f}")
            print(f"    Mean: {weights.mean():.2f} (should be ~1.0)")
        else:
            self.weights = np.ones(self.N_stars, dtype=np.float32)
        
        # Analytical components
        self.use_bulge = use_bulge
        self.use_gas = use_gas
        
        print(f"\n  Components:")
        print(f"    Stars (disk): ON")
        print(f"    Bulge: {'ON' if use_bulge else 'OFF'}")
        print(f"    Gas: {'ON' if use_gas else 'OFF'}")
        
        print(f"\n[OK] Calculator initialized")
    
    def compute_total_velocity(self, R_obs, period_name, multiplier_func, params,
                               M_scale=5e10, n_sample=50000):
        """
        Compute total circular velocity including all components.
        
        Returns:
        --------
        v_total : array
            Total circular velocity (km/s)
        v_components : dict
            Breakdown by component
        """
        
        # Component 1: Stars with multiplier (sampled for speed)
        v_stars = self._compute_stellar_component(
            R_obs, period_name, multiplier_func, params, M_scale, n_sample
        )
        
        # Component 2: Bulge (analytical)
        if self.use_bulge:
            v_bulge = hernquist_potential(R_obs, M_bulge=1.5e10, a_bulge=0.7)
        else:
            v_bulge = np.zeros_like(R_obs)
        
        # Component 3: Gas (analytical)
        if self.use_gas:
            v_gas = exponential_gas_disk(R_obs, M_gas=1e10, R_gas=7.0)
        else:
            v_gas = np.zeros_like(R_obs)
        
        # Total velocity (quadrature sum)
        v_total = np.sqrt(v_stars**2 + v_bulge**2 + v_gas**2)
        
        return v_total, {
            'stars': v_stars,
            'bulge': v_bulge,
            'gas': v_gas
        }
    
    def _compute_stellar_component(self, R_obs, period_name, multiplier_func, params,
                                   M_scale, n_sample):
        """
        Compute stellar contribution with multiplier (core calculation).
        Uses stratified sampling for speed.
        """
        
        N_obs = len(R_obs)
        v_stars = np.zeros(N_obs, dtype=np.float32)
        
        # Sample stars (stratified by radius) - use larger sample
        n_sample_use = min(100000, self.N_stars)  # Use more stars for accuracy
        
        if self.N_stars > n_sample_use:
            R_bins = np.percentile(self.R, np.linspace(0, 100, 21))
            sample_indices = []
            
            for i in range(len(R_bins) - 1):
                in_bin = np.where((self.R >= R_bins[i]) & (self.R < R_bins[i+1]))[0]
                if len(in_bin) > 0:
                    n_bin_sample = min(len(in_bin), n_sample_use // 20)
                    sampled = np.random.choice(in_bin, size=n_bin_sample, replace=False)
                    sample_indices.extend(sampled)
            
            sample_indices = np.array(sample_indices)
        else:
            sample_indices = np.arange(self.N_stars)
        
        # Get sampled data
        x_stars = self.x[sample_indices]
        y_stars = self.y[sample_indices]
        z_stars = self.z[sample_indices]
        M_stars = self.M[sample_indices]
        weights_stars = self.weights[sample_indices]
        lambda_stars = self.periods[period_name][sample_indices]
        
        # Scale masses (with selection weights)
        M_total_weighted = np.sum(M_stars * weights_stars)
        M_scale_factor = M_scale / M_total_weighted
        M_scaled = M_stars * M_scale_factor * weights_stars
        
        # Calculate velocity at each radius
        for i, R in enumerate(R_obs):
            # Observation point
            x_obs, y_obs, z_obs = R, 0, 0
            
            # Distances
            dx = x_stars - x_obs
            dy = y_stars - y_obs
            dz = z_stars - z_obs
            r = np.sqrt(dx**2 + dy**2 + dz**2 + 0.01**2)
            
            # Base gravity
            g_base = G_KPC * M_scaled / r**2
            
            # Apply multiplier
            multiplier = multiplier_func(lambda_stars, r, params, np)
            g_enhanced = g_base * multiplier
            
            # Radial component
            cos_theta = dx / r
            g_radial = np.sum(g_enhanced * cos_theta)
            
            # Velocity
            v_stars[i] = np.sqrt(max(R * g_radial, 0))
        
        return v_stars

# ============================================================================
# OPTIMIZATION
# ============================================================================

def optimize_improved_model(calculator, obs_indices, v_observed, period_name,
                            multiplier_func, param_bounds):
    """
    Optimize with improved calculator.
    """
    
    print(f"\n  Testing: {period_name} + {multiplier_func.__name__}")
    t0 = time.time()
    
    R_obs = calculator.R[obs_indices]
    
    best_rms = np.inf
    n_calls = [0]
    
    def objective(params):
        n_calls[0] += 1
        v_model, _ = calculator.compute_total_velocity(
            R_obs, period_name, multiplier_func, params
        )
        chi_sq = np.sum((v_model - v_observed)**2)
        
        # Track progress
        nonlocal best_rms
        rms = np.sqrt(np.mean((v_model - v_observed)**2))
        if rms < best_rms:
            best_rms = rms
            print(f"      Call {n_calls[0]}: RMS={rms:.1f} km/s")
        
        return chi_sq
    
    result = differential_evolution(
        objective,
        bounds=param_bounds,
        maxiter=15,  # Fewer iterations (already have good starting point)
        popsize=5,
        seed=42,
        workers=1,
        polish=False
    )
    
    t1 = time.time()
    
    # Final model
    v_model, v_components = calculator.compute_total_velocity(
        R_obs, period_name, multiplier_func, result.x
    )
    
    rms = np.sqrt(np.mean((v_model - v_observed)**2))
    
    print(f"    [OK] Complete: RMS={rms:.1f} km/s (time: {t1-t0:.1f}s)")
    print(f"    Params: {result.x}")
    
    # Component contributions
    for comp_name, v_comp in v_components.items():
        contrib = np.mean(v_comp**2) / np.mean(v_model**2) * 100
        print(f"      {comp_name}: {contrib:.1f}% contribution")
    
    return {
        'period_name': period_name,
        'multiplier_func': multiplier_func.__name__,
        'params': result.x.tolist(),
        'rms': float(rms),
        'chi_squared': float(result.fun),
        'v_model': v_model.tolist(),
        'time': t1 - t0
    }

# ============================================================================
# MAIN
# ============================================================================

def run_improved_analysis():
    """
    Run improved analysis with all enhancements.
    """
    
    print("="*80)
    print("IMPROVED INVERSE MULTIPLIER CALCULATION")
    print("="*80)
    print("\nGoal: Reduce RMS from 74.5 km/s to <20 km/s")
    
    # Load data
    print("\nLoading data...")
    gaia = pd.read_parquet('gravitywavebaseline/gaia_with_periods.parquet')
    print(f"  Loaded {len(gaia):,} stars")
    
    # Create hybrid periods
    print("\nCreating hybrid period combinations...")
    hybrid1 = create_hybrid_periods(gaia, 'jeans+orbital')
    hybrid2 = create_hybrid_periods(gaia, 'jeans*mass')
    hybrid3 = create_hybrid_periods(gaia, 'weighted_avg')
    
    # Setup observations
    print("\nSetting up observations...")
    N_obs = min(1000, len(gaia))
    obs_indices = np.linspace(0, len(gaia)-1, N_obs, dtype=int)
    v_observed = np.ones(N_obs) * 220.0  # Flat target
    
    # Initialize improved calculator
    calculator = ImprovedGravityCalculator(
        gaia,
        use_gpu=GPU_AVAILABLE,
        use_bulge=True,
        use_gas=True,
        use_selection_weights=True
    )
    
    # Test suite
    print("\n" + "="*80)
    print("RUNNING IMPROVED TESTS")
    print("="*80)
    
    test_suite = [
        # Best from original
        ('jeans', multiplier_power_law, [(0, 3), (1, 20), (1, 4)]),
        
        # New: distance-dependent
        ('jeans', multiplier_distance_dependent, [(0, 3), (1, 20), (1, 4), (1, 20)]),
        
        # New: hybrid periods
        (hybrid1, multiplier_power_law, [(0, 3), (1, 20), (1, 4)]),
        (hybrid2, multiplier_power_law, [(0, 3), (1, 20), (1, 4)]),
        (hybrid3, multiplier_power_law, [(0, 3), (1, 20), (1, 4)]),
        
        # New: advanced multipliers
        ('jeans', multiplier_hybrid_saturating, [(0, 3), (1, 20), (1, 4), (0, 1), (1, 20), (0, 2)]),
        ('jeans', multiplier_resonant_enhanced, [(0, 3), (1, 20), (1, 20), (1, 4)]),
    ]
    
    results = []
    for period, func, bounds in test_suite:
        try:
            result = optimize_improved_model(
                calculator, obs_indices, v_observed, period, func, bounds
            )
            results.append(result)
        except Exception as e:
            print(f"    [!] Failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*80)
    print("IMPROVED RESULTS")
    print("="*80)
    
    results_sorted = sorted(results, key=lambda x: x['rms'])
    
    print(f"\n{'Rank':<5} {'Period':<20} {'Multiplier':<30} {'RMS (km/s)':<12}")
    print("-"*75)
    for i, res in enumerate(results_sorted, 1):
        print(f"{i:<5} {res['period_name']:<20} {res['multiplier_func']:<30} {res['rms']:<12.1f}")
    
    # Save
    output_file = 'gravitywavebaseline/improved_multiplier_results.json'
    with open(output_file, 'w') as f:
        results_save = [{k: v for k, v in r.items() if k != 'v_model'} 
                       for r in results_sorted]
        json.dump(results_save, f, indent=2)
    
    print(f"\n[OK] Results saved to: {output_file}")
    
    # Best result details
    best = results_sorted[0]
    print(f"\n{'='*80}")
    print("BEST IMPROVED RESULT")
    print("="*80)
    print(f"\nPeriod: {best['period_name']}")
    print(f"Multiplier: {best['multiplier_func']}")
    print(f"RMS: {best['rms']:.1f} km/s")
    print(f"Parameters: {best['params']}")
    
    improvement = (74.5 - best['rms']) / 74.5 * 100
    print(f"\nImprovement over original: {improvement:.1f}%")
    
    if best['rms'] < 20:
        print("\n[SUCCESS] Achieved RMS < 20 km/s!")
    elif best['rms'] < 50:
        print("\n[OK] Good progress! Getting close to target.")
    else:
        print("\n[!] More improvements needed.")
    
    return results_sorted

if __name__ == "__main__":
    results = run_improved_analysis()

