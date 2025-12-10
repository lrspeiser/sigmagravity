"""
Star-by-star rotation curve calculation using GPU acceleration.
CORRECTED to use proper Σ-Gravity physics:
- Force projection (not potential)
- Burr-XII coherence window
- Proper mass weighting (Monte Carlo samples)
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
from typing import Callable
import json
import os

# Physical constants
G_SI = 6.67430e-11  # m^3 kg^-1 s^-2
Msun_kg = 1.989e30
kpc_m = 3.086e19

# Gravitational constant in (km/s)² × kpc / M_☉
# G_kpc = G_SI × M_☉ / (kpc × 10^6 m/km)²
G_KPC = 4.30091e-6  # (km/s)² kpc M_☉^-1

def burr_xii_window(xp, r_kpc, lambda_kpc, p=0.757, n_coh=0.5):
    """
    Burr-XII coherence window from Σ-Gravity paper.
    
    C(r|λ,p,n) = 1 - [1 + (r/λ)^p]^(-n)
    
    Parameters from paper: p=0.757, n_coh=0.5
    """
    ratio = r_kpc / lambda_kpc
    return 1.0 - (1.0 + ratio**p)**(-n_coh)

class StarByStarCalculator:
    """
    Calculate rotation curve from individual stars with Σ-Gravity enhancement.
    
    CORRECTED PHYSICS:
    - Treats stars as Monte Carlo samples of continuous disk
    - Computes radial acceleration (force), not potential
    - Uses Burr-XII coherence window
    - Properly projects to circular velocity: v² = R × g_R
    """
    
    def __init__(self, stars_df: pd.DataFrame, use_gpu: bool = True):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np
        
        print(f"\nInitializing calculator (GPU: {self.use_gpu})")
        
        # Convert to arrays and move to GPU if available
        self.R_stars = self.xp.array(stars_df['R_cyl'].values, dtype=self.xp.float32)  # kpc
        self.z_stars = self.xp.array(stars_df['z'].values, dtype=self.xp.float32)      # kpc
        self.phi_stars = self.xp.array(stars_df['phi'].values, dtype=self.xp.float32)  # radians
        
        self.n_stars = len(self.R_stars)
        
        # Pre-compute Cartesian positions
        self.x_stars = self.R_stars * self.xp.cos(self.phi_stars)
        self.y_stars = self.R_stars * self.xp.sin(self.phi_stars)
        
        print(f"Loaded {self.n_stars} stars (Monte Carlo samples)")
        print(f"Memory usage: {self.n_stars * 5 * 4 / 1e6:.1f} MB")
        
    def compute_lambda_universal(self, ell0: float = 4.993):
        """Hypothesis 1: Universal λ for all stars."""
        return self.xp.full(self.n_stars, ell0, dtype=self.xp.float32)
    
    def compute_lambda_mass_dependent(self, M_weights, gamma: float = 0.5):
        """
        Hypothesis 2: λ_i ∝ M_i^γ.
        
        Note: M_weights are the MC weights, not "stellar masses"
        """
        lambda_0 = 5.0  # kpc
        M_norm = 5.0e10 / self.n_stars  # Typical weight
        
        lambda_i = lambda_0 * (M_weights / M_norm)**gamma
        return lambda_i
    
    def compute_lambda_local_disk(self, R_stars):
        """
        Hypothesis 3: λ from local disk scale height.
        
        h(R) = σ_z² / (π G Σ(R))
        
        With Σ(R) = Σ₀ exp(-R/R_d)
        """
        # MW disk surface density
        Sigma_0 = 800  # M_☉/pc² at R=0
        R_d = 2.5      # kpc
        
        # Convert to M_☉/kpc²
        Sigma_local = Sigma_0 * 1e6 * self.xp.exp(-R_stars / R_d)  # M_☉/kpc²
        
        # Velocity dispersion
        sigma_z = 20.0  # km/s
        
        # Scale height: h = σ² / (π G Σ)
        # Units: (km/s)² / [(km/s)² kpc M_☉^-1 × M_☉/kpc²] = kpc
        h_disk = sigma_z**2 / (self.xp.pi * G_KPC * Sigma_local)
        
        return h_disk
    
    def compute_lambda_hybrid(self, M_weights, R_stars):
        """
        Hypothesis 4: Hybrid scaling from SPARC.
        λ ~ M^0.3 × v^-1 × R^0.3
        """
        v_norm = 220.0
        R_norm = 2.5
        M_norm = 5.0e10 / self.n_stars
        
        v_local = 220.0  # Flat rotation curve
        
        lambda_i = 18.0 * (M_weights / M_norm)**0.3 * (v_local / v_norm)**(-1.0) * (R_stars / R_norm)**0.3
        
        return lambda_i
    
    def compute_enhanced_gravity(self, R_obs, lambda_stars,
                                 A: float = 0.591,
                                 p: float = 0.757,
                                 n_coh: float = 0.5,
                                 M_disk: float = 5.0e10):
        """
        Compute rotation curve with proper Σ-Gravity physics.
        
        CORRECTED:
        1. Stars are MC samples → assign uniform mass weights summing to M_disk
        2. Compute radial acceleration: g_R = Σ_i (G M_i Δx / r³) × [1 + K(r)]
        3. Circular velocity: v² = R × g_R
        4. Use Burr-XII window for K(r)
        
        Parameters:
        -----------
        R_obs : array
            Observation radii (kpc)
        lambda_stars : array
            Coherence length for each star (kpc)
        A : float
            Enhancement amplitude (0.591 from paper)
        p, n_coh : float
            Burr-XII parameters (0.757, 0.5 from paper)
        M_disk : float
            Total disk mass (M_☉)
        
        Returns:
        --------
        v_circ : array
            Circular velocity (km/s)
        """
        xp = self.xp
        n_obs = len(R_obs)
        
        # Radial acceleration at each observation point
        g_R = xp.zeros(n_obs, dtype=xp.float32)
        
        # Monte Carlo mass weights (uniform)
        M_per_star = M_disk / self.n_stars  # M_☉
        M_weights = xp.full(self.n_stars, M_per_star, dtype=xp.float32)
        
        print(f"\nComputing Σ-Gravity forces for {n_obs} radii from {self.n_stars} stars...")
        print(f"  Disk mass: {M_disk:.2e} M_☉")
        print(f"  Mass per star (MC weight): {M_per_star:.2e} M_☉")
        start_time = time.time()
        
        # Observation points on midplane (z=0, phi=0 → x-axis)
        x_obs = R_obs.astype(xp.float32)
        y_obs = xp.zeros_like(x_obs)
        z_obs = xp.zeros_like(x_obs)
        
        # Small epsilon to avoid division by zero
        eps = xp.asarray(1e-6, dtype=xp.float32)
        
        # Batch size for memory management
        batch_size = 10000 if self.use_gpu else 1000
        
        for i_batch in range(0, self.n_stars, batch_size):
            i_end = min(i_batch + batch_size, self.n_stars)
            
            # Get batch
            x_i = self.x_stars[i_batch:i_end]
            y_i = self.y_stars[i_batch:i_end]
            z_i = self.z_stars[i_batch:i_end]
            M_i = M_weights[i_batch:i_end]
            lam_i = lambda_stars[i_batch:i_end]
            
            # Displacement vectors: Δr = r_obs - r_star
            # Shape: (n_obs, n_batch)
            dx = x_obs[:, xp.newaxis] - x_i[xp.newaxis, :]
            dy = y_obs[:, xp.newaxis] - y_i[xp.newaxis, :]
            dz = z_obs[:, xp.newaxis] - z_i[xp.newaxis, :]
            
            # Distance
            r = xp.sqrt(dx*dx + dy*dy + dz*dz) + eps
            
            # Newtonian radial acceleration from each star
            # g_R,i = G M_i Δx / r³  (radial component at phi=0)
            # Units: (km/s)² kpc M_☉^-1 × M_☉ × kpc / kpc³ = (km/s)²/kpc
            g_R_newtonian = G_KPC * M_i[xp.newaxis, :] * dx / (r**3)
            
            # Σ-Gravity coherence kernel
            K = A * burr_xii_window(xp, r, lam_i[xp.newaxis, :], p=p, n_coh=n_coh)
            
            # Enhanced acceleration
            g_R_enhanced = g_R_newtonian * (1.0 + K)
            
            # Sum over stars
            g_R += xp.sum(g_R_enhanced, axis=1)
            
            if i_batch % (batch_size * 10) == 0:
                elapsed = time.time() - start_time
                progress = i_end / self.n_stars
                eta = elapsed / progress - elapsed if progress > 0 else 0
                print(f"  Progress: {i_end}/{self.n_stars} ({100*progress:.1f}%) - ETA: {eta:.1f}s")
        
        elapsed = time.time() - start_time
        print(f"✓ Completed in {elapsed:.2f}s ({self.n_stars/elapsed:.0f} stars/sec)")
        
        # Circular velocity: v² = R × g_R
        # Units: kpc × (km/s)²/kpc = (km/s)²
        v_squared = x_obs * g_R
        v_circ = xp.sqrt(xp.maximum(v_squared, 0.0))
        
        return v_circ, M_weights
    
    def test_hypothesis(self, R_obs: np.ndarray,
                       lambda_hypothesis: Callable,
                       A: float = 0.591,
                       M_disk: float = 5.0e10,
                       **hypothesis_kwargs):
        """Test a specific λ hypothesis."""
        
        # First get MC mass weights for this test
        M_per_star = M_disk / self.n_stars
        M_weights = self.xp.full(self.n_stars, M_per_star, dtype=self.xp.float32)
        
        # Compute λ for each star (may depend on M_weights for some hypotheses)
        if 'M_weights' in hypothesis_kwargs:
            hypothesis_kwargs['M_weights'] = M_weights
        lambda_stars = lambda_hypothesis(**hypothesis_kwargs)
        
        # Convert observation radii to GPU
        R_obs_gpu = self.xp.array(R_obs, dtype=self.xp.float32)
        
        # Compute rotation curve
        v_circ_gpu, M_weights_used = self.compute_enhanced_gravity(
            R_obs_gpu, lambda_stars, A=A, M_disk=M_disk
        )
        
        # Move back to CPU
        if self.use_gpu:
            v_circ = cp.asnumpy(v_circ_gpu)
            lambda_stars_cpu = cp.asnumpy(lambda_stars)
        else:
            v_circ = v_circ_gpu
            lambda_stars_cpu = lambda_stars
        
        return v_circ, lambda_stars_cpu

def run_mw_rotation_curve_test(stars_csv: str = "data/gaia/gaia_processed.csv"):
    """
    Test star-by-star calculation on Milky Way.
    CORRECTED for proper Σ-Gravity physics.
    """
    print("="*80)
    print("MILKY WAY STAR-BY-STAR TEST (CORRECTED Σ-GRAVITY)")
    print("="*80)
    
    # Load data
    print("\nLoading stellar data...")
    stars = pd.read_csv(stars_csv)
    print(f"Loaded {len(stars)} stars (Monte Carlo samples)")
    
    # Initialize calculator
    calc = StarByStarCalculator(stars, use_gpu=GPU_AVAILABLE)
    
    # Observation radii
    R_obs = np.linspace(0.5, 15.0, 30)  # kpc
    
    # MW parameters
    M_disk = 5.0e10  # M_☉ (total disk mass)
    A = 0.591        # Enhancement amplitude (from paper)
    
    # Test hypotheses
    hypotheses = {
        'universal': {
            'func': lambda: calc.compute_lambda_universal(ell0=4.993),
            'kwargs': {},
            'name': 'Universal λ = 4.993 kpc',
            'color': 'blue'
        },
        'mass_05': {
            'func': lambda M_weights: calc.compute_lambda_mass_dependent(M_weights, gamma=0.5),
            'kwargs': {'M_weights': None},  # Will be filled in test_hypothesis
            'name': 'λ ∝ M^0.5 (Tully-Fisher)',
            'color': 'red'
        },
        'mass_03': {
            'func': lambda M_weights: calc.compute_lambda_mass_dependent(M_weights, gamma=0.3),
            'kwargs': {'M_weights': None},
            'name': 'λ ∝ M^0.3 (SPARC)',
            'color': 'green'
        },
        'local_disk': {
            'func': lambda: calc.compute_lambda_local_disk(calc.R_stars),
            'kwargs': {},
            'name': 'λ = h(R) (disk scale height)',
            'color': 'purple'
        },
        'hybrid': {
            'func': lambda M_weights: calc.compute_lambda_hybrid(M_weights, calc.R_stars),
            'kwargs': {'M_weights': None},
            'name': 'λ ~ M^0.3 × R^0.3',
            'color': 'orange'
        }
    }
    
    # MW observed rotation curve (flat at 220 km/s)
    v_obs_MW = 220 * np.ones_like(R_obs)
    
    results = {}
    
    print("\n" + "="*80)
    print("TESTING HYPOTHESES")
    print("="*80)
    
    for hyp_name, hyp_data in hypotheses.items():
        print(f"\nTesting: {hyp_data['name']}")
        
        v_circ, lambda_stars = calc.test_hypothesis(
            R_obs,
            hyp_data['func'],
            A=A,
            M_disk=M_disk,
            **hyp_data['kwargs']
        )
        
        results[hyp_name] = {
            'R': R_obs,
            'v_circ': v_circ,
            'lambda_median': np.median(lambda_stars),
            'lambda_std': np.std(lambda_stars),
            'lambda_min': np.min(lambda_stars),
            'lambda_max': np.max(lambda_stars)
        }
        
        # Compute metrics
        chi2 = np.sum((v_circ - v_obs_MW)**2) / len(R_obs)
        rms = np.sqrt(chi2)
        
        print(f"  λ range: {results[hyp_name]['lambda_min']:.2f} - {results[hyp_name]['lambda_max']:.2f} kpc")
        print(f"  λ median: {results[hyp_name]['lambda_median']:.2f} ± {results[hyp_name]['lambda_std']:.2f} kpc")
        print(f"  v at R=8.2 kpc: {np.interp(8.2, R_obs, v_circ):.1f} km/s (obs: ~220 km/s)")
        print(f"  RMS deviation: {rms:.1f} km/s")
        print(f"  χ²/dof: {chi2:.2f}")
        
        results[hyp_name]['chi2'] = chi2
        results[hyp_name]['rms'] = rms
    
    # Generate plots
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Milky Way Star-by-Star Test (CORRECTED Σ-Gravity)', fontsize=14, fontweight='bold')
    
    # Plot 1: Rotation curves
    ax = axes[0, 0]
    for hyp_name, hyp_data in hypotheses.items():
        ax.plot(results[hyp_name]['R'], results[hyp_name]['v_circ'],
                label=hyp_data['name'], color=hyp_data['color'], linewidth=2)
    
    ax.plot(R_obs, v_obs_MW, 'k--', linewidth=2, label='Observed (220 km/s)')
    ax.axvline(8.2, color='gray', linestyle=':', alpha=0.5, label='Solar radius')
    ax.set_xlabel('R [kpc]', fontsize=12)
    ax.set_ylabel('v_circ [km/s]', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Rotation Curves')
    
    # Plot 2: Residuals
    ax = axes[0, 1]
    for hyp_name, hyp_data in hypotheses.items():
        residuals = results[hyp_name]['v_circ'] - v_obs_MW
        ax.plot(results[hyp_name]['R'], residuals,
                label=hyp_data['name'], color=hyp_data['color'], linewidth=2)
    
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.set_xlabel('R [kpc]', fontsize=12)
    ax.set_ylabel('Δv [km/s]', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Residuals')
    
    # Plot 3: χ² comparison
    ax = axes[1, 0]
    hyp_names = list(hypotheses.keys())
    chi2_values = [results[h]['chi2'] for h in hyp_names]
    colors = [hypotheses[h]['color'] for h in hyp_names]
    
    bars = ax.bar(range(len(hyp_names)), chi2_values, color=colors, alpha=0.7)
    ax.set_xticks(range(len(hyp_names)))
    ax.set_xticklabels([hypotheses[h]['name'] for h in hyp_names],
                       rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('χ² / dof', fontsize=12)
    ax.set_title('Goodness of Fit')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: λ distributions
    ax = axes[1, 1]
    for hyp_name, hyp_data in hypotheses.items():
        ax.errorbar([hyp_name], [results[hyp_name]['lambda_median']],
                   yerr=[[results[hyp_name]['lambda_median'] - results[hyp_name]['lambda_min']],
                         [results[hyp_name]['lambda_max'] - results[hyp_name]['lambda_median']]],
                   fmt='o', color=hyp_data['color'], capsize=5, markersize=8,
                   label=hyp_data['name'])
    
    ax.set_ylabel('λ [kpc]', fontsize=12)
    ax.set_xticklabels([])
    ax.set_title('Coherence Length Distributions')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    output_dir = "GravityWaveTest/mw_star_by_star"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/mw_rotation_comparison.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot to {output_dir}/mw_rotation_comparison.png")
    plt.close()
    
    # Save results
    results_export = {
        'n_stars': len(stars),
        'M_disk_Msun': M_disk,
        'A': A,
        'R_obs': R_obs.tolist(),
        'v_obs': v_obs_MW.tolist(),
        'hypotheses': {}
    }
    
    for hyp_name in hypotheses:
        results_export['hypotheses'][hyp_name] = {
            'name': hypotheses[hyp_name]['name'],
            'v_pred': results[hyp_name]['v_circ'].tolist(),
            'lambda_median': float(results[hyp_name]['lambda_median']),
            'lambda_std': float(results[hyp_name]['lambda_std']),
            'chi2': float(results[hyp_name]['chi2']),
            'rms_km_s': float(results[hyp_name]['rms'])
        }
    
    with open(f"{output_dir}/mw_test_results.json", 'w') as f:
        json.dump(results_export, f, indent=2)
    
    print(f"✓ Saved results to {output_dir}/mw_test_results.json")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    best_hyp = min(hypotheses.keys(), key=lambda h: results[h]['chi2'])
    
    print(f"\nBest fit: {hypotheses[best_hyp]['name']}")
    print(f"  χ²/dof = {results[best_hyp]['chi2']:.2f}")
    print(f"  RMS = {results[best_hyp]['rms']:.1f} km/s")
    
    print("\nRankings:")
    for i, hyp_name in enumerate(sorted(hypotheses.keys(), key=lambda h: results[h]['chi2'])):
        print(f"  {i+1}. {hypotheses[hyp_name]['name']}: χ² = {results[hyp_name]['chi2']:.2f}, RMS = {results[hyp_name]['rms']:.1f} km/s")
    
    return results

if __name__ == "__main__":
    if not os.path.exists("data/gaia/gaia_processed.csv"):
        print("ERROR: Gaia data not found!")
        print("Please run: python GravityWaveTest/generate_synthetic_mw.py first")
    else:
        results = run_mw_rotation_curve_test()

