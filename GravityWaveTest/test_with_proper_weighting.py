"""
Star-by-star test with PROPER mass weighting to correct Gaia selection bias.

KEY FIX:
- Stars are weighted by inverse selection probability
- Over-represented regions (R~8 kpc) get LOWER mass
- Under-represented regions (R<3 kpc, R>15 kpc) get HIGHER mass
- Mass distribution now matches true exp(-R/R_d) disk
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✓ CuPy available - using GPU acceleration")
except ImportError:
    cp = np
    GPU_AVAILABLE = False

import sys
sys.path.insert(0, 'GravityWaveTest')
from test_star_by_star_mw import burr_xii_window, G_KPC
import time
import json
import os

def compute_selection_weights(R_stars, z_stars, M_total=5.0e10):
    """
    Compute mass weights that correct for Gaia selection bias.
    
    True MW disk: Σ(R) ∝ exp(-R/2.5 kpc)
    Gaia sample: Over-represents R~5-10 kpc
    
    Returns weights such that Σ w_i Σ(R_i) ∝ exp(-R/2.5)
    """
    
    # Parameters
    R_d = 2.5  # kpc, disk scale length
    Sigma_0 = 800  # M_☉/pc² at R=0
    
    # Expected surface density at each star's radius
    Sigma_expected = Sigma_0 * np.exp(-R_stars / R_d)  # M_☉/pc²
    
    # Compute actual Gaia density (histogram in R-z bins)
    R_bins = np.linspace(0, 25, 50)
    z_bins = np.linspace(-3, 3, 20)
    
    # 2D histogram
    H, R_edges, z_edges = np.histogram2d(R_stars, z_stars, bins=[R_bins, z_bins])
    
    # Bin areas (annulus × height)
    R_centers = 0.5 * (R_edges[:-1] + R_edges[1:])
    z_heights = z_edges[1:] - z_edges[:-1]
    
    # Volume element (cylindrical)
    dR = R_edges[1:] - R_edges[:-1]
    areas = 2 * np.pi * R_centers[:, np.newaxis] * dR[:, np.newaxis] * z_heights[np.newaxis, :]
    
    # Actual number density (stars per kpc³)
    density_actual = H / (areas + 1e-10)
    
    # Assign weights: w_i ∝ expected_density / actual_density
    # This makes weighted distribution match expected
    
    # Find which bin each star is in
    R_idx = np.digitize(R_stars, R_edges) - 1
    z_idx = np.digitize(z_stars, z_edges) - 1
    
    # Clip to valid range
    R_idx = np.clip(R_idx, 0, len(R_centers)-1)
    z_idx = np.clip(z_idx, 0, len(z_heights)-1)
    
    # Get actual density for each star
    density_actual_per_star = density_actual[R_idx, z_idx]
    
    # Expected density (from exponential disk)
    # ρ(R,z) = Σ(R) × ρ_z(z) where ρ_z ∝ sech²(z/h)
    h = 0.3  # kpc, scale height
    rho_z = 1.0 / np.cosh(z_stars / h)**2
    rho_expected = Sigma_expected * rho_z
    
    # Weight inversely to actual density
    weights = rho_expected / (density_actual_per_star + 1e-10)
    
    # Normalize so total mass is M_total
    weights = weights / weights.sum() * M_total
    
    print(f"\nMass weighting statistics:")
    print(f"  Mean weight: {weights.mean():.2e} M_☉")
    print(f"  Std weight: {weights.std():.2e} M_☉")
    print(f"  Min weight: {weights.min():.2e} M_☉")
    print(f"  Max weight: {weights.max():.2e} M_☉")
    print(f"  Weight range: {weights.max()/weights.min():.1f}×")
    print(f"  Total mass: {weights.sum():.2e} M_☉")
    
    return weights

class WeightedStarCalculator:
    """
    Star-by-star calculator with proper mass weighting.
    """
    
    def __init__(self, stars_df, M_weights, use_gpu=True):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np
        
        # Convert to arrays
        self.R_stars = self.xp.array(stars_df['R_cyl'].values, dtype=self.xp.float32)
        self.z_stars = self.xp.array(stars_df['z'].values, dtype=self.xp.float32)
        self.phi_stars = self.xp.array(stars_df['phi'].values, dtype=self.xp.float32)
        self.M_weights = self.xp.array(M_weights, dtype=self.xp.float32)
        
        self.n_stars = len(self.R_stars)
        
        # Cartesian
        self.x_stars = self.R_stars * self.xp.cos(self.phi_stars)
        self.y_stars = self.R_stars * self.xp.sin(self.phi_stars)
        
        print(f"\nWeighted calculator initialized:")
        print(f"  Stars: {self.n_stars:,}")
        print(f"  Total mass: {self.xp.sum(self.M_weights):.2e} M_☉")
        print(f"  GPU: {self.use_gpu}")
    
    def compute_lambda_universal(self, ell0=4.993):
        """Universal λ."""
        return self.xp.full(self.n_stars, ell0, dtype=self.xp.float32)
    
    def compute_lambda_h_R(self):
        """Position-dependent λ = h(R)."""
        Sigma_0 = 800  # M_☉/pc²
        R_d = 2.5  # kpc
        sigma_z = 20.0  # km/s
        
        Sigma_local = Sigma_0 * 1e6 * self.xp.exp(-self.R_stars / R_d)
        h_disk = sigma_z**2 / (self.xp.pi * G_KPC * Sigma_local)
        
        return h_disk
    
    def compute_rotation_curve(self, R_obs, lambda_stars, A=0.591, p=0.757, n_coh=0.5):
        """Compute rotation curve with proper weighting."""
        
        xp = self.xp
        n_obs = len(R_obs)
        g_R = xp.zeros(n_obs, dtype=xp.float32)
        
        print(f"\nComputing Σ-Gravity with PROPER mass weighting...")
        print(f"  {self.n_stars:,} stars")
        print(f"  Total mass: {xp.sum(self.M_weights):.2e} M_☉")
        
        start = time.time()
        
        # Observation points
        x_obs = xp.asarray(R_obs, dtype=xp.float32)
        y_obs = xp.zeros_like(x_obs)
        z_obs = xp.zeros_like(x_obs)
        eps = xp.asarray(1e-6, dtype=xp.float32)
        
        # Batch processing
        batch_size = 10000 if self.use_gpu else 1000
        
        for i_batch in range(0, self.n_stars, batch_size):
            i_end = min(i_batch + batch_size, self.n_stars)
            
            # Batch
            x_i = self.x_stars[i_batch:i_end]
            y_i = self.y_stars[i_batch:i_end]
            z_i = self.z_stars[i_batch:i_end]
            M_i = self.M_weights[i_batch:i_end]
            lam_i = lambda_stars[i_batch:i_end]
            
            # Displacements
            dx = x_obs[:, xp.newaxis] - x_i[xp.newaxis, :]
            dy = y_obs[:, xp.newaxis] - y_i[xp.newaxis, :]
            dz = z_obs[:, xp.newaxis] - z_i[xp.newaxis, :]
            r = xp.sqrt(dx*dx + dy*dy + dz*dz) + eps
            
            # Newtonian + enhancement
            g_R_newt = G_KPC * M_i[xp.newaxis, :] * dx / (r**3)
            K = A * burr_xii_window(xp, r, lam_i[xp.newaxis, :], p=p, n_coh=n_coh)
            g_R_enh = g_R_newt * (1.0 + K)
            
            g_R += xp.sum(g_R_enh, axis=1)
            
            if i_batch % (batch_size * 50) == 0:
                elapsed = time.time() - start
                progress = i_end / self.n_stars
                eta = elapsed / progress - elapsed if progress > 0 else 0
                print(f"  Progress: {i_end}/{self.n_stars} ({100*progress:.1f}%) - ETA: {eta:.1f}s")
        
        elapsed = time.time() - start
        print(f"✓ Completed in {elapsed:.2f}s ({self.n_stars/elapsed:.0f} stars/sec)")
        
        # Circular velocity
        v_circ = xp.sqrt(xp.maximum(x_obs * g_R, 0.0))
        
        if self.use_gpu:
            v_circ = cp.asnumpy(v_circ)
        
        return v_circ

def run_properly_weighted_test():
    """
    Run star-by-star test with PROPER mass weighting.
    """
    
    print("="*80)
    print("STAR-BY-STAR TEST WITH PROPER MASS WEIGHTING")
    print("="*80)
    print("\n✓ Corrects for Gaia selection bias")
    print("✓ Mass distribution matches true MW disk")
    print("✓ Each star still has its own λ_i!")
    
    # Load data
    stars_csv = 'data/gaia/gaia_processed.csv'
    print(f"\nLoading {stars_csv}...")
    stars = pd.read_csv(stars_csv)
    print(f"Loaded {len(stars):,} stars")
    
    # Compute proper mass weights
    print("\nComputing selection-corrected mass weights...")
    M_weights = compute_selection_weights(
        stars['R_cyl'].values,
        stars['z'].values,
        M_total=5.0e10
    )
    
    # Initialize weighted calculator
    calc = WeightedStarCalculator(stars, M_weights, use_gpu=GPU_AVAILABLE)
    
    # Observation radii
    R_obs = np.linspace(0.5, 15.0, 30)
    v_obs_MW = 220 * np.ones_like(R_obs)
    
    # Test hypotheses
    hypotheses = {
        'universal': {
            'func': lambda: calc.compute_lambda_universal(4.993),
            'name': 'Universal λ = 4.993 kpc',
            'color': 'blue'
        },
        'h_R': {
            'func': lambda: calc.compute_lambda_h_R(),
            'name': 'λ = h(R) (position-dependent)',
            'color': 'green'
        }
    }
    
    print("\n" + "="*80)
    print("TESTING WITH PROPER WEIGHTS")
    print("="*80)
    
    results = {}
    
    for hyp_name, hyp_data in hypotheses.items():
        print(f"\nHypothesis: {hyp_data['name']}")
        
        # Compute λ for each star
        lambda_stars_gpu = hyp_data['func']()
        
        # Compute rotation curve
        v_circ = calc.compute_rotation_curve(R_obs, lambda_stars_gpu)
        
        # Metrics
        chi2 = np.sum((v_circ - v_obs_MW)**2) / len(R_obs)
        rms = np.sqrt(chi2)
        v_solar = np.interp(8.2, R_obs, v_circ)
        
        if calc.use_gpu:
            lambda_stars = cp.asnumpy(lambda_stars_gpu)
        else:
            lambda_stars = lambda_stars_gpu
        
        results[hyp_name] = {
            'R': R_obs,
            'v_circ': v_circ,
            'chi2': chi2,
            'rms': rms,
            'v_solar': v_solar,
            'lambda_median': np.median(lambda_stars),
            'lambda_std': np.std(lambda_stars)
        }
        
        print(f"  λ median: {results[hyp_name]['lambda_median']:.2f} ± {results[hyp_name]['lambda_std']:.2f} kpc")
        print(f"  v @ R=8.2: {v_solar:.1f} km/s (obs: 220 km/s)")
        print(f"  RMS: {rms:.1f} km/s")
        print(f"  χ²/dof: {chi2:.2f}")
    
    # Plots
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Properly Weighted Analysis ({len(stars):,} stars)', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Rotation curves
    ax = axes[0, 0]
    for hyp_name, hyp_data in hypotheses.items():
        ax.plot(results[hyp_name]['R'], results[hyp_name]['v_circ'],
                label=hyp_data['name'], color=hyp_data['color'], linewidth=2)
    ax.plot(R_obs, v_obs_MW, 'k--', linewidth=2, label='Observed (220 km/s)')
    ax.axvline(8.2, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('R [kpc]', fontsize=12)
    ax.set_ylabel('v_circ [km/s]', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Rotation Curves (Selection-Corrected)')
    
    # Plot 2: Mass weighting
    ax = axes[0, 1]
    subsample = np.random.choice(len(stars), min(10000, len(stars)), replace=False)
    scatter = ax.scatter(stars['R_cyl'].iloc[subsample], 
                        M_weights[subsample],
                        c=stars['z'].iloc[subsample], 
                        s=1, alpha=0.5, cmap='viridis')
    ax.set_xlabel('R [kpc]', fontsize=12)
    ax.set_ylabel('Mass weight [M_☉]', fontsize=12)
    ax.set_yscale('log')
    ax.set_title('Selection-Corrected Mass Weights')
    plt.colorbar(scatter, ax=ax, label='z [kpc]')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Residuals
    ax = axes[1, 0]
    for hyp_name, hyp_data in hypotheses.items():
        residuals = results[hyp_name]['v_circ'] - v_obs_MW
        ax.plot(results[hyp_name]['R'], residuals,
                label=hyp_data['name'], color=hyp_data['color'], linewidth=2)
    ax.axhline(0, color='k', linestyle='--')
    ax.set_xlabel('R [kpc]', fontsize=12)
    ax.set_ylabel('Δv [km/s]', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Residuals')
    
    # Plot 4: χ² comparison
    ax = axes[1, 1]
    names = list(hypotheses.keys())
    chi2_values = [results[h]['chi2'] for h in names]
    colors = [hypotheses[h]['color'] for h in names]
    
    ax.bar(range(len(names)), chi2_values, color=colors, alpha=0.7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([hypotheses[h]['name'] for h in names],
                       rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('χ² / dof', fontsize=12)
    ax.set_title('Goodness of Fit')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_dir = "GravityWaveTest/weighted_analysis"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/properly_weighted_results.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot to {output_dir}/properly_weighted_results.png")
    plt.close()
    
    # Save results
    results_export = {
        'method': 'SELECTION_CORRECTED_WEIGHTING',
        'n_stars': int(len(stars)),
        'M_total': float(M_weights.sum()),
        'note': 'Stars weighted by inverse selection probability to match true MW disk',
        'R_obs': R_obs.tolist(),
        'v_obs': v_obs_MW.tolist(),
        'hypotheses': {}
    }
    
    for hyp_name in hypotheses:
        results_export['hypotheses'][hyp_name] = {
            'name': hypotheses[hyp_name]['name'],
            'v_pred': results[hyp_name]['v_circ'].tolist(),
            'v_solar': float(results[hyp_name]['v_solar']),
            'lambda_median': float(results[hyp_name]['lambda_median']),
            'lambda_std': float(results[hyp_name]['lambda_std']),
            'chi2': float(results[hyp_name]['chi2']),
            'rms': float(results[hyp_name]['rms'])
        }
    
    with open(f"{output_dir}/weighted_results.json", 'w') as f:
        json.dump(results_export, f, indent=2)
    
    print(f"✓ Saved results to {output_dir}/weighted_results.json")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY (PROPERLY WEIGHTED)")
    print("="*80)
    
    best_hyp = min(hypotheses.keys(), key=lambda h: results[h]['chi2'])
    
    print(f"\nBest fit: {hypotheses[best_hyp]['name']}")
    print(f"  v @ R=8.2: {results[best_hyp]['v_solar']:.1f} km/s (obs: 220 km/s)")
    print(f"  Deviation: {100*(results[best_hyp]['v_solar']/220 - 1):.1f}%")
    print(f"  RMS: {results[best_hyp]['rms']:.1f} km/s")
    print(f"  χ²/dof: {results[best_hyp]['chi2']:.2f}")
    
    print("\nAll results:")
    for hyp_name in sorted(hypotheses.keys(), key=lambda h: results[h]['chi2']):
        print(f"  {hypotheses[hyp_name]['name']}: v={results[hyp_name]['v_solar']:.1f} km/s, χ²={results[hyp_name]['chi2']:.2f}")
    
    return results

if __name__ == "__main__":
    results = run_properly_weighted_test()

