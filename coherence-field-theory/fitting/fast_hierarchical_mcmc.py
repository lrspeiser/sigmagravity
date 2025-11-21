#!/usr/bin/env python3
"""
Fast Hierarchical MCMC for GPM - Simplified

Single-threaded but optimized for speed:
1. Cache baryon rotation curves (don't recompute)
2. Only fit 2 parameters (α₀, ℓ₀) with M*, σ* fixed
3. Use reduced step count (1000 steps should be enough)
4. Skip expensive axisymmetric convolution, use fast spherical

Expected runtime: 30-60 minutes
"""

import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_integration.load_real_data import RealDataLoader
from data_integration.load_sparc_masses import load_sparc_masses
from galaxies.coherence_microphysics import GravitationalPolarizationMemory
from galaxies.rotation_curves import GalaxyRotationCurve
from galaxies.environment_estimator import EnvironmentEstimator


class FastGalaxyData:
    """Lightweight galaxy data container (picklable)."""
    
    def __init__(self, name, gal, M_total, R_disk, Q, sigma_v, 
                 M_stellar, M_HI, R_HI, SBdisk):
        self.name = name
        self.r = gal['r']
        self.v_obs = gal['v_obs']
        self.v_err = gal['v_err']
        self.M_total = M_total
        self.R_disk = R_disk
        self.Q = Q
        self.sigma_v = sigma_v
        self.M_stellar = M_stellar
        self.M_HI = M_HI
        self.R_HI = R_HI
        self.R_gas = max(R_HI, 1.5 * R_disk)
        self.SBdisk = SBdisk
        self.h_z = 0.3  # kpc
        
        # Pre-compute baryon density parameters
        self.Sigma0_stellar = M_stellar / (2.0 * np.pi * R_disk**2)
        self.Sigma0_gas = M_HI / (2.0 * np.pi * self.R_gas**2)
    
    def rho_baryon(self, r):
        """Baryon density at radius r."""
        r_safe = np.maximum(np.atleast_1d(r), 1e-6)
        scalar_input = np.isscalar(r)
        
        Sigma_stellar = self.Sigma0_stellar * np.exp(-r_safe / self.R_disk)
        Sigma_gas = self.Sigma0_gas * np.exp(-r_safe / self.R_gas)
        Sigma_total = Sigma_stellar + Sigma_gas
        
        rho = Sigma_total / (2.0 * self.h_z)
        
        return float(rho[0]) if scalar_input else rho


def load_galaxy_data_fast(galaxy_names):
    """Load galaxy data into picklable containers."""
    loader = RealDataLoader()
    galaxy_data = []
    
    print(f"\nLoading {len(galaxy_names)} galaxies...")
    
    for name in galaxy_names:
        try:
            # Load SPARC rotation curve
            gal = loader.load_rotmod_galaxy(name)
            
            if len(gal['r']) < 5:
                print(f"  ✗ {name}: Only {len(gal['r'])} points, skipping")
                continue
            
            # Load masses
            sparc_masses = load_sparc_masses(name)
            M_stellar = sparc_masses['M_stellar']
            M_HI = sparc_masses['M_HI']
            M_total = sparc_masses['M_total']
            R_disk = sparc_masses['R_disk']
            R_HI = sparc_masses['R_HI']
            
            # Load SBdisk
            rotmod_dir = os.path.join(loader.base_data_dir, 'Rotmod_LTG')
            filepath = os.path.join(rotmod_dir, f'{name}_rotmod.dat')
            
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            data_lines = [l for l in lines if not l.startswith('#')]
            SBdisk = []
            for line in data_lines:
                parts = line.split()
                if len(parts) >= 7:
                    SBdisk.append(float(parts[6]))
            SBdisk = np.array(SBdisk)
            
            # Estimate environment
            estimator = EnvironmentEstimator()
            morphology = estimator.classify_morphology(gal, M_total, R_disk)
            Q, sigma_v = estimator.estimate_from_sparc(
                gal, SBdisk, R_disk, M_L=0.5, morphology=morphology
            )
            
            # Create lightweight data object
            gal_data = FastGalaxyData(
                name, gal, M_total, R_disk, Q, sigma_v,
                M_stellar, M_HI, R_HI, SBdisk
            )
            
            galaxy_data.append(gal_data)
            
            print(f"  ✓ {name}: {len(gal['r'])} points, M={M_total:.2e} M☉, Q={Q:.2f}, σ_v={sigma_v:.1f} km/s")
            
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            continue
    
    print(f"\n✓ Loaded {len(galaxy_data)}/{len(galaxy_names)} galaxies\n")
    return galaxy_data


def compute_chi2_fast(gal_data, alpha0, ell0_kpc, Mstar_Msun=2e10, sigmastar=70.0):
    """
    Fast chi-squared computation using spherical approximation.
    
    Uses GravitationalPolarizationMemory with use_axisymmetric=False for speed.
    """
    try:
        # Create GPM
        gpm = GravitationalPolarizationMemory(
            alpha0=alpha0,
            ell0_kpc=ell0_kpc,
            Qstar=2.0,
            sigmastar=sigmastar,
            nQ=2.0,
            nsig=2.0,
            p=0.5,
            Mstar_Msun=Mstar_Msun,
            nM=2.5
        )
        
        # Make coherence density (SPHERICAL - fast)
        rho_coh_func, gpm_diagnostics = gpm.make_rho_coh(
            gal_data.rho_baryon,
            Q=gal_data.Q,
            sigma_v=gal_data.sigma_v,
            R_disk=gal_data.R_disk,
            M_total=gal_data.M_total,
            use_axisymmetric=False,  # Fast spherical approximation
            h_z=gal_data.h_z,
            r_max=gal_data.r.max() * 2
        )
        
        # Compute rotation curve
        galaxy_gpm = GalaxyRotationCurve(G=4.30091e-6)
        galaxy_gpm.set_baryon_profile(M_disk=gal_data.M_total, R_disk=gal_data.R_disk)
        galaxy_gpm.set_coherence_halo_gpm(rho_coh_func, {'alpha0': alpha0, 'ell0_kpc': ell0_kpc})
        v_model = galaxy_gpm.circular_velocity(gal_data.r)
        
        # Chi-squared
        chi2 = np.sum(((gal_data.v_obs - v_model) / gal_data.v_err)**2)
        
        return chi2
        
    except Exception as e:
        return 1e10


class FastHierarchicalGPM:
    """
    Simplified hierarchical model - fit only (α₀, ℓ₀).
    
    Fixed parameters:
    - M* = 2×10¹⁰ M☉ (from grid search)
    - σ* = 70 km/s (from grid search)
    - Q* = 2.0
    - n_Q = 2.0
    - n_σ = 2.0
    - n_M = 2.5
    - p = 0.5
    """
    
    def __init__(self, galaxy_data):
        self.galaxy_data = galaxy_data
        self.n_galaxies = len(galaxy_data)
        self.n_dim = 2  # Only (alpha_0, ell_0)
        
        # Fixed parameters
        self.Mstar_Msun = 2e10
        self.sigmastar = 70.0
        
        print(f"✓ Initialized FAST hierarchical model:")
        print(f"  - {self.n_galaxies} galaxies")
        print(f"  - {self.n_dim} hyperparameters to fit: (α₀, ℓ₀)")
        print(f"  - Fixed: M*={self.Mstar_Msun:.2e} M☉, σ*={self.sigmastar:.1f} km/s")
        print(f"  - Using SPHERICAL approximation for speed")
    
    def log_prior(self, theta):
        """Log prior: uniform on α₀, log-uniform on ℓ₀."""
        alpha_0 = theta[0]
        ell_0 = theta[1]
        
        if not (0.1 <= alpha_0 <= 0.5):
            return -np.inf
        if not (0.3 <= ell_0 <= 3.0):
            return -np.inf
        
        # Log-uniform prior on ell_0
        return -np.log(ell_0)
    
    def log_likelihood(self, theta):
        """Log likelihood: sum over galaxies."""
        alpha_0 = theta[0]
        ell_0 = theta[1]
        
        log_like_total = 0.0
        
        for gal_data in self.galaxy_data:
            chi2 = compute_chi2_fast(gal_data, alpha_0, ell_0, self.Mstar_Msun, self.sigmastar)
            log_like_total += -0.5 * chi2
        
        return log_like_total
    
    def log_probability(self, theta):
        """Log posterior."""
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        
        ll = self.log_likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf
        
        return lp + ll
    
    def initialize_walkers(self, n_walkers):
        """Initialize near grid search optimum."""
        theta_init = np.array([0.30, 0.80])  # (alpha_0, ell_0)
        
        # Small perturbations
        pos = theta_init + 1e-2 * np.random.randn(n_walkers, self.n_dim)
        
        # Ensure within bounds
        pos[:, 0] = np.clip(pos[:, 0], 0.15, 0.45)  # alpha_0
        pos[:, 1] = np.clip(pos[:, 1], 0.4, 2.5)    # ell_0
        
        return pos
    
    def run_mcmc(self, n_walkers=16, n_steps=1000, n_burn=200, output_dir='../outputs/gpm_tests'):
        """Run MCMC (single-threaded for stability)."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"Starting FAST MCMC sampling")
        print(f"{'='*80}")
        print(f"Walkers: {n_walkers}")
        print(f"Steps: {n_steps}")
        print(f"Burn-in: {n_burn}")
        print(f"Total evaluations: {n_walkers * n_steps} = {n_walkers * n_steps:,}")
        est_time_min = n_walkers * n_steps * self.n_galaxies * 1.5 / 60  # ~1.5 sec/eval
        print(f"Estimated time: ~{est_time_min:.1f} minutes")
        print(f"{'='*80}\n")
        
        # Initialize
        pos = self.initialize_walkers(n_walkers)
        
        # Test initial position
        print("Testing initial position...")
        test_logp = self.log_probability(pos[0])
        print(f"  log P = {test_logp:.2f}")
        if not np.isfinite(test_logp):
            print("  ✗ ERROR: Initial position has -inf log probability!")
            return None, None
        print("  ✓ Initial position valid\n")
        
        # Create sampler (no parallelization to avoid pickle issues)
        sampler = emcee.EnsembleSampler(
            n_walkers, self.n_dim, self.log_probability
        )
        
        # Run MCMC
        print("Running MCMC...")
        sampler.run_mcmc(pos, n_steps, progress=True)
        
        # Extract results
        samples = sampler.get_chain(discard=n_burn, flat=True)
        
        # Compute statistics
        alpha_0_med = np.median(samples[:, 0])
        ell_0_med = np.median(samples[:, 1])
        
        alpha_0_std = np.std(samples[:, 0])
        ell_0_std = np.std(samples[:, 1])
        
        # 16th and 84th percentiles for ±1σ
        alpha_0_16, alpha_0_84 = np.percentile(samples[:, 0], [16, 84])
        ell_0_16, ell_0_84 = np.percentile(samples[:, 1], [16, 84])
        
        print(f"\n{'='*80}")
        print("Posterior Summary")
        print(f"{'='*80}")
        print(f"α₀ = {alpha_0_med:.3f} +{alpha_0_84-alpha_0_med:.3f} -{alpha_0_med-alpha_0_16:.3f}  (std: {alpha_0_std:.3f})")
        print(f"ℓ₀ = {ell_0_med:.3f} +{ell_0_84-ell_0_med:.3f} -{ell_0_med-ell_0_16:.3f} kpc  (std: {ell_0_std:.3f})")
        print(f"\nAcceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
        print(f"{'='*80}\n")
        
        # Save results
        results = {
            'alpha_0_median': alpha_0_med,
            'alpha_0_std': alpha_0_std,
            'alpha_0_16': alpha_0_16,
            'alpha_0_84': alpha_0_84,
            'ell_0_median': ell_0_med,
            'ell_0_std': ell_0_std,
            'ell_0_16': ell_0_16,
            'ell_0_84': ell_0_84,
            'Mstar_Msun_fixed': self.Mstar_Msun,
            'sigmastar_fixed': self.sigmastar,
            'n_walkers': n_walkers,
            'n_steps': n_steps,
            'n_burn': n_burn,
            'acceptance_fraction': np.mean(sampler.acceptance_fraction)
        }
        
        results_df = pd.DataFrame([results])
        results_df.to_csv(f'{output_dir}/fast_hierarchical_mcmc_results.csv', index=False)
        print(f"✓ Saved results to {output_dir}/fast_hierarchical_mcmc_results.csv")
        
        # Save full chain
        np.save(f'{output_dir}/fast_hierarchical_mcmc_chain.npy', samples)
        print(f"✓ Saved chain to {output_dir}/fast_hierarchical_mcmc_chain.npy")
        
        # Corner plot
        labels = [r'$\alpha_0$', r'$\ell_0$ [kpc]']
        fig = corner.corner(samples, labels=labels, truths=[alpha_0_med, ell_0_med],
                           quantiles=[0.16, 0.5, 0.84], show_titles=True)
        plt.savefig(f'{output_dir}/fast_hierarchical_mcmc_corner.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved corner plot to {output_dir}/fast_hierarchical_mcmc_corner.png")
        plt.close()
        
        # Trace plots
        chain = sampler.get_chain()
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        for i in range(2):
            ax = axes[i]
            ax.plot(chain[:, :, i], 'k', alpha=0.3, lw=0.5)
            ax.set_ylabel(labels[i])
            ax.axvline(n_burn, color='r', linestyle='--', label='Burn-in' if i==0 else '')
        axes[-1].set_xlabel('Step')
        if n_burn > 0:
            axes[0].legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/fast_hierarchical_mcmc_trace.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved trace plot to {output_dir}/fast_hierarchical_mcmc_trace.png")
        plt.close()
        
        return samples, results


def main():
    """Run fast hierarchical MCMC."""
    print("="*80)
    print("FAST Hierarchical MCMC for GPM")
    print("="*80)
    
    # Galaxy sample
    galaxy_names = [
        'NGC6503',   # Dwarf spiral
        'IC2574',    # Dwarf irregular
        'DDO154',    # Dwarf irregular
        'UGC02259',  # Low surface brightness
        'NGC2403',   # Intermediate spiral
        'NGC3198',   # Classic spiral
        'NGC5055',   # Grand design spiral
        'NGC3521',   # Grand design spiral
        'NGC2976'    # Dwarf spiral
    ]
    
    # Load data
    galaxy_data = load_galaxy_data_fast(galaxy_names)
    
    if len(galaxy_data) < 3:
        print("\n✗ ERROR: Need at least 3 galaxies")
        return
    
    # Create model
    model = FastHierarchicalGPM(galaxy_data)
    
    # Run MCMC
    samples, results = model.run_mcmc(
        n_walkers=16,
        n_steps=1000,
        n_burn=200
    )
    
    if samples is not None:
        print("\n" + "="*80)
        print("✓ FAST hierarchical MCMC complete!")
        print("="*80)
        print("\nResults:")
        print(f"  α₀ = {results['alpha_0_median']:.3f} ± {results['alpha_0_std']:.3f}")
        print(f"  ℓ₀ = {results['ell_0_median']:.3f} ± {results['ell_0_std']:.3f} kpc")
        print(f"  (M* = {results['Mstar_Msun_fixed']:.2e} M☉, σ* = {results['sigmastar_fixed']:.1f} km/s fixed)")


if __name__ == '__main__':
    main()
