"""
Hierarchical Bayesian Calibration for GPM

Uses emcee (affine-invariant MCMC) to calibrate GPM hyperparameters:
- Global: (α₀, ℓ₀, M*, σ*)
- Per-galaxy: nuisance parameters (distance modulus, M/L ratios)

Advantages over grid search:
1. Proper uncertainty quantification
2. Correlations between parameters
3. Posterior predictive checks
4. Model comparison (Bayes factors)

See: Foreman-Mackey+ 2013 (emcee paper)
"""

import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt
from multiprocessing import Pool
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_integration.load_real_data import RealDataLoader
from data_integration.load_sparc_masses import load_sparc_masses
from galaxies.coherence_microphysics_axisym import CoherenceMicrophysicsAxiSym
from galaxies.environment_estimator import EnvironmentEstimator


class HierarchicalGPM:
    """
    Hierarchical Bayesian model for GPM parameter inference.
    
    Hyperparameters (global):
    - alpha_0: base coupling strength
    - l_0: coherence length [kpc]
    - log_M_star: log10(mass gate scale [M_sun])
    - log_sigma_star: log10(velocity gate scale [km/s])
    
    Nuisance parameters (per-galaxy):
    - log_M_L_disk: log10(disk mass-to-light ratio)
    - log_M_L_bulge: log10(bulge mass-to-light ratio) (if applicable)
    """
    
    def __init__(self, galaxy_data, use_temporal_memory=True):
        """
        Initialize hierarchical model.
        
        Parameters:
        - galaxy_data: list of dicts with galaxy names and SPARC data
        - use_temporal_memory: whether to include temporal smoothing
        """
        self.galaxy_data = galaxy_data
        self.n_galaxies = len(galaxy_data)
        self.use_temporal_memory = use_temporal_memory
        
        # Fixed parameters (not fitted)
        self.n_M = 2.5
        self.n_sigma = 3.0
        self.Q_crit = 1.5
        self.n_Q = 3.0
        self.h_z = 0.3  # kpc
        self.t_age = 10.0  # Gyr
        
        # Dimensionality
        self.n_hyper = 4  # (alpha_0, l_0, log_M_star, log_sigma_star)
        self.n_nuisance_per_galaxy = 1  # log_M_L_disk (simplified)
        self.n_dim = self.n_hyper + self.n_galaxies * self.n_nuisance_per_galaxy
        
    def log_prior(self, theta):
        """
        Log prior probability.
        
        Priors:
        - alpha_0: uniform [0.1, 0.5]
        - l_0: log-uniform [0.3, 3.0] kpc
        - log_M_star: uniform [9.5, 11.0] (M_star = 3e9 to 1e11 M_sun)
        - log_sigma_star: uniform [1.5, 2.3] (sigma_star = 30 to 200 km/s)
        - log_M_L_disk: uniform [-0.3, 0.7] (M/L = 0.5 to 5.0)
        """
        alpha_0 = theta[0]
        l_0 = theta[1]
        log_M_star = theta[2]
        log_sigma_star = theta[3]
        log_M_L_disk = theta[4:]
        
        # Hyperparameter priors
        if not (0.1 <= alpha_0 <= 0.5):
            return -np.inf
        if not (0.3 <= l_0 <= 3.0):
            return -np.inf
        if not (9.5 <= log_M_star <= 11.0):
            return -np.inf
        if not (1.5 <= log_sigma_star <= 2.3):
            return -np.inf
        
        # Nuisance parameter priors
        if not np.all((-0.3 <= log_M_L_disk) & (log_M_L_disk <= 0.7)):
            return -np.inf
        
        # Log-uniform prior for l_0
        log_prior_l0 = -np.log(l_0)
        
        return log_prior_l0
    
    def log_likelihood(self, theta):
        """
        Log likelihood: sum over galaxies.
        
        L = Π_i P(D_i | theta_global, theta_i)
        log L = Σ_i log P(D_i | theta_global, theta_i)
        
        where D_i = {v_obs, dv, r} for galaxy i
        """
        alpha_0 = theta[0]
        l_0 = theta[1]
        log_M_star = theta[2]
        log_sigma_star = theta[3]
        log_M_L_disk = theta[4:]
        
        M_star = 10**log_M_star
        sigma_star = 10**log_sigma_star
        
        log_like_total = 0.0
        
        for i, gal_dict in enumerate(self.galaxy_data):
            M_L_disk = 10**log_M_L_disk[i]
            
            try:
                # Compute model prediction
                model = CoherenceMicrophysicsAxiSym(
                    alpha_0=alpha_0,
                    l_0=l_0,
                    M_star=M_star,
                    n_M=self.n_M,
                    sigma_star=sigma_star,
                    n_sigma=self.n_sigma,
                    Q_crit=self.Q_crit,
                    n_Q=self.n_Q,
                    h_z=self.h_z
                )
                
                v_model = model.compute_rotation_curve_axisym(
                    gal_dict['gal'],
                    gal_dict['SBdisk'],
                    gal_dict['M_total'],
                    gal_dict['R_disk'],
                    use_bulge=True,
                    use_temporal_memory=self.use_temporal_memory,
                    t_age=self.t_age,
                    M_L_override=M_L_disk
                )
                
                # Chi-squared likelihood
                v_obs = gal_dict['gal']['v_obs']
                dv = gal_dict['gal']['dv']
                
                chi2 = np.sum(((v_obs - v_model) / dv)**2)
                log_like_gal = -0.5 * chi2
                
                log_like_total += log_like_gal
                
            except Exception as e:
                # If model fails, return very low likelihood
                return -1e10
        
        return log_like_total
    
    def log_probability(self, theta):
        """
        Log posterior: log P(theta | D) ∝ log P(D | theta) + log P(theta)
        """
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        
        ll = self.log_likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf
        
        return lp + ll
    
    def initialize_walkers(self, n_walkers):
        """
        Initialize MCMC walkers near optimized values.
        """
        # Starting point (from grid search)
        theta_init = np.zeros(self.n_dim)
        theta_init[0] = 0.30  # alpha_0
        theta_init[1] = 0.80  # l_0
        theta_init[2] = 10.3  # log_M_star (2e10 M_sun)
        theta_init[3] = 1.85  # log_sigma_star (70 km/s)
        theta_init[4:] = 0.0  # log_M_L_disk = 1.0
        
        # Small perturbations around starting point
        pos = theta_init + 1e-3 * np.random.randn(n_walkers, self.n_dim)
        
        # Adjust l_0 perturbations (log-uniform)
        pos[:, 1] = np.abs(pos[:, 1])
        
        return pos
    
    def run_mcmc(self, n_walkers=32, n_steps=5000, n_burn=1000, output_dir='outputs/hierarchical'):
        """
        Run MCMC sampling.
        
        Parameters:
        - n_walkers: number of MCMC walkers (should be >= 2*n_dim)
        - n_steps: number of MCMC steps
        - n_burn: number of burn-in steps to discard
        - output_dir: directory to save results
        """
        print("="*80)
        print("HIERARCHICAL BAYESIAN CALIBRATION")
        print("="*80)
        print()
        print(f"Galaxies: {self.n_galaxies}")
        print(f"Dimensions: {self.n_dim} ({self.n_hyper} hyper + {self.n_galaxies} nuisance)")
        print(f"Walkers: {n_walkers}")
        print(f"Steps: {n_steps} (burn-in: {n_burn})")
        print()
        
        # Initialize walkers
        pos = self.initialize_walkers(n_walkers)
        
        # Setup sampler with parallelization
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(
                n_walkers, self.n_dim, self.log_probability,
                pool=pool
            )
            
            print("Running burn-in...")
            pos, _, _ = sampler.run_mcmc(pos, n_burn, progress=True)
            sampler.reset()
            
            print("\nRunning production...")
            sampler.run_mcmc(pos, n_steps, progress=True)
        
        print("\nMCMC complete!")
        print()
        
        # Extract samples
        samples = sampler.get_chain(discard=0, flat=True)
        
        # Acceptance fraction
        acc_frac = np.mean(sampler.acceptance_fraction)
        print(f"Mean acceptance fraction: {acc_frac:.3f}")
        print(f"Target: 0.2-0.5 (good), <0.1 or >0.7 (adjust step size)")
        print()
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, 'samples.npy'), samples)
        np.save(os.path.join(output_dir, 'chain.npy'), sampler.get_chain())
        
        print(f"Saved samples to {output_dir}/")
        print()
        
        return sampler, samples
    
    def analyze_results(self, samples, output_dir='outputs/hierarchical'):
        """
        Analyze MCMC results and produce diagnostic plots.
        """
        print("="*80)
        print("PARAMETER ESTIMATES")
        print("="*80)
        print()
        
        # Parameter names
        param_names = ['alpha_0', 'l_0', 'log_M_star', 'log_sigma_star']
        for i in range(self.n_galaxies):
            param_names.append(f'log_M_L_{i}')
        
        # Compute percentiles (16th, 50th, 84th)
        percentiles = np.percentile(samples, [16, 50, 84], axis=0)
        
        print("Hyperparameters:")
        for i in range(self.n_hyper):
            med = percentiles[1, i]
            err_lo = percentiles[1, i] - percentiles[0, i]
            err_hi = percentiles[2, i] - percentiles[1, i]
            print(f"  {param_names[i]:15s} = {med:.4f} +{err_hi:.4f} -{err_lo:.4f}")
        
        print()
        print("Nuisance parameters (M/L ratios):")
        for i in range(self.n_galaxies):
            idx = self.n_hyper + i
            med = percentiles[1, idx]
            err_lo = percentiles[1, idx] - percentiles[0, idx]
            err_hi = percentiles[2, idx] - percentiles[1, idx]
            M_L = 10**med
            gal_name = self.galaxy_data[i]['name']
            print(f"  {gal_name:15s} M/L = {M_L:.2f} (log={med:.3f} +{err_hi:.3f} -{err_lo:.3f})")
        
        print()
        
        # Corner plot for hyperparameters
        fig = corner.corner(
            samples[:, :self.n_hyper],
            labels=['α₀', 'ℓ₀ [kpc]', 'log(M*/M☉)', 'log(σ*/km/s)'],
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt='.3f',
            title_kwargs={'fontsize': 10}
        )
        fig.suptitle('Hyperparameter Posteriors', fontsize=14, fontweight='bold', y=1.01)
        
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'corner_hyperparameters.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir}/corner_hyperparameters.png")
        plt.close()
        
        # Trace plots
        chain = np.load(os.path.join(output_dir, 'chain.npy'))
        
        fig, axes = plt.subplots(self.n_hyper, 1, figsize=(10, 8), sharex=True)
        fig.suptitle('MCMC Trace Plots', fontsize=14, fontweight='bold')
        
        for i in range(self.n_hyper):
            ax = axes[i]
            ax.plot(chain[:, :, i], 'k', alpha=0.1)
            ax.set_ylabel(param_names[i], fontsize=11)
            ax.grid(alpha=0.3)
        
        axes[-1].set_xlabel('Step', fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'trace_plots.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir}/trace_plots.png")
        plt.close()
        
        print()
        print("="*80)


def prepare_galaxy_data(galaxy_names):
    """
    Load and prepare galaxy data for hierarchical fitting.
    """
    loader = RealDataLoader()
    estimator = EnvironmentEstimator()
    
    galaxy_data = []
    
    for name in galaxy_names:
        try:
            gal = loader.load_rotmod_galaxy(name)
            sparc_masses = load_sparc_masses(name)
            
            M_total = sparc_masses['M_total']
            R_disk = sparc_masses['R_disk']
            
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
            
            galaxy_data.append({
                'name': name,
                'gal': gal,
                'M_total': M_total,
                'R_disk': R_disk,
                'SBdisk': SBdisk
            })
            
        except Exception as e:
            print(f"Warning: Failed to load {name}: {e}")
    
    return galaxy_data


if __name__ == '__main__':
    print()
    print("="*80)
    print("HIERARCHICAL BAYESIAN CALIBRATION FOR GPM")
    print("="*80)
    print()
    
    # Select diverse galaxy sample (smaller for faster testing)
    galaxy_names = [
        'NGC6503',  # Dwarf spiral
        'NGC2403',  # Intermediate
        'NGC3198',  # Standard spiral
        'UGC128',   # Dwarf
        'DDO154',   # Low mass
    ]
    
    print(f"Loading {len(galaxy_names)} galaxies...")
    galaxy_data = prepare_galaxy_data(galaxy_names)
    print(f"Successfully loaded {len(galaxy_data)} galaxies")
    print()
    
    # Initialize hierarchical model
    model = HierarchicalGPM(galaxy_data, use_temporal_memory=True)
    
    # Run MCMC (small test run)
    sampler, samples = model.run_mcmc(
        n_walkers=32,
        n_steps=1000,   # Use 5000+ for production
        n_burn=200,     # Use 1000+ for production
        output_dir='outputs/hierarchical'
    )
    
    # Analyze results
    model.analyze_results(samples, output_dir='outputs/hierarchical')
    
    print()
    print("="*80)
    print("NEXT STEPS")
    print("="*80)
    print()
    print("1. Increase n_steps to 5000-10000 for production run")
    print("2. Check convergence with Gelman-Rubin statistic")
    print("3. Perform posterior predictive checks")
    print("4. Compare to NFW/Burkert using Bayes factors")
    print()
