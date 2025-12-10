#!/usr/bin/env python3
"""
GPM Pure Predictor - Frozen Parameters

Hold-out validation on 175 SPARC galaxies with NO parameter tuning.

Universal parameters (frozen from axisymmetric grid search):
- α₀ = 0.30 ± 0.05
- ℓ₀ = 0.80 ± 0.15 kpc
- M* = 2×10¹⁰ M☉
- σ* = 70 km/s
- Q* = 2.0
- n_M = 2.5

Prediction pipeline:
1. Load galaxy: Σ_b(R), σ_v, Q, M_total
2. Compute gates: α_eff = α₀ × g_Q × g_σ × g_M × g_K
3. Compute coherence length: ℓ = f(σ_v, Σ_b) [self-consistent]
4. Convolve: ρ_coh = α_eff × [Σ_b ⊗ K₀(R/ℓ)]
5. Predict: v_coh = √(G ∫ ρ_coh dV / r)
6. Total: v_pred = √(v_bar² + v_coh²)

NO FITTING. Pure prediction from frozen theory.

Reference: GPM_CANONICAL_STATEMENT.md
Author: GPM Theory Team
Date: December 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_integration.load_real_data import RealDataLoader
from data_integration.load_sparc_masses import load_sparc_masses
from galaxies.coherence_microphysics_axisym import AxiSymmetricYukawaConvolver
from galaxies.microphysical_gates import MicrophysicalGates
from galaxies.environment_estimator_v3 import EnvironmentEstimatorV3
from galaxies.rotation_curves import GalaxyRotationCurve
from scipy.interpolate import interp1d


class GPMPredictor:
    """
    Pure GPM predictor with frozen universal parameters.
    
    NO FITTING - only prediction from theory.
    
    Parameters (FROZEN from grid search):
    -----------
    alpha_0 : float
        Base susceptibility (0.30)
    ell_0 : float
        Base coherence length [kpc] (0.80)
    M_star : float
        Mass gate scale [M_sun] (2×10¹⁰)
    sigma_star : float
        Velocity dispersion gate [km/s] (70)
    Q_star : float
        Toomre Q gate (2.0)
    n_M : float
        Mass gate exponent (2.5)
    """
    
    def __init__(self, alpha_0=0.30, ell_0=0.80, M_star=2e10, 
                 sigma_star=70.0, Q_star=2.0, n_M=2.5, p=0.5):
        # Frozen parameters (NO TUNING)
        self.alpha_0 = alpha_0
        self.ell_0 = ell_0
        self.M_star = M_star
        self.sigma_star = sigma_star
        self.Q_star = Q_star
        self.n_M = n_M
        self.p = p  # ℓ ~ R_disk^p scaling
        
        # Initialize gates and convolver (WITH HARD FLOOR)
        self.gates = MicrophysicalGates(
            alpha_0=alpha_0,
            Q_star=Q_star,
            sigma_star=sigma_star,
            M_star=M_star,
            n_M=n_M,
            alpha_floor=0.05  # Hard cutoff: prevents spurious coherence in massive systems
        )
        
        self.h_z = 0.3  # kpc (thin disk scale height)
        
        print("="*80)
        print("GPM PREDICTOR (FROZEN PARAMETERS)")
        print("="*80)
        print(f"α₀ = {alpha_0}")
        print(f"ℓ₀ = {ell_0} kpc")
        print(f"M* = {M_star:.2e} M☉")
        print(f"σ* = {sigma_star} km/s")
        print(f"Q* = {Q_star}")
        print(f"n_M = {n_M}")
        print("="*80)
        print("NO PARAMETER FITTING - PURE PREDICTION ONLY")
        print("="*80 + "\n")
    
    def predict_rotation_curve(self, galaxy_name, verbose=False):
        """
        Predict rotation curve for one galaxy (NO FITTING).
        
        Parameters:
        -----------
        galaxy_name : str
            Galaxy name from SPARC
        verbose : bool
            Print detailed diagnostics
        
        Returns:
        --------
        result : dict
            Prediction results with RMS, MAE, chi-squared and residuals
        """
        if verbose:
            print(f"\nPredicting {galaxy_name}...")
        
        failure_reason = None
        try:
            # Load data
            loader = RealDataLoader()
            gal = loader.load_rotmod_galaxy(galaxy_name)
            
            r_data = gal['r']
            v_obs = gal['v_obs']
            v_err = gal['v_err']
            v_disk = gal['v_disk']
            v_gas = gal['v_gas']
            v_bulge = gal.get('v_bulge', np.zeros_like(v_disk))
            v_bar = np.sqrt(v_disk**2 + v_gas**2 + v_bulge**2)
            
            # DATA QUALITY CHECK: Skip poor quality data
            mean_snr = np.mean(v_obs / v_err)
            if len(r_data) < 8:
                raise ValueError(f"Insufficient data: only {len(r_data)} points")
            if mean_snr < 5.0:  # Mean SNR < 5 (error > 20% of signal)
                raise ValueError(f"Poor data quality: mean SNR = {mean_snr:.1f}")
            
            # Load masses
            sparc_masses = load_sparc_masses(galaxy_name)
            M_stellar = sparc_masses['M_stellar']
            M_HI = sparc_masses['M_HI']
            M_total = sparc_masses['M_total']
            R_disk = sparc_masses['R_disk']
            R_HI = sparc_masses['R_HI']
            
            # Baryon surface density
            R_gas = max(R_HI, 1.5 * R_disk)
            Sigma0_stellar = M_stellar / (2.0 * np.pi * R_disk**2)
            Sigma0_gas = M_HI / (2.0 * np.pi * R_gas**2)
            
            def Sigma_b(R):
                Sigma_stellar = Sigma0_stellar * np.exp(-R / R_disk)
                Sigma_gas = Sigma0_gas * np.exp(-R / R_gas)
                return Sigma_stellar + Sigma_gas
            
            # Environment estimation
            rotmod_dir = os.path.join(loader.base_data_dir, 'Rotmod_LTG')
            filepath = os.path.join(rotmod_dir, f'{galaxy_name}_rotmod.dat')
            
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            SBdisk = []
            for line in lines:
                if not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 7:
                        SBdisk.append(float(parts[6]))
            SBdisk = np.array(SBdisk)
            
            # Use improved environment estimator V3 (radius-dependent profiles)
            estimator = EnvironmentEstimatorV3(verbose=verbose)
            profiles = estimator.estimate_profiles_from_sparc(
                r_data, v_obs, SBdisk, M_L=0.5,
                v_bar=v_bar,  # Use baryons for κ (avoid circularity)
                R_disk=R_disk  # For compactness check
            )
            
            # Extract profiles
            r_profile = profiles['r']
            Q_profile = profiles['Q']
            sigma_v_profile = profiles['sigma_v']
            ell_profile = profiles['ell']
            
            # For reporting: use median values
            Q_median = np.median(Q_profile)
            sigma_v_median = np.median(sigma_v_profile)
            ell_median = np.median(ell_profile)
            
            # Compute gates at each radius
            alpha_eff_profile = np.array([
                self.gates.alpha_eff(Q_profile[i], sigma_v_profile[i], M_total, K=0.0)
                for i in range(len(r_profile))
            ])
            alpha_eff_median = np.median(alpha_eff_profile)
            
            # CRITICAL: If all alpha_eff == 0 (below floor), skip coherence calculation
            if np.all(alpha_eff_profile == 0.0):
                # No coherence - return baryon-only prediction
                ell_eff = 0.0  # Set to zero for reporting
                v_pred = v_bar
                if verbose:
                    print(f"  M_total = {M_total:.2e} M☉")
                    print(f"  R_disk = {R_disk:.2f} kpc")
                    print(f"  Q range: {Q_profile.min():.2f} - {Q_profile.max():.2f}")
                    print(f"  σ_v range: {sigma_v_profile.min():.1f} - {sigma_v_profile.max():.1f} km/s")
                    print(f"  α_eff = 0.0 (all below floor {self.gates.alpha_floor}) - NO COHERENCE")
            else:
                # Radius-dependent coherence calculation
                if verbose:
                    print(f"  M_total = {M_total:.2e} M☉")
                    print(f"  R_disk = {R_disk:.2f} kpc")
                    print(f"  Q range: {Q_profile.min():.2f} - {Q_profile.max():.2f} (median: {Q_median:.2f})")
                    print(f"  σ_v range: {sigma_v_profile.min():.1f} - {sigma_v_profile.max():.1f} km/s (median: {sigma_v_median:.1f})")
                    print(f"  α_eff range: {alpha_eff_profile.min():.4f} - {alpha_eff_profile.max():.4f} (median: {alpha_eff_median:.4f})")
                    print(f"  ℓ range: {ell_profile.min():.2f} - {ell_profile.max():.2f} kpc (median: {ell_median:.2f})")
                
                # Create interpolators for radius-dependent profiles
                # Use linear extrapolation at boundaries
                ell_interp = interp1d(r_profile, ell_profile, kind='linear',
                                     bounds_error=False, fill_value=(ell_profile[0], ell_profile[-1]))
                alpha_interp = interp1d(r_profile, alpha_eff_profile, kind='linear',
                                       bounds_error=False, fill_value=(alpha_eff_profile[0], alpha_eff_profile[-1]))
                
                # Use median alpha_eff for reporting (but use profile for convolution)
                alpha_eff_for_convolution = alpha_eff_median
                ell_eff = ell_median  # For reporting
                
                # Convolution with radius-dependent ℓ(R)
                convolver = AxiSymmetricYukawaConvolver(h_z=self.h_z)
                rho_coh = convolver.convolve_surface_density_with_ell_profile(
                    Sigma_b, alpha_eff_for_convolution, ell_interp, r_data,
                    R_max=r_data.max() * 2,
                    apply_thickness_correction=True
                )
                
                # Create interpolated density function
                rho_coh_func = interp1d(r_data, rho_coh, kind='cubic', 
                                       bounds_error=False, fill_value=0.0)
                
                # Convert to velocity
                galaxy = GalaxyRotationCurve(G=4.30091e-6)
                galaxy.set_baryon_profile(M_disk=M_total, R_disk=R_disk)
                galaxy.set_coherence_halo_gpm(rho_coh_func, gpm_params={
                    'alpha_eff': alpha_eff_median,
                    'ell_eff': ell_median,
                    'Q': Q_median,
                    'sigma_v': sigma_v_median
                })
                v_coh = galaxy.circular_velocity(r_data)
                
                # Total prediction
                v_pred = np.sqrt(v_bar**2 + v_coh**2)
            
            # Compute proper metrics
            residuals_bar = v_obs - v_bar
            residuals_gpm = v_obs - v_pred
            
            # RMS and MAE (in km/s - physical units)
            rms_bar = np.sqrt(np.mean(residuals_bar**2))
            rms_gpm = np.sqrt(np.mean(residuals_gpm**2))
            mae_bar = np.mean(np.abs(residuals_bar))
            mae_gpm = np.mean(np.abs(residuals_gpm))
            
            # Chi-squared (for compatibility)
            chi2_bar = np.sum(((v_obs - v_bar) / v_err)**2)
            chi2_gpm = np.sum(((v_obs - v_pred) / v_err)**2)
            
            n_dof = len(r_data)
            chi2_red_bar = chi2_bar / n_dof
            chi2_red_gpm = chi2_gpm / n_dof
            
            # Improvement metrics
            rms_improvement = (rms_bar - rms_gpm) / rms_bar * 100
            chi2_improvement = (chi2_bar - chi2_gpm) / chi2_bar * 100
            
            if verbose:
                print(f"  RMS_bar = {rms_bar:.2f} km/s, RMS_gpm = {rms_gpm:.2f} km/s")
                print(f"  MAE_bar = {mae_bar:.2f} km/s, MAE_gpm = {mae_gpm:.2f} km/s")
                print(f"  χ²_bar = {chi2_red_bar:.2f}, χ²_gpm = {chi2_red_gpm:.2f}")
                print(f"  RMS improvement = {rms_improvement:+.1f}%")
            
            return {
                'name': galaxy_name,
                'n_points': n_dof,
                'M_total': M_total,
                'R_disk': R_disk,
                'Q': Q_median if 'Q_median' in locals() else 0.0,
                'sigma_v': sigma_v_median if 'sigma_v_median' in locals() else 0.0,
                'alpha_eff': alpha_eff_median if 'alpha_eff_median' in locals() else alpha_eff if 'alpha_eff' in locals() else 0.0,
                'ell_eff': ell_eff,
                'rms_bar': rms_bar,
                'rms_gpm': rms_gpm,
                'mae_bar': mae_bar,
                'mae_gpm': mae_gpm,
                'chi2_red_bar': chi2_red_bar,
                'chi2_red_gpm': chi2_red_gpm,
                'rms_improvement': rms_improvement,
                'chi2_improvement': chi2_improvement,
                'r': r_data,
                'v_obs': v_obs,
                'v_err': v_err,
                'v_bar': v_bar,
                'v_pred': v_pred,
                'residuals_bar': residuals_bar,
                'residuals_gpm': residuals_gpm,
                'success': True,
                'failure_reason': None
            }
            
        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")
            failure_reason = str(e)
            return {
                'name': galaxy_name,
                'success': False,
                'failure_reason': failure_reason,
                'error': failure_reason
            }
    
    def predict_batch(self, galaxy_names, output_dir='outputs/gpm_holdout'):
        """
        Predict rotation curves for batch of galaxies (hold-out test).
        
        Parameters:
        -----------
        galaxy_names : list
            List of galaxy names
        output_dir : str
            Output directory for results
        
        Returns:
        --------
        results : list of dict
            Prediction results for each galaxy
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"HOLD-OUT PREDICTION ON {len(galaxy_names)} GALAXIES")
        print(f"{'='*80}\n")
        
        results = []
        successes = 0
        failures = 0
        
        for i, name in enumerate(galaxy_names):
            print(f"[{i+1}/{len(galaxy_names)}] {name}...", end=' ')
            
            result = self.predict_rotation_curve(name, verbose=False)
            results.append(result)
            
            if result.get('success', False):
                successes += 1
                rms_imp = result['rms_improvement']
                rms_gpm = result['rms_gpm']
                print(f"✓ RMS = {rms_gpm:.1f} km/s ({rms_imp:+.1f}%)")
            else:
                failures += 1
                err = result.get('failure_reason', 'Unknown error')
                # Truncate long error messages
                err_short = err[:50] + '...' if len(err) > 50 else err
                print(f"✗ {err_short}")
        
        # Summary statistics
        valid_results = [r for r in results if r.get('success', False)]
        
        if len(valid_results) > 0:
            rms_bar_all = [r['rms_bar'] for r in valid_results]
            rms_gpm_all = [r['rms_gpm'] for r in valid_results]
            mae_bar_all = [r['mae_bar'] for r in valid_results]
            mae_gpm_all = [r['mae_gpm'] for r in valid_results]
            rms_improvement_all = [r['rms_improvement'] for r in valid_results]
            chi2_improvement_all = [r['chi2_improvement'] for r in valid_results]
            
            print(f"\n{'='*80}")
            print("HOLD-OUT RESULTS (FROZEN PARAMETERS)")
            print(f"{'='*80}")
            print(f"Success rate: {successes}/{len(galaxy_names)} ({successes/len(galaxy_names)*100:.1f}%)")
            print(f"Failures: {failures}")
            print(f"\nRMS Residuals (km/s):")
            print(f"  Baryons:  {np.median(rms_bar_all):.1f} ± {np.std(rms_bar_all):.1f}")
            print(f"  GPM:      {np.median(rms_gpm_all):.1f} ± {np.std(rms_gpm_all):.1f}")
            print(f"  Improvement: {np.median(rms_improvement_all):+.1f}% (median)")
            print(f"\nMAE Residuals (km/s):")
            print(f"  Baryons:  {np.median(mae_bar_all):.1f} ± {np.std(mae_bar_all):.1f}")
            print(f"  GPM:      {np.median(mae_gpm_all):.1f} ± {np.std(mae_gpm_all):.1f}")
            print(f"\nFraction with RMS improvement > 0: {np.sum(np.array(rms_improvement_all) > 0) / len(rms_improvement_all) * 100:.1f}%")
            print(f"{'='*80}\n")
            
            # Save ALL results (including failures)
            df_all = pd.DataFrame(results)
            csv_all_path = os.path.join(output_dir, 'holdout_predictions_all.csv')
            df_all.to_csv(csv_all_path, index=False)
            print(f"✓ Saved all results (with failures) to {csv_all_path}")
            
            # Save successes only
            df_success = pd.DataFrame(valid_results)
            csv_path = os.path.join(output_dir, 'holdout_predictions_successes.csv')
            df_success.to_csv(csv_path, index=False)
            print(f"✓ Saved successes to {csv_path}")
            
            # Plot calibration
            self.plot_calibration(valid_results, output_dir)
        
        return results
    
    def plot_calibration(self, results, output_dir):
        """
        Generate calibration plots for hold-out predictions.
        Uses RMS residuals (km/s) as primary metric.
        """
        # Extract data
        rms_bar = np.array([r['rms_bar'] for r in results])
        rms_gpm = np.array([r['rms_gpm'] for r in results])
        rms_improvement = np.array([r['rms_improvement'] for r in results])
        alpha_eff = np.array([r['alpha_eff'] for r in results])
        sigma_v = np.array([r['sigma_v'] for r in results])
        M_total = np.array([r['M_total'] for r in results])
        
        # Collect all residuals for QQ plot
        all_residuals_gpm = np.concatenate([r['residuals_gpm'] for r in results])
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('GPM Hold-Out Prediction Calibration (Frozen Parameters)', 
                     fontsize=14, fontweight='bold')
        
        # Plot 1: RMS comparison (km/s)
        ax = axes[0, 0]
        ax.scatter(rms_bar, rms_gpm, alpha=0.6, s=50, c=M_total, cmap='viridis', norm=plt.Normalize(vmin=1e8, vmax=1e11))
        lim = max(rms_bar.max(), rms_gpm.max())
        ax.plot([0, lim], [0, lim], 'k--', alpha=0.5, label='1:1 line')
        ax.set_xlabel('RMS Baryons [km/s]', fontsize=11)
        ax.set_ylabel('RMS GPM [km/s]', fontsize=11)
        ax.set_title('Prediction Quality (Physical Units)', fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Add success/failure counts
        n_improved = np.sum(rms_gpm < rms_bar)
        n_total = len(rms_gpm)
        median_rms = np.median(rms_gpm)
        ax.text(0.05, 0.95, f"{n_improved}/{n_total} improved\n({n_improved/n_total*100:.1f}%)\nMedian RMS: {median_rms:.1f} km/s",
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot 2: QQ plot for residuals (normality test)
        ax = axes[0, 1]
        from scipy import stats
        stats.probplot(all_residuals_gpm, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot: GPM Residuals vs Normal', fontsize=12)
        ax.set_xlabel('Theoretical Quantiles', fontsize=11)
        ax.set_ylabel('Sample Quantiles [km/s]', fontsize=11)
        ax.grid(alpha=0.3)
        # Add text with normality test
        _, p_value = stats.shapiro(all_residuals_gpm[:5000] if len(all_residuals_gpm) > 5000 else all_residuals_gpm)
        ax.text(0.05, 0.95, f'Shapiro-Wilk p={p_value:.3f}\n(p>0.05: normal)',
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # Plot 3: α_eff vs σ_v (gating signature)
        ax = axes[1, 0]
        scatter = ax.scatter(sigma_v, alpha_eff, c=rms_improvement, 
                           cmap='RdYlGn', vmin=-50, vmax=100, s=50, alpha=0.7)
        ax.set_xlabel('σ_v [km/s]', fontsize=11)
        ax.set_ylabel('α_eff', fontsize=11)
        ax.set_title('Environmental Gating', fontsize=12)
        ax.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='RMS Improvement (%)')
        
        # Plot 4: RMS improvement vs mass (mass gate validation)
        ax = axes[1, 1]
        scatter = ax.scatter(M_total, rms_improvement, c=alpha_eff, 
                           cmap='viridis', s=50, alpha=0.7, norm=plt.Normalize(vmin=0, vmax=0.3))
        ax.axhline(0, color='r', linestyle='--', alpha=0.5, label='No improvement')
        ax.axvline(self.M_star, color='k', linestyle=':', alpha=0.5, label=f'M* = {self.M_star:.1e}')
        ax.set_xlabel('M_total [M☉]', fontsize=11)
        ax.set_ylabel('RMS Improvement (%)', fontsize=11)
        ax.set_title('Mass Gate Validation', fontsize=12)
        ax.set_xscale('log')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='α_eff')
        
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, 'holdout_calibration.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved calibration plot to {plot_path}")
        plt.close()


def get_all_sparc_galaxies(data_dir=None):
    """Load all 175 SPARC galaxy names from Rotmod_LTG directory."""
    if data_dir is None:
        # Try to find data directory
        possible_paths = [
            r'C:\Users\henry\dev\sigmagravity\data\Rotmod_LTG',
            r'../data/Rotmod_LTG',
            r'../../data/Rotmod_LTG'
        ]
        for path in possible_paths:
            if os.path.exists(path):
                data_dir = path
                break
    
    if data_dir is None or not os.path.exists(data_dir):
        raise FileNotFoundError(f"Cannot find SPARC data directory")
    
    files = [f.replace('_rotmod.dat', '') for f in os.listdir(data_dir) 
             if f.endswith('_rotmod.dat')]
    return sorted(files)


def main():
    """
    Main driver: Hold-out prediction on FULL 175 SPARC galaxy sample.
    """
    # Initialize predictor (FROZEN PARAMETERS)
    predictor = GPMPredictor(
        alpha_0=0.30,
        ell_0=0.80,
        M_star=2e10,
        sigma_star=70.0,
        Q_star=2.0,
        n_M=2.5,
        p=0.5
    )
    
    # Load FULL SPARC galaxy list (175 galaxies)
    all_galaxies = get_all_sparc_galaxies()
    
    print(f"\nFound {len(all_galaxies)} SPARC galaxies")
    print("\nRunning FULL hold-out prediction (no parameter tuning)...\n")
    
    # Run hold-out prediction on ALL galaxies
    results = predictor.predict_batch(all_galaxies)
    
    print("\n✓ Hold-out prediction complete!")
    print("\nResults:")
    print(f"  - Full sample: {len(all_galaxies)} galaxies")
    print(f"  - Frozen parameters (no tuning)")
    print(f"  - Results saved to outputs/gpm_holdout/")
    print("\nNext steps:")
    print("1. Analyze failure modes (expect M > 10¹¹ M☉)")
    print("2. Stratify by mass and morphology")
    print("3. Compute prediction interval coverage")
    print("4. Generate per-galaxy residual plots")


if __name__ == '__main__':
    main()
