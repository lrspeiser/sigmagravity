"""
Improved Environment Estimator for GPM - Version 2

Fixes critical issues in Q and σ_v estimation:
1. Uses actual SBdisk data from SPARC (not defaults)
2. Computes κ(R) from observed rotation curve
3. Estimates σ_v from disk stability requirements (not morphology)
4. Logs when using fallbacks

Reference: HOLDOUT_HONEST_ASSESSMENT.md
Author: GPM Theory Team
Date: December 2024
"""

import numpy as np
from typing import Tuple, Dict

class EnvironmentEstimatorV2:
    """
    Improved environment estimator using SPARC observables.
    
    Computes Toomre Q and velocity dispersion σ_v from:
    - Surface brightness SB(R) → Σ_b(R) via M/L
    - Rotation curve v(R) → epicyclic frequency κ(R)
    - Stability requirement: Q ~ 1-2 for observed disks
    """
    
    def __init__(self, G_kpc=4.302e-3, verbose=False):
        """
        Parameters:
        -----------
        G_kpc : float
            G in units (km/s)² kpc / M_sun
        verbose : bool
            Print diagnostic information
        """
        self.G = G_kpc
        self.verbose = verbose
    
    def estimate_from_sparc(self, 
                           r: np.ndarray,
                           v_obs: np.ndarray,
                           SBdisk: np.ndarray,
                           M_L: float = 0.5) -> Tuple[float, float]:
        """
        Estimate Q and σ_v from SPARC data.
        
        Method:
        1. Compute Σ_b(R) from SBdisk × M/L
        2. Compute κ(R) from rotation curve (κ² = 2Ω(Ω + dΩ/dr))
        3. Assume Q ~ 1.5 (marginally stable) → solve for σ_v
        
        Parameters:
        -----------
        r : array
            Radii [kpc]
        v_obs : array
            Observed rotation velocities [km/s]
        SBdisk : array
            Surface brightness [L_sun/pc²]
        M_L : float
            Mass-to-light ratio [M_sun/L_sun]
        
        Returns:
        --------
        Q : float
            Mean Toomre Q parameter
        sigma_v : float
            Mean velocity dispersion [km/s]
        """
        # Convert surface brightness to surface density
        # SB is in L_sun/pc², need M_sun/kpc²
        Sigma_b = SBdisk * M_L * 1e6  # M_sun/kpc²
        
        # Mask valid points
        mask = (r > 0) & (v_obs > 0) & (Sigma_b > 0)
        if np.sum(mask) < 3:
            if self.verbose:
                print(f"  [ENV] Warning: Only {np.sum(mask)} valid points, using fallback")
            return self._fallback_estimate(v_obs)
        
        r_valid = r[mask]
        v_valid = v_obs[mask]
        Sigma_valid = Sigma_b[mask]
        
        # Compute angular velocity Ω = v/r
        Omega = v_valid / r_valid
        
        # Compute dΩ/dr numerically
        dOmega_dr = np.gradient(Omega, r_valid)
        
        # Epicyclic frequency: κ² = 2Ω(Ω + r dΩ/dr)
        # For flat rotation curve (v = const), κ = √2 Ω
        # For general case, use full formula
        kappa_sq = 2.0 * Omega * (Omega + r_valid * dOmega_dr)
        kappa_sq = np.maximum(kappa_sq, 0.0)  # Ensure non-negative
        kappa = np.sqrt(kappa_sq)
        
        # CRITICAL: Skip points with kappa ~ 0 (center, turnover points)
        # These points are either innermost (solid body) or beyond peak (declining)
        kappa_mask = kappa > 0.1 * np.max(kappa)  # Keep points with kappa > 10% of max
        
        if np.sum(kappa_mask) < 3:
            if self.verbose:
                print(f"  [ENV] Warning: Too few points with valid κ, using fallback")
            return self._fallback_estimate(v_obs)
        
        r_use = r_valid[kappa_mask]
        Sigma_use = Sigma_valid[kappa_mask]
        kappa_use = kappa[kappa_mask]
        
        # Assume disks are marginally stable: Q ~ 1.5
        # Q = κ σ_R / (π G Σ) → σ_R = Q π G Σ / κ
        Q_target = 1.5
        sigma_v_at_r = Q_target * np.pi * self.G * Sigma_use / kappa_use
        
        # Weight by surface density (inner disk dominates)
        weights = Sigma_use / np.sum(Sigma_use)
        sigma_v_mean = np.sum(sigma_v_at_r * weights)
        
        # Sanity bounds
        sigma_v_mean = np.clip(sigma_v_mean, 5.0, 80.0)
        
        # Now compute actual Q with this σ_v
        Q_at_r = kappa_use * sigma_v_mean / (np.pi * self.G * Sigma_use)
        Q_mean = np.sum(Q_at_r * weights)
        
        # Sanity bounds
        Q_mean = np.clip(Q_mean, 0.5, 5.0)
        
        if self.verbose:
            print(f"  [ENV] Computed from SBdisk: Q = {Q_mean:.2f}, σ_v = {sigma_v_mean:.1f} km/s")
            print(f"  [ENV] κ range: {kappa.min():.1f} - {kappa.max():.1f} km/s/kpc")
            print(f"  [ENV] Σ_b range: {Sigma_valid.min():.1e} - {Sigma_valid.max():.1e} M_sun/kpc²")
        
        return Q_mean, sigma_v_mean
    
    def _fallback_estimate(self, v_obs: np.ndarray) -> Tuple[float, float]:
        """
        Fallback estimate when SBdisk data is insufficient.
        
        Use empirical relations:
        - σ_v ~ 0.15 v_mean (typical for disk galaxies)
        - Q ~ 1.5 (marginally stable)
        """
        v_mean = np.mean(v_obs[v_obs > 0])
        sigma_v = 0.15 * v_mean
        sigma_v = np.clip(sigma_v, 5.0, 50.0)
        Q = 1.5
        
        if self.verbose:
            print(f"  [ENV] Fallback: Q = {Q:.2f}, σ_v = {sigma_v:.1f} km/s (from v_mean = {v_mean:.1f})")
        
        return Q, sigma_v
    
    def estimate_from_line_width(self,
                                  W20: float,
                                  inclination: float = 45.0) -> float:
        """
        Estimate σ_v from HI line width (if available).
        
        W20 is the width at 20% of peak intensity.
        For a Gaussian line profile:
        W20 ≈ 2.3 × σ_v / sin(i)
        
        Parameters:
        -----------
        W20 : float
            HI line width at 20% [km/s]
        inclination : float
            Disk inclination [degrees]
        
        Returns:
        --------
        sigma_v : float
            Velocity dispersion [km/s]
        """
        sin_i = np.sin(np.radians(inclination))
        sigma_v = W20 * sin_i / 2.3
        return np.clip(sigma_v, 5.0, 50.0)
    
    def classify_morphology(self, 
                           M_total: float,
                           R_disk: float,
                           v_max: float) -> str:
        """
        Classify galaxy morphology from observables.
        
        Parameters:
        -----------
        M_total : float
            Total baryonic mass [M_sun]
        R_disk : float
            Disk scale length [kpc]
        v_max : float
            Maximum rotation velocity [km/s]
        
        Returns:
        --------
        morphology : str
            'dwarf', 'lsb', 'spiral', or 'massive'
        """
        if M_total < 1e9:
            return 'dwarf'
        elif M_total < 1e10:
            if v_max < 80:
                return 'lsb'  # Low surface brightness
            else:
                return 'spiral'
        elif M_total < 1e11:
            return 'spiral'
        else:
            return 'massive'


def compare_estimators(galaxy_name: str):
    """
    Compare old vs new environment estimator for debugging.
    """
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    
    from data_integration.load_real_data import RealDataLoader
    from data_integration.load_sparc_masses import load_sparc_masses
    from galaxies.environment_estimator import EnvironmentEstimator as OldEstimator
    
    # Load data
    loader = RealDataLoader()
    gal = loader.load_rotmod_galaxy(galaxy_name)
    sparc = load_sparc_masses(galaxy_name)
    
    # Parse SBdisk
    rotmod_dir = loader.base_data_dir + '/Rotmod_LTG'
    filepath = f"{rotmod_dir}/{galaxy_name}_rotmod.dat"
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    SBdisk = []
    for line in lines:
        if not line.startswith('#'):
            parts = line.split()
            if len(parts) >= 7:
                SBdisk.append(float(parts[6]))
    SBdisk = np.array(SBdisk)
    
    # Old estimator
    old_est = OldEstimator()
    morphology = old_est.classify_morphology(gal, sparc['M_total'], sparc['R_disk'])
    Q_old, sigma_old = old_est.estimate_from_sparc(gal, SBdisk, sparc['R_disk'], M_L=0.5, morphology=morphology)
    
    # New estimator
    new_est = EnvironmentEstimatorV2(verbose=True)
    Q_new, sigma_new = new_est.estimate_from_sparc(gal['r'], gal['v_obs'], SBdisk, M_L=0.5)
    
    print(f"\n{galaxy_name}:")
    print(f"  M_total = {sparc['M_total']:.2e} M_sun")
    print(f"  OLD: Q = {Q_old:.2f}, σ_v = {sigma_old:.1f} km/s")
    print(f"  NEW: Q = {Q_new:.2f}, σ_v = {sigma_new:.1f} km/s")
    print(f"  Δσ_v = {sigma_new - sigma_old:+.1f} km/s")


if __name__ == '__main__':
    # Test on problematic galaxies
    print("="*80)
    print("Comparing Old vs New Environment Estimators")
    print("="*80)
    
    test_galaxies = ['F561-1', 'UGC04305', 'NGC2976', 'NGC6503', 'NGC2841']
    
    for name in test_galaxies:
        try:
            compare_estimators(name)
        except Exception as e:
            print(f"\n{name}: ERROR - {e}")
