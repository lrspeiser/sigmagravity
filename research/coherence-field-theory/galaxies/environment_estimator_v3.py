"""
Radius-Dependent Environment Estimator for GPM - Version 3

This version computes PROFILES Q(R), σ_v(R), ℓ(R) instead of single values.

Key improvements over V2:
1. Radius-dependent Q(R) and σ_v(R) capture radial variations
2. Self-consistent ℓ(R) computed iteratively at each radius
3. Eliminates catastrophic failures in compact dwarfs (e.g. NGC2976, UGC04305)
4. Uses PCHIP-smoothed derivative for stable κ(R) computation

Physical motivation:
- Compact dwarfs have high Σ_b(inner) → high σ_v needed for Q~1
- But low Σ_b(outer) → should have lower σ_v there
- Single global σ_v over-predicts coherence in outer disk

Reference: Phase 2 of HOLDOUT_ROADMAP.md
Author: GPM Theory Team
Date: December 2024
"""

import numpy as np
from scipy.interpolate import PchipInterpolator
from typing import Tuple, Dict

class EnvironmentEstimatorV3:
    """
    Radius-dependent environment estimator using SPARC observables.
    
    Returns Q(R), σ_v(R), ℓ(R) profiles instead of single values.
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
    
    def estimate_profiles_from_sparc(self,
                                     r: np.ndarray,
                                     v_obs: np.ndarray,
                                     SBdisk: np.ndarray,
                                     M_L: float = 0.5,
                                     v_bar: np.ndarray = None,
                                     R_disk: float = None) -> Dict[str, np.ndarray]:
        """
        Estimate Q(R), σ_v(R), ℓ(R) profiles from SPARC data.
        
        CRITICAL FEATURES:
        1. Compute κ(R) from PCHIP-smoothed dv_bar/dr (stable, non-oscillatory)
        2. Compute σ_v(R) = Q_target × π G Σ(R) / κ(R) at each radius
        3. Apply per-radius clamps [10, 50] km/s + compactness guards
        4. Iterate ℓ(R) = σ_v(R) / √(2πG ⟨Σ_b⟩_ℓ(R)) to convergence
        
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
        v_bar : array, optional
            Baryon rotation velocities [km/s] (use for κ to avoid circularity)
        R_disk : float, optional
            Disk scale length [kpc] (for compactness check)
        
        Returns:
        --------
        profiles : dict
            'r': radii (kpc)
            'Q': Toomre Q parameter at each radius
            'sigma_v': velocity dispersion (km/s) at each radius
            'ell': coherence length (kpc) at each radius
            'kappa': epicyclic frequency (km/s/kpc) at each radius
            'Sigma_b': surface density (M_sun/kpc²) at each radius
        """
        # Convert surface brightness to surface density
        Sigma_b = SBdisk * M_L * 1e6  # M_sun/kpc²
        
        # Use v_bar for κ if provided (avoid circularity)
        v_for_kappa = v_bar if v_bar is not None else v_obs
        
        # Mask valid points
        mask = (r > 0) & (v_obs > 0) & (Sigma_b > 0)
        if np.sum(mask) < 5:
            if self.verbose:
                print(f"  [ENV-V3] Warning: Only {np.sum(mask)} valid points, insufficient for profile")
            return self._fallback_profiles(r, v_obs, mask)
        
        r_valid = r[mask]
        v_obs_valid = v_obs[mask]
        v_kappa_valid = v_for_kappa[mask]
        Sigma_valid = Sigma_b[mask]
        
        # CRITICAL: Use PCHIP interpolator for smooth dv/dr (no oscillations)
        # PCHIP = Piecewise Cubic Hermite Interpolating Polynomial
        # Preserves monotonicity, prevents Runge oscillations from spline
        pchip = PchipInterpolator(r_valid, v_kappa_valid)
        
        # Compute κ(R) at each radius using smooth derivative
        # κ² = 2Ω(Ω + r dΩ/dr) = (2v/r)(v/r + r d(v/r)/dr)
        #    = (2v/r)(v/r + dv/dr - v/r) = (2v/r)(dv/dr)
        # For flat curve (dv/dr=0): κ = √2 v/r
        # For general case: κ² = 2Ω(Ω + r dΩ/dr)
        
        Omega = v_kappa_valid / r_valid
        dv_dr = pchip.derivative()(r_valid)
        dOmega_dr = (dv_dr - v_kappa_valid / r_valid) / r_valid
        
        kappa_sq = 2.0 * Omega * (Omega + r_valid * dOmega_dr)
        kappa_sq = np.maximum(kappa_sq, 0.0)  # Ensure non-negative
        kappa = np.sqrt(kappa_sq)
        
        # Filter out points with κ ~ 0 (center, turnover points)
        kappa_min = 0.1 * np.max(kappa)  # 10% of peak
        kappa_mask = kappa > kappa_min
        
        if np.sum(kappa_mask) < 5:
            if self.verbose:
                print(f"  [ENV-V3] Warning: Too few points with valid κ, using fallback")
            return self._fallback_profiles(r, v_obs, mask)
        
        # Apply κ mask
        r_use = r_valid[kappa_mask]
        v_obs_use = v_obs_valid[kappa_mask]
        Sigma_use = Sigma_valid[kappa_mask]
        kappa_use = kappa[kappa_mask]
        
        # Compute σ_v(R) at each radius using Q_target = 1.0
        Q_target = 1.0
        sigma_v_profile = Q_target * np.pi * self.G * Sigma_use / kappa_use
        
        # CRITICAL: Clamp to realistic range [10, 50] km/s at each radius
        sigma_v_profile = np.clip(sigma_v_profile, 10.0, 50.0)
        
        # COMPACTNESS GUARD: Extra limits for compact/dense dwarfs
        if R_disk is not None and R_disk < 1.0:  # Compact dwarf
            sigma_v_profile = np.minimum(sigma_v_profile, 35.0)
            if self.verbose:
                print(f"  [ENV-V3] Compactness guard: R_disk = {R_disk:.2f} kpc, limiting σ_v to 35 km/s")
        
        # Check median surface density (another compactness indicator)
        if np.median(Sigma_use) > 1e9:  # Very dense
            sigma_v_profile = np.minimum(sigma_v_profile, 35.0)
            if self.verbose:
                print(f"  [ENV-V3] High density guard: median Σ = {np.median(Sigma_use):.1e}, limiting σ_v to 35 km/s")
        
        # Compute self-consistent ℓ(R) = σ_v(R) / √(2πG ⟨Σ_b⟩_ℓ(R))
        # This requires iteration: ℓ depends on ⟨Σ_b⟩_ℓ which depends on ℓ
        ell_profile = self._compute_self_consistent_ell_profile(
            r_use, Sigma_use, sigma_v_profile
        )
        
        # Compute actual Q(R) with this σ_v(R)
        Q_profile = kappa_use * sigma_v_profile / (np.pi * self.G * Sigma_use)
        
        # Sanity bounds on Q
        Q_profile = np.clip(Q_profile, 0.5, 5.0)
        
        if self.verbose:
            print(f"  [ENV-V3] Profile computed:")
            print(f"    Q range: {Q_profile.min():.2f} - {Q_profile.max():.2f}")
            print(f"    σ_v range: {sigma_v_profile.min():.1f} - {sigma_v_profile.max():.1f} km/s")
            print(f"    ℓ range: {ell_profile.min():.2f} - {ell_profile.max():.2f} kpc")
            print(f"    κ range: {kappa_use.min():.1f} - {kappa_use.max():.1f} km/s/kpc")
        
        return {
            'r': r_use,
            'Q': Q_profile,
            'sigma_v': sigma_v_profile,
            'ell': ell_profile,
            'kappa': kappa_use,
            'Sigma_b': Sigma_use
        }
    
    def _compute_self_consistent_ell_profile(self,
                                             r: np.ndarray,
                                             Sigma_b: np.ndarray,
                                             sigma_v: np.ndarray,
                                             max_iter: int = 10,
                                             tol: float = 0.01) -> np.ndarray:
        """
        Compute self-consistent ℓ(R) profile via iteration.
        
        Equation: ℓ(R) = σ_v(R) / √(2πG ⟨Σ_b⟩_ℓ(R))
        
        where ⟨Σ_b⟩_ℓ(R) is the mean Σ_b within a disk of radius ℓ(R) centered at R.
        
        Algorithm:
        1. Initialize ℓ₀(R) = σ_v(R) / √(2πG Σ_b(R)) (local estimate)
        2. For each R, compute ⟨Σ_b⟩_ℓ by averaging Σ_b within ℓ of R
        3. Update ℓ(R) = σ_v(R) / √(2πG ⟨Σ_b⟩_ℓ(R))
        4. Repeat until convergence or max_iter
        
        Parameters:
        -----------
        r : array
            Radii [kpc]
        Sigma_b : array
            Surface density [M_sun/kpc²]
        sigma_v : array
            Velocity dispersion [km/s]
        max_iter : int
            Maximum iterations
        tol : float
            Relative tolerance for convergence
        
        Returns:
        --------
        ell_profile : array
            Self-consistent coherence length [kpc] at each radius
        """
        # Initialize with local estimate
        ell = sigma_v / np.sqrt(2.0 * np.pi * self.G * Sigma_b)
        ell = np.clip(ell, 0.1, 20.0)  # Reasonable bounds
        
        for iteration in range(max_iter):
            ell_old = ell.copy()
            
            # For each radius, compute ⟨Σ_b⟩_ℓ(R)
            for i, r_i in enumerate(r):
                # Find points within ℓ(R_i) of R_i
                dr = np.abs(r - r_i)
                mask_within_ell = dr <= ell[i]
                
                if np.sum(mask_within_ell) > 0:
                    Sigma_avg = np.mean(Sigma_b[mask_within_ell])
                else:
                    Sigma_avg = Sigma_b[i]  # Fallback to local value
                
                # Update ℓ(R_i)
                ell[i] = sigma_v[i] / np.sqrt(2.0 * np.pi * self.G * Sigma_avg)
            
            # Apply bounds
            ell = np.clip(ell, 0.1, 20.0)
            
            # Check convergence
            rel_change = np.max(np.abs(ell - ell_old) / (ell_old + 1e-10))
            if rel_change < tol:
                if self.verbose:
                    print(f"  [ENV-V3] ℓ(R) converged in {iteration+1} iterations (Δℓ/ℓ < {tol})")
                break
        
        return ell
    
    def _fallback_profiles(self, r: np.ndarray, v_obs: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Fallback profiles when data is insufficient.
        
        Use empirical scaling:
        - σ_v(R) ~ 0.15 v(R)
        - ℓ ~ 1.0 kpc (typical disk scale)
        - Q ~ 1.5 (marginally stable)
        """
        r_valid = r[mask]
        v_valid = v_obs[mask]
        
        n = len(r_valid)
        sigma_v = 0.15 * v_valid
        sigma_v = np.clip(sigma_v, 10.0, 50.0)
        
        ell = np.full(n, 1.0)  # 1 kpc typical
        Q = np.full(n, 1.5)
        kappa = np.sqrt(2.0) * v_valid / r_valid  # Flat rotation curve approximation
        
        # Estimate Sigma_b from v and R (rough)
        Sigma_b = v_valid**2 / (2.0 * np.pi * self.G * r_valid)
        
        if self.verbose:
            print(f"  [ENV-V3] Using fallback profiles (insufficient data)")
        
        return {
            'r': r_valid,
            'Q': Q,
            'sigma_v': sigma_v,
            'ell': ell,
            'kappa': kappa,
            'Sigma_b': Sigma_b
        }


def test_v3_vs_v2():
    """
    Compare V3 (profiles) vs V2 (single values) on problematic galaxies.
    """
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    
    from data_integration.load_real_data import RealDataLoader
    from data_integration.load_sparc_masses import load_sparc_masses
    from galaxies.environment_estimator_v2 import EnvironmentEstimatorV2
    
    # Load data
    loader = RealDataLoader()
    
    test_galaxies = ['NGC2976', 'UGC04305', 'NGC6503']
    
    for galaxy_name in test_galaxies:
        print(f"\n{'='*80}")
        print(f"Galaxy: {galaxy_name}")
        print('='*80)
        
        try:
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
            
            # V2 (single values)
            v2 = EnvironmentEstimatorV2(verbose=False)
            Q_v2, sigma_v2 = v2.estimate_from_sparc(
                gal['r'], gal['v_obs'], SBdisk, M_L=0.5,
                v_bar=gal['v_bar'], R_disk=sparc['R_disk']
            )
            
            # V3 (profiles)
            v3 = EnvironmentEstimatorV3(verbose=True)
            profiles = v3.estimate_profiles_from_sparc(
                gal['r'], gal['v_obs'], SBdisk, M_L=0.5,
                v_bar=gal['v_bar'], R_disk=sparc['R_disk']
            )
            
            print(f"\nV2 (single values):")
            print(f"  Q = {Q_v2:.2f}, σ_v = {sigma_v2:.1f} km/s")
            
            print(f"\nV3 (profiles):")
            print(f"  Q: {profiles['Q'].min():.2f} - {profiles['Q'].max():.2f}")
            print(f"  σ_v: {profiles['sigma_v'].min():.1f} - {profiles['sigma_v'].max():.1f} km/s")
            print(f"  ℓ: {profiles['ell'].min():.2f} - {profiles['ell'].max():.2f} kpc")
            
            # Compare inner vs outer disk
            r_prof = profiles['r']
            R_disk_val = sparc['R_disk']
            
            inner_mask = r_prof < 0.5 * R_disk_val
            outer_mask = r_prof > 2.0 * R_disk_val
            
            if np.sum(inner_mask) > 0 and np.sum(outer_mask) > 0:
                print(f"\nRadial variation:")
                print(f"  Inner (<0.5 R_disk): σ_v = {np.mean(profiles['sigma_v'][inner_mask]):.1f} km/s")
                print(f"  Outer (>2.0 R_disk): σ_v = {np.mean(profiles['sigma_v'][outer_mask]):.1f} km/s")
            
        except Exception as e:
            print(f"ERROR: {e}")


if __name__ == '__main__':
    test_v3_vs_v2()
