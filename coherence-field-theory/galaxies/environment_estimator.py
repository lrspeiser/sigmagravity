"""
Environment parameter estimation for GPM gating.

Computes Toomre Q and velocity dispersion σ_v from SPARC data
to properly gate the coherence susceptibility α.
"""

import numpy as np
from typing import Tuple, Dict, Optional


class EnvironmentEstimator:
    """
    Estimate dynamical environment (Q, σ_v) from galaxy observables.
    
    Uses SPARC rotation curves and surface brightness profiles to
    compute Toomre Q parameter and velocity dispersion for GPM gating.
    """
    
    def __init__(self):
        """Initialize with standard constants."""
        self.G_kpc = 4.302e-3  # G in kpc (km/s)^2 / M_sun
    
    def estimate_from_sparc(self,
                           gal: Dict,
                           SBdisk: np.ndarray,
                           R_disk: float,
                           M_L: float = 0.5,
                           morphology: str = 'dwarf') -> Tuple[float, float]:
        """
        Estimate Q and σ_v from SPARC observables.
        
        Parameters
        ----------
        gal : dict
            SPARC galaxy data with keys 'r', 'v_obs', 'v_disk', 'v_gas'
        SBdisk : array
            Surface brightness profile (L_sun/pc^2)
        R_disk : float
            Disk scale length (kpc)
        M_L : float
            Mass-to-light ratio (M_sun/L_sun)
        morphology : str
            Galaxy morphology: 'dwarf', 'lsb', 'spiral', 'massive'
            
        Returns
        -------
        Q : float
            Volume-averaged Toomre Q parameter
        sigma_v : float
            Velocity dispersion (km/s)
        """
        r = gal['r']
        v_obs = gal['v_obs']
        v_disk = gal['v_disk']
        v_gas = gal['v_gas']
        
        # Estimate velocity dispersion from morphology and observed velocities
        sigma_v = self._estimate_sigma_v(v_obs, morphology)
        
        # Estimate Toomre Q from surface density and epicyclic frequency
        Q = self._estimate_toomre_Q(r, SBdisk, v_obs, R_disk, M_L, sigma_v)
        
        return Q, sigma_v
    
    def _estimate_sigma_v(self, v_obs: np.ndarray, morphology: str) -> float:
        """
        Estimate velocity dispersion from observed velocities and morphology.
        
        Based on observational scaling relations:
        - Dwarfs: σ_v ~ 0.05-0.10 v_c (cold, low dispersion)
        - LSBs: σ_v ~ 0.10-0.15 v_c (intermediate)
        - Spirals: σ_v ~ 0.15-0.20 v_c (warmer disks)
        - Massive: σ_v ~ 0.20-0.30 v_c (hot, pressure-supported)
        
        Parameters
        ----------
        v_obs : array
            Observed rotation velocities (km/s)
        morphology : str
            Galaxy type
            
        Returns
        -------
        sigma_v : float
            Velocity dispersion (km/s)
        """
        v_mean = np.mean(v_obs[v_obs > 0])
        v_max = np.max(v_obs)
        
        if morphology == 'dwarf':
            # Cold dwarfs: very low dispersion
            # Typical σ_v/v_c ~ 0.05-0.10
            sigma_v = 0.06 * v_mean
            sigma_v = max(sigma_v, 2.0)  # floor at 2 km/s
            
        elif morphology == 'lsb':
            # Low surface brightness: intermediate
            # σ_v/v_c ~ 0.10-0.15
            sigma_v = 0.12 * v_mean
            sigma_v = max(sigma_v, 5.0)
            
        elif morphology == 'spiral':
            # Normal spirals: warmer disks
            # σ_v/v_c ~ 0.15-0.20
            sigma_v = 0.17 * v_mean
            sigma_v = max(sigma_v, 10.0)
            
        elif morphology == 'massive':
            # Massive galaxies: hot, dispersion-supported
            # σ_v/v_c ~ 0.20-0.30
            sigma_v = 0.25 * v_mean
            sigma_v = max(sigma_v, 20.0)
            
        else:
            # Default: use mean velocity as proxy
            sigma_v = 0.15 * v_mean
            sigma_v = max(sigma_v, 5.0)
        
        return sigma_v
    
    def _estimate_toomre_Q(self,
                          r: np.ndarray,
                          SBdisk: np.ndarray,
                          v_obs: np.ndarray,
                          R_disk: float,
                          M_L: float,
                          sigma_v: float) -> float:
        """
        Estimate Toomre Q parameter from surface density and kinematics.
        
        Q = κ σ_R / (3.36 G Σ)
        
        where:
        - κ = epicyclic frequency ≈ sqrt(2) Ω for flat rotation curve
        - σ_R = radial velocity dispersion ≈ σ_v
        - Σ = surface density from SBdisk × M/L
        
        Parameters
        ----------
        r : array
            Radii (kpc)
        SBdisk : array
            Surface brightness (L_sun/pc^2)
        v_obs : array
            Rotation velocities (km/s)
        R_disk : float
            Scale length (kpc)
        M_L : float
            Mass-to-light ratio
        sigma_v : float
            Velocity dispersion (km/s)
            
        Returns
        -------
        Q : float
            Volume-averaged Toomre Q
        """
        # Convert surface brightness to surface density
        Sigma = SBdisk * M_L * 1e6  # M_sun/kpc^2
        
        # Mask valid points
        mask = (Sigma > 0) & (v_obs > 0) & (r > 0)
        if np.sum(mask) < 3:
            # Insufficient data: return typical value based on morphology
            return 1.5  # marginally stable
        
        r_valid = r[mask]
        Sigma_valid = Sigma[mask]
        v_valid = v_obs[mask]
        
        # Epicyclic frequency: κ ≈ sqrt(2) Ω for flat rotation curve
        # Ω = v/r
        Omega = v_valid / r_valid
        kappa = np.sqrt(2.0) * Omega
        
        # Toomre Q at each radius
        Q_r = kappa * sigma_v / (3.36 * self.G_kpc * Sigma_valid)
        
        # Weight by surface density (denser regions matter more)
        weights = Sigma_valid / np.sum(Sigma_valid)
        Q_avg = np.sum(Q_r * weights)
        
        # Sanity check: Q < 1 implies gravitational instability (unphysical for observed galaxies)
        # Q ~ 1-2 typical for disk galaxies
        Q_avg = np.clip(Q_avg, 1.0, 5.0)
        
        return Q_avg
    
    def classify_morphology(self, 
                           gal: Dict,
                           M_total: float,
                           R_disk: float) -> str:
        """
        Classify galaxy morphology from observables.
        
        Parameters
        ----------
        gal : dict
            SPARC data
        M_total : float
            Total baryon mass (M_sun)
        R_disk : float
            Disk scale length (kpc)
            
        Returns
        -------
        morphology : str
            One of: 'dwarf', 'lsb', 'spiral', 'massive'
        """
        v_max = np.max(gal['v_obs'])
        
        # Mass-based primary classification
        if M_total < 1e8:
            return 'dwarf'
        elif M_total < 5e8:
            # Check if LSB (large R_disk for mass)
            if R_disk > 2.0:
                return 'lsb'
            else:
                return 'dwarf'
        elif M_total < 5e9:
            return 'spiral'
        else:
            return 'massive'
    
    def estimate_simple(self,
                       v_obs: np.ndarray,
                       morphology: Optional[str] = None) -> Tuple[float, float]:
        """
        Simple fallback estimation when full data unavailable.
        
        Uses observational scaling relations only.
        
        Parameters
        ----------
        v_obs : array
            Observed velocities (km/s)
        morphology : str, optional
            If None, infer from v_max
            
        Returns
        -------
        Q : float
            Toomre Q (typical value)
        sigma_v : float
            Velocity dispersion (km/s)
        """
        v_max = np.max(v_obs)
        
        # Infer morphology if not provided
        if morphology is None:
            if v_max < 50:
                morphology = 'dwarf'
            elif v_max < 100:
                morphology = 'lsb'
            elif v_max < 200:
                morphology = 'spiral'
            else:
                morphology = 'massive'
        
        # Estimate sigma_v
        sigma_v = self._estimate_sigma_v(v_obs, morphology)
        
        # Typical Q values by morphology
        Q_typical = {
            'dwarf': 1.5,    # marginally stable
            'lsb': 1.8,      # slightly stable
            'spiral': 2.0,   # stable
            'massive': 2.5   # very stable
        }
        Q = Q_typical.get(morphology, 2.0)
        
        return Q, sigma_v


# Convenience function
def estimate_environment(gal: Dict,
                        SBdisk: Optional[np.ndarray] = None,
                        R_disk: Optional[float] = None,
                        M_total: Optional[float] = None,
                        M_L: float = 0.5) -> Tuple[float, float]:
    """
    Estimate Q and σ_v from SPARC galaxy data.
    
    Convenience wrapper around EnvironmentEstimator.
    
    Parameters
    ----------
    gal : dict
        SPARC data with 'r', 'v_obs', etc.
    SBdisk : array, optional
        Surface brightness profile
    R_disk : float, optional
        Disk scale length
    M_total : float, optional
        Total mass for morphology classification
    M_L : float
        Mass-to-light ratio
        
    Returns
    -------
    Q : float
        Toomre Q parameter
    sigma_v : float
        Velocity dispersion (km/s)
    """
    estimator = EnvironmentEstimator()
    
    if SBdisk is not None and R_disk is not None and M_total is not None:
        # Full estimation
        morphology = estimator.classify_morphology(gal, M_total, R_disk)
        return estimator.estimate_from_sparc(gal, SBdisk, R_disk, M_L, morphology)
    else:
        # Simple fallback
        return estimator.estimate_simple(gal['v_obs'])
