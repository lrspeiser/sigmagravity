"""
Gravitational Channeling Kernel
===============================

Physical Picture:
-----------------
Gravitational field lines from distributed mass sources self-organize into 
coherent "channels" over cosmic time. Like rivers carving canyons, small 
anisotropies grow, reinforcing local gradients.

Why outer disks have deeper channels:
1. Lower σ_v → perturbations persist longer → channels aren't erased
2. Larger R → more "room" for channels to grow before saturating
3. More orbital periods → more time for cumulative deepening

Key prediction: Enhancement *increases* with radius — exactly what flat 
rotation curves require.

Formula:
--------
F(R) = 1 + χ₀ · (Σ/Σ_ref)^ε · D(R) / (1 + D/D_max)

Channel depth:
D(R) = (t_age/τ_ch)^γ · (v_c/σ_v)^β · (R/R_0)^α

Channel formation time:
τ_ch(R) = τ_0 · (σ_v/σ_ref) · (R_0/R)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List
import os


@dataclass
class ChannelingParams:
    """Parameters for gravitational channeling model."""
    
    # Channel growth exponents
    alpha: float = 0.7       # Channel scale grows with R
    beta: float = 1.5        # Cold systems carve deeper channels
    gamma: float = 0.3       # Sublinear accumulation (channel merging)
    
    # Coupling constants
    chi_0: float = 0.8       # Base coupling strength
    epsilon: float = 0.3     # Surface density exponent
    
    # Saturation
    D_max: float = 5.0       # Maximum channel depth (saturation)
    
    # Time scales
    t_age: float = 10.0      # System age [Gyr]
    tau_0: float = 1.0       # Reference channel formation time [Gyr]
    
    # Reference scales
    Sigma_ref: float = 100.0   # Reference surface density [M_sun/pc^2]
    sigma_ref: float = 30.0    # Reference velocity dispersion [km/s]
    R_0: float = 8.0           # Reference radius [kpc] (Solar galactocentric)
    
    def __post_init__(self):
        """Validate parameters."""
        assert self.alpha > 0, "alpha must be positive"
        assert self.beta > 0, "beta must be positive"
        assert 0 < self.gamma < 1, "gamma should be sublinear (0 < γ < 1)"
        assert self.chi_0 > 0, "chi_0 must be positive"
        assert self.D_max > 0, "D_max must be positive"


def channel_formation_time(R: np.ndarray, sigma_v: np.ndarray, 
                          params: ChannelingParams) -> np.ndarray:
    """
    Compute channel formation timescale.
    
    τ_ch(R) = τ_0 · (σ_v/σ_ref) · (R_0/R)
    
    Hot systems (high σ_v) have longer formation times (channels erased).
    Large R has shorter formation time (more room to grow).
    
    Parameters
    ----------
    R : array
        Galactocentric radius [kpc]
    sigma_v : array
        Velocity dispersion [km/s]
    params : ChannelingParams
        Model parameters
        
    Returns
    -------
    tau_ch : array
        Channel formation time [Gyr]
    """
    # Use tiny floor (1e-10) to allow Solar System scales
    R_safe = np.maximum(R, 1e-10)
    sigma_safe = np.maximum(sigma_v, 0.01)
    
    tau_ch = params.tau_0 * (sigma_safe / params.sigma_ref) * (params.R_0 / R_safe)
    
    return tau_ch


def channel_depth(R: np.ndarray, v_c: np.ndarray, sigma_v: np.ndarray,
                  params: ChannelingParams) -> np.ndarray:
    """
    Compute channel depth D(R).
    
    D(R) = (t_age/τ_ch)^γ · (v_c/σ_v)^β · (R/R_0)^α
    
    Parameters
    ----------
    R : array
        Galactocentric radius [kpc]
    v_c : array
        Circular velocity [km/s]
    sigma_v : array
        Velocity dispersion [km/s]
    params : ChannelingParams
        Model parameters
        
    Returns
    -------
    D : array
        Channel depth (dimensionless)
    """
    # Formation time at each radius
    tau_ch = channel_formation_time(R, sigma_v, params)
    
    # Use tiny floors to allow Solar System scales
    R_safe = np.maximum(R, 1e-10)
    sigma_safe = np.maximum(sigma_v, 0.01)
    tau_safe = np.maximum(tau_ch, 0.01)
    
    # Time accumulation term
    time_term = (params.t_age / tau_safe) ** params.gamma
    
    # Velocity coherence term (cold → deep channels)
    coherence_term = (v_c / sigma_safe) ** params.beta
    
    # Radial growth term
    radial_term = (R_safe / params.R_0) ** params.alpha
    
    D = time_term * coherence_term * radial_term
    
    return D


def channeling_enhancement(R: np.ndarray, v_c: np.ndarray, Sigma: np.ndarray,
                           sigma_v: np.ndarray, params: ChannelingParams,
                           diagnostics: Optional[Dict] = None) -> np.ndarray:
    """
    Compute gravitational enhancement factor F(R).
    
    F(R) = 1 + χ₀ · (Σ/Σ_ref)^ε · D(R) / (1 + D/D_max)
    
    The saturation term (1 + D/D_max) prevents runaway enhancement.
    
    Parameters
    ----------
    R : array
        Galactocentric radius [kpc]
    v_c : array
        Circular velocity [km/s]
    Sigma : array
        Surface density [M_sun/pc^2]
    sigma_v : array
        Velocity dispersion [km/s]
    params : ChannelingParams
        Model parameters
    diagnostics : dict, optional
        If provided, filled with intermediate values
        
    Returns
    -------
    F : array
        Enhancement factor (≥ 1)
    """
    # Ensure arrays
    R = np.atleast_1d(np.asarray(R, dtype=float))
    v_c = np.atleast_1d(np.asarray(v_c, dtype=float))
    Sigma = np.atleast_1d(np.asarray(Sigma, dtype=float))
    sigma_v = np.atleast_1d(np.asarray(sigma_v, dtype=float))
    
    # Channel depth
    D = channel_depth(R, v_c, sigma_v, params)
    
    # Surface density term
    Sigma_safe = np.maximum(Sigma, 0.01)
    density_term = (Sigma_safe / params.Sigma_ref) ** params.epsilon
    
    # Saturating channel contribution
    D_contribution = D / (1 + D / params.D_max)
    
    # Full enhancement
    enhancement = params.chi_0 * density_term * D_contribution
    F = 1.0 + enhancement
    
    # Fill diagnostics
    if diagnostics is not None:
        diagnostics['D'] = D
        diagnostics['D_contribution'] = D_contribution
        diagnostics['density_term'] = density_term
        diagnostics['enhancement'] = enhancement
        diagnostics['tau_ch'] = channel_formation_time(R, sigma_v, params)
    
    return F


def predict_v_obs(R: np.ndarray, v_bary: np.ndarray, Sigma: np.ndarray,
                  sigma_v: np.ndarray, params: ChannelingParams) -> np.ndarray:
    """
    Predict observed velocity from baryonic velocity with channeling.
    
    v_obs = v_bary · √F(R)
    
    Parameters
    ----------
    R : array
        Galactocentric radius [kpc]
    v_bary : array
        Baryonic rotation velocity [km/s]
    Sigma : array
        Surface density [M_sun/pc^2]
    sigma_v : array
        Velocity dispersion [km/s]
    params : ChannelingParams
        Model parameters
        
    Returns
    -------
    v_obs : array
        Predicted observed velocity [km/s]
    """
    F = channeling_enhancement(R, v_bary, Sigma, sigma_v, params)
    return v_bary * np.sqrt(F)


def estimate_sigma_v(R: np.ndarray, v_c: np.ndarray, 
                     is_gas_dominated: bool = False) -> np.ndarray:
    """
    Estimate velocity dispersion from rotation curve.
    
    For disks: σ_v ≈ v_c × asymmetric_drift_factor
    Outer disk is colder, inner disk is hotter.
    
    Parameters
    ----------
    R : array
        Galactocentric radius [kpc]
    v_c : array
        Circular velocity [km/s]
    is_gas_dominated : bool
        If True, use gas dispersion (~10 km/s)
        
    Returns
    -------
    sigma_v : array
        Estimated velocity dispersion [km/s]
    """
    if is_gas_dominated:
        # Gas has nearly constant dispersion
        return np.full_like(R, 10.0)
    
    # Stellar disk: σ_v decreases with R
    # Inner (R < 3 kpc): σ ≈ 0.3 × v_c
    # Outer (R > 10 kpc): σ ≈ 0.1 × v_c
    # Interpolate between
    R_inner = 3.0
    R_outer = 10.0
    f_inner = 0.3
    f_outer = 0.1
    
    # Smooth interpolation
    R_safe = np.maximum(R, 0.1)
    x = np.clip((R_safe - R_inner) / (R_outer - R_inner), 0, 1)
    factor = f_inner + (f_outer - f_inner) * x
    
    sigma_v = v_c * factor
    
    # Floor at gas dispersion
    sigma_v = np.maximum(sigma_v, 8.0)
    
    return sigma_v


def compute_surface_density(R: np.ndarray, M_disk: float, R_d: float,
                           M_bulge: float = 0, R_b: float = 1.0) -> np.ndarray:
    """
    Compute surface density from exponential disk + bulge.
    
    Parameters
    ----------
    R : array
        Galactocentric radius [kpc]
    M_disk : float
        Total disk mass [M_sun]
    R_d : float
        Disk scale length [kpc]
    M_bulge : float
        Bulge mass [M_sun]
    R_b : float
        Bulge effective radius [kpc]
        
    Returns
    -------
    Sigma : array
        Surface density [M_sun/pc^2]
    """
    R = np.atleast_1d(np.asarray(R, dtype=float))
    
    # Exponential disk: Σ(R) = Σ_0 × exp(-R/R_d)
    # Σ_0 = M_disk / (2π R_d²)
    Sigma_0_disk = M_disk / (2 * np.pi * (R_d * 1e3)**2)  # M_sun/pc^2
    Sigma_disk = Sigma_0_disk * np.exp(-R / R_d)
    
    # Bulge (de Vaucouleurs-like, simplified as exponential)
    if M_bulge > 0:
        Sigma_0_bulge = M_bulge / (2 * np.pi * (R_b * 1e3)**2)
        Sigma_bulge = Sigma_0_bulge * np.exp(-R / R_b)
    else:
        Sigma_bulge = 0
    
    return Sigma_disk + Sigma_bulge


# =============================================================================
# SPARC Data Loading
# =============================================================================

def load_sparc_galaxy(filepath: str) -> Dict:
    """
    Load a single SPARC galaxy rotation curve.
    
    Parameters
    ----------
    filepath : str
        Path to the .dat file
        
    Returns
    -------
    data : dict
        Dictionary with R, v_obs, v_err, v_gas, v_disk, v_bul, etc.
    """
    data = {
        'R': [], 'v_obs': [], 'v_err': [],
        'v_gas': [], 'v_disk': [], 'v_bul': [],
        'SBdisk': [], 'SBbul': []
    }
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.split()
            if len(parts) >= 6:
                data['R'].append(float(parts[0]))
                data['v_obs'].append(float(parts[1]))
                data['v_err'].append(float(parts[2]))
                data['v_gas'].append(float(parts[3]))
                data['v_disk'].append(float(parts[4]))
                data['v_bul'].append(float(parts[5]))
                if len(parts) >= 8:
                    data['SBdisk'].append(float(parts[6]))
                    data['SBbul'].append(float(parts[7]))
    
    for key in data:
        data[key] = np.array(data[key])
    
    # Compute total baryonic velocity
    v_gas = data['v_gas']
    v_disk = data['v_disk']
    v_bul = data['v_bul']
    
    # Handle signed velocities (gas can be negative for counter-rotation)
    v_gas_sq = np.sign(v_gas) * v_gas**2
    v_disk_sq = np.sign(v_disk) * v_disk**2
    v_bul_sq = v_bul**2
    
    v_bary_sq = v_gas_sq + v_disk_sq + v_bul_sq
    data['v_bary'] = np.sign(v_bary_sq) * np.sqrt(np.abs(v_bary_sq))
    
    # Estimate surface density from surface brightness (approximate)
    # Using M/L ~ 0.5 for disk in 3.6μm
    if len(data['SBdisk']) > 0:
        # SB is in L_sun/pc^2, convert to M_sun/pc^2
        ML_disk = 0.5
        ML_bul = 0.7
        data['Sigma'] = ML_disk * data['SBdisk'] + ML_bul * data['SBbul']
    else:
        # Estimate from velocity
        data['Sigma'] = estimate_sigma_from_v(data['R'], data['v_bary'])
    
    data['name'] = os.path.basename(filepath).replace('_rotmod.dat', '')
    
    return data


def estimate_sigma_from_v(R: np.ndarray, v_bary: np.ndarray) -> np.ndarray:
    """
    Rough estimate of surface density from rotation curve.
    
    Σ ≈ v²/(2πGR) for a thin disk
    """
    G = 4.302e-6  # kpc (km/s)² / M_sun
    R_safe = np.maximum(R, 0.1)
    
    # M(<R) ≈ v² R / G
    M_enc = v_bary**2 * R_safe / G
    
    # Σ ≈ M / (π R²) converted to M_sun/pc²
    Sigma = M_enc / (np.pi * (R_safe * 1e3)**2)
    
    return np.maximum(Sigma, 0.1)


def compute_rms(v_pred: np.ndarray, v_obs: np.ndarray, 
                v_err: Optional[np.ndarray] = None) -> float:
    """Compute RMS deviation."""
    residuals = v_pred - v_obs
    return np.sqrt(np.mean(residuals**2))


def compute_chi2(v_pred: np.ndarray, v_obs: np.ndarray, 
                 v_err: np.ndarray) -> float:
    """Compute chi-squared."""
    v_err_safe = np.maximum(v_err, 1.0)
    chi2 = np.sum(((v_pred - v_obs) / v_err_safe)**2)
    return chi2


# =============================================================================
# Solar System Test
# =============================================================================

def test_solar_system(params: ChannelingParams) -> Tuple[float, bool]:
    """
    Test Cassini constraint: δg/g < 2.3×10⁻⁵ at Saturn.
    
    Solar System has:
    - Nearly zero surface density (point mass)
    - Very low velocity dispersion irrelevant (bound orbits)
    - R ~ 10 AU, v_c ~ 10 km/s
    
    Returns
    -------
    delta_g : float
        Fractional gravity deviation
    passes : bool
        Whether constraint is satisfied
    """
    # Saturn orbital parameters
    R_saturn_au = 9.5
    R_saturn_kpc = R_saturn_au * 4.85e-9  # AU to kpc
    v_saturn = 9.7  # km/s
    
    # Solar System has essentially zero surface density
    # (all mass in point-like Sun)
    Sigma_ss = 1e-10  # M_sun/pc^2 (effectively zero)
    
    # Velocity dispersion in SS is tiny (bound Keplerian orbits)
    sigma_ss = 0.1  # km/s
    
    F = channeling_enhancement(
        R=np.array([R_saturn_kpc]),
        v_c=np.array([v_saturn]),
        Sigma=np.array([Sigma_ss]),
        sigma_v=np.array([sigma_ss]),
        params=params
    )[0]
    
    # δg/g = F - 1
    delta_g = F - 1
    cassini_limit = 2.3e-5
    
    passes = delta_g < cassini_limit
    
    return delta_g, passes


# =============================================================================
# Fitting utilities
# =============================================================================

def fit_galaxy(data: Dict, params: ChannelingParams,
               fit_sigma: bool = True) -> Dict:
    """
    Fit channeling model to a single galaxy.
    
    Parameters
    ----------
    data : dict
        Galaxy data from load_sparc_galaxy
    params : ChannelingParams
        Model parameters
    fit_sigma : bool
        If True, estimate σ_v from rotation curve
        
    Returns
    -------
    results : dict
        Fitting results including v_pred, RMS, chi2
    """
    R = data['R']
    v_obs = data['v_obs']
    v_err = data['v_err']
    v_bary = np.abs(data['v_bary'])  # Use absolute value
    Sigma = data.get('Sigma', estimate_sigma_from_v(R, v_bary))
    
    # Estimate velocity dispersion
    if fit_sigma:
        # Check if gas-dominated (v_gas > v_disk at large R)
        is_gas_dom = np.mean(np.abs(data['v_gas'][-3:])) > np.mean(np.abs(data['v_disk'][-3:]))
        sigma_v = estimate_sigma_v(R, v_bary, is_gas_dominated=is_gas_dom)
    else:
        sigma_v = np.full_like(R, 20.0)  # Default
    
    # Compute enhancement
    diagnostics = {}
    F = channeling_enhancement(R, v_bary, Sigma, sigma_v, params, diagnostics)
    
    # Predicted velocity
    v_pred = v_bary * np.sqrt(F)
    
    # Metrics
    rms_pred = compute_rms(v_pred, v_obs)
    rms_bary = compute_rms(v_bary, v_obs)
    chi2 = compute_chi2(v_pred, v_obs, v_err)
    
    return {
        'name': data.get('name', 'unknown'),
        'R': R,
        'v_obs': v_obs,
        'v_err': v_err,
        'v_bary': v_bary,
        'v_pred': v_pred,
        'F': F,
        'sigma_v': sigma_v,
        'Sigma': Sigma,
        'rms_pred': rms_pred,
        'rms_bary': rms_bary,
        'delta_rms': rms_pred - rms_bary,
        'chi2': chi2,
        'chi2_dof': chi2 / max(len(R) - 6, 1),
        'diagnostics': diagnostics,
        'improved': rms_pred < rms_bary
    }


if __name__ == "__main__":
    # Quick sanity test
    print("Gravitational Channeling Kernel - Sanity Test")
    print("=" * 50)
    
    params = ChannelingParams()
    
    # Test MW-like galaxy
    R = np.array([2, 4, 6, 8, 10, 15, 20, 25])  # kpc
    v_c = np.array([180, 200, 210, 220, 220, 210, 200, 190])  # km/s
    Sigma = np.array([500, 200, 100, 50, 30, 10, 5, 2])  # M_sun/pc^2
    sigma_v = estimate_sigma_v(R, v_c)
    
    diag = {}
    F = channeling_enhancement(R, v_c, Sigma, sigma_v, params, diag)
    
    print("\nMW-like test:")
    print(f"{'R (kpc)':<10} {'σ_v':<10} {'D':<10} {'F':<10}")
    print("-" * 40)
    for i in range(len(R)):
        print(f"{R[i]:<10.1f} {sigma_v[i]:<10.1f} {diag['D'][i]:<10.2f} {F[i]:<10.3f}")
    
    # Solar System test
    print("\n" + "=" * 50)
    delta_g, passes = test_solar_system(params)
    print(f"Solar System test: δg/g = {delta_g:.2e}")
    print(f"Cassini limit: 2.3×10⁻⁵")
    print(f"Result: {'PASS ✓' if passes else 'FAIL ✗'}")
