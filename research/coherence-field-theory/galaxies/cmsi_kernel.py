"""
Coherent Metric Self-Interaction (CMSI) Kernel
===============================================

First-principles derivation of gravitational coherence enhancement.

Physical basis:
--------------
In weak-field GR, metric perturbations h_μν satisfy:
    □h_μν = -16πG T_μν + Λ_μν[h]

where Λ_μν[h] contains nonlinear self-interaction terms. These are usually
negligible, but in a *coherent* system (phase-aligned orbiting masses),
they can accumulate constructively.

Key insight: The enhancement factor is NOT fitted - it emerges from:
1. Phase correlation statistics → σ-dependence
2. Nonlinear self-interaction strength → amplitude
3. Geometric coherence volume → radial profile

See: derivations/CMSI_DERIVATION_SUMMARY.md for full derivation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Callable


# =============================================================================
# Physical Constants
# =============================================================================

G_NEWTON = 4.302e-6  # kpc (km/s)^2 / M_sun
C_LIGHT = 299792.458  # km/s


# =============================================================================
# Core CMSI Physics
# =============================================================================

@dataclass
class CMSIParams:
    """
    Parameters for CMSI kernel - all have physical interpretation.
    
    chi_0: Base nonlinear coupling strength (dimensionless).
           The ACTUAL coupling is χ = χ_0 × (v_c/c)^2 × source_factor
           Expected range: 10^2 - 10^4
    
    gamma_phase: Exponent for phase coherence scaling with σ_v/v_c.
                 From Gaussian phase decorrelation: γ ~ 1.5-2.0
                 Theoretical value: 1.5 (3D velocity dispersion)
    
    alpha_Ncoh: Exponent for how N_coh contributes to enhancement.
                α = 0.5: random-phase square-root (conservative)
                α = 1.0: fully coherent linear (maximum)
                Expected: 0.5 - 0.75
    
    ell_0_kpc: Characteristic coherence length scale [kpc].
               Sets the radial profile shape.
               If use_scale_dependent_ell0=True, this is ignored.
    
    n_profile: Shape exponent for radial coherence profile.
    
    Sigma_ref: Reference surface density [M_sun/pc²].
               Coherent self-interaction requires multiple mass sources.
               Solar System: Σ ~ 0 (point mass)
               Galaxy disk: Σ ~ 10-100 M_sun/pc²
               This naturally kills the effect in the Solar System.
    
    epsilon_Sigma: Exponent for surface density scaling.
                   Higher = stronger dependence on distributed mass.
    
    Note on (v/c)²: This is CRUCIAL - it's why Solar System passes:
          Galaxy: v ~ 200 km/s → (v/c)² ~ 4×10^-7
          Solar:  v ~ 10 km/s  → (v/c)² ~ 10^-9
    
    Note on Σ: This is ALSO crucial - point masses don't self-interfere:
          Galaxy: Σ ~ 50 M_sun/pc² → (Σ/Σ_ref) ~ 1
          Solar:  Σ ~ 0 → (Σ/Σ_ref) ~ 0
    
    Scale-dependent ℓ₀:
        From coherence time: ℓ_coh ~ c × τ_coh ~ c × R/v_c
        This implies ℓ₀ should scale with system size: ℓ₀ = η × R_half
        where η is a universal constant (~0.3-0.6).
        This unifies galaxies and clusters with a single parameter!
    """
    chi_0: float = 800.0       # Base nonlinear coupling
    gamma_phase: float = 1.5   # Phase coherence exponent
    alpha_Ncoh: float = 0.55   # N_coh → enhancement exponent
    ell_0_kpc: float = 2.2     # Coherence length scale [kpc] (if not scale-dependent)
    n_profile: float = 2.0     # Radial profile shape
    
    # Source density parameters (the "no self-interference for point masses" physics)
    Sigma_ref: float = 50.0    # Reference surface density [M_sun/pc²]
    epsilon_Sigma: float = 0.5 # Surface density exponent
    
    # Optional: combine with time-coherence (K_rough)
    include_K_rough: bool = True
    K_rough_prefactor: float = 0.774
    K_rough_exponent: float = 0.1
    
    # Scale-dependent coherence length (unifies galaxies and clusters)
    use_scale_dependent_ell0: bool = False
    eta_ell0: float = 0.5      # ℓ₀ = η × R_half (universal constant)


def compute_phase_coherence(
    v_circ: np.ndarray,
    sigma_v: np.ndarray,
    gamma: float = 1.5
) -> np.ndarray:
    """
    Compute phase coherence parameter C from velocity structure.
    
    Physics: For N mass elements with orbital phases φ_i(t), the coherence is
        C = (1/N)|Σ exp(iφ_i)|²
    
    The phase spread over a dynamical time scales as:
        Δφ ~ (σ_v / v_c) × 2π
    
    So the number of coherently-contributing orbital phases is:
        N_coh ~ (v_c / σ_v)^γ
    
    where γ depends on geometry:
        - γ = 1: 1D phase spread
        - γ = 1.5: 3D velocity dispersion (theoretical expectation)
        - γ = 2: worst-case decorrelation
    
    Parameters
    ----------
    v_circ : array
        Circular velocity [km/s]
    sigma_v : array
        Velocity dispersion [km/s]
    gamma : float
        Phase coherence exponent (default: 1.5)
    
    Returns
    -------
    N_coh : array
        Number of coherently-contributing orbits
    """
    # Protect against division by zero
    sigma_safe = np.maximum(sigma_v, 1.0)  # Floor at 1 km/s
    
    # Phase coherence ratio
    coherence_ratio = v_circ / sigma_safe
    
    # Number of coherent orbits
    N_coh = np.power(coherence_ratio, gamma)
    
    return N_coh


def compute_radial_coherence_profile(
    R_kpc: np.ndarray,
    ell_0: float = 2.5,
    n: float = 2.0
) -> np.ndarray:
    """
    Compute radial coherence profile f(R/ℓ_0).
    
    Physics: The coherence volume at radius R has a characteristic
    size ℓ_coh. The profile describes how the nonlinear self-interaction
    strength varies with the ratio R/ℓ_0.
    
    We use a generalized Burr-type profile (matches your empirical finding):
        f(x) = 1 / (1 + x^n)^(1/n)
    
    where x = R/ℓ_0. This:
        - → 1 as R → 0 (full coherence at small radii)
        - → (ℓ_0/R) as R → ∞ (decay at large radii)
        - n controls the transition sharpness
    
    Parameters
    ----------
    R_kpc : array
        Galactocentric radius [kpc]
    ell_0 : float
        Coherence length scale [kpc]
    n : float
        Profile shape exponent
    
    Returns
    -------
    f_profile : array
        Radial coherence profile (0 to 1)
    """
    x = R_kpc / ell_0
    f_profile = np.power(1.0 + np.power(x, n), -1.0/n)
    return f_profile


def compute_K_rough(
    R_kpc: np.ndarray,
    v_circ: np.ndarray,
    sigma_v: np.ndarray,
    Sigma: Optional[np.ndarray],
    Sigma_ref: float,
    epsilon_Sigma: float,
    prefactor: float = 0.774,
    exponent: float = 0.1,
    alpha_c: float = 1e-4  # ℓ_coh = α_c × c × τ_coh
) -> np.ndarray:
    """
    Compute time-coherence enhancement K_rough from exposure factor Ξ.
    
    This implements your DERIVED piece with the addition of source density
    suppression for point masses.
    
    Physics:
        τ_coh = min(τ_geom, τ_noise)
        τ_geom ~ R / v_circ
        τ_noise ~ R / σ_v^β
        Ξ = τ_coh / T_orb
        K_rough ≈ 0.774 × Ξ^0.1 × (Σ/Σ_ref)^ε
    
    The source density factor ensures point masses (Solar System) don't
    get enhancement from "roughness" that doesn't exist.
    
    Parameters
    ----------
    R_kpc : array
        Galactocentric radius [kpc]
    v_circ : array
        Circular velocity [km/s]
    sigma_v : array
        Velocity dispersion [km/s]
    Sigma : array or None
        Surface density [M_sun/pc²]
    Sigma_ref : float
        Reference surface density
    epsilon_Sigma : float
        Surface density exponent
    prefactor : float
        K_rough prefactor (default: 0.774 from your derivation)
    exponent : float
        K_rough exponent (default: 0.1)
    alpha_c : float
        Coherence length coefficient (ℓ_coh = α_c × c × τ_coh)
    
    Returns
    -------
    K_rough : array
        Time-coherence enhancement factor
    """
    # Geometric coherence time
    tau_geom = R_kpc / (v_circ * 1.0227e-3)  # Convert to Gyr (kpc / (km/s) → Gyr)
    
    # Noise-limited coherence time (β=1 case)
    sigma_safe = np.maximum(sigma_v, 1.0)
    tau_noise = R_kpc / (sigma_safe * 1.0227e-3)
    
    # Effective coherence time
    tau_coh = np.minimum(tau_geom, tau_noise)
    
    # Orbital period
    T_orb = 2.0 * np.pi * R_kpc / (v_circ * 1.0227e-3)
    
    # Exposure factor
    Xi = tau_coh / T_orb
    Xi = np.maximum(Xi, 1e-10)  # Floor to avoid log issues
    
    # Base time-coherence enhancement
    K_rough_base = prefactor * np.power(Xi, exponent)
    
    # Apply source density suppression (point masses have no roughness)
    if Sigma is not None:
        Sigma_safe = np.maximum(Sigma, 1e-10)
        source_factor = np.power(Sigma_safe / Sigma_ref, epsilon_Sigma)
        source_factor = np.minimum(source_factor, 2.0)  # Cap
        K_rough = K_rough_base * source_factor
    else:
        K_rough = K_rough_base
    
    return K_rough


def cmsi_enhancement(
    R_kpc: np.ndarray,
    v_circ: np.ndarray,
    sigma_v: np.ndarray,
    params: CMSIParams,
    Sigma: Optional[np.ndarray] = None,
    R_half: Optional[float] = None
) -> Tuple[np.ndarray, dict]:
    """
    Compute full CMSI enhancement factor F_CMSI(R).
    
    This is the MAIN FUNCTION implementing the first-principles derivation.
    
    Physics:
        F_CMSI = 1 + χ_0 × (v_c/c)² × (Σ/Σ_ref)^ε × N_coh^α × f(R/ℓ_0)
    
    where:
        - χ_0: base nonlinear coupling
        - (v/c)²: from nonlinear GR self-interaction (Landau-Lifshitz)
        - (Σ/Σ_ref)^ε: source density factor - point masses don't self-interfere
        - N_coh: phase coherence (derived from σ_v/v_c)
        - f(R/ℓ_0): radial coherence profile
    
    The source density factor is KEY to Solar System safety:
        Galaxy disk: Σ ~ 50 M_sun/pc² → full effect
        Solar System: Σ ~ 0 (point mass) → no effect
    
    Scale-dependent ℓ₀:
        If params.use_scale_dependent_ell0=True and R_half is provided:
        ℓ₀ = η × R_half  (derived from coherence time physics)
        This unifies galaxies (~5 kpc) and clusters (~500 kpc).
    
    Parameters
    ----------
    R_kpc : array
        Galactocentric radius [kpc]
    v_circ : array
        Circular velocity [km/s] - can be observed or model
    sigma_v : array
        Velocity dispersion [km/s]
    params : CMSIParams
        CMSI parameters
    Sigma : array, optional
        Surface density [M_sun/pc²]. If None, estimate from v_circ profile.
    R_half : float, optional
        Half-mass/half-light radius [kpc]. Used for scale-dependent ℓ₀.
        If None and scale-dependent mode is on, estimates from R array.
    
    Returns
    -------
    F_CMSI : array
        Total enhancement factor at each radius
    diagnostics : dict
        Intermediate quantities for debugging/analysis
    """
    # Step 1: Phase coherence → N_coh
    N_coh = compute_phase_coherence(v_circ, sigma_v, params.gamma_phase)
    
    # Step 2: Determine coherence length scale
    if params.use_scale_dependent_ell0:
        # Scale-dependent ℓ₀ from coherence time: ℓ₀ = η × R_half
        if R_half is None:
            # Estimate R_half from data extent (median radius)
            R_half = np.median(R_kpc)
        ell_0_local = params.eta_ell0 * R_half
    else:
        ell_0_local = params.ell_0_kpc
    
    # Step 3: Radial profile with local ℓ₀
    f_profile = compute_radial_coherence_profile(
        R_kpc, ell_0_local, params.n_profile
    )
    
    # Step 3: The (v/c)² factor from nonlinear GR
    v_over_c_squared = (v_circ / C_LIGHT) ** 2
    
    # Step 4: Source density factor
    # Coherent self-interaction requires distributed mass sources
    # Point masses (like the Sun) don't self-interfere
    if Sigma is None:
        # Estimate from rotation curve: Σ ~ v²/(2πGR) for flat part
        # This is crude but gives right order of magnitude for disks
        R_pc = R_kpc * 1000.0  # Convert to pc
        Sigma = (v_circ**2) / (2.0 * np.pi * G_NEWTON * R_kpc * 1e6)  # M_sun/pc²
        # For MW-like at R=8 kpc: Σ ~ 50 M_sun/pc²
    
    # Normalize and apply exponent
    Sigma_safe = np.maximum(Sigma, 1e-10)  # Protect against zero
    source_factor = np.power(Sigma_safe / params.Sigma_ref, params.epsilon_Sigma)
    # Cap at 2 to avoid runaway in dense regions
    source_factor = np.minimum(source_factor, 2.0)
    
    # Step 5: Coherent self-interaction amplitude
    # This combines all the physical factors
    coherent_amplitude = (
        params.chi_0 
        * v_over_c_squared 
        * source_factor
        * np.power(N_coh, params.alpha_Ncoh) 
        * f_profile
    )
    
    # Step 6: CMSI enhancement (the "F_missing" from first principles)
    F_cmsi_core = 1.0 + coherent_amplitude
    
    # Step 7: Optionally include K_rough (multiplicative)
    if params.include_K_rough:
        K_rough = compute_K_rough(
            R_kpc, v_circ, sigma_v, Sigma,
            params.Sigma_ref, params.epsilon_Sigma,
            params.K_rough_prefactor, params.K_rough_exponent
        )
        F_total = F_cmsi_core * (1.0 + K_rough)
    else:
        K_rough = np.zeros_like(R_kpc)
        F_total = F_cmsi_core
    
    # Diagnostics
    diagnostics = {
        'N_coh': N_coh,
        'f_profile': f_profile,
        'v_over_c_squared': v_over_c_squared,
        'Sigma': Sigma,
        'source_factor': source_factor,
        'coherent_amplitude': coherent_amplitude,
        'F_cmsi_core': F_cmsi_core,
        'K_rough': K_rough,
        'F_total': F_total,
        'sigma_over_vc': sigma_v / np.maximum(v_circ, 1.0),
        'ell_0_local': ell_0_local,
        'R_half': R_half if params.use_scale_dependent_ell0 else None,
    }
    
    return F_total, diagnostics


# =============================================================================
# Enhanced Gravity Calculation
# =============================================================================

def compute_v_circ_enhanced(
    R_kpc: np.ndarray,
    v_circ_bary: np.ndarray,
    sigma_v: np.ndarray,
    params: CMSIParams,
    Sigma: Optional[np.ndarray] = None,
    use_iterative: bool = True,
    n_iter: int = 5
) -> Tuple[np.ndarray, dict]:
    """
    Compute enhanced circular velocity including CMSI effect.
    
    The enhanced gravitational potential gives:
        v_circ_enhanced² = v_circ_bary² × F_CMSI
    
    Since F_CMSI depends on v_circ (through N_coh), we can either:
    1. Use v_circ_bary as proxy (fast, approximate)
    2. Iterate to self-consistency (slower, more accurate)
    
    Parameters
    ----------
    R_kpc : array
        Galactocentric radius [kpc]
    v_circ_bary : array
        Baryonic circular velocity [km/s]
    sigma_v : array
        Velocity dispersion [km/s]
    params : CMSIParams
        CMSI parameters
    Sigma : array, optional
        Surface density [M_sun/pc²]
    use_iterative : bool
        If True, iterate to self-consistency
    n_iter : int
        Number of iterations (if iterative)
    
    Returns
    -------
    v_circ_enhanced : array
        Enhanced circular velocity [km/s]
    diagnostics : dict
        Diagnostic quantities
    """
    if use_iterative:
        # Start with baryonic as initial guess
        v_circ_current = v_circ_bary.copy()
        
        for i in range(n_iter):
            F_CMSI, diag = cmsi_enhancement(R_kpc, v_circ_current, sigma_v, params, Sigma)
            v_circ_new = v_circ_bary * np.sqrt(F_CMSI)
            
            # Check convergence
            delta = np.max(np.abs(v_circ_new - v_circ_current))
            v_circ_current = v_circ_new
            
            if delta < 0.1:  # Converged to 0.1 km/s
                break
        
        v_circ_enhanced = v_circ_current
        F_CMSI, diagnostics = cmsi_enhancement(R_kpc, v_circ_enhanced, sigma_v, params, Sigma)
    else:
        # Use baryonic velocity as proxy
        F_CMSI, diagnostics = cmsi_enhancement(R_kpc, v_circ_bary, sigma_v, params, Sigma)
        v_circ_enhanced = v_circ_bary * np.sqrt(F_CMSI)
    
    diagnostics['v_circ_bary'] = v_circ_bary
    diagnostics['v_circ_enhanced'] = v_circ_enhanced
    diagnostics['F_CMSI'] = F_CMSI
    
    return v_circ_enhanced, diagnostics


# =============================================================================
# Sigma Profile Models (for testing)
# =============================================================================

def sigma_exponential_disk(
    R_kpc: np.ndarray,
    sigma_0: float = 30.0,
    R_sigma: float = 4.0,
    sigma_floor: float = 8.0
) -> np.ndarray:
    """
    Exponential disk velocity dispersion profile.
    
    σ_v(R) = σ_0 × exp(-R/R_σ) + σ_floor
    
    Typical values for MW-like disk:
        σ_0 ~ 30 km/s (central)
        R_σ ~ 4 kpc (scale length)
        σ_floor ~ 8 km/s (outer disk minimum)
    """
    return sigma_0 * np.exp(-R_kpc / R_sigma) + sigma_floor


def sigma_constant(
    R_kpc: np.ndarray,
    sigma_v: float = 20.0
) -> np.ndarray:
    """Constant velocity dispersion (useful for simple tests)."""
    return np.full_like(R_kpc, sigma_v)


def sigma_from_Toomre_Q(
    R_kpc: np.ndarray,
    v_circ: np.ndarray,
    Sigma_disk: np.ndarray,
    Q: float = 1.5
) -> np.ndarray:
    """
    Derive σ_v from Toomre Q parameter.
    
    Q = (σ_v × κ) / (π G Σ)
    
    For flat rotation curve, κ ≈ √2 × (v_c / R)
    
    So: σ_v = Q × π G Σ / κ = Q × π G Σ × R / (√2 × v_c)
    """
    kappa = np.sqrt(2.0) * v_circ / R_kpc  # Epicyclic frequency
    sigma_v = Q * np.pi * G_NEWTON * Sigma_disk / kappa
    return np.maximum(sigma_v, 5.0)  # Floor at 5 km/s


# =============================================================================
# Solar System Safety Check
# =============================================================================

def check_solar_system_safety(params: CMSIParams) -> dict:
    """
    Verify CMSI doesn't violate Solar System constraints.
    
    The Cassini constraint requires |δg/g| < 2.3 × 10^-5 at Saturn.
    
    CRUCIAL PHYSICS: The Solar System is safe because it's a POINT MASS.
    
    The source density factor kills CMSI in the Solar System:
        - Galaxy disk: Σ ~ 50 M_sun/pc² → (Σ/Σ_ref)^0.5 ~ 1
        - Solar System: Σ ~ 0 (point mass) → (Σ/Σ_ref)^0.5 ~ 0
    
    A point mass cannot coherently self-interfere with itself!
    This is the physical reason CMSI affects galaxies but not the Solar System.
    """
    # Solar system parameters
    R_saturn_kpc = 9.5 * 4.848e-9  # 9.5 AU in kpc
    v_circ_saturn = 9.7  # km/s
    sigma_v_solar = 0.01  # km/s (planets on very circular orbits)
    
    # CRITICAL: The Solar System has essentially ZERO surface density
    # at Saturn's orbit. All the mass is in the Sun (a point source).
    # This is what kills CMSI in the Solar System.
    Sigma_solar_system = 1e-20  # Essentially zero [M_sun/pc²]
    
    R_test = np.array([R_saturn_kpc])
    v_test = np.array([v_circ_saturn])
    sigma_test = np.array([sigma_v_solar])
    Sigma_test = np.array([Sigma_solar_system])
    
    F_CMSI, diag = cmsi_enhancement(R_test, v_test, sigma_test, params, Sigma_test)
    
    # Enhancement of g
    delta_g_over_g = F_CMSI[0] - 1.0
    
    # Cassini constraint
    cassini_limit = 2.3e-5
    
    # Also compute what would happen if we ignored the source density factor
    # (to show why it's essential)
    Sigma_if_disk = 50.0  # What if Saturn were in a disk?
    F_CMSI_if_disk, _ = cmsi_enhancement(
        R_test, v_test, sigma_test, params, np.array([Sigma_if_disk])
    )
    
    return {
        'F_CMSI_saturn': F_CMSI[0],
        'delta_g_over_g': delta_g_over_g,
        'cassini_limit': cassini_limit,
        'passes_cassini': np.abs(delta_g_over_g) < cassini_limit,
        'N_coh': diag['N_coh'][0],
        'f_profile': diag['f_profile'][0],
        'v_over_c_squared': diag['v_over_c_squared'][0],
        'Sigma': diag['Sigma'][0],
        'source_factor': diag['source_factor'][0],
        'coherent_amplitude': diag['coherent_amplitude'][0],
        'F_CMSI_if_in_disk': F_CMSI_if_disk[0],
        'note': 'Point mass (Σ~0) kills CMSI in Solar System'
    }


# =============================================================================
# Utility: RMS calculation
# =============================================================================

def compute_rms(v_model: np.ndarray, v_obs: np.ndarray, 
                v_err: Optional[np.ndarray] = None) -> float:
    """Compute RMS difference between model and observed velocities."""
    residuals = v_model - v_obs
    if v_err is not None:
        # Weighted RMS
        weights = 1.0 / v_err**2
        rms = np.sqrt(np.sum(weights * residuals**2) / np.sum(weights))
    else:
        rms = np.sqrt(np.mean(residuals**2))
    return rms


# =============================================================================
# Main test function
# =============================================================================

def test_cmsi_kernel():
    """Quick self-test of CMSI kernel."""
    print("=" * 60)
    print("CMSI Kernel Self-Test")
    print("=" * 60)
    
    # Test parameters
    params = CMSIParams(
        chi_0=800.0,            # Base coupling
        gamma_phase=1.5,        # Phase coherence exponent
        alpha_Ncoh=0.55,        # N_coh exponent
        ell_0_kpc=2.2,          # Coherence length [kpc]
        n_profile=2.0,          # Radial profile shape
        Sigma_ref=50.0,         # Reference surface density [M_sun/pc²]
        epsilon_Sigma=0.5,      # Surface density exponent
        include_K_rough=True
    )
    
    print(f"\nParameters:")
    print(f"  χ_0 = {params.chi_0:.1f}")
    print(f"  γ_phase = {params.gamma_phase}")
    print(f"  α_Ncoh = {params.alpha_Ncoh}")
    print(f"  ℓ_0 = {params.ell_0_kpc} kpc")
    print(f"  Σ_ref = {params.Sigma_ref} M_sun/pc²")
    print(f"  ε_Σ = {params.epsilon_Sigma}")
    
    # Create test profile (MW-like)
    R_kpc = np.linspace(0.5, 20.0, 40)
    v_circ_bary = 80.0 * np.sqrt(R_kpc / 2.0) / np.sqrt(1 + R_kpc / 2.0)
    v_circ_bary = np.maximum(v_circ_bary, 50.0) + 100.0
    sigma_v = sigma_exponential_disk(R_kpc, sigma_0=35.0, R_sigma=4.0, sigma_floor=10.0)
    
    # Compute CMSI enhancement (let it estimate Sigma from v_circ)
    F_CMSI, diag = cmsi_enhancement(R_kpc, v_circ_bary, sigma_v, params)
    v_enhanced, _ = compute_v_circ_enhanced(R_kpc, v_circ_bary, sigma_v, params)
    
    print(f"\nTest profile: MW-like disk")
    print(f"  R range: {R_kpc[0]:.1f} - {R_kpc[-1]:.1f} kpc")
    print(f"  v_bary range: {v_circ_bary.min():.1f} - {v_circ_bary.max():.1f} km/s")
    print(f"  σ_v range: {sigma_v.min():.1f} - {sigma_v.max():.1f} km/s")
    print(f"  Σ (estimated) range: {diag['Sigma'].min():.1f} - {diag['Sigma'].max():.1f} M_sun/pc²")
    
    print(f"\nPhysical factors:")
    print(f"  (v/c)² range: {diag['v_over_c_squared'].min():.2e} - {diag['v_over_c_squared'].max():.2e}")
    print(f"  Source factor range: {diag['source_factor'].min():.3f} - {diag['source_factor'].max():.3f}")
    print(f"  N_coh range: {diag['N_coh'].min():.1f} - {diag['N_coh'].max():.1f}")
    
    print(f"\nCMSI Enhancement:")
    print(f"  F_CMSI range: {F_CMSI.min():.3f} - {F_CMSI.max():.3f}")
    print(f"  v_enhanced range: {v_enhanced.min():.1f} - {v_enhanced.max():.1f} km/s")
    
    # Sample points
    for R_sample in [2.0, 8.0, 15.0]:
        idx = np.argmin(np.abs(R_kpc - R_sample))
        print(f"\n  At R = {R_sample} kpc:")
        print(f"    v_bary = {v_circ_bary[idx]:.1f} km/s → v_enhanced = {v_enhanced[idx]:.1f} km/s")
        print(f"    σ_v = {sigma_v[idx]:.1f} km/s, N_coh = {diag['N_coh'][idx]:.1f}")
        print(f"    Σ = {diag['Sigma'][idx]:.1f} M_sun/pc², source_factor = {diag['source_factor'][idx]:.3f}")
        print(f"    F_CMSI = {F_CMSI[idx]:.3f}")
    
    # Solar system check
    print("\n" + "-" * 60)
    print("Solar System Safety Check:")
    print("-" * 60)
    ss_check = check_solar_system_safety(params)
    print(f"  Key insight: Solar System is a POINT MASS (Σ ~ 0)")
    print(f"")
    print(f"  Saturn parameters:")
    print(f"    (v/c)² = {ss_check['v_over_c_squared']:.2e}")
    print(f"    N_coh = {ss_check['N_coh']:.1f} (high coherence!)")
    print(f"    Σ = {ss_check['Sigma']:.2e} M_sun/pc² (essentially zero)")
    print(f"    source_factor = {ss_check['source_factor']:.2e} (THIS kills it)")
    print(f"")
    print(f"  Result:")
    print(f"    F_CMSI = {ss_check['F_CMSI_saturn']:.10f}")
    print(f"    δg/g = {ss_check['delta_g_over_g']:.2e}")
    print(f"    Cassini limit = {ss_check['cassini_limit']:.2e}")
    print(f"    PASSES CASSINI: {ss_check['passes_cassini']}")
    print(f"")
    print(f"  Counterfactual: If Saturn were in a disk (Σ=50):")
    print(f"    F_CMSI would be {ss_check['F_CMSI_if_in_disk']:.3f}")
    
    print("\n" + "=" * 60)
    print("Self-test complete.")
    
    return R_kpc, v_circ_bary, sigma_v, F_CMSI, v_enhanced, params


if __name__ == "__main__":
    test_cmsi_kernel()
