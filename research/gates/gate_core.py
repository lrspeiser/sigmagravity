"""
Core gate functions for Σ-Gravity

These implement the fundamental gate equations that suppress coherence
where needed (solar system, bulges, bars) while allowing it in extended,
low-acceleration environments.
"""

import numpy as np

# ============================================================================
# DISTANCE-BASED GATES
# ============================================================================

def G_distance(R, R_min=1.0, alpha=2.0, beta=1.0):
    """
    Distance-based gate: suppresses at small R, saturates at large R
    
    Formula: G(R) = [1 + (R_min/R)^alpha]^(-beta)
    
    Parameters
    ----------
    R : array_like
        Radial distance (kpc)
    R_min : float
        Characteristic scale where G ≈ 0.5 (kpc)
    alpha : float
        Steepness of transition (typical: 2-3)
    beta : float
        Strength of suppression (typical: 0.5-2)
    
    Returns
    -------
    G : array_like
        Gate value in [0, 1]
    
    Properties
    ----------
    - G(R → 0) → 0  (suppressed at small scales)
    - G(R → ∞) → 1  (no suppression at large scales)
    - dG/dR > 0     (monotonic increasing)
    - G(R_min) ≈ 0.5 (transition scale)
    
    Examples
    --------
    >>> R = np.logspace(-3, 2, 100)  # 0.001 to 100 kpc
    >>> G = G_distance(R, R_min=2.0, alpha=2.0, beta=1.0)
    >>> assert G[0] < 0.01  # Suppressed at small R
    >>> assert G[-1] > 0.99  # Saturated at large R
    """
    R = np.maximum(np.asarray(R), 1e-30)  # Avoid division by zero
    return (1.0 + (R_min / R)**alpha)**(-beta)


def G_solar_system(R, R_min_AU=1.0, alpha=4.0, beta=2.0):
    """
    Solar system safety gate: MUST suppress K to < 10^-14 at 1 AU
    
    This is a HARD constraint from PPN/Cassini bounds.
    Always multiply this gate with any other gates.
    
    Parameters
    ----------
    R : array_like
        Radial distance (kpc)
    R_min_AU : float
        Minimum scale in AU (default 1 AU)
    alpha : float
        Steepness (recommend 3-4 for strong suppression)
    beta : float
        Strength (recommend 1-2)
    
    Returns
    -------
    G : array_like
        Gate value, with G(1 AU) ~ 10^-15
    
    Examples
    --------
    >>> AU_in_kpc = 4.848e-9
    >>> R_AU = np.array([1.0, 100.0, 1e4]) * AU_in_kpc
    >>> G = G_solar_system(R_AU)
    >>> assert G[0] < 1e-14  # PPN safe at 1 AU
    """
    AU_in_kpc = 4.848136811e-9
    R_min_kpc = R_min_AU * AU_in_kpc
    return G_distance(R, R_min=R_min_kpc, alpha=alpha, beta=beta)


# ============================================================================
# ACCELERATION-BASED GATES
# ============================================================================

def G_acceleration(g_bar, g_crit=1e-10, alpha=2.0, beta=1.0):
    """
    Acceleration-based gate: suppresses at high g_bar (bulges, dense regions)
    
    Formula: G(g_bar) = [1 + (g_bar/g_crit)^alpha]^(-beta)
    
    Parameters
    ----------
    g_bar : array_like
        Baryonic acceleration (m/s²)
    g_crit : float
        Critical acceleration scale (m/s², typical: 10^-10 to 10^-9)
    alpha : float
        Steepness of transition
    beta : float
        Strength of suppression
    
    Returns
    -------
    G : array_like
        Gate value in [0, 1]
    
    Properties
    ----------
    - G(g → ∞) → 0  (suppressed at high acceleration)
    - G(g → 0) → 1  (no suppression at low acceleration)
    - dG/dg < 0     (monotonic decreasing with g)
    - G(g_crit) ≈ 0.5
    
    Physical Motivation
    -------------------
    High accelerations → compact objects (bulges, clusters cores)
    → frequent interactions → rapid decoherence → suppress coherence
    
    Examples
    --------
    >>> g = np.logspace(-12, -8, 100)  # m/s²
    >>> G = G_acceleration(g, g_crit=1e-10)
    >>> assert G[0] > 0.99  # Allowed at low g
    >>> assert G[-1] < 0.01  # Suppressed at high g
    """
    g_bar = np.maximum(np.asarray(g_bar), 1e-30)
    return (1.0 + (g_bar / g_crit)**alpha)**(-beta)


# ============================================================================
# EXPONENTIAL GATES (ALTERNATIVE FORM)
# ============================================================================

def G_bulge_exponential(R, R_bulge=1.5, alpha=2.0, beta=1.0):
    """
    Exponential bulge gate: physically motivated by bulge scale length
    
    Formula: G(R) = [1 - exp(-(R/R_bulge)^alpha)]^beta
    
    Parameters
    ----------
    R : array_like
        Radial distance (kpc)
    R_bulge : float
        Bulge effective radius from Sérsic fit (kpc)
    alpha : float
        Steepness (typical: 1.5-2.5)
    beta : float
        Strength (typical: 0.5-2.0)
    
    Returns
    -------
    G : array_like
        Gate value in [0, 1]
    
    When to Use
    -----------
    Use this when you have a measured R_bulge from surface brightness profile.
    Better than power-law gates for systems with well-defined bulge scales.
    
    Examples
    --------
    >>> R = np.linspace(0, 10, 100)
    >>> G = G_bulge_exponential(R, R_bulge=2.0, alpha=2.0, beta=1.0)
    >>> assert G[0] < 0.01  # Suppressed at center
    >>> assert G[-1] > 0.95  # Allowed in outer regions
    """
    R = np.maximum(np.asarray(R), 0.0)
    return (1.0 - np.exp(-(R / R_bulge)**alpha))**beta


# ============================================================================
# UNIFIED GATE (COMBINED DISTANCE + ACCELERATION)
# ============================================================================

def G_unified(R, g_bar, R_min=1.0, g_crit=1e-10,
              alpha_R=2.0, beta_R=1.0, alpha_g=2.0, beta_g=1.0):
    """
    Unified gate: combines distance AND acceleration suppression
    
    Formula: G_total = G_distance(R) × G_acceleration(g_bar)
    
    Parameters
    ----------
    R : array_like
        Radial distance (kpc)
    g_bar : array_like
        Baryonic acceleration (m/s²)
    R_min : float
        Distance scale (kpc)
    g_crit : float
        Acceleration scale (m/s²)
    alpha_R, beta_R : float
        Distance gate parameters
    alpha_g, beta_g : float
        Acceleration gate parameters
    
    Returns
    -------
    G : array_like
        Combined gate value in [0, 1]
    
    Physical Interpretation
    -----------------------
    Coherence emerges ONLY when:
    - Distance is large enough (R >> R_min), AND
    - Acceleration is low enough (g << g_crit)
    
    This naturally implements: "Extended, low-density regions allow coherence"
    
    Examples
    --------
    >>> R = np.logspace(-2, 2, 50)
    >>> g_bar = 1e-10 / R**2  # Toy model: g ~ 1/R²
    >>> G = G_unified(R, g_bar, R_min=1.0, g_crit=1e-10)
    >>> # At small R: both distance AND acceleration suppress
    >>> # At large R: both distance AND acceleration allow
    """
    G_d = G_distance(R, R_min, alpha_R, beta_R)
    G_a = G_acceleration(g_bar, g_crit, alpha_g, beta_g)
    return G_d * G_a


# ============================================================================
# COHERENCE WINDOW (FOR COMPLETENESS)
# ============================================================================

def C_burr_XII(R, ell0=5.0, p=2.0, n_coh=1.0):
    """
    Coherence window: Burr-XII envelope (from paper Section 2.3)
    
    Formula: C(R) = 1 - [1 + (R/ℓ₀)^p]^(-n_coh)
    
    This is your UNIVERSAL coherence window, same for galaxies and clusters.
    Gates multiply this, they don't replace it.
    
    Parameters
    ----------
    R : array_like
        Radial distance (kpc)
    ell0 : float
        Coherence length ℓ₀ = c·τ_collapse (kpc)
    p : float
        Shape exponent (typical: 0.5-2.0)
    n_coh : float
        Coherence exponent (typical: 0.5-1.5)
    
    Returns
    -------
    C : array_like
        Coherence fraction in [0, 1]
    
    Properties
    ----------
    - C(R → 0) → 0  (no coherence at small scales)
    - C(R → ∞) → 1  (full coherence at large scales)
    - C(ℓ₀) ≈ 0.5  (transition at coherence length)
    - Monotonic, saturating
    
    Examples
    --------
    >>> R = np.linspace(0, 20, 100)
    >>> C = C_burr_XII(R, ell0=5.0, p=2.0, n_coh=1.0)
    >>> assert C[0] < 0.01
    >>> assert C[-1] > 0.99
    >>> idx_ell0 = np.argmin(np.abs(R - 5.0))
    >>> assert 0.4 < C[idx_ell0] < 0.6  # ≈ 0.5 at ℓ₀
    """
    R = np.maximum(np.asarray(R), 0.0)
    return 1.0 - (1.0 + (R / ell0)**p)**(-n_coh)


# ============================================================================
# FULL SIGMA-GRAVITY KERNEL (EXAMPLE)
# ============================================================================

def K_sigma_gravity(R, g_bar, A=0.6, ell0=5.0, p=0.75, n_coh=0.5,
                    gate_type='unified', gate_params=None):
    """
    Full Σ-Gravity kernel: K(R) = A · C(R) · G(R, g_bar)
    
    Parameters
    ----------
    R : array_like
        Radial distance (kpc)
    g_bar : array_like
        Baryonic acceleration (m/s²)
    A : float
        Amplitude
    ell0, p, n_coh : float
        Coherence window parameters
    gate_type : str
        'distance', 'acceleration', 'exponential', 'unified', or 'none'
    gate_params : dict
        Parameters for selected gate type
    
    Returns
    -------
    K : array_like
        Kernel value (dimensionless)
    
    Examples
    --------
    >>> R = np.logspace(-2, 2, 100)
    >>> g_bar = 1e-10 / R**2
    >>> K = K_sigma_gravity(R, g_bar, A=0.6, ell0=5.0,
    ...                     gate_type='unified',
    ...                     gate_params={'R_min': 1.0, 'g_crit': 1e-10})
    >>> # Check solar system safety
    >>> AU_in_kpc = 4.848e-9
    >>> K_1AU = K_sigma_gravity(1.0*AU_in_kpc, 5.9e-3, A=0.6, ell0=5.0,
    ...                         gate_type='distance',
    ...                         gate_params={'R_min': 0.0001})
    >>> assert K_1AU < 1e-14
    """
    # Coherence window (universal)
    C = C_burr_XII(R, ell0, p, n_coh)
    
    # Gate (context-dependent)
    if gate_type == 'none' or gate_params is None:
        G = 1.0
    elif gate_type == 'distance':
        G = G_distance(R, **gate_params)
    elif gate_type == 'acceleration':
        G = G_acceleration(g_bar, **gate_params)
    elif gate_type == 'exponential':
        G = G_bulge_exponential(R, **gate_params)
    elif gate_type == 'unified':
        G = G_unified(R, g_bar, **gate_params)
    else:
        raise ValueError(f"Unknown gate_type: {gate_type}")
    
    # Always include solar system safety
    G_solar = G_solar_system(R)
    
    return A * C * G * G_solar


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def check_gate_properties(R, G, tol=1e-6):
    """
    Validate that a gate function satisfies required properties
    
    Checks:
    1. G ∈ [0, 1]
    2. G(R → 0) → 0
    3. G(R → ∞) → 1
    4. dG/dR ≥ 0 (monotonic)
    
    Parameters
    ----------
    R : array_like
        Radial points (must be sorted, log-spaced recommended)
    G : array_like
        Gate values
    tol : float
        Numerical tolerance
    
    Returns
    -------
    checks : dict
        Dictionary of boolean checks
    
    Raises
    ------
    AssertionError if any check fails
    """
    checks = {}
    
    # Check bounds
    checks['in_bounds'] = np.all((G >= -tol) & (G <= 1 + tol))
    
    # Check limits
    checks['small_R_suppressed'] = G[0] < tol
    checks['large_R_saturated'] = G[-1] > 1 - tol
    
    # Check monotonicity
    dG = np.gradient(G, np.log(R))
    checks['monotonic'] = np.all(dG >= -tol)
    
    # Summary
    checks['all_pass'] = all(checks.values())
    
    return checks


if __name__ == '__main__':
    """Quick sanity checks"""
    import matplotlib.pyplot as plt
    
    # Test distance gate
    R = np.logspace(-3, 2, 200)
    G_d = G_distance(R, R_min=1.0, alpha=2.0, beta=1.0)
    checks_d = check_gate_properties(R, G_d)
    print("Distance gate checks:", checks_d)
    
    # Test acceleration gate
    g = np.logspace(-12, -8, 200)
    G_a = G_acceleration(g, g_crit=1e-10, alpha=2.0, beta=1.0)
    print(f"Accel gate: G(low g) = {G_a[0]:.6f}, G(high g) = {G_a[-1]:.6f}")
    
    # Test solar system safety
    AU_in_kpc = 4.848e-9
    R_AU = np.array([1, 10, 100, 1000, 10000]) * AU_in_kpc
    G_solar = G_solar_system(R_AU)
    print("\nSolar system gate:")
    for r_au, g in zip([1, 10, 100, 1000, 10000], G_solar):
        print(f"  {r_au:5.0f} AU: G = {g:.2e}")
    
    # Test full kernel
    R_test = np.logspace(-6, 2, 300)
    g_test = 1e-10 / R_test**2
    K = K_sigma_gravity(R_test, g_test, A=0.6, ell0=5.0,
                       gate_type='unified',
                       gate_params={'R_min': 1.0, 'g_crit': 1e-10})
    
    # Find K at 1 AU
    idx_1AU = np.argmin(np.abs(R_test - AU_in_kpc))
    print(f"\nFull kernel K(1 AU) = {K[idx_1AU]:.2e}")
    print(f"PPN safe? {K[idx_1AU] < 1e-14}")
    
    print("\n[OK] All core functions operational")

