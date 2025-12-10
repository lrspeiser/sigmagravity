"""
Three First-Principles Approaches to Derive Sigma-Gravity Field Theory
========================================================================

Each approach starts from a different physical picture and derives:
1. The field equation for φ (coherence field)
2. The effective potential V(φ)
3. The connection to your working K(R) phenomenology

Approach A: GRAVITATIONAL WELL MODEL
    "Gravity is a well that generates more coherence"
    φ represents coherence accumulation driven by matter density

Approach B: GRAVITATIONAL WAVE AMPLIFICATION MODEL
    "Gravity is a wave that amplifies in certain situations"
    φ represents the amplitude of a scalar graviton mode

Approach C: QUANTUM DECOHERENCE FIELD MODEL
    "Environment-dependent decoherence controls effective gravity"
    φ represents the coherence order parameter
"""

import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import minimize
from typing import Tuple, Callable, Dict
from dataclasses import dataclass
import matplotlib.pyplot as plt


# ==============================================================================
# APPROACH A: GRAVITATIONAL WELL MODEL
# ==============================================================================
# Physical Picture:
# -----------------
# Matter creates a "well" in coherence space. Gravitational wave modes
# accumulate coherence φ in proportion to:
#   1. Local matter density ρ (deeper well = more accumulation)
#   2. Time spent in well (extended systems = longer time)
#   3. Velocity dispersion (hot systems = less accumulation)
#
# This is analogous to how potential wells trap particles, but here
# the "particle" is gravitational wave coherence.
#
# Field Equation Derivation:
# ---------------------------
# Start with coherence accumulation rate:
#   dφ/dt ∝ ρ(r) - φ/τ_decohere
#
# where τ_decohere depends on environment (smaller in hot systems).
#
# For steady-state: dφ/dt = 0
#   φ(r) = τ_decohere(r) · ρ(r)
#
# In field theory language, this is Klein-Gordon with source:
#   ∇²φ - m_eff²(ρ) φ = -4πG ρ
#
# where m_eff² ~ 1/τ_decohere encodes environment-dependent screening.
#
# Effective gravity:
#   g_eff = g_Newtonian · [1 + α·φ/M_Pl]
#
# Connection to K(R):
#   K(R) ≈ α · φ(R) / M_Pl
# ==============================================================================

@dataclass
class GravitationalWellParams:
    """Parameters for gravitational well model."""
    alpha: float = 0.5  # Coupling strength (dimensionless)
    tau_0: float = 1.0  # Base decoherence time [Gyr]
    rho_crit: float = 1e-21  # Critical density [kg/m³]
    beta: float = 2.0  # Velocity dispersion sensitivity
    sigma_ref: float = 30.0  # Reference velocity dispersion [km/s]
    M_Pl_eff: float = 1.0  # Effective Planck mass scale (normalized)


def gravitational_well_field_equation(
    r: np.ndarray,
    rho_bar: np.ndarray,
    sigma_v: float,
    params: GravitationalWellParams
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve field equation for gravitational well model.
    
    ∇²φ - m_eff²(ρ,σ_v) φ = -4πG ρ
    
    where m_eff² = 1/[τ_0 · (σ_ref/σ_v)^β · (1 + ρ/ρ_crit)]
    
    Returns:
        phi: Coherence field [dimensionless]
        K_R: Boost factor K(R) = α·φ/M_Pl
    """
    # Effective mass (inverse decoherence time)
    # Smaller in cold systems (small σ_v), larger in hot systems
    # Also decreases in dense regions (ρ large) - longer coherence
    tau_eff = params.tau_0 * (params.sigma_ref / sigma_v)**params.beta
    m_eff_sq = 1.0 / (tau_eff * (1.0 + rho_bar / params.rho_crit))
    
    # For spherical symmetry: ∇²φ = d²φ/dr² + (2/r) dφ/dr
    # Rewrite as system: φ' = ψ, ψ' = -2ψ/r + m²φ - 4πG·ρ
    
    def field_ode(y, r_val):
        """ODE system for φ and φ'."""
        phi, psi = y
        
        # Interpolate density and m_eff² at this radius
        rho_interp = np.interp(r_val, r, rho_bar, left=0, right=0)
        m2_interp = np.interp(r_val, r, m_eff_sq)
        
        # Source term: -4πG·ρ (normalized units)
        G_norm = 1.0  # Set units where 4πG = 1
        source = -G_norm * rho_interp
        
        dphi_dr = psi
        dpsi_dr = -2*psi/r_val + m2_interp*phi + source
        
        return [dphi_dr, dpsi_dr]
    
    # Boundary conditions: φ(0) = finite, φ'(0) = 0 (regularity)
    # Integrate outward from small radius
    r_min = r[0] if r[0] > 0 else 0.01
    r_max = r[-1]
    r_solve = np.linspace(r_min, r_max, len(r))
    
    # Initial conditions: φ(r_min) ~ ∫ρ, φ'(r_min) ~ 0
    G_norm = 1.0  # Same normalization as in ODE
    rho_center = np.interp(r_min, r, rho_bar)
    phi_0 = G_norm * rho_center * r_min**2 / 6.0  # Taylor expansion
    psi_0 = 0.0
    
    sol = odeint(field_ode, [phi_0, psi_0], r_solve)
    phi_sol = sol[:, 0]
    
    # Interpolate back to original r grid
    phi = np.interp(r, r_solve, phi_sol)
    
    # Boost factor
    K_R = params.alpha * phi / params.M_Pl_eff
    
    return phi, K_R


def infer_potential_from_well_model(
    phi: np.ndarray,
    rho: np.ndarray,
    params: GravitationalWellParams
) -> Callable[[float], float]:
    """
    Given solution φ(r), infer the effective potential V(φ).
    
    From field equation: □φ = dV_eff/dφ + coupling to matter
    
    This works backwards: given φ and ρ, what V(φ) is consistent?
    """
    # Compute □φ numerically from solution
    # Then: dV/dφ ≈ □φ - coupling·ρ
    
    # For now, return analytic form consistent with well model
    # V_eff(φ) = (1/2) m_eff² φ² + V_0
    
    def V_effective(phi_val):
        """Effective potential (quadratic well)."""
        m_eff_typical = 1.0 / params.tau_0
        return 0.5 * m_eff_typical**2 * phi_val**2
    
    return V_effective


# ==============================================================================
# APPROACH B: GRAVITATIONAL WAVE AMPLIFICATION MODEL
# ==============================================================================
# Physical Picture:
# -----------------
# Gravitational waves (or scalar graviton modes) propagate through matter.
# In extended, cold systems, they experience coherent amplification via
# stimulated emission / parametric resonance.
#
# This is analogous to how light is amplified in a laser cavity, but here
# the "cavity" is the matter distribution and orbital dynamics.
#
# Field Equation Derivation:
# ---------------------------
# Start with wave equation for scalar mode:
#   □φ + m² φ = J(ρ, v)
#
# where J is a source/gain term that depends on:
#   - Matter density ρ (provides energy for amplification)
#   - Orbital velocity structure v (provides phase matching)
#
# Gain is maximized when:
#   1. Wavelength λ_gw ~ orbital scale (resonance)
#   2. Velocity dispersion small (coherent phase matching)
#   3. Extended geometry (long interaction length)
#
# Effective gravity:
#   g_eff = g_bar · [1 + β·|φ|²]
#
# where |φ|² is intensity (not amplitude) of the amplified mode.
#
# Connection to K(R):
#   K(R) ≈ β · |φ(R)|²
# ==============================================================================

@dataclass
class WaveAmplificationParams:
    """Parameters for wave amplification model."""
    beta: float = 1.0  # Coupling to intensity
    gain_0: float = 0.1  # Base gain coefficient
    lambda_res: float = 5.0  # Resonant wavelength [kpc]
    Delta_lambda: float = 2.0  # Resonance width [kpc]
    sigma_ref: float = 30.0  # Reference dispersion [km/s]
    gamma: float = 2.0  # Gain suppression by velocity dispersion


def wave_amplification_field_equation(
    r: np.ndarray,
    rho_bar: np.ndarray,
    v_circ: np.ndarray,
    sigma_v: float,
    params: WaveAmplificationParams
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve wave equation with gain:
    
    □φ + m² φ = g(r) φ
    
    where g(r) = gain_0 · ρ/ρ_ref · resonance(λ_orbital) · (σ_ref/σ_v)^γ
    
    Resonance condition: λ_orbital(r) ≈ 2πr matches λ_res
    
    Returns:
        phi: Wave amplitude [dimensionless]
        K_R: Boost factor K(R) = β·|φ|²
    """
    # Orbital wavelength at each radius
    lambda_orbital = 2 * np.pi * r
    
    # Resonance factor (Lorentzian)
    resonance = params.Delta_lambda**2 / (
        (lambda_orbital - params.lambda_res)**2 + params.Delta_lambda**2
    )
    
    # Gain coefficient (density × resonance × coherence factor)
    rho_ref = np.mean(rho_bar[rho_bar > 0])
    coherence_factor = (params.sigma_ref / sigma_v)**params.gamma
    gain = params.gain_0 * (rho_bar / rho_ref) * resonance * coherence_factor
    
    # Wave equation: φ'' + (2/r)φ' + gain·φ = 0
    # This is exponential growth/decay depending on sign of gain
    
    def wave_ode(y, r_val):
        """ODE for wave amplitude."""
        phi, psi = y  # φ and dφ/dr
        
        # Interpolate gain
        gain_interp = np.interp(r_val, r, gain)
        
        dphi_dr = psi
        dpsi_dr = -2*psi/r_val - gain_interp*phi  # Negative gain -> growth
        
        return [dphi_dr, dpsi_dr]
    
    # Boundary condition: φ(r_min) = small seed amplitude
    r_min = r[0] if r[0] > 0 else 0.1
    r_solve = r[r >= r_min]
    
    phi_seed = 0.01  # Vacuum fluctuation
    psi_seed = 0.0
    
    sol = odeint(wave_ode, [phi_seed, psi_seed], r_solve)
    phi_sol = sol[:, 0]
    
    # Map back to full grid
    phi = np.zeros_like(r)
    phi[r >= r_min] = phi_sol
    
    # Boost factor: intensity |φ|²
    K_R = params.beta * np.abs(phi)**2
    
    return phi, K_R


def infer_potential_from_wave_model(
    phi: np.ndarray,
    gain: np.ndarray,
    params: WaveAmplificationParams
) -> Callable[[float], float]:
    """
    Infer potential V(φ) consistent with wave amplification.
    
    Effective action includes kinetic + potential + gain:
    S = ∫ [(∇φ)²/2 - V(φ) - g(x)·φ²/2] d³x
    
    For positive gain g > 0, this is unstable (tachyonic) region
    where field grows. For g < 0, field decays.
    
    Effective potential:
    V_eff(φ) = V₀ - <g>·φ²/2
    """
    gain_avg = np.mean(gain)
    
    def V_effective(phi_val):
        """Effective potential with tachyonic mass."""
        V_0 = 1.0
        m_tach_sq = -gain_avg  # Negative mass² -> instability
        return V_0 + 0.5 * m_tach_sq * phi_val**2
    
    return V_effective


# ==============================================================================
# APPROACH C: QUANTUM DECOHERENCE FIELD MODEL
# ==============================================================================
# Physical Picture:
# -----------------
# Gravitational interaction strength is controlled by a "coherence order
# parameter" φ that interpolates between:
#   - φ = 0: Classical gravity (collapsed wavefunction, no coherence)
#   - φ = 1: Quantum-enhanced gravity (coherent superposition)
#
# The field φ obeys dynamics similar to order parameters in phase transitions:
#   - Driven toward φ=1 in cold, extended systems (favors coherence)
#   - Relaxes toward φ=0 in hot, compact systems (favors decoherence)
#
# This is analogous to superconductivity, where Cooper pairs form below T_c.
#
# Field Equation Derivation:
# ---------------------------
# Landau-Ginzburg style effective action:
#   S = ∫ [α(T)|∇φ|² + β(T)φ² + γ φ⁴] d³x
#
# where "temperature" T ∝ σ_v² / (density × length_scale)
#
# For T < T_c: β < 0 → spontaneous symmetry breaking, φ ≠ 0
# For T > T_c: β > 0 → φ = 0 (decoherence)
#
# Equation of motion:
#   ∇²φ = -2βφ - 4γφ³
#
# Effective gravity:
#   g_eff = g_bar · [1 + φ²]
#
# Connection to K(R):
#   K(R) = φ(R)²
# ==============================================================================

@dataclass
class DecoherenceFieldParams:
    """Parameters for decoherence field model."""
    alpha: float = 1.0  # Gradient energy coefficient
    beta_0: float = -0.1  # Quadratic term at T=0
    gamma: float = 0.01  # Quartic term (self-interaction)
    T_c: float = 50.0  # Critical "temperature" [km/s]²
    rho_scale: float = 1e-21  # Density scale [kg/m³]
    L_scale: float = 5.0  # Length scale [kpc]


def decoherence_field_equation(
    r: np.ndarray,
    rho_bar: np.ndarray,
    sigma_v: float,
    params: DecoherenceFieldParams
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve Landau-Ginzburg equation for coherence order parameter.
    
    ∇²φ = -2β(T)φ - 4γφ³
    
    where β(T) = β₀·[1 - (T/T_c)²]
    and T_eff² ∝ σ_v² / (ρ·L)
    
    Returns:
        phi: Order parameter φ ∈ [0, 1]
        K_R: Boost factor K(R) = φ²
    """
    # Effective temperature at each radius
    # High σ_v, low ρ, small L → high T → φ = 0
    # Low σ_v, high ρ, large L → low T → φ ≠ 0
    L_eff = r  # Radial scale
    T_eff_sq = (sigma_v**2) / ((rho_bar / params.rho_scale) * (L_eff / params.L_scale) + 1e-10)
    
    # β parameter (negative below T_c, positive above)
    beta_eff = params.beta_0 * (1.0 - T_eff_sq / params.T_c**2)
    
    # Field equation: ∇²φ = -2β·φ - 4γ·φ³
    # In spherical symmetry: d²φ/dr² + (2/r)dφ/dr = -2β·φ - 4γ·φ³
    
    def lg_ode(y, r_val):
        """Landau-Ginzburg ODE."""
        phi, psi = y
        
        # Interpolate β
        beta_interp = np.interp(r_val, r, beta_eff)
        
        dphi_dr = psi
        dpsi_dr = -2*psi/r_val - 2*beta_interp*phi - 4*params.gamma*phi**3
        
        return [dphi_dr, dpsi_dr]
    
    # Boundary condition: φ(r_min) ≈ equilibrium value
    # At origin, if T < T_c: φ ~ sqrt(-β/2γ)
    r_min = r[0] if r[0] > 0 else 0.1
    beta_center = np.interp(r_min, r, beta_eff)
    
    if beta_center < 0:
        phi_eq = np.sqrt(-beta_center / (2*params.gamma))
    else:
        phi_eq = 0.0
    
    r_solve = r[r >= r_min]
    
    sol = odeint(lg_ode, [phi_eq, 0.0], r_solve)
    phi_sol = sol[:, 0]
    
    # Clip to physical range [0, 1]
    phi_sol = np.clip(phi_sol, 0.0, 1.0)
    
    phi = np.zeros_like(r)
    phi[r >= r_min] = phi_sol
    
    # Boost factor
    K_R = phi**2
    
    return phi, K_R


def infer_potential_from_decoherence_model(
    phi: np.ndarray,
    T_eff: np.ndarray,
    params: DecoherenceFieldParams
) -> Callable[[float], float]:
    """
    Effective potential for decoherence field.
    
    V(φ) = β(T)φ² + γφ⁴
    
    For T < T_c: β < 0 → double-well, minima at φ = ±√(-β/2γ)
    For T > T_c: β > 0 → single well at φ = 0
    """
    beta_avg = params.beta_0 * (1.0 - np.mean(T_eff**2) / params.T_c**2)
    
    def V_effective(phi_val):
        """Landau-Ginzburg potential."""
        return beta_avg * phi_val**2 + params.gamma * phi_val**4
    
    return V_effective


# ==============================================================================
# COMPARISON WITH SIGMA-GRAVITY PHENOMENOLOGY
# ==============================================================================

def compare_with_sigma_gravity(
    r: np.ndarray,
    K_sigma: np.ndarray,
    K_model: np.ndarray,
    model_name: str
) -> Dict[str, float]:
    """
    Compare model-derived K(R) with fitted Sigma-Gravity K(R).
    
    Returns metrics:
        - rms_error
        - max_error
        - correlation
    """
    # RMS fractional error
    rms = np.sqrt(np.mean(((K_model - K_sigma) / (K_sigma + 1e-10))**2))
    
    # Max error
    max_err = np.max(np.abs((K_model - K_sigma) / (K_sigma + 1e-10)))
    
    # Pearson correlation
    corr = np.corrcoef(K_sigma, K_model)[0, 1]
    
    return {
        'model': model_name,
        'rms_error': rms,
        'max_error': max_err,
        'correlation': corr
    }


# ==============================================================================
# MAIN: TEST ALL THREE APPROACHES ON SYNTHETIC GALAXY
# ==============================================================================

if __name__ == "__main__":
    print("="*80)
    print("FIRST-PRINCIPLES DERIVATIONS OF SIGMA-GRAVITY")
    print("="*80)
    print("\nTesting three physical mechanisms:")
    print("  A. Gravitational Well (coherence accumulation)")
    print("  B. Wave Amplification (parametric resonance)")
    print("  C. Decoherence Field (order parameter)")
    print()
    
    # Create synthetic galaxy
    r = np.linspace(0.5, 30, 300)
    R_d = 3.0  # Disk scale [kpc]
    Sigma_0 = 1e9  # Surface density [M☉/kpc²]
    Sigma = Sigma_0 * np.exp(-r / R_d)
    h_z = 0.3  # Scale height [kpc]
    rho_bar = Sigma / (2 * h_z) * 1e-24  # Convert to kg/m³
    
    v_max = 200.0  # km/s
    v_circ = v_max * np.sqrt(1 - np.exp(-r / R_d))
    sigma_v = 30.0  # Velocity dispersion [km/s]
    
    print(f"Synthetic galaxy:")
    print(f"  R_d = {R_d:.1f} kpc")
    print(f"  v_max = {v_max:.0f} km/s")
    print(f"  σ_v = {sigma_v:.0f} km/s")
    print()
    
    # Generate "true" Sigma-Gravity K(R) as reference
    # Using Burr-XII form from your coherence_models.py
    ell0, p, n_coh, A = 5.0, 1.0, 1.0, 0.5
    x = r / ell0
    C_sigma = 1.0 - (1.0 + x**p)**(-n_coh)
    K_sigma = A * C_sigma
    
    print("Reference Sigma-Gravity K(R) generated (Burr-XII)")
    print(f"  Parameters: A={A}, ℓ₀={ell0} kpc, p={p}, n_coh={n_coh}")
    print()
    
    # -------------------------------------------------------------------------
    # Test Approach A: Gravitational Well
    # -------------------------------------------------------------------------
    print("-"*80)
    print("APPROACH A: GRAVITATIONAL WELL MODEL")
    print("-"*80)
    
    params_well = GravitationalWellParams(
        alpha=0.5, tau_0=1.0, rho_crit=1e-21,
        beta=2.0, sigma_ref=30.0
    )
    
    phi_well, K_well = gravitational_well_field_equation(
        r, rho_bar, sigma_v, params_well
    )
    
    metrics_well = compare_with_sigma_gravity(r, K_sigma, K_well, "Gravitational Well")
    
    print(f"Results:")
    print(f"  RMS error: {metrics_well['rms_error']:.4f}")
    print(f"  Max error: {metrics_well['max_error']:.4f}")
    print(f"  Correlation: {metrics_well['correlation']:.4f}")
    print()
    
    V_well = infer_potential_from_well_model(phi_well, rho_bar, params_well)
    phi_test = np.linspace(0, np.max(phi_well), 100)
    print(f"Inferred potential: V(φ) ∝ φ² (harmonic well)")
    print()
    
    # -------------------------------------------------------------------------
    # Test Approach B: Wave Amplification
    # -------------------------------------------------------------------------
    print("-"*80)
    print("APPROACH B: WAVE AMPLIFICATION MODEL")
    print("-"*80)
    
    params_wave = WaveAmplificationParams(
        beta=1.0, gain_0=0.1, lambda_res=15.0,
        Delta_lambda=5.0, sigma_ref=30.0, gamma=2.0
    )
    
    phi_wave, K_wave = wave_amplification_field_equation(
        r, rho_bar, v_circ, sigma_v, params_wave
    )
    
    metrics_wave = compare_with_sigma_gravity(r, K_sigma, K_wave, "Wave Amplification")
    
    print(f"Results:")
    print(f"  RMS error: {metrics_wave['rms_error']:.4f}")
    print(f"  Max error: {metrics_wave['max_error']:.4f}")
    print(f"  Correlation: {metrics_wave['correlation']:.4f}")
    print()
    
    # Resonance condition
    lambda_orbital = 2 * np.pi * r
    idx_res = np.argmin(np.abs(lambda_orbital - params_wave.lambda_res))
    print(f"Resonance at R ~ {r[idx_res]:.1f} kpc (λ_orbital ≈ λ_res)")
    print()
    
    # -------------------------------------------------------------------------
    # Test Approach C: Decoherence Field
    # -------------------------------------------------------------------------
    print("-"*80)
    print("APPROACH C: DECOHERENCE FIELD MODEL")
    print("-"*80)
    
    params_decoh = DecoherenceFieldParams(
        alpha=1.0, beta_0=-0.1, gamma=0.01,
        T_c=50.0, rho_scale=1e-21, L_scale=5.0
    )
    
    phi_decoh, K_decoh = decoherence_field_equation(
        r, rho_bar, sigma_v, params_decoh
    )
    
    metrics_decoh = compare_with_sigma_gravity(r, K_sigma, K_decoh, "Decoherence Field")
    
    print(f"Results:")
    print(f"  RMS error: {metrics_decoh['rms_error']:.4f}")
    print(f"  Max error: {metrics_decoh['max_error']:.4f}")
    print(f"  Correlation: {metrics_decoh['correlation']:.4f}")
    print()
    
    # Check phase transition
    T_eff = sigma_v**2 / ((rho_bar / params_decoh.rho_scale) * (r / params_decoh.L_scale))
    print(f"Effective temperature: T_eff ~ {np.mean(np.sqrt(T_eff)):.1f} km/s (avg)")
    print(f"Critical temperature: T_c = {params_decoh.T_c:.1f} km/s")
    
    if np.mean(np.sqrt(T_eff)) < params_decoh.T_c:
        print("→ System is in COHERENT phase (φ ≠ 0)")
    else:
        print("→ System is in INCOHERENT phase (φ = 0)")
    print()
    
    # -------------------------------------------------------------------------
    # Summary and Visualization
    # -------------------------------------------------------------------------
    print("="*80)
    print("SUMMARY")
    print("="*80)
    
    results = [metrics_well, metrics_wave, metrics_decoh]
    results_sorted = sorted(results, key=lambda x: x['rms_error'])
    
    print("\nBest fit to Sigma-Gravity phenomenology:")
    for i, res in enumerate(results_sorted, 1):
        print(f"{i}. {res['model']}: RMS = {res['rms_error']:.4f}, corr = {res['correlation']:.4f}")
    
    print("\n" + "="*80)
    print("PHYSICAL INTERPRETATION")
    print("="*80)
    
    print("""
Each model provides a different lens on the SAME physics:

A. GRAVITATIONAL WELL
   - Coherence "trapped" by matter density
   - φ represents accumulated phase correlation
   - Natural in static systems (galaxies, clusters)
   - Field equation: ∇²φ = ρ/τ_decohere

B. WAVE AMPLIFICATION
   - Gravitational modes amplified via resonance
   - φ represents wave amplitude (intensity = |φ|²)
   - Natural in rotating systems (disks)
   - Field equation: □φ = gain(λ,ρ,σ_v)·φ

C. DECOHERENCE FIELD
   - Order parameter for quantum→classical transition
   - φ ∈ [0,1] represents coherence fraction
   - Natural for phase-transition-like behavior
   - Field equation: ∇²φ = -dV_eff/dφ (Landau-Ginzburg)

NEXT STEPS:
-----------
1. Fit each model to your REAL SPARC data
2. Extract best-fit parameters for each approach
3. Compare effective potentials V(φ)
4. Test cosmological evolution φ(z)
5. See which physical picture predicts NEW phenomena
    """)
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: K(R) comparison
    ax = axes[0, 0]
    ax.plot(r, K_sigma, 'k-', lw=3, label='Sigma-Gravity (ref)')
    ax.plot(r, K_well, 'r--', lw=2, label=f'Well (RMS={metrics_well["rms_error"]:.3f})')
    ax.plot(r, K_wave, 'b--', lw=2, label=f'Wave (RMS={metrics_wave["rms_error"]:.3f})')
    ax.plot(r, K_decoh, 'g--', lw=2, label=f'Decoh (RMS={metrics_decoh["rms_error"]:.3f})')
    ax.set_xlabel('Radius [kpc]')
    ax.set_ylabel('Boost Factor K(R)')
    ax.set_title('Model Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Field profiles
    ax = axes[0, 1]
    ax.plot(r, phi_well / np.max(phi_well), 'r-', lw=2, label='φ_well (norm)')
    ax.plot(r, phi_wave / np.max(phi_wave), 'b-', lw=2, label='φ_wave (norm)')
    ax.plot(r, phi_decoh, 'g-', lw=2, label='φ_decoh')
    ax.set_xlabel('Radius [kpc]')
    ax.set_ylabel('Field φ (normalized)')
    ax.set_title('Field Profiles')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Residuals
    ax = axes[1, 0]
    ax.plot(r, K_well - K_sigma, 'r-', lw=2, label='Well')
    ax.plot(r, K_wave - K_sigma, 'b-', lw=2, label='Wave')
    ax.plot(r, K_decoh - K_sigma, 'g-', lw=2, label='Decoh')
    ax.axhline(0, color='k', ls='--', lw=1)
    ax.set_xlabel('Radius [kpc]')
    ax.set_ylabel('Residual ΔK(R)')
    ax.set_title('Residuals vs Sigma-Gravity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Potentials
    ax = axes[1, 1]
    phi_range = np.linspace(0, 1, 100)
    
    # Well: V ∝ φ²
    V_well_vals = 0.5 * phi_range**2
    ax.plot(phi_range, V_well_vals, 'r-', lw=2, label='Well: V∝φ²')
    
    # Wave: V ∝ -φ² (tachyonic)
    V_wave_vals = -0.5 * phi_range**2
    ax.plot(phi_range, V_wave_vals, 'b-', lw=2, label='Wave: V∝-φ²')
    
    # Decoh: V ∝ -φ² + φ⁴ (double-well)
    V_decoh_vals = -phi_range**2 + phi_range**4
    ax.plot(phi_range, V_decoh_vals, 'g-', lw=2, label='Decoh: V∝-φ²+φ⁴')
    
    ax.set_xlabel('Field φ')
    ax.set_ylabel('Potential V(φ)')
    ax.set_title('Effective Potentials')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', ls='--', lw=1)
    
    plt.tight_layout()
    
    outpath = 'coherence-field-theory/outputs/first_principles_comparison.png'
    import os
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved: {outpath}")
    print("="*80)
