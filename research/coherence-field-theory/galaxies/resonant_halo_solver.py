"""
Resonant Halo Solver: Wave Amplification (Approach B)
======================================================

Field equation with environment-dependent gain from disk dynamics:

    ∇²φ - μ²(r)φ = β ρ_b(r)

where μ²(r) = m₀² - g(r) and g(r) is the **resonant gain function**.

Physical picture:
-----------------
Cold, rotating disks act as GAIN MEDIA for scalar gravitational modes.
Where disk shear + orbital resonance align, the field amplifies coherently.
Outside resonant zones or in hot systems, the field is massive and decays.

Key advantages:
- Cosmology safe: g→0 in homogeneous background (no disk structure)
- PPN safe: g→0 in Solar System (no cold disk)
- Predictive: Global parameters, per-galaxy only baryonic profiles
- Testable: Predicts R_res ~ ξ R_disk, morphology dependence

Gain function g(r) has THREE gates (all smooth, dimensionless):
1. Coldness (Toomre Q): Unstable disks amplify
2. Dispersion (σ_v): Hot systems suppress
3. Resonance (cavity): 2πr ~ m λ_φ (standing wave condition)
"""

import numpy as np
from scipy.integrate import solve_bvp
from typing import Tuple, Dict
from dataclasses import dataclass


@dataclass
class ResonantParams:
    """
    Parameters for wave amplification / resonant gain model.
    
    GLOBAL (shared across galaxies):
    ---------------------------------
    m0 : Baseline scalar mass [kpc⁻¹]
        Sets decay length ~1/m₀ outside resonant zones
        
    R_coh : Coherence length [kpc]
        Sets gain amplitude via g₀ = α/R_coh²
        
    alpha : Dimensionless gain strength
        Typical ~O(1), controls how strong amplification is
        
    lambda_phi : Scalar wavelength [kpc]
        Standing wave condition: 2πr ~ m λ_φ
        
    Q_c : Toomre threshold
        Disks with Q < Q_c amplify (typical 1.5)
        
    Delta_Q : Coldness gate width
        Softness of Q-threshold (typical 0.2)
        
    sigma_c : Critical velocity dispersion [km/s]
        Suppression scale (typical 30 km/s)
        
    sigma_m : Resonance width in mode space
        Typical 0.2-0.3 (dimensionless)
        
    m_max : Number of resonant modes
        Usually 2-3 sufficient (m=1,2,3)
        
    beta : Baryon coupling strength
        From your existing field-driven framework
        
    PER-GALAXY (baryonic observables only):
    ----------------------------------------
    Sigma_b(r) : Surface density profile
    sigma_v(r) : Velocity dispersion
    v_c(r) : Circular velocity (or Omega(r))
    """
    # Global parameters
    m0: float = 0.01  # kpc⁻¹
    R_coh: float = 5.0  # kpc
    alpha: float = 1.0  # dimensionless
    lambda_phi: float = 6.0  # kpc
    Q_c: float = 1.5
    Delta_Q: float = 0.2
    sigma_c: float = 30.0  # km/s
    sigma_m: float = 0.25
    m_max: int = 2
    beta: float = 1.0  # baryon coupling
    lambda_4: float = 0.5  # saturation (φ⁴ term) - INCREASED for stability


# ==============================================================================
# GAIN FUNCTION: g(r) with three gates
# ==============================================================================

def kappa_from_Omega(r: np.ndarray, Omega: np.ndarray, 
                     dlnOm_dlnr: np.ndarray) -> np.ndarray:
    """
    Epicyclic frequency κ from angular frequency Ω.
    
    κ = √2 Ω √(1 + d ln Ω / d ln r)
    
    Notes:
    ------
    For real galaxies, occasionally dlnOm/dlnr < -1 at inner/outer edges
    due to noise or data artifacts. We clip to avoid NaNs.
    """
    discriminant = np.maximum(1.0 + dlnOm_dlnr, 0.0)  # Clip negative values
    return np.sqrt(2.0) * Omega * np.sqrt(discriminant)


def toomre_Q(r: np.ndarray, Sigma_b: np.ndarray, sigma_v: np.ndarray,
             Omega: np.ndarray, dlnOm_dlnr: np.ndarray,
             G: float = 4.30091e-6) -> np.ndarray:
    """
    Toomre stability parameter.
    
    Q = (κ σ_v) / (π G Σ_b)
    
    Q < 1: gravitationally unstable (amplifies)
    Q > 1: stable (suppresses)
    
    Parameters:
    -----------
    G : Gravitational constant [kpc km² s⁻² M☉⁻¹]
    """
    kappa = kappa_from_Omega(r, Omega, dlnOm_dlnr)
    return (kappa * sigma_v) / (np.pi * G * Sigma_b + 1e-30)


def gain_function(r: np.ndarray, Sigma_b: np.ndarray, sigma_v: np.ndarray,
                  Omega: np.ndarray, dlnOm_dlnr: np.ndarray,
                  params: ResonantParams) -> np.ndarray:
    """
    Full gain function g(r) with three gates.
    
    g(r) = g₀ · S_Q(r) · S_σ(r) · S_res(r)
    
    Gates:
    ------
    S_Q : Coldness (Toomre Q < Q_c amplifies)
    S_σ : Dispersion (hot systems suppress)
    S_res : Resonance (cavity condition 2πr ~ m λ_φ)
    
    Returns:
    --------
    g(r) : Gain [kpc⁻²]
    """
    # Amplitude
    g0 = params.alpha / params.R_coh**2  # kpc⁻²
    
    # Gate 1: Coldness (Toomre Q)
    Q = toomre_Q(r, Sigma_b, sigma_v, Omega, dlnOm_dlnr)
    S_Q = 0.5 * (1.0 + np.tanh((params.Q_c - Q) / params.Delta_Q))
    
    # Gate 2: Dispersion (hot → suppress)
    S_sigma = np.exp(-(sigma_v / params.sigma_c)**2)
    
    # Gate 3: Resonance (standing wave)
    x = (2.0 * np.pi * r) / params.lambda_phi  # mode number
    S_res = np.zeros_like(r)
    for m in range(1, params.m_max + 1):
        S_res += np.exp(-((x - m)**2) / (2.0 * params.sigma_m**2))
    
    g = g0 * S_Q * S_sigma * S_res
    
    # Replace any NaNs with zero (no gain)
    g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    
    return g


# ==============================================================================
# FIELD SOLVER
# ==============================================================================

class ResonantHaloSolver:
    """
    Solve field equation with resonant gain:
    
    ∇²φ - μ²(r)φ - λ₄ φ³ = β ρ_b(r)
    
    where μ²(r) = m₀² - g(r)
    
    In tachyonic zones (g > m₀²), field amplifies but saturates via φ⁴ term.
    """
    
    def __init__(self, params: ResonantParams):
        self.params = params
    
    def mu_eff_squared(self, r: np.ndarray, g: np.ndarray) -> np.ndarray:
        """
        Effective mass² including gain.
        
        μ²_eff(r) = m₀² - g(r)
        
        Where g > m₀²: tachyonic (amplification)
        Where g < m₀²: massive (decay)
        """
        return self.params.m0**2 - g
    
    def solve_phi(self, r: np.ndarray, rho_b: np.ndarray, g: np.ndarray,
                  phi_init: np.ndarray = None) -> Tuple[np.ndarray, Dict]:
        """
        Solve static field equation in spherical symmetry.
        
        d²φ/dr² + (2/r) dφ/dr - μ²(r)φ - λ₄ φ³ = β ρ_b(r)
        
        Boundary conditions:
        - φ'(0) = 0 (regularity)
        - φ(∞) = 0 (decay)
        
        Returns:
        --------
        phi : Field profile
        diagnostics : Dict with convergence info, energy, etc.
        """
        mu_sq = self.mu_eff_squared(r, g)
        
        # Rewrite as first-order system: y = [φ, φ']
        def field_ode(r_val, y):
            """ODE system for BVP solver."""
            phi, phi_prime = y
            
            # Interpolate coefficients at this r
            rho_interp = np.interp(r_val, r, rho_b, left=0, right=0)
            mu2_interp = np.interp(r_val, r, mu_sq)
            
            # d²φ/dr² = -2/r φ' + μ² φ + λ₄ φ³ + β ρ_b
            phi_double_prime = (-2.0 / r_val * phi_prime 
                               + mu2_interp * phi 
                               + self.params.lambda_4 * phi**3 
                               + self.params.beta * rho_interp)
            
            return np.vstack((phi_prime, phi_double_prime))
        
        # Boundary conditions
        def bc(ya, yb):
            """
            ya: φ, φ' at r=r_min
            yb: φ, φ' at r=r_max
            
            Conditions:
            - φ'(r_min) = 0 (regularity)
            - φ(r_max) = 0 (decay to zero)
            """
            return np.array([ya[1], yb[0]])  # [φ'(r_min), φ(r_max)]
        
        # Initial guess (if not provided)
        if phi_init is None:
            # Perturbative solution: φ ~ β ρ_b / (|μ²| + ε)
            epsilon = 1e-4  # Regularization to avoid blow-up where μ²≈0
            mu_sq_abs = np.abs(mu_sq) + epsilon
            phi_init = (self.params.beta * rho_b) / mu_sq_abs
            # Damp initial amplitude to stay in linear regime
            phi_init = phi_init / (1.0 + np.abs(phi_init))
        
        y_init = np.vstack((phi_init, np.gradient(phi_init, r)))
        
        # Use relaxation solver directly (more robust for tachyonic zones)
        print("   Using relaxation solver (robust for tachyonic regions)...")
        phi, diagnostics = self._solve_relaxation(r, rho_b, mu_sq, phi_init)
        
        return phi, diagnostics
    
    def _solve_relaxation(self, r: np.ndarray, rho_b: np.ndarray,
                         mu_sq: np.ndarray, phi_init: np.ndarray,
                         max_iter: int = 500, tol: float = 1e-5,
                         omega: float = 0.3) -> Tuple[np.ndarray, Dict]:
        """
        Relaxation method with under-relaxation for stability.
        
        Iterate: φ^(n+1) = φ^(n) + ω [φ_new - φ^(n)]
        where φ_new solves: ∇²φ - μ²φ = β ρ_b + λ₄ (φ^(n))³
        
        Parameters:
        -----------
        omega : Under-relaxation parameter (0 < ω ≤ 1)
            Smaller = more stable but slower convergence
        """
        phi = phi_init.copy()
        
        for iteration in range(max_iter):
            # Clip field to prevent overflow (physical saturation)
            phi_clipped = np.clip(phi, -100, 100)
            
            # Source term with current φ (with saturation)
            source = self.params.beta * rho_b + self.params.lambda_4 * phi_clipped**3
            
            # Solve linear Helmholtz: ∇²φ - μ²φ = source
            phi_new = self._solve_helmholtz(r, mu_sq, source)
            
            # Clip solution to reasonable range
            phi_new = np.clip(phi_new, -100, 100)
            
            # Under-relaxation (damping for stability)
            phi_update = phi + omega * (phi_new - phi)
            
            # Check convergence
            residual = np.max(np.abs(phi_update - phi))
            phi = phi_update
            
            if residual < tol:
                diagnostics = {
                    'success': True,
                    'message': f'Relaxation converged in {iteration+1} iterations (residual={residual:.2e})',
                    'niter': iteration + 1,
                    'residual': residual
                }
                return phi, diagnostics
        
        # Did not converge, but return best solution
        diagnostics = {
            'success': False,
            'message': f'Relaxation max iterations ({max_iter}) reached (residual={residual:.2e})',
            'niter': max_iter,
            'residual': residual
        }
        return phi, diagnostics
    
    def _solve_helmholtz(self, r: np.ndarray, mu_sq: np.ndarray, 
                        source: np.ndarray) -> np.ndarray:
        """
        Solve linear Helmholtz equation in spherical symmetry.
        
        d²φ/dr² + (2/r) dφ/dr - μ²(r) φ = source(r)
        
        Using finite differences.
        """
        n = len(r)
        dr = np.diff(r)
        
        # Build tridiagonal matrix
        diag = np.zeros(n)
        lower = np.zeros(n-1)
        upper = np.zeros(n-1)
        rhs = source.copy()
        
        for i in range(1, n-1):
            dr_left = dr[i-1]
            dr_right = dr[i]
            dr_avg = 0.5 * (dr_left + dr_right)
            
            # Coefficients
            c_left = 1.0 / (dr_left * dr_avg)
            c_right = 1.0 / (dr_right * dr_avg)
            c_center = -(c_left + c_right) - mu_sq[i]
            c_grad = 2.0 / (r[i] * dr_avg)
            
            lower[i-1] = c_left - c_grad
            diag[i] = c_center
            upper[i] = c_right + c_grad
        
        # Boundary conditions
        # Left: φ'(0) = 0 → φ[1] = φ[0]
        diag[0] = 1.0
        upper[0] = -1.0
        rhs[0] = 0.0
        
        # Right: φ(∞) = 0
        diag[-1] = 1.0
        rhs[-1] = 0.0
        
        # Solve tridiagonal system
        from scipy.linalg import solve_banded
        ab = np.zeros((3, n))
        ab[0, 1:] = upper
        ab[1, :] = diag
        ab[2, :-1] = lower
        
        phi = solve_banded((1, 1), ab, rhs)
        
        return phi
    
    def field_energy_density(self, r: np.ndarray, phi: np.ndarray, 
                            g: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute field energy density components.
        
        ρ_φ = (1/2)(∇φ)² + V(φ)
        
        where V(φ) = (1/2)μ²(r)φ² + (λ₄/4)φ⁴
        
        Returns dict with:
        - kinetic: (1/2)(dφ/dr)²
        - potential: V(φ)
        - total: ρ_φ
        """
        phi_prime = np.gradient(phi, r)
        
        kinetic = 0.5 * phi_prime**2
        
        mu_sq = self.mu_eff_squared(r, g)
        potential = 0.5 * mu_sq * phi**2 + 0.25 * self.params.lambda_4 * phi**4
        
        return {
            'kinetic': kinetic,
            'potential': potential,
            'total': kinetic + potential
        }


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    """
    Test on synthetic exponential disk.
    """
    print("="*80)
    print("RESONANT HALO SOLVER TEST")
    print("="*80)
    
    # Create test galaxy: exponential disk
    r = np.linspace(0.1, 30, 300)
    R_d = 3.0  # Disk scale [kpc]
    Sigma_0 = 1e9  # Central surface density [M☉/kpc²]
    
    # Profiles
    Sigma = Sigma_0 * np.exp(-r / R_d)
    h_z = 0.3  # Scale height [kpc]
    rho_b = Sigma / (2 * h_z)  # Volume density
    
    # Kinematics
    v_max = 200.0  # km/s
    v_c = v_max * np.sqrt(1 - np.exp(-r / R_d))
    Omega = v_c / r
    dlnOm_dlnr = np.gradient(np.log(Omega), np.log(r))
    
    sigma_v = 20.0 * np.ones_like(r)  # Cold disk [km/s]
    
    print(f"\nSynthetic galaxy:")
    print(f"  R_d = {R_d:.1f} kpc")
    print(f"  v_max = {v_max:.0f} km/s")
    print(f"  σ_v = {sigma_v[0]:.0f} km/s (cold!)")
    
    # Setup parameters
    params = ResonantParams(
        m0=0.02,
        R_coh=5.0,
        alpha=1.5,
        lambda_phi=8.0,
        Q_c=1.5,
        sigma_c=30.0,
        sigma_m=0.25,
        m_max=2,
        beta=0.5
    )
    
    print(f"\nParameters:")
    print(f"  m₀ = {params.m0:.3f} kpc⁻¹")
    print(f"  R_coh = {params.R_coh:.1f} kpc")
    print(f"  λ_φ = {params.lambda_phi:.1f} kpc")
    print(f"  Q_c = {params.Q_c:.1f}")
    
    # Compute gain
    g = gain_function(r, Sigma, sigma_v, Omega, dlnOm_dlnr, params)
    
    print(f"\nGain function:")
    print(f"  max(g) = {np.max(g):.4f} kpc⁻²")
    print(f"  m₀² = {params.m0**2:.4f} kpc⁻²")
    
    tachyonic = g > params.m0**2
    if np.any(tachyonic):
        r_tach = r[tachyonic]
        print(f"  Tachyonic zone: r ∈ [{r_tach[0]:.1f}, {r_tach[-1]:.1f}] kpc")
    else:
        print(f"  No tachyonic zone (g < m₀² everywhere)")
    
    # Solve field
    solver = ResonantHaloSolver(params)
    
    print(f"\nSolving field equation...")
    phi, diag = solver.solve_phi(r, rho_b, g)
    
    print(f"  Status: {diag['success']}")
    print(f"  Message: {diag['message']}")
    if diag['residual'] is not None:
        print(f"  Residual: {diag['residual']:.2e}")
    
    # Field energy
    energy = solver.field_energy_density(r, phi, g)
    
    print(f"\nField profile:")
    print(f"  max(φ) = {np.max(np.abs(phi)):.4f}")
    print(f"  E_kinetic (integrated) = {np.trapz(energy['kinetic'] * 4*np.pi*r**2, r):.2e}")
    print(f"  E_potential (integrated) = {np.trapz(energy['potential'] * 4*np.pi*r**2, r):.2e}")
    
    # Plot
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Gain function
    ax = axes[0, 0]
    ax.plot(r, g, 'b-', lw=2, label='g(r)')
    ax.axhline(params.m0**2, color='r', ls='--', label='m₀²')
    ax.fill_between(r, 0, g, where=tachyonic, alpha=0.2, color='red', label='Tachyonic')
    ax.set_xlabel('Radius [kpc]')
    ax.set_ylabel('Gain g(r) [kpc⁻²]')
    ax.set_title('Resonant Gain Function')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Field profile
    ax = axes[0, 1]
    ax.plot(r, phi, 'g-', lw=2)
    ax.set_xlabel('Radius [kpc]')
    ax.set_ylabel('Field φ(r)')
    ax.set_title('Coherence Field')
    ax.grid(True, alpha=0.3)
    
    # Energy density
    ax = axes[1, 0]
    ax.plot(r, energy['kinetic'], 'b-', lw=2, label='Kinetic')
    ax.plot(r, energy['potential'], 'r-', lw=2, label='Potential')
    ax.plot(r, energy['total'], 'k--', lw=2, label='Total')
    ax.set_xlabel('Radius [kpc]')
    ax.set_ylabel('Energy Density')
    ax.set_title('Field Energy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Toomre Q
    Q = toomre_Q(r, Sigma, sigma_v, Omega, dlnOm_dlnr)
    ax = axes[1, 1]
    ax.plot(r, Q, 'purple', lw=2)
    ax.axhline(params.Q_c, color='k', ls='--', label=f'Q_c = {params.Q_c}')
    ax.set_xlabel('Radius [kpc]')
    ax.set_ylabel('Toomre Q')
    ax.set_title('Disk Stability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    import os
    os.makedirs('coherence-field-theory/outputs', exist_ok=True)
    outpath = 'coherence-field-theory/outputs/resonant_halo_test.png'
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved: {outpath}")
    
    print("\n" + "="*80)
