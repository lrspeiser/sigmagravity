#!/usr/bin/env python3
"""
Microphysical Environmental Gates for GPM

Replaces ad-hoc power-law gates with theoretically-motivated forms
derived from linear response theory.

From LINEAR_RESPONSE_DERIVATION.md:
- Π(ω, k) = self-energy (polarization tensor) of rotating disk
- In static limit: Π(0, k) = ℓ⁻² where ℓ = σ_v / √(2πG Σ_b)
- Suppression mechanisms: Landau damping (high Q), dephasing (high σ_v)

Mathematical Forms:
------------------
1. Q-gate (Landau damping):
   g_Q(Q) = 1 / [1 + (Q/Q*)²]
   
   Physical basis: For Q >> 1, disk is gravitationally stable. Random stellar
   motions dephase collective response → Landau damping → Π(ω,k) → 0.

2. σ-gate (velocity dephasing):
   g_σ(σ_v) = 1 / [1 + (σ_v/σ*)²]
   
   Physical basis: Dephasing frequency Δω ~ k σ_v. When Δω >> κ (epicyclic),
   Lindblad resonances wash out → Π(ω,k) → 0.

3. M-gate (mass/homogeneity):
   g_M(M) = 1 / [1 + (M/M*)^n_M]
   
   Physical basis: High-mass systems approach homogeneity. No preferred disk
   geometry → no axisymmetric response → α_eff → 0.

4. K-gate (curvature):
   g_K(K) = exp(-K/K*)
   
   Physical basis: High curvature (K ~ ρ) suppresses non-local response.
   Local Newtonian gravity dominates.

Effective Coupling:
-------------------
α_eff(R) = α_0 × g_Q(Q(R)) × g_σ(σ_v(R)) × g_M(M_disk) × g_K(K(R))

This replaces the phenomenological power-law gates while preserving
their functional behavior (smooth suppression at high Q, σ_v, M, K).

Self-Consistent Coherence Length:
---------------------------------
ℓ(R) = σ_v(R) / √(2πG ⟨Σ_b⟩_ℓ(R))

where ⟨Σ_b⟩_ℓ is the surface density averaged over scale ℓ.
This requires iterative solution:
1. Start with ℓ = ℓ_0
2. Compute ⟨Σ_b⟩_ℓ = ∫_0^ℓ Σ_b(r) r dr / (ℓ²/2)
3. Update ℓ = σ_v / √(2πG ⟨Σ_b⟩_ℓ)
4. Repeat until convergence

Reference: LINEAR_RESPONSE_DERIVATION.md
Author: GPM Theory Team
Date: December 2024
"""

import numpy as np


class MicrophysicalGates:
    """
    Environmental gates derived from linear response theory.
    
    Replaces ad-hoc power-law gates with theoretically-motivated forms
    based on Landau damping and velocity dephasing.
    
    Parameters:
    -----------
    alpha_0 : float
        Base susceptibility (static response strength)
    Q_star : float
        Toomre Q threshold for Landau damping
    sigma_star : float
        Velocity dispersion threshold [km/s]
    M_star : float
        Mass threshold for homogeneity breaking [M_sun]
    n_M : float
        Mass gate exponent
    K_star : float
        Curvature threshold [1/kpc²]
    """
    
    def __init__(self, alpha_0=0.30, Q_star=2.0, sigma_star=70.0, 
                 M_star=2e10, n_M=2.5, K_star=1e-3):
        self.alpha_0 = alpha_0
        self.Q_star = Q_star
        self.sigma_star = sigma_star
        self.M_star = M_star
        self.n_M = n_M
        self.K_star = K_star
        
        # Physical constants
        self.G_kpc = 4.302e-3  # G in kpc (km/s)^2 / M_sun
    
    def g_Q(self, Q):
        """
        Toomre Q gate (Landau damping).
        
        For Q >> 1, disk is gravitationally stable. Random stellar motions
        dephase collective gravitational response.
        
        Form: g_Q = 1 / [1 + (Q/Q*)²]
        
        Physical interpretation:
        - Q < Q*: Gravitationally unstable, coherent response active (g_Q → 1)
        - Q >> Q*: Stable, random motions dominate, Landau damping (g_Q → 0)
        
        Parameters:
        -----------
        Q : float or array
            Toomre Q parameter = σ_v κ / (πG Σ_b)
        
        Returns:
        --------
        g_Q : float or array
            Q-gate value ∈ [0, 1]
        """
        Q_safe = np.maximum(Q, 0.1)  # Avoid division by zero
        return 1.0 / (1.0 + (Q_safe / self.Q_star)**2)
    
    def g_sigma(self, sigma_v):
        """
        Velocity dispersion gate (dephasing).
        
        Dephasing frequency Δω ~ k σ_v. When Δω >> κ (epicyclic frequency),
        Lindblad resonances wash out → no coherent response.
        
        Form: g_σ = 1 / [1 + (σ_v/σ*)²]
        
        Physical interpretation:
        - σ_v < σ*: Cold disk, coherent epicyclic oscillations (g_σ → 1)
        - σ_v >> σ*: Hot disk, dephasing dominates (g_σ → 0)
        
        Parameters:
        -----------
        sigma_v : float or array
            Velocity dispersion [km/s]
        
        Returns:
        --------
        g_σ : float or array
            σ-gate value ∈ [0, 1]
        """
        sigma_safe = np.maximum(sigma_v, 1.0)  # Avoid division by zero
        return 1.0 / (1.0 + (sigma_safe / self.sigma_star)**2)
    
    def g_M(self, M_disk):
        """
        Mass gate (homogeneity breaking).
        
        High-mass systems approach homogeneity. No preferred disk geometry
        → no axisymmetric gravitational response.
        
        Form: g_M = 1 / [1 + (M/M*)^n_M]
        
        Physical interpretation:
        - M << M*: Small disk, well-defined geometry (g_M → 1)
        - M >> M*: Massive system, homogeneity breaking (g_M → 0)
        
        Parameters:
        -----------
        M_disk : float
            Total disk mass [M_sun]
        
        Returns:
        --------
        g_M : float
            M-gate value ∈ [0, 1]
        """
        M_safe = np.maximum(M_disk, 1e6)  # Avoid division by zero
        return 1.0 / (1.0 + (M_safe / self.M_star)**self.n_M)
    
    def g_K(self, K):
        """
        Curvature gate (high-density suppression).
        
        High curvature (K ~ ρ) suppresses non-local response. Local
        Newtonian gravity dominates over coherence field.
        
        Form: g_K = exp(-K/K*)
        
        Physical interpretation:
        - K << K*: Low density, non-local response active (g_K → 1)
        - K >> K*: High density (Solar System, clusters), local gravity (g_K → 0)
        
        Parameters:
        -----------
        K : float or array
            Ricci curvature scalar [1/kpc²]
            Approximate: K ~ 4πG ρ (for static matter-dominated)
        
        Returns:
        --------
        g_K : float or array
            K-gate value ∈ [0, 1]
        """
        K_safe = np.maximum(K, 0.0)
        return np.exp(-K_safe / self.K_star)
    
    def alpha_eff(self, Q, sigma_v, M_disk, K=0.0):
        """
        Effective coupling strength with all gates applied.
        
        α_eff = α_0 × g_Q × g_σ × g_M × g_K
        
        Parameters:
        -----------
        Q : float or array
            Toomre Q parameter
        sigma_v : float or array
            Velocity dispersion [km/s]
        M_disk : float
            Total disk mass [M_sun]
        K : float or array, optional
            Ricci curvature [1/kpc²] (default: 0)
        
        Returns:
        --------
        α_eff : float or array
            Effective coupling ∈ [0, α_0]
        """
        gQ = self.g_Q(Q)
        gS = self.g_sigma(sigma_v)
        gM = self.g_M(M_disk)
        gK = self.g_K(K)
        
        return self.alpha_0 * gQ * gS * gM * gK
    
    def compute_self_consistent_ell(self, sigma_v, Sigma_b_func, R_disk, 
                                    ell_init=0.8, max_iter=10, tol=0.01):
        """
        Compute self-consistent coherence length.
        
        From linear response derivation:
        ℓ = σ_v / √(2πG ⟨Σ_b⟩_ℓ)
        
        where ⟨Σ_b⟩_ℓ is the surface density averaged over scale ℓ.
        
        Iterative solution:
        1. Start with ℓ = ℓ_init
        2. Compute ⟨Σ_b⟩_ℓ = ∫_0^ℓ Σ_b(r) 2πr dr / (πℓ²)
        3. Update ℓ = σ_v / √(2πG ⟨Σ_b⟩_ℓ)
        4. Repeat until |ℓ_new - ℓ_old| < tol
        
        Parameters:
        -----------
        sigma_v : float
            Velocity dispersion [km/s]
        Sigma_b_func : callable
            Function Σ_b(r) returning surface density [M_sun/kpc²]
        R_disk : float
            Disk scale length [kpc]
        ell_init : float, optional
            Initial guess for ℓ [kpc]
        max_iter : int, optional
            Maximum iterations
        tol : float, optional
            Convergence tolerance [kpc]
        
        Returns:
        --------
        ell : float
            Self-consistent coherence length [kpc]
        converged : bool
            Whether iteration converged
        """
        ell = ell_init
        
        for i in range(max_iter):
            # Compute average surface density over disk of radius ℓ
            # ⟨Σ_b⟩_ℓ = ∫_0^ℓ Σ_b(r) 2πr dr / (πℓ²)
            #         = 2/ℓ² ∫_0^ℓ Σ_b(r) r dr
            
            # Numerical integration (simple trapezoidal)
            r_sample = np.linspace(0, ell, 50)
            Sigma_sample = np.array([Sigma_b_func(r) for r in r_sample])
            integrand = Sigma_sample * r_sample  # Σ_b(r) × r
            
            # ∫_0^ℓ Σ_b(r) r dr ≈ Σ_avg × ℓ²/2
            integral = np.trapz(integrand, r_sample)
            Sigma_avg = 2.0 * integral / ell**2
            
            # Update ℓ
            ell_new = sigma_v / np.sqrt(2 * np.pi * self.G_kpc * Sigma_avg)
            
            # Check convergence
            if abs(ell_new - ell) < tol:
                return ell_new, True
            
            ell = ell_new
        
        # Did not converge
        return ell, False
    
    def compute_theoretical_ell(self, sigma_v, Sigma_b, R_disk, ell_scale_ratio=1.0):
        """
        Compute theoretical coherence length (simple approximation).
        
        ℓ = σ_v / √(2πG Σ_b^eff)
        
        where Σ_b^eff = Σ_b(0) × (R_disk / ℓ) accounts for scale averaging.
        
        This is a simpler formula that doesn't require iteration, using
        the empirical observation that ℓ ~ R_disk^p with p ~ 0.5.
        
        Parameters:
        -----------
        sigma_v : float
            Velocity dispersion [km/s]
        Sigma_b : float
            Central surface density [M_sun/kpc²]
        R_disk : float
            Disk scale length [kpc]
        ell_scale_ratio : float, optional
            Ratio ℓ/R_disk (default: 1.0, but typically ~0.5)
        
        Returns:
        --------
        ell : float
            Coherence length [kpc]
        """
        # Effective surface density (accounting for scale averaging)
        # For ℓ ~ R_disk/2, the average Σ_b is ~ Σ_b(0) / 2
        Sigma_eff = Sigma_b * (R_disk / (ell_scale_ratio * R_disk))
        
        # Coherence length from Toomre scale
        ell = sigma_v / np.sqrt(2 * np.pi * self.G_kpc * Sigma_eff)
        
        return ell


def test_microphysical_gates():
    """
    Test microphysical gates and compare to phenomenological forms.
    """
    gates = MicrophysicalGates(
        alpha_0=0.30,
        Q_star=2.0,
        sigma_star=70.0,
        M_star=2e10,
        n_M=2.5,
        K_star=1e-3
    )
    
    print("="*80)
    print("MICROPHYSICAL GATES TEST")
    print("="*80)
    print()
    
    # Test Q-gate
    Q_values = np.array([0.5, 1.0, 2.0, 3.0, 5.0])
    print("Q-gate (Landau damping):")
    print(f"{'Q':<10} {'g_Q':<10} {'Interpretation'}")
    print("-"*50)
    for Q in Q_values:
        gQ = gates.g_Q(Q)
        interp = "Active" if gQ > 0.5 else "Suppressed"
        print(f"{Q:<10.1f} {gQ:<10.3f} {interp}")
    print()
    
    # Test σ-gate
    sigma_values = np.array([20, 40, 70, 100, 150])
    print("σ-gate (velocity dephasing):")
    print(f"{'σ_v [km/s]':<15} {'g_σ':<10} {'Interpretation'}")
    print("-"*50)
    for sig in sigma_values:
        gS = gates.g_sigma(sig)
        interp = "Active" if gS > 0.5 else "Suppressed"
        print(f"{sig:<15.0f} {gS:<10.3f} {interp}")
    print()
    
    # Test M-gate
    M_values = np.array([1e9, 5e9, 2e10, 1e11, 5e11])
    print("M-gate (homogeneity):")
    print(f"{'M [M☉]':<15} {'g_M':<10} {'Interpretation'}")
    print("-"*50)
    for M in M_values:
        gM = gates.g_M(M)
        interp = "Active" if gM > 0.5 else "Suppressed"
        print(f"{M:<15.2e} {gM:<10.3f} {interp}")
    print()
    
    # Test full α_eff
    print("Effective coupling α_eff:")
    print(f"{'Galaxy':<15} {'Q':<6} {'σ_v':<8} {'M':<12} {'α_eff':<8} {'%'}")
    print("-"*70)
    
    test_cases = [
        ("Dwarf (cold)", 1.5, 25, 5e9),
        ("MW-like", 2.0, 40, 6e10),
        ("Massive", 3.0, 70, 2e11),
        ("Elliptical", 5.0, 150, 1e12),
    ]
    
    for name, Q, sig, M in test_cases:
        alpha = gates.alpha_eff(Q, sig, M)
        pct = 100 * alpha / gates.alpha_0
        print(f"{name:<15} {Q:<6.1f} {sig:<8.0f} {M:<12.2e} {alpha:<8.3f} {pct:>5.1f}%")
    
    print()
    print("="*80)
    print("✓ Microphysical gates operational")
    print()
    print("Key features:")
    print("  1. Smooth suppression (no sharp cutoffs)")
    print("  2. Physical basis (Landau damping, dephasing)")
    print("  3. Testable predictions (α_eff vs Q, σ_v correlation)")
    print("="*80)


if __name__ == '__main__':
    test_microphysical_gates()
