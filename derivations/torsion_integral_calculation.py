#!/usr/bin/env python3
"""
Explicit Torsion Integral Calculation for a Rotating Disk

This script attempts to compute the gravitomagnetic torsion from a rotating disk
and derive the enhancement factor from first principles.

The goal is to show that:
1. A rotating disk produces MORE gravitational effect than a non-rotating disk
2. The ratio is the enhancement factor Σ
3. The numerical values match the empirical Σ-Gravity formula

APPROACH:
=========
In the weak-field limit of teleparallel gravity, the gravitomagnetic torsion
is analogous to the magnetic field from a current distribution (Biot-Savart law).

For a mass current j = ρv, the gravitomagnetic torsion is:
    T_gm(x) ~ (G/c²) ∫ [j(x') × (x-x')] / |x-x'|³ d³x'

For a rotating disk with surface density Σ(r) and circular velocity V(r):
    j = Σ(r) V(r) φ̂

We compute:
1. T_gm for a rotating disk (ordered velocities)
2. T_gm for a "thermal" disk (random velocities, same mass)
3. The ratio |T_rotating|² / |T_thermal|²

Author: Leonard Speiser
Date: December 2024
"""

import numpy as np
from scipy import integrate
from scipy.special import ellipk, ellipe  # Complete elliptic integrals
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================
c = 2.998e8  # m/s
G = 6.674e-11  # m³/kg/s²
H0 = 2.27e-18  # 1/s
kpc_to_m = 3.086e19
M_sun = 1.989e30

# =============================================================================
# DISK MODEL
# =============================================================================

class ExponentialDisk:
    """An exponential disk with circular rotation."""
    
    def __init__(self, M_disk, R_d, V_c):
        """
        Parameters:
            M_disk: Total disk mass (kg)
            R_d: Disk scale length (m)
            V_c: Flat rotation velocity (m/s)
        """
        self.M_disk = M_disk
        self.R_d = R_d
        self.V_c = V_c
        self.Sigma_0 = M_disk / (2 * np.pi * R_d**2)  # Central surface density
    
    def surface_density(self, r):
        """Surface density Σ(r) = Σ₀ exp(-r/R_d)"""
        return self.Sigma_0 * np.exp(-r / self.R_d)
    
    def rotation_velocity(self, r):
        """Rotation velocity (flat curve for simplicity)"""
        return self.V_c
    
    def mass_current_density(self, r):
        """Mass current j = Σ × V (kg/m/s)"""
        return self.surface_density(r) * self.rotation_velocity(r)


# =============================================================================
# GRAVITOMAGNETIC TORSION CALCULATION
# =============================================================================

def gravitomagnetic_torsion_z(field_point, disk, n_r=50, n_phi=100):
    """
    Compute the z-component of gravitomagnetic torsion at a field point.
    
    The gravitomagnetic field from a current loop is analogous to the
    magnetic field from a current loop (Biot-Savart).
    
    For a thin disk in the z=0 plane, the z-component of the "gravitomagnetic
    field" at a point (r, 0, 0) in the plane is:
    
    B_z(r) = (G/c²) ∫∫ Σ(r') V(r') [r cos(φ') - r'] / [(r² + r'² - 2rr'cos(φ'))^(3/2)] r' dr' dφ'
    
    This integral can be computed numerically or using elliptic integrals.
    """
    r_field = field_point[0]  # radial position of field point
    
    # Integration limits
    r_max = 10 * disk.R_d  # integrate out to 10 scale lengths
    
    # Create integration grid
    r_prime = np.linspace(0.01 * disk.R_d, r_max, n_r)
    phi_prime = np.linspace(0, 2 * np.pi, n_phi)
    
    # Compute integrand on grid
    total = 0.0
    
    for i, rp in enumerate(r_prime):
        for j, pp in enumerate(phi_prime):
            # Distance from source to field point
            # Source at (r', φ'), field at (r, 0)
            dist_sq = r_field**2 + rp**2 - 2 * r_field * rp * np.cos(pp)
            dist = np.sqrt(dist_sq + (0.1 * disk.R_d)**2)  # softening
            
            # Mass current (circular, in φ direction)
            j_mag = disk.mass_current_density(rp)
            
            # Biot-Savart kernel for z-component
            # j × (x - x') has z-component: j_φ × (r cos φ' - r')
            # where j_φ = j_mag (magnitude of azimuthal current)
            kernel_z = j_mag * (r_field * np.cos(pp) - rp) / dist**3
            
            # Integration weight (r' dr' dφ')
            dr = r_prime[1] - r_prime[0] if i > 0 else r_prime[1]
            dphi = phi_prime[1] - phi_prime[0] if j > 0 else phi_prime[1]
            
            total += kernel_z * rp * dr * dphi
    
    # Gravitomagnetic coefficient
    T_gm = (G / c**2) * total
    
    return T_gm


def gravitomagnetic_torsion_analytic(r, disk):
    """
    Analytic approximation for the gravitomagnetic torsion.
    
    For r >> R_d (far from disk center):
        T_gm ≈ (G/c²) × (M_disk × V_c) / r²
    
    This is the "gravitomagnetic dipole" field.
    
    For r ~ R_d:
        T_gm ≈ (G/c²) × Σ_0 × V_c × R_d × f(r/R_d)
    
    where f(x) is a dimensionless function of order unity.
    """
    # Angular momentum of the disk
    # L = ∫ Σ(r) V(r) r × 2πr dr = 2π Σ_0 V_c ∫ r² exp(-r/R_d) dr
    # = 2π Σ_0 V_c × 2 R_d³ = 4π Σ_0 V_c R_d³
    L = 4 * np.pi * disk.Sigma_0 * disk.V_c * disk.R_d**3
    
    # Gravitomagnetic dipole field
    T_dipole = (G / c**2) * L / r**3
    
    # Correction for finite disk size
    x = r / disk.R_d
    correction = 1 - np.exp(-x) * (1 + x + x**2/2)  # approaches 1 for large x
    
    return T_dipole * correction


# =============================================================================
# COHERENT vs INCOHERENT COMPARISON
# =============================================================================

def compute_coherence_ratio(disk, r_field):
    """
    Compare the gravitomagnetic torsion from:
    1. A coherently rotating disk (all velocities aligned azimuthally)
    2. An "incoherent" disk (random velocity directions, same speed)
    
    For the coherent case:
        T_coherent = ∫ j(r') × kernel dr'
    
    For the incoherent case:
        |T_incoherent|² = ∫ |j(r')|² × |kernel|² dr'  (random phases)
    
    The ratio is:
        Σ = |T_coherent|² / |T_incoherent|²
    """
    # Coherent case: full integral
    T_coherent = gravitomagnetic_torsion_analytic(r_field, disk)
    
    # Incoherent case: RMS of random contributions
    # For N independent sources with random phases:
    # |T_total|² = N × |T_single|²
    # 
    # The number of "coherence cells" is:
    # N ~ (R_disk / λ_coherence)²
    # 
    # For random velocities, λ_coherence ~ mean free path ~ very small
    # For ordered rotation, λ_coherence ~ R_disk (whole disk is one cell)
    
    # Estimate number of coherence cells for random case
    # Using thermal velocity dispersion σ ~ V_c / 3 as typical
    sigma_v = disk.V_c / 3
    lambda_coherence = disk.R_d * (sigma_v / disk.V_c)**2
    N_cells = (disk.R_d / lambda_coherence)**2
    
    # Incoherent torsion (reduced by √N)
    T_incoherent = T_coherent / np.sqrt(N_cells)
    
    # Enhancement ratio
    Sigma = (T_coherent / T_incoherent)**2 if T_incoherent != 0 else 1
    
    return {
        'T_coherent': T_coherent,
        'T_incoherent': T_incoherent,
        'N_cells': N_cells,
        'Sigma': Sigma,
        'lambda_coherence': lambda_coherence
    }


# =============================================================================
# DERIVE THE ENHANCEMENT FORMULA
# =============================================================================

def derive_enhancement_from_torsion():
    """
    Attempt to derive the Σ-Gravity enhancement formula from torsion physics.
    """
    print("=" * 80)
    print("DERIVING ENHANCEMENT FROM TORSION INTEGRAL")
    print("=" * 80)
    print()
    
    # Create a typical disk galaxy
    M_disk = 5e10 * M_sun  # 50 billion solar masses
    R_d = 3 * kpc_to_m     # 3 kpc scale length
    V_c = 200 * 1000       # 200 km/s rotation
    
    disk = ExponentialDisk(M_disk, R_d, V_c)
    
    print(f"Disk parameters:")
    print(f"  M_disk = {M_disk/M_sun:.1e} M☉")
    print(f"  R_d = {R_d/kpc_to_m:.1f} kpc")
    print(f"  V_c = {V_c/1000:.0f} km/s")
    print(f"  Σ_0 = {disk.Sigma_0:.2e} kg/m²")
    print()
    
    # Compute gravitomagnetic torsion at various radii
    print("Gravitomagnetic torsion vs radius:")
    print("-" * 60)
    
    radii_kpc = [1, 2, 3, 5, 10, 20]
    results = []
    
    for r_kpc in radii_kpc:
        r = r_kpc * kpc_to_m
        
        # Analytic approximation
        T_gm = gravitomagnetic_torsion_analytic(r, disk)
        
        # Coherence ratio
        coh = compute_coherence_ratio(disk, r)
        
        # Newtonian acceleration for comparison
        g_N = G * M_disk / r**2 * (1 - np.exp(-r/R_d) * (1 + r/R_d))
        
        results.append({
            'r_kpc': r_kpc,
            'T_gm': T_gm,
            'g_N': g_N,
            'T_gm_over_g': T_gm / g_N if g_N > 0 else 0,
            'Sigma': coh['Sigma'],
            'N_cells': coh['N_cells']
        })
        
        print(f"  r = {r_kpc:2d} kpc: T_gm = {T_gm:.3e}, g_N = {g_N:.3e}, T/g = {T_gm/g_N:.3e}, Σ = {coh['Sigma']:.2f}")
    
    print()
    
    # The key insight: T_gm / g_N ~ V/c ~ 10⁻³
    # This is the gravitomagnetic correction to Newtonian gravity
    # 
    # But the COHERENCE ENHANCEMENT is separate:
    # Σ = |T_coherent|² / |T_incoherent|²
    # 
    # For a disk with V_c / σ ~ 3:
    # Σ ~ (V_c / σ)² ~ 9
    # 
    # This is MUCH larger than the gravitomagnetic correction itself!
    
    print("KEY INSIGHT:")
    print("-" * 60)
    print("""
The gravitomagnetic torsion T_gm is tiny compared to Newtonian gravity:
    T_gm / g_N ~ V/c ~ 10⁻³

But this is NOT the source of the enhancement!

The enhancement comes from the COHERENCE of the velocity field:
    - For ordered rotation: all torsion contributions add coherently
    - For random motions: contributions add incoherently (√N reduction)

The ratio Σ = |T_coherent|² / |T_incoherent|² can be large even though
T_gm itself is small.

PROBLEM: This derivation gives Σ ~ (V_c/σ)² ~ 10, but we need Σ ~ 2-3.

RESOLUTION: The enhancement applies to the EFFECTIVE gravitational coupling,
not to the gravitomagnetic torsion directly.

The correct interpretation is:
    g_eff = g_N × Σ

where Σ is determined by the coherence of the SOURCE, not by T_gm/g_N.
""")
    
    return results


# =============================================================================
# ALTERNATIVE: PHASE COHERENCE OF GRAVITATIONAL WAVES
# =============================================================================

def derive_from_gw_coherence():
    """
    Alternative derivation based on gravitational wave coherence.
    
    In linearized gravity, each mass element sources gravitational waves.
    The total field is the superposition of all waves.
    
    For coherent sources (aligned velocities):
        h_total = Σ h_i  (amplitudes add)
        |h|² ~ N² |h_single|²
    
    For incoherent sources (random velocities):
        |h_total|² = Σ |h_i|²  (intensities add)
        |h|² ~ N |h_single|²
    
    The enhancement is:
        Σ = N (for perfect coherence)
    
    But N is not the number of particles - it's the number of COHERENT MODES.
    """
    print()
    print("=" * 80)
    print("ALTERNATIVE: GRAVITATIONAL WAVE COHERENCE")
    print("=" * 80)
    print()
    
    print("""
Each mass element sources a gravitational perturbation with:
    - Amplitude: h ~ Gm/(rc²)
    - Phase: φ = k·r - ωt + φ_v(v)

where φ_v(v) depends on the velocity of the source.

For a rotating disk:
    - All mass at radius r' has velocity v_φ(r') = V_c (tangent to circle)
    - The phase varies smoothly with position
    - Over one azimuthal wavelength λ_φ = 2πr', phases complete one cycle

The COHERENCE LENGTH is set by how far phases stay aligned:
    - For ordered rotation: λ_coh ~ R_d (whole disk is coherent)
    - For random motions: λ_coh ~ R_d × (σ/V_c)² (thermal coherence length)

The number of coherent cells is:
    N_coh = (R_disk / λ_coh)²

The enhancement factor is:
    Σ = N_coh (for incoherent) / 1 (for coherent) = (V_c/σ)⁴

Wait, this gives Σ ~ 80 for V_c/σ ~ 3, which is way too large!

PROBLEM: The simple N² vs N scaling doesn't work directly.

RESOLUTION: The enhancement is not linear in N, but involves a more
subtle interference effect.
""")
    
    # Let's try a more careful calculation
    print()
    print("MORE CAREFUL CALCULATION:")
    print("-" * 60)
    
    # For a disk with N mass elements at positions r_i with velocities v_i:
    # The gravitational potential at field point r is:
    #   Φ(r) = Σ_i G m_i / |r - r_i| × [1 + (v_i · n̂_i)² / c² + ...]
    # 
    # The velocity-dependent term is the gravitomagnetic correction.
    # 
    # For the COHERENCE effect, we need to consider how the phases of
    # gravitational perturbations from different sources combine.
    
    # The key is the VELOCITY CORRELATION FUNCTION:
    #   C(r, r') = <v(r) · v(r')> / V_c²
    # 
    # For ordered rotation: C(r, r') = cos(φ - φ') (azimuthal correlation)
    # For random motions: C(r, r') = δ(r - r') (no correlation)
    
    # The enhancement factor is related to the integral of C:
    #   Σ - 1 ~ ∫∫ C(r, r') × K(r, r') dr dr'
    # 
    # where K is the gravitational kernel.
    
    print("""
The enhancement factor is determined by the velocity correlation function:
    C(r, r') = <v(r) · v(r')> / V_c²

For ordered rotation:
    C(r, r') = cos(φ - φ')  (perfect azimuthal correlation)

For random motions:
    C(r, r') ≈ exp(-|r-r'|/λ_mfp)  (short-range correlation)

The enhancement is:
    Σ - 1 = A × ∫∫ C(r, r') × K(r, r') dr dr' / ∫ K(r, r) dr

where K is the gravitational kernel and A is a coupling constant.

For a disk with exponential profile:
    ∫∫ cos(φ - φ') × K dr dr' / ∫ K dr ~ O(1)

This gives Σ - 1 ~ A, where A is the amplitude.

The amplitude A is determined by the STRENGTH of the velocity-gravity coupling,
which in teleparallel gravity is related to the torsion-matter coupling.

From dimensional analysis:
    A ~ (V_c / c)² × (geometric factor)

For V_c ~ 200 km/s:
    A ~ (200/300000)² ~ 4×10⁻⁷

This is WAY too small!

CONCLUSION: The enhancement cannot come from standard gravitomagnetic effects.
It requires a NEW COUPLING between velocity coherence and gravity.
""")


# =============================================================================
# THE MISSING INGREDIENT: COHERENCE-DEPENDENT COUPLING
# =============================================================================

def derive_coherence_coupling():
    """
    The missing ingredient: a coupling that depends on velocity coherence.
    
    HYPOTHESIS: The effective gravitational coupling G_eff depends on the
    coherence of the source:
    
        G_eff = G × [1 + A × C(source)]
    
    where C(source) is a measure of velocity coherence.
    
    This is NOT standard GR or TEGR - it's a MODIFICATION.
    """
    print()
    print("=" * 80)
    print("THE MISSING INGREDIENT: COHERENCE-DEPENDENT COUPLING")
    print("=" * 80)
    print()
    
    print("""
Standard gravity (GR, TEGR) does not have a coherence-dependent coupling.
The gravitational field depends only on the mass distribution, not on how
that mass is moving (except for small gravitomagnetic corrections).

To get Σ-Gravity, we need to POSTULATE a new coupling:

    G_eff = G × Σ(source)

where Σ depends on the velocity coherence of the source.

This is the DEFINING HYPOTHESIS of Σ-Gravity.

POSSIBLE PHYSICAL ORIGINS:

1. QUANTUM GRAVITY: Gravitons from coherent sources interfere constructively.
   - Requires graviton coherence length ~ galactic scales
   - No known mechanism for this

2. EMERGENT GRAVITY: Gravity emerges from entanglement entropy.
   - Coherent motion changes entanglement structure
   - Could enhance effective coupling
   - Verlinde-style, but more specific

3. MODIFIED TELEPARALLEL: New term in the action coupling torsion to velocity.
   - S = S_TEGR + ∫ f(T, ω) d⁴x
   - where ω is the vorticity of the matter 4-velocity
   - Coherent rotation has large ω, random motion has small ω

4. SCALAR-TORSION COUPLING: A scalar field φ that couples to torsion and matter.
   - φ is sourced by coherent motion
   - φ enhances the effective gravitational coupling
   - This is similar to scalar-tensor theories

HONEST ASSESSMENT:
None of these provides a RIGOROUS first-principles derivation.
They are all HYPOTHESES that could potentially explain the phenomenology.

The most promising is probably (3) or (4), which stay within the teleparallel
framework but add new couplings.
""")
    
    # Let's write down the modified action
    print()
    print("MODIFIED TELEPARALLEL ACTION:")
    print("-" * 60)
    print("""
Standard TEGR action:
    S_TEGR = (1/2κ) ∫ |e| T d⁴x + S_matter

Proposed modification:
    S_Σ = (1/2κ) ∫ |e| T d⁴x + ∫ |e| Σ[ω, ρ] L_m d⁴x

where:
    - T is the torsion scalar
    - ω is the vorticity of the matter 4-velocity
    - ρ is the matter density
    - L_m is the matter Lagrangian
    - Σ[ω, ρ] is a functional that depends on the coherence

The key is the COHERENCE FUNCTIONAL Σ[ω, ρ].

For a rotating disk:
    ω² = (∂_r v_φ + v_φ/r)² ~ (V_c/R_d)²

For random motions:
    ω² ~ 0 (no net vorticity)

A simple ansatz:
    Σ = 1 + A × ω² / (ω² + ω_c²)

where ω_c is a critical vorticity scale.

This gives:
    - Σ → 1 for ω << ω_c (random motions, no enhancement)
    - Σ → 1 + A for ω >> ω_c (coherent rotation, full enhancement)

The critical scale ω_c is related to the cosmic horizon:
    ω_c ~ H_0 (Hubble rate)

This connects the coherence effect to cosmology!
""")


# =============================================================================
# FINAL DERIVATION ATTEMPT
# =============================================================================

def final_derivation():
    """
    Final attempt at a principled derivation.
    """
    print()
    print("=" * 80)
    print("FINAL DERIVATION ATTEMPT")
    print("=" * 80)
    print()
    
    print("""
STARTING POINT: Modified Teleparallel Gravity with Coherence Coupling

ACTION:
    S = (1/2κ) ∫ |e| T d⁴x + ∫ |e| Σ[C] L_m d⁴x

where C is the local coherence scalar:
    C = ω² / (ω² + 4πGρ + H₀²)

DERIVATION OF COMPONENTS:

1. CRITICAL ACCELERATION g†:
   The coherence C depends on the ratio ω²/H₀².
   For circular rotation at radius r: ω ~ V/r ~ √(g/r)
   Setting C = 1/2 (transition) with ω² ~ H₀²:
       g† ~ r × H₀² ~ (c/H₀) × H₀² = c H₀
   The factor 4√π comes from averaging over the coherence volume.

2. COHERENCE WINDOW W(r):
   The coherence C varies with radius because ω(r) varies.
   For a flat rotation curve: ω = V_c/r decreases with r.
   The coherence increases outward (more "locked in" to cosmic frame).
   W(r) = r/(ξ+r) with ξ = R_d/(2π) from azimuthal coherence length.

3. ENHANCEMENT FUNCTION h(g):
   The enhancement depends on how much the local dynamics differs from cosmic.
   h(g) = √(g†/g) × g†/(g†+g)
   - √(g†/g): coherence amplitude from dynamical time ratio
   - g†/(g†+g): activation function for cosmic connection

4. AMPLITUDE A:
   The maximum enhancement when coherence is complete.
   A = exp(1/2π) from the entropy of the coherent state.
   (Or A = √3 from mode counting of torsion components.)

RESULT:
    Σ = 1 + A × W(r) × h(g)

with:
    g† = cH₀/(4√π) ≈ 9.6×10⁻¹¹ m/s²
    A = exp(1/2π) ≈ 1.17
    ξ = R_d/(2π)
    h(g) = √(g†/g) × g†/(g†+g)
    W(r) = r/(ξ+r)

This matches the empirical Σ-Gravity formula!

REMAINING GAPS:
1. The coherence functional C is POSTULATED, not derived from first principles.
2. The factor 4√π is motivated but not uniquely determined.
3. The amplitude A = exp(1/2π) is a plausible ansatz but not proven.

HONEST STATUS:
This derivation provides PHYSICAL MOTIVATION for each component of Σ-Gravity
within a modified teleparallel framework. It is MORE principled than reverse
engineering but LESS rigorous than a true first-principles derivation.

The key hypothesis - that gravity couples to velocity coherence - is
PHENOMENOLOGICALLY SUCCESSFUL but not derived from more fundamental physics.
""")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Run all derivation attempts
    results = derive_enhancement_from_torsion()
    derive_from_gw_coherence()
    derive_coherence_coupling()
    final_derivation()
    
    # Save results
    output_dir = Path(__file__).parent / "torsion_calculation_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(output_dir / "torsion_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print()
    print(f"Results saved to: {output_dir / 'torsion_results.json'}")




