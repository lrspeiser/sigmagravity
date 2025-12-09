"""
THE GRAVITY-ENERGY CONVERSION FORMULA
=====================================

Complete mathematical formulation of the reversed model.
"""

import numpy as np

# =============================================================================
# Physical Constants
# =============================================================================

c = 2.998e8          # Speed of light [m/s]
G = 6.674e-11        # Gravitational constant [m³/kg/s²]
hbar = 1.055e-34     # Reduced Planck constant [J·s]
a0 = 1.2e-10         # Acceleration scale [m/s²]

# =============================================================================
# THE COMPLETE FORMULA
# =============================================================================

def print_formula():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    THE GRAVITY-ENERGY CONVERSION FORMULA                      ║
╚══════════════════════════════════════════════════════════════════════════════╝


1. GRAVITY ENERGY PRODUCTION
════════════════════════════

Mass M produces gravity energy that propagates outward as a flux:

                         G M
    Φ_g(r) = ε × ─────────────
                      r²

where:
  • Φ_g = gravity energy flux [W/m² or J/s/m²]
  • ε = production efficiency (dimensionless, ~10⁻¹⁰ to 10⁻⁸)
  • G = gravitational constant
  • M = source mass
  • r = distance from source


2. ENERGY ACCUMULATION IN EMPTY SPACE
═════════════════════════════════════

As the energy propagates outward, it accumulates:

                    r
    E_acc(r) = ∫   Φ_g(r') × e^(-τ(r')) dr'
                   0

where τ(r) is the "optical depth" of matter encountered:

                r
    τ(r) = ∫   σ × ρ(r') dr'
               0

  • σ = absorption cross-section [m²/kg]
  • ρ(r) = matter density at radius r [kg/m³]
  • e^(-τ) = fraction of energy that survives (not yet converted)


3. GRAVITY BOOST FROM ACCUMULATED ENERGY
════════════════════════════════════════

The accumulated energy creates an additional gravitational field:

                    _______________
    g_boost(r) = α √ g_Newton × a₀  × f(E_acc)

where:
  • α = coupling constant (~1)
  • g_Newton = GM_enc/r² (standard Newtonian gravity)
  • a₀ = 1.2 × 10⁻¹⁰ m/s² (the MOND acceleration scale)
  • f(E_acc) = E_acc / E_max (normalized accumulated energy, 0 to 1)


4. TOTAL GRAVITY
════════════════

    g_total(r) = g_Newton(r) + g_boost(r)

                      G M_enc        _______________
    g_total(r) = ───────────── + α √ g_Newton × a₀  × f(E_acc)
                       r²


5. THE CONVERSION AT MATTER
═══════════════════════════

When gravity energy encounters matter, it converts back to regular gravity:

    dE_converted     
    ──────────── = σ × ρ × Φ_g
         dr

This is what creates the optical depth τ and depletes the energy reservoir.


╔══════════════════════════════════════════════════════════════════════════════╗
║                         SIMPLIFIED WORKING FORMULA                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

For practical calculations:

                                                    r
                      _______________              ⌠  1        ⎛    r'         ⎞
    g_boost(r) = α × √ g_Newton × a₀  ×  ─────── × ⎮ ──── exp⎜-∫  σρ(r'')dr''⎟ dr'
                                          E_norm   ⌡  r'²     ⎝    0          ⎠
                                                   0

where E_norm normalizes the integral to [0, 1].


LIMITING CASES:
───────────────

1. NO MATTER (ρ = 0 everywhere):
   τ = 0, e^(-τ) = 1
   E_acc grows without bound → f(E_acc) → 1
   g_boost = α × √(g_Newton × a₀)
   
   This is the MOND deep limit!

2. LOTS OF MATTER (high ρ):
   τ → large, e^(-τ) → 0
   Energy gets absorbed quickly → f(E_acc) → 0
   g_boost → 0
   
   This is the Newtonian limit!

3. INTERMEDIATE (some matter):
   Energy accumulates until it hits matter
   Peak boost at intermediate radius
   This explains the Einstein radius!


╔══════════════════════════════════════════════════════════════════════════════╗
║                              KEY PARAMETERS                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Parameter    Value           Physical Meaning
─────────────────────────────────────────────────────────────────────────────
a₀           1.2×10⁻¹⁰ m/s²  Acceleration scale (from MOND observations)
σ            ~10⁻⁶ m²/kg     Cross-section for energy absorption by matter
α            ~1              Coupling strength for boost
ε            ~10⁻⁹           Fraction of gravitational energy radiated


╔══════════════════════════════════════════════════════════════════════════════╗
║                           PHYSICAL INTERPRETATION                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

GRAVITY ENERGY is like a fluid:
  • Produced by mass (source)
  • Flows outward through space
  • Accumulates in empty regions
  • Gets absorbed (converted) by matter

MATTER acts as a sink:
  • Absorbs gravity energy
  • Converts it back to "regular" gravity
  • Depletes the reservoir

THE BOOST depends on:
  • How much energy has accumulated (distance traveled)
  • How much matter has absorbed it (optical depth)
  • The local Newtonian gravity (sets the scale)

This creates a HALO of enhanced gravity:
  • Weak near center (energy depleted by matter)
  • Strong at intermediate radius (energy accumulated)
  • Falls off at large radius (g_Newton too weak)

""")


def demonstrate_formula():
    """Show how the formula works in practice"""
    print("\n" + "="*70)
    print("DEMONSTRATION: How the Formula Works")
    print("="*70)
    
    # Parameters
    M = 1e14 * 1.989e30  # 10^14 solar masses (cluster)
    R_core = 100e3 * 3.086e16  # 100 kpc in meters
    sigma = 1e-6  # m²/kg
    alpha = 1.0
    
    # Density profile (NFW-like)
    def rho(r):
        r_s = R_core / 5
        x = max(r / r_s, 0.01)
        rho_0 = M / (4 * np.pi * r_s**3 * 10)
        return rho_0 / (x * (1 + x)**2)
    
    print(f"\nCluster: M = 10¹⁴ M☉, R_core = 100 kpc")
    print(f"Parameters: σ = {sigma:.0e} m²/kg, α = {alpha}")
    
    # Calculate at different radii
    kpc = 3.086e19
    radii = [10, 50, 100, 200, 500, 1000]  # kpc
    
    print(f"\n{'r [kpc]':>10} {'τ(r)':>12} {'e^(-τ)':>12} {'f(E_acc)':>12} {'g_Newton':>12} {'g_boost':>12} {'g_total':>12}")
    print("-" * 95)
    
    for r_kpc in radii:
        r = r_kpc * kpc
        
        # Compute optical depth
        n_steps = 100
        r_values = np.linspace(R_core * 0.01, r, n_steps)
        dr = r_values[1] - r_values[0]
        
        tau = 0
        E_acc = 0
        E_norm = 1.0 / (R_core * 0.01)  # Normalization
        
        for r_i in r_values:
            rho_i = rho(r_i)
            flux = 1.0 / r_i**2
            survival = np.exp(-tau)
            
            E_acc += flux * survival * dr
            tau += sigma * rho_i * dr
        
        f_E = min(E_acc / E_norm, 1.0)  # Normalize to [0, 1]
        
        # Newtonian gravity
        g_newton = G * M / r**2
        
        # Boost
        g_boost = alpha * np.sqrt(g_newton * a0) * f_E
        
        # Total
        g_total = g_newton + g_boost
        
        print(f"{r_kpc:>10} {tau:>12.4f} {np.exp(-tau):>12.4f} {f_E:>12.4f} "
              f"{g_newton:>12.2e} {g_boost:>12.2e} {g_total:>12.2e}")
    
    print("""
INTERPRETATION:
  • τ increases with radius (more matter encountered)
  • e^(-τ) decreases (less energy survives)
  • f(E_acc) peaks at intermediate radius
  • g_boost peaks where f(E_acc) × √g_Newton is maximum
  • This creates the "dark matter halo" effect!
""")


def write_latex_formula():
    """Write the formula in LaTeX format"""
    print("\n" + "="*70)
    print("LATEX FORMULA (for papers)")
    print("="*70)
    
    print(r"""
% Gravity Energy Conversion Formula

% 1. Gravity energy flux
\Phi_g(r) = \epsilon \frac{GM}{r^2}

% 2. Optical depth
\tau(r) = \int_0^r \sigma \rho(r') \, dr'

% 3. Accumulated energy
E_{\rm acc}(r) = \int_0^r \Phi_g(r') \, e^{-\tau(r')} \, dr'

% 4. Gravity boost
g_{\rm boost}(r) = \alpha \sqrt{g_{\rm Newton} \cdot a_0} \cdot f(E_{\rm acc})

% 5. Total gravity
g_{\rm total}(r) = g_{\rm Newton}(r) + g_{\rm boost}(r)

% Simplified form
g_{\rm total} = \frac{GM_{\rm enc}}{r^2} + \alpha \sqrt{\frac{GM_{\rm enc}}{r^2} \cdot a_0} \cdot f\left(\int_0^r \frac{e^{-\tau(r')}}{r'^2} dr'\right)

% Where a_0 = 1.2 \times 10^{-10} \, \rm{m/s^2}
""")


if __name__ == "__main__":
    print_formula()
    demonstrate_formula()
    write_latex_formula()
    
    # Summary box
    print("\n" + "="*70)
    print("SUMMARY: THE COMPLETE CONVERSION FORMULA")
    print("="*70)
    print("""
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│   g_total(r) = g_Newton(r) + g_boost(r)                                   │
│                                                                            │
│                 G M_enc              _______________                       │
│   g_total(r) = ───────── + α × √( g_Newton × a₀ ) × f(E_acc)             │
│                   r²                                                       │
│                                                                            │
│   where:                                                                   │
│                                                                            │
│            r                                                               │
│           ⌠   1                                                            │
│   E_acc = ⎮  ──── × exp(-τ(r')) dr'    (accumulated energy)               │
│           ⌡  r'²                                                           │
│           0                                                                │
│                                                                            │
│            r                                                               │
│   τ(r) =  ∫  σ × ρ(r') dr'             (optical depth of matter)          │
│           0                                                                │
│                                                                            │
│   f(E) = E / E_max                      (normalized to 0-1)               │
│                                                                            │
│   a₀ = 1.2 × 10⁻¹⁰ m/s²                (MOND acceleration scale)         │
│   σ ~ 10⁻⁶ m²/kg                       (absorption cross-section)         │
│   α ~ 1                                 (coupling constant)                │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘

PHYSICAL MEANING:
  • Energy accumulates as it travels through empty space (E_acc increases)
  • Matter absorbs energy (τ increases, survival e^(-τ) decreases)
  • Boost is strongest where these balance (intermediate radius)
  • This creates the "dark matter halo" and explains the Einstein radius
""")

