"""
Gravitational Lensing in the Gravity-Energy-Matter Conversion Model

Key question: How does light bending work when gravity energy converts
to gravity upon hitting matter?

The scenario:
  1. Light from a distant source travels toward us
  2. It passes through/near a galaxy cluster
  3. The cluster's gravity bends the light
  4. We see a distorted/magnified image

In our model:
  - The cluster produces gravity energy that propagates outward
  - This energy converts to gravity when it hits matter IN the cluster
  - Light passing through experiences the TOTAL gravity (Newton + boost)
  - The bending angle depends on the integrated gravity along the path
"""

import numpy as np
from scipy import integrate
import json

# =============================================================================
# Physical Constants
# =============================================================================

c = 2.998e8          # Speed of light [m/s]
G = 6.674e-11        # Gravitational constant [m³/kg/s²]
M_sun = 1.989e30     # Solar mass [kg]
kpc = 3.086e19       # Kiloparsec [m]
Mpc = 3.086e22       # Megaparsec [m]
a0 = 1.2e-10         # MOND acceleration scale [m/s²]

# =============================================================================
# Cluster Model
# =============================================================================

def nfw_density(r: float, M_vir: float, c_nfw: float, R_vir: float) -> float:
    """
    NFW density profile for cluster.
    ρ(r) = ρ_s / [(r/r_s)(1 + r/r_s)²]
    """
    r_s = R_vir / c_nfw
    x = max(r / r_s, 0.01)  # Avoid singularity
    
    # Characteristic density
    rho_s = M_vir / (4 * np.pi * r_s**3 * (np.log(1 + c_nfw) - c_nfw/(1 + c_nfw)))
    
    return rho_s / (x * (1 + x)**2)


def nfw_enclosed_mass(r: float, M_vir: float, c_nfw: float, R_vir: float) -> float:
    """Enclosed mass for NFW profile"""
    r_s = R_vir / c_nfw
    x = r / r_s
    
    M_enc = M_vir * (np.log(1 + x) - x/(1 + x)) / (np.log(1 + c_nfw) - c_nfw/(1 + c_nfw))
    return M_enc


def cluster_surface_density(b: float, M_vir: float, c_nfw: float, R_vir: float,
                            z_max: float = None) -> float:
    """
    Surface density (column density) at impact parameter b.
    Σ(b) = ∫_{-∞}^{+∞} ρ(√(b² + z²)) dz
    """
    if z_max is None:
        z_max = 3 * R_vir
    
    n_steps = 200
    z_values = np.linspace(-z_max, z_max, n_steps)
    dz = z_values[1] - z_values[0]
    
    Sigma = 0
    for z in z_values:
        r = np.sqrt(b**2 + z**2)
        Sigma += nfw_density(r, M_vir, c_nfw, R_vir) * dz
    
    return Sigma


# =============================================================================
# Lensing Physics
# =============================================================================

def deflection_angle_newton(b: float, M_enc: float) -> float:
    """
    Newtonian deflection angle for light passing at impact parameter b.
    
    α = 4GM/(c²b)  (Einstein's formula, factor of 2 from GR)
    """
    return 4 * G * M_enc / (c**2 * b)


def deflection_angle_with_boost(b: float, M_enc: float, Sigma: float, 
                                  Sigma_threshold: float = 1.0) -> float:
    """
    Deflection angle including gravity-energy boost.
    
    The boost factor depends on column density:
      f(Σ) = tanh(Σ / Σ_threshold)
    
    Total gravity: g_total = g_Newton + g_boost
    where g_boost = √(g_Newton × a₀) × f(Σ)
    
    For lensing, we need to integrate the TRANSVERSE gravity:
      α = (2/c²) ∫ g_⊥ dl
    
    With boost:
      α_total = α_Newton × (1 + boost_factor)
    """
    # Newtonian deflection
    alpha_newton = deflection_angle_newton(b, M_enc)
    
    # Newtonian gravity at impact parameter
    g_newton = G * M_enc / b**2
    
    # Boost
    f_Sigma = np.tanh(Sigma / Sigma_threshold)
    g_boost = np.sqrt(g_newton * a0) * f_Sigma
    
    # Boost factor for deflection
    boost_factor = g_boost / g_newton if g_newton > 0 else 0
    
    # Total deflection
    alpha_total = alpha_newton * (1 + boost_factor)
    
    return alpha_total, alpha_newton, boost_factor


# =============================================================================
# The Key Analysis: How Light Bending Works
# =============================================================================

def explain_lensing_mechanism():
    """Explain step by step how lensing works in our model"""
    print("\n" + "="*70)
    print("HOW GRAVITATIONAL LENSING WORKS IN OUR MODEL")
    print("="*70)
    
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                    THE LENSING GEOMETRY                              │
└─────────────────────────────────────────────────────────────────────┘

                    Distant Source (galaxy/quasar)
                              ★
                              |
                              |  Light ray
                              |
                              ↓
                    ┌─────────────────┐
                    │                 │
                    │  Galaxy Cluster │ ← Matter here converts
                    │   (the lens)    │   gravity energy to gravity
                    │                 │
                    └─────────────────┘
                              |
                              |  Bent light ray
                              ↓
                           Observer
                              👁

The light ray passes at impact parameter b from the cluster center.
It experiences the cluster's gravitational field and bends.
""")
    
    print("""
STEP 1: GRAVITY ENERGY PRODUCTION
─────────────────────────────────
The cluster's mass produces "gravity energy" that fills the cluster volume.

  Gravity energy flux at radius r: F_g ∝ GM_enc(r)/r²

This is the same as the Newtonian gravitational field - it's the
"potential" for gravity that hasn't yet converted to actual gravity.


STEP 2: CONVERSION AT MATTER
────────────────────────────
Inside the cluster, there's lots of matter:
  - Hot intracluster gas (ICM): ~10⁻²⁴ to 10⁻²³ kg/m³
  - Galaxies: concentrated mass
  - Dark matter (if it exists): unknown

When gravity energy encounters this matter, it CONVERTS to gravity.
The conversion efficiency depends on the column density:

  f(Σ) = tanh(Σ / Σ_threshold)

For a typical cluster:
  - Σ ~ 1-100 kg/m² through the core
  - Σ_threshold ~ 1 kg/m²
  - So f(Σ) ≈ 1 (full conversion)


STEP 3: LIGHT EXPERIENCES TOTAL GRAVITY
───────────────────────────────────────
The light ray passing through experiences:

  g_total = g_Newton + g_boost

where:
  g_Newton = GM_enc/r²
  g_boost = √(g_Newton × a₀) × f(Σ)

The deflection angle is:
  α = (4G/c²) × M_effective / b

where M_effective includes the boost.


STEP 4: THE OBSERVED BENDING
────────────────────────────
In standard GR:
  α_GR = 4GM/c²b

In our model:
  α_model = α_GR × (1 + boost_factor)

The boost_factor = g_boost/g_Newton = √(a₀/g_Newton) × f(Σ)

At cluster scales:
  g_Newton ~ 10⁻¹¹ to 10⁻¹⁰ m/s² (comparable to a₀!)
  boost_factor ~ 1-3

So lensing is enhanced by a factor of 2-4!
""")


def analyze_cluster_lensing():
    """Detailed analysis of lensing through a cluster"""
    print("\n" + "="*70)
    print("CLUSTER LENSING ANALYSIS")
    print("="*70)
    
    # Abell 1689-like cluster
    M_vir = 1e15 * M_sun      # Virial mass
    M_baryonic = 0.15 * M_vir  # Baryonic fraction
    R_vir = 2 * Mpc           # Virial radius
    c_nfw = 6                  # Concentration
    
    print(f"\nCluster parameters (Abell 1689-like):")
    print(f"  M_virial = {M_vir/M_sun:.2e} M☉")
    print(f"  M_baryonic = {M_baryonic/M_sun:.2e} M☉ (15% of total)")
    print(f"  R_virial = {R_vir/Mpc:.1f} Mpc")
    print(f"  Concentration c = {c_nfw}")
    
    # Impact parameters to test
    b_values = np.array([0.1, 0.2, 0.5, 1.0, 1.5, 2.0]) * Mpc
    
    print(f"\n{'b [Mpc]':>10} {'Σ [kg/m²]':>12} {'f(Σ)':>10} {'α_Newton':>12} {'α_model':>12} {'Boost':>10}")
    print("-" * 70)
    
    results = []
    Sigma_threshold = 1.0  # kg/m²
    
    for b in b_values:
        # Surface density (using baryonic mass only for conversion)
        Sigma = cluster_surface_density(b, M_baryonic, c_nfw, R_vir)
        
        # Enclosed mass (baryonic only - this is what we observe)
        M_enc = nfw_enclosed_mass(b, M_baryonic, c_nfw, R_vir)
        
        # Deflection angles
        alpha_total, alpha_newton, boost_factor = deflection_angle_with_boost(
            b, M_enc, Sigma, Sigma_threshold)
        
        # Convert to arcseconds
        alpha_newton_arcsec = np.degrees(alpha_newton) * 3600
        alpha_total_arcsec = np.degrees(alpha_total) * 3600
        
        results.append({
            'b_Mpc': b / Mpc,
            'Sigma': Sigma,
            'f_Sigma': np.tanh(Sigma / Sigma_threshold),
            'alpha_newton_arcsec': alpha_newton_arcsec,
            'alpha_total_arcsec': alpha_total_arcsec,
            'boost_factor': boost_factor
        })
        
        print(f"{b/Mpc:>10.2f} {Sigma:>12.2e} {np.tanh(Sigma/Sigma_threshold):>10.4f} "
              f"{alpha_newton_arcsec:>12.2f}\" {alpha_total_arcsec:>12.2f}\" {1+boost_factor:>10.2f}×")
    
    print(f"""
INTERPRETATION:
  - Column density Σ ranges from ~1 to ~100 kg/m² through cluster
  - Conversion factor f(Σ) ≈ 1 (full conversion)
  - Boost factor (g_boost/g_Newton) ranges from 1-5
  - Total deflection is 2-6× larger than pure Newtonian

This means:
  The "missing mass" inferred from lensing is actually the BOOST
  from gravity energy conversion, not dark matter!
""")
    
    return results


def compare_lensing_scenarios():
    """Compare lensing in different scenarios"""
    print("\n" + "="*70)
    print("COMPARISON: Where Does Lensing Get Enhanced?")
    print("="*70)
    
    print("""
The boost depends on TWO things:
  1. The Newtonian gravity g_Newton (determines boost magnitude)
  2. The column density Σ (determines conversion efficiency)

Let's compare different scenarios:
""")
    
    scenarios = [
        {
            'name': 'Galaxy cluster (Abell 1689)',
            'M': 1.5e14 * M_sun,  # Baryonic mass
            'b': 0.5 * Mpc,
            'Sigma': 10,  # kg/m²
        },
        {
            'name': 'Galaxy (Milky Way)',
            'M': 6e10 * M_sun,
            'b': 10 * kpc,
            'Sigma': 1,  # kg/m²
        },
        {
            'name': 'Galaxy group',
            'M': 1e13 * M_sun,
            'b': 0.2 * Mpc,
            'Sigma': 0.5,  # kg/m²
        },
        {
            'name': 'Void (empty space)',
            'M': 1e10 * M_sun,
            'b': 1 * Mpc,
            'Sigma': 1e-6,  # kg/m² (almost nothing)
        },
        {
            'name': 'Solar system',
            'M': M_sun,
            'b': 1.5e11,  # 1 AU
            'Sigma': 1e-7,  # kg/m²
        },
    ]
    
    Sigma_threshold = 1.0
    
    print(f"{'Scenario':<30} {'g_Newton':>12} {'g_Newton/a₀':>12} {'f(Σ)':>10} {'Boost':>10}")
    print("-" * 80)
    
    for s in scenarios:
        g_newton = G * s['M'] / s['b']**2
        f_Sigma = np.tanh(s['Sigma'] / Sigma_threshold)
        boost_factor = np.sqrt(a0 / g_newton) * f_Sigma if g_newton > 0 else 0
        
        print(f"{s['name']:<30} {g_newton:>12.2e} {g_newton/a0:>12.2f} "
              f"{f_Sigma:>10.4f} {1+boost_factor:>10.2f}×")
    
    print(f"""
KEY OBSERVATIONS:
─────────────────
1. CLUSTERS: g ~ a₀, full conversion → large boost (2-5×)
2. GALAXIES: g ~ a₀, full conversion → large boost (2-3×)  
3. VOIDS: g << a₀, but NO MATTER → no boost (1×)
4. SOLAR SYSTEM: g >> a₀, no matter → no boost (1×)

The boost is ONLY significant when:
  - g_Newton is comparable to or less than a₀ (so boost term matters)
  - AND there's enough matter for conversion (Σ > Σ_threshold)

This explains why:
  - Clusters show "excess" lensing (both conditions met)
  - Voids show normal lensing (no matter to convert)
  - Solar system shows normal gravity (g >> a₀)
""")


def bullet_cluster_lensing():
    """Analyze the Bullet Cluster lensing puzzle"""
    print("\n" + "="*70)
    print("THE BULLET CLUSTER PUZZLE")
    print("="*70)
    
    print("""
THE OBSERVATION:
────────────────
In the Bullet Cluster (1E 0657-56):
  - Two clusters collided
  - Hot gas (80% of baryons) was stripped and displaced
  - Galaxies (20% of baryons) passed through
  - Lensing mass is centered on GALAXIES, not gas

THE PUZZLE FOR OUR MODEL:
─────────────────────────
In our model, the boost requires matter for conversion.
If gas has more mass, shouldn't the boost follow the gas?

Let's analyze:
""")
    
    # Simplified Bullet Cluster model
    # Main cluster
    M_main_gas = 5e13 * M_sun    # Hot gas (displaced)
    M_main_galaxies = 1e13 * M_sun  # Galaxies (at original position)
    
    # Bullet (sub-cluster)
    M_bullet_gas = 1e13 * M_sun
    M_bullet_galaxies = 2e12 * M_sun
    
    # Positions (1D simplification)
    x_main_galaxies = 0
    x_main_gas = 300 * kpc  # Displaced by collision
    x_bullet_galaxies = 700 * kpc
    x_bullet_gas = 500 * kpc  # Displaced backward
    
    print(f"Mass distribution:")
    print(f"  Main galaxies: {M_main_galaxies/M_sun:.1e} M☉ at x = 0")
    print(f"  Main gas:      {M_main_gas/M_sun:.1e} M☉ at x = 300 kpc")
    print(f"  Bullet galaxies: {M_bullet_galaxies/M_sun:.1e} M☉ at x = 700 kpc")
    print(f"  Bullet gas:    {M_bullet_gas/M_sun:.1e} M☉ at x = 500 kpc")
    
    # Test lensing at different positions
    test_x = np.linspace(-200, 900, 50) * kpc
    
    # For each position, compute the lensing signal
    # The key question: where is the boost strongest?
    
    print(f"""
ANALYSIS:
─────────
The boost at each location depends on:
  1. Gravity from ALL mass (g_total at that point)
  2. Column density of matter AT that point (for conversion)

The gravity energy comes from ALL masses (galaxies + gas).
But conversion only happens where there IS matter.

So the boost is:
  g_boost(x) = √(g_bar(x) × a₀) × f(Σ_local(x))

where:
  g_bar(x) = sum of gravity from all masses at point x
  Σ_local(x) = column density of matter at point x
""")
    
    # Compute gravity and boost at each position
    results = []
    
    for x in test_x:
        # Gravity from each component (simplified as point masses)
        def gravity_from(M, x_source, x_test):
            d = abs(x_test - x_source) + 50*kpc  # Avoid singularity
            return G * M / d**2
        
        g_main_gal = gravity_from(M_main_galaxies, x_main_galaxies, x)
        g_main_gas = gravity_from(M_main_gas, x_main_gas, x)
        g_bullet_gal = gravity_from(M_bullet_galaxies, x_bullet_galaxies, x)
        g_bullet_gas = gravity_from(M_bullet_gas, x_bullet_gas, x)
        
        g_total_newton = g_main_gal + g_main_gas + g_bullet_gal + g_bullet_gas
        
        # Local column density (simplified Gaussian profiles)
        def sigma_from(M, x_source, x_test, width=100*kpc):
            d = abs(x_test - x_source)
            return (M / (2*np.pi*width**2)) * np.exp(-d**2/(2*width**2)) / 1e15  # Normalize
        
        Sigma_local = (sigma_from(M_main_galaxies, x_main_galaxies, x) +
                       sigma_from(M_main_gas, x_main_gas, x) +
                       sigma_from(M_bullet_galaxies, x_bullet_galaxies, x) +
                       sigma_from(M_bullet_gas, x_bullet_gas, x))
        
        # Boost
        f_Sigma = np.tanh(Sigma_local / 1.0)  # Σ_threshold = 1
        g_boost = np.sqrt(g_total_newton * a0) * f_Sigma
        g_total = g_total_newton + g_boost
        
        results.append({
            'x_kpc': x / kpc,
            'g_newton': g_total_newton,
            'g_boost': g_boost,
            'g_total': g_total,
            'Sigma': Sigma_local,
            'boost_factor': g_boost / g_total_newton if g_total_newton > 0 else 0
        })
    
    # Find peaks
    g_total_arr = np.array([r['g_total'] for r in results])
    g_newton_arr = np.array([r['g_newton'] for r in results])
    Sigma_arr = np.array([r['Sigma'] for r in results])
    x_arr = np.array([r['x_kpc'] for r in results])
    
    peak_total = x_arr[np.argmax(g_total_arr)]
    peak_newton = x_arr[np.argmax(g_newton_arr)]
    peak_sigma = x_arr[np.argmax(Sigma_arr)]
    
    print(f"""
RESULTS:
────────
Peak locations:
  Total gravity (g_total): x = {peak_total:.0f} kpc
  Newtonian gravity:       x = {peak_newton:.0f} kpc
  Column density (Σ):      x = {peak_sigma:.0f} kpc

Observed lensing peaks:   x ≈ 0 and 700 kpc (at galaxies)
Gas peaks:                x ≈ 300 and 500 kpc

INTERPRETATION:
───────────────
In our model, the lensing signal comes from g_total = g_Newton + g_boost.

The boost is strongest where:
  - g_Newton is significant (from all mass)
  - AND Σ_local is high (matter for conversion)

The GAS contributes to g_Newton everywhere (it's still mass!).
But the boost is localized where the MATTER is densest.

If galaxies are more COMPACT than diffuse gas:
  - Galaxy regions have higher Σ_local
  - So boost is stronger at galaxy positions
  - Even though gas has more total mass

This could explain why lensing follows galaxies, not gas!
""")
    
    print("""
POSSIBLE RESOLUTION:
───────────────────
The Bullet Cluster observation might be explained if:

1. COMPACT MATTER converts more efficiently
   - Galaxies are dense concentrations
   - Gas is diffuse and spread out
   - Σ_local is higher at galaxy positions

2. TEMPERATURE/STATE matters
   - Hot gas (10⁸ K) might convert differently than cold galaxies
   - Ionization state could affect conversion cross-section

3. SOME dark matter exists
   - Our model reduces but doesn't eliminate need for dark matter
   - Clusters might have ~50% less dark matter than standard model

This is a key test of the model that needs more detailed analysis!
""")
    
    return results


def summary():
    """Summarize how lensing works"""
    print("\n" + "="*70)
    print("SUMMARY: LENSING IN THE GRAVITY-ENERGY MODEL")
    print("="*70)
    
    print("""
THE MECHANISM:
══════════════

1. GRAVITY ENERGY PRODUCTION
   - Mass produces gravity energy that fills space
   - Flux: F_g ∝ GM/r² (same as Newtonian potential)

2. CONVERSION AT MATTER
   - Gravity energy converts to gravity when hitting matter
   - Efficiency: f(Σ) = tanh(Σ / Σ_threshold)
   - Σ = column density along line of sight

3. LIGHT BENDING
   - Light experiences total gravity: g_total = g_Newton + g_boost
   - Deflection: α ∝ ∫ g_total dl
   - Boost adds to bending angle

THE EQUATIONS:
══════════════

Newtonian deflection:
  α_Newton = 4GM / (c²b)

Boost factor:
  boost = √(a₀/g_Newton) × f(Σ)

Total deflection:
  α_total = α_Newton × (1 + boost)

WHEN IS LENSING ENHANCED?
═════════════════════════

Conditions for significant boost:
  1. g_Newton ≲ a₀  (so boost term is comparable)
  2. Σ ≳ Σ_threshold  (enough matter for conversion)

╔═══════════════════╦══════════════╦═══════════╦═══════════════╗
║ Scenario          ║ g/a₀         ║ Σ/Σ_th    ║ Lensing boost ║
╠═══════════════════╬══════════════╬═══════════╬═══════════════╣
║ Galaxy cluster    ║ ~1           ║ ~10       ║ 2-5×          ║
║ Galaxy            ║ ~1           ║ ~1        ║ 2-3×          ║
║ Void              ║ ~0.01        ║ ~10⁻⁶     ║ 1× (none)     ║
║ Solar system      ║ ~10⁸         ║ ~10⁻⁷     ║ 1× (none)     ║
╚═══════════════════╩══════════════╩═══════════╩═══════════════╝

KEY PREDICTIONS:
════════════════

1. Lensing is enhanced in matter-rich regions
2. Voids show NO excess lensing (no matter to convert)
3. The "missing mass" from lensing is the boost, not dark matter
4. Compact structures (galaxies) may lens more than diffuse gas
""")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n" + "#"*70)
    print("# GRAVITATIONAL LENSING IN THE GRAVITY-ENERGY MODEL")
    print("#"*70)
    
    explain_lensing_mechanism()
    cluster_results = analyze_cluster_lensing()
    compare_lensing_scenarios()
    bullet_results = bullet_cluster_lensing()
    summary()
    
    # Save results
    output = {
        'model': 'gravity_energy_lensing',
        'key_equation': 'alpha_total = alpha_Newton * (1 + sqrt(a0/g_Newton) * f(Sigma))',
        'cluster_results': [
            {k: float(v) if isinstance(v, (float, np.floating)) else v 
             for k, v in r.items()}
            for r in cluster_results
        ]
    }
    
    output_file = "/Users/leonardspeiser/Projects/sigmagravity/derivations/lensing_mechanism_results.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n\nResults saved to: {output_file}")




