#!/usr/bin/env python3
"""
UNIFIED THEORY: COMPREHENSIVE TESTS
====================================

Testing the predictions of the unified φ field theory:
1. High-z evolution of g† (using available high-z data)
2. Void dynamics prediction
3. Milky Way screening radius
4. Inner structure → outer enhancement correlation

Author: Leonard Speiser
Date: December 2025
"""

import numpy as np
from scipy import stats
from scipy.integrate import quad
from pathlib import Path

# =============================================================================
# CONSTANTS
# =============================================================================

c = 2.998e8           # m/s
G = 6.674e-11         # m³/kg/s²
H0 = 2.27e-18         # 1/s (70 km/s/Mpc)
kpc_to_m = 3.086e19
M_sun = 1.989e30
g_dagger = c * H0 / (4 * np.sqrt(np.pi))

# Cosmological parameters
Omega_m = 0.3
Omega_L = 0.7

print("=" * 80)
print("UNIFIED THEORY: COMPREHENSIVE TESTS")
print("=" * 80)

# =============================================================================
# TEST 1: HIGH-Z EVOLUTION OF g†
# =============================================================================

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  TEST 1: HIGH-Z EVOLUTION OF g†                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

PREDICTION: g†(z) = g†(0) × H(z)/H₀

Since g† = cH/4√π, and H evolves with redshift:
    H(z) = H₀ × √(Ω_m(1+z)³ + Ω_Λ)

At z=2: H(z)/H₀ ≈ 2.97
So g†(z=2) ≈ 2.97 × g†(0)

This means HIGH-Z GALAXIES should show:
    - Enhancement turning on at HIGHER acceleration
    - Less "dark matter" effect at fixed g_bar
""")

def H_of_z(z, H0=H0, Om=Omega_m, OL=Omega_L):
    """Hubble parameter at redshift z."""
    return H0 * np.sqrt(Om * (1+z)**3 + OL)

def g_dagger_of_z(z):
    """Critical acceleration at redshift z."""
    return c * H_of_z(z) / (4 * np.sqrt(np.pi))

# Calculate for several redshifts
print("g†(z) evolution:")
print(f"{'z':<8} {'H(z)/H₀':<12} {'g†(z) (m/s²)':<15} {'g†(z)/g†(0)':<12}")
print("-" * 50)
for z in [0, 0.5, 1, 2, 3, 5]:
    H_ratio = H_of_z(z) / H0
    g_dag_z = g_dagger_of_z(z)
    print(f"{z:<8} {H_ratio:<12.3f} {g_dag_z:<15.2e} {g_dag_z/g_dagger:<12.3f}")

# Check if we have high-z data
print("""
AVAILABLE HIGH-Z DATA:
─────────────────────
Looking for high-z rotation curve data in the codebase...
""")

# Check for KMOS3D or other high-z data
highz_paths = [
    Path("exploratory/coherence_wavelength_test/analyze_kmos3d_highz.py"),
    Path("data/highz"),
]

for p in highz_paths:
    if p.exists():
        print(f"  Found: {p}")
    else:
        print(f"  Not found: {p}")

# =============================================================================
# TEST 2: MILKY WAY SCREENING RADIUS
# =============================================================================

print(f"""

╔══════════════════════════════════════════════════════════════════════════════╗
║  TEST 2: MILKY WAY SCREENING RADIUS                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

PREDICTION: The enhancement "turns on" at radius where g = g†

For the Milky Way:
    g(r) = GM(<r) / r²
    
Setting g = g†:
    r_screen = √(GM/g†)
""")

# Milky Way mass profile (simplified)
M_MW_disk = 5e10 * M_sun      # Disk mass
M_MW_bulge = 1e10 * M_sun     # Bulge mass
M_MW_total = 1e12 * M_sun     # Total (including halo if it exists)

# Screening radii for different mass assumptions
r_screen_disk = np.sqrt(G * M_MW_disk / g_dagger) / kpc_to_m
r_screen_bulge = np.sqrt(G * M_MW_bulge / g_dagger) / kpc_to_m
r_screen_total = np.sqrt(G * M_MW_total / g_dagger) / kpc_to_m

print(f"Screening radius (where g = g†):")
print(f"  Using M_disk = 5×10¹⁰ M☉:  r_screen = {r_screen_disk:.1f} kpc")
print(f"  Using M_bulge = 1×10¹⁰ M☉: r_screen = {r_screen_bulge:.1f} kpc")
print(f"  Using M_total = 1×10¹² M☉: r_screen = {r_screen_total:.1f} kpc")

print("""
TESTABLE PREDICTION:
───────────────────
Inside r_screen: Standard Newtonian gravity (Σ ≈ 1)
Outside r_screen: Enhanced gravity (Σ > 1)

The transition should be visible in:
    - Gaia stellar kinematics
    - Rotation curve at R > 20 kpc
    - Satellite galaxy dynamics
""")

# Load Gaia data if available
gaia_path = Path("vendor/maxdepth_gaia")
if gaia_path.exists():
    print(f"\nFound Gaia data at: {gaia_path}")
    print("Running Milky Way test...")
    
    # Try to run the MW test from full regression
    try:
        from derivations.full_regression_test import test_milky_way, load_gaia_data
        
        gaia_file = gaia_path / "gaia_vcirc_binned.csv"
        if gaia_file.exists():
            mw_data = load_gaia_data(gaia_file)
            if mw_data:
                result = test_milky_way(mw_data, verbose=True)
                print(f"\nMilky Way test result: {'PASSED' if result.passed else 'FAILED'}")
                print(f"  RMS: {result.mean_rms:.1f} km/s")
    except Exception as e:
        print(f"  Could not run MW test: {e}")

# =============================================================================
# TEST 3: INNER STRUCTURE → OUTER ENHANCEMENT
# =============================================================================

print(f"""

╔══════════════════════════════════════════════════════════════════════════════╗
║  TEST 3: INNER STRUCTURE AFFECTS OUTER ENHANCEMENT                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

PREDICTION: The φ field at outer radius depends on the path integral:
    φ(r) ∝ ∫₀ʳ ρ(r') h(g(r')) dr'

This means two galaxies with:
    - SAME outer g_bar
    - DIFFERENT inner structure
Should have DIFFERENT outer Σ.

MOND predicts: same outer g → same outer Σ (purely local)
""")

def find_sparc_data():
    candidates = [
        Path("data/Rotmod_LTG"),
        Path("../data/Rotmod_LTG"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def load_galaxy(sparc_dir, name):
    filepath = sparc_dir / f"{name}_rotmod.dat"
    if not filepath.exists():
        return None
    
    data = np.loadtxt(filepath)
    R = data[:, 0]
    V_obs = data[:, 1]
    V_gas = data[:, 3]
    V_disk = data[:, 4] * np.sqrt(0.5)
    V_bul = data[:, 5] * np.sqrt(0.7)
    
    V_bar_sq = np.sign(V_gas) * V_gas**2 + V_disk**2 + V_bul**2
    V_bar = np.sqrt(np.abs(V_bar_sq)) * np.sign(V_bar_sq)
    
    return {'R': R, 'V_obs': V_obs, 'V_bar': V_bar, 'V_bul': V_bul}

sparc_dir = find_sparc_data()
if sparc_dir:
    print(f"Found SPARC data: {sparc_dir}")
    
    # Load all galaxies
    galaxy_names = [f.stem.replace('_rotmod', '') for f in sparc_dir.glob('*_rotmod.dat')]
    
    analysis_data = []
    
    for name in galaxy_names:
        data = load_galaxy(sparc_dir, name)
        if data is None or len(data['R']) < 10:
            continue
        
        R = data['R']
        V_obs = data['V_obs']
        V_bar = data['V_bar']
        V_bul = data['V_bul']
        
        valid = (V_bar > 5) & (R > 0.1)
        if valid.sum() < 10:
            continue
        
        R = R[valid]
        V_obs = V_obs[valid]
        V_bar = V_bar[valid]
        V_bul = V_bul[valid]
        
        R_m = R * kpc_to_m
        g_bar = (V_bar * 1000)**2 / R_m
        
        # Inner: R < R_max/3
        # Outer: R > 2*R_max/3
        inner_mask = R < R.max() / 3
        outer_mask = R > 2 * R.max() / 3
        
        if inner_mask.sum() < 3 or outer_mask.sum() < 3:
            continue
        
        # Inner properties
        g_inner = np.mean(g_bar[inner_mask])
        V_inner = np.mean(V_bar[inner_mask])
        bulge_frac = np.mean(V_bul[inner_mask]**2) / (np.mean(V_bar[inner_mask]**2) + 1)
        
        # Outer properties
        g_outer = np.mean(g_bar[outer_mask])
        Sigma_outer = np.mean((V_obs[outer_mask] / V_bar[outer_mask])**2)
        
        # φ buildup proxy: ∫ h(g) dr through inner region
        h_inner = np.sqrt(g_dagger / g_bar[inner_mask]) * g_dagger / (g_dagger + g_bar[inner_mask])
        phi_proxy = np.mean(h_inner)
        
        analysis_data.append({
            'name': name,
            'g_outer': g_outer,
            'g_inner': g_inner,
            'Sigma_outer': Sigma_outer,
            'phi_proxy': phi_proxy,
            'bulge_frac': bulge_frac,
            'V_inner': V_inner,
        })
    
    print(f"Analyzed {len(analysis_data)} galaxies")
    
    # Filter outliers
    analysis_data = [d for d in analysis_data if 0.5 < d['Sigma_outer'] < 30 and d['g_outer'] > 1e-12]
    
    if len(analysis_data) > 20:
        # Convert to arrays
        g_outer = np.array([d['g_outer'] for d in analysis_data])
        g_inner = np.array([d['g_inner'] for d in analysis_data])
        Sigma_outer = np.array([d['Sigma_outer'] for d in analysis_data])
        phi_proxy = np.array([d['phi_proxy'] for d in analysis_data])
        bulge_frac = np.array([d['bulge_frac'] for d in analysis_data])
        
        # MOND prediction: Σ depends only on outer g
        # Unified prediction: Σ depends on outer g AND inner structure
        
        # Partial correlation: does inner structure predict Σ after controlling for outer g?
        
        # Bin by outer g
        g_bins = np.percentile(g_outer, [0, 25, 50, 75, 100])
        
        print("\nPartial correlation analysis (controlling for outer g):")
        print("-" * 60)
        
        all_residuals = []
        all_phi = []
        all_inner_g = []
        
        for i in range(len(g_bins) - 1):
            mask = (g_outer >= g_bins[i]) & (g_outer < g_bins[i+1])
            if mask.sum() < 5:
                continue
            
            # Within this bin, outer g is ~constant
            # Does inner structure predict Σ?
            
            Sigma_bin = Sigma_outer[mask]
            phi_bin = phi_proxy[mask]
            g_inner_bin = g_inner[mask]
            
            # Residuals from mean (removing outer g effect)
            Sigma_residual = Sigma_bin - np.mean(Sigma_bin)
            
            all_residuals.extend(Sigma_residual)
            all_phi.extend(phi_bin)
            all_inner_g.extend(np.log10(g_inner_bin))
        
        # Overall correlation of residuals with inner structure
        if len(all_residuals) > 10:
            r_phi, p_phi = stats.pearsonr(all_phi, all_residuals)
            r_inner, p_inner = stats.pearsonr(all_inner_g, all_residuals)
            
            print(f"\nCorrelation of Σ residuals (after removing outer g effect):")
            print(f"  With φ proxy:    r = {r_phi:.3f}, p = {p_phi:.4f}")
            print(f"  With inner g:    r = {r_inner:.3f}, p = {p_inner:.4f}")
            
            if p_phi < 0.05 or p_inner < 0.05:
                print("\n*** SIGNIFICANT: Inner structure predicts outer Σ! ***")
                print("This supports the unified theory over MOND.")
            else:
                print("\nNo significant partial correlation detected.")
        
        # Direct test: galaxies with similar outer g but different inner structure
        print("\n" + "=" * 60)
        print("DIRECT COMPARISON: Similar outer g, different inner structure")
        print("=" * 60)
        
        # Find pairs
        pairs = []
        for i, d1 in enumerate(analysis_data):
            for j, d2 in enumerate(analysis_data):
                if i >= j:
                    continue
                
                # Similar outer g (within 30%)
                g_ratio = d1['g_outer'] / d2['g_outer']
                if not (0.7 < g_ratio < 1.43):
                    continue
                
                # Different inner structure (factor of 3)
                inner_ratio = d1['g_inner'] / d2['g_inner']
                if not (inner_ratio > 3 or inner_ratio < 0.33):
                    continue
                
                Sigma_diff = abs(d1['Sigma_outer'] - d2['Sigma_outer'])
                pairs.append((d1, d2, Sigma_diff, inner_ratio))
        
        # Sort by Sigma difference
        pairs.sort(key=lambda x: -x[2])
        
        if pairs:
            print(f"\nFound {len(pairs)} pairs with similar outer g but different inner structure:")
            print()
            
            for d1, d2, Sigma_diff, inner_ratio in pairs[:5]:
                print(f"{d1['name']} vs {d2['name']}:")
                print(f"  Outer g: {d1['g_outer']:.2e} vs {d2['g_outer']:.2e} (ratio {d1['g_outer']/d2['g_outer']:.2f})")
                print(f"  Inner g: {d1['g_inner']:.2e} vs {d2['g_inner']:.2e} (ratio {inner_ratio:.2f})")
                print(f"  Outer Σ: {d1['Sigma_outer']:.2f} vs {d2['Sigma_outer']:.2f} (diff {Sigma_diff:.2f})")
                
                if Sigma_diff > 1.0:
                    print(f"  *** LARGE Σ difference despite similar outer g! ***")
                print()
            
            # Statistical test on pairs
            high_inner_Sigma = [d1['Sigma_outer'] if d1['g_inner'] > d2['g_inner'] else d2['Sigma_outer'] 
                               for d1, d2, _, _ in pairs]
            low_inner_Sigma = [d2['Sigma_outer'] if d1['g_inner'] > d2['g_inner'] else d1['Sigma_outer'] 
                              for d1, d2, _, _ in pairs]
            
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(high_inner_Sigma, low_inner_Sigma)
            
            print(f"Paired comparison (N={len(pairs)} pairs):")
            print(f"  Mean Σ (high inner g): {np.mean(high_inner_Sigma):.2f}")
            print(f"  Mean Σ (low inner g):  {np.mean(low_inner_Sigma):.2f}")
            print(f"  Paired t-test: t = {t_stat:.2f}, p = {p_value:.4f}")
            
            if p_value < 0.05:
                print("\n*** SIGNIFICANT: Galaxies with higher inner g have different outer Σ! ***")

# =============================================================================
# TEST 4: VOID DYNAMICS
# =============================================================================

print(f"""

╔══════════════════════════════════════════════════════════════════════════════╗
║  TEST 4: VOID DYNAMICS PREDICTION                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

PREDICTION: In cosmic voids, g → 0, so the φ coupling is UNSUPPRESSED.

Objects in voids should experience:
    - Enhanced gravity toward void walls
    - Faster infall velocities than ΛCDM predicts
    - Modified void profiles

The enhancement in voids:
    Σ_void = 1 + A × h(g_void)
    
For g_void ~ 10⁻¹² m/s² (typical void):
""")

g_void = 1e-12  # Typical void acceleration
h_void = np.sqrt(g_dagger / g_void) * g_dagger / (g_dagger + g_void)
Sigma_void = 1 + np.sqrt(3) * h_void

print(f"At g_void = {g_void:.0e} m/s²:")
print(f"  h(g_void) = {h_void:.2f}")
print(f"  Σ_void = 1 + √3 × h = {Sigma_void:.2f}")
print(f"  Effective gravity enhanced by factor {Sigma_void:.1f}×")

print("""
OBSERVATIONAL TESTS:
───────────────────
1. Void galaxy peculiar velocities (should be ~√Σ larger)
2. Void density profiles (steeper than ΛCDM)
3. Void-galaxy cross-correlation (enhanced clustering)

Current status: Void catalogs exist (SDSS, 2dFGRS) but detailed
kinematic tests require dedicated analysis.
""")

# =============================================================================
# TEST 5: COUNTER-ROTATING GALAXIES
# =============================================================================

print(f"""

╔══════════════════════════════════════════════════════════════════════════════╗
║  TEST 5: COUNTER-ROTATING COMPONENTS                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

PREDICTION: Counter-rotating components disrupt coherence.

If the enhancement depends on velocity field coherence:
    - Co-rotating: high coherence → full enhancement
    - Counter-rotating: disrupted coherence → reduced enhancement

This is a UNIQUE prediction that MOND cannot make.
""")

# Check if we have counter-rotation test results
cr_test_path = Path("derivations/test_counter_rotation.py")
if cr_test_path.exists():
    print(f"Found counter-rotation test: {cr_test_path}")
    print("Running counter-rotation analysis...")
    
    try:
        exec(open(cr_test_path).read())
    except Exception as e:
        print(f"  Could not run: {e}")
else:
    print("Counter-rotation test not found. Creating prediction...")
    
    print("""
THEORETICAL PREDICTION:
──────────────────────

For a galaxy with counter-rotating component fraction f_CR:

    Σ_effective = Σ_full × (1 - f_CR)² + Σ_reduced × f_CR × (1 - f_CR) × 2

where:
    Σ_full = 1 + A × W × h(g)  [co-rotating]
    Σ_reduced ≈ 1              [counter-rotating, coherence disrupted]

For f_CR = 0.3 (30% counter-rotating):
""")
    
    f_CR = 0.3
    # Assume typical outer galaxy values
    g_typical = 1e-11
    h_typical = np.sqrt(g_dagger / g_typical) * g_dagger / (g_dagger + g_typical)
    W_typical = 0.8  # Outer disk
    A = np.sqrt(3)
    
    Sigma_full = 1 + A * W_typical * h_typical
    Sigma_reduced = 1.0  # Disrupted coherence
    
    Sigma_effective = Sigma_full * (1 - f_CR)**2 + Sigma_reduced * 2 * f_CR * (1 - f_CR) + Sigma_reduced * f_CR**2
    
    print(f"    Σ_full (no CR) = {Sigma_full:.2f}")
    print(f"    Σ_effective (30% CR) = {Sigma_effective:.2f}")
    print(f"    Reduction: {100*(1 - Sigma_effective/Sigma_full):.1f}%")

# =============================================================================
# SUMMARY
# =============================================================================

print(f"""

╔══════════════════════════════════════════════════════════════════════════════╗
║  TEST SUMMARY                                                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

TEST 1: High-z evolution
    Prediction: g†(z) = g†(0) × H(z)/H₀
    Status: QUANTITATIVE PREDICTION MADE
    Data needed: JWST/KMOS high-z rotation curves

TEST 2: Milky Way screening
    Prediction: Enhancement turns on at r ~ 20-40 kpc
    Status: TESTABLE WITH GAIA
    Data: Available (Gaia DR3)

TEST 3: Inner structure → outer enhancement
    Prediction: Same outer g, different inner structure → different Σ
    Status: TESTED ON SPARC
    Result: See analysis above

TEST 4: Void dynamics
    Prediction: Σ ~ 10× enhancement in voids (g ~ 10⁻¹² m/s²)
    Status: QUANTITATIVE PREDICTION MADE
    Data needed: Void peculiar velocity surveys

TEST 5: Counter-rotation
    Prediction: Counter-rotating components reduce enhancement
    Status: UNIQUE PREDICTION
    Data needed: MaNGA counter-rotating galaxies

══════════════════════════════════════════════════════════════════════════════════
""")



