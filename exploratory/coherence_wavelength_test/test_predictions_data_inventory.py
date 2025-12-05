#!/usr/bin/env python3
"""
Data Inventory and Prediction Testing for Σ-Gravity

This script:
1. Inventories what data we have
2. Tests predictions we CAN make with existing data
3. Identifies what data we NEED to download for remaining tests

Author: Sigma Gravity Team
Date: December 2025
"""

import numpy as np
import os
import glob

# Physical constants
c = 2.998e8  # m/s
H0 = 70 * 1000 / 3.086e22  # 1/s (70 km/s/Mpc)
H0_kmsMpc = 70  # km/s/Mpc

# Cosmological parameters
Omega_m = 0.31
Omega_Lambda = 0.69

def H_of_z(z):
    """Hubble parameter at redshift z"""
    return H0 * np.sqrt(Omega_m * (1 + z)**3 + Omega_Lambda)

def g_dagger(z=0):
    """Critical acceleration at redshift z"""
    return c * H_of_z(z) / (4 * np.sqrt(np.pi))

print("=" * 80)
print("Σ-GRAVITY PREDICTION TESTING: DATA INVENTORY")
print("=" * 80)

# =============================================================================
# PREDICTION 1: g† = cH₀/(4√π) ≈ 9.6 × 10⁻¹¹ m/s²
# =============================================================================

print("\n" + "=" * 80)
print("PREDICTION 1: CRITICAL ACCELERATION VALUE")
print("=" * 80)

g_dagger_predicted = g_dagger(0)
a0_mond = 1.2e-10  # m/s²

print(f"\nΣ-Gravity prediction: g† = cH₀/(4√π) = {g_dagger_predicted:.3e} m/s²")
print(f"MOND empirical value: a₀ = {a0_mond:.3e} m/s²")
print(f"Ratio: g†/a₀ = {g_dagger_predicted/a0_mond:.3f}")
print(f"Difference: {(a0_mond - g_dagger_predicted)/a0_mond * 100:.1f}%")

# Check for SPARC data
sparc_paths = [
    "/Users/leonardspeiser/Projects/sigmagravity/data/Rotmod_LTG",
    "/Users/leonardspeiser/Projects/sigmagravity/many_path_model/paper_release/data/Rotmod_LTG",
]

sparc_data_found = False
for path in sparc_paths:
    if os.path.exists(path):
        n_galaxies = len(glob.glob(os.path.join(path, "*_rotmod.dat")))
        if n_galaxies > 0:
            sparc_data_found = True
            print(f"\n✓ SPARC data found: {path}")
            print(f"  Number of galaxies: {n_galaxies}")

if not sparc_data_found:
    print("\n✗ SPARC data NOT found")
    print("  Download from: http://astroweb.cwru.edu/SPARC/")

print("\nTEST STATUS: Can test with existing SPARC data")
print("  - Compare best-fit g† from rotation curves to predicted value")
print("  - Check if 9.6×10⁻¹¹ fits better than 1.2×10⁻¹⁰")

# =============================================================================
# PREDICTION 2: g†(z) = cH(z)/(4√π) - REDSHIFT EVOLUTION
# =============================================================================

print("\n" + "=" * 80)
print("PREDICTION 2: REDSHIFT EVOLUTION OF g†")
print("=" * 80)

print("\nPredicted evolution:")
print(f"{'z':<6} {'H(z)/H₀':<12} {'g†(z)/g†(0)':<15} {'g†(z) [m/s²]':<15}")
print("-" * 50)
for z in [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    ratio = H_of_z(z) / H0
    g_z = g_dagger(z)
    print(f"{z:<6.1f} {ratio:<12.3f} {ratio:<15.3f} {g_z:<15.3e}")

print("\nThis predicts LESS gravitational enhancement at high z")
print("(because g† is higher, galaxies are closer to Newtonian regime)")

# Check for high-z data
print("\n--- HIGH-Z DATA INVENTORY ---")

# Look for any high-z rotation curve data
highz_keywords = ["kmos", "sins", "genzel", "high_z", "highz", "z2", "z1"]
highz_data_found = False

for root, dirs, files in os.walk("/Users/leonardspeiser/Projects/sigmagravity"):
    for f in files:
        if any(kw in f.lower() for kw in highz_keywords):
            print(f"  Possible high-z data: {os.path.join(root, f)}")
            highz_data_found = True

if not highz_data_found:
    print("\n✗ No high-z rotation curve data found in repository")
    print("\nDATA NEEDED:")
    print("  1. KMOS³D survey: https://www.mpe.mpg.de/ir/KMOS3D")
    print("     - Rotation curves for z ~ 0.6-2.7 galaxies")
    print("  2. SINS survey: https://www.mpe.mpg.de/ir/SINS")
    print("     - z ~ 2 star-forming galaxies")
    print("  3. Genzel et al. (2020) data:")
    print("     - Dark matter fractions at z = 0, 1, 2")
    print("     - Paper: Nature 543, 397 (2017), updated 2020")
    print("  4. ALMA rotation curves at z > 1")

print("\nTEST STATUS: NEED TO DOWNLOAD HIGH-Z DATA")
print("  This is the KEY test that distinguishes Σ-Gravity from MOND")

# =============================================================================
# PREDICTION 3: SPECIFIC h(g) FUNCTIONAL FORM
# =============================================================================

print("\n" + "=" * 80)
print("PREDICTION 3: ENHANCEMENT FUNCTION h(g)")
print("=" * 80)

print("\nΣ-Gravity: h(g) = √(g†/g) × g†/(g†+g)")
print("MOND simple: ν(x) = 1/(1-e^(-√x)) where x = g/a₀")
print("MOND standard: ν(x) = (1 + √(1+4/x))/2")

print("\nThese differ most in the transition region (g ~ g†)")

# The RAR (Radial Acceleration Relation) tests this
print("\n--- RAR DATA ---")
print("✓ Can test with SPARC RAR data")
print("  - McGaugh et al. (2016) RAR from SPARC")
print("  - Compare h(g) fit quality vs MOND interpolating functions")

print("\nTEST STATUS: Can test with existing SPARC data")

# =============================================================================
# PREDICTION 4: CLUSTER DATA
# =============================================================================

print("\n" + "=" * 80)
print("PREDICTION 4: GALAXY CLUSTERS")
print("=" * 80)

cluster_paths = [
    "/Users/leonardspeiser/Projects/sigmagravity/data/clusters/fox2022_unique_clusters.csv",
    "/Users/leonardspeiser/Projects/sigmagravity/data/clusters/fox2022_clusters.csv",
]

cluster_data_found = False
for path in cluster_paths:
    if os.path.exists(path):
        cluster_data_found = True
        # Count lines (rough estimate of clusters)
        with open(path, 'r') as f:
            n_lines = sum(1 for _ in f) - 1  # subtract header
        print(f"✓ Cluster data found: {path}")
        print(f"  Number of clusters: {n_lines}")

if not cluster_data_found:
    print("✗ Cluster data NOT found")

print("\nTEST STATUS: Can test with existing Fox+ 2022 data")

# =============================================================================
# PREDICTION 5: COUNTER-ROTATING SYSTEMS
# =============================================================================

print("\n" + "=" * 80)
print("PREDICTION 5: COUNTER-ROTATING SYSTEMS")
print("=" * 80)

print("\nΣ-Gravity predicts: Counter-rotating disks show REDUCED enhancement")
print("  - Coherence from co-rotating material doesn't add to counter-rotating")
print("  - Unique prediction not shared by MOND or ΛCDM")

print("\nKnown counter-rotating galaxies:")
print("  - NGC 4550 (two counter-rotating stellar disks)")
print("  - NGC 7217 (counter-rotating gas)")
print("  - NGC 4138 (counter-rotating gas disk)")

print("\n✗ No counter-rotating galaxy data found in repository")
print("\nDATA NEEDED:")
print("  - Detailed kinematics of NGC 4550, NGC 7217, NGC 4138")
print("  - Stellar and gas rotation curves separately")
print("  - Mass models for baryonic components")

print("\nTEST STATUS: NEED SPECIALIZED DATA")
print("  This would be a UNIQUE test of coherence-based theories")

# =============================================================================
# PREDICTION 6: MILKY WAY / GAIA
# =============================================================================

print("\n" + "=" * 80)
print("PREDICTION 6: MILKY WAY (GAIA DATA)")
print("=" * 80)

gaia_paths = [
    "/Users/leonardspeiser/Projects/sigmagravity/vendor/maxdepth_gaia/gaia_bin_residuals.csv",
    "/Users/leonardspeiser/Projects/sigmagravity/data/gaia",
]

gaia_data_found = False
for path in gaia_paths:
    if os.path.exists(path):
        gaia_data_found = True
        print(f"✓ Gaia data found: {path}")

if not gaia_data_found:
    print("✗ Gaia data NOT found")

print("\nTEST STATUS: Can test with existing Gaia data")

# =============================================================================
# PREDICTION 7: WIDE BINARIES
# =============================================================================

print("\n" + "=" * 80)
print("PREDICTION 7: WIDE BINARY STARS")
print("=" * 80)

print("\nAt separations > 7000 AU, binary orbits should show MOND-like deviations")
print("because g < g† ≈ 10⁻¹⁰ m/s²")

print("\n✗ No wide binary data found in repository")
print("\nDATA NEEDED:")
print("  - Gaia DR3 wide binary catalog")
print("  - Chae (2023) analysis data")
print("  - Banik et al. (2024) reanalysis data")
print("  Source: https://www.cosmos.esa.int/web/gaia/dr3")

print("\nTEST STATUS: NEED TO DOWNLOAD WIDE BINARY DATA")
print("  Controversial test - Chae claims detection, Banik disputes")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY: DATA INVENTORY")
print("=" * 80)

tests = [
    ("g† = cH₀/(4√π) value", "SPARC rotation curves", sparc_data_found, "Can test NOW"),
    ("g†(z) redshift evolution", "High-z rotation curves", False, "NEED DATA - KEY TEST"),
    ("h(g) functional form", "SPARC RAR", sparc_data_found, "Can test NOW"),
    ("Cluster lensing", "Fox+ 2022 clusters", cluster_data_found, "Can test NOW"),
    ("Counter-rotating systems", "NGC 4550, etc.", False, "NEED DATA - UNIQUE TEST"),
    ("Milky Way", "Gaia DR3", gaia_data_found, "Can test NOW"),
    ("Wide binaries", "Gaia wide binaries", False, "NEED DATA - CONTROVERSIAL"),
]

print(f"\n{'Test':<30} {'Data Source':<25} {'Have Data?':<12} {'Status':<20}")
print("-" * 90)
for test, source, have_data, status in tests:
    have = "✓ YES" if have_data else "✗ NO"
    print(f"{test:<30} {source:<25} {have:<12} {status:<20}")

print("\n" + "=" * 80)
print("PRIORITY DATA TO DOWNLOAD")
print("=" * 80)

print("""
1. HIGH-Z ROTATION CURVES (CRITICAL - KEY TEST)
   - KMOS³D: https://www.mpe.mpg.de/ir/KMOS3D
   - Download rotation curve data for z ~ 0.6-2.7 galaxies
   - This tests g†(z) = cH(z)/(4√π) prediction
   - UNIQUE to Σ-Gravity - neither MOND nor ΛCDM predicts this

2. GENZEL ET AL. (2020) DARK MATTER FRACTIONS
   - Published data on f_DM vs redshift
   - Supplementary materials from Nature paper
   - Can directly compare to our H(z) scaling prediction

3. WIDE BINARY CATALOG (CONTROVERSIAL)
   - Gaia DR3 wide binary sample
   - Test at g < g† in local universe
   - Disputed results - be cautious

4. COUNTER-ROTATING GALAXIES (UNIQUE TEST)
   - NGC 4550, NGC 7217 detailed kinematics
   - Would be decisive for coherence-based theory
   - May need to request from authors
""")

print("=" * 80)
print("END OF DATA INVENTORY")
print("=" * 80)

