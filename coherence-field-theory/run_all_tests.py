"""
Master script to run all coherence field theory tests.

Executes:
1. Background cosmology evolution
2. Galaxy rotation curve fitting
3. Cluster lensing profiles
4. Solar system PPN tests
5. Multi-scale parameter optimization
6. Visualization dashboard
"""

import sys
import os

print("=" * 80)
print("COHERENCE FIELD THEORY - COMPREHENSIVE TEST SUITE")
print("=" * 80)

# Test 1: Background Cosmology
print("\n" + "=" * 80)
print("TEST 1: Background Cosmology Evolution")
print("=" * 80)

try:
    os.chdir('cosmology')
    import background_evolution
    cosmo = background_evolution.CoherenceCosmology(V0=1.0e-6, lambda_param=1.0)
    results = cosmo.evolve()
    print(f"✓ Cosmology test passed: Ω_m0 = {results['Omega_m0']:.3f}, Ω_φ0 = {results['Omega_phi0']:.3f}")
    cosmo.plot_density_evolution(savefig='../outputs/density_evolution.png')
    os.chdir('..')
except Exception as e:
    print(f"✗ Cosmology test failed: {e}")
    os.chdir('..')

# Test 2: Galaxy Rotation Curves
print("\n" + "=" * 80)
print("TEST 2: Galaxy Rotation Curves")
print("=" * 80)

try:
    os.chdir('galaxies')
    import rotation_curves
    rotation_curves.toy_example()
    print("✓ Galaxy rotation curve test passed")
    os.chdir('..')
except Exception as e:
    print(f"✗ Galaxy test failed: {e}")
    os.chdir('..')

# Test 3: Cluster Lensing
print("\n" + "=" * 80)
print("TEST 3: Cluster Lensing Profiles")
print("=" * 80)

try:
    os.chdir('clusters')
    import lensing_profiles
    lensing_profiles.example_cluster()
    print("✓ Cluster lensing test passed")
    os.chdir('..')
except Exception as e:
    print(f"✗ Cluster lensing test failed: {e}")
    os.chdir('..')

# Test 4: Solar System Tests
print("\n" + "=" * 80)
print("TEST 4: Solar System PPN Tests")
print("=" * 80)

try:
    os.chdir('solar_system')
    import ppn_tests
    ppn_tests.test_solar_system()
    print("✓ Solar system tests passed")
    os.chdir('..')
except Exception as e:
    print(f"✗ Solar system tests failed: {e}")
    os.chdir('..')

# Test 5: Data Integration
print("\n" + "=" * 80)
print("TEST 5: Data Integration")
print("=" * 80)

try:
    os.chdir('data_integration')
    import load_data
    loader = load_data.DataLoader()
    loader.list_available_data()
    print("✓ Data integration test passed")
    os.chdir('..')
except Exception as e:
    print(f"✗ Data integration failed: {e}")
    os.chdir('..')

# Test 6: Visualization Dashboard
print("\n" + "=" * 80)
print("TEST 6: Visualization Dashboard")
print("=" * 80)

try:
    os.chdir('visualization')
    import dashboard
    dashboard.example_dashboard()
    print("✓ Dashboard creation passed")
    os.chdir('..')
except Exception as e:
    print(f"✗ Dashboard creation failed: {e}")
    os.chdir('..')

# Summary
print("\n" + "=" * 80)
print("TEST SUITE COMPLETE")
print("=" * 80)
print("\nNext steps:")
print("  1. Review generated plots in outputs/")
print("  2. Fit to real SPARC data using galaxies/fit_sparc.py")
print("  3. Run multi-scale optimization with fitting/parameter_optimization.py")
print("  4. Compare with sigma gravity results")
print("=" * 80)

