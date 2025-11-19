"""
Master script to run all coherence field theory tests.

Executes:
1. Background cosmology evolution
2. Galaxy rotation curve fitting
3. Cluster lensing profiles
4. Solar system PPN tests
5. Data integration
6. Visualization dashboard
"""

import sys
import os

# Ensure we're in the coherence-field-theory directory
if not os.path.exists('cosmology'):
    print("[ERROR] Must run from coherence-field-theory/ directory")
    sys.exit(1)

# Create outputs directory if needed
os.makedirs('outputs', exist_ok=True)

print("=" * 80)
print("COHERENCE FIELD THEORY - COMPREHENSIVE TEST SUITE")
print("=" * 80)

# Keep track of test results
test_results = []

# Test 1: Background Cosmology
print("\n" + "=" * 80)
print("TEST 1: Background Cosmology Evolution")
print("=" * 80)

try:
    sys.path.insert(0, 'cosmology')
    import background_evolution
    cosmo = background_evolution.CoherenceCosmology(V0=1.0e-6, lambda_param=1.0)
    results = cosmo.evolve(n_steps=2000)
    print(f"[OK] Cosmology test passed: Omega_m0 = {results['Omega_m0']:.3f}, Omega_phi0 = {results['Omega_phi0']:.3f}")
    cosmo.plot_density_evolution(savefig='outputs/density_evolution.png')
    print(f"[OK] Generated: outputs/density_evolution.png")
    sys.path.pop(0)
    test_results.append(("Cosmology", True))
except Exception as e:
    print(f"[ERROR] Cosmology test failed: {e}")
    import traceback
    traceback.print_exc()
    if 'cosmology' in sys.path[0]:
        sys.path.pop(0)
    test_results.append(("Cosmology", False))

# Test 2: Galaxy Rotation Curves
print("\n" + "=" * 80)
print("TEST 2: Galaxy Rotation Curves")
print("=" * 80)

try:
    sys.path.insert(0, 'galaxies')
    import rotation_curves
    
    # Create toy example
    galaxy = rotation_curves.GalaxyRotationCurve(G=1.0)
    galaxy.set_baryon_profile(M_disk=1.0, R_disk=2.0)
    galaxy.set_coherence_halo_simple(rho_c0=0.2, R_c=8.0)
    galaxy.plot_rotation_curve(r_max=30.0, savefig='outputs/toy_rotation_curve.png')
    
    print("[OK] Galaxy rotation curve test passed")
    print(f"[OK] Generated: outputs/toy_rotation_curve.png")
    sys.path.pop(0)
    test_results.append(("Galaxy Rotation", True))
except Exception as e:
    print(f"[ERROR] Galaxy test failed: {e}")
    import traceback
    traceback.print_exc()
    if 'galaxies' in sys.path[0]:
        sys.path.pop(0)
    test_results.append(("Galaxy Rotation", False))

# Test 3: Cluster Lensing
print("\n" + "=" * 80)
print("TEST 3: Cluster Lensing Profiles")
print("=" * 80)

try:
    sys.path.insert(0, 'clusters')
    import lensing_profiles
    import numpy as np
    
    # Create example cluster
    lens = lensing_profiles.ClusterLensing(z_lens=0.3, z_source=1.0)
    lens.set_baryonic_profile_NFW(M200=1e15, c=4.0, r_vir=2000)
    lens.set_coherence_profile_simple(rho_c0=1e8, R_c=500)
    
    # Compute profiles (smaller sample for speed)
    R_array = np.logspace(1, 3.0, 20)
    profiles = lens.compute_lensing_profile(R_array, include_coherence=True)
    lens.plot_lensing_profiles(profiles, savefig='outputs/cluster_lensing_example.png')
    
    print("[OK] Cluster lensing test passed")
    print(f"[OK] Generated: outputs/cluster_lensing_example.png")
    sys.path.pop(0)
    test_results.append(("Cluster Lensing", True))
except Exception as e:
    print(f"[ERROR] Cluster lensing test failed: {e}")
    import traceback
    traceback.print_exc()
    if 'clusters' in sys.path[0]:
        sys.path.pop(0)
    test_results.append(("Cluster Lensing", False))

# Test 4: Solar System Tests
print("\n" + "=" * 80)
print("TEST 4: Solar System PPN Tests")
print("=" * 80)

try:
    sys.path.insert(0, 'solar_system')
    import ppn_tests
    
    ppn = ppn_tests.PPNCalculator(V0=1e-6, lambda_param=1.0, coupling=1e-3)
    ppn.plot_solar_system_tests(savefig='outputs/solar_system_tests.png')
    
    print("[OK] Solar system tests passed")
    print(f"[OK] Generated: outputs/solar_system_tests.png")
    sys.path.pop(0)
    test_results.append(("Solar System", True))
except Exception as e:
    print(f"[ERROR] Solar system tests failed: {e}")
    import traceback
    traceback.print_exc()
    if 'solar_system' in sys.path[0]:
        sys.path.pop(0)
    test_results.append(("Solar System", False))

# Test 5: Data Integration
print("\n" + "=" * 80)
print("TEST 5: Data Integration")
print("=" * 80)

try:
    sys.path.insert(0, 'data_integration')
    import load_data
    loader = load_data.DataLoader()
    loader.list_available_data()
    print("[OK] Data integration test passed")
    sys.path.pop(0)
    test_results.append(("Data Integration", True))
except Exception as e:
    print(f"[ERROR] Data integration failed: {e}")
    import traceback
    traceback.print_exc()
    if 'data_integration' in sys.path[0]:
        sys.path.pop(0)
    test_results.append(("Data Integration", False))

# Test 6: Visualization Dashboard
print("\n" + "=" * 80)
print("TEST 6: Visualization Dashboard")
print("=" * 80)

try:
    sys.path.insert(0, 'visualization')
    import dashboard
    import numpy as np
    try:
        from scipy.integrate import trapz
    except ImportError:
        from scipy.integrate import trapezoid as trapz
    
    # Create example dashboard with synthetic data
    dash = dashboard.CoherenceFieldDashboard()
    
    # Add cosmology
    z = np.linspace(0.1, 2.0, 30)
    H_obs = np.sqrt(0.3 * (1+z)**3 + 0.7) * (1 + 0.02 * np.random.randn(len(z)))
    H_model = np.sqrt(0.3 * (1+z)**3 + 0.7)
    
    dL_obs = []
    dL_model = []
    for zi in z:
        zs = np.linspace(0, zi, 500)
        chi = trapz(1.0 / np.sqrt(0.3 * (1+zs)**3 + 0.7), zs)
        dL = (1 + zi) * chi
        dL_model.append(dL)
        dL_obs.append(dL * (1 + 0.03 * np.random.randn()))
    
    dash.add_cosmology_results(z, H_obs, H_model, 
                               np.array(dL_obs), np.array(dL_model),
                               H_err=H_obs*0.05, dL_err=np.array(dL_model)*0.03)
    
    # Add galaxies
    for i in range(3):
        r = np.linspace(1, 25, 30)
        v_baryon = 120 * np.sqrt(r / (r + 3))
        v_model = np.sqrt(v_baryon**2 + 100**2 * (1 - np.exp(-r/5)))
        v_obs = v_model + np.random.normal(0, 5, len(r))
        
        dash.add_galaxy_results(f'Galaxy{i+1}', r, v_obs, v_model, v_baryon, 
                               v_err=np.ones_like(r)*5)
    
    # Add clusters
    for i in range(3):
        R = np.logspace(1.5, 3, 20)
        Sigma_NFW = 1e10 / R**2
        Sigma_coh = 5e9 / (R + 100)
        Sigma_model = Sigma_NFW + Sigma_coh
        Sigma_obs = Sigma_model * (1 + 0.1 * np.random.randn(len(R)))
        
        dash.add_cluster_results(f'Cluster{i+1}', R, Sigma_obs, Sigma_model,
                                Sigma_NFW, Sigma_coh, 
                                Sigma_err=Sigma_model*0.1)
    
    # Create dashboard
    dash.create_full_dashboard(savefig='outputs/example_dashboard.png')
    
    print("[OK] Dashboard creation passed")
    print(f"[OK] Generated: outputs/example_dashboard.png")
    sys.path.pop(0)
    test_results.append(("Visualization", True))
except Exception as e:
    print(f"[ERROR] Dashboard creation failed: {e}")
    import traceback
    traceback.print_exc()
    if 'visualization' in sys.path[0]:
        sys.path.pop(0)
    test_results.append(("Visualization", False))

# Summary
print("\n" + "=" * 80)
print("TEST SUITE COMPLETE")
print("=" * 80)

# Print summary
print("\nTest Results:")
all_passed = True
for test_name, passed in test_results:
    status = "[OK]" if passed else "[FAILED]"
    print(f"  {status} {test_name}")
    if not passed:
        all_passed = False

print("\nGenerated Outputs:")
output_files = [
    'outputs/density_evolution.png',
    'outputs/toy_rotation_curve.png',
    'outputs/cluster_lensing_example.png',
    'outputs/solar_system_tests.png',
    'outputs/example_dashboard.png'
]

for filepath in output_files:
    if os.path.exists(filepath):
        size_kb = os.path.getsize(filepath) / 1024
        print(f"  [OK] {filepath} ({size_kb:.1f} KB)")
    else:
        print(f"  [MISSING] {filepath}")

print("\n" + "=" * 80)
if all_passed:
    print("SUCCESS: All tests passed!")
else:
    print("WARNING: Some tests failed. Check output above.")
print("=" * 80)

print("\nNext steps:")
print("  1. Review generated plots in outputs/")
print("  2. Fit to real SPARC data using galaxies/fit_sparc.py")
print("  3. Run multi-scale optimization with fitting/parameter_optimization.py")
print("  4. Compare with sigma gravity results")
print("=" * 80)
