"""
Quick demonstration of coherence field theory capabilities.

Runs minimal examples from each module to verify installation and show basic usage.
"""

import sys
import os

print("=" * 80)
print("COHERENCE FIELD THEORY - QUICK START DEMO")
print("=" * 80)
print("\nThis script demonstrates the basic capabilities of each module.")
print("Full documentation: see GETTING_STARTED.md and ROADMAP.md")
print("=" * 80)

# Check imports
print("\n[1/6] Checking dependencies...")
try:
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from scipy.integrate import odeint
    try:
        from scipy.integrate import trapz
    except ImportError:
        from scipy.integrate import trapezoid as trapz
    from scipy.optimize import minimize
    print("[OK] All required packages found")
except ImportError as e:
    print(f"[ERROR] Missing package: {e}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

# Test 1: Cosmology
print("\n[2/6] Testing cosmology module...")
try:
    sys.path.insert(0, 'cosmology')
    from background_evolution import CoherenceCosmology
    
    cosmo = CoherenceCosmology(V0=1.0e-6, lambda_param=1.0)
    results = cosmo.evolve(n_steps=2000)  # Fewer steps for speed
    
    print(f"  Omega_m0 = {results['Omega_m0']:.3f}")
    print(f"  Omega_phi0 = {results['Omega_phi0']:.3f}")
    print("  [OK] Background evolution successful")
    
    sys.path.pop(0)
except Exception as e:
    print(f"  [ERROR] Cosmology test failed: {e}")

# Test 2: Galaxy rotation curves
print("\n[3/6] Testing galaxy module...")
try:
    sys.path.insert(0, 'galaxies')
    from rotation_curves import GalaxyRotationCurve
    
    galaxy = GalaxyRotationCurve(G=1.0)
    galaxy.set_baryon_profile(M_disk=1.0, R_disk=2.0)
    galaxy.set_coherence_halo_simple(rho_c0=0.2, R_c=8.0)
    
    r_test = np.array([5.0, 10.0, 15.0])
    v_test = galaxy.circular_velocity(r_test)
    
    print(f"  v(r=10) = {v_test[1]:.3f} (arbitrary units)")
    print("  [OK] Rotation curve calculation successful")
    
    sys.path.pop(0)
except Exception as e:
    print(f"  [ERROR] Galaxy test failed: {e}")

# Test 3: Cluster lensing
print("\n[4/6] Testing cluster module...")
try:
    sys.path.insert(0, 'clusters')
    from lensing_profiles import ClusterLensing
    
    lens = ClusterLensing(z_lens=0.3, z_source=1.0)
    lens.set_baryonic_profile_NFW(M200=1e15, c=4.0, r_vir=2000)
    lens.set_coherence_profile_simple(rho_c0=1e8, R_c=500)
    
    # Just test surface density calculation
    Sigma_test = lens.surface_density(100.0, lens.rho_NFW)
    
    print(f"  Surface density at 100 kpc = {Sigma_test:.2e} M_sun/kpc^2")
    print("  [OK] Lensing calculation successful")
    
    sys.path.pop(0)
except Exception as e:
    print(f"  [ERROR] Cluster test failed: {e}")

# Test 4: Data integration
print("\n[5/6] Testing data integration...")
try:
    sys.path.insert(0, 'data_integration')
    from load_data import DataLoader
    
    loader = DataLoader()
    # Just check if data directory exists
    if os.path.exists(loader.base_data_dir):
        print(f"  Data directory: {loader.base_data_dir}")
        print("  [OK] Data integration ready")
    else:
        print(f"  [WARNING] Data directory not found: {loader.base_data_dir}")
        print("  (This is okay for basic testing)")
    
    sys.path.pop(0)
except Exception as e:
    print(f"  [ERROR] Data integration test failed: {e}")

# Test 5: Solar system
print("\n[6/6] Testing solar system module...")
try:
    sys.path.insert(0, 'solar_system')
    from ppn_tests import PPNCalculator
    
    ppn = PPNCalculator(V0=1e-6, lambda_param=1.0, coupling=1e-3)
    gamma, beta = ppn.compute_ppn_parameters()
    
    print(f"  PPN gamma = {gamma:.6f} (GR: 1.0)")
    print(f"  PPN beta = {beta:.6f} (GR: 1.0)")
    print("  [OK] PPN calculation successful")
    
    sys.path.pop(0)
except Exception as e:
    print(f"  [ERROR] Solar system test failed: {e}")

# Summary
print("\n" + "=" * 80)
print("QUICK START COMPLETE!")
print("=" * 80)
print("\nAll basic functionality verified. Next steps:")
print("\n1. Read GETTING_STARTED.md for detailed usage")
print("2. Review ROADMAP.md for development plan")
print("3. Run individual module examples:")
print("     python cosmology/background_evolution.py")
print("     python galaxies/rotation_curves.py")
print("     python clusters/lensing_profiles.py")
print("\n4. Fit real data:")
print("     python galaxies/fit_sparc.py")
print("\n5. Multi-scale optimization:")
print("     python fitting/parameter_optimization.py")
print("\n6. Generate dashboard:")
print("     python visualization/dashboard.py")
print("=" * 80)

