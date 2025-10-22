"""
Option B Test Suite Runner
===========================
Runs all three sanity checks to verify that the linear-regime cosmological
framework (Ω_eff in FRW, μ=1) does NOT break halo-scale predictions from
the main paper.

Tests:
1. Solar System (Cassini bounds)
2. Galaxy rotation curves
3. Cluster lensing (distances + kernel)

Run from repo root:
  python cosmo/tests_option_b/run_all_tests.py
"""

import sys
from pathlib import Path
import subprocess

def run_test(script_path):
    """Run a test script and return True if it succeeds."""
    print("\n" + "="*70)
    print(f"Running: {script_path.name}")
    print("="*70)
    result = subprocess.run([sys.executable, str(script_path)], 
                          capture_output=False, text=True)
    return result.returncode == 0

def main():
    tests_dir = Path(__file__).parent
    
    tests = [
        tests_dir / "test_solar_system.py",
        tests_dir / "test_galaxy_vc.py",
        tests_dir / "test_cluster_lensing.py"
    ]
    
    print("="*70)
    print("OPTION B FULL TEST SUITE")
    print("="*70)
    print("Testing whether Ω_eff FRW cosmology breaks halo-scale physics.")
    print(f"Running {len(tests)} tests...\n")
    
    results = {}
    for test in tests:
        if test.exists():
            success = run_test(test)
            results[test.name] = success
        else:
            print(f"✗ Test not found: {test}")
            results[test.name] = False
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUITE SUMMARY")
    print("="*70)
    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}  {name}")
    
    all_pass = all(results.values())
    print("\n" + "="*70)
    if all_pass:
        print("✓ ALL TESTS PASS")
        print("Option B (Ω_eff FRW) preserves halo-scale predictions.")
        print("Solar System, galaxy, and cluster results unchanged.")
    else:
        print("✗ SOME TESTS FAILED")
        print("Review individual test outputs above.")
    print("="*70)
    
    outdir = tests_dir / "outputs"
    print(f"\n✓ Individual test results in: {outdir}")
    
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())
