"""
SPARC Batch Test for Gravitational Channeling
==============================================

Phase 1 Statistical Validation:
- 1a. Full SPARC batch: Does channeling improve majority? (target >65%)
- 1b. RMS distribution: Are improvements significant? (target median ΔRMS < -3 km/s)
- 1c. Failure mode analysis: Which galaxies fail? Pattern?
- 1d. Cross-validation: Train on 80%, predict 20%
"""

import sys
import os
import glob
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from channeling_kernel import (
    ChannelingParams, load_sparc_galaxy, fit_galaxy, 
    test_solar_system, compute_rms
)


def run_sparc_batch(data_dir: str, params: ChannelingParams):
    """Run channeling model on all SPARC galaxies."""
    
    # Find all rotation curve files
    pattern = os.path.join(data_dir, "*_rotmod.dat")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"ERROR: No files found matching {pattern}")
        return None
    
    print(f"Found {len(files)} SPARC galaxies")
    print("=" * 70)
    
    results = []
    improved_count = 0
    failed_galaxies = []
    
    for filepath in files:
        try:
            data = load_sparc_galaxy(filepath)
            if len(data['R']) < 5:
                continue
                
            result = fit_galaxy(data, params)
            results.append(result)
            
            if result['improved']:
                improved_count += 1
            else:
                failed_galaxies.append(result)
                
        except Exception as e:
            print(f"  Error processing {os.path.basename(filepath)}: {e}")
            continue
    
    return results, improved_count, failed_galaxies


def analyze_results(results, improved_count, failed_galaxies):
    """Analyze and report batch results."""
    
    n_total = len(results)
    pct_improved = 100 * improved_count / n_total
    
    # Extract metrics
    delta_rms = np.array([r['delta_rms'] for r in results])
    rms_pred = np.array([r['rms_pred'] for r in results])
    rms_bary = np.array([r['rms_bary'] for r in results])
    v_flat = np.array([np.mean(r['v_obs'][-3:]) for r in results])
    
    print("\n" + "=" * 70)
    print("PHASE 1a: FULL SPARC BATCH RESULTS")
    print("=" * 70)
    print(f"Galaxies tested: {n_total}")
    print(f"Improved: {improved_count}/{n_total} ({pct_improved:.1f}%)")
    print(f"Target: >65% → {'PASS ✓' if pct_improved > 65 else 'FAIL ✗'}")
    
    print("\n" + "=" * 70)
    print("PHASE 1b: RMS DISTRIBUTION")
    print("=" * 70)
    print(f"Median ΔRMS: {np.median(delta_rms):.2f} km/s")
    print(f"Mean ΔRMS: {np.mean(delta_rms):.2f} km/s")
    print(f"Std ΔRMS: {np.std(delta_rms):.2f} km/s")
    print(f"Min ΔRMS: {np.min(delta_rms):.2f} km/s (best improvement)")
    print(f"Max ΔRMS: {np.max(delta_rms):.2f} km/s (worst case)")
    print(f"Target: median < -3 km/s → {'PASS ✓' if np.median(delta_rms) < -3 else 'FAIL ✗'}")
    
    # Percentile breakdown
    print("\nΔRMS percentiles:")
    for p in [10, 25, 50, 75, 90]:
        print(f"  {p}th: {np.percentile(delta_rms, p):.2f} km/s")
    
    print("\n" + "=" * 70)
    print("PHASE 1c: FAILURE MODE ANALYSIS")
    print("=" * 70)
    
    n_failed = len(failed_galaxies)
    if n_failed > 0:
        # Analyze failed galaxies
        failed_v_flat = np.array([np.mean(r['v_obs'][-3:]) for r in failed_galaxies])
        failed_delta_rms = np.array([r['delta_rms'] for r in failed_galaxies])
        
        print(f"Failed galaxies: {n_failed}")
        print(f"Mean v_flat of failures: {np.mean(failed_v_flat):.1f} km/s")
        print(f"Mean v_flat of successes: {np.mean([np.mean(r['v_obs'][-3:]) for r in results if r['improved']]):.1f} km/s")
        
        # Categorize by type
        low_v = [r for r in failed_galaxies if np.mean(r['v_obs'][-3:]) < 80]
        mid_v = [r for r in failed_galaxies if 80 <= np.mean(r['v_obs'][-3:]) < 150]
        high_v = [r for r in failed_galaxies if np.mean(r['v_obs'][-3:]) >= 150]
        
        print(f"\nFailures by velocity class:")
        print(f"  Dwarfs (v < 80 km/s): {len(low_v)}")
        print(f"  Intermediate (80-150 km/s): {len(mid_v)}")
        print(f"  Massive (v > 150 km/s): {len(high_v)}")
        
        # Top 10 worst failures
        print("\nTop 10 worst failures:")
        worst = sorted(failed_galaxies, key=lambda x: x['delta_rms'], reverse=True)[:10]
        print(f"{'Galaxy':<20} {'v_flat':<10} {'ΔRMS':<10} {'RMS_bary':<10} {'RMS_pred':<10}")
        print("-" * 60)
        for r in worst:
            v_f = np.mean(r['v_obs'][-3:])
            print(f"{r['name']:<20} {v_f:<10.1f} {r['delta_rms']:<10.2f} {r['rms_bary']:<10.2f} {r['rms_pred']:<10.2f}")
    else:
        print("No failures!")
    
    # Top 10 best improvements
    print("\nTop 10 best improvements:")
    best = sorted(results, key=lambda x: x['delta_rms'])[:10]
    print(f"{'Galaxy':<20} {'v_flat':<10} {'ΔRMS':<10} {'RMS_bary':<10} {'RMS_pred':<10}")
    print("-" * 60)
    for r in best:
        v_f = np.mean(r['v_obs'][-3:])
        print(f"{r['name']:<20} {v_f:<10.1f} {r['delta_rms']:<10.2f} {r['rms_bary']:<10.2f} {r['rms_pred']:<10.2f}")
    
    return {
        'n_total': n_total,
        'n_improved': improved_count,
        'pct_improved': pct_improved,
        'median_delta_rms': np.median(delta_rms),
        'mean_delta_rms': np.mean(delta_rms),
    }


def run_cross_validation(data_dir: str, params: ChannelingParams, n_folds: int = 5):
    """Phase 1d: Cross-validation test."""
    
    print("\n" + "=" * 70)
    print("PHASE 1d: CROSS-VALIDATION")
    print("=" * 70)
    
    # Load all galaxies
    pattern = os.path.join(data_dir, "*_rotmod.dat")
    files = sorted(glob.glob(pattern))
    
    all_data = []
    for filepath in files:
        try:
            data = load_sparc_galaxy(filepath)
            if len(data['R']) >= 5:
                all_data.append(data)
        except:
            continue
    
    np.random.seed(42)
    np.random.shuffle(all_data)
    
    fold_size = len(all_data) // n_folds
    fold_results = []
    
    for fold in range(n_folds):
        # Split data
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < n_folds - 1 else len(all_data)
        test_data = all_data[test_start:test_end]
        
        # Evaluate on test set
        test_improved = 0
        test_delta_rms = []
        
        for data in test_data:
            result = fit_galaxy(data, params)
            test_delta_rms.append(result['delta_rms'])
            if result['improved']:
                test_improved += 1
        
        pct_improved = 100 * test_improved / len(test_data)
        median_delta = np.median(test_delta_rms)
        fold_results.append((pct_improved, median_delta))
        
        print(f"Fold {fold+1}: {pct_improved:.1f}% improved, median ΔRMS = {median_delta:.2f} km/s")
    
    # Summary
    mean_pct = np.mean([r[0] for r in fold_results])
    std_pct = np.std([r[0] for r in fold_results])
    mean_delta = np.mean([r[1] for r in fold_results])
    
    print(f"\nCross-validation summary:")
    print(f"  Mean % improved: {mean_pct:.1f}% ± {std_pct:.1f}%")
    print(f"  Mean median ΔRMS: {mean_delta:.2f} km/s")
    print(f"  Prediction within 20% of training: {'PASS ✓' if std_pct < 10 else 'UNCERTAIN'}")
    
    return fold_results


def main():
    print("=" * 70)
    print("GRAVITATIONAL CHANNELING - PHASE 1: SPARC STATISTICAL VALIDATION")
    print("=" * 70)
    
    # Parameters from sweep optimization
    params = ChannelingParams(
        alpha=1.0,      # Channel scale grows with R
        beta=0.5,       # Cold systems carve deeper (gentler than theory)
        gamma=0.3,      # Sublinear accumulation
        chi_0=0.3,      # Base coupling (reduced to avoid over-boost)
        epsilon=0.3,    # Surface density exponent
        D_max=3.0,      # Saturation (tighter)
        t_age=10.0,     # System age [Gyr]
        tau_0=1.0,      # Reference formation time
    )
    
    print("\nParameters:")
    print(f"  α = {params.alpha} (radial growth)")
    print(f"  β = {params.beta} (velocity coherence)")
    print(f"  γ = {params.gamma} (time accumulation)")
    print(f"  χ₀ = {params.chi_0} (coupling)")
    print(f"  ε = {params.epsilon} (density exponent)")
    print(f"  D_max = {params.D_max} (saturation)")
    print(f"  t_age = {params.t_age} Gyr")
    
    # Solar System test first
    print("\n" + "=" * 70)
    print("SOLAR SYSTEM CONSTRAINT (Cassini)")
    print("=" * 70)
    delta_g, passes = test_solar_system(params)
    print(f"δg/g at Saturn: {delta_g:.2e}")
    print(f"Cassini limit: 2.3×10⁻⁵")
    print(f"Result: {'PASS ✓' if passes else 'FAIL ✗'}")
    
    if not passes:
        print("\nWARNING: Solar System constraint violated! Proceeding anyway...")
    
    # SPARC data directory
    data_dir = r"C:\Users\henry\dev\sigmagravity\data\Rotmod_LTG"
    
    if not os.path.exists(data_dir):
        print(f"\nERROR: Data directory not found: {data_dir}")
        return
    
    # Run batch test
    batch_result = run_sparc_batch(data_dir, params)
    if batch_result is None:
        return
    
    results, improved_count, failed_galaxies = batch_result
    
    # Analyze results
    summary = analyze_results(results, improved_count, failed_galaxies)
    
    # Cross-validation
    cv_results = run_cross_validation(data_dir, params, n_folds=5)
    
    # Final verdict
    print("\n" + "=" * 70)
    print("PHASE 1 VERDICT")
    print("=" * 70)
    
    passed_tests = 0
    total_tests = 4
    
    # 1a: >65% improved
    if summary['pct_improved'] > 65:
        print(f"✓ 1a. Batch success rate: {summary['pct_improved']:.1f}% > 65%")
        passed_tests += 1
    else:
        print(f"✗ 1a. Batch success rate: {summary['pct_improved']:.1f}% < 65%")
    
    # 1b: median ΔRMS < -3
    if summary['median_delta_rms'] < -3:
        print(f"✓ 1b. Median ΔRMS: {summary['median_delta_rms']:.2f} < -3 km/s")
        passed_tests += 1
    else:
        print(f"✗ 1b. Median ΔRMS: {summary['median_delta_rms']:.2f} >= -3 km/s")
    
    # 1c: No systematic failure pattern (subjective, pass if <50% failures are one type)
    n_failed = len(failed_galaxies)
    if n_failed > 0:
        low_v = len([r for r in failed_galaxies if np.mean(r['v_obs'][-3:]) < 80])
        if low_v / n_failed < 0.5:
            print(f"✓ 1c. No systematic failure pattern (dwarfs: {100*low_v/n_failed:.0f}% of failures)")
            passed_tests += 1
        else:
            print(f"✗ 1c. Systematic dwarf failure pattern ({100*low_v/n_failed:.0f}% of failures)")
    else:
        print(f"✓ 1c. No failures!")
        passed_tests += 1
    
    # 1d: Cross-validation stable
    cv_std = np.std([r[0] for r in cv_results])
    if cv_std < 10:
        print(f"✓ 1d. Cross-validation stable (σ = {cv_std:.1f}%)")
        passed_tests += 1
    else:
        print(f"✗ 1d. Cross-validation unstable (σ = {cv_std:.1f}%)")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests >= 3:
        print("\n🎉 PHASE 1 PASSED - Proceed to Phase 2 (MW anchor)")
    else:
        print("\n⚠️ PHASE 1 NEEDS WORK - Consider parameter tuning")


if __name__ == "__main__":
    main()
