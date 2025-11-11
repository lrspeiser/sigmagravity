"""
Main runner script for all scale-finding tests.
Runs tests in optimal order and generates comprehensive report.
"""

import os
import sys
import time
from datetime import datetime

def run_all_scale_tests():
    """Run complete test suite in optimal order."""
    
    print("="*80)
    print("ΣGRAVITY SCALE-FINDING TEST SUITE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Check if SPARC data is prepared
    sparc_path = "data/sparc/sparc_combined.csv"
    if not os.path.exists(sparc_path):
        print("\n⚠ SPARC data not found. Running data preparation...")
        from prepare_sparc_data import prepare_sparc_combined
        df = prepare_sparc_combined()
        
        if df is None:
            print("\n❌ Failed to prepare SPARC data. Aborting.")
            return
        
        print("\n✓ SPARC data prepared successfully!")
    else:
        print(f"\n✓ Found SPARC data: {sparc_path}")
    
    # Test 1: Tully-Fisher scaling (fastest, most important)
    print("\n" + "="*80)
    print("TEST 1: TULLY-FISHER SCALING")
    print("="*80)
    print("This test checks if the coherence length scales as λ ∝ √M_b")
    print("Expected time: 30 seconds")
    
    start = time.time()
    try:
        from test_tully_fisher_scaling import test_tully_fisher_scaling
        tf_results = test_tully_fisher_scaling(sparc_path)
        print(f"\n✓ Tully-Fisher test completed in {time.time() - start:.1f}s")
    except Exception as e:
        print(f"\n❌ Tully-Fisher test failed: {e}")
        import traceback
        traceback.print_exc()
        tf_results = None
    
    # Test 2: Power-law optimization
    print("\n" + "="*80)
    print("TEST 2: POWER-LAW OPTIMIZATION")
    print("="*80)
    print("Finding best-fit exponents for λ ~ M^α × v^β × R^γ")
    print("Expected time: 1-2 minutes")
    
    start = time.time()
    try:
        from optimize_power_law import run_power_law_optimization
        pl_results = run_power_law_optimization(sparc_path)
        print(f"\n✓ Power-law optimization completed in {time.time() - start:.1f}s")
    except Exception as e:
        print(f"\n❌ Power-law optimization failed: {e}")
        import traceback
        traceback.print_exc()
        pl_results = None
    
    # Test 3: Comprehensive scale library
    print("\n" + "="*80)
    print("TEST 3: COMPREHENSIVE SCALE LIBRARY")
    print("="*80)
    print("Testing multiple physical scale hypotheses")
    print("Expected time: 2-5 minutes")
    
    start = time.time()
    try:
        from scale_finder import run_scale_finder_tests
        scale_results = run_scale_finder_tests(sparc_path)
        print(f"\n✓ Scale library test completed in {time.time() - start:.1f}s")
    except Exception as e:
        print(f"\n❌ Scale library test failed: {e}")
        import traceback
        traceback.print_exc()
        scale_results = None
    
    # Generate summary report
    print("\n" + "="*80)
    print("GENERATING SUMMARY REPORT")
    print("="*80)
    
    report = generate_summary_report(tf_results, pl_results, scale_results)
    
    report_path = "GravityWaveTest/SCALE_FINDING_REPORT.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n✓ Summary report saved to {report_path}")
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    print("\nResults:")
    print(f"  - Tully-Fisher results: GravityWaveTest/tully_fisher_results.json")
    print(f"  - Power-law results: GravityWaveTest/power_law_fits/optimized_params.json")
    print(f"  - Scale library results: GravityWaveTest/scale_tests/scale_test_results.json")
    print(f"  - Summary report: {report_path}")
    print(f"\nPlots:")
    print(f"  - Tully-Fisher: GravityWaveTest/tully_fisher_scaling_test.png")
    print(f"  - Power-law: GravityWaveTest/power_law_fits/optimized_power_law.png")
    print(f"  - Scale library: GravityWaveTest/scale_tests/*_diagnostic.png")

def generate_summary_report(tf_results, pl_results, scale_results):
    """Generate markdown summary report."""
    
    report = f"""# Σ-Gravity Scale-Finding Test Results

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report summarizes the results of comprehensive tests to determine the physical origin of the coherence length scale in Σ-Gravity theory.

---

## Test 1: Tully-Fisher Scaling

**Hypothesis**: If λ ∝ √M_b, then v⁴ ∝ M_b (baryonic Tully-Fisher relation)

"""
    
    if tf_results:
        gamma = tf_results['gamma_fit']
        tf_slope = tf_results['TF_slope']
        
        report += f"""**Results**:
- Power-law index: γ = {gamma:.4f} (expected: 0.5)
- Tully-Fisher slope: {tf_slope:.4f} (expected: 1.0)
- Median λ_g: {tf_results['median_lambda_kpc']:.3f} kpc
- Fitted ℓ₀: {tf_results['fitted_ell0_kpc']:.3f} kpc
- Scatter: {tf_results['scatter_dex']:.4f} dex
- R²: {tf_results['r_squared']:.4f}

**Interpretation**:
"""
        
        if abs(gamma - 0.5) < 0.1 and abs(tf_slope - 1.0) < 0.2:
            report += "✓✓ **STRONG EVIDENCE**: λ_g ∝ √M_b as predicted by Tully-Fisher!\n"
        elif abs(gamma - 0.5) < 0.2:
            report += "✓ **MODERATE EVIDENCE**: λ_g weakly scales with √M_b\n"
        else:
            report += "✗ **NO EVIDENCE**: λ_g does not scale as √M_b\n"
    else:
        report += "❌ Test failed or not run\n"
    
    report += "\n![Tully-Fisher Plot](tully_fisher_scaling_test.png)\n\n"
    
    report += "---\n\n## Test 2: Power-Law Optimization\n\n"
    report += "**Hypothesis**: λ ~ M_b^α × v^β × R^γ (optimize exponents)\n\n"
    
    if pl_results:
        params, diag = pl_results
        report += f"""**Results**:
- Optimized formula: ℓ₀ = {diag['scale']:.4f} × M_b^{diag['alpha_M']:.4f} × v^{diag['alpha_v']:.4f} × R^{diag['alpha_R']:.4f}
- Median predicted: {diag['median_ell0']:.4f} kpc (target: 4.993 kpc)
- Scatter: {diag['scatter_dex']:.5f} dex
- Valid galaxies: {diag['n_valid']}

**Physical Interpretation**:
"""
        
        if abs(diag['alpha_M'] - 0.5) < 0.1:
            report += "- α_M ≈ 0.5 → consistent with Tully-Fisher (v⁴ ∝ M)\n"
        if abs(diag['alpha_v'] + 2.0) < 0.2:
            report += "- α_v ≈ -2 → consistent with ℓ ~ GM/v²\n"
        if abs(diag['alpha_R']) < 0.1:
            report += "- α_R ≈ 0 → scale independent of R_disk\n"
    else:
        report += "❌ Test failed or not run\n"
    
    report += "\n![Power-Law Plot](power_law_fits/optimized_power_law.png)\n\n"
    
    report += "---\n\n## Test 3: Scale Library Comparison\n\n"
    report += "**Multiple hypotheses tested** against SPARC data\n\n"
    
    if scale_results and len(scale_results) > 0:
        report += "**Top 5 Best-Fitting Scales** (by scatter):\n\n"
        report += "| Rank | Hypothesis | Median ℓ₀ [kpc] | Scatter [dex] | BIC |\n"
        report += "|------|------------|----------------|---------------|-----|\n"
        
        for i, r in enumerate(scale_results[:5]):
            report += f"| {i+1} | {r.hypothesis_name} | {r.ell0_median:.3f} | {r.scatter_dex:.4f} | {r.bic:.1f} |\n"
        
        report += f"\n**Total hypotheses tested**: {len(scale_results)}\n"
        report += "\nSee `scale_tests/scale_test_results.json` for full results.\n"
    else:
        report += "❌ Test failed or not run\n"
    
    report += """
---

## Conclusions

### Key Findings

1. **Tully-Fisher Consistency**: [Summarize if γ ≈ 0.5]
2. **Best-Fit Power Law**: [Summarize exponents]
3. **Physical Scale**: [Summarize which physical scale works best]

### Recommended Next Steps

1. If Tully-Fisher scaling confirmed → λ is intrinsically mass-dependent
2. If power-law optimization finds simple exponents → dimensional analysis path
3. If specific physical scale wins → connect to that mechanism

### Files Generated

- `tully_fisher_results.json` - Detailed TF test results
- `tully_fisher_scaling_test.png` - Diagnostic plots
- `power_law_fits/optimized_params.json` - Best-fit power law
- `power_law_fits/optimized_power_law.png` - Optimization diagnostics
- `scale_tests/scale_test_results.json` - All hypothesis results
- `scale_tests/*_diagnostic.png` - Individual hypothesis plots

---

*Generated by GravityWaveTest suite*
"""
    
    return report

if __name__ == "__main__":
    run_all_scale_tests()

