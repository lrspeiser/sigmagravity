"""
Quick-start script for global viability scan.

This is the decisive test for the exponential + chameleon potential.
"""

import sys
from pathlib import Path

# Add analysis directory to path
sys.path.insert(0, str(Path(__file__).parent / 'analysis'))

from global_viability_scan import run_coarse_scan, run_fine_scan_near_viable

def main():
    print("\n" + "="*70)
    print("GLOBAL VIABILITY SCAN")
    print("="*70)
    print("\nTesting: V(φ) = V₀ exp(-λφ) + M⁵/φ, A(φ) = exp(βφ)")
    print("\nThis will test ~10,000 parameter combinations.")
    print("Estimated time: 10-30 minutes (depends on failures).")
    print("\nConstraints:")
    print("  1. Cosmology: Ω_m0 ∈ [0.25, 0.35], Ω_φ0 ∈ [0.65, 0.75]")
    print("  2. Screening: R_c^spiral ≤ 10 kpc, R_c^dwarf ≤ 50 kpc")
    print("  3. PPN: |γ-1| < 2.3×10⁻⁵ (placeholder for now)")
    print("="*70 + "\n")
    
    # Run coarse scan
    scanner, df, viable, summary = run_coarse_scan(n_per_param=10)
    
    # Interpret results
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    if len(viable) > 0:
        print(f"\n✅ SUCCESS: Found {len(viable)} viable parameter sets!")
        print("\nThis means the exponential + chameleon potential CAN work globally.")
        print("\nBest 5 parameter sets:")
        print(viable[['V0', 'lambda', 'M4', 'beta', 
                     'cosmo_Omega_m0', 'cosmo_Omega_phi0',
                     'screening_R_c_spiral', 'screening_R_c_dwarf']].head())
        
        print("\n" + "-"*70)
        print("NEXT STEPS:")
        print("-"*70)
        print("1. Review outputs/viability_scan/viability_scan_summary.png")
        print("2. Check outputs/viability_scan/viability_scan_viable.csv")
        print("3. Run full galaxy fits with these parameters")
        print("4. Implement proper PPN test to verify Solar System safety")
        print("\nOptionally, run fine scan around viable regions to refine parameters.")
        
    else:
        print("\n❌ FAILURE: No viable parameter sets found.")
        print("\nThis means the exponential + chameleon potential CANNOT simultaneously")
        print("satisfy cosmology, galaxy screening, and PPN constraints.")
        print("\n" + "-"*70)
        print("NEXT STEPS:")
        print("-"*70)
        print("1. Review outputs/viability_scan/viability_scan_summary.png")
        print("   to see which constraints are hardest to satisfy")
        print("\n2. Try next potential form:")
        print("   - Symmetron: V(φ) = -μ²φ²/2 + λφ⁴/4")
        print("   - K-mouflage: non-canonical kinetic term")
        print("   - Vainshtein: derivative interactions")
        print("\n3. See THEORY_LEVELS.md for the decision flowchart")
    
    print("="*70 + "\n")
    
    return scanner, df, viable, summary

if __name__ == '__main__':
    scanner, df, viable, summary = main()
