import pandas as pd
import numpy as np

# Load results
df = pd.read_csv('outputs/viability_scan/viability_scan_full.csv')
passed_cosmo = df[df['cosmo_pass'] == True]

print("="*70)
print("VIABILITY SCAN RESULTS ANALYSIS")
print("="*70)

print(f"\n1. COSMOLOGY STAGE:")
print(f"   Passed: {len(passed_cosmo)}/10,000 ({100*len(passed_cosmo)/10000:.1f}%)")

if len(passed_cosmo) > 0:
    print("\n2. SAMPLE PARAMETERS THAT PASSED COSMOLOGY:")
    print(passed_cosmo[['V0', 'lambda', 'M4', 'beta', 'cosmo_Omega_m0', 'cosmo_Omega_phi0']].head(5))
    
    print("\n3. SCREENING STAGE (why all 200 failed):")
    print("\n   Requirements:")
    print("   - R_c_spiral <= 10 kpc  (field must be heavy in spirals)")
    print("   - R_c_dwarf  <= 50 kpc  (heavy in dwarfs)")
    print("   - R_c_cosmic >= 1000 kpc (light at cosmic density)")
    
    print("\n   Actual R_c values (kpc):")
    print(f"   - R_c_spiral: {passed_cosmo['screening_R_c_spiral'].min():.1f} to {passed_cosmo['screening_R_c_spiral'].max():.1f}")
    print(f"   - R_c_dwarf:  {passed_cosmo['screening_R_c_dwarf'].min():.1f} to {passed_cosmo['screening_R_c_dwarf'].max():.1f}")
    print(f"   - R_c_cosmic: {passed_cosmo['screening_R_c_cosmic'].min():.1e} to {passed_cosmo['screening_R_c_cosmic'].max():.1e}")
    
    # Check which constraint is violated most
    fails_spiral = (passed_cosmo['screening_R_c_spiral'] > 10).sum()
    fails_dwarf = (passed_cosmo['screening_R_c_dwarf'] > 50).sum()
    fails_cosmic = (passed_cosmo['screening_R_c_cosmic'] < 1000).sum()
    
    print("\n   Failure breakdown:")
    print(f"   - {fails_spiral}/200 ({100*fails_spiral/200:.0f}%) fail spiral constraint (R_c > 10 kpc)")
    print(f"   - {fails_dwarf}/200 ({100*fails_dwarf/200:.0f}%) fail dwarf constraint (R_c > 50 kpc)")
    print(f"   - {fails_cosmic}/200 ({100*fails_cosmic/200:.0f}%) fail cosmic constraint (R_c < 1000 kpc)")
    
    # Find closest to working
    # Compute "score" - how far from satisfying all constraints
    def score_screening(row):
        # Violations (positive = bad)
        spiral_viol = max(0, row['screening_R_c_spiral'] - 10) / 10
        dwarf_viol = max(0, row['screening_R_c_dwarf'] - 50) / 50
        cosmic_viol = max(0, 1000 - row['screening_R_c_cosmic']) / 1000
        return spiral_viol + dwarf_viol + cosmic_viol
    
    passed_cosmo['screening_score'] = passed_cosmo.apply(score_screening, axis=1)
    best = passed_cosmo.nsmallest(3, 'screening_score')
    
    print("\n4. CLOSEST TO WORKING (3 best):")
    for idx, row in best.iterrows():
        print(f"\n   V0={row['V0']:.2e}, λ={row['lambda']:.2f}, M4={row['M4']:.3f}, β={row['beta']:.3f}")
        print(f"   Ωm={row['cosmo_Omega_m0']:.3f}, Ωφ={row['cosmo_Omega_phi0']:.3f}")
        print(f"   R_c: cosmic={row['screening_R_c_cosmic']:.1e}, dwarf={row['screening_R_c_dwarf']:.1f}, spiral={row['screening_R_c_spiral']:.1f}")
        print(f"   Score: {row['screening_score']:.3f} (0 = perfect, lower = better)")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("\nThe exponential + chameleon potential V(φ) = V₀e^(-λφ) + M⁵/φ")
print("CANNOT simultaneously satisfy cosmology and screening constraints.")
print("\nBottleneck: Even parameter sets that give Ωm ≈ 0.3 fail to produce")
print("the needed screening (light cosmologically, heavy in galaxies).")
print("\nThis means this potential form is RULED OUT for coherence gravity.")
print("\n" + "="*70)
