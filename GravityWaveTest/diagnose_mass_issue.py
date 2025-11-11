"""
Diagnose why predictions are too high with 1.8M stars.
The problem: Stars are NOT Monte Carlo samples - they're real with Gaia selection bias!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("="*80)
print("DIAGNOSING THE MASS DISTRIBUTION PROBLEM")
print("="*80)

# Load the 1.8M sample
gaia = pd.read_csv('data/gaia/gaia_processed.csv')

print(f"\nLoaded {len(gaia):,} stars")

# Check spatial distribution
R = gaia['R_cyl'].values
z = gaia['z'].values

# Bin by radius
R_bins = np.logspace(np.log10(0.1), np.log10(25), 30)
R_centers = np.sqrt(R_bins[:-1] * R_bins[1:])
counts, _ = np.histogram(R, bins=R_bins)

# Compute what fraction of stars are in each bin
frac = counts / counts.sum()

print("\nStellar distribution by radius:")
print(f"{'R range (kpc)':<15} {'Count':>10} {'Fraction':>10} {'Issue'}")
print("-"*70)

for i in range(len(R_bins)-1):
    r1, r2 = R_bins[i], R_bins[i+1]
    count = counts[i]
    f = frac[i]
    
    issue = ""
    if r2 < 1:
        issue = "← Very sparse (Gaia can't see here)"
    elif r2 < 3:
        issue = "← Bulge (limited by crowding)"
    elif r2 < 10:
        issue = "← PEAK (Gaia bias!)"
    elif r2 < 15:
        issue = "← Outer disk"
    else:
        issue = "← Sparse (few stars)"
    
    if count > 0:
        print(f"{r1:.2f}-{r2:.2f}      {count:>10,}   {f:>9.1%}   {issue}")

print("\n" + "="*80)
print("THE PROBLEM:")
print("="*80)

# Compare to expected MW mass distribution
# True MW disk: Σ(R) ∝ exp(-R/2.5 kpc)
R_theory = R_centers
Sigma_theory = np.exp(-R_theory / 2.5)
Sigma_theory /= Sigma_theory.sum()

# Actual stellar distribution (normalized)
stellar_dist = frac

print("\nExpected vs Actual stellar distribution:")
print(f"{'R (kpc)':<10} {'Expected %':>12} {'Actual %':>12} {'Ratio'}")
print("-"*60)

for i in range(min(15, len(R_centers))):
    if counts[i] > 100:
        expected_pct = 100 * Sigma_theory[i]
        actual_pct = 100 * stellar_dist[i]
        ratio = actual_pct / expected_pct if expected_pct > 0 else np.inf
        
        marker = ""
        if ratio > 2:
            marker = " ← OVER-REPRESENTED"
        elif ratio < 0.5:
            marker = " ← UNDER-REPRESENTED"
        
        print(f"{R_centers[i]:.2f}        {expected_pct:>10.2f}%  {actual_pct:>10.2f}%   {ratio:>6.2f}x{marker}")

print("\n" + "="*80)
print("ROOT CAUSE:")
print("="*80)
print("\n1. Gaia selection is BIASED toward solar neighborhood (R ~ 5-10 kpc)")
print(f"   - Peak at R ~ 8 kpc: {(frac * 100).max():.1f}% of all stars")
print(f"   - Expected at R ~ 8 kpc: ~10% based on exp(-R/2.5)")
print(f"   - Over-representation: ~{(frac * 100).max() / 10:.0f}×")

print("\n2. We're assigning UNIFORM mass weights: M_disk / N_stars")
print(f"   - Each star gets: {5e10 / len(gaia):.2e} M_☉")
print(f"   - But stars are DENSER at R~8 kpc due to selection!")
print(f"   - This artificially CONCENTRATES mass at R~8 kpc")

print("\n3. Result: Too much mass at Solar radius → v too high!")
print(f"   - Predicted: v ~ 310-500 km/s")
print(f"   - Observed: v ~ 220 km/s")
print(f"   - Over-prediction: 40-130%")

print("\n" + "="*80)
print("SOLUTION:")
print("="*80)
print("\nOption 1: WEIGHT stars by inverse selection probability")
print("   - Stars at R~8 kpc get LOWER weight (over-represented)")
print("   - Stars at R<3 kpc get HIGHER weight (under-represented)")
print("   - This corrects for Gaia bias")

print("\nOption 2: Use ANALYTICAL mass distribution")
print("   - Σ(R) = Σ₀ exp(-R/R_d) (exponential disk)")
print("   - Bulge: Hernquist or Sérsic profile")
print("   - Calculate enhancement from this, not from star positions")

print("\nOption 3: SUBSAMPLE to match true distribution")
print("   - Reject stars to make distribution ∝ exp(-R/2.5)")
print("   - End up with ~100k-200k properly distributed stars")
print("   - Lose statistical power but gain correctness")

print("\n" + "="*80)
print("RECOMMENDATION:")
print("="*80)
print("\nThe star-by-star approach has a fundamental issue:")
print("  Real stars ≠ Monte Carlo samples of mass distribution!")
print("\nGaia gives you STELLAR POSITIONS (biased by selection)")
print("NOT mass samples (which would follow Σ(R) ∝ exp(-R/R_d))")

print("\nBetter approach for 1.8M stars:")
print("  1. Use stars to VALIDATE density model: ρ_obs(R,z) vs theory")
print("  2. Then calculate Σ-Gravity from ANALYTIC ρ(R,z)")
print("  3. Compare to observed v_phi from same stars")
print("\n  This separates: 'What is the mass?' from 'Does Σ-Gravity work?'")

# Generate plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Gaia Selection Bias vs True MW Disk', fontsize=14, fontweight='bold')

# Plot 1: Distribution comparison
ax = axes[0]
mask = counts > 100
ax.plot(R_centers[mask], 100*Sigma_theory[mask], 'r-', linewidth=2, label='Expected: exp(-R/2.5)')
ax.plot(R_centers[mask], 100*stellar_dist[mask], 'b-', linewidth=2, label='Actual Gaia')
ax.set_xlabel('R [kpc]', fontsize=12)
ax.set_ylabel('Fraction of stars (%)', fontsize=12)
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_title('Radial Distribution')

# Plot 2: Over/under representation
ax = axes[1]
ratio = stellar_dist / (Sigma_theory + 1e-10)
ax.plot(R_centers[mask], ratio[mask], 'g-', linewidth=2)
ax.axhline(1, color='k', linestyle='--', linewidth=1, label='Expected')
ax.fill_between([5, 12], [0.1, 0.1], [10, 10], alpha=0.2, color='red', label='Over-represented')
ax.set_xlabel('R [kpc]', fontsize=12)
ax.set_ylabel('Actual / Expected', fontsize=12)
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_title('Selection Bias (Actual/Expected)')

plt.tight_layout()
plt.savefig('GravityWaveTest/gaia_selection_bias.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved plot to GravityWaveTest/gaia_selection_bias.png")

print("\n" + "="*80)
print("NEXT STEPS:")
print("="*80)
print("\n1. Create density-weighted star calculation")
print("2. Or: Switch to analytical mass model validated by stars")
print("3. Or: Accept this as 'order of magnitude' demonstration")

