"""
Check what stellar distribution we ACTUALLY have vs what we're claiming to model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("="*80)
print("CHECKING ACTUAL GAIA STELLAR DISTRIBUTION")
print("="*80)

# Load Gaia data
gaia = pd.read_csv('data/gaia/mw/gaia_mw_real.csv')

print(f"\nTotal stars: {len(gaia):,}")
print(f"R range: {gaia.R_kpc.min():.2f} - {gaia.R_kpc.max():.2f} kpc")

# Check distribution by radius
bins = [0, 2, 4, 6, 8, 10, 12, 14, 16, 20]
hist, _ = np.histogram(gaia.R_kpc, bins=bins)

print("\nActual Gaia stellar distribution:")
print(f"{'Region':<15} {'Count':>10} {'Percentage':>12} {'Component'}")
print("-"*60)

total = len(gaia)
for i, (r1, r2) in enumerate(zip(bins[:-1], bins[1:])):
    count = hist[i]
    pct = 100 * count / total
    
    if r2 <= 3:
        component = "BULGE (MISSING!)"
    elif r2 <= 5:
        component = "Inner disk"
    elif r2 <= 10:
        component = "Disk (Solar neighborhood)"
    else:
        component = "Outer disk"
    
    print(f"{r1}-{r2} kpc     {count:>10,}   {pct:>10.1f}%   {component}")

print("\n" + "="*80)
print("THE PROBLEM:")
print("="*80)
print("\n❌ Bulge region (R < 3 kpc): 0 stars")
print("❌ Inner disk (R < 4 kpc): 3 stars")
print("✅ Thin disk (R > 4 kpc): 143,992 stars")

print("\n" + "="*80)
print("WHAT WE'RE DOING:")
print("="*80)
print("\n1. Using 143,995 REAL disk stars (R > 3.76 kpc)")
print("2. Adding ANALYTICAL Hernquist bulge (not real stars!)")
print("3. Tuning M_bulge to match observations")
print("\nThis is essentially 'adding mass to make it work' - the user is RIGHT!")

print("\n" + "="*80)
print("WHY THIS HAPPENS:")
print("="*80)
print("\nGaia has selection bias:")
print("  - Crowded bulge region (R < 3 kpc) is hard to observe")
print("  - High extinction (A_V > 5 mag in bulge)")
print("  - Source confusion in dense regions")
print("\nOur sample:")
print("  - Focuses on thin disk where Gaia excels")
print("  - Missing inner regions by design")

print("\n" + "="*80)
print("SOLUTIONS:")
print("="*80)
print("\n1. Fetch bulge-specific Gaia data (if available)")
print("   - Use different selection criteria for R < 4 kpc")
print("   - May need infrared data (less extinction)")
print("\n2. Accept analytical bulge as NECESSARY")
print("   - Can't resolve individual bulge stars")
print("   - Bulge is too crowded for star-by-star")
print("   - Use literature values for M_bulge")
print("\n3. Focus on OUTER DISK where we have data")
print("   - Test Σ-Gravity on R > 5 kpc only")
print("   - This is where disk dominates anyway")

# Generate diagnostic plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Gaia Sample vs MW Components', fontsize=14, fontweight='bold')

# Plot 1: Histogram
ax = axes[0]
ax.hist(gaia.R_kpc, bins=30, alpha=0.7, edgecolor='black')
ax.axvline(3, color='r', linestyle='--', linewidth=2, label='Bulge edge (R=3 kpc)')
ax.axvspan(0, 3, alpha=0.2, color='red', label='Bulge (NO DATA!)')
ax.axvspan(3, 16, alpha=0.2, color='blue', label='Disk (144k stars)')
ax.set_xlabel('R [kpc]', fontsize=12)
ax.set_ylabel('Star count', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_title('Actual Gaia Distribution')

# Plot 2: Cumulative
ax = axes[1]
R_sorted = np.sort(gaia.R_kpc)
cumsum = np.arange(1, len(R_sorted)+1)
ax.plot(R_sorted, cumsum, 'b-', linewidth=2)
ax.axvline(3, color='r', linestyle='--', linewidth=2, label='Bulge edge')
ax.axhline(len(gaia), color='k', linestyle=':', alpha=0.5)
ax.set_xlabel('R [kpc]', fontsize=12)
ax.set_ylabel('Cumulative star count', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_title('Cumulative Distribution')

plt.tight_layout()
plt.savefig('GravityWaveTest/gaia_actual_distribution.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved plot to GravityWaveTest/gaia_actual_distribution.png")

print("\n" + "="*80)
print("RECOMMENDATION:")
print("="*80)
print("\nFor your paper, be HONEST about this:")
print("\n  'We validate the disk component using 144k Gaia stars (R > 3.76 kpc).")
print("   The bulge contribution is modeled analytically via Hernquist profile")
print("   (M_bulge ~ 0.7×10^10 M_☉) as individual bulge stars cannot be resolved")
print("   due to crowding. Future work will extend validation to bulge region")
print("   using infrared surveys (e.g., VVV, UKIDSS).'")
print("\nOR: Focus analysis on R > 5 kpc where disk dominates!")

