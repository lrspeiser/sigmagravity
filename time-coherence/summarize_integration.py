"""Summarize integration results."""

import pandas as pd
import json

# Load data
df = pd.read_csv("time-coherence/sparc_roughness_amplitude.csv")
corr = json.load(open("time-coherence/F_missing_correlations.json"))

print("=" * 80)
print("INTEGRATION SUMMARY")
print("=" * 80)

print(f"\nSPARC Roughness Amplitude Test ({len(df)} galaxies):")
print(f"  Mean K_rough: {df['K_rough'].mean():.3f}")
print(f"  Median K_rough: {df['K_rough'].median():.3f}")
print(f"  Mean A_empirical: {df['A_empirical'].mean():.2f}")
print(f"  Median A_empirical: {df['A_empirical'].median():.2f}")

valid_F = df['F_missing'].dropna()
print(f"\nF_missing Statistics:")
print(f"  Mean: {valid_F.mean():.2f}")
print(f"  Median: {valid_F.median():.2f}")
print(f"  Std: {valid_F.std():.2f}")
print(f"  Min: {valid_F.min():.2f}")
print(f"  Max: {valid_F.max():.2f}")
print(f"  Roughness explains ~{100/valid_F.mean():.1f}% of enhancement")

print(f"\nPerformance:")
print(f"  Mean delta_RMS: {df['delta_rms'].mean():.2f} km/s")
print(f"  Median delta_RMS: {df['delta_rms'].median():.2f} km/s")
print(f"  Galaxies improved: {(df['delta_rms'] > 0).sum()}/{len(df)}")

print(f"\n" + "=" * 80)
print("F_MISSING CORRELATIONS")
print("=" * 80)

for prop, stats in sorted(corr.items(), key=lambda x: abs(x[1]['spearman_r']), reverse=True):
    print(f"\n{prop}:")
    print(f"  Spearman r: {stats['spearman_r']:.3f} (p={stats['spearman_p']:.2e})")
    print(f"  Pearson r: {stats['pearson_r']:.3f} (p={stats['pearson_p']:.2e})")
    print(f"  N points: {stats['n_points']}")
    if abs(stats['spearman_r']) > 0.3 and stats['spearman_p'] < 0.05:
        print(f"  *** SIGNIFICANT CORRELATION ***")

print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)
print("\n1. Roughness explains ~10% of total enhancement")
print("2. F_missing shows strong NEGATIVE correlations:")
print("   - sigma_v: r = -0.585 (strongest)")
print("   - R_d: r = -0.489")
print("   - bulge_frac: r = -0.442")
print("\n3. Second mechanism must be:")
print("   - Velocity/dispersion-dependent")
print("   - Suppressed in high-sigma_v systems")
print("   - Suppressed in large discs")
print("\n4. Next step: Fit F_missing(sigma_v, R_d) functional form")

