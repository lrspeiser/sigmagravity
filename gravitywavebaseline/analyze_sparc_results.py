"""Quick analysis of SPARC theory kernel results."""

import pandas as pd
import numpy as np

df = pd.read_csv("gravitywavebaseline/theory_kernel_sparc_from_mw_improved.csv")

print("=== SPARC Theory Kernel Results ===")
print(f"Total galaxies: {len(df)}")
print(f"\nOverall Statistics:")
print(f"  Mean delta_rms: {df['delta_rms'].mean():.3f} km/s")
print(f"  Median delta_rms: {df['delta_rms'].median():.3f} km/s")
print(f"  Std delta_rms: {df['delta_rms'].std():.3f} km/s")
print(f"\n  Mean RMS GR: {df['rms_gr'].mean():.3f} km/s")
print(f"  Mean RMS Theory: {df['rms_theory'].mean():.3f} km/s")
print(f"\n  Galaxies improved (delta_rms < 0): {(df['delta_rms'] < 0).sum()} / {len(df)} ({(df['delta_rms'] < 0).sum()/len(df)*100:.1f}%)")
print(f"  Galaxies worsened (delta_rms > 0): {(df['delta_rms'] > 0).sum()} / {len(df)} ({(df['delta_rms'] > 0).sum()/len(df)*100:.1f}%)")

print(f"\nBy sigma_v bins:")
df['sigma_bin'] = pd.cut(df['sigma_v_true'], [0, 20, 25, 30, 35, 40, 100], labels=['<20', '20-25', '25-30', '30-35', '35-40', '>40'])
binned = df.groupby('sigma_bin')['delta_rms'].agg(['count', 'mean', 'median', 'std']).round(3)
print(binned)

print(f"\nCorrelation analysis:")
print(f"  corr(sigma_v, delta_rms): {df['sigma_v_true'].corr(df['delta_rms']):.3f}")
print(f"  corr(sigma_v, K_mean): {df['sigma_v_true'].corr(df['K_mean']):.3f}")

print(f"\nWorst performers (largest delta_rms):")
worst = df.nlargest(10, 'delta_rms')[['galaxy', 'sigma_v_true', 'delta_rms', 'rms_gr', 'rms_theory']]
print(worst.to_string(index=False))

print(f"\nBest performers (most negative delta_rms):")
best = df.nsmallest(10, 'delta_rms')[['galaxy', 'sigma_v_true', 'delta_rms', 'rms_gr', 'rms_theory']]
print(best.to_string(index=False))


