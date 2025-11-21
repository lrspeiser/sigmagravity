import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\henry\dev\sigmagravity\coherence-field-theory\outputs\gpm_holdout\holdout_predictions_successes.csv')

print(f'Total galaxies: {len(df)}')
print(f'Galaxies with alpha_eff = 0: {np.sum(df["alpha_eff"] == 0.0)}')
print(f'Galaxies with 0 < alpha_eff < 0.05: {np.sum((df["alpha_eff"] > 0) & (df["alpha_eff"] < 0.05))}')
print(f'Galaxies with alpha_eff >= 0.05: {np.sum(df["alpha_eff"] >= 0.05)}')

print(f'\nMassive galaxies (M>10^10) with alpha=0:')
massive_zero = df[(df["alpha_eff"] == 0) & (df["M_total"] > 1e10)][['name', 'M_total', 'rms_bar', 'rms_gpm', 'rms_improvement']]
print(massive_zero.sort_values('M_total', ascending=False).head(10))

print(f'\nComparison before/after floor:')
print(f'% with RMS improvement > 0: {np.sum(df["rms_improvement"] > 0) / len(df) * 100:.1f}%')
print(f'Median RMS improvement: {np.median(df["rms_improvement"]):.1f}%')

print(f'\nWorst performers (RMS < 0):')
worst = df[df["rms_improvement"] < 0][['name', 'M_total', 'alpha_eff', 'rms_bar', 'rms_gpm', 'rms_improvement']]
print(worst.sort_values('rms_improvement').head(10))
