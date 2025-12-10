"""Compare default vs fitted parameter results."""

import pandas as pd
import json

df1 = pd.read_csv('time-coherence/sparc_coherence_test.csv')
df2 = pd.read_csv('time-coherence/sparc_coherence_fitted_params.csv')

# Load fitted parameters
with open('time-coherence/time_coherence_fit_hyperparams.json', 'r') as f:
    fit_params = json.load(f)

print('=' * 80)
print('COMPARISON: Default vs Fitted Parameters')
print('=' * 80)

print('\nDefault Parameters (A=1.0, p=0.757, n_coh=0.5, alpha_length=0.037, beta_sigma=1.5):')
print(f'  Mean delta_rms: {df1["delta_rms"].mean():.3f} km/s')
print(f'  Median delta_rms: {df1["delta_rms"].median():.3f} km/s')
print(f'  Improved: {(df1["delta_rms"] < 0).sum()}/{len(df1)} ({(df1["delta_rms"] < 0).sum()/len(df1)*100:.1f}%)')
print(f'  Mean ell_coh: {df1["ell_coh_mean_kpc"].mean():.2f} kpc')
print(f'  Median ell_coh: {df1["ell_coh_mean_kpc"].median():.2f} kpc')

print(f'\nFitted Parameters (A={fit_params["A_global"]:.3f}, p={fit_params["p"]:.3f}, n_coh={fit_params["n_coh"]:.3f}, delta_R={fit_params["delta_R_kpc"]:.3f} kpc):')
print(f'  Mean delta_rms: {df2["delta_rms"].mean():.3f} km/s')
print(f'  Median delta_rms: {df2["delta_rms"].median():.3f} km/s')
print(f'  Improved: {(df2["delta_rms"] < 0).sum()}/{len(df2)} ({(df2["delta_rms"] < 0).sum()/len(df2)*100:.1f}%)')
print(f'  Mean ell_coh: {df2["ell_coh_mean_kpc"].mean():.2f} kpc')
print(f'  Median ell_coh: {df2["ell_coh_mean_kpc"].median():.2f} kpc')

print('\nChange:')
print(f'  Mean delta_rms: {df2["delta_rms"].mean() - df1["delta_rms"].mean():+.3f} km/s')
print(f'  Median delta_rms: {df2["delta_rms"].median() - df1["delta_rms"].median():+.3f} km/s')
print(f'  Improved rate: {(df2["delta_rms"] < 0).sum()/len(df2)*100 - (df1["delta_rms"] < 0).sum()/len(df1)*100:+.1f}%')
print(f'  Mean ell_coh: {df2["ell_coh_mean_kpc"].mean() - df1["ell_coh_mean_kpc"].mean():+.2f} kpc')

print('\n' + '=' * 80)


