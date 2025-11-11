#!/usr/bin/env python3
"""Quick summary of PCA + Sigma-Gravity integration results"""
import pandas as pd
import numpy as np

print("=" * 80)
print("SIGMA-GRAVITY + PCA INTEGRATION: FINAL RESULTS")
print("=" * 80)

# Load Sigma-Gravity fits
fits = pd.read_csv('pca/outputs/sigmagravity_fits/sparc_sigmagravity_fits.csv')

print(f"\n{'='*80}")
print("1. SIGMA-GRAVITY FIT QUALITY")
print("=" * 80)

print(f"\nTotal galaxies fitted: {len(fits)}")

print(f"\nResidual Statistics:")
print(f"  Mean RMS:   {fits['residual_rms'].mean():.2f} km/s")
print(f"  Median RMS: {fits['residual_rms'].median():.2f} km/s")
print(f"  Std RMS:    {fits['residual_rms'].std():.2f} km/s")
print(f"  Min RMS:    {fits['residual_rms'].min():.2f} km/s")
print(f"  Max RMS:    {fits['residual_rms'].max():.2f} km/s")

print(f"\nChi-squared Statistics:")
print(f"  Mean chi2_red:   {fits['chi2_red'].mean():.2f}")
print(f"  Median chi2_red: {fits['chi2_red'].median():.2f}")

print(f"\nWorst 10 Fits (by RMS):")
worst = fits.nlargest(10, 'residual_rms')[['name', 'residual_rms', 'Mbar', 'Rd', 'Sigma0']]
print(worst.to_string(index=False))

print(f"\nBest 10 Fits (by RMS):")
best = fits.nsmallest(10, 'residual_rms')[['name', 'residual_rms', 'Mbar', 'Rd', 'Sigma0']]
print(best.to_string(index=False))

# Load PCA comparison
print(f"\n{'='*80}")
print("2. PCA COMPARISON TEST RESULTS")
print("=" * 80)

with open('pca/outputs/model_comparison/comparison_summary.txt', 'r') as f:
    print(f.read())

# Load PCA results
pca = np.load('pca/outputs/pca_results_curve_only.npz', allow_pickle=True)
names_pca = pca['names']
scores = pca['scores']
evr = pca['evr']

print(f"\n{'='*80}")
print("3. PCA STRUCTURE (REMINDER)")
print("=" * 80)

print(f"\nVariance Explained:")
print(f"  PC1: {evr[0]*100:.1f}% (mass-velocity mode)")
print(f"  PC2: {evr[1]*100:.1f}% (scale-length mode)")
print(f"  PC3: {evr[2]*100:.1f}% (density mode)")
print(f"  PC1-3 total: {evr[:3].sum()*100:.1f}%")

# Load metadata for correlations
meta = pd.read_csv('pca/data/raw/metadata/sparc_meta.csv')

# Merge everything
pc_df = pd.DataFrame({
    'name': names_pca,
    'PC1': scores[:, 0],
    'PC2': scores[:, 1],
    'PC3': scores[:, 2]
})

print(f"\n{'='*80}")
print("4. DIAGNOSTIC PATTERNS")
print("=" * 80)

from scipy.stats import spearmanr

print(f"\nResidual RMS correlations with physical parameters:")
# Fits already has Mbar, Rd, Sigma0, Vf from the fitting script
for col in ['Mbar', 'Rd', 'Sigma0', 'Vf']:
    if col in fits.columns:
        mask = np.isfinite(fits['residual_rms']) & np.isfinite(fits[col]) & (fits[col] > 0)
        if mask.sum() > 10:
            rho, p = spearmanr(np.log10(fits.loc[mask, col]), fits.loc[mask, 'residual_rms'])
            print(f"  log10({col:8s}) vs residual: rho = {rho:+.3f}, p = {p:.3e}")

print(f"\n{'='*80}")
print("5. CONCLUSIONS")
print("=" * 80)

print("""
CURRENT MODEL (Fixed Parameters):
  - A = 0.6 (fixed)
  - l0 = 5.0 kpc (fixed)
  - Mean RMS = 33.9 km/s
  - FAILS PCA test (rho = +0.459 with PC1)

DIAGNOSIS:
  - PC1 correlation (+0.459): Need mass-dependent amplitude A(Mbar)
  - PC2 correlation (+0.406): Need scale-dependent coherence l0(Rd)
  - PC3 correlation (-0.316): Need density-dependent shape p(Sigma0)

RECOMMENDED MODEL (Parameter Scalings):
  - A = A0 * (Mbar / 10^9)^alpha
  - l0 = l0_base * (Rd / 5 kpc)^beta
  - p = p0 + p1 * log10(Sigma0 / 100)
  
  Total: ~7 global parameters for entire population
  vs ~525 parameters for LCDM (3 per galaxy x 175 galaxies)

EXPECTED IMPROVEMENT:
  - |rho(residual, PC1)| < 0.2 (pass threshold)
  - Mean RMS < 20 km/s (50% improvement)
  - Physically motivated parameter scalings
  - Competitive with MOND, works for clusters too

NEXT STEP:
  Implement parameter scalings in 10_fit_sigmagravity_to_sparc.py
  and re-run analysis.
""")

print("=" * 80)
print("Analysis complete. See pca/SIGMAGRAVITY_RESULTS.md for details.")
print("=" * 80)

