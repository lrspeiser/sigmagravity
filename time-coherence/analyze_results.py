"""Analyze time-coherence test results."""

import pandas as pd
import json
from pathlib import Path

print("=" * 80)
print("TIME-COHERENCE TEST RESULTS SUMMARY")
print("=" * 80)

# SPARC results
if Path("time-coherence/sparc_coherence_test.csv").exists():
    df_sparc = pd.read_csv("time-coherence/sparc_coherence_test.csv")
    print("\n--- SPARC Galaxies ---")
    print(f"Total galaxies: {len(df_sparc)}")
    print(f"Mean delta_rms: {df_sparc['delta_rms'].mean():.3f} km/s")
    print(f"Median delta_rms: {df_sparc['delta_rms'].median():.3f} km/s")
    print(f"Std delta_rms: {df_sparc['delta_rms'].std():.3f} km/s")
    print(f"\nImproved (delta_rms < 0): {(df_sparc['delta_rms'] < 0).sum()}/{len(df_sparc)} ({(df_sparc['delta_rms'] < 0).sum()/len(df_sparc)*100:.1f}%)")
    print(f"Worsened (delta_rms > 0): {(df_sparc['delta_rms'] > 0).sum()}/{len(df_sparc)} ({(df_sparc['delta_rms'] > 0).sum()/len(df_sparc)*100:.1f}%)")
    print(f"\nCoherence scales:")
    print(f"  Mean ell_coh: {df_sparc['ell_coh_mean_kpc'].mean():.2f} kpc")
    print(f"  Median ell_coh: {df_sparc['ell_coh_mean_kpc'].median():.2f} kpc")
    print(f"  Mean tau_coh: {df_sparc['tau_coh_mean_yr'].mean():.2e} yr")
    print(f"\nBest 5 galaxies:")
    best = df_sparc.nsmallest(5, 'delta_rms')[['galaxy', 'delta_rms', 'ell_coh_mean_kpc', 'sigma_v_kms']]
    print(best.to_string(index=False))
    print(f"\nWorst 5 galaxies:")
    worst = df_sparc.nlargest(5, 'delta_rms')[['galaxy', 'delta_rms', 'ell_coh_mean_kpc', 'sigma_v_kms']]
    print(worst.to_string(index=False))

# MW results
if Path("time-coherence/mw_coherence_test.json").exists():
    mw_data = json.loads(Path("time-coherence/mw_coherence_test.json").read_text())
    print("\n--- Milky Way ---")
    print(f"R range: {mw_data['R_range_kpc']} kpc")
    print(f"sigma_v: {mw_data['sigma_v_kms']} km/s")
    print(f"\nBest match to ell0 ~ 5 kpc:")
    best_mw = min(mw_data['results'], key=lambda x: abs(x['ell_coh_mean_kpc'] - 5.0))
    print(f"  A={best_mw['A_global']}, p={best_mw['p']}, method={best_mw['tau_geom_method']}")
    print(f"  ell_coh_mean = {best_mw['ell_coh_mean_kpc']:.2f} kpc")
    print(f"  tau_coh = {best_mw['tau_coh_mean_yr']:.2e} yr")
    print(f"  K_max = {best_mw['K_max']:.4f}")

print("\n" + "=" * 80)


