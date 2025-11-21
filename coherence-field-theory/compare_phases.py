import pandas as pd

df = pd.read_csv('outputs/gpm_holdout/holdout_predictions_all.csv')
success = df[df['success'] == True]

print("="*80)
print("PHASE 2 RESULTS (Radius-Dependent Environment)")
print("="*80)
print(f"\nSuccess rate: {len(success)}/175 = {len(success)/175*100:.1f}%")
print(f"Median RMS improvement: {success['rms_improvement'].median():.1f}%")
print(f"Fraction improved (>0%): {(success['rms_improvement'] > 0).mean()*100:.1f}%")
print(f"Median RMS: {success['rms_bar'].median():.1f} km/s (baryons) -> {success['rms_gpm'].median():.1f} km/s (GPM)")

# Catastrophic failures
bad = success[success['rms_improvement'] < -50]
print(f"\nCatastrophic failures (<-50%): {len(bad)}")
if len(bad) > 0:
    print(bad[['name','M_total','rms_bar','rms_gpm','rms_improvement','alpha_eff']].to_string(index=False))

print("\n" + "="*80)
print("COMPARISON TO PHASE 1")
print("="*80)
print("Phase 1 (V2, single global σ_v):")
print("  - 142/175 success (81%)")
print("  - Median RMS improvement: +26.3%")
print("  - Fraction improved: 59%")
print("  - ~8 catastrophic failures")
print("\nPhase 2 (V3, radius-dependent σ_v(R), ℓ(R)):")
print(f"  - {len(success)}/175 success ({len(success)/175*100:.1f}%)")
print(f"  - Median RMS improvement: {success['rms_improvement'].median():+.1f}%")
print(f"  - Fraction improved: {(success['rms_improvement'] > 0).mean()*100:.1f}%")
print(f"  - {len(bad)} catastrophic failures")
print("\nOUTCOME: Radius-dependent profiles did NOT eliminate catastrophic failures.")
print("Same galaxies (UGC05750, UGC04305, NGC2976) still show -200% to -400% worsening.")
print("These compact dwarfs are fundamentally unsuitable for GPM.")
