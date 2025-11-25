"""Quick analysis of CMSI outliers."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fitting.cmsi_parameter_sweep import *

data_dir = str(Path(__file__).parent.parent.parent / "data" / "Rotmod_LTG")
galaxies = load_all_sparc(data_dir)

params = CMSIParams(chi_0=500, alpha_Ncoh=0.45, ell_0_kpc=3.0)
eval_result = evaluate_batch(galaxies, params)

# Sort by delta_rms to find worst
results = sorted(eval_result['results'], key=lambda r: r['delta_rms'], reverse=True)

print("WORST 15 GALAXIES (over-boosted by CMSI):")
print("=" * 80)
for r in results[:15]:
    print(f"{r['name']:20s} | v_flat={r['v_flat']:6.1f} | RMS: {r['rms_bary']:5.1f} -> {r['rms_cmsi']:5.1f} | dRMS={r['delta_rms']:+6.1f} | F={r['mean_F']:.2f}")

print()
print("BEST 15 GALAXIES (helped by CMSI):")
print("=" * 80)
for r in results[-15:]:
    print(f"{r['name']:20s} | v_flat={r['v_flat']:6.1f} | RMS: {r['rms_bary']:5.1f} -> {r['rms_cmsi']:5.1f} | dRMS={r['delta_rms']:+6.1f} | F={r['mean_F']:.2f}")

# Stats
improved = [r for r in results if r['improved']]
worsened = [r for r in results if not r['improved']]

print()
print("SUMMARY:")
print("=" * 80)
print(f"Total: {len(results)} galaxies")
print(f"Improved: {len(improved)} ({100*len(improved)/len(results):.1f}%)")
print(f"Worsened: {len(worsened)} ({100*len(worsened)/len(results):.1f}%)")

# Check patterns
print()
print("PATTERN ANALYSIS:")
print("-" * 40)
imp_vflat = [r['v_flat'] for r in improved]
wor_vflat = [r['v_flat'] for r in worsened]
print(f"Improved galaxies: mean v_flat = {np.mean(imp_vflat):.1f} km/s")
print(f"Worsened galaxies: mean v_flat = {np.mean(wor_vflat):.1f} km/s")

imp_bary_rms = [r['rms_bary'] for r in improved]
wor_bary_rms = [r['rms_bary'] for r in worsened]
print(f"Improved galaxies: mean bary RMS = {np.mean(imp_bary_rms):.1f} km/s")
print(f"Worsened galaxies: mean bary RMS = {np.mean(wor_bary_rms):.1f} km/s")
