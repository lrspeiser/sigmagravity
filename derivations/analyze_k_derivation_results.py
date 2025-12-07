#!/usr/bin/env python3
"""
Analyze the k derivation results and understand the discrepancy.

Key finding: Per-galaxy optimal k has median 0.37, but fixed k=0.24 works well.
This suggests k may not be the right parameter to vary, or there's a 
systematic in the fitting.

Author: Leonard Speiser
"""

import numpy as np
import json
from pathlib import Path

# Load results
results_path = Path("/Users/leonardspeiser/Projects/sigmagravity/derivations/k_derivation_sparc_test_results.json")

with open(results_path) as f:
    data = json.load(f)

galaxies = data['galaxies']

print("=" * 70)
print("ANALYSIS OF k DERIVATION RESULTS")
print("=" * 70)

# Extract arrays
k_values = np.array([g['k_optimal'] for g in galaxies])
rms_values = np.array([g['rms'] for g in galaxies])
V_sigma = np.array([g['V_over_sigma'] for g in galaxies])
R_d = np.array([g['R_d'] for g in galaxies])
V_at_Rd = np.array([g['V_at_Rd'] for g in galaxies])
sigma_eff = np.array([g['sigma_eff'] for g in galaxies])
gas_frac = np.array([g['gas_frac'] for g in galaxies])

# Check for boundary hits
at_lower = np.sum(k_values <= 0.051)
at_upper = np.sum(k_values >= 0.799)
print(f"\nBoundary hits:")
print(f"  k at lower bound (0.05): {at_lower} galaxies ({100*at_lower/len(k_values):.1f}%)")
print(f"  k at upper bound (0.80): {at_upper} galaxies ({100*at_upper/len(k_values):.1f}%)")

# Exclude boundary hits for analysis
interior = (k_values > 0.051) & (k_values < 0.799)
print(f"  Interior solutions: {np.sum(interior)} galaxies ({100*np.sum(interior)/len(k_values):.1f}%)")

k_interior = k_values[interior]
print(f"\nInterior k distribution:")
print(f"  Mean = {np.mean(k_interior):.4f}")
print(f"  Median = {np.median(k_interior):.4f}")
print(f"  Std = {np.std(k_interior):.4f}")

# Analyze what drives high k values
print("\n" + "=" * 70)
print("WHAT DRIVES HIGH k VALUES?")
print("=" * 70)

# Split by k
low_k = k_values < 0.24
mid_k = (k_values >= 0.24) & (k_values < 0.5)
high_k = k_values >= 0.5

print(f"\nGalaxy properties by k range:")
print("-" * 60)
print(f"{'Property':>15} | {'k<0.24':>12} | {'0.24≤k<0.5':>12} | {'k≥0.5':>12}")
print("-" * 60)

for name, arr in [('V/σ', V_sigma), ('R_d (kpc)', R_d), ('V(R_d)', V_at_Rd), 
                   ('σ_eff', sigma_eff), ('gas_frac', gas_frac), ('RMS', rms_values)]:
    low_val = np.median(arr[low_k]) if np.sum(low_k) > 0 else 0
    mid_val = np.median(arr[mid_k]) if np.sum(mid_k) > 0 else 0
    high_val = np.median(arr[high_k]) if np.sum(high_k) > 0 else 0
    print(f"{name:>15} | {low_val:>12.2f} | {mid_val:>12.2f} | {high_val:>12.2f}")

print("-" * 60)
print(f"{'N galaxies':>15} | {np.sum(low_k):>12} | {np.sum(mid_k):>12} | {np.sum(high_k):>12}")

# The key question: Does fixed k=0.24 give similar RMS to optimal k?
print("\n" + "=" * 70)
print("COMPARISON: FIXED k=0.24 vs OPTIMAL k")
print("=" * 70)

# We need to compute RMS with fixed k=0.24
# This requires re-running the predictions, which we don't have here
# But we can estimate from the distribution

print(f"""
The per-galaxy optimal k has:
  - Median = {np.median(k_values):.3f}
  - Mean = {np.mean(k_values):.3f}
  - Std = {np.std(k_values):.3f}

The fixed k = 0.24 is at the {100*np.mean(k_values <= 0.24):.1f}th percentile.

This suggests that:
1. Many galaxies prefer higher k than 0.24
2. But k=0.24 may still work well because:
   - The RMS surface is flat near the optimum
   - Other parameters (A, G) compensate
   - The fitting may be overfitting noise
""")

# Check RMS sensitivity
print("\n" + "=" * 70)
print("RMS SENSITIVITY TO k")
print("=" * 70)

# For galaxies with k_optimal near 0.24, what's their RMS?
near_024 = np.abs(k_values - 0.24) < 0.05
print(f"\nGalaxies with k_optimal ≈ 0.24 (±0.05): {np.sum(near_024)}")
if np.sum(near_024) > 0:
    print(f"  Mean RMS = {np.mean(rms_values[near_024]):.2f} km/s")
    print(f"  Median RMS = {np.median(rms_values[near_024]):.2f} km/s")

# For all galaxies, what's the RMS?
print(f"\nAll galaxies at their optimal k:")
print(f"  Mean RMS = {np.mean(rms_values):.2f} km/s")
print(f"  Median RMS = {np.median(rms_values):.2f} km/s")

# Key insight
print("\n" + "=" * 70)
print("KEY INSIGHT")
print("=" * 70)
print("""
The wide range of per-galaxy optimal k (0.05 to 0.80) with many galaxies
preferring k > 0.24 suggests that:

1. k IS NOT UNIVERSAL - it varies with galaxy properties
   
2. OR the σ_eff estimation is wrong:
   - We use mass-weighted component dispersions
   - But actual σ may be different (line-of-sight, inclination effects)
   
3. OR other parameters should vary instead:
   - A(G) might need galaxy-specific G values
   - The h(g) function shape might vary

4. The correlation k vs V/σ is weak (r = 0.16), suggesting
   V/σ is not the main driver of k variation.

NEXT STEPS:
- Test if fixed k=0.24 gives similar total RMS to per-galaxy optimal k
- Check if k correlates with galaxy type or morphology
- Test alternative σ_eff definitions
""")

# Compute correlation matrix
print("\n" + "=" * 70)
print("CORRELATION MATRIX")
print("=" * 70)

variables = {
    'k': k_values,
    'RMS': rms_values,
    'V/σ': V_sigma,
    'R_d': R_d,
    'V(Rd)': V_at_Rd,
    'σ_eff': sigma_eff,
    'f_gas': gas_frac
}

print(f"\n{'':>8}", end='')
for name in variables:
    print(f"{name:>8}", end='')
print()

for name1, arr1 in variables.items():
    print(f"{name1:>8}", end='')
    for name2, arr2 in variables.items():
        corr = np.corrcoef(arr1, arr2)[0, 1]
        print(f"{corr:>8.2f}", end='')
    print()

# Save analysis
analysis = {
    'n_galaxies': len(galaxies),
    'boundary_hits': {
        'lower': int(at_lower),
        'upper': int(at_upper)
    },
    'k_interior': {
        'mean': float(np.mean(k_interior)),
        'median': float(np.median(k_interior)),
        'std': float(np.std(k_interior))
    },
    'k_by_range': {
        'low_k_count': int(np.sum(low_k)),
        'mid_k_count': int(np.sum(mid_k)),
        'high_k_count': int(np.sum(high_k))
    }
}

output_path = Path("/Users/leonardspeiser/Projects/sigmagravity/derivations/k_derivation_analysis.json")
with open(output_path, 'w') as f:
    json.dump(analysis, f, indent=2)
print(f"\nAnalysis saved to: {output_path}")

