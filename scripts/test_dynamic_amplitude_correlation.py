#!/usr/bin/env python3
"""
Dynamic Amplitude Validation Test

This script tests whether C = 1 - R_coh/R_outer correlates with
model residuals, which would indicate dynamic amplitude could improve fits.

Hypothesis: If dynamic A would help, galaxies where C differs most from 1
(i.e., where A_dynamic differs from fixed √3) should show systematic residuals.
"""

import math
from pathlib import Path

# Physical constants
c = 2.998e8  # m/s
kpc_to_m = 3.086e19
km_to_m = 1000.0
H0 = 70  # km/s/Mpc
Mpc_to_m = 3.086e22
H0_SI = H0 * 1000 / Mpc_to_m
e_const = math.e
g_dagger = c * H0_SI / (2 * e_const)
k_coh = 0.65
A_geometry = math.sqrt(3)

def R_coh_kpc(V_kms):
    """R_coh = k × V² / g†"""
    V_ms = V_kms * km_to_m
    return k_coh * V_ms**2 / (g_dagger * kpc_to_m)

def C_formula(R_coh, R_outer):
    """C = 1 - R_coh/R_outer"""
    if R_outer <= 0:
        return 0.5
    ratio = R_coh / R_outer
    if ratio >= 1.0:
        return 0.1
    return 1 - ratio

# Load SPARC metadata
meta_file = Path("/home/user/sigmagravity/pca/data/raw/metadata/sparc_meta.csv")
meta = {}
with open(meta_file) as f:
    header = f.readline().strip().split(',')
    for line in f:
        parts = line.strip().split(',')
        if len(parts) >= 15:
            try:
                name = parts[0]
                Rd = float(parts[10])  # Disk scale length
                RHI = float(parts[13])  # HI radius
                Vf = float(parts[14])   # Flat velocity
                if Vf > 0 and Rd > 0:
                    R_outer = RHI if RHI > 0 else 4 * Rd
                    meta[name] = {'Vf': Vf, 'Rd': Rd, 'RHI': RHI, 'R_outer': R_outer}
            except (ValueError, IndexError):
                continue

print(f"Loaded {len(meta)} galaxies from metadata")

# Load model results
results_file = Path("/home/user/sigmagravity/time-coherence/unified_kernel_sparc_results.csv")
results = {}
with open(results_file) as f:
    header = f.readline().strip().split(',')
    for line in f:
        parts = line.strip().split(',')
        if len(parts) >= 4:
            try:
                name = parts[0]
                rms_gr = float(parts[1])
                rms_model = float(parts[2])
                delta_rms = float(parts[3])
                results[name] = {
                    'rms_gr': rms_gr,
                    'rms_model': rms_model,
                    'delta_rms': delta_rms,
                    'improvement': rms_gr - rms_model
                }
            except (ValueError, IndexError):
                continue

print(f"Loaded {len(results)} galaxies from results")

# Combine and analyze
combined = []
for name in meta:
    if name in results:
        m = meta[name]
        r = results[name]

        R_coh = R_coh_kpc(m['Vf'])
        C = C_formula(R_coh, m['R_outer'])
        A_dynamic = A_geometry * C
        A_fixed = A_geometry

        combined.append({
            'name': name,
            'Vf': m['Vf'],
            'R_coh': R_coh,
            'R_outer': m['R_outer'],
            'ratio': R_coh / m['R_outer'],
            'C': C,
            'A_dynamic': A_dynamic,
            'A_diff': A_fixed - A_dynamic,  # How much we'd reduce A
            'rms_model': r['rms_model'],
            'improvement': r['improvement'],
            'delta_rms': r['delta_rms']
        })

print(f"\nCombined {len(combined)} galaxies")

# Sort by C (low to high) - galaxies where dynamic A would differ most
combined.sort(key=lambda x: x['C'])

print("\n" + "=" * 90)
print("GALAXIES WHERE DYNAMIC A DIFFERS MOST FROM FIXED (low C)")
print("=" * 90)
print(f"\n{'Galaxy':<15} {'V_flat':<8} {'R_coh':<8} {'R_out':<8} {'C':<6} {'A_dyn':<6} {'RMS':<8} {'Δ_rms':<8}")
print("-" * 85)

# Show bottom 15 (where C is lowest, so dynamic A would differ most)
for g in combined[:15]:
    print(f"{g['name']:<15} {g['Vf']:<8.1f} {g['R_coh']:<8.1f} {g['R_outer']:<8.1f} "
          f"{g['C']:<6.2f} {g['A_dynamic']:<6.2f} {g['rms_model']:<8.1f} {g['delta_rms']:<8.1f}")

print("\n" + "=" * 90)
print("GALAXIES WHERE DYNAMIC A ≈ FIXED (high C)")
print("=" * 90)
print(f"\n{'Galaxy':<15} {'V_flat':<8} {'R_coh':<8} {'R_out':<8} {'C':<6} {'A_dyn':<6} {'RMS':<8} {'Δ_rms':<8}")
print("-" * 85)

# Show top 15 (where C is highest, so dynamic A ≈ fixed A)
for g in combined[-15:]:
    print(f"{g['name']:<15} {g['Vf']:<8.1f} {g['R_coh']:<8.1f} {g['R_outer']:<8.1f} "
          f"{g['C']:<6.2f} {g['A_dynamic']:<6.2f} {g['rms_model']:<8.1f} {g['delta_rms']:<8.1f}")

# Correlation analysis
print("\n" + "=" * 90)
print("CORRELATION ANALYSIS")
print("=" * 90)

# Calculate correlation between C and model RMS
n = len(combined)
sum_C = sum(g['C'] for g in combined)
sum_rms = sum(g['rms_model'] for g in combined)
sum_C2 = sum(g['C']**2 for g in combined)
sum_rms2 = sum(g['rms_model']**2 for g in combined)
sum_C_rms = sum(g['C'] * g['rms_model'] for g in combined)

# Pearson correlation
mean_C = sum_C / n
mean_rms = sum_rms / n
cov = sum_C_rms / n - mean_C * mean_rms
std_C = math.sqrt(sum_C2 / n - mean_C**2)
std_rms = math.sqrt(sum_rms2 / n - mean_rms**2)
corr_C_rms = cov / (std_C * std_rms) if std_C * std_rms > 0 else 0

print(f"\nCorrelation between C and RMS: r = {corr_C_rms:.3f}")

# Correlation between C and improvement (delta_rms)
sum_delta = sum(g['delta_rms'] for g in combined)
sum_delta2 = sum(g['delta_rms']**2 for g in combined)
sum_C_delta = sum(g['C'] * g['delta_rms'] for g in combined)

mean_delta = sum_delta / n
cov_delta = sum_C_delta / n - mean_C * mean_delta
std_delta = math.sqrt(sum_delta2 / n - mean_delta**2)
corr_C_delta = cov_delta / (std_C * std_delta) if std_C * std_delta > 0 else 0

print(f"Correlation between C and Δ_rms (improvement): r = {corr_C_delta:.3f}")

# Split into low-C and high-C groups
low_C = [g for g in combined if g['C'] < 0.7]
high_C = [g for g in combined if g['C'] >= 0.9]

avg_rms_low = sum(g['rms_model'] for g in low_C) / len(low_C) if low_C else 0
avg_rms_high = sum(g['rms_model'] for g in high_C) / len(high_C) if high_C else 0

avg_delta_low = sum(g['delta_rms'] for g in low_C) / len(low_C) if low_C else 0
avg_delta_high = sum(g['delta_rms'] for g in high_C) / len(high_C) if high_C else 0

print(f"\nLow C group (C < 0.7): n={len(low_C)}")
print(f"  Average RMS: {avg_rms_low:.1f} km/s")
print(f"  Average Δ_rms: {avg_delta_low:.1f} km/s")

print(f"\nHigh C group (C ≥ 0.9): n={len(high_C)}")
print(f"  Average RMS: {avg_rms_high:.1f} km/s")
print(f"  Average Δ_rms: {avg_delta_high:.1f} km/s")

# Interpretation
print("\n" + "=" * 90)
print("INTERPRETATION")
print("=" * 90)

if corr_C_rms < -0.2:
    print("""
NEGATIVE correlation (r < -0.2) between C and RMS suggests:
→ Low-C galaxies (where dynamic A would reduce amplitude) have HIGHER residuals
→ Dynamic amplitude might HELP by reducing enhancement for these galaxies
→ RECOMMEND: Proceed with full rotation curve testing
""")
elif corr_C_rms > 0.2:
    print("""
POSITIVE correlation (r > 0.2) between C and RMS suggests:
→ Low-C galaxies actually have LOWER residuals
→ Fixed A = √3 may already be working well
→ Dynamic amplitude unlikely to help much
""")
else:
    print(f"""
WEAK correlation (|r| < 0.2, actual r = {corr_C_rms:.3f}) between C and RMS suggests:
→ No strong systematic pattern
→ Dynamic amplitude may or may not help
→ Need full rotation curve testing to determine impact
""")

if abs(avg_rms_low - avg_rms_high) > 10:
    direction = "higher" if avg_rms_low > avg_rms_high else "lower"
    print(f"""
Low-C galaxies have {direction} average RMS ({avg_rms_low:.1f} vs {avg_rms_high:.1f} km/s)
This is a {abs(avg_rms_low - avg_rms_high):.1f} km/s difference.
""")
