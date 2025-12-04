#!/usr/bin/env python3
"""
Dynamic Amplitude Test: C = 1 - R_coh/R_outer
=============================================

This script tests whether using dynamic amplitude A = A_geometry × C
improves rotation curve predictions compared to fixed A = √3.

The hypothesis: Galaxies where R_coh approaches R_outer should have
reduced enhancement (lower effective A), which the C formula captures.

Author: Sigma Gravity Validation
Date: December 2025
"""

import math
from pathlib import Path
from typing import List, Tuple, Dict

# ==============================================================================
# PHYSICAL CONSTANTS
# ==============================================================================
c = 2.998e8              # Speed of light [m/s]
kpc_to_m = 3.086e19      # kpc to meters
km_to_m = 1000.0         # km to meters
Mpc_to_m = 3.086e22      # Mpc to meters

H0 = 70                  # Hubble constant [km/s/Mpc]
H0_SI = H0 * 1000 / Mpc_to_m
g_dagger = c * H0_SI / (2 * math.e)  # Critical acceleration
k_coh = 0.65             # Coherence coefficient

A_geometry = math.sqrt(3)  # Base amplitude for disk geometry

print("=" * 80)
print("DYNAMIC AMPLITUDE TEST: C = 1 - R_coh/R_outer")
print("=" * 80)
print(f"\nFormula: A_dynamic = √3 × C = √3 × (1 - R_coh/R_outer)")
print(f"Where: R_coh = {k_coh} × V²/g†")


# ==============================================================================
# CORE FUNCTIONS
# ==============================================================================

def R_coh_kpc(V_kms: float) -> float:
    """R_coh = k × V² / g†"""
    V_ms = V_kms * km_to_m
    return k_coh * V_ms**2 / (g_dagger * kpc_to_m)


def C_factor(R_coh: float, R_outer: float) -> float:
    """C = 1 - R_coh/R_outer, clamped to [0.1, 1.0]"""
    if R_outer <= 0:
        return 0.5
    ratio = R_coh / R_outer
    C = 1 - ratio
    return max(0.1, min(1.0, C))


def h_function(g: float) -> float:
    """h(g) = √(g†/g) × g†/(g† + g)"""
    if g <= 0:
        return 0.0
    return math.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)


def predict_v_fixed(V_bar: float, r_kpc: float) -> float:
    """Predict V_obs with fixed A = √3"""
    if V_bar <= 0 or r_kpc <= 0:
        return 0.0
    r_m = r_kpc * kpc_to_m
    V_bar_ms = V_bar * km_to_m
    g_bar = V_bar_ms**2 / r_m
    h = h_function(g_bar)
    V_obs_ms = V_bar_ms * math.sqrt(1 + A_geometry * h)
    return V_obs_ms / km_to_m


def predict_v_dynamic(V_bar: float, r_kpc: float, C: float) -> float:
    """Predict V_obs with dynamic A = √3 × C"""
    if V_bar <= 0 or r_kpc <= 0:
        return 0.0
    r_m = r_kpc * kpc_to_m
    V_bar_ms = V_bar * km_to_m
    g_bar = V_bar_ms**2 / r_m
    h = h_function(g_bar)
    A_dynamic = A_geometry * C
    V_obs_ms = V_bar_ms * math.sqrt(1 + A_dynamic * h)
    return V_obs_ms / km_to_m


def load_sparc_metadata() -> Dict[str, Dict]:
    """Load SPARC galaxy metadata."""
    meta_file = Path("/home/user/sigmagravity/pca/data/raw/metadata/sparc_meta.csv")
    meta = {}
    with open(meta_file) as f:
        header = f.readline().strip().split(',')
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 15:
                try:
                    name = parts[0]
                    Rd = float(parts[10])
                    RHI = float(parts[13])
                    Vf = float(parts[14])
                    if Vf > 0 and Rd > 0:
                        R_outer = RHI if RHI > 0 else 4 * Rd
                        meta[name] = {'Vf': Vf, 'Rd': Rd, 'RHI': RHI, 'R_outer': R_outer}
                except (ValueError, IndexError):
                    continue
    return meta


def load_rotation_curve(galaxy_name: str) -> List[Tuple[float, float, float]]:
    """Load rotation curve: (r, V_obs, V_bar)"""
    data_dir = Path("/home/user/sigmagravity/many_path_model/paper_release/data/Rotmod_LTG")
    rc_file = data_dir / f"{galaxy_name}_rotmod.dat"

    if not rc_file.exists():
        return []

    data = []
    with open(rc_file) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 6:
                try:
                    r = float(parts[0])
                    V_obs = float(parts[1])
                    V_gas = float(parts[3])
                    V_disk = float(parts[4])
                    V_bul = float(parts[5])
                    V_bar = math.sqrt(V_gas**2 + V_disk**2 + V_bul**2)
                    if r > 0:
                        data.append((r, V_obs, V_bar))
                except (ValueError, IndexError):
                    continue
    return data


# ==============================================================================
# MAIN TEST
# ==============================================================================

def main():
    meta = load_sparc_metadata()
    print(f"\nLoaded {len(meta)} galaxies")

    results = []

    for name, m in meta.items():
        rc_data = load_rotation_curve(name)
        if not rc_data:
            continue

        # Calculate R_coh and C
        R_coh = R_coh_kpc(m['Vf'])
        C = C_factor(R_coh, m['R_outer'])

        # Calculate RMS for different models
        rms_bar = 0      # Baryonic only (no enhancement)
        rms_fixed = 0    # Fixed A = √3
        rms_dynamic = 0  # Dynamic A = √3 × C
        n = 0

        for r, V_obs, V_bar in rc_data:
            if V_obs > 0 and V_bar > 0 and r > 0:
                V_fixed = predict_v_fixed(V_bar, r)
                V_dyn = predict_v_dynamic(V_bar, r, C)

                rms_bar += (V_bar - V_obs)**2
                rms_fixed += (V_fixed - V_obs)**2
                rms_dynamic += (V_dyn - V_obs)**2
                n += 1

        if n > 0:
            rms_bar = math.sqrt(rms_bar / n)
            rms_fixed = math.sqrt(rms_fixed / n)
            rms_dynamic = math.sqrt(rms_dynamic / n)

            results.append({
                'name': name,
                'Vf': m['Vf'],
                'R_coh': R_coh,
                'R_outer': m['R_outer'],
                'ratio': R_coh / m['R_outer'],
                'C': C,
                'A_dynamic': A_geometry * C,
                'rms_bar': rms_bar,
                'rms_fixed': rms_fixed,
                'rms_dynamic': rms_dynamic,
                'n_points': n
            })

    print(f"Tested {len(results)} galaxies with rotation curves\n")

    # ===========================================================================
    # SUMMARY STATISTICS
    # ===========================================================================

    # Calculate mean RMS for each model
    mean_rms_bar = sum(r['rms_bar'] for r in results) / len(results)
    mean_rms_fixed = sum(r['rms_fixed'] for r in results) / len(results)
    mean_rms_dynamic = sum(r['rms_dynamic'] for r in results) / len(results)

    print("=" * 80)
    print("OVERALL COMPARISON")
    print("=" * 80)
    print(f"\n{'Model':<25} {'Mean RMS (km/s)':<20}")
    print("-" * 45)
    print(f"{'Baryonic only':<25} {mean_rms_bar:<20.1f}")
    print(f"{'Fixed A = √3':<25} {mean_rms_fixed:<20.1f}")
    print(f"{'Dynamic A = √3×C':<25} {mean_rms_dynamic:<20.1f}")

    # Count which model is best for each galaxy
    n_bar_best = sum(1 for r in results if r['rms_bar'] <= r['rms_fixed'] and r['rms_bar'] <= r['rms_dynamic'])
    n_fixed_best = sum(1 for r in results if r['rms_fixed'] < r['rms_bar'] and r['rms_fixed'] <= r['rms_dynamic'])
    n_dynamic_best = sum(1 for r in results if r['rms_dynamic'] < r['rms_bar'] and r['rms_dynamic'] < r['rms_fixed'])

    print(f"\n{'Model':<25} {'Best for N galaxies':<20}")
    print("-" * 45)
    print(f"{'Baryonic only':<25} {n_bar_best:<20}")
    print(f"{'Fixed A = √3':<25} {n_fixed_best:<20}")
    print(f"{'Dynamic A = √3×C':<25} {n_dynamic_best:<20}")

    # ===========================================================================
    # BREAKDOWN BY R_coh/R_outer RATIO
    # ===========================================================================

    print("\n" + "=" * 80)
    print("BREAKDOWN BY COHERENCE RATIO (R_coh/R_outer)")
    print("=" * 80)

    bins = [
        ("Low ratio (< 0.15)", lambda r: r['ratio'] < 0.15),
        ("Medium ratio (0.15-0.35)", lambda r: 0.15 <= r['ratio'] < 0.35),
        ("High ratio (≥ 0.35)", lambda r: r['ratio'] >= 0.35),
    ]

    for bin_name, condition in bins:
        subset = [r for r in results if condition(r)]
        if not subset:
            continue

        mean_bar = sum(r['rms_bar'] for r in subset) / len(subset)
        mean_fixed = sum(r['rms_fixed'] for r in subset) / len(subset)
        mean_dynamic = sum(r['rms_dynamic'] for r in subset) / len(subset)

        n_fixed_wins = sum(1 for r in subset if r['rms_fixed'] < r['rms_dynamic'])
        n_dynamic_wins = sum(1 for r in subset if r['rms_dynamic'] < r['rms_fixed'])

        print(f"\n{bin_name}: N = {len(subset)}")
        print(f"  Mean C = {sum(r['C'] for r in subset)/len(subset):.2f}")
        print(f"  Mean RMS - Baryonic: {mean_bar:.1f} km/s")
        print(f"  Mean RMS - Fixed:    {mean_fixed:.1f} km/s")
        print(f"  Mean RMS - Dynamic:  {mean_dynamic:.1f} km/s")
        print(f"  Fixed wins: {n_fixed_wins}, Dynamic wins: {n_dynamic_wins}")

    # ===========================================================================
    # TOP EXAMPLES WHERE DYNAMIC HELPS
    # ===========================================================================

    print("\n" + "=" * 80)
    print("GALAXIES WHERE DYNAMIC AMPLITUDE HELPS MOST")
    print("=" * 80)

    # Sort by improvement from fixed to dynamic
    improvement = [(r, r['rms_fixed'] - r['rms_dynamic']) for r in results]
    improvement.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'Galaxy':<15} {'Vf':<8} {'R_coh':<8} {'Ratio':<8} {'C':<6} {'RMS_fix':<10} {'RMS_dyn':<10} {'Δ':<8}")
    print("-" * 90)

    for r, delta in improvement[:10]:
        print(f"{r['name']:<15} {r['Vf']:<8.0f} {r['R_coh']:<8.1f} {r['ratio']:<8.2f} "
              f"{r['C']:<6.2f} {r['rms_fixed']:<10.1f} {r['rms_dynamic']:<10.1f} {delta:<8.1f}")

    # ===========================================================================
    # GALAXIES WHERE DYNAMIC HURTS MOST
    # ===========================================================================

    print("\n" + "=" * 80)
    print("GALAXIES WHERE DYNAMIC AMPLITUDE HURTS MOST")
    print("=" * 80)

    print(f"\n{'Galaxy':<15} {'Vf':<8} {'R_coh':<8} {'Ratio':<8} {'C':<6} {'RMS_fix':<10} {'RMS_dyn':<10} {'Δ':<8}")
    print("-" * 90)

    for r, delta in improvement[-10:]:
        print(f"{r['name']:<15} {r['Vf']:<8.0f} {r['R_coh']:<8.1f} {r['ratio']:<8.2f} "
              f"{r['C']:<6.2f} {r['rms_fixed']:<10.1f} {r['rms_dynamic']:<10.1f} {delta:<8.1f}")

    # ===========================================================================
    # STATISTICAL SIGNIFICANCE
    # ===========================================================================

    print("\n" + "=" * 80)
    print("STATISTICAL SUMMARY")
    print("=" * 80)

    # Paired comparison: dynamic vs fixed
    deltas = [r['rms_fixed'] - r['rms_dynamic'] for r in results]
    mean_delta = sum(deltas) / len(deltas)
    std_delta = math.sqrt(sum((d - mean_delta)**2 for d in deltas) / len(deltas))

    n_dynamic_better = sum(1 for d in deltas if d > 0)
    n_fixed_better = sum(1 for d in deltas if d < 0)
    n_tie = sum(1 for d in deltas if d == 0)

    print(f"\nDynamic vs Fixed comparison:")
    print(f"  Mean improvement (Δ = RMS_fixed - RMS_dynamic): {mean_delta:.2f} km/s")
    print(f"  Std of improvement: {std_delta:.2f} km/s")
    print(f"  Dynamic better: {n_dynamic_better} galaxies ({100*n_dynamic_better/len(results):.1f}%)")
    print(f"  Fixed better: {n_fixed_better} galaxies ({100*n_fixed_better/len(results):.1f}%)")

    # Check if improvement correlates with ratio
    ratios = [r['ratio'] for r in results]
    mean_ratio = sum(ratios) / len(ratios)

    cov = sum((d - mean_delta) * (r - mean_ratio) for d, r in zip(deltas, ratios)) / len(deltas)
    std_ratio = math.sqrt(sum((r - mean_ratio)**2 for r in ratios) / len(ratios))
    correlation = cov / (std_delta * std_ratio) if std_delta * std_ratio > 0 else 0

    print(f"\n  Correlation between improvement and R_coh/R_outer: r = {correlation:.3f}")

    if correlation > 0.2:
        print("  → POSITIVE: Dynamic amplitude helps more for high-ratio galaxies")
    elif correlation < -0.2:
        print("  → NEGATIVE: Dynamic amplitude helps more for low-ratio galaxies")
    else:
        print("  → WEAK: No clear pattern with coherence ratio")

    # ===========================================================================
    # FINAL VERDICT
    # ===========================================================================

    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    if mean_delta > 2 and n_dynamic_better > n_fixed_better:
        print(f"""
DYNAMIC AMPLITUDE IMPROVES FITS

Mean improvement: {mean_delta:.1f} km/s
Dynamic better for {n_dynamic_better}/{len(results)} = {100*n_dynamic_better/len(results):.0f}% of galaxies

The formula A = √3 × (1 - R_coh/R_outer) provides better rotation curve
predictions than fixed A = √3, supporting the coherence scale framework.
""")
    elif mean_delta < -2 and n_fixed_better > n_dynamic_better:
        print(f"""
FIXED AMPLITUDE IS BETTER

Mean change: {mean_delta:.1f} km/s (negative = fixed is better)
Fixed better for {n_fixed_better}/{len(results)} = {100*n_fixed_better/len(results):.0f}% of galaxies

The dynamic amplitude formula does not improve fits. The fixed A = √3
approach remains preferable.
""")
    else:
        print(f"""
MIXED RESULTS - NO CLEAR WINNER

Mean improvement: {mean_delta:.1f} km/s
Dynamic better: {n_dynamic_better} galaxies
Fixed better: {n_fixed_better} galaxies

Neither approach is clearly superior across the full sample. The dynamic
amplitude helps for some galaxies but hurts for others.
""")

    # Save detailed results
    output_file = Path("/home/user/sigmagravity/data/dynamic_amplitude_results.csv")
    with open(output_file, 'w') as f:
        f.write("galaxy,Vf,R_coh,R_outer,ratio,C,A_dyn,rms_bar,rms_fixed,rms_dynamic,delta\n")
        for r in results:
            delta = r['rms_fixed'] - r['rms_dynamic']
            f.write(f"{r['name']},{r['Vf']:.1f},{r['R_coh']:.2f},{r['R_outer']:.2f},"
                    f"{r['ratio']:.3f},{r['C']:.3f},{r['A_dynamic']:.3f},"
                    f"{r['rms_bar']:.2f},{r['rms_fixed']:.2f},{r['rms_dynamic']:.2f},{delta:.2f}\n")

    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
