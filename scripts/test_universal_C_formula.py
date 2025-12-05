#!/usr/bin/env python3
"""
Test Universal C Formula on SPARC Galaxies

This script tests the cluster-validated C formula (C = 1 - R_coh/R_outer)
on SPARC galaxies to verify it works universally across all scales.

Key formulas (CONSISTENT with cluster tests):
- R_coh = k × V² / g† where k = 0.65
- g† = cH₀/(4√π) ≈ 1.25×10⁻¹⁰ m/s²
- A = A_geometry × C_realization
- A_geometry = √3 for disk galaxies
- Σ = 1 + A × h(g)
- h(g) = √(g†/g) × g†/(g†+g)
"""

import math
import os
from pathlib import Path

# Physical constants
c = 2.998e8  # m/s
G = 6.674e-11  # m³/kg/s²
M_sun = 1.989e30  # kg
kpc_to_m = 3.086e19
km_to_m = 1000.0

# Cosmology
H0 = 70  # km/s/Mpc
Mpc_to_m = 3.086e22
H0_SI = H0 * 1000 / Mpc_to_m

# Σ-Gravity parameters (CONSISTENT)
e_const = math.e
g_dagger = c * H0_SI / (2 * e_const)  # Critical acceleration
A_geometry = math.sqrt(3)  # Disk geometry (2D)
k_coh = 0.65  # Universal coherence coefficient

# MOND
a0_MOND = 1.2e-10  # m/s²

print("=" * 80)
print("UNIVERSAL C FORMULA TEST ON SPARC GALAXIES")
print("=" * 80)
print(f"\nParameters (CONSISTENT with cluster tests):")
print(f"  g† = cH₀/(4√π) = {g_dagger:.3e} m/s²")
print(f"  k_coh = {k_coh}")
print(f"  A_geometry = √3 = {A_geometry:.4f}")

def R_coherence_kpc(V_kms):
    """Universal coherence radius: R = k × V² / g†"""
    V_ms = V_kms * km_to_m
    return k_coh * V_ms**2 / (g_dagger * kpc_to_m)


def h_universal(g):
    """Universal acceleration function h(g)."""
    g = max(g, 1e-15)
    return math.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)


def C_formula(R_coh, R_outer):
    """
    C = 1 - R_coh/R_outer (cluster-validated formula)

    Behavior:
    - C → 1 when R_coh << R_outer (full coherence zone)
    - C → 0 when R_coh → R_outer (no room for coherence)
    """
    if R_outer <= 0:
        return 0.5
    ratio = R_coh / R_outer
    if ratio >= 1.0:
        return 0.1  # Minimum for galaxies where R_coh > R_outer
    return 1 - ratio


def MOND_mu(g):
    """MOND interpolation function (simple form)."""
    return g / (g + a0_MOND)


def load_rotation_curve(filepath):
    """Load SPARC rotation curve data."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.split()
            if len(parts) >= 6:
                try:
                    r = float(parts[0])  # kpc
                    v_obs = float(parts[1])  # km/s
                    v_err = float(parts[2])  # km/s
                    v_gas = float(parts[3])  # km/s
                    v_disk = float(parts[4])  # km/s
                    v_bul = float(parts[5])  # km/s
                    data.append({
                        'r': r, 'v_obs': v_obs, 'v_err': v_err,
                        'v_gas': v_gas, 'v_disk': v_disk, 'v_bul': v_bul
                    })
                except ValueError:
                    continue
    return data


def analyze_galaxy(galaxy_name, data_dir, master_data):
    """Analyze a single galaxy."""
    # Load rotation curve
    rc_file = data_dir / f"{galaxy_name}_rotmod.dat"
    if not rc_file.exists():
        return None

    rc = load_rotation_curve(rc_file)
    if len(rc) < 5:
        return None

    # Get galaxy properties from master data
    if galaxy_name not in master_data:
        return None

    props = master_data[galaxy_name]
    V_flat = props['Vflat']

    if V_flat <= 0:
        return None

    # Calculate coherence radius
    R_coh = R_coherence_kpc(V_flat)

    # Outer radius from rotation curve
    R_outer = max([p['r'] for p in rc])

    # Calculate C factor
    C = C_formula(R_coh, R_outer)

    # Dynamic amplitude
    A_dynamic = A_geometry * C

    # Analyze each point in rotation curve
    chi2_sigma = 0.0
    chi2_mond = 0.0
    chi2_fixed_A = 0.0
    n_points = 0
    sigma_wins = 0
    mond_wins = 0

    for point in rc:
        r_kpc = point['r']
        v_obs = point['v_obs']
        v_err = max(point['v_err'], 3.0)  # Minimum 3 km/s error

        # Baryonic velocity (disk + gas + bulge)
        v_bar_sq = (point['v_disk']**2 + point['v_gas']**2 +
                    abs(point['v_bul'])**2)
        v_bar = math.sqrt(max(v_bar_sq, 0.01))

        if v_bar < 1.0:
            continue

        # Baryonic acceleration
        r_m = r_kpc * kpc_to_m
        g_bar = (v_bar * km_to_m)**2 / r_m

        # Σ-Gravity prediction with dynamic A
        h_val = h_universal(g_bar)
        Sigma = 1 + A_dynamic * h_val
        g_sigma = Sigma * g_bar
        v_sigma = math.sqrt(g_sigma * r_m) / km_to_m

        # Σ-Gravity with fixed A = √3
        Sigma_fixed = 1 + A_geometry * h_val
        g_fixed = Sigma_fixed * g_bar
        v_fixed = math.sqrt(g_fixed * r_m) / km_to_m

        # MOND prediction
        mu = MOND_mu(g_bar)
        if mu > 0:
            g_mond = g_bar / mu
            v_mond = math.sqrt(g_mond * r_m) / km_to_m
        else:
            v_mond = v_bar

        # Chi-squared contributions
        chi2_sigma += ((v_obs - v_sigma) / v_err)**2
        chi2_mond += ((v_obs - v_mond) / v_err)**2
        chi2_fixed_A += ((v_obs - v_fixed) / v_err)**2
        n_points += 1

        # Count wins
        if abs(v_obs - v_sigma) < abs(v_obs - v_mond):
            sigma_wins += 1
        else:
            mond_wins += 1

    if n_points < 5:
        return None

    return {
        'galaxy': galaxy_name,
        'V_flat': V_flat,
        'R_coh': R_coh,
        'R_outer': R_outer,
        'C': C,
        'A_dynamic': A_dynamic,
        'chi2_sigma': chi2_sigma / n_points,
        'chi2_mond': chi2_mond / n_points,
        'chi2_fixed_A': chi2_fixed_A / n_points,
        'n_points': n_points,
        'sigma_wins': sigma_wins,
        'mond_wins': mond_wins,
        'sigma_beats_mond': chi2_sigma < chi2_mond,
        'dynamic_beats_fixed': chi2_sigma < chi2_fixed_A
    }


def parse_master_table(filepath):
    """Parse SPARC master table."""
    master_data = {}
    with open(filepath, 'r') as f:
        for line in f:
            # Skip header lines
            if line.startswith(('Title', 'Authors', 'Table', '=', '-',
                                'Byte', 'Note', ' ')):
                continue

            # Parse data lines
            parts = line.split()
            if len(parts) >= 16:
                try:
                    galaxy = parts[0]
                    L_36 = float(parts[7])  # 10^9 L_sun
                    R_disk = float(parts[11])  # kpc
                    M_HI = float(parts[13])  # 10^9 M_sun
                    Vflat = float(parts[15])  # km/s

                    master_data[galaxy] = {
                        'L_36': L_36,
                        'R_disk': R_disk,
                        'M_HI': M_HI,
                        'Vflat': Vflat
                    }
                except (ValueError, IndexError):
                    continue
    return master_data


# Main analysis
data_dir = Path(__file__).parent.parent / "data" / "Rotmod_LTG"
master_file = data_dir / "MasterSheet_SPARC.mrt"

# Try alternative paths
if not master_file.exists():
    alt_paths = [
        Path(__file__).parent.parent / "many_path_model" / "paper_release" / "data" / "Rotmod_LTG" / "MasterSheet_SPARC.mrt",
        Path("/home/user/sigmagravity/many_path_model/paper_release/data/Rotmod_LTG/MasterSheet_SPARC.mrt"),
    ]
    for alt in alt_paths:
        if alt.exists():
            master_file = alt
            data_dir = alt.parent
            break

print(f"\nData directory: {data_dir}")

# Parse master table
master_data = parse_master_table(master_file)
print(f"Loaded {len(master_data)} galaxies from master table")

# Find rotation curve files
rc_files = list(data_dir.glob("*_rotmod.dat"))
print(f"Found {len(rc_files)} rotation curve files")

# Analyze all galaxies
results = []
for rc_file in rc_files:
    galaxy_name = rc_file.stem.replace("_rotmod", "")
    result = analyze_galaxy(galaxy_name, data_dir, master_data)
    if result:
        results.append(result)

print(f"\nAnalyzed {len(results)} galaxies successfully")

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

# Categorize by coherence regime
galaxies_Rcoh_gt_Rout = [r for r in results if r['R_coh'] > r['R_outer']]
galaxies_Rcoh_lt_Rout = [r for r in results if r['R_coh'] <= r['R_outer']]

print(f"\nCoherence regimes:")
print(f"  R_coh > R_outer: {len(galaxies_Rcoh_gt_Rout)} galaxies (like clusters, but inverted)")
print(f"  R_coh ≤ R_outer: {len(galaxies_Rcoh_lt_Rout)} galaxies (standard regime)")

# Chi-squared comparison
total_sigma_wins = sum(1 for r in results if r['sigma_beats_mond'])
total_mond_wins = len(results) - total_sigma_wins

print(f"\nΣ-Gravity (dynamic A) vs MOND:")
print(f"  Σ-Gravity wins: {total_sigma_wins}/{len(results)} ({100*total_sigma_wins/len(results):.1f}%)")
print(f"  MOND wins:      {total_mond_wins}/{len(results)} ({100*total_mond_wins/len(results):.1f}%)")

# Dynamic vs fixed A
dynamic_beats_fixed = sum(1 for r in results if r['dynamic_beats_fixed'])
print(f"\nDynamic A vs Fixed A (√3):")
print(f"  Dynamic wins: {dynamic_beats_fixed}/{len(results)} ({100*dynamic_beats_fixed/len(results):.1f}%)")

# Average chi-squared
avg_chi2_sigma = sum(r['chi2_sigma'] for r in results) / len(results)
avg_chi2_mond = sum(r['chi2_mond'] for r in results) / len(results)
avg_chi2_fixed = sum(r['chi2_fixed_A'] for r in results) / len(results)

print(f"\nAverage reduced χ² (lower is better):")
print(f"  Σ-Gravity (dynamic A): {avg_chi2_sigma:.2f}")
print(f"  Σ-Gravity (fixed A):   {avg_chi2_fixed:.2f}")
print(f"  MOND:                  {avg_chi2_mond:.2f}")

# Breakdown by galaxy type
print("\n" + "=" * 80)
print("BREAKDOWN BY VELOCITY CLASS")
print("=" * 80)

velocity_classes = [
    ("Dwarfs (V < 80 km/s)", lambda r: r['V_flat'] < 80),
    ("Low-mass (80-120 km/s)", lambda r: 80 <= r['V_flat'] < 120),
    ("Mid-mass (120-180 km/s)", lambda r: 120 <= r['V_flat'] < 180),
    ("High-mass (V > 180 km/s)", lambda r: r['V_flat'] >= 180),
]

print(f"\n{'Class':<25} {'N':<5} {'Σ wins':<10} {'MOND wins':<10} {'<C>':<8} {'<χ²_Σ>':<8} {'<χ²_M>':<8}")
print("-" * 85)

for name, condition in velocity_classes:
    subset = [r for r in results if condition(r)]
    if len(subset) == 0:
        continue

    sigma_wins = sum(1 for r in subset if r['sigma_beats_mond'])
    mond_wins = len(subset) - sigma_wins
    avg_C = sum(r['C'] for r in subset) / len(subset)
    avg_chi2_s = sum(r['chi2_sigma'] for r in subset) / len(subset)
    avg_chi2_m = sum(r['chi2_mond'] for r in subset) / len(subset)

    print(f"{name:<25} {len(subset):<5} {sigma_wins:<10} {mond_wins:<10} {avg_C:<8.2f} {avg_chi2_s:<8.2f} {avg_chi2_m:<8.2f}")

# Show some individual results
print("\n" + "=" * 80)
print("SAMPLE RESULTS (sorted by V_flat)")
print("=" * 80)

sorted_results = sorted(results, key=lambda x: x['V_flat'])
sample = sorted_results[::max(1, len(sorted_results)//20)]  # ~20 samples

print(f"\n{'Galaxy':<15} {'V_flat':<8} {'R_coh':<8} {'R_out':<8} {'C':<6} {'A_dyn':<6} {'χ²_Σ':<8} {'χ²_M':<8} {'Winner':<8}")
print("-" * 95)

for r in sample:
    winner = "Σ" if r['sigma_beats_mond'] else "MOND"
    print(f"{r['galaxy']:<15} {r['V_flat']:<8.1f} {r['R_coh']:<8.2f} {r['R_outer']:<8.2f} "
          f"{r['C']:<6.2f} {r['A_dynamic']:<6.2f} {r['chi2_sigma']:<8.2f} {r['chi2_mond']:<8.2f} {winner:<8}")

# Final verdict
print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)

if total_sigma_wins > total_mond_wins and avg_chi2_sigma < avg_chi2_mond:
    print(f"""
✓ UNIVERSAL C FORMULA WORKS FOR GALAXIES

The cluster-validated formula C = 1 - R_coh/R_outer also works for galaxies:
- Σ-Gravity wins {total_sigma_wins}/{len(results)} galaxies vs MOND
- Average χ² improved: {avg_chi2_sigma:.2f} vs {avg_chi2_mond:.2f}

The formula is UNIVERSAL across:
- Galaxy clusters (R_coh < R_outer → C close to 1)
- Massive galaxies (R_coh ~ R_outer → C moderate)
- Dwarf galaxies (R_coh > R_outer → C capped at 0.1)
""")
elif avg_chi2_sigma < avg_chi2_fixed:
    print(f"""
○ PARTIAL SUCCESS

Dynamic amplitude improves over fixed A:
- Dynamic A: χ² = {avg_chi2_sigma:.2f}
- Fixed A:   χ² = {avg_chi2_fixed:.2f}

But MOND still competitive in some regimes.
Consider refining C formula for galaxy scales.
""")
else:
    print(f"""
✗ C FORMULA NEEDS ADJUSTMENT FOR GALAXIES

The cluster-validated C = 1 - R_coh/R_outer doesn't work as well for galaxies.
Consider:
- Different C formula for R_coh > R_outer regime
- Scale-dependent k_coh
- Alternative amplitude scaling
""")

# Save detailed results
output_file = Path(__file__).parent.parent / "data" / "universal_C_galaxy_results.csv"
output_file.parent.mkdir(exist_ok=True)

with open(output_file, 'w') as f:
    f.write("galaxy,V_flat,R_coh,R_outer,C,A_dynamic,chi2_sigma,chi2_mond,chi2_fixed,sigma_wins,mond_wins\n")
    for r in results:
        f.write(f"{r['galaxy']},{r['V_flat']},{r['R_coh']:.3f},{r['R_outer']:.3f},"
                f"{r['C']:.4f},{r['A_dynamic']:.4f},{r['chi2_sigma']:.4f},"
                f"{r['chi2_mond']:.4f},{r['chi2_fixed_A']:.4f},"
                f"{r['sigma_wins']},{r['mond_wins']}\n")

print(f"\nDetailed results saved to: {output_file}")
