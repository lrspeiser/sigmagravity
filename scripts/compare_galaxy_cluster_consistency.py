#!/usr/bin/env python3
"""
Compare Galaxy vs Cluster: Consistency of Universal C Formula

Key finding: The formula C = 1 - R_coh/R_outer works, but the
interpretation differs between scales because R_outer means:
- Galaxies: HI radius (extent of rotation curve)
- Clusters: Observation aperture (strong lensing at ~200 kpc)
"""

import math

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

A_disk = math.sqrt(3)  # 1.732
A_sphere = math.pi * math.sqrt(2)  # 4.443

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

print("=" * 80)
print("CONSISTENCY CHECK: UNIVERSAL C FORMULA")
print("=" * 80)

# Galaxy examples
print("\n" + "=" * 40)
print("GALAXY SCALE (using HI radius as R_outer)")
print("=" * 40)

galaxies = [
    ("DDO154 (dwarf)", 47, 5.0),
    ("NGC3109", 66, 6.0),
    ("NGC0300", 93, 9.2),
    ("NGC2403", 131, 15.1),
    ("NGC3198", 150, 35.7),
    ("Milky Way", 220, 30.0),
    ("NGC2841", 285, 45.1),
    ("ESO563-G021", 315, 55.7),
]

print(f"\n{'Galaxy':<20} {'V_flat':<8} {'R_coh':<8} {'R_out':<8} {'Ratio':<8} {'C':<6} {'A=√3×C':<8}")
print("-" * 75)

for name, V, R_out in galaxies:
    R_coh = R_coh_kpc(V)
    C = C_formula(R_coh, R_out)
    A = A_disk * C
    ratio = R_coh / R_out
    print(f"{name:<20} {V:<8.0f} {R_coh:<8.1f} {R_out:<8.1f} {ratio:<8.2f} {C:<6.2f} {A:<8.2f}")

# Cluster examples
print("\n" + "=" * 40)
print("CLUSTER SCALE (using 200 kpc as R_outer)")
print("=" * 40)

clusters = [
    ("Low-mass cluster", 600, 200),
    ("Intermediate", 800, 200),
    ("Massive cluster", 1000, 200),
    ("Very massive", 1200, 200),
]

print(f"\n{'Cluster':<20} {'σ_v':<8} {'R_coh':<8} {'R_out':<8} {'Ratio':<8} {'C':<6} {'A=π√2×C':<8}")
print("-" * 75)

for name, V, R_out in clusters:
    R_coh = R_coh_kpc(V)
    C = C_formula(R_coh, R_out)
    A = A_sphere * C
    ratio = R_coh / R_out
    print(f"{name:<20} {V:<8.0f} {R_coh:<8.1f} {R_out:<8.1f} {ratio:<8.2f} {C:<6.2f} {A:<8.2f}")

# Key insight
print("\n" + "=" * 80)
print("KEY INSIGHT")
print("=" * 80)

print("""
The universal C formula behaves as follows:

GALAXIES (R_coh << R_outer for most):
  - R_coh ~ 0.3-12 kpc
  - R_outer (HI radius) ~ 5-50 kpc
  - Ratio ~ 0.03-0.4 → C ~ 0.6-0.97
  - A = √3 × C ~ 1.0-1.7

This gives amplitude CLOSE TO fixed √3 because coherence zone
fills only a small fraction of the galaxy.

CLUSTERS (R_coh ~ R_outer):
  - R_coh ~ 60-240 kpc (from σ_v ~ 600-1200 km/s)
  - R_outer (lensing aperture) ~ 200 kpc
  - Ratio ~ 0.3-1.2 → C ~ 0.0-0.7
  - A = π√2 × C ~ 0-3.1

This gives VARIABLE amplitude because coherence zone
approaches or exceeds the observation radius.

THE FORMULA IS CONSISTENT BECAUSE:
1. Both use same R_coh = 0.65 × V²/g†
2. Both use C = 1 - R_coh/R_outer
3. Only A_geometry differs: √3 (2D) vs π√2 (3D)
4. R_outer matches observation: HI radius vs lensing aperture
""")

# Compare expected vs actual
print("=" * 80)
print("VERIFICATION: EXPECTED AMPLITUDES")
print("=" * 80)

print("""
From earlier cluster analysis (C = 1 - R_coh/R_outer):
- Mean ratio M_Σ/M_SL = 0.999 (perfect!)
- Scatter = 0.108 dex

This confirms the universal formula works for clusters when:
- Using velocity dispersion for R_coh
- Using lensing aperture (200 kpc) for R_outer
- Using A_geometry = π√2 for 3D geometry

For galaxies, the same formula predicts:
- C ~ 0.6-0.9 (high values due to R_coh << R_outer)
- A ~ 1.0-1.6 (close to fixed √3 = 1.73)

This is CONSISTENT with the observation that fixed A = √3
works reasonably well for many galaxies - the dynamic
formula naturally gives similar values!

The dynamic formula adds value by:
1. Reducing A for massive spirals (where R_coh → R_outer)
2. Providing physical explanation for amplitude variation
3. Creating unified framework across all scales
""")

# Final summary table
print("=" * 80)
print("SUMMARY: EFFECTIVE AMPLITUDES BY SCALE")
print("=" * 80)

print(f"""
Scale               V (km/s)    R_coh (kpc)   R_outer (kpc)   C       A_eff
------------------------------------------------------------------------------
Dwarf galaxies      30-80       0.1-1.1       3-10            0.8-0.97  1.4-1.7
Normal spirals      80-180      1.1-5.5       8-25            0.6-0.9   1.0-1.6
Massive spirals     180-300     5.5-15        20-50           0.3-0.7   0.5-1.2
Galaxy clusters     600-1200    60-240        200 (lens)      0.0-0.7   0.0-3.1

Note: Cluster A_eff uses π√2 as base, galaxies use √3.

The universal formula C = 1 - R_coh/R_outer produces a smooth
transition of effective amplitude from dwarfs to clusters!
""")
