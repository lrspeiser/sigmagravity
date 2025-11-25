"""
Spiral Channel Winding Hypothesis
==================================

Physical idea: Gravitational field lines don't stay radial — they get
wound up by differential rotation, like magnetic fields in a dynamo.

Channel pitch angle: θ_pitch = arctan(v_r / v_circ)

For pure circular rotation, v_r ≈ 0, so channels become tightly wound.
But there's a subtlety: TIGHTLY wound channels might INTERFERE with
each other (like over-twisted rope), reducing effective enhancement!

Key prediction:
- Few orbits (outer disk, dwarfs) → loose winding → strong channels
- Many orbits (inner disk, massive spirals) → tight winding → saturation/interference

This could explain why massive spirals need LESS enhancement!
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cooperative_channeling import (
    CooperativeParams, load_sparc_galaxy, estimate_sigma_v
)


def compute_orbital_periods(R: np.ndarray, v_c: np.ndarray, t_age: float = 10.0) -> np.ndarray:
    """
    Compute number of orbital periods at each radius.
    
    N_orbits = t_age / T_orbital = t_age × v_c / (2πR)
    
    Convert: R [kpc], v_c [km/s], t_age [Gyr]
    T_orbital = 2πR / v_c [kpc / (km/s)] = 2πR / v_c × (3.086e16 km/kpc) / (3.156e16 s/Gyr) [Gyr]
                                         ≈ 2πR / v_c × 0.978 [Gyr]
    """
    T_orbital_gyr = 2 * np.pi * R / v_c * 0.978  # Gyr
    N_orbits = t_age / T_orbital_gyr
    return N_orbits


def spiral_winding_factor(N_orbits: np.ndarray, N_crit: float = 30.0, steepness: float = 2.0) -> np.ndarray:
    """
    Model channel interference from over-winding.
    
    f_wind = 1 / (1 + (N_orbits / N_crit)^steepness)
    
    - Few orbits (N << N_crit): f_wind ≈ 1 (no interference)
    - Many orbits (N >> N_crit): f_wind → 0 (channels interfere)
    
    Parameters:
    - N_crit: Critical number of orbits where interference kicks in
    - steepness: How sharply interference grows
    """
    return 1.0 / (1.0 + (N_orbits / N_crit) ** steepness)


def spiral_channeling_enhancement(
    R: np.ndarray, v_c: np.ndarray, Sigma: np.ndarray, sigma_v: np.ndarray,
    chi_0: float = 0.5,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.3,
    epsilon: float = 0.3,
    D_max: float = 3.0,
    t_age: float = 10.0,
    tau_0: float = 1.0,
    N_crit: float = 30.0,      # Critical orbits for interference
    wind_steepness: float = 2.0,
    Sigma_ref: float = 100.0,
    sigma_ref: float = 30.0,
    R_0: float = 8.0,
):
    """
    Channeling enhancement with spiral winding suppression.
    
    F = 1 + χ₀ × (Σ/Σ_ref)^ε × D(R) × f_wind(N_orbits)
    
    The f_wind term suppresses enhancement when channels are over-wound.
    """
    R_safe = np.maximum(R, 0.1)
    sigma_safe = np.maximum(sigma_v, 1.0)
    Sigma_safe = np.maximum(Sigma, 0.01)
    
    # Channel depth (same as before)
    tau_ch = tau_0 * (sigma_safe / sigma_ref) * (R_0 / R_safe)
    tau_ch = np.maximum(tau_ch, 0.01)
    
    time_term = (t_age / tau_ch) ** gamma
    coherence_term = (v_c / sigma_safe) ** beta
    radial_term = (R_safe / R_0) ** alpha
    
    D_raw = time_term * coherence_term * radial_term
    D = D_raw / (1.0 + D_raw / D_max)
    
    # Surface density factor
    density_factor = (Sigma_safe / Sigma_ref) ** epsilon
    density_factor = np.minimum(density_factor, 5.0)
    
    # NEW: Spiral winding suppression
    N_orbits = compute_orbital_periods(R_safe, v_c, t_age)
    f_wind = spiral_winding_factor(N_orbits, N_crit, wind_steepness)
    
    # Full enhancement
    F = 1.0 + chi_0 * density_factor * D * f_wind
    
    return F, {
        'D': D,
        'N_orbits': N_orbits,
        'f_wind': f_wind,
        'density_factor': density_factor,
    }


def test_spiral_winding():
    """Test spiral winding hypothesis on synthetic galaxies."""
    
    print("=" * 70)
    print("SPIRAL WINDING HYPOTHESIS TEST")
    print("=" * 70)
    
    # Compare dwarf (v_flat=60) vs massive spiral (v_flat=220)
    R = np.array([2, 5, 8, 10, 15, 20])  # kpc
    
    # Dwarf galaxy
    v_dwarf = np.array([40, 55, 60, 60, 58, 55])  # km/s
    Sigma_dwarf = np.array([50, 30, 20, 15, 10, 8])  # M_sun/pc^2
    sigma_dwarf = estimate_sigma_v(R, v_dwarf, is_gas_dominated=True)
    
    # Massive spiral
    v_spiral = np.array([180, 210, 220, 220, 215, 210])  # km/s
    Sigma_spiral = np.array([300, 150, 80, 50, 20, 10])  # M_sun/pc^2
    sigma_spiral = estimate_sigma_v(R, v_spiral, is_gas_dominated=False)
    
    print("\n" + "=" * 70)
    print("WITHOUT WINDING SUPPRESSION")
    print("=" * 70)
    
    # Without winding (N_crit very large)
    F_dwarf_no, diag_d_no = spiral_channeling_enhancement(
        R, v_dwarf, Sigma_dwarf, sigma_dwarf, N_crit=1000
    )
    F_spiral_no, diag_s_no = spiral_channeling_enhancement(
        R, v_spiral, Sigma_spiral, sigma_spiral, N_crit=1000
    )
    
    print(f"\nDwarf (v~60 km/s):")
    print(f"  N_orbits: {diag_d_no['N_orbits'].min():.1f} - {diag_d_no['N_orbits'].max():.1f}")
    print(f"  F range: {F_dwarf_no.min():.3f} - {F_dwarf_no.max():.3f}")
    
    print(f"\nMassive spiral (v~220 km/s):")
    print(f"  N_orbits: {diag_s_no['N_orbits'].min():.1f} - {diag_s_no['N_orbits'].max():.1f}")
    print(f"  F range: {F_spiral_no.min():.3f} - {F_spiral_no.max():.3f}")
    
    print("\n" + "=" * 70)
    print("WITH WINDING SUPPRESSION (N_crit=30)")
    print("=" * 70)
    
    # With winding suppression
    F_dwarf, diag_d = spiral_channeling_enhancement(
        R, v_dwarf, Sigma_dwarf, sigma_dwarf, N_crit=30, wind_steepness=2.0
    )
    F_spiral, diag_s = spiral_channeling_enhancement(
        R, v_spiral, Sigma_spiral, sigma_spiral, N_crit=30, wind_steepness=2.0
    )
    
    print(f"\nDwarf (v~60 km/s):")
    print(f"  N_orbits: {diag_d['N_orbits'].min():.1f} - {diag_d['N_orbits'].max():.1f}")
    print(f"  f_wind: {diag_d['f_wind'].min():.3f} - {diag_d['f_wind'].max():.3f}")
    print(f"  F range: {F_dwarf.min():.3f} - {F_dwarf.max():.3f}")
    
    print(f"\nMassive spiral (v~220 km/s):")
    print(f"  N_orbits: {diag_s['N_orbits'].min():.1f} - {diag_s['N_orbits'].max():.1f}")
    print(f"  f_wind: {diag_s['f_wind'].min():.3f} - {diag_s['f_wind'].max():.3f}")
    print(f"  F range: {F_spiral.min():.3f} - {F_spiral.max():.3f}")
    
    # Detailed radial profiles
    print("\n" + "=" * 70)
    print("RADIAL PROFILES")
    print("=" * 70)
    
    print(f"\n{'':>3} {'DWARF':<35} {'MASSIVE SPIRAL':<35}")
    print(f"{'R':>3} {'N_orb':>7} {'f_wind':>7} {'F':>7} {'F_no':>7} | "
          f"{'N_orb':>7} {'f_wind':>7} {'F':>7} {'F_no':>7}")
    print("-" * 80)
    
    for i in range(len(R)):
        print(f"{R[i]:3.0f} "
              f"{diag_d['N_orbits'][i]:7.1f} {diag_d['f_wind'][i]:7.3f} "
              f"{F_dwarf[i]:7.3f} {F_dwarf_no[i]:7.3f} | "
              f"{diag_s['N_orbits'][i]:7.1f} {diag_s['f_wind'][i]:7.3f} "
              f"{F_spiral[i]:7.3f} {F_spiral_no[i]:7.3f}")
    
    # Key metric: how much is F_spiral suppressed vs F_dwarf?
    print("\n" + "=" * 70)
    print("SUPPRESSION RATIO")
    print("=" * 70)
    
    ratio_no_wind = np.mean(F_spiral_no) / np.mean(F_dwarf_no)
    ratio_with_wind = np.mean(F_spiral) / np.mean(F_dwarf)
    
    print(f"\nWithout winding: F_spiral / F_dwarf = {ratio_no_wind:.2f}")
    print(f"With winding: F_spiral / F_dwarf = {ratio_with_wind:.2f}")
    print(f"\nWinding reduces massive spiral enhancement by {(1 - ratio_with_wind/ratio_no_wind)*100:.0f}%")
    
    # Sweep N_crit
    print("\n" + "=" * 70)
    print("N_crit SWEEP (finding optimal interference threshold)")
    print("=" * 70)
    
    print(f"\n{'N_crit':>8} {'F_dwarf':>10} {'F_spiral':>10} {'Ratio':>8}")
    print("-" * 40)
    
    for N_crit in [10, 20, 30, 40, 50, 75, 100, 200]:
        F_d, _ = spiral_channeling_enhancement(
            R, v_dwarf, Sigma_dwarf, sigma_dwarf, N_crit=N_crit
        )
        F_s, _ = spiral_channeling_enhancement(
            R, v_spiral, Sigma_spiral, sigma_spiral, N_crit=N_crit
        )
        ratio = np.mean(F_s) / np.mean(F_d)
        print(f"{N_crit:8.0f} {np.mean(F_d):10.3f} {np.mean(F_s):10.3f} {ratio:8.2f}")
    
    print("\n" + "=" * 70)
    print("PHYSICAL INTERPRETATION")
    print("=" * 70)
    print("""
The spiral winding hypothesis:

1. Field lines in a rotating disk get wound up over time
2. After N_crit orbits, channels become so tightly wound they interfere
3. Massive spirals have MORE orbits → stronger interference → LESS enhancement

This naturally explains why:
- Dwarfs (slow rotation, few orbits) → strong channeling
- Massive spirals (fast rotation, many orbits) → suppressed channeling

The effect is STRONGEST in the inner disk of massive spirals,
where N_orbits is highest — exactly where we over-predicted!
""")


if __name__ == "__main__":
    test_spiral_winding()
