"""
Spiral Winding Gate for Σ-Gravity
==================================

Adds morphology-dependent suppression based on differential rotation.

Physical motivation: After ~10 orbits, coherent paths wind so tightly
that adjacent paths interfere destructively.

This is a G_j gate that can be multiplied into the existing kernel:
K(R) = A × C(R) × G_bulge × G_bar × G_winding

Author: Leonard Speiser
Date: 2025-11-25
"""

import numpy as np


def compute_winding_gate(
    R: np.ndarray,
    v_c: np.ndarray,
    t_age: float = 10.0,
    N_crit: float = 10.0,
    wind_power: float = 2.0,
) -> np.ndarray:
    """
    Compute spiral winding suppression gate.
    
    Parameters
    ----------
    R : array
        Galactocentric radius [kpc]
    v_c : array
        Circular velocity [km/s]
    t_age : float
        System age [Gyr]
    N_crit : float
        Critical winding number (default: 10)
    wind_power : float
        Steepness of suppression (default: 2)
    
    Returns
    -------
    G_wind : array
        Winding gate (1 = no suppression, 0 = full suppression)
    
    Notes
    -----
    The number of orbits is:
        N = t_age × v_c / (2π R × 0.978)
    
    where 0.978 Gyr converts from kpc·km/s to Gyr.
    
    Physical interpretation:
    - N << N_crit: Loosely wound → no suppression (G_wind ≈ 1)
    - N >> N_crit: Tightly wound → strong suppression (G_wind ≈ 0)
    
    Critical winding N_crit ~ v_c/σ_v ~ 10 from coherence geometry.
    
    Examples
    --------
    Dwarf galaxy:
        R = 10 kpc, v_c = 60 km/s, t = 10 Gyr
        N = 10 × 60 / (2π × 10 × 0.978) ≈ 10
        G_wind = 1/(1 + 1²) = 0.5
    
    Massive spiral:
        R = 15 kpc, v_c = 220 km/s, t = 10 Gyr
        N = 10 × 220 / (2π × 15 × 0.978) ≈ 24
        G_wind = 1/(1 + 2.4²) ≈ 0.15
    
    This naturally gives MORE suppression to fast rotators!
    """
    # Protect against edge cases
    R_safe = np.maximum(R, 0.1)
    v_safe = np.maximum(v_c, 1.0)
    
    # Number of orbits over system lifetime
    # Conversion: 1 orbit = 2πR/v_c [kpc·s/km]
    #           = 2πR/v_c × (3.086e16 km/kpc) × (1 Gyr/3.154e16 s)
    #           ≈ 2πR/v_c × 0.978 Gyr
    N_orbits = t_age * v_safe / (2.0 * np.pi * R_safe * 0.978)
    
    # Winding suppression
    G_wind = 1.0 / (1.0 + (N_orbits / N_crit) ** wind_power)
    
    return G_wind


def compute_N_orbits(R: np.ndarray, v_c: np.ndarray, t_age: float = 10.0) -> np.ndarray:
    """Compute number of orbits for diagnostics."""
    R_safe = np.maximum(R, 0.1)
    v_safe = np.maximum(v_c, 1.0)
    return t_age * v_safe / (2.0 * np.pi * R_safe * 0.978)


def derive_N_crit_from_physics(v_c: float = 200.0, sigma_v: float = 20.0) -> float:
    """
    Derive N_crit from coherence geometry.
    
    Physical setup:
    - Coherent paths organized at scale ℓ₀
    - Differential rotation winds paths with period 2πR/v_c
    - Velocity dispersion σ_v sets azimuthal coherence length

    Coherence in azimuthal direction:
        ℓ_azimuthal ~ (σ_v/v_c) × 2πR

    After N orbits, paths wind to spacing:
        λ_wound ~ 2πR/N

    Destructive interference when:
        λ_wound ~ ℓ_azimuthal
        2πR/N ~ (σ_v/v_c) × 2πR
        N ~ v_c/σ_v

    For typical galaxies:
        v_c ~ 200 km/s, σ_v ~ 20 km/s
        → N_crit ~ 10 ✓
    """
    return v_c / sigma_v


if __name__ == "__main__":
    # Test the winding gate
    print("=" * 70)
    print("SPIRAL WINDING GATE TEST")
    print("=" * 70)
    
    systems = [
        ("Dwarf (DDO 154)", 10, 60),
        ("Intermediate (NGC 3198)", 15, 150),
        ("Massive (NGC 2403)", 15, 220),
        ("Milky Way", 8, 220),
    ]
    
    t_age = 10.0  # Gyr
    N_crit = 10.0
    
    print(f"\nAge: {t_age} Gyr")
    print(f"N_crit: {N_crit}")
    print()
    print(f"{'System':<25} {'R [kpc]':<10} {'v_c [km/s]':<12} {'N_orbits':<12} {'G_wind':<10}")
    print("-" * 80)
    
    for name, R, v_c in systems:
        R_arr = np.array([R])
        v_arr = np.array([v_c])
        
        G_wind = compute_winding_gate(R_arr, v_arr, t_age, N_crit)
        N_orbits = compute_N_orbits(R_arr, v_arr, t_age)[0]
        
        print(f"{name:<25} {R:<10.1f} {v_c:<12.0f} {N_orbits:<12.1f} {G_wind[0]:<10.3f}")
    
    print("\n" + "=" * 70)
    print("N_CRIT DERIVATION")
    print("=" * 70)
    N_derived = derive_N_crit_from_physics(200.0, 20.0)
    print(f"\nFrom v_c/σ_v = 200/20 = {N_derived:.1f}")
    print(f"Using N_crit = 10 ✓")
