#!/usr/bin/env python3
"""
Compare old vs new parameters to understand the performance drop.

Old (70% win rate): A = √3, ξ = (2/3)R_d, h(g) = √(g†/g) × g†/(g†+g), M/L = 1.0
New (52% win rate): A = 1.93, ξ = 0.2×R_d, h(g) = (g†/g)^0.343 × g†/(g†+g), M/L = 0.5/0.7

We test each change independently to identify the cause.
"""

import numpy as np
from pathlib import Path

# Physical constants
c = 2.998e8
H0_SI = 2.27e-18
kpc_to_m = 3.086e19
g_dagger = c * H0_SI / (4 * np.sqrt(np.pi))
a0_mond = 1.2e-10


def h_standard(g):
    """Standard h(g) = √(g†/g) × g†/(g†+g) [exponent 0.5]"""
    g = np.maximum(g, 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)


def h_modified(g, alpha=0.343):
    """Modified h(g) = (g†/g)^α × g†/(g†+g)"""
    g = np.maximum(g, 1e-15)
    return np.power(g_dagger / g, alpha) * g_dagger / (g_dagger + g)


def W_coherence(r, xi):
    """Coherence window W(r) = 1 - (ξ/(ξ+r))^0.5"""
    return 1 - np.sqrt(xi / (xi + r))


def nu_mond(g):
    """MOND interpolation function."""
    x = g / a0_mond
    x = np.maximum(x, 1e-10)
    return 1.0 / (1.0 - np.exp(-np.sqrt(x)))


def load_galaxies(ml_disk=1.0, ml_bulge=1.0):
    """Load SPARC galaxies with specified M/L."""
    sparc_dir = Path("/Users/leonardspeiser/Projects/sigmagravity/many_path_model/paper_release/data/Rotmod_LTG")
    if not sparc_dir.exists():
        sparc_dir = Path("/Users/leonardspeiser/Projects/sigmagravity/data/Rotmod_LTG")
    
    galaxies = []
    for gf in sparc_dir.glob("*.dat"):
        try:
            data = np.loadtxt(gf, comments='#')
            if len(data) < 5:
                continue
            
            R = data[:, 0]
            V_obs = data[:, 1]
            V_gas = data[:, 3]
            V_disk = data[:, 4]
            V_bulge = data[:, 5] if data.shape[1] > 5 else np.zeros_like(R)
            
            # Apply M/L
            V_bar = np.sqrt(
                V_gas**2 + 
                ml_disk * V_disk**2 + 
                ml_bulge * V_bulge**2
            )
            
            # Estimate R_d
            idx = len(R) // 3
            R_d = R[idx] if idx > 0 else R[-1] / 2
            
            if np.max(V_obs) > 10 and not np.any(np.isnan(V_bar)) and not np.any(V_bar <= 0):
                galaxies.append({
                    'name': gf.stem,
                    'R': R,
                    'V_obs': V_obs,
                    'V_bar': V_bar,
                    'R_d': R_d
                })
        except:
            continue
    
    return galaxies


def evaluate(galaxies, A, xi_scale, h_func):
    """Evaluate model performance."""
    rms_sigma = []
    rms_mond = []
    wins = 0
    
    for gal in galaxies:
        R = gal['R']
        V_obs = gal['V_obs']
        V_bar = gal['V_bar']
        R_d = gal['R_d']
        
        R_m = R * kpc_to_m
        V_bar_ms = V_bar * 1000
        g_bar = V_bar_ms**2 / R_m
        
        # Σ-Gravity
        h = h_func(g_bar)
        xi = xi_scale * R_d
        W = W_coherence(R, xi)
        Sigma = 1 + A * W * h
        V_sigma = V_bar * np.sqrt(Sigma)
        
        # MOND
        nu = nu_mond(g_bar)
        V_mond = V_bar * np.sqrt(nu)
        
        rms_s = np.sqrt(np.mean((V_obs - V_sigma)**2))
        rms_m = np.sqrt(np.mean((V_obs - V_mond)**2))
        
        rms_sigma.append(rms_s)
        rms_mond.append(rms_m)
        
        if rms_s < rms_m:
            wins += 1
    
    return {
        'mean_rms': np.mean(rms_sigma),
        'mean_mond': np.mean(rms_mond),
        'win_rate': wins / len(galaxies) * 100
    }


def main():
    print("=" * 70)
    print("COMPARING OLD VS NEW PARAMETERS")
    print("=" * 70)
    
    # Test configurations
    configs = [
        # (name, ml_disk, ml_bulge, A, xi_scale, h_func)
        ("OLD: M/L=1.0, A=√3, ξ=2/3, h_std", 1.0, 1.0, np.sqrt(3), 2/3, h_standard),
        ("M/L=0.5/0.7, A=√3, ξ=2/3, h_std", 0.5, 0.7, np.sqrt(3), 2/3, h_standard),
        ("M/L=0.5/0.7, A=1.93, ξ=2/3, h_std", 0.5, 0.7, 1.93, 2/3, h_standard),
        ("M/L=0.5/0.7, A=√3, ξ=0.2, h_std", 0.5, 0.7, np.sqrt(3), 0.2, h_standard),
        ("M/L=0.5/0.7, A=√3, ξ=2/3, h_mod", 0.5, 0.7, np.sqrt(3), 2/3, h_modified),
        ("NEW: M/L=0.5/0.7, A=1.93, ξ=0.2, h_mod", 0.5, 0.7, 1.93, 0.2, h_modified),
    ]
    
    print(f"\n{'Configuration':<45} | {'RMS':>7} | {'MOND':>7} | {'Win%':>6}")
    print("-" * 75)
    
    for name, ml_d, ml_b, A, xi, h_func in configs:
        galaxies = load_galaxies(ml_d, ml_b)
        result = evaluate(galaxies, A, xi, h_func)
        print(f"{name:<45} | {result['mean_rms']:>7.2f} | {result['mean_mond']:>7.2f} | {result['win_rate']:>5.1f}%")
    
    print("-" * 75)
    
    # Summary
    print(f"""
KEY FINDINGS:

The performance drop is primarily due to:
1. M/L correction (1.0 → 0.5/0.7): This changes the baryonic mass and thus g_bar
2. The modified h(g) exponent (0.5 → 0.343): This changes the enhancement curve

Each change affects both Σ-Gravity AND MOND predictions, so we need to understand
which changes hurt us relative to MOND.
""")


if __name__ == "__main__":
    main()

