#!/usr/bin/env python3
"""
Morphology-Based Prediction Test
================================

This tests the KEY UNIQUE PREDICTION of the coherence survival model:

    Galaxies with internal disruptions (bars, warps, rings) should show
    REDUCED outer enhancement compared to smooth disks at the same acceleration.

This is NOT predicted by MOND, which depends only on local g.

We use morphological classifications from the literature to identify:
- Smooth disks (SA, Sc, Sd types without bars)
- Barred galaxies (SB types)
- Disturbed/interacting systems

Author: Sigma Gravity Team
Date: December 2025
"""

import numpy as np
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

c = 2.998e8
H0_SI = 2.27e-18
cH0 = c * H0_SI
kpc_to_m = 3.086e19
G_SI = 6.674e-11

g_dagger = cH0 / (4 * math.sqrt(math.pi))
a0_mond = 1.2e-10

print("=" * 80)
print("MORPHOLOGY-BASED PREDICTION TEST")
print("=" * 80)


# =============================================================================
# GALAXY MORPHOLOGY CLASSIFICATION
# =============================================================================

# Manual classification based on NED, HyperLeda, and literature
# Format: galaxy_name -> (morphology_type, features, notes)

GALAXY_MORPHOLOGY = {
    # =========== SMOOTH DISKS (SA, Sc, Sd without bars) ===========
    'NGC2403': ('SAcd', 'smooth', 'Prototypical smooth disk, no bar'),
    'NGC3198': ('SBc', 'weak_bar', 'Weak bar, mostly smooth'),
    'NGC6946': ('SABcd', 'weak_bar', 'Very weak bar, multiple arms'),
    'NGC2841': ('SAb', 'smooth', 'Flocculent spiral, no bar'),
    'NGC7331': ('SAb', 'smooth', 'Unbarred, ring structure'),
    'NGC5055': ('SAbc', 'smooth', 'M63, Sunflower galaxy, no bar'),
    'NGC3521': ('SABbc', 'weak_bar', 'Flocculent, weak bar'),
    'NGC4736': ('SAab', 'smooth', 'M94, ring galaxy'),
    'NGC925': ('SABd', 'weak_bar', 'Asymmetric, weak bar'),
    'NGC2976': ('SAc', 'smooth', 'Dwarf spiral, no bar'),
    
    # =========== STRONGLY BARRED GALAXIES (SB types) ===========
    'NGC1300': ('SBbc', 'strong_bar', 'Classic grand-design barred spiral'),
    'NGC4548': ('SBb', 'strong_bar', 'M91, strong bar'),
    'NGC5383': ('SBb', 'strong_bar', 'Strong bar with ring'),
    'NGC3992': ('SBbc', 'strong_bar', 'M109, prominent bar'),
    'NGC2903': ('SBd', 'strong_bar', 'Strong bar, starburst'),
    'NGC4321': ('SABbc', 'moderate_bar', 'M100, moderate bar'),
    'NGC1097': ('SBb', 'strong_bar', 'Very strong bar'),
    'NGC4303': ('SABbc', 'moderate_bar', 'M61, moderate bar'),
    'NGC4535': ('SABc', 'moderate_bar', 'Moderate bar'),
    'NGC4579': ('SABb', 'moderate_bar', 'M58, moderate bar'),
    
    # =========== DISTURBED/INTERACTING ===========
    'NGC4631': ('SBd', 'disturbed', 'Edge-on, interacting with NGC4627'),
    'NGC4656': ('SBm', 'disturbed', 'Hockey Stick galaxy, tidal'),
    'NGC3034': ('I0', 'disturbed', 'M82, starburst, outflows'),
    'NGC5194': ('SABbc', 'disturbed', 'M51, interacting with NGC5195'),
    'NGC4038': ('SBm', 'merger', 'Antennae galaxy'),
    
    # =========== DWARF IRREGULARS (control group) ===========
    'DDO154': ('IBm', 'irregular', 'Dwarf irregular, dark matter dominated'),
    'DDO168': ('IBm', 'irregular', 'Dwarf irregular'),
    'DDO170': ('IBm', 'irregular', 'Dwarf irregular'),
    'UGC128': ('Im', 'irregular', 'Dwarf irregular'),
    'IC2574': ('SABm', 'irregular', 'Dwarf irregular'),
}


def classify_galaxy(name: str) -> Tuple[str, str]:
    """
    Classify a galaxy as smooth, barred, or disturbed.
    
    Returns:
        (category, confidence)
        category: 'smooth', 'barred', 'disturbed', 'irregular', 'unknown'
        confidence: 'high', 'medium', 'low'
    """
    if name in GALAXY_MORPHOLOGY:
        morph_type, features, notes = GALAXY_MORPHOLOGY[name]
        
        if features in ['smooth', 'weak_bar']:
            return ('smooth', 'high' if features == 'smooth' else 'medium')
        elif features in ['strong_bar', 'moderate_bar']:
            return ('barred', 'high' if features == 'strong_bar' else 'medium')
        elif features in ['disturbed', 'merger']:
            return ('disturbed', 'high')
        elif features == 'irregular':
            return ('irregular', 'high')
    
    # Try to infer from name patterns
    if 'DDO' in name or 'UGC' in name:
        return ('irregular', 'low')
    
    return ('unknown', 'low')


# =============================================================================
# MODELS
# =============================================================================

def h_function(g: np.ndarray, g_dag: float = g_dagger) -> np.ndarray:
    g = np.maximum(g, 1e-15)
    return np.sqrt(g_dag / g) * g_dag / (g_dag + g)


def survival_model(R_kpc: np.ndarray, V_bar: np.ndarray,
                   r_char: float = 20.0, alpha: float = 0.1, beta: float = 0.3,
                   A: float = np.sqrt(3)) -> np.ndarray:
    """Coherence survival threshold model."""
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m
    
    r_ratio = r_char / np.maximum(R_kpc, 0.01)
    g_ratio = g_bar / g_dagger
    
    exponent = -np.power(r_ratio, beta) * np.power(g_ratio, alpha)
    P_survive = np.exp(exponent)
    
    h = h_function(g_bar)
    Sigma = 1.0 + A * P_survive * h
    return V_bar * np.sqrt(Sigma)


def predict_mond(R_kpc: np.ndarray, V_bar: np.ndarray) -> np.ndarray:
    """MOND simple interpolation."""
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m
    
    x = g_bar / a0_mond
    x = np.maximum(x, 1e-10)
    nu = 1.0 / (1.0 - np.exp(-np.sqrt(x)))
    
    g_obs = g_bar * nu
    return np.sqrt(g_obs * R_m) / 1000


# =============================================================================
# DATA LOADING
# =============================================================================

def find_sparc_data() -> Optional[Path]:
    possible_paths = [
        Path("/Users/leonardspeiser/Projects/sigmagravity/data/Rotmod_LTG"),
        Path("/Users/leonardspeiser/Projects/sigmagravity/many_path_model/paper_release/data/Rotmod_LTG"),
    ]
    for p in possible_paths:
        if p.exists():
            return p
    return None


def load_galaxy_rotmod(rotmod_file: Path) -> Optional[Dict]:
    R, V_obs, V_err, V_gas, V_disk, V_bulge = [], [], [], [], [], []
    
    with open(rotmod_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 6:
                try:
                    R.append(float(parts[0]))
                    V_obs.append(float(parts[1]))
                    V_err.append(float(parts[2]))
                    V_gas.append(float(parts[3]))
                    V_disk.append(float(parts[4]))
                    V_bulge.append(float(parts[5]) if len(parts) > 5 else 0.0)
                except ValueError:
                    continue
    
    if len(R) < 3:
        return None
    
    R = np.array(R)
    V_obs = np.array(V_obs)
    V_err = np.array(V_err)
    V_gas = np.array(V_gas)
    V_disk = np.array(V_disk)
    V_bulge = np.array(V_bulge)
    
    V_bar_sq = np.sign(V_gas) * V_gas**2 + np.sign(V_disk) * V_disk**2 + V_bulge**2
    if np.any(V_bar_sq < 0):
        return None
    V_bar = np.sqrt(V_bar_sq)
    
    return {'R': R, 'V_obs': V_obs, 'V_err': V_err, 'V_bar': V_bar}


def load_all_galaxies(sparc_dir: Path) -> Dict[str, Dict]:
    galaxies = {}
    for rotmod_file in sorted(sparc_dir.glob('*_rotmod.dat')):
        name = rotmod_file.stem.replace('_rotmod', '')
        data = load_galaxy_rotmod(rotmod_file)
        if data is not None:
            galaxies[name] = data
    return galaxies


# =============================================================================
# METRICS
# =============================================================================

def compute_rms(V_obs: np.ndarray, V_pred: np.ndarray) -> float:
    return np.sqrt(np.mean((V_obs - V_pred)**2))


def compute_outer_residual(data: Dict, model_func, outer_frac: float = 0.5) -> float:
    """Compute residual in outer region."""
    R = data['R']
    V_obs = data['V_obs']
    V_bar = data['V_bar']
    
    R_cut = outer_frac * R.max()
    outer_mask = R > R_cut
    
    if outer_mask.sum() < 2:
        return np.nan
    
    V_pred = model_func(R, V_bar)
    return np.sqrt(np.mean((V_obs[outer_mask] - V_pred[outer_mask])**2))


def compute_enhancement_ratio(data: Dict, outer_frac: float = 0.5) -> float:
    """Compute observed V_obs/V_bar ratio in outer region."""
    R = data['R']
    V_obs = data['V_obs']
    V_bar = data['V_bar']
    
    R_cut = outer_frac * R.max()
    outer_mask = R > R_cut
    
    if outer_mask.sum() < 2:
        return np.nan
    
    V_bar_outer = np.mean(V_bar[outer_mask])
    V_obs_outer = np.mean(V_obs[outer_mask])
    
    if V_bar_outer < 5:
        return np.nan
    
    return V_obs_outer / V_bar_outer


# =============================================================================
# MAIN TEST
# =============================================================================

def run_morphology_test():
    """
    Test whether morphology correlates with model performance.
    
    PREDICTION: Survival model should perform better on smooth disks
    than on barred/disturbed galaxies (relative to MOND).
    """
    
    sparc_dir = find_sparc_data()
    if sparc_dir is None:
        print("ERROR: SPARC data not found!")
        return
    
    galaxies = load_all_galaxies(sparc_dir)
    print(f"\nLoaded {len(galaxies)} galaxies")
    
    # Classify all galaxies
    categories = {'smooth': [], 'barred': [], 'disturbed': [], 'irregular': [], 'unknown': []}
    
    for name in galaxies.keys():
        cat, conf = classify_galaxy(name)
        categories[cat].append((name, conf))
    
    print("\n" + "=" * 80)
    print("GALAXY CLASSIFICATION")
    print("=" * 80)
    
    for cat, gals in categories.items():
        high_conf = [g for g, c in gals if c == 'high']
        print(f"\n{cat.upper()}: {len(gals)} total, {len(high_conf)} high-confidence")
        if len(high_conf) > 0 and len(high_conf) <= 10:
            print(f"  High-confidence: {', '.join(high_conf)}")
    
    # =========================================================================
    # TEST 1: Compare model performance by morphology
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST 1: MODEL PERFORMANCE BY MORPHOLOGY")
    print("=" * 80)
    
    results = {}
    
    for cat in ['smooth', 'barred', 'disturbed', 'irregular']:
        gal_list = [g for g, c in categories[cat] if c in ['high', 'medium']]
        
        survival_rms = []
        mond_rms = []
        survival_outer = []
        mond_outer = []
        
        for name in gal_list:
            if name not in galaxies:
                continue
            
            data = galaxies[name]
            
            try:
                rms_s = compute_rms(data['V_obs'], survival_model(data['R'], data['V_bar']))
                rms_m = compute_rms(data['V_obs'], predict_mond(data['R'], data['V_bar']))
                
                outer_s = compute_outer_residual(data, survival_model)
                outer_m = compute_outer_residual(data, predict_mond)
                
                if np.isfinite(rms_s) and np.isfinite(rms_m):
                    survival_rms.append(rms_s)
                    mond_rms.append(rms_m)
                
                if np.isfinite(outer_s) and np.isfinite(outer_m):
                    survival_outer.append(outer_s)
                    mond_outer.append(outer_m)
            except:
                continue
        
        if len(survival_rms) > 0:
            results[cat] = {
                'n': len(survival_rms),
                'survival_rms': np.mean(survival_rms),
                'mond_rms': np.mean(mond_rms),
                'survival_outer': np.mean(survival_outer) if len(survival_outer) > 0 else np.nan,
                'mond_outer': np.mean(mond_outer) if len(mond_outer) > 0 else np.nan,
                'survival_wins': np.sum(np.array(survival_rms) < np.array(mond_rms)),
            }
    
    print(f"\n{'Category':<12} {'N':<5} {'Surv RMS':<12} {'MOND RMS':<12} {'Surv Wins':<12} {'Δ(S-M)':<10}")
    print("-" * 70)
    
    for cat, res in results.items():
        delta = res['survival_rms'] - res['mond_rms']
        win_pct = 100 * res['survival_wins'] / res['n'] if res['n'] > 0 else 0
        print(f"{cat:<12} {res['n']:<5} {res['survival_rms']:<12.2f} {res['mond_rms']:<12.2f} {win_pct:<12.1f}% {delta:<10.2f}")
    
    # =========================================================================
    # TEST 2: Outer region enhancement by morphology
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST 2: OUTER ENHANCEMENT BY MORPHOLOGY")
    print("=" * 80)
    print("\nPREDICTION: Barred/disturbed galaxies should show LESS outer enhancement")
    print("            at the same acceleration (coherence disrupted).\n")
    
    for cat in ['smooth', 'barred', 'disturbed']:
        gal_list = [g for g, c in categories[cat] if c in ['high', 'medium']]
        
        enhancements = []
        g_ratios = []  # g/g† at outer radii
        
        for name in gal_list:
            if name not in galaxies:
                continue
            
            data = galaxies[name]
            R = data['R']
            V_bar = data['V_bar']
            V_obs = data['V_obs']
            
            # Outer region
            R_cut = 0.5 * R.max()
            outer_mask = R > R_cut
            
            if outer_mask.sum() < 2:
                continue
            
            # Enhancement ratio
            enh = compute_enhancement_ratio(data)
            if not np.isfinite(enh):
                continue
            
            # Mean g/g† in outer region
            R_m = R[outer_mask] * kpc_to_m
            V_bar_ms = V_bar[outer_mask] * 1000
            g_bar = V_bar_ms**2 / R_m
            g_ratio = np.mean(g_bar / g_dagger)
            
            enhancements.append(enh)
            g_ratios.append(g_ratio)
        
        if len(enhancements) > 0:
            print(f"\n{cat.upper()} ({len(enhancements)} galaxies):")
            print(f"  Mean enhancement (V_obs/V_bar): {np.mean(enhancements):.3f}")
            print(f"  Mean g/g†:                      {np.mean(g_ratios):.3f}")
            print(f"  Std enhancement:                {np.std(enhancements):.3f}")
    
    # =========================================================================
    # TEST 3: Detailed comparison of matched pairs
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST 3: MATCHED PAIR COMPARISON")
    print("=" * 80)
    print("\nComparing smooth vs barred galaxies with similar properties:\n")
    
    # Find pairs with similar V_max
    smooth_gals = [(g, galaxies[g]['V_obs'].max()) for g, c in categories['smooth'] 
                   if g in galaxies and c in ['high', 'medium']]
    barred_gals = [(g, galaxies[g]['V_obs'].max()) for g, c in categories['barred'] 
                   if g in galaxies and c in ['high', 'medium']]
    
    print(f"{'Smooth Galaxy':<15} {'V_max':<8} {'Barred Galaxy':<15} {'V_max':<8} {'Smooth Δ':<10} {'Barred Δ':<10}")
    print("-" * 80)
    
    matched_pairs = []
    for sg, sv in sorted(smooth_gals, key=lambda x: x[1]):
        # Find closest barred galaxy by V_max
        best_match = None
        best_diff = np.inf
        for bg, bv in barred_gals:
            diff = abs(sv - bv)
            if diff < best_diff and diff < 30:  # Within 30 km/s
                best_diff = diff
                best_match = (bg, bv)
        
        if best_match is not None:
            bg, bv = best_match
            
            # Compute Δ = survival_rms - mond_rms for each
            s_data = galaxies[sg]
            b_data = galaxies[bg]
            
            s_surv = compute_rms(s_data['V_obs'], survival_model(s_data['R'], s_data['V_bar']))
            s_mond = compute_rms(s_data['V_obs'], predict_mond(s_data['R'], s_data['V_bar']))
            s_delta = s_surv - s_mond
            
            b_surv = compute_rms(b_data['V_obs'], survival_model(b_data['R'], b_data['V_bar']))
            b_mond = compute_rms(b_data['V_obs'], predict_mond(b_data['R'], b_data['V_bar']))
            b_delta = b_surv - b_mond
            
            print(f"{sg:<15} {sv:<8.0f} {bg:<15} {bv:<8.0f} {s_delta:<10.2f} {b_delta:<10.2f}")
            
            matched_pairs.append((s_delta, b_delta))
            
            # Remove used barred galaxy
            barred_gals = [(g, v) for g, v in barred_gals if g != bg]
    
    if len(matched_pairs) > 0:
        smooth_deltas = [p[0] for p in matched_pairs]
        barred_deltas = [p[1] for p in matched_pairs]
        
        print(f"\n{'Summary':<30} {'Smooth':<15} {'Barred':<15}")
        print("-" * 60)
        print(f"{'Mean Δ(Survival-MOND):':<30} {np.mean(smooth_deltas):<15.2f} {np.mean(barred_deltas):<15.2f}")
        print(f"{'Std Δ(Survival-MOND):':<30} {np.std(smooth_deltas):<15.2f} {np.std(barred_deltas):<15.2f}")
        
        print(f"""
INTERPRETATION:
    If Δ(Survival-MOND) is MORE NEGATIVE for smooth galaxies than barred,
    this supports the coherence survival prediction.
    
    Smooth mean Δ: {np.mean(smooth_deltas):.2f}
    Barred mean Δ: {np.mean(barred_deltas):.2f}
    Difference:    {np.mean(smooth_deltas) - np.mean(barred_deltas):.2f}
    
    {'SUPPORTS' if np.mean(smooth_deltas) < np.mean(barred_deltas) else 'DOES NOT SUPPORT'} the prediction.
""")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY: MORPHOLOGY PREDICTION TEST")
    print("=" * 80)
    print("""
The coherence survival model predicts that:

1. SMOOTH DISKS should show the strongest enhancement because coherence
   can propagate uninterrupted from inner to outer regions.

2. BARRED GALAXIES should show reduced outer enhancement because the bar
   creates a disruption zone that "breaks the chain."

3. DISTURBED/INTERACTING galaxies should show the weakest enhancement
   because tidal forces and non-equilibrium dynamics constantly reset
   coherence.

This is a UNIQUE PREDICTION not made by MOND or standard dark matter models.

To definitively test this:
1. Obtain proper morphological classifications for all SPARC galaxies
2. Control for mass, size, and acceleration profiles
3. Compare residual patterns in outer rotation curves
""")


if __name__ == "__main__":
    run_morphology_test()




