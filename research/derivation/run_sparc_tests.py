#!/usr/bin/env python3
"""
Σ-GRAVITY HYPOTHESIS DISTINGUISHING TESTS
==========================================
Adapted for sigmagravity project data structure.

Tests:
1. LSB vs HSB at same x = R/R_disk
2. RAR vs Geometry (partial correlation)
"""

import numpy as np
from pathlib import Path
import json

try:
    from scipy.stats import pearsonr, ks_2samp
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available")

# =============================================================================
# CONFIGURATION - adapted to project structure
# =============================================================================

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
ROTMOD_DIR = DATA_DIR / "Rotmod_LTG"
SPARC_TABLE = DATA_DIR / "sparc" / "Table1_SPARC.dat"
RESULTS_DIR = PROJECT_DIR / "derivation" / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_sparc_table():
    """Load SPARC galaxy properties from Table1_SPARC.dat"""
    
    galaxies = {}
    
    with open(SPARC_TABLE, 'r') as f:
        for line in f:
            if line.startswith('#') or len(line.strip()) == 0:
                continue
            
            parts = line.split()
            if len(parts) < 10:
                continue
            
            name = parts[0]
            try:
                # Parse columns based on SPARC format
                # Name T D e_D f Inc e_Inc L36 e_L36 Rdisk SBdisk e_SBdisk Vflat ...
                galaxies[name] = {
                    'T': int(parts[1]),           # Hubble type
                    'D': float(parts[2]),          # Distance (Mpc)
                    'Inc': float(parts[4]),        # Inclination (deg)
                    'L36': float(parts[6]),        # 3.6μm luminosity (10^9 L_sun)
                    'Rdisk': float(parts[8]),      # Disk scale length (kpc)
                    'SBdisk': float(parts[9]),     # Central surface brightness (L/pc²)
                    'Vflat': float(parts[13]) if len(parts) > 13 and parts[13] != '0.0' else None,
                }
            except (ValueError, IndexError):
                continue
    
    print(f"Loaded {len(galaxies)} galaxies from SPARC table")
    return galaxies


def load_rotation_curves():
    """Load individual rotation curves from Rotmod_LTG/"""
    
    rotation_curves = {}
    
    for rc_file in ROTMOD_DIR.glob("*_rotmod.dat"):
        name = rc_file.stem.replace('_rotmod', '')
        
        try:
            # Read data, skip header lines
            data = []
            with open(rc_file, 'r') as f:
                for line in f:
                    if line.startswith('#') or len(line.strip()) == 0:
                        continue
                    parts = line.split()
                    if len(parts) >= 6:
                        data.append([float(p) for p in parts[:6]])
            
            if len(data) > 0:
                data = np.array(data)
                rotation_curves[name] = {
                    'R': data[:, 0],      # kpc
                    'v_obs': data[:, 1],  # km/s
                    'e_v': data[:, 2],    # km/s error
                    'v_gas': data[:, 3],  # km/s
                    'v_disk': data[:, 4], # km/s
                    'v_bul': data[:, 5],  # km/s
                }
        except Exception as e:
            print(f"Warning: Could not load {rc_file}: {e}")
    
    print(f"Loaded {len(rotation_curves)} rotation curves")
    return rotation_curves


def compute_K(rc, Rdisk):
    """Compute enhancement factor K = v_obs² / v_bar²"""
    R = rc['R']
    v_obs = rc['v_obs']
    v_bar = np.sqrt(rc['v_gas']**2 + rc['v_disk']**2 + rc['v_bul']**2)
    
    # Avoid division by zero
    v_bar = np.maximum(v_bar, 1.0)
    
    K = (v_obs / v_bar)**2
    x = R / Rdisk
    
    return x, K, v_bar


# =============================================================================
# TEST 1: LSB vs HSB AT SAME x
# =============================================================================

def test_lsb_vs_hsb(galaxies, rotation_curves):
    """
    Test if K differs between LSB and HSB galaxies at same x = R/R_disk.
    
    If Σ matters: LSB should have HIGHER K than HSB at same x
    If only x matters: K should be SAME regardless of surface brightness
    """
    
    print("\n" + "="*70)
    print("  TEST 1: LSB vs HSB at same x = R/R_disk")
    print("="*70)
    
    # Classify by surface brightness
    # Note: SPARC uses L/pc² not mag/arcsec²
    # Lower SBdisk value = Lower surface brightness
    
    # Get median SB for classification
    sb_values = [g['SBdisk'] for g in galaxies.values() if g['SBdisk'] > 0]
    sb_median = np.median(sb_values)
    sb_25 = np.percentile(sb_values, 25)
    sb_75 = np.percentile(sb_values, 75)
    
    print(f"\n  Surface brightness (L/pc²): median={sb_median:.1f}, Q1={sb_25:.1f}, Q3={sb_75:.1f}")
    
    lsb_galaxies = []
    hsb_galaxies = []
    
    for name, props in galaxies.items():
        if name not in rotation_curves:
            continue
        if props['Rdisk'] <= 0 or props['SBdisk'] <= 0:
            continue
        
        galaxy_info = {
            'name': name,
            'SBdisk': props['SBdisk'],
            'Rdisk': props['Rdisk'],
            'rc': rotation_curves[name]
        }
        
        # LSB = below 25th percentile, HSB = above 75th percentile
        if props['SBdisk'] < sb_25:
            lsb_galaxies.append(galaxy_info)
        elif props['SBdisk'] > sb_75:
            hsb_galaxies.append(galaxy_info)
    
    print(f"\n  LSB galaxies (SB < {sb_25:.0f} L/pc²): {len(lsb_galaxies)}")
    print(f"  HSB galaxies (SB > {sb_75:.0f} L/pc²): {len(hsb_galaxies)}")
    
    # Bin by x and compare K
    x_bins = np.linspace(0.5, 5.0, 19)
    x_centers = 0.5 * (x_bins[:-1] + x_bins[1:])
    
    lsb_K_binned = {i: [] for i in range(len(x_centers))}
    hsb_K_binned = {i: [] for i in range(len(x_centers))}
    
    for gal in lsb_galaxies:
        x, K, _ = compute_K(gal['rc'], gal['Rdisk'])
        for j in range(len(K)):
            bin_idx = np.digitize(x[j], x_bins) - 1
            if 0 <= bin_idx < len(x_centers) and 0.1 < K[j] < 50:
                lsb_K_binned[bin_idx].append(K[j])
    
    for gal in hsb_galaxies:
        x, K, _ = compute_K(gal['rc'], gal['Rdisk'])
        for j in range(len(K)):
            bin_idx = np.digitize(x[j], x_bins) - 1
            if 0 <= bin_idx < len(x_centers) and 0.1 < K[j] < 50:
                hsb_K_binned[bin_idx].append(K[j])
    
    # Compare
    print(f"\n  {'x bin':<12s} {'LSB K':<15s} {'HSB K':<15s} {'Diff':<12s} {'p-value':<10s}")
    print("  " + "-"*65)
    
    results = []
    for i, xc in enumerate(x_centers):
        if len(lsb_K_binned[i]) > 3 and len(hsb_K_binned[i]) > 3:
            lsb_med = np.median(lsb_K_binned[i])
            hsb_med = np.median(hsb_K_binned[i])
            lsb_err = np.std(lsb_K_binned[i]) / np.sqrt(len(lsb_K_binned[i]))
            hsb_err = np.std(hsb_K_binned[i]) / np.sqrt(len(hsb_K_binned[i]))
            diff = lsb_med - hsb_med
            
            if SCIPY_AVAILABLE:
                _, p = ks_2samp(lsb_K_binned[i], hsb_K_binned[i])
            else:
                p = np.nan
            
            print(f"  {xc:<12.2f} {lsb_med:.3f}±{lsb_err:.3f}    "
                  f"{hsb_med:.3f}±{hsb_err:.3f}    {diff:+.3f}      {p:.3f}")
            
            results.append({
                'x': float(xc), 
                'lsb_K': float(lsb_med), 
                'hsb_K': float(hsb_med),
                'diff': float(diff), 
                'p': float(p),
                'n_lsb': len(lsb_K_binned[i]),
                'n_hsb': len(hsb_K_binned[i])
            })
    
    # Conclusion
    if len(results) > 0:
        avg_diff = np.mean([r['diff'] for r in results])
        weighted_diff = np.average([r['diff'] for r in results], 
                                   weights=[r['n_lsb'] + r['n_hsb'] for r in results])
        sig_diffs = sum(1 for r in results if r['p'] < 0.05)
        
        print(f"\n  RESULTS:")
        print(f"  Average K difference (LSB - HSB): {avg_diff:+.4f}")
        print(f"  Weighted average difference: {weighted_diff:+.4f}")
        print(f"  Bins with p < 0.05: {sig_diffs}/{len(results)}")
        
        print(f"\n  CONCLUSION:")
        if abs(weighted_diff) < 0.3:
            print(f"  → K is SIMILAR for LSB and HSB at same x")
            print(f"  → SUPPORTS: Geometry hypothesis (x = R/R_disk)")
            print(f"  → DISFAVORS: Surface brightness Σ hypothesis")
        elif weighted_diff > 0.3:
            print(f"  → LSB has HIGHER K than HSB at same x")
            print(f"  → SUPPORTS: Σ-dependent hypothesis")
        else:
            print(f"  → HSB has HIGHER K than LSB at same x")
            print(f"  → Unexpected - needs investigation")
    
    # Save
    with open(RESULTS_DIR / "test1_lsb_hsb.json", 'w') as f:
        json.dump({
            'results': results, 
            'n_lsb': len(lsb_galaxies), 
            'n_hsb': len(hsb_galaxies),
            'avg_diff': float(avg_diff) if len(results) > 0 else None
        }, f, indent=2)
    
    return results


# =============================================================================
# TEST 2: RAR vs GEOMETRY
# =============================================================================

def test_rar_vs_geometry(galaxies, rotation_curves):
    """
    Test if K depends more on g_bar (RAR) or x (geometry).
    
    Uses partial correlation analysis.
    """
    
    print("\n" + "="*70)
    print("  TEST 2: RAR vs Geometry (Partial Correlation)")
    print("="*70)
    
    # Collect all data points
    all_K = []
    all_x = []
    all_g = []
    all_galaxies = []
    
    for name, props in galaxies.items():
        if name not in rotation_curves:
            continue
        if props['Rdisk'] <= 0:
            continue
        
        rc = rotation_curves[name]
        x, K, v_bar = compute_K(rc, props['Rdisk'])
        R = rc['R']
        
        for i in range(len(R)):
            g_bar = v_bar[i]**2 / max(R[i], 0.1)  # (km/s)²/kpc
            
            if 0.1 < K[i] < 50 and 0.1 < x[i] < 10 and g_bar > 10:
                all_K.append(K[i])
                all_x.append(x[i])
                all_g.append(g_bar)
                all_galaxies.append(name)
    
    K = np.array(all_K)
    x = np.array(all_x)
    log_g = np.log10(all_g)
    
    print(f"\n  Total data points: {len(K)}")
    print(f"  From {len(set(all_galaxies))} galaxies")
    
    if not SCIPY_AVAILABLE:
        print("  ERROR: scipy required for correlation analysis")
        return None
    
    # Simple correlations
    r_Kx, p_Kx = pearsonr(K, x)
    r_Kg, p_Kg = pearsonr(K, log_g)
    
    print(f"\n  Simple correlations:")
    print(f"    K vs x:        r = {r_Kx:+.4f}  (p = {p_Kx:.2e})")
    print(f"    K vs log(g):   r = {r_Kg:+.4f}  (p = {p_Kg:.2e})")
    
    # Partial correlations using residual method
    from numpy.linalg import lstsq
    
    # K vs x, controlling for g
    A1 = np.column_stack([np.ones(len(K)), log_g])
    coef_K1, _, _, _ = lstsq(A1, K, rcond=None)
    K_resid1 = K - A1 @ coef_K1
    coef_x1, _, _, _ = lstsq(A1, x, rcond=None)
    x_resid1 = x - A1 @ coef_x1
    r_partial_Kx, _ = pearsonr(K_resid1, x_resid1)
    
    # K vs g, controlling for x
    A2 = np.column_stack([np.ones(len(K)), x])
    coef_K2, _, _, _ = lstsq(A2, K, rcond=None)
    K_resid2 = K - A2 @ coef_K2
    coef_g2, _, _, _ = lstsq(A2, log_g, rcond=None)
    g_resid2 = log_g - A2 @ coef_g2
    r_partial_Kg, _ = pearsonr(K_resid2, g_resid2)
    
    print(f"\n  Partial correlations:")
    print(f"    K vs x (controlling g):       r_partial = {r_partial_Kx:+.4f}")
    print(f"    K vs log(g) (controlling x):  r_partial = {r_partial_Kg:+.4f}")
    
    # Test at fixed x: K variation with g
    print(f"\n  Test at fixed x bins - K variation with g_bar:")
    print(f"  {'x range':<15s} {'r(K,g)':<10s} {'n':<8s}")
    print("  " + "-"*40)
    
    fixed_x_results = []
    for x_lo, x_hi in [(0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 5.0)]:
        mask = (x >= x_lo) & (x < x_hi)
        if mask.sum() > 20:
            r_local, p_local = pearsonr(K[mask], log_g[mask])
            print(f"  [{x_lo:.1f}, {x_hi:.1f}]       {r_local:+.3f}      {mask.sum()}")
            fixed_x_results.append({
                'x_range': f"[{x_lo}, {x_hi}]",
                'r': float(r_local),
                'n': int(mask.sum())
            })
    
    # Conclusion
    print(f"\n  CONCLUSION:")
    if abs(r_partial_Kx) > abs(r_partial_Kg) + 0.1:
        print(f"  → x has STRONGER partial correlation with K")
        print(f"    ({r_partial_Kx:+.3f} vs {r_partial_Kg:+.3f})")
        print(f"  → SUPPORTS: Geometry hypothesis (x = R/R_disk)")
        print(f"  → DISFAVORS: RAR (g_bar dependence)")
        winner = "geometry"
    elif abs(r_partial_Kg) > abs(r_partial_Kx) + 0.1:
        print(f"  → g_bar has STRONGER partial correlation with K")
        print(f"    ({r_partial_Kg:+.3f} vs {r_partial_Kx:+.3f})")
        print(f"  → SUPPORTS: RAR hypothesis")
        print(f"  → DISFAVORS: Geometry hypothesis")
        winner = "RAR"
    else:
        print(f"  → Both have similar partial correlations")
        print(f"    (K vs x: {r_partial_Kx:+.3f}, K vs g: {r_partial_Kg:+.3f})")
        print(f"  → INCONCLUSIVE")
        winner = "inconclusive"
    
    # Save
    results = {
        'n_points': len(K),
        'n_galaxies': len(set(all_galaxies)),
        'r_K_x': float(r_Kx),
        'r_K_g': float(r_Kg),
        'r_partial_K_x': float(r_partial_Kx),
        'r_partial_K_g': float(r_partial_Kg),
        'fixed_x_results': fixed_x_results,
        'winner': winner
    }
    
    with open(RESULTS_DIR / "test2_rar_geometry.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


# =============================================================================
# TEST 3: Tanh fit quality by galaxy type
# =============================================================================

def test_tanh_universality(galaxies, rotation_curves):
    """
    Test if tanh form fits all galaxy types equally well.
    """
    
    print("\n" + "="*70)
    print("  TEST 3: Tanh Universality by Galaxy Type")
    print("="*70)
    
    from scipy.optimize import curve_fit
    
    def tanh_model(x, x_c, w, K_inf):
        return np.tanh((x - x_c) / w) + K_inf
    
    # Group by Hubble type
    type_groups = {
        'Early (T<2)': [],
        'Mid (2≤T<6)': [],
        'Late (6≤T<9)': [],
        'Dwarf (T≥9)': []
    }
    
    for name, props in galaxies.items():
        if name not in rotation_curves:
            continue
        if props['Rdisk'] <= 0:
            continue
        
        T = props['T']
        rc = rotation_curves[name]
        x, K, _ = compute_K(rc, props['Rdisk'])
        
        # Filter valid points
        mask = (K > 0.1) & (K < 50) & (x > 0.1) & (x < 10)
        if mask.sum() < 5:
            continue
        
        galaxy_info = {
            'name': name,
            'T': T,
            'x': x[mask],
            'K': K[mask]
        }
        
        if T < 2:
            type_groups['Early (T<2)'].append(galaxy_info)
        elif T < 6:
            type_groups['Mid (2≤T<6)'].append(galaxy_info)
        elif T < 9:
            type_groups['Late (6≤T<9)'].append(galaxy_info)
        else:
            type_groups['Dwarf (T≥9)'].append(galaxy_info)
    
    print(f"\n  Galaxy counts by type:")
    for group, gals in type_groups.items():
        print(f"    {group}: {len(gals)}")
    
    # Fit tanh to each group's combined data
    print(f"\n  Tanh fits by galaxy type:")
    print(f"  {'Type':<18s} {'x_c':<10s} {'w':<10s} {'R²':<10s}")
    print("  " + "-"*50)
    
    results = {}
    for group, gals in type_groups.items():
        if len(gals) < 3:
            continue
        
        # Combine all data
        all_x = np.concatenate([g['x'] for g in gals])
        all_K = np.concatenate([g['K'] for g in gals])
        
        try:
            popt, _ = curve_fit(tanh_model, all_x, all_K, 
                               p0=[1.0, 0.5, 1.0],
                               bounds=([0, 0.1, 0.5], [5, 3, 2]),
                               maxfev=5000)
            
            K_pred = tanh_model(all_x, *popt)
            ss_res = np.sum((all_K - K_pred)**2)
            ss_tot = np.sum((all_K - np.mean(all_K))**2)
            r2 = 1 - ss_res / ss_tot
            
            print(f"  {group:<18s} {popt[0]:.3f}     {popt[1]:.3f}     {r2:.3f}")
            
            results[group] = {
                'x_c': float(popt[0]),
                'w': float(popt[1]),
                'K_inf': float(popt[2]),
                'r2': float(r2),
                'n_galaxies': len(gals),
                'n_points': len(all_x)
            }
        except Exception as e:
            print(f"  {group:<18s} Fit failed: {e}")
    
    # Conclusion
    if len(results) > 1:
        x_c_values = [r['x_c'] for r in results.values()]
        x_c_spread = max(x_c_values) - min(x_c_values)
        
        print(f"\n  CONCLUSION:")
        print(f"  x_c range across types: {min(x_c_values):.2f} - {max(x_c_values):.2f}")
        
        if x_c_spread < 0.5:
            print(f"  → x_c is SIMILAR across galaxy types (spread = {x_c_spread:.2f})")
            print(f"  → SUPPORTS: Universal tanh form")
        else:
            print(f"  → x_c VARIES significantly (spread = {x_c_spread:.2f})")
            print(f"  → Galaxy type may influence transition location")
    
    # Save
    with open(RESULTS_DIR / "test3_tanh_universality.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("  Σ-GRAVITY HYPOTHESIS DISTINGUISHING TESTS")
    print("="*70)
    
    # Load data
    print("\n  Loading SPARC data...")
    galaxies = load_sparc_table()
    rotation_curves = load_rotation_curves()
    
    # Match names (handle case differences)
    galaxies_lower = {k.lower(): v for k, v in galaxies.items()}
    rc_lower = {k.lower(): v for k, v in rotation_curves.items()}
    
    matched = {}
    for name_lower, props in galaxies_lower.items():
        if name_lower in rc_lower:
            # Find original case name
            orig_name = [k for k in galaxies.keys() if k.lower() == name_lower][0]
            matched[orig_name] = props
            matched[orig_name]['_rc_key'] = [k for k in rotation_curves.keys() if k.lower() == name_lower][0]
    
    print(f"  Matched {len(matched)} galaxies between table and rotation curves")
    
    # Update rotation_curves dict to use matched names
    matched_rc = {}
    for name, props in matched.items():
        rc_key = props.pop('_rc_key')
        matched_rc[name] = rotation_curves[rc_key]
    
    # Run tests
    print("\n" + "="*70)
    print("  RUNNING TESTS")
    print("="*70)
    
    all_results = {}
    
    # Test 1
    all_results['test1'] = test_lsb_vs_hsb(matched, matched_rc)
    
    # Test 2
    all_results['test2'] = test_rar_vs_geometry(matched, matched_rc)
    
    # Test 3
    all_results['test3'] = test_tanh_universality(matched, matched_rc)
    
    # Summary
    print("\n" + "="*70)
    print("  OVERALL SUMMARY")
    print("="*70)
    
    print(f"\n  Results saved to: {RESULTS_DIR}")
    
    return all_results


if __name__ == "__main__":
    results = main()
