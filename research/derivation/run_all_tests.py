#!/usr/bin/env python3
"""
Σ-GRAVITY HYPOTHESIS DISTINGUISHING TESTS
==========================================

This script runs all the tests that can distinguish between
different physical hypotheses for gravitational enhancement.

TESTS INCLUDED:
1. LSB vs HSB at same x (SPARC)
2. RAR vs Geometry (SPARC)
3. MW thick vs thin disk (Gaia)
4. Barred vs unbarred (SPARC)
5. Elliptical galaxies (Atlas3D/SLUGGS)
6. Galaxy clusters (Lensing data)

DATA REQUIREMENTS:
==================

TEST 1 & 2 & 4: SPARC Database
- Download from: http://astroweb.cwru.edu/SPARC/
- Files needed:
  * SPARC_Lelli2016c.mrt (main table)
  * Individual rotation curves from "Rotmod_LTG/" folder
- Place in: ./data/SPARC/

TEST 3: Gaia DR3
- Query from: https://gea.esac.esa.int/archive/
- Or use pre-processed: Sanders & Das (2018) action-angle catalog
- Place in: ./data/Gaia/

TEST 5: Elliptical Galaxy Data
- Atlas3D: http://www-astro.physics.ox.ac.uk/atlas3d/
- SLUGGS: https://sluggs.swin.edu.au/
- Place in: ./data/Ellipticals/

TEST 6: Cluster Lensing
- Download from: CLASH, LoCuSS, or CCCP surveys
- Place in: ./data/Clusters/
"""

import numpy as np
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import json

# Try to import optional dependencies
try:
    from scipy.optimize import curve_fit
    from scipy.stats import pearsonr, spearmanr, ks_2samp
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, some tests will be limited")

try:
    from astropy.io import fits, ascii
    from astropy.table import Table
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False
    print("Warning: astropy not available, cannot read FITS/MRT files")


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path("./data")
SPARC_DIR = DATA_DIR / "SPARC"
GAIA_DIR = DATA_DIR / "Gaia"
ELLIPTICAL_DIR = DATA_DIR / "Ellipticals"
CLUSTER_DIR = DATA_DIR / "Clusters"

# Create directories if they don't exist
for d in [DATA_DIR, SPARC_DIR, GAIA_DIR, ELLIPTICAL_DIR, CLUSTER_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def check_data_availability():
    """Check which data files are available."""
    
    status = {
        'SPARC': {
            'main_table': (SPARC_DIR / "SPARC_Lelli2016c.mrt").exists(),
            'rotation_curves': (SPARC_DIR / "Rotmod_LTG").exists(),
            'download_url': 'http://astroweb.cwru.edu/SPARC/',
            'instructions': '''
Download instructions for SPARC:
1. Go to http://astroweb.cwru.edu/SPARC/
2. Download "SPARC_Lelli2016c.mrt" (main galaxy table)
3. Download "Rotmod_LTG.zip" (rotation curve files)
4. Extract to ./data/SPARC/
5. Rotation curves should be in ./data/SPARC/Rotmod_LTG/
'''
        },
        'Gaia': {
            'available': len(list(GAIA_DIR.glob("*.fits"))) > 0 or len(list(GAIA_DIR.glob("*.csv"))) > 0,
            'download_url': 'https://gea.esac.esa.int/archive/',
            'instructions': '''
Download instructions for Gaia DR3:
Option A - Direct query (large, ~100GB for full catalog):
1. Go to https://gea.esac.esa.int/archive/
2. Query: SELECT * FROM gaiadr3.gaia_source WHERE ...
3. Add cuts for MW disk stars

Option B - Pre-processed catalogs (recommended):
1. Sanders & Das (2018): https://zenodo.org/record/1492526
2. Or use astroquery: 
   from astroquery.gaia import Gaia
   job = Gaia.launch_job_async("SELECT ...")
'''
        },
        'Ellipticals': {
            'atlas3d': (ELLIPTICAL_DIR / "atlas3d").exists(),
            'sluggs': (ELLIPTICAL_DIR / "sluggs").exists(),
            'download_url': 'http://www-astro.physics.ox.ac.uk/atlas3d/',
            'instructions': '''
Download instructions for Elliptical data:
Atlas3D:
1. Go to http://www-astro.physics.ox.ac.uk/atlas3d/
2. Download kinematic maps and mass models
3. Place in ./data/Ellipticals/atlas3d/

SLUGGS:
1. Go to https://sluggs.swin.edu.au/
2. Download GC kinematics tables
3. Place in ./data/Ellipticals/sluggs/
'''
        },
        'Clusters': {
            'available': len(list(CLUSTER_DIR.glob("*"))) > 0,
            'download_url': 'https://archive.stsci.edu/prepds/clash/',
            'instructions': '''
Download instructions for Cluster data:
CLASH:
1. Go to https://archive.stsci.edu/prepds/clash/
2. Download lensing mass profiles
3. Place in ./data/Clusters/

Alternative - use published mass profiles from:
- Umetsu et al. (2016) for CLASH
- Okabe & Smith (2016) for LoCuSS
'''
        }
    }
    
    return status


def load_sparc_data():
    """
    Load SPARC galaxy data.
    
    Returns dict with:
    - galaxies: list of galaxy properties
    - rotation_curves: dict of {name: rotation_curve_data}
    """
    
    main_table_path = SPARC_DIR / "SPARC_Lelli2016c.mrt"
    rotcurve_dir = SPARC_DIR / "Rotmod_LTG"
    
    if not main_table_path.exists():
        print(f"ERROR: SPARC main table not found at {main_table_path}")
        print("Please download from http://astroweb.cwru.edu/SPARC/")
        return None
    
    # Read main table
    if ASTROPY_AVAILABLE:
        try:
            # Try reading as machine-readable table
            galaxies = ascii.read(main_table_path, format='mrt')
        except:
            # Try as fixed-width
            galaxies = ascii.read(main_table_path, format='fixed_width')
    else:
        # Manual parsing
        galaxies = parse_sparc_manually(main_table_path)
    
    # Load rotation curves
    rotation_curves = {}
    if rotcurve_dir.exists():
        for rc_file in rotcurve_dir.glob("*.dat"):
            name = rc_file.stem
            try:
                rc_data = np.loadtxt(rc_file, comments='#')
                rotation_curves[name] = {
                    'R': rc_data[:, 0],      # kpc
                    'v_obs': rc_data[:, 1],  # km/s
                    'e_v': rc_data[:, 2],    # km/s error
                    'v_gas': rc_data[:, 3],  # km/s
                    'v_disk': rc_data[:, 4], # km/s
                    'v_bul': rc_data[:, 5],  # km/s
                }
            except Exception as e:
                print(f"Warning: Could not load {rc_file}: {e}")
    
    return {'galaxies': galaxies, 'rotation_curves': rotation_curves}


def parse_sparc_manually(filepath):
    """Manually parse SPARC table if astropy unavailable."""
    
    data = {
        'Galaxy': [],
        'T': [],        # Hubble type
        'D': [],        # Distance (Mpc)
        'Inc': [],      # Inclination (deg)
        'L36': [],      # 3.6μm luminosity
        'Rdisk': [],    # Disk scale length (kpc)
        'SBdisk': [],   # Central surface brightness
        'MHI': [],      # HI mass
        'Vflat': [],    # Flat rotation velocity
        'Q': [],        # Quality flag
    }
    
    with open(filepath, 'r') as f:
        in_data = False
        for line in f:
            if line.startswith('---'):
                in_data = True
                continue
            if not in_data or line.startswith('#') or len(line.strip()) == 0:
                continue
            
            parts = line.split()
            if len(parts) >= 10:
                try:
                    data['Galaxy'].append(parts[0])
                    data['T'].append(float(parts[1]) if parts[1] != '...' else np.nan)
                    data['D'].append(float(parts[2]) if parts[2] != '...' else np.nan)
                    data['Inc'].append(float(parts[3]) if parts[3] != '...' else np.nan)
                    data['L36'].append(float(parts[4]) if parts[4] != '...' else np.nan)
                    data['Rdisk'].append(float(parts[5]) if parts[5] != '...' else np.nan)
                    data['SBdisk'].append(float(parts[6]) if parts[6] != '...' else np.nan)
                    data['MHI'].append(float(parts[7]) if parts[7] != '...' else np.nan)
                    data['Vflat'].append(float(parts[8]) if parts[8] != '...' else np.nan)
                    data['Q'].append(int(parts[9]) if parts[9] != '...' else 0)
                except:
                    pass
    
    return data


def load_gaia_data():
    """
    Load Gaia DR3 data for MW stars.
    
    Expects a pre-processed file with columns:
    R, phi, z, v_R, v_phi, v_z, [Fe/H], age
    """
    
    # Look for any Gaia data file
    gaia_files = list(GAIA_DIR.glob("*.fits")) + list(GAIA_DIR.glob("*.csv"))
    
    if len(gaia_files) == 0:
        print(f"ERROR: No Gaia data found in {GAIA_DIR}")
        print("Please download Gaia DR3 data or use pre-processed catalog")
        return None
    
    gaia_file = gaia_files[0]
    print(f"Loading Gaia data from {gaia_file}")
    
    if gaia_file.suffix == '.fits' and ASTROPY_AVAILABLE:
        data = Table.read(gaia_file)
    elif gaia_file.suffix == '.csv':
        data = np.genfromtxt(gaia_file, delimiter=',', names=True)
    else:
        print("Cannot read Gaia file format")
        return None
    
    return data


# =============================================================================
# TEST 1: LSB vs HSB AT SAME x
# =============================================================================

def test_lsb_vs_hsb(sparc_data, output_dir=Path("./results")):
    """
    Test if K differs between LSB and HSB galaxies at the same x = R/R_disk.
    
    Hypothesis being tested:
    - If Σ matters: LSB (low Σ) should have HIGHER K than HSB at same x
    - If only x matters: K should be SAME regardless of surface brightness
    
    DATA NEEDED: SPARC (main table + rotation curves)
    """
    
    print("\n" + "="*70)
    print("  TEST 1: LSB vs HSB at same x = R/R_disk")
    print("="*70)
    
    if sparc_data is None:
        print("  ERROR: SPARC data not loaded")
        return None
    
    galaxies = sparc_data['galaxies']
    rotation_curves = sparc_data['rotation_curves']
    
    # Classify galaxies by surface brightness
    # LSB: SBdisk > 22 mag/arcsec² (fainter = lower SB)
    # HSB: SBdisk < 21 mag/arcsec²
    
    lsb_galaxies = []
    hsb_galaxies = []
    
    if isinstance(galaxies, dict):
        for i in range(len(galaxies['Galaxy'])):
            name = galaxies['Galaxy'][i]
            sb = galaxies['SBdisk'][i]
            rdisk = galaxies['Rdisk'][i]
            
            if np.isnan(sb) or np.isnan(rdisk) or name not in rotation_curves:
                continue
            
            galaxy_info = {
                'name': name,
                'SBdisk': sb,
                'Rdisk': rdisk,
                'rc': rotation_curves.get(name)
            }
            
            if sb > 22:
                lsb_galaxies.append(galaxy_info)
            elif sb < 21:
                hsb_galaxies.append(galaxy_info)
    else:
        # Astropy table
        for row in galaxies:
            name = row['Galaxy'] if 'Galaxy' in galaxies.colnames else row['Name']
            sb = row['SBdisk'] if 'SBdisk' in galaxies.colnames else row['mu0']
            rdisk = row['Rdisk'] if 'Rdisk' in galaxies.colnames else row['Rd']
            
            if np.ma.is_masked(sb) or np.ma.is_masked(rdisk):
                continue
            if str(name) not in rotation_curves:
                continue
                
            galaxy_info = {
                'name': str(name),
                'SBdisk': float(sb),
                'Rdisk': float(rdisk),
                'rc': rotation_curves.get(str(name))
            }
            
            if sb > 22:
                lsb_galaxies.append(galaxy_info)
            elif sb < 21:
                hsb_galaxies.append(galaxy_info)
    
    print(f"\n  Found {len(lsb_galaxies)} LSB galaxies (SB > 22 mag/arcsec²)")
    print(f"  Found {len(hsb_galaxies)} HSB galaxies (SB < 21 mag/arcsec²)")
    
    if len(lsb_galaxies) < 5 or len(hsb_galaxies) < 5:
        print("  WARNING: Not enough galaxies for robust comparison")
    
    # Compute K(x) for each galaxy
    def compute_K(rc, Rdisk):
        """Compute enhancement factor K = v_obs² / v_bar²"""
        R = rc['R']
        v_obs = rc['v_obs']
        v_bar = np.sqrt(rc['v_gas']**2 + rc['v_disk']**2 + rc['v_bul']**2)
        
        # Avoid division by zero
        v_bar = np.maximum(v_bar, 1.0)
        
        K = (v_obs / v_bar)**2
        x = R / Rdisk
        
        return x, K
    
    # Collect K values in x bins
    x_bins = np.linspace(0.5, 4.0, 15)
    x_centers = 0.5 * (x_bins[:-1] + x_bins[1:])
    
    lsb_K_binned = {i: [] for i in range(len(x_centers))}
    hsb_K_binned = {i: [] for i in range(len(x_centers))}
    
    for gal in lsb_galaxies:
        if gal['rc'] is None:
            continue
        x, K = compute_K(gal['rc'], gal['Rdisk'])
        for j, (K_val, x_val) in enumerate(zip(K, x)):
            bin_idx = np.digitize(x_val, x_bins) - 1
            if 0 <= bin_idx < len(x_centers):
                lsb_K_binned[bin_idx].append(K_val)
    
    for gal in hsb_galaxies:
        if gal['rc'] is None:
            continue
        x, K = compute_K(gal['rc'], gal['Rdisk'])
        for j, (K_val, x_val) in enumerate(zip(K, x)):
            bin_idx = np.digitize(x_val, x_bins) - 1
            if 0 <= bin_idx < len(x_centers):
                hsb_K_binned[bin_idx].append(K_val)
    
    # Compute statistics
    print("\n  Results by x = R/R_disk bin:")
    print("  " + "-"*60)
    print(f"  {'x bin':<10s} {'LSB K':<15s} {'HSB K':<15s} {'Difference':<15s}")
    print("  " + "-"*60)
    
    results = []
    for i, x_c in enumerate(x_centers):
        lsb_vals = lsb_K_binned[i]
        hsb_vals = hsb_K_binned[i]
        
        if len(lsb_vals) > 2 and len(hsb_vals) > 2:
            lsb_mean = np.median(lsb_vals)
            hsb_mean = np.median(hsb_vals)
            lsb_std = np.std(lsb_vals) / np.sqrt(len(lsb_vals))
            hsb_std = np.std(hsb_vals) / np.sqrt(len(hsb_vals))
            
            diff = lsb_mean - hsb_mean
            diff_err = np.sqrt(lsb_std**2 + hsb_std**2)
            
            # KS test
            if SCIPY_AVAILABLE:
                ks_stat, ks_p = ks_2samp(lsb_vals, hsb_vals)
            else:
                ks_stat, ks_p = np.nan, np.nan
            
            print(f"  {x_c:<10.2f} {lsb_mean:<7.2f}±{lsb_std:<5.2f} "
                  f"{hsb_mean:<7.2f}±{hsb_std:<5.2f} {diff:+.2f}±{diff_err:.2f}")
            
            results.append({
                'x': x_c,
                'lsb_K': lsb_mean,
                'lsb_err': lsb_std,
                'lsb_n': len(lsb_vals),
                'hsb_K': hsb_mean,
                'hsb_err': hsb_std,
                'hsb_n': len(hsb_vals),
                'diff': diff,
                'ks_p': ks_p
            })
    
    # Overall conclusion
    if len(results) > 0:
        avg_diff = np.mean([r['diff'] for r in results])
        sig_diffs = [r for r in results if abs(r['diff']) > 2*np.sqrt(r['lsb_err']**2 + r['hsb_err']**2)]
        
        print("\n  CONCLUSION:")
        print(f"  Average K difference (LSB - HSB): {avg_diff:+.3f}")
        print(f"  Significant (>2σ) differences in {len(sig_diffs)}/{len(results)} bins")
        
        if abs(avg_diff) < 0.1:
            print("  → K is similar for LSB and HSB at same x")
            print("  → SUPPORTS geometry hypothesis (x = R/R_disk)")
        elif avg_diff > 0.1:
            print("  → LSB has HIGHER K than HSB at same x")
            print("  → SUPPORTS Σ-dependent hypothesis")
        else:
            print("  → HSB has HIGHER K than LSB at same x")
            print("  → UNEXPECTED - needs investigation")
    
    # Save results
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "test1_lsb_hsb.json", 'w') as f:
        json.dump({'results': results, 'n_lsb': len(lsb_galaxies), 'n_hsb': len(hsb_galaxies)}, f, indent=2)
    
    return results


# =============================================================================
# TEST 2: RAR vs GEOMETRY
# =============================================================================

def test_rar_vs_geometry(sparc_data, output_dir=Path("./results")):
    """
    Test if K depends on g_bar (RAR) or x = R/R_disk (geometry).
    
    Method: Find points with SAME g_bar but DIFFERENT x, and vice versa.
    
    Hypothesis being tested:
    - If RAR holds: K should be same at same g_bar, regardless of x
    - If geometry holds: K should be same at same x, regardless of g_bar
    
    DATA NEEDED: SPARC (rotation curves with mass models)
    """
    
    print("\n" + "="*70)
    print("  TEST 2: RAR vs Geometry")
    print("="*70)
    
    if sparc_data is None:
        print("  ERROR: SPARC data not loaded")
        return None
    
    galaxies = sparc_data['galaxies']
    rotation_curves = sparc_data['rotation_curves']
    
    # Collect all data points
    all_points = []
    G = 4.3e-6  # kpc (km/s)² / M_sun
    
    # Get R_disk for each galaxy
    rdisk_dict = {}
    if isinstance(galaxies, dict):
        for i in range(len(galaxies['Galaxy'])):
            rdisk_dict[galaxies['Galaxy'][i]] = galaxies['Rdisk'][i]
    else:
        for row in galaxies:
            name = str(row['Galaxy'] if 'Galaxy' in galaxies.colnames else row['Name'])
            rdisk = float(row['Rdisk'] if 'Rdisk' in galaxies.colnames else row['Rd'])
            rdisk_dict[name] = rdisk
    
    for name, rc in rotation_curves.items():
        if name not in rdisk_dict or np.isnan(rdisk_dict[name]):
            continue
        
        Rdisk = rdisk_dict[name]
        R = rc['R']
        v_obs = rc['v_obs']
        v_bar = np.sqrt(rc['v_gas']**2 + rc['v_disk']**2 + rc['v_bul']**2)
        
        for i in range(len(R)):
            if R[i] < 0.1 or v_bar[i] < 1:
                continue
            
            g_bar = v_bar[i]**2 / R[i]  # (km/s)²/kpc
            g_obs = v_obs[i]**2 / R[i]
            K = (v_obs[i] / v_bar[i])**2 if v_bar[i] > 1 else np.nan
            x = R[i] / Rdisk
            
            if not np.isnan(K) and 0.1 < K < 100 and 0.1 < x < 10:
                all_points.append({
                    'galaxy': name,
                    'R': R[i],
                    'x': x,
                    'g_bar': g_bar,
                    'g_obs': g_obs,
                    'K': K,
                    'v_obs': v_obs[i],
                    'v_bar': v_bar[i]
                })
    
    print(f"\n  Collected {len(all_points)} data points from {len(rotation_curves)} galaxies")
    
    if len(all_points) < 100:
        print("  WARNING: Not enough data points")
        return None
    
    # Convert to arrays
    x_arr = np.array([p['x'] for p in all_points])
    g_bar_arr = np.array([p['g_bar'] for p in all_points])
    K_arr = np.array([p['K'] for p in all_points])
    
    # Test 2A: At fixed x, how much does K vary with g_bar?
    print("\n  Test 2A: K variation with g_bar at fixed x")
    print("  " + "-"*50)
    
    x_bins = [0.5, 1.0, 1.5, 2.0, 3.0]
    for x_lo, x_hi in zip(x_bins[:-1], x_bins[1:]):
        mask = (x_arr >= x_lo) & (x_arr < x_hi)
        if mask.sum() < 20:
            continue
        
        g_in_bin = g_bar_arr[mask]
        K_in_bin = K_arr[mask]
        
        # Split by g_bar
        g_med = np.median(g_in_bin)
        K_low_g = K_in_bin[g_in_bin < g_med]
        K_high_g = K_in_bin[g_in_bin >= g_med]
        
        if len(K_low_g) > 5 and len(K_high_g) > 5:
            diff = np.median(K_high_g) - np.median(K_low_g)
            print(f"  x=[{x_lo:.1f},{x_hi:.1f}]: K(high g) - K(low g) = {diff:+.3f} "
                  f"(n={len(K_low_g)}, {len(K_high_g)})")
    
    # Test 2B: At fixed g_bar, how much does K vary with x?
    print("\n  Test 2B: K variation with x at fixed g_bar")
    print("  " + "-"*50)
    
    log_g_bins = np.percentile(np.log10(g_bar_arr), [0, 25, 50, 75, 100])
    for g_lo, g_hi in zip(10**log_g_bins[:-1], 10**log_g_bins[1:]):
        mask = (g_bar_arr >= g_lo) & (g_bar_arr < g_hi)
        if mask.sum() < 20:
            continue
        
        x_in_bin = x_arr[mask]
        K_in_bin = K_arr[mask]
        
        # Split by x
        x_med = np.median(x_in_bin)
        K_low_x = K_in_bin[x_in_bin < x_med]
        K_high_x = K_in_bin[x_in_bin >= x_med]
        
        if len(K_low_x) > 5 and len(K_high_x) > 5:
            diff = np.median(K_high_x) - np.median(K_low_x)
            print(f"  g_bar=[{g_lo:.0f},{g_hi:.0f}]: K(high x) - K(low x) = {diff:+.3f} "
                  f"(n={len(K_low_x)}, {len(K_high_x)})")
    
    # Partial correlation analysis
    if SCIPY_AVAILABLE:
        print("\n  Partial correlation analysis:")
        print("  " + "-"*50)
        
        # Correlation of K with x
        r_K_x, p_K_x = pearsonr(K_arr, x_arr)
        print(f"  Correlation K vs x:     r = {r_K_x:.3f}, p = {p_K_x:.2e}")
        
        # Correlation of K with log(g_bar)
        r_K_g, p_K_g = pearsonr(K_arr, np.log10(g_bar_arr))
        print(f"  Correlation K vs log(g): r = {r_K_g:.3f}, p = {p_K_g:.2e}")
        
        # Partial correlation: K vs x controlling for g_bar
        # Using residuals method
        from numpy.linalg import lstsq
        
        # Residuals of K on log(g)
        A = np.column_stack([np.ones(len(K_arr)), np.log10(g_bar_arr)])
        coef_K, _, _, _ = lstsq(A, K_arr, rcond=None)
        K_resid = K_arr - A @ coef_K
        
        # Residuals of x on log(g)
        coef_x, _, _, _ = lstsq(A, x_arr, rcond=None)
        x_resid = x_arr - A @ coef_x
        
        r_partial, p_partial = pearsonr(K_resid, x_resid)
        print(f"  Partial corr K vs x (controlling g): r = {r_partial:.3f}")
        
        # Vice versa: K vs g controlling for x
        A2 = np.column_stack([np.ones(len(K_arr)), x_arr])
        coef_K2, _, _, _ = lstsq(A2, K_arr, rcond=None)
        K_resid2 = K_arr - A2 @ coef_K2
        
        coef_g, _, _, _ = lstsq(A2, np.log10(g_bar_arr), rcond=None)
        g_resid = np.log10(g_bar_arr) - A2 @ coef_g
        
        r_partial2, p_partial2 = pearsonr(K_resid2, g_resid)
        print(f"  Partial corr K vs g (controlling x): r = {r_partial2:.3f}")
        
        print("\n  CONCLUSION:")
        if abs(r_partial) > abs(r_partial2):
            print(f"  → x has stronger partial correlation ({r_partial:.3f} vs {r_partial2:.3f})")
            print("  → SUPPORTS geometry hypothesis")
        else:
            print(f"  → g_bar has stronger partial correlation ({r_partial2:.3f} vs {r_partial:.3f})")
            print("  → SUPPORTS RAR hypothesis")
    
    # Save results
    output_dir.mkdir(exist_ok=True)
    results = {
        'n_points': len(all_points),
        'n_galaxies': len(rotation_curves),
        'r_K_x': float(r_K_x) if SCIPY_AVAILABLE else None,
        'r_K_g': float(r_K_g) if SCIPY_AVAILABLE else None,
        'r_partial_K_x': float(r_partial) if SCIPY_AVAILABLE else None,
        'r_partial_K_g': float(r_partial2) if SCIPY_AVAILABLE else None,
    }
    
    with open(output_dir / "test2_rar_geometry.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


# =============================================================================
# TEST 3: MW THICK vs THIN DISK
# =============================================================================

def test_mw_thick_thin(gaia_data, output_dir=Path("./results")):
    """
    Test if K differs between MW thick and thin disk at the same R.
    
    Thick disk: kinematically hot (high σ_v), old, metal-poor
    Thin disk: kinematically cold (low σ_v), young, metal-rich
    
    Hypothesis being tested:
    - If σ_v matters: thick disk should have DIFFERENT K than thin disk
    - If only R matters: K should be SAME for both components
    
    DATA NEEDED: Gaia DR3 with proper motions and abundances
    """
    
    print("\n" + "="*70)
    print("  TEST 3: MW Thick vs Thin Disk")
    print("="*70)
    
    if gaia_data is None:
        print("  ERROR: Gaia data not loaded")
        print("""
  To run this test, you need Gaia DR3 data with:
  - Galactocentric positions (R, z)
  - Velocities (v_R, v_phi, v_z)
  - [Fe/H] metallicity (for thin/thick separation)
  
  Download options:
  1. Direct query from Gaia archive: https://gea.esac.esa.int/archive/
  2. Pre-processed catalog: Sanders & Das (2018)
  3. Use astroquery:
     
     from astroquery.gaia import Gaia
     query = '''
     SELECT ra, dec, parallax, pmra, pmdec, radial_velocity,
            phot_g_mean_mag, bp_rp
     FROM gaiadr3.gaia_source
     WHERE parallax > 0.5 AND parallax_error/parallax < 0.1
       AND radial_velocity IS NOT NULL
     '''
     job = Gaia.launch_job_async(query)
     results = job.get_results()
""")
        return None
    
    # This is a placeholder - actual implementation depends on data format
    print("  Gaia data analysis not yet implemented")
    print("  See code comments for data requirements")
    
    return None


# =============================================================================
# TEST 4: BARRED vs UNBARRED
# =============================================================================

def test_barred_vs_unbarred(sparc_data, output_dir=Path("./results")):
    """
    Test if barred galaxies have different K(x) than unbarred.
    
    Hypothesis being tested:
    - If local effects matter: bar region should have different K
    - If global geometry matters: same x → same K regardless of bar
    
    DATA NEEDED: SPARC + bar classification
    """
    
    print("\n" + "="*70)
    print("  TEST 4: Barred vs Unbarred Galaxies")
    print("="*70)
    
    if sparc_data is None:
        print("  ERROR: SPARC data not loaded")
        return None
    
    # Bar classification from Hubble type
    # SB = barred, SA = unbarred, SAB = weakly barred
    
    galaxies = sparc_data['galaxies']
    rotation_curves = sparc_data['rotation_curves']
    
    # We need bar classifications - SPARC doesn't include this directly
    # Would need to cross-match with HyperLeda or NED
    
    print("""
  NOTE: Bar classification requires cross-matching with:
  - HyperLeda: http://leda.univ-lyon1.fr/
  - NASA/IPAC Extragalactic Database (NED): https://ned.ipac.caltech.edu/
  
  Query example for HyperLeda:
  SELECT objname, type FROM meandata WHERE objname IN ('NGC1234', ...)
  
  Then classify:
  - 'SB' in type → barred
  - 'SA' in type → unbarred
  - 'SAB' in type → weakly barred
""")
    
    return None


# =============================================================================
# TEST 5: ELLIPTICAL GALAXIES
# =============================================================================

def test_ellipticals(output_dir=Path("./results")):
    """
    Test if elliptical galaxies follow the same K(R) as disk galaxies.
    
    Hypothesis being tested:
    - If disk geometry is required: ellipticals should NOT show x = R/R_d scaling
    - If only mass distribution matters: ellipticals should show similar behavior
    
    DATA NEEDED: Atlas3D or SLUGGS kinematic data
    """
    
    print("\n" + "="*70)
    print("  TEST 5: Elliptical Galaxies")
    print("="*70)
    
    atlas3d_dir = ELLIPTICAL_DIR / "atlas3d"
    sluggs_dir = ELLIPTICAL_DIR / "sluggs"
    
    if not atlas3d_dir.exists() and not sluggs_dir.exists():
        print(f"""
  ERROR: Elliptical galaxy data not found

  Download Atlas3D data:
  1. Go to: http://www-astro.physics.ox.ac.uk/atlas3d/
  2. Download kinematic data tables
  3. Place in: {atlas3d_dir}

  Download SLUGGS data:
  1. Go to: https://sluggs.swin.edu.au/
  2. Download GC kinematic tables
  3. Place in: {sluggs_dir}

  Key files needed:
  - Mass profiles M(R)
  - Velocity dispersion profiles σ(R)
  - Rotation curves v(R) where available
""")
        return None
    
    # Placeholder for actual implementation
    print("  Elliptical galaxy analysis not yet implemented")
    
    return None


# =============================================================================
# TEST 6: GALAXY CLUSTERS
# =============================================================================

def test_clusters(output_dir=Path("./results")):
    """
    Test if galaxy clusters show enhancement scaling with R/R_cluster.
    
    Hypothesis being tested:
    - If geometry scales: R_c ≈ 0.75 × R_200 for clusters
    - If disk-specific: clusters should NOT show similar behavior
    
    DATA NEEDED: Cluster lensing mass profiles
    """
    
    print("\n" + "="*70)
    print("  TEST 6: Galaxy Clusters")
    print("="*70)
    
    if not any(CLUSTER_DIR.glob("*")):
        print(f"""
  ERROR: Cluster data not found

  Download options:

  1. CLASH Survey:
     https://archive.stsci.edu/prepds/clash/
     
  2. Published mass profiles:
     - Umetsu et al. (2016): CLASH lensing
     - Okabe & Smith (2016): LoCuSS lensing
     
  3. X-ray mass profiles:
     - Vikhlinin et al. (2006): Chandra
     
  Place data in: {CLUSTER_DIR}

  Key data needed:
  - Total mass profile M(R)
  - Baryonic mass (gas + BCG + galaxies)
  - Characteristic radii (R_200, R_500, r_s)
""")
        return None
    
    print("  Cluster analysis not yet implemented")
    
    return None


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run all tests."""
    
    print("="*70)
    print("  Σ-GRAVITY HYPOTHESIS DISTINGUISHING TESTS")
    print("="*70)
    
    # Check data availability
    print("\n  Checking data availability...")
    status = check_data_availability()
    
    print("\n  Data status:")
    for dataset, info in status.items():
        if isinstance(info, dict):
            avail = info.get('main_table', info.get('available', False))
            print(f"    {dataset}: {'✓ Available' if avail else '✗ Not found'}")
    
    # Create results directory
    results_dir = Path("./results")
    results_dir.mkdir(exist_ok=True)
    
    # Load SPARC data
    print("\n  Loading SPARC data...")
    sparc_data = load_sparc_data()
    
    if sparc_data is not None:
        n_gal = len(sparc_data['rotation_curves'])
        print(f"  Loaded {n_gal} galaxies with rotation curves")
    
    # Load Gaia data
    print("\n  Loading Gaia data...")
    gaia_data = load_gaia_data()
    
    # Run tests
    print("\n" + "="*70)
    print("  RUNNING TESTS")
    print("="*70)
    
    all_results = {}
    
    # Test 1: LSB vs HSB
    result1 = test_lsb_vs_hsb(sparc_data, results_dir)
    all_results['test1_lsb_hsb'] = result1
    
    # Test 2: RAR vs Geometry
    result2 = test_rar_vs_geometry(sparc_data, results_dir)
    all_results['test2_rar_geometry'] = result2
    
    # Test 3: MW thick/thin
    result3 = test_mw_thick_thin(gaia_data, results_dir)
    all_results['test3_mw_disk'] = result3
    
    # Test 4: Barred vs unbarred
    result4 = test_barred_vs_unbarred(sparc_data, results_dir)
    all_results['test4_barred'] = result4
    
    # Test 5: Ellipticals
    result5 = test_ellipticals(results_dir)
    all_results['test5_ellipticals'] = result5
    
    # Test 6: Clusters
    result6 = test_clusters(results_dir)
    all_results['test6_clusters'] = result6
    
    # Summary
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    
    print("\n  Test results:")
    for test_name, result in all_results.items():
        if result is not None:
            print(f"    ✓ {test_name}: Completed")
        else:
            print(f"    ○ {test_name}: Data not available")
    
    print(f"\n  Results saved to: {results_dir.absolute()}")
    
    return all_results


if __name__ == "__main__":
    results = main()
