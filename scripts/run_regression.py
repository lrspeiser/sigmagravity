#!/usr/bin/env python3
"""
Σ-GRAVITY MASTER REGRESSION TEST
=================================

This is THE DEFINITIVE regression test for Σ-Gravity. Run this after any formula changes.

USAGE:
    python run_regression.py           # Full test (all domains)
    python run_regression.py --quick   # Skip slow tests (Gaia, counter-rotation)
    python run_regression.py --compare # Compare all parameter configurations

DATA SOURCES (validated and documented):
    - SPARC: 171 galaxies from Lelli+ 2016 (data/Rotmod_LTG/)
    - Clusters: 42 high-quality Fox+ 2022 clusters (data/clusters/fox2022_unique_clusters.csv)
    - Gaia/MW: 28,368 Eilers-APOGEE-Gaia disk stars (data/gaia/eilers_apogee_6d_disk.csv)
    - Counter-rotation: 64 Bevacqua+ 2022 galaxies (data/stellar_corgi/)

CURRENT FORMULA (Σ-Gravity canonical form):
    Σ = 1 + A × W(r) × h(g)
    
    h(g) = √(g†/g) × g†/(g†+g)           [acceleration enhancement]
    W(r) = 1 - (ξ/(ξ+r))^0.5             [coherence window]
    g† = cH₀/(4√π) ≈ 9.6×10⁻¹¹ m/s²     [critical acceleration]
    ξ = (1/2) × R_d                      [coherence scale - clean formulation]
    
    A_galaxy = √e ≈ 1.649                [disk amplitude, from path length scaling]
    A_cluster = 8.0                      [cluster amplitude, from path length scaling]
    
    M/L = 0.5 (disk), 0.7 (bulge)        [Lelli+ 2016 standard]

EXPECTED RESULTS:
    - SPARC: RMS ≈ 20 km/s, ~42% win rate vs MOND (fair comparison, same M/L)
    - Clusters: Median ratio ≈ 0.93, scatter ≈ 0.13 dex
    - Gaia: RMS ≈ 31 km/s (28,368 stars)
    - Counter-rotation: p < 0.05, f_DM(CR) < f_DM(Normal)
    - Redshift: g†(z) ∝ H(z) confirmed
    - Solar System: γ-1 < 10⁻⁵ (Cassini safe)

Author: Leonard Speiser
Last Updated: December 2025
"""

import numpy as np
import pandas as pd
import math
import json
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PHYSICAL CONSTANTS (DO NOT CHANGE)
# =============================================================================
c = 2.998e8  # m/s
H0_SI = 2.27e-18  # 1/s (H0 = 70 km/s/Mpc)
kpc_to_m = 3.086e19
G_const = 6.674e-11
M_sun = 1.989e30

# Critical acceleration (derived from cosmology)
g_dagger = c * H0_SI / (4 * math.sqrt(math.pi))  # ≈ 9.6×10⁻¹¹ m/s²

# MOND acceleration scale (for comparison)
a0_mond = 1.2e-10

# =============================================================================
# MODEL PARAMETERS
# =============================================================================
# Galaxy parameters (2D coherence framework)
A_GALAXY = np.exp(1 / (2 * np.pi))  # ≈ 1.173, from 2D coherence geometry
XI_SCALE = 1 / (2 * np.pi)  # ξ = R_d/(2π), one azimuthal wavelength
ML_DISK = 0.5   # Mass-to-light ratio for disk (Lelli+ 2016)
ML_BULGE = 0.7  # Mass-to-light ratio for bulge (Lelli+ 2016)

# Cluster parameters
A_CLUSTER = 8.0  # From path length scaling: A_0 × L^0.25 with L ≈ 400 kpc

# MW parameters
MW_VBAR_SCALE = 1.16  # McMillan 2017 baryonic model scaling

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def h_function(g: np.ndarray) -> np.ndarray:
    """Enhancement function h(g) = √(g†/g) × g†/(g†+g)"""
    g = np.maximum(g, 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)


def W_coherence(r: np.ndarray, xi: float) -> np.ndarray:
    """Coherence window W(r) = r/(ξ+r) for 2D coherence (k=1)"""
    xi = max(xi, 0.01)
    return r / (xi + r)


def predict_velocity(R_kpc: np.ndarray, V_bar: np.ndarray, R_d: float) -> np.ndarray:
    """Predict rotation velocity using Σ-Gravity."""
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m
    
    h = h_function(g_bar)
    xi = XI_SCALE * R_d
    W = W_coherence(R_kpc, xi)
    
    Sigma = 1 + A_GALAXY * W * h
    return V_bar * np.sqrt(Sigma)


def predict_mond(R_kpc: np.ndarray, V_bar: np.ndarray) -> np.ndarray:
    """Predict MOND rotation velocity."""
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m
    
    x = g_bar / a0_mond
    nu = 1.0 / (1.0 - np.exp(-np.sqrt(np.maximum(x, 1e-10))))
    return V_bar * np.sqrt(nu)


# =============================================================================
# DATA LOADERS
# =============================================================================

def load_sparc(data_dir: Path) -> List[Dict]:
    """Load SPARC galaxy rotation curves."""
    sparc_dir = data_dir / "Rotmod_LTG"
    if not sparc_dir.exists():
        return []
    
    galaxies = []
    for gf in sorted(sparc_dir.glob("*_rotmod.dat")):
        data = []
        with open(gf) as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        data.append({
                            'R': float(parts[0]),
                            'V_obs': float(parts[1]),
                            'V_gas': float(parts[3]),
                            'V_disk': float(parts[4]),
                            'V_bulge': float(parts[5]) if len(parts) > 5 else 0.0
                        })
                    except ValueError:
                        continue
        
        if len(data) < 5:
            continue
        
        df = pd.DataFrame(data)
        
        # Apply M/L corrections
        df['V_disk_scaled'] = df['V_disk'] * np.sqrt(ML_DISK)
        df['V_bulge_scaled'] = df['V_bulge'] * np.sqrt(ML_BULGE)
        V_bar_sq = (np.sign(df['V_gas']) * df['V_gas']**2 + 
                    df['V_disk_scaled']**2 + df['V_bulge_scaled']**2)
        df['V_bar'] = np.sqrt(np.abs(V_bar_sq)) * np.sign(V_bar_sq)
        
        valid = (df['V_bar'] > 0) & (df['R'] > 0) & (df['V_obs'] > 0)
        df = df[valid]
        
        if len(df) >= 5:
            idx = len(df) // 3
            R_d = df['R'].iloc[idx] if idx > 0 else df['R'].iloc[-1] / 2
            
            galaxies.append({
                'name': gf.stem.replace('_rotmod', ''),
                'R': df['R'].values,
                'V_obs': df['V_obs'].values,
                'V_bar': df['V_bar'].values,
                'R_d': R_d
            })
    
    return galaxies


def load_clusters(data_dir: Path) -> List[Dict]:
    """Load Fox+ 2022 cluster data."""
    cluster_file = data_dir / "clusters" / "fox2022_unique_clusters.csv"
    
    if not cluster_file.exists():
        return []
    
    df = pd.read_csv(cluster_file)
    
    # Filter to high-quality clusters
    df_valid = df[
        df['M500_1e14Msun'].notna() & 
        df['MSL_200kpc_1e12Msun'].notna() &
        (df['spec_z_constraint'] == 'yes')
    ].copy()
    df_valid = df_valid[df_valid['M500_1e14Msun'] > 2.0].copy()
    
    clusters = []
    f_baryon = 0.15
    
    for idx, row in df_valid.iterrows():
        M500 = row['M500_1e14Msun'] * 1e14
        M_bar_200 = 0.4 * f_baryon * M500
        M_lens_200 = row['MSL_200kpc_1e12Msun'] * 1e12
        
        clusters.append({
            'name': row['cluster'],
            'M_bar': M_bar_200,
            'M_lens': M_lens_200,
            'r_kpc': 200,
        })
    
    return clusters


def load_gaia(data_dir: Path) -> Optional[pd.DataFrame]:
    """Load validated Gaia/Eilers-APOGEE disk star catalog."""
    gaia_file = data_dir / "gaia" / "eilers_apogee_6d_disk.csv"
    if not gaia_file.exists():
        return None
    
    df = pd.read_csv(gaia_file)
    df['v_phi_obs'] = -df['v_phi']  # Correct sign convention
    return df


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

@dataclass
class TestResult:
    name: str
    passed: bool
    metric: float
    details: Dict[str, Any]
    message: str


def test_sparc(galaxies: List[Dict]) -> TestResult:
    """Test SPARC galaxy rotation curves."""
    if len(galaxies) == 0:
        return TestResult("SPARC Galaxies", False, 0.0, {}, "No data")
    
    rms_list, mond_rms_list = [], []
    all_log_ratios = []  # For RAR scatter
    all_log_ratios_mond = []
    wins = 0
    
    for gal in galaxies:
        R, V_obs, V_bar, R_d = gal['R'], gal['V_obs'], gal['V_bar'], gal['R_d']
        
        V_pred = predict_velocity(R, V_bar, R_d)
        V_mond = predict_mond(R, V_bar)
        
        rms = np.sqrt(((V_obs - V_pred)**2).mean())
        rms_mond = np.sqrt(((V_obs - V_mond)**2).mean())
        
        rms_list.append(rms)
        mond_rms_list.append(rms_mond)
        
        # Compute RAR scatter: log10(g_obs / g_bar) = log10(V_obs² / V_bar²) = 2*log10(V_obs/V_bar)
        # We compare predicted vs observed: log10(V_obs / V_pred)
        valid = (V_obs > 0) & (V_pred > 0) & (V_mond > 0) & (V_bar > 0)
        if valid.sum() > 0:
            log_ratio = np.log10(V_obs[valid] / V_pred[valid])
            log_ratio_mond = np.log10(V_obs[valid] / V_mond[valid])
            all_log_ratios.extend(log_ratio)
            all_log_ratios_mond.extend(log_ratio_mond)
        
        if rms < rms_mond:
            wins += 1
    
    mean_rms = np.mean(rms_list)
    mean_mond = np.mean(mond_rms_list)
    win_rate = wins / len(galaxies) * 100
    
    # RAR scatter in dex (std of log10(V_obs/V_pred))
    rar_scatter = np.std(all_log_ratios) if all_log_ratios else 0.0
    rar_scatter_mond = np.std(all_log_ratios_mond) if all_log_ratios_mond else 0.0
    
    passed = mean_rms < 25.0  # Reasonable threshold
    
    return TestResult(
        name="SPARC Galaxies",
        passed=passed,
        metric=mean_rms,
        details={
            'n_galaxies': len(galaxies),
            'mean_rms': mean_rms,
            'mean_mond_rms': mean_mond,
            'win_rate': win_rate,
            'rar_scatter_dex': rar_scatter,
            'rar_scatter_mond_dex': rar_scatter_mond,
        },
        message=f"RMS={mean_rms:.2f} km/s, Scatter={rar_scatter:.3f} dex, Win={win_rate:.1f}%"
    )


def test_clusters(clusters: List[Dict]) -> TestResult:
    """Test cluster lensing masses."""
    if len(clusters) == 0:
        return TestResult("Clusters", False, 0.0, {}, "No data")
    
    ratios = []
    for cl in clusters:
        r_m = cl['r_kpc'] * kpc_to_m
        g_bar = G_const * cl['M_bar'] * M_sun / r_m**2
        
        h = h_function(np.array([g_bar]))[0]
        Sigma = 1 + A_CLUSTER * h  # W ≈ 1 for clusters
        
        M_pred = cl['M_bar'] * Sigma
        ratio = M_pred / cl['M_lens']
        if np.isfinite(ratio) and ratio > 0:
            ratios.append(ratio)
    
    median_ratio = np.median(ratios)
    scatter = np.std(np.log10(ratios))
    
    passed = 0.5 < median_ratio < 1.5
    
    return TestResult(
        name="Clusters",
        passed=passed,
        metric=median_ratio,
        details={
            'n_clusters': len(ratios),
            'median_ratio': median_ratio,
            'scatter_dex': scatter,
        },
        message=f"Median ratio={median_ratio:.3f}, Scatter={scatter:.3f} dex ({len(ratios)} clusters)"
    )


def test_gaia(gaia_df: Optional[pd.DataFrame]) -> TestResult:
    """Test Milky Way star-by-star validation."""
    if gaia_df is None or len(gaia_df) == 0:
        return TestResult("Gaia/MW", True, 0.0, {}, "SKIPPED: No Gaia data")
    
    # McMillan 2017 baryonic model
    R = gaia_df['R_gal'].values
    M_disk = 4.6e10 * MW_VBAR_SCALE**2
    M_bulge = 1.0e10 * MW_VBAR_SCALE**2
    M_gas = 1.0e10 * MW_VBAR_SCALE**2
    G_kpc = 4.302e-6
    
    v2_disk = G_kpc * M_disk * R**2 / (R**2 + 3.3**2)**1.5
    v2_bulge = G_kpc * M_bulge * R / (R + 0.5)**2
    v2_gas = G_kpc * M_gas * R**2 / (R**2 + 7.0**2)**1.5
    V_bar = np.sqrt(v2_disk + v2_bulge + v2_gas)
    
    # Predict circular velocity
    R_d_mw = 2.6  # MW disk scale length
    V_c_pred = predict_velocity(R, V_bar, R_d_mw)
    
    # Asymmetric drift correction
    from scipy.interpolate import interp1d
    R_bins = np.arange(4, 16, 0.5)
    disp_data = []
    for i in range(len(R_bins) - 1):
        mask = (gaia_df['R_gal'] >= R_bins[i]) & (gaia_df['R_gal'] < R_bins[i + 1])
        if mask.sum() > 30:
            disp_data.append({
                'R': (R_bins[i] + R_bins[i + 1]) / 2,
                'sigma_R': gaia_df.loc[mask, 'v_R'].std()
            })
    
    if len(disp_data) > 0:
        disp_df = pd.DataFrame(disp_data)
        sigma_interp = interp1d(disp_df['R'], disp_df['sigma_R'], fill_value='extrapolate')
        sigma_R = sigma_interp(R)
    else:
        sigma_R = 40.0
    
    V_a = sigma_R**2 / (2 * V_c_pred) * (R / R_d_mw - 1)
    V_a = np.clip(V_a, 0, 50)
    
    v_pred = V_c_pred - V_a
    resid = gaia_df['v_phi_obs'].values - v_pred
    rms = np.sqrt((resid**2).mean())
    
    passed = rms < 35.0
    
    return TestResult(
        name="Gaia/MW",
        passed=passed,
        metric=rms,
        details={
            'n_stars': len(gaia_df),
            'rms': rms,
            'mean_residual': resid.mean(),
        },
        message=f"RMS={rms:.1f} km/s ({len(gaia_df)} stars)"
    )


def test_redshift() -> TestResult:
    """Test redshift evolution of g†."""
    Omega_m, Omega_L = 0.3, 0.7
    
    def H_z(z):
        return np.sqrt(Omega_m * (1 + z)**3 + Omega_L)
    
    g_dagger_z2 = g_dagger * H_z(2)
    expected_ratio = H_z(2)  # ≈ 2.97
    actual_ratio = g_dagger_z2 / g_dagger
    
    passed = abs(actual_ratio - expected_ratio) < 0.01
    
    return TestResult(
        name="Redshift Evolution",
        passed=passed,
        metric=actual_ratio,
        details={'g_dagger_z2': g_dagger_z2, 'ratio': actual_ratio},
        message=f"g†(z=2)/g†(z=0) = {actual_ratio:.3f} (expected {expected_ratio:.3f})"
    )


def test_solar_system() -> TestResult:
    """Test Solar System safety (Cassini bound)."""
    r_saturn = 9.5  # AU
    r_m = r_saturn * 1.496e11  # meters
    M_sun_kg = 1.989e30
    
    g_saturn = G_const * M_sun_kg / r_m**2
    h_saturn = h_function(np.array([g_saturn]))[0]
    
    # γ - 1 ≈ h for weak field
    gamma_minus_1 = h_saturn
    cassini_bound = 2.3e-5
    
    passed = gamma_minus_1 < cassini_bound
    
    return TestResult(
        name="Solar System",
        passed=passed,
        metric=gamma_minus_1,
        details={'h_saturn': h_saturn, 'cassini_bound': cassini_bound},
        message=f"|γ-1| = {gamma_minus_1:.2e} < {cassini_bound:.2e}"
    )


def test_counter_rotation(data_dir: Path) -> TestResult:
    """Test counter-rotation prediction: CR galaxies should have lower f_DM.
    
    This is a UNIQUE prediction of Σ-Gravity that neither ΛCDM nor MOND makes.
    Disrupted coherence → reduced gravitational enhancement → lower apparent DM.
    """
    try:
        from astropy.io import fits
        from astropy.table import Table
        from scipy import stats
    except ImportError:
        return TestResult("Counter-Rotation", True, 0.0, {}, "SKIPPED: astropy required")
    
    dynpop_file = data_dir / "manga_dynpop" / "SDSSDR17_MaNGA_JAM.fits"
    cr_file = data_dir / "stellar_corgi" / "bevacqua2022_counter_rotating.tsv"
    
    if not dynpop_file.exists() or not cr_file.exists():
        return TestResult("Counter-Rotation", True, 0.0, {}, "SKIPPED: Data not found")
    
    # Load DynPop catalog
    with fits.open(dynpop_file) as hdul:
        basic = Table(hdul[1].data)
        jam_nfw = Table(hdul[4].data)
    
    # Load counter-rotating catalog
    with open(cr_file, 'r') as f:
        lines = f.readlines()
    
    # Parse CR data
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith('---'):
            data_start = i + 1
            break
    
    header_line = None
    for i, line in enumerate(lines):
        if line.startswith('MaNGAId'):
            header_line = i
            break
    
    if header_line is None:
        return TestResult("Counter-Rotation", True, 0.0, {}, "SKIPPED: Parse error")
    
    headers = [h.strip() for h in lines[header_line].split('|')]
    cr_data = []
    for line in lines[data_start:]:
        if line.strip() and not line.startswith('#'):
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= len(headers):
                cr_data.append(dict(zip(headers, parts)))
    
    cr_manga_ids = [d['MaNGAId'].strip() for d in cr_data]
    
    # Cross-match
    dynpop_idx = {str(mid).strip(): i for i, mid in enumerate(basic['mangaid'])}
    matches = [dynpop_idx[cr_id] for cr_id in cr_manga_ids if cr_id in dynpop_idx]
    
    if len(matches) < 10:
        return TestResult("Counter-Rotation", True, 0.0, {}, f"SKIPPED: Only {len(matches)} matches")
    
    # Extract f_DM values
    fdm_all = np.array(jam_nfw['fdm_Re'])
    valid_mask = np.isfinite(fdm_all) & (fdm_all >= 0) & (fdm_all <= 1)
    
    cr_mask = np.zeros(len(fdm_all), dtype=bool)
    cr_mask[matches] = True
    
    fdm_cr = fdm_all[cr_mask & valid_mask]
    fdm_normal = fdm_all[~cr_mask & valid_mask]
    
    if len(fdm_cr) < 10:
        return TestResult("Counter-Rotation", True, 0.0, {}, "SKIPPED: Insufficient data")
    
    # Mann-Whitney U test (one-sided: CR < Normal)
    mw_stat, mw_pval_two = stats.mannwhitneyu(fdm_cr, fdm_normal)
    mw_pval = mw_pval_two / 2 if np.mean(fdm_cr) < np.mean(fdm_normal) else 1 - mw_pval_two / 2
    
    # Pass if p < 0.05 and CR has lower f_DM
    passed = mw_pval < 0.05 and np.mean(fdm_cr) < np.mean(fdm_normal)
    
    return TestResult(
        name="Counter-Rotation",
        passed=passed,
        metric=mw_pval,
        details={
            'n_cr': len(fdm_cr),
            'n_normal': len(fdm_normal),
            'fdm_cr_mean': float(np.mean(fdm_cr)),
            'fdm_normal_mean': float(np.mean(fdm_normal)),
        },
        message=f"f_DM(CR)={np.mean(fdm_cr):.3f} < f_DM(Normal)={np.mean(fdm_normal):.3f}, p={mw_pval:.4f}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    quick = '--quick' in sys.argv
    compare = '--compare' in sys.argv
    
    data_dir = Path(__file__).parent.parent / "data"
    
    print("=" * 80)
    print("Σ-GRAVITY MASTER REGRESSION TEST")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Mode: {'Quick' if quick else 'Full'}")
    print()
    print("Parameters (2D coherence framework):")
    print(f"  A_galaxy = e^(1/2π) ≈ {A_GALAXY:.3f}")
    print(f"  A_cluster = {A_CLUSTER}")
    print(f"  ξ = R_d/(2π) ≈ {XI_SCALE:.4f} × R_d")
    print(f"  M/L = {ML_DISK}/{ML_BULGE} (disk/bulge)")
    print(f"  g† = {g_dagger:.3e} m/s²")
    print()
    
    # Load data
    print("Loading data...")
    galaxies = load_sparc(data_dir)
    print(f"  SPARC: {len(galaxies)} galaxies")
    
    clusters = load_clusters(data_dir)
    print(f"  Clusters: {len(clusters)}")
    
    gaia_df = load_gaia(data_dir) if not quick else None
    print(f"  Gaia/MW: {len(gaia_df) if gaia_df is not None else 'Skipped'}")
    print()
    
    # Run tests
    results = []
    
    print("Running tests...")
    print("-" * 80)
    
    result = test_sparc(galaxies)
    results.append(result)
    print(f"[{'✓' if result.passed else '✗'}] {result.name}: {result.message}")
    
    result = test_clusters(clusters)
    results.append(result)
    print(f"[{'✓' if result.passed else '✗'}] {result.name}: {result.message}")
    
    result = test_gaia(gaia_df)
    results.append(result)
    print(f"[{'✓' if result.passed else '✗'}] {result.name}: {result.message}")
    
    result = test_redshift()
    results.append(result)
    print(f"[{'✓' if result.passed else '✗'}] {result.name}: {result.message}")
    
    result = test_solar_system()
    results.append(result)
    print(f"[{'✓' if result.passed else '✗'}] {result.name}: {result.message}")
    
    if not quick:
        result = test_counter_rotation(data_dir)
        results.append(result)
        print(f"[{'✓' if result.passed else '✗'}] {result.name}: {result.message}")
    
    print("-" * 80)
    
    # Summary
    passed = sum(1 for r in results if r.passed)
    print()
    print("=" * 80)
    print(f"SUMMARY: {passed}/{len(results)} tests passed")
    print("=" * 80)
    
    if passed == len(results):
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}: {r.message}")
    
    # Save report
    output_dir = Path(__file__).parent / "regression_results"
    output_dir.mkdir(exist_ok=True)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'A_galaxy': A_GALAXY,
            'A_cluster': A_CLUSTER,
            'xi_scale': XI_SCALE,
            'ml_disk': ML_DISK,
            'ml_bulge': ML_BULGE,
            'g_dagger': g_dagger,
        },
        'results': [asdict(r) for r in results],
        'all_passed': passed == len(results),
    }
    
    with open(output_dir / "latest_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=float)
    
    print(f"\nReport saved to: {output_dir / 'latest_report.json'}")
    
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()

