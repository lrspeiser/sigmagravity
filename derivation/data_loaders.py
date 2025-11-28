#!/usr/bin/env python3
"""
Data Loaders for Theory Discovery Engine

Connects real Σ-Gravity data sources to the symbolic regression engine:
1. SPARC galaxy rotation curves (166 galaxies)
2. Gaia MW star-level RAR (1.8M stars)
3. Galaxy cluster profiles

Each loader returns (X: Dict[str, ndarray], y: ndarray, metadata: dict)
compatible with GPUTheoryDiscoveryEngine.evolve()
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import warnings

warnings.filterwarnings('ignore')

# Base data directory
DATA_DIR = Path(__file__).parent.parent / "data"


# =============================================================================
# SPARC GALAXY DATA
# =============================================================================

def load_sparc_galaxies(
    n_galaxies: Optional[int] = None,
    seed: int = 42
) -> Tuple[Dict[str, np.ndarray], np.ndarray, dict]:
    """
    Load SPARC galaxy summary data.
    
    Returns data for discovering: v_flat = f(M_baryon, R_disk, ...)
    
    Variables available:
    - M_baryon: Total baryonic mass (solar masses)
    - M_stellar: Stellar mass
    - M_gas: Gas mass  
    - R_disk: Disk scale radius (kpc)
    - bulge_frac: Bulge fraction
    
    Target: v_flat (flat rotation velocity in km/s)
    """
    csv_path = DATA_DIR / "sparc" / "sparc_combined.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"SPARC data not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Sample if requested
    if n_galaxies is not None and n_galaxies < len(df):
        np.random.seed(seed)
        df = df.sample(n=n_galaxies, random_state=seed)
    
    # Build feature dict
    X = {
        'M': np.log10(df['M_baryon'].values),  # Log mass
        'R': df['R_disk'].values,
        'M_star': np.log10(df['M_stellar'].values + 1),
        'M_gas': np.log10(df['M_gas'].values + 1),
        'f_b': df['bulge_frac'].values,
    }
    
    # Target: flat rotation velocity
    y = df['v_flat'].values
    
    # Metadata
    meta = {
        'source': 'SPARC',
        'n_galaxies': len(df),
        'target': 'v_flat (km/s)',
        'expected_relation': 'Baryonic Tully-Fisher: v^4 ∝ M_baryon',
        'variables': list(X.keys()),
    }
    
    print(f"✓ Loaded SPARC: {len(df)} galaxies")
    print(f"  Features: {list(X.keys())}")
    print(f"  Target: v_flat range [{y.min():.1f}, {y.max():.1f}] km/s")
    
    return X, y, meta


def load_sparc_rar(
    max_points: Optional[int] = None,
    seed: int = 42
) -> Tuple[Dict[str, np.ndarray], np.ndarray, dict]:
    """
    Load SPARC Radial Acceleration Relation data.
    
    This is the key target for Σ-Gravity: discovering g_obs = g_bar × [1 + K(R)]
    
    Variables:
    - g_bar: Baryonic acceleration (log10, m/s²)
    - R: Galactic radius (kpc)
    
    Target: g_obs (observed acceleration, log10 m/s²)
    
    Note: Needs point-by-point rotation curve data. If not available,
    generates from galaxy summaries using circular velocity formula.
    """
    # Try to load pre-computed RAR data
    rar_path = DATA_DIR / "sparc" / "sparc_rar_points.csv"
    
    if rar_path.exists():
        df = pd.read_csv(rar_path)
        if max_points and len(df) > max_points:
            np.random.seed(seed)
            df = df.sample(n=max_points, random_state=seed)
        
        X = {
            'g_bar': df['log10_g_bar'].values,
            'R': df['R_kpc'].values,
        }
        y = df['log10_g_obs'].values
        
    else:
        # Generate RAR points from galaxy summaries
        print("  Generating RAR from galaxy summaries...")
        galaxies_path = DATA_DIR / "sparc" / "sparc_combined.csv"
        df = pd.read_csv(galaxies_path)
        
        G = 4.302e-6  # kpc (km/s)² / M_sun
        
        g_bars, g_obs_list, radii = [], [], []
        
        for _, row in df.iterrows():
            M = row['M_baryon']
            R_d = row['R_disk']
            v_flat = row['v_flat']
            
            # Generate points at multiple radii
            for r_mult in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
                R = R_d * r_mult
                if R < 0.1:
                    continue
                    
                # Baryonic acceleration (exponential disk approximation)
                g_bar = G * M / R**2 * (1 - np.exp(-R/R_d))
                
                # Observed acceleration from flat rotation curve
                g_obs = v_flat**2 / R / 3.086e16  # Convert to m/s²
                g_bar_si = g_bar * 3.086e16 / 1e6  # Convert to m/s²
                
                if g_bar_si > 1e-13 and g_obs > 1e-13:
                    g_bars.append(np.log10(g_bar_si))
                    g_obs_list.append(np.log10(g_obs))
                    radii.append(R)
        
        X = {
            'g_bar': np.array(g_bars),
            'R': np.array(radii),
        }
        y = np.array(g_obs_list)
        
        if max_points and len(y) > max_points:
            np.random.seed(seed)
            idx = np.random.choice(len(y), max_points, replace=False)
            X = {k: v[idx] for k, v in X.items()}
            y = y[idx]
    
    meta = {
        'source': 'SPARC RAR',
        'n_points': len(y),
        'target': 'log10(g_obs) [m/s²]',
        'expected_relation': 'MOND: g_obs = g_bar × ν(g_bar/a0)',
        'sigma_gravity': 'g_obs = g_bar × [1 + K(R)]',
        'variables': list(X.keys()),
    }
    
    print(f"✓ Loaded SPARC RAR: {len(y)} points")
    print(f"  g_bar range: [{X['g_bar'].min():.2f}, {X['g_bar'].max():.2f}] (log10 m/s²)")
    print(f"  g_obs range: [{y.min():.2f}, {y.max():.2f}] (log10 m/s²)")
    
    return X, y, meta


# =============================================================================
# GAIA MILKY WAY DATA
# =============================================================================

def load_gaia_rar(
    max_stars: Optional[int] = None,
    use_extended: bool = True,
    seed: int = 42
) -> Tuple[Dict[str, np.ndarray], np.ndarray, dict]:
    """
    Load Gaia Milky Way star-level RAR data.
    
    This is the 1.8M star dataset for discovering:
    g_obs = f(g_bar, R, z, ...)
    
    Variables:
    - g_bar: Baryonic acceleration (log10 m/s²)
    - R: Galactic radius (kpc)
    - z: Height above plane (kpc)
    - v_obs: Observed circular velocity (km/s)
    - Sigma_loc: Local surface density (M_sun/pc²)
    
    Target: g_obs (observed acceleration, log10 m/s²)
    """
    if use_extended:
        csv_path = DATA_DIR / "gaia" / "outputs" / "mw_rar_starlevel_full.csv"
        if not csv_path.exists():
            csv_path = DATA_DIR / "gaia" / "outputs" / "mw_rar_starlevel_extended.csv"
    else:
        csv_path = DATA_DIR / "gaia" / "outputs" / "mw_rar_starlevel.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Gaia RAR data not found: {csv_path}")
    
    print(f"  Loading from {csv_path.name}...")
    df = pd.read_csv(csv_path)
    
    # Sample if too large
    if max_stars is not None and len(df) > max_stars:
        np.random.seed(seed)
        df = df.sample(n=max_stars, random_state=seed)
    
    # Build feature dict
    X = {
        'g_bar': df['log10_g_bar'].values,
        'R': df['R_kpc'].values,
        'z': np.abs(df['z_kpc'].values),
        'v': df['v_obs_kms'].values,
    }
    
    # Add local density if available
    if 'Sigma_loc_Msun_pc2' in df.columns:
        X['Sigma'] = np.log10(df['Sigma_loc_Msun_pc2'].values + 1)
    
    # Target
    y = df['log10_g_obs'].values
    
    # Filter invalid values
    valid = np.isfinite(y) & np.isfinite(X['g_bar'])
    for k in X:
        valid &= np.isfinite(X[k])
    
    X = {k: v[valid] for k, v in X.items()}
    y = y[valid]
    
    meta = {
        'source': 'Gaia MW RAR',
        'n_stars': len(y),
        'target': 'log10(g_obs) [m/s²]',
        'expected_relation': 'MW rotation curve anomaly',
        'sigma_gravity': 'g_obs = g_bar × [1 + K(R)]',
        'variables': list(X.keys()),
    }
    
    print(f"✓ Loaded Gaia MW: {len(y):,} stars")
    print(f"  R range: [{X['R'].min():.2f}, {X['R'].max():.2f}] kpc")
    print(f"  g_bar range: [{X['g_bar'].min():.2f}, {X['g_bar'].max():.2f}]")
    
    return X, y, meta


def load_gaia_enhancement_factor(
    max_stars: Optional[int] = None,
    seed: int = 42
) -> Tuple[Dict[str, np.ndarray], np.ndarray, dict]:
    """
    Load Gaia data formatted for discovering the enhancement factor K(R).
    
    Target: K = g_obs/g_bar - 1 (the gravitational enhancement)
    
    This is the core Σ-Gravity relationship to discover.
    """
    X, y_gobs, meta = load_gaia_rar(max_stars=max_stars, seed=seed)
    
    # Compute enhancement factor K
    # K = g_obs/g_bar - 1 = 10^(log10_g_obs - log10_g_bar) - 1
    log_ratio = y_gobs - X['g_bar']
    K = 10**log_ratio - 1
    
    # Filter extreme values
    valid = (K > -0.5) & (K < 100) & np.isfinite(K)
    X = {k: v[valid] for k, v in X.items()}
    K = K[valid]
    
    meta['target'] = 'K = g_obs/g_bar - 1 (enhancement factor)'
    meta['sigma_gravity'] = 'K(R) ~ coherence window function'
    meta['n_stars'] = len(K)
    
    print(f"✓ Computed enhancement factor K for {len(K):,} stars")
    print(f"  K range: [{K.min():.3f}, {K.max():.3f}]")
    print(f"  K median: {np.median(K):.3f}")
    
    return X, K, meta


# =============================================================================
# GALAXY CLUSTER DATA
# =============================================================================

def load_cluster_profiles(
    cluster_name: Optional[str] = None
) -> Tuple[Dict[str, np.ndarray], np.ndarray, dict]:
    """
    Load galaxy cluster gas/mass profiles.
    
    Clusters available: ABELL_1689, MACSJ0416, MACSJ0717
    
    Variables:
    - R: Radius (kpc)
    - n_e: Electron density (cm^-3)
    - T: Temperature (keV)
    
    Target: log10(n_e) - used to discover density profile relationships
    """
    clusters_dir = DATA_DIR / "clusters"
    
    if cluster_name:
        cluster_dirs = [clusters_dir / cluster_name]
    else:
        cluster_dirs = [d for d in clusters_dir.iterdir() 
                       if d.is_dir() and not d.name.startswith('.') 
                       and not d.name == 'figures']
    
    all_R, all_n_e, all_T = [], [], []
    cluster_names = []
    
    for cdir in cluster_dirs:
        gas_path = cdir / "gas_profile.csv"
        temp_path = cdir / "temp_profile.csv"
        
        if not gas_path.exists():
            continue
            
        gas_df = pd.read_csv(gas_path)
        
        # Get radius (column is r_kpc)
        if 'r_kpc' in gas_df.columns:
            R = gas_df['r_kpc'].values
        elif 'R_kpc' in gas_df.columns:
            R = gas_df['R_kpc'].values
        else:
            print(f"  Skipping {cdir.name}: no radius column")
            continue
        
        # Get electron density (n_e_cm3)
        if 'n_e_cm3' in gas_df.columns:
            n_e = gas_df['n_e_cm3'].values
        else:
            print(f"  Skipping {cdir.name}: no n_e column")
            continue
        
        # Get temperature if available
        if temp_path.exists():
            temp_df = pd.read_csv(temp_path)
            if 'T_keV' in temp_df.columns and 'r_kpc' in temp_df.columns:
                T = np.interp(R, temp_df['r_kpc'].values, temp_df['T_keV'].values)
            else:
                T = np.ones_like(R) * 5.0
        else:
            T = np.ones_like(R) * 5.0
        
        all_R.extend(R)
        all_n_e.extend(n_e)
        all_T.extend(T)
        cluster_names.extend([cdir.name] * len(R))
    
    if len(all_R) == 0:
        print("  Warning: No cluster data loaded")
        return {'R': np.array([])}, np.array([]), {'source': 'Galaxy Clusters', 'n_points': 0}
    
    X = {
        'R': np.array(all_R),
        'T': np.array(all_T),
    }
    
    # Target: electron density (log scale)
    y = np.log10(np.array(all_n_e) + 1e-10)
    
    meta = {
        'source': 'Galaxy Clusters',
        'clusters': list(set(cluster_names)),
        'n_points': len(y),
        'target': 'log10(n_e) [cm^-3]',
        'expected_relation': 'Beta-model: n_e ∝ (1 + (r/r_c)²)^(-3β/2)',
        'variables': list(X.keys()),
    }
    
    print(f"✓ Loaded cluster profiles: {len(y)} points from {len(set(cluster_names))} clusters")
    
    return X, y, meta


# =============================================================================
# COMBINED LOADERS
# =============================================================================

def load_multi_scale_rar(
    n_sparc: int = 1000,
    n_gaia: int = 10000,
    seed: int = 42
) -> Tuple[Dict[str, np.ndarray], np.ndarray, dict]:
    """
    Load combined RAR data from SPARC galaxies and Gaia MW.
    
    This tests whether a single formula works across scales.
    """
    # Load both datasets
    X_sparc, y_sparc, _ = load_sparc_rar(max_points=n_sparc, seed=seed)
    X_gaia, y_gaia, _ = load_gaia_rar(max_stars=n_gaia, seed=seed)
    
    # Find common features
    common_keys = set(X_sparc.keys()) & set(X_gaia.keys())
    
    X = {}
    for k in common_keys:
        X[k] = np.concatenate([X_sparc[k], X_gaia[k]])
    
    y = np.concatenate([y_sparc, y_gaia])
    
    # Add source indicator
    X['is_MW'] = np.concatenate([
        np.zeros(len(y_sparc)),
        np.ones(len(y_gaia))
    ])
    
    meta = {
        'source': 'Multi-scale RAR (SPARC + Gaia)',
        'n_sparc': len(y_sparc),
        'n_gaia': len(y_gaia),
        'n_total': len(y),
        'target': 'log10(g_obs) [m/s²]',
        'variables': list(X.keys()),
    }
    
    print(f"✓ Combined multi-scale: {len(y):,} total points")
    
    return X, y, meta


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  DATA LOADER TEST")
    print("=" * 60)
    
    # Test each loader
    print("\n1. SPARC Galaxies:")
    try:
        X, y, meta = load_sparc_galaxies(n_galaxies=50)
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n2. SPARC RAR:")
    try:
        X, y, meta = load_sparc_rar(max_points=1000)
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n3. Gaia MW RAR:")
    try:
        X, y, meta = load_gaia_rar(max_stars=5000)
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n4. Gaia Enhancement Factor K:")
    try:
        X, K, meta = load_gaia_enhancement_factor(max_stars=5000)
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n5. Cluster Profiles:")
    try:
        X, y, meta = load_cluster_profiles()
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n" + "=" * 60)
