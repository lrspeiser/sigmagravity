"""
Compute Σ-Gravity enhancement factor from rotation curves and lensing.

This module:
1. Loads rotation curve data (SPARC)
2. Computes the "anomalous" acceleration (beyond Newtonian from baryons)
3. Derives the effective Σ enhancement as a function of position
4. Provides the target variable for correlation with quietness factors
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SPARC_DIR, G_GRAV

# =============================================================================
# LOAD SPARC DATA
# =============================================================================

def load_sparc_rotcurve(galaxy_name: str) -> Dict:
    """
    Load rotation curve for a single SPARC galaxy.
    
    Returns dict with:
        r_kpc: Radii (kpc)
        v_obs: Observed rotation velocity (km/s)
        v_err: Velocity uncertainty (km/s)
        v_bar: Baryonic velocity (sqrt(v_gas² + v_disk² + v_bulge²))
        v_gas, v_disk, v_bulge: Individual components
    """
    # Try different possible locations
    possible_paths = [
        SPARC_DIR / "SPARC_Rotmod" / f"{galaxy_name}_rotmod.dat",
        SPARC_DIR / "rotcurves" / f"{galaxy_name}_rotmod.dat",
        SPARC_DIR / f"{galaxy_name}_rotmod.dat",
    ]
    
    filepath = None
    for p in possible_paths:
        if p.exists():
            filepath = p
            break
    
    if filepath is None:
        raise FileNotFoundError(f"Rotation curve not found for {galaxy_name}")
    
    # Parse file
    data = {
        'r_kpc': [],
        'v_obs': [],
        'v_err': [],
        'v_gas': [],
        'v_disk': [],
        'v_bulge': []
    }
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) < 5:
                continue
            
            try:
                data['r_kpc'].append(float(parts[0]))
                data['v_obs'].append(float(parts[1]))
                data['v_err'].append(float(parts[2]))
                data['v_gas'].append(float(parts[3]))
                data['v_disk'].append(float(parts[4]))
                data['v_bulge'].append(float(parts[5]) if len(parts) > 5 else 0.0)
            except ValueError:
                continue
    
    # Convert to arrays
    for key in data:
        data[key] = np.array(data[key])
    
    # Compute baryonic velocity
    data['v_bar'] = np.sqrt(
        data['v_gas']**2 + 
        data['v_disk']**2 + 
        data['v_bulge']**2
    )
    
    data['galaxy_name'] = galaxy_name
    
    return data


def load_all_sparc_galaxies() -> list:
    """Load all available SPARC rotation curves."""
    
    # Find rotation curve files
    rotcurve_dirs = [
        SPARC_DIR / "SPARC_Rotmod",
        SPARC_DIR / "rotcurves",
        SPARC_DIR,
    ]
    
    files = []
    for d in rotcurve_dirs:
        if d.exists():
            files.extend(d.glob("*_rotmod.dat"))
    
    # Remove duplicates
    galaxy_names = list(set(f.stem.replace("_rotmod", "") for f in files))
    
    galaxies = []
    for name in sorted(galaxy_names):
        try:
            data = load_sparc_rotcurve(name)
            galaxies.append(data)
        except Exception as e:
            print(f"Warning: Could not load {name}: {e}")
    
    print(f"Loaded {len(galaxies)} galaxies")
    return galaxies


# =============================================================================
# COMPUTE SIGMA ENHANCEMENT
# =============================================================================

def compute_sigma_enhancement(data: Dict) -> Dict:
    """
    Compute the Σ enhancement factor from rotation curve.
    
    The enhancement is defined as the ratio of total gravity to baryonic gravity:
        Σ = g_obs / g_bar = (v_obs² / r) / (v_bar² / r) = (v_obs / v_bar)²
    
    Where g_bar = v_bar²/r is what we'd expect from baryons alone.
    
    Returns data dict with additional columns:
        g_obs: Observed centripetal acceleration (km/s)²/kpc
        g_bar: Baryonic acceleration
        sigma: Enhancement factor Σ = g_obs / g_bar
        sigma_err: Uncertainty in Σ
    """
    data = data.copy()
    
    r = data['r_kpc']
    v_obs = data['v_obs']
    v_err = data['v_err']
    v_bar = data['v_bar']
    
    # Avoid division by zero
    r_safe = np.where(r > 0, r, 0.1)
    v_bar_safe = np.where(v_bar > 0, v_bar, 1.0)
    
    # Accelerations (in (km/s)²/kpc)
    g_obs = v_obs**2 / r_safe
    g_bar = v_bar**2 / r_safe
    
    # Sigma enhancement
    sigma = g_obs / np.where(g_bar > 0, g_bar, 1.0)
    
    # Error propagation: σ_Σ/Σ ≈ 2 * σ_v/v (assuming v_bar is well-determined)
    sigma_err = sigma * 2 * v_err / np.where(v_obs > 0, v_obs, 1.0)
    
    data['g_obs'] = g_obs
    data['g_bar'] = g_bar
    data['sigma'] = sigma
    data['sigma_err'] = sigma_err
    
    return data


def compute_rar_residual(data: Dict) -> Dict:
    """
    Compute residual from Radial Acceleration Relation (RAR).
    
    The RAR predicts: g_obs = g_bar / (1 - exp(-sqrt(g_bar/g†)))
    where g† ≈ 1.2 × 10⁻¹⁰ m/s² = 3.7 (km/s)²/kpc
    
    The residual tells us how much Σ-Gravity deviates from MOND-like behavior.
    """
    data = data.copy()
    
    g_bar = data['g_bar']
    g_obs = data['g_obs']
    
    # RAR critical acceleration (converted to (km/s)²/kpc)
    g_dagger = 3.7  # (km/s)²/kpc
    
    # RAR prediction
    x = np.sqrt(g_bar / g_dagger)
    g_rar = g_bar / (1 - np.exp(-x))
    
    # Residual
    rar_residual = np.log10(g_obs / g_rar)
    
    data['g_rar'] = g_rar
    data['rar_residual'] = rar_residual
    
    return data


# =============================================================================
# AGGREGATE STATISTICS
# =============================================================================

def compute_sigma_profile(galaxies: list, 
                          r_bins: np.ndarray = None) -> Dict:
    """
    Compute stacked Σ(r) profile across all galaxies.
    
    Parameters
    ----------
    galaxies : list of dicts
        Each dict is output from compute_sigma_enhancement()
    r_bins : array, optional
        Radial bins in kpc
    
    Returns
    -------
    profile : dict
        r_mid: Bin centers
        sigma_median: Median Σ in each bin
        sigma_16, sigma_84: 16th and 84th percentiles
        n_points: Number of data points per bin
    """
    if r_bins is None:
        r_bins = np.array([0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50])
    
    # Collect all (r, Σ) pairs
    all_r = []
    all_sigma = []
    
    for gal in galaxies:
        if 'sigma' not in gal:
            gal = compute_sigma_enhancement(gal)
        
        valid = (gal['sigma'] > 0) & np.isfinite(gal['sigma'])
        all_r.extend(gal['r_kpc'][valid])
        all_sigma.extend(gal['sigma'][valid])
    
    all_r = np.array(all_r)
    all_sigma = np.array(all_sigma)
    
    # Bin
    n_bins = len(r_bins) - 1
    r_mid = np.sqrt(r_bins[:-1] * r_bins[1:])  # Geometric mean
    sigma_median = np.zeros(n_bins)
    sigma_16 = np.zeros(n_bins)
    sigma_84 = np.zeros(n_bins)
    n_points = np.zeros(n_bins, dtype=int)
    
    for i in range(n_bins):
        mask = (all_r >= r_bins[i]) & (all_r < r_bins[i+1])
        n_points[i] = np.sum(mask)
        
        if n_points[i] > 0:
            sigma_in_bin = all_sigma[mask]
            sigma_median[i] = np.median(sigma_in_bin)
            sigma_16[i] = np.percentile(sigma_in_bin, 16)
            sigma_84[i] = np.percentile(sigma_in_bin, 84)
    
    return {
        'r_mid': r_mid,
        'sigma_median': sigma_median,
        'sigma_16': sigma_16,
        'sigma_84': sigma_84,
        'n_points': n_points,
        'r_bins': r_bins,
    }


def compute_sigma_vs_gbar(galaxies: list,
                          gbar_bins: np.ndarray = None) -> Dict:
    """
    Compute Σ as function of baryonic acceleration g_bar.
    
    This is essentially the RAR but expressed as enhancement factor.
    """
    if gbar_bins is None:
        # Logarithmic bins from 0.01 to 100 (km/s)²/kpc
        gbar_bins = np.logspace(-2, 2, 20)
    
    all_gbar = []
    all_sigma = []
    
    for gal in galaxies:
        if 'sigma' not in gal:
            gal = compute_sigma_enhancement(gal)
        
        valid = (gal['g_bar'] > 0) & (gal['sigma'] > 0) & np.isfinite(gal['sigma'])
        all_gbar.extend(gal['g_bar'][valid])
        all_sigma.extend(gal['sigma'][valid])
    
    all_gbar = np.array(all_gbar)
    all_sigma = np.array(all_sigma)
    
    n_bins = len(gbar_bins) - 1
    gbar_mid = np.sqrt(gbar_bins[:-1] * gbar_bins[1:])
    sigma_median = np.zeros(n_bins)
    sigma_16 = np.zeros(n_bins)
    sigma_84 = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (all_gbar >= gbar_bins[i]) & (all_gbar < gbar_bins[i+1])
        if np.sum(mask) > 0:
            sigma_in_bin = all_sigma[mask]
            sigma_median[i] = np.median(sigma_in_bin)
            sigma_16[i] = np.percentile(sigma_in_bin, 16)
            sigma_84[i] = np.percentile(sigma_in_bin, 84)
    
    return {
        'gbar_mid': gbar_mid,
        'sigma_median': sigma_median,
        'sigma_16': sigma_16,
        'sigma_84': sigma_84,
        'gbar_bins': gbar_bins,
    }


# =============================================================================
# EXPORT FOR CORRELATION
# =============================================================================

def export_sigma_for_correlation(galaxies: list) -> Tuple[np.ndarray, ...]:
    """
    Export Σ values for correlation with quietness variables.
    
    Returns arrays that can be matched with quietness measurements:
        r_kpc: Radii (flattened across all galaxies)
        sigma: Enhancement factors
        g_bar: Baryonic accelerations
        galaxy_id: Index identifying which galaxy each point belongs to
    """
    all_r = []
    all_sigma = []
    all_gbar = []
    all_gal_id = []
    
    for i, gal in enumerate(galaxies):
        if 'sigma' not in gal:
            gal = compute_sigma_enhancement(gal)
        
        valid = (gal['sigma'] > 0) & np.isfinite(gal['sigma'])
        n_valid = np.sum(valid)
        
        all_r.extend(gal['r_kpc'][valid])
        all_sigma.extend(gal['sigma'][valid])
        all_gbar.extend(gal['g_bar'][valid])
        all_gal_id.extend([i] * n_valid)
    
    return (
        np.array(all_r),
        np.array(all_sigma),
        np.array(all_gbar),
        np.array(all_gal_id)
    )


# =============================================================================
# MAIN / TEST
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("Loading SPARC data...")
    galaxies = load_all_sparc_galaxies()
    
    if len(galaxies) == 0:
        print("No galaxies loaded. Run download_sparc.py first.")
        sys.exit(1)
    
    # Compute Σ for all
    print("Computing Σ enhancement...")
    for i in range(len(galaxies)):
        galaxies[i] = compute_sigma_enhancement(galaxies[i])
        galaxies[i] = compute_rar_residual(galaxies[i])
    
    # Compute profiles
    print("Computing stacked profiles...")
    profile_r = compute_sigma_profile(galaxies)
    profile_gbar = compute_sigma_vs_gbar(galaxies)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Example galaxy
    gal = galaxies[0]
    ax = axes[0]
    ax.errorbar(gal['r_kpc'], gal['v_obs'], yerr=gal['v_err'], 
                fmt='o', label='Observed', ms=4)
    ax.plot(gal['r_kpc'], gal['v_bar'], 'r--', label='Baryonic')
    ax.set_xlabel('r (kpc)')
    ax.set_ylabel('v (km/s)')
    ax.set_title(f"{gal['galaxy_name']}")
    ax.legend()
    
    # Σ(r) profile
    ax = axes[1]
    valid = profile_r['n_points'] > 0
    ax.fill_between(profile_r['r_mid'][valid], 
                    profile_r['sigma_16'][valid],
                    profile_r['sigma_84'][valid],
                    alpha=0.3)
    ax.plot(profile_r['r_mid'][valid], profile_r['sigma_median'][valid], 'b-', lw=2)
    ax.axhline(1, color='gray', ls='--')
    ax.set_xlabel('r (kpc)')
    ax.set_ylabel('Σ enhancement')
    ax.set_xscale('log')
    ax.set_title('Stacked Σ(r)')
    
    # Σ(g_bar) - the RAR
    ax = axes[2]
    valid = profile_gbar['sigma_median'] > 0
    ax.fill_between(profile_gbar['gbar_mid'][valid],
                    profile_gbar['sigma_16'][valid],
                    profile_gbar['sigma_84'][valid],
                    alpha=0.3)
    ax.plot(profile_gbar['gbar_mid'][valid], 
            profile_gbar['sigma_median'][valid], 'b-', lw=2)
    ax.axhline(1, color='gray', ls='--')
    ax.set_xlabel('g_bar [(km/s)²/kpc]')
    ax.set_ylabel('Σ enhancement')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Σ(g_bar) - RAR as enhancement')
    
    plt.tight_layout()
    plt.savefig('sigma_enhancement.png', dpi=150)
    print("Saved sigma_enhancement.png")
    
    # Export for correlation
    r, sigma, gbar, gal_id = export_sigma_for_correlation(galaxies)
    print(f"\nExported {len(r)} data points for correlation analysis")
    print(f"  Σ range: {sigma.min():.2f} to {sigma.max():.2f}")
    print(f"  g_bar range: {gbar.min():.3f} to {gbar.max():.1f}")
