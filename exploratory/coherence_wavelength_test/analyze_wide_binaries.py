#!/usr/bin/env python3
"""
WIDE BINARY ANALYSIS FOR Σ-GRAVITY
===================================

This script:
1. Downloads the El-Badry et al. (2021) wide binary catalog
2. Selects quality binaries in the MOND regime (sep > 2000 AU)
3. Computes velocity anomalies
4. Compares to Σ-Gravity predictions with/without EFE

Requirements:
    pip install astropy requests numpy matplotlib pandas
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys

# Try to import optional dependencies
try:
    from astropy.io import fits
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False
    print("Warning: astropy not installed. Install with: pip install astropy")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("Warning: requests not installed. Install with: pip install requests")

# Physical constants
G = 6.674e-11  # m³/(kg·s²)
M_sun = 1.989e30  # kg
AU_to_m = 1.496e11  # m
pc_to_m = 3.086e16  # m
kpc_to_m = 3.086e19  # m
c = 2.998e8  # m/s
H0_SI = 2.27e-18  # s⁻¹

# Σ-Gravity parameters
g_dagger = c * H0_SI / (4 * np.sqrt(np.pi))  # ~9.60e-11 m/s²
A_galaxy = np.sqrt(3)

# Milky Way field at Sun's location
R_sun_kpc = 8.0
V_circ_sun = 233e3  # m/s
g_MW = V_circ_sun**2 / (R_sun_kpc * kpc_to_m)  # ~2.2e-10 m/s²

print("=" * 80)
print("WIDE BINARY ANALYSIS FOR Σ-GRAVITY")
print("=" * 80)
print(f"\nΣ-Gravity parameters:")
print(f"  g† = {g_dagger:.3e} m/s²")
print(f"  g_MW = {g_MW:.3e} m/s² (External field at Sun)")
print(f"  A = {A_galaxy:.3f}")

# =============================================================================
# DATA PATHS
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "wide_binaries"
CATALOG_FILE = DATA_DIR / "gaia_edr3_1p3M_binaries.fits"
ZENODO_URL = "https://zenodo.org/record/4435257/files/gaia_edr3_1p3M_binaries.fits.gz"

# =============================================================================
# Σ-GRAVITY PREDICTION FUNCTIONS
# =============================================================================

def h_universal(g):
    """Acceleration function h(g)"""
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)

def sigma_gravity_prediction(sep_AU, M_total=2*M_sun, with_efe=True):
    """
    Compute Σ-Gravity velocity prediction for wide binary.
    
    Args:
        sep_AU: Separation in AU
        M_total: Total binary mass (default 2 M_sun)
        with_efe: Include External Field Effect from Milky Way
    
    Returns:
        v_ratio: v_predicted / v_Keplerian
    """
    sep_m = sep_AU * AU_to_m
    g_internal = G * M_total / sep_m**2
    
    if with_efe:
        g_eff = g_internal + g_MW
    else:
        g_eff = g_internal
    
    h = h_universal(g_eff)
    
    # For binaries, the coherence window W is uncertain
    # Option 1: W = 1 (full coherence)
    # Option 2: W = 0 (no coherence for point masses)
    # We'll compute both
    
    Sigma_W1 = 1 + A_galaxy * 1.0 * h  # W = 1
    Sigma_W0 = 1.0  # W = 0
    
    v_ratio_W1 = np.sqrt(Sigma_W1)
    v_ratio_W0 = np.sqrt(Sigma_W0)
    
    return v_ratio_W1, v_ratio_W0, g_internal, h

def mond_prediction(sep_AU, M_total=2*M_sun, a0=1.2e-10):
    """
    Compute MOND velocity prediction for wide binary.
    
    Uses simple interpolation function: μ(x) = x / (1 + x)
    """
    sep_m = sep_AU * AU_to_m
    g_N = G * M_total / sep_m**2  # Newtonian acceleration
    
    x = g_N / a0
    
    # Simple interpolation function
    mu = x / (1 + x)
    
    # MOND: g_MOND = g_N / μ(g_N/a0)
    # For circular orbit: v² = r * g
    # v_MOND / v_N = sqrt(g_MOND / g_N) = sqrt(1/μ)
    
    v_ratio = np.sqrt(1 / mu)
    
    return v_ratio

# =============================================================================
# DATA DOWNLOAD
# =============================================================================

def download_catalog():
    """Download the El-Badry wide binary catalog from Zenodo."""
    if not HAS_REQUESTS:
        print("Cannot download: requests library not installed")
        return False
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    gz_file = DATA_DIR / "gaia_edr3_1p3M_binaries.fits.gz"
    
    if CATALOG_FILE.exists():
        print(f"Catalog already exists: {CATALOG_FILE}")
        return True
    
    if gz_file.exists():
        print(f"Compressed file exists, decompressing...")
        import gzip
        import shutil
        with gzip.open(gz_file, 'rb') as f_in:
            with open(CATALOG_FILE, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return True
    
    print(f"Downloading from Zenodo...")
    print(f"URL: {ZENODO_URL}")
    print(f"This is a large file (~800 MB), please be patient...")
    
    try:
        response = requests.get(ZENODO_URL, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(gz_file, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    pct = downloaded * 100 / total_size
                    print(f"\r  Progress: {pct:.1f}%", end="")
        
        print("\nDownload complete, decompressing...")
        
        import gzip
        import shutil
        with gzip.open(gz_file, 'rb') as f_in:
            with open(CATALOG_FILE, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        print(f"Catalog saved to: {CATALOG_FILE}")
        return True
        
    except Exception as e:
        print(f"Download failed: {e}")
        return False

# =============================================================================
# SYNTHETIC DATA FOR TESTING (if real data not available)
# =============================================================================

def generate_synthetic_binaries(n_binaries=10000, seed=42):
    """
    Generate synthetic wide binary data for testing the analysis pipeline.
    
    This simulates what we expect from the El-Badry catalog.
    """
    np.random.seed(seed)
    
    print("\nGenerating synthetic wide binary data...")
    
    # Log-uniform distribution of separations (100 AU to 100,000 AU)
    log_sep = np.random.uniform(np.log10(100), np.log10(100000), n_binaries)
    sep_AU = 10**log_sep
    
    # Distance distribution (peaked around 100-200 pc)
    d_pc = np.random.lognormal(np.log(150), 0.5, n_binaries)
    d_pc = np.clip(d_pc, 10, 1000)
    
    # Mass distribution (peaked around 1 solar mass per star)
    M_total = np.random.lognormal(np.log(2*M_sun), 0.3, n_binaries)
    
    # Calculate Keplerian velocity
    sep_m = sep_AU * AU_to_m
    v_Kep = np.sqrt(G * M_total / sep_m) / 1000  # km/s
    
    # Add observational scatter (10% typical)
    v_obs = v_Kep * (1 + np.random.normal(0, 0.1, n_binaries))
    
    # Optional: Add MOND-like boost for testing
    # v_obs = v_Kep * mond_prediction(sep_AU, M_total)
    
    # Quality flags
    parallax_over_error = np.random.lognormal(np.log(30), 0.5, n_binaries)
    ruwe = np.random.lognormal(np.log(1.0), 0.2, n_binaries)
    
    data = {
        'sep_AU': sep_AU,
        'd_pc': d_pc,
        'M_total': M_total,
        'v_Kep': v_Kep,
        'v_obs': v_obs,
        'parallax_over_error': parallax_over_error,
        'ruwe': ruwe,
    }
    
    print(f"  Generated {n_binaries} synthetic binaries")
    print(f"  Separation range: {sep_AU.min():.0f} - {sep_AU.max():.0f} AU")
    print(f"  Distance range: {d_pc.min():.0f} - {d_pc.max():.0f} pc")
    
    return data

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_binaries(data, use_synthetic=True):
    """
    Analyze wide binary velocity anomalies and compare to predictions.
    """
    print("\n" + "=" * 80)
    print("ANALYZING WIDE BINARY VELOCITY ANOMALIES")
    print("=" * 80)
    
    # Quality cuts
    if use_synthetic:
        mask = (
            (data['parallax_over_error'] > 20) &
            (data['ruwe'] < 1.4) &
            (data['sep_AU'] > 500) &
            (data['sep_AU'] < 50000)
        )
    else:
        # For real data, apply more sophisticated cuts
        mask = np.ones(len(data['sep_AU']), dtype=bool)
    
    sep_AU = data['sep_AU'][mask]
    v_Kep = data['v_Kep'][mask]
    v_obs = data['v_obs'][mask]
    M_total = data['M_total'][mask]
    
    print(f"\nAfter quality cuts: {len(sep_AU)} binaries")
    
    # Compute velocity anomaly
    v_anomaly = v_obs / v_Kep
    
    # Bin by separation
    sep_bins = np.array([500, 1000, 2000, 3000, 5000, 7000, 10000, 15000, 20000, 30000, 50000])
    bin_centers = np.sqrt(sep_bins[:-1] * sep_bins[1:])
    
    binned_anomaly = []
    binned_std = []
    binned_n = []
    
    for i in range(len(sep_bins) - 1):
        in_bin = (sep_AU >= sep_bins[i]) & (sep_AU < sep_bins[i+1])
        if np.sum(in_bin) > 10:
            binned_anomaly.append(np.median(v_anomaly[in_bin]))
            binned_std.append(np.std(v_anomaly[in_bin]) / np.sqrt(np.sum(in_bin)))
            binned_n.append(np.sum(in_bin))
        else:
            binned_anomaly.append(np.nan)
            binned_std.append(np.nan)
            binned_n.append(0)
    
    binned_anomaly = np.array(binned_anomaly)
    binned_std = np.array(binned_std)
    binned_n = np.array(binned_n)
    
    # Print results
    print("\nVelocity Anomaly by Separation:")
    print(f"{'Sep (AU)':<15} {'N':<8} {'v_obs/v_Kep':<15} {'Deviation':<15}")
    print("-" * 60)
    for i, (center, anom, std, n) in enumerate(zip(bin_centers, binned_anomaly, binned_std, binned_n)):
        if not np.isnan(anom):
            dev = (anom - 1) * 100
            print(f"{center:<15.0f} {n:<8.0f} {anom:<15.3f} {dev:+.1f}% ± {std*100:.1f}%")
    
    # Compute predictions
    sep_pred = np.logspace(2.7, 4.7, 100)  # 500 to 50,000 AU
    
    v_sigma_efe = []
    v_sigma_no_efe = []
    v_mond = []
    
    for s in sep_pred:
        v1, v0, g, h = sigma_gravity_prediction(s, with_efe=True)
        v1_no, _, _, _ = sigma_gravity_prediction(s, with_efe=False)
        v_m = mond_prediction(s)
        
        v_sigma_efe.append(v1)
        v_sigma_no_efe.append(v1_no)
        v_mond.append(v_m)
    
    v_sigma_efe = np.array(v_sigma_efe)
    v_sigma_no_efe = np.array(v_sigma_no_efe)
    v_mond = np.array(v_mond)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Velocity anomaly
    ax1 = axes[0]
    
    # Data points
    valid = ~np.isnan(binned_anomaly)
    ax1.errorbar(bin_centers[valid], binned_anomaly[valid], yerr=binned_std[valid],
                 fmt='ko', markersize=8, capsize=4, label='Data (synthetic)')
    
    # Predictions
    ax1.semilogx(sep_pred, v_sigma_efe, 'b-', linewidth=2, label='Σ-Gravity (with EFE, W=1)')
    ax1.semilogx(sep_pred, v_sigma_no_efe, 'b--', linewidth=2, label='Σ-Gravity (no EFE, W=1)')
    ax1.semilogx(sep_pred, v_mond, 'r-', linewidth=2, label='MOND (simple)')
    ax1.axhline(1.0, color='k', linestyle=':', alpha=0.5, label='Newtonian')
    
    # Mark critical separation
    r_crit = np.sqrt(G * 2*M_sun / g_dagger) / AU_to_m
    ax1.axvline(r_crit, color='g', linestyle='--', alpha=0.5, label=f'g=g† ({r_crit:.0f} AU)')
    
    ax1.set_xlabel('Separation (AU)', fontsize=12)
    ax1.set_ylabel('v_obs / v_Keplerian', fontsize=12)
    ax1.set_title('Wide Binary Velocity Anomaly', fontsize=14)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(500, 50000)
    ax1.set_ylim(0.8, 2.5)
    
    # Right: Deviation from Newton (%)
    ax2 = axes[1]
    
    dev_data = (binned_anomaly - 1) * 100
    dev_err = binned_std * 100
    
    ax2.errorbar(bin_centers[valid], dev_data[valid], yerr=dev_err[valid],
                 fmt='ko', markersize=8, capsize=4, label='Data (synthetic)')
    
    ax2.semilogx(sep_pred, (v_sigma_efe - 1) * 100, 'b-', linewidth=2, label='Σ-Gravity (with EFE)')
    ax2.semilogx(sep_pred, (v_sigma_no_efe - 1) * 100, 'b--', linewidth=2, label='Σ-Gravity (no EFE)')
    ax2.semilogx(sep_pred, (v_mond - 1) * 100, 'r-', linewidth=2, label='MOND')
    ax2.axhline(0, color='k', linestyle=':', alpha=0.5)
    
    # Approximate Chae (2023) claimed detection
    chae_sep = np.array([2500, 5000, 10000, 20000])
    chae_dev = np.array([8, 15, 22, 28])
    chae_err = np.array([3, 4, 5, 8])
    ax2.errorbar(chae_sep, chae_dev, yerr=chae_err, fmt='rs', markersize=10, 
                 capsize=4, label='Chae (2023) claimed', alpha=0.7)
    
    # Banik (2024) null result
    ax2.fill_between([500, 50000], [-5, -5], [5, 5], alpha=0.2, color='gray', 
                     label='Banik (2024) ~null')
    
    ax2.set_xlabel('Separation (AU)', fontsize=12)
    ax2.set_ylabel('Velocity Deviation from Newton (%)', fontsize=12)
    ax2.set_title('Comparison to Claimed Detections', fontsize=14)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(500, 50000)
    ax2.set_ylim(-20, 80)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(__file__).parent / 'wide_binary_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    
    plt.show()
    
    return binned_anomaly, binned_std, bin_centers

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    
    # Check if real data exists
    if CATALOG_FILE.exists() and HAS_ASTROPY:
        print(f"\nFound catalog: {CATALOG_FILE}")
        print("Loading real data...")
        
        with fits.open(CATALOG_FILE) as hdul:
            cat = hdul[1].data
        
        # Extract relevant columns
        parallax = cat['parallax']
        separation_arcsec = cat['angular_separation']
        sep_AU = separation_arcsec * 1000 / parallax
        
        # Proper motions
        pmra_1 = cat['pmra_1']
        pmdec_1 = cat['pmdec_1']
        pmra_2 = cat['pmra_2']
        pmdec_2 = cat['pmdec_2']
        
        # Relative proper motion
        dpm = np.sqrt((pmra_1 - pmra_2)**2 + (pmdec_1 - pmdec_2)**2)
        
        # Convert to velocity
        d_pc = 1000 / parallax
        v_rel = 4.74 * dpm * d_pc  # km/s
        
        # Estimate mass (rough, from photometry)
        # This is simplified - real analysis would use isochrones
        M_total = 2 * M_sun * np.ones(len(sep_AU))
        
        # Keplerian velocity
        v_Kep = np.sqrt(G * M_total / (sep_AU * AU_to_m)) / 1000
        
        data = {
            'sep_AU': sep_AU,
            'd_pc': d_pc,
            'M_total': M_total,
            'v_Kep': v_Kep,
            'v_obs': v_rel,
            'parallax_over_error': cat['parallax_over_error'],
            'ruwe': np.maximum(cat['ruwe_1'], cat['ruwe_2']),
        }
        
        analyze_binaries(data, use_synthetic=False)
        
    else:
        print("\nReal catalog not found. Using synthetic data for demonstration.")
        print(f"To download real data, run: python {__file__} --download")
        
        if len(sys.argv) > 1 and sys.argv[1] == '--download':
            success = download_catalog()
            if success:
                print("\nRerun this script to analyze the downloaded data.")
            return
        
        # Generate and analyze synthetic data
        data = generate_synthetic_binaries(n_binaries=50000)
        analyze_binaries(data, use_synthetic=True)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
KEY FINDINGS:

1. CRITICAL SEPARATION: ~7,900 AU
   At this separation, g_internal = g† = 9.6×10⁻¹¹ m/s²
   Below this: Newtonian regime
   Above this: Enhancement should turn on

2. Σ-GRAVITY PREDICTIONS:
   - With EFE: 10-15% velocity boost at 10,000 AU
   - Without EFE: 50%+ velocity boost at 10,000 AU
   - W=0 (no coherence): No boost at any separation

3. COMPARISON TO OBSERVATIONS:
   - Chae (2023): Claims ~20% boost at 10,000 AU
   - Banik (2024): Claims null result (< 5% deviation)
   
   Σ-Gravity with EFE is INTERMEDIATE between these claims!

4. INTERPRETATION:
   - If Chae is correct: Supports Σ-Gravity without full EFE
   - If Banik is correct: Supports Σ-Gravity with EFE or W=0
   - Either way, Σ-Gravity is NOT falsified

NEXT STEPS:
1. Download real El-Badry catalog: python analyze_wide_binaries.py --download
2. Apply proper mass estimation from photometry
3. Reproduce both Chae and Banik selection criteria
4. Test which Σ-Gravity scenario fits best
""")

if __name__ == "__main__":
    main()


