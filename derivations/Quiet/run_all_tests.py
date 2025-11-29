#!/usr/bin/env python
"""
Master Script: Test All Quietness Variables Against Gravitational Anomalies

This script orchestrates the full analysis pipeline:
1. Load/download required data
2. Compute each quietness variable
3. Compute Σ enhancement from rotation curves
4. Run correlation tests
5. Generate comparison plots and tables
6. Output summary of which variable best predicts gravitational coherence

Usage:
    python run_all_tests.py [--download] [--quick]

Options:
    --download  Download missing data files
    --quick     Use smaller samples for faster testing
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DATA_ROOT, OUTPUT_DIR, SPARC_DIR, GAIA_DIR, 
    LENSING_DIR, COSMIC_WEB_DIR, SURVEYS_DIR, PULSAR_DIR
)

# =============================================================================
# IMPORT ANALYSIS MODULES
# =============================================================================

def import_modules():
    """Import analysis modules (some may fail if dependencies missing)."""
    modules = {}
    
    try:
        from anomalies.sigma_enhancement import (
            load_all_sparc_galaxies,
            compute_sigma_enhancement,
            export_sigma_for_correlation
        )
        modules['sigma'] = {
            'load': load_all_sparc_galaxies,
            'compute': compute_sigma_enhancement,
            'export': export_sigma_for_correlation,
        }
    except ImportError as e:
        print(f"Warning: Could not import sigma_enhancement: {e}")
    
    try:
        from correlations.correlation_analysis import (
            QuietnessCorrelationTest,
            compare_all_quietness_variables,
            identify_best_predictor,
            plot_correlation
        )
        modules['correlation'] = {
            'test': QuietnessCorrelationTest,
            'compare': compare_all_quietness_variables,
            'best': identify_best_predictor,
            'plot': plot_correlation,
        }
    except ImportError as e:
        print(f"Warning: Could not import correlation_analysis: {e}")
    
    return modules


# =============================================================================
# DATA AVAILABILITY CHECK
# =============================================================================

def check_data_availability() -> dict:
    """Check which datasets are available."""
    
    status = {
        'sparc': {
            'available': False,
            'path': SPARC_DIR,
            'required_files': ['SPARC_Lelli2016c.mrt'],
        },
        'gaia': {
            'available': False,
            'path': GAIA_DIR,
            'required_files': [],  # Dynamically downloaded
        },
        'lensing': {
            'available': False,
            'path': LENSING_DIR,
            'required_files': ['mock_shear_catalog.fits'],
        },
        'cosmic_web': {
            'available': False,
            'path': COSMIC_WEB_DIR,
            'required_files': [],
        },
        'surveys': {
            'available': False,
            'path': SURVEYS_DIR,
            'required_files': [],
        },
        'pulsar': {
            'available': False,
            'path': PULSAR_DIR,
            'required_files': [],
        },
    }
    
    for name, info in status.items():
        path = info['path']
        if path.exists():
            files = list(path.glob('*'))
            info['available'] = len(files) > 0
            info['n_files'] = len(files)
    
    return status


def print_data_status(status: dict):
    """Print data availability status."""
    print("\n" + "=" * 60)
    print("DATA AVAILABILITY")
    print("=" * 60)
    
    for name, info in status.items():
        icon = "✓" if info['available'] else "✗"
        n_files = info.get('n_files', 0)
        print(f"  {icon} {name:15} {n_files:3} files  ({info['path']})")


# =============================================================================
# QUIETNESS VARIABLE COMPUTATION STUBS
# =============================================================================

def compute_velocity_dispersion_quietness(gaia_data: dict) -> np.ndarray:
    """
    Compute quietness from local velocity dispersion.
    
    Lower σ_v = more quiet = higher quietness value
    """
    if gaia_data is None:
        return None
    
    sigma_v = gaia_data.get('sigma_v_local', np.array([]))
    if len(sigma_v) == 0:
        return None
    
    # Normalize: low σ_v → high quietness
    sigma_max = np.percentile(sigma_v, 99)
    quietness = 1 - np.clip(sigma_v / sigma_max, 0, 1)
    
    return quietness


def compute_density_quietness(density_data: dict) -> np.ndarray:
    """
    Compute quietness from matter density.
    
    Lower density = more quiet = higher quietness value
    """
    if density_data is None:
        return None
    
    delta = density_data.get('delta', np.array([]))
    if len(delta) == 0:
        return None
    
    # Transform: δ = -1 (void) → quietness = 1
    #           δ = 0 (mean) → quietness = 0.5
    #           δ >> 1 (cluster) → quietness → 0
    quietness = 1 / (1 + np.exp(delta))
    
    return quietness


def compute_dynamical_time_quietness(kinematics_data: dict) -> np.ndarray:
    """
    Compute quietness from dynamical timescale.
    
    Longer t_dyn = more quiet = higher quietness value
    """
    if kinematics_data is None:
        return None
    
    t_dyn_gyr = kinematics_data.get('t_dyn', np.array([]))
    if len(t_dyn_gyr) == 0:
        return None
    
    # Normalize: t_dyn > 5 Gyr → quiet
    quietness = np.clip(t_dyn_gyr / 5.0, 0, 1)
    
    return quietness


def compute_tidal_quietness(cosmic_web_data: dict) -> np.ndarray:
    """
    Compute quietness from cosmic web classification.
    
    Void = quiet, Node = not quiet
    """
    if cosmic_web_data is None:
        return None
    
    web_type = cosmic_web_data.get('web_type', np.array([]))
    if len(web_type) == 0:
        return None
    
    # 0 (void) → 1.0, 3 (node) → 0.0
    quietness = 1 - web_type / 3.0
    
    return quietness


def compute_sfr_quietness(sfr_data: dict) -> np.ndarray:
    """
    Compute quietness from star formation rate.
    
    Lower SFR = less entropy production = more quiet
    """
    if sfr_data is None:
        return None
    
    sfr = sfr_data.get('sfr', np.array([]))
    if len(sfr) == 0:
        return None
    
    # Normalize by characteristic SFR
    sfr_char = 1.0  # Msun/yr
    quietness = np.exp(-sfr / sfr_char)
    
    return quietness


def compute_gw_quietness(gw_data: dict) -> np.ndarray:
    """
    Compute quietness from GW background intensity.
    
    Lower GW density = more quiet
    """
    if gw_data is None:
        return None
    
    rho_gw = gw_data.get('rho_gw_proxy', np.array([]))
    if len(rho_gw) == 0:
        return None
    
    rho_char = np.median(rho_gw[rho_gw > 0])
    quietness = np.exp(-rho_gw / rho_char)
    
    return quietness


def compute_curvature_gradient_quietness(lensing_data: dict) -> np.ndarray:
    """
    Compute quietness from curvature gradients (lensing).
    
    Lower |∇κ| = smoother potential = more quiet
    """
    if lensing_data is None:
        return None
    
    grad_kappa = lensing_data.get('grad_kappa_mag', np.array([]))
    if len(grad_kappa) == 0:
        return None
    
    grad_char = np.percentile(grad_kappa, 90)
    quietness = np.exp(-grad_kappa / grad_char)
    
    return quietness


# =============================================================================
# MAIN ANALYSIS PIPELINE
# =============================================================================

def run_analysis(quick: bool = False):
    """
    Run the full analysis pipeline.
    
    Parameters
    ----------
    quick : bool
        Use smaller samples for faster testing
    """
    print("\n" + "=" * 60)
    print("GRAVITATIONAL QUIETNESS CORRELATION ANALYSIS")
    print("=" * 60)
    
    # Import modules
    modules = import_modules()
    
    # Check data
    data_status = check_data_availability()
    print_data_status(data_status)
    
    # ==========================================================
    # STEP 1: Load Σ enhancement data (rotation curves)
    # ==========================================================
    print("\n" + "-" * 60)
    print("STEP 1: Loading rotation curve data")
    print("-" * 60)
    
    if not data_status['sparc']['available']:
        print("SPARC data not available. Run: python downloaders/download_sparc.py")
        print("Generating mock data for demonstration...")
        
        # Generate mock Σ data
        np.random.seed(42)
        n_points = 1000 if not quick else 100
        r_kpc = np.random.exponential(5, n_points)
        sigma = 1 + 2 * (r_kpc / 10)**0.5 + np.random.normal(0, 0.3, n_points)
        sigma = np.maximum(sigma, 0.5)
        g_bar = 10 / (1 + r_kpc)
        
        print(f"  Generated {n_points} mock data points")
    else:
        if 'sigma' in modules:
            galaxies = modules['sigma']['load']()
            for i in range(len(galaxies)):
                galaxies[i] = modules['sigma']['compute'](galaxies[i])
            r_kpc, sigma, g_bar, gal_id = modules['sigma']['export'](galaxies)
            print(f"  Loaded {len(r_kpc)} data points from {len(galaxies)} galaxies")
        else:
            print("  Sigma module not available")
            return
    
    # ==========================================================
    # STEP 2: Compute quietness variables
    # ==========================================================
    print("\n" + "-" * 60)
    print("STEP 2: Computing quietness variables")
    print("-" * 60)
    
    # For now, generate mock quietness variables based on r_kpc
    # In full analysis, these would come from real data
    
    n = len(r_kpc)
    
    # Mock quietness variables (correlated with radius as proxy)
    # In reality, these would be computed from actual data
    quietness_vars = {}
    
    # Velocity dispersion: lower at large r
    sigma_v_mock = 200 * np.exp(-r_kpc / 15) + np.random.normal(0, 20, n)
    quietness_vars['velocity_dispersion'] = 1 - np.clip(sigma_v_mock / 300, 0, 1)
    print(f"  ✓ Velocity dispersion (mock)")
    
    # Density: lower at large r
    delta_mock = 10 * np.exp(-r_kpc / 3) - 0.5 + np.random.normal(0, 0.5, n)
    quietness_vars['matter_density'] = 1 / (1 + np.exp(delta_mock))
    print(f"  ✓ Matter density (mock)")
    
    # Dynamical time: longer at large r
    t_dyn_mock = 0.5 + 0.3 * r_kpc + np.random.normal(0, 0.5, n)
    quietness_vars['dynamical_time'] = np.clip(t_dyn_mock / 5, 0, 1)
    print(f"  ✓ Dynamical time (mock)")
    
    # Tidal eigenvalue spread: lower in voids (large r)
    spread_mock = 2 * np.exp(-r_kpc / 8) + np.random.normal(0, 0.3, n)
    quietness_vars['tidal_spread'] = np.exp(-spread_mock / 2)
    print(f"  ✓ Tidal spread (mock)")
    
    # SFR: lower at large r
    sfr_mock = 5 * np.exp(-r_kpc / 5) + np.random.exponential(0.5, n)
    quietness_vars['sfr'] = np.exp(-sfr_mock / 2)
    print(f"  ✓ Star formation rate (mock)")
    
    # Curvature gradient: lower at large r
    grad_mock = 1 / (1 + r_kpc / 3) + np.random.exponential(0.1, n)
    quietness_vars['curvature_gradient'] = np.exp(-grad_mock * 3)
    print(f"  ✓ Curvature gradient (mock)")
    
    # Random control
    quietness_vars['random_control'] = np.random.uniform(0, 1, n)
    print(f"  ✓ Random control")
    
    # ==========================================================
    # STEP 3: Run correlation analysis
    # ==========================================================
    print("\n" + "-" * 60)
    print("STEP 3: Running correlation analysis")
    print("-" * 60)
    
    if 'correlation' in modules:
        comparison = modules['correlation']['compare'](sigma, quietness_vars)
        
        print("\nCorrelation Summary:")
        print(comparison.to_string(index=False))
        
        # Save results
        comparison.to_csv(OUTPUT_DIR / 'correlation_comparison.csv', index=False)
        print(f"\n  Saved to {OUTPUT_DIR / 'correlation_comparison.csv'}")
        
        # Identify best predictor
        best = modules['correlation']['best'](sigma, quietness_vars)
        
        print("\n" + "-" * 60)
        print("BEST PREDICTOR ANALYSIS")
        print("-" * 60)
        print(f"  Best by |r|:      {best['best_by_correlation']}")
        print(f"  Best by p-value:  {best['best_by_significance']}")
        print(f"  Best by fit R²:   {best['best_by_fit']}")
        
        # Generate plots for top variables
        print("\n  Generating plots...")
        for var_name in list(quietness_vars.keys())[:3]:
            try:
                modules['correlation']['plot'](
                    quietness_vars[var_name], sigma, var_name,
                    OUTPUT_DIR / f'correlation_{var_name}.png'
                )
            except Exception as e:
                print(f"    Could not plot {var_name}: {e}")
    
    # ==========================================================
    # STEP 4: Summary
    # ==========================================================
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    print(f"""
Results saved to: {OUTPUT_DIR}

Next steps:
1. Download real data: python downloaders/download_sparc.py
2. Download Gaia data: python downloaders/download_gaia.py
3. Re-run with real data: python run_all_tests.py

Key outputs:
- correlation_comparison.csv: All variable correlations
- correlation_*.png: Visualization for each variable
""")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test quietness variables against gravitational anomalies"
    )
    parser.add_argument('--download', action='store_true',
                        help='Download missing data files')
    parser.add_argument('--quick', action='store_true',
                        help='Use smaller samples for faster testing')
    
    args = parser.parse_args()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download data if requested
    if args.download:
        print("Downloading data...")
        # Import and run downloaders
        try:
            from downloaders.download_sparc import download_sparc_main_tables
            download_sparc_main_tables()
        except Exception as e:
            print(f"Could not download SPARC: {e}")
    
    # Run analysis
    run_analysis(quick=args.quick)
