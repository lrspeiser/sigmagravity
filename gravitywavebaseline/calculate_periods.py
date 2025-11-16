"""
STEP 1: Calculate Gravitational Wave Period Lengths for Every Gaia Star

This script:
1. Loads 1.8M Gaia MW stellar data (positions, masses, velocities)
2. Computes multiple period length hypotheses for each star
3. Saves results to file for use in inverse multiplier calculation

Output: gaia_with_periods.parquet
Columns: source_id, x, y, z, R, phi, M_star, v_phi, lambda_*, M_interior, rho_local
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path

try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("[OK] GPU (CuPy) available for calculations")
    device = cp.cuda.Device(0)
    print(f"  GPU: {device.compute_capability}")
    meminfo = device.mem_info
    print(f"  Memory: {meminfo[1] / 1e9:.1f} GB total, {meminfo[0] / 1e9:.1f} GB free")
except ImportError:
    cp = np
    GPU_AVAILABLE = False
    print("[!] GPU not available, using CPU (slower)")

# Constants
G_SI = 6.67430e-11  # m^3 kg^-1 s^-2
M_SUN = 1.989e30  # kg
KPC_TO_M = 3.086e19  # m
G_KPC = G_SI * M_SUN / (KPC_TO_M)**3 * 1e6  # kpc^3 / M_sun / (km/s)^2
G_KPC_ACTUAL = 4.498e-12  # More precise value

def load_gaia_data():
    """
    Load Gaia MW data from available files.
    
    Returns DataFrame with: source_id, R, z, phi, M_star, v_phi, x, y
    """
    
    print("\n" + "="*80)
    print("LOADING GAIA DATA")
    print("="*80)
    
    # Try different possible file locations
    possible_paths = [
        'data/gaia/gaia_processed_corrected.csv',
        'data/gaia/gaia_processed.csv',
        '../data/gaia/gaia_processed.csv',
    ]
    
    gaia = None
    for path in possible_paths:
        try:
            print(f"  Attempting to load: {path}")
            gaia = pd.read_csv(path)
            print(f"[OK] Loaded from: {path}")
            break
        except FileNotFoundError:
            continue
    
    if gaia is None:
        raise FileNotFoundError("Could not find gaia_processed.csv")
    
    print(f"  Total stars: {len(gaia):,}")
    print(f"  Columns: {list(gaia.columns)}")
    
    # Standardize column names
    if 'R_cyl' in gaia.columns and 'R' not in gaia.columns:
        gaia['R'] = gaia['R_cyl']
    
    # Ensure we have required columns
    required = ['R', 'z', 'phi', 'source_id']
    for col in required:
        if col not in gaia.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Use M_star_estimated if available, otherwise M_star
    if 'M_star_estimated' in gaia.columns:
        gaia['M_star'] = gaia['M_star_estimated']
    elif 'M_star' not in gaia.columns:
        # If no mass column, use default
        print("  Using default stellar mass of 1.0 M_sun")
        gaia['M_star'] = 1.0
    
    # Ensure M_star is reasonable
    gaia['M_star'] = gaia['M_star'].clip(0.08, 100.0)  # Physical limits
    
    # Compute Cartesian coordinates
    gaia['x'] = gaia['R'] * np.cos(gaia['phi'])
    gaia['y'] = gaia['R'] * np.sin(gaia['phi'])
    
    # Keep v_phi if it exists (observed velocities)
    has_vphi = 'v_phi' in gaia.columns
    
    print(f"\n  Final columns: {list(gaia.columns)}")
    print(f"  R range: {gaia['R'].min():.2f} - {gaia['R'].max():.2f} kpc")
    print(f"  z range: {gaia['z'].min():.2f} - {gaia['z'].max():.2f} kpc")
    print(f"  M_star range: {gaia['M_star'].min():.3f} - {gaia['M_star'].max():.3f} M_sun")
    print(f"  Total stellar mass: {gaia['M_star'].sum():.2e} M_sun")
    if has_vphi:
        vphi_data = gaia['v_phi'][gaia['v_phi'] != 0]
        if len(vphi_data) > 0:
            print(f"  v_phi range: {vphi_data.min():.1f} - {vphi_data.max():.1f} km/s")
            print(f"  Stars with v_phi: {len(vphi_data):,}")
    
    return gaia

def calculate_enclosed_mass_gpu(R, M_star, batch_size=50000):
    """
    Calculate mass interior to each star's radius.
    Uses GPU if available for faster computation.
    
    For 1.8M stars, this is O(N log N) via sorting method.
    """
    
    print("\nCalculating enclosed masses...")
    t0 = time.time()
    
    N = len(R)
    
    # Use sorting method (efficient for large N)
    print(f"  Processing {N:,} stars (sorting method)...")
    
    # Sort by radius
    sort_idx = np.argsort(R)
    R_sorted = R[sort_idx]
    M_sorted = M_star[sort_idx]
    
    # Cumulative sum
    M_interior_sorted = np.cumsum(M_sorted)
    
    # Unsort back to original order
    unsort_idx = np.argsort(sort_idx)
    M_interior = M_interior_sorted[unsort_idx]
    
    t1 = time.time()
    print(f"  [OK] Complete in {t1-t0:.1f}s")
    print(f"    M_interior range: {M_interior.min():.2e} - {M_interior.max():.2e} M_sun")
    
    return M_interior

def calculate_local_density_kdtree(x, y, z, M_star, r_neighbor=1.0, n_neighbors=100):
    """
    Estimate local density using k-nearest neighbors with KDTree.
    Much faster than pairwise distances for large datasets.
    
    Parameters:
    -----------
    x, y, z : arrays
        Cartesian coordinates (kpc)
    M_star : array
        Stellar masses (M_☉)
    r_neighbor : float
        Maximum radius for neighbors (kpc)
    n_neighbors : int
        Number of neighbors to use for density estimate
    """
    
    print(f"\nCalculating local densities (k-NN method, k={n_neighbors})...")
    t0 = time.time()
    
    from scipy.spatial import cKDTree
    
    N = len(x)
    
    # Build KDTree
    print(f"  Building KDTree for {N:,} stars...")
    positions = np.column_stack([x, y, z])
    tree = cKDTree(positions)
    
    # Query k-nearest neighbors
    print(f"  Querying {n_neighbors} nearest neighbors...")
    distances, indices = tree.query(positions, k=n_neighbors+1)  # +1 because star finds itself
    
    # Remove self (first neighbor)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    
    # Calculate local mass and volume
    M_local = np.sum(M_star[indices], axis=1)
    
    # Volume of sphere containing k neighbors
    r_max = distances[:, -1]  # Distance to farthest neighbor
    V_local = (4/3) * np.pi * r_max**3
    
    rho_local = M_local / V_local
    
    t1 = time.time()
    print(f"  [OK] Complete in {t1-t0:.1f}s")
    print(f"    rho_local range: {rho_local.min():.2e} - {rho_local.max():.2e} M_sun/kpc^3")
    
    return rho_local

def calculate_all_period_hypotheses(gaia):
    """
    Calculate multiple gravitational wave period length hypotheses.
    
    These are different physical scales that might relate to gravity enhancement.
    """
    
    print("\n" + "="*80)
    print("CALCULATING PERIOD LENGTH HYPOTHESES")
    print("="*80)
    
    R = gaia['R'].values
    z = gaia['z'].values
    M_star = gaia['M_star'].values
    x = gaia['x'].values
    y = gaia['y'].values
    
    # Calculate derived quantities
    print("\n[1/2] Calculating enclosed masses...")
    M_interior = calculate_enclosed_mass_gpu(R, M_star)
    gaia['M_interior'] = M_interior
    
    print("\n[2/2] Calculating local densities...")
    rho_local = calculate_local_density_kdtree(x, y, z, M_star, n_neighbors=50)
    gaia['rho_local'] = rho_local
    
    print("\n" + "-"*80)
    print("Calculating period hypotheses...")
    print("-"*80)
    
    # Hypothesis 1: ORBITAL PERIOD -> wavelength
    # lambda = 2*pi*R (circumference of orbit)
    lambda_orbital = 2 * np.pi * R
    gaia['lambda_orbital'] = lambda_orbital
    print(f"  1. Orbital (2*pi*R): median={np.median(lambda_orbital):.2f} kpc")
    
    # Hypothesis 2: DYNAMICAL TIME -> wavelength  
    # t_dyn = sqrt(R^3/GM_interior), lambda ~ R (characteristic scale)
    lambda_dynamical = R
    gaia['lambda_dynamical'] = lambda_dynamical
    print(f"  2. Dynamical (R): median={np.median(lambda_dynamical):.2f} kpc")
    
    # Hypothesis 3: JEANS LENGTH (pressure support scale)
    # lambda_J = c_s / sqrt(G*rho), where c_s ~ 10 km/s (ISM sound speed)
    c_sound = 10.0  # km/s
    lambda_jeans = c_sound / np.sqrt(G_KPC_ACTUAL * rho_local + 1e-10)
    lambda_jeans = np.clip(lambda_jeans, 0.01, 1000.0)
    gaia['lambda_jeans'] = lambda_jeans
    print(f"  3. Jeans (c_s/sqrt(G*rho)): median={np.median(lambda_jeans):.2f} kpc")
    
    # Hypothesis 4: MASS-DEPENDENT
    # lambda ~ M_star^alpha (heavier stars -> longer periods?)
    lambda_mass = M_star ** 0.5  # alpha = 0.5
    gaia['lambda_mass'] = lambda_mass
    print(f"  4. Mass-dependent (M^0.5): median={np.median(lambda_mass):.2f} kpc")
    
    # Hypothesis 5: HYBRID (mass x radius)
    # lambda ~ sqrt(M x R)
    lambda_hybrid = np.sqrt(M_star * R)
    gaia['lambda_hybrid'] = lambda_hybrid
    print(f"  5. Hybrid (sqrt(M*R)): median={np.median(lambda_hybrid):.2f} kpc")
    
    # Hypothesis 6: GRAVITATIONAL WAVE FREQUENCY
    # For binary-like oscillation: f_GW ~ sqrt(GM/R^3)
    # lambda_GW = v_circ / f_GW
    v_circ = np.sqrt(G_KPC_ACTUAL * (M_interior + M_star) / (R + 0.1))
    f_gw = v_circ / (2 * np.pi * R)
    lambda_gw = v_circ / (f_gw + 1e-10)
    lambda_gw = np.clip(lambda_gw, 0.01, 1000.0)
    gaia['lambda_gw'] = lambda_gw
    print(f"  6. GW frequency (v/f): median={np.median(lambda_gw):.2f} kpc")
    
    # Hypothesis 7: SCALE HEIGHT (vertical oscillation)
    # lambda = h(R), typical disk scale height
    # Approximate: h ~ 0.3 kpc at solar radius, grows with R
    lambda_scale_height = 0.3 * (R / 8.0) ** 0.5
    lambda_scale_height = np.clip(lambda_scale_height, 0.05, 10.0)
    gaia['lambda_scale_height'] = lambda_scale_height
    print(f"  7. Scale height (0.3*sqrt(R/8)): median={np.median(lambda_scale_height):.2f} kpc")
    
    # Hypothesis 8: TOOMRE LENGTH (stability scale)
    # Q = sigma*kappa / (pi*G*Sigma), where sigma = velocity dispersion
    sigma_v = 30.0  # km/s, typical velocity dispersion
    kappa = v_circ / R  # epicyclic frequency (approximate)
    Sigma_local = rho_local * 2 * np.abs(z + 0.3)  # Approximate surface density
    lambda_toomre = sigma_v * kappa / (np.pi * G_KPC_ACTUAL * Sigma_local + 1e-10)
    lambda_toomre = np.clip(lambda_toomre, 0.01, 1000.0)
    gaia['lambda_toomre'] = lambda_toomre
    print(f"  8. Toomre (stability): median={np.median(lambda_toomre):.2f} kpc")
    
    print("\n[OK] All period hypotheses calculated")
    
    return gaia

def save_results(gaia, output_path='gaia_with_periods.parquet'):
    """
    Save results to file.
    
    Uses parquet for efficiency with large datasets.
    """
    
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Select columns to save
    period_columns = [col for col in gaia.columns if col.startswith('lambda_')]
    
    base_columns = ['source_id', 'R', 'z', 'phi', 'x', 'y', 'M_star', 'M_interior', 'rho_local']
    
    # Include v_phi if it exists
    if 'v_phi' in gaia.columns:
        base_columns.append('v_phi')
    
    save_columns = base_columns + period_columns
    
    # Filter to only existing columns
    save_columns = [col for col in save_columns if col in gaia.columns]
    
    gaia_save = gaia[save_columns].copy()
    
    print(f"\nSaving {len(gaia_save):,} stars with {len(save_columns)} columns:")
    print(f"  Columns: {save_columns}")
    
    # Save as parquet
    gaia_save.to_parquet(output_path, index=False, compression='snappy')
    print(f"\n[OK] Saved to: {output_path}")
    
    # Report file size
    size_mb = Path(output_path).stat().st_size / (1024**2)
    print(f"  File size: {size_mb:.1f} MB")
    
    print(f"\nThis file can now be used in the inverse multiplier calculation!")
    
    return gaia_save

def main():
    """Run complete period calculation pipeline."""
    
    print("="*80)
    print("GRAVITATIONAL WAVE PERIOD CALCULATION FOR 1.8M GAIA STARS")
    print("="*80)
    print("\nThis script calculates multiple period length hypotheses (lambda) for each star.")
    print("These will be used to test which periods, when used as gravity multipliers,")
    print("can reproduce the observed Milky Way rotation curve.")
    
    t_start = time.time()
    
    # Step 1: Load data
    gaia = load_gaia_data()
    
    # Step 2: Calculate periods
    gaia = calculate_all_period_hypotheses(gaia)
    
    # Step 3: Save results
    output_path = 'gravitywavebaseline/gaia_with_periods.parquet'
    gaia_save = save_results(gaia, output_path)
    
    t_end = time.time()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\n[OK] Processing complete in {t_end - t_start:.1f} seconds")
    print(f"[OK] Processed {len(gaia):,} stars")
    print(f"[OK] Calculated {len([c for c in gaia.columns if c.startswith('lambda_')])} period hypotheses per star")
    
    print("\nPeriod statistics (median values):")
    for col in sorted(gaia.columns):
        if col.startswith('lambda_'):
            print(f"  {col:25s}: {gaia[col].median():>10.2f} kpc "
                  f"(range: {gaia[col].min():.2f}-{gaia[col].max():.2f})")
    
    print(f"\n[OK] Output file ready for inverse multiplier calculation:")
    print(f"  {output_path}")
    
    return gaia_save

if __name__ == "__main__":
    gaia_with_periods = main()

