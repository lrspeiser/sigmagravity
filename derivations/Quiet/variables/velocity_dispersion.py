"""
Velocity Dispersion Computation
================================

Computes local velocity dispersion σ_v(r) from Gaia stellar kinematics.
This measures "metric fluctuations" - how noisy the local gravitational
environment is.

Theory: In Σ-Gravity, high σ_v implies decoherence of graviton states,
reducing the enhancement factor K.

Expected correlation: K decreases with increasing σ_v

Data source: Gaia DR3 with radial velocities (1.8M stars already downloaded)
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import sys

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GAIA_DIR, G_GRAV

try:
    import cupy as cp
    USE_GPU = True
except ImportError:
    cp = np
    USE_GPU = False


def load_gaia_kinematics(data_path: Optional[Path] = None) -> Dict[str, np.ndarray]:
    """
    Load Gaia DR3 6D phase space data.
    
    Returns dict with: ra, dec, parallax, pmra, pmdec, radial_velocity, distance
    """
    if data_path is None:
        # Check common locations
        candidates = [
            GAIA_DIR / "gaia_rv_sample.csv",
            GAIA_DIR / "gaia_kinematics.npy",
            Path("C:/Users/henry/dev/sigmagravity/data/gaia_rv_sample.csv"),
        ]
        for p in candidates:
            if p.exists():
                data_path = p
                break
    
    if data_path is None or not data_path.exists():
        print(f"No Gaia data found. Creating synthetic sample for testing.")
        return create_synthetic_gaia_sample()
    
    if data_path.suffix == '.csv':
        import pandas as pd
        df = pd.read_csv(data_path)
        return {
            'ra': df['ra'].values,
            'dec': df['dec'].values,
            'parallax': df['parallax'].values,
            'pmra': df['pmra'].values,
            'pmdec': df['pmdec'].values,
            'radial_velocity': df['radial_velocity'].values if 'radial_velocity' in df else np.zeros(len(df)),
            'distance': 1000 / df['parallax'].values,  # pc
        }
    else:
        data = np.load(data_path, allow_pickle=True).item()
        return data


def create_synthetic_gaia_sample(n_stars: int = 100000) -> Dict[str, np.ndarray]:
    """Create synthetic Gaia-like sample for testing."""
    np.random.seed(42)
    
    # Distances (pc) - exponential disk
    distances = np.random.exponential(500, n_stars)
    distances = distances[distances < 10000]
    n = len(distances)
    
    # Positions
    ra = np.random.uniform(0, 360, n)
    dec = np.random.uniform(-90, 90, n)
    
    # Velocities - disk-like kinematics
    # Azimuthal: ~220 km/s + dispersion
    # Radial: near zero + dispersion
    # Vertical: near zero + dispersion
    v_phi = 220 + np.random.normal(0, 30, n)
    v_r = np.random.normal(0, 40, n)
    v_z = np.random.normal(0, 20, n)
    
    # Add bulge/halo component for some stars
    halo_frac = distances > 3000
    v_phi[halo_frac] = np.random.normal(0, 150, halo_frac.sum())
    v_r[halo_frac] = np.random.normal(0, 150, halo_frac.sum())
    v_z[halo_frac] = np.random.normal(0, 150, halo_frac.sum())
    
    return {
        'ra': ra,
        'dec': dec,
        'parallax': 1000 / distances,  # mas
        'pmra': (v_phi * 4.74 / distances),  # mas/yr
        'pmdec': (v_z * 4.74 / distances),  # mas/yr
        'radial_velocity': v_r,  # km/s
        'distance': distances,
    }


def compute_3d_velocities(data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert proper motions + RV to 3D Galactocentric velocities.
    
    Returns: vx, vy, vz in km/s
    """
    xp = cp if USE_GPU else np
    
    ra = xp.asarray(data['ra']) * np.pi / 180
    dec = xp.asarray(data['dec']) * np.pi / 180
    dist = xp.asarray(data['distance'])  # pc
    pmra = xp.asarray(data['pmra'])  # mas/yr
    pmdec = xp.asarray(data['pmdec'])  # mas/yr
    rv = xp.asarray(data['radial_velocity'])  # km/s
    
    # Proper motion to km/s: v = 4.74047 * d(kpc) * pm(mas/yr)
    dist_kpc = dist / 1000
    v_ra = 4.74047 * dist_kpc * pmra
    v_dec = 4.74047 * dist_kpc * pmdec
    
    # Convert to Galactocentric Cartesian
    # Simplified: assume Sun at (8, 0, 0) kpc with (U,V,W) = (11, 232, 7) km/s
    cos_ra, sin_ra = xp.cos(ra), xp.sin(ra)
    cos_dec, sin_dec = xp.cos(dec), xp.sin(dec)
    
    # Heliocentric Cartesian
    x_hel = dist_kpc * cos_dec * cos_ra
    y_hel = dist_kpc * cos_dec * sin_ra
    z_hel = dist_kpc * sin_dec
    
    vx_hel = rv * cos_dec * cos_ra - v_ra * sin_ra - v_dec * sin_dec * cos_ra
    vy_hel = rv * cos_dec * sin_ra + v_ra * cos_ra - v_dec * sin_dec * sin_ra
    vz_hel = rv * sin_dec + v_dec * cos_dec
    
    # Transform to Galactocentric (simplified)
    R_sun = 8.0  # kpc
    V_sun = xp.array([11.0, 232.0, 7.0])  # km/s
    
    x_gal = x_hel - R_sun
    y_gal = y_hel
    z_gal = z_hel
    
    vx_gal = vx_hel + V_sun[0]
    vy_gal = vy_hel + V_sun[1]
    vz_gal = vz_hel + V_sun[2]
    
    if USE_GPU:
        return vx_gal.get(), vy_gal.get(), vz_gal.get()
    return vx_gal, vy_gal, vz_gal


def compute_velocity_dispersion(data: Dict[str, np.ndarray],
                                 r_bins: np.ndarray = None,
                                 method: str = 'radial') -> Dict[str, np.ndarray]:
    """
    Compute velocity dispersion σ_v as a function of radius.
    
    Args:
        data: Gaia kinematic data
        r_bins: Radial bins in kpc (default: 0.5 to 20 kpc)
        method: 'radial' for σ_r, 'total' for sqrt(σ_x² + σ_y² + σ_z²)/sqrt(3)
    
    Returns:
        Dictionary with:
            r_mid: Midpoint of radial bins (kpc)
            sigma_v: Velocity dispersion (km/s)
            n_stars: Number of stars in each bin
    """
    xp = cp if USE_GPU else np
    
    if r_bins is None:
        r_bins = np.array([0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30])
    
    # Get 3D velocities
    vx, vy, vz = compute_3d_velocities(data)
    
    # Compute Galactocentric radius
    dist = data['distance'] / 1000  # kpc
    ra = data['ra'] * np.pi / 180
    dec = data['dec'] * np.pi / 180
    
    x = dist * np.cos(dec) * np.cos(ra) - 8.0  # Subtract Sun's position
    y = dist * np.cos(dec) * np.sin(ra)
    z = dist * np.sin(dec)
    r_gal = np.sqrt(x**2 + y**2 + z**2)
    
    # Move to GPU if available
    r_gal = xp.asarray(r_gal)
    vx, vy, vz = xp.asarray(vx), xp.asarray(vy), xp.asarray(vz)
    
    # Compute dispersion in each bin
    r_mid = []
    sigma_v = []
    n_stars = []
    
    for i in range(len(r_bins) - 1):
        mask = (r_gal >= r_bins[i]) & (r_gal < r_bins[i + 1])
        n = int(xp.sum(mask))
        
        if n > 10:
            if method == 'radial':
                # Compute radial velocity component
                x_bin = x[mask.get() if USE_GPU else mask]
                y_bin = y[mask.get() if USE_GPU else mask]
                z_bin = z[mask.get() if USE_GPU else mask]
                r_bin = np.sqrt(x_bin**2 + y_bin**2 + z_bin**2)
                
                vx_bin = vx[mask]
                vy_bin = vy[mask]
                vz_bin = vz[mask]
                
                # Radial velocity: v·r̂
                v_radial = (vx_bin * xp.asarray(x_bin) + vy_bin * xp.asarray(y_bin) + vz_bin * xp.asarray(z_bin)) / xp.asarray(r_bin)
                sigma = float(xp.std(v_radial))
            else:
                # Total 3D dispersion
                sigma_x = float(xp.std(vx[mask]))
                sigma_y = float(xp.std(vy[mask]))
                sigma_z = float(xp.std(vz[mask]))
                sigma = np.sqrt((sigma_x**2 + sigma_y**2 + sigma_z**2) / 3)
            
            r_mid.append((r_bins[i] + r_bins[i + 1]) / 2)
            sigma_v.append(sigma)
            n_stars.append(n)
    
    return {
        'r_mid': np.array(r_mid),
        'sigma_v': np.array(sigma_v),
        'n_stars': np.array(n_stars),
    }


def compute_local_sigma(data: Dict[str, np.ndarray],
                        target_coords: np.ndarray,
                        search_radius: float = 0.5) -> np.ndarray:
    """
    Compute local velocity dispersion at specific coordinates.
    
    Args:
        data: Gaia kinematic data
        target_coords: (N, 3) array of (x, y, z) coordinates in kpc
        search_radius: Radius for local average (kpc)
    
    Returns:
        sigma_v at each target location
    """
    xp = cp if USE_GPU else np
    
    vx, vy, vz = compute_3d_velocities(data)
    
    dist = data['distance'] / 1000
    ra = data['ra'] * np.pi / 180
    dec = data['dec'] * np.pi / 180
    
    x = dist * np.cos(dec) * np.cos(ra) - 8.0
    y = dist * np.cos(dec) * np.sin(ra)
    z = dist * np.sin(dec)
    
    # GPU acceleration for distance calculations
    x, y, z = xp.asarray(x), xp.asarray(y), xp.asarray(z)
    vx, vy, vz = xp.asarray(vx), xp.asarray(vy), xp.asarray(vz)
    target_coords = xp.asarray(target_coords)
    
    sigma_local = []
    
    for i in range(len(target_coords)):
        tx, ty, tz = target_coords[i]
        dist_sq = (x - tx)**2 + (y - ty)**2 + (z - tz)**2
        mask = dist_sq < search_radius**2
        
        n = int(xp.sum(mask))
        if n > 5:
            sigma_x = float(xp.std(vx[mask]))
            sigma_y = float(xp.std(vy[mask]))
            sigma_z = float(xp.std(vz[mask]))
            sigma = np.sqrt((sigma_x**2 + sigma_y**2 + sigma_z**2) / 3)
        else:
            sigma = np.nan
        
        sigma_local.append(sigma)
    
    return np.array(sigma_local)


def run_velocity_dispersion_analysis():
    """Run the full velocity dispersion analysis."""
    print("=" * 70)
    print("   VELOCITY DISPERSION ANALYSIS")
    print("=" * 70)
    
    # Load data
    print("\nLoading Gaia kinematic data...")
    data = load_gaia_kinematics()
    n_stars = len(data['ra'])
    print(f"  Loaded {n_stars:,} stars")
    
    # Compute σ_v(r)
    print("\nComputing velocity dispersion profile...")
    result = compute_velocity_dispersion(data)
    
    print("\nResults:")
    print("  R (kpc)  |  σ_v (km/s)  |  N_stars")
    print("-" * 40)
    for r, s, n in zip(result['r_mid'], result['sigma_v'], result['n_stars']):
        print(f"  {r:6.1f}   |   {s:6.1f}     |  {n:7,}")
    
    # Expected trend: σ_v increases outward (disk → halo transition)
    print("\n" + "=" * 70)
    print("   INTERPRETATION FOR Σ-GRAVITY")
    print("=" * 70)
    print("""
    In the Σ-Gravity framework, velocity dispersion σ_v measures
    "metric fluctuations" - the local gravitational noise level.
    
    Expected correlation with enhancement K:
      - Low σ_v (quiet environment): Strong coherence → High K
      - High σ_v (noisy environment): Decoherence → Low K
    
    This explains why:
      - Inner disk (low σ_v): Enhancement is active
      - Halo (high σ_v): Enhancement suppressed
""")
    
    return result


if __name__ == "__main__":
    run_velocity_dispersion_analysis()
