"""
Download Gaia DR3 stellar data via TAP queries.

Source: ESA Gaia Archive (https://gea.esac.esa.int/archive/)
Reference: Gaia Collaboration (2023)

We use ADQL queries rather than bulk downloads to get manageable subsets.
Key tables:
    - gaiadr3.gaia_source: Main source catalog (1.8 billion sources)
    - gaiadr3.astrophysical_parameters: Stellar parameters
    
For gravitational quietness tests, we need:
    - 6D phase space (position + velocity) for dynamical timescales
    - Spatial density for matter density gradients
    - Velocity dispersion for metric fluctuations proxy
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent to path for config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GAIA_DIR, GAIA_MAX_ROWS, GAIA_PARALLAX_MIN, GAIA_RUWE_MAX, GAIA_SAMPLE_REGIONS

# =============================================================================
# TAP QUERY SETUP
# =============================================================================

GAIA_TAP_URL = "https://gea.esac.esa.int/tap-server/tap"

def setup_gaia_tap():
    """Set up connection to Gaia TAP service."""
    try:
        from astroquery.gaia import Gaia
        Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
        Gaia.ROW_LIMIT = GAIA_MAX_ROWS
        return Gaia
    except ImportError:
        print("ERROR: astroquery not installed")
        print("Run: pip install astroquery")
        sys.exit(1)


# =============================================================================
# QUERY TEMPLATES
# =============================================================================

# Query for 6D phase space data (full kinematics)
QUERY_6D_PHASE_SPACE = """
SELECT 
    source_id,
    ra, dec,
    l, b,
    parallax, parallax_error,
    pmra, pmra_error,
    pmdec, pmdec_error,
    radial_velocity, radial_velocity_error,
    ruwe,
    phot_g_mean_mag,
    bp_rp
FROM gaiadr3.gaia_source
WHERE 
    parallax > {parallax_min}
    AND parallax_error/parallax < 0.2
    AND ruwe < {ruwe_max}
    AND radial_velocity IS NOT NULL
    AND l BETWEEN {l_min} AND {l_max}
    AND b BETWEEN {b_min} AND {b_max}
    AND 1000.0/parallax < {dist_max}
"""

# Query for density mapping (positions only, larger sample)
QUERY_DENSITY_MAP = """
SELECT 
    source_id,
    l, b,
    parallax, parallax_error,
    phot_g_mean_mag
FROM gaiadr3.gaia_source
WHERE 
    parallax > {parallax_min}
    AND parallax_error/parallax < 0.3
    AND l BETWEEN {l_min} AND {l_max}
    AND b BETWEEN {b_min} AND {b_max}
"""

# Query for velocity dispersion in a region
QUERY_VELOCITY_DISPERSION = """
SELECT 
    source_id,
    l, b,
    parallax,
    pmra, pmdec,
    radial_velocity
FROM gaiadr3.gaia_source
WHERE 
    parallax > {parallax_min}
    AND parallax_error/parallax < 0.1
    AND ruwe < {ruwe_max}
    AND radial_velocity IS NOT NULL
    AND SQRT(POWER(l - {l_center}, 2) + POWER(b - {b_center}, 2)) < {radius}
"""


def run_gaia_query(query: str, output_file: Path, Gaia) -> pd.DataFrame:
    """
    Execute a Gaia TAP query and save results.
    
    Parameters
    ----------
    query : str
        ADQL query string
    output_file : Path
        Where to save results (CSV or Parquet)
    Gaia : module
        astroquery.gaia.Gaia module
    
    Returns
    -------
    pd.DataFrame
        Query results
    """
    print(f"  Executing query...")
    print(f"  (This may take several minutes for large queries)")
    
    try:
        job = Gaia.launch_job_async(query)
        results = job.get_results()
        
        # Convert to pandas
        df = results.to_pandas()
        
        # Save
        if output_file.suffix == '.parquet':
            df.to_parquet(output_file, index=False)
        else:
            df.to_csv(output_file, index=False)
        
        print(f"  Saved {len(df)} rows to {output_file}")
        return df
        
    except Exception as e:
        print(f"  Query failed: {e}")
        return pd.DataFrame()


# =============================================================================
# DOWNLOAD FUNCTIONS FOR EACH REGION
# =============================================================================

def download_6d_sample(region: dict, Gaia):
    """Download 6D phase space data for a region."""
    name = region['name']
    output_file = GAIA_DIR / f"gaia_6d_{name}.parquet"
    
    if output_file.exists():
        print(f"  {name}: already downloaded")
        return
    
    print(f"\nDownloading 6D data for {name}...")
    
    query = QUERY_6D_PHASE_SPACE.format(
        parallax_min=GAIA_PARALLAX_MIN,
        ruwe_max=GAIA_RUWE_MAX,
        l_min=region['l_min'],
        l_max=region['l_max'],
        b_min=region['b_min'],
        b_max=region['b_max'],
        dist_max=region['dist_max']
    )
    
    run_gaia_query(query, output_file, Gaia)


def download_density_sample(region: dict, Gaia):
    """Download position data for density mapping."""
    name = region['name']
    output_file = GAIA_DIR / f"gaia_density_{name}.parquet"
    
    if output_file.exists():
        print(f"  {name}: already downloaded")
        return
    
    print(f"\nDownloading density data for {name}...")
    
    query = QUERY_DENSITY_MAP.format(
        parallax_min=GAIA_PARALLAX_MIN / 2,  # Go deeper for density
        l_min=region['l_min'],
        l_max=region['l_max'],
        b_min=region['b_min'],
        b_max=region['b_max']
    )
    
    run_gaia_query(query, output_file, Gaia)


# =============================================================================
# UTILITY: CONVERT GAIA OBSERVABLES TO PHYSICAL COORDINATES
# =============================================================================

def gaia_to_galactocentric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Gaia observables to Galactocentric coordinates.
    
    Adds columns:
        - x, y, z: Galactocentric Cartesian (kpc)
        - vx, vy, vz: Galactocentric velocities (km/s)
        - r_gal: Galactocentric radius (kpc)
        - v_total: Total velocity (km/s)
    
    Uses astropy coordinates for proper transformation.
    """
    from astropy.coordinates import SkyCoord, Galactocentric
    import astropy.units as u
    
    # Create SkyCoord from Gaia data
    coords = SkyCoord(
        ra=df['ra'].values * u.deg,
        dec=df['dec'].values * u.deg,
        distance=(1000.0 / df['parallax'].values) * u.pc,
        pm_ra_cosdec=df['pmra'].values * u.mas/u.yr,
        pm_dec=df['pmdec'].values * u.mas/u.yr,
        radial_velocity=df['radial_velocity'].values * u.km/u.s,
        frame='icrs'
    )
    
    # Transform to Galactocentric
    galcen = coords.galactocentric
    
    # Add to dataframe
    df = df.copy()
    df['x_kpc'] = galcen.x.to(u.kpc).value
    df['y_kpc'] = galcen.y.to(u.kpc).value
    df['z_kpc'] = galcen.z.to(u.kpc).value
    df['vx'] = galcen.v_x.to(u.km/u.s).value
    df['vy'] = galcen.v_y.to(u.km/u.s).value
    df['vz'] = galcen.v_z.to(u.km/u.s).value
    
    df['r_gal'] = np.sqrt(df['x_kpc']**2 + df['y_kpc']**2 + df['z_kpc']**2)
    df['v_total'] = np.sqrt(df['vx']**2 + df['vy']**2 + df['vz']**2)
    
    return df


def compute_local_velocity_dispersion(df: pd.DataFrame, 
                                       n_neighbors: int = 50) -> pd.DataFrame:
    """
    Compute local velocity dispersion for each star.
    
    Uses k-nearest neighbors in position space to estimate local σ_v.
    This serves as a proxy for metric fluctuation amplitude.
    """
    from sklearn.neighbors import NearestNeighbors
    
    # Fit neighbors in position space
    positions = df[['x_kpc', 'y_kpc', 'z_kpc']].values
    velocities = df[['vx', 'vy', 'vz']].values
    
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(positions)
    
    # For each star, find neighbors and compute velocity dispersion
    distances, indices = nn.kneighbors(positions)
    
    sigma_v = np.zeros(len(df))
    for i in range(len(df)):
        neighbor_vels = velocities[indices[i]]
        mean_vel = np.mean(neighbor_vels, axis=0)
        sigma_v[i] = np.sqrt(np.mean(np.sum((neighbor_vels - mean_vel)**2, axis=1)))
    
    df = df.copy()
    df['sigma_v_local'] = sigma_v
    
    return df


# =============================================================================
# PRE-COMPUTED SAMPLES (for quick testing without full download)
# =============================================================================

def download_precomputed_samples():
    """
    Download pre-computed Gaia samples if available.
    
    These are smaller, curated datasets for quick testing:
    - Milky Way halo stars with 6D kinematics
    - Solar neighborhood with full phase space
    """
    # URLs for any pre-computed samples (if hosted somewhere)
    # For now, we'll generate our own via queries
    
    print("No pre-computed samples available.")
    print("Use the TAP queries to download custom samples.")


# =============================================================================
# VERIFY DOWNLOADS
# =============================================================================

def verify_downloads():
    """Check what Gaia data is available."""
    print("\n" + "=" * 60)
    print("Verifying Gaia data")
    print("=" * 60)
    
    parquet_files = list(GAIA_DIR.glob("*.parquet"))
    csv_files = list(GAIA_DIR.glob("*.csv"))
    
    if not parquet_files and not csv_files:
        print("  No Gaia data downloaded yet")
        return
    
    for f in parquet_files + csv_files:
        if f.suffix == '.parquet':
            df = pd.read_parquet(f)
        else:
            df = pd.read_csv(f)
        print(f"  {f.name}: {len(df):,} rows, {len(df.columns)} columns")
    
    print("\nGaia data available for analysis!")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Gaia DR3 Data Downloader")
    print("=" * 60)
    print(f"Data directory: {GAIA_DIR}")
    print(f"Max rows per query: {GAIA_MAX_ROWS:,}")
    print()
    
    # Set up TAP connection
    Gaia = setup_gaia_tap()
    
    # Download each region
    print("\nDownloading sample regions:")
    for region in GAIA_SAMPLE_REGIONS:
        print(f"\n--- {region['name']} ---")
        download_6d_sample(region, Gaia)
    
    # Verify
    verify_downloads()
    
    print("\n" + "=" * 60)
    print("USAGE NOTE:")
    print("=" * 60)
    print("""
After downloading, use gaia_to_galactocentric() to convert
to physical coordinates, then compute_local_velocity_dispersion()
to get the metric fluctuation proxy.

Example:
    df = pd.read_parquet('gaia_6d_halo_north.parquet')
    df = gaia_to_galactocentric(df)
    df = compute_local_velocity_dispersion(df)
    
    # Now df has sigma_v_local for correlation with Σ
""")
