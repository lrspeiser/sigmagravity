"""
Download weak lensing data for curvature gradient analysis.

Primary sources:
1. DES Y3 - Dark Energy Survey Year 3 shape catalogs
2. KiDS - Kilo-Degree Survey 
3. HSC - Hyper Suprime-Cam
4. CLASH - Cluster Lensing And Supernova survey with Hubble

Weak lensing shear γ relates to the projected mass (convergence κ):
    κ = Σ / Σ_crit  (surface mass density / critical density)
    γ = complex shear from image ellipticities

For curvature gradients, we compute ∇κ from shear maps.
"""

import sys
from pathlib import Path
import numpy as np
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import LENSING_DIR

# =============================================================================
# DATA SOURCES
# =============================================================================

LENSING_SOURCES = {
    # DES Y3 - requires NCSA authentication, so we use public catalogs
    "des_y3": {
        "description": "Dark Energy Survey Year 3 weak lensing",
        "url_base": "https://des.ncsa.illinois.edu/releases/y3a2/",
        "auth_required": True,
        "size_gb": 10,
        "columns": ["ra", "dec", "e1", "e2", "weight", "z_phot"],
    },
    
    # KiDS DR4 - public access
    "kids_dr4": {
        "description": "Kilo-Degree Survey DR4",
        "url_base": "https://kids.strw.leidenuniv.nl/DR4/",
        "catalogs": [
            "KiDS_DR4.0_ugriZYJHKs_SOM_gold_WL_cat.fits"
        ],
        "auth_required": False,
        "size_gb": 5,
    },
    
    # CLASH cluster lensing - HST data, public
    "clash": {
        "description": "Cluster Lensing And Supernova survey with Hubble",
        "url_base": "https://archive.stsci.edu/prepds/clash/",
        "catalog_url": "https://archive.stsci.edu/hlsps/clash/",
        "auth_required": False,
        "size_gb": 2,
    },
}

# CLASH cluster list (25 clusters)
CLASH_CLUSTERS = [
    "abell383", "abell209", "abell1423", "abell2261", "abell611",
    "macs0329", "macs0429", "macs0744", "macs1115", "macs1149",
    "macs1206", "macs1311", "macs1423", "macs1720", "macs1931",
    "macs2129", "ms2137", "rxj1347", "rxj1532", "rxj2129",
    "rxj2248", "clj1226", "abell1835", "abell370", "zw1358"
]


# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================

def download_file(url: str, dest: Path, chunk_size: int = 8192) -> bool:
    """Download with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))
        
        with open(dest, 'wb') as f:
            with tqdm(total=total, unit='B', unit_scale=True, desc=dest.name) as pbar:
                for chunk in response.iter_content(chunk_size):
                    f.write(chunk)
                    pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def download_clash_catalogs():
    """
    Download CLASH cluster lensing catalogs.
    
    These provide mass maps and shear catalogs for 25 galaxy clusters.
    Excellent for testing Σ-enhancement in cluster environments.
    """
    print("\n" + "=" * 60)
    print("Downloading CLASH cluster lensing data")
    print("=" * 60)
    
    clash_dir = LENSING_DIR / "clash"
    clash_dir.mkdir(exist_ok=True)
    
    # Download mass model catalogs for each cluster
    base_url = "https://archive.stsci.edu/hlsps/clash/"
    
    for cluster in tqdm(CLASH_CLUSTERS, desc="CLASH clusters"):
        cluster_dir = clash_dir / cluster
        cluster_dir.mkdir(exist_ok=True)
        
        # Mass map file pattern
        # Note: actual URLs may vary, this is a template
        files_to_try = [
            f"{cluster}_massmap.fits",
            f"{cluster}_kappa.fits",
            f"{cluster}_gamma.fits",
        ]
        
        for filename in files_to_try:
            url = f"{base_url}{cluster}/{filename}"
            dest = cluster_dir / filename
            
            if dest.exists():
                continue
            
            # Try to download (may fail if file doesn't exist)
            try:
                response = requests.head(url, timeout=10)
                if response.status_code == 200:
                    download_file(url, dest)
            except:
                pass
    
    print(f"\nCLASH data saved to {clash_dir}")


def download_kids_public_sample():
    """
    Download a public KiDS weak lensing sample.
    
    The full KiDS DR4 is ~100GB, so we download a smaller sample
    or provide instructions for the full catalog.
    """
    print("\n" + "=" * 60)
    print("KiDS Weak Lensing Data")
    print("=" * 60)
    
    kids_dir = LENSING_DIR / "kids"
    kids_dir.mkdir(exist_ok=True)
    
    print("""
KiDS DR4 full catalog requires ~100GB.
For full access, visit: https://kids.strw.leidenuniv.nl/DR4/

For quick testing, we provide a smaller area sample.
""")
    
    # Download a sample region if available
    # (In practice, you'd need to query their database)
    sample_info_file = kids_dir / "README.txt"
    with open(sample_info_file, 'w') as f:
        f.write("""KiDS DR4 Weak Lensing Data
==========================

Full catalog: https://kids.strw.leidenuniv.nl/DR4/
Data access: https://kids.strw.leidenuniv.nl/DR4/data_files.php

Key files:
- KiDS_DR4.0_ugriZYJHKs_SOM_gold_WL_cat.fits (main shape catalog)
- Contains: RA, Dec, e1, e2, weight, photo-z

For curvature gradient analysis:
1. Download shape catalog
2. Create shear maps on a grid
3. Convert γ → κ using Kaiser-Squires
4. Compute ∇κ for curvature gradients

Python example:
    from astropy.io import fits
    hdu = fits.open('KiDS_cat.fits')
    data = hdu[1].data
    ra, dec = data['ALPHA_J2000'], data['DELTA_J2000']
    e1, e2 = data['e1'], data['e2']
    weight = data['weight']
""")
    
    print(f"Instructions saved to {sample_info_file}")


def create_mock_shear_catalog():
    """
    Create a mock shear catalog for testing the pipeline.
    
    This generates realistic-looking shear data that can be used
    to test the curvature gradient computation pipeline.
    """
    print("\n" + "=" * 60)
    print("Creating mock shear catalog for testing")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate random positions (10 deg x 10 deg field)
    n_galaxies = 100_000
    ra = np.random.uniform(150, 160, n_galaxies)
    dec = np.random.uniform(20, 30, n_galaxies)
    
    # Add a mock cluster at center
    cluster_ra, cluster_dec = 155, 25
    cluster_mass = 1e15  # Msun
    
    # Compute tangential shear from cluster (simplified)
    dx = (ra - cluster_ra) * np.cos(np.radians(dec))
    dy = dec - cluster_dec
    r = np.sqrt(dx**2 + dy**2)  # degrees
    r_mpc = r * 60  # rough conversion at z~0.3
    
    # NFW-ish shear profile
    gamma_t = 0.1 * (r_mpc / 0.5) / (1 + (r_mpc / 0.5)**2)
    phi = np.arctan2(dy, dx)
    
    # Convert tangential shear to e1, e2
    e1 = -gamma_t * np.cos(2 * phi) + np.random.normal(0, 0.3, n_galaxies)
    e2 = -gamma_t * np.sin(2 * phi) + np.random.normal(0, 0.3, n_galaxies)
    
    # Weights (higher for brighter galaxies)
    weight = np.random.uniform(0.5, 1.5, n_galaxies)
    
    # Photo-z
    z_phot = np.random.uniform(0.2, 1.5, n_galaxies)
    
    # Save as FITS
    from astropy.table import Table
    from astropy.io import fits
    
    table = Table({
        'ra': ra,
        'dec': dec,
        'e1': e1,
        'e2': e2,
        'weight': weight,
        'z_phot': z_phot,
        'r_from_cluster': r,
        'gamma_t_input': gamma_t,
    })
    
    output_file = LENSING_DIR / "mock_shear_catalog.fits"
    table.write(output_file, format='fits', overwrite=True)
    
    print(f"Mock catalog saved to {output_file}")
    print(f"  {n_galaxies:,} galaxies")
    print(f"  Mock cluster at (RA={cluster_ra}, Dec={cluster_dec})")
    
    return output_file


# =============================================================================
# SHEAR TO CONVERGENCE (Kaiser-Squires)
# =============================================================================

def kaiser_squires_reconstruction(e1: np.ndarray, e2: np.ndarray,
                                   ra: np.ndarray, dec: np.ndarray,
                                   npix: int = 256) -> tuple:
    """
    Reconstruct convergence κ from shear (e1, e2) using Kaiser-Squires.
    
    Parameters
    ----------
    e1, e2 : arrays
        Shear components
    ra, dec : arrays
        Sky positions (degrees)
    npix : int
        Grid resolution
    
    Returns
    -------
    kappa : 2D array
        Convergence map
    ra_grid, dec_grid : 2D arrays
        Coordinate grids
    """
    # Create grid
    ra_min, ra_max = ra.min(), ra.max()
    dec_min, dec_max = dec.min(), dec.max()
    
    ra_edges = np.linspace(ra_min, ra_max, npix + 1)
    dec_edges = np.linspace(dec_min, dec_max, npix + 1)
    
    ra_centers = 0.5 * (ra_edges[:-1] + ra_edges[1:])
    dec_centers = 0.5 * (dec_edges[:-1] + dec_edges[1:])
    ra_grid, dec_grid = np.meshgrid(ra_centers, dec_centers)
    
    # Bin shear onto grid
    from scipy.stats import binned_statistic_2d
    
    g1_map, _, _, _ = binned_statistic_2d(
        ra, dec, e1, statistic='mean', bins=[ra_edges, dec_edges]
    )
    g2_map, _, _, _ = binned_statistic_2d(
        ra, dec, e2, statistic='mean', bins=[ra_edges, dec_edges]
    )
    
    # Replace NaN with 0
    g1_map = np.nan_to_num(g1_map.T)
    g2_map = np.nan_to_num(g2_map.T)
    
    # Kaiser-Squires in Fourier space
    # κ̂(k) = D*(k) γ̂(k) where D(k) = (k1² - k2² + 2ik1k2) / |k|²
    
    # FFT
    g1_fft = np.fft.fft2(g1_map)
    g2_fft = np.fft.fft2(g2_map)
    
    # k-space coordinates
    kx = np.fft.fftfreq(npix)
    ky = np.fft.fftfreq(npix)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k2 = kx_grid**2 + ky_grid**2
    k2[0, 0] = 1  # avoid division by zero
    
    # D* kernel
    D_star = (kx_grid**2 - ky_grid**2 - 2j * kx_grid * ky_grid) / k2
    D_star[0, 0] = 0
    
    # Reconstruct κ
    gamma_complex = g1_fft + 1j * g2_fft
    kappa_fft = D_star * gamma_complex
    kappa = np.real(np.fft.ifft2(kappa_fft))
    
    return kappa, ra_grid, dec_grid


def compute_convergence_gradient(kappa: np.ndarray, 
                                  pixel_scale_deg: float) -> tuple:
    """
    Compute gradient of convergence map.
    
    Parameters
    ----------
    kappa : 2D array
        Convergence map
    pixel_scale_deg : float
        Pixel scale in degrees
    
    Returns
    -------
    grad_kappa_x, grad_kappa_y : 2D arrays
        Gradient components
    grad_kappa_mag : 2D array
        Gradient magnitude |∇κ|
    """
    # Convert to radians for proper gradient
    pixel_scale_rad = np.radians(pixel_scale_deg)
    
    # Compute gradient
    grad_y, grad_x = np.gradient(kappa, pixel_scale_rad)
    
    # Magnitude
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    return grad_x, grad_y, grad_mag


# =============================================================================
# VERIFY
# =============================================================================

def verify_downloads():
    """Check available lensing data."""
    print("\n" + "=" * 60)
    print("Verifying lensing data")
    print("=" * 60)
    
    # Check CLASH
    clash_dir = LENSING_DIR / "clash"
    if clash_dir.exists():
        clusters = list(clash_dir.iterdir())
        print(f"  CLASH: {len(clusters)} cluster directories")
    
    # Check mock catalog
    mock = LENSING_DIR / "mock_shear_catalog.fits"
    if mock.exists():
        print(f"  Mock catalog: {mock}")
    
    # Check KiDS
    kids_dir = LENSING_DIR / "kids"
    if kids_dir.exists():
        print(f"  KiDS: directory exists (see README for download)")
    
    fits_files = list(LENSING_DIR.glob("**/*.fits"))
    print(f"\n  Total FITS files: {len(fits_files)}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Weak Lensing Data Downloader")
    print("=" * 60)
    print(f"Data directory: {LENSING_DIR}")
    print()
    
    # Create mock for testing
    create_mock_shear_catalog()
    
    # Download CLASH (public)
    download_clash_catalogs()
    
    # Instructions for KiDS
    download_kids_public_sample()
    
    # Verify
    verify_downloads()
