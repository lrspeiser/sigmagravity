"""
Download galaxy survey data for matter density and star formation rate.

Sources:
1. SDSS DR17 - Spectroscopic catalog, stellar masses, SFR
2. 2MRS - 2MASS Redshift Survey for local density field
3. GALEX - UV luminosities for SFR
4. WISE - IR luminosities for SFR
5. GSWLC - GALEX-SDSS-WISE Legacy Catalog

These provide:
- Matter density ρ(r) from galaxy counts + M/L
- Density gradients ∇ρ
- Star formation rates (entropy production proxy)
"""

import sys
from pathlib import Path
import numpy as np
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SURVEYS_DIR

# =============================================================================
# DATA SOURCES
# =============================================================================

SURVEY_SOURCES = {
    # SDSS spectroscopic
    "sdss_dr17": {
        "description": "SDSS DR17 spectroscopic catalog",
        "url": "https://data.sdss.org/sas/dr17/",
        "casjobs": "https://skyserver.sdss.org/CasJobs/",
        "size_gb": 50,
    },
    
    # MPA-JHU value-added catalog (stellar masses, SFR)
    "mpa_jhu": {
        "description": "MPA-JHU stellar masses and SFR for SDSS",
        "url": "https://wwwmpa.mpa-garching.mpg.de/SDSS/DR7/",
        "files": [
            "gal_totsfr_dr7_v5_2.fits.gz",
            "gal_totspecsfr_dr7_v5_2.fits.gz", 
            "totlgm_dr7_v5_2.fit.gz",
        ],
    },
    
    # 2MASS Redshift Survey
    "2mrs": {
        "description": "2MASS Redshift Survey",
        "vizier": "J/ApJS/199/26",
        "size_mb": 50,
    },
    
    # GSWLC - GALEX-SDSS-WISE Legacy Catalog
    "gswlc": {
        "description": "GALEX-SDSS-WISE Legacy Catalog",
        "url": "https://salims.pages.iu.edu/gswlc/",
        "paper": "https://arxiv.org/abs/1610.00712",
    },
    
    # GALEX
    "galex": {
        "description": "GALEX UV catalog",
        "url": "https://galex.stsci.edu/GR6/",
        "mast": "https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html",
    },
}


# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================

def download_mpa_jhu_catalog():
    """
    Download MPA-JHU value-added catalog.
    
    This contains stellar masses and SFR for SDSS galaxies,
    derived from spectral fitting.
    
    Key columns:
    - log_mass: log10(M*/Msun)
    - sfr: Star formation rate (Msun/yr)
    - ssfr: Specific SFR (1/yr)
    """
    print("\n" + "=" * 60)
    print("MPA-JHU Value-Added Catalog")
    print("=" * 60)
    
    mpa_dir = SURVEYS_DIR / "mpa_jhu"
    mpa_dir.mkdir(exist_ok=True)
    
    base_url = "https://wwwmpa.mpa-garching.mpg.de/SDSS/DR7/Data/"
    
    files = [
        ("gal_totsfr_dr7_v5_2.fits.gz", "Total SFR"),
        ("gal_totspecsfr_dr7_v5_2.fits.gz", "Spectroscopic SFR"),
        ("totlgm_dr7_v5_2.fit.gz", "Stellar masses"),
    ]
    
    for filename, description in files:
        url = base_url + filename
        dest = mpa_dir / filename
        
        if dest.exists():
            print(f"  {description}: already downloaded")
            continue
        
        print(f"\nDownloading {description}...")
        print(f"  URL: {url}")
        
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            total = int(response.headers.get('content-length', 0))
            with open(dest, 'wb') as f:
                with tqdm(total=total, unit='B', unit_scale=True) as pbar:
                    for chunk in response.iter_content(8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
                        
            print(f"  Saved to {dest}")
            
        except Exception as e:
            print(f"  Error: {e}")
            print(f"  Download manually from: {url}")


def download_2mrs_catalog():
    """
    Download 2MASS Redshift Survey catalog.
    
    Contains ~45,000 galaxies with K-band magnitudes and redshifts
    covering the whole sky. Good for local density field.
    """
    print("\n" + "=" * 60)
    print("2MASS Redshift Survey (2MRS)")
    print("=" * 60)
    
    try:
        from astroquery.vizier import Vizier
        
        Vizier.ROW_LIMIT = -1
        
        catalogs = Vizier.get_catalogs("J/ApJS/199/26")
        
        if catalogs:
            cat = catalogs[0]
            
            output_file = SURVEYS_DIR / "2mrs.fits"
            cat.write(output_file, format='fits', overwrite=True)
            print(f"  Saved {len(cat)} galaxies to {output_file}")
            
            # Quick stats
            if 'Kmag' in cat.colnames:
                print(f"  K-mag range: {cat['Kmag'].min():.1f} to {cat['Kmag'].max():.1f}")
                
        else:
            print("  Catalog not found")
            
    except ImportError:
        print("  astroquery not installed")
        create_2mrs_instructions()


def create_2mrs_instructions():
    """Create manual download instructions for 2MRS."""
    
    readme = SURVEYS_DIR / "2mrs_download.txt"
    with open(readme, 'w') as f:
        f.write("""2MASS Redshift Survey (2MRS) Download Instructions
==================================================

VizieR catalog: J/ApJS/199/26

Web download:
1. Go to: https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/ApJS/199/26
2. Select all columns
3. Download as FITS

Key columns:
- RAJ2000, DEJ2000: Coordinates
- Kmag: K-band magnitude
- cz: Heliocentric velocity (km/s)
- Dist: Distance (Mpc) when available
- Type: Morphological type

Python example:
    from astroquery.vizier import Vizier
    Vizier.ROW_LIMIT = -1
    cat = Vizier.get_catalogs("J/ApJS/199/26")[0]
    cat.write("2mrs.fits", format='fits')
""")
    
    print(f"  Instructions saved to {readme}")


def download_gswlc_sample():
    """
    Download GSWLC (GALEX-SDSS-WISE Legacy Catalog) sample.
    
    This combines UV (GALEX), optical (SDSS), and IR (WISE)
    for robust SFR and stellar mass estimates.
    """
    print("\n" + "=" * 60)
    print("GSWLC - GALEX-SDSS-WISE Legacy Catalog")
    print("=" * 60)
    
    gswlc_dir = SURVEYS_DIR / "gswlc"
    gswlc_dir.mkdir(exist_ok=True)
    
    # GSWLC is available from Salim's website
    readme = gswlc_dir / "DOWNLOAD_INSTRUCTIONS.txt"
    with open(readme, 'w') as f:
        f.write("""GSWLC Download Instructions
============================

Main website: https://salims.pages.iu.edu/gswlc/

GSWLC-2 (recommended):
- Contains ~700,000 SDSS galaxies
- SED fitting with GALEX UV + SDSS optical + WISE IR
- Robust stellar masses and SFR

Download:
1. Go to https://salims.pages.iu.edu/gswlc/
2. Download GSWLC-2 catalog (FITS format)
3. Save to this directory

Key columns:
- logMstar: log10(M*/Msun)
- logSFR: log10(SFR / Msun/yr)
- logsSFR: log10(sSFR / yr^-1)
- z: Spectroscopic redshift
- ra, dec: Coordinates

For entropy production proxy:
- Use logSFR as indicator of ongoing gravitational collapse
- High SFR = high entropy production = less "quiet"

Python loading:
    from astropy.io import fits
    hdu = fits.open('GSWLC-2.fits')
    data = hdu[1].data
    sfr = 10**data['logSFR']  # Msun/yr
    mass = 10**data['logMstar']  # Msun
    ssfr = sfr / mass  # Specific SFR
""")
    
    print(f"  Instructions saved to {readme}")


def create_sdss_casjobs_query():
    """
    Create CasJobs query for SDSS spectroscopic sample.
    
    CasJobs allows SQL queries on SDSS database for custom selections.
    """
    print("\n" + "=" * 60)
    print("SDSS CasJobs Query")
    print("=" * 60)
    
    query_file = SURVEYS_DIR / "sdss_casjobs_query.sql"
    
    query = """-- SDSS DR17 Galaxy Sample for Density/SFR Analysis
-- Run this query at https://skyserver.sdss.org/CasJobs/

SELECT TOP 500000
    p.objID,
    p.ra, p.dec,
    p.petroMag_r, p.petroMag_g,
    p.petroR50_r,  -- Petrosian half-light radius
    s.z, s.zErr,
    s.velDisp, s.velDispErr,
    -- Spectroscopic classifications
    s.class, s.subclass,
    -- Emission line fluxes for SFR
    s.h_alpha_flux, s.h_beta_flux,
    s.oiii_5007_flux, s.nii_6584_flux,
    -- Stellar mass and SFR from Portsmouth group
    g.logMass, g.logMass_err,
    g.sfr, g.sfr_err,
    g.ssfr
FROM PhotoObj AS p
JOIN SpecObj AS s ON s.bestObjID = p.objID
LEFT JOIN stellarMassPCAWiscBC03 AS g ON g.specObjID = s.specObjID
WHERE 
    s.class = 'GALAXY'
    AND s.z > 0.01 AND s.z < 0.3
    AND s.zWarning = 0
    AND p.petroMag_r < 17.77
    AND p.petroMag_r > 14
ORDER BY p.ra
"""
    
    with open(query_file, 'w') as f:
        f.write(query)
    
    print(f"  Query saved to {query_file}")
    print("\n  To use:")
    print("  1. Go to https://skyserver.sdss.org/CasJobs/")
    print("  2. Create account if needed")
    print("  3. Paste query and submit")
    print("  4. Download results as FITS")


def create_density_field_calculator():
    """
    Create module for computing density fields from galaxy catalogs.
    """
    print("\n" + "=" * 60)
    print("Creating density field calculator")
    print("=" * 60)
    
    calc_file = SURVEYS_DIR / "density_calculator.py"
    
    code = '''"""
Compute 3D density field from galaxy catalog.

Methods:
1. Simple number counts in cells
2. Weighted by stellar mass
3. Voronoi tessellation
4. Adaptive kernel smoothing

Output:
- δ(x) = ρ(x)/ρ̄ - 1: Density contrast
- ∇δ: Density gradient (quietness proxy)
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from typing import Tuple


def galaxy_to_cartesian(ra: np.ndarray, dec: np.ndarray, 
                        redshift: np.ndarray,
                        H0: float = 70.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert galaxy coordinates to Cartesian (comoving Mpc).
    
    Uses simple Hubble law: d = cz/H0
    """
    c = 299792.458  # km/s
    
    # Comoving distance (Mpc)
    d = c * redshift / H0
    
    # Convert to Cartesian
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    
    x = d * np.cos(dec_rad) * np.cos(ra_rad)
    y = d * np.cos(dec_rad) * np.sin(ra_rad)
    z = d * np.sin(dec_rad)
    
    return x, y, z


def density_on_grid(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                    weights: np.ndarray = None,
                    box_size: float = None,
                    n_cells: int = 64,
                    smooth_mpc: float = 5.0) -> Tuple[np.ndarray, dict]:
    """
    Compute density field on a regular grid.
    
    Parameters
    ----------
    x, y, z : arrays
        Galaxy positions (Mpc)
    weights : array, optional
        Weights (e.g., stellar mass)
    box_size : float, optional
        Grid size (Mpc). If None, use data extent.
    n_cells : int
        Number of grid cells per dimension
    smooth_mpc : float
        Gaussian smoothing scale (Mpc)
    
    Returns
    -------
    delta : 3D array
        Density contrast δ = ρ/ρ̄ - 1
    grid_info : dict
        Grid metadata (edges, centers, etc.)
    """
    if weights is None:
        weights = np.ones_like(x)
    
    # Determine grid extent
    if box_size is None:
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        z_min, z_max = z.min(), z.max()
        box_size = max(x_max - x_min, y_max - y_min, z_max - z_min) * 1.1
        center = [(x_max + x_min)/2, (y_max + y_min)/2, (z_max + z_min)/2]
    else:
        center = [0, 0, 0]
    
    # Grid edges
    half_box = box_size / 2
    x_edges = np.linspace(center[0] - half_box, center[0] + half_box, n_cells + 1)
    y_edges = np.linspace(center[1] - half_box, center[1] + half_box, n_cells + 1)
    z_edges = np.linspace(center[2] - half_box, center[2] + half_box, n_cells + 1)
    
    # Bin galaxies
    density, _ = np.histogramdd(
        (x, y, z), 
        bins=[x_edges, y_edges, z_edges],
        weights=weights
    )
    
    # Smooth
    cell_size = box_size / n_cells
    sigma_cells = smooth_mpc / cell_size
    if sigma_cells > 0.5:
        density = gaussian_filter(density, sigma_cells)
    
    # Convert to density contrast
    mean_density = density.mean()
    if mean_density > 0:
        delta = density / mean_density - 1
    else:
        delta = np.zeros_like(density)
    
    grid_info = {
        'x_edges': x_edges,
        'y_edges': y_edges,
        'z_edges': z_edges,
        'cell_size': cell_size,
        'box_size': box_size,
        'center': center,
        'mean_density': mean_density,
    }
    
    return delta, grid_info


def density_gradient(delta: np.ndarray, 
                     cell_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute gradient of density field.
    
    Parameters
    ----------
    delta : 3D array
        Density contrast
    cell_size : float
        Grid cell size (Mpc)
    
    Returns
    -------
    grad_mag : 3D array
        |∇δ| at each point
    grad_components : tuple of 3D arrays
        (∂δ/∂x, ∂δ/∂y, ∂δ/∂z)
    """
    grad_x, grad_y, grad_z = np.gradient(delta, cell_size)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    
    return grad_mag, (grad_x, grad_y, grad_z)


def local_density_knn(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                      k: int = 10) -> np.ndarray:
    """
    Compute local density using k-nearest neighbors.
    
    ρ_local ∝ k / V_k where V_k is the volume containing k neighbors.
    
    Parameters
    ----------
    x, y, z : arrays
        Galaxy positions
    k : int
        Number of neighbors
    
    Returns
    -------
    density : array
        Local density at each galaxy position
    """
    positions = np.column_stack([x, y, z])
    tree = cKDTree(positions)
    
    # Distance to k-th neighbor
    distances, _ = tree.query(positions, k=k+1)  # +1 because first is self
    r_k = distances[:, -1]  # Distance to k-th neighbor
    
    # Volume of sphere with radius r_k
    V_k = (4/3) * np.pi * r_k**3
    
    # Density ∝ k / V_k
    density = k / V_k
    
    return density


def quietness_from_density(delta: np.ndarray,
                           delta_threshold: float = 0.5) -> np.ndarray:
    """
    Map density contrast to quietness factor.
    
    Low density (underdense) = quiet
    High density (overdense) = not quiet
    
    Parameters
    ----------
    delta : array
        Density contrast
    delta_threshold : float
        Characteristic scale
    
    Returns
    -------
    quietness : array
        0 (dense) to 1 (underdense)
    """
    # Sigmoid-like transition
    quietness = 1 / (1 + np.exp(delta / delta_threshold))
    return quietness


def quietness_from_gradient(grad_mag: np.ndarray,
                            grad_threshold: float = 0.1) -> np.ndarray:
    """
    Map density gradient to quietness.
    
    Low gradient = smooth = quiet
    High gradient = turbulent = not quiet
    """
    quietness = np.exp(-grad_mag / grad_threshold)
    return quietness


# =============================================================================
# EXAMPLE
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Generate mock galaxy catalog
    np.random.seed(42)
    n_gal = 10000
    
    # Clustered distribution
    n_clusters = 20
    cluster_centers = np.random.uniform(-100, 100, (n_clusters, 3))
    cluster_sizes = np.random.exponential(10, n_clusters)
    
    x, y, z = [], [], []
    for i in range(n_clusters):
        n_in_cluster = int(n_gal / n_clusters * (1 + np.random.randn() * 0.3))
        x.extend(cluster_centers[i, 0] + np.random.randn(n_in_cluster) * cluster_sizes[i])
        y.extend(cluster_centers[i, 1] + np.random.randn(n_in_cluster) * cluster_sizes[i])
        z.extend(cluster_centers[i, 2] + np.random.randn(n_in_cluster) * cluster_sizes[i])
    
    x, y, z = np.array(x), np.array(y), np.array(z)
    
    print(f"Mock catalog: {len(x)} galaxies")
    
    # Compute density field
    delta, grid_info = density_on_grid(x, y, z, n_cells=32, smooth_mpc=5.0)
    
    # Compute gradient
    grad_mag, _ = density_gradient(delta, grid_info['cell_size'])
    
    # Quietness
    q_delta = quietness_from_density(delta)
    q_grad = quietness_from_gradient(grad_mag)
    
    print(f"Density contrast: {delta.min():.2f} to {delta.max():.2f}")
    print(f"Gradient magnitude: {grad_mag.min():.3f} to {grad_mag.max():.3f}")
    print(f"Mean quietness (density): {q_delta.mean():.3f}")
    print(f"Mean quietness (gradient): {q_grad.mean():.3f}")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    z_slice = 16
    
    im0 = axes[0, 0].imshow(delta[:, :, z_slice].T, cmap='RdBu_r', 
                            extent=[grid_info['x_edges'][0], grid_info['x_edges'][-1],
                                   grid_info['y_edges'][0], grid_info['y_edges'][-1]])
    axes[0, 0].set_title('Density contrast δ')
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(grad_mag[:, :, z_slice].T, cmap='plasma',
                            extent=[grid_info['x_edges'][0], grid_info['x_edges'][-1],
                                   grid_info['y_edges'][0], grid_info['y_edges'][-1]])
    axes[0, 1].set_title('Gradient magnitude |∇δ|')
    plt.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[1, 0].imshow(q_delta[:, :, z_slice].T, cmap='viridis', vmin=0, vmax=1,
                            extent=[grid_info['x_edges'][0], grid_info['x_edges'][-1],
                                   grid_info['y_edges'][0], grid_info['y_edges'][-1]])
    axes[1, 0].set_title('Quietness (from density)')
    plt.colorbar(im2, ax=axes[1, 0])
    
    im3 = axes[1, 1].imshow(q_grad[:, :, z_slice].T, cmap='viridis', vmin=0, vmax=1,
                            extent=[grid_info['x_edges'][0], grid_info['x_edges'][-1],
                                   grid_info['y_edges'][0], grid_info['y_edges'][-1]])
    axes[1, 1].set_title('Quietness (from gradient)')
    plt.colorbar(im3, ax=axes[1, 1])
    
    for ax in axes.flat:
        ax.set_xlabel('x (Mpc)')
        ax.set_ylabel('y (Mpc)')
    
    plt.tight_layout()
    plt.savefig('density_field_example.png', dpi=150)
    print("\\nSaved density_field_example.png")
'''
    
    with open(calc_file, 'w') as f:
        f.write(code)
    
    print(f"Calculator saved to {calc_file}")


# =============================================================================
# VERIFY
# =============================================================================

def verify_downloads():
    """Check available survey data."""
    print("\n" + "=" * 60)
    print("Verifying survey data")
    print("=" * 60)
    
    # Check MPA-JHU
    mpa_dir = SURVEYS_DIR / "mpa_jhu"
    if mpa_dir.exists():
        files = list(mpa_dir.glob("*.fits*"))
        print(f"  MPA-JHU: {len(files)} files")
    
    # Check 2MRS
    tmrs = SURVEYS_DIR / "2mrs.fits"
    if tmrs.exists():
        print(f"  2MRS: {tmrs}")
    
    # Check GSWLC
    gswlc_dir = SURVEYS_DIR / "gswlc"
    if gswlc_dir.exists():
        print(f"  GSWLC: directory exists (see README)")
    
    # List all files
    all_files = list(SURVEYS_DIR.glob("**/*"))
    print(f"\\n  Total files: {len([f for f in all_files if f.is_file()])}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Galaxy Survey Downloader")
    print("=" * 60)
    print(f"Data directory: {SURVEYS_DIR}")
    print()
    
    # Download catalogs
    download_mpa_jhu_catalog()
    download_2mrs_catalog()
    download_gswlc_sample()
    
    # Create queries and calculators
    create_sdss_casjobs_query()
    create_density_field_calculator()
    
    # Verify
    verify_downloads()
