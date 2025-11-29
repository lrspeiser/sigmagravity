"""
Download cosmic web classification catalogs for tidal tensor analysis.

Sources:
1. DisPerSE void/filament catalogs
2. Cosmic Flows (Tully et al.) - local velocity field
3. SDSS-based cosmic web classifications
4. 2MRS-based local density field

The cosmic web classification (void/sheet/filament/node) is determined
by the eigenvalues of the tidal tensor:
    T_ij = ∂²Φ/∂x_i∂x_j
    
- Voids: 3 positive eigenvalues (expanding in all directions)
- Sheets: 2 positive, 1 negative
- Filaments: 1 positive, 2 negative  
- Nodes: 3 negative eigenvalues (collapsing in all directions)

This directly gives us the "tidal tensor eigenvalue spread" variable.
"""

import sys
from pathlib import Path
import numpy as np
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import COSMIC_WEB_DIR

# =============================================================================
# DATA SOURCES
# =============================================================================

COSMIC_WEB_SOURCES = {
    # SDSS void catalog (Pan et al. 2012)
    "sdss_voids": {
        "description": "SDSS DR7 void catalog",
        "vizier": "J/MNRAS/421/926",
        "paper": "https://arxiv.org/abs/1110.3771",
    },
    
    # Cosmic Flows - local peculiar velocities
    "cosmicflows4": {
        "description": "Cosmic Flows-4 peculiar velocity catalog",
        "url": "https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/ApJ/944/94",
        "paper": "https://arxiv.org/abs/2209.11238",
    },
    
    # 2MRS density field
    "2mrs_density": {
        "description": "2MASS Redshift Survey density reconstruction",
        "url": "https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/ApJS/199/26",
    },
    
    # BOSS/eBOSS void catalogs
    "boss_voids": {
        "description": "BOSS DR12 void catalog",
        "paper": "https://arxiv.org/abs/1607.03155",
    },
}


# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================

def download_sdss_void_catalog():
    """
    Download SDSS void catalog from VizieR.
    
    Pan et al. (2012) catalog contains ~1000 voids identified
    in SDSS DR7 using ZOBOV algorithm.
    """
    print("\n" + "=" * 60)
    print("Downloading SDSS Void Catalog")
    print("=" * 60)
    
    try:
        from astroquery.vizier import Vizier
        
        # Query VizieR for Pan et al. void catalog
        Vizier.ROW_LIMIT = -1  # Get all rows
        
        catalogs = Vizier.get_catalogs("J/MNRAS/421/926")
        
        if catalogs:
            void_table = catalogs[0]
            
            # Save to file
            output_file = COSMIC_WEB_DIR / "sdss_voids.fits"
            void_table.write(output_file, format='fits', overwrite=True)
            print(f"  Saved {len(void_table)} voids to {output_file}")
            
            # Also save as CSV for easy inspection
            csv_file = COSMIC_WEB_DIR / "sdss_voids.csv"
            void_table.to_pandas().to_csv(csv_file, index=False)
            print(f"  Also saved to {csv_file}")
        else:
            print("  No catalog found")
            
    except ImportError:
        print("  astroquery not installed, creating manual download instructions")
        create_vizier_instructions("J/MNRAS/421/926", "sdss_voids")


def download_cosmicflows_catalog():
    """
    Download Cosmic Flows-4 peculiar velocity catalog.
    
    Contains ~56,000 galaxy distances and peculiar velocities
    in the local universe (z < 0.1).
    """
    print("\n" + "=" * 60)
    print("Downloading Cosmic Flows-4 Catalog")
    print("=" * 60)
    
    try:
        from astroquery.vizier import Vizier
        
        Vizier.ROW_LIMIT = -1
        
        # CF4 catalog
        catalogs = Vizier.get_catalogs("J/ApJ/944/94")
        
        if catalogs:
            for i, cat in enumerate(catalogs):
                output_file = COSMIC_WEB_DIR / f"cosmicflows4_table{i+1}.fits"
                cat.write(output_file, format='fits', overwrite=True)
                print(f"  Table {i+1}: {len(cat)} rows -> {output_file}")
        else:
            print("  No catalog found")
            
    except ImportError:
        print("  astroquery not installed")
        create_vizier_instructions("J/ApJ/944/94", "cosmicflows4")


def create_vizier_instructions(catalog_id: str, name: str):
    """Create manual download instructions for VizieR catalogs."""
    
    instructions_file = COSMIC_WEB_DIR / f"{name}_download.txt"
    
    with open(instructions_file, 'w') as f:
        f.write(f"""Manual Download Instructions for {name}
{'=' * 50}

VizieR Catalog ID: {catalog_id}

Option 1: Web interface
-----------------------
1. Go to: https://vizier.cds.unistra.fr/viz-bin/VizieR?-source={catalog_id}
2. Select all columns
3. Choose output format (FITS or CSV)
4. Download and save to {COSMIC_WEB_DIR}

Option 2: Command line (with curl)
----------------------------------
curl -o {name}.tsv "https://vizier.cds.unistra.fr/viz-bin/votable/-A?-source={catalog_id}&-out.all"

Option 3: Python with astroquery
--------------------------------
pip install astroquery

from astroquery.vizier import Vizier
Vizier.ROW_LIMIT = -1
catalogs = Vizier.get_catalogs("{catalog_id}")
catalogs[0].write("{name}.fits", format='fits')
""")
    
    print(f"  Instructions saved to {instructions_file}")


def create_cosmic_web_classifier():
    """
    Create a module for classifying cosmic web environment.
    
    This implements the T-web algorithm based on tidal tensor eigenvalues.
    """
    print("\n" + "=" * 60)
    print("Creating cosmic web classifier")
    print("=" * 60)
    
    classifier_file = COSMIC_WEB_DIR / "cosmic_web_classifier.py"
    
    code = '''"""
Cosmic Web Classification from Tidal Tensor

The tidal tensor T_ij = ∂²Φ/∂x_i∂x_j has eigenvalues λ1 ≥ λ2 ≥ λ3.
The number of eigenvalues above a threshold λ_th determines the web type:

- Void: 0 eigenvalues > λ_th (all expanding)
- Sheet: 1 eigenvalue > λ_th
- Filament: 2 eigenvalues > λ_th
- Node: 3 eigenvalues > λ_th (all collapsing)

Reference: Hahn et al. (2007), Forero-Romero et al. (2009)
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from enum import IntEnum


class WebType(IntEnum):
    """Cosmic web environment types."""
    VOID = 0
    SHEET = 1
    FILAMENT = 2
    NODE = 3


def compute_tidal_tensor(density_field: np.ndarray, 
                         box_size: float,
                         smooth_scale: float = None) -> np.ndarray:
    """
    Compute tidal tensor from density field.
    
    Parameters
    ----------
    density_field : 3D array
        Density contrast δ = ρ/ρ̄ - 1
    box_size : float
        Physical size of box (Mpc)
    smooth_scale : float, optional
        Gaussian smoothing scale (Mpc)
    
    Returns
    -------
    tidal_tensor : 5D array
        Shape (nx, ny, nz, 3, 3) - tidal tensor at each point
    """
    nx, ny, nz = density_field.shape
    dx = box_size / nx
    
    # Smooth if requested
    if smooth_scale is not None:
        sigma_pix = smooth_scale / dx
        density_field = gaussian_filter(density_field, sigma_pix)
    
    # Solve Poisson equation in Fourier space: ∇²Φ = 4πGρ̄δ
    # We work in units where 4πGρ̄ = 1
    
    delta_k = np.fft.fftn(density_field)
    
    # k-vectors
    kx = np.fft.fftfreq(nx, dx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, dx) * 2 * np.pi
    kz = np.fft.fftfreq(nz, dx) * 2 * np.pi
    kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing='ij')
    
    k2 = kx_grid**2 + ky_grid**2 + kz_grid**2
    k2[0, 0, 0] = 1  # Avoid division by zero
    
    # Potential in Fourier space: Φ̂ = -δ̂/k²
    phi_k = -delta_k / k2
    phi_k[0, 0, 0] = 0
    
    # Tidal tensor components: T_ij = -k_i k_j Φ̂
    k_components = [kx_grid, ky_grid, kz_grid]
    
    tidal_tensor = np.zeros((nx, ny, nz, 3, 3))
    
    for i in range(3):
        for j in range(3):
            T_ij_k = -k_components[i] * k_components[j] * phi_k
            tidal_tensor[:, :, :, i, j] = np.real(np.fft.ifftn(T_ij_k))
    
    return tidal_tensor


def classify_web_type(tidal_tensor: np.ndarray, 
                      lambda_th: float = 0.0) -> np.ndarray:
    """
    Classify cosmic web type from tidal tensor eigenvalues.
    
    Parameters
    ----------
    tidal_tensor : 5D array
        Shape (..., 3, 3) - tidal tensor
    lambda_th : float
        Eigenvalue threshold (default 0)
    
    Returns
    -------
    web_type : array
        WebType classification at each point
    eigenvalues : array
        Sorted eigenvalues (λ1 ≥ λ2 ≥ λ3)
    """
    # Compute eigenvalues at each point
    original_shape = tidal_tensor.shape[:-2]
    T_flat = tidal_tensor.reshape(-1, 3, 3)
    
    eigenvalues = np.linalg.eigvalsh(T_flat)  # Returns sorted ascending
    eigenvalues = eigenvalues[:, ::-1]  # Sort descending: λ1 ≥ λ2 ≥ λ3
    
    # Count eigenvalues above threshold
    n_above = np.sum(eigenvalues > lambda_th, axis=1)
    
    # Classify
    web_type = np.zeros(len(n_above), dtype=int)
    web_type[n_above == 0] = WebType.VOID
    web_type[n_above == 1] = WebType.SHEET
    web_type[n_above == 2] = WebType.FILAMENT
    web_type[n_above == 3] = WebType.NODE
    
    # Reshape back
    web_type = web_type.reshape(original_shape)
    eigenvalues = eigenvalues.reshape(original_shape + (3,))
    
    return web_type, eigenvalues


def eigenvalue_spread(eigenvalues: np.ndarray) -> np.ndarray:
    """
    Compute eigenvalue spread as quietness proxy.
    
    Spread = (λ1 - λ3) / |λ_mean|
    
    Low spread = isotropic (quiet)
    High spread = anisotropic (active)
    """
    lambda_mean = np.mean(eigenvalues, axis=-1)
    lambda_mean = np.where(np.abs(lambda_mean) < 1e-10, 1e-10, lambda_mean)
    
    spread = (eigenvalues[..., 0] - eigenvalues[..., 2]) / np.abs(lambda_mean)
    return spread


def quietness_from_web_type(web_type: np.ndarray) -> np.ndarray:
    """
    Map web type to quietness factor.
    
    Voids are quietest, nodes are noisiest.
    
    Returns
    -------
    quietness : array
        Values from 0 (node) to 1 (void)
    """
    quietness_map = {
        WebType.VOID: 1.0,
        WebType.SHEET: 0.66,
        WebType.FILAMENT: 0.33,
        WebType.NODE: 0.0,
    }
    
    quietness = np.zeros_like(web_type, dtype=float)
    for wt, q in quietness_map.items():
        quietness[web_type == wt] = q
    
    return quietness


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create mock density field (Gaussian random field)
    np.random.seed(42)
    nx = 64
    box_size = 100  # Mpc
    
    # Generate in Fourier space with P(k) ∝ k^(-2)
    kx = np.fft.fftfreq(nx) * nx
    ky, kz = kx.copy(), kx.copy()
    kx_g, ky_g, kz_g = np.meshgrid(kx, ky, kz, indexing='ij')
    k = np.sqrt(kx_g**2 + ky_g**2 + kz_g**2)
    k[0, 0, 0] = 1
    
    # Random phases, amplitude ∝ k^(-1)
    phases = np.random.uniform(0, 2*np.pi, (nx, nx, nx))
    amplitude = 1.0 / k
    amplitude[0, 0, 0] = 0
    
    delta_k = amplitude * np.exp(1j * phases)
    density = np.real(np.fft.ifftn(delta_k))
    density = (density - density.mean()) / density.std() * 0.5  # Normalize
    
    print("Computing tidal tensor...")
    tidal = compute_tidal_tensor(density, box_size, smooth_scale=2.0)
    
    print("Classifying cosmic web...")
    web_type, eigenvalues = classify_web_type(tidal)
    spread = eigenvalue_spread(eigenvalues)
    quietness = quietness_from_web_type(web_type)
    
    # Statistics
    print(f"\\nWeb type fractions:")
    for wt in WebType:
        frac = np.sum(web_type == wt) / web_type.size
        print(f"  {wt.name}: {frac:.1%}")
    
    print(f"\\nEigenvalue spread: {spread.mean():.2f} ± {spread.std():.2f}")
    print(f"Mean quietness: {quietness.mean():.2f}")
    
    # Plot a slice
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    z_slice = nx // 2
    
    im0 = axes[0].imshow(density[:, :, z_slice], cmap='RdBu_r')
    axes[0].set_title('Density contrast δ')
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(web_type[:, :, z_slice], cmap='viridis', vmin=0, vmax=3)
    axes[1].set_title('Web type (0=void, 3=node)')
    plt.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].imshow(spread[:, :, z_slice], cmap='plasma')
    axes[2].set_title('Eigenvalue spread')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('cosmic_web_example.png', dpi=150)
    print("\\nSaved cosmic_web_example.png")
'''
    
    with open(classifier_file, 'w') as f:
        f.write(code)
    
    print(f"Classifier saved to {classifier_file}")


def download_precomputed_web_catalog():
    """
    Download a pre-computed cosmic web catalog if available.
    """
    print("\n" + "=" * 60)
    print("Looking for pre-computed cosmic web catalogs")
    print("=" * 60)
    
    # Sutter et al. void catalog
    # https://www.cosmicvoids.net/
    
    readme = COSMIC_WEB_DIR / "VOID_CATALOGS.txt"
    with open(readme, 'w') as f:
        f.write("""Pre-computed Void and Cosmic Web Catalogs
==========================================

1. SDSS Void Catalog (Sutter et al.)
   URL: https://www.cosmicvoids.net/
   Contains: ~1500 voids from SDSS DR7/DR9
   Format: ASCII with void centers, radii, and shapes

2. BOSS DR12 Voids (Mao et al. 2017)
   arXiv: 1602.02771
   Contains: Voids identified in BOSS CMASS/LOWZ

3. 2MTF Cosmic Web (Courtois et al.)
   URL: https://projets.ip2i.in2p3.fr/cosmicflows/
   Contains: Local velocity field and web classification

4. Millennium Simulation Web Catalogs
   URL: https://wwwmpa.mpa-garching.mpg.de/millennium/
   Contains: Full cosmic web from simulations

Download void catalogs from cosmicvoids.net:
   wget https://www.cosmicvoids.net/data/voids_sdss_dr7.dat
""")
    
    print(f"  Catalog list saved to {readme}")


# =============================================================================
# VERIFY
# =============================================================================

def verify_downloads():
    """Check available cosmic web data."""
    print("\n" + "=" * 60)
    print("Verifying cosmic web data")
    print("=" * 60)
    
    files = list(COSMIC_WEB_DIR.glob("*"))
    for f in files:
        if f.is_file():
            size = f.stat().st_size / 1024
            print(f"  {f.name}: {size:.1f} KB")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Cosmic Web Catalog Downloader")
    print("=" * 60)
    print(f"Data directory: {COSMIC_WEB_DIR}")
    print()
    
    # Download void catalogs
    download_sdss_void_catalog()
    download_cosmicflows_catalog()
    
    # Pre-computed catalogs
    download_precomputed_web_catalog()
    
    # Create classifier
    create_cosmic_web_classifier()
    
    # Verify
    verify_downloads()
