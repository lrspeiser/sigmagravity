"""
Tidal Tensor and Cosmic Web Classification
============================================

Computes the tidal tensor eigenvalues (λ₁, λ₂, λ₃) which classify
the cosmic web environment:

    Void:     λ₁, λ₂, λ₃ < 0 (all negative → expansion)
    Sheet:    λ₁ > 0, λ₂, λ₃ < 0 (one direction collapsed)
    Filament: λ₁, λ₂ > 0, λ₃ < 0 (two directions collapsed)
    Node:     λ₁, λ₂, λ₃ > 0 (all collapsed → cluster)

Theory: In Σ-Gravity, voids have the quietest environments (low density,
long dynamical times) → maximum coherence → maximum enhancement K.
Nodes (clusters) have the noisiest environments → minimum enhancement.

Expected correlation: K(void) > K(filament) > K(sheet) > K(node)

Data sources:
    - SDSS void catalogs
    - DisPerSE cosmic web catalogs
    - 2MRS density field
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import COSMIC_WEB_DIR, DATA_URLS

try:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter as gpu_gaussian_filter
    USE_GPU = True
except ImportError:
    cp = np
    USE_GPU = False


# Cosmic web classification thresholds (from literature)
LAMBDA_THRESHOLD = 0.0  # Zel'dovich threshold for collapse


def classify_cosmic_web(lambda1: float, lambda2: float, lambda3: float,
                        threshold: float = LAMBDA_THRESHOLD) -> str:
    """
    Classify cosmic web environment from tidal eigenvalues.
    
    Convention: λ₁ ≤ λ₂ ≤ λ₃ (sorted)
    """
    n_positive = sum([l > threshold for l in [lambda1, lambda2, lambda3]])
    
    if n_positive == 0:
        return 'void'
    elif n_positive == 1:
        return 'sheet'
    elif n_positive == 2:
        return 'filament'
    else:
        return 'node'


def compute_tidal_tensor_from_density(density_field: np.ndarray,
                                       smoothing_scale: float = 2.0,
                                       box_size: float = 100.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute tidal tensor eigenvalues from a density field.
    
    T_ij = ∂²Φ / ∂x_i ∂x_j
    
    where Φ is the gravitational potential from Poisson: ∇²Φ = 4πGρ
    
    Args:
        density_field: 3D density array (N x N x N)
        smoothing_scale: Gaussian smoothing scale in grid units
        box_size: Physical box size in Mpc
    
    Returns:
        lambda1, lambda2, lambda3: Eigenvalue arrays (same shape as input)
    """
    xp = cp if USE_GPU else np
    
    # Move to GPU if available
    if USE_GPU:
        density = cp.asarray(density_field)
        gf = gpu_gaussian_filter
    else:
        density = density_field
        gf = gaussian_filter
    
    # Smooth the density field
    delta = gf(density, smoothing_scale)
    
    # Solve Poisson equation in Fourier space
    # Φ(k) = -4πG * δ(k) / k²
    n = density.shape[0]
    k = xp.fft.fftfreq(n) * 2 * np.pi * n / box_size
    kx, ky, kz = xp.meshgrid(k, k, k, indexing='ij')
    k_sq = kx**2 + ky**2 + kz**2
    k_sq[0, 0, 0] = 1  # Avoid division by zero
    
    delta_k = xp.fft.fftn(delta)
    phi_k = -delta_k / k_sq
    phi_k[0, 0, 0] = 0  # Zero mean
    
    # Compute second derivatives (tidal tensor components)
    # T_ij = -k_i * k_j * Φ(k) in Fourier space
    T_xx_k = -kx * kx * phi_k
    T_yy_k = -ky * ky * phi_k
    T_zz_k = -kz * kz * phi_k
    T_xy_k = -kx * ky * phi_k
    T_xz_k = -kx * kz * phi_k
    T_yz_k = -ky * kz * phi_k
    
    # Transform back to real space
    T_xx = xp.real(xp.fft.ifftn(T_xx_k))
    T_yy = xp.real(xp.fft.ifftn(T_yy_k))
    T_zz = xp.real(xp.fft.ifftn(T_zz_k))
    T_xy = xp.real(xp.fft.ifftn(T_xy_k))
    T_xz = xp.real(xp.fft.ifftn(T_xz_k))
    T_yz = xp.real(xp.fft.ifftn(T_yz_k))
    
    # Compute eigenvalues at each point
    # For 3x3 symmetric matrix, use analytical formula for eigenvalues
    shape = T_xx.shape
    
    # Flatten for vectorized eigenvalue computation
    T_xx_f = T_xx.flatten()
    T_yy_f = T_yy.flatten()
    T_zz_f = T_zz.flatten()
    T_xy_f = T_xy.flatten()
    T_xz_f = T_xz.flatten()
    T_yz_f = T_yz.flatten()
    
    # Build tensor matrices and compute eigenvalues
    # This is the bottleneck - would benefit from custom CUDA kernel
    n_points = len(T_xx_f)
    lambda1 = xp.zeros(n_points)
    lambda2 = xp.zeros(n_points)
    lambda3 = xp.zeros(n_points)
    
    # Batch eigenvalue computation
    if USE_GPU:
        # Use batched eigenvalue solver on GPU
        tensors = cp.zeros((n_points, 3, 3))
        tensors[:, 0, 0] = T_xx_f
        tensors[:, 1, 1] = T_yy_f
        tensors[:, 2, 2] = T_zz_f
        tensors[:, 0, 1] = tensors[:, 1, 0] = T_xy_f
        tensors[:, 0, 2] = tensors[:, 2, 0] = T_xz_f
        tensors[:, 1, 2] = tensors[:, 2, 1] = T_yz_f
        
        eigenvalues = cp.linalg.eigvalsh(tensors)
        lambda1 = eigenvalues[:, 0].reshape(shape)
        lambda2 = eigenvalues[:, 1].reshape(shape)
        lambda3 = eigenvalues[:, 2].reshape(shape)
        
        return lambda1.get(), lambda2.get(), lambda3.get()
    else:
        # CPU fallback - slower but works
        for i in range(n_points):
            T = np.array([
                [T_xx_f[i], T_xy_f[i], T_xz_f[i]],
                [T_xy_f[i], T_yy_f[i], T_yz_f[i]],
                [T_xz_f[i], T_yz_f[i], T_zz_f[i]]
            ])
            eigvals = np.linalg.eigvalsh(T)
            lambda1[i], lambda2[i], lambda3[i] = eigvals
        
        return lambda1.reshape(shape), lambda2.reshape(shape), lambda3.reshape(shape)


def classify_grid(lambda1: np.ndarray, lambda2: np.ndarray, lambda3: np.ndarray,
                  threshold: float = LAMBDA_THRESHOLD) -> np.ndarray:
    """
    Classify every point in a grid into cosmic web types.
    
    Returns integer array: 0=void, 1=sheet, 2=filament, 3=node
    """
    n_positive = ((lambda1 > threshold).astype(int) +
                  (lambda2 > threshold).astype(int) +
                  (lambda3 > threshold).astype(int))
    return n_positive


def load_cosmic_web_catalog(catalog_path: Optional[Path] = None) -> Dict[str, np.ndarray]:
    """
    Load pre-computed cosmic web classification catalog.
    
    Returns dict with: ra, dec, z, web_type, lambda1, lambda2, lambda3
    """
    if catalog_path is None:
        candidates = [
            COSMIC_WEB_DIR / "cosmic_web_catalog.csv",
            COSMIC_WEB_DIR / "sdss_voids.csv",
            Path("C:/Users/henry/dev/sigmagravity/data/cosmic_web/cosmic_web.csv"),
        ]
        for p in candidates:
            if p.exists():
                catalog_path = p
                break
    
    if catalog_path is None or not catalog_path.exists():
        print("No cosmic web catalog found. Creating synthetic sample.")
        return create_synthetic_cosmic_web()
    
    import pandas as pd
    df = pd.read_csv(catalog_path)
    return {col: df[col].values for col in df.columns}


def create_synthetic_cosmic_web(n_points: int = 10000,
                                 box_size: float = 100.0) -> Dict[str, np.ndarray]:
    """
    Create synthetic cosmic web classification for testing.
    
    Based on log-normal density field statistics.
    """
    np.random.seed(42)
    
    # Random positions
    ra = np.random.uniform(0, 360, n_points)
    dec = np.random.uniform(-90, 90, n_points)
    z = np.random.uniform(0.01, 0.1, n_points)
    
    # Distance from void centers (statistical model)
    # Voids are roughly spherical, ~20 Mpc radius
    n_voids = 50
    void_centers = np.random.uniform(0, box_size, (n_voids, 3))
    void_radii = np.random.uniform(10, 30, n_voids)
    
    # Convert RA/Dec/z to Cartesian for distance calculations
    r = z * 3000  # Approximate Mpc
    x = r * np.cos(dec * np.pi/180) * np.cos(ra * np.pi/180)
    y = r * np.cos(dec * np.pi/180) * np.sin(ra * np.pi/180)
    z_cart = r * np.sin(dec * np.pi/180)
    
    # Compute local density (simple model)
    local_density = np.ones(n_points)
    
    for i in range(n_voids):
        dx = x - void_centers[i, 0]
        dy = y - void_centers[i, 1]
        dz = z_cart - void_centers[i, 2]
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Inside void: low density
        inside = dist < void_radii[i]
        local_density[inside] *= 0.2
        
        # Near edge: filament/sheet
        edge = (dist >= void_radii[i]) & (dist < void_radii[i] * 1.5)
        local_density[edge] *= 2.0
    
    # Generate eigenvalues based on density
    # High density → positive eigenvalues (collapsed)
    # Low density → negative eigenvalues (expanding)
    sigma = 0.5
    base = np.log10(local_density + 0.1)
    
    lambda1 = base + np.random.normal(0, sigma, n_points)
    lambda2 = base + np.random.normal(0, sigma, n_points)
    lambda3 = base + np.random.normal(0, sigma, n_points)
    
    # Sort eigenvalues
    eigenvalues = np.sort(np.vstack([lambda1, lambda2, lambda3]).T, axis=1)
    lambda1, lambda2, lambda3 = eigenvalues[:, 0], eigenvalues[:, 1], eigenvalues[:, 2]
    
    # Classify
    web_type = np.array([classify_cosmic_web(l1, l2, l3) 
                         for l1, l2, l3 in zip(lambda1, lambda2, lambda3)])
    
    return {
        'ra': ra,
        'dec': dec,
        'redshift': z,
        'web_type': web_type,
        'lambda1': lambda1,
        'lambda2': lambda2,
        'lambda3': lambda3,
        'local_density': local_density,
    }


def get_web_type_at_position(catalog: Dict[str, np.ndarray],
                              ra: float, dec: float, z: float,
                              search_radius: float = 5.0) -> str:
    """
    Get cosmic web classification at a given position using nearest neighbor.
    
    Args:
        catalog: Cosmic web catalog
        ra, dec: Coordinates in degrees
        z: Redshift
        search_radius: Search radius in Mpc
    
    Returns:
        Web type: 'void', 'sheet', 'filament', or 'node'
    """
    # Convert to Cartesian
    r = z * 3000  # Approximate Mpc
    x = r * np.cos(dec * np.pi/180) * np.cos(ra * np.pi/180)
    y = r * np.cos(dec * np.pi/180) * np.sin(ra * np.pi/180)
    z_cart = r * np.sin(dec * np.pi/180)
    
    # Catalog positions
    r_cat = catalog['redshift'] * 3000
    x_cat = r_cat * np.cos(catalog['dec'] * np.pi/180) * np.cos(catalog['ra'] * np.pi/180)
    y_cat = r_cat * np.cos(catalog['dec'] * np.pi/180) * np.sin(catalog['ra'] * np.pi/180)
    z_cat = r_cat * np.sin(catalog['dec'] * np.pi/180)
    
    # Build KD-tree for fast nearest neighbor
    tree = cKDTree(np.vstack([x_cat, y_cat, z_cat]).T)
    
    # Find nearest neighbor
    dist, idx = tree.query([x, y, z_cart])
    
    if dist < search_radius:
        return catalog['web_type'][idx]
    else:
        return 'unknown'


def run_cosmic_web_analysis():
    """Run cosmic web analysis and compute statistics."""
    print("=" * 70)
    print("   COSMIC WEB CLASSIFICATION ANALYSIS")
    print("=" * 70)
    
    # Load catalog
    print("\nLoading cosmic web catalog...")
    catalog = load_cosmic_web_catalog()
    n_points = len(catalog['ra'])
    print(f"  Loaded {n_points:,} points")
    
    # Statistics
    web_types = ['void', 'sheet', 'filament', 'node']
    print("\nCosmic web classification:")
    print("-" * 40)
    
    for wt in web_types:
        count = np.sum(catalog['web_type'] == wt)
        frac = 100 * count / n_points
        
        # Get mean eigenvalues for this type
        mask = catalog['web_type'] == wt
        if mask.sum() > 0:
            l1_mean = np.mean(catalog['lambda1'][mask])
            l2_mean = np.mean(catalog['lambda2'][mask])
            l3_mean = np.mean(catalog['lambda3'][mask])
            print(f"  {wt:10s}: {count:6,} ({frac:5.1f}%)  λ = ({l1_mean:+.2f}, {l2_mean:+.2f}, {l3_mean:+.2f})")
    
    # Expected quietness ranking
    print("\n" + "=" * 70)
    print("   INTERPRETATION FOR Σ-GRAVITY")
    print("=" * 70)
    print("""
    In the Σ-Gravity framework, the cosmic web environment affects
    gravitational coherence through:
    
    1. Local density → matter fluctuations
    2. Tidal forces → metric gradients
    3. Dynamical timescales → coherence time
    
    Expected "quietness" ranking (from quietest to noisiest):
    
      VOID > SHEET > FILAMENT > NODE
      
    This predicts:
      - Void galaxies: Maximum enhancement K
      - Cluster galaxies: Minimum enhancement K
      
    This may explain why:
      - Void galaxies appear more "dark matter dominated"
      - Cluster galaxies follow Newtonian predictions better
""")
    
    return catalog


if __name__ == "__main__":
    run_cosmic_web_analysis()
