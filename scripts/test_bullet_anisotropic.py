#!/usr/bin/env python3
"""
Test anisotropic gravity on Bullet Cluster directional offset.

This test uses the 2D anisotropic Poisson solver to predict the Bullet Cluster
lensing offset (directional observable) and compares κ=0 vs κ>0 predictions
against the observed offset of ~150 kpc.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

try:
    from stream_seeking_anisotropic import (
        solve_anisotropic_poisson_dirichlet,
        normalize_vector_field,
    )
    _HAS_ANISOTROPIC = True
except ImportError:
    _HAS_ANISOTROPIC = False

# Physical constants
G = 6.67430e-11  # m³/kg/s²
M_sun = 1.98892e30  # kg
kpc_to_m = 3.0856775814913673e19  # m/kpc


@dataclass
class BulletClusterScene:
    """Bullet Cluster scene: gas and stellar mass distributions."""
    rho_gas: np.ndarray  # Gas density (H, W)
    rho_stars: np.ndarray  # Stellar density (H, W)
    stream_direction: np.ndarray  # Stream direction field (H, W, 2)
    stream_weight: np.ndarray  # Stream intensity weight (H, W)
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    dx: float
    dy: float
    N: int
    L: float


def build_bullet_scene(
    N: int = 256,
    L: float = 2000.0,  # 2 Mpc domain
    gas_offset_x: float = -50.0,  # Gas displaced toward center (kpc)
    stars_offset_x: float = -200.0,  # Stars passed through (kpc)
    gas_sigma: float = 250.0,  # Gas core radius (kpc)
    stars_sigma: float = 100.0,  # Stellar scale (kpc)
    M_gas: float = 1.5e14,  # M_sun
    M_stars: float = 0.3e14,  # M_sun
) -> BulletClusterScene:
    """
    Build a 2D Bullet Cluster scene from observational data.
    
    Based on Clowe+ 2006 observations:
    - Main cluster gas: M_gas = 1.5e14 M_sun, offset ~50 kpc toward center
    - Main cluster stars: M_stars = 0.3e14 M_sun, offset ~200 kpc (passed through)
    - Observed lensing offset: ~150 kpc between lensing peak and gas peak
    """
    x_min, x_max = -L, L
    y_min, y_max = -L, L
    xs = np.linspace(x_min, x_max, N)
    ys = np.linspace(y_min, y_max, N)
    dx = (x_max - x_min) / (N - 1)
    dy = (y_max - y_min) / (N - 1)
    X, Y = np.meshgrid(xs, ys)
    
    # Gas density: Gaussian blob at gas_offset_x
    rho_gas = np.exp(-((X - gas_offset_x)**2 + Y**2) / (2 * gas_sigma**2))
    # Normalize to total mass M_gas
    rho_gas = rho_gas / np.sum(rho_gas) * (M_gas * M_sun) / (dx * dy * kpc_to_m**2)
    
    # Stellar density: Gaussian blob at stars_offset_x
    rho_stars = np.exp(-((X - stars_offset_x)**2 + Y**2) / (2 * stars_sigma**2))
    # Normalize to total mass M_stars
    rho_stars = rho_stars / np.sum(rho_stars) * (M_stars * M_sun) / (dx * dy * kpc_to_m**2)
    
    # Total baryonic density
    rho_total = rho_gas + rho_stars
    
    # Stream direction: spatially varying, pointing from gas toward stars
    # This represents the coherent flow direction, stronger near stars
    # The direction should point from each point toward the stellar peak
    dx_stream = stars_offset_x - X  # Point toward stars
    dy_stream = 0.0 - Y  # Point toward y=0 (collision axis)
    norm = np.sqrt(dx_stream**2 + dy_stream**2) + 1e-12
    stream_direction = np.zeros((N, N, 2), dtype=float)
    stream_direction[..., 0] = dx_stream / norm  # x-component (normalized)
    stream_direction[..., 1] = dy_stream / norm  # y-component (normalized)
    
    # Stream weight: stronger where stars are (coherent stellar flow)
    stream_weight = np.exp(-((X - stars_offset_x)**2 + Y**2) / (2 * stars_sigma**2))
    stream_weight = np.clip(stream_weight / np.max(stream_weight), 0.0, 1.0)
    
    return BulletClusterScene(
        rho_gas=rho_gas,
        rho_stars=rho_stars,
        stream_direction=stream_direction,
        stream_weight=stream_weight,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        dx=dx,
        dy=dy,
        N=N,
        L=L,
    )


def find_lensing_peak(Phi: np.ndarray, scene: BulletClusterScene) -> Tuple[float, float]:
    """
    Find the lensing peak position from the potential field.
    
    For weak lensing, the convergence κ is related to the Laplacian of the potential.
    The peak of the convergence field indicates the lensing mass peak.
    
    Returns:
        (x_peak, y_peak) in kpc
    """
    # Compute convergence: κ = (1/2) ∇²Φ (in appropriate units)
    # For 2D, we compute the Laplacian numerically
    Phi_x = np.gradient(Phi, scene.dx, axis=1)
    Phi_xx = np.gradient(Phi_x, scene.dx, axis=1)
    Phi_y = np.gradient(Phi, scene.dy, axis=0)
    Phi_yy = np.gradient(Phi_y, scene.dy, axis=0)
    laplacian = Phi_xx + Phi_yy
    
    # Find peak (minimum of Laplacian, since Φ < 0 for attractive potential)
    peak_idx = np.unravel_index(np.argmin(laplacian), laplacian.shape)
    
    # Convert to physical coordinates
    xs = np.linspace(scene.x_min, scene.x_max, scene.N)
    ys = np.linspace(scene.y_min, scene.y_max, scene.N)
    x_peak = xs[peak_idx[1]]
    y_peak = ys[peak_idx[0]]
    
    return x_peak, y_peak


def find_baryon_peak(scene: BulletClusterScene) -> Tuple[float, float]:
    """Find the peak of the baryonic mass distribution."""
    rho_total = scene.rho_gas + scene.rho_stars
    peak_idx = np.unravel_index(np.argmax(rho_total), rho_total.shape)
    
    xs = np.linspace(scene.x_min, scene.x_max, scene.N)
    ys = np.linspace(scene.y_min, scene.y_max, scene.N)
    x_peak = xs[peak_idx[1]]
    y_peak = ys[peak_idx[0]]
    
    return x_peak, y_peak


def predict_bullet_offset(
    scene: BulletClusterScene,
    kappa: float,
    G: float = G,
    tol: float = 1e-7,
    max_iter: int = 6000,
) -> Dict[str, float]:
    """
    Predict Bullet Cluster lensing offset using anisotropic solver.
    
    Returns:
        Dict with predicted offset, lensing peak position, baryon peak position
    """
    if not _HAS_ANISOTROPIC:
        raise ImportError("stream_seeking_anisotropic module not available")
    
    # Total baryonic density
    rho_total = scene.rho_gas + scene.rho_stars
    
    # Solve anisotropic Poisson equation
    Phi, info = solve_anisotropic_poisson_dirichlet(
        rho=rho_total,
        shat=scene.stream_direction,
        kappa=kappa,
        weight=scene.stream_weight,
        dx=scene.dx * kpc_to_m,  # Convert to meters
        dy=scene.dy * kpc_to_m,
        G=G,
        tol=tol,
        max_iter=max_iter,
    )
    
    # Find lensing peak (from potential)
    x_lens, y_lens = find_lensing_peak(Phi, scene)
    
    # Find baryon peak (from density)
    x_baryon, y_baryon = find_baryon_peak(scene)
    
    # Compute offset
    offset = np.sqrt((x_lens - x_baryon)**2 + (y_lens - y_baryon)**2)
    
    return {
        'offset_kpc': float(offset),
        'x_lens': float(x_lens),
        'y_lens': float(y_lens),
        'x_baryon': float(x_baryon),
        'y_baryon': float(y_baryon),
        'solver_info': info,
    }


def test_bullet_anisotropic_ab() -> Dict[str, any]:
    """
    A/B test: Compare baseline (κ=0) vs anisotropic (κ>0) predictions
    for Bullet Cluster lensing offset.
    
    Returns:
        Dict with comparison results
    """
    if not _HAS_ANISOTROPIC:
        return {
            'success': False,
            'error': 'stream_seeking_anisotropic module not available',
        }
    
    # Observed offset (from Clowe+ 2006)
    obs_offset_kpc = 150.0
    
    # Build scene from observational data
    scene = build_bullet_scene(
        N=256,
        L=2000.0,  # 2 Mpc domain
        gas_offset_x=-50.0,  # Gas displaced toward center
        stars_offset_x=-200.0,  # Stars passed through
        gas_sigma=250.0,
        stars_sigma=100.0,
        M_gas=1.5e14,
        M_stars=0.3e14,
    )
    
    # Baseline prediction (κ=0, isotropic)
    baseline = predict_bullet_offset(scene, kappa=0.0)
    
    # Anisotropic prediction (κ>0, test different values)
    kappa_test = 6.0
    anisotropic = predict_bullet_offset(scene, kappa=kappa_test)
    
    # Score both against observations
    baseline_error = abs(baseline['offset_kpc'] - obs_offset_kpc)
    aniso_error = abs(anisotropic['offset_kpc'] - obs_offset_kpc)
    
    improvement = (baseline_error - aniso_error) / baseline_error if baseline_error > 0 else 0.0
    aniso_better = aniso_error < baseline_error
    
    return {
        'success': True,
        'observed_offset_kpc': obs_offset_kpc,
        'baseline': {
            'kappa': 0.0,
            'predicted_offset_kpc': baseline['offset_kpc'],
            'error_kpc': baseline_error,
            'x_lens': baseline['x_lens'],
            'x_baryon': baseline['x_baryon'],
        },
        'anisotropic': {
            'kappa': kappa_test,
            'predicted_offset_kpc': anisotropic['offset_kpc'],
            'error_kpc': aniso_error,
            'x_lens': anisotropic['x_lens'],
            'x_baryon': anisotropic['x_baryon'],
        },
        'comparison': {
            'improvement': improvement,
            'aniso_better': aniso_better,
            'baseline_rms': baseline_error,
            'aniso_rms': aniso_error,
        },
    }


if __name__ == "__main__":
    print("=" * 80)
    print("BULLET CLUSTER ANISOTROPIC GRAVITY TEST")
    print("=" * 80)
    print()
    
    results = test_bullet_anisotropic_ab()
    
    if not results['success']:
        print(f"ERROR: {results.get('error', 'Unknown error')}")
        exit(1)
    
    print(f"Observed offset: {results['observed_offset_kpc']:.1f} kpc")
    print()
    print("Baseline (κ=0, isotropic):")
    print(f"  Predicted offset: {results['baseline']['predicted_offset_kpc']:.1f} kpc")
    print(f"  Error: {results['baseline']['error_kpc']:.1f} kpc")
    print(f"  Lensing peak: x={results['baseline']['x_lens']:.1f} kpc")
    print(f"  Baryon peak: x={results['baseline']['x_baryon']:.1f} kpc")
    print()
    print(f"Anisotropic (κ={results['anisotropic']['kappa']:.1f}):")
    print(f"  Predicted offset: {results['anisotropic']['predicted_offset_kpc']:.1f} kpc")
    print(f"  Error: {results['anisotropic']['error_kpc']:.1f} kpc")
    print(f"  Lensing peak: x={results['anisotropic']['x_lens']:.1f} kpc")
    print(f"  Baryon peak: x={results['anisotropic']['x_baryon']:.1f} kpc")
    print()
    print("Comparison:")
    print(f"  Improvement: {results['comparison']['improvement']*100:.1f}%")
    print(f"  Anisotropic better: {results['comparison']['aniso_better']}")

