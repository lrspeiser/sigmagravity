"""
Quietness Variable Computation Modules
======================================

Each module computes a specific "quietness" variable that may control
gravitational coherence in Σ-Gravity theory.

Modules:
    velocity_dispersion - Local velocity dispersion σ_v from Gaia
    matter_density - Matter overdensity δ from galaxy counts
    tidal_tensor - Cosmic web classification (void/filament/node)
    dynamical_timescale - Orbital/crossing times from kinematics
    curvature_gradients - ∇κ from weak lensing shear
    entropy_sfr - Star formation rate as entropy proxy
    gw_background - Gravitational wave background density

GPU Acceleration:
    All modules support CuPy for GPU acceleration on NVIDIA GPUs.
    Set USE_GPU=True in config to enable (requires RTX 5090 or similar).
"""

import numpy as np

# Try to use CuPy for GPU acceleration
USE_GPU = False
try:
    import cupy as cp
    USE_GPU = True
    print("CuPy available - GPU acceleration enabled")
except ImportError:
    cp = np  # Fallback to NumPy
    print("CuPy not available - using CPU")


def get_array_module():
    """Get the appropriate array module (CuPy or NumPy)."""
    return cp if USE_GPU else np


def to_gpu(arr):
    """Move array to GPU if available."""
    if USE_GPU and isinstance(arr, np.ndarray):
        return cp.asarray(arr)
    return arr


def to_cpu(arr):
    """Move array to CPU."""
    if USE_GPU and hasattr(arr, 'get'):
        return arr.get()
    return arr
