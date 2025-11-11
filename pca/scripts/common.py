#!/usr/bin/env python3
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from scipy.interpolate import splrep, splev

def zscore_weighted(X: np.ndarray, W: np.ndarray, eps: float=1e-12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Weighted per-feature z-score.
    X: [N,D] features
    W: [N,D] non-negative weights (e.g., 1/sigma^2 per point)
    Returns: Xz, mu[D], sd[D]
    """
    assert X.shape == W.shape, "X and W must have the same shape for per-feature weights."
    Wsum = np.sum(W, axis=0) + eps
    mu = np.sum(W * X, axis=0) / Wsum
    var = np.sum(W * (X - mu)**2, axis=0) / Wsum
    sd = np.sqrt(np.maximum(var, eps))
    Xz = (X - mu) / sd
    return Xz, mu, sd

def aggregate_row_weights(W: np.ndarray, method: str = "median") -> np.ndarray:
    """
    Aggregate per-feature weights W[N,D] into row (sample) weights w[N].
    """
    if method == "median":
        w = np.median(W, axis=1)
    elif method == "mean":
        w = np.mean(W, axis=1)
    else:
        raise ValueError("method must be 'median' or 'mean'")
    # Normalize to mean 1 for stability
    w = np.asarray(w, dtype=float)
    w = w / (np.mean(w) + 1e-12)
    # avoid zeros
    w = np.clip(w, 1e-8, None)
    return w

def pca_weighted_svd(Xz: np.ndarray, w: np.ndarray, n_components: int=10, random_state: int=0) -> Dict[str, np.ndarray]:
    """
    Approximate sample-weighted PCA by scaling rows with sqrt(w) and computing SVD.
    Returns dict with components, explained_variance_ratio, scores, etc.
    """
    # center is already done via z-scoring; ensure zero-mean to numerical tolerance
    # scale each row by sqrt(w)
    sw = np.sqrt(w)[:, None]
    Xw = Xz * sw
    # randomized SVD
    # We implement thin SVD via numpy.linalg.svd for portability;
    # for large N,D consider sklearn randomized SVD or cupy for GPU.
    U, S, VT = np.linalg.svd(Xw, full_matrices=False)
    components = VT[:n_components, :]            # [n_comp, D]
    scores = (U * S)[:, :n_components]           # [N, n_comp]
    # explained variance ratio from singular values
    S2 = S**2
    total_var = np.sum(S2)
    ev = S2[:n_components] / (total_var + 1e-12)
    return {
        "components": components,
        "scores": scores,
        "explained_variance_ratio": ev
    }

def resample_curve(R: np.ndarray, V: np.ndarray, eV: np.ndarray, x_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample V(R) onto x_grid where x=R/Rd is supplied externally by caller.
    We assume caller provides x for spline fit; here we fit in x-space.
    """
    # The caller passes already x=R/Rd values; here we just spline vs x directly.
    # This function is a placeholder as we do the spline in the ingest stage.
    raise NotImplementedError("Use splrep/splev directly in ingest script where x is known.")

def compute_acceleration(V: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Compute g = V^2 / R with SI conversion expected to be handled outside if needed."""
    eps = 1e-12
    return (V**2) / np.maximum(R, eps)

def principal_angles(U: np.ndarray, V: np.ndarray, k: int) -> np.ndarray:
    """
    Compute principal angles between k-dim subspaces spanned by columns of U and V.
    U: [D, k], V: [D, k] with orthonormal columns
    Returns angles in radians, ascending order.
    """
    # Orthonormalize (QR)
    Uq, _ = np.linalg.qr(U[:, :k])
    Vq, _ = np.linalg.qr(V[:, :k])
    M = Uq.T @ Vq
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.arccos(s)
