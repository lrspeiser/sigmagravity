#!/usr/bin/env python3
"""
stream_seeking_anisotropic.py
=============================

A small, dependency-light prototype for the *actual* "stream-seeking" idea:

1) Anisotropic gravity operator (direction matters):
   Solve a variable-coefficient anisotropic Poisson-like equation in 2D:

       ∇·[(I + κ w(x) ŝ ŝᵀ) ∇Φ] = 4π G ρ

   where:
     - ŝ(x) is a *local stream direction* (unit vector field)
     - w(x) ∈ [0,1] is an optional "stream intensity / coherence weight"
     - κ controls anisotropy strength (κ > -1 for SPD)

   This is implemented with a symmetric finite-volume discretization and a
   matrix-free Conjugate Gradient solve on the interior (Dirichlet Φ=0 boundary).

2) A synthetic 2D regression test (inherently non-radial):
   A toy "beam / filament" channels the potential and changes a ray-traced
   deflection statistic, which a scalar amplitude-rescaling cannot mimic.

No SciPy required; only numpy.

NOTE ON PHYSICS:
- This is a *toy* solver/test. In real gravity:
  - the Newtonian Poisson equation is 3D: ∇²Φ = 4πGρ
  - lensing uses projected potentials / GR corrections
- Here we use a 2D plane as a synthetic "lens plane" to test directionality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any, Tuple, Optional
import numpy as np


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def normalize_vector_field(vx: np.ndarray, vy: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return unit vectors ŝ = v / ||v|| with shape (H, W, 2)."""
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)
    n = np.sqrt(vx * vx + vy * vy) + eps
    shat = np.stack([vx / n, vy / n], axis=-1)
    return shat


def build_diffusion_tensor(
    shat: np.ndarray,
    kappa: float,
    weight: Optional[np.ndarray] = None,
    kappa_min: float = -0.99,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build symmetric positive-definite diffusion tensor components for:

      D = I + κ_eff ŝ ŝᵀ
      κ_eff(x) = κ * weight(x)  (if weight is provided), else κ

    Returns:
      (Dxx, Dxy, Dyy) each shape (H, W)
    """
    shat = np.asarray(shat, dtype=float)
    assert shat.ndim == 3 and shat.shape[-1] == 2, "shat must be (H,W,2)"
    sx = shat[..., 0]
    sy = shat[..., 1]

    if weight is None:
        keff = float(kappa)
    else:
        w = np.clip(np.asarray(weight, dtype=float), 0.0, 1.0)
        keff = float(kappa) * w

    # Ensure SPD: keff > -1 everywhere
    if np.isscalar(keff):
        keff = max(float(keff), kappa_min)
    else:
        keff = np.maximum(keff, kappa_min)

    Dxx = 1.0 + keff * sx * sx
    Dyy = 1.0 + keff * sy * sy
    Dxy = keff * sx * sy
    return Dxx, Dxy, Dyy


def _central_grad_x(phi: np.ndarray, dx: float) -> np.ndarray:
    """Central gradient ∂φ/∂x on cell centers; one-sided at boundaries."""
    phi = np.asarray(phi, dtype=float)
    gx = np.zeros_like(phi)
    gx[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2.0 * dx)
    gx[:, 0] = (phi[:, 1] - phi[:, 0]) / dx
    gx[:, -1] = (phi[:, -1] - phi[:, -2]) / dx
    return gx


def _central_grad_y(phi: np.ndarray, dy: float) -> np.ndarray:
    """Central gradient ∂φ/∂y on cell centers; one-sided at boundaries."""
    phi = np.asarray(phi, dtype=float)
    gy = np.zeros_like(phi)
    gy[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2.0 * dy)
    gy[0, :] = (phi[1, :] - phi[0, :]) / dy
    gy[-1, :] = (phi[-1, :] - phi[-2, :]) / dy
    return gy


def apply_div_D_grad(phi: np.ndarray, Dxx: np.ndarray, Dxy: np.ndarray, Dyy: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Compute A(phi) = ∇·(D ∇phi) using a symmetric finite-volume style scheme.

    - Gradients are computed on faces using neighbor differences for the normal
      component and averaged cell-centered gradients for the tangential component.
    - D is averaged to faces.

    Returns:
      Aphi shape (H, W)
    """
    phi = np.asarray(phi, dtype=float)
    H, W = phi.shape
    assert Dxx.shape == (H, W) and Dxy.shape == (H, W) and Dyy.shape == (H, W)

    # Cell-centered grads (for tangential averaging)
    phix = _central_grad_x(phi, dx)   # (H,W)
    phiy = _central_grad_y(phi, dy)   # (H,W)

    # -----------------------------
    # Flux on vertical faces (x-faces): between j and j+1
    # shape: (H, W-1)
    # -----------------------------
    Dxx_xface = 0.5 * (Dxx[:, :-1] + Dxx[:, 1:])
    Dxy_xface = 0.5 * (Dxy[:, :-1] + Dxy[:, 1:])

    gradx_xface = (phi[:, 1:] - phi[:, :-1]) / dx
    grady_xface = 0.5 * (phiy[:, :-1] + phiy[:, 1:])  # tangential component

    Fx = Dxx_xface * gradx_xface + Dxy_xface * grady_xface

    # -----------------------------
    # Flux on horizontal faces (y-faces): between i and i+1
    # shape: (H-1, W)
    # -----------------------------
    Dxy_yface = 0.5 * (Dxy[:-1, :] + Dxy[1:, :])  # Dyx = Dxy
    Dyy_yface = 0.5 * (Dyy[:-1, :] + Dyy[1:, :])

    grady_yface = (phi[1:, :] - phi[:-1, :]) / dy
    gradx_yface = 0.5 * (phix[:-1, :] + phix[1:, :])  # tangential component

    Fy = Dxy_yface * gradx_yface + Dyy_yface * grady_yface

    # Divergence back to cell centers
    Aphi = np.zeros_like(phi)

    # x-div: for interior j=1..W-2
    Aphi[:, 1:-1] += (Fx[:, 1:] - Fx[:, :-1]) / dx

    # y-div: for interior i=1..H-2
    Aphi[1:-1, :] += (Fy[1:, :] - Fy[:-1, :]) / dy

    return Aphi


def cg_solve(
    apply_A: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    tol: float = 1e-8,
    max_iter: int = 2000,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Matrix-free Conjugate Gradient solve for SPD systems A x = b.

    Returns:
      x, info dict (iters, converged, rel_resid)
    """
    b = np.asarray(b, dtype=float)
    x = np.zeros_like(b) if x0 is None else np.asarray(x0, dtype=float).copy()

    r = b - apply_A(x)
    p = r.copy()
    rsold = float(np.dot(r, r))

    bnorm = float(np.sqrt(np.dot(b, b))) + 1e-30
    rel = float(np.sqrt(rsold) / bnorm)
    if rel < tol:
        return x, {"iters": 0, "converged": True, "rel_resid": rel}

    for it in range(1, max_iter + 1):
        Ap = apply_A(p)
        denom = float(np.dot(p, Ap)) + 1e-30
        alpha = rsold / denom
        x += alpha * p
        r -= alpha * Ap
        rsnew = float(np.dot(r, r))
        rel = float(np.sqrt(rsnew) / bnorm)
        if rel < tol:
            return x, {"iters": it, "converged": True, "rel_resid": rel}
        p = r + (rsnew / (rsold + 1e-30)) * p
        rsold = rsnew

    return x, {"iters": max_iter, "converged": False, "rel_resid": rel}


def solve_anisotropic_poisson_dirichlet(
    rho: np.ndarray,
    shat: np.ndarray,
    kappa: float,
    *,
    weight: Optional[np.ndarray] = None,
    dx: float = 1.0,
    dy: float = 1.0,
    G: float = 1.0,
    tol: float = 1e-8,
    max_iter: int = 4000,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Solve ∇·(D ∇Φ) = 4πG ρ  on a 2D grid with Dirichlet boundary Φ=0.

    We solve the SPD system:
        L Φ = b
    where:
        L = -∇·(D∇·)   (SPD with Dirichlet BCs)
        b = -4πG ρ

    Returns:
      Phi (H,W), info
    """
    rho = np.asarray(rho, dtype=float)
    H, W = rho.shape
    Dxx, Dxy, Dyy = build_diffusion_tensor(shat=shat, kappa=kappa, weight=weight)

    # Interior mask: 1..H-2, 1..W-2
    def pack(phi_full: np.ndarray) -> np.ndarray:
        return phi_full[1:-1, 1:-1].ravel()

    def unpack(x: np.ndarray) -> np.ndarray:
        phi = np.zeros((H, W), dtype=float)
        phi[1:-1, 1:-1] = x.reshape((H - 2, W - 2))
        return phi

    rhs_full = 4.0 * np.pi * float(G) * rho
    b = -pack(rhs_full)  # b for LΦ=b

    def apply_L_vec(x: np.ndarray) -> np.ndarray:
        phi = unpack(x)
        Aphi = apply_div_D_grad(phi, Dxx, Dxy, Dyy, dx, dy)  # A=div(D grad)
        Lphi = -Aphi
        return pack(Lphi)

    x0 = np.zeros_like(b)
    x_sol, info = cg_solve(apply_L_vec, b, x0=x0, tol=tol, max_iter=max_iter)
    Phi = unpack(x_sol)
    return Phi, info


# ---------------------------------------------------------------------
# Ray tracing helpers (toy, 2D)
# ---------------------------------------------------------------------

def _bilinear_sample(grid: np.ndarray, x: float, y: float, x_min: float, y_min: float, dx: float, dy: float) -> float:
    """
    Bilinear sample a grid defined on:
      x_j = x_min + j*dx,  j=0..W-1
      y_i = y_min + i*dy,  i=0..H-1
    """
    H, W = grid.shape
    fx = (x - x_min) / dx
    fy = (y - y_min) / dy

    # Clamp to valid interior for bilinear indexing
    fx = np.clip(fx, 0.0, W - 1.000001)
    fy = np.clip(fy, 0.0, H - 1.000001)

    j0 = int(np.floor(fx))
    i0 = int(np.floor(fy))
    j1 = min(j0 + 1, W - 1)
    i1 = min(i0 + 1, H - 1)

    tx = fx - j0
    ty = fy - i0

    v00 = grid[i0, j0]
    v10 = grid[i0, j1]
    v01 = grid[i1, j0]
    v11 = grid[i1, j1]
    v0 = (1 - tx) * v00 + tx * v10
    v1 = (1 - tx) * v01 + tx * v11
    return (1 - ty) * v0 + ty * v1


def trace_rays_2d(
    Phi: np.ndarray,
    *,
    x0s: np.ndarray,
    y_start: float,
    y_end: float,
    n_steps: int,
    x_min: float,
    y_min: float,
    dx: float,
    dy: float,
    c2: float = 1.0,
) -> np.ndarray:
    """
    Toy ray tracing through a weak potential field Phi(x,y).

    We evolve rays approximately via:
        dθ/ds = -(2/c^2) * ∂Φ/∂x
        dx/ds = θ
        y decreases linearly with s (ray propagates in -y direction)

    This is not full GR lensing. It's a simple way to create a 2D/3D
    directional-focusing regression statistic.

    Returns:
      x_end for each ray.
    """
    Phi = np.asarray(Phi, dtype=float)
    H, W = Phi.shape
    # Precompute ∂Φ/∂x on grid
    Phi_x = _central_grad_x(Phi, dx)

    x0s = np.asarray(x0s, dtype=float)
    x = x0s.copy()
    theta = np.zeros_like(x0s)

    total_len = float(y_start - y_end)
    ds = total_len / float(n_steps)

    y = float(y_start)
    for _ in range(n_steps):
        # sample ∂Φ/∂x at each ray position
        gx = np.array([_bilinear_sample(Phi_x, xi, y, x_min, y_min, dx, dy) for xi in x], dtype=float)
        theta += -(2.0 / float(c2)) * gx * ds
        x += theta * ds
        y -= ds

    return x


# ---------------------------------------------------------------------
# Synthetic regression test
# ---------------------------------------------------------------------

@dataclass
class SyntheticStreamTestResult:
    capture_ratio: float
    mean_shift_ratio: float
    capture_iso: float
    capture_aniso: float
    mean_x_iso: float
    mean_x_aniso: float
    solver_iso: Dict[str, Any]
    solver_aniso: Dict[str, Any]


def synthetic_stream_lensing_regression(
    *,
    N: int = 160,
    L: float = 1.0,
    mass_offset_x: float = 0.30,
    mass_sigma: float = 0.12,
    stream_sigma: float = 0.06,
    kappa: float = 6.0,
    G: float = 1.0,
    n_rays: int = 80,
    n_steps: int = 240,
    capture_width: float = 0.08,
    tol: float = 1e-7,
    max_iter: int = 6000,
) -> SyntheticStreamTestResult:
    """
    A 2D toy test that *forces* a directional effect:

    - A positive mass blob sits at (x=+mass_offset_x, y=0).
    - A "stream/filament" sits at x=0 with direction ŝ=(0,1) (vertical).
      We localize anisotropy with weight w(x)=exp(-x^2/(2 stream_sigma^2)).
    - We solve:
        baseline:  κ = 0   (isotropic)
        anisotropic: κ > 0 (stream-seeking)

    Then we trace rays from y=+L to y=-L and compute:
      - capture fraction within |x| < capture_width (toward filament)
      - mean x_end shift (should move toward 0 with anisotropy)

    Returns a structured result for easy use in your regression harness.
    """
    x_min, x_max = -L, L
    y_min, y_max = -L, L
    dx = (x_max - x_min) / (N - 1)
    dy = (y_max - y_min) / (N - 1)

    xs = np.linspace(x_min, x_max, N)
    ys = np.linspace(y_min, y_max, N)
    X, Y = np.meshgrid(xs, ys)

    # Mass density: Gaussian blob
    rho = np.exp(-((X - mass_offset_x) ** 2 + (Y - 0.0) ** 2) / (2.0 * mass_sigma ** 2))

    # Stream intensity weight (localize anisotropy around x=0)
    w = np.exp(-(X ** 2) / (2.0 * stream_sigma ** 2))
    w = np.clip(w, 0.0, 1.0)

    # Stream direction: vertical (0,1) everywhere (could be spatially varying)
    shat = np.zeros((N, N, 2), dtype=float)
    shat[..., 1] = 1.0

    # Solve isotropic baseline (κ=0)
    Phi_iso, info_iso = solve_anisotropic_poisson_dirichlet(
        rho=rho,
        shat=shat,
        kappa=0.0,
        weight=None,
        dx=dx,
        dy=dy,
        G=G,
        tol=tol,
        max_iter=max_iter,
    )

    # Solve anisotropic (κ>0) localized by w
    Phi_aniso, info_aniso = solve_anisotropic_poisson_dirichlet(
        rho=rho,
        shat=shat,
        kappa=kappa,
        weight=w,
        dx=dx,
        dy=dy,
        G=G,
        tol=tol,
        max_iter=max_iter,
    )

    # Rays start positions
    x0s = np.linspace(-0.8 * L, 0.8 * L, n_rays)

    x_end_iso = trace_rays_2d(
        Phi_iso,
        x0s=x0s,
        y_start=y_max,
        y_end=y_min,
        n_steps=n_steps,
        x_min=x_min,
        y_min=y_min,
        dx=dx,
        dy=dy,
        c2=1.0,
    )

    x_end_aniso = trace_rays_2d(
        Phi_aniso,
        x0s=x0s,
        y_start=y_max,
        y_end=y_min,
        n_steps=n_steps,
        x_min=x_min,
        y_min=y_min,
        dx=dx,
        dy=dy,
        c2=1.0,
    )

    # "Capture" = how many rays end near filament center x=0
    capture_iso = float(np.mean(np.abs(x_end_iso) < capture_width))
    capture_aniso = float(np.mean(np.abs(x_end_aniso) < capture_width))
    capture_ratio = (capture_aniso + 1e-12) / (capture_iso + 1e-12)

    mean_x_iso = float(np.mean(x_end_iso))
    mean_x_aniso = float(np.mean(x_end_aniso))

    # Expectation: anisotropy pulls mean toward the filament (x=0), i.e. |mean| shrinks
    mean_shift_ratio = (abs(mean_x_aniso) + 1e-12) / (abs(mean_x_iso) + 1e-12)

    return SyntheticStreamTestResult(
        capture_ratio=capture_ratio,
        mean_shift_ratio=mean_shift_ratio,
        capture_iso=capture_iso,
        capture_aniso=capture_aniso,
        mean_x_iso=mean_x_iso,
        mean_x_aniso=mean_x_aniso,
        solver_iso=info_iso,
        solver_aniso=info_aniso,
    )


if __name__ == "__main__":
    # Quick demo run
    out = synthetic_stream_lensing_regression()
    print("Synthetic stream lensing regression")
    print(f"  capture_iso   = {out.capture_iso:.3f}")
    print(f"  capture_aniso = {out.capture_aniso:.3f}")
    print(f"  capture_ratio = {out.capture_ratio:.3f}  (want > 1)")
    print(f"  mean_x_iso    = {out.mean_x_iso:+.4f}")
    print(f"  mean_x_aniso  = {out.mean_x_aniso:+.4f}")
    print(f"  mean_shift_ratio = {out.mean_shift_ratio:.3f}  (want < 1)")
    print(f"  solver_iso:   {out.solver_iso}")
    print(f"  solver_aniso: {out.solver_aniso}")

