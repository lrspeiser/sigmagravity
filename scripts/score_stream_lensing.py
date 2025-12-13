#!/usr/bin/env python3
"""
Scoring framework for A/B testing anisotropic gravity predictions.

Compares κ=0 (baseline isotropic) vs κ>0 (anisotropic) predictions
against observational targets to determine if anisotropy improves
predictions of how gravity impacts objects (light rays, particles, etc.).
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass

try:
    from stream_seeking_anisotropic import (
        solve_anisotropic_poisson_dirichlet,
        trace_rays_2d,
        _central_grad_x,
        _central_grad_y,
        _bilinear_sample,
    )
    _HAS_ANISOTROPIC = True
except ImportError:
    _HAS_ANISOTROPIC = False


@dataclass
class Scene:
    """A gravitational scene: density, stream direction, and metadata."""
    rho: np.ndarray  # (H, W) density field
    shat: np.ndarray  # (H, W, 2) unit stream direction field
    weight: Optional[np.ndarray]  # (H, W) stream intensity weight [0,1]
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    dx: float
    dy: float
    N: int
    L: float


@dataclass
class PredictionScore:
    """Scores comparing predictions to observations."""
    rms: float
    chi2: Optional[float] = None
    mean_abs_error: float = None
    max_error: float = None
    n_points: int = None


def build_scene(
    N: int = 160,
    L: float = 1.0,
    mass_offset_x: float = 0.30,
    mass_sigma: float = 0.12,
    stream_sigma: float = 0.06,
    stream_direction: Tuple[float, float] = (0.0, 1.0),
) -> Scene:
    """
    Build a synthetic gravitational scene.
    
    Args:
        N: Grid resolution
        L: Domain size [-L, L]
        mass_offset_x: x-position of mass blob
        mass_sigma: Width of mass blob
        stream_sigma: Width of stream/filament
        stream_direction: (sx, sy) unit direction of stream
    
    Returns:
        Scene object
    """
    x_min, x_max = -L, L
    y_min, y_max = -L, L
    xs = np.linspace(x_min, x_max, N)
    ys = np.linspace(y_min, y_max, N)
    dx = (x_max - x_min) / (N - 1)
    dy = (y_max - y_min) / (N - 1)
    X, Y = np.meshgrid(xs, ys)

    # Baryonic density: Gaussian blob
    rho = np.exp(-((X - mass_offset_x)**2 + Y**2) / (2 * mass_sigma**2))

    # Stream intensity weight (localized around stream)
    w = np.exp(-(X**2) / (2 * stream_sigma**2))
    w = np.clip(w, 0.0, 1.0)

    # Stream direction field
    sx, sy = stream_direction
    norm = np.sqrt(sx**2 + sy**2) + 1e-12
    shat = np.zeros((N, N, 2), dtype=float)
    shat[..., 0] = sx / norm
    shat[..., 1] = sy / norm

    return Scene(
        rho=rho,
        shat=shat,
        weight=w,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        dx=dx,
        dy=dy,
        N=N,
        L=L,
    )


def solve_potential(
    scene: Scene,
    kappa: float,
    G: float = 1.0,
    tol: float = 1e-7,
    max_iter: int = 6000,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Solve for gravitational potential given a scene and anisotropy parameter.
    
    Returns:
        Phi (H, W), solver info dict
    """
    if not _HAS_ANISOTROPIC:
        raise ImportError("stream_seeking_anisotropic module not available")
    
    return solve_anisotropic_poisson_dirichlet(
        rho=scene.rho,
        shat=scene.shat,
        kappa=kappa,
        weight=scene.weight,
        dx=scene.dx,
        dy=scene.dy,
        G=G,
        tol=tol,
        max_iter=max_iter,
    )


def predict_ray_endpoints(
    Phi: np.ndarray,
    scene: Scene,
    n_rays: int = 80,
    n_steps: int = 240,
    c2: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict ray endpoint positions for a bundle of rays.
    
    Args:
        Phi: Potential field (H, W)
        scene: Scene metadata
        n_rays: Number of rays
        n_steps: Integration steps
        c2: "Lens strength" calibration factor
    
    Returns:
        x0s: Initial x positions
        x_end: Final x positions after ray tracing
    """
    if not _HAS_ANISOTROPIC:
        raise ImportError("stream_seeking_anisotropic module not available")
    
    x0s = np.linspace(-0.8 * scene.L, 0.8 * scene.L, n_rays)
    x_end = trace_rays_2d(
        Phi,
        x0s=x0s,
        y_start=scene.y_max,
        y_end=scene.y_min,
        n_steps=n_steps,
        x_min=scene.x_min,
        y_min=scene.y_min,
        dx=scene.dx,
        dy=scene.dy,
        c2=c2,
    )
    return x0s, x_end


def integrate_particle_trajectory(
    Phi: np.ndarray,
    scene: Scene,
    x0: float,
    y0: float,
    vx0: float,
    vy0: float,
    dt: float,
    n_steps: int,
) -> np.ndarray:
    """
    Integrate a test particle trajectory through potential field.
    
    Uses velocity-Verlet integrator: ẍ = -∇Φ
    
    Args:
        Phi: Potential field (H, W)
        scene: Scene metadata
        x0, y0: Initial position
        vx0, vy0: Initial velocity
        dt: Time step
        n_steps: Number of integration steps
    
    Returns:
        traj: Array of shape (n_steps, 4) with columns [x, y, vx, vy]
    """
    if not _HAS_ANISOTROPIC:
        raise ImportError("stream_seeking_anisotropic module not available")
    
    Phi_x = _central_grad_x(Phi, scene.dx)
    Phi_y = _central_grad_y(Phi, scene.dy)

    x, y, vx, vy = float(x0), float(y0), float(vx0), float(vy0)
    traj = []

    for _ in range(n_steps):
        # Compute acceleration from potential gradient
        ax = -_bilinear_sample(Phi_x, x, y, scene.x_min, scene.y_min, scene.dx, scene.dy)
        ay = -_bilinear_sample(Phi_y, x, y, scene.x_min, scene.y_min, scene.dx, scene.dy)

        # Velocity-Verlet half-step
        vx_half = vx + 0.5 * dt * ax
        vy_half = vy + 0.5 * dt * ay

        # Position update
        x += dt * vx_half
        y += dt * vy_half

        # Second half-step
        ax2 = -_bilinear_sample(Phi_x, x, y, scene.x_min, scene.y_min, scene.dx, scene.dy)
        ay2 = -_bilinear_sample(Phi_y, x, y, scene.x_min, scene.y_min, scene.dx, scene.dy)

        vx = vx_half + 0.5 * dt * ax2
        vy = vy_half + 0.5 * dt * ay2

        traj.append([x, y, vx, vy])

    return np.array(traj)


def compute_scattering_angle(vx: float, vy: float, initial_direction: Tuple[float, float] = (0.0, -1.0)) -> float:
    """
    Compute scattering angle relative to initial direction.
    
    Args:
        vx, vy: Final velocity components
        initial_direction: (vx0, vy0) initial direction
    
    Returns:
        Scattering angle in radians
    """
    vx0, vy0 = initial_direction
    # Angle of final velocity
    angle_final = np.arctan2(vy, vx)
    # Angle of initial velocity
    angle_initial = np.arctan2(vy0, vx0)
    # Scattering angle
    return angle_final - angle_initial


def score_predictions(
    pred: np.ndarray,
    obs: np.ndarray,
    obs_uncertainty: Optional[np.ndarray] = None,
) -> PredictionScore:
    """
    Score predictions against observations.
    
    Args:
        pred: Predicted values
        obs: Observed values
        obs_uncertainty: Optional uncertainties for χ² calculation
    
    Returns:
        PredictionScore with RMS, χ², etc.
    """
    pred = np.asarray(pred, dtype=float)
    obs = np.asarray(obs, dtype=float)
    
    residuals = pred - obs
    rms = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    max_err = float(np.max(np.abs(residuals)))
    
    chi2 = None
    if obs_uncertainty is not None:
        sigma = np.asarray(obs_uncertainty, dtype=float)
        chi2 = float(np.sum((residuals / sigma)**2))
    
    return PredictionScore(
        rms=rms,
        chi2=chi2,
        mean_abs_error=mae,
        max_error=max_err,
        n_points=len(pred),
    )


def compare_models_ab(
    scene: Scene,
    kappa_baseline: float,
    kappa_aniso: float,
    observations: np.ndarray,
    obs_uncertainty: Optional[np.ndarray] = None,
    prediction_type: str = "ray_endpoints",
    **prediction_kwargs,
) -> Dict[str, Any]:
    """
    A/B test: Compare baseline (κ=0) vs anisotropic (κ>0) predictions.
    
    Args:
        scene: Gravitational scene
        kappa_baseline: Baseline anisotropy (typically 0.0)
        kappa_aniso: Anisotropic model parameter
        observations: Observed values to compare against
        obs_uncertainty: Optional uncertainties
        prediction_type: "ray_endpoints" or "particle_trajectory"
        **prediction_kwargs: Additional arguments for prediction function
    
    Returns:
        Dict with scores for both models and comparison metrics
    """
    # Solve potentials
    Phi_baseline, info_baseline = solve_potential(scene, kappa_baseline)
    Phi_aniso, info_aniso = solve_potential(scene, kappa_aniso)
    
    # Make predictions
    if prediction_type == "ray_endpoints":
        _, pred_baseline = predict_ray_endpoints(Phi_baseline, scene, **prediction_kwargs)
        _, pred_aniso = predict_ray_endpoints(Phi_aniso, scene, **prediction_kwargs)
    elif prediction_type == "particle_trajectory":
        # For trajectory, extract final position or scattering angle
        traj_baseline = integrate_particle_trajectory(Phi_baseline, scene, **prediction_kwargs)
        traj_aniso = integrate_particle_trajectory(Phi_aniso, scene, **prediction_kwargs)
        # Use final position or scattering angle as prediction
        pred_baseline = traj_baseline[-1, :2]  # Final (x, y)
        pred_aniso = traj_aniso[-1, :2]
    else:
        raise ValueError(f"Unknown prediction_type: {prediction_type}")
    
    # Score both models
    score_baseline = score_predictions(pred_baseline, observations, obs_uncertainty)
    score_aniso = score_predictions(pred_aniso, observations, obs_uncertainty)
    
    # Comparison metrics
    rms_improvement = (score_baseline.rms - score_aniso.rms) / score_baseline.rms if score_baseline.rms > 0 else 0.0
    chi2_delta = None
    if score_baseline.chi2 is not None and score_aniso.chi2 is not None:
        chi2_delta = score_baseline.chi2 - score_aniso.chi2
    
    return {
        "baseline": {
            "kappa": kappa_baseline,
            "score": score_baseline,
            "solver_info": info_baseline,
        },
        "anisotropic": {
            "kappa": kappa_aniso,
            "score": score_aniso,
            "solver_info": info_aniso,
        },
        "comparison": {
            "rms_improvement": rms_improvement,
            "chi2_delta": chi2_delta,
            "aniso_better": score_aniso.rms < score_baseline.rms,
        },
    }


if __name__ == "__main__":
    # Example: Synthetic A/B test
    print("Building scene...")
    scene = build_scene(N=160, L=1.0)
    
    print("Generating synthetic 'observations' (κ_true=6.0)...")
    Phi_true, _ = solve_potential(scene, kappa=6.0)
    x0s, x_obs = predict_ray_endpoints(Phi_true, scene)
    # Add measurement noise
    x_obs = x_obs + np.random.normal(0.0, 0.002, size=x_obs.shape)
    obs_uncertainty = np.full_like(x_obs, 0.002)
    
    print("Running A/B comparison...")
    results = compare_models_ab(
        scene=scene,
        kappa_baseline=0.0,
        kappa_aniso=6.0,
        observations=x_obs,
        obs_uncertainty=obs_uncertainty,
        prediction_type="ray_endpoints",
        n_rays=80,
        n_steps=240,
    )
    
    print("\n" + "=" * 60)
    print("A/B TEST RESULTS")
    print("=" * 60)
    print(f"\nBaseline (κ={results['baseline']['kappa']}):")
    print(f"  RMS: {results['baseline']['score'].rms:.6f}")
    print(f"  χ²:  {results['baseline']['score'].chi2:.2f}" if results['baseline']['score'].chi2 else "  χ²:  N/A")
    
    print(f"\nAnisotropic (κ={results['anisotropic']['kappa']}):")
    print(f"  RMS: {results['anisotropic']['score'].rms:.6f}")
    print(f"  χ²:  {results['anisotropic']['score'].chi2:.2f}" if results['anisotropic']['score'].chi2 else "  χ²:  N/A")
    
    print(f"\nComparison:")
    print(f"  RMS improvement: {results['comparison']['rms_improvement']*100:.2f}%")
    if results['comparison']['chi2_delta'] is not None:
        print(f"  Δχ²: {results['comparison']['chi2_delta']:.2f}")
    print(f"  Anisotropic better: {results['comparison']['aniso_better']}")

