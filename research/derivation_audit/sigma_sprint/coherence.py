"""Coherence estimators that do not use the gravitational outcome."""

from __future__ import annotations

import numpy as np


def coherence_from_moments(streaming_velocity, dispersion_tensor) -> float:
    """Compute |v_stream|^2/(|v_stream|^2 + tr(sigma^2))."""
    stream = np.asarray(streaming_velocity, dtype=float)
    dispersion = np.asarray(dispersion_tensor, dtype=float)
    if stream.shape != (3,) or dispersion.shape != (3, 3):
        raise ValueError("expected a 3-vector and a 3x3 dispersion tensor")
    ordered = float(stream @ stream)
    random = float(np.trace(dispersion))
    if random < -1e-12:
        raise ValueError("dispersion trace cannot be negative")
    denominator = ordered + max(random, 0.0)
    return 0.0 if denominator == 0 else ordered / denominator


def phase_space_coherence(positions, velocities, weights=None, axis=None) -> float:
    """Estimate rotational coherence from independent phase-space samples.

    Positions and velocities must already be expressed in the system
    barycentric frame.  The ordered component is the weighted mean signed
    azimuthal velocity about ``axis``.  Random energy includes the weighted
    variance of azimuthal velocity and the radial/vertical second moments.
    Counterrotating components therefore cancel in the ordered numerator.
    """
    r = np.asarray(positions, dtype=float)
    v = np.asarray(velocities, dtype=float)
    if r.shape != v.shape or r.ndim != 2 or r.shape[1] != 3:
        raise ValueError("positions and velocities must both have shape (N,3)")
    if len(r) < 2:
        raise ValueError("at least two phase-space samples are required")
    w = np.ones(len(r), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (len(r),) or np.any(w < 0) or not np.any(w > 0):
        raise ValueError("weights must be non-negative with at least one positive value")
    w = w / w.sum()
    if axis is None:
        angular_momentum = np.sum(w[:, None] * np.cross(r, v), axis=0)
        norm = np.linalg.norm(angular_momentum)
        if norm == 0:
            return 0.0
        axis_vector = angular_momentum / norm
    else:
        axis_vector = np.asarray(axis, dtype=float)
        if axis_vector.shape != (3,) or np.linalg.norm(axis_vector) == 0:
            raise ValueError("axis must be a non-zero 3-vector")
        axis_vector = axis_vector / np.linalg.norm(axis_vector)
    vertical_position = (r @ axis_vector)[:, None] * axis_vector
    cylindrical_r = r - vertical_position
    radius = np.linalg.norm(cylindrical_r, axis=1)
    valid = radius > 0
    if np.count_nonzero(valid) < 2:
        return 0.0
    e_r = cylindrical_r[valid] / radius[valid, None]
    e_phi = np.cross(np.broadcast_to(axis_vector, e_r.shape), e_r)
    v_valid = v[valid]
    w_valid = w[valid]
    w_valid = w_valid / w_valid.sum()
    v_phi = np.sum(v_valid * e_phi, axis=1)
    v_radial = np.sum(v_valid * e_r, axis=1)
    v_vertical = v_valid @ axis_vector
    mean_phi = float(w_valid @ v_phi)
    ordered = mean_phi**2
    random = float(
        w_valid @ ((v_phi - mean_phi) ** 2 + v_radial**2 + v_vertical**2)
    )
    return 0.0 if ordered + random == 0 else ordered / (ordered + random)


def assert_no_outcome_leakage(feature_names) -> None:
    """Reject coherence features derived from gravity-model outcomes."""
    forbidden = {
        "v_pred",
        "predicted_velocity",
        "gtot",
        "dark_matter_fraction",
        "fdm_re",
        "model_residual",
    }
    normalized = {str(name).strip().lower() for name in feature_names}
    leaked = sorted(normalized & forbidden)
    if leaked:
        raise ValueError(f"coherence feature leakage: {', '.join(leaked)}")
