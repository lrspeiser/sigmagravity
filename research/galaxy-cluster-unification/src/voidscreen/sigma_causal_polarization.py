from __future__ import annotations

import numpy as np


def transition_bandpass(acceleration_ratio) -> np.ndarray:
    """Return the v5-family geometric source ``x^4/(1+x^4)^2``.

    In the static weak branch, ``x=g_phi/a_sigma`` and the squared trace
    invariant is ``Z=x^4``.  The source is zero in flat space, peaks at the
    universal acceleration transition, and falls as ``x^-4`` at high field.
    """
    ratio = np.asarray(acceleration_ratio, dtype=float)
    if np.any(~np.isfinite(ratio)) or np.any(ratio < 0.0):
        raise ValueError("acceleration_ratio must be finite and non-negative")
    fourth = np.square(np.square(ratio))
    return fourth / np.square(1.0 + fourth)


def signed_trace_bandpass(trace_ratio) -> np.ndarray:
    """Return the globally real source ``Y^2/(1+Y^2)^2`` for signed Y."""
    value = np.asarray(trace_ratio, dtype=float)
    if np.any(~np.isfinite(value)):
        raise ValueError("trace_ratio must be finite")
    squared = np.square(value)
    return squared / np.square(1.0 + squared)


def transition_bandpass_y_derivative(y_squared) -> np.ndarray:
    """Return ``d[Y^2/(1+Y^2)^2]/dY`` for signed trace ratio ``Y``."""
    value = np.asarray(y_squared, dtype=float)
    if np.any(~np.isfinite(value)):
        raise ValueError("y_squared must be finite")
    return 2.0 * value * (1.0 - np.square(value)) / np.power(
        1.0 + np.square(value), 3
    )


def bounded_disformal_fraction(nonmetricity_ratio, anisotropy: float) -> np.ndarray:
    """Magnitude of the v5-family rank-one disformal correction.

    ``nonmetricity_ratio`` is ``|W_a W^a|/(4 q_sigma)^2``.  The returned
    fraction is strictly below ``alpha/(1+alpha)`` and therefore below one for
    every finite non-negative anisotropy.
    """
    ratio = np.asarray(nonmetricity_ratio, dtype=float)
    alpha = float(anisotropy)
    if np.any(~np.isfinite(ratio)) or np.any(ratio < 0.0):
        raise ValueError("nonmetricity_ratio must be finite and non-negative")
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("anisotropy must be finite and non-negative")
    return (alpha / (1.0 + alpha)) * ratio / np.sqrt(1.0 + np.square(ratio))


def local_transport_eigenvalues(
    nonmetricity_ratio,
    anisotropy: float,
    *,
    orientation: str,
) -> dict[str, np.ndarray]:
    """Return adapted-frame kinetic eigenvalues for spacelike/timelike W.

    With ``G_sigma^ab=g^ab-B W^a W^b``, a spacelike W reduces one spatial
    eigenvalue to ``1-f``.  A timelike W increases the magnitude of the time
    eigenvalue to ``1+f``.  Both cases keep a Lorentzian, subluminal scalar cone.
    """
    fraction = bounded_disformal_fraction(nonmetricity_ratio, anisotropy)
    ones = np.ones_like(fraction)
    if orientation == "spacelike":
        return {
            "time_magnitude": ones,
            "parallel_spatial": ones - fraction,
            "transverse_spatial": ones,
        }
    if orientation == "timelike":
        return {
            "time_magnitude": ones + fraction,
            "parallel_spatial": ones,
            "transverse_spatial": ones,
        }
    raise ValueError("orientation must be 'spacelike' or 'timelike'")


def maximum_characteristic_speed(
    nonmetricity_ratio,
    anisotropy: float,
    *,
    orientation: str,
) -> np.ndarray:
    """Return the largest local high-frequency scalar speed in metric units."""
    eigenvalues = local_transport_eigenvalues(
        nonmetricity_ratio, anisotropy, orientation=orientation
    )
    time = eigenvalues["time_magnitude"]
    spatial = np.maximum(
        eigenvalues["parallel_spatial"], eigenvalues["transverse_spatial"]
    )
    return np.sqrt(spatial / time)


def flrw_polarization_mode(hubble_over_q, anisotropy: float) -> dict[str, np.ndarray]:
    """Return the quadratic v5B scalar mode on its ``sigma=0`` FLRW branch.

    In coincident gauge ``W_0=6H``.  The dimensionless rank-one transport
    argument is therefore ``|W^2|/(4q_sigma)^2=(9/4)(H/q_sigma)^2``.
    The free polarization has unit spatial kinetic coefficient, time kinetic
    magnitude ``1+f``, sound speed squared ``1/(1+f)``, and the same factor
    multiplying its inverse-range mass squared.
    """
    ratio = np.asarray(hubble_over_q, dtype=float)
    if np.any(~np.isfinite(ratio)):
        raise ValueError("hubble_over_q must be finite")
    invariant_ratio = 2.25 * np.square(ratio)
    eigenvalues = local_transport_eigenvalues(
        invariant_ratio, anisotropy, orientation="timelike"
    )
    time = eigenvalues["time_magnitude"]
    spatial = eigenvalues["parallel_spatial"]
    return {
        "time_kinetic": time,
        "spatial_gradient": spatial,
        "sound_speed_squared": spatial / time,
        "mass_squared_times_L_squared": 1.0 / time,
    }


def static_reduced_velocity_lagrangian(
    velocities,
    weyl_spatial,
    polarization_gradient,
    trace_spatial,
    sigma_background: float,
    anisotropy: float,
    *,
    q_sigma: float = 1.0,
    polarization_weight: float = 1.0,
) -> float:
    """Return the exact local v5B kinetic reduction used by the ADM screen.

    ``velocities=(x,h,s)`` denotes ``x=dot(log N)``, the isotropic metric
    velocity ``h=dot(log a)``, and ``s=dot(sigma)``.  At a locally static
    background, ``tilde(Q)_0=2x`` and ``W_0=6(h-x)``.  The omitted terms are
    independent of these velocities and cannot change the Hessian rank.

    Units are rescaled locally; the positive overall action normalization and
    spatial volume are omitted.  This reduction is a necessary degeneracy
    screen, not a replacement for the full field equations.
    """
    velocity = np.asarray(velocities, dtype=float)
    weyl = np.asarray(weyl_spatial, dtype=float)
    gradient = np.asarray(polarization_gradient, dtype=float)
    trace = np.asarray(trace_spatial, dtype=float)
    alpha = float(anisotropy)
    sigma = float(sigma_background)
    q_value = float(q_sigma)
    eta = float(polarization_weight)
    if velocity.shape != (3,):
        raise ValueError("velocities must contain (lapse, scale, sigma) rates")
    if weyl.ndim != 1 or gradient.shape != weyl.shape or trace.shape != weyl.shape:
        raise ValueError("the three spatial vectors must have matching shapes")
    if np.any(~np.isfinite(np.concatenate((velocity, weyl, gradient, trace)))):
        raise ValueError("all velocities and background vectors must be finite")
    if not np.isfinite(sigma) or not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("sigma must be finite and anisotropy non-negative")
    if not np.isfinite(q_value) or q_value <= 0.0:
        raise ValueError("q_sigma must be finite and positive")
    if not np.isfinite(eta) or eta <= 0.0:
        raise ValueError("polarization_weight must be finite and positive")

    lapse_rate, scale_rate, sigma_rate = velocity
    weyl_time = 6.0 * (scale_rate - lapse_rate)
    spatial_weyl_squared = float(np.dot(weyl, weyl))
    weyl_contraction = spatial_weyl_squared - weyl_time**2
    denominator = np.sqrt(weyl_contraction**2 + (4.0 * q_value) ** 4)
    coefficient = alpha / (1.0 + alpha)
    effective_00 = -1.0 - coefficient * weyl_time**2 / denominator
    effective_0i = coefficient * weyl_time * weyl / denominator
    effective_ij = np.eye(weyl.size) - coefficient * np.outer(weyl, weyl) / denominator
    scalar_kinetic = (
        effective_00 * sigma_rate**2
        + 2.0 * sigma_rate * float(np.dot(effective_0i, gradient))
        + float(gradient @ effective_ij @ gradient)
    )
    trace_invariant = (
        float(np.dot(trace, trace)) - 4.0 * lapse_rate**2
    ) / (4.0 * q_value**2)
    source = float(signed_trace_bandpass(trace_invariant))
    return float(
        -6.0 * scale_rate**2
        - eta * scalar_kinetic
        + 2.0 * eta * sigma * source
    )


def static_reduced_kinetic_hessian(
    weyl_spatial,
    polarization_gradient,
    trace_spatial,
    sigma_background: float,
    anisotropy: float,
    *,
    q_sigma: float = 1.0,
    polarization_weight: float = 1.0,
) -> np.ndarray:
    """Return the analytic ``(dot N, dot a, dot sigma)`` Hessian at rest.

    STEGR plus a canonical scalar has rank two: the lapse row is null and its
    constraint removes the negative conformal metric direction.  A healthy
    v5B completion must retain an identically degenerate kinetic matrix.  The
    source and transport terms below instead make the matrix full rank on
    generic polarized static backgrounds.
    """
    weyl = np.asarray(weyl_spatial, dtype=float)
    gradient = np.asarray(polarization_gradient, dtype=float)
    trace = np.asarray(trace_spatial, dtype=float)
    alpha = float(anisotropy)
    sigma = float(sigma_background)
    q_value = float(q_sigma)
    eta = float(polarization_weight)
    if weyl.ndim != 1 or gradient.shape != weyl.shape or trace.shape != weyl.shape:
        raise ValueError("the three finite spatial vectors must have matching shapes")
    if np.any(~np.isfinite(np.concatenate((weyl, gradient, trace)))):
        raise ValueError("the three spatial vectors must be finite")
    if not np.isfinite(sigma) or not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("sigma must be finite and anisotropy non-negative")
    if not np.isfinite(q_value) or q_value <= 0.0:
        raise ValueError("q_sigma must be finite and positive")
    if not np.isfinite(eta) or eta <= 0.0:
        raise ValueError("polarization_weight must be finite and positive")

    weyl_squared = float(np.dot(weyl, weyl))
    projection = float(np.dot(weyl, gradient))
    denominator = np.sqrt(weyl_squared**2 + (4.0 * q_value) ** 4)
    coefficient = alpha / (1.0 + alpha)
    cross = coefficient * projection / denominator
    curvature = (
        coefficient
        * projection**2
        * weyl_squared
        / denominator**3
    )
    trace_invariant = float(np.dot(trace, trace)) / (4.0 * q_value**2)
    source_curvature = (
        -4.0
        * sigma
        * float(transition_bandpass_y_derivative(trace_invariant))
        / q_value**2
    )
    return np.asarray(
        [
            [
                eta * (72.0 * curvature + source_curvature),
                -72.0 * eta * curvature,
                12.0 * eta * cross,
            ],
            [
                -72.0 * eta * curvature,
                -12.0 + 72.0 * eta * curvature,
                -12.0 * eta * cross,
            ],
            [12.0 * eta * cross, -12.0 * eta * cross, 2.0 * eta],
        ],
        dtype=float,
    )


def minimum_static_operator_eigenvalue(
    nonmetricity_ratio, anisotropy: float
) -> np.ndarray:
    """Return the smallest eigenvalue of ``1-L^2 div(K grad)``'s symbol.

    The zero-wavenumber mass contribution is normalized to one.  Positivity of
    the spatial kinetic eigenvalues then makes every Fourier symbol at least
    one, proving uniqueness of the regular static decaying branch.
    """
    spatial = local_transport_eigenvalues(
        nonmetricity_ratio, anisotropy, orientation="spacelike"
    )
    minimum_spatial = np.minimum(
        spatial["parallel_spatial"], spatial["transverse_spatial"]
    )
    if np.any(minimum_spatial <= 0.0):
        raise FloatingPointError("the spatial transport tensor lost positivity")
    return np.ones_like(minimum_spatial)


def weak_transport_tensor(weyl_gradient_ratio, anisotropy: float) -> np.ndarray:
    """Return the static weak ``K^ij`` for ``u=grad(W)/a_sigma``."""
    vector = np.asarray(weyl_gradient_ratio, dtype=float)
    alpha = float(anisotropy)
    if vector.ndim == 0 or np.any(~np.isfinite(vector)):
        raise ValueError("weyl_gradient_ratio must be a finite vector array")
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("anisotropy must be finite and non-negative")
    squared = np.sum(np.square(vector), axis=-1)
    denominator = np.sqrt(1.0 + np.square(squared))
    coefficient = alpha / (1.0 + alpha)
    identity = np.eye(vector.shape[-1])
    return identity - coefficient * (
        np.einsum("...i,...j->...ij", vector, vector) / denominator[..., None, None]
    )


def weak_transport_gradient_contraction(
    weyl_gradient_ratio,
    polarization_gradient,
    anisotropy: float,
) -> np.ndarray:
    """Return ``(partial K^jk/partial u_i) sigma_j sigma_k`` analytically."""
    vector = np.asarray(weyl_gradient_ratio, dtype=float)
    gradient = np.asarray(polarization_gradient, dtype=float)
    alpha = float(anisotropy)
    if vector.shape != gradient.shape or vector.ndim == 0:
        raise ValueError("the two finite vector arrays must have matching shapes")
    if np.any(~np.isfinite(vector + gradient)):
        raise ValueError("the two vector arrays must be finite")
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("anisotropy must be finite and non-negative")
    squared = np.sum(np.square(vector), axis=-1)
    denominator = np.sqrt(1.0 + np.square(squared))
    dot = np.sum(vector * gradient, axis=-1)
    coefficient = alpha / (1.0 + alpha)
    return -2.0 * coefficient * (
        gradient * (dot / denominator)[..., None]
        - vector
        * (squared * np.square(dot) / np.power(denominator, 3))[..., None]
    )
