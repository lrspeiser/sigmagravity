from __future__ import annotations

import numpy as np


def transition_bandpass(acceleration_ratio) -> np.ndarray:
    """Return the v5A geometric source ``x^4/(1+x^4)^2``.

    In the static weak branch, ``x=g_phi/a_sigma`` and the squared trace
    invariant is ``Z=x^4``.  The source is zero in flat space, peaks at the
    universal acceleration transition, and falls as ``x^-4`` at high field.
    """
    ratio = np.asarray(acceleration_ratio, dtype=float)
    if np.any(~np.isfinite(ratio)) or np.any(ratio < 0.0):
        raise ValueError("acceleration_ratio must be finite and non-negative")
    fourth = np.square(np.square(ratio))
    return fourth / np.square(1.0 + fourth)


def bounded_disformal_fraction(nonmetricity_ratio, anisotropy: float) -> np.ndarray:
    """Magnitude of the v5A rank-one disformal correction.

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
