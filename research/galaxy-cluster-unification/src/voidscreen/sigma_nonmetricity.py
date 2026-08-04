from __future__ import annotations

import numpy as np


def standard_action_primitive(x) -> np.ndarray:
    """Return the regular primitive with dF/dx=sqrt(x/(1+x))."""
    value = np.asarray(x, dtype=float)
    if np.any(value < 0.0):
        raise ValueError("the dimensionless nonmetricity invariant must be non-negative")
    output = np.empty_like(value)
    small = value < 1e-4
    small_value = value[small]
    output[small] = (
        (2.0 / 3.0) * np.power(small_value, 1.5)
        - (1.0 / 5.0) * np.power(small_value, 2.5)
        + (3.0 / 28.0) * np.power(small_value, 3.5)
        - (5.0 / 72.0) * np.power(small_value, 4.5)
    )
    regular = value[~small]
    output[~small] = np.sqrt(regular * (1.0 + regular)) - np.arcsinh(np.sqrt(regular))
    return output


def standard_mu(x) -> np.ndarray:
    """Derivative of :func:`standard_action_primitive`."""
    value = np.asarray(x, dtype=float)
    if np.any(value < 0.0):
        raise ValueError("the dimensionless nonmetricity invariant must be non-negative")
    return np.sqrt(value / (1.0 + value))


def weak_field_contractions(grad_psi, grad_phi) -> dict[str, np.ndarray]:
    """Quadratic nonmetricity contractions for a static scalar metric.

    The weak metric is

        ds^2 = -(1+2 Psi/c^2)c^2dt^2 + (1-2 Phi/c^2) dx^2.

    Gradients can have any final Cartesian dimension.  The returned quantities
    omit the common powers of c; they are the exact quadratic contractions in
    the coincident gauge at leading weak-field order.
    """
    psi = np.asarray(grad_psi, dtype=float)
    phi = np.asarray(grad_phi, dtype=float)
    if psi.shape != phi.shape or psi.ndim == 0:
        raise ValueError("grad_psi and grad_phi must have the same vector shape")
    psi2 = np.sum(np.square(psi), axis=-1)
    phi2 = np.sum(np.square(phi), axis=-1)
    cross = np.sum(psi * phi, axis=-1)
    spatial_dimension = psi.shape[-1]

    # Q_{alpha mu nu} Q^{alpha mu nu}
    q1 = 4.0 * psi2 + 4.0 * spatial_dimension * phi2
    # Q_{alpha mu nu} Q^{mu alpha nu}
    q2 = 4.0 * phi2
    # Q_alpha Q^alpha, with Q_i=2 Psi_i-2d Phi_i.
    q3 = 4.0 * psi2 + 4.0 * spatial_dimension**2 * phi2 - 8.0 * spatial_dimension * cross
    # tilde Q_alpha tilde Q^alpha, with tilde Q_i=-2 Phi_i.
    q4 = 4.0 * phi2
    # Q_alpha tilde Q^alpha.
    q5 = -4.0 * cross + 4.0 * spatial_dimension * phi2

    return {
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "q4": q4,
        "q5": q5,
        "psi2": psi2,
        "phi2": phi2,
        "cross": cross,
    }


def stegr_nonmetricity(grad_psi, grad_phi) -> np.ndarray:
    """STEGR nonmetricity scalar in the convention used by Sigma v1.

    Q = 1/4 Q1 - 1/2 Q2 - 1/4 Q3 + 1/2 Q5.  In three static
    spatial dimensions this reduces to 4 grad(Psi).grad(Phi)-2|grad(Phi)|^2.
    """
    contractions = weak_field_contractions(grad_psi, grad_phi)
    return (
        0.25 * contractions["q1"]
        - 0.5 * contractions["q2"]
        - 0.25 * contractions["q3"]
        + 0.5 * contractions["q5"]
    )


def slip_nonmetricity(grad_psi, grad_phi) -> np.ndarray:
    """Independent quadratic combination reducing to |grad(Psi-Phi)|^2."""
    contractions = weak_field_contractions(grad_psi, grad_phi)
    return (
        0.25 * contractions["q1"]
        - 2.0 * contractions["q2"]
        + 0.5 * contractions["q5"]
    )


def dimensionless_action_invariant(
    grad_psi, grad_phi, acceleration_scale: float
) -> np.ndarray:
    """Return X=Q/(2 a_sigma^2) for weak physical accelerations."""
    if acceleration_scale <= 0.0:
        raise ValueError("acceleration_scale must be positive")
    return stegr_nonmetricity(grad_psi, grad_phi) / (2.0 * acceleration_scale**2)


def standard_mu_spherical_acceleration(gbar, acceleration_scale: float) -> np.ndarray:
    """Positive spherical solution of gbar=mu(g/a_sigma) g."""
    baryonic = np.asarray(gbar, dtype=float)
    if np.any(baryonic <= 0.0) or acceleration_scale <= 0.0:
        raise ValueError("gbar and acceleration_scale must be positive")
    ratio = baryonic / acceleration_scale
    squared = 0.5 * (
        np.square(baryonic)
        + np.sqrt(np.power(baryonic, 4) + 4.0 * np.square(baryonic * acceleration_scale))
    )
    result = np.sqrt(squared)
    if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
        raise FloatingPointError("the spherical branch was not finite and positive")
    # This identity catches accidental changes to the analytic root.
    reconstructed = standard_mu(np.square(result / acceleration_scale)) * result
    if not np.allclose(reconstructed, baryonic, rtol=2e-12, atol=1e-30):
        raise RuntimeError(f"the spherical root failed its constitutive identity at ratio {ratio}")
    return result


def regular_isolated_branch_has_zero_slip(mu_minimum: float) -> bool:
    """Energy-integral result for div[mu grad(Psi-Phi)]=0.

    With isolated Dirichlet data and a strictly positive coefficient, multiplying
    the equation by Psi-Phi makes the volume integral of
    mu |grad(Psi-Phi)|^2 vanish.  The only regular branch is therefore zero slip.
    """
    return bool(np.isfinite(mu_minimum) and mu_minimum > 0.0)
