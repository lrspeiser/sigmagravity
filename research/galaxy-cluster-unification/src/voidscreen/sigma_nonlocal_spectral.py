from __future__ import annotations

import math

import numpy as np
from scipy.special import erf


def positive_spectral_transfer(
    momentum_squared,
    mass_squared,
    residues,
    *,
    massless_residue: float = 1.0,
) -> np.ndarray:
    """Return a UV-normalized Kallen--Lehmann static transfer.

    For Euclidean ``s=k^2`` a standard positive spectral propagator is

    ``D(s)=Z0/s + sum_i rho_i/(s+m_i^2)``.

    The returned quantity is ``s D(s)/(Z0+sum rho_i)``, normalized to one
    at high momentum.  Nonnegative residues make it monotone increasing in
    ``s`` and therefore no larger in the infrared than in the ultraviolet.
    """
    s = np.asarray(momentum_squared, dtype=float)
    masses = np.asarray(mass_squared, dtype=float)
    weights = np.asarray(residues, dtype=float)
    if np.any(~np.isfinite(s)) or np.any(s < 0.0):
        raise ValueError("momentum_squared must be finite and nonnegative")
    if masses.ndim != 1 or weights.ndim != 1 or masses.shape != weights.shape:
        raise ValueError("mass_squared and residues must be matching vectors")
    if np.any(~np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("mass_squared must be finite and positive")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("residues must be finite and nonnegative")
    if not math.isfinite(massless_residue) or massless_residue <= 0.0:
        raise ValueError("massless_residue must be finite and positive")
    normalization = massless_residue + float(np.sum(weights))
    terms = np.expand_dims(s, -1) / (np.expand_dims(s, -1) + masses)
    return (massless_residue + np.sum(weights * terms, axis=-1)) / normalization


def positive_spectral_transfer_derivative(
    momentum_squared,
    mass_squared,
    residues,
    *,
    massless_residue: float = 1.0,
) -> np.ndarray:
    """Derivative of :func:`positive_spectral_transfer` with respect to ``s``."""
    s = np.asarray(momentum_squared, dtype=float)
    masses = np.asarray(mass_squared, dtype=float)
    weights = np.asarray(residues, dtype=float)
    # Reuse validation and shape checks.
    positive_spectral_transfer(
        s, masses, weights, massless_residue=massless_residue
    )
    normalization = massless_residue + float(np.sum(weights))
    derivative = weights * masses / np.square(np.expand_dims(s, -1) + masses)
    return np.sum(derivative, axis=-1) / normalization


def rational_far_enhancing_transfer(momentum_squared_times_length_squared, amplitude: float):
    """Return ``T(s)=1+A/(1+s)`` for the simplest IR-enhancing filter."""
    s = np.asarray(momentum_squared_times_length_squared, dtype=float)
    if np.any(~np.isfinite(s)) or np.any(s < 0.0):
        raise ValueError("dimensionless momentum squared must be finite and nonnegative")
    if not math.isfinite(amplitude) or amplitude < 0.0:
        raise ValueError("amplitude must be finite and nonnegative")
    return 1.0 + amplitude / (1.0 + s)


def rational_propagator_residues(amplitude: float) -> dict[str, float]:
    """Residues in ``T(k^2)/k^2=(1+A)/k^2-A/(k^2+L^-2)``."""
    if not math.isfinite(amplitude) or amplitude < 0.0:
        raise ValueError("amplitude must be finite and nonnegative")
    return {
        "massless_residue": 1.0 + amplitude,
        "massive_residue": -amplitude,
    }


def rational_point_force_ratio(radius_over_length, amplitude: float) -> np.ndarray:
    """Point-force ratio for the rational far-enhancing transfer."""
    x = np.asarray(radius_over_length, dtype=float)
    if np.any(~np.isfinite(x)) or np.any(x < 0.0):
        raise ValueError("radius_over_length must be finite and nonnegative")
    if not math.isfinite(amplitude) or amplitude < 0.0:
        raise ValueError("amplitude must be finite and nonnegative")
    shape = -np.expm1(-x) - x * np.exp(-x)
    return 1.0 + amplitude * shape


def entire_ir_transfer(momentum_squared_times_length_squared, log_ir_boost: float):
    """No-zero nested-exponential transfer ``exp[A exp(-k^2 L^2)]``."""
    s = np.asarray(momentum_squared_times_length_squared, dtype=float)
    if np.any(~np.isfinite(s)) or np.any(s < 0.0):
        raise ValueError("dimensionless momentum squared must be finite and nonnegative")
    if not math.isfinite(log_ir_boost) or log_ir_boost < 0.0:
        raise ValueError("log_ir_boost must be finite and nonnegative")
    return np.exp(log_ir_boost * np.exp(-s))


def entire_ir_transfer_derivative(
    momentum_squared_times_length_squared, log_ir_boost: float
) -> np.ndarray:
    """Derivative of the entire transfer with respect to ``k^2 L^2``."""
    s = np.asarray(momentum_squared_times_length_squared, dtype=float)
    transfer = entire_ir_transfer(s, log_ir_boost)
    return -log_ir_boost * np.exp(-s) * transfer


def entire_point_force_correction(
    radius_over_length,
    log_ir_boost: float,
    *,
    maximum_terms: int = 256,
    weight_tolerance: float = 1e-15,
) -> np.ndarray:
    """Extra point-force fraction for ``exp[A exp(-k^2 L^2)]``.

    Expanding the transfer gives positive Gaussian source images.  The ``n``th
    image has weight ``A^n/n!`` and width ``sqrt(n) L``.  This evaluates the
    exact convergent force series to the requested weight tolerance.
    """
    x = np.asarray(radius_over_length, dtype=float)
    if np.any(~np.isfinite(x)) or np.any(x < 0.0):
        raise ValueError("radius_over_length must be finite and nonnegative")
    if not math.isfinite(log_ir_boost) or log_ir_boost < 0.0:
        raise ValueError("log_ir_boost must be finite and nonnegative")
    if maximum_terms < 1 or weight_tolerance <= 0.0:
        raise ValueError("maximum_terms and weight_tolerance must be positive")
    result = np.zeros_like(x)
    weight = 1.0
    for index in range(1, maximum_terms + 1):
        weight *= log_ir_boost / index
        y = x / (2.0 * math.sqrt(index))
        shape = erf(y) - 2.0 * y * np.exp(-np.square(y)) / math.sqrt(math.pi)
        small_shape = (
            4.0 * np.power(y, 3) / (3.0 * math.sqrt(math.pi))
            - 4.0 * np.power(y, 5) / (5.0 * math.sqrt(math.pi))
            + 2.0 * np.power(y, 7) / (7.0 * math.sqrt(math.pi))
            - 2.0 * np.power(y, 9) / (27.0 * math.sqrt(math.pi))
        )
        shape = np.where(y < 1e-3, small_shape, shape)
        result += weight * shape
        if index > log_ir_boost and weight < weight_tolerance:
            break
    else:
        raise RuntimeError("entire point-force series did not converge")
    return result


def entire_point_force_ratio(
    radius_over_length,
    log_ir_boost: float,
    *,
    maximum_terms: int = 256,
    weight_tolerance: float = 1e-15,
) -> np.ndarray:
    """Point-force ratio, including the local Newtonian unit contribution."""
    return 1.0 + entire_point_force_correction(
        radius_over_length,
        log_ir_boost,
        maximum_terms=maximum_terms,
        weight_tolerance=weight_tolerance,
    )


def periodic_lensing_hessian(
    surface_density,
    pixel_scale: float,
    *,
    log_ir_boost: float = 0.0,
    response_length: float = 1.0,
) -> dict[str, np.ndarray]:
    """Return convergence and shear for a periodic thin-lens manufactured map.

    The zero mode is removed.  The nested-exponential transfer multiplies the
    same lensing potential before all Hessian components are calculated, so no
    lensing-only shear or orientation is inserted.
    """
    density = np.asarray(surface_density, dtype=float)
    if density.ndim != 2 or min(density.shape) < 4 or np.any(~np.isfinite(density)):
        raise ValueError("surface_density must be a finite two-dimensional map")
    if pixel_scale <= 0.0 or response_length <= 0.0:
        raise ValueError("pixel_scale and response_length must be positive")
    ky = 2.0 * np.pi * np.fft.fftfreq(density.shape[0], d=pixel_scale)
    kx = 2.0 * np.pi * np.fft.fftfreq(density.shape[1], d=pixel_scale)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k_squared = np.square(kx_grid) + np.square(ky_grid)
    source = np.fft.fft2(density - np.mean(density))
    transfer = entire_ir_transfer(
        k_squared * response_length**2, log_ir_boost
    )
    potential = np.zeros_like(source, dtype=complex)
    nonzero = k_squared > 0.0
    potential[nonzero] = -2.0 * transfer[nonzero] * source[nonzero] / k_squared[nonzero]
    h_xx = np.fft.ifft2(-np.square(kx_grid) * potential).real
    h_yy = np.fft.ifft2(-np.square(ky_grid) * potential).real
    h_xy = np.fft.ifft2(-kx_grid * ky_grid * potential).real
    return {
        "convergence": 0.5 * (h_xx + h_yy),
        "shear_1": 0.5 * (h_xx - h_yy),
        "shear_2": h_xy,
    }
