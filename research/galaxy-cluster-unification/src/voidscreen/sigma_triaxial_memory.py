from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TidalMemoryField:
    """Static spectral realization of the screened trace-free memory field."""

    density: np.ndarray
    acceleration: np.ndarray
    tidal: np.ndarray
    screen: np.ndarray
    source: np.ndarray
    propagated_memory: np.ndarray
    memory: np.ndarray
    invariant_i2: np.ndarray
    invariant_i3: np.ndarray
    discriminant: np.ndarray
    bounded_potential: np.ndarray


def symmetric_trace_free(matrix: Any) -> np.ndarray:
    """Return the symmetric trace-free projection on the final two axes."""
    values = np.asarray(matrix, dtype=float)
    if values.shape[-2:] != (3, 3) or np.any(~np.isfinite(values)):
        raise ValueError("matrix must be finite with final shape (3, 3)")
    symmetric = 0.5 * (values + np.swapaxes(values, -1, -2))
    trace = np.trace(symmetric, axis1=-2, axis2=-1)
    return symmetric - trace[..., None, None] * np.eye(3) / 3.0


def triaxial_invariants(matrix: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return I2, I3, and the trace-free eigenvalue discriminant."""
    values = symmetric_trace_free(matrix)
    squared = np.matmul(values, values)
    i2 = np.trace(squared, axis1=-2, axis2=-1)
    i3 = np.trace(np.matmul(squared, values), axis1=-2, axis2=-1)
    discriminant = np.power(i2, 3) - 6.0 * np.square(i3)
    return i2, i3, discriminant


def bounded_triaxial_potential(matrix: Any) -> np.ndarray:
    """Evaluate the bounded discriminant potential, clipping roundoff below zero."""
    i2, _, discriminant = triaxial_invariants(matrix)
    scale = np.maximum(np.power(i2, 3), 1.0)
    if np.any(discriminant < -2.0e-12 * scale):
        raise ValueError("symmetric trace-free discriminant is unexpectedly negative")
    nonnegative = np.maximum(discriminant, 0.0)
    return nonnegative / np.power(1.0 + i2, 3)


def bounded_triaxial_gradient(matrix: Any) -> np.ndarray:
    """Analytic gradient of the bounded potential on the STF subspace."""
    values = symmetric_trace_free(matrix)
    squared = np.matmul(values, values)
    i2 = np.trace(squared, axis1=-2, axis2=-1)
    i3 = np.trace(np.matmul(squared, values), axis1=-2, axis2=-1)
    numerator = np.power(i2, 3) - 6.0 * np.square(i3)
    denominator = np.power(1.0 + i2, 3)
    squared_stf = squared - i2[..., None, None] * np.eye(3) / 3.0
    numerator_gradient = (
        6.0 * np.square(i2)[..., None, None] * values - 36.0 * i3[..., None, None] * squared_stf
    )
    denominator_gradient = 6.0 * np.square(1.0 + i2)[..., None, None] * values
    gradient = (
        numerator_gradient * denominator[..., None, None]
        - numerator[..., None, None] * denominator_gradient
    ) / np.square(denominator)[..., None, None]
    return symmetric_trace_free(gradient)


def acceleration_power_screen(g_over_a_sigma: Any, power: float) -> np.ndarray:
    """Return S=1/(1+(g/a_sigma)^power) with stable large-input handling."""
    ratio = np.asarray(g_over_a_sigma, dtype=float)
    if (
        np.any(~np.isfinite(ratio))
        or np.any(ratio < 0.0)
        or not math.isfinite(power)
        or power <= 0.0
    ):
        raise ValueError("g/a_sigma and screen power are invalid")
    result = np.zeros_like(ratio)
    logarithm = np.full_like(ratio, -np.inf)
    positive = ratio > 0.0
    logarithm[positive] = power * np.log(ratio[positive])
    moderate = logarithm <= math.log(np.finfo(float).max)
    result[moderate] = 1.0 / (1.0 + np.exp(logarithm[moderate]))
    return result


def high_acceleration_screen(g_over_a_sigma: Any) -> np.ndarray:
    """Return the frozen Sigma v3D screen S=1/(1+(g/a_sigma)^4)."""
    return acceleration_power_screen(g_over_a_sigma, 4.0)


def axisymmetric_tidal_tensor(direction: Any, amplitude: float = 1.0) -> np.ndarray:
    """Return an isolated-source STF tide with two degenerate eigenvalues."""
    unit = np.asarray(direction, dtype=float)
    if unit.shape != (3,) or np.any(~np.isfinite(unit)) or not math.isfinite(amplitude):
        raise ValueError("direction and amplitude are invalid")
    norm = float(np.linalg.norm(unit))
    if norm <= 0.0:
        raise ValueError("direction must be nonzero")
    unit = unit / norm
    return float(amplitude) * (np.eye(3) - 3.0 * np.outer(unit, unit))


def centered_axis(points: int, half_width: float) -> np.ndarray:
    """Return an odd, cell-centered coordinate axis spanning the requested box."""
    if points < 9 or points % 2 != 1 or not math.isfinite(half_width) or half_width <= 0.0:
        raise ValueError("points must be an odd integer >=9 and half_width must be positive")
    return np.linspace(-half_width, half_width, points, dtype=float)


def gaussian_mixture_density(
    axis: Any,
    components: list[dict[str, Any]],
    *,
    total_mass: float,
) -> np.ndarray:
    """Build a finite-box Gaussian mixture with exact discrete component masses."""
    coordinates = np.asarray(axis, dtype=float)
    if (
        coordinates.ndim != 1
        or coordinates.size < 9
        or np.any(~np.isfinite(coordinates))
        or np.any(np.diff(coordinates) <= 0.0)
        or not math.isfinite(total_mass)
        or total_mass <= 0.0
        or not components
    ):
        raise ValueError("axis, components, or total mass are invalid")
    spacing = float(np.mean(np.diff(coordinates)))
    if not np.allclose(np.diff(coordinates), spacing, rtol=1e-10, atol=1e-12):
        raise ValueError("axis must be uniformly spaced")
    x, y, z = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    density = np.zeros((coordinates.size,) * 3, dtype=float)
    fraction_sum = sum(float(component["mass_fraction"]) for component in components)
    if not math.isclose(fraction_sum, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("component mass fractions must sum to one")
    for component in components:
        fraction = float(component["mass_fraction"])
        center = np.asarray(component["center_L_sigma"], dtype=float)
        sigma = np.asarray(component["sigma_L_sigma"], dtype=float)
        if (
            fraction <= 0.0
            or center.shape != (3,)
            or sigma.shape != (3,)
            or np.any(~np.isfinite(center))
            or np.any(~np.isfinite(sigma))
            or np.any(sigma <= 0.0)
        ):
            raise ValueError("Gaussian component is invalid")
        exponent = -0.5 * (
            np.square((x - center[0]) / sigma[0])
            + np.square((y - center[1]) / sigma[1])
            + np.square((z - center[2]) / sigma[2])
        )
        profile = np.exp(exponent)
        normalization = float(np.sum(profile) * spacing**3)
        density += total_mass * fraction * profile / normalization
    return density


def spectral_tidal_memory(
    density: Any,
    *,
    spacing: float,
    gravitational_constant: float,
    a_sigma: float,
    memory_length: float,
    screen_power: float = 4.0,
    screen_order: str = "before_memory",
) -> TidalMemoryField:
    """Solve the static periodic Poisson, screened-source, and Helmholtz system."""
    rho = np.asarray(density, dtype=float)
    if (
        rho.ndim != 3
        or len(set(rho.shape)) != 1
        or min(rho.shape) < 9
        or np.any(~np.isfinite(rho))
        or np.any(rho < 0.0)
        or screen_order not in {"before_memory", "after_memory"}
        or not math.isfinite(screen_power)
        or screen_power <= 0.0
        or not all(
            math.isfinite(value) and value > 0.0
            for value in (spacing, gravitational_constant, a_sigma, memory_length)
        )
    ):
        raise ValueError("density or physical scales are invalid")

    size = rho.shape[0]
    wave = 2.0 * np.pi * np.fft.fftfreq(size, d=spacing)
    kx, ky, kz = np.meshgrid(wave, wave, wave, indexing="ij", sparse=True)
    wavevectors = (kx, ky, kz)
    k_squared = np.square(kx) + np.square(ky) + np.square(kz)
    nonzero = k_squared > 0.0
    rho_fourier = np.fft.fftn(rho)
    potential_fourier = np.zeros_like(rho_fourier, dtype=complex)
    potential_fourier[nonzero] = (
        -4.0 * np.pi * gravitational_constant * rho_fourier[nonzero] / k_squared[nonzero]
    )

    acceleration = np.empty(rho.shape + (3,), dtype=float)
    for index, component in enumerate(wavevectors):
        acceleration[..., index] = np.fft.ifftn(-1.0j * component * potential_fourier).real
    magnitude = np.linalg.norm(acceleration, axis=-1)

    laplacian_fourier = -k_squared * potential_fourier
    tidal = np.empty(rho.shape + (3, 3), dtype=float)
    for row, row_wave in enumerate(wavevectors):
        for column, column_wave in enumerate(wavevectors):
            hessian_fourier = -row_wave * column_wave * potential_fourier
            if row == column:
                hessian_fourier = hessian_fourier - laplacian_fourier / 3.0
            tidal[..., row, column] = np.fft.ifftn(hessian_fourier).real
    tidal = symmetric_trace_free(tidal)

    screen = acceleration_power_screen(magnitude / a_sigma, screen_power)
    curvature_scale = a_sigma / memory_length
    helmholtz = 1.0 + memory_length**2 * k_squared
    if screen_order == "before_memory":
        source = screen[..., None, None] * tidal / curvature_scale
    else:
        source = tidal / curvature_scale
    propagated_memory = np.empty_like(source)
    for row in range(3):
        for column in range(3):
            propagated_memory[..., row, column] = np.fft.ifftn(
                np.fft.fftn(source[..., row, column]) / helmholtz
            ).real
    propagated_memory = symmetric_trace_free(propagated_memory)
    memory = propagated_memory.copy()
    if screen_order == "after_memory":
        memory *= screen[..., None, None]
    memory = symmetric_trace_free(memory)
    i2, i3, discriminant = triaxial_invariants(memory)
    potential = bounded_triaxial_potential(memory)
    return TidalMemoryField(
        density=rho,
        acceleration=acceleration,
        tidal=tidal,
        screen=screen,
        source=source,
        propagated_memory=propagated_memory,
        memory=memory,
        invariant_i2=i2,
        invariant_i3=i3,
        discriminant=discriminant,
        bounded_potential=potential,
    )


def integrated_response(
    potential: Any,
    axis: Any,
    *,
    analysis_half_width: float,
) -> float:
    """Integrate the bounded potential over a central cubic analysis volume."""
    values = np.asarray(potential, dtype=float)
    coordinates = np.asarray(axis, dtype=float)
    if (
        values.shape != (coordinates.size,) * 3
        or np.any(~np.isfinite(values))
        or np.any(values < 0.0)
        or analysis_half_width <= 0.0
    ):
        raise ValueError("potential, axis, or analysis volume is invalid")
    spacing = float(np.mean(np.diff(coordinates)))
    selected = np.abs(coordinates) <= analysis_half_width
    if not np.any(selected):
        raise ValueError("analysis volume contains no cells")
    return float(np.sum(values[np.ix_(selected, selected, selected)]) * spacing**3)
