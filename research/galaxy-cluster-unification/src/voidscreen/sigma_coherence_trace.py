from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from voidscreen.sigma_variational_source import kappa_to_shear


@dataclass(frozen=True)
class ProjectedCoherenceTrace:
    """Baryon-seeded directional-disorder fields and unit trace correction."""

    vector_mean: np.ndarray
    vector_second_moment: np.ndarray
    raw_directional_variance: np.ndarray
    directional_disorder: np.ndarray
    high_field_activation: np.ndarray
    baryonic_seed: np.ndarray
    full_baryonic_seed: np.ndarray
    full_trace_state: np.ndarray
    trace_state: np.ndarray
    unit_eta_kappa: np.ndarray
    unit_eta_shear_1: np.ndarray
    unit_eta_shear_2: np.ndarray
    crop_slices: tuple[slice, slice]


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _scalar_field(values: Any, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or min(result.shape) < 4 or np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be a finite two-dimensional field")
    return result


def _vector_field(values: Any, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 3 or result.shape[-1] != 2 or np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape (ny, nx, 2)")
    if min(result.shape[:2]) < 4:
        raise ValueError(f"{name} must have at least four pixels per spatial dimension")
    return result


def _wavenumber_squared(shape: tuple[int, int], spacing: float) -> np.ndarray:
    pixel = _positive_finite(spacing, "spacing")
    if len(shape) != 2 or min(shape) < 4:
        raise ValueError("shape must contain two dimensions of at least four pixels")
    ky = 2.0 * np.pi * np.fft.fftfreq(shape[0], d=pixel)
    kx = 2.0 * np.pi * np.fft.fftfreq(shape[1], d=pixel)
    grid_x, grid_y = np.meshgrid(kx, ky)
    return np.square(grid_x) + np.square(grid_y)


def helmholtz_response(values: Any, *, spacing: float, length: float) -> np.ndarray:
    """Apply the periodic positive Helmholtz inverse to a scalar or vector map."""
    array = np.asarray(values, dtype=float)
    if array.ndim not in {2, 3} or np.any(~np.isfinite(array)):
        raise ValueError("values must be a finite scalar or channel-last vector map")
    if min(array.shape[:2]) < 4:
        raise ValueError("values must have at least four pixels per spatial dimension")
    memory_length = _positive_finite(length, "length")
    squared = _wavenumber_squared(array.shape[:2], spacing)
    denominator = 1.0 + memory_length**2 * squared
    if array.ndim == 3:
        denominator = denominator[..., None]
    transformed = np.fft.fft2(array, axes=(0, 1), norm="ortho")
    return np.fft.ifft2(
        transformed / denominator, axes=(0, 1), norm="ortho"
    ).real


def helmholtz_relative_residual(
    response: Any,
    source: Any,
    *,
    spacing: float,
    length: float,
) -> float:
    """Return ||(1-L^2 Laplacian) response-source||/||source|| spectrally."""
    result = np.asarray(response, dtype=float)
    forcing = np.asarray(source, dtype=float)
    if result.shape != forcing.shape or result.ndim not in {2, 3}:
        raise ValueError("response and source must have matching scalar/vector shapes")
    squared = _wavenumber_squared(result.shape[:2], spacing)
    multiplier = 1.0 + _positive_finite(length, "length") ** 2 * squared
    if result.ndim == 3:
        multiplier = multiplier[..., None]
    transformed = np.fft.fft2(result, axes=(0, 1), norm="ortho")
    recovered = np.fft.ifft2(
        multiplier * transformed, axes=(0, 1), norm="ortho"
    ).real
    numerator = float(np.sqrt(np.mean(np.square(recovered - forcing))))
    denominator = float(np.sqrt(np.mean(np.square(forcing))))
    return numerator / max(denominator, np.finfo(float).tiny)


def directional_disorder(
    physical_vector: Any,
    *,
    spacing: float,
    memory_length: float,
    vector_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return first/second moments, variance, bounded disorder, and activation."""
    vector = _vector_field(physical_vector, "physical_vector")
    scale = _positive_finite(vector_scale, "vector_scale")
    mean = helmholtz_response(vector, spacing=spacing, length=memory_length)
    second = helmholtz_response(
        np.sum(np.square(vector), axis=-1), spacing=spacing, length=memory_length
    )
    raw_variance = second - np.sum(np.square(mean), axis=-1)
    variance = np.maximum(raw_variance, 0.0)
    numerical_floor = 64.0 * np.finfo(float).eps * max(float(np.max(second)), 1.0)
    disorder = np.divide(
        variance,
        second,
        out=np.zeros_like(second),
        where=second > numerical_floor,
    )
    disorder = np.clip(disorder, 0.0, 1.0)
    activation = scale**2 / (scale**2 + np.maximum(second, 0.0))
    return mean, second, raw_variance, disorder, activation


def coherence_trace_state(
    physical_vector: Any,
    baryonic_convergence: Any,
    *,
    spacing: float,
    memory_length: float,
    vector_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the v4C moments, nonnegative baryonic seed, and trace state."""
    vector = _vector_field(physical_vector, "physical_vector")
    baryons = _scalar_field(baryonic_convergence, "baryonic_convergence")
    if vector.shape[:2] != baryons.shape:
        raise ValueError("vector and baryonic convergence grids must match")
    mean, second, raw_variance, disorder, activation = directional_disorder(
        vector,
        spacing=spacing,
        memory_length=memory_length,
        vector_scale=vector_scale,
    )
    seed = np.maximum(baryons, 0.0) * activation * disorder
    trace = helmholtz_response(seed, spacing=spacing, length=memory_length)
    return mean, second, raw_variance, disorder, activation, seed, trace


def _padding(shape: tuple[int, int], factor: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if isinstance(factor, bool) or int(factor) != factor or factor < 1:
        raise ValueError("padding_factor must be a positive integer")
    target = (int(factor) * shape[0], int(factor) * shape[1])
    result = []
    for original, padded in zip(shape, target, strict=True):
        total = padded - original
        result.append((total // 2, total - total // 2))
    return result[0], result[1]


def projected_coherence_trace(
    vector_east: Any,
    vector_north: Any,
    baryonic_convergence: Any,
    *,
    spacing: float,
    memory_length: float,
    vector_scale: float,
    padding_factor: int = 2,
) -> ProjectedCoherenceTrace:
    """Evaluate v4C on zero-padded maps and crop its E-mode correction."""
    east = _scalar_field(vector_east, "vector_east")
    north = _scalar_field(vector_north, "vector_north")
    baryons = _scalar_field(baryonic_convergence, "baryonic_convergence")
    if east.shape != north.shape or east.shape != baryons.shape:
        raise ValueError("all input maps must have identical shapes")
    pad_y, pad_x = _padding(east.shape, padding_factor)
    pad_spec = (pad_y, pad_x)
    padded_vector = np.stack(
        [
            np.pad(east, pad_spec, mode="constant"),
            np.pad(north, pad_spec, mode="constant"),
        ],
        axis=-1,
    )
    padded_baryons = np.pad(baryons, pad_spec, mode="constant")
    mean, second, raw_variance, disorder, activation, seed, full_trace = (
        coherence_trace_state(
            padded_vector,
            padded_baryons,
            spacing=spacing,
            memory_length=memory_length,
            vector_scale=vector_scale,
        )
    )
    shear_1, shear_2 = kappa_to_shear(full_trace, spacing=spacing)
    crop = (
        slice(pad_y[0], pad_y[0] + east.shape[0]),
        slice(pad_x[0], pad_x[0] + east.shape[1]),
    )
    return ProjectedCoherenceTrace(
        vector_mean=mean[crop],
        vector_second_moment=second[crop],
        raw_directional_variance=raw_variance[crop],
        directional_disorder=disorder[crop],
        high_field_activation=activation[crop],
        baryonic_seed=seed[crop],
        full_baryonic_seed=seed,
        full_trace_state=full_trace,
        trace_state=full_trace[crop],
        unit_eta_kappa=full_trace[crop],
        unit_eta_shear_1=shear_1[crop],
        unit_eta_shear_2=shear_2[crop],
        crop_slices=crop,
    )
