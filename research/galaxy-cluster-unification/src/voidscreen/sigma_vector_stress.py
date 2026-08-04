from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from voidscreen.sigma_variational_source import (
    helmholtz_memory,
    kappa_to_shear,
    misalignment_potential_and_gradients,
)


@dataclass(frozen=True)
class ProjectedVectorStressSource:
    """Vector-stress memory fields and the unit-positive-coupling correction."""

    normalized_vector: np.ndarray
    local_stress: np.ndarray
    memory_stress: np.ndarray
    potential: np.ndarray
    stress_gradient: np.ndarray
    vector_gradient: np.ndarray
    full_source: np.ndarray
    source: np.ndarray
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


def _stf_tensor(values: Any, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 4 or result.shape[-2:] != (2, 2) or np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape (ny, nx, 2, 2)")
    symmetric = 0.5 * (result + np.swapaxes(result, -1, -2))
    trace = np.trace(symmetric, axis1=-2, axis2=-1)
    return symmetric - 0.5 * trace[..., None, None] * np.eye(2)


def _wavenumber_grids(shape: tuple[int, int], spacing: float) -> tuple[np.ndarray, np.ndarray]:
    pixel = _positive_finite(spacing, "spacing")
    if len(shape) != 2 or min(shape) < 4:
        raise ValueError("shape must contain two dimensions of at least four pixels")
    ky = 2.0 * np.pi * np.fft.fftfreq(shape[0], d=pixel)
    kx = 2.0 * np.pi * np.fft.fftfreq(shape[1], d=pixel)
    return np.meshgrid(kx, ky)


def spectral_gradient(potential: Any, *, spacing: float) -> np.ndarray:
    """Return (east, north) derivatives of a periodic scalar field."""
    values = _scalar_field(potential, "potential")
    kx, ky = _wavenumber_grids(values.shape, spacing)
    transformed = np.fft.fft2(values, norm="ortho")
    result = np.empty(values.shape + (2,), dtype=float)
    result[..., 0] = np.fft.ifft2(1j * kx * transformed, norm="ortho").real
    result[..., 1] = np.fft.ifft2(1j * ky * transformed, norm="ortho").real
    return result


def spectral_divergence(vector: Any, *, spacing: float) -> np.ndarray:
    """Return partial_east v_east + partial_north v_north periodically."""
    values = _vector_field(vector, "vector")
    kx, ky = _wavenumber_grids(values.shape[:2], spacing)
    transformed = np.fft.fft2(values, axes=(0, 1), norm="ortho")
    divergence_transform = 1j * (
        kx * transformed[..., 0] + ky * transformed[..., 1]
    )
    return np.fft.ifft2(divergence_transform, norm="ortho").real


def vector_stress(physical_vector: Any, *, vector_scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Return u=a/ell and S_ij=u_i u_j-delta_ij u^2/2."""
    vector = _vector_field(physical_vector, "physical_vector")
    scale = _positive_finite(vector_scale, "vector_scale")
    normalized = vector / scale
    squared = np.sum(np.square(normalized), axis=-1)
    stress = np.einsum("...i,...j->...ij", normalized, normalized)
    stress -= 0.5 * squared[..., None, None] * np.eye(2)
    return normalized, _stf_tensor(stress, "stress")


def vector_chain_gradient(
    normalized_vector: Any,
    stress_gradient: Any,
    *,
    vector_scale: float,
) -> np.ndarray:
    """Pull an STF stress gradient back to the physical field vector."""
    vector = _vector_field(normalized_vector, "normalized_vector")
    gradient = _stf_tensor(stress_gradient, "stress_gradient")
    if vector.shape[:2] != gradient.shape[:2]:
        raise ValueError("normalized vector and stress gradient grids must match")
    scale = _positive_finite(vector_scale, "vector_scale")
    return 2.0 * np.einsum("...ij,...j->...i", gradient, vector) / scale


def variational_source_from_vector(
    physical_vector: Any,
    *,
    spacing: float,
    memory_length: float,
    vector_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return v4B fields and R=-div(dV/d(grad psi))."""
    normalized, local = vector_stress(physical_vector, vector_scale=vector_scale)
    memory = helmholtz_memory(local, spacing=spacing, length=memory_length)
    potential, gradient_local, gradient_memory = misalignment_potential_and_gradients(
        local, memory
    )
    pulled_memory_gradient = helmholtz_memory(
        gradient_memory, spacing=spacing, length=memory_length
    )
    stress_gradient = _stf_tensor(
        gradient_local + pulled_memory_gradient, "stress_gradient"
    )
    field_gradient = vector_chain_gradient(
        normalized, stress_gradient, vector_scale=vector_scale
    )
    source = -spectral_divergence(field_gradient, spacing=spacing)
    return normalized, local, memory, potential, stress_gradient, source


def variational_source_from_potential(
    potential: Any,
    *,
    spacing: float,
    memory_length: float,
    vector_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return interaction density and Euler--Lagrange source from a scalar potential."""
    vector = spectral_gradient(potential, spacing=spacing)
    _, _, _, density, _, source = variational_source_from_vector(
        vector,
        spacing=spacing,
        memory_length=memory_length,
        vector_scale=vector_scale,
    )
    return density, source


def _padding(shape: tuple[int, int], factor: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if isinstance(factor, bool) or int(factor) != factor or factor < 1:
        raise ValueError("padding_factor must be a positive integer")
    target = (int(factor) * shape[0], int(factor) * shape[1])
    result = []
    for original, padded in zip(shape, target, strict=True):
        total = padded - original
        result.append((total // 2, total - total // 2))
    return result[0], result[1]


def projected_vector_stress_source(
    vector_east: Any,
    vector_north: Any,
    *,
    spacing: float,
    memory_length: float,
    vector_scale: float,
    padding_factor: int = 2,
) -> ProjectedVectorStressSource:
    """Evaluate v4B on padded vector maps and crop the correction to the data field."""
    east = _scalar_field(vector_east, "vector_east")
    north = _scalar_field(vector_north, "vector_north")
    if east.shape != north.shape:
        raise ValueError("vector channels must have identical shapes")
    pad_y, pad_x = _padding(east.shape, padding_factor)
    padded = np.empty(
        (east.shape[0] + sum(pad_y), east.shape[1] + sum(pad_x), 2), dtype=float
    )
    padded[..., 0] = np.pad(east, (pad_y, pad_x), mode="constant")
    padded[..., 1] = np.pad(north, (pad_y, pad_x), mode="constant")
    normalized, local, memory, potential, stress_gradient, full_source = (
        variational_source_from_vector(
            padded,
            spacing=spacing,
            memory_length=memory_length,
            vector_scale=vector_scale,
        )
    )
    field_gradient = vector_chain_gradient(
        normalized, stress_gradient, vector_scale=vector_scale
    )
    crop = (
        slice(pad_y[0], pad_y[0] + east.shape[0]),
        slice(pad_x[0], pad_x[0] + east.shape[1]),
    )
    source = full_source[crop]
    unit_eta_kappa_full = -0.5 * full_source
    shear_1_full, shear_2_full = kappa_to_shear(unit_eta_kappa_full, spacing=spacing)
    return ProjectedVectorStressSource(
        normalized_vector=normalized[crop],
        local_stress=local[crop],
        memory_stress=memory[crop],
        potential=potential[crop],
        stress_gradient=stress_gradient[crop],
        vector_gradient=field_gradient[crop],
        full_source=full_source,
        source=source,
        unit_eta_kappa=unit_eta_kappa_full[crop],
        unit_eta_shear_1=shear_1_full[crop],
        unit_eta_shear_2=shear_2_full[crop],
        crop_slices=crop,
    )
