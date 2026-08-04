from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ProjectedVariationalSource:
    """Projected v3E fields and the unit-positive-coupling lensing correction."""

    local_tide: np.ndarray
    memory_tide: np.ndarray
    potential: np.ndarray
    tensor_gradient: np.ndarray
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


def shear_to_stf(shear_1: Any, shear_2: Any) -> np.ndarray:
    """Return the 2D trace-free lensing Hessian from Cartesian shear channels."""
    first = _scalar_field(shear_1, "shear_1")
    second = _scalar_field(shear_2, "shear_2")
    if first.shape != second.shape:
        raise ValueError("shear channels must have identical shapes")
    result = np.empty(first.shape + (2, 2), dtype=float)
    result[..., 0, 0] = first
    result[..., 0, 1] = second
    result[..., 1, 0] = second
    result[..., 1, 1] = -first
    return result


def stf_to_shear(tensor: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return Cartesian shear channels from a 2D STF tensor."""
    values = _stf_tensor(tensor, "tensor")
    return values[..., 0, 0], values[..., 0, 1]


def spectral_stf_hessian(potential: Any, *, spacing: float) -> np.ndarray:
    """Return D_ij potential on a periodic grid using spectral derivatives."""
    values = _scalar_field(potential, "potential")
    kx, ky = _wavenumber_grids(values.shape, spacing)
    transformed = np.fft.fft2(values, norm="ortho")
    first = np.fft.ifft2(0.5 * (np.square(ky) - np.square(kx)) * transformed, norm="ortho").real
    second = np.fft.ifft2(-kx * ky * transformed, norm="ortho").real
    return shear_to_stf(first, second)


def helmholtz_memory(tensor: Any, *, spacing: float, length: float) -> np.ndarray:
    """Apply (1-L^2 Laplacian)^-1 to every component of a 2D STF tensor."""
    values = _stf_tensor(tensor, "tensor")
    scale = _positive_finite(length, "length")
    kx, ky = _wavenumber_grids(values.shape[:2], spacing)
    transfer = 1.0 / (1.0 + scale**2 * (np.square(kx) + np.square(ky)))
    transformed = np.fft.fft2(values, axes=(0, 1), norm="ortho")
    memory = np.fft.ifft2(
        transformed * transfer[..., None, None], axes=(0, 1), norm="ortho"
    ).real
    return _stf_tensor(memory, "memory")


def misalignment_potential_and_gradients(
    local_tide: Any, memory_tide: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return bounded v3E potential and its two analytic STF gradients."""
    local = _stf_tensor(local_tide, "local_tide")
    memory = _stf_tensor(memory_tide, "memory_tide")
    if local.shape != memory.shape:
        raise ValueError("local and memory tides must have identical shapes")

    commutator = np.matmul(local, memory) - np.matmul(memory, local)
    numerator = np.sum(np.square(commutator), axis=(-2, -1))
    local_norm = np.sum(np.square(local), axis=(-2, -1))
    memory_norm = np.sum(np.square(memory), axis=(-2, -1))
    denominator = 2.0 * (1.0 + local_norm) * (1.0 + memory_norm)
    potential = numerator / denominator

    numerator_local = 2.0 * (
        np.matmul(commutator, memory) - np.matmul(memory, commutator)
    )
    numerator_memory = 2.0 * (
        np.matmul(local, commutator) - np.matmul(commutator, local)
    )
    denominator_local = 4.0 * (1.0 + memory_norm)[..., None, None] * local
    denominator_memory = 4.0 * (1.0 + local_norm)[..., None, None] * memory
    common = 1.0 / np.square(denominator)[..., None, None]
    gradient_local = common * (
        numerator_local * denominator[..., None, None]
        - numerator[..., None, None] * denominator_local
    )
    gradient_memory = common * (
        numerator_memory * denominator[..., None, None]
        - numerator[..., None, None] * denominator_memory
    )
    if np.any(potential < -1e-14) or np.any(potential > 1.0 + 1e-12):
        raise ValueError("misalignment potential violated its analytic bound")
    return (
        potential,
        _stf_tensor(gradient_local, "gradient_local"),
        _stf_tensor(gradient_memory, "gradient_memory"),
    )


def spectral_stf_double_divergence(tensor: Any, *, spacing: float) -> np.ndarray:
    """Return D_ij tensor^ij for a periodic symmetric trace-free tensor."""
    values = _stf_tensor(tensor, "tensor")
    kx, ky = _wavenumber_grids(values.shape[:2], spacing)
    transformed = np.fft.fft2(values, axes=(0, 1), norm="ortho")
    k_squared = np.square(kx) + np.square(ky)
    multipliers = np.empty(values.shape, dtype=float)
    multipliers[..., 0, 0] = -np.square(kx) + 0.5 * k_squared
    multipliers[..., 0, 1] = -kx * ky
    multipliers[..., 1, 0] = -kx * ky
    multipliers[..., 1, 1] = -np.square(ky) + 0.5 * k_squared
    source_transform = np.sum(multipliers * transformed, axis=(-2, -1))
    return np.fft.ifft2(source_transform, norm="ortho").real


def kappa_to_shear(kappa: Any, *, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the periodic E-mode shear associated with a convergence map."""
    values = _scalar_field(kappa, "kappa")
    kx, ky = _wavenumber_grids(values.shape, spacing)
    k_squared = np.square(kx) + np.square(ky)
    transformed = np.fft.fft2(values, norm="ortho")
    first_kernel = np.divide(
        np.square(kx) - np.square(ky),
        k_squared,
        out=np.zeros_like(k_squared),
        where=k_squared > 0.0,
    )
    second_kernel = np.divide(
        2.0 * kx * ky,
        k_squared,
        out=np.zeros_like(k_squared),
        where=k_squared > 0.0,
    )
    first = np.fft.ifft2(first_kernel * transformed, norm="ortho").real
    second = np.fft.ifft2(second_kernel * transformed, norm="ortho").real
    return first, second


def variational_source_from_tide(
    physical_tide: Any,
    *,
    spacing: float,
    memory_length: float,
    tensor_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized fields, potential, tensor gradient, and signed source."""
    tide = _stf_tensor(physical_tide, "physical_tide")
    scale = _positive_finite(tensor_scale, "tensor_scale")
    local = tide / scale
    memory = helmholtz_memory(local, spacing=spacing, length=memory_length)
    potential, gradient_local, gradient_memory = misalignment_potential_and_gradients(
        local, memory
    )
    pulled_memory_gradient = helmholtz_memory(
        gradient_memory, spacing=spacing, length=memory_length
    )
    tensor_gradient = _stf_tensor(
        (gradient_local + pulled_memory_gradient) / scale, "tensor_gradient"
    )
    source = spectral_stf_double_divergence(tensor_gradient, spacing=spacing)
    return local, memory, potential, tensor_gradient, source


def variational_source_from_potential(
    potential: Any,
    *,
    spacing: float,
    memory_length: float,
    tensor_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return interaction-density and Euler--Lagrange source from a scalar potential."""
    physical_tide = spectral_stf_hessian(potential, spacing=spacing)
    _, _, density, _, source = variational_source_from_tide(
        physical_tide,
        spacing=spacing,
        memory_length=memory_length,
        tensor_scale=tensor_scale,
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


def projected_variational_source(
    shear_1: Any,
    shear_2: Any,
    *,
    spacing: float,
    memory_length: float,
    tensor_scale: float,
    padding_factor: int = 2,
) -> ProjectedVariationalSource:
    """Evaluate the projected v4A source on padded shear maps and crop it back."""
    first = _scalar_field(shear_1, "shear_1")
    second = _scalar_field(shear_2, "shear_2")
    if first.shape != second.shape:
        raise ValueError("shear channels must have identical shapes")
    pad_y, pad_x = _padding(first.shape, padding_factor)
    padded_first = np.pad(first, (pad_y, pad_x), mode="constant")
    padded_second = np.pad(second, (pad_y, pad_x), mode="constant")
    physical_tide = shear_to_stf(padded_first, padded_second)
    local, memory, potential, tensor_gradient, full_source = variational_source_from_tide(
        physical_tide,
        spacing=spacing,
        memory_length=memory_length,
        tensor_scale=tensor_scale,
    )
    crop = (
        slice(pad_y[0], pad_y[0] + first.shape[0]),
        slice(pad_x[0], pad_x[0] + first.shape[1]),
    )
    source = full_source[crop]
    unit_eta_kappa = -0.5 * source
    unit_eta_shear_1, unit_eta_shear_2 = kappa_to_shear(
        -0.5 * full_source, spacing=spacing
    )
    return ProjectedVariationalSource(
        local_tide=local[crop],
        memory_tide=memory[crop],
        potential=potential[crop],
        tensor_gradient=tensor_gradient[crop],
        full_source=full_source,
        source=source,
        unit_eta_kappa=unit_eta_kappa,
        unit_eta_shear_1=unit_eta_shear_1[crop],
        unit_eta_shear_2=unit_eta_shear_2[crop],
        crop_slices=crop,
    )
