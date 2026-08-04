"""Finite-window internal/boundary decomposition for spent cluster lens maps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voidscreen.sigma_covariant_feature_inference import convergence_to_shear


@dataclass(frozen=True)
class HarmonicFit:
    """Least-squares harmonic-potential description of a shear field."""

    coefficients: dict[str, float]
    predicted_shear_1: np.ndarray
    predicted_shear_2: np.ndarray
    normalized_RMSE: float
    power_closed: float


@dataclass(frozen=True)
class BoundaryDecomposition:
    """Internal E-mode shear plus the remaining boundary candidate."""

    tapered_internal_convergence: np.ndarray
    internal_shear_1: np.ndarray
    internal_shear_2: np.ndarray
    boundary_shear_1: np.ndarray
    boundary_shear_2: np.ndarray
    boundary_to_total_shear_power_ratio: float
    harmonic_fit: HarmonicFit


def radial_taper(
    radius: np.ndarray,
    *,
    start: float,
    end: float,
) -> np.ndarray:
    """Raised-cosine disk taper equal to one inside start and zero outside end."""
    values = np.asarray(radius, dtype=float)
    if values.ndim != 2 or np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("radius must be a finite nonnegative two-dimensional map")
    if not 0.0 <= start < end:
        raise ValueError("taper radii must satisfy zero <= start < end")
    result = np.ones_like(values)
    transition = (values > start) & (values < end)
    result[values >= end] = 0.0
    phase = (values[transition] - start) / (end - start)
    result[transition] = 0.5 * (1.0 + np.cos(np.pi * phase))
    return result


def harmonic_shear_basis(
    east_kpc: np.ndarray,
    north_kpc: np.ndarray,
    *,
    minimum_order: int,
    maximum_order: int,
    reference_radius_kpc: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return exact zero-convergence shear bases from Re/Im[(E+iN)^m]."""
    east = np.asarray(east_kpc, dtype=float)
    north = np.asarray(north_kpc, dtype=float)
    if east.shape != north.shape or east.ndim != 2:
        raise ValueError("east and north coordinates must be matching grids")
    if any(np.any(~np.isfinite(values)) for values in (east, north)):
        raise ValueError("coordinates must be finite")
    if minimum_order < 2 or maximum_order < minimum_order:
        raise ValueError("harmonic orders must satisfy 2 <= minimum <= maximum")
    if reference_radius_kpc <= 0.0:
        raise ValueError("reference radius must be positive")
    conjugate_coordinate = (east - 1j * north) / reference_radius_kpc
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for order in range(minimum_order, maximum_order + 1):
        complex_shear = np.power(conjugate_coordinate, order - 2)
        result[f"harmonic_m{order}_cos"] = (
            complex_shear.real,
            complex_shear.imag,
        )
        sine_shear = 1j * complex_shear
        result[f"harmonic_m{order}_sin"] = (
            sine_shear.real,
            sine_shear.imag,
        )
    return result


def fit_harmonic_shear(
    shear_1: np.ndarray,
    shear_2: np.ndarray,
    mask: np.ndarray,
    basis: dict[str, tuple[np.ndarray, np.ndarray]],
) -> HarmonicFit:
    """Fit a fixed harmonic basis jointly to both spin-two components."""
    first = np.asarray(shear_1, dtype=float)
    second = np.asarray(shear_2, dtype=float)
    selected = np.asarray(mask, dtype=bool)
    if first.shape != second.shape or first.shape != selected.shape or not np.any(selected):
        raise ValueError("shear components and nonempty mask must match")
    if not basis:
        raise ValueError("harmonic basis must be nonempty")
    names = sorted(basis)
    columns = []
    for name in names:
        basis_1, basis_2 = basis[name]
        if basis_1.shape != first.shape or basis_2.shape != first.shape:
            raise ValueError("every harmonic basis map must match the shear grid")
        columns.append(np.concatenate([basis_1[selected], basis_2[selected]]))
    design = np.column_stack(columns)
    target = np.concatenate([first[selected], second[selected]])
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    predicted_1 = np.zeros_like(first)
    predicted_2 = np.zeros_like(second)
    for coefficient, name in zip(coefficients, names, strict=True):
        predicted_1 += coefficient * basis[name][0]
        predicted_2 += coefficient * basis[name][1]
    denominator = float(np.sum(np.square(target)))
    if denominator <= 0.0:
        raise ValueError("target shear must have nonzero power")
    error = np.concatenate(
        [
            predicted_1[selected] - first[selected],
            predicted_2[selected] - second[selected],
        ]
    )
    normalized_rmse = float(np.sqrt(np.sum(np.square(error)) / denominator))
    return HarmonicFit(
        coefficients={name: float(value) for name, value in zip(names, coefficients, strict=True)},
        predicted_shear_1=predicted_1,
        predicted_shear_2=predicted_2,
        normalized_RMSE=normalized_rmse,
        power_closed=float(1.0 - normalized_rmse**2),
    )


def decompose_boundary_shear(
    missing_convergence: np.ndarray,
    missing_shear_1: np.ndarray,
    missing_shear_2: np.ndarray,
    radius_kpc: np.ndarray,
    mask: np.ndarray,
    basis: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    taper_start_kpc: float,
    taper_end_kpc: float,
    padding_factor: int,
) -> BoundaryDecomposition:
    """Subtract internal E-mode shear and fit the finite-window harmonic remainder."""
    convergence = np.asarray(missing_convergence, dtype=float)
    shear_1 = np.asarray(missing_shear_1, dtype=float)
    shear_2 = np.asarray(missing_shear_2, dtype=float)
    radius = np.asarray(radius_kpc, dtype=float)
    selected = np.asarray(mask, dtype=bool)
    if any(values.shape != convergence.shape for values in (shear_1, shear_2, radius, selected)):
        raise ValueError("missing-field maps, radius, and mask must have matching shapes")
    if any(np.any(~np.isfinite(values)) for values in (convergence, shear_1, shear_2)):
        raise ValueError("missing-field maps must be finite")
    taper = radial_taper(radius, start=taper_start_kpc, end=taper_end_kpc)
    tapered_convergence = taper * convergence
    internal_1, internal_2 = convergence_to_shear(
        tapered_convergence,
        padding_factor=padding_factor,
    )
    boundary_1 = shear_1 - internal_1
    boundary_2 = shear_2 - internal_2
    boundary_power = float(
        np.sum(np.square(boundary_1[selected]) + np.square(boundary_2[selected]))
    )
    total_power = float(np.sum(np.square(shear_1[selected]) + np.square(shear_2[selected])))
    if total_power <= 0.0:
        raise ValueError("missing shear must have nonzero power")
    harmonic = fit_harmonic_shear(boundary_1, boundary_2, selected, basis)
    return BoundaryDecomposition(
        tapered_internal_convergence=tapered_convergence,
        internal_shear_1=internal_1,
        internal_shear_2=internal_2,
        boundary_shear_1=boundary_1,
        boundary_shear_2=boundary_2,
        boundary_to_total_shear_power_ratio=boundary_power / total_power,
        harmonic_fit=harmonic,
    )


def shear_alignment_and_power_closed(
    predicted_shear_1: np.ndarray,
    predicted_shear_2: np.ndarray,
    target_shear_1: np.ndarray,
    target_shear_2: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """Score a predicted boundary field without using convergence."""
    selected = np.asarray(mask, dtype=bool)
    predicted = np.column_stack([predicted_shear_1[selected], predicted_shear_2[selected]])
    target = np.column_stack([target_shear_1[selected], target_shear_2[selected]])
    denominator = float(np.linalg.norm(predicted) * np.linalg.norm(target))
    alignment = float(np.sum(predicted * target) / denominator) if denominator > 0.0 else 0.0
    target_power = float(np.sum(np.square(target)))
    if target_power <= 0.0:
        raise ValueError("target boundary shear must have nonzero power")
    error_ratio = float(np.sum(np.square(predicted - target)) / target_power)
    return {
        "boundary_shear_alignment_cosine": alignment,
        "boundary_shear_normalized_RMSE": float(np.sqrt(error_ratio)),
        "boundary_shear_power_closed": float(1.0 - error_ratio),
    }
