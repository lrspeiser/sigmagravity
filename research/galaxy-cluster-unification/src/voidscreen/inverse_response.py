"""Theory-neutral inversion of stationary baryon-to-response kernels.

This module estimates the *effective response kernel* that makes a collection
of baryonic source maps reproduce supplied discovery-target maps.  It does not
interpret those targets as direct dark-matter observations and it does not
claim that a recovered kernel is a physical trajectory.  The intended use is
hypothesis generation followed by a separate, frozen forward test.

The forward convention exactly matches the generic field worker::

    response(y) = sum_x source(x) * kernel(y - x) * cell_volume

The convolution is linear, same-sized, centered on an odd kernel, and zero
padded outside the submitted domain.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
from scipy.optimize import lsq_linear
from scipy.signal import fftconvolve

Array = np.ndarray


@dataclass(frozen=True)
class StationaryKernelFit:
    raw_kernel: Array
    normalized_kernel: Array
    amplitude: float
    predictions: tuple[Array, ...]
    residuals: tuple[Array, ...]
    system_metrics: tuple[dict[str, float], ...]
    aggregate_metrics: dict[str, float]
    identifiability: dict[str, Any]
    optimizer: dict[str, Any]


@dataclass(frozen=True)
class InverseResponseAnalysis:
    fit: StationaryKernelFit
    kernel_lower: Array
    kernel_median: Array
    kernel_upper: Array
    amplitude_interval: dict[str, float]
    null_controls: tuple[dict[str, float | int | str | None], ...]
    null_summary: dict[str, Any]
    regularization_sensitivity: tuple[dict[str, Any], ...]
    non_identifiability: dict[str, Any]


def _spacing(values: float | Sequence[float], dimensions: int) -> tuple[float, ...]:
    if np.isscalar(values):
        result = (float(values),) * dimensions
    else:
        result = tuple(float(value) for value in values)
    if len(result) != dimensions or any(
        not math.isfinite(value) or value <= 0.0 for value in result
    ):
        raise ValueError(
            f"spacing must contain {dimensions} finite positive values"
        )
    return result


def _kernel_shape(values: Sequence[int], source_shape: tuple[int, ...]) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if len(result) != len(source_shape):
        raise ValueError("kernel shape must have one entry per map dimension")
    if any(value < 3 or value % 2 == 0 for value in result):
        raise ValueError("kernel shape requires at least three odd cells per dimension")
    if any(value > size for value, size in zip(result, source_shape, strict=True)):
        raise ValueError("kernel shape cannot exceed the source-map shape")
    return result


def convolve_stationary_response(
    source: Array,
    kernel: Array,
    spacing: float | Sequence[float],
) -> Array:
    """Apply the worker's centered, zero-padded physical convolution."""

    source_values = np.asarray(source, dtype=float)
    kernel_values = np.asarray(kernel, dtype=float)
    if source_values.ndim not in {2, 3}:
        raise ValueError("inverse response supports Cartesian 2D or 3D maps")
    steps = _spacing(spacing, source_values.ndim)
    _kernel_shape(kernel_values.shape, source_values.shape)
    if np.any(~np.isfinite(source_values)) or np.any(~np.isfinite(kernel_values)):
        raise ValueError("source and kernel must be finite")
    return np.asarray(
        fftconvolve(source_values, kernel_values, mode="same")
        * float(np.prod(steps)),
        dtype=float,
    )


def convolution_design_matrix(
    source: Array,
    kernel_shape: Sequence[int],
    spacing: float | Sequence[float],
) -> Array:
    """Return the exact linear design matrix for a compact response kernel."""

    source_values = np.asarray(source, dtype=float)
    if source_values.ndim not in {2, 3}:
        raise ValueError("inverse response supports Cartesian 2D or 3D maps")
    steps = _spacing(spacing, source_values.ndim)
    shape = _kernel_shape(kernel_shape, source_values.shape)
    columns: list[Array] = []
    for index in np.ndindex(shape):
        basis = np.zeros(shape, dtype=float)
        basis[index] = 1.0
        columns.append(
            convolve_stationary_response(source_values, basis, steps).ravel()
        )
    return np.column_stack(columns)


def _smoothness_matrix(shape: tuple[int, ...]) -> Array:
    rows: list[Array] = []
    count = int(np.prod(shape))
    for index in product(*(range(size) for size in shape)):
        flat = int(np.ravel_multi_index(index, shape))
        for axis in range(len(shape)):
            if index[axis] + 1 >= shape[axis]:
                continue
            neighbor = list(index)
            neighbor[axis] += 1
            neighbor_flat = int(np.ravel_multi_index(tuple(neighbor), shape))
            row = np.zeros(count, dtype=float)
            row[flat] = -1.0
            row[neighbor_flat] = 1.0
            rows.append(row)
    return np.vstack(rows) if rows else np.zeros((0, count), dtype=float)


def _validate_systems(
    sources: Sequence[Array],
    targets: Sequence[Array],
    uncertainties: Sequence[Array] | None,
    masks: Sequence[Array] | None,
) -> tuple[
    tuple[Array, ...],
    tuple[Array, ...],
    tuple[Array, ...],
    tuple[Array, ...],
]:
    if not sources or len(sources) != len(targets):
        raise ValueError("sources and targets require the same non-zero system count")
    source_values = tuple(np.asarray(value, dtype=float) for value in sources)
    target_values = tuple(np.asarray(value, dtype=float) for value in targets)
    dimensions = source_values[0].ndim
    shape = source_values[0].shape
    if dimensions not in {2, 3}:
        raise ValueError("inverse response supports Cartesian 2D or 3D maps")
    for source, target in zip(source_values, target_values, strict=True):
        if source.shape != shape or target.shape != shape:
            raise ValueError("all source and target maps must share one shape")
        if np.any(~np.isfinite(source)) or np.any(~np.isfinite(target)):
            raise ValueError("source and target maps must be finite")

    if uncertainties is None:
        uncertainty_values = tuple(np.ones(shape, dtype=float) for _ in source_values)
    else:
        if len(uncertainties) != len(source_values):
            raise ValueError("uncertainties must match the system count")
        uncertainty_values = tuple(np.asarray(value, dtype=float) for value in uncertainties)
        for uncertainty in uncertainty_values:
            if uncertainty.shape != shape or np.any(~np.isfinite(uncertainty)):
                raise ValueError("uncertainty maps must be finite and match the source shape")
            if np.any(uncertainty <= 0.0):
                raise ValueError("uncertainty maps must be strictly positive")

    if masks is None:
        mask_values = tuple(np.ones(shape, dtype=bool) for _ in source_values)
    else:
        if len(masks) != len(source_values):
            raise ValueError("masks must match the system count")
        mask_values = tuple(np.asarray(value, dtype=bool) for value in masks)
        for mask in mask_values:
            if mask.shape != shape or not np.any(mask):
                raise ValueError("each mask must match the source shape and select pixels")
    return source_values, target_values, uncertainty_values, mask_values


def _metrics(target: Array, prediction: Array, uncertainty: Array, mask: Array) -> dict[str, float]:
    observed = target[mask]
    predicted = prediction[mask]
    residual = predicted - observed
    scaled = residual / uncertainty[mask]
    denominator = float(np.sum(np.square(observed - np.mean(observed))))
    r_squared = (
        1.0 - float(np.sum(np.square(residual))) / denominator
        if denominator > np.finfo(float).tiny
        else 0.0
    )
    return {
        "pixels": int(np.sum(mask)),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "weighted_rmse": float(np.sqrt(np.mean(np.square(scaled)))),
        "chi_square": float(np.sum(np.square(scaled))),
        "r_squared": r_squared,
    }


def fit_stationary_response_kernel(
    sources: Sequence[Array],
    targets: Sequence[Array],
    spacing: float | Sequence[float],
    kernel_shape: Sequence[int],
    *,
    uncertainties: Sequence[Array] | None = None,
    masks: Sequence[Array] | None = None,
    ridge: float = 1.0e-8,
    smoothness: float = 1.0e-4,
    nonnegative: bool = True,
) -> StationaryKernelFit:
    """Fit one stationary kernel and amplitude across all submitted systems."""

    source_values, target_values, uncertainty_values, mask_values = _validate_systems(
        sources, targets, uncertainties, masks
    )
    steps = _spacing(spacing, source_values[0].ndim)
    shape = _kernel_shape(kernel_shape, source_values[0].shape)
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge regularization must be finite and non-negative")
    if not math.isfinite(smoothness) or smoothness < 0.0:
        raise ValueError("smoothness regularization must be finite and non-negative")

    design_blocks: list[Array] = []
    target_blocks: list[Array] = []
    for source, target, uncertainty, mask in zip(
        source_values,
        target_values,
        uncertainty_values,
        mask_values,
        strict=True,
    ):
        design = convolution_design_matrix(source, shape, steps)
        selected = mask.ravel()
        sigma = uncertainty.ravel()[selected]
        design_blocks.append(design[selected] / sigma[:, None])
        target_blocks.append(target.ravel()[selected] / sigma)
    weighted_design = np.vstack(design_blocks)
    weighted_target = np.concatenate(target_blocks)
    parameters = int(np.prod(shape))
    scale = max(
        float(np.square(np.linalg.norm(weighted_design, ord="fro")) / parameters),
        np.finfo(float).tiny,
    )
    augmented_design = [weighted_design]
    augmented_target = [weighted_target]
    if ridge > 0.0:
        augmented_design.append(np.sqrt(ridge * scale) * np.eye(parameters))
        augmented_target.append(np.zeros(parameters, dtype=float))
    difference = _smoothness_matrix(shape)
    if smoothness > 0.0 and len(difference):
        augmented_design.append(np.sqrt(smoothness * scale) * difference)
        augmented_target.append(np.zeros(len(difference), dtype=float))
    lower = 0.0 if nonnegative else -np.inf
    optimization = lsq_linear(
        np.vstack(augmented_design),
        np.concatenate(augmented_target),
        bounds=(lower, np.inf),
        method="trf",
        tol=1.0e-12,
        lsmr_tol="auto",
        max_iter=1000,
    )
    if not optimization.success or np.any(~np.isfinite(optimization.x)):
        raise RuntimeError(f"kernel inversion failed: {optimization.message}")
    raw_kernel = np.asarray(optimization.x.reshape(shape), dtype=float)
    cell_volume = float(np.prod(steps))
    amplitude = float(np.sum(np.abs(raw_kernel)) * cell_volume)
    if not math.isfinite(amplitude) or amplitude <= np.finfo(float).tiny:
        raise RuntimeError("kernel inversion returned zero response amplitude")
    normalized_kernel = raw_kernel / amplitude
    predictions = tuple(
        convolve_stationary_response(source, raw_kernel, steps)
        for source in source_values
    )
    residuals = tuple(
        prediction - target
        for prediction, target in zip(predictions, target_values, strict=True)
    )
    system_metrics = tuple(
        _metrics(target, prediction, uncertainty, mask)
        for target, prediction, uncertainty, mask in zip(
            target_values,
            predictions,
            uncertainty_values,
            mask_values,
            strict=True,
        )
    )
    all_scaled = np.concatenate(
        [
            residual[mask] / uncertainty[mask]
            for residual, uncertainty, mask in zip(
                residuals, uncertainty_values, mask_values, strict=True
            )
        ]
    )
    all_observed = np.concatenate(
        [target[mask] for target, mask in zip(target_values, mask_values, strict=True)]
    )
    all_predicted = np.concatenate(
        [
            prediction[mask]
            for prediction, mask in zip(predictions, mask_values, strict=True)
        ]
    )
    denominator = float(np.sum(np.square(all_observed - np.mean(all_observed))))
    singular_values = np.linalg.svd(weighted_design, compute_uv=False)
    threshold = (
        float(singular_values[0]) * 1.0e-6 if len(singular_values) else 0.0
    )
    effective_rank = int(np.sum(singular_values > threshold))
    positive = singular_values[singular_values > threshold]
    condition = (
        float(positive[0] / positive[-1]) if len(positive) else float("inf")
    )
    aggregate_metrics = {
        "systems": len(source_values),
        "pixels": len(all_scaled),
        "rmse": float(np.sqrt(np.mean(np.square(all_predicted - all_observed)))),
        "weighted_rmse": float(np.sqrt(np.mean(np.square(all_scaled)))),
        "chi_square": float(np.sum(np.square(all_scaled))),
        "r_squared": (
            1.0
            - float(np.sum(np.square(all_predicted - all_observed))) / denominator
            if denominator > np.finfo(float).tiny
            else 0.0
        ),
    }
    return StationaryKernelFit(
        raw_kernel=raw_kernel,
        normalized_kernel=normalized_kernel,
        amplitude=amplitude,
        predictions=predictions,
        residuals=residuals,
        system_metrics=system_metrics,
        aggregate_metrics=aggregate_metrics,
        identifiability={
            "kernel_cells": parameters,
            "effective_rank": effective_rank,
            "nullity": parameters - effective_rank,
            "condition_number_at_threshold": condition,
            "singular_value_threshold": threshold,
            "largest_singular_value": (
                float(singular_values[0]) if len(singular_values) else 0.0
            ),
            "smallest_retained_singular_value": (
                float(positive[-1]) if len(positive) else 0.0
            ),
        },
        optimizer={
            "success": bool(optimization.success),
            "status": int(optimization.status),
            "message": str(optimization.message),
            "iterations": int(optimization.nit),
            "cost": float(optimization.cost),
            "optimality": float(optimization.optimality),
            "ridge": float(ridge),
            "smoothness": float(smoothness),
            "nonnegative": bool(nonnegative),
        },
    )


def radial_angle_shuffle(values: Array, rng: np.random.Generator) -> Array:
    """Shuffle values within pixel-radius shells while preserving the radial profile."""

    array = np.asarray(values, dtype=float)
    coordinates = np.indices(array.shape, dtype=float)
    center = np.asarray([(size - 1.0) / 2.0 for size in array.shape], dtype=float)
    radius_squared = np.zeros(array.shape, dtype=float)
    for axis in range(array.ndim):
        radius_squared += np.square(coordinates[axis] - center[axis])
    shells = np.rint(np.sqrt(radius_squared)).astype(int)
    result = array.copy()
    for shell in np.unique(shells):
        selected = np.flatnonzero(shells.ravel() == shell)
        if len(selected) > 1:
            result.ravel()[selected] = array.ravel()[rng.permutation(selected)]
    return result


def phase_scramble(values: Array, rng: np.random.Generator) -> Array:
    """Randomize Fourier phase while preserving power and the map mean."""

    array = np.asarray(values, dtype=float)
    if array.ndim not in {2, 3} or np.any(~np.isfinite(array)):
        raise ValueError("phase scrambling requires a finite 2D or 3D map")
    spectrum = np.fft.fftn(array)
    random_spectrum = np.fft.fftn(rng.normal(size=array.shape))
    phase = np.ones_like(random_spectrum, dtype=complex)
    nonzero = np.abs(random_spectrum) > np.finfo(float).tiny
    phase[nonzero] = random_spectrum[nonzero] / np.abs(random_spectrum[nonzero])
    scrambled_spectrum = np.abs(spectrum) * phase
    origin = (0,) * array.ndim
    scrambled_spectrum[origin] = spectrum[origin]
    return np.asarray(np.fft.ifftn(scrambled_spectrum).real, dtype=float)


def missing_baryon_dropout(
    values: Array,
    rng: np.random.Generator,
    dropout_fraction: float,
) -> Array:
    """Remove random source cells and rescale to preserve total baryonic input."""

    array = np.asarray(values, dtype=float)
    fraction = float(dropout_fraction)
    if array.ndim not in {2, 3} or np.any(~np.isfinite(array)):
        raise ValueError("missing-baryon dropout requires a finite 2D or 3D map")
    if np.any(array < 0.0) or float(np.sum(array)) <= 0.0:
        raise ValueError("missing-baryon dropout requires a non-negative source")
    if not math.isfinite(fraction) or not 0.0 < fraction <= 0.5:
        raise ValueError("dropout_fraction must be greater than zero and at most 0.5")
    kept = rng.random(array.shape) >= fraction
    if not np.any(kept & (array > 0.0)):
        kept.flat[int(np.argmax(array))] = True
    result = np.where(kept, array, 0.0)
    result *= float(np.sum(array)) / float(np.sum(result))
    return np.asarray(result, dtype=float)


NULL_PRESERVED_QUANTITIES = {
    "source_radial_angle_shuffle": "source radial shell values",
    "source_phase_scramble": "source Fourier power spectrum and mean",
    "target_system_permutation": "target, uncertainty, and mask maps as a system-level multiset",
    "target_radial_angle_shuffle": "target radial shell values",
    "source_missing_baryon_dropout": "source total integral after random-cell dropout and rescaling",
}


def _normalized_null_families(
    null_families: Sequence[Mapping[str, Any]] | None,
    null_count: int,
    null_seed: int,
) -> tuple[dict[str, Any], ...]:
    raw_families: Sequence[Mapping[str, Any]] = (
        (
            {
                "kind": "source_radial_angle_shuffle",
                "count": null_count,
                "seed": null_seed,
            },
        )
        if null_families is None
        else null_families
    )
    normalized: list[dict[str, Any]] = []
    seen_kinds: set[str] = set()
    for family_index, family in enumerate(raw_families):
        if not isinstance(family, Mapping):
            raise TypeError("each null family must be an object")
        unknown = set(family) - {"kind", "count", "seed", "dropoutFraction"}
        if unknown:
            raise ValueError(f"unsupported null family properties: {sorted(unknown)}")
        kind = str(family.get("kind", ""))
        if kind not in NULL_PRESERVED_QUANTITIES:
            raise ValueError(f"unsupported null family: {kind}")
        if kind in seen_kinds:
            raise ValueError(f"duplicate null family: {kind}")
        seen_kinds.add(kind)
        count = family.get("count", 19)
        seed = family.get("seed", family_index + 1)
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 999:
            raise ValueError("null family count must be an integer from 0 to 999")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**31 - 1:
            raise ValueError("null family seed must be an integer from 0 to 2147483647")
        record: dict[str, Any] = {"kind": kind, "count": count, "seed": seed}
        if kind == "source_missing_baryon_dropout":
            fraction = float(family.get("dropoutFraction", 0.15))
            if not math.isfinite(fraction) or not 0.0 < fraction <= 0.5:
                raise ValueError(
                    "source_missing_baryon_dropout dropoutFraction must be greater than zero and at most 0.5"
                )
            record["dropoutFraction"] = fraction
        elif "dropoutFraction" in family:
            raise ValueError("dropoutFraction is only valid for source_missing_baryon_dropout")
        normalized.append(record)
    return tuple(normalized)


def _null_inputs(
    family: Mapping[str, Any],
    sources: tuple[Array, ...],
    targets: tuple[Array, ...],
    uncertainties: tuple[Array, ...],
    masks: tuple[Array, ...],
    rng: np.random.Generator,
) -> tuple[tuple[Array, ...], tuple[Array, ...], tuple[Array, ...], tuple[Array, ...]]:
    kind = str(family["kind"])
    if kind == "source_radial_angle_shuffle":
        return (
            tuple(radial_angle_shuffle(source, rng) for source in sources),
            targets,
            uncertainties,
            masks,
        )
    if kind == "source_phase_scramble":
        return (
            tuple(phase_scramble(source, rng) for source in sources),
            targets,
            uncertainties,
            masks,
        )
    if kind == "target_radial_angle_shuffle":
        return (
            sources,
            tuple(radial_angle_shuffle(target, rng) for target in targets),
            uncertainties,
            masks,
        )
    if kind == "target_system_permutation":
        if len(targets) < 2:
            raise ValueError("target_system_permutation requires at least two systems")
        shift = int(rng.integers(1, len(targets)))
        indices = np.roll(np.arange(len(targets)), shift)
        return (
            sources,
            tuple(targets[int(index)] for index in indices),
            tuple(uncertainties[int(index)] for index in indices),
            tuple(masks[int(index)] for index in indices),
        )
    if kind == "source_missing_baryon_dropout":
        fraction = float(family["dropoutFraction"])
        return (
            tuple(missing_baryon_dropout(source, rng, fraction) for source in sources),
            targets,
            uncertainties,
            masks,
        )
    raise ValueError(f"unsupported null family: {kind}")


def _kernel_distance(left: Array, right: Array, cell_volume: float) -> dict[str, float]:
    left_flat = np.asarray(left, dtype=float).ravel()
    right_flat = np.asarray(right, dtype=float).ravel()
    norm = float(np.linalg.norm(left_flat) * np.linalg.norm(right_flat))
    cosine = float(np.dot(left_flat, right_flat) / norm) if norm > 0.0 else 0.0
    return {
        "kernel_cosine": cosine,
        "kernel_l1": float(np.sum(np.abs(left_flat - right_flat)) * cell_volume),
    }


def analyze_stationary_response(
    sources: Sequence[Array],
    targets: Sequence[Array],
    spacing: float | Sequence[float],
    kernel_shape: Sequence[int],
    *,
    uncertainties: Sequence[Array] | None = None,
    masks: Sequence[Array] | None = None,
    ridge: float = 1.0e-8,
    smoothness: float = 1.0e-4,
    nonnegative: bool = True,
    ensemble_size: int = 32,
    ensemble_seed: int = 0,
    null_count: int = 19,
    null_seed: int = 1,
    null_families: Sequence[Mapping[str, Any]] | None = None,
    regularization_multipliers: Sequence[float] = (0.1, 1.0, 10.0),
) -> InverseResponseAnalysis:
    """Fit a kernel, uncertainty ensemble, declared nulls, and sensitivity grid."""

    source_values, target_values, uncertainty_values, mask_values = _validate_systems(
        sources, targets, uncertainties, masks
    )
    steps = _spacing(spacing, source_values[0].ndim)
    if not isinstance(ensemble_size, int) or not 0 <= ensemble_size <= 512:
        raise ValueError("ensemble_size must be an integer from 0 to 512")
    if not isinstance(null_count, int) or not 0 <= null_count <= 999:
        raise ValueError("null_count must be an integer from 0 to 999")
    families = _normalized_null_families(null_families, null_count, null_seed)
    if any(
        family["kind"] == "target_system_permutation" and family["count"] > 0
        for family in families
    ) and len(source_values) < 2:
        raise ValueError("target_system_permutation requires at least two systems")
    fit = fit_stationary_response_kernel(
        source_values,
        target_values,
        steps,
        kernel_shape,
        uncertainties=uncertainty_values,
        masks=mask_values,
        ridge=ridge,
        smoothness=smoothness,
        nonnegative=nonnegative,
    )

    ensemble_kernels = [fit.normalized_kernel]
    ensemble_amplitudes = [fit.amplitude]
    if ensemble_size:
        rng = np.random.default_rng(int(ensemble_seed))
        for _ in range(ensemble_size):
            perturbed = tuple(
                target + rng.normal(0.0, uncertainty)
                for target, uncertainty in zip(
                    target_values, uncertainty_values, strict=True
                )
            )
            sample = fit_stationary_response_kernel(
                source_values,
                perturbed,
                steps,
                kernel_shape,
                uncertainties=uncertainty_values,
                masks=mask_values,
                ridge=ridge,
                smoothness=smoothness,
                nonnegative=nonnegative,
            )
            ensemble_kernels.append(sample.normalized_kernel)
            ensemble_amplitudes.append(sample.amplitude)
    kernel_samples = np.stack(ensemble_kernels)
    amplitude_samples = np.asarray(ensemble_amplitudes, dtype=float)

    null_rows: list[dict[str, float | int | str | None]] = []
    family_summaries: list[dict[str, Any]] = []
    observed_error = float(fit.aggregate_metrics["weighted_rmse"])
    observed_r_squared = float(fit.aggregate_metrics["r_squared"])
    for family_index, family in enumerate(families):
        family_rows: list[dict[str, float | int | str | None]] = []
        rng = np.random.default_rng(int(family["seed"]))
        for replicate_index in range(int(family["count"])):
            (
                null_sources,
                null_targets,
                null_uncertainties,
                null_masks,
            ) = _null_inputs(
                family,
                source_values,
                target_values,
                uncertainty_values,
                mask_values,
                rng,
            )
            null_fit = fit_stationary_response_kernel(
                null_sources,
                null_targets,
                steps,
                kernel_shape,
                uncertainties=null_uncertainties,
                masks=null_masks,
                ridge=ridge,
                smoothness=smoothness,
                nonnegative=nonnegative,
            )
            row: dict[str, float | int | str | None] = {
                "family_index": family_index,
                "replicate_index": replicate_index,
                "kind": str(family["kind"]),
                "seed": int(family["seed"]),
                "dropout_fraction": family.get("dropoutFraction"),
                "weighted_rmse": null_fit.aggregate_metrics["weighted_rmse"],
                "rmse": null_fit.aggregate_metrics["rmse"],
                "r_squared": null_fit.aggregate_metrics["r_squared"],
                "amplitude": null_fit.amplitude,
            }
            family_rows.append(row)
            null_rows.append(row)
        as_good = sum(
            float(row["weighted_rmse"]) <= observed_error for row in family_rows
        )
        p_value = (1.0 + as_good) / (1.0 + len(family_rows))
        family_summaries.append(
            {
                "kind": family["kind"],
                "count": len(family_rows),
                "seed": family["seed"],
                **(
                    {"dropout_fraction": family["dropoutFraction"]}
                    if "dropoutFraction" in family
                    else {}
                ),
                "observed_weighted_rmse": observed_error,
                "median_null_weighted_rmse": (
                    float(
                        np.median(
                            [row["weighted_rmse"] for row in family_rows]
                        )
                    )
                    if family_rows
                    else None
                ),
                "monte_carlo_p_value": float(p_value),
                "permutation_p_value": float(p_value),
                "minimum_r_squared_gate": 0.25,
                "signal_against_null": bool(
                    family_rows
                    and p_value <= 0.05
                    and observed_r_squared >= 0.25
                ),
                "preserved_quantity": NULL_PRESERVED_QUANTITIES[str(family["kind"])],
            }
        )
    maximum_p_value = max(
        (float(summary["monte_carlo_p_value"]) for summary in family_summaries),
        default=1.0,
    )
    null_summary = {
        "combination_rule": "all_declared_families",
        "family_count": len(family_summaries),
        "total_count": len(null_rows),
        "observed_weighted_rmse": observed_error,
        "maximum_family_p_value": maximum_p_value,
        "minimum_r_squared_gate": 0.25,
        "signal_against_null": bool(
            family_summaries
            and all(summary["signal_against_null"] for summary in family_summaries)
        ),
        "families": family_summaries,
    }
    if len(family_summaries) == 1:
        null_summary.update(family_summaries[0])

    cell_volume = float(np.prod(steps))
    sensitivity_rows: list[dict[str, Any]] = []
    compatible_kernel_l1: list[float] = []
    for multiplier in regularization_multipliers:
        multiplier_value = float(multiplier)
        if not math.isfinite(multiplier_value) or multiplier_value <= 0.0:
            raise ValueError("regularization multipliers must be finite and positive")
        alternative = fit_stationary_response_kernel(
            source_values,
            target_values,
            steps,
            kernel_shape,
            uncertainties=uncertainty_values,
            masks=mask_values,
            ridge=ridge * multiplier_value,
            smoothness=smoothness * multiplier_value,
            nonnegative=nonnegative,
        )
        distance = _kernel_distance(
            fit.normalized_kernel, alternative.normalized_kernel, cell_volume
        )
        relative_error = (
            float(alternative.aggregate_metrics["weighted_rmse"]) / observed_error - 1.0
            if observed_error > np.finfo(float).tiny
            else 0.0
        )
        compatible = relative_error <= 0.02
        if compatible:
            compatible_kernel_l1.append(distance["kernel_l1"])
        sensitivity_rows.append(
            {
                "multiplier": multiplier_value,
                "ridge": ridge * multiplier_value,
                "smoothness": smoothness * multiplier_value,
                "weighted_rmse": alternative.aggregate_metrics["weighted_rmse"],
                "relative_weighted_rmse_change": relative_error,
                "amplitude": alternative.amplitude,
                **distance,
                "compatible_within_two_percent": compatible,
            }
        )
    rank_deficient = int(fit.identifiability["nullity"]) > 0
    ill_conditioned = float(fit.identifiability["condition_number_at_threshold"]) > 1.0e6
    maximum_compatible_l1 = max(compatible_kernel_l1, default=0.0)
    materially_distinct = maximum_compatible_l1 >= 0.25
    non_identifiability = {
        "rank_deficient": rank_deficient,
        "ill_conditioned": ill_conditioned,
        "materially_distinct_compatible_kernels": materially_distinct,
        "maximum_compatible_normalized_kernel_l1": maximum_compatible_l1,
        "non_identifiable": bool(
            rank_deficient or ill_conditioned or materially_distinct
        ),
        "interpretation": (
            "Multiple kernel directions are weakly constrained; report a family of compatible responses."
            if rank_deficient or ill_conditioned or materially_distinct
            else "This submitted design resolves the compact kernel at the declared numerical threshold."
        ),
    }
    return InverseResponseAnalysis(
        fit=fit,
        kernel_lower=np.quantile(kernel_samples, 0.025, axis=0),
        kernel_median=np.quantile(kernel_samples, 0.5, axis=0),
        kernel_upper=np.quantile(kernel_samples, 0.975, axis=0),
        amplitude_interval={
            "lower_2_5": float(np.quantile(amplitude_samples, 0.025)),
            "median": float(np.quantile(amplitude_samples, 0.5)),
            "upper_97_5": float(np.quantile(amplitude_samples, 0.975)),
            "samples": len(amplitude_samples),
        },
        null_controls=tuple(null_rows),
        null_summary=null_summary,
        regularization_sensitivity=tuple(sensitivity_rows),
        non_identifiability=non_identifiability,
    )
