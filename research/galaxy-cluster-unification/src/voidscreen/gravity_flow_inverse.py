"""Inverse diagnostics for baryon-sourced, nonlocal gravity-flow models.

The routines in this module are deliberately descriptive.  They infer a
minimum-cost coupling between a measured baryonic source distribution and a
lensing-derived destination map.  The coupling is not a measured trajectory
and it does not determine an unobserved line-of-sight arc height.
"""

from __future__ import annotations

import numpy as np

from voidscreen.gravity_arc_tomography import sinkhorn_transport


def normalized_positive(image: np.ndarray, aperture: np.ndarray) -> np.ndarray:
    """Clip an image to positive values and normalize it inside ``aperture``."""
    result = np.zeros_like(np.asarray(image, dtype=float))
    values = np.maximum(np.asarray(image, dtype=float)[aperture], 0.0)
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("image has no positive weight inside the aperture")
    result[aperture] = values / total
    return result


def local_projection_excess(
    target: np.ndarray,
    baryon_template: np.ndarray,
    aperture: np.ndarray,
    *,
    scale_fraction: float = 1.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Remove the best non-negative local-light projection from a target.

    Both inputs are normalized spatial shapes.  Consequently the fitted scale
    is a morphological nuisance coefficient, not a stellar mass-to-light
    ratio or a physical dark-matter fraction.
    """
    target_norm = normalized_positive(target, aperture)
    baryon_norm = normalized_positive(baryon_template, aperture)
    t = target_norm[aperture]
    b = baryon_norm[aperture]
    denominator = float(np.dot(b, b))
    fitted = 0.0 if denominator <= 0.0 else max(0.0, float(np.dot(b, t) / denominator))
    applied = float(np.clip(scale_fraction, 0.0, 1.0)) * fitted
    raw = np.maximum(t - applied * b, 0.0)
    raw_total = float(np.sum(raw))
    if raw_total <= np.finfo(float).tiny:
        raise ValueError("local-light subtraction removed the entire target")
    excess = np.zeros_like(target_norm)
    excess[aperture] = raw / raw_total
    removed = 1.0 - raw_total
    return excess, {
        "fitted_local_projection": fitted,
        "applied_local_projection": applied,
        "positive_target_weight_removed": float(removed),
        "positive_residual_weight_before_renormalization": raw_total,
    }


def coarsen_destination(
    image: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    *,
    factor: int,
    radius_kpc: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return coarse destination positions, weights, and their 2-D indices."""
    if int(factor) < 1:
        raise ValueError("factor must be positive")
    size_y, size_x = image.shape
    usable_y = (size_y // int(factor)) * int(factor)
    usable_x = (size_x // int(factor)) * int(factor)
    shape = (usable_y // factor, factor, usable_x // factor, factor)
    coarse = image[:usable_y, :usable_x].reshape(shape).sum(axis=(1, 3))
    coarse_x = x_grid[:usable_y, :usable_x].reshape(shape).mean(axis=(1, 3))
    coarse_y = y_grid[:usable_y, :usable_x].reshape(shape).mean(axis=(1, 3))
    mask = (np.hypot(coarse_x, coarse_y) <= float(radius_kpc)) & (coarse > 0.0)
    positions = np.column_stack([coarse_x[mask], coarse_y[mask]])
    weights = coarse[mask]
    weights /= np.sum(weights)
    indices = np.column_stack(np.nonzero(mask))
    return positions, weights, indices


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    """Weighted quantile for non-negative weights."""
    values = np.asarray(values, dtype=float).ravel()
    weights = np.asarray(weights, dtype=float).ravel()
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    return float(np.interp(float(quantile), cumulative, values[order]))


def solve_transport(
    source_positions: np.ndarray,
    source_weights: np.ndarray,
    destination_positions: np.ndarray,
    destination_weights: np.ndarray,
    *,
    entropy_length_kpc: float,
    iterations: int = 1000,
    tolerance: float = 1.0e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve balanced regularized transport and return plan and squared cost."""
    displacement = (
        destination_positions[None, :, :] - source_positions[:, None, :]
    )
    cost = np.sum(np.square(displacement), axis=2)
    plan = sinkhorn_transport(
        source_weights,
        destination_weights,
        cost,
        entropy=float(entropy_length_kpc) ** 2,
        iterations=int(iterations),
        tolerance=float(tolerance),
    )
    return plan, cost


def transport_diagnostics(
    plan: np.ndarray,
    source_positions: np.ndarray,
    source_weights: np.ndarray,
    destination_positions: np.ndarray,
    destination_weights: np.ndarray,
    *,
    baryonic_center: np.ndarray,
) -> dict[str, float]:
    """Summarize a source-to-destination coupling."""
    displacement = destination_positions[None, :, :] - source_positions[:, None, :]
    distance = np.linalg.norm(displacement, axis=2)
    unit = displacement / np.maximum(distance[:, :, None], np.finfo(float).tiny)
    inward = np.asarray(baryonic_center)[None, :] - source_positions
    inward /= np.maximum(np.linalg.norm(inward, axis=1)[:, None], np.finfo(float).tiny)
    moving = distance > 5.0
    moving_weight = np.where(moving, plan, 0.0)
    moving_weight /= max(float(np.sum(moving_weight)), np.finfo(float).tiny)
    inward_cosine = np.sum(unit * inward[:, None, :], axis=2)
    radial_source = np.linalg.norm(source_positions - baryonic_center[None, :], axis=1)
    radial_destination = np.linalg.norm(
        destination_positions - baryonic_center[None, :], axis=1
    )
    radial_change = radial_destination[None, :] - radial_source[:, None]
    return {
        "rms_transport_kpc": float(np.sqrt(np.sum(plan * np.square(distance)))),
        "mean_path_kpc": float(np.sum(plan * distance)),
        "median_path_kpc": weighted_quantile(distance, plan, 0.5),
        "p90_path_kpc": weighted_quantile(distance, plan, 0.9),
        "fraction_le_50_kpc": float(np.sum(plan[distance <= 50.0])),
        "fraction_le_100_kpc": float(np.sum(plan[distance <= 100.0])),
        "fraction_gt_150_kpc": float(np.sum(plan[distance > 150.0])),
        "mean_cos_inward": float(np.sum(moving_weight * inward_cosine)),
        "mean_radial_change_kpc": float(np.sum(plan * radial_change)),
        "fraction_ending_inward": float(np.sum(plan[radial_change < 0.0])),
        "source_marginal_max_error": float(
            np.max(np.abs(np.sum(plan, axis=1) - source_weights / np.sum(source_weights)))
        ),
        "target_marginal_max_error": float(
            np.max(
                np.abs(
                    np.sum(plan, axis=0)
                    - destination_weights / np.sum(destination_weights)
                )
            )
        ),
    }


def source_route_table(
    plan: np.ndarray,
    source_positions: np.ndarray,
    source_weights: np.ndarray,
    destination_positions: np.ndarray,
    *,
    baryonic_center: np.ndarray,
) -> list[dict[str, float | int]]:
    """Return per-source expected endpoints and routing features."""
    source = source_weights / np.sum(source_weights)
    displacement = destination_positions[None, :, :] - source_positions[:, None, :]
    distance = np.linalg.norm(displacement, axis=2)
    endpoint = np.einsum("ij,jk->ik", plan, destination_positions)
    endpoint /= np.maximum(source[:, None], np.finfo(float).tiny)
    mean_distance = np.sum(plan * distance, axis=1) / np.maximum(
        source, np.finfo(float).tiny
    )
    center = np.asarray(baryonic_center, dtype=float)
    rows: list[dict[str, float | int]] = []
    for index in range(len(source_positions)):
        delta = endpoint[index] - source_positions[index]
        inward = center - source_positions[index]
        norm = float(np.linalg.norm(delta) * np.linalg.norm(inward))
        cosine = 0.0 if norm == 0.0 else float(np.dot(delta, inward) / norm)
        rows.append(
            {
                "source_index": int(index),
                "source_x_kpc": float(source_positions[index, 0]),
                "source_y_kpc": float(source_positions[index, 1]),
                "source_weight": float(source[index]),
                "expected_endpoint_x_kpc": float(endpoint[index, 0]),
                "expected_endpoint_y_kpc": float(endpoint[index, 1]),
                "expected_displacement_kpc": float(np.linalg.norm(delta)),
                "conditional_mean_path_kpc": float(mean_distance[index]),
                "expected_direction_cos_inward": cosine,
            }
        )
    return rows


def rasterize_transport_paths(
    plan: np.ndarray,
    source_positions: np.ndarray,
    destination_positions: np.ndarray,
    axis_kpc: np.ndarray,
    *,
    samples_per_path: int = 17,
    retained_weight: float = 0.995,
) -> np.ndarray:
    """Rasterize the highest-weight projected transport chords.

    This is a minimum-path density diagnostic.  Curvature in an unobserved
    spatial direction is not encoded by a 2-D lens map and is therefore not
    invented here.
    """
    flat = plan.ravel()
    order = np.argsort(flat)[::-1]
    cumulative = np.cumsum(flat[order])
    count = int(np.searchsorted(cumulative, float(retained_weight), side="left") + 1)
    chosen = order[:count]
    source_index, destination_index = np.unravel_index(chosen, plan.shape)
    edge_weight = flat[chosen]
    fractions = np.linspace(0.0, 1.0, int(samples_per_path))
    start = source_positions[source_index]
    end = destination_positions[destination_index]
    points = start[:, None, :] + fractions[None, :, None] * (
        end[:, None, :] - start[:, None, :]
    )
    spacing = float(axis_kpc[1] - axis_kpc[0])
    pixel_x = np.rint((points[..., 0] - axis_kpc[0]) / spacing).astype(int)
    pixel_y = np.rint((points[..., 1] - axis_kpc[0]) / spacing).astype(int)
    weights = np.repeat(edge_weight / len(fractions), len(fractions))
    pixel_x = pixel_x.ravel()
    pixel_y = pixel_y.ravel()
    valid = (
        (pixel_x >= 0)
        & (pixel_x < len(axis_kpc))
        & (pixel_y >= 0)
        & (pixel_y < len(axis_kpc))
    )
    image = np.zeros((len(axis_kpc), len(axis_kpc)), dtype=float)
    np.add.at(image, (pixel_y[valid], pixel_x[valid]), weights[valid])
    total = float(np.sum(image))
    if total > 0.0:
        image /= total
    return image


def map_similarity(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    """Jensen-Shannon divergence and Pearson correlation for two path maps."""
    p = np.maximum(np.asarray(left, dtype=float).ravel(), 0.0)
    q = np.maximum(np.asarray(right, dtype=float).ravel(), 0.0)
    p /= np.sum(p)
    q /= np.sum(q)
    middle = 0.5 * (p + q)
    positive_p = p > 0.0
    positive_q = q > 0.0
    js = 0.5 * np.sum(p[positive_p] * np.log(p[positive_p] / middle[positive_p]))
    js += 0.5 * np.sum(q[positive_q] * np.log(q[positive_q] / middle[positive_q]))
    pearson = 0.0
    if np.std(p) > 0.0 and np.std(q) > 0.0:
        pearson = float(np.corrcoef(p, q)[0, 1])
    return {"jensen_shannon": float(js), "pearson": pearson}


def off_plane_arc_length(projected_distance: np.ndarray, height_ratio: float) -> np.ndarray:
    """Length of a parabolic off-plane arc z=4*h*t*(1-t), h=ratio*d.

    The result provides a transparent family of possible 3-D paths.  The
    height ratio is not inferable from the 2-D transport endpoints.
    """
    distance = np.asarray(projected_distance, dtype=float)
    ratio = abs(float(height_ratio))
    if ratio == 0.0:
        return distance.copy()
    a = 4.0 * ratio
    multiplier = 0.5 * np.sqrt(1.0 + a * a) + np.arcsinh(a) / (2.0 * a)
    return distance * multiplier
