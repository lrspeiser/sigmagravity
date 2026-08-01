"""Rasterized conservative center-return templates from baryonic sources."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def weighted_radius(radius, weights, fraction: float) -> float:
    radius = np.asarray(radius, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if radius.shape != weights.shape or radius.ndim != 1:
        raise ValueError("radius and weights must be matching vectors")
    if np.any(radius < 0.0) or np.any(weights < 0.0) or np.sum(weights) <= 0.0:
        raise ValueError("invalid radius or weight values")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must lie in [0, 1]")
    order = np.argsort(radius)
    cumulative = np.cumsum(weights[order]) / np.sum(weights)
    return float(np.interp(float(fraction), cumulative, radius[order]))


def center_return_endpoints(
    positions,
    weights,
    *,
    return_scale: float,
    radius_exponent: float,
    reference_radius: float,
    travel_mode: str = "constant",
    center=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return endpoints and the baryonic light centroid used for routing."""
    xy = np.asarray(positions, dtype=float)
    weight = np.asarray(weights, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or weight.shape != (len(xy),):
        raise ValueError("positions and weights do not match")
    if np.any(~np.isfinite(xy)) or np.any(weight < 0.0) or np.sum(weight) <= 0.0:
        raise ValueError("invalid source positions or weights")
    if return_scale <= 0.0 or reference_radius <= 0.0:
        raise ValueError("return and reference scales must be positive")
    weight = weight / np.sum(weight)
    centroid = (
        np.sum(xy * weight[:, None], axis=0)
        if center is None
        else np.asarray(center, dtype=float)
    )
    inward = centroid[None, :] - xy
    radius = np.linalg.norm(inward, axis=1)
    direction = inward / np.maximum(radius[:, None], np.finfo(float).tiny)
    scale = float(return_scale) * np.power(
        np.maximum(radius / float(reference_radius), 1.0e-6),
        float(radius_exponent),
    )
    scale = np.clip(scale, 0.2 * float(return_scale), 3.0 * float(return_scale))
    mode = str(travel_mode)
    if mode == "hard_no_cross":
        scale = np.minimum(scale, radius)
    elif mode == "tanh_no_cross":
        scale = scale * np.tanh(radius / np.maximum(scale, np.finfo(float).tiny))
    elif mode == "rational_no_cross":
        scale = scale * radius / np.maximum(scale + radius, np.finfo(float).tiny)
    elif mode != "constant":
        raise ValueError(
            "travel_mode must be constant, hard_no_cross, tanh_no_cross, or rational_no_cross"
        )
    return xy + scale[:, None] * direction, centroid


def deposit_sources(axis, positions, weights, *, smoothing: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    xy = np.asarray(positions, dtype=float)
    weight = np.asarray(weights, dtype=float)
    if axis.ndim != 1 or len(axis) < 16 or not np.all(np.diff(axis) > 0.0):
        raise ValueError("axis must be a strictly increasing vector")
    if xy.ndim != 2 or xy.shape[1] != 2 or weight.shape != (len(xy),):
        raise ValueError("positions and weights do not match")
    spacing = float(axis[1] - axis[0])
    if not np.allclose(np.diff(axis), spacing) or smoothing <= 0.0:
        raise ValueError("axis must be uniform and smoothing positive")
    edges = np.r_[axis - 0.5 * spacing, axis[-1] + 0.5 * spacing]
    histogram, _, _ = np.histogram2d(
        xy[:, 1], xy[:, 0], bins=(edges, edges), weights=weight
    )
    image = gaussian_filter(
        histogram,
        sigma=float(smoothing) / spacing,
        mode="constant",
        cval=0.0,
    )
    total = float(np.sum(image))
    if total <= 0.0:
        raise ValueError("all deposited source weight fell outside the grid")
    return image / total


def conservative_route_template(
    axis,
    positions,
    weights,
    *,
    routing_fraction: float,
    return_scale: float,
    radius_exponent: float,
    reference_radius: float,
    smoothing: float,
    travel_mode: str = "constant",
    center=None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Blend local and returned endpoint maps with exact unit normalization."""
    if not 0.0 <= routing_fraction <= 1.0:
        raise ValueError("routing_fraction must lie in [0, 1]")
    endpoints, centroid = center_return_endpoints(
        positions,
        weights,
        return_scale=return_scale,
        radius_exponent=radius_exponent,
        reference_radius=reference_radius,
        travel_mode=travel_mode,
        center=center,
    )
    local = deposit_sources(axis, positions, weights, smoothing=smoothing)
    routed = deposit_sources(axis, endpoints, weights, smoothing=smoothing)
    combined = (1.0 - float(routing_fraction)) * local + float(routing_fraction) * routed
    combined /= np.sum(combined)
    source_vector = np.asarray(positions, dtype=float) - centroid[None, :]
    endpoint_vector = endpoints - centroid[None, :]
    crossed = np.sum(source_vector * endpoint_vector, axis=1) < 0.0
    normalized_weight = np.asarray(weights, dtype=float) / np.sum(weights)
    travel = np.linalg.norm(endpoints - np.asarray(positions, dtype=float), axis=1)
    return combined, {
        "centroid": centroid,
        "endpoints": endpoints,
        "local_map": local,
        "routed_map": routed,
        "normalization_error": abs(float(np.sum(combined)) - 1.0),
        "travel_mode": str(travel_mode),
        "sources_crossing_center": int(np.sum(crossed)),
        "source_weight_crossing_center": float(np.sum(normalized_weight[crossed])),
        "maximum_travel": float(np.max(travel)),
        "median_travel": float(np.median(travel)),
    }


def local_baryonic_attractor_endpoints(
    positions,
    weights,
    *,
    return_scale: float,
    softening: float,
    distance_power: float,
    local_mix: float,
    travel_mode: str = "tanh_no_cross",
    center=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approach baryon-derived local attractors without crossing them."""
    xy = np.asarray(positions, dtype=float)
    weight = np.asarray(weights, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or weight.shape != (len(xy),):
        raise ValueError("positions and weights do not match")
    if np.any(~np.isfinite(xy)) or np.any(weight < 0.0) or np.sum(weight) <= 0.0:
        raise ValueError("invalid source positions or weights")
    if return_scale <= 0.0 or softening <= 0.0 or distance_power <= 0.0:
        raise ValueError("return scale, softening, and distance power must be positive")
    if not 0.0 <= local_mix <= 1.0:
        raise ValueError("local_mix must lie in [0, 1]")
    weight = weight / np.sum(weight)
    centroid = (
        np.sum(xy * weight[:, None], axis=0)
        if center is None
        else np.asarray(center, dtype=float)
    )
    if len(xy) == 1:
        local_attractor = np.repeat(centroid[None, :], 1, axis=0)
    else:
        separation = xy[None, :, :] - xy[:, None, :]
        distance2 = np.sum(np.square(separation), axis=2)
        kernel = np.power(
            distance2 + float(softening) ** 2,
            -0.5 * float(distance_power),
        )
        np.fill_diagonal(kernel, 0.0)
        influence = kernel * weight[None, :]
        influence_sum = np.sum(influence, axis=1)
        local_attractor = np.divide(
            influence @ xy,
            influence_sum[:, None],
            out=np.repeat(centroid[None, :], len(xy), axis=0),
            where=influence_sum[:, None] > np.finfo(float).tiny,
        )
    target = (
        (1.0 - float(local_mix)) * centroid[None, :]
        + float(local_mix) * local_attractor
    )
    displacement = target - xy
    distance = np.linalg.norm(displacement, axis=1)
    direction = displacement / np.maximum(
        distance[:, None], np.finfo(float).tiny
    )
    mode = str(travel_mode)
    scale = np.full(len(xy), float(return_scale), dtype=float)
    if mode == "hard_no_cross":
        scale = np.minimum(scale, distance)
    elif mode == "tanh_no_cross":
        scale *= np.tanh(distance / float(return_scale))
    elif mode == "rational_no_cross":
        scale = float(return_scale) * distance / (float(return_scale) + distance)
    elif mode != "constant":
        raise ValueError(
            "travel_mode must be constant, hard_no_cross, tanh_no_cross, or rational_no_cross"
        )
    endpoints = xy + scale[:, None] * direction
    endpoints[distance <= np.finfo(float).tiny] = xy[distance <= np.finfo(float).tiny]
    return endpoints, target, centroid


def conservative_local_attractor_route_template(
    axis,
    positions,
    weights,
    *,
    routing_fraction: float,
    return_scale: float,
    smoothing: float,
    softening: float,
    distance_power: float,
    local_mix: float,
    travel_mode: str = "tanh_no_cross",
    center=None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Conservative endpoint map with one baryon-derived attractor per source."""
    if not 0.0 <= routing_fraction <= 1.0:
        raise ValueError("routing_fraction must lie in [0, 1]")
    xy = np.asarray(positions, dtype=float)
    weight = np.asarray(weights, dtype=float)
    weight = weight / np.sum(weight)
    endpoints, targets, centroid = local_baryonic_attractor_endpoints(
        xy,
        weight,
        return_scale=return_scale,
        softening=softening,
        distance_power=distance_power,
        local_mix=local_mix,
        travel_mode=travel_mode,
        center=center,
    )
    local = deposit_sources(axis, xy, weight, smoothing=smoothing)
    routed = deposit_sources(axis, endpoints, weight, smoothing=smoothing)
    combined = (1.0 - float(routing_fraction)) * local + float(routing_fraction) * routed
    combined /= np.sum(combined)
    target_distance = np.linalg.norm(targets - xy, axis=1)
    travel = np.linalg.norm(endpoints - xy, axis=1)
    target_vector = targets - xy
    endpoint_vector = targets - endpoints
    crossed = np.sum(target_vector * endpoint_vector, axis=1) < 0.0
    return combined, {
        "centroid": centroid,
        "targets": targets,
        "endpoints": endpoints,
        "local_map": local,
        "routed_map": routed,
        "normalization_error": abs(float(np.sum(combined)) - 1.0),
        "travel_mode": str(travel_mode),
        "local_mix": float(local_mix),
        "softening": float(softening),
        "distance_power": float(distance_power),
        "sources_crossing_target": int(np.sum(crossed)),
        "source_weight_crossing_target": float(np.sum(weight[crossed])),
        "median_target_distance": float(np.median(target_distance)),
        "maximum_target_distance": float(np.max(target_distance)),
        "median_travel": float(np.median(travel)),
        "maximum_travel": float(np.max(travel)),
    }


def baryonic_route_directions(
    positions,
    weights,
    *,
    local_mix: float,
    softening: float,
    distance_power: float = 2.0,
    neighbor_weights=None,
    center=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blend global-center and baryonic-neighbor directions.

    The local vector is the softened inverse-square attraction from every
    *other* cataloged baryonic source.  Both global and local vectors are
    normalized before mixing, so ``local_mix`` changes direction rather than
    introducing an implicit amplitude parameter.
    """
    xy = np.asarray(positions, dtype=float)
    weight = np.asarray(weights, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or weight.shape != (len(xy),):
        raise ValueError("positions and weights do not match")
    if np.any(~np.isfinite(xy)) or np.any(weight < 0.0) or np.sum(weight) <= 0.0:
        raise ValueError("invalid source positions or weights")
    if not np.isfinite(local_mix) or not 0.0 <= local_mix <= 1.0:
        raise ValueError("local_mix must lie in [0, 1]")
    if not np.isfinite(softening) or softening <= 0.0:
        raise ValueError("softening must be positive")
    if not np.isfinite(distance_power) or distance_power <= 0.0:
        raise ValueError("distance_power must be positive")
    weight = weight / np.sum(weight)
    neighbor_weight = (
        weight.copy()
        if neighbor_weights is None
        else np.asarray(neighbor_weights, dtype=float)
    )
    if neighbor_weight.shape != weight.shape or np.any(~np.isfinite(neighbor_weight)):
        raise ValueError("neighbor_weights must match weights and be finite")
    if np.any(neighbor_weight < 0.0) or np.sum(neighbor_weight) <= 0.0:
        raise ValueError("neighbor_weights must be nonnegative and not all zero")
    neighbor_weight = neighbor_weight / np.sum(neighbor_weight)
    centroid = (
        np.sum(xy * weight[:, None], axis=0)
        if center is None
        else np.asarray(center, dtype=float)
    )
    global_vector = centroid[None, :] - xy
    separation = xy[None, :, :] - xy[:, None, :]
    distance2 = np.sum(np.square(separation), axis=2)
    inverse = np.power(
        distance2 + float(softening) ** 2,
        -0.5 * (float(distance_power) + 1.0),
    )
    np.fill_diagonal(inverse, 0.0)
    local_vector = np.sum(
        separation * (neighbor_weight[None, :] * inverse)[:, :, None], axis=1
    )

    tiny = np.finfo(float).tiny
    global_norm = np.linalg.norm(global_vector, axis=1)
    local_norm = np.linalg.norm(local_vector, axis=1)
    global_unit = global_vector / np.maximum(global_norm[:, None], tiny)
    local_unit = local_vector / np.maximum(local_norm[:, None], tiny)
    local_unit[local_norm <= tiny] = global_unit[local_norm <= tiny]
    global_unit[global_norm <= tiny] = local_unit[global_norm <= tiny]
    mixed = (1.0 - float(local_mix)) * global_unit + float(local_mix) * local_unit
    mixed_norm = np.linalg.norm(mixed, axis=1)
    mixed[mixed_norm <= tiny] = global_unit[mixed_norm <= tiny]
    mixed /= np.maximum(np.linalg.norm(mixed, axis=1)[:, None], tiny)
    return mixed, centroid, local_unit


def conservative_directional_route_template(
    axis,
    positions,
    weights,
    *,
    routing_fraction: float,
    return_scale: float,
    radius_exponent: float,
    reference_radius: float,
    smoothing: float,
    local_mix: float,
    softening: float,
    distance_power: float = 2.0,
    neighbor_weights=None,
    symmetric_bend_degrees: float = 0.0,
    center=None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Conservative route map with baryon-local direction and symmetric bends.

    A nonzero bend deposits equal weight along clockwise and counterclockwise
    endpoint directions.  This changes the projected arc geometry without
    inserting a preferred handedness.
    """
    if not 0.0 <= routing_fraction <= 1.0:
        raise ValueError("routing_fraction must lie in [0, 1]")
    if return_scale <= 0.0 or reference_radius <= 0.0 or smoothing <= 0.0:
        raise ValueError("route scales must be positive")
    if not np.isfinite(symmetric_bend_degrees) or not 0.0 <= symmetric_bend_degrees < 90.0:
        raise ValueError("symmetric_bend_degrees must lie in [0, 90)")
    xy = np.asarray(positions, dtype=float)
    weight = np.asarray(weights, dtype=float)
    weight = weight / np.sum(weight)
    direction, centroid, local_direction = baryonic_route_directions(
        xy,
        weight,
        local_mix=float(local_mix),
        softening=float(softening),
        distance_power=float(distance_power),
        neighbor_weights=neighbor_weights,
        center=center,
    )
    combined, audit = conservative_explicit_direction_route_template(
        axis,
        xy,
        weight,
        direction,
        routing_fraction=routing_fraction,
        return_scale=return_scale,
        radius_exponent=radius_exponent,
        reference_radius=reference_radius,
        smoothing=smoothing,
        symmetric_bend_degrees=symmetric_bend_degrees,
        center=centroid,
    )
    alignment = np.sum(direction * local_direction, axis=1)
    return combined, {
        **audit,
        "local_directions": local_direction,
        "mean_global_local_alignment": float(np.sum(weight * alignment)),
    }


def conservative_explicit_direction_route_template(
    axis,
    positions,
    weights,
    directions,
    *,
    routing_fraction: float,
    return_scale: float,
    radius_exponent: float,
    reference_radius: float,
    smoothing: float,
    symmetric_bend_degrees: float = 0.0,
    center=None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Conservative route map for externally reconstructed unit directions."""
    if not 0.0 <= routing_fraction <= 1.0:
        raise ValueError("routing_fraction must lie in [0, 1]")
    if return_scale <= 0.0 or reference_radius <= 0.0 or smoothing <= 0.0:
        raise ValueError("route scales must be positive")
    if not np.isfinite(symmetric_bend_degrees) or not 0.0 <= symmetric_bend_degrees < 90.0:
        raise ValueError("symmetric_bend_degrees must lie in [0, 90)")
    xy = np.asarray(positions, dtype=float)
    weight = np.asarray(weights, dtype=float)
    direction = np.asarray(directions, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or direction.shape != xy.shape:
        raise ValueError("positions and directions must be matching Nx2 arrays")
    if weight.shape != (len(xy),) or np.any(weight < 0.0) or np.sum(weight) <= 0.0:
        raise ValueError("weights must match positions and be nonnegative")
    if np.any(~np.isfinite(xy)) or np.any(~np.isfinite(direction)):
        raise ValueError("positions and directions must be finite")
    norm = np.linalg.norm(direction, axis=1)
    if np.any(norm <= np.finfo(float).tiny):
        raise ValueError("every explicit route direction must be nonzero")
    direction = direction / norm[:, None]
    weight = weight / np.sum(weight)
    centroid = (
        np.sum(xy * weight[:, None], axis=0)
        if center is None
        else np.asarray(center, dtype=float)
    )
    radius = np.linalg.norm(centroid[None, :] - xy, axis=1)
    scale = float(return_scale) * np.power(
        np.maximum(radius / float(reference_radius), 1.0e-6),
        float(radius_exponent),
    )
    scale = np.clip(scale, 0.2 * float(return_scale), 3.0 * float(return_scale))
    angle = np.deg2rad(float(symmetric_bend_degrees))
    if angle == 0.0:
        endpoint_direction = direction
        endpoints = xy + scale[:, None] * endpoint_direction
        endpoint_weights = weight
    else:
        cosine, sine = np.cos(angle), np.sin(angle)
        plus = np.column_stack(
            [
                cosine * direction[:, 0] - sine * direction[:, 1],
                sine * direction[:, 0] + cosine * direction[:, 1],
            ]
        )
        minus = np.column_stack(
            [
                cosine * direction[:, 0] + sine * direction[:, 1],
                -sine * direction[:, 0] + cosine * direction[:, 1],
            ]
        )
        endpoint_direction = np.vstack([plus, minus])
        endpoints = np.vstack(
            [xy + scale[:, None] * plus, xy + scale[:, None] * minus]
        )
        endpoint_weights = np.r_[0.5 * weight, 0.5 * weight]
    local = deposit_sources(axis, xy, weight, smoothing=smoothing)
    routed = deposit_sources(axis, endpoints, endpoint_weights, smoothing=smoothing)
    combined = (1.0 - float(routing_fraction)) * local + float(routing_fraction) * routed
    combined /= np.sum(combined)
    return combined, {
        "centroid": centroid,
        "endpoints": endpoints,
        "endpoint_weights": endpoint_weights,
        "route_directions": direction,
        "local_map": local,
        "routed_map": routed,
        "normalization_error": abs(float(np.sum(combined)) - 1.0),
    }


def baryonic_network_transitions(
    positions,
    weights,
    *,
    target_weight_power: float,
    distance_power: float,
    softening: float,
    link_scale: float,
    top_k: int | None = None,
) -> np.ndarray:
    """Return a row-normalized, baryon-only source-to-neighbor kernel.

    For source ``i`` and distinct baryonic target ``j`` the unnormalized link is

    ``w_j**target_weight_power * (d_ij**2+s**2)**(-distance_power/2)
    * exp(-d_ij/link_scale)``.

    This differs from :func:`baryonic_route_directions`: it retains the
    individual target branches instead of reducing them to one vector sum.
    """
    xy = np.asarray(positions, dtype=float)
    weight = np.asarray(weights, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or weight.shape != (len(xy),):
        raise ValueError("positions and weights do not match")
    if len(xy) < 2 or np.any(~np.isfinite(xy)):
        raise ValueError("at least two finite baryonic positions are required")
    if np.any(~np.isfinite(weight)) or np.any(weight < 0.0) or np.sum(weight) <= 0.0:
        raise ValueError("weights must be finite, nonnegative, and not all zero")
    if not np.isfinite(target_weight_power) or target_weight_power < 0.0:
        raise ValueError("target_weight_power must be finite and nonnegative")
    if not np.isfinite(distance_power) or distance_power < 0.0:
        raise ValueError("distance_power must be finite and nonnegative")
    if not np.isfinite(softening) or softening <= 0.0:
        raise ValueError("softening must be positive")
    if not np.isfinite(link_scale) or link_scale <= 0.0:
        raise ValueError("link_scale must be positive")
    if top_k is not None and not 1 <= int(top_k) < len(xy):
        raise ValueError("top_k must be between one and N-1")

    normalized_weight = weight / np.sum(weight)
    displacement = xy[None, :, :] - xy[:, None, :]
    distance2 = np.sum(np.square(displacement), axis=2)
    distance = np.sqrt(distance2)
    transition = (
        np.power(np.maximum(normalized_weight, np.finfo(float).tiny), float(target_weight_power))[None, :]
        * np.power(distance2 + float(softening) ** 2, -0.5 * float(distance_power))
        * np.exp(-distance / float(link_scale))
    )
    np.fill_diagonal(transition, 0.0)
    if top_k is not None:
        keep = np.zeros_like(transition, dtype=bool)
        nearest = np.argpartition(distance2, kth=int(top_k), axis=1)[:, : int(top_k) + 1]
        rows = np.arange(len(xy))[:, None]
        keep[rows, nearest] = True
        np.fill_diagonal(keep, False)
        transition[~keep] = 0.0
    row_sum = np.sum(transition, axis=1)
    if np.any(row_sum <= 0.0):
        raise RuntimeError("network kernel left a source without a destination")
    return transition / row_sum[:, None]


def conservative_network_route_template(
    axis,
    positions,
    weights,
    *,
    routing_fraction: float,
    target_weight_power: float,
    distance_power: float,
    softening: float,
    link_scale: float,
    hop_fraction: float,
    smoothing: float,
    top_k: int | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Route a conserved fraction through explicit baryon-to-baryon links."""
    if not 0.0 <= routing_fraction <= 1.0:
        raise ValueError("routing_fraction must lie in [0, 1]")
    if not np.isfinite(hop_fraction) or hop_fraction <= 0.0:
        raise ValueError("hop_fraction must be positive")
    if not np.isfinite(smoothing) or smoothing <= 0.0:
        raise ValueError("smoothing must be positive")
    xy = np.asarray(positions, dtype=float)
    weight = np.asarray(weights, dtype=float)
    weight /= np.sum(weight)
    transition = baryonic_network_transitions(
        xy,
        weight,
        target_weight_power=target_weight_power,
        distance_power=distance_power,
        softening=softening,
        link_scale=link_scale,
        top_k=top_k,
    )
    source_index, target_index = np.nonzero(transition > 0.0)
    branch_weight = weight[source_index] * transition[source_index, target_index]
    displacement = xy[target_index] - xy[source_index]
    endpoints = xy[source_index] + float(hop_fraction) * displacement
    local = deposit_sources(axis, xy, weight, smoothing=smoothing)
    routed = deposit_sources(axis, endpoints, branch_weight, smoothing=smoothing)
    combined = (1.0 - float(routing_fraction)) * local + float(routing_fraction) * routed
    combined /= np.sum(combined)

    receiving = np.sum(weight[:, None] * transition, axis=0)
    entropy = -np.sum(
        np.where(transition > 0.0, transition * np.log(np.maximum(transition, np.finfo(float).tiny)), 0.0),
        axis=1,
    )
    route_length = np.linalg.norm(displacement, axis=1) * float(hop_fraction)
    return combined, {
        "transition": transition,
        "endpoints": endpoints,
        "endpoint_weights": branch_weight,
        "local_map": local,
        "routed_map": routed,
        "normalization_error": abs(float(np.sum(combined)) - 1.0),
        "branch_count": int(len(branch_weight)),
        "mean_route_length": float(np.sum(branch_weight * route_length)),
        "rms_route_length": float(np.sqrt(np.sum(branch_weight * np.square(route_length)))),
        "mean_source_transition_entropy": float(np.sum(weight * entropy)),
        "effective_receiving_targets": float(1.0 / np.sum(np.square(receiving))),
        "largest_receiving_fraction": float(np.max(receiving)),
    }
