"""Spherical transfer form of the source-conserving routing operator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid

from voidscreen.potential_channel_qumond import (
    path_diluted_potential_channel_acceleration,
)


@dataclass(frozen=True)
class SphericalRoutingResponse:
    radius_m: np.ndarray
    base_acceleration_m_s2: np.ndarray
    local_generator_acceleration_m_s2: np.ndarray
    routed_acceleration_m_s2: np.ndarray
    baryonic_source_s2: np.ndarray
    local_extra_source_s2: np.ndarray
    positive_routed_source_s2: np.ndarray
    negative_shell_source_s2: np.ndarray
    transition_shell_weight_m_inv: np.ndarray
    local_channel_exponent: np.ndarray
    positive_generator_strength_m3_s2: float
    net_added_flux_fraction: float


def _spherical_source(radius: np.ndarray, acceleration: np.ndarray) -> np.ndarray:
    flux = 4.0 * np.pi * np.square(radius) * acceleration
    return np.gradient(flux, radius, edge_order=2) / (4.0 * np.pi * np.square(radius))


def source_conserving_spherical_response(
    radius_m,
    gbar_m_s2,
    potential_depth,
    potential_path_ratio,
    *,
    a0_m_s2: float,
    transition_depth: float,
    transition_power: float,
    extra_spatial_channels: float,
    path_power: float,
) -> SphericalRoutingResponse:
    radius, gbar, depth, path = np.broadcast_arrays(
        np.asarray(radius_m, dtype=float),
        np.asarray(gbar_m_s2, dtype=float),
        np.asarray(potential_depth, dtype=float),
        np.asarray(potential_path_ratio, dtype=float),
    )
    if (
        radius.ndim != 1
        or radius.size < 5
        or np.any(~np.isfinite(radius))
        or np.any(radius <= 0.0)
        or np.any(np.diff(radius) <= 0.0)
        or np.any(~np.isfinite(gbar))
        or np.any(gbar <= 0.0)
        or np.any(~np.isfinite(depth))
        or np.any(depth < 0.0)
        or np.any(~np.isfinite(path))
        or np.any(path <= 0.0)
    ):
        raise ValueError("spherical routing profiles must be finite, positive, and ordered")
    local = path_diluted_potential_channel_acceleration(
        gbar,
        depth,
        path,
        a0_m_s2=a0_m_s2,
        transition_depth=transition_depth,
        transition_power=transition_power,
        extra_spatial_channels=extra_spatial_channels,
        path_power=path_power,
    )
    base = path_diluted_potential_channel_acceleration(
        gbar,
        depth,
        path,
        a0_m_s2=a0_m_s2,
        transition_depth=transition_depth,
        transition_power=transition_power,
        extra_spatial_channels=0.0,
        path_power=path_power,
    )
    base_acceleration = base["predicted_acceleration_m_s2"]
    local_acceleration = local["predicted_acceleration_m_s2"]
    base_source = _spherical_source(radius, base_acceleration)
    local_source = _spherical_source(radius, local_acceleration)
    local_extra_source = local_source - base_source
    volume_weight = 4.0 * np.pi * np.square(radius)
    positive_strength = float(
        np.trapezoid(np.maximum(local_extra_source, 0.0) * volume_weight, radius)
    )
    if not np.isfinite(positive_strength) or positive_strength <= 0.0:
        raise ValueError("spherical generator has no positive source strength")

    baryonic_source = np.maximum(_spherical_source(radius, gbar), 0.0)
    baryonic_integral = float(np.trapezoid(baryonic_source * volume_weight, radius))
    if not np.isfinite(baryonic_integral) or baryonic_integral <= 0.0:
        raise ValueError("spherical baryonic source has no positive integral")
    coordinate = np.power(depth / float(transition_depth), float(transition_power))
    onset = coordinate / (1.0 + coordinate)
    depth_gradient = np.abs(np.gradient(depth, radius, edge_order=2))
    shell_weight = 4.0 * onset * (1.0 - onset) * depth_gradient
    shell_integral = float(np.trapezoid(shell_weight * volume_weight, radius))
    if not np.isfinite(shell_integral) or shell_integral <= 0.0:
        raise ValueError("spherical transition shell has no positive integral")
    positive_route = positive_strength * baryonic_source / baryonic_integral
    negative_shell = positive_strength * shell_weight / shell_integral
    added_source = positive_route - negative_shell
    added_flux = cumulative_trapezoid(
        added_source * volume_weight,
        radius,
        initial=0.0,
    )
    routed_acceleration = base_acceleration + added_flux / (4.0 * np.pi * np.square(radius))
    if np.any(~np.isfinite(routed_acceleration)) or np.any(routed_acceleration <= 0.0):
        raise ValueError("routed spherical acceleration must be finite and positive")
    net_fraction = abs(float(added_flux[-1])) / positive_strength
    return SphericalRoutingResponse(
        radius_m=radius,
        base_acceleration_m_s2=base_acceleration,
        local_generator_acceleration_m_s2=local_acceleration,
        routed_acceleration_m_s2=routed_acceleration,
        baryonic_source_s2=baryonic_source,
        local_extra_source_s2=local_extra_source,
        positive_routed_source_s2=positive_route,
        negative_shell_source_s2=negative_shell,
        transition_shell_weight_m_inv=shell_weight,
        local_channel_exponent=local["channel_exponent"],
        positive_generator_strength_m3_s2=positive_strength,
        net_added_flux_fraction=net_fraction,
    )
