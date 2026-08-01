"""Controlled Sigma/Refracted-Gravity hybrids for sensitivity experiments.

These laws are deliberately phenomenological weak-field diagnostics.  They
share one high-acceleration screen so the same equation can be evaluated in
galaxies, clusters, and the Solar System without silently changing constants.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from functools import lru_cache

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply

from .sigma_refracted import sigma_h


G_SI = 6.67430e-11
M_SUN_KG = 1.988409870698051e30
AU_M = 149_597_870_700.0
KPC_M = 3.085677581491367e19
JULIAN_YEAR_DAYS = 365.25
RAD_TO_MAS = 206_264_806.24709636


def _positive(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must be finite and positive")
    return array


def _logistic(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.exp(-np.logaddexp(0.0, -values))


def _variant_number(variant: Mapping[str, object], name: str, default: float) -> float:
    value = float(variant.get(name, default))
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _smooth_cap(
    values: np.ndarray,
    ceiling: object | None,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a differentiable positive ceiling and return its retained fraction."""

    if ceiling is None:
        return values, np.ones_like(values)
    ceiling_value = float(ceiling)
    if not math.isfinite(ceiling_value) or ceiling_value <= 0.0:
        raise ValueError(f"{name} must be positive when supplied")
    capped = ceiling_value * np.tanh(values / ceiling_value)
    fraction = np.divide(
        capped,
        values,
        out=np.ones_like(values),
        where=values > 0.0,
    )
    return capped, fraction


def tidal_shape_property(tidal_eigenvalues_s2, property_name: str) -> np.ndarray:
    """Return one dimensionless invariant of three tidal eigenvalues."""

    eigenvalues = np.asarray(tidal_eigenvalues_s2, dtype=np.float64)
    if (
        eigenvalues.ndim < 1
        or eigenvalues.shape[-1] != 3
        or np.any(~np.isfinite(eigenvalues))
    ):
        raise ValueError("tidal eigenvalues must be finite with final dimension three")
    absolute = np.abs(eigenvalues)
    l1 = np.sum(absolute, axis=-1)
    safe_l1 = np.maximum(l1, np.finfo(float).tiny)
    l2 = np.linalg.norm(eigenvalues, axis=-1)
    safe_l2 = np.maximum(l2, np.finfo(float).tiny)
    sorted_absolute = np.sort(absolute, axis=-1)
    safe_maximum = np.maximum(
        sorted_absolute[..., 2], np.finfo(float).tiny
    )
    if property_name == "tidal_l1_dominance":
        values = sorted_absolute[..., 2] / safe_l1
    elif property_name == "tidal_middle_to_max":
        values = sorted_absolute[..., 1] / safe_maximum
    elif property_name == "tidal_minimum_to_max":
        values = sorted_absolute[..., 0] / safe_maximum
    elif property_name == "tidal_third_axis_abs_fraction":
        values = absolute[..., 2] / safe_l1
    elif property_name == "tidal_traceless_fraction":
        traceless = eigenvalues - np.mean(
            eigenvalues, axis=-1, keepdims=True
        )
        values = np.linalg.norm(traceless, axis=-1) / safe_l2
    elif property_name == "tidal_trace_fraction":
        values = np.abs(np.sum(eigenvalues, axis=-1)) / (
            math.sqrt(3.0) * safe_l2
        )
    elif property_name == "tidal_positive_fraction":
        values = np.sum(np.maximum(eigenvalues, 0.0), axis=-1) / safe_l1
    elif property_name == "tidal_signed_determinant_shape":
        values = np.prod(eigenvalues, axis=-1) / np.power(safe_l2, 3.0)
    elif property_name == "tidal_radial_abs_fraction":
        values = absolute[..., 0] / safe_l1
    else:
        raise ValueError(f"unknown tidal shape property {property_name}")
    return values


def _spherical_tidal_eigenvalues(
    gbar: np.ndarray,
    density_g_cm3: np.ndarray,
    radius_kpc: np.ndarray,
) -> np.ndarray:
    radius_m = radius_kpc * KPC_M
    tangential = gbar / radius_m
    poisson_trace = 4.0 * math.pi * G_SI * density_g_cm3 * 1000.0
    radial = poisson_trace - 2.0 * tangential
    return np.stack([radial, tangential, tangential], axis=-1)


def _channel_gate(
    gbar: np.ndarray,
    density_g_cm3: np.ndarray,
    radius_kpc: np.ndarray,
    variant: Mapping[str, object],
    property_values=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cluster-like weight and the gate's formula-facing coordinate."""

    property_name = variant.get("channel_gate_property")
    if property_name is None:
        return np.zeros_like(gbar), np.full_like(gbar, np.nan)
    radius_m = radius_kpc * KPC_M
    if property_name == "equivalent_mass":
        property_value = (
            gbar * np.square(radius_m) / (G_SI * M_SUN_KG)
        )
    elif property_name == "local_to_mean_density":
        mean_density_g_cm3 = (
            3.0 * gbar / (4.0 * math.pi * G_SI * radius_m) * 1.0e-3
        )
        property_value = density_g_cm3 / mean_density_g_cm3
    elif property_name == "radius":
        property_value = radius_kpc
    elif property_name == "tidal_curvature":
        property_value = gbar / radius_m
    elif str(property_name).startswith("tidal_"):
        if property_values is None:
            eigenvalues = _spherical_tidal_eigenvalues(
                gbar, density_g_cm3, radius_kpc
            )
            property_value = tidal_shape_property(
                eigenvalues, str(property_name)
            )
        else:
            property_value = np.broadcast_to(
                np.asarray(property_values, dtype=np.float64), gbar.shape
            )
    else:
        raise ValueError(f"unknown channel_gate_property {property_name}")
    direct_tidal_coordinate = (
        str(property_name).startswith("tidal_")
        and property_name != "tidal_curvature"
    )
    if np.any(~np.isfinite(property_value)):
        raise ValueError("channel gate property must be finite")
    if not direct_tidal_coordinate and np.any(property_value <= 0.0):
        raise ValueError("logarithmic channel gate property must be positive")
    if direct_tidal_coordinate:
        coordinate = property_value
        pivot = _variant_number(variant, "channel_gate_pivot", 0.5)
    else:
        coordinate = np.log10(property_value)
        pivot = _variant_number(variant, "channel_gate_log10_pivot", 0.0)
    sharpness = _variant_number(variant, "channel_gate_sharpness", 1.0)
    if sharpness <= 0.0:
        raise ValueError("channel_gate_sharpness must be positive")
    topology = str(variant.get("channel_gate_topology", "monotonic"))
    if topology == "monotonic":
        orientation = (
            1.0
            if bool(variant.get("channel_gate_cluster_high", True))
            else -1.0
        )
        cluster_weight = _logistic(
            orientation * sharpness * (coordinate - pivot)
        )
    elif topology in {"band", "tails"}:
        lower = _variant_number(variant, "channel_gate_lower_pivot", 0.25)
        upper = _variant_number(variant, "channel_gate_upper_pivot", 0.75)
        if not lower < upper:
            raise ValueError(
                "channel gate lower pivot must be below upper pivot"
            )
        middle_weight = _logistic(
            sharpness * (coordinate - lower)
        ) * _logistic(sharpness * (upper - coordinate))
        cluster_weight = (
            middle_weight if topology == "band" else 1.0 - middle_weight
        )
    elif topology == "constant":
        constant = _variant_number(
            variant, "channel_gate_constant_weight", 0.0
        )
        if not 0.0 <= constant <= 1.0:
            raise ValueError(
                "channel_gate_constant_weight must lie in [0, 1]"
            )
        cluster_weight = np.full_like(coordinate, constant)
    else:
        raise ValueError(f"unknown channel_gate_topology {topology}")
    return cluster_weight, coordinate


def screened_hybrid_response(
    gbar_m_s2,
    density_g_cm3,
    radius_kpc,
    parameters,
    variant: Mapping[str, object],
    *,
    g_reference_m_s2: float = 1.0e-10,
    g_dagger_m_s2: float = 9.6e-11,
    acceleration_screen_m_s2: float = 1.2e-10,
    channel_gate_property_values=None,
) -> dict[str, np.ndarray]:
    """Return one controlled hybrid enhancement and its intermediate terms.

    ``parameters`` has the universal order
    ``(epsilon_0, log10_rho_c_g_cm3, Q, B)``.

    The common form is

    ``g/g_N = 1 + W_a * F(R_eff, S_eff)``,

    where ``R=epsilon_RG^-1-1``, ``S=B h(g_N)``, and
    ``W_a=[a_s/(a_s+g_N)]^n``.  Optional channel-specific smooth ceilings
    create ``R_eff`` and ``S_eff`` before they are combined.  A separate
    common ceiling can still be applied after combination.
    """

    gbar = _positive(gbar_m_s2, "gbar_m_s2")
    density = _positive(density_g_cm3, "density_g_cm3")
    radius = _positive(radius_kpc, "radius_kpc")
    gbar, density, radius = np.broadcast_arrays(gbar, density, radius)
    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != (4,) or np.any(~np.isfinite(values)):
        raise ValueError("parameters must be four finite values")
    epsilon_0, log10_rho_c, sharpness, sigma_amplitude = map(float, values)
    if not 0.0 < epsilon_0 <= 1.0:
        raise ValueError("epsilon_0 must be in (0, 1]")
    if sharpness <= 0.0 or sigma_amplitude < 0.0:
        raise ValueError("Q must be positive and B nonnegative")
    if min(g_reference_m_s2, g_dagger_m_s2, acceleration_screen_m_s2) <= 0.0:
        raise ValueError("acceleration scales must be positive")

    alpha_g = _variant_number(variant, "threshold_acceleration_power", 0.0)
    critical_density = (
        np.power(10.0, log10_rho_c)
        * np.power(gbar / g_reference_m_s2, alpha_g)
    )
    log_odds = 2.0 * sharpness * np.log(density / critical_density)
    transition = _logistic(log_odds)
    epsilon = epsilon_0 + (1.0 - epsilon_0) * transition
    rg_excess = 1.0 / epsilon - 1.0
    sigma_excess = sigma_amplitude * sigma_h(
        gbar, g_dagger_m_s2=g_dagger_m_s2
    )
    capped_rg_excess, _ = _smooth_cap(
        rg_excess,
        variant.get("rg_saturation_ceiling"),
        "rg_saturation_ceiling",
    )
    capped_sigma_excess, _ = _smooth_cap(
        sigma_excess,
        variant.get("sigma_saturation_ceiling"),
        "sigma_saturation_ceiling",
    )
    channel_gate, channel_gate_property_coordinate = _channel_gate(
        gbar,
        density,
        radius,
        variant,
        property_values=channel_gate_property_values,
    )
    if variant.get("channel_gate_property") is None:
        effective_rg_excess = capped_rg_excess
        effective_sigma_excess = capped_sigma_excess
        rg_cap_weight = np.ones_like(channel_gate)
        sigma_cap_weight = np.ones_like(channel_gate)
    else:
        rg_cap_weight = (
            (
                channel_gate
                if bool(variant.get("rg_cap_cluster_weight", True))
                else 1.0 - channel_gate
            )
            if bool(variant.get("rg_cap_gate_enabled", True))
            else np.ones_like(channel_gate)
        )
        sigma_cap_weight = (
            (
                channel_gate
                if bool(variant.get("sigma_cap_cluster_weight", False))
                else 1.0 - channel_gate
            )
            if bool(variant.get("sigma_cap_gate_enabled", True))
            else np.ones_like(channel_gate)
        )
        effective_rg_excess = (
            rg_excess
            + rg_cap_weight * (capped_rg_excess - rg_excess)
        )
        effective_sigma_excess = (
            sigma_excess
            + sigma_cap_weight * (capped_sigma_excess - sigma_excess)
        )
    rg_saturation_fraction = np.divide(
        effective_rg_excess,
        rg_excess,
        out=np.ones_like(rg_excess),
        where=rg_excess > 0.0,
    )
    sigma_saturation_fraction = np.divide(
        effective_sigma_excess,
        sigma_excess,
        out=np.ones_like(sigma_excess),
        where=sigma_excess > 0.0,
    )

    combination = str(variant.get("combination", "interaction"))
    if combination == "interaction":
        interaction = _variant_number(variant, "interaction_eta", 0.0)
        raw_excess = (
            effective_rg_excess
            + effective_sigma_excess
            + interaction * effective_rg_excess * effective_sigma_excess
        )
    elif combination == "power_mean":
        power = _variant_number(variant, "combination_power", 1.0)
        if power <= 0.0:
            raise ValueError("combination_power must be positive")
        raw_excess = np.power(
            np.power(effective_rg_excess, power)
            + np.power(effective_sigma_excess, power),
            1.0 / power,
        )
    else:
        raise ValueError(f"unknown hybrid combination {combination}")

    combined_excess, saturation_fraction = _smooth_cap(
        raw_excess,
        variant.get("saturation_ceiling"),
        "saturation_ceiling",
    )

    screen_power = _variant_number(variant, "screen_power", 1.0)
    screen_scale_multiplier = _variant_number(
        variant, "screen_scale_multiplier", 1.0
    )
    if screen_power <= 0.0 or screen_scale_multiplier <= 0.0:
        raise ValueError("screen power and scale multiplier must be positive")
    effective_screen_scale = (
        acceleration_screen_m_s2 * screen_scale_multiplier
    )
    acceleration_screen = np.power(
        effective_screen_scale / (effective_screen_scale + gbar),
        screen_power,
    )
    excess = acceleration_screen * combined_excess
    enhancement = 1.0 + excess
    if np.any(~np.isfinite(enhancement)) or np.any(enhancement < 1.0):
        raise ValueError("hybrid enhancement must be finite and at least one")
    return {
        "enhancement": enhancement,
        "fractional_excess": excess,
        "gbar_m_s2": gbar,
        "radius_kpc": radius,
        "g_reference_m_s2": np.full_like(gbar, g_reference_m_s2),
        "rg_excess": rg_excess,
        "sigma_excess": sigma_excess,
        "effective_rg_excess": effective_rg_excess,
        "effective_sigma_excess": effective_sigma_excess,
        "rg_saturation_fraction": rg_saturation_fraction,
        "sigma_saturation_fraction": sigma_saturation_fraction,
        "channel_gate_cluster_weight": channel_gate,
        "channel_gate_property_coordinate": channel_gate_property_coordinate,
        "rg_channel_cap_weight": rg_cap_weight,
        "sigma_channel_cap_weight": sigma_cap_weight,
        "raw_combined_excess": raw_excess,
        "combined_excess": combined_excess,
        "saturation_fraction": saturation_fraction,
        "acceleration_screen": acceleration_screen,
        "acceleration_screen_scale_m_s2": np.full_like(
            acceleration_screen, effective_screen_scale
        ),
        "critical_density_g_cm3": critical_density,
        "permittivity": epsilon,
    }


def radial_memory_blend(
    radius_kpc,
    local_excess,
    *,
    strength: float,
    log_scale: float,
    outer_to_inner: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend a local excess with a causal exponential memory in log radius.

    The running memory obeys

    ``M_i = exp(-Delta ln(r)/ell) M_(i-1) + (1-exp(...)) F_i``.

    ``outer_to_inner=False`` carries smaller-radius information outward.
    Reversing it tests a spatial exterior-pressure interpretation.  A
    one-point profile and ``strength=0`` both reduce exactly to the local law.
    """

    radius = np.asarray(radius_kpc, dtype=np.float64)
    excess = np.asarray(local_excess, dtype=np.float64)
    if radius.ndim != 1 or excess.ndim != 1 or radius.shape != excess.shape:
        raise ValueError("radial memory requires matching one-dimensional arrays")
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("radial memory radii must be finite and positive")
    if np.any(~np.isfinite(excess)) or np.any(excess < 0.0):
        raise ValueError("radial memory excess must be finite and nonnegative")
    strength_value = float(strength)
    scale_value = float(log_scale)
    if not np.isfinite(strength_value) or not 0.0 <= strength_value <= 1.0:
        raise ValueError("radial memory strength must lie in [0, 1]")
    if not np.isfinite(scale_value) or scale_value <= 0.0:
        raise ValueError("radial memory log scale must be positive")
    if len(radius) <= 1 or strength_value == 0.0:
        return excess.copy(), excess.copy()

    order = np.argsort(radius, kind="stable")
    if outer_to_inner:
        order = order[::-1]
    ordered_radius = radius[order]
    if np.any(np.diff(np.log(ordered_radius)) == 0.0):
        raise ValueError("radial memory profile radii must be unique")
    ordered_excess = excess[order]
    memory = np.empty_like(ordered_excess)
    memory[0] = ordered_excess[0]
    for index in range(1, len(memory)):
        delta = abs(
            math.log(ordered_radius[index] / ordered_radius[index - 1])
        )
        retained = math.exp(-delta / scale_value)
        memory[index] = (
            retained * memory[index - 1]
            + (1.0 - retained) * ordered_excess[index]
        )
    blended = (
        (1.0 - strength_value) * ordered_excess
        + strength_value * memory
    )
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return blended[inverse], memory[inverse]


def apply_channel_gate_memory_to_response(
    local: Mapping[str, np.ndarray],
    radius_kpc,
    variant: Mapping[str, object],
) -> dict[str, np.ndarray]:
    """Carry a bounded tidal/channel classification across log radius.

    This is distinct from force memory.  The local channel gate first classifies
    the baryonic geometry, then an exponential running average lets that
    classification persist across neighboring radii.  The remembered gate is
    used only to place the pre-existing RG and Sigma ceilings; it does not add a
    new force amplitude.
    """

    radius = np.asarray(radius_kpc, dtype=np.float64)
    local_weight = np.asarray(
        local["channel_gate_cluster_weight"], dtype=np.float64
    )
    if (
        radius.ndim != 1
        or local_weight.ndim != 1
        or radius.shape != local_weight.shape
    ):
        raise ValueError(
            "channel-gate memory requires matching one-dimensional arrays"
        )
    strength = _variant_number(variant, "channel_gate_memory_strength", 0.0)
    scale = _variant_number(variant, "channel_gate_memory_log_scale", 1.0)
    outer_to_inner = bool(
        variant.get("channel_gate_memory_outer_to_inner", False)
    )
    if not 0.0 <= strength <= 1.0:
        raise ValueError("channel_gate_memory_strength must lie in [0, 1]")
    if scale <= 0.0:
        raise ValueError("channel_gate_memory_log_scale must be positive")
    if strength > 0.0 and variant.get("channel_gate_property") is None:
        raise ValueError("channel-gate memory requires channel_gate_property")
    if np.any(~np.isfinite(local_weight)) or np.any(
        (local_weight < 0.0) | (local_weight > 1.0)
    ):
        raise ValueError("local channel-gate weights must lie in [0, 1]")

    remembered_weight, memory = radial_memory_blend(
        radius,
        local_weight,
        strength=strength,
        log_scale=scale,
        outer_to_inner=outer_to_inner,
    )
    remembered_weight = np.clip(remembered_weight, 0.0, 1.0)

    rg_excess = np.asarray(local["rg_excess"], dtype=np.float64)
    sigma_excess = np.asarray(local["sigma_excess"], dtype=np.float64)
    capped_rg, _ = _smooth_cap(
        rg_excess,
        variant.get("rg_saturation_ceiling"),
        "rg_saturation_ceiling",
    )
    capped_sigma, _ = _smooth_cap(
        sigma_excess,
        variant.get("sigma_saturation_ceiling"),
        "sigma_saturation_ceiling",
    )
    if variant.get("channel_gate_property") is None:
        rg_cap_weight = np.ones_like(remembered_weight)
        sigma_cap_weight = np.ones_like(remembered_weight)
    else:
        rg_cap_weight = (
            (
                remembered_weight
                if bool(variant.get("rg_cap_cluster_weight", True))
                else 1.0 - remembered_weight
            )
            if bool(variant.get("rg_cap_gate_enabled", True))
            else np.ones_like(remembered_weight)
        )
        sigma_cap_weight = (
            (
                remembered_weight
                if bool(variant.get("sigma_cap_cluster_weight", False))
                else 1.0 - remembered_weight
            )
            if bool(variant.get("sigma_cap_gate_enabled", True))
            else np.ones_like(remembered_weight)
        )
    effective_rg = rg_excess + rg_cap_weight * (capped_rg - rg_excess)
    effective_sigma = sigma_excess + sigma_cap_weight * (
        capped_sigma - sigma_excess
    )
    combined, raw_combined = _combine_memory_channels(
        effective_rg, effective_sigma, variant
    )
    screen = np.asarray(local["acceleration_screen"], dtype=np.float64)
    fractional = screen * combined
    enhancement = 1.0 + fractional
    if np.any(~np.isfinite(enhancement)) or np.any(enhancement < 1.0):
        raise ValueError(
            "channel-gate memory enhancement must be finite and at least one"
        )

    output = dict(local)
    output["local_without_channel_gate_memory_fractional_excess"] = np.asarray(
        local["fractional_excess"], dtype=np.float64
    )
    output["local_channel_gate_cluster_weight"] = local_weight
    output["channel_gate_memory_average"] = memory
    output["channel_gate_cluster_weight"] = remembered_weight
    output["channel_gate_memory_strength"] = np.full_like(
        enhancement, strength
    )
    output["channel_gate_memory_log_scale"] = np.full_like(
        enhancement, scale
    )
    output["channel_gate_memory_outer_to_inner"] = np.full_like(
        enhancement, float(outer_to_inner)
    )
    output["rg_channel_cap_weight"] = rg_cap_weight
    output["sigma_channel_cap_weight"] = sigma_cap_weight
    output["effective_rg_excess"] = effective_rg
    output["effective_sigma_excess"] = effective_sigma
    output["rg_saturation_fraction"] = np.divide(
        effective_rg,
        rg_excess,
        out=np.ones_like(rg_excess),
        where=rg_excess > 0.0,
    )
    output["sigma_saturation_fraction"] = np.divide(
        effective_sigma,
        sigma_excess,
        out=np.ones_like(sigma_excess),
        where=sigma_excess > 0.0,
    )
    output["raw_combined_excess"] = raw_combined
    output["combined_excess"] = combined
    output["fractional_excess"] = fractional
    output["enhancement"] = enhancement
    return output


def _combine_memory_channels(
    rg_excess: np.ndarray,
    sigma_excess: np.ndarray,
    variant: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    """Recombine profile-modified channels with the local formula's closure."""

    combination = str(variant.get("combination", "interaction"))
    if combination == "interaction":
        interaction = _variant_number(variant, "interaction_eta", 0.0)
        raw_excess = (
            rg_excess
            + sigma_excess
            + interaction * rg_excess * sigma_excess
        )
    elif combination == "power_mean":
        power = _variant_number(variant, "combination_power", 1.0)
        if power <= 0.0:
            raise ValueError("combination_power must be positive")
        raw_excess = np.power(
            np.power(rg_excess, power) + np.power(sigma_excess, power),
            1.0 / power,
        )
    else:
        raise ValueError(f"unknown hybrid combination {combination}")
    return _smooth_cap(
        raw_excess,
        variant.get("saturation_ceiling"),
        "saturation_ceiling",
    )[0], raw_excess


def _profile_log_slope(radius_kpc, positive_profile) -> np.ndarray:
    """Return d ln(profile) / d ln(radius) in the input row order."""

    radius = _positive(radius_kpc, "profile slope radius_kpc")
    profile = _positive(positive_profile, "profile slope values")
    if radius.ndim != 1 or profile.ndim != 1 or radius.shape != profile.shape:
        raise ValueError("profile slope requires matching one-dimensional arrays")
    if len(radius) <= 1:
        return np.zeros_like(radius)
    order = np.argsort(radius, kind="stable")
    ordered_log_radius = np.log(radius[order])
    if np.any(np.diff(ordered_log_radius) == 0.0):
        raise ValueError("profile slope radii must be unique")
    ordered_log_profile = np.log(profile[order])
    ordered_slope = np.gradient(
        ordered_log_profile,
        ordered_log_radius,
        edge_order=1,
    )
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return ordered_slope[inverse]


def _profile_log_linear_slope(radius_kpc, positive_profile) -> float:
    """Return one least-squares log slope for an entire radial profile."""

    radius = _positive(radius_kpc, "profile slope radius_kpc")
    profile = _positive(positive_profile, "profile slope values")
    if radius.ndim != 1 or profile.ndim != 1 or radius.shape != profile.shape:
        raise ValueError("profile slope requires matching one-dimensional arrays")
    if len(radius) <= 1:
        return 0.0
    order = np.argsort(radius, kind="stable")
    ordered_log_radius = np.log(radius[order])
    if np.any(np.diff(ordered_log_radius) == 0.0):
        raise ValueError("profile slope radii must be unique")
    ordered_log_profile = np.log(profile[order])
    centered_radius = ordered_log_radius - np.mean(ordered_log_radius)
    denominator = float(np.dot(centered_radius, centered_radius))
    if denominator == 0.0:
        raise ValueError("profile slope radii must span a nonzero range")
    centered_profile = ordered_log_profile - np.mean(ordered_log_profile)
    return float(np.dot(centered_radius, centered_profile) / denominator)


def _profile_smoothed_log_slope(
    radius_kpc,
    positive_profile,
    *,
    log_scale: float,
) -> np.ndarray:
    """Return Gaussian-local linear slopes in natural-log radius.

    ``log_scale`` is a universal bandwidth in ``ln(radius)``.  The local
    regression estimates a slope without differentiating a point-dependent
    exponent inside the force law.  If a bandwidth is too narrow to support a
    local regression at one row, that row falls back to the ordinary finite-
    difference profile slope.
    """

    radius = _positive(radius_kpc, "smoothed profile slope radius_kpc")
    profile = _positive(positive_profile, "smoothed profile slope values")
    if radius.ndim != 1 or profile.ndim != 1 or radius.shape != profile.shape:
        raise ValueError(
            "smoothed profile slope requires matching one-dimensional arrays"
        )
    scale = float(log_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("profile slope smoothing log scale must be positive")
    if len(radius) <= 1:
        return np.zeros_like(radius)

    order = np.argsort(radius, kind="stable")
    ordered_x = np.log(radius[order])
    if np.any(np.diff(ordered_x) == 0.0):
        raise ValueError("smoothed profile slope radii must be unique")
    ordered_y = np.log(profile[order])
    fallback = np.gradient(ordered_y, ordered_x, edge_order=1)
    slopes = np.empty_like(ordered_x)
    span_squared = float(np.square(np.ptp(ordered_x)))
    denominator_floor = np.finfo(float).eps * max(span_squared, 1.0)
    for index, center in enumerate(ordered_x):
        normalized_distance = (ordered_x - center) / scale
        with np.errstate(under="ignore"):
            weight = np.exp(-0.5 * np.square(normalized_distance))
        total_weight = float(np.sum(weight))
        weighted_x = float(np.dot(weight, ordered_x) / total_weight)
        weighted_y = float(np.dot(weight, ordered_y) / total_weight)
        centered_x = ordered_x - weighted_x
        denominator = float(np.dot(weight, np.square(centered_x)))
        if denominator <= denominator_floor:
            slopes[index] = fallback[index]
        else:
            slopes[index] = float(
                np.dot(weight, centered_x * (ordered_y - weighted_y))
                / denominator
            )
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return slopes[inverse]


def _log_radius_cell_widths(ordered_log_radius: np.ndarray) -> np.ndarray:
    """Finite-volume cell widths for a strictly ordered log-radius grid."""

    count = len(ordered_log_radius)
    if count == 1:
        return np.ones(1, dtype=np.float64)
    gaps = np.diff(ordered_log_radius)
    if np.any(~np.isfinite(gaps)) or np.any(gaps <= 0.0):
        raise ValueError("profile-diffusion radii must be finite and unique")
    boundaries = np.empty(count + 1, dtype=np.float64)
    boundaries[1:-1] = 0.5 * (
        ordered_log_radius[:-1] + ordered_log_radius[1:]
    )
    boundaries[0] = ordered_log_radius[0] - 0.5 * gaps[0]
    boundaries[-1] = ordered_log_radius[-1] + 0.5 * gaps[-1]
    return np.diff(boundaries)


def _no_flux_log_radius_generator(
    ordered_log_radius: np.ndarray,
) -> tuple[csr_matrix, np.ndarray]:
    """Return the conservative finite-volume heat generator and cell widths."""

    count = len(ordered_log_radius)
    widths = _log_radius_cell_widths(ordered_log_radius)
    if count == 1:
        return csr_matrix((1, 1), dtype=np.float64), widths
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    diagonal = np.zeros(count, dtype=np.float64)
    for left, gap in enumerate(np.diff(ordered_log_radius)):
        right = left + 1
        left_rate = 1.0 / (widths[left] * gap)
        right_rate = 1.0 / (widths[right] * gap)
        rows.extend((left, right))
        columns.extend((right, left))
        values.extend((left_rate, right_rate))
        diagonal[left] -= left_rate
        diagonal[right] -= right_rate
    rows.extend(range(count))
    columns.extend(range(count))
    values.extend(diagonal.tolist())
    generator = csr_matrix(
        (values, (rows, columns)), shape=(count, count), dtype=np.float64
    )
    return generator, widths


@lru_cache(maxsize=4096)
def _dense_log_radius_diffusion_matrix(
    ordered_log_radius: tuple[float, ...],
    log_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Cache small profile propagators used repeatedly by bridge fitting."""

    coordinate = np.asarray(ordered_log_radius, dtype=np.float64)
    generator, widths = _no_flux_log_radius_generator(coordinate)
    diffusion_time = 0.5 * log_scale**2
    propagator = expm(diffusion_time * generator.toarray())
    return propagator, widths


def no_flux_log_radius_diffusion(
    radius_kpc,
    source,
    *,
    log_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Diffuse a positive radial carrier without changing its log-radius integral.

    This is the finite-volume solution of ``dX/dtau=d2X/d(ln r)^2`` with
    zero flux at both profile boundaries.  The returned cell widths make the
    conserved quadrature measure explicit.
    """

    radius = _positive(radius_kpc, "profile-diffusion radius_kpc")
    values = np.asarray(source, dtype=np.float64)
    if (
        radius.ndim != 1
        or values.ndim != 1
        or radius.shape != values.shape
        or np.any(~np.isfinite(values))
        or np.any(values < 0.0)
    ):
        raise ValueError(
            "profile-diffusion source must be finite, nonnegative, and match radius"
        )
    if not math.isfinite(log_scale) or log_scale <= 0.0:
        raise ValueError("radial_diffusion_log_scale must be positive")
    if len(radius) == 1:
        return values.copy(), np.ones_like(values)
    order = np.argsort(radius)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    ordered_log_radius = np.log(radius[order])
    ordered_source = values[order]
    if len(radius) <= 128:
        propagator, widths = _dense_log_radius_diffusion_matrix(
            tuple(ordered_log_radius.tolist()), float(log_scale)
        )
        diffused = propagator @ ordered_source
    else:
        generator, widths = _no_flux_log_radius_generator(ordered_log_radius)
        operator = 0.5 * log_scale**2 * generator
        diffused = expm_multiply(
            operator,
            ordered_source,
            traceA=float(operator.diagonal().sum()),
        )
    negative_tolerance = 1.0e-12 * max(
        1.0, float(np.max(np.abs(ordered_source)))
    )
    if np.any(~np.isfinite(diffused)) or np.min(diffused) < -negative_tolerance:
        raise ValueError("profile diffusion produced a nonphysical carrier")
    diffused = np.maximum(diffused, 0.0)
    integral_before = float(np.dot(widths, ordered_source))
    integral_after = float(np.dot(widths, diffused))
    if integral_before > 0.0:
        if integral_after <= 0.0:
            raise ValueError("profile diffusion lost a positive carrier integral")
        diffused *= integral_before / integral_after
    elif integral_after != 0.0:
        raise ValueError("profile diffusion created carrier from a zero source")
    return diffused[inverse], widths[inverse]


def apply_radial_diffusion_to_response(
    response: Mapping[str, np.ndarray],
    radius_kpc,
    variant: Mapping[str, object],
) -> dict[str, np.ndarray]:
    """Symmetrically redistribute a selected response carrier in log radius."""

    radius = _positive(radius_kpc, "profile-diffusion radius_kpc")
    gbar = _positive(response["gbar_m_s2"], "profile-diffusion gbar_m_s2")
    fractional = np.asarray(response["fractional_excess"], dtype=np.float64)
    if (
        radius.ndim != 1
        or gbar.ndim != 1
        or fractional.ndim != 1
        or radius.shape != gbar.shape
        or radius.shape != fractional.shape
        or np.any(~np.isfinite(fractional))
        or np.any(fractional < 0.0)
    ):
        raise ValueError("profile diffusion requires matching nonnegative profiles")
    strength = _variant_number(variant, "radial_diffusion_strength", 0.0)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("radial_diffusion_strength must lie in [0, 1]")
    if strength == 0.0:
        output = dict(response)
        output["pre_diffusion_fractional_excess"] = fractional
        output["radial_diffusion_local_source"] = fractional
        output["radial_diffusion_smoothed_source"] = fractional.copy()
        output["radial_diffusion_blended_source"] = fractional.copy()
        output["radial_diffusion_source_factor"] = np.ones_like(fractional)
        output["radial_diffusion_log_radius_cell_width"] = np.ones_like(
            fractional
        )
        output["radial_diffusion_strength"] = np.zeros_like(fractional)
        output["radial_diffusion_log_scale"] = np.full_like(fractional, math.nan)
        output["radial_diffusion_gbar_power"] = np.full_like(fractional, math.nan)
        output["radial_diffusion_radius_power"] = np.full_like(
            fractional, math.nan
        )
        return output
    log_scale = _variant_number(variant, "radial_diffusion_log_scale", 0.35)
    gbar_power = _variant_number(variant, "radial_diffusion_gbar_power", 0.0)
    radius_power = _variant_number(variant, "radial_diffusion_radius_power", 0.0)
    if log_scale <= 0.0:
        raise ValueError("radial_diffusion_log_scale must be positive")
    reference = _positive(
        response["g_reference_m_s2"], "profile-diffusion g_reference_m_s2"
    )
    with np.errstate(over="raise", under="ignore", invalid="raise"):
        source_factor = np.power(gbar / reference, gbar_power) * np.power(
            radius, radius_power
        )
    if np.any(~np.isfinite(source_factor)) or np.any(source_factor <= 0.0):
        raise ValueError("profile-diffusion source factor must be finite and positive")
    local_source = fractional * source_factor
    diffused_source, cell_width = no_flux_log_radius_diffusion(
        radius,
        local_source,
        log_scale=log_scale,
    )
    blended_source = (1.0 - strength) * local_source + strength * diffused_source
    effective_fractional = blended_source / source_factor
    enhancement = 1.0 + effective_fractional
    if np.any(~np.isfinite(enhancement)) or np.any(enhancement < 1.0):
        raise ValueError("profile-diffusion enhancement must be finite and at least one")
    output = dict(response)
    output["pre_diffusion_fractional_excess"] = fractional
    output["radial_diffusion_local_source"] = local_source
    output["radial_diffusion_smoothed_source"] = diffused_source
    output["radial_diffusion_blended_source"] = blended_source
    output["radial_diffusion_source_factor"] = source_factor
    output["radial_diffusion_log_radius_cell_width"] = cell_width
    output["radial_diffusion_strength"] = np.full_like(enhancement, strength)
    output["radial_diffusion_log_scale"] = np.full_like(enhancement, log_scale)
    output["radial_diffusion_gbar_power"] = np.full_like(enhancement, gbar_power)
    output["radial_diffusion_radius_power"] = np.full_like(
        enhancement, radius_power
    )
    output["fractional_excess"] = effective_fractional
    output["enhancement"] = enhancement
    return output


def apply_radial_memory_to_response(
    local: Mapping[str, np.ndarray],
    radius_kpc,
    variant: Mapping[str, object],
) -> dict[str, np.ndarray]:
    """Apply profile memory to a selected physical quantity and force channel.

    The transported source is

    ``X = F (g_N/g_ref)^p (r/1 kpc)^q``.

    ``p=q=0`` remembers fractional enhancement, ``p=1,q=0`` remembers
    additional acceleration, and ``p=1,q=1`` remembers the contribution to
    circular speed squared.  ``radial_memory_channel_code`` selects combined
    excess (0), RG only (1), Sigma only (2), or both channels independently
    (3).  An optional label-free slope gate can interpolate either the powers
    or two completed memory responses.  Mode 0 uses the local pointwise slope
    and interpolates powers; mode 1 uses one log-linear slope for the profile
    and interpolates powers; mode 2 uses the profile slope and blends completed
    responses; mode 3 uses the local finite-difference slope and blends
    completed responses; and mode 4 uses a universally smoothed local slope
    and blends completed responses.
    """

    radius = np.asarray(radius_kpc, dtype=np.float64)
    gbar = np.asarray(local["gbar_m_s2"], dtype=np.float64)
    if radius.ndim != 1 or gbar.ndim != 1 or radius.shape != gbar.shape:
        raise ValueError("profile response requires matching one-dimensional arrays")
    strength = _variant_number(variant, "radial_memory_strength", 0.0)
    scale = _variant_number(variant, "radial_memory_log_scale", 1.0)
    gbar_power = _variant_number(variant, "radial_memory_gbar_power", 0.0)
    radius_power = _variant_number(variant, "radial_memory_radius_power", 0.0)
    slope_gate_strength = _variant_number(
        variant, "radial_memory_slope_gate_strength", 0.0
    )
    if not 0.0 <= strength <= 1.0:
        raise ValueError("radial_memory_strength must lie in [0, 1]")
    if scale <= 0.0:
        raise ValueError("radial_memory_log_scale must be positive")
    if not 0.0 <= slope_gate_strength <= 1.0:
        raise ValueError("radial_memory_slope_gate_strength must lie in [0, 1]")
    memory_gate_mode = str(variant.get("radial_memory_gate_mode", "none"))
    if strength > 0.0 and memory_gate_mode not in (
        "none",
        "channel",
        "complement",
    ):
        raise ValueError(
            "radial_memory_gate_mode must be none, channel, or complement"
        )
    if strength > 0.0 and memory_gate_mode != "none":
        if variant.get("channel_gate_property") is None:
            raise ValueError("gated radial memory requires channel_gate_property")
        channel_weight = np.asarray(
            local["channel_gate_cluster_weight"], dtype=np.float64
        )
        if np.any(~np.isfinite(channel_weight)) or np.any(
            (channel_weight < 0.0) | (channel_weight > 1.0)
        ):
            raise ValueError("radial-memory gate weights must lie in [0, 1]")
        memory_gate_weight = (
            channel_weight
            if memory_gate_mode == "channel"
            else 1.0 - channel_weight
        )
    else:
        memory_gate_mode = "none"
        memory_gate_weight = np.ones_like(gbar)
    effective_memory_strength = strength * memory_gate_weight
    channel_value = _variant_number(variant, "radial_memory_channel_code", 0.0)
    channel_code = int(round(channel_value))
    if abs(channel_value - channel_code) > 1.0e-12 or channel_code not in range(4):
        raise ValueError("radial_memory_channel_code must be 0, 1, 2, or 3")
    reference = np.asarray(local["g_reference_m_s2"], dtype=np.float64)
    slope_gate_mode = 0
    slope_smoothing_log_scale = math.nan
    base_source_factor = np.empty_like(gbar)
    steep_source_factor = np.empty_like(gbar)
    if slope_gate_strength == 0.0:
        local_slope = np.zeros_like(gbar)
        slope_coordinate = np.zeros_like(gbar)
        slope_gate = np.zeros_like(gbar)
        effective_gbar_power = np.full_like(gbar, gbar_power)
        effective_radius_power = np.full_like(gbar, radius_power)
        with np.errstate(over="raise", under="ignore", invalid="raise"):
            source_factor = np.power(gbar / reference, gbar_power) * np.power(
                radius, radius_power
            )
        base_source_factor[:] = source_factor
        steep_source_factor[:] = source_factor
    else:
        mode_value = _variant_number(
            variant, "radial_memory_slope_gate_mode", 0.0
        )
        slope_gate_mode = int(round(mode_value))
        if (
            abs(mode_value - slope_gate_mode) > 1.0e-12
            or slope_gate_mode not in range(5)
        ):
            raise ValueError(
                "radial_memory_slope_gate_mode must be 0, 1, 2, 3, or 4"
            )
        local_slope = _profile_log_slope(radius, gbar)
        if slope_gate_mode in (1, 2):
            profile_slope = _profile_log_linear_slope(radius, gbar)
            slope_coordinate = np.full_like(gbar, profile_slope)
        elif slope_gate_mode == 4:
            slope_smoothing_log_scale = _variant_number(
                variant, "radial_memory_slope_smoothing_log_scale", 0.5
            )
            slope_coordinate = _profile_smoothed_log_slope(
                radius,
                gbar,
                log_scale=slope_smoothing_log_scale,
            )
        else:
            slope_coordinate = local_slope
        slope_pivot = _variant_number(
            variant, "radial_memory_slope_gate_pivot", -1.0
        )
        slope_sharpness = _variant_number(
            variant, "radial_memory_slope_gate_sharpness", 1.0
        )
        if slope_sharpness <= 0.0:
            raise ValueError(
                "radial_memory_slope_gate_sharpness must be positive"
            )
        steep_gbar_power = _variant_number(
            variant, "radial_memory_steep_gbar_power", gbar_power
        )
        steep_radius_power = _variant_number(
            variant, "radial_memory_steep_radius_power", radius_power
        )
        slope_gate = slope_gate_strength * _logistic(
            slope_sharpness * (slope_pivot - slope_coordinate)
        )
        effective_gbar_power = gbar_power + slope_gate * (
            steep_gbar_power - gbar_power
        )
        effective_radius_power = radius_power + slope_gate * (
            steep_radius_power - radius_power
        )
        with np.errstate(over="raise", under="ignore", invalid="raise"):
            log_gbar_ratio = np.log(gbar / reference)
            log_radius = np.log(radius)
            base_source_factor = np.exp(
                gbar_power * log_gbar_ratio + radius_power * log_radius
            )
            steep_source_factor = np.exp(
                steep_gbar_power * log_gbar_ratio
                + steep_radius_power * log_radius
            )
            source_factor = np.exp(
                effective_gbar_power * log_gbar_ratio
                + effective_radius_power * log_radius
            )
    for factor in (source_factor, base_source_factor, steep_source_factor):
        if np.any(~np.isfinite(factor)) or np.any(factor <= 0.0):
            raise ValueError(
                "radial-memory source factor must be finite and positive"
            )

    outer_to_inner = bool(variant.get("radial_memory_outer_to_inner", False))

    def transport_with_factor(
        component: np.ndarray,
        factor: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        transported_source = np.asarray(component, dtype=np.float64) * factor
        if memory_gate_mode == "none":
            blended, memory = radial_memory_blend(
                radius,
                transported_source,
                strength=strength,
                log_scale=scale,
                outer_to_inner=outer_to_inner,
            )
        else:
            _, memory = radial_memory_blend(
                radius,
                transported_source,
                strength=1.0,
                log_scale=scale,
                outer_to_inner=outer_to_inner,
            )
            blended = transported_source + effective_memory_strength * (
                memory - transported_source
            )
        return blended / factor, memory / factor

    def transport(component: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if slope_gate_strength > 0.0 and slope_gate_mode in (2, 3, 4):
            base_blended, base_memory = transport_with_factor(
                component, base_source_factor
            )
            steep_blended, steep_memory = transport_with_factor(
                component, steep_source_factor
            )
            blended = (
                (1.0 - slope_gate) * base_blended
                + slope_gate * steep_blended
            )
            memory = (
                (1.0 - slope_gate) * base_memory
                + slope_gate * steep_memory
            )
            return blended, memory
        return transport_with_factor(component, source_factor)

    pre_screen = bool(variant.get("radial_memory_pre_screen", False))
    local_fractional = np.asarray(local["fractional_excess"], dtype=np.float64)
    effective_rg = np.asarray(local["effective_rg_excess"], dtype=np.float64)
    effective_sigma = np.asarray(
        local["effective_sigma_excess"], dtype=np.float64
    )
    memory_effective_rg = effective_rg.copy()
    memory_effective_sigma = effective_sigma.copy()

    if channel_code == 0:
        source_name = "combined_excess" if pre_screen else "fractional_excess"
        local_source = np.asarray(local[source_name], dtype=np.float64)
        blended_source, memory = transport(local_source)
        effective_excess = (
            np.asarray(local["acceleration_screen"], dtype=np.float64)
            * blended_source
            if pre_screen
            else blended_source
        )
        memory_combined = np.asarray(local["combined_excess"], dtype=np.float64)
        memory_raw_combined = np.asarray(
            local["raw_combined_excess"], dtype=np.float64
        )
    else:
        if pre_screen:
            raise ValueError(
                "radial_memory_pre_screen is only defined for combined-channel memory"
            )
        if channel_code in (1, 3):
            memory_effective_rg, rg_memory = transport(effective_rg)
        else:
            rg_memory = effective_rg.copy()
        if channel_code in (2, 3):
            memory_effective_sigma, sigma_memory = transport(effective_sigma)
        else:
            sigma_memory = effective_sigma.copy()
        memory_combined, memory_raw_combined = _combine_memory_channels(
            memory_effective_rg,
            memory_effective_sigma,
            variant,
        )
        effective_excess = (
            np.asarray(local["acceleration_screen"], dtype=np.float64)
            * memory_combined
        )
        if channel_code == 1:
            local_source = effective_rg
            blended_source = memory_effective_rg
            memory = rg_memory
        elif channel_code == 2:
            local_source = effective_sigma
            blended_source = memory_effective_sigma
            memory = sigma_memory
        else:
            local_source = effective_rg + effective_sigma
            blended_source = memory_effective_rg + memory_effective_sigma
            memory = rg_memory + sigma_memory

    enhancement = 1.0 + effective_excess
    if np.any(~np.isfinite(enhancement)) or np.any(enhancement < 1.0):
        raise ValueError("radial-memory enhancement must be finite and at least one")
    output = dict(local)
    output["local_fractional_excess"] = local_fractional
    output["radial_memory_local_source"] = local_source
    output["radial_memory_average"] = memory
    output["radial_memory_blended_source"] = blended_source
    output["radial_memory_source_factor"] = source_factor
    output["radial_memory_base_source_factor"] = base_source_factor
    output["radial_memory_steep_source_factor"] = steep_source_factor
    output["radial_memory_pre_screen"] = np.full_like(
        enhancement, float(pre_screen)
    )
    output["radial_memory_channel_code"] = np.full_like(
        enhancement, float(channel_code)
    )
    output["radial_memory_gbar_power"] = np.full_like(
        enhancement, gbar_power
    )
    output["radial_memory_radius_power"] = np.full_like(
        enhancement, radius_power
    )
    output["radial_memory_local_log_gbar_slope"] = local_slope
    output["radial_memory_slope_gate_coordinate"] = slope_coordinate
    output["radial_memory_slope_gate_mode"] = np.full_like(
        enhancement, float(slope_gate_mode)
    )
    output["radial_memory_slope_smoothing_log_scale"] = np.full_like(
        enhancement, slope_smoothing_log_scale
    )
    output["radial_memory_slope_gate_weight"] = slope_gate
    output["radial_memory_gate_weight"] = memory_gate_weight
    output["radial_memory_effective_strength"] = effective_memory_strength
    output["radial_memory_gate_mode_code"] = np.full_like(
        enhancement,
        {"none": 0.0, "channel": 1.0, "complement": 2.0}[
            memory_gate_mode
        ],
    )
    output["radial_memory_effective_gbar_power"] = effective_gbar_power
    output["radial_memory_effective_radius_power"] = effective_radius_power
    output["memory_effective_rg_excess"] = memory_effective_rg
    output["memory_effective_sigma_excess"] = memory_effective_sigma
    output["memory_raw_combined_excess"] = memory_raw_combined
    output["memory_combined_excess"] = memory_combined
    output["fractional_excess"] = effective_excess
    output["enhancement"] = enhancement
    return apply_radial_diffusion_to_response(output, radius, variant)


def screened_hybrid_profile_response(
    gbar_m_s2,
    density_g_cm3,
    radius_kpc,
    parameters,
    variant: Mapping[str, object],
    **kwargs,
) -> dict[str, np.ndarray]:
    """Apply the local law plus optional gate memory, memory, and diffusion."""

    result = screened_hybrid_response(
        gbar_m_s2,
        density_g_cm3,
        radius_kpc,
        parameters,
        variant,
        **kwargs,
    )
    result = apply_channel_gate_memory_to_response(
        result, radius_kpc, variant
    )
    return apply_radial_memory_to_response(result, radius_kpc, variant)


def solar_system_diagnostics(
    parameters,
    variant: Mapping[str, object],
    *,
    cassini_fractional_limit: float,
    interplanetary_density_g_cm3: float = 1.0e-30,
    acceleration_screen_m_s2: float = 1.2e-10,
) -> dict[str, float | bool]:
    """Evaluate the same law from 1.6 Solar radii through Saturn.

    The fractional-force gate is a conservative phenomenological proxy.  A
    complete metric theory would still need a derived PPN gamma.
    """

    solar_radius_m = 6.957e8
    radius_m = np.geomspace(1.6 * solar_radius_m, 8.43 * AU_M, 1000)
    gbar = G_SI * M_SUN_KG / np.square(radius_m)
    result = screened_hybrid_profile_response(
        gbar,
        np.full_like(gbar, interplanetary_density_g_cm3),
        radius_m / KPC_M,
        parameters,
        variant,
        acceleration_screen_m_s2=acceleration_screen_m_s2,
    )
    fractional = result["fractional_excess"]
    earth_change = float(np.interp(AU_M, radius_m, fractional))
    mercury_change = float(np.interp(0.38709893 * AU_M, radius_m, fractional))
    saturn_change = float(np.interp(8.43 * AU_M, radius_m, fractional))
    maximum = float(np.max(np.abs(fractional)))
    return {
        "PPN_gamma_assumption": 1.0,
        "PPN_gamma_minus_one": 0.0,
        "near_solar_limb_fractional_change": float(fractional[0]),
        "Mercury_orbit_fractional_change": mercury_change,
        "Earth_orbit_fractional_change": earth_change,
        "Saturn_orbit_fractional_change": saturn_change,
        "maximum_fractional_change_limb_to_Saturn": maximum,
        "Cassini_fractional_proxy_limit": float(cassini_fractional_limit),
        "Cassini_proxy_pass": bool(maximum <= cassini_fractional_limit),
    }


def mercury_precession_mas_per_century(
    parameters,
    variant: Mapping[str, object],
    *,
    interplanetary_density_g_cm3: float = 1.0e-30,
    acceleration_screen_m_s2: float = 1.2e-10,
    quadrature_points: int = 32_768,
) -> float:
    """First-order Mercury precession from the hybrid's radial extra force."""

    semimajor_axis_m = 0.38709893 * AU_M
    eccentricity = 0.205630
    orbital_period_days = 87.9691
    anomaly = np.linspace(0.0, 2.0 * np.pi, quadrature_points, endpoint=False)
    cosine = np.cos(anomaly)
    one_minus_e2 = 1.0 - eccentricity**2
    radius_m = (
        semimajor_axis_m
        * one_minus_e2
        / (1.0 + eccentricity * cosine)
    )
    gbar = G_SI * M_SUN_KG / np.square(radius_m)
    if (
        _variant_number(variant, "radial_memory_strength", 0.0) > 0.0
        or _variant_number(variant, "channel_gate_memory_strength", 0.0)
        > 0.0
        or _variant_number(variant, "radial_diffusion_strength", 0.0) > 0.0
    ):
        solar_radius_m = 6.957e8
        profile_radius_m = np.geomspace(
            1.6 * solar_radius_m,
            float(np.max(radius_m)) * 1.001,
            4096,
        )
        profile_gbar = G_SI * M_SUN_KG / np.square(profile_radius_m)
        profile_response = screened_hybrid_profile_response(
            profile_gbar,
            np.full_like(profile_gbar, interplanetary_density_g_cm3),
            profile_radius_m / KPC_M,
            parameters,
            variant,
            acceleration_screen_m_s2=acceleration_screen_m_s2,
        )
        fractional_excess = np.interp(
            radius_m,
            profile_radius_m,
            profile_response["fractional_excess"],
        )
    else:
        response = screened_hybrid_response(
            gbar,
            np.full_like(gbar, interplanetary_density_g_cm3),
            radius_m / KPC_M,
            parameters,
            variant,
            acceleration_screen_m_s2=acceleration_screen_m_s2,
        )
        fractional_excess = response["fractional_excess"]
    radial_perturbation = -gbar * fractional_excess
    time_weight = one_minus_e2**1.5 / np.square(
        1.0 + eccentricity * cosine
    )
    mean_r_cosine = float(np.mean(radial_perturbation * cosine * time_weight))
    period_seconds = orbital_period_days * 86_400.0
    mean_motion = 2.0 * np.pi / period_seconds
    mean_rate = (
        -math.sqrt(one_minus_e2)
        / (mean_motion * semimajor_axis_m * eccentricity)
        * mean_r_cosine
    )
    radians_per_orbit = mean_rate * period_seconds
    orbits_per_century = 100.0 * JULIAN_YEAR_DAYS / orbital_period_days
    return radians_per_orbit * orbits_per_century * RAD_TO_MAS
