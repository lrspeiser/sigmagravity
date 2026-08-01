"""Slowly unbounded effective-gravity laws with Solar-System screening."""

from __future__ import annotations

import math

import numpy as np

from .data import KPC_M

G_SI = 6.67430e-11
M_SUN_KG = 1.988409870698051e30

CURVATURE_MODELS = {
    "curvature_log",
    "curvature_loglog",
    "curvature_rootlog",
    "curvature_power",
    "curvature_stretched_power",
    "curvature_mixed_power",
    "curvature_additive_power",
    "curvature_decelerating_power",
    "curvature_variable_mass_power",
    "curvature_variable_density_power",
    "curvature_variable_shape_power",
}
VARIABLE_EXPONENT_MODELS = {
    "curvature_variable_mass_power",
    "curvature_variable_density_power",
    "curvature_variable_shape_power",
}
VARIABLE_EXPONENT_DENSITY_MODELS = {
    "curvature_variable_density_power",
    "curvature_variable_shape_power",
}
PATH_RUNNING_MODELS = {"path_log_running", "path_power_running"}
TENSOR_RUNNING_MODELS = {
    "tensor_alignment_log",
    "tensor_dominance_log",
    "tensor_alignment_power",
    "tensor_dominance_power",
}
RUNNING_MODELS = CURVATURE_MODELS | PATH_RUNNING_MODELS | TENSOR_RUNNING_MODELS


def scalar_tidal_proxy(gbar_m_s2, radius_kpc) -> np.ndarray:
    """Return the weak-field scalar tidal scale ``g_bar/r`` in s^-2."""
    gbar = np.asarray(gbar_m_s2, dtype=float)
    radius = np.asarray(radius_kpc, dtype=float)
    gbar, radius = np.broadcast_arrays(gbar, radius)
    if np.any(~np.isfinite(gbar)) or np.any(gbar <= 0.0):
        raise ValueError("gbar must be finite and positive")
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("radius must be finite and positive")
    return gbar / (radius * KPC_M)


def _log_one_plus_power(log_ratio: np.ndarray, power: float) -> np.ndarray:
    """Evaluate ``log(1 + exp(power*log_ratio))`` without overflow."""
    return np.logaddexp(0.0, float(power) * log_ratio)


def equivalent_enclosed_baryonic_mass_msun(gbar_m_s2, radius_kpc) -> np.ndarray:
    """Return the spherical-equivalent baryonic mass implied by ``g_bar``.

    For a spherical source this is the actual enclosed mass.  For a disk it is
    deliberately only a force-equivalent mass proxy, not a claim of spherical
    geometry.
    """
    gbar = np.asarray(gbar_m_s2, dtype=float)
    radius_m = np.asarray(radius_kpc, dtype=float) * KPC_M
    gbar, radius_m = np.broadcast_arrays(gbar, radius_m)
    return gbar * np.square(radius_m) / (G_SI * M_SUN_KG)


def mean_equivalent_baryonic_density_g_cm3(gbar_m_s2, radius_kpc) -> np.ndarray:
    """Return the spherical mean density associated with ``g_bar/r``."""
    gbar = np.asarray(gbar_m_s2, dtype=float)
    radius_m = np.asarray(radius_kpc, dtype=float) * KPC_M
    gbar, radius_m = np.broadcast_arrays(gbar, radius_m)
    density_kg_m3 = 3.0 * gbar / (4.0 * math.pi * G_SI * radius_m)
    return density_kg_m3 * 1.0e-3


def variable_exponent(
    gbar_m_s2,
    radius_kpc,
    model: str,
    parameters,
    *,
    local_density_g_cm3=None,
) -> dict[str, np.ndarray]:
    """Evaluate the universal bounded exponent ``p(X)``.

    ``p(X) = p0 exp(beta tanh(ln(X/X*)))``.  The three candidate definitions
    of X are a force-equivalent enclosed baryonic mass, local baryonic density,
    or the local-to-mean density ratio (a concentration/profile-shape proxy).
    """
    if model not in VARIABLE_EXPONENT_MODELS:
        raise ValueError(f"model {model} has no variable exponent")
    values = np.asarray(parameters, dtype=float)
    if values.shape != (5,):
        raise ValueError(f"{model} requires five parameters")
    _, p0, beta, log10_pivot, _ = map(float, values)
    if not math.isfinite(p0) or p0 <= 0.0 or not math.isfinite(beta):
        raise ValueError("p0 must be positive and beta must be finite")

    gbar = np.asarray(gbar_m_s2, dtype=float)
    radius = np.asarray(radius_kpc, dtype=float)
    gbar, radius = np.broadcast_arrays(gbar, radius)
    if model == "curvature_variable_mass_power":
        property_value = equivalent_enclosed_baryonic_mass_msun(gbar, radius)
        property_name = "equivalent_enclosed_baryonic_mass_msun"
    else:
        if local_density_g_cm3 is None:
            raise ValueError(f"{model} requires local baryonic density")
        density = np.broadcast_to(np.asarray(local_density_g_cm3, dtype=float), gbar.shape)
        if np.any(~np.isfinite(density)) or np.any(density < 0.0):
            raise ValueError("local baryonic density must be finite and non-negative")
        if model == "curvature_variable_density_power":
            property_value = density
            property_name = "local_baryonic_density_g_cm3"
        else:
            mean_density = mean_equivalent_baryonic_density_g_cm3(gbar, radius)
            property_value = density / np.maximum(mean_density, 1.0e-300)
            property_name = "local_to_mean_baryonic_density_ratio"

    pivot = 10.0**log10_pivot
    safe_property = np.maximum(property_value, np.finfo(float).tiny)
    transition_coordinate = np.tanh(np.log(safe_property / pivot))
    exponent = p0 * np.exp(beta * transition_coordinate)
    if np.any(~np.isfinite(exponent)) or np.any(exponent <= 0.0):
        raise ValueError("effective exponent must be finite and positive")
    return {
        "effective_exponent": exponent,
        "exponent_property": property_value,
        "exponent_property_name": property_name,
        "exponent_transition_coordinate": transition_coordinate,
    }


def _directional_availability(tidal_eigenvalues_s2, model: str, q: float) -> np.ndarray:
    tidal = np.asarray(tidal_eigenvalues_s2, dtype=float)
    if tidal.ndim < 1 or tidal.shape[-1] != 3 or np.any(~np.isfinite(tidal)):
        raise ValueError("tidal eigenvalues must be finite with final dimension three")
    magnitude = np.abs(tidal)
    if "alignment" in model:
        denominator = np.linalg.norm(magnitude, axis=-1)
    elif "dominance" in model:
        denominator = np.max(magnitude, axis=-1)
    else:
        raise ValueError(f"model {model} has no directional definition")
    if np.any(denominator <= 0.0):
        raise ValueError("tidal tensor must have nonzero norm")
    return np.power(magnitude[..., 0] / denominator, float(q))


def running_enhancement(
    gbar_m_s2,
    radius_kpc,
    model: str,
    parameters,
    *,
    tidal_eigenvalues_s2=None,
    local_density_g_cm3=None,
) -> dict[str, np.ndarray]:
    """Return an unbounded, positive enhancement relative to local ``G``."""
    if model not in RUNNING_MODELS:
        raise ValueError(f"unknown running model {model}")
    gbar = np.asarray(gbar_m_s2, dtype=float)
    radius = np.asarray(radius_kpc, dtype=float)
    gbar, radius = np.broadcast_arrays(gbar, radius)
    if np.any(~np.isfinite(gbar)) or np.any(gbar <= 0.0):
        raise ValueError("gbar must be finite and positive")
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("radius must be finite and positive")
    values = np.asarray(parameters, dtype=float)

    availability = np.ones_like(gbar)
    exponent_result = None
    if model in VARIABLE_EXPONENT_MODELS:
        if values.shape != (5,):
            raise ValueError(f"{model} requires five parameters")
        log10_transition, _, _, _, strength = map(float, values)
        tidal = scalar_tidal_proxy(gbar, radius)
        log_ratio = math.log(10.0) * log10_transition - np.log(tidal)
        exponent_result = variable_exponent(
            gbar,
            radius,
            model,
            values,
            local_density_g_cm3=local_density_g_cm3,
        )
        growth_coordinate = np.logaddexp(
            0.0, exponent_result["effective_exponent"] * log_ratio
        )
    elif model in CURVATURE_MODELS:
        four_parameter_curvature = {
            "curvature_stretched_power",
            "curvature_mixed_power",
            "curvature_additive_power",
            "curvature_decelerating_power",
        }
        expected = 4 if model in four_parameter_curvature else 3
        if values.shape != (expected,):
            raise ValueError(f"{model} requires {expected} parameters")
        log10_transition, scale_power, strength = map(float, values[:3])
        shape = float(values[3]) if expected == 4 else None
        tidal = scalar_tidal_proxy(gbar, radius)
        log_ratio = math.log(10.0) * log10_transition - np.log(tidal)
        growth_coordinate = _log_one_plus_power(log_ratio, scale_power)
    elif model in PATH_RUNNING_MODELS:
        if values.shape != (5,):
            raise ValueError(f"{model} requires five parameters")
        log10_transition, transition_power, log10_radius, radius_power, strength = map(
            float, values
        )
        tidal = scalar_tidal_proxy(gbar, radius)
        with np.errstate(over="ignore"):
            low_curvature = 1.0 / (
                1.0 + np.power(tidal / (10.0**log10_transition), transition_power)
            )
        log_radius_ratio = np.log(radius) - math.log(10.0) * log10_radius
        growth_coordinate = low_curvature * _log_one_plus_power(
            log_radius_ratio, radius_power
        )
    else:
        if values.shape != (4,):
            raise ValueError(f"{model} requires four parameters")
        log10_transition, scale_power, strength, q = map(float, values)
        if tidal_eigenvalues_s2 is None:
            raise ValueError(f"{model} requires tidal eigenvalues")
        eigenvalues = np.asarray(tidal_eigenvalues_s2, dtype=float)
        tidal = np.linalg.norm(eigenvalues, axis=-1)
        log_ratio = math.log(10.0) * log10_transition - np.log(tidal)
        growth_coordinate = _log_one_plus_power(log_ratio, scale_power)
        availability = _directional_availability(eigenvalues, model, q)
        growth_coordinate = growth_coordinate * availability

    if not np.isfinite(strength) or strength <= 0.0:
        raise ValueError("running strength must be finite and positive")
    if model == "curvature_stretched_power":
        log_enhancement = np.minimum(
            strength * np.power(growth_coordinate, shape), 700.0
        )
        fractional_excess = np.expm1(log_enhancement)
        enhancement = 1.0 + fractional_excess
    elif model == "curvature_mixed_power":
        log_enhancement = np.minimum(
            strength * growth_coordinate + shape * np.log1p(growth_coordinate),
            700.0,
        )
        fractional_excess = np.expm1(log_enhancement)
        enhancement = 1.0 + fractional_excess
    elif model == "curvature_additive_power":
        fractional_excess = shape * np.expm1(
            np.minimum(strength * growth_coordinate, 680.0)
        )
        enhancement = 1.0 + fractional_excess
    elif model == "curvature_decelerating_power":
        log_enhancement = np.minimum(
            strength
            * growth_coordinate
            / np.power(1.0 + growth_coordinate, shape),
            700.0,
        )
        fractional_excess = np.expm1(log_enhancement)
        enhancement = 1.0 + fractional_excess
    elif "loglog" in model:
        fractional_excess = strength * np.log1p(growth_coordinate)
        enhancement = 1.0 + fractional_excess
    elif "rootlog" in model:
        log_enhancement = strength * np.log1p(growth_coordinate)
        fractional_excess = np.expm1(log_enhancement)
        enhancement = 1.0 + fractional_excess
    elif "power" in model:
        log_enhancement = np.minimum(strength * growth_coordinate, 700.0)
        fractional_excess = np.expm1(log_enhancement)
        enhancement = 1.0 + fractional_excess
    else:
        fractional_excess = strength * growth_coordinate
        enhancement = 1.0 + fractional_excess
    if np.any(~np.isfinite(enhancement)) or np.any(enhancement < 1.0):
        raise ValueError("running enhancement must be finite and at least one")
    output = {
        "enhancement_relative_to_local_G": enhancement,
        "fractional_enhancement_above_local_G": fractional_excess,
        "running_coordinate": growth_coordinate,
        "directional_availability": availability,
        "tidal_scale_s2": tidal,
    }
    if exponent_result is not None:
        output.update(exponent_result)
    return output


def predict_running_acceleration(
    gbar_m_s2,
    radius_kpc,
    model: str,
    parameters,
    *,
    tidal_eigenvalues_s2=None,
    local_density_g_cm3=None,
) -> dict[str, np.ndarray]:
    result = running_enhancement(
        gbar_m_s2,
        radius_kpc,
        model,
        parameters,
        tidal_eigenvalues_s2=tidal_eigenvalues_s2,
        local_density_g_cm3=local_density_g_cm3,
    )
    result["predicted_acceleration_m_s2"] = (
        np.asarray(gbar_m_s2, dtype=float) * result["enhancement_relative_to_local_G"]
    )
    return result


def solar_system_diagnostics(model: str, parameters, *, cassini_limit: float) -> dict:
    """Evaluate a zero-slip running law from the solar limb to Saturn."""
    solar_radius_m = 6.957e8
    astronomical_unit_m = 149597870700.0
    radius_m = np.geomspace(1.6 * solar_radius_m, 8.43 * astronomical_unit_m, 800)
    radius_kpc = radius_m / KPC_M
    gm = G_SI * M_SUN_KG
    gbar = gm / np.square(radius_m)
    eigenvalues = None
    if model in TENSOR_RUNNING_MODELS:
        tidal = gm / np.power(radius_m, 3)
        eigenvalues = np.stack([-2.0 * tidal, tidal, tidal], axis=-1)
    result = running_enhancement(
        gbar,
        radius_kpc,
        model,
        parameters,
        tidal_eigenvalues_s2=eigenvalues,
        local_density_g_cm3=(
            np.zeros_like(gbar) if model in VARIABLE_EXPONENT_DENSITY_MODELS else None
        ),
    )
    enhancement = result["enhancement_relative_to_local_G"]
    fractional_excess = result["fractional_enhancement_above_local_G"]
    earth_change = float(np.interp(astronomical_unit_m, radius_m, fractional_excess))
    saturn_change = float(
        np.interp(8.43 * astronomical_unit_m, radius_m, fractional_excess)
    )
    earth = float(
        1.0 + earth_change
    )
    cassini = float(
        1.0 + saturn_change
    )
    maximum_change = float(np.max(np.abs(fractional_excess)))
    return {
        "PPN_gamma_assumption": 1.0,
        "PPN_gamma_minus_one": 0.0,
        "Cassini_fractional_limit": float(cassini_limit),
        "near_solar_limb_enhancement": float(enhancement[0]),
        "Earth_orbit_enhancement": earth,
        "Earth_orbit_fractional_change": earth_change,
        "Saturn_orbit_enhancement": cassini,
        "Saturn_orbit_fractional_change": saturn_change,
        "maximum_fractional_change_limb_to_Saturn": maximum_change,
        "coupling_spread_limb_to_Saturn": float(
            np.max(fractional_excess) - np.min(fractional_excess)
        ),
        "Cassini_pass": bool(maximum_change <= cassini_limit),
    }


def point_mass_scale_diagnostics(
    model: str,
    parameters,
    *,
    mass_solar: float = 1.0e11,
    radii_kpc=(1.0, 10.0, 100.0, 1000.0, 1.0e6),
) -> dict:
    """Illustrate unbounded growth around a fixed isolated baryonic mass."""
    radius = np.asarray(radii_kpc, dtype=float)
    gm = G_SI * M_SUN_KG * float(mass_solar)
    radius_m = radius * KPC_M
    gbar = gm / np.square(radius_m)
    eigenvalues = None
    if model in TENSOR_RUNNING_MODELS:
        tidal = gm / np.power(radius_m, 3)
        eigenvalues = np.stack([-2.0 * tidal, tidal, tidal], axis=-1)
    result = running_enhancement(
        gbar,
        radius,
        model,
        parameters,
        tidal_eigenvalues_s2=eigenvalues,
        local_density_g_cm3=(
            np.zeros_like(gbar) if model in VARIABLE_EXPONENT_DENSITY_MODELS else None
        ),
    )
    return {
        "mass_solar": float(mass_solar),
        "warning": "isolated point-mass extrapolation, not a homogeneous cosmology",
        "enhancement_by_radius_kpc": {
            f"{value:g}": float(enhancement)
            for value, enhancement in zip(
                radius, result["enhancement_relative_to_local_G"], strict=True
            )
        },
    }
