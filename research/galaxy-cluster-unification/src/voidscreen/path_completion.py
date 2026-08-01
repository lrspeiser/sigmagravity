"""Bounded, path-dependent gravitational-vector completion laws."""

from __future__ import annotations

import numpy as np

from .data import KPC_M

PATH_MODELS = (
    "distance_path",
    "tidal_path",
    "matter_path",
    "hybrid_path",
)
MASS_PATH_MODELS = (
    "mass_weighted_path",
    "mass_amplified_path",
    "mass_ceiling_path",
)

G_SI = 6.67430e-11
M_SUN_KG = 1.98847e30


def _sigmoid(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=float)
    positive = values >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _cumulative_path(radius_kpc: np.ndarray, weight: np.ndarray, *, center_weight: float) -> np.ndarray:
    extended_radius = np.concatenate(([0.0], radius_kpc))
    extended_weight = np.concatenate(([float(center_weight)], weight))
    segments = (
        0.5
        * (extended_weight[:-1] + extended_weight[1:])
        * np.diff(extended_radius)
    )
    return np.cumsum(segments)


def path_completion_profile(
    radius_kpc,
    gbar_m_s2,
    model: str,
    parameters,
) -> dict[str, np.ndarray]:
    """Return a completed radial field for one ordered source profile.

    Completion evolves as ``logit(C)=logit(C_solar)+tau``.  The nonnegative
    optical depth ``tau`` is a path integral, so ``C_solar <= C < 1`` and the
    proposed maximum field is never exceeded.  The four model names select
    different recovery weights along the path.
    """
    if model not in PATH_MODELS:
        raise ValueError(f"unknown path-completion model: {model}")
    radius = np.asarray(radius_kpc, dtype=float)
    gbar = np.asarray(gbar_m_s2, dtype=float)
    values = np.asarray(parameters, dtype=float)
    if radius.ndim != 1 or gbar.ndim != 1 or radius.shape != gbar.shape:
        raise ValueError("radius and gbar must be matching one-dimensional arrays")
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("radius must be finite and positive")
    if np.any(~np.isfinite(gbar)) or np.any(gbar <= 0.0):
        raise ValueError("gbar must be finite and positive")
    if np.any(np.diff(radius) <= 0.0):
        raise ValueError("radius must be strictly increasing")

    expected = {
        "distance_path": 3,
        "tidal_path": 4,
        "matter_path": 4,
        "hybrid_path": 4,
    }[model]
    if values.shape != (expected,) or np.any(~np.isfinite(values)):
        raise ValueError(f"{model} requires {expected} finite parameters")
    solar_completion = float(values[0])
    recovery_length_kpc = float(10.0 ** values[1])
    if not 0.0 < solar_completion < 1.0 or recovery_length_kpc <= 0.0:
        raise ValueError("invalid completion fraction or recovery length")

    tidal = gbar / (radius * KPC_M)
    if model == "distance_path":
        gamma = float(values[2])
        if gamma <= 0.0:
            raise ValueError("distance power must be positive")
        optical_depth = np.power(radius / recovery_length_kpc, gamma)
        path_weight = np.ones_like(radius)
    elif model == "tidal_path":
        tidal_transition = float(10.0 ** values[2])
        power = float(values[3])
        if tidal_transition <= 0.0 or power <= 0.0:
            raise ValueError("invalid tidal path parameters")
        with np.errstate(over="ignore"):
            path_weight = 1.0 / (1.0 + np.power(tidal / tidal_transition, power))
        optical_depth = _cumulative_path(radius, path_weight, center_weight=0.0) / recovery_length_kpc
    elif model == "matter_path":
        acceleration_transition = float(10.0 ** values[2])
        power = float(values[3])
        if acceleration_transition <= 0.0 or power <= 0.0:
            raise ValueError("invalid matter path parameters")
        with np.errstate(over="ignore"):
            path_weight = 1.0 / (
                1.0 + np.power(gbar / acceleration_transition, power)
            )
        optical_depth = _cumulative_path(radius, path_weight, center_weight=0.0) / recovery_length_kpc
    else:
        tidal_transition = float(10.0 ** values[2])
        acceleration_transition = float(10.0 ** values[3])
        if tidal_transition <= 0.0 or acceleration_transition <= 0.0:
            raise ValueError("invalid hybrid path parameters")
        tidal_weight = 1.0 / (1.0 + tidal / tidal_transition)
        matter_weight = 1.0 / (1.0 + gbar / acceleration_transition)
        path_weight = tidal_weight * matter_weight
        optical_depth = _cumulative_path(radius, path_weight, center_weight=0.0) / recovery_length_kpc

    initial_logit = np.log(solar_completion / (1.0 - solar_completion))
    completion = _sigmoid(initial_logit + optical_depth)
    enhancement = completion / solar_completion
    return {
        "tidal_curvature_s2": tidal,
        "path_weight": path_weight,
        "recovery_optical_depth": optical_depth,
        "completion_fraction": completion,
        "enhancement_relative_to_local_G": enhancement,
        "predicted_acceleration_m_s2": gbar * enhancement,
    }


def predict_path_completion_frame(
    frame,
    model: str,
    parameters,
    *,
    system_column: str = "system",
    radius_column: str = "radius_kpc",
    gbar_column: str = "gbar_m_s2",
) -> dict[str, np.ndarray]:
    """Evaluate complete radial systems and restore the input row order."""
    size = len(frame)
    names = (
        "tidal_curvature_s2",
        "path_weight",
        "recovery_optical_depth",
        "completion_fraction",
        "enhancement_relative_to_local_G",
        "predicted_acceleration_m_s2",
    )
    output = {name: np.full(size, np.nan) for name in names}
    indexed = frame.reset_index(drop=True)
    for _, group in indexed.groupby(system_column, sort=True):
        ordered = group.sort_values(radius_column, kind="stable")
        result = path_completion_profile(
            ordered[radius_column].to_numpy(float),
            ordered[gbar_column].to_numpy(float),
            model,
            parameters,
        )
        indices = ordered.index.to_numpy(int)
        for name in names:
            output[name][indices] = result[name]
    if any(np.any(~np.isfinite(values)) for values in output.values()):
        raise RuntimeError("path completion left non-finite rows")
    return output


def mass_path_completion_profile(
    radius_kpc,
    gbar_m_s2,
    model: str,
    parameters,
) -> dict[str, np.ndarray]:
    """Return a bounded path law whose recovery depends on enclosed mass.

    These are explicitly second-stage exploratory variants.  The spherical
    effective mass ``g_bar r^2/G`` is converted into a monotonic path history
    before it controls recovery, so a vector does not forget mass it already
    passed when a noisy radial proxy decreases.
    """
    if model not in MASS_PATH_MODELS:
        raise ValueError(f"unknown mass path model: {model}")
    radius = np.asarray(radius_kpc, dtype=float)
    gbar = np.asarray(gbar_m_s2, dtype=float)
    values = np.asarray(parameters, dtype=float)
    if radius.ndim != 1 or gbar.ndim != 1 or radius.shape != gbar.shape:
        raise ValueError("radius and gbar must be matching one-dimensional arrays")
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0) or np.any(np.diff(radius) <= 0.0):
        raise ValueError("radius must be finite, positive, and strictly increasing")
    if np.any(~np.isfinite(gbar)) or np.any(gbar <= 0.0):
        raise ValueError("gbar must be finite and positive")
    if values.shape != (4,) or np.any(~np.isfinite(values)):
        raise ValueError("mass path models require four finite parameters")

    solar_completion = float(values[0])
    recovery_length_kpc = float(10.0 ** values[1])
    mass_transition_solar = float(10.0 ** values[2])
    mass_power = float(values[3])
    if not 0.0 < solar_completion < 1.0:
        raise ValueError("solar completion must lie in (0, 1)")
    if recovery_length_kpc <= 0.0 or mass_transition_solar <= 0.0 or mass_power <= 0.0:
        raise ValueError("mass path scales and power must be positive")

    radius_m = radius * KPC_M
    effective_mass = gbar * radius_m**2 / G_SI / M_SUN_KG
    mass_history = np.maximum.accumulate(effective_mass)
    mass_ratio = mass_history / mass_transition_solar
    availability = 1.0 / (1.0 + np.power(1.0 / mass_ratio, mass_power))
    initial_logit = np.log(solar_completion / (1.0 - solar_completion))

    if model == "mass_weighted_path":
        path_weight = availability
        optical_depth = _cumulative_path(radius, path_weight, center_weight=0.0) / recovery_length_kpc
        completion = _sigmoid(initial_logit + optical_depth)
    elif model == "mass_amplified_path":
        path_weight = np.power(mass_ratio, mass_power)
        optical_depth = _cumulative_path(radius, path_weight, center_weight=0.0) / recovery_length_kpc
        completion = _sigmoid(initial_logit + optical_depth)
    else:
        path_weight = availability
        optical_depth = radius / recovery_length_kpc
        unrestricted = _sigmoid(initial_logit + optical_depth)
        completion = solar_completion + availability * (unrestricted - solar_completion)

    enhancement = completion / solar_completion
    tidal = gbar / radius_m
    return {
        "tidal_curvature_s2": tidal,
        "effective_enclosed_mass_solar": effective_mass,
        "mass_history_solar": mass_history,
        "path_weight": path_weight,
        "recovery_optical_depth": optical_depth,
        "completion_fraction": completion,
        "enhancement_relative_to_local_G": enhancement,
        "predicted_acceleration_m_s2": gbar * enhancement,
    }


def predict_mass_path_completion_frame(
    frame,
    model: str,
    parameters,
    *,
    system_column: str = "system",
    radius_column: str = "radius_kpc",
    gbar_column: str = "gbar_m_s2",
) -> dict[str, np.ndarray]:
    """Evaluate mass-dependent paths by complete system in input row order."""
    names = (
        "tidal_curvature_s2",
        "effective_enclosed_mass_solar",
        "mass_history_solar",
        "path_weight",
        "recovery_optical_depth",
        "completion_fraction",
        "enhancement_relative_to_local_G",
        "predicted_acceleration_m_s2",
    )
    output = {name: np.full(len(frame), np.nan) for name in names}
    indexed = frame.reset_index(drop=True)
    for _, group in indexed.groupby(system_column, sort=True):
        ordered = group.sort_values(radius_column, kind="stable")
        result = mass_path_completion_profile(
            ordered[radius_column].to_numpy(float),
            ordered[gbar_column].to_numpy(float),
            model,
            parameters,
        )
        indices = ordered.index.to_numpy(int)
        for name in names:
            output[name][indices] = result[name]
    if any(np.any(~np.isfinite(values)) for values in output.values()):
        raise RuntimeError("mass path completion left non-finite rows")
    return output
