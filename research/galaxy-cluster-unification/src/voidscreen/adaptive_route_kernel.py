"""Scale-free controls for a conservative, extent-adaptive route kernel."""

from __future__ import annotations

import numpy as np


def transformed_source_weights(weights, power: float = 1.0) -> np.ndarray:
    """Change source dominance while preserving unit total weight."""
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("weights must be a nonempty vector")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0) or np.sum(values) <= 0.0:
        raise ValueError("weights must be finite, nonnegative, and not all zero")
    if not np.isfinite(power) or power <= 0.0:
        raise ValueError("power must be finite and positive")
    result = np.power(values / np.max(values), float(power))
    return result / np.sum(result)


def multiplicity_gate(weights, power: float) -> float:
    """Return a decomposition-stable concentration gate in [0, 1].

    The base quantity ``1-sum(w_i^2)`` is exactly zero for one source and
    approaches one when influence is distributed across many sources.
    """
    values = transformed_source_weights(weights, 1.0)
    if not np.isfinite(power) or power < 0.0:
        raise ValueError("power must be finite and nonnegative")
    base = float(np.clip(1.0 - np.sum(np.square(values)), 0.0, 1.0))
    return float(np.power(base, float(power)))


def extent_coordinate(r50_kpc: float, concentration: float, feature: str) -> float:
    """Dimensionless, predeclared baryonic-extent coordinate."""
    if r50_kpc <= 0.0 or concentration <= 0.0:
        raise ValueError("extent inputs must be positive")
    z_r50 = float(np.log(float(r50_kpc) / 150.0) / 0.35)
    z_concentration = float(np.log(float(concentration) / 0.65) / 0.25)
    if feature == "r50":
        return z_r50
    if feature == "concentration":
        return z_concentration
    if feature == "combined":
        return 0.5 * (z_r50 + z_concentration)
    if feature == "none":
        return 0.0
    raise ValueError(f"unknown extent feature {feature}")


def adaptive_route_parameters(
    *,
    r50_kpc: float,
    concentration: float,
    source_weights,
    feature: str,
    base_fraction: float,
    extent_slope: float,
    base_length_kpc: float,
    length_power: float,
    base_width_kpc: float,
    width_power: float,
    gate_power: float,
) -> dict[str, float]:
    """Evaluate one universal kernel for a particular baryonic morphology."""
    if not 0.0 < base_fraction < 1.0:
        raise ValueError("base_fraction must lie strictly between zero and one")
    if base_length_kpc <= 0.0 or base_width_kpc <= 0.0:
        raise ValueError("base length and width must be positive")
    coordinate = extent_coordinate(r50_kpc, concentration, feature)
    logit = np.log(base_fraction / (1.0 - base_fraction)) + float(extent_slope) * coordinate
    fraction = float(1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0))))
    fraction *= multiplicity_gate(source_weights, float(gate_power))
    length = float(base_length_kpc) * np.power(float(r50_kpc) / 150.0, float(length_power))
    width = float(base_width_kpc) * np.power(float(concentration) / 0.65, float(width_power))
    return {
        "extent_coordinate": coordinate,
        "multiplicity_gate": multiplicity_gate(source_weights, float(gate_power)),
        "routing_fraction": float(np.clip(fraction, 0.0, 1.0)),
        "return_scale_kpc": float(np.clip(length, 75.0, 500.0)),
        "width_kpc": float(np.clip(width, 20.0, 120.0)),
    }
