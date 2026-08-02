"""Nonperturbative compound-path activation for 3D tensor AQUAL."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voidscreen.metric_lensing_3d import TensorActivation3D, exact_tensor_activation_3d
from voidscreen.multipole_activation_3d import MultipoleGate3D, baryonic_multipole_gate_3d


@dataclass(frozen=True)
class CompoundPathActivation3D:
    sigma: np.ndarray
    elementary_probability: np.ndarray
    coherent_opportunities: np.ndarray
    minimum_eigenvalue_proxy: np.ndarray
    amplitude_gate: float
    multipole: MultipoleGate3D
    local: TensorActivation3D


def exact_compound_path_activation_3d(
    stellar_density: np.ndarray,
    gas_density: np.ndarray,
    spacing: float,
    *,
    coherence_length: float,
    coherence_power: float = 2.0,
    **activation_kwargs,
) -> CompoundPathActivation3D:
    """Compound an elementary routed fraction over a physical tidal path."""
    local = exact_tensor_activation_3d(
        stellar_density,
        gas_density,
        spacing,
        coherence_length=coherence_length,
        coherence_power=coherence_power,
        **activation_kwargs,
    )
    multipole = baryonic_multipole_gate_3d(stellar_density, gas_density, spacing)
    amplitude_gate = float(np.sqrt(np.clip(multipole.gate, 0.0, 1.0)))
    elementary = np.clip(
        amplitude_gate * local.high_acceleration_screen * local.transverse_mismatch,
        0.0,
        1.0 - np.finfo(float).eps,
    )
    opportunities = np.power(
        np.maximum(local.trace_length / float(coherence_length), 0.0),
        float(coherence_power),
    )
    sigma = -np.expm1(opportunities * np.log1p(-elementary))
    mu_floor = float(activation_kwargs.get("mu_floor", 1e-6))
    sigma = np.clip(sigma, 0.0, 1.0 - mu_floor)
    return CompoundPathActivation3D(
        sigma=sigma,
        elementary_probability=elementary,
        coherent_opportunities=opportunities,
        minimum_eigenvalue_proxy=local.mu_newtonian_proxy * (1.0 - sigma),
        amplitude_gate=amplitude_gate,
        multipole=multipole,
        local=local,
    )
