"""Amplitude-level baryonic multipole gate for 3D tensor activation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voidscreen.metric_lensing_3d import TensorActivation3D, exact_tensor_activation_3d
from voidscreen.multipole_activation_3d import MultipoleGate3D, baryonic_multipole_gate_3d


@dataclass(frozen=True)
class AmplitudeMultipoleActivation3D:
    sigma: np.ndarray
    minimum_eigenvalue_proxy: np.ndarray
    amplitude_gate: float
    multipole: MultipoleGate3D
    local: TensorActivation3D


def exact_amplitude_multipole_activation_3d(
    stellar_density: np.ndarray,
    gas_density: np.ndarray,
    spacing: float,
    **activation_kwargs,
) -> AmplitudeMultipoleActivation3D:
    """Apply sqrt(power-fraction) to a field-amplitude coefficient.

    The multipole gate is global for one registered baryonic system. Therefore
    its nonnegative square root scales the complete local coefficient field.
    """
    local = exact_tensor_activation_3d(
        stellar_density,
        gas_density,
        spacing,
        **activation_kwargs,
    )
    multipole = baryonic_multipole_gate_3d(stellar_density, gas_density, spacing)
    amplitude_gate = float(np.sqrt(np.clip(multipole.gate, 0.0, 1.0)))
    sigma = np.clip(local.sigma * amplitude_gate, 0.0, 1.0)
    return AmplitudeMultipoleActivation3D(
        sigma=sigma,
        minimum_eigenvalue_proxy=local.mu_newtonian_proxy * (1.0 - sigma),
        amplitude_gate=amplitude_gate,
        multipole=multipole,
        local=local,
    )
