from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from voidscreen.sigma_triaxial_memory import (
    TidalMemoryField,
    spectral_tidal_memory,
    symmetric_trace_free,
)


@dataclass(frozen=True)
class TidalMisalignmentField:
    """Local and scale-memory tidal tensors with their bounded mismatch response."""

    base: TidalMemoryField
    local_tide: np.ndarray
    memory_tide: np.ndarray
    commutator: np.ndarray
    bounded_potential: np.ndarray


def _validated_screen(screen: Any, shape: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(screen, dtype=float)
    try:
        broadcast = np.broadcast_to(values, shape)
    except ValueError as error:
        raise ValueError("screen is not broadcastable to the tensor batch") from error
    if np.any(~np.isfinite(broadcast)) or np.any(broadcast < 0.0) or np.any(broadcast > 1.0):
        raise ValueError("screen must be finite and lie between zero and one")
    return broadcast


def tidal_commutator(local_tide: Any, memory_tide: Any) -> np.ndarray:
    """Return the antisymmetric commutator of two STF tidal tensors."""
    local = symmetric_trace_free(local_tide)
    memory = symmetric_trace_free(memory_tide)
    if local.shape != memory.shape:
        raise ValueError("local and memory tensors must have identical shapes")
    return np.matmul(local, memory) - np.matmul(memory, local)


def bounded_misalignment_potential(
    local_tide: Any,
    memory_tide: Any,
    *,
    screen: Any = 1.0,
) -> np.ndarray:
    """Return the screened, Böttcher-Wenzel-bounded commutator potential."""
    local = symmetric_trace_free(local_tide)
    memory = symmetric_trace_free(memory_tide)
    if local.shape != memory.shape:
        raise ValueError("local and memory tensors must have identical shapes")
    response_screen = _validated_screen(screen, local.shape[:-2])
    commutator = np.matmul(local, memory) - np.matmul(memory, local)
    numerator = np.sum(np.square(commutator), axis=(-2, -1))
    local_norm = np.sum(np.square(local), axis=(-2, -1))
    memory_norm = np.sum(np.square(memory), axis=(-2, -1))
    denominator = 2.0 * (1.0 + local_norm) * (1.0 + memory_norm)
    potential = response_screen * numerator / denominator
    if np.any(potential < -1e-14) or np.any(potential > 1.0 + 1e-12):
        raise ValueError("misalignment potential violated its analytic bound")
    return potential


def bounded_misalignment_gradients(
    local_tide: Any,
    memory_tide: Any,
    *,
    screen: Any = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Gradients with respect to the two STF tensors, holding the screen fixed."""
    local = symmetric_trace_free(local_tide)
    memory = symmetric_trace_free(memory_tide)
    if local.shape != memory.shape:
        raise ValueError("local and memory tensors must have identical shapes")
    response_screen = _validated_screen(screen, local.shape[:-2])
    commutator = np.matmul(local, memory) - np.matmul(memory, local)
    numerator = np.sum(np.square(commutator), axis=(-2, -1))
    local_norm = np.sum(np.square(local), axis=(-2, -1))
    memory_norm = np.sum(np.square(memory), axis=(-2, -1))
    denominator = 2.0 * (1.0 + local_norm) * (1.0 + memory_norm)

    numerator_local = 2.0 * (np.matmul(commutator, memory) - np.matmul(memory, commutator))
    numerator_memory = 2.0 * (np.matmul(local, commutator) - np.matmul(commutator, local))
    denominator_local = 4.0 * (1.0 + memory_norm)[..., None, None] * local
    denominator_memory = 4.0 * (1.0 + local_norm)[..., None, None] * memory
    common = response_screen[..., None, None] / np.square(denominator)[..., None, None]
    gradient_local = common * (
        numerator_local * denominator[..., None, None]
        - numerator[..., None, None] * denominator_local
    )
    gradient_memory = common * (
        numerator_memory * denominator[..., None, None]
        - numerator[..., None, None] * denominator_memory
    )
    return symmetric_trace_free(gradient_local), symmetric_trace_free(gradient_memory)


def spectral_tidal_misalignment(
    density: Any,
    *,
    spacing: float,
    gravitational_constant: float,
    a_sigma: float,
    memory_length: float,
) -> TidalMisalignmentField:
    """Construct the local tide, unscreened scale memory, and locally screened mismatch."""
    base = spectral_tidal_memory(
        density,
        spacing=spacing,
        gravitational_constant=gravitational_constant,
        a_sigma=a_sigma,
        memory_length=memory_length,
        screen_power=4.0,
        screen_order="after_memory",
    )
    curvature_scale = a_sigma / memory_length
    local = symmetric_trace_free(base.tidal / curvature_scale)
    memory = symmetric_trace_free(base.propagated_memory)
    commutator = tidal_commutator(local, memory)
    potential = bounded_misalignment_potential(local, memory, screen=base.screen)
    return TidalMisalignmentField(
        base=base,
        local_tide=local,
        memory_tide=memory,
        commutator=commutator,
        bounded_potential=potential,
    )
