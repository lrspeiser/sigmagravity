from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from voidscreen.sigma_tidal_misalignment import _validated_screen
from voidscreen.sigma_triaxial_memory import (
    TidalMemoryField,
    spectral_tidal_memory,
    symmetric_trace_free,
)


@dataclass(frozen=True)
class ScaleHomologyField:
    """Local and remembered tidal tensors with their bounded Gram response."""

    base: TidalMemoryField
    local_tide: np.ndarray
    memory_tide: np.ndarray
    inner_product: np.ndarray
    gram_determinant: np.ndarray
    bounded_potential: np.ndarray


def homology_invariants(
    local_tide: Any,
    memory_tide: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the two norms, inner product, and Gram determinant."""
    local = symmetric_trace_free(local_tide)
    memory = symmetric_trace_free(memory_tide)
    if local.shape != memory.shape:
        raise ValueError("local and memory tensors must have identical shapes")
    local_norm = np.sum(np.square(local), axis=(-2, -1))
    memory_norm = np.sum(np.square(memory), axis=(-2, -1))
    inner = np.sum(local * memory, axis=(-2, -1))
    gram = local_norm * memory_norm - np.square(inner)
    return local_norm, memory_norm, inner, gram


def bounded_homology_potential(
    local_tide: Any,
    memory_tide: Any,
    *,
    screen: Any = 1.0,
) -> np.ndarray:
    """Return the locally screened, Cauchy-Schwarz-bounded Gram potential."""
    local = symmetric_trace_free(local_tide)
    memory = symmetric_trace_free(memory_tide)
    if local.shape != memory.shape:
        raise ValueError("local and memory tensors must have identical shapes")
    response_screen = _validated_screen(screen, local.shape[:-2])
    local_norm, memory_norm, _, gram = homology_invariants(local, memory)
    roundoff_scale = np.maximum(local_norm * memory_norm, 1.0)
    if np.any(gram < -2e-12 * roundoff_scale):
        raise ValueError("Gram determinant violated Cauchy-Schwarz")
    denominator = (1.0 + local_norm) * (1.0 + memory_norm)
    potential = response_screen * np.maximum(gram, 0.0) / denominator
    if np.any(potential < -1e-14) or np.any(potential > 1.0 + 1e-12):
        raise ValueError("scale-homology potential violated its analytic bound")
    return potential


def bounded_homology_gradients(
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
    local_norm, memory_norm, inner, gram = homology_invariants(local, memory)
    denominator = (1.0 + local_norm) * (1.0 + memory_norm)
    numerator_local = (
        2.0 * memory_norm[..., None, None] * local - 2.0 * inner[..., None, None] * memory
    )
    numerator_memory = (
        2.0 * local_norm[..., None, None] * memory - 2.0 * inner[..., None, None] * local
    )
    denominator_local = 2.0 * (1.0 + memory_norm)[..., None, None] * local
    denominator_memory = 2.0 * (1.0 + local_norm)[..., None, None] * memory
    common = response_screen[..., None, None] / np.square(denominator)[..., None, None]
    gradient_local = common * (
        numerator_local * denominator[..., None, None] - gram[..., None, None] * denominator_local
    )
    gradient_memory = common * (
        numerator_memory * denominator[..., None, None] - gram[..., None, None] * denominator_memory
    )
    return symmetric_trace_free(gradient_local), symmetric_trace_free(gradient_memory)


def spectral_scale_homology(
    density: Any,
    *,
    spacing: float,
    gravitational_constant: float,
    a_sigma: float,
    memory_length: float,
) -> ScaleHomologyField:
    """Construct the local tide, unscreened scale memory, and screened Gram response."""
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
    local_norm, memory_norm, inner, gram = homology_invariants(local, memory)
    del local_norm, memory_norm
    potential = bounded_homology_potential(local, memory, screen=base.screen)
    return ScaleHomologyField(
        base=base,
        local_tide=local,
        memory_tide=memory,
        inner_product=inner,
        gram_determinant=gram,
        bounded_potential=potential,
    )
