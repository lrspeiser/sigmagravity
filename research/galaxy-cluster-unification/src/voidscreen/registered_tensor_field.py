"""Projected scalar/tensor AQUAL pairs sourced by registered baryonic maps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voidscreen.geometric_transport import KPC_M, thin_sheet_newtonian_field
from voidscreen.physical_tensor_activation import (
    PhysicalTensorActivation2D,
    exact_physical_tensor_activation,
)
from voidscreen.tensor_aqual import (
    TensorAQUAL2DSolution,
    solve_projected_tensor_aqual,
    tensor_graph_laplacian,
)


@dataclass(frozen=True)
class RegisteredTensorFieldPair:
    newtonian_potential: np.ndarray
    source: np.ndarray
    activation: PhysicalTensorActivation2D
    scalar: TensorAQUAL2DSolution
    tensor: TensorAQUAL2DSolution
    tensor_effect_relative_rms: float
    scalar_newtonian_enhancement_rms: float
    tensor_normalized_curl_rms: float


def constant_mu(values: np.ndarray) -> np.ndarray:
    return np.ones_like(np.asarray(values, dtype=float))


def projected_source_from_newtonian_potential(
    newtonian_potential: np.ndarray,
    spacing_m: float,
) -> np.ndarray:
    """Construct the source that exactly recovers ``Phi_N`` for ``mu=1``."""

    potential = np.asarray(newtonian_potential, dtype=float)
    if potential.ndim != 2 or min(potential.shape) < 9 or spacing_m <= 0.0:
        raise ValueError("potential must be a 2D map and spacing positive")
    ones = np.ones_like(potential)
    zeros = np.zeros_like(potential)
    laplacian = tensor_graph_laplacian(
        ones,
        zeros,
        ones,
        zeros,
        spacing_m,
    )
    return (-laplacian @ potential.ravel()).reshape(potential.shape)


def central_mask(shape: tuple[int, int], border_fraction: float) -> np.ndarray:
    if not 0.0 <= border_fraction < 0.45:
        raise ValueError("border_fraction must lie in [0,0.45)")
    border = max(int(np.floor(min(shape) * float(border_fraction))), 1)
    mask = np.zeros(shape, dtype=bool)
    mask[border:-border, border:-border] = True
    return mask


def vector_relative_rms(
    first_x: np.ndarray,
    first_y: np.ndarray,
    reference_x: np.ndarray,
    reference_y: np.ndarray,
    mask: np.ndarray,
) -> float:
    numerator = float(
        np.sqrt(np.mean((first_x[mask] - reference_x[mask]) ** 2 + (first_y[mask] - reference_y[mask]) ** 2))
    )
    denominator = float(
        np.sqrt(np.mean(reference_x[mask] ** 2 + reference_y[mask] ** 2))
    )
    return numerator / max(denominator, np.finfo(float).tiny)


def normalized_acceleration_curl(
    acceleration_x: np.ndarray,
    acceleration_y: np.ndarray,
    spacing_m: float,
    mask: np.ndarray,
) -> float:
    curl = np.gradient(acceleration_y, spacing_m, axis=1, edge_order=2) - np.gradient(
        acceleration_x,
        spacing_m,
        axis=0,
        edge_order=2,
    )
    divergence = np.gradient(acceleration_x, spacing_m, axis=1, edge_order=2) + np.gradient(
        acceleration_y,
        spacing_m,
        axis=0,
        edge_order=2,
    )
    return float(np.sqrt(np.mean(curl[mask] ** 2))) / max(
        float(np.sqrt(np.mean(divergence[mask] ** 2))),
        np.finfo(float).tiny,
    )


def solve_registered_tensor_field_pair(
    stars_msun_kpc2: np.ndarray,
    gas_msun_kpc2: np.ndarray,
    cell_kpc: float,
    *,
    a0_m_s2: float = 1.2e-10,
    coherence_length_kpc: float = 10.0,
    coherence_power: float = 2.0,
    border_fraction: float = 0.1,
    residual_tolerance: float = 1e-5,
    maximum_nonlinear_iterations: int = 80,
    maximum_linear_iterations: int = 5000,
    linear_relative_tolerance: float = 1e-10,
    damping: float = 0.65,
    mu_floor: float = 1e-6,
) -> RegisteredTensorFieldPair:
    stars = np.asarray(stars_msun_kpc2, dtype=float)
    gas = np.asarray(gas_msun_kpc2, dtype=float)
    newtonian = thin_sheet_newtonian_field(stars + gas, cell_kpc)
    activation = exact_physical_tensor_activation(
        stars,
        gas,
        cell_kpc,
        a0_m_s2=a0_m_s2,
        coherence_length_kpc=coherence_length_kpc,
        coherence_power=coherence_power,
        mu_floor=mu_floor,
    )
    spacing_m = float(cell_kpc) * KPC_M
    source = projected_source_from_newtonian_potential(
        newtonian.potential_m2_s2,
        spacing_m,
    )
    common = {
        "a0": float(a0_m_s2),
        "residual_tolerance": float(residual_tolerance),
        "maximum_nonlinear_iterations": int(maximum_nonlinear_iterations),
        "maximum_linear_iterations": int(maximum_linear_iterations),
        "linear_relative_tolerance": float(linear_relative_tolerance),
        "damping": float(damping),
        "mu_floor": float(mu_floor),
    }
    scalar = solve_projected_tensor_aqual(
        source,
        spacing_m,
        newtonian.potential_m2_s2,
        np.zeros_like(activation.sigma),
        activation.transport_direction_x,
        activation.transport_direction_y,
        **common,
    )
    tensor = solve_projected_tensor_aqual(
        source,
        spacing_m,
        newtonian.potential_m2_s2,
        activation.sigma,
        activation.transport_direction_x,
        activation.transport_direction_y,
        **common,
    )
    mask = central_mask(stars.shape, border_fraction)
    effect = vector_relative_rms(
        tensor.acceleration_x,
        tensor.acceleration_y,
        scalar.acceleration_x,
        scalar.acceleration_y,
        mask,
    )
    enhancement = float(
        np.sqrt(np.mean(scalar.acceleration_x[mask] ** 2 + scalar.acceleration_y[mask] ** 2))
        / max(
            float(
                np.sqrt(
                    np.mean(
                        newtonian.acceleration_x_m_s2[mask] ** 2
                        + newtonian.acceleration_y_m_s2[mask] ** 2
                    )
                )
            ),
            np.finfo(float).tiny,
        )
    )
    curl = normalized_acceleration_curl(
        tensor.acceleration_x,
        tensor.acceleration_y,
        spacing_m,
        mask,
    )
    return RegisteredTensorFieldPair(
        newtonian_potential=newtonian.potential_m2_s2,
        source=source,
        activation=activation,
        scalar=scalar,
        tensor=tensor,
        tensor_effect_relative_rms=float(effect),
        scalar_newtonian_enhancement_rms=enhancement,
        tensor_normalized_curl_rms=curl,
    )
