from __future__ import annotations

import numpy as np

from voidscreen.geometric_transport import KPC_M, thin_sheet_newtonian_field
from voidscreen.registered_tensor_field import (
    constant_mu,
    projected_source_from_newtonian_potential,
    solve_registered_tensor_field_pair,
)
from voidscreen.tensor_aqual import solve_projected_tensor_aqual


def maps(cells=33):
    axis = np.linspace(-4.0, 4.0, cells)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    stars = np.exp(-0.5 * (((xx + 0.4) / 0.45) ** 2 + (yy / 0.7) ** 2))
    gas = np.exp(-0.5 * (((xx - 0.6) / 0.9) ** 2 + (yy / 1.1) ** 2))
    cell = float(axis[1] - axis[0])
    stars *= 3.0e9 / (np.sum(stars) * cell**2)
    gas *= 7.0e9 / (np.sum(gas) * cell**2)
    return cell, stars, gas


def test_projected_source_recovers_newtonian_potential_in_constant_mu_limit():
    cell, stars, gas = maps()
    field = thin_sheet_newtonian_field(stars + gas, cell)
    spacing = cell * KPC_M
    source = projected_source_from_newtonian_potential(field.potential_m2_s2, spacing)
    solution = solve_projected_tensor_aqual(
        source,
        spacing,
        field.potential_m2_s2,
        np.zeros_like(stars),
        np.ones_like(stars),
        np.zeros_like(stars),
        a0=1.2e-10,
        mu_function=constant_mu,
    )
    error = np.sqrt(np.mean((solution.potential - field.potential_m2_s2) ** 2))
    scale = np.sqrt(np.mean(field.potential_m2_s2**2))
    assert error / scale < 1e-9


def test_scalar_and_tensor_field_pair_converges_and_is_conservative():
    cell, stars, gas = maps()
    pair = solve_registered_tensor_field_pair(stars, gas, cell)
    assert pair.scalar.converged
    assert pair.tensor.converged
    assert pair.scalar.normalized_residual_rms < 1e-5
    assert pair.tensor.normalized_residual_rms < 1e-5
    assert pair.tensor_effect_relative_rms > 0.0
    assert pair.scalar_newtonian_enhancement_rms > 1.0
    assert pair.tensor_normalized_curl_rms < 1e-10
