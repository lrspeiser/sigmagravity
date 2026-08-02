from __future__ import annotations

import numpy as np

from voidscreen.geometric_transport import symmetric_field_line_diffusion


def inputs():
    axis = np.arange(-32.0, 33.0)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    radius = np.hypot(xx, yy)
    flux_x = np.exp(-0.5 * ((xx + 5.0) ** 2 + (yy - 2.0) ** 2) / 5.0**2)
    flux_y = -0.4 * np.exp(-0.5 * ((xx - 6.0) ** 2 + (yy + 3.0) ** 2) / 8.0**2)
    norm = np.maximum(radius, 1.0)
    direction_x = -xx / norm
    direction_y = -yy / norm
    lengths = 3.0 + 8.0 * np.exp(-0.5 * radius**2 / 14.0**2)
    conductance = np.ones_like(radius)
    transition = (radius > 22.0) & (radius < 27.0)
    conductance[transition] = 0.5 * (1.0 + np.cos(np.pi * (radius[transition] - 22.0) / 5.0))
    conductance[radius >= 27.0] = 0.0
    flux_x *= conductance
    flux_y *= conductance
    return flux_x, flux_y, direction_x, direction_y, lengths, conductance


def solve(direction_sign=1.0):
    flux_x, flux_y, direction_x, direction_y, lengths, conductance = inputs()
    return (
        symmetric_field_line_diffusion(
            flux_x,
            flux_y,
            direction_sign * direction_x,
            direction_sign * direction_y,
            lengths,
            conductance,
        ),
        flux_x,
        flux_y,
        conductance,
    )


def test_diffusion_conserves_vector_flux_and_has_no_component_overshoot():
    (output_x, output_y, audit), flux_x, flux_y, _ = solve()
    assert np.isclose(np.sum(output_x), np.sum(flux_x), rtol=1e-9, atol=1e-9)
    assert np.isclose(np.sum(output_y), np.sum(flux_y), rtol=1e-9, atol=1e-9)
    assert audit["transport_flux_sum_relative_error"] < 1e-9
    assert audit["transport_component_overshoot_fraction"] < 1e-10
    assert audit["transport_is_source_conservative"] is True
    assert audit["transport_is_self_adjoint_diffusion"] is True


def test_diffusion_is_direction_reversal_invariant():
    forward = solve(1.0)[0]
    reversed_direction = solve(-1.0)[0]
    assert np.allclose(forward[0], reversed_direction[0], rtol=1e-9, atol=1e-9)
    assert np.allclose(forward[1], reversed_direction[1], rtol=1e-9, atol=1e-9)


def test_diffusion_is_nontrivial_smoothing_with_zero_outer_support():
    (output_x, output_y, audit), _, _, conductance = solve()
    assert audit["transport_relative_change_RMS"] > 0.05
    assert audit["transport_flux_RMS_ratio"] < 1.0
    assert audit["transport_graph_edges"] > 1000
    assert audit["transport_solver_information_x"] == 0
    assert audit["transport_solver_information_y"] == 0
    assert np.max(np.abs(output_x[conductance == 0.0])) == 0.0
    assert np.max(np.abs(output_y[conductance == 0.0])) == 0.0
