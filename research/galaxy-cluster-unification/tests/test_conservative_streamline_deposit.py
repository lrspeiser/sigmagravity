from __future__ import annotations

import numpy as np

from voidscreen.geometric_transport import symmetric_streamline_deposit


def gaussian(axis, x0, y0, width):
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return np.exp(-0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / width**2)


def inputs():
    axis = np.arange(-32.0, 33.0)
    flux_x = gaussian(axis, -5.0, 2.0, 5.0)
    flux_y = -0.4 * gaussian(axis, 6.0, -3.0, 8.0)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    norm = np.maximum(np.hypot(xx, yy), 1.0)
    direction_x = -xx / norm
    direction_y = -yy / norm
    lengths = 4.0 + 10.0 * gaussian(axis, 0.0, 0.0, 14.0)
    return flux_x, flux_y, direction_x, direction_y, lengths


def test_deposition_conserves_both_vector_component_sums():
    flux_x, flux_y, direction_x, direction_y, lengths = inputs()
    output_x, output_y, audit = symmetric_streamline_deposit(
        flux_x, flux_y, direction_x, direction_y, lengths, steps=8
    )
    assert np.isclose(np.sum(output_x), np.sum(flux_x), rtol=1e-12, atol=1e-12)
    assert np.isclose(np.sum(output_y), np.sum(flux_y), rtol=1e-12, atol=1e-12)
    assert audit["transport_flux_sum_relative_error"] < 1e-12
    assert audit["transport_is_source_conservative"] is True
    assert audit["transport_relative_change_RMS"] > 0.05


def test_deposition_is_invariant_to_direction_reversal():
    flux_x, flux_y, direction_x, direction_y, lengths = inputs()
    forward = symmetric_streamline_deposit(
        flux_x, flux_y, direction_x, direction_y, lengths, steps=8
    )
    reversed_direction = symmetric_streamline_deposit(
        flux_x, flux_y, -direction_x, -direction_y, lengths, steps=8
    )
    assert np.allclose(forward[0], reversed_direction[0], rtol=1e-12, atol=1e-12)
    assert np.allclose(forward[1], reversed_direction[1], rtol=1e-12, atol=1e-12)


def test_deposition_returns_finite_shape_preserving_fields():
    flux_x, flux_y, direction_x, direction_y, lengths = inputs()
    output_x, output_y, audit = symmetric_streamline_deposit(
        flux_x, flux_y, direction_x, direction_y, lengths, steps=8
    )
    assert output_x.shape == flux_x.shape
    assert output_y.shape == flux_y.shape
    assert np.isfinite(output_x).all()
    assert np.isfinite(output_y).all()
    assert audit["minimum_samples_per_source_cell"] >= 1.0
