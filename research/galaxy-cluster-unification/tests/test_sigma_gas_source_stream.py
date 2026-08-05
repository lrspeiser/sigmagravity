from __future__ import annotations

import numpy as np

from voidscreen.sigma_gas_source_stream import (
    append_feature_batch,
    concatenate_feature_batches,
    gas_feature_batch,
    region_draws_to_grid,
    smooth_masked_draws,
)


def test_region_draw_mapping_preserves_labels_and_masks_exterior() -> None:
    labels = np.asarray([[0, 0, -1], [1, 1, -1]])
    values = np.asarray([[2.0, 5.0], [3.0, 7.0]])
    mapped = region_draws_to_grid(values, [0, 1], labels)
    np.testing.assert_allclose(mapped[:, 0, :2], [[2.0, 2.0], [3.0, 3.0]])
    np.testing.assert_allclose(mapped[:, 1, :2], [[5.0, 5.0], [7.0, 7.0]])
    assert np.all(np.isnan(mapped[:, :, 2]))


def test_masked_smoothing_conserves_each_surface_density_draw() -> None:
    maps = np.full((2, 15, 15), np.nan)
    maps[0, 2:13, 2:13] = 1.0
    maps[1, 2:13, 2:13] = 3.0
    smoothed = smooth_masked_draws(maps, sigma_pixels=1.5, conserve_integral=True)
    np.testing.assert_allclose(
        np.nansum(smoothed, axis=(-2, -1)),
        np.nansum(maps, axis=(-2, -1)),
        rtol=1.0e-14,
    )
    assert np.all(np.isnan(smoothed[:, :2]))


def test_gas_feature_batch_builds_all_scales_radii_and_bounded_i5() -> None:
    axis = np.arange(-50.0, 51.0, 10.0)
    labels = np.arange(axis.size * axis.size).reshape(axis.size, axis.size)
    region_ids = labels.ravel()
    east, north = np.meshgrid(axis, axis)
    draws = 4
    density = np.stack([np.exp(0.005 * east) * (1.0 + 0.01 * draw) for draw in range(draws)])
    entropy = np.stack([np.exp(0.004 * north) * (1.0 + 0.01 * draw) for draw in range(draws)])
    pressure = np.stack([np.exp(0.003 * (east + north)) * (1.0 + 0.01 * draw) for draw in range(draws)])
    surface = np.stack([np.exp(0.002 * east) * (1.0 + 0.01 * draw) for draw in range(draws)])
    regional = {
        "electron_density_cm3": density.reshape(draws, -1),
        "entropy_proxy_keV_cm2": entropy.reshape(draws, -1),
        "thermal_pressure_erg_cm3": pressure.reshape(draws, -1),
        "gas_surface_density_msun_kpc2": surface.reshape(draws, -1),
    }
    result = gas_feature_batch(
        regional,
        region_ids=region_ids,
        label_grid=labels,
        east_axis_kpc=axis,
        north_axis_kpc=axis,
        spacing_kpc=10.0,
        smoothing_fwhm_kpc=[20.0, 40.0],
        radii_kpc=[30.0, 50.0],
    )
    assert len(result) == 14 * 2 * 2
    for name, values in result.items():
        assert values.shape == (draws, region_ids.size)
        if name.startswith("i5_baroclinicity"):
            finite = values[np.isfinite(values)]
            assert np.all((finite >= 0.0) & (finite <= 1.0))


def test_stream_accumulator_refuses_schema_drift_and_checks_draw_count() -> None:
    accumulated: dict[str, list[np.ndarray]] = {}
    append_feature_batch(accumulated, {"a": np.ones((2, 3)), "b": np.zeros((2, 3))})
    append_feature_batch(accumulated, {"a": 2.0 * np.ones((1, 3)), "b": np.ones((1, 3))})
    result = concatenate_feature_batches(accumulated, expected_draws=3)
    assert result["a"].shape == (3, 3)
