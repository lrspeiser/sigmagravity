from __future__ import annotations

import numpy as np
import pytest
from scipy.special import ndtri

from voidscreen.sigma_gas_posterior import (
    cluster_sobol_uniforms,
    common_grid_axis,
    log_uniform_depth_factors,
    map_region_values,
    positive_profile_draws,
    quantile_summary,
    resample_bin_labels_to_physical_grid,
    smooth_masked_field,
)


def test_sobol_draws_are_reproducible_and_have_declared_latent_correlation() -> None:
    first = cluster_sobol_uniforms(4, 4096, 17, rank_correlation=0.7)
    second = cluster_sobol_uniforms(4, 4096, 17, rank_correlation=0.7)
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left, right)
        assert np.all((left > 0.0) & (left < 1.0))
    correlations = [
        np.corrcoef(ndtri(first[0][index]), ndtri(first[1][index]))[0, 1]
        for index in range(4)
    ]
    assert correlations == pytest.approx([0.7] * 4, abs=0.015)
    assert first[2].shape == (4096,)


def test_profile_draws_use_ordered_interval_or_frozen_full_bound() -> None:
    uniforms = np.array([0.1586552539, 0.5, 0.8413447461])
    draws, mode = positive_profile_draws(10.0, 8.0, 15.0, (1.0, 30.0), uniforms)
    assert mode == "asymmetric_log_profile"
    assert draws == pytest.approx([8.0, 10.0, 15.0], rel=1e-8)

    fallback, mode = positive_profile_draws(10.0, None, None, (1.0, 100.0), uniforms)
    assert mode == "full_frozen_log_bound_fallback"
    assert np.all((fallback > 1.0) & (fallback < 100.0))


def test_depth_draws_are_log_uniform_with_geometric_median() -> None:
    factors = log_uniform_depth_factors(np.array([0.25, 0.5, 0.75]), 0.5, 2.0)
    assert factors[1] == pytest.approx(1.0)
    assert factors[0] * factors[2] == pytest.approx(1.0)


def test_physical_grid_resampling_uses_east_left_and_north_up() -> None:
    binmap = np.arange(25).reshape(5, 5)
    axis = common_grid_axis(2.0, 1.0)
    sampled = resample_bin_labels_to_physical_grid(
        binmap,
        center_logical_x=3.0,
        center_logical_y=3.0,
        native_pixel_kpc=1.0,
        common_axis_kpc=axis,
    )
    assert sampled[2, 2] == binmap[2, 2]
    assert sampled[2, -1] == binmap[2, 0]
    assert sampled[-1, 2] == binmap[4, 2]


def test_region_mapping_preserves_missing_labels() -> None:
    labels = np.array([[0, 2], [-1, 5]])
    mapped = map_region_values(labels, np.array([0, 2]), np.array([10.0, 20.0]))
    assert mapped[0].tolist() == [10.0, 20.0]
    assert np.isnan(mapped[1]).all()


def test_masked_mass_smoothing_conserves_integral_and_missing_support() -> None:
    field = np.full((21, 21), np.nan)
    field[3:18, 3:18] = 0.0
    field[10, 10] = 100.0
    smoothed = smooth_masked_field(field, sigma_pixels=2.0, conserve_integral=True)
    valid = np.isfinite(field)
    assert np.sum(smoothed[valid]) == pytest.approx(np.sum(field[valid]), rel=1e-12)
    assert np.isnan(smoothed[~valid]).all()
    assert smoothed[10, 10] < field[10, 10]


def test_quantile_summary_is_regionwise() -> None:
    summary = quantile_summary(np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]))
    assert summary["median"].tolist() == [2.0, 20.0]
    assert summary["q05"].shape == (2,)


@pytest.mark.parametrize(
    "call",
    [
        lambda: cluster_sobol_uniforms(1, 1000, 1, rank_correlation=0.0),
        lambda: positive_profile_draws(2.0, 1.0, 3.0, (2.0, 4.0), [0.5]),
        lambda: common_grid_axis(10.0, 3.0),
        lambda: smooth_masked_field(np.full((3, 3), np.nan), sigma_pixels=1.0, conserve_integral=False),
    ],
)
def test_invalid_posterior_inputs_fail_closed(call) -> None:
    with pytest.raises(ValueError):
        call()
