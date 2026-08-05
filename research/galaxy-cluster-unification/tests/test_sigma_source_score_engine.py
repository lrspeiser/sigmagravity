from __future__ import annotations

import numpy as np

from voidscreen.sigma_source_score_engine import (
    gradient_support_mask,
    i4_draw_summary,
    i5_draw_summary,
    joint_variant_draw_pass_fraction,
    leave_one_region_out_stability,
    posterior_feature_summary,
    posterior_novelty_scores,
)


def test_gradient_support_requires_every_gradient() -> None:
    rng = np.random.default_rng(4)
    first_east = rng.normal(2.0, 0.1, (256, 3))
    first_north = rng.normal(0.0, 0.1, (256, 3))
    second_east = rng.normal(1.0, 0.1, (256, 3))
    second_north = rng.normal(0.0, 0.1, (256, 3))
    second_east[:, 1] = rng.normal(0.0, 1.0, 256)
    support = gradient_support_mask(
        [(first_east, first_north), (second_east, second_north)],
        minimum_detection_sigma=3.0,
    )
    assert support.tolist() == [True, False, True]


def test_i4_i5_draw_summaries_and_posterior_summary() -> None:
    plus = np.asarray([[1.0, 1.0], [1.1, 1.1], [0.9, 0.9]])
    cross = np.zeros_like(plus)
    i4 = i4_draw_summary(plus, cross, [True, True])
    assert np.allclose(i4["axis_deg"], 0.0)
    assert posterior_feature_summary(i4)["detection_sigma"] > 3.0
    i5 = i5_draw_summary(np.asarray([[0.2, 0.4], [0.3, 0.5]]), [True, True])
    assert np.allclose(i5["activation"], [0.3, 0.4])


def test_posterior_novelty_separates_controlled_and_independent_responses() -> None:
    rng = np.random.default_rng(19)
    draws, regions = 12, 80
    controls = rng.normal(size=(draws, regions, 5))
    controlled = 2.0 * controls[..., 0] - controls[..., 1]
    independent = rng.normal(size=(draws, regions))
    support = np.ones(regions, dtype=bool)
    controlled_score = posterior_novelty_scores(
        controls,
        controlled,
        support,
        minimum_unexplained_fraction=0.2,
    )
    independent_score = posterior_novelty_scores(
        controls,
        independent,
        support,
        minimum_unexplained_fraction=0.2,
    )
    assert controlled_score["pass_fraction"] == 0.0
    assert independent_score["pass_fraction"] > 0.9


def test_leave_one_out_and_variant_stability_are_candidate_aware() -> None:
    draws, regions = 20, 40
    plus = np.ones((draws, regions))
    cross = np.zeros_like(plus)
    response = np.stack([plus, cross], axis=-1)
    support = np.ones(regions, dtype=bool)
    loo = leave_one_region_out_stability(
        response,
        support,
        candidate="I4",
        maximum_activation_change_fraction=0.1,
        maximum_axis_change_deg=10.0,
    )
    assert loo["pass_fraction"] == 1.0
    primary = i4_draw_summary(plus, cross, support)
    close = i4_draw_summary(1.02 * plus, cross, support)
    rotated = i4_draw_summary(np.zeros_like(plus), plus, support)
    assert joint_variant_draw_pass_fraction(
        primary,
        [close],
        maximum_activation_change_fraction=0.1,
        maximum_axis_change_deg=10.0,
    ) == 1.0
    assert joint_variant_draw_pass_fraction(
        primary,
        [rotated],
        maximum_activation_change_fraction=0.1,
        maximum_axis_change_deg=10.0,
    ) == 0.0
