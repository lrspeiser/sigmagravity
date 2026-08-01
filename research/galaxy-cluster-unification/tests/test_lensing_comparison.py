import numpy as np
import pytest

from voidscreen.lensing_comparison import (
    equal_system_rmse,
    lensing_metrics,
    paired_system_bootstrap,
)


def test_equal_system_rmse_does_not_overweight_system_with_more_points():
    systems = np.array(["a", "a", "a", "b"])
    residual = np.array([1.0, 1.0, 1.0, 3.0])
    assert equal_system_rmse(systems, residual) == pytest.approx(np.sqrt(5.0))


def test_metrics_report_multiplier_and_symmetric_factor_coverage():
    residual = np.log10(np.array([0.5, 1.0, 2.0]))
    metrics = lensing_metrics(["a", "a", "b"], residual)
    assert metrics["posthoc_multiplier_to_remove_mean_bias"] == pytest.approx(1.0)
    assert metrics["coverage_within_symmetric_factor"]["2.00"] == pytest.approx(1.0)
    assert metrics["median_absolute_error_factor"] == pytest.approx(2.0)


def test_paired_system_bootstrap_detects_uniformly_better_candidate():
    systems = np.array(["a", "a", "b", "b", "c", "c"])
    candidate = np.full(6, 0.1)
    reference = np.full(6, 0.3)
    result = paired_system_bootstrap(
        systems, candidate, reference, draws=500, seed=7
    )
    assert result["observed_delta_dex"] == pytest.approx(-0.2)
    assert result["probability_candidate_better"] == pytest.approx(1.0)
