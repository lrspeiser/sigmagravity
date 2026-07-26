import numpy as np
import pandas as pd

from sigma_sprint.cluster_audit import (
    _weighted_radial_residual_slope,
    cluster_bootstrap_radial_trend,
    fit_constant_B,
)
from sigma_sprint.model import predict_acceleration


def test_constant_B_recovered_from_grouped_synthetic_data():
    gbar = np.logspace(-12, -9, 24)
    B = 4.25
    frame = pd.DataFrame(
        {
            "cluster": np.repeat(["a", "b", "c"], 8),
            "gbar": gbar,
            "gtot": predict_acceleration(gbar, B),
            "log_gbar": np.log10(gbar),
            "log_gtot": np.log10(predict_acceleration(gbar, B)),
            "err_log_gbar": 0.03,
            "err_log_gtot": 0.05,
            "radius_kpc": np.tile(np.arange(1, 9) * 50.0, 3),
        }
    )
    fit = fit_constant_B(frame)
    assert abs(fit.parameters["B"] - B) < 1e-7


def test_cluster_bootstrap_radial_trend_preserves_positive_slope():
    frame = pd.DataFrame(
        {
            "cluster": np.repeat(["a", "b", "c"], 4),
            "radius_kpc": np.tile([50.0, 100.0, 200.0, 400.0], 3),
            "residual_dex": np.tile([-0.2, -0.1, 0.0, 0.1], 3),
            "sigma_residual_dex": 0.05,
        }
    )
    assert _weighted_radial_residual_slope(frame) > 0
    result = cluster_bootstrap_radial_trend(frame, draws=20, seed=7)
    assert result["bootstrap_95_percent_interval"][0] > 0
