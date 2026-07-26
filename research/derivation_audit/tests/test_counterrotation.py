import pandas as pd
import pytest

from sigma_sprint.counterrotation import greedy_match_controls


def _frame(prefix: str, count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mangaid": [f"{prefix}-{index}" for index in range(count)],
            "plateifu": [f"plate-{prefix}-{index}" for index in range(count)],
            "log_stellar_mass": [10.0 + 0.01 * index for index in range(count)],
            "log_Re_kpc": [0.5 + 0.01 * index for index in range(count)],
            "sersic_n": [2.0] * count,
            "axis_ratio": [0.7] * count,
            "inclination_deg": [50.0] * count,
            "redshift": [0.03] * count,
            "jam_chi2_dof": [0.05] * count,
            "fdm_Re_secondary": [0.2] * count,
        }
    )


def test_matching_is_without_replacement():
    matches = greedy_match_controls(_frame("case", 2), _frame("control", 10))
    assert len(matches) == 10
    assert matches["control_mangaid"].nunique() == 10


def test_matching_rejects_outcome_leakage():
    with pytest.raises(ValueError):
        greedy_match_controls(
            _frame("case", 1),
            _frame("control", 5),
            features=["fdm_Re_secondary"],
        )
