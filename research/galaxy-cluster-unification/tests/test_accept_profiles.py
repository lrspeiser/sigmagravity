from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from voidscreen.accept_profiles import (
    ACCEPT_COLUMNS,
    interpolate_electron_density_cm3,
    load_accept_profiles,
)


def test_load_accept_profiles_parses_midpoint(tmp_path) -> None:
    row = "TEST 0.1 0.3 1e-3 1e-4 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\n"
    path = tmp_path / "profiles.dat"
    path.write_text("# ignored\n" + row, encoding="utf-8")
    frame = load_accept_profiles(path)
    assert tuple(frame.columns[:-1]) == ACCEPT_COLUMNS
    assert frame.loc[0, "radius_kpc"] == 200.0


def test_log_interpolation_recovers_power_law() -> None:
    profile = pd.DataFrame(
        {"radius_kpc": [10.0, 100.0], "nelec_cm3": [1.0e-2, 1.0e-4]}
    )
    value = interpolate_electron_density_cm3(profile, [np.sqrt(1000.0)])
    assert np.isclose(value[0], 1.0e-3)


def test_interpolation_rejects_extrapolation() -> None:
    profile = pd.DataFrame(
        {"radius_kpc": [10.0, 100.0], "nelec_cm3": [1.0e-2, 1.0e-4]}
    )
    with pytest.raises(ValueError, match="extrapolate"):
        interpolate_electron_density_cm3(profile, [101.0])
