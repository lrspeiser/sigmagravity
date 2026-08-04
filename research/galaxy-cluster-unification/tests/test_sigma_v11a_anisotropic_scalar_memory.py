from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_v11a_anisotropic_scalar_memory import (
    audit_v11a_anisotropic_scalar_memory,
    bounded_alignment,
    effective_memory_speed_squared,
    mixed_speed_squared_roots,
)


def test_bounded_alignment_is_regular_monotone_and_saturating() -> None:
    ratios = np.array([0.0, 1.0e-6, 0.1, 1.0, 10.0, 1.0e6])
    values = bounded_alignment(ratios)
    assert values[0] == 0.0
    assert values[3] == pytest.approx(0.5)
    assert np.all(np.diff(values) > 0.0)
    assert values[-1] == pytest.approx(1.0, abs=2.0e-12)
    assert bounded_alignment(1.0e308) == pytest.approx(1.0)


def test_effective_memory_speed_has_exact_global_bounds() -> None:
    speed = 3.0 / 11.0
    fraction = 0.25
    assert effective_memory_speed_squared(
        0.0, 1.0, maximum_speed_squared=speed, anisotropy_fraction=fraction
    ) == pytest.approx(speed)
    assert effective_memory_speed_squared(
        1.0e12,
        1.0,
        maximum_speed_squared=speed,
        anisotropy_fraction=fraction,
    ) == pytest.approx(9.0 / 44.0)
    assert effective_memory_speed_squared(
        1.0e12,
        0.0,
        maximum_speed_squared=speed,
        anisotropy_fraction=fraction,
    ) == pytest.approx(speed)


def test_selected_endpoint_cones_are_positive_and_causal() -> None:
    zero = mixed_speed_squared_roots(
        aether_speed_squared=3.0 / 4.0,
        memory_speed_squared=3.0 / 11.0,
        normalized_mixing_squared=2.0 / 11.0,
    )
    saturated = mixed_speed_squared_roots(
        aether_speed_squared=3.0 / 4.0,
        memory_speed_squared=9.0 / 44.0,
        normalized_mixing_squared=2.0 / 11.0,
    )
    assert zero == pytest.approx([9.0 / 44.0, 1.0])
    assert np.all(saturated > 0.0)
    assert np.all(saturated < 1.0)


def test_v11a_selection_passes_without_observational_data() -> None:
    report = audit_v11a_anisotropic_scalar_memory(
        k_b=1.0,
        aether_speed_squared=3.0 / 4.0,
        maximum_memory_speed_squared=3.0 / 11.0,
        normalized_mixing_squared=2.0 / 11.0,
        anisotropy_fraction=1.0 / 4.0,
        physical_parameter_count=5,
        maximum_physical_parameters=5,
        ratio_scan_maximum=1.0e8,
        ratio_scan_samples=101,
        angle_scan_samples=51,
    )
    assert all(report["selection_gates"].values())
    assert report["analytic_bounds"][
        "minimum_memory_speed_squared"
    ] == pytest.approx(9.0 / 44.0)
    assert report["analytic_bounds"][
        "minimum_static_schur_margin"
    ] == pytest.approx(1.0 / 44.0)
    assert report["scan"]["maximum_root"] <= 1.0 + 1.0e-12
    assert not report["observational_data_accessed"]
    assert not report["raw_holdout_opened"]
    assert not report["unresolved"]["nonlinear_ADM_constraint_and_global_rank"]


def test_invalid_v11a_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        bounded_alignment(-1.0)
    with pytest.raises(ValueError):
        effective_memory_speed_squared(
            1.0,
            1.1,
            maximum_speed_squared=3.0 / 11.0,
            anisotropy_fraction=0.25,
        )
    with pytest.raises(ValueError):
        mixed_speed_squared_roots(
            aether_speed_squared=0.75,
            memory_speed_squared=-0.1,
            normalized_mixing_squared=2.0 / 11.0,
        )
