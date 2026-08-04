from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voidscreen.sigma_v9b_local_state_closure import (
    audit_v9b_local_state_closure,
    local_state_overlap_metrics,
    scale_equivalent_spherical_pair,
)

ROOT = Path(__file__).resolve().parents[1]


def test_scale_equivalent_pair_has_same_surface_field_but_different_environment() -> None:
    pair = scale_equivalent_spherical_pair(mass_ratio=100.0)
    assert pair["second_radius"] == pytest.approx(10.0)
    assert pair["surface_field_ratio_second_to_first"] == pytest.approx(1.0)
    assert pair["potential_depth_ratio_second_to_first"] == pytest.approx(10.0)
    assert pair["tidal_or_mean_density_ratio_second_to_first"] == pytest.approx(0.1)


def test_overlap_metric_detects_same_acceleration_different_enhancement() -> None:
    sparc_g = np.linspace(-12.0, -9.0, 301)
    sparc_e = 0.5 + 0.05 * (sparc_g + 10.5)
    cluster_g = np.linspace(-10.8, -9.7, 25)
    cluster_e = 0.55 + 0.05 * (cluster_g + 10.5) + 0.5
    result = local_state_overlap_metrics(
        sparc_log_gbar=sparc_g,
        sparc_log_enhancement=sparc_e,
        cluster_log_gbar=cluster_g,
        cluster_log_enhancement=cluster_e,
        nearest_neighbors=5,
    )
    assert result["cluster_fraction_inside_SPARC_range"] == pytest.approx(1.0)
    assert result["nearest_log_gbar_distance_dex"]["maximum"] <= 0.0051
    assert result["neighbor_median_required_enhancement_gap_dex"][
        "median"
    ] == pytest.approx(0.55, abs=0.002)
    assert result["local_first_gradient_conflict_gate"]


def test_actual_spent_development_products_pass_declared_closure_gate() -> None:
    report = audit_v9b_local_state_closure(
        sparc_predictions_path=(
            ROOT / "results" / "sparc_independent_nuisance_refit" / "point_predictions.csv"
        ),
        cluster_sample_path=(
            ROOT / "results" / "phenomenology_formula_sweep" / "sample.csv"
        ),
        nearest_neighbors=10,
    )
    overlap = report["development_overlap"]
    assert report["closure_gate_passed"]
    assert overlap["SPARC_points"] == 968
    assert overlap["cluster_points"] == 72
    assert overlap["cluster_fraction_inside_SPARC_range"] == pytest.approx(1.0)
    assert overlap["nearest_log_gbar_distance_dex"]["median"] == pytest.approx(
        0.00144787001, rel=2.0e-6
    )
    assert overlap["nearest_required_enhancement_gap_dex"]["median"] == pytest.approx(
        0.50933966, rel=2.0e-6
    )
    assert overlap["fraction_cluster_gap_above_0p2_dex"] == pytest.approx(
        70.0 / 72.0
    )
    assert report["existing_spent_observational_products_accessed"]
    assert not report["new_observational_product_accessed"]
    assert not report["new_holdout_opened"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "sparc_log_gbar": [],
                "sparc_log_enhancement": [],
                "cluster_log_gbar": [1.0],
                "cluster_log_enhancement": [1.0],
            },
            "non-empty",
        ),
        (
            {
                "sparc_log_gbar": [1.0, 2.0],
                "sparc_log_enhancement": [1.0],
                "cluster_log_gbar": [1.0],
                "cluster_log_enhancement": [1.0],
            },
            "match",
        ),
    ],
)
def test_invalid_overlap_inputs_are_rejected(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        local_state_overlap_metrics(**kwargs)


def test_invalid_scale_pair_is_rejected() -> None:
    with pytest.raises(ValueError):
        scale_equivalent_spherical_pair(mass_ratio=0.0)
