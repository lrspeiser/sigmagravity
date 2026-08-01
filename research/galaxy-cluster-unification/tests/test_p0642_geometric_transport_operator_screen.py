from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from voidscreen.geometric_transport import (
    component_cancellation,
    normalized_discrete_curl,
    spectral_poisson_acceleration_2d,
    thin_sheet_newtonian_field,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0642_geometric_transport_operator_screen"


def report() -> dict:
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def test_negative_result_is_preserved_without_opening_sealed_targets():
    result = report()
    assert result["status"] == "fail"
    assert result["selected_operator"] is None
    assert result["provisional_operator_before_universal_gates"] == "component_cancellation"
    assert result["sealed_P0633_kinematics_opened"] is False
    assert result["sealed_P0640_lensing_constraints_opened"] is False
    assert result["selection_used_sealed_target_outcomes"] is False


def test_registered_map_coverage_and_no_per_object_gravity_parameters():
    result = report()
    assert result["coverage"]["registered_galaxies"] == 13
    assert result["coverage"]["registered_clusters"] == 4
    assert result["coverage"]["per_object_gravity_parameters"] == 0
    scores = pd.read_csv(RESULTS / "registered_map_operator_scores.csv")
    assert len(scores) == 17 * 3
    assert np.isfinite(scores.select_dtypes(include=[float, int])).all().all()


def test_observed_geometry_does_not_artificially_separate_clusters():
    ratios = report()["cluster_to_galaxy_median_activation_ratio"]
    assert set(ratios) == {"path_incoherence", "component_cancellation", "hybrid"}
    assert max(ratios.values()) < 1.0


def test_identical_component_fields_have_zero_cancellation():
    axis = np.linspace(-5.0, 5.0, 65)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    surface = np.exp(-(xx * xx + yy * yy))
    field = thin_sheet_newtonian_field(surface, axis[1] - axis[0])
    cancellation = component_cancellation(field, field)
    assert float(np.max(cancellation)) < 1e-12


def test_spectral_poisson_output_is_conservative_away_from_edges():
    axis = np.linspace(-5.0, 5.0, 129)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    source = np.exp(-0.5 * ((xx - 1.0) ** 2 + yy**2))
    source -= np.mean(source)
    _, ax, ay = spectral_poisson_acceleration_2d(source, axis[1] - axis[0])
    assert normalized_discrete_curl(ax, ay, axis[1] - axis[0]) < 0.08


def test_machine_readable_outputs_and_figure_exist():
    assert (RESULTS / "synthetic_operator_scores.csv").stat().st_size > 1000
    assert (RESULTS / "domain_summary.csv").stat().st_size > 200
    assert (RESULTS / "geometric_operator_screen.png").stat().st_size > 20000
