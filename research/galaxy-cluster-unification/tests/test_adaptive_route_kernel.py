import numpy as np
import pytest

from voidscreen.adaptive_route_kernel import (
    adaptive_route_parameters,
    extent_coordinate,
    multiplicity_gate,
    transformed_source_weights,
)


def test_source_reweighting_preserves_budget_and_changes_dominance():
    weights = np.array([0.1, 0.2, 0.7])
    shallow = transformed_source_weights(weights, 0.5)
    steep = transformed_source_weights(weights, 1.5)
    assert np.isclose(shallow.sum(), 1.0)
    assert np.isclose(steep.sum(), 1.0)
    assert steep[-1] / steep[0] > shallow[-1] / shallow[0]


def test_multiplicity_gate_vanishes_for_one_source():
    assert multiplicity_gate([1.0], 2.0) == 0.0
    assert 0.0 < multiplicity_gate([0.5, 0.5], 2.0) < 1.0


def test_positive_extent_slope_routes_more_for_extended_systems():
    common = dict(
        concentration=0.65,
        source_weights=[0.5, 0.5],
        feature="r50",
        base_fraction=0.5,
        extent_slope=1.0,
        base_length_kpc=250.0,
        length_power=0.0,
        base_width_kpc=50.0,
        width_power=0.0,
        gate_power=0.0,
    )
    compact = adaptive_route_parameters(r50_kpc=100.0, **common)
    extended = adaptive_route_parameters(r50_kpc=225.0, **common)
    assert extended["routing_fraction"] > compact["routing_fraction"]


def test_extent_features_and_invalid_name():
    combined = extent_coordinate(150.0, 0.65, "combined")
    assert np.isclose(combined, 0.0)
    with pytest.raises(ValueError):
        extent_coordinate(150.0, 0.65, "unknown")
