from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from voidscreen.sigma_nonmetricity import (
    simple_nu,
    trace_action_derivative,
    trace_action_primitive,
    trace_nonmetricity,
    trace_split_spherical_accelerations,
)

ROOT = Path(__file__).resolve().parents[1]


def test_second_trace_reduces_to_spatial_potential_gradient() -> None:
    rng = np.random.default_rng(8202)
    grad_psi = rng.normal(size=(2000, 3))
    grad_phi = rng.normal(size=(2000, 3))
    assert np.allclose(
        trace_nonmetricity(grad_psi, grad_phi),
        4.0 * np.sum(np.square(grad_phi), axis=1),
        atol=2e-14,
    )


def test_trace_action_primitive_has_declared_derivative() -> None:
    invariant = np.geomspace(1e-9, 1e7, 4000)
    step = 1e-5
    numerical = (
        trace_action_primitive(invariant * np.exp(step))
        - trace_action_primitive(invariant * np.exp(-step))
    ) / (2.0 * step * invariant)
    assert np.allclose(numerical, trace_action_derivative(invariant), rtol=3e-5, atol=1e-8)


def test_spherical_branch_is_qumond_with_fixed_weyl_average() -> None:
    acceleration_scale = 1.2e-10
    gbar = acceleration_scale * np.geomspace(1e-6, 1e8, 1000)
    result = trace_split_spherical_accelerations(gbar, acceleration_scale)
    expected_matter = simple_nu(gbar / acceleration_scale) * gbar
    assert np.allclose(result["spatial_phi"], gbar)
    assert np.allclose(result["matter_psi"], expected_matter)
    assert np.allclose(result["photon_weyl"], 0.5 * (gbar + expected_matter))
    assert abs(expected_matter[0] / np.sqrt(gbar[0] * acceleration_scale) - 1.0) < 0.001
    assert expected_matter[-1] / gbar[-1] - 1.0 < 1e-8


def test_frozen_protocol_has_no_object_or_lensing_parameters() -> None:
    protocol = json.loads(
        (ROOT / "configs" / "sigma_v2_trace_nonmetricity_cycle.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["parameters"]["global_physical_parameter_count"] == 1
    assert protocol["parameters"]["per_object_gravity_parameters"] == 0
    assert protocol["parameters"]["lensing_only_parameters"] == 0


def test_completed_cycle_passes_galaxies_and_fails_raw_cluster_topology() -> None:
    output = ROOT / "results" / "sigma_v2_trace_nonmetricity_cycle"
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    clusters = pd.read_csv(output / "cluster_scores.csv")
    assert report["gate_results"]["mathematical_and_limit_checks"]
    assert report["gate_results"]["galaxy"]
    assert not report["gate_results"]["novel_weak_field_response"]
    assert not report["gate_results"]["raw_cluster_lensing"]
    assert not report["advances"]
    assert set(clusters.cluster) == {"AS295", "PLCKG287"}
    assert np.allclose(clusters.root_convergence_fraction, 1.0 / 3.0)
    assert not clusters.all_heldout_topologies_correct.any()
