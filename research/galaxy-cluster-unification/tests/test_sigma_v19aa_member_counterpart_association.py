from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_sigma_v19aa_member_counterpart_association import (
    Detection,
    angular_separation_arcsec,
    association_posterior,
    global_assignment,
    quantized_axis_pdf,
    reciprocal_crossmatches,
)


def test_quantized_axis_pdf_is_normalized_and_symmetric() -> None:
    axis = np.linspace(-10.0, 10.0, 200_001)
    values = np.asarray([quantized_axis_pdf(value, 2.0, 0.3) for value in axis])
    assert np.trapezoid(values, axis) == pytest.approx(1.0, abs=2e-10)
    assert values[0] == pytest.approx(values[-1])
    assert values[len(values) // 2] > values[0]


def test_association_posterior_includes_null_and_normalizes() -> None:
    candidates, null = association_posterior([20.0, 2.0], 0.9)
    assert float(candidates.sum()) + null == pytest.approx(1.0)
    assert candidates[0] / candidates[1] == pytest.approx(10.0)
    assert candidates[0] > null


def test_association_posterior_without_candidates_is_all_null() -> None:
    candidates, null = association_posterior([], 0.9)
    assert candidates.size == 0
    assert null == pytest.approx(1.0)


def test_global_assignment_prevents_duplicate_counterpart() -> None:
    members = [
        {"cluster": "C", "object_id": "1"},
        {"cluster": "C", "object_id": "2"},
    ]
    member_candidates = {
        ("C", "1"): ["shared", "alternate"],
        ("C", "2"): ["shared"],
    }
    log_likelihood_ratios = {
        ("C", "1", "shared"): np.log(100.0),
        ("C", "1", "alternate"): np.log(20.0),
        ("C", "2", "shared"): np.log(90.0),
    }
    assignment = global_assignment(
        members,
        member_candidates,
        log_likelihood_ratios,
        counterpart_prior=0.9,
    )
    assert assignment == {("C", "1"): "alternate", ("C", "2"): "shared"}


def test_reciprocal_crossmatch_does_not_force_conflicted_neighbor() -> None:
    def detection(survey: str, identifier: str, ra: float) -> Detection:
        return Detection(survey, "C", identifier, ra, 0.0, 0.05, {})

    hsc = {
        "HSC:a": detection("HSC", "a", 0.0),
        "HSC:b": detection("HSC", "b", 0.00020),
    }
    nsc = {
        "NSC:x": detection("NSC", "x", 0.00001),
        "NSC:y": detection("NSC", "y", 0.00021),
    }
    matches = reciprocal_crossmatches(hsc, nsc, radius_arcsec=0.5)
    assert {(left, right) for left, right, _ in matches} == {
        ("HSC:a", "NSC:x"),
        ("HSC:b", "NSC:y"),
    }
    assert all(separation < 0.05 for _, _, separation in matches)


def test_angular_separation_handles_ra_wrap() -> None:
    separation = angular_separation_arcsec(359.9999, 0.0, 0.0001, 0.0)
    assert separation == pytest.approx(0.72, rel=1e-8)
