from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from voidscreen.sigma_nonlocal_spectral import (
    entire_ir_transfer,
    entire_ir_transfer_derivative,
    entire_point_force_correction,
    entire_point_force_ratio,
    periodic_lensing_hessian,
    positive_spectral_transfer,
    positive_spectral_transfer_derivative,
    rational_far_enhancing_transfer,
    rational_propagator_residues,
)

ROOT = Path(__file__).resolve().parents[1]


def test_positive_standard_spectrum_cannot_be_stronger_in_the_ir() -> None:
    s = np.geomspace(1e-10, 1e10, 5000)
    masses = np.geomspace(1e-4, 1e4, 41)
    residues = np.linspace(0.02, 1.0, len(masses))
    transfer = positive_spectral_transfer(s, masses, residues)
    derivative = positive_spectral_transfer_derivative(s, masses, residues)

    assert np.all(derivative >= 0.0)
    assert np.all(np.diff(transfer) >= 0.0)
    assert transfer[0] < 1.0
    assert np.isclose(transfer[-1], 1.0, rtol=1e-5)


def test_rational_far_enhancement_requires_a_negative_massive_residue() -> None:
    amplitude = 5.7
    s = np.geomspace(1e-8, 1e8, 1000)
    transfer = rational_far_enhancing_transfer(s, amplitude)
    residues = rational_propagator_residues(amplitude)
    direct_propagator = transfer / s
    partial_fraction = residues["massless_residue"] / s + residues[
        "massive_residue"
    ] / (s + 1.0)

    assert np.allclose(direct_propagator, partial_fraction, rtol=2e-14)
    assert residues["massless_residue"] > 0.0
    assert residues["massive_residue"] < 0.0


def test_entire_escape_has_no_real_zero_but_reverses_standard_spectral_monotonicity() -> None:
    log_boost = np.log(6.7)
    s = np.geomspace(1e-10, 1e10, 2000)
    transfer = entire_ir_transfer(s, log_boost)
    derivative = entire_ir_transfer_derivative(s, log_boost)

    assert np.all(transfer > 0.0)
    assert np.all(derivative <= 0.0)
    assert np.any(derivative < 0.0)
    assert np.isclose(transfer[0], 6.7, rtol=1e-9)
    assert np.isclose(transfer[-1], 1.0)


def test_entire_point_force_is_locally_screened_and_reaches_ir_boost() -> None:
    log_boost = np.log(6.7)
    local_x = 4.84813681109536e-11
    correction = entire_point_force_correction(local_x, log_boost)
    ratio = entire_point_force_ratio(np.asarray([0.1, 100.0]), log_boost)

    assert 0.0 < correction < 1e-25
    assert ratio[0] - 1.0 < 5e-4
    assert np.isclose(ratio[-1], 6.7, rtol=1e-10)


def test_common_metric_filter_changes_a_manufactured_shear_map() -> None:
    size = 96
    coordinate = np.linspace(-6.0, 6.0, size, endpoint=False)
    x, y = np.meshgrid(coordinate, coordinate)
    density = np.exp(-((x + 1.1) ** 2 + y**2) / (2.0 * 0.25**2))
    density += 0.7 * np.exp(-((x - 0.9) ** 2 + (y + 0.5) ** 2) / (2.0 * 0.4**2))
    base = periodic_lensing_hessian(density, 12.0 / size)
    filtered = periodic_lensing_hessian(
        density,
        12.0 / size,
        log_ir_boost=np.log(6.7),
        response_length=1.0,
    )

    assert np.allclose(base["convergence"], density - np.mean(density), atol=2e-14)
    shear_change = np.hypot(
        filtered["shear_1"] - base["shear_1"],
        filtered["shear_2"] - base["shear_2"],
    )
    assert np.linalg.norm(shear_change) > 0.1 * np.linalg.norm(base["shear_1"])


def test_completed_audit_selects_a_nonlinear_not_linear_next_mechanism() -> None:
    protocol = json.loads(
        (
            ROOT
            / "configs"
            / "sigma_v3b_linear_nonlocal_spectral_audit.json"
        ).read_text(encoding="utf-8")
    )
    report = json.loads(
        (
            ROOT
            / "results"
            / "sigma_v3b_linear_nonlocal_spectral_audit"
            / "report.json"
        ).read_text(encoding="utf-8")
    )

    assert protocol["parameters"]["total_provisional_physical_parameter_count"] == 3
    assert protocol["parameters"]["per_object_gravity_parameters"] == 0
    assert protocol["parameters"]["lensing_only_parameters"] == 0
    gates = report["gate_results"]
    assert not gates["positive_spectral_linear_exchange_can_enhance_IR"]
    assert not gates["rational_IR_filter_has_no_negative_residue"]
    assert gates["entire_filter_has_no_extra_finite_pole"]
    assert not gates["entire_filter_passes_standard_positive_spectral_test"]
    assert not gates["entire_filter_has_proved_causal_Lorentzian_prescription"]
    assert gates["luminal_massless_tensor_pole"]
    assert gates["nonzero_baryon_registered_shear_response"]
    assert not report["advances_to_frozen_linear_sigma_v3"]
    assert "nonlinear retarded tidal interaction" in report["next_mechanism"]
