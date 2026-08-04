from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from voidscreen.sigma_operator_inference import (
    angular_wavenumber_grid,
    apodization_window,
    normalized_channel_rmse,
    radial_transfer_spectrum,
    transfer_grid_from_spectrum,
    wavelength_band_mask,
    windowed_fourier,
)

ROOT = Path(__file__).resolve().parents[1]


def test_windowed_fourier_removes_a_constant_field() -> None:
    window = apodization_window((32, 48), 0.3)
    transformed = windowed_fourier(np.full((32, 48), 7.5), window)
    assert window.shape == (32, 48)
    assert np.all(window >= 0.0)
    assert np.all(window <= 1.0)
    assert np.max(np.abs(transformed)) < 1e-12


def test_radial_spectrum_recovers_a_shared_real_transfer() -> None:
    rng = np.random.default_rng(8303)
    shape = (48, 48)
    spacing = 2.0
    wavenumber = angular_wavenumber_grid(shape, spacing)
    band = wavelength_band_mask(wavenumber, 10.0, 80.0)
    source = {
        "convergence": rng.normal(size=shape) + 1j * rng.normal(size=shape),
        "shear_1": rng.normal(size=shape) + 1j * rng.normal(size=shape),
        "shear_2": rng.normal(size=shape) + 1j * rng.normal(size=shape),
    }
    target = {name: 3.25 * values for name, values in source.items()}
    spectrum = radial_transfer_spectrum(
        source, target, wavenumber, band, bins=12
    )
    reconstructed = transfer_grid_from_spectrum(spectrum, wavenumber)

    assert np.allclose(spectrum.best_real_transfer, 3.25)
    assert np.allclose(spectrum.coherence, 1.0)
    assert normalized_channel_rmse(source, target, reconstructed, band) < 1e-14


def test_completed_spent_inference_rejects_wavelength_only_linear_response() -> None:
    protocol = json.loads(
        (
            ROOT / "configs" / "sigma_v3c_spent_operator_inference.json"
        ).read_text(encoding="utf-8")
    )
    report = json.loads(
        (
            ROOT
            / "results"
            / "sigma_v3c_spent_operator_inference"
            / "report.json"
        ).read_text(encoding="utf-8")
    )

    assert protocol["sample"]["sample_is_spent"]
    assert protocol["sample"]["clusters"] == ["AS295", "PLCKG287"]
    gates = report["gate_results"]
    assert not gates["joint_entire_linear_plausibility"]
    assert not gates["cross_cluster_binned_isotropic_transfer"]
    assert not gates["radial_phase_coherence"]
    assert not gates["nonnegative_real_transfer"]
    assert not gates["all_isotropic_linear_diagnostics"]
    assert report["joint_entire_fit"]["normalized_RMSE"] > 0.8 - 1e-3
    sensitivity = report["post_failure_lower_length_sensitivity"]
    assert sensitivity["primary_fit_at_lower_bound"]
    assert sensitivity["result"]["normalized_RMSE"] > 0.78
    for row in report["scores"]:
        assert row["median_radial_coherence"] < 0.3
        assert row["cross_cluster_binned_oracle_normalized_RMSE"] > 0.8
    assert "local tensor orientation" in report["inferred_action_requirement"]
