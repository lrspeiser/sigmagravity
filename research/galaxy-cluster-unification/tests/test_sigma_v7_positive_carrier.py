from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_v7_positive_carrier import (
    audit_linear_positive_carrier,
    locally_calibrated_dynamical_ratio,
    maximum_residue_from_high_field_force,
    maximum_residue_from_ppn,
    positive_spin2_spectrum,
    ppn_gamma,
    spin2_force_enhancements,
    yukawa_force_kernel,
)


def test_positive_fierz_pauli_carrier_has_seven_healthy_spin2_modes() -> None:
    spectrum = positive_spin2_spectrum()
    assert spectrum["massless_spin2_degrees_of_freedom"] == 2
    assert spectrum["massive_spin2_degrees_of_freedom"] == 5
    assert spectrum["total_degrees_of_freedom"] == 7
    assert spectrum["negative_kinetic_directions"] == 0
    assert np.all(np.asarray(spectrum["kinetic_eigenvalues"]) > 0.0)


def test_massive_spin2_has_fixed_dynamics_and_lensing_residues() -> None:
    response = spin2_force_enhancements(0.0, 0.3)
    assert float(response["dynamics"]) == pytest.approx(1.4)
    assert float(response["spatial"]) == pytest.approx(1.2)
    assert float(response["lensing"]) == pytest.approx(1.3)
    assert float(ppn_gamma(0.3)) == pytest.approx(1.2 / 1.4)


def test_positive_yukawa_carrier_weakens_with_distance_after_local_calibration() -> None:
    ratio = np.geomspace(1.0e-8, 1.0e8, 10001)
    kernel = yukawa_force_kernel(ratio)
    calibrated = locally_calibrated_dynamical_ratio(ratio, 2.0)
    assert np.all(np.diff(kernel) <= 1.0e-14)
    assert calibrated[0] == pytest.approx(1.0, abs=1.0e-12)
    assert calibrated[-1] == pytest.approx(1.0 / (1.0 + 8.0 / 3.0))
    assert np.all(np.diff(calibrated) <= 1.0e-14)


def test_solar_gates_limit_unscreened_cluster_lensing_to_parts_per_million() -> None:
    ppn_limit = maximum_residue_from_ppn(2.3e-5)
    force_limit = maximum_residue_from_high_field_force(1.0e-5)
    assert ppn_limit == pytest.approx(3.450158707300536e-5)
    assert force_limit == pytest.approx(7.5e-6)
    audit = audit_linear_positive_carrier(
        ppn_bound=2.3e-5,
        high_field_force_bound=1.0e-5,
        required_lensing_enhancement=1.5,
        radius_over_range=np.geomspace(1.0e-12, 1.0e8, 20001),
    )
    assert audit["maximum_jointly_allowed_residue"] == pytest.approx(7.5e-6)
    assert audit["maximum_lensing_enhancement"] < 1.000008
    assert audit["gates"]["positive_local_kinetic_spectrum"]
    assert audit["gates"]["solar_ppn_gamma"]
    assert not audit["gates"]["large_scale_lensing_amplitude"]
    assert not audit["gates"]["turns_on_with_distance"]


@pytest.mark.parametrize(
    ("function", "arguments"),
    [
        (yukawa_force_kernel, (-1.0,)),
        (spin2_force_enhancements, (1.0, -0.1)),
        (ppn_gamma, (-0.1,)),
        (maximum_residue_from_ppn, (0.0,)),
        (maximum_residue_from_high_field_force, (0.0,)),
        (locally_calibrated_dynamical_ratio, (1.0, -0.1)),
    ],
)
def test_invalid_positive_carrier_inputs_are_rejected(function, arguments) -> None:
    with pytest.raises(ValueError):
        function(*arguments)
