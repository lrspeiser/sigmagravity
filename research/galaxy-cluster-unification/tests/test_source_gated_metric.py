import numpy as np

from voidscreen.source_gated_metric import (
    gated_extra_acceleration,
    radial_source_concentration,
    source_gate,
    source_gated_metric_eta,
)


def test_radial_concentration_recovers_limiting_mass_slopes():
    radius = np.geomspace(1.0, 100.0, 20)
    point_mass_gbar = radius**-2
    uniform_density_gbar = radius
    assert np.allclose(radial_source_concentration(radius, point_mass_gbar), 1.0)
    assert np.allclose(
        radial_source_concentration(radius, uniform_density_gbar), 0.25
    )


def test_gate_turns_off_for_concentrated_or_high_acceleration_sources():
    concentrated = source_gate(
        [1.0e-12],
        [1.0],
        acceleration_scale_m_s2=1.2e-10,
    )
    high_acceleration = source_gate(
        [1.0],
        [0.0],
        acceleration_scale_m_s2=1.2e-10,
    )
    assert np.allclose(concentrated, 0.0)
    assert high_acceleration[0] < 1.0e-18


def test_metric_gate_changes_only_the_existing_extra_response():
    gbar = np.asarray([1.0])
    gdyn = np.asarray([3.0])
    gate = np.asarray([0.5])
    assert np.allclose(gated_extra_acceleration(gbar, gdyn, gate, 2.0), 5.0)
    assert np.allclose(source_gated_metric_eta(gbar, gdyn, gate, 2.0), 7.0 / 3.0)
    assert np.allclose(gated_extra_acceleration(gbar, gbar, gate, 30.0), gbar)
