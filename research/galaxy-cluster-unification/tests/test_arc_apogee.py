import numpy as np

from voidscreen.arc_apogee import (
    acceleration_screen,
    arc_apogee_enhancement,
    extent_gate,
    mass_radius_kpc,
    residence_coordinate,
    solar_diagnostics,
)


def test_residence_coordinate_grows_then_saturates():
    radius = np.array([0.1, 1.0, 10.0, 1.0e6])
    coordinate = residence_coordinate(radius, 1.0, alpha=1.0, apogee_ratio=2.0)
    assert np.all(np.diff(coordinate) > 0.0)
    assert coordinate[-1] < 2.0001
    assert coordinate[-1] > 1.999


def test_extent_gate_and_screen_are_bounded():
    gate = extent_gate(np.array([0.4, 0.65, 0.9]), "cluster_logistic")
    assert np.all((gate > 0.0) & (gate < 1.0))
    assert np.all(np.diff(gate) > 0.0)
    screen = acceleration_screen(
        np.array([1.2e-12, 1.2e-10, 1.2e-8]),
        a0_m_s2=1.2e-10,
        exponent=2.0,
    )
    assert np.all(np.diff(screen) < 0.0)
    assert np.isclose(screen[1], 0.5)


def test_enhancement_and_mass_radius_are_physical():
    result = arc_apogee_enhancement(
        np.array([1e-12, 1e-10]),
        np.array([10.0, 1.0]),
        np.array([5.0, 5.0]),
        np.array([0.8, 0.8]),
        residence_strength=1.0,
        alpha=1.0,
        apogee_ratio=2.0,
        gate_mode="cluster_logistic",
        screen_a0_m_s2=1.2e-10,
        screen_exponent=2.0,
    )
    assert np.all(result["enhancement_relative_to_local_G"] >= 1.0)
    assert float(mass_radius_kpc(1e11, a0_m_s2=1.2e-10)) > 0.0


def test_quadratic_screen_can_hide_worst_case_solar_response():
    diagnostics = solar_diagnostics(
        residence_strength=1.0,
        alpha=1.0,
        apogee_ratio=2.0,
        gate_mode="cluster_logistic",
        scale_mode="baryon_r80",
        screen_a0_m_s2=1.2e-10,
        screen_exponent=2.0,
    )
    assert diagnostics["Cassini_proxy_pass"]
    assert diagnostics["Earth_proxy_pass"]
    assert diagnostics["Mercury_proxy_pass"]
