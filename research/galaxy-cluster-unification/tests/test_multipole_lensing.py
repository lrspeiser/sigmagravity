from __future__ import annotations

import numpy as np
import pytest

from voidscreen.multipole_lensing import build_matched_multipole_deflection_field


def build(order=3, phase=0.31):
    axis = np.arange(-80.0, 80.5, 0.5)
    return build_matched_multipole_deflection_field(
        axis,
        order=order,
        phase_rad=phase,
        radial_scale_arcsec=30.0,
        taper_inner_arcsec=50.0,
        support_radius_arcsec=58.0,
        target_deflection_rms_arcsec=0.07,
    )


def test_field_is_matched_curl_free_and_compensated():
    field = build()
    assert np.isclose(field.audit["unit_deflection_RMS_arcsec"], 0.07)
    assert field.audit["normalized_curl_RMS"] < 1e-10
    assert field.audit["source_integral_fraction"] < 1e-8
    assert field.audit["maximum_edge_correction_arcsec"] == 0.0


def test_rotation_covariance_for_multipole_phase():
    order = 4
    rotation = np.pi / (2.0 * order)
    field = build(order=order, phase=0.0)
    rotated = build(order=order, phase=rotation)
    x = np.array([5.0, 12.0, 24.0])
    y = np.array([3.0, -7.0, 10.0])
    alpha = field.alpha_arcsec(x, y, distance_ratio=1.0)
    cosine, sine = np.cos(rotation), np.sin(rotation)
    beta = rotated.alpha_arcsec(
        cosine * x - sine * y,
        sine * x + cosine * y,
        distance_ratio=1.0,
    )
    assert np.allclose(beta[0], cosine * alpha[0] - sine * alpha[1], atol=3e-3)
    assert np.allclose(beta[1], sine * alpha[0] + cosine * alpha[1], atol=3e-3)


def test_signed_amplitude_is_linear():
    field = build()
    positive = field.alpha_arcsec([8.0], [11.0], distance_ratio=1.0)
    negative = tuple(-component for component in positive)
    assert np.allclose(negative[0], -positive[0])
    assert np.allclose(negative[1], -positive[1])


@pytest.mark.parametrize("order", [0, 1, 2.5])
def test_invalid_orders_are_rejected(order):
    axis = np.arange(-80.0, 81.0, 1.0)
    with pytest.raises(ValueError):
        build_matched_multipole_deflection_field(
            axis,
            order=order,
            phase_rad=0.0,
            radial_scale_arcsec=30.0,
            taper_inner_arcsec=50.0,
            support_radius_arcsec=58.0,
            target_deflection_rms_arcsec=0.07,
        )
