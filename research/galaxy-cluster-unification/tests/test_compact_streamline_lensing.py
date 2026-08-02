from __future__ import annotations

import numpy as np

from voidscreen.accumulated_lensing import build_accumulated_transport_deflection_field


def gaussian(axis, x0, y0, width):
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return np.exp(-0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / width**2)


def build(closure):
    axis = np.arange(-80.0, 81.0, 1.0)
    stars = gaussian(axis, -12.0, 4.0, 9.0)
    gas = gaussian(axis, 10.0, -5.0, 16.0)

    def carrier(radius):
        return 18.0 * np.asarray(radius) / (12.0 + np.asarray(radius))

    def gbar(radius_kpc):
        return 3e-10 / np.maximum(np.asarray(radius_kpc) / 30.0, 0.2) ** 2

    return build_accumulated_transport_deflection_field(
        axis,
        stars,
        gas,
        angular_scale_kpc_per_arcsec=5.0,
        carrier_alpha_arcsec=carrier,
        radial_gbar_m_s2=gbar,
        mismatch_mode="transverse_tensor_mix",
        closure=closure,
        common_smoothing_kpc=10.0,
        taper_inner_arcsec=50.0,
        support_radius_arcsec=58.0,
        transport_steps=8,
    )


def test_compact_transport_closes_edge_flux_and_source():
    compact = build("compact_streamline_averaged_gas_minus_star_flux")
    open_field = build("streamline_averaged_gas_minus_star_flux")
    assert compact.audit["post_transport_compact_taper_applied"] is True
    assert compact.audit["maximum_flux_edge_fraction_of_RMS"] == 0.0
    assert compact.audit["source_integral_fraction"] < 1e-8
    assert open_field.audit["maximum_flux_edge_fraction_of_RMS"] > 0.0
    assert compact.audit["normalized_curl_RMS"] < 1e-10


def test_compact_transport_is_nonlocal_and_finite():
    compact = build("compact_streamline_averaged_gas_minus_star_flux")
    assert compact.audit["transport_relative_change_RMS"] > 0.05
    assert compact.audit["unit_deflection_RMS_arcsec"] > 0.0
    assert np.isfinite(compact.raw_alpha_x_arcsec).all()
    assert np.isfinite(compact.raw_alpha_y_arcsec).all()
