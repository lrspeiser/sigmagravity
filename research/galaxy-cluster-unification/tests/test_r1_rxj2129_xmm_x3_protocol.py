from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_rxj2129_x3_annular_protocol_is_numeric_residual_blind_and_narrow() -> None:
    protocol = json.loads(
        (ROOT / "configs/r1_rxj2129_xmm_x3_annular_protocol.json").read_text()
    )
    geometry = protocol["fixed_geometry"]
    assert protocol["status"] == (
        "corrected_and_refrozen_after_ESAS_pn_badpixel_resolution_and_quadrant_support_audit_before_admissible_X3_products"
    )
    assert protocol["protocol_version"].endswith("0.4")
    assert protocol["correction_log"][-1]["invalid_partial_products_admitted"] is False
    assert protocol["correction_log"][0][
        "annular_edges_energy_band_background_multipliers_or_thresholds_changed"
    ] is False
    assert protocol["correction_log"][-1][
        "catalog_source_position_radius_annular_edge_energy_band_background_multiplier_or_threshold_changed"
    ] is False
    assert protocol["prerequisites"]["required_passing_instruments"] == ["MOS2", "pn"]
    assert geometry["radial_edges_kpc"] == [10.0, 50.0, 100.0, 175.0, 275.0, 380.0, 500.0]
    assert len(geometry["annulus_ids"]) == 6
    assert geometry["minimum_accepted_annuli"] == 5
    assert geometry["minimum_total_net_counts"] == 2000.0
    assert geometry["minimum_signal_to_noise_each_annulus"] == 5.0
    pn = protocol["extraction"]["pn"]
    assert pn["badpixelresolution_arcsec"] == 1.0
    assert pn["active_quadrants_by_annulus"] == {
        "a01_010_050kpc": "F T F F",
        "a02_050_100kpc": "F T F F",
        "a03_100_175kpc": "F T F F",
        "a04_175_275kpc": "F T T F",
        "a05_275_380kpc": "T T T F",
        "a06_380_500kpc": "T T T T",
    }
    detector_center = pn["frozen_detector_center"]
    expected_mapping = {}
    for index, annulus_id in enumerate(geometry["annulus_ids"]):
        inner = geometry["radial_edges_detector_units"][index]
        outer = geometry["radial_edges_detector_units"][index + 1]
        flags = []
        for quadrant in ("1", "2", "3", "4"):
            box = pn["quadrant_detector_boxes"][quadrant]
            xmin = box["center"][0] - box["half_widths"][0]
            xmax = box["center"][0] + box["half_widths"][0]
            ymin = box["center"][1] - box["half_widths"][1]
            ymax = box["center"][1] + box["half_widths"][1]
            dx_min = max(xmin - detector_center[0], 0.0, detector_center[0] - xmax)
            dy_min = max(ymin - detector_center[1], 0.0, detector_center[1] - ymax)
            minimum_radius = math.hypot(dx_min, dy_min)
            maximum_radius = max(
                math.hypot(x - detector_center[0], y - detector_center[1])
                for x in (xmin, xmax)
                for y in (ymin, ymax)
            )
            flags.append("T" if minimum_radius < outer and maximum_radius > inner else "F")
        expected_mapping[annulus_id] = " ".join(flags)
    assert pn["active_quadrants_by_annulus"] == expected_mapping
    kpc_per_arcsec = geometry["cosmology_for_coordinate_conversion_only"][
        "kpc_per_arcsec"
    ]
    for kpc, arcsec, detector in zip(
        geometry["radial_edges_kpc"],
        geometry["radial_edges_arcsec"],
        geometry["radial_edges_detector_units"],
        strict=True,
    ):
        assert math.isclose(arcsec, kpc / kpc_per_arcsec, rel_tol=1e-12)
        assert math.isclose(detector, 20.0 * arcsec, rel_tol=1e-12)
    adequacy = protocol["prefit_adequacy"]
    assert adequacy["conservative_QPB_multiplier"]["MOS2"] > 1.0
    assert adequacy["conservative_QPB_multiplier"]["pn"] == 1.0
    assert "compact subset" in protocol["extraction"]["source_mask"]
    assert "source_id 50" in protocol["extraction"]["central_target_component"]
    assert protocol["next_gate_after_pass"]["protocol_required"].startswith(
        "A new XMM-specific"
    )
    assert protocol["authorization"]["construct_region_specific_MOS2_and_pn_products"] is True
    assert protocol["authorization"]["fit_temperature_or_density"] is False
    assert protocol["authorization"]["infer_gas_mass"] is False
    assert protocol["authorization"]["infer_dynamical_or_Weyl_response"] is False
    assert protocol["authorization"]["fit_new_force_or_action"] is False
