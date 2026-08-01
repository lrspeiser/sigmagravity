from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_m1206_ppxf_protocol_is_blind_and_not_gravity_authorized() -> None:
    config = json.loads(
        (ROOT / "configs/r1_m1206_ppxf_protocol.json").read_text(encoding="utf-8")
    )
    assert config["spatial_extraction"]["annulus_semimajor_edges_arcsec"] == [
        0.0,
        0.6,
        1.5,
        3.0,
        5.0,
        8.0,
        12.0,
    ]
    assert config["spectral_fit"]["software_version"] == "9.4.8"
    assert config["success_thresholds"][
        "maximum_opposite_half_velocity_difference_km_s"
    ] == 100.0
    assert config["success_thresholds"]["publication_figure_use"].startswith(
        "visual validation only"
    )
    assert config["authorization"]["engineering_profile_reconstruction"]
    assert not config["authorization"]["gravity_response_fit"]
    assert not config["authorization"]["claim_as_published_likelihood"]
