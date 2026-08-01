import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_chandra_reduction_protocol_is_frozen_and_noninferential() -> None:
    protocol = json.loads(
        (ROOT / "configs/r1_rxj2129_chandra_reduction_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["status"] == (
        "frozen_before_ciao_reprocessing_or_calibrated_product_inspection"
    )
    assert protocol["software"] == {
        "ciao_required": "4.18",
        "caldb_required": "4.12.4",
        "environment": "/home/henry/ciao-4.18",
        "installation_source": (
            "official CXC conda channel with conda-forge dependencies"
        ),
    }
    assert protocol["inputs"]["obsids"] == [552, 9370]
    assert protocol["reprocessing"]["pix_adj"] == "EDSER"
    assert protocol["flare_filter"]["nsigma"] == 3.0
    assert protocol["blank_sky"]["weight_method"] == "particle"
    assert protocol["spectra"]["minimum_counts_for_any_independent_temperature_annulus"] == 1500
    assert protocol["authorization"]["ciao_reprocessing_and_calibration_audit"] is True
    assert protocol["authorization"]["gas_density_or_mass_fit"] is False
    assert protocol["authorization"]["gravity_response_fit"] is False
    assert protocol["authorization"]["weyl_response_reconstruction"] is False
    assert protocol["authorization"]["strict_r1_ready"] is False


def test_chandra_reduction_regions_are_fixed_at_the_shared_center() -> None:
    regions = ROOT / "configs/regions"
    expected = {
        "r1_rxj2129_flare_background.reg": 'annulus(21:29:39.9624,+00:05:21.228,120",480")',
        "r1_rxj2129_global_60arcsec.reg": 'circle(21:29:39.9624,+00:05:21.228,60")',
        "r1_rxj2129_annulus_0_5arcsec.reg": 'circle(21:29:39.9624,+00:05:21.228,5")',
        "r1_rxj2129_annulus_5_15arcsec.reg": 'annulus(21:29:39.9624,+00:05:21.228,5",15")',
        "r1_rxj2129_annulus_15_30arcsec.reg": 'annulus(21:29:39.9624,+00:05:21.228,15",30")',
        "r1_rxj2129_annulus_30_60arcsec.reg": 'annulus(21:29:39.9624,+00:05:21.228,30",60")',
    }
    for filename, shape in expected.items():
        assert shape in (regions / filename).read_text(encoding="utf-8")
