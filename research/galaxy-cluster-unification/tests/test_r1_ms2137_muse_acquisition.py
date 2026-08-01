import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ms2137_acquisition_report_passes_without_pixel_inspection():
    report = json.loads((ROOT / "results/r1_ms2137_muse_acquisition/report.json").read_text())
    assert report["pixel_arrays_inspected"] is False
    assert report["size_bytes"] == 477322560
    assert report["sha256"] == "EEC593788696A9F640A6DA7C17FEACEC9AABBF998773A4BA1C9676B10F95716D"
    assert report["extension_names"] == ["PRIMARY", "DATA", "STAT"]
    assert report["data_metadata"]["shape_spectral_y_x"] == [1841, 180, 180]
    assert report["data_metadata"]["wavelength_first_angstrom"] == 4860.3134765625
    assert report["data_metadata"]["wavelength_last_angstrom"] == 7160.3134765625
    assert report["protocol_amendments"][0]["amendment_id"] == "A1-native-spectral-grid-semantics"
    assert all(report["gates"].values())
    assert report["decision"] == "authorize_numerical_protocol_freeze"
    assert report["authorization"]["freeze_numerical_protocol"] is True
    assert report["authorization"]["inspect_science_pixels"] is False
