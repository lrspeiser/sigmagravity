import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_report() -> dict:
    return json.loads(
        (
            ROOT / "results/p0585_local_attractor_screen/report.json"
        ).read_text(encoding="utf-8")
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p0585_protocol_and_local_attractor_grid_are_frozen():
    result = load_report()
    protocol = ROOT / result["protocol"]["path"]
    assert result["protocol"]["sha256"] == sha256(protocol)
    assert result["coverage"] == {
        "candidates": 33,
        "amplitudes": 7,
        "scores": 231,
        "heldout_images": 7,
    }


def test_p0585_best_local_attractor_only_barely_beats_global_destination():
    result = load_report()
    winner = result["winner"]
    assert winner["candidate_id"] == "L0001"
    assert winner["local_mix"] == 0.25
    assert winner["softening_over_R80"] == 0.1
    assert winner["distance_power"] == 1.0
    assert winner["epsilon"] == 0.005
    assert winner["heldout_RMS_arcsec"] == pytest.approx(1.2423488955707376)
    assert result["global_best"]["heldout_RMS_arcsec"] == pytest.approx(
        1.2430077939002733
    )
    assert result["local_improvement_over_global_fraction"] < 0.0006


def test_p0585_destination_coordinates_have_sub_milliarcsecond_impacts():
    impacts = {row["parameter"]: row for row in load_report()["parameter_impacts"]}
    assert impacts["local_mix"]["median_RMS_span_arcsec"] < 0.0007
    assert impacts["softening_over_R80"]["median_RMS_span_arcsec"] < 0.0005
    assert impacts["distance_power"]["median_RMS_span_arcsec"] < 0.00013


def test_p0585_local_fields_are_conservative_and_do_not_cross_destinations():
    audit = load_report()["field_audit"]
    assert audit["maximum_route_normalization_error"] < 1e-12
    assert audit["maximum_annular_convergence_mean_fraction"] < 1e-12
    assert audit["maximum_normalized_curl_RMS"] < 1e-12
    assert audit["maximum_source_weight_crossing_destination"] == 0.0
