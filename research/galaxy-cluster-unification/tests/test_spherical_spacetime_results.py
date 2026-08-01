import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "spherical_spacetime_cavity"


def test_spherical_spacetime_reports_reproduce_the_verdict():
    galaxy = json.loads((RESULTS / "galaxy_report.json").read_text(encoding="utf-8"))
    raw = json.loads((RESULTS / "raw_lensing_report.json").read_text(encoding="utf-8"))
    protocol_path = ROOT / galaxy["protocol"]["path"]
    assert hashlib.sha256(protocol_path.read_bytes()).hexdigest() == galaxy["protocol"]["sha256"]
    assert hashlib.sha256(protocol_path.read_bytes()).hexdigest() == raw["protocol"]["sha256"]

    assert galaxy["advanced_to_raw_lensing"] == []
    assert all(not result["advance"] for result in galaxy["models"].values())
    cavity = galaxy["hard_cavity"]
    assert cavity["fraction_axis_upper_bound_meets_required"] == 0.0
    assert cavity["axis_factor_quantiles"]["median"] < 1.01
    assert cavity["required_factor_quantiles"]["median"] > 3.0
    assert cavity["stellar_covering_fraction_quantiles"]["maximum"] < 1.0e-10

    assert raw["transfer_status"].startswith("post-failure")
    sphere = raw["cross_cluster_validation"]["equal_system_radial_RMS_arcsec"]
    baryons = raw["comparators"]["baryons_GR"]["equal_system_radial_RMS_arcsec"]
    halo = raw["comparators"]["GR_plus_cluster_halo"]["equal_system_radial_RMS_arcsec"]
    assert abs(sphere - baryons) < 0.1
    assert sphere > 2.0 * halo
    assert raw["verdict"]["cutoff_robust_within_20_percent"]
    assert not raw["verdict"]["spherical_spacetime_candidate_survives"]
