import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p0554_all_baryon_route_input_audit.json"
RESULTS = ROOT / "results/p0554_all_baryon_route_input_audit"


def load_report():
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def test_input_audit_was_frozen_before_pixel_audit():
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert protocol["status"].startswith("frozen_")
    assert "before_fits_pixel_audit_or_route_score" in protocol["status"]
    assert protocol["gates"]["expected_systems"] == 5
    assert protocol["gates"]["expected_chandra_evt2_observations"] == 11


def test_all_registered_inputs_pass_adequacy():
    report = load_report()
    assert report["input_adequacy_pass"]
    assert report["coverage"] == {
        "systems": 5,
        "hst_files": 10,
        "chandra_evt2_observations": 11,
        "total_chandra_exposure_ks": 307.2584805788856,
        "known_lens_images": 84,
    }
    assert all(report["checks"][key] for key in report["checks"] if key not in {"gas_mass_inferred", "gravity_or_lens_residual_scored"})
    assert not report["checks"]["gas_mass_inferred"]
    assert not report["checks"]["gravity_or_lens_residual_scored"]


def test_every_image_has_hst_weight_and_every_cluster_has_xray_counts():
    hst = pd.read_csv(RESULTS / "hst_ledger.csv")
    assert len(hst) == 5
    assert (hst.known_images == hst.known_images_positive_weight).all()
    assert (hst.covered_fraction_within_60_arcsec == 1.0).all()
    totals = pd.DataFrame(load_report()["system_chandra_totals"])
    assert (totals.exposure_ks >= 20.0).all()
    assert (totals.soft_events_inside_100_arcsec >= 1000).all()
