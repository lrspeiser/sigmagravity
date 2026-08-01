from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "p0636_little_things_baryon_acquisition.json"
RESULTS = ROOT / "results" / "p0636_little_things_baryon_acquisition"


def test_every_frozen_target_has_exactly_four_permitted_products():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    provenance = pd.read_csv(RESULTS / "provenance.csv")
    assert len(config["targets"]) == 13
    assert len(provenance) == 52
    assert set(provenance.groupby("galaxy").size()) == {4}
    assert set(provenance["role"]) == {
        "H_I_moment_0",
        "B_band",
        "V_band",
        "UBV_calibration",
    }
    assert provenance["valid"].all()


def test_no_sealed_kinematic_product_was_acquired_or_opened():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    report = json.loads((RESULTS / "provenance.json").read_text(encoding="utf-8"))
    filenames = " ".join(row["filename"].lower() for row in report["products_detail"])
    assert not any(fragment.lower() in filenames for fragment in config["forbidden_filename_fragments"])
    assert report["P0633_target_observables_opened"] is False
    assert not any(report["sealed_state"].values())


def test_all_radio_and_optical_fits_are_ready_for_map_ingestion():
    report = json.loads((RESULTS / "provenance.json").read_text(encoding="utf-8"))
    assert report["status"] == "ready"
    assert report["all_products_valid"] is True
    assert report["errors"] == []
    radio = [row for row in report["products_detail"] if row["role"] == "H_I_moment_0"]
    assert len(radio) == 13
    assert all(row["beam_major_deg"] > 0.0 and row["beam_minor_deg"] > 0.0 for row in radio)
    assert all(row["finite"] for row in radio)
