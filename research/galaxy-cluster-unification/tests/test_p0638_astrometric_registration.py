from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0638_gaia_astrometric_registration"


def test_all_thirteen_optical_images_pass_the_frozen_registration_gates():
    report = json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(RESULTS / "astrometry_audit.csv")
    assert report["status"] == "pass"
    assert report["all_gates_pass"] is True
    assert len(frame) == 13
    assert frame["galaxy"].is_unique
    assert frame["all_gates_pass"].all()


def test_astrometry_did_not_open_a_sealed_target_observable():
    report = json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))
    assert report["sealed_target_observables_opened"] is False
    assert report["failures"] == []
