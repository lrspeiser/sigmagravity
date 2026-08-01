import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results" / "formula_scorecard" / "formula_scorecard.json"


def _rows_by_name() -> dict[str, dict]:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    return {row["formula"]: row for row in payload["rows"]}


def test_scorecard_has_broad_formula_coverage() -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    assert payload["formula_rows"] == len(payload["rows"])
    assert payload["formula_rows"] >= 90
    assert len({row["family"] for row in payload["rows"]}) >= 10


def test_raw_and_derived_lensing_are_not_merged() -> None:
    rows = _rows_by_name()
    nfw = rows["CLASH NFW construction"]
    assert nfw["derived_lensing_proximity_percent"] == 100.0
    assert nfw["raw_lensing_proximity_percent"] is None
    assert "not an independent" in nfw["verdict"]


def test_current_candidate_and_new_tensor_are_recorded() -> None:
    rows = _rows_by_name()
    candidate = rows["RAR + squared coherence-gated RG (current empirical bridge)"]
    full_tensor = rows["Full member tidal metric (new test)"]

    assert 93.0 < candidate["galaxy_proximity_percent"] < 94.0
    assert 92.0 < candidate["raw_lensing_proximity_percent"] < 93.0
    assert full_tensor["raw_lensing_error_arcsec"] > 18.43
    assert "selected t=0" in full_tensor["verdict"]


def test_solar_screened_cluster_survivor_is_recorded() -> None:
    row = _rows_by_name()["Solar-screened baryon-normalized isothermal tail"]
    assert 18.60 < row["galaxy_error"] < 18.61
    assert 5.26 < row["raw_lensing_error_arcsec"] < 5.27
    assert row["all_raw_roots_complete"] is True
    assert "Mercury" in row["verdict"]


def test_incomplete_roots_never_receive_raw_percentage() -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    incomplete = [row for row in payload["rows"] if row["all_raw_roots_complete"] is False]
    assert incomplete
    assert all(row["raw_lensing_proximity_percent"] is None for row in incomplete)
