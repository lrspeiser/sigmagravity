import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_slacs_kcwi_sample_screen_is_complete_and_does_not_promote_rank() -> None:
    report = json.loads(
        (ROOT / "results/r1_slacs_kcwi_sample_feasibility/report.json").read_text(
            encoding="utf-8"
        )
    )
    ledger = pd.read_csv(ROOT / "data/derived/r1_slacs_kcwi_candidate_ledger.csv")
    inventory = pd.read_csv(ROOT / "data/derived/r1_slacs_kcwi_archive_inventory.csv")

    assert report["protocol"] == "R1-SLACS-KCWI-sample-feasibility-0.1"
    assert report["selection_blind"] is True
    assert report["science_arrays_downloaded"] is False
    assert all(report["source_checks"].values())
    assert len(ledger) == 14
    assert ledger["system"].nunique() == 14
    assert set(inventory["archive"]) == {"KOA_KCWI", "MAST_HST"}
    assert not ledger["counts_toward_ten_system_target"].any()
    assert report["sample_summary"]["strict_rank_three_promotions"] == 0
    assert report["sample_summary"]["structural_ceiling_after_screen"] == 3
    assert report["sample_summary"]["strict_ready_systems_after_screen"] == 0
    assert report["authorization"]["count_candidate_as_rank_three"] is False
    assert report["authorization"]["fit_gravity_response"] is False
    assert report["authorization"]["authorize_R2"] is False
