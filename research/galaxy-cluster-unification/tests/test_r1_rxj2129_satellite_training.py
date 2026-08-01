import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_rxj2129_satellite_training_audit_is_residual_blind() -> None:
    report = json.loads(
        (
            ROOT
            / "results/r1_rxj2129_satellite_membership/crossmatch_report.json"
        ).read_text(encoding="utf-8")
    )
    assert report["gravity_or_lens_residual_read"] is False
    assert report["membership_classifier_fit"] is False
    assert report["metrics"]["parsed_muse_redshifts"] == 156
    assert report["source_count_audit"]["published_prose_count"] == 158
    assert report["metrics"]["member_matches"] > 0
    assert report["metrics"]["nonmember_matches"] > 0


def test_rxj2129_satellite_training_ledger_is_unique() -> None:
    training = pd.read_csv(
        ROOT / "data/derived/r1_rxj2129_muse_molino_training.csv"
    )
    assert training["molino_CLASHID"].is_unique
    assert training["match_separation_arcsec"].max() <= 0.5
    assert set(training["is_cluster_member"].unique()) == {False, True}
