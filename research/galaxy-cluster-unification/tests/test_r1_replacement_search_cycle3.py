import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_cycle3_exhausts_loubser_sample_and_reaches_host_boundary() -> None:
    report = json.loads(
        (ROOT / "results/r1_replacement_search_cycle3/report.json").read_text(
            encoding="utf-8"
        )
    )
    summary = report["summary"]
    assert summary["source_bcg_hosts"] == 32
    assert summary["previously_screened_overlaps"] == 5
    assert summary["cycle3_new_unique_hosts"] == 27
    assert summary["cumulative_unique_hosts_source_screened"] == 45
    assert summary["inventory_boundary_reached"] is True
    assert summary["new_structural_promotions"] == 0
    assert summary["strict_r1_ready_systems"] == 0
    assert report["decision"]["host_count_gate"] == "passed"
    assert report["decision"]["strict_readiness_gate"].startswith("failed")


def test_cycle3_ledger_preserves_exact_public_data_blockers() -> None:
    ledger = pd.read_csv(
        ROOT / "data/derived/r1_replacement_cycle3_candidate_ledger.csv"
    )
    assert len(ledger) == 32
    assert ledger["system"].nunique() == 32
    assert ledger["new_unique_host_in_cycle3"].sum() == 27
    assert ledger["radial_profile_machine_readable"].sum() == 0
    assert ledger["central_sigma_and_power_law_slope_machine_readable"].all()
    assert ledger["measurement_covariance_published"].sum() == 0
    assert ledger["profile_plot_sha256"].str.len().eq(64).all()
    assert ledger["exclusion_reason"].str.len().gt(30).all()
    assert not ledger["full_r1_ready"].any()
