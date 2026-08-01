import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_rxj2129_chandra_raw_event_gate() -> None:
    report = json.loads(
        (ROOT / "results/r1_rxj2129_chandra_event_audit/report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["archive_file_records"] == 61
    assert report["evt2_hashes_verified"] is True
    assert len(report["observations"]) == 2
    assert np.isclose(report["combined_exposure_ks"], 39.6040961730402)
    assert report["combined_soft_events_inside_5arcsec"] == 1434
    assert report["four_inner_soft_event_counts"] == [114, 275, 513, 532]
    assert report["raw_event_adequacy_gate_pass"] is True
    assert report["raw_public_data_shortfall"] is False
    assert report["gas_density_or_mass_inferred"] is False
    assert report["gravity_or_independent_lens_residual_used"] is False


def test_rxj2129_chandra_event_ledger_is_complete() -> None:
    ledger = pd.read_csv(
        ROOT / "data/derived/r1_rxj2129_chandra_event_adequacy.csv"
    )
    assert set(ledger["scope"]) == {"obsid_9370", "obsid_552", "combined"}
    assert set(ledger["band"]) == {
        "soft_imaging",
        "spectral_screen",
        "particle_control",
    }
    assert len(ledger) == 3 * 3 * 8
    assert np.isfinite(ledger["events"]).all()
    combined_soft = ledger[
        (ledger["scope"] == "combined") & (ledger["band"] == "soft_imaging")
    ]
    inner = combined_soft[combined_soft["outer_arcsec"] <= 5.0]
    assert inner["events"].sum() == 1434
    assert (inner["conservative_source_fraction"] >= 0.75).all()
