from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_hst_psf_protocol_passes_without_authorizing_gravity() -> None:
    protocol = json.loads(
        (ROOT / "configs/r1_rxj2129_hst_psf_protocol.json").read_text(encoding="utf-8")
    )
    report = json.loads(
        (ROOT / "results/r1_rxj2129_hst_psf/report.json").read_text(encoding="utf-8")
    )
    assert protocol["authorization"]["gravity_response_fit"] is False
    assert report["candidate_count"] >= protocol["candidate_selection"]["minimum_candidates"]
    assert report["both_filters_gate_pass"] is True
    assert report["bcg_icl_decomposition_authorized"] is True
    assert report["gravity_response_fit_authorized"] is False
    assert report["strict_r1_ready"] is False


def test_empirical_psfs_are_finite_nonnegative_and_normalized() -> None:
    package = np.load(ROOT / "data/derived/r1_rxj2129_empirical_hst_psf.npz")
    for key in ("f125w", "f814w"):
        psf = package[key]
        yy, xx = np.indices(psf.shape)
        center = 0.5 * (psf.shape[0] - 1)
        aperture = np.hypot(xx - center, yy - center) <= 10.0
        assert psf.shape == (51, 51)
        assert np.all(np.isfinite(psf))
        assert np.min(psf) >= 0
        assert np.isclose(psf[aperture].sum(), 1.0)
        assert np.unravel_index(np.argmax(psf), psf.shape) == (25, 25)


def test_psf_star_ledger_records_both_filters_and_all_gates() -> None:
    ledger = pd.read_csv(ROOT / "data/derived/r1_rxj2129_empirical_hst_psf_stars.csv")
    assert set(ledger["filter"]) == {"F125W", "F814W"}
    assert ledger.groupby("filter")["clash_id"].nunique().to_dict() == {
        "F125W": 3,
        "F814W": 3,
    }
    assert ledger["centroid_gate_pass"].all()
    assert ledger["fwhm_gate_pass"].all()
