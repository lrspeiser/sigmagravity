import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_chandra_calibrated_reduction_gate() -> None:
    report = json.loads(
        (ROOT / "results/r1_rxj2129_chandra_reduction/report.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(report["observations"]) == 2
    assert {item["obsid"] for item in report["observations"]} == {552, 9370}
    assert report["combined_retained_exposure_ks"] >= 35.0
    assert report["combined_global_0p7_7keV_counts"] >= 3000
    assert report["calibrated_reduction_gate_pass"] is False
    assert report["status"] == "calibrated_reduction_gate_failed"
    assert report["checks"]["software_versions"] is False
    assert report["checks"]["retained_exposure_fraction_each_observation"] is False
    assert report["checks"]["blank_sky_particle_scales"] is False
    assert report["checks"]["combined_retained_exposure"] is True
    assert report["checks"]["valid_global_responses"] is True
    assert report["checks"]["combined_global_counts"] is True
    assert report["checks"]["inner_soft_observation_compatibility"] is True
    assert report["gas_density_or_mass_inferred"] is False
    assert report["gravity_or_independent_lens_residual_used"] is False
    assert report["gas_profile_fit_authorized"] is False
    assert report["weyl_or_dynamical_response_authorized"] is False
    assert report["strict_r1_ready"] is False


def test_chandra_reduction_audit_ledger() -> None:
    ledger = pd.read_csv(
        ROOT / "data/derived/r1_rxj2129_chandra_reduction_audit.csv"
    )
    assert len(ledger) == 8
    assert set(ledger["obsid"]) == {552, 9370}
    assert np.allclose(sorted(ledger["inner_arcsec"].unique()), [0.0, 1.0, 2.0, 3.5])
    assert np.allclose(sorted(ledger["outer_arcsec"].unique()), [1.0, 2.0, 3.5, 5.0])
    assert np.isfinite(ledger.select_dtypes(include=["number"])).all().all()
    assert (ledger["source_soft_counts"] > 0).all()
    assert (ledger["net_soft_counts"] > 0).all()
