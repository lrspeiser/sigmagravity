import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "path_completion_full_test"


def load_report() -> dict:
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def test_all_predefined_path_models_and_samples_were_run():
    report = load_report()
    assert list(report["models"]) == [
        "distance_path",
        "tidal_path",
        "matter_path",
        "hybrid_path",
    ]
    assert report["coverage"] == {
        "bridge_rows": 116,
        "bridge_systems": 64,
        "SPARC_galaxies": 131,
        "SPARC_inner_points": 2066,
        "SPARC_outer_points": 968,
        "raw_images_per_model": 22,
    }


def test_universal_G_is_allowed_above_measured_G_and_solar_limit_is_recovered():
    report = load_report()
    for model in report["models"].values():
        assert 8.0 < model["G_max_over_G_measured"] < 8.5
        assert abs(model["solar"]["1_AU"]["fractional_change"]) < 1.0e-10
        assert model["bridge_completion"]["maximum"] <= 1.0
        assert model["gate_audit"]["solar_Earth_pass"] is True
        assert model["gate_audit"]["bounded_completion_pass"] is True


def test_path_family_improves_raw_lensing_but_fails_sparc_transfer():
    report = load_report()
    raw_reference = report["references"]["raw_lensing"]["compact_halo"]
    for model in report["models"].values():
        assert model["raw_lensing"]["heldout"]["exact_radial_RMS_arcsec"] < raw_reference
        assert model["SPARC_metrics"]["outer_holdout"]["RMSE_km_s"] > 30.0
        assert model["gate_audit"]["raw_lensing_pass"] is True
        assert model["gate_audit"]["SPARC_transfer_pass"] is False


def test_no_path_law_survives_and_extra_switches_hit_boundaries():
    report = load_report()
    assert report["selection"]["all_gate_survivors"] == []
    assert report["verdict"]["any_universal_path_law_passes"] is False
    assert report["models"]["matter_path"]["bridge_metrics"][
        "equal_domain_RMSE_dex"
    ] == pytest.approx(0.1311311749)
    assert report["models"]["matter_path"]["full_fit_at_boundary"][
        "log10_g_star_m_s2"
    ] is True
    assert report["models"]["tidal_path"]["full_fit_at_boundary"][
        "log10_T_star_s2"
    ] is True
    assert report["models"]["hybrid_path"]["full_fit_at_boundary"][
        "log10_T_star_s2"
    ] is True


def test_result_tables_have_complete_model_coverage():
    bridge = pd.read_csv(RESULTS / "bridge_predictions.csv")
    sparc = pd.read_csv(RESULTS / "sparc_predictions.csv")
    raw = pd.read_csv(RESULTS / "raw_lensing_predictions.csv")
    assert len(bridge) == 116 * 4
    assert len(sparc) == (2066 + 968) * 4
    assert len(raw) == 22 * 4
