import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "mass_path_completion_full_test"
MODELS = {"mass_weighted_path", "mass_amplified_path", "mass_ceiling_path"}


def _report():
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def test_mass_path_run_has_complete_blinded_coverage():
    report = _report()
    assert set(report["models"]) == MODELS
    assert report["coverage"] == {
        "bridge_rows": 116,
        "bridge_systems": 64,
        "SPARC_galaxies": 131,
        "SPARC_inner_points": 2066,
        "SPARC_outer_points": 968,
        "raw_images_per_model": 22,
    }

    bridge = pd.read_csv(RESULTS / "bridge_predictions.csv")
    sparc = pd.read_csv(RESULTS / "sparc_predictions.csv")
    raw = pd.read_csv(RESULTS / "raw_lensing_predictions.csv")

    assert len(bridge) == 116 * len(MODELS)
    assert len(sparc) == 3034 * len(MODELS)
    assert len(raw) == 22 * len(MODELS)
    assert set(bridge["model"]) == MODELS
    assert set(sparc["model"]) == MODELS
    assert set(raw["mass_path_model"]) == MODELS


def test_universal_g_is_larger_but_local_solar_gravity_is_recovered():
    for result in _report()["models"].values():
        assert result["G_max_over_G_measured"] > 8.0
        assert result["gate_audit"]["solar_Earth_pass"] is True
        assert result["gate_audit"]["bounded_completion_pass"] is True
        assert abs(result["solar"]["1_AU"]["fractional_change"]) <= 1.0e-10


def test_no_mass_dependent_option_passes_all_domains():
    report = _report()
    rows = report["models"]
    assert report["selection"]["all_gate_survivors"] == []
    assert report["verdict"]["any_mass_path_survives"] is False
    assert not any(row["gate_audit"]["bridge_equal_domain_pass"] for row in rows.values())
    assert not any(row["gate_audit"]["SPARC_transfer_pass"] for row in rows.values())
    assert all(row["gate_audit"]["raw_lensing_pass"] for row in rows.values())

    # The best galaxy-transfer result is still substantially worse than fixed RAR.
    ceiling_rmse = rows["mass_ceiling_path"]["SPARC_metrics"]["outer_holdout"]["RMSE_km_s"]
    assert ceiling_rmse == pytest.approx(24.6706907606)
    assert ceiling_rmse > report["references"]["fixed_RAR_SPARC"]


def test_data_push_mass_dependence_to_degenerate_limits():
    rows = _report()["models"]
    assert rows["mass_weighted_path"]["full_fit_at_boundary"]["log10_M_star_solar"] is True
    assert rows["mass_weighted_path"]["full_fit_at_boundary"]["mass_power"] is True
    assert rows["mass_amplified_path"]["full_fit_at_boundary"]["mass_power"] is True
    assert rows["mass_ceiling_path"]["full_fit_at_boundary"]["log10_M_star_solar"] is True
    assert rows["mass_ceiling_path"]["full_fit_at_boundary"]["mass_power"] is True
