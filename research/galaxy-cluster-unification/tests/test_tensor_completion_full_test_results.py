import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "results" / "tensor_completion_full_test"
TRADEOFF = ROOT / "results" / "tensor_directional_tradeoff"
MODELS = {
    "tensor_isotropic",
    "tensor_alignment",
    "tensor_competition",
    "tensor_dominance",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_primary_tensor_run_has_complete_locked_coverage():
    report = load(PRIMARY / "report.json")
    assert set(report["models"]) == MODELS
    assert report["coverage"] == {
        "bridge": {"rows": 116, "systems": 64},
        "SPARC": {"galaxies": 131, "inner_points": 2066, "outer_points": 968},
        "raw_lensing_images_per_model": 22,
    }
    assert len(pd.read_csv(PRIMARY / "bridge_predictions.csv")) == 116 * 4
    assert len(pd.read_csv(PRIMARY / "sparc_predictions.csv")) == 3034 * 4
    assert len(pd.read_csv(PRIMARY / "raw_lensing_predictions.csv")) == 22 * 4


def test_directionality_does_not_survive_blind_selection_or_sparc_transfer():
    report = load(PRIMARY / "report.json")
    assert report["selection"]["selected_model"] == "tensor_isotropic"
    for name, result in report["models"].items():
        assert result["gate_audit"]["bridge_equal_domain_pass"] is True
        assert result["gate_audit"]["SPARC_transfer_pass"] is False
        assert result["gate_audit"]["raw_lensing_pass"] is True
        assert result["gate_audit"]["solar_Earth_pass"] is True
        assert result["gate_audit"]["bounded_completion_pass"] is True
        if name != "tensor_isotropic":
            assert result["full_fit_at_boundary"]["q"] is True
            assert result["full_fit_parameters"]["q"] == pytest.approx(0.1)

    assert report["models"]["tensor_isotropic"]["SPARC_metrics"]["outer_holdout"][
        "RMSE_km_s"
    ] == pytest.approx(28.9181353935)
    assert report["models"]["tensor_competition"]["raw_lensing"]["heldout"][
        "exact_radial_RMS_arcsec"
    ] == pytest.approx(1.1419384811)


def test_larger_universal_g_and_solar_limit_are_both_realized():
    report = load(PRIMARY / "report.json")
    for result in report["models"].values():
        assert result["completion"]["G_max_over_G_solar"] > 9.0
        assert result["completion"]["maximum_completion_fraction"] <= 1.0
        assert abs(result["solar"]["Earth_fractional_change"]) <= 1.0e-10


def test_exact_postfailure_directional_refits_confirm_the_tradeoff():
    report = load(TRADEOFF / "report.json")
    assert set(report["results"]) == {"alignment_q035", "dominance_q035"}
    assert "not independent" in report["claim"]
    assert len(pd.read_csv(TRADEOFF / "bridge_predictions.csv")) == 116 * 2
    assert len(pd.read_csv(TRADEOFF / "sparc_predictions.csv")) == 3034 * 2
    assert len(pd.read_csv(TRADEOFF / "raw_lensing_predictions.csv")) == 22 * 2

    for result in report["results"].values():
        assert result["bridge_metrics"]["equal_domain_RMSE_dex"] < 0.15
        assert result["SPARC_metrics"]["outer_holdout"]["RMSE_km_s"] > 36.0
        assert result["raw_lensing"]["training"]["all_roots_converged"] is False
        assert result["raw_lensing"]["heldout"]["all_roots_converged"] is False
        assert result["raw_lensing"]["heldout"]["exact_radial_RMS_arcsec"] is None
