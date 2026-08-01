import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "vector_completion_full_test"


def load_report() -> dict:
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def test_full_test_uses_all_frozen_samples():
    coverage = load_report()["coverage"]
    assert coverage["bridge"] == {"rows": 116, "systems": 64}
    assert coverage["SPARC"] == {
        "galaxies": 131,
        "inner_points": 2066,
        "outer_points": 968,
    }
    assert coverage["raw_lensing_images_per_model"] == 22


def test_isotropic_completion_is_selected_without_sparc_or_raw_tuning():
    report = load_report()
    selection = report["selection"]
    model = report["models"]["isotropic_completion"]

    assert selection["selected_model"] == "isotropic_completion"
    assert selection["parameters_selected_from_SPARC"] == 0
    assert selection["parameters_selected_from_raw_images"] == 0
    assert model["full_fit_parameters"]["C_solar"] == pytest.approx(0.1047188747)
    assert model["completion"]["G_max_over_G_solar"] == pytest.approx(9.5493768663)
    assert model["completion"]["maximum_completion_fraction"] <= 1.0


def test_completion_fits_bridge_and_raw_lensing_but_fails_galaxy_transfer():
    report = load_report()
    model = report["models"]["isotropic_completion"]
    gates = model["gate_audit"]

    assert model["bridge_metrics"]["equal_domain_RMSE_dex"] == pytest.approx(
        0.1214954231
    )
    assert model["raw_lensing"]["heldout"]["exact_radial_RMS_arcsec"] == pytest.approx(
        1.2875348092
    )
    assert model["SPARC_metrics"]["outer_holdout"]["RMSE_km_s"] == pytest.approx(
        26.2275997449
    )
    assert gates["bridge_equal_domain_pass"] is True
    assert gates["raw_lensing_pass"] is True
    assert gates["SPARC_transfer_pass"] is False
    assert gates["all_observational_gates_pass"] is False


def test_coherence_extension_is_rejected_at_its_parameter_boundary():
    model = load_report()["models"]["coherence_completion"]
    assert model["full_fit_parameters"]["q"] == pytest.approx(0.1)
    assert model["full_fit_at_boundary"]["q"] is True
    assert model["SPARC_metrics"]["outer_holdout"]["RMSE_km_s"] > 30.0


def test_result_tables_have_all_model_rows():
    bridge = pd.read_csv(RESULTS / "bridge_predictions.csv")
    sparc = pd.read_csv(RESULTS / "sparc_predictions.csv")
    raw = pd.read_csv(RESULTS / "raw_lensing_predictions.csv")
    assert len(bridge) == 116 * 2
    assert len(sparc) == (2066 + 968) * 2
    assert len(raw) == 22 * 2
