import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "sigma_complete_action"


def test_complete_action_report_preserves_plain_language_outcome():
    report = json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))
    best = report["best_feedback_row"]

    assert best["chi"] == pytest.approx(0.003)
    assert best["joint_RMSE_dex"] == pytest.approx(0.2303616380)
    assert best["galaxy_typical_factor"] == pytest.approx(1.7372933097)
    assert best["cluster_typical_factor"] == pytest.approx(1.6612695667)
    assert report["scope"]["same_chi_controls_feedback_and_field_energy"] is True
    assert report["scope"]["independent_scalar_mass_amplitude_fit"] is False
    assert "less than one part in a thousand" in report["plain_language_results"][
        "feedback_change"
    ]


def test_complete_action_grids_capture_the_field_mass_tradeoff():
    feedback = pd.read_csv(RESULTS / "backreaction_sweep.csv")
    stress = pd.read_csv(RESULTS / "stress_energy_grid.csv")

    assert len(feedback) == 4
    assert len(stress) == 800
    assert feedback["galaxy_solver_converged"].all()
    assert feedback["cluster_solver_converged"].all()

    row = stress[
        stress["eta"].eq(0.6)
        & stress["log10_rho_s_g_cm3"].eq(-23.5)
        & stress["L_Sigma_kpc"].eq(3.0)
        & stress["log10_chi"].eq(-10.0)
    ].iloc[0]
    assert row["field_to_baryon_mass_at_20kpc"] == pytest.approx(20.2813650)
    assert row["field_to_baryon_mass_at_100kpc"] == pytest.approx(1.5499441)
    assert row["galaxy_typical_factor"] > 20.0
