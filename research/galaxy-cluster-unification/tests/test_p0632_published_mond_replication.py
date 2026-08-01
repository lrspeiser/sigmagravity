from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0632_published_mond_replication"


def report():
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def test_published_replication_passes_every_frozen_gate():
    result = report()
    assert result["published_replication_pass"] is True
    assert all(result["gate_results"].values())
    reproduction = result["published_reproduction"]
    assert reproduction["published_table_entries"] == 175
    assert reproduction["scatter_sample_galaxies"] == 153
    assert reproduction["scatter_points"] == 2694
    assert abs(reproduction["replayed_scatter_dex"] - 0.057) < 0.001
    assert reproduction["reduced_chi_square_correlation"] > 0.999


def test_fixed_input_scatter_reproduces_published_value():
    result = report()
    fixed = result["strict_no_nuisance"]["li2018_rar_mond"]
    assert fixed["points"] == 2694
    assert abs(fixed["log_acceleration_residual"]["standard_deviation"] - 0.13) < 0.01
    assert fixed["equal_galaxy_velocity_RMSE_km_s"] < 25.0


def test_whole_galaxy_holdout_is_strict_and_complete():
    scores = pd.read_csv(RESULTS / "whole_galaxy_holdout_scores.csv")
    assert set(scores.law) == {
        "baryons",
        "li2018_rar_mond",
        "simple_mond",
        "standard_mond",
    }
    assert (scores.galaxies == 23).all()
    assert (scores.points == 427).all()
    mond = scores.loc[scores.law.eq("li2018_rar_mond")].iloc[0]
    assert np.isclose(mond.equal_galaxy_velocity_RMSE_km_s, 23.325730076930643)


def test_representative_virtual_telescope_outputs_exist():
    for galaxy in ("DDO154", "IC2574", "NGC2403", "NGC2841"):
        path = RESULTS / "representatives" / f"{galaxy}_mond_comparison.png"
        assert path.stat().st_size > 10000
