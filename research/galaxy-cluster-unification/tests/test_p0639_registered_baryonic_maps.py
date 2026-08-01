from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0639_registered_baryonic_maps"


def test_all_thirteen_registered_maps_pass_mass_and_boundary_gates():
    report = json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(RESULTS / "map_audit.csv")
    assert report["status"] == "pass"
    assert report["all_gates_pass"] is True
    assert len(frame) == 13
    assert frame["all_gates_pass"].all()


def test_every_map_is_finite_nonnegative_and_contains_the_declared_mass():
    frame = pd.read_csv(RESULTS / "map_audit.csv").set_index("galaxy")
    for galaxy, row in frame.iterrows():
        with np.load(RESULTS / "maps" / f"{galaxy}.npz") as maps:
            axis = maps["axis_kpc"]
            spacing = float(axis[1] - axis[0])
            for key in ("gas", "stars", "total"):
                assert maps[key].shape == (int(row["cells_per_axis"]),) * 2
                assert 65 <= maps[key].shape[0] <= 513
                assert maps[key].shape[0] % 2 == 1
                assert np.isfinite(maps[key]).all()
                assert (maps[key] >= 0.0).all()
            assert np.isclose(np.sum(maps["gas"]) * spacing**2, row["gas_mass_solar"])
            assert np.isclose(np.sum(maps["stars"]) * spacing**2, row["stellar_mass_solar"])
            assert np.allclose(maps["total"], maps["gas"] + maps["stars"])


def test_no_sealed_outcome_or_per_object_gravity_parameter_was_used():
    report = json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))
    assert report["sealed_target_observables_opened"] is False
    assert report["per_object_gravity_parameters"] == 0
