from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_adaptive_route_multicluster_raw import (
    adjusted_candidate,
    load_member_sources,
    matched_comparison,
)


def test_member_loader_normalizes_light_and_applies_aperture(tmp_path: Path):
    rows = []
    for index in range(12):
        rows.append(f"{index} {index * 0.0001:g} 0 1 1 0 {18 + index * 0.1:g} 1")
    rows.append("99 1 0 1 1 0 20 1")
    path = tmp_path / "members.dat"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    system = {"label": "TEST", "center_ra_deg": 0.0, "center_dec_deg": 0.0}
    local = {"cosmology_and_coordinates": {"angular_scale_kpc_per_arcsec": 1.0}}
    settings = {
        "columns": [
            "member_id", "RA_deg", "Dec_deg", "axis_a", "axis_b",
            "position_angle", "magnitude", "flag",
        ],
        "aperture_kpc": 300.0,
        "minimum_members": 10,
    }
    result = load_member_sources(path, system, local, settings)
    assert len(result) == 12
    assert np.isclose(result.base_weight.sum(), 1.0)
    assert result.radius_kpc.max() <= 300.0
    assert result.iloc[0].base_weight > result.iloc[-1].base_weight


def test_adjusted_candidate_changes_only_declared_parameter():
    candidate = pd.Series({"base_fraction": 0.45, "base_length_kpc": 250.0})
    changed = adjusted_candidate(
        candidate,
        {"parameter": "base_length_kpc", "multiplier": 0.8},
    )
    assert changed.base_fraction == candidate.base_fraction
    assert changed.base_length_kpc == 200.0
    assert candidate.base_length_kpc == 250.0


def test_matched_comparison_does_not_treat_failed_root_as_improvement():
    frame = pd.DataFrame(
        [
            {"system_label": "A", "variant": "scalar", "heldout_all_roots": True, "heldout_RMS_arcsec": 10.0},
            {"system_label": "B", "variant": "scalar", "heldout_all_roots": False, "heldout_RMS_arcsec": np.inf},
            {"system_label": "A", "variant": "route", "heldout_all_roots": True, "heldout_RMS_arcsec": 9.0},
            {"system_label": "B", "variant": "route", "heldout_all_roots": True, "heldout_RMS_arcsec": 8.0},
        ]
    )
    result = matched_comparison(frame, "scalar", "route")
    assert result["matched_labels"] == ["A"]
    assert result["all_requested_systems_comparable"] is False
    assert np.isclose(result["fractional_improvement"], 0.1)
