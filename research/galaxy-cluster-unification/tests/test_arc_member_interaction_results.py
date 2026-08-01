import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/arc_member_interaction"


def test_directional_interaction_is_a_controlled_negative_result():
    report = json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))
    assert report["inputs"] == {
        "members": 66,
        "training_images": 15,
        "heldout_images": 7,
        "screen_variants": 722,
        "total_effective_member_stellar_mass_msun": report["inputs"][
            "total_effective_member_stellar_mass_msun"
        ],
    }
    assert report["preservation_checks"]["net_added_radial_member_mass_msun"] == 0.0
    assert report["preservation_checks"]["galaxy_prediction_change"] == 0.0
    assert report["preservation_checks"]["Solar_System_prediction_change"] == 0.0
    assert report["verdict"]["meaningful_improvement_gate_pass"] is False
    assert report["verdict"]["strong_absolute_gate_pass"] is False
    assert report["verdict"]["measured_layout_randomization_gate_pass"] is False


def test_training_selection_does_not_transfer_to_heldout_images():
    final = pd.read_csv(RESULTS / "final_scores.csv")
    for parent in ("P0554", "P0396"):
        chosen = final[(final.parent == parent) & (final.variant == "selected")].iloc[0]
        assert chosen.routing_fraction == 2.0
        assert chosen.member_mass_power == 0.5
        assert chosen.softening_scale == 0.5
        assert chosen.radial_dressing == "none"
    p0554 = final[final.parent.eq("P0554")].set_index("variant")
    assert p0554.loc["selected", "training_RMS_arcsec"] < p0554.loc["baseline", "training_RMS_arcsec"]
    assert p0554.loc["selected", "heldout_RMS_arcsec"] > p0554.loc["baseline", "heldout_RMS_arcsec"]
    p0396 = final[final.parent.eq("P0396")].set_index("variant")
    assert int(p0396.loc["selected", "heldout_converged_roots"]) == 6
    assert not np.isfinite(p0396.loc["selected", "heldout_RMS_arcsec"])


def test_routed_weights_preserve_each_parent_total_budget():
    weights = pd.read_csv(RESULTS / "selected_member_weights.csv")
    expected = 206741121935.85123
    for _, block in weights.groupby("parent"):
        assert np.isclose(block.selected_route_weight_msun.sum(), expected)
        assert np.isclose(block.selected_weight_share.sum(), 1.0)
