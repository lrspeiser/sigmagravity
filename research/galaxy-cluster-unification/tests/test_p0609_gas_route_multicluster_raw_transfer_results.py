import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/p0609_gas_route_multicluster_raw_transfer"


def test_locked_gas_route_is_conservative_but_does_not_transfer():
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    fields = pd.read_csv(OUTPUT / "field_audits.csv")
    assert report["coverage"]["systems"] == 4
    assert report["coverage"]["variant_system_refits"] == 12
    assert fields.route_map_normalization_error.max() < 1.0e-12
    assert fields.maximum_annular_convergence_mean_fraction.max() < 1.0e-12
    assert fields.normalized_curl_RMS.max() < 1.0e-12
    assert report["transfer_gate"]["all_gates_pass"] is False
    assert report["interpretation"]["standard_gas_route_transfers"] is False
    assert report["interpretation"]["absolute_gas_mass_claimed"] is False


def test_aggregate_gain_is_one_cluster_not_a_universal_effect():
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    comparison = next(
        row for row in report["matched_comparisons"] if row["variant_id"] == "gas_route_gamma1"
    )
    assert comparison["fractional_improvement"] > 0.05
    assert comparison["systems_improved"] == 1
    assert comparison["matched_systems"] == 3
    assert report["transfer_gate"]["systems_improved_pass"] is False
    assert report["transfer_gate"]["absolute_RMS_arcsec"] > 10.0
    assert report["transfer_gate"]["all_heldout_roots_pass"] is False


def test_macs0429_drives_the_only_material_change():
    scores = pd.read_csv(OUTPUT / "system_scores.csv").set_index(
        ["system_label", "variant_id"]
    )
    base = scores.loc[("MACS0429", "P0599_no_route"), "heldout_RMS_arcsec"]
    route = scores.loc[("MACS0429", "gas_route_gamma1"), "heldout_RMS_arcsec"]
    assert 1.0 - route / base > 0.40
    for label in ("MACS0329", "MACS1115"):
        base = scores.loc[(label, "P0599_no_route"), "heldout_RMS_arcsec"]
        route = scores.loc[(label, "gas_route_gamma1"), "heldout_RMS_arcsec"]
        assert route >= base
