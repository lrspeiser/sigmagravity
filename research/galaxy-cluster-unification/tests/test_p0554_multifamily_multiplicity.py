import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "p0554_multifamily_multiplicity_protocol.json"
RESULTS = ROOT / "results" / "p0554_multifamily_multiplicity"


def load_report():
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def test_protocol_was_frozen_before_other_family_root_counts():
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert protocol["status"].startswith("frozen_")
    assert "before_any_other_family_global_root_counts" in protocol["status"]
    assert protocol["variants"] == [
        "baseline",
        "lensing_softness_098",
        "route_parent",
        "combined_parent",
        "combined_power_240",
    ]
    assert protocol["evaluation"]["formula_parameters_fit"] == 0
    assert protocol["evaluation"]["geometry_parameters_refit"] == 0
    assert protocol["evaluation"]["global_grid_spacing_arcsec"] == 12.0
    assert protocol["evaluation"]["interlaced_grid_offset_arcsec"] == 6.0


def test_complete_coverage_and_root_closure():
    report = load_report()
    assert report["report_version"] == "P0554-MULTIFAMILY-MULTIPLICITY-RESULTS-0.2.0"
    assert report["coverage"] == {
        "variants": 5,
        "systems": 5,
        "source_families": 27,
        "formula_family_searches": 135,
        "published_images": 77,
        "accepted_global_roots": 415,
    }
    roots = pd.read_csv(RESULTS / "global_roots.csv")
    assert len(roots) == 415
    assert roots.closure_arcsec.max() < 1.0e-6
    assert len(pd.read_csv(RESULTS / "assignments.csv")) == 385
    assert len(pd.read_csv(RESULTS / "family_summary.csv")) == 135


def test_variant_level_multiplicity_counts_are_frozen():
    summary = pd.read_csv(RESULTS / "variant_summary.csv").set_index("variant_id")
    expected = {
        "baseline": (7, 12, 1, 7, 8, 10.530940033025285),
        "lensing_softness_098": (8, 12, 1, 6, 7, 9.985848591525956),
        "route_parent": (7, 11, 0, 9, 12, 9.227803332709083),
        "combined_parent": (7, 11, 0, 9, 12, 9.177935244299963),
        "combined_power_240": (7, 11, 1, 8, 10, 9.060496716205899),
    }
    for variant_id, values in expected.items():
        row = summary.loc[variant_id]
        assert tuple(
            int(row[column])
            for column in (
                "families_missing_multiplicity",
                "families_exact_multiplicity",
                "families_demagnified_only_surplus",
                "families_potentially_observable_surplus",
                "potentially_observable_surplus_roots",
            )
        ) == values[:5]
        assert np.isclose(row.equal_family_assignment_RMS_arcsec, values[5])


def test_route_changes_root_count_only_in_two_macs1931_families():
    families = pd.read_csv(RESULTS / "family_summary.csv")
    baseline = families[families.variant_id.eq("baseline")].set_index(
        ["system_label", "source_family"]
    )
    route = families[families.variant_id.eq("route_parent")].set_index(
        ["system_label", "source_family"]
    )
    changed = (route.global_roots - baseline.global_roots)[lambda values: values.ne(0)]
    assert changed.to_dict() == {("MACS1931", 2): 2, ("MACS1931", 3): 2}

    power = families[families.variant_id.eq("combined_power_240")].set_index(
        ["system_label", "source_family"]
    )
    assert (power.global_roots - baseline.global_roots)[lambda values: values.ne(0)].to_dict() == {
        ("MACS1931", 2): 2,
        ("MACS1931", 3): 2,
    }


def test_photon_softness_changes_only_macs1931_family1_root_count():
    families = pd.read_csv(RESULTS / "family_summary.csv")
    baseline = families[families.variant_id.eq("baseline")].set_index(
        ["system_label", "source_family"]
    )
    lens = families[families.variant_id.eq("lensing_softness_098")].set_index(
        ["system_label", "source_family"]
    )
    changed = (lens.global_roots - baseline.global_roots)[lambda values: values.ne(0)]
    assert changed.to_dict() == {("MACS1931", 1): -2}


def test_macs1931_family2_reproduces_prior_three_to_five_transition():
    families = pd.read_csv(RESULTS / "family_summary.csv")
    block = families[
        families.system_label.eq("MACS1931") & families.source_family.eq(2)
    ].set_index("variant_id")
    assert block.loc["baseline", "global_roots"] == 3
    assert block.loc["lensing_softness_098", "global_roots"] == 3
    assert block.loc["route_parent", "global_roots"] == 5
    assert block.loc["combined_parent", "global_roots"] == 5
    assert block.loc["combined_power_240", "global_roots"] == 5


def test_surplus_conclusion_is_robust_to_descriptive_threshold_sweep():
    sensitivity = pd.read_csv(RESULTS / "threshold_sensitivity.csv")
    assert len(sensitivity) == 20
    assert set(sensitivity.relative_magnification_threshold) == {0.1, 0.25, 0.5, 1.0}
    for _, block in sensitivity.groupby("relative_magnification_threshold"):
        indexed = block.set_index("variant_id")
        assert (
            indexed.loc["route_parent", "potentially_observable_surplus_roots"]
            > indexed.loc["baseline", "potentially_observable_surplus_roots"]
        )
    primary = sensitivity[sensitivity.relative_magnification_threshold.eq(0.25)].set_index(
        "variant_id"
    )
    assert primary.loc["baseline", "potentially_observable_surplus_roots"] == 8
    assert primary.loc["route_parent", "potentially_observable_surplus_roots"] == 12


def test_no_formula_has_universal_exact_multiplicity_or_is_promoted():
    verdict = load_report()["verdict"]
    assert verdict == {
        "potentially_observable_surplus_occurs_outside_MACS1931_family2": True,
        "any_variant_has_exact_multiplicity_for_every_family": False,
        "observable_surplus_is_a_recurring_issue": True,
        "no_formula_promoted": True,
        "route_has_more_surplus_roots_than_baseline_at_every_checked_threshold": True,
    }
