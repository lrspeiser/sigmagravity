from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0675_spent_rxj2129_compound_raw_topology"


def report() -> dict:
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_compound_raw_topology_fails_declared_gates():
    result = report()
    assert result["status"] == "fail"
    assert result["all_progression_gates_pass"] is False
    assert result["candidate_advanced_to_spent_robustness"] is False
    failed = {name for name, passed in result["gate_results"].items() if not passed}
    assert failed == {
        "training_improvement",
        "heldout_absolute_RMS",
        "compact_halo_comparison",
        "no_missing_multiplicity",
        "acceptable_multiplicity",
        "parity_diversity",
        "critical_curves",
        "nuisance_bounds",
    }


def test_compound_field_does_not_recover_lens_topology():
    result = report()
    comparison = result["comparisons"]
    topology = result["topology"]["compound_absolute_P0673"]
    assert comparison["compound_training_improvement_fraction_vs_scalar"] < 0.0
    assert comparison["compound_to_compact_halo_heldout_RMS_ratio"] > 7.0
    assert topology["missing_multiplicity_families"] == 7
    assert topology["total_global_roots"] == 7
    assert topology["total_observed_images"] == 22
    assert topology["parity_diverse_families"] == 0
    assert topology["critical_curve_present_families"] == 0


def test_every_family_has_one_root_and_no_critical_curve():
    families = pd.read_csv(RESULTS / "family_topology.csv")
    compound = families[families.variant_id.eq("compound_absolute_P0673")]
    assert len(compound) == 7
    assert compound.global_roots.eq(1).all()
    assert compound.multiplicity_classification.eq("missing_multiplicity").all()
    assert ~compound.parity_diverse.astype(bool).any()
    assert ~compound.critical_curve_present.astype(bool).any()


def test_sources_parameter_accounting_and_seals_are_preserved():
    result = report()
    assert result["protocol_sha256"] == digest(
        ROOT / "configs/p0675_spent_rxj2129_compound_raw_topology.json"
    )
    assert result["source_sha256"] == digest(
        ROOT / "scripts/run_p0675_spent_rxj2129_compound_raw_topology.py"
    )
    assert result["coverage"]["gravity_parameters"] == 0
    assert result["coverage"]["photon_amplitudes"] == 0
    assert result["sealed_P0633_kinematics_opened"] is False
    assert result["sealed_P0640_lensing_constraints_opened"] is False
    assert (RESULTS / "p0675_compound_raw_topology.png").stat().st_size > 100000
