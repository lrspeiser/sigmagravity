from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_sigma_v19dh_direct_response_parity as v19dh


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_frozen_runner_hashes_are_exact() -> None:
    for stem in (
        "sigma_v19dg_hierarchical_response_equivalence",
        "sigma_v19dg2_hierarchical_response_equivalence",
        "sigma_v19dh_direct_response_parity",
    ):
        config = load(f"configs/{stem}.json")
        runner = ROOT / config["implementation"]["runner"]
        assert sha256(runner) == config["implementation"]["runner_sha256"]


def test_hierarchical_method_remains_rejected() -> None:
    report = load(
        "results/sigma_v19dg2_hierarchical_response_equivalence/report.json"
    )
    assert report["status"] == (
        "hierarchical_response_equivalence_failed_no_successor_authorized"
    )
    assert report["aggregate_pass"] is False
    assert all(control["passed"] is False for control in report["controls"])
    assert report["full_combination_executed"] is False
    assert report["spectrum_fitted"] is False


def test_direct_array_suffix_controls_pass_every_gate() -> None:
    report = load("results/sigma_v19dh_direct_response_parity/report.json")
    assert report["status"] == (
        "direct_array_response_parity_passed_full_successor_may_be_frozen"
    )
    assert report["aggregate_pass"] is True
    assert len(report["controls"]) == 2
    for control in report["controls"]:
        assert control["cells"] == 128
        assert control["passed"] is True
        assert all(control["gates"].values())
        assert control["evidence"]["rmf_elements_at_or_above_addresp_threshold"] == 0
    assert report["full_combination_executed"] is False
    assert report["spectrum_fitted"] is False


def test_suffix_controls_do_not_overlap_exploratory_prefixes() -> None:
    index = ROOT / (
        "results/sigma_v19x2_unified_spectral_combination_commissioning/"
        "validated_cell_index.csv"
    )
    with index.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for cluster in ("BULLET", "ABELL2146"):
        names = [row["cell_name"] for row in rows if row["cluster"] == cluster]
        assert set(names[:64]).isdisjoint(names[-128:])


def test_folded_diagnostic_is_zero_for_identical_matrices() -> None:
    energy_lo = np.array([0.5, 1.0])
    energy_hi = np.array([1.0, 2.0])
    arf = np.array([10.0, 20.0])
    matrix = np.array([[0.8, 0.2], [0.1, 0.9]])
    diagnostics = v19dh.folded_diagnostics(
        energy_lo, energy_hi, arf, matrix, matrix.copy()
    )
    assert diagnostics
    assert all(value == 0.0 for value in diagnostics.values())
