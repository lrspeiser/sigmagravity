from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0640_relics_input_audit"
ACQUISITION = ROOT / "results" / "p0640_relics_input_acquisition" / "provenance.json"


def test_all_preregistered_cluster_baryonic_inputs_are_real_and_ready():
    report = json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))
    systems = pd.read_csv(RESULTS / "systems.csv")
    assert report["status"] == "ready"
    assert report["totals"]["systems"] == 4
    assert len(systems) == 4
    assert (systems["hard_photoz_members"] >= 200).all()
    assert (systems["hard_photoz_f160w_5sigma_members"] >= 90).all()
    assert (systems["segmentation_catalog_match_fraction"] >= 0.995).all()
    assert (systems["hst_finite_fraction"] >= 0.15).all()
    assert (systems["hst_infinite_pixels"] == 0).all()


def test_chandra_morphology_has_real_2d_coverage_for_every_cluster():
    systems = pd.read_csv(RESULTS / "systems.csv")
    observations = pd.read_csv(RESULTS / "chandra_observations.csv")
    assert len(observations) == 19
    assert observations["valid"].all()
    assert (observations.groupby("system").size() >= 3).all()
    assert (systems["chandra_exposure_ks"] >= 70.0).all()
    assert (systems["chandra_counts"] > 0).all()


def test_lensing_constraints_remain_opaque_and_derived_maps_are_absent():
    report = json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))
    acquisition = json.loads(ACQUISITION.read_text(encoding="utf-8"))
    sealed = [row for row in acquisition["records"] if row["kind"] == "sealed_constraint_container"]
    assert len(sealed) == 2
    assert not any(report["sealed_state"].values())
    assert report["gates"]["two_lensing_sources_sealed_opaque"] is True
    assert report["gates"]["no_derived_lens_map_in_baryon_inputs"] is True


def test_every_acquired_artifact_has_size_and_sha256_provenance():
    acquisition = json.loads(ACQUISITION.read_text(encoding="utf-8"))
    assert acquisition["status"] == "downloaded_and_hashed_without_opening_sealed_payloads"
    assert acquisition["files"] == 34
    assert all(row["bytes"] > 0 for row in acquisition["records"])
    assert all(len(row["sha256"]) == 64 for row in acquisition["records"])
