from __future__ import annotations

import json
from pathlib import Path

from voidscreen.preregistration import protocol_sha256, validate_p0633_protocol

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "p0633_external_validation_preregistration.json"
RESULTS = ROOT / "results" / "p0633_external_validation_preregistration"


def load_protocol() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def load_ledger() -> dict:
    return json.loads((RESULTS / "ledger.json").read_text(encoding="utf-8"))


def test_protocol_is_frozen_and_structurally_strict():
    protocol = load_protocol()
    validate_p0633_protocol(protocol)
    assert protocol["selection_boundary"]["maximum_per_object_gravity_parameters"] == 0
    assert len(protocol["galaxy_validation"]["systems"]) == 13
    assert len(protocol["cluster_validation"]["systems"]) == 4


def test_ledger_proves_targets_were_absent_and_unopened():
    ledger = load_ledger()
    assert ledger["status"] == "frozen_targets_verified_and_unopened"
    assert ledger["targets"]["baseline_alias_matches"] == 0
    assert ledger["target_directories_present"] == []
    assert ledger["target_products_opened"] is False
    assert all(row["matches"] == 0 for row in ledger["contamination_scan"])


def test_ledger_hashes_the_exact_protocol():
    assert load_ledger()["protocol_sha256"] == protocol_sha256(load_protocol())


def test_rejection_thresholds_cannot_be_rescued_by_domain_averaging():
    gates = load_protocol()["rejection_thresholds"]
    assert gates["galaxy"]["equal_galaxy_RMSE_ratio_to_best_frozen_MOND_max"] == 1.05
    assert gates["cluster"]["heldout_image_RMS_ratio_to_compact_halo_max"] == 1.25
    assert gates["cluster"]["heldout_root_convergence_fraction_min"] == 1.0
    assert gates["cluster"]["all_heldout_family_topologies_correct"] is True
    assert gates["solar_system"]["metric_PPN_quantities_must_be_derived_not_assumed"] is True
    assert "cannot rescue" in gates["overall"]
