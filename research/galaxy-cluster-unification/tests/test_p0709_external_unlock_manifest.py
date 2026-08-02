from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_p0709_authorizes_one_frozen_external_parse() -> None:
    manifest = json.loads(
        (ROOT / "results/p0633_external_validation/unlock_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"] == "authorized_for_exactly_one_external_parse"
    assert manifest["candidate_lock_commit"] == (
        "857feeb85af3b94f3c486c72db00c8f486318d6b"
    )
    assert manifest["universal_parameter_sha256"] == (
        "bf3f12d6b32ee3f1b0e3bf48a9603c4aafbcd34b2cbdd3de021d689514099a15"
    )
    assert manifest["prediction_systems"] == 17
    assert len(manifest["prediction_hashes"]) == 17
    assert len(manifest["galaxy_moment_products"]) == 26
    assert manifest["authorization"]["external_evaluations_authorized"] == 1
    assert manifest["authorization"]["per_object_gravity_parameters"] == 0
    assert manifest["outcomes_opened_at_manifest_creation"] is False
    assert (
        manifest["formula_change_after_this_manifest_invalidates_P0633_validation"]
        is True
    )


def test_p0709_sealed_container_hashes_still_match() -> None:
    manifest = json.loads(
        (ROOT / "results/p0633_external_validation/unlock_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    for item in manifest["sealed_constraint_containers"]:
        path = ROOT / item["path"]
        assert path.stat().st_size == item["bytes"]
        assert sha256(path) == item["sha256"]


def test_p0709_morphology_splits_are_preoutcome_and_complete() -> None:
    manifest = json.loads(
        (ROOT / "results/p0633_external_validation/unlock_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(manifest["morphology_median_splits"]) == {
        "concentration_5log_r80_r20",
        "lopsidedness_180",
        "clumpiness_positive_highpass",
        "inclination_deg",
    }
    comparator = manifest["compact_halo_comparator"]
    assert comparator["primary_method"] == "glafic"
    assert comparator["primary_version"] == "v2"
    assert "Never choose a method per cluster" in comparator["selection_rule"]
