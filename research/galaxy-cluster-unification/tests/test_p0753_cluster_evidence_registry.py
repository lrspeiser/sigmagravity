from __future__ import annotations

import json
from pathlib import Path

from scripts.build_p0753_cluster_evidence_registry import build_registry, canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "p0753_cluster_evidence_registry" / "registry.json"
HOSTED = ROOT / "hosted-simulator" / "data" / "resolved-cluster-evidence-v1.json"


def test_registry_is_deterministic_and_matches_hosted_copy():
    generated = build_registry()
    committed = json.loads(RESULT.read_text(encoding="utf-8"))
    hosted = json.loads(HOSTED.read_text(encoding="utf-8"))
    assert generated == committed == hosted
    core = {key: value for key, value in committed.items() if key != "registrySha256"}
    assert committed["registrySha256"] == canonical_sha256(core)


def test_scientific_roles_and_readiness_cannot_be_conflated():
    registry = json.loads(RESULT.read_text(encoding="utf-8"))
    assert registry["sample"]["registeredBaryonMaps"] == 4
    assert registry["sample"]["inverseDiscoverySystems"] == 4
    assert registry["sample"]["rawCatalogReadinessGateSystems"] == 2
    assert registry["sample"]["rawForwardScoreReadySystems"] == 0
    for system in registry["systems"]:
        assert system["baryonicEvidence"]["scientificRole"] == "baryonic_input"
        assert (
            system["modelDerivedLensingEvidence"]["scientificRole"]
            == "model_derived_discovery_target"
        )
        assert system["rawLensingEvidence"]["scientificRole"] == "raw_observation"
        assert not system["rawLensingEvidence"]["scoreReadyNow"]
        assert not system["rawLensingEvidence"]["prospectiveHoldoutEligible"]


def test_registry_preserves_zero_per_cluster_gravity_parameters_and_spent_state():
    registry = json.loads(RESULT.read_text(encoding="utf-8"))
    assert registry["provenance"]["p0710SampleSpent"] is True
    assert registry["sample"]["prospectiveHoldoutSystems"] == 0
    for system in registry["systems"]:
        assert system["sampleState"] == "spent_development"
        policy = system["baryonicEvidence"]["sharedUncertaintyPolicy"]
        assert policy["perClusterGravityParameters"] == 0
        assert len(system["modelDerivedLensingEvidence"]["models"]) == 2
        assert all(
            model["scientificRole"] == "model_derived_discovery_target"
            for model in system["modelDerivedLensingEvidence"]["models"]
        )
