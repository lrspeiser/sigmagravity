from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def report() -> dict:
    return json.loads((ROOT / "results" / "sigma_v19co_checkpoint_attempt_metadata_correction" / "report.json").read_text(encoding="utf-8"))


def test_v19co_passes_full_chain() -> None:
    payload = report()
    assert payload["status"] == "checkpoint_metadata_corrected_full_archive_and_source_chain_complete"
    assert payload["decision"] == "run_frozen_v19bs_disposition_next"
    assert all(payload["gate_results"].values())


def test_v19co_only_changes_attempt_metadata() -> None:
    assert report()["json_differences"] == {"attempt": {"before": 3, "after": 1}}
    assert report()["independent_checkpoint_audit"]["attempt"] == 1


def test_v19co_preserves_products_and_physics_boundary() -> None:
    payload = report()
    assert payload["product_hashes_after"] == {
        "source_pha": "0d86f31b56ca6083b6e9ae084d530ec0298b23ddd3e21914d04b3d581a32ad11",
        "background_pha": "21fb9599f0261b0a2efee09f5f4ac7383df911e411496727b24b4a60938fb333",
        "arf": "94a6f720f8e74e08fb4639fde76c925f1d047b302741b7b3e7e20712d3b9deeb",
        "rmf": "d5f833951150257ce2f0e7b66b9f76005d1cc6485d5e7f832b5f47627e2471ce",
    }
    boundary = payload["authorization_boundary"]
    assert not boundary["v19bs_run"] and not boundary["action_derived"]
    assert not boundary["target_or_gravity_opened"] and not boundary["solar_optimized"]
