from __future__ import annotations

import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_complete_x4_response_package_passes_frozen_gate() -> None:
    report = json.loads(
        (ROOT / "results/r1_rxj2129_xmm_x4_responses/report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["report_version"] == "R1B3-RXJ2129-XMM-X4-products-0.2-input-bound"
    assert set(report["inputs"]) == {
        "protocol",
        "map_convergence",
        "production_runner",
        "audit_implementation",
    }
    for record in report["inputs"].values():
        path = ROOT / record["path"]
        assert path.is_file()
        assert path.stat().st_size == record["bytes"]
        assert sha256(path) == record["sha256"]
    assert report["status"] == "pass"
    assert report["product_counts"] == report["expected_product_counts"] == {
        "rmfs": 12,
        "direct_arfs": 12,
        "central_source_arfs": 12,
        "cross_region_arfs": 72,
    }
    assert set(report["instruments"]) == {"MOS2", "pn"}
    for instrument in report["instruments"].values():
        assert instrument["status"] == "pass"
        assert all(item["valid"] for item in instrument["detector_maps"].values())
        assert all(item["passed"] for item in instrument["outputs"].values())
        assert instrument["file_counts"] == instrument["expected_file_counts"]
    assert report["authorization"]["construct_X5_joint_likelihood_scaffold"] is True
    assert report["authorization"]["fit_temperature_or_density"] is False
    assert report["authorization"]["fit_new_force_or_action"] is False


def test_x4_manifest_has_all_response_products_and_hashes() -> None:
    manifest = json.loads(
        (
            ROOT / "data/derived/r1_rxj2129_xmm_x4_response_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["manifest_version"] == (
        "R1B3-RXJ2129-XMM-X4-products-0.2-input-bound"
    )
    products = manifest["products"]
    response_products = [item for item in products if item["kind"] != "detector_map"]
    detector_maps = [item for item in products if item["kind"] == "detector_map"]
    assert len(response_products) == 108
    assert len(detector_maps) == 8
    assert len(products) == 116
    assert len({item["path"] for item in products}) == len(products)
    assert all(item["bytes"] > 0 for item in products)
    assert all(len(item["sha256"]) == 64 for item in products)
