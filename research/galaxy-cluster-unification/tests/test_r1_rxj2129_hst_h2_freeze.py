from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def test_rxj2129_hst_h2_is_hash_frozen_before_arc_pixel_access() -> None:
    config = json.loads(
        (ROOT / "configs/r1_rxj2129_hst_h2_centroid_execution_protocol.json").read_text()
    )
    report = json.loads(
        (ROOT / "results/r1_rxj2129_hst_h2_freeze/report.json").read_text()
    )
    runner = ROOT / config["implementation"]["runner"]

    assert report["status"] == "pass"
    assert report["HST_arc_pixels_accessed_before_original_runner_hash_freeze"] is False
    assert report["HST_arc_pixels_accessed_during_invalid_first_execution"] is True
    assert report["HST_arc_pixels_accessed_during_this_static_audit"] is False
    assert report["selected_ledger_rows"] == 21
    assert all(report["gates"].values())
    assert report["runner_sha256"] == config["implementation"]["runner_sha256"]
    assert _sha256(runner) == config["implementation"]["runner_sha256"]
    assert report["synthetic_self_test"]["status"] == "pass"
    assert report["synthetic_self_test"]["HST_pixels_accessed"] is False
    assert report["authorization"] == {
        "execute_H2_arc_centroids": True,
        "assemble_H3_covariance": False,
        "use_lens_or_gravity_model": False,
        "infer_dynamical_or_Weyl_response": False,
        "fit_new_force_or_action": False,
    }
