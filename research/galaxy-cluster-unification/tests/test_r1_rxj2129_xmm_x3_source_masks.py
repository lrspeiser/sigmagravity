from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_rxj2129_x3_mask_retains_only_the_frozen_central_target_component() -> None:
    manifest = json.loads(
        (ROOT / "data/derived/r1_rxj2129_xmm_x3_source_mask_manifest.json").read_text()
    )
    assert manifest["manifest_version"].endswith("0.3")
    assert manifest["catalog_detection_count_unchanged"] == 87
    assert manifest["X3_exclusion_count"] == 86
    central = manifest["central_target_component"]
    assert central["source_id"] == 50
    assert central["separation_from_frozen_center_arcsec"] < central[
        "original_mask_radius_arcsec"
    ]
    assert central["separation_from_frozen_center_arcsec"] < 1.1
    assert all(manifest["gates"].values())
    for instrument in ("MOS2", "pn"):
        for coordinate in ("detector", "sky"):
            product = manifest["products"][instrument][coordinate]
            assert product["original_rows"] == 87
            assert product["derived_rows"] == 86
            assert len(product["original_sha256"]) == 64
            assert len(product["derived_sha256"]) == 64
    compact = manifest["annular_compact_masks"]
    assert list(compact) == [
        "a01_010_050kpc",
        "a02_050_100kpc",
        "a03_100_175kpc",
        "a04_175_275kpc",
        "a05_275_380kpc",
        "a06_380_500kpc",
    ]
    assert compact["a05_275_380kpc"]["exclusion_count"] == 0
    assert compact["a06_380_500kpc"]["exclusion_count"] == 0
    for annulus in compact.values():
        assert 0 <= annulus["exclusion_count"] <= 86
        assert len(annulus["intersecting_source_ids"]) == annulus["exclusion_count"]
        assert 50 not in annulus["intersecting_source_ids"]
        for instrument in ("MOS2", "pn"):
            assert annulus["products"][instrument]["detector"]["derived_rows"] == annulus[
                "exclusion_count"
            ]
            assert annulus["products"][instrument]["sky"]["derived_rows"] == annulus[
                "exclusion_count"
            ]
