from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_level2_inventory_is_complete_and_residual_blind() -> None:
    config = json.loads(
        (ROOT / "configs/r1_m1206_level2_products.json").read_text(encoding="utf-8")
    )
    assert len(config["products"]) == 6
    assert all(product["dp_id"].startswith("ADP.2016-06-") for product in config["products"])
    assert abs(
        sum(product["exposure_seconds"] for product in config["products"])
        - config["total_exposure_seconds"]
    ) < 0.001
    assert abs(config["cutout"]["radius_deg"] * 3600 - 15.0) < 1e-6
    assert config["authorization"]["download_and_uniform_coadd"]
    assert not config["authorization"]["gravity_response_fit"]
