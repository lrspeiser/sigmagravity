#!/usr/bin/env python3
"""Run the frozen V19AJ compact-core foreground-star astrometry audit."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "scripts" / "run_sigma_v19ai_fors1_subpixel_astrometry.py"
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19aj_fors1_core_astrometry.json"
SPEC = importlib.util.spec_from_file_location("sigma_v19ai_frozen_base", BASE_SCRIPT)
BASE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BASE)

sha256 = BASE.sha256
load_json = BASE.load_json
refine_centroid = BASE.refine_centroid
fit_and_loo = BASE.fit_and_loo
json_wcs = BASE.json_wcs
center_separations = BASE.center_separations


def validate_config(config_path: Path, config: dict[str, Any]) -> dict[str, str]:
    if config["status"] != "frozen_before_any_v19aj_foreground_star_cutout_or_centroid":
        raise RuntimeError("V19AJ protocol is not frozen")
    runner = ROOT / config["implementation"]["runner"]
    if sha256(runner) != config["implementation"]["runner_sha256"]:
        raise RuntimeError("frozen V19AJ runner hash mismatch")
    if sha256(BASE_SCRIPT) != config["implementation"]["frozen_base_runner_sha256"]:
        raise RuntimeError("frozen V19AI base runner hash mismatch")
    hashes = {
        "config": sha256(config_path),
        "runner": sha256(runner),
        "frozen_base_runner": sha256(BASE_SCRIPT),
    }
    for artifact in config["parent_artifacts"]:
        path = ROOT / artifact["path"]
        actual = sha256(path)
        if actual != artifact["sha256"]:
            raise RuntimeError(f"V19AJ parent hash mismatch: {artifact['path']}")
        hashes[artifact["path"]] = actual
    if len(config["science_products"]) != int(config["gates"]["exact_filter_count"]):
        raise RuntimeError("V19AJ filter count changed")
    for product in config["science_products"]:
        path = ROOT / product["path"]
        actual = sha256(path)
        if actual != product["sha256"]:
            raise RuntimeError(f"V19AJ science hash mismatch: {product['filter']}")
        hashes[product["path"]] = actual
    prohibited = [
        "detect_or_rematch_sources",
        "inspect_member_or_candidate_coordinates_or_cutouts",
        "fit_photometry_or_deblending",
        "infer_stellar_mass_or_current",
        "read_lensing_or_halo_payload",
        "change_gravity_physics_or_parameters",
        "open_holdout",
    ]
    if any(config["authorization"][name] for name in prohibited):
        raise RuntimeError("V19AJ authorizes a prohibited action")
    return hashes


def run(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    original_validator = BASE.validate_config
    BASE.validate_config = validate_config
    try:
        return BASE.run(config_path)
    finally:
        BASE.validate_config = original_validator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    report = run(args.config)
    print(
        json.dumps(
            {
                "status": report["status"],
                "filters": report["filters"],
                "global_gates": report["global_gates"],
                "failures": report["failures"],
            },
            indent=2,
        )
    )
    return 0 if report["all_subpixel_astrometry_gates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
