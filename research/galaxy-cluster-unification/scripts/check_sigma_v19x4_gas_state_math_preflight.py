#!/usr/bin/env python3
"""Audit the frozen V19X4 gas-state algebra before regional results exist."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from voidscreen.sigma_gas_thermodynamics import (
    KPC_CM,
    PROTON_MASS_G,
    compression_mach_number,
    json_scalars,
    temperature_jump_from_mach,
    uniform_slab_thermodynamics,
)

DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19x4_gas_state_math_preflight.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19x4_gas_state_math_preflight" / "report.json"
ARCSEC_PER_RADIAN = 206264.80624709636


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_positive(value: Any) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > 0.0
    except (TypeError, ValueError):
        return False


def region_inventory(path: Path, pixel_kpc: float) -> dict[str, Any]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["valid"].lower() == "true"]
    pixels = np.asarray([float(row["pixels"]) for row in rows], dtype=float)
    if not len(pixels) or not np.all(np.isfinite(pixels)) or np.any(pixels <= 0.0):
        raise RuntimeError(f"invalid accepted-region pixels: {path}")
    areas = pixels * pixel_kpc**2
    return {
        "accepted_regions": len(rows),
        "pixel_count_minimum": int(np.min(pixels)),
        "pixel_count_median": float(np.median(pixels)),
        "pixel_count_maximum": int(np.max(pixels)),
        "projected_area_kpc2_minimum": float(np.min(areas)),
        "projected_area_kpc2_median": float(np.median(areas)),
        "projected_area_kpc2_maximum": float(np.max(areas)),
        "reference_depth_kpc_median": float(np.sqrt(np.median(areas))),
    }


def representative_state(cluster: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    redshift = float(cluster["redshift"])
    kpc_per_arcsec = float(cluster["kpc_per_arcsec"])
    distance_cm = kpc_per_arcsec * ARCSEC_PER_RADIAN * KPC_CM
    area = float(inventory["projected_area_kpc2_median"])
    depth = math.sqrt(area)
    normalization = 1.0e-3
    temperature = 8.0
    state = uniform_slab_thermodynamics(
        normalization,
        temperature,
        distance_cm,
        redshift,
        area,
        depth,
    )
    emission_measure_per_area = float(state["emission_measure_per_area_cm5"])
    historical_surface_density = (
        1.17
        * PROTON_MASS_G
        * math.sqrt(emission_measure_per_area * depth * KPC_CM / 1.2)
        * KPC_CM**2
        / 1.988409870698051e33
    )
    scalars = json_scalars(state)
    return {
        "synthetic_input_only": {
            "normalization": normalization,
            "temperature_keV": temperature,
            "angular_diameter_distance_cm": distance_cm,
            "projected_area_kpc2": area,
            "reference_depth_kpc": depth,
        },
        "corrected_state": scalars,
        "historical_surface_density_msun_kpc2": historical_surface_density,
        "corrected_to_historical_surface_density_ratio": (
            float(state["gas_surface_density_msun_kpc2"])
            / historical_surface_density
        ),
    }


def execute(config: dict[str, Any]) -> dict[str, Any]:
    parent_checks: dict[str, bool] = {}
    parents = config["parents"]
    for key, value in parents.items():
        if not key.endswith("_sha256"):
            continue
        path_key = key.removesuffix("_sha256")
        path = ROOT / parents[path_key]
        parent_checks[path_key] = path.is_file() and sha256(path) == value

    clusters: dict[str, Any] = {}
    for name, cluster in config["geometry"]["clusters"].items():
        pixel_kpc = (
            float(config["geometry"]["output_pixel_arcsec"])
            * float(cluster["kpc_per_arcsec"])
        )
        inventory = region_inventory(ROOT / cluster["region_statistics"], pixel_kpc)
        clusters[name] = {
            "inventory": inventory,
            "representative_algebra_check": representative_state(cluster, inventory),
        }

    ratios = [
        row["representative_algebra_check"][
            "corrected_to_historical_surface_density_ratio"
        ]
        for row in clusters.values()
    ]
    mach = float(compression_mach_number(3.0))
    temperature_jump = float(temperature_jump_from_mach(mach))
    gates = {
        "all_parent_hashes_match": bool(parent_checks) and all(parent_checks.values()),
        "registered_region_inventory_is_366_plus_128": all(
            clusters[name]["inventory"]["accepted_regions"]
            == int(cluster["expected_valid_regions"])
            for name, cluster in config["geometry"]["clusters"].items()
        ),
        "corrected_to_historical_ratio_is_exactly_1_2": all(
            math.isclose(value, 1.2, rel_tol=1e-12, abs_tol=0.0)
            for value in ratios
        ),
        "representative_gas_states_are_finite_positive": all(
            all(
                finite_positive(value)
                for value in row["representative_algebra_check"][
                    "corrected_state"
                ].values()
            )
            for row in clusters.values()
        ),
        "rankine_hugoniot_compression_three_returns_mach_three": math.isclose(
            mach, 3.0, rel_tol=1e-12, abs_tol=0.0
        ),
        "shock_temperature_jump_is_finite_positive": finite_positive(
            temperature_jump
        ),
        "lensing_halo_and_gravity_selection_remain_sealed": not any(
            config["authorization"][key]
            for key in (
                "open_lensing_or_halo_payload",
                "select_source_invariant_or_action",
                "fit_gravity_parameter",
            )
        ),
    }
    return {
        "status": (
            "gas_state_math_preflight_passed_awaiting_v19x3_measurements"
            if all(gates.values())
            else "gas_state_math_preflight_failed"
        ),
        "config_sha256": sha256(DEFAULT_CONFIG),
        "parent_hash_checks": parent_checks,
        "historical_error": {
            "surface_density_understatement_fraction": 1.0 - 1.0 / 1.2,
            "corrected_surface_density_increase_fraction": 1.2 - 1.0,
            "historical_parent_was_not_mutated": True,
        },
        "clusters": clusters,
        "shock_identity_check": {
            "density_compression": 3.0,
            "mach_number": mach,
            "temperature_jump": temperature_jump,
        },
        "gates": gates,
        "observed_regional_spectra_opened": False,
        "source_invariant_selected": False,
        "gravity_theory_tested": False,
    }


def main() -> None:
    global DEFAULT_CONFIG
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = load_json(args.config.resolve())
    DEFAULT_CONFIG = args.config.resolve()
    report = execute(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())
    print(report["status"])


if __name__ == "__main__":
    main()
