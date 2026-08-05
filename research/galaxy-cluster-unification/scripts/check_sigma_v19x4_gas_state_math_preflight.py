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
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from voidscreen.sigma_gas_posterior import (
    common_grid_axis,
    resample_bin_labels_to_physical_grid,
)
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
        "accepted_bin_ids": [int(row["bin_id"]) for row in rows],
        "accepted_regions": len(rows),
        "pixel_count_minimum": int(np.min(pixels)),
        "pixel_count_median": float(np.median(pixels)),
        "pixel_count_maximum": int(np.max(pixels)),
        "projected_area_kpc2_minimum": float(np.min(areas)),
        "projected_area_kpc2_median": float(np.median(areas)),
        "projected_area_kpc2_maximum": float(np.max(areas)),
        "reference_depth_kpc_median": float(np.sqrt(np.median(areas))),
    }


def frozen_product(
    report: dict[str, Any], cluster: str, role: str
) -> tuple[Path, bool]:
    cluster_row = next(row for row in report["clusters"] if row["cluster"] == cluster)
    matches = [row for row in cluster_row["products"] if row["role"] == role]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {cluster} {role} product")
    item = matches[0]
    path = ROOT / item["relative_path"]
    return path, path.is_file() and sha256(path) == item["sha256"]


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

    implementation_checks: dict[str, bool] = {}
    implementation = config["implementation"]
    for key in ("runner", "posterior_module"):
        path = ROOT / implementation[key]
        implementation_checks[key] = (
            path.is_file() and sha256(path) == implementation[f"{key}_sha256"]
        )

    v19m_report = load_json(ROOT / parents["v19m_region_report"])
    source_report = load_json(ROOT / parents["source_map_report"])
    source_by_cluster = {row["cluster"]: row for row in source_report["clusters"]}
    common_axis = common_grid_axis(
        config["common_grid"]["half_width_kpc"],
        config["common_grid"]["spacing_kpc"],
    )
    clusters: dict[str, Any] = {}
    for name, cluster in config["geometry"]["clusters"].items():
        pixel_kpc = (
            float(config["geometry"]["output_pixel_arcsec"])
            * float(cluster["kpc_per_arcsec"])
        )
        inventory = region_inventory(ROOT / cluster["region_statistics"], pixel_kpc)
        binmap_path, binmap_hash_matches = frozen_product(v19m_report, name, "binmap")
        with fits.open(binmap_path, memmap=False) as handle:
            binmap = np.asarray(handle[0].data, dtype=np.int64)
        labels = resample_bin_labels_to_physical_grid(
            binmap,
            center_logical_x=source_by_cluster[name]["final_center"]["logicalx"],
            center_logical_y=source_by_cluster[name]["final_center"]["logicaly"],
            native_pixel_kpc=pixel_kpc,
            common_axis_kpc=common_axis,
        )
        accepted_ids = set(inventory["accepted_bin_ids"])
        admitted_labels = np.where(np.isin(labels, list(accepted_ids)), labels, -1)
        clusters[name] = {
            "inventory": inventory,
            "common_grid_admission": {
                "binmap_hash_matches": binmap_hash_matches,
                "represented_region_ids": len(
                    set(admitted_labels[admitted_labels >= 0].tolist())
                ),
                "finite_label_cells": int(np.count_nonzero(admitted_labels >= 0)),
            },
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
        "posterior_runner_and_module_hashes_match": all(
            implementation_checks.values()
        ),
        "registered_region_inventory_is_366_plus_128": all(
            clusters[name]["inventory"]["accepted_regions"]
            == int(cluster["expected_valid_regions"])
            for name, cluster in config["geometry"]["clusters"].items()
        ),
        "every_accepted_region_is_represented_on_common_grid": all(
            row["common_grid_admission"]["binmap_hash_matches"]
            and row["common_grid_admission"]["represented_region_ids"]
            == row["inventory"]["accepted_regions"]
            for row in clusters.values()
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
        "common_grid_is_241_cells_and_two_frozen_resolutions": (
            len(common_axis) == int(config["common_grid"]["cells_per_axis"]) == 241
            and config["common_grid"]["smoothing_fwhm_kpc"] == [50.0, 100.0]
        ),
        "three_dependence_branches_are_frozen": config["posterior"][
            "rank_correlations"
        ]
        == [-0.9, 0.0, 0.9],
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
        "implementation_hash_checks": implementation_checks,
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
