#!/usr/bin/env python3
"""Construct target-blind regional and common-grid gas-state posteriors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from voidscreen.sigma_gas_posterior import (
    cluster_sobol_uniforms,
    common_grid_axis,
    log_uniform_depth_factors,
    map_region_values,
    positive_profile_draws,
    quantile_summary,
    resample_bin_labels_to_physical_grid,
    smooth_masked_field,
)
from voidscreen.sigma_gas_thermodynamics import (
    KPC_CM,
    uniform_slab_thermodynamics,
)

DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19x4_gas_state_math_preflight.json"
DEFAULT_X3_CONFIG = ROOT / "configs" / "sigma_v19x3_full_regional_spectral_production.json"
DEFAULT_X3_REPORT = ROOT / "results" / "sigma_v19x3_full_regional_spectral_production" / "report.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19x4_gas_state_posterior"
AUTHORIZED_X3_STATUS = (
    "all_494_regional_spectra_and_finite_temperature_fits_passed_source_map_authorized"
)
ARCSEC_PER_RADIAN = 206264.80624709636
STATE_KEYS = (
    "emission_measure_cm3",
    "emission_measure_per_area_cm5",
    "electron_density_cm3",
    "gas_surface_density_msun_kpc2",
    "gas_mass_msun",
    "thermal_pressure_erg_cm3",
    "entropy_proxy_keV_cm2",
    "sound_speed_km_s",
)
COMMON_MAP_KEYS = (
    "temperature_keV",
    "electron_density_cm3",
    "gas_surface_density_msun_kpc2",
    "thermal_pressure_erg_cm3",
    "entropy_proxy_keV_cm2",
    "sound_speed_km_s",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def product_path(report: dict[str, Any], cluster: str, role: str) -> Path:
    rows = next(row for row in report["clusters"] if row["cluster"] == cluster)
    matches = [row for row in rows["products"] if row["role"] == role]
    if len(matches) != 1:
        raise RuntimeError(f"V19X4 expected one {cluster} {role} product")
    item = matches[0]
    path = ROOT / item["relative_path"]
    if not path.is_file() or sha256(path) != item["sha256"]:
        raise RuntimeError(f"V19X4 changed {cluster} {role} product")
    return path


def validate_preconditions(
    config: dict[str, Any], x3_config_path: Path, x3_report_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    for key, value in config["parents"].items():
        if key.endswith("_sha256"):
            continue
        expected = config["parents"].get(f"{key}_sha256")
        if expected is not None:
            path = ROOT / value
            if not path.is_file() or sha256(path) != expected:
                raise RuntimeError(f"V19X4 parent changed: {value}")
    implementation = config.get("implementation", {})
    if implementation:
        runner = ROOT / implementation["runner"]
        module = ROOT / implementation["posterior_module"]
        if runner.resolve() != Path(__file__).resolve():
            raise RuntimeError("V19X4 configuration names another runner")
        if sha256(runner) != implementation["runner_sha256"]:
            raise RuntimeError("V19X4 runner changed after pre-registration")
        if sha256(module) != implementation["posterior_module_sha256"]:
            raise RuntimeError("V19X4 posterior module changed after pre-registration")

    if not x3_config_path.is_file() or not x3_report_path.is_file():
        raise RuntimeError("V19X4 requires terminal V19X3 config and report")
    x3_config = load_json(x3_config_path)
    x3_report = load_json(x3_report_path)
    if x3_report.get("status") != AUTHORIZED_X3_STATUS:
        raise RuntimeError("V19X3 did not authorize gas-source construction")
    if x3_report.get("source_map_construction_authorized") is not True:
        raise RuntimeError("V19X3 source-map authorization is false")
    if not x3_report.get("gates") or not all(x3_report["gates"].values()):
        raise RuntimeError("V19X3 contains a failed production gate")
    if x3_report.get("config_sha256") != sha256(x3_config_path):
        raise RuntimeError("V19X3 report names another frozen config")
    expected_x3_runner = config["parents"]["v19x3_runner_sha256"]
    if x3_report.get("runner_sha256") != expected_x3_runner:
        raise RuntimeError("V19X3 report names another runner")
    if x3_config.get("implementation", {}).get("runner_sha256") != expected_x3_runner:
        raise RuntimeError("V19X3 frozen config names another runner")
    if x3_report.get("lensing_or_halo_payload_opened") is not False:
        raise RuntimeError("V19X3 opened a prohibited target")

    v19m_report = load_json(ROOT / config["parents"]["v19m_region_report"])
    source_report = load_json(ROOT / config["parents"]["source_map_report"])
    return x3_report, v19m_report, source_report


def load_valid_region_geometry(path: Path) -> dict[int, dict[str, float]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    geometry: dict[int, dict[str, float]] = {}
    for row in rows:
        if row["valid"].lower() != "true":
            continue
        bin_id = int(row["bin_id"])
        if bin_id in geometry:
            raise RuntimeError(f"duplicate valid bin {bin_id}: {path}")
        pixels = float(row["pixels"])
        if not math.isfinite(pixels) or pixels <= 0.0:
            raise RuntimeError(f"invalid pixel count for bin {bin_id}: {path}")
        geometry[bin_id] = {"pixels": pixels}
    return geometry


def interval(fit: dict[str, Any], parameter: str) -> tuple[Any, Any]:
    if parameter == "temperature":
        record = fit.get("temperature_confidence_68_percent", {})
        return record.get("lower_keV"), record.get("upper_keV")
    if parameter == "normalization":
        record = fit.get("normalization_confidence_68_percent", {})
        return record.get("lower"), record.get("upper")
    raise ValueError(f"unknown interval parameter: {parameter}")


def atomic_npz(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def cluster_branch(
    config: dict[str, Any],
    cluster: str,
    region_rows: list[dict[str, Any]],
    geometry: dict[int, dict[str, float]],
    rank_correlation: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ordered = sorted(region_rows, key=lambda row: int(row["bin_id"]))
    bin_ids = np.asarray([int(row["bin_id"]) for row in ordered], dtype=np.int64)
    if set(bin_ids.tolist()) != set(geometry):
        raise RuntimeError(f"{cluster} V19X3 and V19M bin inventories differ")
    posterior = config["posterior"]
    draws = int(posterior["draws"])
    seed_offset = int(config["geometry"]["clusters"][cluster]["seed_offset"])
    u_temperature, u_normalization, u_depth = cluster_sobol_uniforms(
        len(ordered),
        draws,
        int(posterior["seed"]) + seed_offset,
        rank_correlation=rank_correlation,
    )
    depth_rule = posterior["line_of_sight_depth_factor"]
    depth_factor = log_uniform_depth_factors(
        u_depth, float(depth_rule["minimum"]), float(depth_rule["maximum"])
    )
    cluster_config = config["geometry"]["clusters"][cluster]
    pixel_kpc = (
        float(config["geometry"]["output_pixel_arcsec"])
        * float(cluster_config["kpc_per_arcsec"])
    )
    distance_cm = (
        float(cluster_config["kpc_per_arcsec"]) * ARCSEC_PER_RADIAN * KPC_CM
    )
    redshift = float(cluster_config["redshift"])
    temperature = np.empty((len(ordered), draws), dtype=float)
    normalization = np.empty_like(temperature)
    depth = np.empty_like(temperature)
    state = {key: np.empty_like(temperature) for key in STATE_KEYS}
    temperature_modes: list[str] = []
    normalization_modes: list[str] = []
    quality = np.empty(len(ordered), dtype=bool)

    for index, row in enumerate(ordered):
        fit = row["fit"]
        parameters = fit["parameters"]
        low_t, high_t = interval(fit, "temperature")
        low_n, high_n = interval(fit, "normalization")
        temperature[index], temperature_mode = positive_profile_draws(
            parameters["temperature_keV"],
            low_t,
            high_t,
            tuple(posterior["temperature_bounds_keV"]),
            u_temperature[index],
        )
        normalization[index], normalization_mode = positive_profile_draws(
            parameters["normalization"],
            low_n,
            high_n,
            tuple(posterior["normalization_bounds"]),
            u_normalization[index],
        )
        area = geometry[int(row["bin_id"])]["pixels"] * pixel_kpc**2
        reference_depth = math.sqrt(area)
        depth[index] = reference_depth * depth_factor
        gas = uniform_slab_thermodynamics(
            normalization[index],
            temperature[index],
            distance_cm,
            redshift,
            area,
            depth[index],
            electron_to_hydrogen_ratio=float(
                config["physical_constants_and_composition"][
                    "electron_to_hydrogen_ratio"
                ]
            ),
            mean_mass_per_electron_proton_masses=float(
                config["physical_constants_and_composition"][
                    "mean_mass_per_electron_proton_masses"
                ]
            ),
            mean_particle_mass_proton_masses=float(
                config["physical_constants_and_composition"][
                    "mean_particle_mass_proton_masses"
                ]
            ),
            adiabatic_index=float(
                config["physical_constants_and_composition"]["adiabatic_index"]
            ),
        )
        for key in STATE_KEYS:
            state[key][index] = gas[key]
        temperature_modes.append(temperature_mode)
        normalization_modes.append(normalization_mode)
        quality[index] = bool(fit.get("gates", {}).get("all_passed"))

    arrays: dict[str, Any] = {
        "bin_id": bin_ids,
        "quality_gate_passed": quality,
        "temperature_keV": temperature.astype(np.float32),
        "normalization": normalization.astype(np.float32),
        "line_of_sight_depth_kpc": depth.astype(np.float32),
        "shared_depth_factor": depth_factor.astype(np.float32),
        "rank_correlation": np.asarray(rank_correlation),
    }
    arrays.update(state)
    summary = {
        "cluster": cluster,
        "regions": len(ordered),
        "draws_per_region": draws,
        "rank_correlation": rank_correlation,
        "individual_quality_passes": int(np.count_nonzero(quality)),
        "temperature_sampling_modes": dict(Counter(temperature_modes)),
        "normalization_sampling_modes": dict(Counter(normalization_modes)),
        "all_draws_finite_positive": all(
            np.all(np.isfinite(value)) and np.all(value > 0.0)
            for key, value in arrays.items()
            if key
            not in {
                "bin_id",
                "quality_gate_passed",
                "rank_correlation",
            }
        ),
    }
    return arrays, summary


def build_common_maps(
    config: dict[str, Any],
    cluster: str,
    arrays: dict[str, Any],
    binmap: np.ndarray,
    center: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    common = config["common_grid"]
    axis = common_grid_axis(
        float(common["half_width_kpc"]), float(common["spacing_kpc"])
    )
    cluster_config = config["geometry"]["clusters"][cluster]
    native_pixel_kpc = (
        float(config["geometry"]["output_pixel_arcsec"])
        * float(cluster_config["kpc_per_arcsec"])
    )
    labels = resample_bin_labels_to_physical_grid(
        binmap,
        center_logical_x=float(center["logicalx"]),
        center_logical_y=float(center["logicaly"]),
        native_pixel_kpc=native_pixel_kpc,
        common_axis_kpc=axis,
    )
    bin_ids = np.asarray(arrays["bin_id"], dtype=np.int64)
    labels = np.where(np.isin(labels, bin_ids), labels, -1)
    output: dict[str, Any] = {
        "axis_east_kpc": axis,
        "axis_north_kpc": axis,
        "bin_id": labels,
    }
    mass_errors: dict[str, float] = {}
    spacing = float(common["spacing_kpc"])
    for key in COMMON_MAP_KEYS:
        summaries = quantile_summary(arrays[key])
        for quantile, values in summaries.items():
            mapped = map_region_values(labels, bin_ids, values)
            output[f"{key}_{quantile}_unsmoothed"] = mapped.astype(np.float32)
            for fwhm in common["smoothing_fwhm_kpc"]:
                sigma_pixels = float(fwhm) / (2.0 * math.sqrt(2.0 * math.log(2.0))) / spacing
                conserve = key == "gas_surface_density_msun_kpc2"
                smoothed = smooth_masked_field(
                    mapped,
                    sigma_pixels=sigma_pixels,
                    conserve_integral=conserve,
                )
                token = f"{float(fwhm):g}kpc"
                output[f"{key}_{quantile}_{token}"] = smoothed.astype(np.float32)
                if conserve:
                    valid = np.isfinite(mapped)
                    relative = abs(
                        float(np.sum(smoothed[valid]) - np.sum(mapped[valid]))
                    ) / float(np.sum(mapped[valid]))
                    mass_errors[f"{quantile}_{token}"] = relative
    summary = {
        "grid_cells_per_axis": len(axis),
        "finite_label_cells": int(np.count_nonzero(labels >= 0)),
        "represented_region_ids": len(set(labels[labels >= 0].tolist())),
        "surface_density_smoothing_mass_relative_errors": mass_errors,
        "maximum_surface_density_smoothing_mass_relative_error": max(
            mass_errors.values(), default=math.inf
        ),
    }
    return output, summary


def execute(
    config: dict[str, Any],
    x3_config_path: Path,
    x3_report_path: Path,
    output: Path,
) -> dict[str, Any]:
    x3_report, v19m_report, source_report = validate_preconditions(
        config, x3_config_path, x3_report_path
    )
    x3_by_cluster: dict[str, list[dict[str, Any]]] = {
        cluster: [] for cluster in config["geometry"]["clusters"]
    }
    for row in x3_report["regions"]:
        x3_by_cluster[row["cluster"]].append(row)
    source_by_cluster = {row["cluster"]: row for row in source_report["clusters"]}
    correlations = [float(value) for value in config["posterior"]["rank_correlations"]]
    products: list[dict[str, Any]] = []
    branch_summaries: list[dict[str, Any]] = []

    for cluster, cluster_config in config["geometry"]["clusters"].items():
        geometry = load_valid_region_geometry(ROOT / cluster_config["region_statistics"])
        binmap_path = product_path(v19m_report, cluster, "binmap")
        with fits.open(binmap_path, memmap=False) as handle:
            binmap = np.asarray(handle[0].data, dtype=np.int64)
        for correlation in correlations:
            token = f"rho_{correlation:+.1f}".replace("+", "p").replace("-", "m").replace(".", "p")
            arrays, regional_summary = cluster_branch(
                config,
                cluster,
                x3_by_cluster[cluster],
                geometry,
                correlation,
            )
            regional_path = output / cluster / f"regional_posterior_{token}.npz"
            atomic_npz(regional_path, arrays)
            maps, map_summary = build_common_maps(
                config,
                cluster,
                arrays,
                binmap,
                source_by_cluster[cluster]["final_center"],
            )
            map_path = output / cluster / f"common_grid_summary_{token}.npz"
            atomic_npz(map_path, maps)
            for role, path in (
                ("regional_posterior", regional_path),
                ("common_grid_summary", map_path),
            ):
                products.append(
                    {
                        "cluster": cluster,
                        "rank_correlation": correlation,
                        "role": role,
                        "relative_path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
            branch_summaries.append(
                {**regional_summary, "common_grid": map_summary}
            )

    minimum_quality = int(config["future_runtime_gates"]["minimum_individual_quality_passes_per_cluster"])
    expected_draws = int(config["posterior"]["draws"])
    gates = {
        "three_registered_dependence_branches_per_cluster": len(branch_summaries)
        == 3 * len(config["geometry"]["clusters"]),
        "all_494_regions_reconstructed_in_every_branch": all(
            row["regions"]
            == int(config["geometry"]["clusters"][row["cluster"]]["expected_valid_regions"])
            for row in branch_summaries
        ),
        "exact_4096_draws_per_region": all(
            row["draws_per_region"] == expected_draws == 4096
            for row in branch_summaries
        ),
        "minimum_quality_passes_per_cluster": all(
            row["individual_quality_passes"] >= minimum_quality
            for row in branch_summaries
        ),
        "every_stored_draw_finite_positive": all(
            row["all_draws_finite_positive"] for row in branch_summaries
        ),
        "common_grid_represents_every_region": all(
            row["common_grid"]["represented_region_ids"] == row["regions"]
            for row in branch_summaries
        ),
        "surface_density_smoothing_mass_conserved_to_1e_6": all(
            row["common_grid"][
                "maximum_surface_density_smoothing_mass_relative_error"
            ]
            <= 1.0e-6
            for row in branch_summaries
        ),
        "all_products_hash_bound": len(products)
        == 2 * len(branch_summaries),
    }
    return {
        "status": (
            "gas_state_posterior_and_common_grids_passed_source_invariant_scoring_authorized"
            if all(gates.values())
            else "gas_state_posterior_or_common_grid_gate_failed"
        ),
        "x3_config_sha256": sha256(x3_config_path),
        "x3_report_sha256": sha256(x3_report_path),
        "branch_summaries": branch_summaries,
        "products": products,
        "gates": gates,
        "source_invariant_scoring_authorized": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--x3-config", type=Path, default=DEFAULT_X3_CONFIG)
    parser.add_argument("--x3-report", type=Path, default=DEFAULT_X3_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        result = execute(
            config,
            args.x3_config.resolve(),
            args.x3_report.resolve(),
            output,
        )
    except Exception as exc:  # noqa: BLE001 - retain a terminal admission failure
        result = {
            "status": "gas_state_posterior_execution_failed",
            "execution_exception": f"{type(exc).__name__}: {exc}",
            "gates": {"execution_completed": False},
            "source_invariant_scoring_authorized": False,
        }
    report = {
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        **result,
        "lensing_or_halo_payload_opened": False,
        "source_invariant_or_action_selected": False,
        "gravity_formula_or_parameter_changed": False,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report_path)
    print(f"status: {report['status']}")
    if not report["source_invariant_scoring_authorized"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
