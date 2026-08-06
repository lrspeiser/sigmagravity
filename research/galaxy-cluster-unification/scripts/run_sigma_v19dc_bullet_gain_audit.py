#!/usr/bin/env python3
"""Measure the frozen Bullet blank-sky Ni/Au gain nuisance per ObsID."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from scipy.optimize import brentq, minimize
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19dc_bullet_gain_audit.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19dc_bullet_gain_audit"
AUTHORIZED_STATUS = "bullet_per_obsid_gain_audit_passed"
C_KM_S = 299792.458


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_parent(item: dict[str, Any]) -> Path:
    path = ROOT / item["path"]
    if not path.is_file() or sha256(path) != item["sha256"]:
        raise RuntimeError(f"V19DC frozen parent changed: {path}")
    return path


def validate_frozen(config: dict[str, Any]) -> dict[str, Path]:
    expected = "frozen_after_v19db_combination_pass_before_any_v19dc_gain_fit_or_source_redshift_fit"
    if config.get("freeze_state") != expected:
        raise RuntimeError("V19DC is not frozen before gain execution")
    implementation = config["implementation"]
    if implementation["runner"] != Path(__file__).resolve().relative_to(ROOT).as_posix():
        raise RuntimeError("V19DC config names another runner")
    if implementation["runner_sha256"] != sha256(Path(__file__).resolve()):
        raise RuntimeError("V19DC runner changed after freeze")
    parents = {
        key: validate_parent(item)
        for key, item in config["parents"].items()
        if isinstance(item, dict) and "path" in item
    }
    report = load_json(parents["v19db_report"])
    if report.get("status") != config["parents"]["v19db_report"]["required_status"]:
        raise RuntimeError("V19DB combination parent is not a terminal pass")
    auth = config["authorization"]
    if not (
        auth["open_original_bullet_background_pha_and_rmf_ebounds"]
        and not auth["open_bullet_source_pha"]
        and not auth["fit_temperature_abundance_redshift_or_velocity"]
        and not auth["open_obsid554_or_abell2146"]
        and not auth["open_lensing_halo_gravity_or_action"]
    ):
        raise RuntimeError("V19DC authorization boundary is open")
    return parents


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def external_product_root(raw_path: str) -> Path:
    """Resolve the frozen WSL archive without changing its scientific path."""
    if os.name == "nt" and raw_path.startswith("/"):
        return Path(r"\\wsl.localhost\Ubuntu-24.04") / raw_path.lstrip("/").replace("/", os.sep)
    return Path(raw_path)


def build_plan(config: dict[str, Any], index_path: Path) -> dict[int, list[dict[str, Any]]]:
    workload = config["workload"]
    obsids = {int(value) for value in workload["obsids"]}
    excluded = {int(value) for value in workload["excluded_obsids"]}
    selected = [
        row
        for row in read_csv(index_path)
        if row["cluster"] == workload["cluster"] and int(row["obsid"]) in obsids
    ]
    if len(selected) != int(workload["response_cells"]):
        raise RuntimeError("V19DC primary response-cell count changed")
    keys = [(row["cluster"], int(row["bin_id"]), int(row["obsid"]), int(row["ccd_id"])) for row in selected]
    if len(keys) != len(set(keys)) or any(int(row["obsid"]) in excluded for row in selected):
        raise RuntimeError("V19DC selected a duplicate or excluded cell")
    plan: dict[int, list[dict[str, Any]]] = {obsid: [] for obsid in sorted(obsids)}
    for row in selected:
        product_root = external_product_root(row["cell_directory"]) / "products"
        plan[int(row["obsid"])].append(
            {
                "cluster": row["cluster"],
                "bin_id": int(row["bin_id"]),
                "obsid": int(row["obsid"]),
                "ccd_id": int(row["ccd_id"]),
                "cell_name": row["cell_name"],
                "background": product_root / row["background_pha_name"],
                "background_bytes": int(row["background_pha_bytes"]),
                "background_sha256": row["background_pha_sha256"],
                "rmf": product_root / row["rmf_name"],
                "rmf_bytes": int(row["rmf_bytes"]),
                "rmf_sha256": row["rmf_sha256"],
            }
        )
    for cells in plan.values():
        cells.sort(key=lambda row: (row["bin_id"], row["ccd_id"], row["cell_name"]))
    return plan


def validate_file(path: Path, size: int, digest: str) -> None:
    if not path.is_file() or path.stat().st_size != size or sha256(path) != digest:
        raise RuntimeError(f"V19DC frozen input changed: {path}")


def load_obsid_background(obsid: int, cells: list[dict[str, Any]]) -> dict[str, Any]:
    counts: np.ndarray | None = None
    channel: np.ndarray | None = None
    energy_lo: np.ndarray | None = None
    energy_hi: np.ndarray | None = None
    inventory: list[dict[str, Any]] = []
    for cell in cells:
        validate_file(cell["background"], cell["background_bytes"], cell["background_sha256"])
        validate_file(cell["rmf"], cell["rmf_bytes"], cell["rmf_sha256"])
        with fits.open(cell["background"], memmap=False) as hdus:
            spectrum = hdus["SPECTRUM"].data
            current_channel = np.asarray(spectrum["CHANNEL"], dtype=np.int64)
            current_counts = np.asarray(spectrum["COUNTS"], dtype=np.int64)
        with fits.open(cell["rmf"], memmap=False) as hdus:
            ebounds = hdus["EBOUNDS"].data
            rmf_channel = np.asarray(ebounds["CHANNEL"], dtype=np.int64)
            current_lo = np.asarray(ebounds["E_MIN"], dtype=float)
            current_hi = np.asarray(ebounds["E_MAX"], dtype=float)
        if len(current_channel) != 1024 or not np.array_equal(current_channel, rmf_channel):
            raise RuntimeError(f"V19DC invalid PHA/RMF channel grid for {cell['cell_name']}")
        if counts is None:
            channel = current_channel
            energy_lo = current_lo
            energy_hi = current_hi
            counts = np.zeros_like(current_counts, dtype=np.int64)
        elif not (
            np.array_equal(channel, current_channel)
            and np.array_equal(energy_lo, current_lo)
            and np.array_equal(energy_hi, current_hi)
        ):
            raise RuntimeError(f"V19DC EBOUNDS changed within ObsID {obsid}")
        if np.any(current_counts < 0):
            raise RuntimeError(f"V19DC negative blank-sky count for {cell['cell_name']}")
        counts += current_counts
        inventory.append(
            {
                "obsid": obsid,
                "bin_id": cell["bin_id"],
                "ccd_id": cell["ccd_id"],
                "cell_name": cell["cell_name"],
                "background_path": str(cell["background"]),
                "background_bytes": cell["background_bytes"],
                "background_sha256": cell["background_sha256"],
                "rmf_path": str(cell["rmf"]),
                "rmf_bytes": cell["rmf_bytes"],
                "rmf_sha256": cell["rmf_sha256"],
            }
        )
    if counts is None or channel is None or energy_lo is None or energy_hi is None:
        raise RuntimeError(f"V19DC ObsID {obsid} has no cells")
    return {
        "obsid": obsid,
        "channel": channel,
        "energy_lo": energy_lo,
        "energy_hi": energy_hi,
        "counts": counts,
        "inventory": inventory,
    }


def cash_stat(counts: np.ndarray, model: np.ndarray) -> float:
    safe = np.maximum(model, np.finfo(float).tiny)
    return float(2.0 * np.sum(safe - counts * np.log(safe)))


def deviance(counts: np.ndarray, model: np.ndarray) -> float:
    safe = np.maximum(model, np.finfo(float).tiny)
    term = np.zeros_like(safe)
    positive = counts > 0
    term[positive] = counts[positive] * np.log(counts[positive] / safe[positive])
    return float(2.0 * np.sum(safe - counts + term))


def line_model(parameters: np.ndarray, lo: np.ndarray, hi: np.ndarray, reference: float) -> np.ndarray:
    centroid, log_sigma, log_amplitude, log_continuum, continuum_slope = parameters
    sigma = math.exp(float(log_sigma))
    amplitude = math.exp(float(log_amplitude))
    midpoint = 0.5 * (lo + hi)
    continuum = np.exp(log_continuum + continuum_slope * (midpoint - reference)) * (hi - lo)
    line = amplitude * (ndtr((hi - centroid) / sigma) - ndtr((lo - centroid) / sigma))
    return np.asarray(continuum + line, dtype=float)


def fit_continuum(counts: np.ndarray, lo: np.ndarray, hi: np.ndarray, reference: float) -> float:
    midpoint = 0.5 * (lo + hi)

    def objective(parameters: np.ndarray) -> float:
        model = np.exp(parameters[0] + parameters[1] * (midpoint - reference)) * (hi - lo)
        return cash_stat(counts, model)

    rate = max(float(np.sum(counts)) / max(float(np.sum(hi - lo)), 1e-12), 1e-3)
    result = minimize(
        objective,
        np.array([math.log(rate), 0.0]),
        method="L-BFGS-B",
        bounds=[(math.log(1e-3), math.log(max(10.0, 10.0 * rate))), (-5.0, 5.0)],
        options={"maxiter": 3000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not np.isfinite(result.fun):
        raise RuntimeError("V19DC continuum-only fit is non-finite")
    return float(result.fun)


def fit_line_once(
    counts_all: np.ndarray,
    energy_lo_all: np.ndarray,
    energy_hi_all: np.ndarray,
    reference: float,
    half_window: float,
    model_config: dict[str, Any],
    *,
    profile: bool,
) -> dict[str, Any]:
    mask = (energy_lo_all >= reference - half_window) & (energy_hi_all <= reference + half_window)
    counts = np.asarray(counts_all[mask], dtype=float)
    lo = np.asarray(energy_lo_all[mask], dtype=float)
    hi = np.asarray(energy_hi_all[mask], dtype=float)
    if len(counts) < 20 or np.sum(counts) <= 0:
        raise RuntimeError("V19DC line window has insufficient channels or counts")
    search = float(model_config["centroid_search_half_width_keV"])
    sigma_min, sigma_max = [float(value) for value in model_config["sigma_bounds_keV"]]
    slope_min, slope_max = [float(value) for value in model_config["continuum_log_slope_bounds_per_keV"]]
    rate = max(float(np.sum(counts)) / (2.0 * half_window), 1e-3)
    bounds = [
        (reference - search, reference + search),
        (math.log(sigma_min), math.log(sigma_max)),
        (math.log(1.0), math.log(max(10.0, 10.0 * float(np.sum(counts))))),
        (math.log(1e-3), math.log(max(10.0, 10.0 * rate))),
        (slope_min, slope_max),
    ]

    def objective(parameters: np.ndarray) -> float:
        return cash_stat(counts, line_model(parameters, lo, hi, reference))

    candidates = []
    for offset in model_config["multistart_centroid_offsets_keV"]:
        for sigma in model_config["multistart_sigma_keV"]:
            start = np.array(
                [
                    reference + float(offset),
                    math.log(float(sigma)),
                    math.log(max(10.0, 0.3 * float(np.sum(counts)))),
                    math.log(max(1.0, 0.5 * rate)),
                    0.0,
                ]
            )
            candidates.append(
                minimize(
                    objective,
                    start,
                    method="Nelder-Mead",
                    bounds=bounds,
                    options={"maxiter": 5000, "xatol": 1e-9, "fatol": 1e-6},
                )
            )
    seed = min(candidates, key=lambda result: float(result.fun))
    best = minimize(
        objective,
        seed.x,
        method="Powell",
        bounds=bounds,
        options={"maxiter": 5000, "xtol": 1e-10, "ftol": 1e-11},
    )
    if not np.isfinite(best.fun):
        raise RuntimeError("V19DC line fit is non-finite")
    model = line_model(best.x, lo, hi, reference)
    null_cash = fit_continuum(counts, lo, hi, reference)
    result: dict[str, Any] = {
        "reference_energy_keV": reference,
        "half_window_keV": half_window,
        "channels": len(counts),
        "counts": int(np.sum(counts)),
        "centroid_recorded_keV": float(best.x[0]),
        "sigma_keV": math.exp(float(best.x[1])),
        "line_counts": math.exp(float(best.x[2])),
        "continuum_rate_at_line_counts_per_keV": math.exp(float(best.x[3])),
        "continuum_log_slope_per_keV": float(best.x[4]),
        "cash": float(best.fun),
        "null_cash": null_cash,
        "line_delta_cash": null_cash - float(best.fun),
        "deviance": deviance(counts, model),
        "deviance_per_channel": deviance(counts, model) / len(counts),
        "optimizer_success": bool(best.success),
    }
    if not profile:
        return result

    nuisance_start = np.asarray(best.x[1:], dtype=float)
    nuisance_bounds = bounds[1:]
    cache: dict[float, float] = {}

    def profiled(centroid: float) -> float:
        key = round(float(centroid), 12)
        if key not in cache:
            def nuisance_objective(nuisance: np.ndarray) -> float:
                return objective(np.concatenate(([centroid], nuisance)))

            nuisance_fit = minimize(
                nuisance_objective,
                nuisance_start,
                method="L-BFGS-B",
                bounds=nuisance_bounds,
                options={"maxiter": 3000, "ftol": 1e-11, "gtol": 1e-7},
            )
            cache[key] = float(nuisance_fit.fun)
        return cache[key]

    target = float(best.fun) + float(model_config["profile_interval_delta_cash"])

    def find_crossing(direction: int) -> float:
        center = float(best.x[0])
        step = 0.002
        previous = center
        bound = bounds[0][0] if direction < 0 else bounds[0][1]
        for _ in range(100):
            current = center + direction * step
            current = max(bounds[0][0], min(bounds[0][1], current))
            if profiled(current) >= target:
                lower, upper = (current, previous) if direction < 0 else (previous, current)
                return float(
                    brentq(
                        lambda value: profiled(value) - target,
                        lower,
                        upper,
                        xtol=float(model_config["profile_crossing_tolerance_keV"]),
                    )
                )
            if current == bound:
                break
            previous = current
            step *= 1.25
        raise RuntimeError("V19DC line profile did not cross Delta-Cash=1 inside bounds")

    lower = find_crossing(-1)
    upper = find_crossing(1)
    result.update(
        {
            "profile_lower_keV": lower,
            "profile_upper_keV": upper,
            "profile_minus_keV": float(best.x[0]) - lower,
            "profile_plus_keV": upper - float(best.x[0]),
            "profile_sigma_conservative_keV": max(float(best.x[0]) - lower, upper - float(best.x[0])),
        }
    )
    return result


def gain_parameters(
    ni_centroid: float,
    au_centroid: float,
    ni_variance: float,
    au_variance: float,
    ni_reference: float,
    au_reference: float,
) -> tuple[np.ndarray, np.ndarray]:
    def transform(recorded: np.ndarray) -> np.ndarray:
        slope = (au_reference - ni_reference) / (recorded[1] - recorded[0])
        intercept = ni_reference - slope * recorded[0]
        return np.array([intercept, slope], dtype=float)

    recorded = np.array([ni_centroid, au_centroid], dtype=float)
    parameters = transform(recorded)
    jacobian = np.zeros((2, 2), dtype=float)
    step = 1e-6
    for column in range(2):
        offset = np.zeros(2)
        offset[column] = step
        jacobian[:, column] = (transform(recorded + offset) - transform(recorded - offset)) / (2.0 * step)
    covariance = jacobian @ np.diag([ni_variance, au_variance]) @ jacobian.T
    return parameters, covariance


def fit_obsid(payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    line_config = config["line_model"]
    lines = line_config["lines_keV"]
    primary_half = float(line_config["primary_half_window_keV"])
    primary = {
        name: fit_line_once(
            payload["counts"], payload["energy_lo"], payload["energy_hi"], float(reference),
            primary_half, line_config, profile=True,
        )
        for name, reference in lines.items()
    }
    robustness: dict[str, dict[str, Any]] = {}
    for half_window in line_config["robustness_half_windows_keV"]:
        key = f"half_window_{float(half_window):.2f}_keV"
        robustness[key] = {
            name: fit_line_once(
                payload["counts"], payload["energy_lo"], payload["energy_hi"], float(reference),
                float(half_window), line_config, profile=False,
            )
            for name, reference in lines.items()
        }
    ni = primary["Ni_Kalpha"]
    au = primary["Au_Lalpha"]
    parameters, covariance = gain_parameters(
        ni["centroid_recorded_keV"],
        au["centroid_recorded_keV"],
        ni["profile_sigma_conservative_keV"] ** 2,
        au["profile_sigma_conservative_keV"] ** 2,
        float(lines["Ni_Kalpha"]),
        float(lines["Au_Lalpha"]),
    )
    observed_fe = float(config["propagation"]["representative_fe_rest_energy_keV"]) / (
        1.0 + float(config["propagation"]["bullet_optical_redshift"])
    )
    vector = np.array([1.0, observed_fe])
    gain_energy_sigma = math.sqrt(max(0.0, float(vector @ covariance @ vector)))
    eigenvalues = np.linalg.eigvalsh(covariance)
    maximum_shift = max(
        abs(
            branch[name]["centroid_recorded_keV"] - primary[name]["centroid_recorded_keV"]
        )
        for branch in robustness.values()
        for name in primary
    )
    minimum_delta_cash = min(item["line_delta_cash"] for item in primary.values())
    gates = {
        "both_lines_detected": minimum_delta_cash >= float(line_config["minimum_line_delta_cash"]),
        "profiles_finite_and_interior": all(
            np.isfinite(item["profile_lower_keV"])
            and np.isfinite(item["profile_upper_keV"])
            and item["profile_lower_keV"] < item["centroid_recorded_keV"] < item["profile_upper_keV"]
            for item in primary.values()
        ),
        "window_robustness": maximum_shift <= float(line_config["maximum_window_robustness_centroid_shift_keV"]),
        "gain_covariance_finite_psd": bool(
            np.isfinite(covariance).all()
            and np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-14)
            and float(np.min(eigenvalues)) >= -1e-14
        ),
    }
    return {
        "obsid": int(payload["obsid"]),
        "cells": len(payload["inventory"]),
        "blank_sky_full_pha_counts": int(np.sum(payload["counts"])),
        "primary_lines": primary,
        "window_robustness": robustness,
        "maximum_window_centroid_shift_keV": maximum_shift,
        "minimum_line_delta_cash": minimum_delta_cash,
        "gain": {
            "relation": "E_cal_keV = intercept_keV + slope * E_recorded_keV",
            "intercept_keV": float(parameters[0]),
            "slope": float(parameters[1]),
            "covariance_intercept_slope": covariance.tolist(),
            "covariance_eigenvalues": eigenvalues.tolist(),
            "representative_observed_fe_energy_keV": observed_fe,
            "correction_at_representative_fe_keV": float(parameters[0] + (parameters[1] - 1.0) * observed_fe),
            "one_sigma_energy_uncertainty_at_representative_fe_keV": gain_energy_sigma,
            "one_sigma_equivalent_velocity_uncertainty_km_s": C_KM_S * gain_energy_sigma / observed_fe,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }


def write_union_spectra(payloads: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    path = output / "blank_sky_union_spectra.csv"
    fields = ["obsid", "channel", "energy_lo_keV", "energy_hi_keV", "counts"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for payload in sorted(payloads, key=lambda item: int(item["obsid"])):
            for channel, lo, hi, counts in zip(
                payload["channel"], payload["energy_lo"], payload["energy_hi"], payload["counts"], strict=True
            ):
                writer.writerow(
                    {
                        "obsid": payload["obsid"],
                        "channel": int(channel),
                        "energy_lo_keV": f"{float(lo):.12g}",
                        "energy_hi_keV": f"{float(hi):.12g}",
                        "counts": int(counts),
                    }
                )
    return {"path": path.relative_to(ROOT).as_posix(), "rows": 9 * 1024, "bytes": path.stat().st_size, "sha256": sha256(path)}


def write_inventory(payloads: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    path = output / "background_input_manifest.csv"
    fields = [
        "obsid", "bin_id", "ccd_id", "cell_name", "background_path", "background_bytes",
        "background_sha256", "rmf_path", "rmf_bytes", "rmf_sha256",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for payload in sorted(payloads, key=lambda item: int(item["obsid"])):
            writer.writerows(payload["inventory"])
    rows = sum(len(payload["inventory"]) for payload in payloads)
    return {"path": path.relative_to(ROOT).as_posix(), "rows": rows, "bytes": path.stat().st_size, "sha256": sha256(path)}


def execute(config: dict[str, Any], output: Path) -> dict[str, Any]:
    parents = validate_frozen(config)
    plan = build_plan(config, parents["unified_product_index"])
    payloads: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=int(config["runtime"]["parallel_obsids"])) as pool:
        futures = {pool.submit(load_obsid_background, obsid, cells): obsid for obsid, cells in plan.items()}
        for future in as_completed(futures):
            payloads.append(future.result())
    payloads.sort(key=lambda item: int(item["obsid"]))
    reference_lo = payloads[0]["energy_lo"]
    reference_hi = payloads[0]["energy_hi"]
    cross_obsid_grid_exact = all(
        np.array_equal(reference_lo, payload["energy_lo"]) and np.array_equal(reference_hi, payload["energy_hi"])
        for payload in payloads[1:]
    )
    if not cross_obsid_grid_exact:
        raise RuntimeError("V19DC EBOUNDS changed across primary ObsIDs")
    results = []
    with ThreadPoolExecutor(max_workers=int(config["runtime"]["parallel_obsids"])) as pool:
        futures = {pool.submit(fit_obsid, payload, config): int(payload["obsid"]) for payload in payloads}
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: int(item["obsid"]))
    inventory = write_inventory(payloads, output)
    spectra = write_union_spectra(payloads, output)
    cells = [item for payload in payloads for item in payload["inventory"]]
    gates = {
        "nine_primary_obsids_exact": len(results) == len(config["workload"]["obsids"]) == 9,
        "3483_background_cells_used_once": len(cells) == len({item["cell_name"] for item in cells}) == 3483,
        "channel_and_ebounds_grids_exact": cross_obsid_grid_exact,
        "both_lines_detected_in_every_obsid": all(item["gates"]["both_lines_detected"] for item in results),
        "all_profiles_finite_and_interior": all(item["gates"]["profiles_finite_and_interior"] for item in results),
        "all_window_robustness_gates_pass": all(item["gates"]["window_robustness"] for item in results),
        "all_gain_covariances_finite_psd": all(item["gates"]["gain_covariance_finite_psd"] for item in results),
    }
    return {
        "status": AUTHORIZED_STATUS if all(gates.values()) else "bullet_per_obsid_gain_audit_gate_failed",
        "background_input_manifest": inventory,
        "blank_sky_union_spectra": spectra,
        "obsids": results,
        "gates": gates,
        "bullet_source_redshift_fitting_authorized": all(gates.values()),
    }


def preflight(config: dict[str, Any]) -> dict[str, Any]:
    parents = validate_frozen(config)
    plan = build_plan(config, parents["unified_product_index"])
    return {
        "status": "v19dc_payload_blind_gain_plan_passed",
        "obsids": len(plan),
        "response_cells": sum(len(cells) for cells in plan.values()),
        "background_or_rmf_payload_opened": False,
        "source_pha_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    output = args.output.resolve()
    config = load_json(config_path)
    try:
        result = preflight(config) if args.preflight_only else execute(config, output)
    except Exception as exc:  # noqa: BLE001
        result = {
            "status": "v19dc_execution_failed_closed",
            "execution_exception": f"{type(exc).__name__}: {exc}",
            "gates": {"execution_completed": False},
            "bullet_source_redshift_fitting_authorized": False,
        }
    report = {
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        **result,
        "bullet_source_pha_opened": False,
        "temperature_abundance_redshift_or_velocity_fitted": False,
        "obsid554_or_abell2146_opened": False,
        "lensing_halo_gravity_or_action_opened": False,
    }
    path = output / ("preflight_report.json" if args.preflight_only else "report.json")
    atomic_json(path, report)
    print(json.dumps({key: report.get(key) for key in ("status", "execution_exception")}, indent=2, sort_keys=True))
    required = "v19dc_payload_blind_gain_plan_passed" if args.preflight_only else AUTHORIZED_STATUS
    if report["status"] != required:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
