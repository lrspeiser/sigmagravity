#!/usr/bin/env python3
"""Fit frozen Mn K-alpha calibration candidates without cluster-event access."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shlex
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from scipy.optimize import minimize
from scipy.special import voigt_profile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19cy_a2319_calibration_line_shape_gate.json"
BLOCK_BYTES = 4 * 1024 * 1024
FWHM_TO_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    if len(drive) != 1 or not drive.isalpha():
        raise RuntimeError(f"unsupported Windows path for WSL: {resolved}")
    return f"/mnt/{drive}{resolved.as_posix().split(':', 1)[1]}"


def validate_inputs(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_json(config_path)
    if config.get("protocol_version") != "SIGMA-V19CY-A2319-CALIBRATION-LINE-SHAPE-1.0.0":
        raise RuntimeError("unexpected calibration line-shape protocol")
    if config.get("status") != (
        "frozen after all 21 calibration applications passed, but before any candidate energy distribution, centroid, or width was inspected"
    ):
        raise RuntimeError("calibration line-shape protocol is not frozen")
    parent = ROOT / config["parents"]["application_report"]
    if not parent.is_file() or sha256(parent) != config["parents"]["application_report_sha256"]:
        raise RuntimeError("frozen calibration-application parent changed")
    application = load_json(parent)
    if not application.get("terminal_gate_passed"):
        raise RuntimeError("calibration-application terminal gate did not pass")
    authorization = config["authorization"]
    for key in (
        "read_cluster_sky_event_rows",
        "apply_selected_candidate_to_cluster_events",
        "fit_cluster_velocity",
        "access_validation_or_holdout_assets",
        "open_lensing_halo_or_gravity_targets",
        "change_gravity_formula_or_parameters",
        "derive_or_select_action",
    ):
        if authorization[key]:
            raise RuntimeError(f"sealed calibration line-shape boundary is open: {key}")
    return config, application


def copy_line_template(config: dict[str, Any], destination: Path) -> dict[str, Any]:
    distribution = "Ubuntu-24.04"
    source = config["line_template"]["wsl_path"]
    command = f"cp -- {shlex.quote(source)} {shlex.quote(to_wsl_path(destination))}"
    process = subprocess.run(
        ["wsl.exe", "-d", distribution, "--", "bash", "-lc", command],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if process.returncode != 0 or not destination.is_file():
        raise RuntimeError(f"failed to copy frozen MnKa template: {process.stderr}")
    if sha256(destination) != config["line_template"]["sha256"]:
        raise RuntimeError("MnKa line-template hash changed")
    return {
        "command": command,
        "exit_code": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    }


def read_template(path: Path, extension: str) -> dict[str, np.ndarray]:
    with fits.open(path, memmap=True, mode="readonly") as hdus:
        data = hdus[extension].data
        output = {
            "energy": np.asarray(data["ENERGY"], dtype=float).copy(),
            "width": np.asarray(data["WIDTH"], dtype=float).copy(),
            "area": np.asarray(data["AREA"], dtype=float).copy(),
        }
    output["area"] /= np.sum(output["area"])
    return output


def expected_counts(
    centers: np.ndarray,
    bin_width: float,
    template: dict[str, np.ndarray],
    parameters: np.ndarray,
) -> np.ndarray:
    shift, fwhm, normalization, background = parameters
    sigma = fwhm / FWHM_TO_SIGMA
    profile = np.zeros_like(centers)
    for energy, width, area in zip(
        template["energy"], template["width"], template["area"], strict=True
    ):
        profile += area * voigt_profile(centers - (energy + shift), sigma, width / 2.0)
    return np.maximum((normalization * profile + background) * bin_width, 1e-12)


def cash_statistic(observed: np.ndarray, expected: np.ndarray) -> float:
    return float(2.0 * np.sum(expected - observed * np.log(expected)))


def fit_histogram(
    observed: np.ndarray,
    centers: np.ndarray,
    bin_width: float,
    template: dict[str, np.ndarray],
    bounds: dict[str, list[float]],
) -> dict[str, Any]:
    total = float(np.sum(observed))
    background = float(np.median(np.r_[observed[:10], observed[-10:]]) / bin_width)
    initial = np.asarray([0.0, 4.5, max(total * 0.9, 1.0), max(background, 0.0)])
    ordered_bounds = [
        bounds["common_centroid_shift_ev"],
        bounds["instrument_gaussian_fwhm_ev"],
        bounds["line_normalization"],
        bounds["constant_background_per_ev"],
    ]

    def objective(parameters: np.ndarray) -> float:
        return cash_statistic(
            observed,
            expected_counts(centers, bin_width, template, parameters),
        )

    result = minimize(objective, initial, method="L-BFGS-B", bounds=ordered_bounds)
    parameters = np.asarray(result.x, dtype=float)
    return {
        "converged": bool(result.success and np.isfinite(parameters).all()),
        "optimizer_message": str(result.message),
        "iterations": int(result.nit),
        "cash_statistic": float(result.fun),
        "centroid_shift_ev": float(parameters[0]),
        "instrument_fwhm_ev": float(parameters[1]),
        "line_normalization": float(parameters[2]),
        "constant_background_per_ev": float(parameters[3]),
        "events_in_fit_window": int(np.sum(observed)),
    }


def read_candidate_energies(
    application_root: Path,
    application: dict[str, Any],
    obsid: str,
    candidate: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    arrays: list[np.ndarray] = []
    sources: list[dict[str, Any]] = []
    branches = sorted(
        {
            item["branch"]
            for item in application["applications"]
            if item["obsid"] == obsid and item["candidate"] == candidate
        }
    )
    for branch in branches:
        path = application_root / branch / f"{candidate}.evt"
        expected = next(
            item["output"]
            for item in application["applications"]
            if item["branch"] == branch and item["candidate"] == candidate
        )
        if not path.is_file() or path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise RuntimeError(f"candidate calibration output changed: {path}")
        with fits.open(path, memmap=True, mode="readonly") as hdus:
            arrays.append(np.asarray(hdus["EVENTS"].data["EPI2"], dtype=float).copy())
        sources.append(
            {"branch": branch, "path": str(path.relative_to(ROOT)), **expected}
        )
    return np.concatenate(arrays), sources


def select_candidate(
    fits_by_candidate: dict[str, dict[str, dict[str, Any]]], config: dict[str, Any]
) -> dict[str, Any]:
    targets = config["published_targets"]
    maximum_z = config["candidate_selection"]["maximum_absolute_statistical_z_per_observable"]
    complexity = config["candidate_selection"]["complexity_order"]
    summaries = []
    for candidate in complexity:
        score = 0.0
        maximum = 0.0
        observables = []
        for obsid in ("000101000", "000102000", "000103000"):
            fit = fits_by_candidate[candidate][obsid]
            target = targets[obsid]
            shift_z = (fit["centroid_shift_ev"] - target["centroid_shift_ev"]) / target[
                "centroid_stat_error_ev"
            ]
            fwhm_z = (fit["instrument_fwhm_ev"] - target["fwhm_ev"]) / target[
                "fwhm_stat_error_ev"
            ]
            score += shift_z**2 + fwhm_z**2
            maximum = max(maximum, abs(shift_z), abs(fwhm_z))
            observables.append({"obsid": obsid, "centroid_z": shift_z, "fwhm_z": fwhm_z})
        summaries.append(
            {
                "candidate": candidate,
                "score": float(score),
                "maximum_absolute_z": float(maximum),
                "passed": maximum <= maximum_z,
                "observables": observables,
            }
        )
    passing = [item for item in summaries if item["passed"]]
    if not passing:
        return {"summaries": summaries, "selected": None, "passed": False}
    passing.sort(key=lambda item: (item["score"], complexity.index(item["candidate"])))
    best = passing[0]
    tied = [item for item in passing if item["score"] - best["score"] < 1.0]
    selected = min(tied, key=lambda item: complexity.index(item["candidate"]))
    return {"summaries": summaries, "selected": selected["candidate"], "passed": True}


def build_report(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config, application = validate_inputs(config_path)
    application_root = (ROOT / config["paths"]["application_scratch_root"]).resolve()
    scratch_root = (ROOT / config["paths"]["scratch_root"]).resolve()
    if not application_root.is_dir():
        raise RuntimeError("calibration-application scratch root is absent")
    if scratch_root.exists():
        raise RuntimeError(f"refusing to overwrite line-shape scratch root: {scratch_root}")
    scratch_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=scratch_root.name + ".installing.", dir=scratch_root.parent))
    template_path = staging / "xa_gen_linefit_20190101vx001.fits"
    template_copy = copy_line_template(config, template_path)
    template = read_template(template_path, config["line_template"]["extension"])
    if len(template["energy"]) != config["line_template"]["rows"]:
        raise RuntimeError("unexpected MnKa component count")

    fit_config = config["fit"]
    edges = np.arange(
        fit_config["energy_min_ev"],
        fit_config["energy_max_ev"] + fit_config["bin_width_ev"],
        fit_config["bin_width_ev"],
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    candidates = config["candidate_selection"]["complexity_order"]
    fits_by_candidate: dict[str, dict[str, dict[str, Any]]] = {}
    source_audits: list[dict[str, Any]] = []
    for candidate in candidates:
        fits_by_candidate[candidate] = {}
        for obsid in ("000101000", "000102000", "000103000"):
            energies, sources = read_candidate_energies(
                application_root, application, obsid, candidate
            )
            selected = energies[
                np.isfinite(energies)
                & (energies >= fit_config["energy_min_ev"])
                & (energies <= fit_config["energy_max_ev"])
            ]
            observed, _ = np.histogram(selected, bins=edges)
            fit = fit_histogram(
                observed,
                centers,
                fit_config["bin_width_ev"],
                template,
                fit_config["bounds"],
            )
            if fit_config["require_convergence"] and not fit["converged"]:
                raise RuntimeError(f"MnKa fit did not converge: {candidate} {obsid}")
            fits_by_candidate[candidate][obsid] = fit
            source_audits.extend(
                {"candidate": candidate, "obsid": obsid, **source} for source in sources
            )
    selection = select_candidate(fits_by_candidate, config)
    os.replace(staging, scratch_root)
    report = {
        "protocol_version": config["protocol_version"],
        "status": "a2319_calibration_line_shape_candidates_fit",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "template_copy": template_copy,
        "template_components": {
            "energy": template["energy"].tolist(),
            "width": template["width"].tolist(),
            "area": template["area"].tolist(),
        },
        "fit_results": fits_by_candidate,
        "source_audits": source_audits,
        "selection": selection,
        "line_shape_gate_passed": selection["passed"],
        "selected_candidate": selection["selected"],
        "cluster_sky_event_accessed": False,
        "cluster_velocity_fit": False,
        "validation_or_holdout_accessed": False,
        "decision": (
            "authorize_selected_candidate_cluster_event_application_protocol_freeze"
            if selection["passed"]
            else "stop_before_cluster_event_application"
        ),
        "authorization": {
            "freeze_selected_candidate_cluster_event_application": selection["passed"],
            "read_cluster_sky_event_rows": False,
            "apply_selected_candidate_to_cluster_events": False,
            "fit_cluster_velocity": False,
            "access_validation_or_holdout_assets": False,
            "open_lensing_halo_or_gravity_targets": False,
            "derive_or_select_action": False,
        },
    }
    output = ROOT / config["paths"]["report"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = build_report()
    print(
        json.dumps(
            {
                "status": result["status"],
                "fit_results": result["fit_results"],
                "selection": result["selection"],
                "line_shape_gate_passed": result["line_shape_gate_passed"],
                "selected_candidate": result["selected_candidate"],
                "cluster_sky_event_accessed": result["cluster_sky_event_accessed"],
                "decision": result["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
