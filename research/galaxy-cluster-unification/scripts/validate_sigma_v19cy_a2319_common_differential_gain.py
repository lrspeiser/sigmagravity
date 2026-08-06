#!/usr/bin/env python3
"""Validate common/differential A2319 gain reconstruction on held-out Fe-55."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shlex
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import apply_sigma_v19cy_a2319_calibration_candidates as application
import fit_sigma_v19cy_a2319_calibration_line_shape as line_shape

DEFAULT_CONFIG = (
    ROOT / "configs" / "sigma_v19cy_a2319_common_differential_gain_closure.json"
)
BLOCK_BYTES = 4 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_inputs(
    config_path: Path = DEFAULT_CONFIG,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = load_json(config_path)
    if config.get("protocol_version") != (
        "SIGMA-V19CY-A2319-COMMON-DIFFERENTIAL-GAIN-CLOSURE-1.0.0"
    ):
        raise RuntimeError("unexpected common/differential closure protocol")
    expected_status = (
        "frozen after the calibration-only failure diagnosis passed its continuous "
        "control and identified curvature in all seven straight-line branches, but "
        "before any held-out filter-wheel event energy was recalculated or inspected"
    )
    if config.get("status") != expected_status:
        raise RuntimeError("common/differential closure is not frozen")
    parents: dict[str, dict[str, Any]] = {}
    for name in ("diagnosis_report", "topology_report", "download_provenance"):
        path = ROOT / config["parents"][name]
        if not path.is_file() or sha256(path) != config["parents"][f"{name}_sha256"]:
            raise RuntimeError(f"common/differential parent changed: {path}")
        parents[name] = load_json(path)
    line_config_path = ROOT / config["parents"]["line_shape_config"]
    if (
        not line_config_path.is_file()
        or sha256(line_config_path)
        != config["parents"]["line_shape_config_sha256"]
    ):
        raise RuntimeError("frozen line-shape configuration changed")
    diagnosis = parents["diagnosis_report"]
    topology = parents["topology_report"]
    if not diagnosis.get("continuous_control_passed"):
        raise RuntimeError("continuous-control parent did not pass")
    if diagnosis.get("cluster_sky_event_accessed"):
        raise RuntimeError("diagnosis parent opened a cluster event")
    if not topology.get("topology_gate_passed"):
        raise RuntimeError("topology parent did not pass")
    segments = topology["segment_sets"].get("10800")
    if segments is None or len(segments) != config["cross_validation"]["folds"]:
        raise RuntimeError("frozen four-segment topology changed")
    authorization = config["authorization"]
    for key in (
        "read_cluster_sky_event_rows",
        "apply_gain_to_cluster_sky_events",
        "fit_cluster_velocity",
        "access_validation_or_holdout_assets",
        "open_lensing_halo_or_gravity_targets",
        "change_gravity_formula_or_parameters",
        "derive_or_select_action",
    ):
        if authorization[key]:
            raise RuntimeError(f"sealed common/differential boundary is open: {key}")
    return config, topology, parents["download_provenance"], load_json(line_config_path)


def verified_raw_path(
    raw_root: Path,
    relative: str,
    provenance_by_path: dict[str, dict[str, Any]],
) -> tuple[Path, dict[str, Any]]:
    path = (raw_root / relative).resolve()
    if not path.is_relative_to(raw_root):
        raise RuntimeError(f"common/differential path escapes raw root: {relative}")
    record = provenance_by_path.get(relative)
    if record is None:
        raise RuntimeError(f"source absent from provenance: {relative}")
    if not path.is_file() or path.stat().st_size != record["bytes"]:
        raise RuntimeError(f"source size changed: {relative}")
    if sha256(path) != record["sha256"]:
        raise RuntimeError(f"source hash changed: {relative}")
    return path, record


def read_gain_history(
    path: Path, extension: str, minimum_events: int
) -> tuple[list[fits.hdu.base.ExtensionHDU], np.ndarray, np.ndarray]:
    with gzip.open(path, "rb") as stream, fits.open(
        stream, memmap=False, mode="readonly"
    ) as hdus:
        copies = [hdu.copy() for hdu in hdus]
        data = np.asarray(hdus[extension].data).copy()
    finite = (
        np.isfinite(np.asarray(data["TIME"], dtype=float))
        & np.isfinite(np.asarray(data["PIXEL"], dtype=float))
        & np.isfinite(np.asarray(data["TEMP_FIT"], dtype=float))
        & np.isfinite(np.asarray(data["NEVENT"], dtype=float))
        & np.isfinite(np.asarray(data["CHISQ"], dtype=float))
    )
    valid = finite & (np.asarray(data["NEVENT"], dtype=int) >= minimum_events)
    return copies, data, data[valid].copy()


def common_temperature(rows: np.ndarray, times: np.ndarray) -> np.ndarray:
    order = np.argsort(np.asarray(rows["TIME"], dtype=float))
    source_times = np.asarray(rows["TIME"], dtype=float)[order]
    source_temps = np.asarray(rows["TEMP_FIT"], dtype=float)[order]
    if times.min() < source_times.min() or times.max() > source_times.max():
        raise RuntimeError("calibration-pixel history does not cover requested times")
    return np.interp(times, source_times, source_temps)


def fit_differential_models(
    held_out_obsid: str,
    fe55_valid: dict[str, np.ndarray],
    pxcal_valid: dict[str, np.ndarray],
) -> dict[int, dict[str, float | int]]:
    models: dict[int, dict[str, float | int]] = {}
    training_obsids = sorted(obsid for obsid in fe55_valid if obsid != held_out_obsid)
    for pixel in range(36):
        if pixel == 12:
            models[pixel] = {
                "rows": 0,
                "time_center": 0.0,
                "differential_at_center": 0.0,
                "slope_per_second": 0.0,
            }
            continue
        times_parts: list[np.ndarray] = []
        differential_parts: list[np.ndarray] = []
        for obsid in training_obsids:
            selected = fe55_valid[obsid][fe55_valid[obsid]["PIXEL"] == pixel]
            times = np.asarray(selected["TIME"], dtype=float)
            temperatures = np.asarray(selected["TEMP_FIT"], dtype=float)
            common = common_temperature(pxcal_valid[obsid], times)
            times_parts.append(times)
            differential_parts.append(temperatures - common)
        all_times = np.concatenate(times_parts)
        all_differential = np.concatenate(differential_parts)
        if len(all_times) < 2 or np.ptp(all_times) <= 0:
            raise RuntimeError(f"insufficient differential training rows for pixel {pixel}")
        center = float(np.mean(all_times))
        slope, intercept = np.polyfit(all_times - center, all_differential, 1)
        if not np.isfinite([slope, intercept]).all():
            raise RuntimeError(f"non-finite differential model for pixel {pixel}")
        models[pixel] = {
            "rows": len(all_times),
            "time_center": center,
            "differential_at_center": float(intercept),
            "slope_per_second": float(slope),
        }
    return models


def predict_differential(model: dict[str, float | int], times: np.ndarray) -> np.ndarray:
    return float(model["differential_at_center"]) + float(
        model["slope_per_second"]
    ) * (times - float(model["time_center"]))


def nearest_record(data: np.ndarray, pixel: int, time: float) -> np.void:
    indexes = np.flatnonzero(np.asarray(data["PIXEL"], dtype=int) == pixel)
    if not len(indexes):
        raise RuntimeError(f"gain template has no pixel {pixel}")
    times = np.asarray(data["TIME"][indexes], dtype=float)
    return data[indexes[int(np.argmin(np.abs(times - time)))]]


def build_candidate_history(
    output: Path,
    template_hdus: list[fits.hdu.base.ExtensionHDU],
    template_data: np.ndarray,
    target_common_rows: np.ndarray,
    segment: dict[str, Any],
    models: dict[int, dict[str, float | int]],
    extension: str,
) -> dict[str, Any]:
    start = float(segment["start"])
    stop = float(segment["stop"])
    inside = target_common_rows[
        (np.asarray(target_common_rows["TIME"], dtype=float) >= start)
        & (np.asarray(target_common_rows["TIME"], dtype=float) <= stop)
    ]
    common_times = np.unique(
        np.r_[start, np.asarray(inside["TIME"], dtype=float), stop]
    )
    common_values = common_temperature(target_common_rows, common_times)
    records: list[np.void] = []
    for time, common in zip(common_times, common_values, strict=True):
        for pixel in range(36):
            record = nearest_record(template_data, pixel, float(time)).copy()
            record["TIME"] = time
            record["PIXEL"] = pixel
            record["TEMP_FIT"] = common + predict_differential(
                models[pixel], np.asarray([time])
            )[0]
            records.append(record)
    output_data = np.asarray(records, dtype=template_data.dtype)
    output_hdus = [hdu.copy() for hdu in template_hdus]
    index = next(
        i for i, hdu in enumerate(output_hdus) if hdu.name.casefold() == extension.casefold()
    )
    header = output_hdus[index].header.copy()
    header["TSTART"] = start
    header["TSTOP"] = stop
    output_hdus[index] = fits.BinTableHDU(
        data=output_data, header=header, name=extension
    )
    fits.HDUList(output_hdus).writeto(output, overwrite=False, checksum=True)
    return {
        "rows": len(output_data),
        "time_grid_rows": len(common_times),
        "rows_per_time": 36,
        "temp_fit_finite": bool(
            np.isfinite(np.asarray(output_data["TEMP_FIT"], dtype=float)).all()
        ),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
    }


def decompress_gzip(source: Path, destination: Path) -> dict[str, Any]:
    with gzip.open(source, "rb") as input_stream, destination.open("xb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream)
    return {"bytes": destination.stat().st_size, "sha256": sha256(destination)}


def ftcopy_command(
    config: dict[str, Any], source: Path, output: Path, segment: dict[str, Any]
) -> str:
    selection = (
        f"{application.to_wsl_path(source)}[EVENTS][TIME>={segment['start']}"
        f"&&TIME<={segment['stop']}&&PIXEL!=12&&ITYPE==0]"
    )
    return (
        application.runtime_environment(config)
        + "ftcopy "
        + shlex.quote(selection)
        + " "
        + shlex.quote(application.to_wsl_path(output))
        + " copyall=yes clobber=yes history=yes"
    )


def fit_energies(
    energies: np.ndarray,
    template: dict[str, np.ndarray],
    fit_config: dict[str, Any],
) -> dict[str, Any]:
    edges = np.arange(
        fit_config["energy_min_ev"],
        fit_config["energy_max_ev"] + fit_config["bin_width_ev"],
        fit_config["bin_width_ev"],
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    selected = energies[
        np.isfinite(energies)
        & (energies >= fit_config["energy_min_ev"])
        & (energies <= fit_config["energy_max_ev"])
    ]
    observed, _ = np.histogram(selected, bins=edges)
    fit = line_shape.fit_histogram(
        observed,
        centers,
        fit_config["bin_width_ev"],
        template,
        fit_config["bounds"],
    )
    if fit_config["require_convergence"] and not fit["converged"]:
        raise RuntimeError("held-out Mn K-alpha fit did not converge")
    return fit


def fit_output(
    path: Path,
    template: dict[str, np.ndarray],
    fit_config: dict[str, Any],
    requirements: dict[str, Any],
) -> dict[str, Any]:
    with fits.open(path, memmap=True, mode="readonly") as hdus:
        data = hdus["EVENTS"].data
        pixels = np.asarray(data["PIXEL"], dtype=int).copy()
        energies = np.asarray(data["EPI2"], dtype=float).copy()
    whole = fit_energies(energies, template, fit_config)
    if whole["events_in_fit_window"] < requirements[
        "require_at_least_events_per_whole_array_fit_window"
    ]:
        raise RuntimeError("held-out whole-array fit has too few events")
    per_pixel: dict[str, dict[str, Any]] = {}
    for pixel in [value for value in range(36) if value != 12]:
        fit = fit_energies(energies[pixels == pixel], template, fit_config)
        if fit["events_in_fit_window"] < requirements[
            "require_at_least_events_per_pixel_fit_window"
        ]:
            raise RuntimeError(f"held-out pixel {pixel} fit has too few events")
        per_pixel[str(pixel)] = fit
    return {"whole_array": whole, "per_pixel": per_pixel}


def compare_fits(
    candidate: dict[str, Any], control: dict[str, Any], gate: dict[str, Any]
) -> dict[str, Any]:
    whole_centroid = (
        candidate["whole_array"]["centroid_shift_ev"]
        - control["whole_array"]["centroid_shift_ev"]
    )
    whole_fwhm = (
        candidate["whole_array"]["instrument_fwhm_ev"]
        - control["whole_array"]["instrument_fwhm_ev"]
    )
    centroid_deltas = []
    fwhm_increases = []
    per_pixel = {}
    for pixel in sorted(control["per_pixel"], key=int):
        centroid_delta = (
            candidate["per_pixel"][pixel]["centroid_shift_ev"]
            - control["per_pixel"][pixel]["centroid_shift_ev"]
        )
        fwhm_increase = (
            candidate["per_pixel"][pixel]["instrument_fwhm_ev"]
            - control["per_pixel"][pixel]["instrument_fwhm_ev"]
        )
        centroid_deltas.append(abs(centroid_delta))
        fwhm_increases.append(fwhm_increase)
        per_pixel[pixel] = {
            "centroid_delta_ev": float(centroid_delta),
            "absolute_centroid_delta_ev": float(abs(centroid_delta)),
            "fwhm_increase_ev": float(fwhm_increase),
        }
    centroid_p90 = float(np.quantile(centroid_deltas, 0.9))
    fwhm_p90 = float(np.quantile(fwhm_increases, 0.9))
    passed = (
        abs(whole_centroid)
        <= gate["maximum_absolute_candidate_minus_control_whole_array_centroid_ev"]
        and whole_fwhm <= gate["maximum_candidate_minus_control_whole_array_fwhm_ev"]
        and centroid_p90
        <= gate[
            "maximum_per_fold_90th_percentile_absolute_per_pixel_centroid_delta_ev"
        ]
        and fwhm_p90
        <= gate["maximum_per_fold_90th_percentile_per_pixel_fwhm_increase_ev"]
    )
    return {
        "whole_array_centroid_delta_ev": float(whole_centroid),
        "whole_array_fwhm_increase_ev": float(whole_fwhm),
        "per_pixel_absolute_centroid_delta_p90_ev": centroid_p90,
        "per_pixel_fwhm_increase_p90_ev": fwhm_p90,
        "per_pixel": per_pixel,
        "passed": passed,
    }


def build_report(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config, topology, provenance, line_config = validate_inputs(config_path)
    raw_root = (ROOT / config["paths"]["raw_root"]).resolve()
    scratch_root = (ROOT / config["paths"]["scratch_root"]).resolve()
    if scratch_root.exists():
        raise RuntimeError(f"refusing to overwrite closure scratch root: {scratch_root}")
    if not scratch_root.is_relative_to((ROOT / "tmp").resolve()):
        raise RuntimeError("closure scratch root must remain under repository tmp")
    scratch_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=scratch_root.name + ".installing.", dir=scratch_root.parent)
    )
    provenance_by_path = {item["download_path"]: item for item in provenance["records"]}
    segments = topology["segment_sets"]["10800"]
    obsids = [segment["obsids"][0] for segment in segments]
    extension = config["inputs"]["gain_history_extension"]
    minimum = config["inputs"]["minimum_gain_events"]

    fe55_paths: dict[str, Path] = {}
    pxcal_paths: dict[str, Path] = {}
    event_paths: dict[str, Path] = {}
    fe55_hdus: dict[str, list[fits.hdu.base.ExtensionHDU]] = {}
    fe55_data: dict[str, np.ndarray] = {}
    fe55_valid: dict[str, np.ndarray] = {}
    pxcal_valid: dict[str, np.ndarray] = {}
    sources: list[dict[str, Any]] = []
    for obsid in obsids:
        roles = (
            ("fe55_history_template", "fe55_history"),
            ("calibration_pixel_history_template", "calibration_pixel_history"),
            ("filter_wheel_event_template", "filter_wheel_events"),
        )
        verified: dict[str, tuple[Path, dict[str, Any]]] = {}
        for template_name, role in roles:
            relative = config["inputs"][template_name].format(obsid=obsid)
            verified[role] = verified_raw_path(raw_root, relative, provenance_by_path)
            _, source_record = verified[role]
            sources.append(
                {
                    "obsid": obsid,
                    "kind": role,
                    "path": relative,
                    "bytes": source_record["bytes"],
                    "sha256": source_record["sha256"],
                }
            )
        fe55_paths[obsid] = verified["fe55_history"][0]
        pxcal_paths[obsid] = verified["calibration_pixel_history"][0]
        event_paths[obsid] = verified["filter_wheel_events"][0]
        hdus, data, valid = read_gain_history(fe55_paths[obsid], extension, minimum)
        fe55_hdus[obsid] = hdus
        fe55_data[obsid] = data
        fe55_valid[obsid] = valid
        _, _, pxvalid = read_gain_history(pxcal_paths[obsid], extension, minimum)
        pxcal_valid[obsid] = pxvalid

    template_path = (ROOT / config["paths"]["line_template"]).resolve()
    if (
        not template_path.is_file()
        or sha256(template_path) != line_config["line_template"]["sha256"]
    ):
        raise RuntimeError("frozen line template changed")
    template = line_shape.read_template(
        template_path, line_config["line_template"]["extension"]
    )

    commands: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    distribution = config["runtime"]["wsl_distribution"]
    for segment, obsid in zip(segments, obsids, strict=True):
        fold_dir = staging / f"segment_{segment['segment']}_{obsid}"
        fold_dir.mkdir()
        models = fit_differential_models(obsid, fe55_valid, pxcal_valid)
        training_obsid = next(value for value in obsids if value != obsid)
        candidate_history = fold_dir / "common_differential.ghf"
        candidate_history_audit = build_candidate_history(
            candidate_history,
            fe55_hdus[training_obsid],
            fe55_data[training_obsid],
            pxcal_valid[obsid],
            segment,
            models,
            extension,
        )
        control_history = fold_dir / "native_fe55_control.ghf"
        control_history_audit = decompress_gzip(fe55_paths[obsid], control_history)
        selected = fold_dir / "selected_main_array_fe55_hp.evt"
        copy_result = application.run_wsl(
            distribution,
            ftcopy_command(config, event_paths[obsid], selected, segment),
            timeout=1200,
        )
        commands.append({"stage": "ftcopy", "obsid": obsid, **copy_result})
        if copy_result["exit_code"] != 0 or not selected.is_file():
            raise RuntimeError(f"held-out Fe-55 selection failed: {obsid}")
        selected_rows = application.event_rows(selected)
        outputs: dict[str, dict[str, Any]] = {}
        fitted: dict[str, dict[str, Any]] = {}
        for name, history in (
            ("control", control_history),
            ("candidate", candidate_history),
        ):
            output = fold_dir / f"{name}.evt"
            result = application.run_wsl(
                distribution,
                application.rslpha2pi_command(config, selected, output, history),
                timeout=2400,
            )
            commands.append(
                {"stage": "rslpha2pi", "obsid": obsid, "model": name, **result}
            )
            if result["exit_code"] != 0 or not output.is_file():
                raise RuntimeError(f"held-out Fe-55 application failed: {obsid} {name}")
            audit = application.audit_output(output)
            if (
                audit["rows"] != selected_rows
                or audit["null_epi2"] != 0
                or audit["null_temp"] != 0
                or audit["null_pi_not_explained_by_negative_epi2"] != 0
                or audit["negative_epi2_without_null_pi"] != 0
            ):
                raise RuntimeError(f"held-out Fe-55 output audit failed: {obsid} {name}")
            outputs[name] = audit
            fitted[name] = fit_output(
                output, template, line_config["fit"], config["fit"]
            )
        comparison = compare_fits(
            fitted["candidate"], fitted["control"], config["terminal_gate"]
        )
        folds.append(
            {
                "segment": segment["segment"],
                "obsid": obsid,
                "validation_role": (
                    "endpoint_extrapolation"
                    if segment["segment"] in (0, len(segments) - 1)
                    else "interior_interpolation"
                ),
                "training_obsids": sorted(value for value in obsids if value != obsid),
                "selected_rows": selected_rows,
                "differential_models": {str(key): value for key, value in models.items()},
                "candidate_history": candidate_history_audit,
                "control_history": control_history_audit,
                "outputs": outputs,
                "fits": fitted,
                "comparison": comparison,
            }
        )
    gate_passed = len(folds) == 4 and all(fold["comparison"]["passed"] for fold in folds)
    os.replace(staging, scratch_root)
    report = {
        "protocol_version": config["protocol_version"],
        "status": "a2319_common_differential_gain_closure_completed",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "source_files": sources,
        "folds": folds,
        "commands": commands,
        "terminal_gate_passed": gate_passed,
        "cluster_sky_event_accessed": False,
        "cluster_velocity_fit": False,
        "validation_or_holdout_accessed": False,
        "decision": (
            "authorize_freeze_of_common_differential_science_branch_application"
            if gate_passed
            else "stop_without_science_branch_application"
        ),
        "authorization": {
            "freeze_science_branch_application_protocol": gate_passed,
            "read_cluster_sky_event_rows": False,
            "apply_gain_to_cluster_sky_events": False,
            "fit_cluster_velocity": False,
            "access_validation_or_holdout_assets": False,
            "open_lensing_halo_or_gravity_targets": False,
            "change_gravity_formula_or_parameters": False,
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
                "folds": [
                    {
                        "segment": fold["segment"],
                        "obsid": fold["obsid"],
                        "validation_role": fold["validation_role"],
                        "comparison": fold["comparison"],
                    }
                    for fold in result["folds"]
                ],
                "terminal_gate_passed": result["terminal_gate_passed"],
                "cluster_sky_event_accessed": result["cluster_sky_event_accessed"],
                "decision": result["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
