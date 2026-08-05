#!/usr/bin/env python3
"""Run the frozen V19AG global four-port FORS1 calibration audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19ag_fors1_role_calibration.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def windows_path_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive
    if len(drive) != 2 or drive[1] != ":":
        raise RuntimeError(f"V19AG requires a drive-letter Windows path: {resolved}")
    return f"/mnt/{drive[0].lower()}/{resolved.as_posix()[3:]}"


def decompress_unix_compress(
    source: Path, target: Path, *, distro: str, executable: str
) -> None:
    command = [
        "wsl.exe",
        "-d",
        distro,
        "--",
        executable,
        "-dc",
        "--",
        windows_path_to_wsl(source),
    ]
    with target.open("wb") as output:
        result = subprocess.run(command, stdout=output, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        target.unlink(missing_ok=True)
        raise RuntimeError(
            f"V19AG decompression failed for {source.name}: "
            + result.stderr.decode("utf-8", "replace")
        )


def robust_sigma(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return math.nan
    median = np.median(finite)
    return float(1.4826 * np.median(np.abs(finite - median)))


def _keyword(header: fits.Header, name: str) -> Any:
    if name not in header:
        raise RuntimeError(f"missing required FITS keyword {name}")
    return header[name]


def detector_regions(header: fits.Header, model: dict[str, Any]) -> list[dict[str, Any]]:
    ny, nx = tuple(int(v) for v in model["required_shape_yx"])
    if (int(header["NAXIS2"]), int(header["NAXIS1"])) != (ny, nx):
        raise RuntimeError("selected frame shape does not match frozen detector model")
    if int(_keyword(header, "ESO DET OUTPUTS")) != int(model["required_outputs"]):
        raise RuntimeError("selected frame is not four-port")
    if str(_keyword(header, "ESO DET CHIP1 ID")).strip() != model["required_chip_id"]:
        raise RuntimeError("selected frame chip ID changed")
    if not str(_keyword(header, "ESO DET READ CLOCK")).startswith(
        model["required_read_clock_prefix"]
    ):
        raise RuntimeError("selected frame read-clock changed")

    output_positions = []
    for port in range(1, 5):
        output_positions.append(
            (
                int(_keyword(header, f"ESO DET OUT{port} X")),
                int(_keyword(header, f"ESO DET OUT{port} Y")),
            )
        )
    min_x = min(x for x, _ in output_positions)
    min_y = min(y for _, y in output_positions)
    expected = model["required_per_output"]
    regions: list[dict[str, Any]] = []

    for port, (position_x, position_y) in enumerate(output_positions, start=1):
        valid_nx = int(_keyword(header, f"ESO DET OUT{port} NX"))
        valid_ny = int(_keyword(header, f"ESO DET OUT{port} NY"))
        prescan_x = int(_keyword(header, f"ESO DET OUT{port} PRSCX"))
        overscan_x = int(_keyword(header, f"ESO DET OUT{port} OVSCX"))
        if (
            valid_nx != int(expected["valid_nx"])
            or valid_ny != int(expected["valid_ny"])
            or prescan_x != int(expected["prescan_x"])
            or overscan_x != int(expected["overscan_x"])
        ):
            raise RuntimeError(f"port {port} dimensions changed")

        offset_x = position_x - min_x if position_x == 1 else nx - valid_nx - prescan_x - overscan_x
        offset_y = position_y - min_y if position_y == 1 else ny - valid_ny
        if position_x == 1:
            prescan = (slice(offset_y, offset_y + valid_ny), slice(offset_x, offset_x + prescan_x))
            valid = (
                slice(offset_y, offset_y + valid_ny),
                slice(offset_x + prescan_x, offset_x + prescan_x + valid_nx),
            )
            overscan = (
                slice(offset_y, offset_y + valid_ny),
                slice(offset_x + prescan_x + valid_nx, offset_x + prescan_x + valid_nx + overscan_x),
            )
            side = "left"
        else:
            overscan = (
                slice(offset_y, offset_y + valid_ny),
                slice(offset_x, offset_x + overscan_x),
            )
            valid = (
                slice(offset_y, offset_y + valid_ny),
                slice(offset_x + overscan_x, offset_x + overscan_x + valid_nx),
            )
            prescan = (
                slice(offset_y, offset_y + valid_ny),
                slice(offset_x + overscan_x + valid_nx, offset_x + overscan_x + valid_nx + prescan_x),
            )
            side = "right"
        half = "lower" if position_y == 1 else "upper"
        regions.append(
            {
                "port": port,
                "label": f"{half}_{side}",
                "mosaic_y": slice(0, valid_ny) if half == "lower" else slice(valid_ny, 2 * valid_ny),
                "mosaic_x": slice(0, valid_nx) if side == "left" else slice(valid_nx, 2 * valid_nx),
                "prescan": prescan,
                "valid": valid,
                "overscan": overscan,
            }
        )

    mosaic_shape = tuple(int(v) for v in model["active_mosaic_shape_yx"])
    coverage = np.zeros(mosaic_shape, dtype=np.uint8)
    for region in regions:
        coverage[region["mosaic_y"], region["mosaic_x"]] += 1
    if not np.all(coverage == 1):
        raise RuntimeError("frozen active-mosaic port coverage is not exact")
    return regions


def correct_ports(
    raw: np.ndarray, regions: list[dict[str, Any]], shape: tuple[int, int]
) -> tuple[np.ndarray, list[dict[str, float]]]:
    mosaic = np.full(shape, np.nan, dtype=np.float32)
    diagnostics: list[dict[str, float]] = []
    for region in regions:
        prescan = np.asarray(raw[region["prescan"]], dtype=np.float32)
        valid = np.asarray(raw[region["valid"]], dtype=np.float32)
        overscan = np.asarray(raw[region["overscan"]], dtype=np.float32)
        row_level = np.median(prescan, axis=1, keepdims=True)
        corrected = valid - row_level
        mosaic[region["mosaic_y"], region["mosaic_x"]] = corrected
        diagnostics.append(
            {
                "port": int(region["port"]),
                "prescan_global_median_adu": float(np.median(prescan)),
                "prescan_row_level_sigma_adu": robust_sigma(row_level),
                "overscan_global_median_adu": float(np.median(overscan)),
                "corrected_valid_median_adu": float(np.median(corrected)),
                "corrected_valid_sigma_adu": robust_sigma(corrected),
            }
        )
    if not np.all(np.isfinite(mosaic)):
        raise RuntimeError("port correction left uncovered or nonfinite active pixels")
    return mosaic, diagnostics


def port_slices(shape: tuple[int, int]) -> list[tuple[slice, slice]]:
    ny, nx = shape
    return [
        (slice(0, ny // 2), slice(0, nx // 2)),
        (slice(0, ny // 2), slice(nx // 2, nx)),
        (slice(ny // 2, ny), slice(0, nx // 2)),
        (slice(ny // 2, ny), slice(nx // 2, nx)),
    ]


def boundary_fraction(image: np.ndarray, strip: int) -> dict[str, float]:
    ny, nx = image.shape
    xmid, ymid = nx // 2, ny // 2
    global_level = float(np.nanmedian(image))
    denominator = max(abs(global_level), np.finfo(float).eps)
    left = float(np.nanmedian(image[:, xmid - strip : xmid]))
    right = float(np.nanmedian(image[:, xmid : xmid + strip]))
    lower = float(np.nanmedian(image[ymid - strip : ymid, :]))
    upper = float(np.nanmedian(image[ymid : ymid + strip, :]))
    vertical = abs(right - left) / denominator
    horizontal = abs(upper - lower) / denominator
    return {
        "global_level_adu": global_level,
        "vertical_fraction": float(vertical),
        "horizontal_fraction": float(horizontal),
        "maximum_fraction": float(max(vertical, horizontal)),
    }


def validate_config(config_path: Path, config: dict[str, Any]) -> dict[str, str]:
    if config["status"] != "frozen_before_any_fits_pixel_access":
        raise RuntimeError("V19AG protocol is not frozen")
    runner = ROOT / config["implementation"]["runner"]
    if sha256(runner) != config["implementation"]["runner_sha256"]:
        raise RuntimeError("frozen V19AG runner hash mismatch")
    hashes = {"config": sha256(config_path), "runner": sha256(runner)}
    for artifact in config["parent_artifacts"]:
        path = ROOT / artifact["path"]
        actual = sha256(path)
        if actual != artifact["sha256"]:
            raise RuntimeError(f"V19AG parent hash mismatch: {artifact['path']}")
        hashes[artifact["path"]] = actual
    authorization = config["authorization"]
    prohibited = [
        "inspect_member_or_candidate_coordinates_or_cutouts",
        "fit_source_photometry_or_counterparts",
        "infer_stellar_mass_or_current",
        "read_lensing_or_halo_payload",
        "change_gravity_physics_or_parameters",
        "open_holdout",
    ]
    if any(authorization[name] for name in prohibited):
        raise RuntimeError("V19AG authorizes a prohibited action")
    return hashes


def choose_records(config: dict[str, Any], report: dict[str, Any]) -> list[dict[str, Any]]:
    records = report["records"]
    science_ids = set(config["selection"]["primary_science_dp_ids"])
    chosen = [record for record in records if record["role"] in {"bias", "flat"}]
    chosen.extend(record for record in records if record["dp_id"] in science_ids)
    if len({record["dp_id"] for record in chosen}) != len(chosen):
        raise RuntimeError("duplicate selected DP ID")
    roles = Counter(record["role"] for record in chosen)
    gates = config["gates"]
    if roles != Counter(
        {
            "science": int(gates["exact_science_count"]),
            "bias": int(gates["exact_bias_count"]),
            "flat": sum(int(value) for value in gates["exact_flat_counts_by_filter"].values()),
        }
    ):
        raise RuntimeError(f"selected role counts changed: {roles}")
    filters = Counter(record["filter_path"] for record in chosen if record["role"] == "flat")
    if filters != Counter({key: int(value) for key, value in gates["exact_flat_counts_by_filter"].items()}):
        raise RuntimeError(f"selected flat counts changed: {filters}")
    science_filters = [record["filter_path"] for record in chosen if record["role"] == "science"]
    if Counter(science_filters) != Counter(config["selection"]["required_science_filters"]):
        raise RuntimeError("selected science filter triplet changed")
    return sorted(chosen, key=lambda record: (record["role"], record["dp_id"]))


def load_selected_pixels(
    record: dict[str, Any], config: dict[str, Any], temp_dir: Path
) -> tuple[np.ndarray, fits.Header, list[dict[str, Any]], list[dict[str, float]], float]:
    source = ROOT / record["compressed_path"]
    if sha256(source) != record["compressed_sha256"]:
        raise RuntimeError(f"compressed hash mismatch: {record['dp_id']}")
    target = temp_dir / f"{record['dp_id'].replace(':', '_')}.fits"
    dec = config["decompression"]
    decompress_unix_compress(
        source,
        target,
        distro=dec["wsl_distribution"],
        executable=dec["executable"],
    )
    try:
        with fits.open(target, memmap=False, do_not_scale_image_data=False) as hdul:
            if len(hdul) != 1:
                raise RuntimeError("selected raw file has unexpected FITS extensions")
            header = hdul[0].header.copy()
            raw = np.asarray(hdul[0].data, dtype=np.float32)
    finally:
        target.unlink(missing_ok=True)
    regions = detector_regions(header, config["detector_model"])
    shape = tuple(int(value) for value in config["detector_model"]["active_mosaic_shape_yx"])
    corrected, port_diagnostics = correct_ports(raw, regions, shape)
    raw_saturated_fraction = float(
        np.mean(raw >= float(config["statistics"]["saturation_adu"]))
    )
    return corrected, header, regions, port_diagnostics, raw_saturated_fraction


def block_medians(image: np.ndarray, block_shape: tuple[int, int]) -> np.ndarray:
    by, bx = block_shape
    ny, nx = image.shape
    if ny % by or nx % bx:
        raise RuntimeError("frozen bias block shape no longer tiles active mosaic")
    return np.median(image.reshape(ny // by, by, nx // bx, bx), axis=(1, 3))


def write_product(
    path: Path,
    data: np.ndarray,
    header: fits.Header,
    *,
    product: str,
    source_dp_id: str | None = None,
) -> None:
    product_header = header.copy()
    product_header["NAXIS1"] = data.shape[1]
    product_header["NAXIS2"] = data.shape[0]
    product_header["SIGV19AG"] = (True, "Frozen V19AG global detector calibration")
    product_header["SIGPROD"] = (product, "Sigma V19AG product type")
    product_header["SIGWCS"] = ("APPROX", "Astrometry not yet independently solved")
    if "CRPIX1" in product_header:
        product_header["CRPIX1"] = (data.shape[1] + 1) / 2.0
    if source_dp_id is not None:
        product_header["SIGDPID"] = source_dp_id
    fits.PrimaryHDU(np.asarray(data, dtype=np.float32), header=product_header).writeto(
        path, overwrite=True
    )


def run(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_json(config_path)
    input_hashes = validate_config(config_path, config)
    parent_report = load_json(ROOT / "results/sigma_v19af_fors1_header_compatibility/report.json")
    selected = choose_records(config, parent_report)
    gates_config = config["gates"]
    mosaic_shape = tuple(int(value) for value in config["detector_model"]["active_mosaic_shape_yx"])

    arrays: dict[
        str, list[tuple[dict[str, Any], np.ndarray, fits.Header, float]]
    ] = defaultdict(list)
    file_metrics: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="sigma-v19ag-") as temporary:
        temp_dir = Path(temporary)
        for record in selected:
            corrected, header, _, port_diagnostics, raw_saturated_fraction = load_selected_pixels(
                record, config, temp_dir
            )
            arrays[record["role"]].append(
                (record, corrected, header, raw_saturated_fraction)
            )
            file_metrics.append(
                {
                    "dp_id": record["dp_id"],
                    "role": record["role"],
                    "filter": record["filter_path"],
                    "compressed_sha256": record["compressed_sha256"],
                    "active_shape_yx": list(corrected.shape),
                    "corrected_global_median_adu": float(np.median(corrected)),
                    "corrected_global_sigma_adu": robust_sigma(corrected),
                    "corrected_finite_fraction": float(np.mean(np.isfinite(corrected))),
                    "raw_saturated_fraction_proxy": raw_saturated_fraction,
                    "ports": port_diagnostics,
                }
            )
    decompressed_payload_persisted = False

    bias_stack = np.stack([array for _, array, _, _ in arrays["bias"]]).astype(np.float32)
    master_bias_residual = np.median(bias_stack, axis=0)
    blocks = block_medians(
        master_bias_residual,
        tuple(int(value) for value in config["statistics"]["bias_block_shape_yx"]),
    )
    master_bias_port_medians = [
        float(np.median(master_bias_residual[region])) for region in port_slices(mosaic_shape)
    ]
    frame_port_medians = [
        [float(np.median(frame[region])) for region in port_slices(mosaic_shape)]
        for frame in bias_stack
    ]
    pair_rons: list[list[float]] = []
    for index in range(0, bias_stack.shape[0] - 1, 2):
        difference = bias_stack[index] - bias_stack[index + 1]
        pair_rons.append(
            [robust_sigma(difference[region]) / math.sqrt(2.0) for region in port_slices(mosaic_shape)]
        )
    empirical_ron_by_port = np.median(np.asarray(pair_rons), axis=0)
    bias_metrics = {
        "master_residual_global_median_adu": float(np.median(master_bias_residual)),
        "master_residual_global_sigma_adu": robust_sigma(master_bias_residual),
        "block_peak_to_peak_adu": float(np.max(blocks) - np.min(blocks)),
        "port_medians_adu": master_bias_port_medians,
        "port_median_spread_adu": float(max(master_bias_port_medians) - min(master_bias_port_medians)),
        "frame_port_abs_median_max_adu": float(np.max(np.abs(frame_port_medians))),
        "empirical_ron_adu_by_port": [float(value) for value in empirical_ron_by_port],
        "empirical_ron_adu_min": float(np.min(empirical_ron_by_port)),
        "empirical_ron_adu_max": float(np.max(empirical_ron_by_port)),
    }

    flat_groups: dict[str, list[np.ndarray]] = defaultdict(list)
    flat_headers: dict[str, fits.Header] = {}
    flat_frame_metrics: list[dict[str, Any]] = []
    for record, frame, header, raw_saturated_fraction in arrays["flat"]:
        median = float(np.median(frame))
        normalized = frame / median
        flat_groups[record["filter_path"]].append(normalized.astype(np.float32))
        flat_headers.setdefault(record["filter_path"], header)
        flat_frame_metrics.append(
            {
                "dp_id": record["dp_id"],
                "filter": record["filter_path"],
                "global_median_adu": median,
                "saturated_fraction_proxy": raw_saturated_fraction,
            }
        )

    master_flats: dict[str, np.ndarray] = {}
    flat_filter_metrics: dict[str, Any] = {}
    valid_low, valid_high = (float(value) for value in config["statistics"]["flat_valid_response"])
    strip = int(config["statistics"]["boundary_strip_pixels"])
    for filter_name, frames in sorted(flat_groups.items()):
        stack = np.stack(frames).astype(np.float32)
        master = np.median(stack, axis=0)
        master /= float(np.median(master))
        temporal_sigma = 1.4826 * np.median(np.abs(stack - master[None, :, :]), axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            temporal_snr = master * math.sqrt(stack.shape[0]) / temporal_sigma
        valid = np.isfinite(master) & (master >= valid_low) & (master <= valid_high)
        calibrated_boundaries = [boundary_fraction(frame / master, strip) for frame in stack]
        master_flats[filter_name] = master.astype(np.float32)
        flat_filter_metrics[filter_name] = {
            "frame_count": int(stack.shape[0]),
            "master_global_median": float(np.median(master)),
            "master_global_sigma": robust_sigma(master),
            "master_response_min": float(np.min(master)),
            "master_response_max": float(np.max(master)),
            "valid_fraction": float(np.mean(valid)),
            "temporal_snr100_fraction": float(np.mean(np.isinf(temporal_snr) | (temporal_snr >= 100.0))),
            "calibrated_frame_boundaries": calibrated_boundaries,
            "calibrated_boundary_fraction_max": float(
                max(item["maximum_fraction"] for item in calibrated_boundaries)
            ),
        }

    science_products: list[tuple[dict[str, Any], np.ndarray, fits.Header]] = []
    science_metrics: list[dict[str, Any]] = []
    for record, frame, header, raw_saturated_fraction in arrays["science"]:
        master = master_flats[record["filter_path"]]
        valid = np.isfinite(master) & (master >= valid_low) & (master <= valid_high)
        calibrated = np.full(mosaic_shape, np.nan, dtype=np.float32)
        calibrated[valid] = frame[valid] / master[valid]
        finite_fraction = float(np.mean(np.isfinite(calibrated)))
        boundary = boundary_fraction(calibrated, strip)
        background = float(np.nanmedian(calibrated))
        science_products.append((record, calibrated, header))
        science_metrics.append(
            {
                "dp_id": record["dp_id"],
                "filter": record["filter_path"],
                "finite_fraction": finite_fraction,
                "saturated_fraction_proxy": raw_saturated_fraction,
                "background_median_adu": background,
                "background_sigma_adu": robust_sigma(calibrated),
                "boundary": boundary,
            }
        )

    flat_medians = [item["global_median_adu"] for item in flat_frame_metrics]
    flat_saturated = [item["saturated_fraction_proxy"] for item in flat_frame_metrics]
    gate_results = {
        "exact_selected_counts": True,
        "all_selected_compressed_hashes_match_v19ae": True,
        "all_selected_geometry_matches_detector_model": True,
        "bias_block_peak_to_peak": bias_metrics["block_peak_to_peak_adu"]
        <= float(gates_config["bias_block_peak_to_peak_adu_max"]),
        "bias_port_median_spread": bias_metrics["port_median_spread_adu"]
        <= float(gates_config["bias_port_median_spread_adu_max"]),
        "bias_frame_port_abs_median": bias_metrics["frame_port_abs_median_max_adu"]
        <= float(gates_config["bias_frame_port_abs_median_adu_max"]),
        "bias_empirical_ron_range": bias_metrics["empirical_ron_adu_min"]
        >= float(gates_config["bias_empirical_ron_adu_min"])
        and bias_metrics["empirical_ron_adu_max"]
        <= float(gates_config["bias_empirical_ron_adu_max"]),
        "flat_frame_median_range": min(flat_medians)
        >= float(gates_config["flat_frame_global_median_adu_min"])
        and max(flat_medians) <= float(gates_config["flat_frame_global_median_adu_max"]),
        "flat_frame_saturation": max(flat_saturated)
        <= float(gates_config["flat_frame_saturated_fraction_max"]),
        "master_flat_valid_fraction": all(
            metrics["valid_fraction"] >= float(gates_config["master_flat_valid_fraction_min"])
            for metrics in flat_filter_metrics.values()
        ),
        "master_flat_temporal_snr": all(
            metrics["temporal_snr100_fraction"]
            >= float(gates_config["master_flat_temporal_snr100_fraction_min"])
            for metrics in flat_filter_metrics.values()
        ),
        "calibrated_flat_boundaries": all(
            metrics["calibrated_boundary_fraction_max"]
            <= float(gates_config["calibrated_flat_boundary_fraction_max"])
            for metrics in flat_filter_metrics.values()
        ),
        "science_finite_fraction": all(
            metrics["finite_fraction"] >= float(gates_config["science_finite_fraction_min"])
            for metrics in science_metrics
        ),
        "science_saturation": all(
            metrics["saturated_fraction_proxy"] <= float(gates_config["science_saturated_fraction_max"])
            for metrics in science_metrics
        ),
        "science_background_positive": all(metrics["background_median_adu"] > 0 for metrics in science_metrics),
        "science_boundaries": all(
            metrics["boundary"]["maximum_fraction"]
            <= float(gates_config["science_boundary_fraction_max"])
            for metrics in science_metrics
        ),
        "no_decompressed_payload_persisted": not decompressed_payload_persisted,
        "no_prohibited_science_or_gravity_access": True,
    }
    all_pass = all(gate_results.values())
    output_dir = ROOT / config["outputs"]["directory"]
    output_dir.mkdir(parents=True, exist_ok=True)
    products: list[dict[str, str]] = []
    if all_pass:
        for filter_name, master in sorted(master_flats.items()):
            path = output_dir / f"master_flat_{filter_name}.fits.gz"
            write_product(path, master, flat_headers[filter_name], product="MASTER_FLAT")
            products.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(path)})
        for record, calibrated, header in science_products:
            path = output_dir / f"calibrated_{record['filter_path']}_{record['dp_id'].replace(':', '_')}.fits.gz"
            write_product(
                path,
                calibrated,
                header,
                product="CALIBRATED_SCIENCE",
                source_dp_id=record["dp_id"],
            )
            products.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(path)})

    report = {
        "report_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "status": "passed_global_calibration" if all_pass else "failed_global_calibration",
        "input_hashes": input_hashes,
        "selection": {
            "science_dp_ids": config["selection"]["primary_science_dp_ids"],
            "selected_role_counts": dict(Counter(record["role"] for record in selected)),
            "flat_counts_by_filter": dict(
                Counter(record["filter_path"] for record in selected if record["role"] == "flat")
            ),
            "excluded_science_dp_ids": config["selection"]["excluded_science_dp_ids"],
        },
        "detector_model": config["detector_model"],
        "file_metrics": file_metrics,
        "bias_metrics": bias_metrics,
        "flat_frame_metrics": flat_frame_metrics,
        "flat_filter_metrics": flat_filter_metrics,
        "science_metrics": science_metrics,
        "gates": gate_results,
        "all_global_calibration_gates_pass": all_pass,
        "products": products,
        "decompressed_payload_persisted": decompressed_payload_persisted,
        "member_candidate_coordinate_or_cutout_opened": False,
        "photometry_or_counterpart_fitted": False,
        "stellar_mass_or_current_inferred": False,
        "lensing_or_halo_payload_opened": False,
        "gravity_formula_or_parameter_changed": False,
        "claim_boundary": config["claim_boundary"],
    }
    report_path = ROOT / config["outputs"]["report"]
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    report = run(args.config)
    print(json.dumps({"status": report["status"], "gates": report["gates"]}, indent=2))
    return 0 if report["all_global_calibration_gates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
