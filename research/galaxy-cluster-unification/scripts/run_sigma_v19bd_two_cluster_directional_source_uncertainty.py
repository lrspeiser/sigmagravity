#!/usr/bin/env python3
"""Quantify source-only directional morphology over both member ensembles."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19bd_two_cluster_directional_source_uncertainty.json"
SPEED_OF_LIGHT_KM_S = 299792.458
PERCENTILES = [2.5, 16.0, 50.0, 84.0, 97.5]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_parent_hashes(config: dict[str, Any]) -> dict[str, str]:
    actual: dict[str, str] = {}
    for name, spec in config["parents"].items():
        path = ROOT / spec["path"]
        value = sha256(path)
        if value != spec["sha256"]:
            raise ValueError(f"parent hash mismatch for {name}: {value} != {spec['sha256']}")
        actual[name] = value
    return actual


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def cic_weights(x: float, y: float, nx: int, ny: int) -> list[tuple[int, int, float]]:
    ix = math.floor(x)
    iy = math.floor(y)
    if ix < 0 or iy < 0 or ix + 1 >= nx or iy + 1 >= ny:
        raise ValueError(f"position ({x}, {y}) does not have four in-grid neighbors")
    dx = x - ix
    dy = y - iy
    result = [
        (iy, ix, (1.0 - dx) * (1.0 - dy)),
        (iy, ix + 1, dx * (1.0 - dy)),
        (iy + 1, ix, (1.0 - dx) * dy),
        (iy + 1, ix + 1, dx * dy),
    ]
    if abs(math.fsum(weight for _, _, weight in result) - 1.0) > 1e-14:
        raise ValueError("cloud-in-cell weights do not sum to one")
    return result


def weighted_centroid(xy: np.ndarray, weights: np.ndarray) -> np.ndarray:
    total = float(np.sum(weights))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("centroid weights must have a positive finite sum")
    return np.sum(xy * weights[:, None], axis=0) / total


def compute_morphology(
    xy_arcsec: np.ndarray, luminosity: np.ndarray, beta: np.ndarray
) -> dict[str, Any]:
    xy = np.asarray(xy_arcsec, dtype=float)
    ell = np.asarray(luminosity, dtype=float)
    velocity = np.asarray(beta, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] != ell.size or ell.size != velocity.size:
        raise ValueError("morphology arrays have incompatible shapes")
    if ell.size < 2 or np.any(~np.isfinite(xy)) or np.any(~np.isfinite(ell)):
        raise ValueError("morphology inputs are nonfinite or undersized")
    if np.any(ell <= 0.0) or np.any(~np.isfinite(velocity)):
        raise ValueError("luminosity and velocity inputs are invalid")

    luminosity_centroid = weighted_centroid(xy, ell)
    centered = xy - luminosity_centroid
    covariance = np.einsum("n,ni,nj->ij", ell, centered, centered) / float(np.sum(ell))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if np.any(~np.isfinite(eigenvalues)) or eigenvalues[0] < -1e-12:
        raise ValueError("luminosity covariance is invalid")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    radius = math.sqrt(float(np.sum(eigenvalues)))
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("luminosity RMS radius is not positive")
    major_axis = eigenvectors[:, -1]
    eigenvalue_sum = float(np.sum(eigenvalues))
    ellipticity = float((eigenvalues[-1] - eigenvalues[0]) / eigenvalue_sum)

    second_weights = ell * velocity * velocity
    positive_weights = ell * np.maximum(velocity, 0.0)
    negative_weights = ell * np.maximum(-velocity, 0.0)
    second_centroid = weighted_centroid(xy, second_weights)
    positive_centroid = weighted_centroid(xy, positive_weights)
    negative_centroid = weighted_centroid(xy, negative_weights)
    second_vector = second_centroid - luminosity_centroid
    current_vector = positive_centroid - negative_centroid
    second_distance = float(np.linalg.norm(second_vector))
    current_distance = float(np.linalg.norm(current_vector))
    if second_distance <= 0.0 or current_distance <= 0.0:
        raise ValueError("directional centroid displacement is zero")
    current_unit = current_vector / current_distance
    second_unit = second_vector / second_distance
    current_axis_alignment = float(
        np.clip(2.0 * np.dot(major_axis, current_unit) ** 2 - 1.0, -1.0, 1.0)
    )
    second_current_alignment = float(np.clip(abs(np.dot(second_unit, current_unit)), 0.0, 1.0))
    return {
        "finite_luminosity_member_count": int(ell.size),
        "luminosity_total_within_band": float(np.sum(ell)),
        "positive_current_weight": float(np.sum(positive_weights)),
        "negative_current_weight": float(np.sum(negative_weights)),
        "second_moment_weight": float(np.sum(second_weights)),
        "luminosity_centroid_x_arcsec": float(luminosity_centroid[0]),
        "luminosity_centroid_y_arcsec": float(luminosity_centroid[1]),
        "second_centroid_x_arcsec": float(second_centroid[0]),
        "second_centroid_y_arcsec": float(second_centroid[1]),
        "positive_current_centroid_x_arcsec": float(positive_centroid[0]),
        "positive_current_centroid_y_arcsec": float(positive_centroid[1]),
        "negative_current_centroid_x_arcsec": float(negative_centroid[0]),
        "negative_current_centroid_y_arcsec": float(negative_centroid[1]),
        "luminosity_rms_radius_arcsec": radius,
        "luminosity_axis_ellipticity": ellipticity,
        "second_moment_offset_arcsec": second_distance,
        "opposite_current_separation_arcsec": current_distance,
        "normalized_second_offset": second_distance / radius,
        "normalized_current_separation": current_distance / radius,
        "current_axis_alignment_cos2": current_axis_alignment,
        "second_current_axis_alignment_abs_cos": second_current_alignment,
    }


def load_cluster_draws(
    cluster: str,
    spec: dict[str, Any],
    paths: dict[str, Path],
) -> tuple[list[dict[str, Any]], dict[str, float], int]:
    with fits.open(paths[spec["grid_parent"]], memmap=False) as hdul:
        wcs = WCS(hdul[0].header)
        ny, nx = hdul[0].data.shape
    pixel_scale_arcsec = float(np.mean(np.abs(wcs.wcs.cdelt)) * 3600.0)
    ensemble_path = paths[spec["ensemble_parent"]]
    luminosity_field = spec["luminosity_field"]
    rows_out: list[dict[str, Any]] = []
    aggregate_maps = np.zeros((3, ny, nx), dtype=np.float64)
    total_rows = 0
    reference_members: set[str] | None = None

    def process_sample(sample_id: int, rows: list[dict[str, str]]) -> None:
        nonlocal total_rows
        nonlocal reference_members
        expected_members = int(spec["expected_members_per_draw"])
        if len(rows) != expected_members:
            raise ValueError(f"{cluster} sample {sample_id} has {len(rows)} rows")
        members = {row["member_id"] for row in rows}
        if len(members) != expected_members:
            raise ValueError(f"{cluster} sample {sample_id} has duplicate members")
        if reference_members is None:
            reference_members = members
        elif members != reference_members:
            raise ValueError(f"{cluster} sample {sample_id} member inventory changed")

        finite_rows = [row for row in rows if row[luminosity_field]]
        ra = np.asarray([float(row["ra_deg"]) for row in finite_rows])
        dec = np.asarray([float(row["dec_deg"]) for row in finite_rows])
        x, y = wcs.world_to_pixel_values(ra, dec)
        xy = np.column_stack([x, y]) * pixel_scale_arcsec
        ell = np.asarray([float(row[luminosity_field]) for row in finite_rows])
        beta = np.asarray([float(row["v_los_rest_km_s"]) for row in finite_rows]) / (
            SPEED_OF_LIGHT_KM_S
        )
        morphology = compute_morphology(xy, ell, beta)
        rows_out.append({"sample_id": sample_id, "cluster": cluster, **morphology})

        second_weights = ell * beta * beta
        values = np.column_stack([ell, ell * beta, second_weights])
        for pixel_xy, particle_values in zip(np.column_stack([x, y]), values, strict=True):
            for iy, ix, weight in cic_weights(float(pixel_xy[0]), float(pixel_xy[1]), nx, ny):
                aggregate_maps[:, iy, ix] += weight * particle_values
        total_rows += len(rows)

    with gzip.open(ensemble_path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        active_id: int | None = None
        active_rows: list[dict[str, str]] = []
        for row in reader:
            sample_id = int(row["sample_id"])
            if active_id is None:
                if sample_id != 0:
                    raise ValueError(f"{cluster} sample sequence starts at {sample_id}")
                active_id = sample_id
            if sample_id != active_id:
                if sample_id != active_id + 1:
                    raise ValueError(
                        f"{cluster} sample sequence jumps from {active_id} to {sample_id}"
                    )
                process_sample(active_id, active_rows)
                active_id = sample_id
                active_rows = []
            active_rows.append(row)
        if active_id is not None:
            process_sample(active_id, active_rows)
    if len(rows_out) != int(spec["expected_draws"]):
        raise ValueError(f"{cluster} draw count changed")
    return rows_out, aggregate_diagnostics(aggregate_maps, pixel_scale_arcsec), total_rows


def map_centroid(values: np.ndarray) -> np.ndarray:
    total = float(np.sum(values))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("map centroid has invalid total")
    y, x = np.indices(values.shape, dtype=float)
    return np.asarray([np.sum(x * values) / total, np.sum(y * values) / total])


def aggregate_diagnostics(maps: np.ndarray, pixel_scale_arcsec: float) -> dict[str, float]:
    luminosity_centroid = map_centroid(maps[0])
    second_centroid = map_centroid(maps[2])
    positive_centroid = map_centroid(np.maximum(maps[1], 0.0))
    negative_centroid = map_centroid(np.maximum(-maps[1], 0.0))
    return {
        "second_moment_to_luminosity_centroid_offset_arcsec": float(
            np.linalg.norm(second_centroid - luminosity_centroid) * pixel_scale_arcsec
        ),
        "opposite_current_centroid_separation_arcsec": float(
            np.linalg.norm(positive_centroid - negative_centroid) * pixel_scale_arcsec
        ),
    }


def percentile_summary(rows: list[dict[str, Any]], fields: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in fields:
        values = np.asarray([float(row[field]) for row in rows])
        quantiles = np.percentile(values, PERCENTILES)
        result[field] = {
            "percentiles": {
                str(percentile): float(value)
                for percentile, value in zip(PERCENTILES, quantiles, strict=True)
            },
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values)),
        }
    return result


def make_figure(path: Path, cluster_rows: dict[str, list[dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        ("normalized_second_offset", "second-moment offset / light RMS radius"),
        ("normalized_current_separation", "opposite-current separation / light RMS radius"),
        ("current_axis_alignment_cos2", "current/light-axis alignment cos(2 delta theta)"),
    ]
    colors = {"BULLET": "tab:blue", "ABELL2146": "tab:orange"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    for axis, (field, label) in zip(axes, fields, strict=True):
        for cluster, rows in cluster_rows.items():
            values = np.asarray([float(row[field]) for row in rows])
            axis.hist(
                values,
                bins=60,
                density=True,
                histtype="step",
                linewidth=1.7,
                color=colors[cluster],
                label=cluster,
            )
            axis.axvline(np.median(values), color=colors[cluster], linewidth=1.0, linestyle="--")
        axis.set_xlabel(label)
        axis.set_ylabel("posterior density")
        axis.legend()
    fig.suptitle(
        "Two-cluster directional collisionless-source uncertainty\n"
        "source-only posterior draws; no lensing, halo, gas-response, or gravity target"
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_hash = sha256(config_path)
    implementation = config["implementation"]
    runner_path = (ROOT / implementation["runner"]).resolve()
    if runner_path != Path(__file__).resolve():
        raise ValueError("frozen implementation path does not identify this runner")
    runner_hash = sha256(runner_path)
    if runner_hash != implementation["runner_sha256"]:
        raise ValueError("frozen implementation hash mismatch")
    input_hashes = verify_parent_hashes(config)
    paths = {name: ROOT / spec["path"] for name, spec in config["parents"].items()}
    moment_reports = {
        cluster: json.loads(paths[spec["moment_report_parent"]].read_text(encoding="utf-8"))
        for cluster, spec in config["cluster_inputs"].items()
    }
    if any(report["decision"] != "passed" for report in moment_reports.values()):
        raise ValueError("a parent moment report did not pass")
    if any(report["lensing_or_halo_payload_opened"] for report in moment_reports.values()):
        raise ValueError("a parent moment report opened lensing or halo data")

    cluster_rows: dict[str, list[dict[str, Any]]] = {}
    aggregate_results: dict[str, dict[str, float]] = {}
    total_rows: dict[str, int] = {}
    for cluster, spec in config["cluster_inputs"].items():
        rows, aggregate, row_count = load_cluster_draws(cluster, spec, paths)
        cluster_rows[cluster] = rows
        aggregate_results[cluster] = aggregate
        total_rows[cluster] = row_count

    per_draw_rows = cluster_rows["BULLET"] + cluster_rows["ABELL2146"]
    paired_rows: list[dict[str, Any]] = []
    for bullet, abell in zip(cluster_rows["BULLET"], cluster_rows["ABELL2146"], strict=True):
        if bullet["sample_id"] != abell["sample_id"]:
            raise ValueError("paired sample IDs differ")
        paired_rows.append(
            {
                "sample_id": bullet["sample_id"],
                "abell_minus_bullet_normalized_second_offset": abell["normalized_second_offset"]
                - bullet["normalized_second_offset"],
                "abell_minus_bullet_normalized_current_separation": abell[
                    "normalized_current_separation"
                ]
                - bullet["normalized_current_separation"],
                "abell_minus_bullet_current_axis_alignment_cos2": abell[
                    "current_axis_alignment_cos2"
                ]
                - bullet["current_axis_alignment_cos2"],
            }
        )

    outputs = config["outputs"]
    morphology_path = ROOT / outputs["per_draw_morphology"]
    paired_path = ROOT / outputs["paired_comparison"]
    report_path = ROOT / outputs["report"]
    figure_path = ROOT / outputs["figure"]
    write_csv(morphology_path, per_draw_rows, list(per_draw_rows[0]))
    write_csv(paired_path, paired_rows, list(paired_rows[0]))
    make_figure(figure_path, cluster_rows)

    summary_fields = [
        "finite_luminosity_member_count",
        "luminosity_rms_radius_arcsec",
        "luminosity_axis_ellipticity",
        "second_moment_offset_arcsec",
        "opposite_current_separation_arcsec",
        "normalized_second_offset",
        "normalized_current_separation",
        "current_axis_alignment_cos2",
        "second_current_axis_alignment_abs_cos",
    ]
    paired_fields = [field for field in paired_rows[0] if field != "sample_id"]
    aggregate_errors: dict[str, dict[str, float]] = {}
    for cluster in cluster_rows:
        parent = moment_reports[cluster]["target_blind_morphology"]
        aggregate_errors[cluster] = {
            field: abs(aggregate_results[cluster][field] - float(parent[field]))
            for field in aggregate_results[cluster]
        }
    max_aggregate_error = max(
        value for cluster_errors in aggregate_errors.values() for value in cluster_errors.values()
    )
    mathematical_bounds = all(
        row["luminosity_rms_radius_arcsec"] > 0.0
        and 0.0 <= row["luminosity_axis_ellipticity"] <= 1.0
        and row["normalized_second_offset"] >= 0.0
        and row["normalized_current_separation"] >= 0.0
        and -1.0 <= row["current_axis_alignment_cos2"] <= 1.0
        and 0.0 <= row["second_current_axis_alignment_abs_cos"] <= 1.0
        and all(math.isfinite(float(value)) for key, value in row.items() if key not in {"cluster"})
        for row in per_draw_rows
    )
    exact_counts = all(
        len(cluster_rows[cluster]) == int(spec["expected_draws"])
        and total_rows[cluster]
        == int(spec["expected_draws"]) * int(spec["expected_members_per_draw"])
        for cluster, spec in config["cluster_inputs"].items()
    )
    positive_weights = all(
        row["luminosity_total_within_band"] > 0.0
        and row["second_moment_weight"] > 0.0
        and row["positive_current_weight"] > 0.0
        and row["negative_current_weight"] > 0.0
        for row in per_draw_rows
    )
    gates = config["gates"]
    gate_results = {
        "all_parent_hashes_exact": True,
        "both_parent_moment_reports_passed": True,
        "exact_draw_member_and_row_counts": exact_counts,
        "positive_luminosity_second_and_signed_current_weights": positive_weights,
        "positive_finite_luminosity_rms_radius": all(
            row["luminosity_rms_radius_arcsec"] > 0.0 for row in per_draw_rows
        ),
        "dimensionless_statistics_within_bounds": mathematical_bounds,
        "aggregate_centroid_diagnostics_match_parent_reports": max_aggregate_error
        <= gates["aggregate_centroid_diagnostics_match_parent_reports_arcsec"],
        "no_cross_cluster_photometric_amplitude_comparison": True,
        "no_lensing_halo_gas_response_or_gravity_payload": True,
    }
    gate_results = {name: bool(value) for name, value in gate_results.items()}
    decision = "passed" if all(gate_results.values()) else "failed_closed"
    report = {
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "decision": decision,
        "config": str(config_path.relative_to(ROOT)).replace("\\", "/"),
        "config_sha256": config_hash,
        "implementation": {
            "runner": implementation["runner"],
            "runner_sha256": runner_hash,
        },
        "input_hashes": input_hashes,
        "cluster_row_counts": total_rows,
        "cluster_summaries": {
            cluster: percentile_summary(rows, summary_fields)
            for cluster, rows in cluster_rows.items()
        },
        "paired_comparison_summary": percentile_summary(paired_rows, paired_fields),
        "aggregate_reproduction": {
            "diagnostics": aggregate_results,
            "absolute_errors_arcsec": aggregate_errors,
            "maximum_absolute_error_arcsec": max_aggregate_error,
        },
        "gate_results": gate_results,
        "outputs": {
            "per_draw_morphology": {
                "path": str(morphology_path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(morphology_path),
                "bytes": morphology_path.stat().st_size,
                "rows": len(per_draw_rows),
            },
            "paired_comparison": {
                "path": str(paired_path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(paired_path),
                "bytes": paired_path.stat().st_size,
                "rows": len(paired_rows),
            },
            "figure": {
                "path": str(figure_path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(figure_path),
                "bytes": figure_path.stat().st_size,
            },
        },
        "claim_boundary": config["claim_boundary"],
        "long_wave_operator_or_parameter_selected": False,
        "cross_filter_luminosity_amplitudes_compared": False,
        "missing_luminosity_or_transverse_velocity_imputed": False,
        "lensing_halo_gas_response_or_gravity_payload_opened": False,
        "gravity_formula_or_parameter_changed": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if decision != "passed":
        raise RuntimeError(f"V19BD failed closed: {gate_results}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    report = run(args.config.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
