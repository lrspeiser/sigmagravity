#!/usr/bin/env python3
"""Run the frozen V19DF public component-current identifiability audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import tarfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19df_macsj0018_component_current.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def download_exact(record: dict[str, Any]) -> Path:
    output = ROOT / record["output"]
    if output.is_file():
        if output.stat().st_size != record["bytes"] or sha256(output) != record["sha256"]:
            raise RuntimeError("existing V19DF paper archive does not match the frozen bytes/hash")
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".download")
    request = urllib.request.Request(
        record["url"], headers={"User-Agent": "SigmaGravity-V19DF/1.0 reproducibility audit"}
    )
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
        while block := response.read(1024 * 1024):
            handle.write(block)
    if temporary.stat().st_size != record["bytes"] or sha256(temporary) != record["sha256"]:
        raise RuntimeError("downloaded V19DF paper archive does not match the frozen bytes/hash")
    temporary.replace(output)
    return output


def archive_member(archive_path: Path, name: str, expected_hash: str) -> str:
    with tarfile.open(archive_path, mode="r:*") as archive:
        handle = archive.extractfile(name)
        if handle is None:
            raise RuntimeError(f"missing frozen archive member: {name}")
        payload = handle.read()
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        raise RuntimeError("V19DF TeX member hash changed")
    return payload.decode("utf-8", errors="replace")


def ra_hms_to_deg(value: str) -> float:
    hours, minutes, seconds = (float(item) for item in value.split(":"))
    return 15.0 * (hours + minutes / 60.0 + seconds / 3600.0)


def dec_dms_to_deg(value: str) -> float:
    sign = -1.0 if value.startswith("-") else 1.0
    degrees, minutes, seconds = (float(item) for item in value.lstrip("+-").split(":"))
    return sign * (degrees + minutes / 60.0 + seconds / 3600.0)


def table_body(text: str, label: str) -> str:
    match = re.search(
        rf"\\begin\{{deluxetable\*\}}[^\n]*\\label\{{{re.escape(label)}\}}"
        r".*?\\startdata\s*(.*?)\\enddata",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        raise RuntimeError(f"missing V19DF TeX table {label}")
    return match.group(1)


def angular_offsets(ra_deg: float, dec_deg: float, ra0: float, dec0: float) -> tuple[float, float]:
    return (
        (ra_deg - ra0) * math.cos(math.radians(dec0)) * 3600.0,
        (dec_deg - dec0) * 3600.0,
    )


def parse_table(text: str, label: str, source_group: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in table_body(text, label).splitlines():
        cells = [cell.strip().replace(r"\\", "").strip() for cell in line.split("&")]
        if len(cells) != 5 or re.fullmatch(r"\d{2}:\d{2}:\d{2}\.\d{2}", cells[0]) is None:
            continue
        rows.append(
            {
                "source_group": source_group,
                "ra_hms": cells[0],
                "dec_dms": cells[1],
                "ra_deg": ra_hms_to_deg(cells[0]),
                "dec_deg": dec_dms_to_deg(cells[1]),
                "redshift": float(cells[2]),
                "provenance": cells[3],
                "internally_marked_duplicate": cells[4] == "*",
            }
        )
    return rows


def separation_arcsec(first: dict[str, Any], second: dict[str, Any], dec0: float) -> float:
    dx = (first["ra_deg"] - second["ra_deg"]) * math.cos(math.radians(dec0)) * 3600.0
    dy = (first["dec_deg"] - second["dec_deg"]) * 3600.0
    return math.hypot(dx, dy)


def build_catalog(text: str, config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    definition = config["member_catalog"]
    ra0 = ra_hms_to_deg(definition["center_ra_hms"])
    dec0 = dec_dms_to_deg(definition["center_dec_dms"])
    half_width = float(definition["box_half_width_arcsec"])
    literature = parse_table(text, definition["literature_table_label"], "literature")
    keck = parse_table(text, definition["keck_table_label"], "keck")
    counts = {"literature_all": len(literature), "keck_all": len(keck)}

    inside: list[dict[str, Any]] = []
    for row in literature + keck:
        x, y = angular_offsets(row["ra_deg"], row["dec_deg"], ra0, dec0)
        row["x_arcsec_east"] = x
        row["y_arcsec_north"] = y
        if not row["internally_marked_duplicate"] and abs(x) <= half_width and abs(y) <= half_width:
            inside.append(row)
    counts["inside_box_after_internal_duplicate_removal"] = len(inside)

    cross_matches: list[dict[str, Any]] = []
    removed_literature_ids: set[int] = set()
    threshold = float(definition["cross_table_match_arcsec"])
    for index, row in enumerate(inside):
        if row["source_group"] != "literature":
            continue
        candidates = [
            (separation_arcsec(row, other, dec0), other)
            for other in inside
            if other["source_group"] == "keck"
            and separation_arcsec(row, other, dec0) <= threshold
        ]
        if not candidates:
            continue
        separation, other = min(candidates, key=lambda item: item[0])
        removed_literature_ids.add(index)
        cross_matches.append(
            {
                "separation_arcsec": separation,
                "literature_ra_hms": row["ra_hms"],
                "literature_dec_dms": row["dec_dms"],
                "literature_redshift": row["redshift"],
                "keck_ra_hms": other["ra_hms"],
                "keck_dec_dms": other["dec_dms"],
                "keck_redshift": other["redshift"],
                "retained": "keck",
            }
        )
    catalog = [row for index, row in enumerate(inside) if index not in removed_literature_ids]
    counts["cross_table_matches"] = len(cross_matches)
    counts["final_rows"] = len(catalog)

    velocity = config["velocity_definition"]
    for object_id, row in enumerate(catalog, start=1):
        row["object_id"] = f"MACSJ0018-{object_id:03d}"
        row["velocity_km_s"] = (
            velocity["speed_of_light_km_s"]
            * (row["redshift"] - velocity["cluster_reference_redshift"])
            / (1.0 + velocity["cluster_reference_redshift"])
            + velocity["bulk_offset_km_s"]
        )
    return catalog, cross_matches, counts


def design(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(
        [[1.0, row["x_arcsec_east"], row["y_arcsec_north"]] for row in rows], dtype=float
    )
    velocity = np.asarray([row["velocity_km_s"] for row in rows], dtype=float)
    return matrix, velocity


def plane_summary(coefficients: np.ndarray, matrix: np.ndarray, velocity: np.ndarray, half_width: float) -> dict[str, Any]:
    angle = float(math.degrees(math.atan2(coefficients[2], coefficients[1])) % 360.0)
    corners = np.asarray(
        [[1.0, x, y] for x in (-half_width, half_width) for y in (-half_width, half_width)]
    )
    predictions = corners @ coefficients
    return {
        "intercept_km_s": float(coefficients[0]),
        "east_gradient_km_s_per_arcsec": float(coefficients[1]),
        "north_gradient_km_s_per_arcsec": float(coefficients[2]),
        "gradient_magnitude_km_s_per_arcsec": float(math.hypot(coefficients[1], coefficients[2])),
        "positive_velocity_direction_deg_east_ccw": angle,
        "peak_to_peak_over_frozen_square_km_s": float(np.ptp(predictions)),
        "residual_rmse_km_s": float(np.sqrt(np.mean((velocity - matrix @ coefficients) ** 2))),
    }


def ols_plane(rows: list[dict[str, Any]], half_width: float) -> dict[str, Any]:
    matrix, velocity = design(rows)
    coefficients = np.linalg.lstsq(matrix, velocity, rcond=None)[0]
    return plane_summary(coefficients, matrix, velocity, half_width)


def huber_plane(rows: list[dict[str, Any]], half_width: float) -> dict[str, Any]:
    matrix, velocity = design(rows)
    coefficients = np.linalg.lstsq(matrix, velocity, rcond=None)[0]
    iterations = 0
    for iterations in range(1, 101):
        residual = velocity - matrix @ coefficients
        centered = residual - np.median(residual)
        scale = 1.4826 * float(np.median(np.abs(centered)))
        if not math.isfinite(scale) or scale <= 0:
            break
        ratio = np.abs(residual) / (1.345 * scale)
        weights = np.ones_like(ratio)
        mask = ratio > 1.0
        weights[mask] = 1.0 / ratio[mask]
        weighted_matrix = matrix * np.sqrt(weights)[:, None]
        updated = np.linalg.lstsq(weighted_matrix, velocity * np.sqrt(weights), rcond=None)[0]
        if float(np.linalg.norm(updated - coefficients)) < 1e-9:
            coefficients = updated
            break
        coefficients = updated
    summary = plane_summary(coefficients, matrix, velocity, half_width)
    summary["iterations"] = iterations
    return summary


def signed_angle_difference(first: float, second: float) -> float:
    return float((first - second + 180.0) % 360.0 - 180.0)


def axial_difference(first: float, second: float) -> float:
    signed = abs(signed_angle_difference(first, second))
    return float(min(signed, 180.0 - signed))


def resampling(rows: list[dict[str, Any]], config: dict[str, Any], nominal: dict[str, Any]) -> dict[str, Any]:
    diagnostic = config["registered_diagnostics"]
    half_width = float(config["member_catalog"]["box_half_width_arcsec"])
    bootstrap_rng = np.random.default_rng(int(diagnostic["bootstrap_seed"]))
    bootstrap_angles: list[float] = []
    bootstrap_amplitudes: list[float] = []
    for _ in range(int(diagnostic["bootstrap_draws"])):
        indices = bootstrap_rng.integers(0, len(rows), len(rows))
        sample = [rows[int(index)] for index in indices]
        fitted = ols_plane(sample, half_width)
        bootstrap_angles.append(fitted["positive_velocity_direction_deg_east_ccw"])
        bootstrap_amplitudes.append(fitted["peak_to_peak_over_frozen_square_km_s"])
    nominal_angle = nominal["positive_velocity_direction_deg_east_ccw"]
    differences = np.asarray(
        [signed_angle_difference(angle, nominal_angle) for angle in bootstrap_angles], dtype=float
    )

    matrix, velocity = design(rows)
    observed = nominal["gradient_magnitude_km_s_per_arcsec"]
    permutation_rng = np.random.default_rng(int(diagnostic["permutation_seed"]))
    exceedances = 0
    for _ in range(int(diagnostic["permutation_draws"])):
        coefficients = np.linalg.lstsq(matrix, permutation_rng.permutation(velocity), rcond=None)[0]
        exceedances += math.hypot(coefficients[1], coefficients[2]) >= observed
    permutation_p = (exceedances + 1.0) / (int(diagnostic["permutation_draws"]) + 1.0)
    return {
        "bootstrap_direction_difference_deg_quantiles": {
            key: float(value)
            for key, value in zip(
                ("q2p5", "q16", "q50", "q84", "q97p5"),
                np.percentile(differences, [2.5, 16.0, 50.0, 84.0, 97.5]),
                strict=True,
            )
        },
        "bootstrap_peak_to_peak_km_s_quantiles": {
            key: float(value)
            for key, value in zip(
                ("q2p5", "q16", "q50", "q84", "q97p5"),
                np.percentile(bootstrap_amplitudes, [2.5, 16.0, 50.0, 84.0, 97.5]),
                strict=True,
            )
        },
        "permutation_exceedances": int(exceedances),
        "permutation_p": float(permutation_p),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in columns} for row in rows)


def diagnostic_plot(path: Path, rows: list[dict[str, Any]], planes: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    x = np.asarray([row["x_arcsec_east"] for row in rows])
    y = np.asarray([row["y_arcsec_north"] for row in rows])
    velocity = np.asarray([row["velocity_km_s"] for row in rows])
    scatter = axis.scatter(x, y, c=velocity, cmap="coolwarm", s=35, edgecolor="black", linewidth=0.25)
    colors = {"ols": "black", "huber": "#21a179"}
    for label in ("ols", "huber"):
        angle = math.radians(planes[label]["positive_velocity_direction_deg_east_ccw"])
        axis.arrow(0, 0, 110 * math.cos(angle), 110 * math.sin(angle), width=2.5, color=colors[label], label=label)
    axis.set(xlabel="east offset (arcsec)", ylabel="north offset (arcsec)", title="MACS J0018 public member velocities")
    axis.set_aspect("equal")
    axis.legend(loc="lower left")
    figure.colorbar(scatter, ax=axis, label="line-of-sight velocity (km/s)")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def validate_frozen(config: dict[str, Any], config_path: Path) -> None:
    if config["freeze_state"] != "frozen_after_public_availability_audit_and_development_payload_exposure_before_any_v19df_map_or_source_decision":
        raise RuntimeError("V19DF protocol is not in the frozen state")
    implementation = config["implementation"]
    runner_path = Path(__file__).resolve()
    if implementation["runner"] != runner_path.relative_to(ROOT).as_posix():
        raise RuntimeError("V19DF config names a different runner")
    if implementation["runner_sha256"] != sha256(runner_path):
        raise RuntimeError("V19DF runner changed after freeze")
    authorization = config["authorization"]
    if not (
        authorization["download_exact_arxiv_source"]
        and authorization["parse_named_member_tables"]
        and authorization["construct_registered_member_velocity_diagnostics"]
        and not authorization["open_or_digitize_figure_pixels"]
        and not authorization["open_unreleased_ksz_or_noise_pixels"]
        and not authorization["use_lensing_halo_or_dark_matter_map"]
        and not authorization["fit_or_change_gravity_formula_or_constants"]
        and not authorization["derive_or_select_covariant_action"]
        and not authorization["open_validation_or_holdout_system"]
    ):
        raise RuntimeError("V19DF authorization boundary is open")
    if not config_path.is_file():
        raise RuntimeError("V19DF config is missing")


def execute(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_frozen(config, config_path)
    archive_path = download_exact(config["paper_source"])
    text = archive_member(
        archive_path,
        config["paper_source"]["tex_member"],
        config["paper_source"]["tex_sha256"],
    )
    rows, cross_matches, counts = build_catalog(text, config)
    expected = config["member_catalog"]["expected_table_rows"]
    if counts != expected:
        raise RuntimeError(f"V19DF catalog counts changed: {counts} != {expected}")

    catalog_path = ROOT / config["member_catalog"]["output"]
    write_csv(
        catalog_path,
        rows,
        [
            "object_id", "source_group", "ra_hms", "dec_dms", "ra_deg", "dec_deg",
            "x_arcsec_east", "y_arcsec_north", "redshift", "velocity_km_s", "provenance",
        ],
    )
    cross_path = ROOT / config["outputs"]["cross_matches"]
    write_csv(
        cross_path,
        cross_matches,
        [
            "separation_arcsec", "literature_ra_hms", "literature_dec_dms", "literature_redshift",
            "keck_ra_hms", "keck_dec_dms", "keck_redshift", "retained",
        ],
    )

    half_width = float(config["member_catalog"]["box_half_width_arcsec"])
    ols = ols_plane(rows, half_width)
    huber = huber_plane(rows, half_width)
    literature = ols_plane([row for row in rows if row["source_group"] == "literature"], half_width)
    keck = ols_plane([row for row in rows if row["source_group"] == "keck"], half_width)
    samples = resampling(rows, config, ols)
    bootstrap = samples["bootstrap_direction_difference_deg_quantiles"]
    gates_config = config["member_gradient_gates"]
    gates = {
        "catalog_reproduces_156_rows": len(rows) == 156,
        "permutation_p_at_most_0p05": samples["permutation_p"] <= gates_config["permutation_p_max"],
        "ols_huber_direction_difference_at_most_15_deg": axial_difference(
            ols["positive_velocity_direction_deg_east_ccw"], huber["positive_velocity_direction_deg_east_ccw"]
        ) <= gates_config["ols_huber_direction_difference_deg_max"],
        "literature_keck_axial_difference_at_most_30_deg": axial_difference(
            literature["positive_velocity_direction_deg_east_ccw"], keck["positive_velocity_direction_deg_east_ccw"]
        ) <= gates_config["literature_keck_axial_difference_deg_max"],
        "bootstrap_95_direction_half_width_at_most_45_deg": max(abs(bootstrap["q2p5"]), abs(bootstrap["q97p5"]))
        <= gates_config["bootstrap_95_direction_half_width_deg_max"],
    }
    member_admitted = all(gates.values())
    gas_available = bool(config["ksz_gas_branch"]["raw_branch_authorized"])
    plot_path = ROOT / config["outputs"]["plot"]
    diagnostic_plot(plot_path, rows, {"ols": ols, "huber": huber})
    report = {
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "status": "macsj0018_public_component_current_not_admitted",
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "input": {
            "archive": archive_path.relative_to(ROOT).as_posix(),
            "archive_bytes": archive_path.stat().st_size,
            "archive_sha256": sha256(archive_path),
            "tex_member": config["paper_source"]["tex_member"],
            "tex_sha256": config["paper_source"]["tex_sha256"],
        },
        "catalog": {
            "counts": counts,
            "source_group_counts": {
                "literature": sum(row["source_group"] == "literature" for row in rows),
                "keck": sum(row["source_group"] == "keck" for row in rows),
            },
            "output": catalog_path.relative_to(ROOT).as_posix(),
            "output_sha256": sha256(catalog_path),
            "cross_matches": cross_path.relative_to(ROOT).as_posix(),
            "cross_matches_sha256": sha256(cross_path),
            "redshift_uncertainties_invented": False,
            "stellar_mass_or_luminosity_weights_used": False,
        },
        "diagnostics": {
            "ols": ols,
            "huber": huber,
            "literature_only_ols": literature,
            "keck_only_ols": keck,
            "ols_huber_axial_difference_deg": axial_difference(
                ols["positive_velocity_direction_deg_east_ccw"], huber["positive_velocity_direction_deg_east_ccw"]
            ),
            "literature_keck_axial_difference_deg": axial_difference(
                literature["positive_velocity_direction_deg_east_ccw"], keck["positive_velocity_direction_deg_east_ccw"]
            ),
            **samples,
        },
        "gates": gates,
        "member_velocity_gradient_admitted": member_admitted,
        "analysis_grade_ksz_gas_products_publicly_available": gas_available,
        "component_resolved_current_source_admitted": member_admitted and gas_available,
        "published_gas_galaxy_misalignment_used_as_primary_score": False,
        "published_summary_diagnostic": config["ksz_gas_branch"]["published_diagnostics_not_scored"],
        "plot": plot_path.relative_to(ROOT).as_posix(),
        "plot_sha256": sha256(plot_path),
        "figure_pixels_digitized": False,
        "ksz_or_noise_pixels_opened": False,
        "lensing_halo_or_dark_matter_map_opened": False,
        "gravity_formula_or_constant_fit": False,
        "covariant_action_selected_or_derived": False,
        "validation_or_holdout_opened": False,
        "scientific_disposition": "The public member catalog alone does not identify a stable global directional source under the registered reconstruction, and the analysis-grade gas kSZ maps/covariance are not public. Keep gas and galaxy stress-energy components separate in future actions, but do not calibrate a current coupling from this system.",
        "next_gate": "obtain untouched analysis-grade multi-frequency kSZ maps plus covariance, or select a genuinely public independent direct-velocity system and freeze its source protocol before pixel access",
    }
    atomic_json(ROOT / config["outputs"]["report"], report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    report = execute(args.config)
    print(json.dumps({key: report[key] for key in ("status", "member_velocity_gradient_admitted", "component_resolved_current_source_admitted")}, indent=2))


if __name__ == "__main__":
    main()
