from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any
import warnings

from astropy.io import fits
from astropy.wcs import FITSFixedWarning, WCS
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19bz_hi_optical_spatial_information_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def finite_float(value: str, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"non-numeric {field}: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"non-finite {field}: {value!r}")
    return result


def format_float(value: float) -> str:
    if math.isinf(value):
        return "inf"
    if math.isnan(value):
        return ""
    return f"{value:.12g}"


def kernel_label(multiplier: float) -> str:
    return f"k{multiplier:g}".replace(".", "p")


def pixel_scale_arcsec(wcs: WCS) -> float:
    scale = math.sqrt(abs(float(np.linalg.det(wcs.pixel_scale_matrix)))) * 3600.0
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("invalid celestial pixel scale")
    return scale


def score_candidate_positions(
    data: np.ndarray,
    header: fits.Header,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    kernel_multipliers: list[float],
) -> dict[str, Any]:
    image = np.asarray(data, dtype=np.float64)
    if image.ndim != 2 or min(image.shape) <= 0 or not np.isfinite(image).all():
        raise ValueError("moment-zero image must be a finite nonempty 2D array")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        wcs = WCS(header).celestial
    scale_arcsec = pixel_scale_arcsec(wcs)
    bmaj_deg = finite_float(str(header.get("BMAJ", "")), "BMAJ")
    bmin_deg = finite_float(str(header.get("BMIN", "")), "BMIN")
    if bmaj_deg <= 0 or bmin_deg <= 0:
        raise ValueError("FITS beam axes must be positive")
    beam_arcsec = math.sqrt(bmaj_deg * bmin_deg) * 3600.0
    beam_pixels = beam_arcsec / scale_arcsec

    x, y = wcs.world_to_pixel_values(ra_deg, dec_deg)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    inside = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0.0)
        & (x <= image.shape[1] - 1)
        & (y >= 0.0)
        & (y <= image.shape[0] - 1)
    )
    coordinates = np.vstack([y, x])

    support_distance_pixels = distance_transform_edt(image == 0.0)
    support_distance_beams = np.full(len(ra_deg), np.inf, dtype=np.float64)
    if np.any(inside):
        support_distance_beams[inside] = map_coordinates(
            support_distance_pixels,
            coordinates[:, inside],
            order=1,
            mode="constant",
            cval=float("inf"),
        ) / beam_pixels

    positive = np.clip(image, 0.0, None)
    if float(positive.sum()) <= 0.0:
        raise ValueError("moment-zero image has no positive signal")
    likelihood_ratios: dict[float, np.ndarray] = {}
    for multiplier in kernel_multipliers:
        if multiplier < 0:
            raise ValueError("kernel multiplier cannot be negative")
        if multiplier == 0:
            density = positive.copy()
        else:
            sigma_pixels = multiplier * beam_pixels / 2.35482004503
            density = gaussian_filter(positive, sigma=sigma_pixels, mode="constant")
        density_sum = float(density.sum())
        if density_sum <= 0 or not math.isfinite(density_sum):
            raise ValueError("invalid normalized spatial density")
        density /= density_sum
        ratios = np.zeros(len(ra_deg), dtype=np.float64)
        if np.any(inside):
            ratios[inside] = map_coordinates(
                density,
                coordinates[:, inside],
                order=1,
                mode="constant",
                cval=0.0,
            ) * image.size
        likelihood_ratios[multiplier] = ratios
    return {
        "inside": inside,
        "support_distance_beams": support_distance_beams,
        "likelihood_ratios": likelihood_ratios,
        "pixel_scale_arcsec": scale_arcsec,
        "beam_fwhm_arcsec": beam_arcsec,
    }


def ranked_indices(values: np.ndarray, object_ids: list[str]) -> list[int]:
    return sorted(range(len(values)), key=lambda index: (-float(values[index]), object_ids[index]))


def top_margin(values: np.ndarray, order: list[int]) -> float:
    if not order:
        raise ValueError("cannot rank an empty candidate set")
    if len(order) == 1:
        return float("inf")
    top = float(values[order[0]])
    second = float(values[order[1]])
    if second <= 0:
        return float("inf") if top > 0 else 1.0
    return top / second


def field_name(team_release: str) -> str:
    for suffix in (" TR1", " TR2"):
        if team_release.endswith(suffix):
            return team_release[: -len(suffix)]
    raise ValueError(f"unsupported release label: {team_release}")


def median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def run(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_json(config_path)
    if not config["honesty_boundary"]["source_data_were_inspected_before_this_audit_was_frozen"]:
        raise RuntimeError("V19BZ must not claim prospective source preregistration")
    if config["honesty_boundary"]["gravity_or_kinematic_target_was_inspected"]:
        raise RuntimeError("V19BZ access boundary claims target access")

    for section in ("parents", "inputs"):
        for item in config[section].values():
            path = ROOT / item["path"]
            if not path.is_file() or sha256(path) != item["sha256"]:
                raise RuntimeError(f"V19BZ {section[:-1]} hash mismatch: {item['path']}")

    _, manifest = load_csv(ROOT / config["inputs"]["moment0_manifest"]["path"])
    candidate_fields, candidates = load_csv(
        ROOT / config["inputs"]["skymapper_candidates"]["path"]
    )
    if len(manifest) != int(config["inputs"]["moment0_manifest"]["rows"]):
        raise RuntimeError("unexpected moment-zero manifest row count")
    if len(candidates) != int(config["inputs"]["skymapper_candidates"]["rows"]):
        raise RuntimeError("unexpected SkyMapper candidate row count")
    required_candidate_fields = {
        "wallaby_name",
        "object_id",
        "raj2000",
        "dej2000",
        "separation_arcsec",
        "extended_candidate",
    }
    if not required_candidate_fields.issubset(candidate_fields):
        raise RuntimeError("SkyMapper candidate projection is incomplete")

    candidates_by_name: dict[str, list[dict[str, str]]] = defaultdict(list)
    for candidate in candidates:
        candidates_by_name[candidate["wallaby_name"]].append(candidate)
    if len(candidates_by_name) != int(config["inputs"]["skymapper_candidates"]["wallaby_names"]):
        raise RuntimeError("unexpected WALLABY name coverage in candidate table")

    multipliers = [float(value) for value in config["spatial_information_model"]["kernel_fwhm_multipliers"]]
    primary = float(config["spatial_information_model"]["primary_kernel_fwhm_multiplier"])
    if primary not in multipliers or len(set(multipliers)) != len(multipliers):
        raise RuntimeError("invalid V19BZ kernel branch declaration")
    labels = {multiplier: kernel_label(multiplier) for multiplier in multipliers}

    candidate_output: list[dict[str, Any]] = []
    release_output: list[dict[str, Any]] = []
    for map_row in manifest:
        map_path = ROOT / map_row["local_path"]
        if not map_path.is_file() or sha256(map_path) != map_row["local_sha256"]:
            raise RuntimeError(f"moment-zero map hash mismatch: {map_row['local_path']}")
        source_candidates = candidates_by_name.get(map_row["name"], [])
        if not source_candidates:
            raise RuntimeError(f"no candidates for {map_row['name']}")
        ra = np.asarray(
            [finite_float(row["raj2000"], "raj2000") for row in source_candidates],
            dtype=np.float64,
        )
        dec = np.asarray(
            [finite_float(row["dej2000"], "dej2000") for row in source_candidates],
            dtype=np.float64,
        )
        with fits.open(map_path, mode="readonly", memmap=False) as hdul:
            scored = score_candidate_positions(hdul[0].data, hdul[0].header, ra, dec, multipliers)

        object_ids = [row["object_id"] for row in source_candidates]
        orders = {
            multiplier: ranked_indices(scored["likelihood_ratios"][multiplier], object_ids)
            for multiplier in multipliers
        }
        ranks = {
            multiplier: {index: rank + 1 for rank, index in enumerate(orders[multiplier])}
            for multiplier in multipliers
        }
        top_ids = {multiplier: object_ids[orders[multiplier][0]] for multiplier in multipliers}
        margins = {
            multiplier: top_margin(scored["likelihood_ratios"][multiplier], orders[multiplier])
            for multiplier in multipliers
        }
        primary_order = orders[primary]
        primary_top = primary_order[0]
        same_top = len(set(top_ids.values())) == 1
        minimum_margin = min(margins.values())

        for index, candidate in enumerate(source_candidates):
            output_row: dict[str, Any] = {
                "source_row_id": map_row["source_row_id"],
                "wallaby_name": map_row["name"],
                "team_release": map_row["team_release"],
                "object_id": candidate["object_id"],
                "separation_arcsec": candidate["separation_arcsec"],
                "inside_map": str(bool(scored["inside"][index])).lower(),
                "distance_to_nonzero_support_beams": format_float(
                    float(scored["support_distance_beams"][index])
                ),
                "extended_candidate_diagnostic": candidate["extended_candidate"],
            }
            for multiplier in multipliers:
                label = labels[multiplier]
                output_row[f"spatial_lr_{label}"] = format_float(
                    float(scored["likelihood_ratios"][multiplier][index])
                )
                output_row[f"spatial_rank_{label}"] = ranks[multiplier][index]
            candidate_output.append(output_row)

        release_row: dict[str, Any] = {
            "source_row_id": map_row["source_row_id"],
            "wallaby_name": map_row["name"],
            "team_release": map_row["team_release"],
            "field": field_name(map_row["team_release"]),
            "candidate_count": len(source_candidates),
            "inside_map_count": int(np.sum(scored["inside"])),
            "within_one_beam_of_support_count": int(
                np.sum(scored["support_distance_beams"] <= 1.0)
            ),
            "primary_top_object_id": object_ids[primary_top],
            "primary_top_extended_diagnostic": source_candidates[primary_top]["extended_candidate"],
            "primary_top_separation_arcsec": source_candidates[primary_top]["separation_arcsec"],
            "same_top_all_kernel_branches": str(same_top).lower(),
            "minimum_top_to_second_margin": format_float(minimum_margin),
            "robust_margin_ge_3": str(same_top and minimum_margin >= 3.0).lower(),
            "pixel_scale_arcsec": format_float(scored["pixel_scale_arcsec"]),
            "beam_fwhm_arcsec": format_float(scored["beam_fwhm_arcsec"]),
        }
        for multiplier in multipliers:
            label = labels[multiplier]
            release_row[f"top_object_id_{label}"] = top_ids[multiplier]
            release_row[f"top_margin_{label}"] = format_float(margins[multiplier])
        release_output.append(release_row)

    candidate_fields_out = [
        "source_row_id",
        "wallaby_name",
        "team_release",
        "object_id",
        "separation_arcsec",
        "inside_map",
        "distance_to_nonzero_support_beams",
        "extended_candidate_diagnostic",
    ]
    for multiplier in multipliers:
        candidate_fields_out.extend(
            [f"spatial_lr_{labels[multiplier]}", f"spatial_rank_{labels[multiplier]}"]
        )
    release_fields_out = list(release_output[0])

    outputs = config["outputs"]
    candidate_path = ROOT / outputs["candidate_scores"]
    release_path = ROOT / outputs["release_information"]
    report_path = ROOT / outputs["report"]
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    release_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_bytes(csv_bytes(candidate_output, candidate_fields_out))
    release_path.write_bytes(csv_bytes(release_output, release_fields_out))

    margins_primary = [float(row[f"top_margin_{labels[primary]}"]) for row in release_output]
    minimum_margins = [float(row["minimum_top_to_second_margin"]) for row in release_output]
    same_kernel_top_count = sum(row["same_top_all_kernel_branches"] == "true" for row in release_output)
    robust_count = sum(row["robust_margin_ge_3"] == "true" for row in release_output)
    release_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in release_output:
        release_by_name[row["wallaby_name"]].append(row)
    duplicate_groups = [rows for rows in release_by_name.values() if len(rows) > 1]
    duplicate_stable = sum(
        len(
            {
                row[f"top_object_id_{labels[multiplier]}"]
                for row in rows
                for multiplier in multipliers
            }
        )
        == 1
        for rows in duplicate_groups
    )
    duplicate_stability_fraction = duplicate_stable / len(duplicate_groups)
    robust_fraction = robust_count / len(release_output)

    field_summary: dict[str, Any] = {}
    for field in sorted({row["field"] for row in release_output}):
        rows = [row for row in release_output if row["field"] == field]
        field_summary[field] = {
            "release_rows": len(rows),
            "median_candidates": median([float(row["candidate_count"]) for row in rows]),
            "median_primary_top_margin": median(
                [float(row[f"top_margin_{labels[primary]}"]) for row in rows]
            ),
            "same_top_all_kernel_branches": sum(
                row["same_top_all_kernel_branches"] == "true" for row in rows
            ),
            "robust_margin_ge_3": sum(row["robust_margin_ge_3"] == "true" for row in rows),
        }

    margin_grid = [float(value) for value in config["descriptive_margin_grid"]]
    sufficiency = config["exploratory_information_sufficiency_rule"]
    information_sufficient = (
        robust_fraction >= float(sufficiency["minimum_fraction_of_release_rows_meeting_robust_margin"])
        and duplicate_stability_fraction
        >= float(sufficiency["minimum_duplicate_release_top_identity_stability"])
    )
    boundary = dict(config["access_boundary"])
    gates = {
        "parent_and_input_hashes_exact": True,
        "all_711_release_maps_scored": len(release_output) == 711,
        "every_candidate_retained_for_every_release_alternative": len(candidate_output)
        == sum(len(candidates_by_name[row["name"]]) for row in manifest),
        "all_four_kernel_branches_reported": len(multipliers) == 4
        and all(
            f"spatial_lr_{labels[multiplier]}" in candidate_fields_out
            for multiplier in multipliers
        ),
        "no_optical_weight_prior_posterior_or_hard_assignment": not boundary[
            "skymapper_extendedness_used_as_weight"
        ]
        and not boundary["hard_counterpart_selected"]
        and not boundary["candidate_removed"],
        "kinematic_gravity_lensing_and_solar_targets_remain_sealed": not any(
            boundary[key]
            for key in (
                "wallaby_kinematic_table_row_read",
                "rotation_speed_or_velocity_field_read",
                "gravity_formula_residual_or_halo_result_read",
                "development_validation_holdout_split_selected",
                "gravity_action_or_constant_changed",
                "lensing_payload_opened",
                "solar_system_optimization_performed",
            )
        ),
        "exploratory_status_reported_honestly": config["honesty_boundary"][
            "source_data_were_inspected_before_this_audit_was_frozen"
        ]
        and not config["honesty_boundary"]["this_is_a_preregistered_theory_or_holdout_gate"],
    }
    report = {
        "protocol_version": config["protocol_version"],
        "status": "completed_exploratory_source_only_spatial_information_audit",
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "input_audit": {
            "release_maps": len(manifest),
            "unique_wallaby_names": len(candidates_by_name),
            "unique_candidate_rows": len(candidates),
            "candidate_release_pair_rows": len(candidate_output),
            "release_counts": dict(sorted(Counter(row["team_release"] for row in manifest).items())),
        },
        "spatial_model": config["spatial_information_model"],
        "information_audit": {
            "primary_kernel_fwhm_multiplier": primary,
            "median_primary_top_margin": median(margins_primary),
            "median_minimum_margin_across_kernels": median(minimum_margins),
            "same_top_all_kernel_branches": same_kernel_top_count,
            "robust_margin_ge_3": robust_count,
            "robust_margin_fraction": robust_fraction,
            "primary_margin_grid_counts": {
                format_float(threshold): sum(value >= threshold for value in margins_primary)
                for threshold in margin_grid
            },
            "duplicate_release_names": len(duplicate_groups),
            "duplicate_top_stable_all_releases_and_kernels": duplicate_stable,
            "duplicate_top_stability_fraction": duplicate_stability_fraction,
            "field_summary": field_summary,
        },
        "exploratory_information_sufficiency_rule": sufficiency,
        "information_sufficient_for_hard_counterpart": information_sufficient,
        "decision": (
            "source_spatial_information_supports_future_preregistered_association_gate"
            if information_sufficient
            else "source_spatial_information_insufficient_for_hard_counterpart"
        ),
        "next_evidence_if_insufficient": [
            "uniform optical image cutouts centered on every H I source",
            "foreground-star masks and source-deblending uncertainty",
            "a source-only probabilistic mixture that retains ambiguous counterparts",
            "an independent association-validation subset that contains no rotation or gravity target",
        ],
        "outputs": {
            "candidate_scores": {
                "path": outputs["candidate_scores"],
                "rows": len(candidate_output),
                "sha256": sha256(candidate_path),
                "bytes": candidate_path.stat().st_size,
            },
            "release_information": {
                "path": outputs["release_information"],
                "rows": len(release_output),
                "sha256": sha256(release_path),
                "bytes": release_path.stat().st_size,
            },
        },
        "access_boundary_audit": boundary,
        "gate_results": gates,
        "claim_boundary": config["claim_boundary"],
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not all(gates.values()):
        raise SystemExit(1)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    report = run(args.config)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "information_audit": report["information_audit"],
                "gate_results": report["gate_results"],
                "outputs": report["outputs"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
