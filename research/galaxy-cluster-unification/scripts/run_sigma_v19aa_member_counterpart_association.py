#!/usr/bin/env python3
"""Associate V19 spectroscopic members with HSC/NSC candidates.

The primary likelihood uses only the published-coordinate quantization and
catalog astrometry.  Photometry, morphology, and proper motion are retained as
diagnostics; no stellar mass or gravity quantity is inferred here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19aa_member_counterpart_association.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def finite_float(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def angular_separation_arcsec(
    ra1_deg: float, dec1_deg: float, ra2_deg: float, dec2_deg: float
) -> float:
    ra1 = math.radians(ra1_deg)
    dec1 = math.radians(dec1_deg)
    ra2 = math.radians(ra2_deg)
    dec2 = math.radians(dec2_deg)
    delta_ra = (ra2 - ra1 + math.pi) % (2.0 * math.pi) - math.pi
    delta_dec = dec2 - dec1
    haversine = (
        math.sin(delta_dec / 2.0) ** 2
        + math.cos(dec1) * math.cos(dec2) * math.sin(delta_ra / 2.0) ** 2
    )
    return math.degrees(2.0 * math.asin(min(1.0, math.sqrt(max(0.0, haversine))))) * 3600.0


def tangent_offsets_arcsec(
    center_ra_deg: float,
    center_dec_deg: float,
    candidate_ra_deg: float,
    candidate_dec_deg: float,
) -> tuple[float, float]:
    delta_ra = (candidate_ra_deg - center_ra_deg + 180.0) % 360.0 - 180.0
    east = delta_ra * math.cos(math.radians(center_dec_deg)) * 3600.0
    north = (candidate_dec_deg - center_dec_deg) * 3600.0
    return east, north


def quantized_axis_pdf(offset_arcsec: float, half_width_arcsec: float, sigma_arcsec: float) -> float:
    """Uniform rounding bin convolved with a Gaussian astrometric error."""
    if half_width_arcsec <= 0.0 or sigma_arcsec <= 0.0:
        raise ValueError("quantization half-width and sigma must be positive")
    upper = ndtr((half_width_arcsec - offset_arcsec) / sigma_arcsec)
    lower = ndtr((-half_width_arcsec - offset_arcsec) / sigma_arcsec)
    return max(float((upper - lower) / (2.0 * half_width_arcsec)), np.finfo(float).tiny)


def quantized_position_pdf_arcsec2(
    east_arcsec: float,
    north_arcsec: float,
    *,
    east_half_width_arcsec: float,
    north_half_width_arcsec: float,
    east_sigma_arcsec: float,
    north_sigma_arcsec: float,
) -> float:
    return quantized_axis_pdf(east_arcsec, east_half_width_arcsec, east_sigma_arcsec) * quantized_axis_pdf(
        north_arcsec, north_half_width_arcsec, north_sigma_arcsec
    )


def association_posterior(likelihood_ratios: Iterable[float], counterpart_prior: float) -> tuple[np.ndarray, float]:
    ratios = np.asarray(list(likelihood_ratios), dtype=float)
    if np.any(~np.isfinite(ratios)) or np.any(ratios < 0.0):
        raise ValueError("likelihood ratios must be finite and nonnegative")
    if not 0.0 < counterpart_prior < 1.0:
        raise ValueError("counterpart prior must lie strictly between zero and one")
    denominator = (1.0 - counterpart_prior) + counterpart_prior * float(np.sum(ratios))
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise ValueError("invalid posterior denominator")
    return counterpart_prior * ratios / denominator, (1.0 - counterpart_prior) / denominator


@dataclass(frozen=True)
class Detection:
    survey: str
    cluster: str
    survey_id: str
    ra_deg: float
    dec_deg: float
    sigma_arcsec: float
    payload: dict[str, str]

    @property
    def key(self) -> str:
        return f"{self.survey}:{self.survey_id}"


@dataclass(frozen=True)
class UnifiedCandidate:
    cluster: str
    candidate_id: str
    ra_deg: float
    dec_deg: float
    sigma_arcsec: float
    hsc: Detection | None
    nsc: Detection | None
    cross_survey_separation_arcsec: float | None

    @property
    def dual_survey(self) -> bool:
        return self.hsc is not None and self.nsc is not None


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_members(path: Path, expected_rows: int) -> list[dict[str, str]]:
    rows = read_csv_rows(path)
    if len(rows) != expected_rows:
        raise RuntimeError(f"member count mismatch for {path}: {len(rows)} != {expected_rows}")
    keys = [(row["cluster"], row["object_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"duplicate member key in {path}")
    return rows


def _register_detection(
    store: dict[str, Detection], detection: Detection, *, tolerance_arcsec: float = 1.0e-4
) -> None:
    prior = store.get(detection.key)
    if prior is None:
        store[detection.key] = detection
        return
    if prior.cluster != detection.cluster:
        raise RuntimeError(f"survey ID crosses clusters: {detection.key}")
    if angular_separation_arcsec(prior.ra_deg, prior.dec_deg, detection.ra_deg, detection.dec_deg) > tolerance_arcsec:
        raise RuntimeError(f"inconsistent duplicate survey position: {detection.key}")


def load_survey_detections(
    report: dict[str, Any],
    *,
    survey: str,
    astrometric_floor_arcsec: float,
) -> tuple[dict[str, Detection], dict[tuple[str, str], set[str]]]:
    detections: dict[str, Detection] = {}
    member_ids: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in report["records"]:
        path = ROOT / record["csv_path"]
        if sha256(path) != record["csv_sha256"]:
            raise RuntimeError(f"raw {survey} payload hash mismatch: {path}")
        rows = read_csv_rows(path)
        if len(rows) != int(record["candidate_rows"]):
            raise RuntimeError(f"raw {survey} row count mismatch: {path}")
        member_key = (str(record["cluster"]), str(record["object_id"]))
        for row in rows:
            if survey == "HSC":
                survey_id = str(row["MatchID"])
                ra_deg = float(row["MatchRA"])
                dec_deg = float(row["MatchDec"])
                dsigma_mas = finite_float(row.get("DSigma"))
                sigma = astrometric_floor_arcsec
                if dsigma_mas is not None and dsigma_mas > 0.0:
                    sigma = max(sigma, dsigma_mas / 1000.0)
            elif survey == "NSC":
                survey_id = str(row["id"])
                ra_deg = float(row["ra"])
                dec_deg = float(row["dec"])
                ra_sigma = finite_float(row.get("raerr"))
                dec_sigma = finite_float(row.get("decerr"))
                supplied = [value for value in (ra_sigma, dec_sigma) if value is not None and value > 0.0]
                sigma = max(astrometric_floor_arcsec, float(np.mean(supplied)) if supplied else 0.0)
            else:
                raise ValueError(survey)
            if not all(math.isfinite(value) for value in (ra_deg, dec_deg, sigma)):
                raise RuntimeError(f"non-finite {survey} astrometry in {path}")
            detection = Detection(survey, member_key[0], survey_id, ra_deg, dec_deg, sigma, row)
            _register_detection(detections, detection)
            member_ids[member_key].add(detection.key)
    return detections, member_ids


def reciprocal_crossmatches(
    hsc: dict[str, Detection],
    nsc: dict[str, Detection],
    *,
    radius_arcsec: float,
) -> list[tuple[str, str, float]]:
    if radius_arcsec <= 0.0:
        raise ValueError("crossmatch radius must be positive")
    pairs: list[tuple[str, str, float]] = []
    clusters = sorted({row.cluster for row in hsc.values()} | {row.cluster for row in nsc.values()})
    for cluster in clusters:
        left = sorted((row for row in hsc.values() if row.cluster == cluster), key=lambda row: row.key)
        right = sorted((row for row in nsc.values() if row.cluster == cluster), key=lambda row: row.key)
        if not left or not right:
            continue
        distances = np.asarray(
            [
                [angular_separation_arcsec(a.ra_deg, a.dec_deg, b.ra_deg, b.dec_deg) for b in right]
                for a in left
            ],
            dtype=float,
        )
        nearest_right = np.argmin(distances, axis=1)
        nearest_left = np.argmin(distances, axis=0)
        for left_index, right_index in enumerate(nearest_right):
            if nearest_left[right_index] != left_index:
                continue
            separation = float(distances[left_index, right_index])
            if separation <= radius_arcsec:
                pairs.append((left[left_index].key, right[right_index].key, separation))
    return pairs


def _weighted_coordinate(a: Detection, b: Detection) -> tuple[float, float, float]:
    weights = np.asarray([1.0 / a.sigma_arcsec**2, 1.0 / b.sigma_arcsec**2])
    reference = a.ra_deg
    delta = (b.ra_deg - reference + 180.0) % 360.0 - 180.0
    ra = (reference + weights[1] * delta / float(np.sum(weights))) % 360.0
    dec = float(np.average([a.dec_deg, b.dec_deg], weights=weights))
    sigma = float(1.0 / math.sqrt(float(np.sum(weights))))
    return ra, dec, sigma


def build_unified_candidates(
    hsc: dict[str, Detection],
    nsc: dict[str, Detection],
    *,
    crossmatch_radius_arcsec: float,
) -> tuple[dict[str, UnifiedCandidate], dict[str, str], list[tuple[str, str, float]]]:
    matches = reciprocal_crossmatches(hsc, nsc, radius_arcsec=crossmatch_radius_arcsec)
    hsc_to_nsc = {left: (right, separation) for left, right, separation in matches}
    nsc_to_hsc = {right: (left, separation) for left, right, separation in matches}
    candidates: dict[str, UnifiedCandidate] = {}
    survey_to_unified: dict[str, str] = {}
    for key, detection in sorted(hsc.items()):
        if key in hsc_to_nsc:
            other_key, separation = hsc_to_nsc[key]
            other = nsc[other_key]
            candidate_id = f"{key}|{other_key}"
            ra, dec, sigma = _weighted_coordinate(detection, other)
            candidate = UnifiedCandidate(detection.cluster, candidate_id, ra, dec, sigma, detection, other, separation)
            survey_to_unified[other_key] = candidate_id
        else:
            candidate_id = key
            candidate = UnifiedCandidate(
                detection.cluster,
                candidate_id,
                detection.ra_deg,
                detection.dec_deg,
                detection.sigma_arcsec,
                detection,
                None,
                None,
            )
        candidates[candidate_id] = candidate
        survey_to_unified[key] = candidate_id
    for key, detection in sorted(nsc.items()):
        if key in nsc_to_hsc:
            continue
        candidate_id = key
        candidates[candidate_id] = UnifiedCandidate(
            detection.cluster,
            candidate_id,
            detection.ra_deg,
            detection.dec_deg,
            detection.sigma_arcsec,
            None,
            detection,
            None,
        )
        survey_to_unified[key] = candidate_id
    return candidates, survey_to_unified, matches


def nsc_has_band(row: dict[str, str], band: str) -> bool:
    magnitude = finite_float(row.get(f"{band}mag"))
    uncertainty = finite_float(row.get(f"{band}err"))
    detections = finite_float(row.get(f"nphot{band}"))
    return bool(
        magnitude is not None
        and uncertainty is not None
        and detections is not None
        and magnitude < 90.0
        and uncertainty < 9.0
        and detections > 0.0
    )


def candidate_diagnostics(candidate: UnifiedCandidate) -> dict[str, Any]:
    hsc_images = finite_float(candidate.hsc.payload.get("NumImages")) if candidate.hsc else None
    nsc_phot = finite_float(candidate.nsc.payload.get("nphot")) if candidate.nsc else None
    class_star = finite_float(candidate.nsc.payload.get("class_star")) if candidate.nsc else None
    pm_significance = None
    if candidate.nsc:
        pmra = finite_float(candidate.nsc.payload.get("pmra"))
        pmdec = finite_float(candidate.nsc.payload.get("pmdec"))
        pmraerr = finite_float(candidate.nsc.payload.get("pmraerr"))
        pmdecerr = finite_float(candidate.nsc.payload.get("pmdecerr"))
        if None not in (pmra, pmdec, pmraerr, pmdecerr) and pmraerr > 0.0 and pmdecerr > 0.0:
            pm_significance = math.hypot(pmra / pmraerr, pmdec / pmdecerr)
    repeated = bool((hsc_images is not None and hsc_images > 1.0) or (nsc_phot is not None and nsc_phot > 1.0))
    probable_foreground_star = bool(
        pm_significance is not None
        and pm_significance >= 5.0
        and class_star is not None
        and class_star >= 0.8
    )
    return {
        "dual_survey": candidate.dual_survey,
        "repeated_detection_support": repeated,
        "hsc_num_images": hsc_images,
        "nsc_nphot": nsc_phot,
        "nsc_class_star": class_star,
        "nsc_pm_significance": pm_significance,
        "probable_foreground_star_diagnostic": probable_foreground_star,
        "nsc_has_g": bool(candidate.nsc and nsc_has_band(candidate.nsc.payload, "g")),
        "nsc_has_r": bool(candidate.nsc and nsc_has_band(candidate.nsc.payload, "r")),
        "nsc_has_i": bool(candidate.nsc and nsc_has_band(candidate.nsc.payload, "i")),
        "nsc_has_z": bool(candidate.nsc and nsc_has_band(candidate.nsc.payload, "z")),
        "hsc_has_f435w": finite_float(candidate.hsc.payload.get("A_F435W")) is not None if candidate.hsc else False,
        "hsc_has_f606w": finite_float(candidate.hsc.payload.get("A_F606W")) is not None if candidate.hsc else False,
        "hsc_has_f814w": finite_float(candidate.hsc.payload.get("A_F814W")) is not None if candidate.hsc else False,
    }


def member_half_widths(member: dict[str, str], spec: dict[str, Any]) -> tuple[float, float]:
    if spec["coordinate_model"] == "sexagesimal_ra_dec_rounding":
        east = 7.5 * float(spec["ra_time_quantization_seconds"]) * math.cos(math.radians(float(member["dec_deg"])))
        north = 0.5 * float(spec["dec_quantization_arcsec"])
        return east, north
    if spec["coordinate_model"] == "decimal_degree_rounding":
        east = 0.5 * float(spec["ra_quantization_deg"]) * 3600.0 * math.cos(math.radians(float(member["dec_deg"])))
        north = 0.5 * float(spec["dec_quantization_deg"]) * 3600.0
        return east, north
    raise ValueError(f"unknown coordinate model: {spec['coordinate_model']}")


def global_assignment(
    members: list[dict[str, str]],
    member_candidates: dict[tuple[str, str], list[str]],
    log_likelihood_ratios: dict[tuple[str, str, str], float],
    *,
    counterpart_prior: float,
) -> dict[tuple[str, str], str | None]:
    candidate_ids = sorted(
        {
            candidate_id
            for member in members
            for candidate_id in member_candidates[(member["cluster"], member["object_id"])]
        }
    )
    candidate_column = {candidate_id: index for index, candidate_id in enumerate(candidate_ids)}
    rows = len(members)
    columns = len(candidate_ids) + rows
    impossible = -1.0e12
    score = np.full((rows, columns), impossible, dtype=float)
    for index, member in enumerate(members):
        member_key = (member["cluster"], member["object_id"])
        for candidate_id in member_candidates[member_key]:
            score[index, candidate_column[candidate_id]] = math.log(counterpart_prior) + log_likelihood_ratios[
                (member_key[0], member_key[1], candidate_id)
            ]
        score[index, len(candidate_ids) + index] = math.log(1.0 - counterpart_prior)
    row_indices, column_indices = linear_sum_assignment(-score)
    if not np.array_equal(row_indices, np.arange(rows)):
        raise RuntimeError("global assignment did not cover every member")
    output: dict[tuple[str, str], str | None] = {}
    reverse = {index: candidate_id for candidate_id, index in candidate_column.items()}
    for row_index, column_index in zip(row_indices, column_indices, strict=True):
        member = members[int(row_index)]
        key = (member["cluster"], member["object_id"])
        output[key] = reverse.get(int(column_index))
    return output


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def validate_inputs(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    for name, item in config["parents"].items():
        if not name.endswith("_sha256"):
            continue
        path_key = name.removesuffix("_sha256")
        path = ROOT / config["parents"][path_key]
        if sha256(path) != item:
            raise RuntimeError(f"parent hash mismatch: {path}")
    runner = ROOT / config["implementation"]["runner"]
    if sha256(runner) != config["implementation"]["runner_sha256"]:
        raise RuntimeError("frozen V19AA runner hash mismatch")
    hsc_report = json.loads((ROOT / config["parents"]["v19y_report"]).read_text(encoding="utf-8"))
    nsc_report = json.loads((ROOT / config["parents"]["v19z_report"]).read_text(encoding="utf-8"))
    members: list[dict[str, str]] = []
    for cluster, spec in config["member_catalogs"].items():
        rows = load_members(ROOT / spec["path"], int(spec["expected_rows"]))
        if any(row["cluster"] != cluster for row in rows):
            raise RuntimeError(f"cluster label mismatch in {spec['path']}")
        members.extend(rows)
    if len(members) != int(config["gates"]["exact_member_count"]):
        raise RuntimeError("total member count mismatch")
    return hsc_report, nsc_report, members


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["status"] != "frozen_before_any_counterpart_likelihood_or_assignment":
        raise RuntimeError("V19AA protocol is not in its frozen pre-association state")
    hsc_report, nsc_report, members = validate_inputs(config)
    astrometry = config["astrometry"]
    hsc, hsc_member_ids = load_survey_detections(
        hsc_report,
        survey="HSC",
        astrometric_floor_arcsec=float(astrometry["hsc_astrometric_floor_arcsec"]),
    )
    nsc, nsc_member_ids = load_survey_detections(
        nsc_report,
        survey="NSC",
        astrometric_floor_arcsec=float(astrometry["nsc_astrometric_floor_arcsec"]),
    )
    candidates, survey_to_unified, crossmatches = build_unified_candidates(
        hsc,
        nsc,
        crossmatch_radius_arcsec=float(astrometry["hsc_nsc_reciprocal_crossmatch_radius_arcsec"]),
    )

    member_candidates: dict[tuple[str, str], list[str]] = {}
    for member in members:
        key = (member["cluster"], member["object_id"])
        survey_ids = hsc_member_ids.get(key, set()) | nsc_member_ids.get(key, set())
        member_candidates[key] = sorted({survey_to_unified[survey_id] for survey_id in survey_ids})

    background_density: dict[str, float] = {}
    for cluster, spec in config["member_catalogs"].items():
        cluster_members = [row for row in members if row["cluster"] == cluster]
        excess = sum(max(len(member_candidates[(cluster, row["object_id"])]) - 1, 0) for row in cluster_members)
        area = len(cluster_members) * math.pi * float(spec["query_radius_arcsec"]) ** 2
        background_density[cluster] = (float(astrometry["background_jeffreys_pseudocount"]) + excess) / area

    priors = [float(value) for value in config["association"]["counterpart_prior_sensitivity"]]
    primary_prior = float(config["association"]["primary_counterpart_prior"])
    log_lr: dict[tuple[str, str, str], float] = {}
    posterior_rows: list[dict[str, Any]] = []
    member_work: dict[tuple[str, str], dict[str, Any]] = {}
    for member in members:
        key = (member["cluster"], member["object_id"])
        spec = config["member_catalogs"][member["cluster"]]
        east_half, north_half = member_half_widths(member, spec)
        candidate_ids = member_candidates[key]
        likelihood_ratios: list[float] = []
        working_rows: list[dict[str, Any]] = []
        for candidate_id in candidate_ids:
            candidate = candidates[candidate_id]
            east, north = tangent_offsets_arcsec(
                float(member["ra_deg"]),
                float(member["dec_deg"]),
                candidate.ra_deg,
                candidate.dec_deg,
            )
            sigma = math.hypot(candidate.sigma_arcsec, float(astrometry["published_centroid_floor_arcsec"]))
            positional_pdf = quantized_position_pdf_arcsec2(
                east,
                north,
                east_half_width_arcsec=east_half,
                north_half_width_arcsec=north_half,
                east_sigma_arcsec=sigma,
                north_sigma_arcsec=sigma,
            )
            ratio = positional_pdf / background_density[member["cluster"]]
            likelihood_ratios.append(ratio)
            log_lr[(key[0], key[1], candidate_id)] = math.log(max(ratio, np.finfo(float).tiny))
            working_rows.append(
                {
                    "cluster": key[0],
                    "object_id": key[1],
                    "candidate_id": candidate_id,
                    "east_offset_arcsec": east,
                    "north_offset_arcsec": north,
                    "angular_separation_arcsec": angular_separation_arcsec(
                        float(member["ra_deg"]),
                        float(member["dec_deg"]),
                        candidate.ra_deg,
                        candidate.dec_deg,
                    ),
                    "position_pdf_per_arcsec2": positional_pdf,
                    "background_density_per_arcsec2": background_density[member["cluster"]],
                    "likelihood_ratio": ratio,
                }
            )
        posterior_by_prior: dict[float, tuple[np.ndarray, float]] = {
            prior: association_posterior(likelihood_ratios, prior) for prior in priors
        }
        for index, row in enumerate(working_rows):
            candidate = candidates[row["candidate_id"]]
            diagnostic = candidate_diagnostics(candidate)
            for prior, (candidate_posterior, _) in posterior_by_prior.items():
                row[f"posterior_q_{prior:.2f}"] = float(candidate_posterior[index])
            row["posterior_min"] = min(float(values[0][index]) for values in posterior_by_prior.values())
            row["posterior_max"] = max(float(values[0][index]) for values in posterior_by_prior.values())
            row.update(diagnostic)
            posterior_rows.append(row)
        primary_posteriors, primary_null = posterior_by_prior[primary_prior]
        ordering = np.argsort(-primary_posteriors) if len(primary_posteriors) else np.asarray([], dtype=int)
        top_index = int(ordering[0]) if len(ordering) else None
        second_index = int(ordering[1]) if len(ordering) > 1 else None
        top_ratio = likelihood_ratios[top_index] if top_index is not None else 0.0
        second_ratio = likelihood_ratios[second_index] if second_index is not None else 0.0
        member_work[key] = {
            "top_candidate_id": candidate_ids[top_index] if top_index is not None else None,
            "top_posterior_primary": float(primary_posteriors[top_index]) if top_index is not None else 0.0,
            "top_posterior_min": min(float(values[0][top_index]) for values in posterior_by_prior.values()) if top_index is not None else 0.0,
            "top_to_second_likelihood_ratio": (top_ratio / second_ratio if second_ratio > 0.0 else math.inf if top_ratio > 0.0 else 0.0),
            "null_posterior_primary": float(primary_null),
            "null_posterior_by_prior": {
                prior: float(values[1]) for prior, values in posterior_by_prior.items()
            },
            "candidate_count": len(candidate_ids),
        }

    assignment: dict[tuple[str, str], str | None] = {}
    for cluster in sorted(config["member_catalogs"]):
        cluster_members = [row for row in members if row["cluster"] == cluster]
        assignment.update(
            global_assignment(
                cluster_members,
                member_candidates,
                log_lr,
                counterpart_prior=primary_prior,
            )
        )

    secure = config["association"]["secure_match_gates"]
    member_rows: list[dict[str, Any]] = []
    for member in members:
        key = (member["cluster"], member["object_id"])
        work = member_work[key]
        top_id = work["top_candidate_id"]
        top_diagnostic = candidate_diagnostics(candidates[top_id]) if top_id is not None else {}
        global_id = assignment[key]
        passes = bool(
            top_id is not None
            and work["top_posterior_min"] >= float(secure["minimum_posterior_across_prior_grid"])
            and work["top_to_second_likelihood_ratio"] >= float(secure["minimum_top_to_second_likelihood_ratio"])
            and global_id == top_id
            and (
                not bool(secure["require_dual_survey_or_repeated_detection"])
                or top_diagnostic["dual_survey"]
                or top_diagnostic["repeated_detection_support"]
            )
            and not top_diagnostic["probable_foreground_star_diagnostic"]
        )
        member_row = {
                "cluster": key[0],
                "object_id": key[1],
                "candidate_count": work["candidate_count"],
                "top_candidate_id": top_id,
                "global_map_candidate_id": global_id,
                "secure_counterpart_id": top_id if passes else None,
                "association_state": "secure" if passes else "no_candidate" if work["candidate_count"] == 0 else "ambiguous",
                "top_posterior_primary": work["top_posterior_primary"],
                "top_posterior_min": work["top_posterior_min"],
                "top_to_second_likelihood_ratio": work["top_to_second_likelihood_ratio"],
                "null_posterior_primary": work["null_posterior_primary"],
                "top_dual_survey": top_diagnostic.get("dual_survey"),
                "top_repeated_detection_support": top_diagnostic.get("repeated_detection_support"),
                "top_probable_foreground_star_diagnostic": top_diagnostic.get("probable_foreground_star_diagnostic"),
            }
        for prior, null_posterior in work["null_posterior_by_prior"].items():
            member_row[f"null_posterior_q_{prior:.2f}"] = null_posterior
        member_rows.append(member_row)

    candidate_rows = []
    for candidate in sorted(candidates.values(), key=lambda row: (row.cluster, row.candidate_id)):
        diagnostic = candidate_diagnostics(candidate)
        candidate_rows.append(
            {
                "cluster": candidate.cluster,
                "candidate_id": candidate.candidate_id,
                "ra_deg": candidate.ra_deg,
                "dec_deg": candidate.dec_deg,
                "astrometric_sigma_arcsec": candidate.sigma_arcsec,
                "hsc_match_id": candidate.hsc.survey_id if candidate.hsc else None,
                "nsc_id": candidate.nsc.survey_id if candidate.nsc else None,
                "cross_survey_separation_arcsec": candidate.cross_survey_separation_arcsec,
                **diagnostic,
            }
        )

    outputs = config["outputs"]
    candidate_path = ROOT / outputs["unified_candidates"]
    posterior_path = ROOT / outputs["candidate_posteriors"]
    member_path = ROOT / outputs["member_associations"]
    write_csv(candidate_path, candidate_rows, list(candidate_rows[0]))
    posterior_columns = list(posterior_rows[0]) if posterior_rows else ["cluster", "object_id", "candidate_id"]
    write_csv(posterior_path, posterior_rows, posterior_columns)
    write_csv(member_path, member_rows, list(member_rows[0]))

    cluster_summary: dict[str, Any] = {}
    for cluster in sorted(config["member_catalogs"]):
        selected = [row for row in member_rows if row["cluster"] == cluster]
        states = defaultdict(int)
        for row in selected:
            states[row["association_state"]] += 1
        cluster_candidates = [row for row in candidate_rows if row["cluster"] == cluster]
        cluster_summary[cluster] = {
            "members": len(selected),
            "association_states": dict(sorted(states.items())),
            "unified_candidates": len(cluster_candidates),
            "dual_survey_candidates": sum(bool(row["dual_survey"]) for row in cluster_candidates),
            "background_density_per_arcsec2": background_density[cluster],
            "secure_fraction": states["secure"] / len(selected),
        }

    report_path = ROOT / outputs["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "report_version": "SIGMA-V19AA-MEMBER-COUNTERPART-ASSOCIATION-1.0.0",
        "status": "completed_positional_counterpart_association_without_mass_or_physics",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "implementation": config["implementation"],
        "input_hashes": {name: value for name, value in config["parents"].items() if name.endswith("_sha256")},
        "catalog_counts": {
            "unique_hsc_detections": len(hsc),
            "unique_nsc_detections": len(nsc),
            "reciprocal_hsc_nsc_pairs": len(crossmatches),
            "unified_candidates": len(candidates),
        },
        "clusters": cluster_summary,
        "gates": {
            "exact_member_count": len(member_rows) == int(config["gates"]["exact_member_count"]),
            "posterior_normalization": all(
                abs(
                    sum(float(row[f"posterior_q_{prior:.2f}"]) for row in posterior_rows if row["cluster"] == member["cluster"] and row["object_id"] == member["object_id"])
                    + next(row[f"null_posterior_q_{prior:.2f}"] for row in member_rows if row["cluster"] == member["cluster"] and row["object_id"] == member["object_id"])
                    - 1.0
                ) <= float(config["gates"]["posterior_normalization_tolerance"])
                for member in members
                for prior in priors
            ),
            "global_one_to_one_assignment": len([value for value in assignment.values() if value is not None])
            == len({value for value in assignment.values() if value is not None}),
            "no_mass_or_physics": True,
        },
        "outputs": {
            "unified_candidates": candidate_path.relative_to(ROOT).as_posix(),
            "unified_candidates_sha256": sha256(candidate_path),
            "candidate_posteriors": posterior_path.relative_to(ROOT).as_posix(),
            "candidate_posteriors_sha256": sha256(posterior_path),
            "member_associations": member_path.relative_to(ROOT).as_posix(),
            "member_associations_sha256": sha256(member_path),
        },
        "claim_boundary": config["claim_boundary"],
        "counterpart_selected_only_when_preregistered_secure_gates_pass": True,
        "photometric_transformation_performed": False,
        "stellar_mass_inference_performed": False,
        "mass_current_constructed": False,
        "lensing_or_halo_payload_opened": False,
        "gravity_formula_or_parameter_changed": False,
    }
    report["gates"]["all_integrity_gates_pass"] = all(report["gates"].values())
    report_path.write_text(json.dumps(strict_json(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    report = run(args.config)
    print(json.dumps(strict_json(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
