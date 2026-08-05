#!/usr/bin/env python3
"""Build the frozen V19BB Abell 2146 luminosity-current ensemble."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19bb_abell2146_luminosity_current_ensemble.json"
SPEED_OF_LIGHT_KM_S = 299792.458


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def numeric_member_key(member_id: str) -> tuple[int, str]:
    try:
        return int(member_id), member_id
    except ValueError:
        return 10**9, member_id


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


def quantized_axis_pdf(
    offset_arcsec: float, half_width_arcsec: float, sigma_arcsec: float
) -> float:
    """Uniform rounding interval convolved with a Gaussian centroid error."""
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
    sigma_arcsec: float,
) -> float:
    return quantized_axis_pdf(
        east_arcsec, east_half_width_arcsec, sigma_arcsec
    ) * quantized_axis_pdf(north_arcsec, north_half_width_arcsec, sigma_arcsec)


def association_posterior(
    likelihood_ratios: list[float], counterpart_prior: float
) -> tuple[np.ndarray, float]:
    ratios = np.asarray(likelihood_ratios, dtype=float)
    if np.any(~np.isfinite(ratios)) or np.any(ratios < 0.0):
        raise ValueError("likelihood ratios must be finite and nonnegative")
    denominator = (1.0 - counterpart_prior) + counterpart_prior * float(np.sum(ratios))
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise ValueError("invalid posterior denominator")
    return counterpart_prior * ratios / denominator, (1.0 - counterpart_prior) / denominator


def member_half_widths(member: dict[str, str]) -> tuple[float, float]:
    east = 0.5 * 0.0001 * 3600.0 * math.cos(math.radians(float(member["dec_deg"])))
    north = 0.5 * 0.0001 * 3600.0
    return east, north


def candidate_ratios(
    member: dict[str, str],
    candidates: list[dict[str, Any]],
    sigma_extra_arcsec: float,
    centroid_floor_arcsec: float,
    background_density: float,
) -> list[float]:
    east_half, north_half = member_half_widths(member)
    ratios: list[float] = []
    for candidate in candidates:
        sigma = math.sqrt(
            float(candidate["astrometric_sigma_arcsec"]) ** 2
            + centroid_floor_arcsec**2
            + sigma_extra_arcsec**2
        )
        density = quantized_position_pdf_arcsec2(
            float(candidate["east_offset_arcsec"]),
            float(candidate["north_offset_arcsec"]),
            east_half_width_arcsec=east_half,
            north_half_width_arcsec=north_half,
            sigma_arcsec=sigma,
        )
        ratios.append(density / background_density)
    return ratios


def member_log_evidence(
    member: dict[str, str],
    candidates: list[dict[str, Any]],
    sigma_extra_arcsec: float,
    counterpart_prior: float,
    centroid_floor_arcsec: float,
    background_density: float,
) -> float:
    ratios = candidate_ratios(
        member,
        candidates,
        sigma_extra_arcsec,
        centroid_floor_arcsec,
        background_density,
    )
    evidence = (1.0 - counterpart_prior) + counterpart_prior * math.fsum(ratios)
    return math.log(evidence)


def select_grid_index(scores: list[float]) -> int:
    """Return the first maximum so exact ties choose the smaller grid value."""
    values = np.asarray(scores, dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError("calibration score is nonfinite")
    return int(np.argmax(values))


def fold_assignments(member_ids: list[str], folds: int) -> dict[str, int]:
    ordered = sorted(member_ids, key=numeric_member_key)
    return {member_id: index % folds for index, member_id in enumerate(ordered)}


def null_position(
    rng: np.random.Generator, member_ra_deg: float, member_dec_deg: float
) -> tuple[float, float]:
    east_half = 0.5 * 0.0001 * 3600.0 * math.cos(math.radians(member_dec_deg))
    north_half = 0.5 * 0.0001 * 3600.0
    east = rng.uniform(-east_half, east_half)
    north = rng.uniform(-north_half, north_half)
    ra = member_ra_deg + east / (3600.0 * math.cos(math.radians(member_dec_deg)))
    dec = member_dec_deg + north / 3600.0
    return ra, dec


def verify_parent_hashes(config: dict[str, Any]) -> dict[str, str]:
    actual: dict[str, str] = {}
    for name, item in config["parents"].items():
        path = ROOT / item["path"]
        value = sha256(path)
        if value != item["sha256"]:
            raise ValueError(f"parent hash mismatch for {name}: {value} != {item['sha256']}")
        actual[name] = value
    return actual


def load_and_verify_hsc_payloads(provenance_path: Path) -> tuple[dict[str, dict[str, str]], int]:
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    records = [row for row in provenance["records"] if row["cluster"] == "ABELL2146"]
    if len(records) != 63:
        raise ValueError(f"expected 63 Abell HSC records, found {len(records)}")
    detections: dict[str, dict[str, str]] = {}
    verified_files = 0
    for record in records:
        for path_key, hash_key in (
            ("csv_path", "csv_sha256"),
            ("query_url_path", "query_url_sha256"),
        ):
            path = ROOT / record[path_key]
            if sha256(path) != record[hash_key]:
                raise ValueError(f"raw HSC hash mismatch: {path}")
            verified_files += 1
        csv_path = ROOT / record["csv_path"]
        if csv_path.stat().st_size == 0:
            continue
        for row in read_csv(csv_path):
            match_id = row["MatchID"]
            if match_id in detections and detections[match_id] != row:
                raise ValueError(f"HSC MatchID {match_id} has inconsistent repeated payloads")
            detections[match_id] = row
    return detections, verified_files


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def output_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def make_figure(
    path: Path,
    grid: list[float],
    priors: list[float],
    full_scores: dict[float, list[float]],
    cv_rows: list[dict[str, Any]],
    member_summary: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), constrained_layout=True)
    for prior in priors:
        scores = np.asarray(full_scores[prior])
        axes[0].plot(grid, scores - np.max(scores), marker="o", label=f"q={prior:.2f}")
    axes[0].axhline(0.0, color="black", linewidth=0.7)
    axes[0].set_xlabel("shared extra astrometric scatter (arcsec)")
    axes[0].set_ylabel("log evidence minus maximum")
    axes[0].set_title("source-only catalog scale")
    axes[0].legend()

    folds = [int(row["fold"]) for row in cv_rows]
    improvements = [float(row["heldout_log_evidence_improvement_over_zero"]) for row in cv_rows]
    axes[1].bar(
        folds,
        improvements,
        color=["tab:blue" if value > 0 else "tab:red" for value in improvements],
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("held-out fold")
    axes[1].set_ylabel("predictive log-evidence improvement")
    axes[1].set_title("seven-fold predictive check")

    ordered = sorted(member_summary, key=lambda row: float(row["finite_f814w_probability"]))
    axes[2].plot(
        np.arange(len(ordered)),
        [float(row["finite_f814w_probability"]) for row in ordered],
        color="tab:green",
    )
    axes[2].axhline(0.5, color="black", linewidth=0.7, linestyle="--")
    axes[2].set_xlabel("member rank")
    axes[2].set_ylabel("posterior probability of measured F814W")
    axes[2].set_ylim(-0.03, 1.03)
    axes[2].set_title("honest luminosity coverage")

    fig.suptitle(
        "Abell 2146 target-blind catalog calibration and luminosity coverage\n"
        "no lensing, halo, gravity, subcluster label, or magnitude used to select the width"
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
    paths = {name: ROOT / item["path"] for name, item in config["parents"].items()}

    v19aa_report = json.loads(paths["v19aa_report"].read_text(encoding="utf-8"))
    if (
        v19aa_report["status"]
        != "completed_positional_counterpart_association_without_mass_or_physics"
    ):
        raise ValueError("V19AA did not authorize positional association inputs")
    if v19aa_report["lensing_or_halo_payload_opened"]:
        raise ValueError("V19AA unexpectedly opened lensing or halo payload")

    hsc_detections, raw_hsc_files_verified = load_and_verify_hsc_payloads(
        paths["v19y_hsc_provenance"]
    )
    members = {
        row["object_id"]: row
        for row in read_csv(paths["spectroscopic_members"])
        if row["cluster"] == "ABELL2146"
    }
    unified = {
        row["candidate_id"]: row
        for row in read_csv(paths["unified_candidates"])
        if row["cluster"] == "ABELL2146"
    }
    posterior_links = [
        row for row in read_csv(paths["candidate_posteriors"]) if row["cluster"] == "ABELL2146"
    ]
    associations = {
        row["object_id"]: row
        for row in read_csv(paths["member_associations"])
        if row["cluster"] == "ABELL2146"
    }
    expected = config["population"]
    if len(members) != expected["expected_spectroscopic_members"]:
        raise ValueError("Abell member count changed")
    if len(posterior_links) != expected["expected_candidate_hypotheses"]:
        raise ValueError("Abell candidate hypothesis count changed")
    if len(unified) != expected["expected_unique_candidates"]:
        raise ValueError("Abell unified candidate count changed")
    if set(members) != set(associations):
        raise ValueError("member association inventory does not match spectroscopy")

    linked_members: dict[str, set[str]] = defaultdict(set)
    links_by_member: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for link in posterior_links:
        member_id = link["object_id"]
        candidate_id = link["candidate_id"]
        if member_id not in members or candidate_id not in unified:
            raise ValueError("candidate link references an unknown member or candidate")
        linked_members[candidate_id].add(member_id)
        candidate = dict(unified[candidate_id])
        east, north = tangent_offsets_arcsec(
            float(members[member_id]["ra_deg"]),
            float(members[member_id]["dec_deg"]),
            float(candidate["ra_deg"]),
            float(candidate["dec_deg"]),
        )
        candidate["east_offset_arcsec"] = east
        candidate["north_offset_arcsec"] = north
        hsc_id = candidate["hsc_match_id"]
        hsc_row = hsc_detections.get(hsc_id) if hsc_id else None
        f814w = finite_float(hsc_row.get("A_F814W")) if hsc_row else None
        candidate["f814w_mag"] = f814w
        candidate["relative_f814w_luminosity"] = (
            10.0 ** (-0.4 * (f814w - config["luminosity_and_current"]["reference_magnitude"]))
            if f814w is not None
            else None
        )
        links_by_member[member_id].append(candidate)
    for candidates in links_by_member.values():
        candidates.sort(key=lambda row: row["candidate_id"])

    shared_candidates = sum(len(member_ids) > 1 for member_ids in linked_members.values())
    candidate_distribution = Counter(len(links_by_member[member_id]) for member_id in members)
    hsc_matches = sum(bool(row["hsc_match_id"]) for row in unified.values())
    finite_hsc = sum(
        bool(row["hsc_match_id"])
        and row["hsc_match_id"] in hsc_detections
        and finite_float(hsc_detections[row["hsc_match_id"]].get("A_F814W")) is not None
        for row in unified.values()
    )
    inventory_exact = (
        shared_candidates == expected["expected_shared_candidates"]
        and candidate_distribution
        == Counter(
            {
                int(key): value
                for key, value in expected["expected_member_candidate_count_distribution"].items()
            }
        )
        and hsc_matches == expected["expected_candidates_with_hsc_match"]
        and finite_hsc == expected["expected_candidates_with_finite_hsc_f814w"]
    )
    if not inventory_exact:
        raise ValueError("frozen Abell candidate inventory changed")

    calibration = config["catalog_level_astrometric_calibration"]
    grid = [float(value) for value in calibration["sigma_extra_grid_arcsec"]]
    priors = [float(value) for value in calibration["counterpart_prior_sensitivity"]]
    primary_prior = float(calibration["primary_counterpart_prior"])
    centroid_floor = float(calibration["baseline_published_centroid_floor_arcsec"])
    background_density = float(calibration["background_density_per_arcsec2"])
    ordered_ids = sorted(members, key=numeric_member_key)

    score_rows: list[dict[str, Any]] = []
    full_scores: dict[float, list[float]] = {}
    selected_by_prior: dict[float, int] = {}
    for prior in priors:
        scores = [
            math.fsum(
                member_log_evidence(
                    members[member_id],
                    links_by_member[member_id],
                    sigma,
                    prior,
                    centroid_floor,
                    background_density,
                )
                for member_id in ordered_ids
            )
            for sigma in grid
        ]
        full_scores[prior] = scores
        selected_index = select_grid_index(scores)
        selected_by_prior[prior] = selected_index
        for index, (sigma, score) in enumerate(zip(grid, scores, strict=True)):
            score_rows.append(
                {
                    "row_type": "full_sample_grid",
                    "counterpart_prior": prior,
                    "sigma_extra_arcsec": sigma,
                    "fold": "",
                    "training_log_evidence": score,
                    "heldout_log_evidence": "",
                    "heldout_zero_log_evidence": "",
                    "heldout_log_evidence_improvement_over_zero": "",
                    "selected": index == selected_index,
                }
            )

    primary_index = selected_by_prior[primary_prior]
    selected_sigma = grid[primary_index]
    folds = int(calibration["cross_validation"]["folds"])
    fold_map = fold_assignments(ordered_ids, folds)
    cv_rows: list[dict[str, Any]] = []
    for fold in range(folds):
        training_ids = [member_id for member_id in ordered_ids if fold_map[member_id] != fold]
        heldout_ids = [member_id for member_id in ordered_ids if fold_map[member_id] == fold]
        training_scores = [
            math.fsum(
                member_log_evidence(
                    members[member_id],
                    links_by_member[member_id],
                    sigma,
                    primary_prior,
                    centroid_floor,
                    background_density,
                )
                for member_id in training_ids
            )
            for sigma in grid
        ]
        fold_index = select_grid_index(training_scores)
        fold_sigma = grid[fold_index]
        heldout = math.fsum(
            member_log_evidence(
                members[member_id],
                links_by_member[member_id],
                fold_sigma,
                primary_prior,
                centroid_floor,
                background_density,
            )
            for member_id in heldout_ids
        )
        heldout_zero = math.fsum(
            member_log_evidence(
                members[member_id],
                links_by_member[member_id],
                grid[0],
                primary_prior,
                centroid_floor,
                background_density,
            )
            for member_id in heldout_ids
        )
        row = {
            "row_type": "cross_validation_fold",
            "counterpart_prior": primary_prior,
            "sigma_extra_arcsec": fold_sigma,
            "fold": fold,
            "training_log_evidence": training_scores[fold_index],
            "heldout_log_evidence": heldout,
            "heldout_zero_log_evidence": heldout_zero,
            "heldout_log_evidence_improvement_over_zero": heldout - heldout_zero,
            "selected": True,
        }
        score_rows.append(row)
        cv_rows.append(row)

    state_rows: list[dict[str, Any]] = []
    member_summary: list[dict[str, Any]] = []
    state_probabilities: dict[str, np.ndarray] = {}
    state_keys: dict[str, list[tuple[str, str]]] = {}
    max_normalization_error = 0.0
    expected_finite_f814w = 0.0
    nominal_systemic = float(
        np.median([float(members[member_id]["heliocentric_cz_km_s"]) for member_id in ordered_ids])
    )
    for member_id in ordered_ids:
        member = members[member_id]
        candidates = links_by_member[member_id]
        ratios = candidate_ratios(
            member,
            candidates,
            selected_sigma,
            centroid_floor,
            background_density,
        )
        candidate_probs, null_prob = association_posterior(ratios, primary_prior)
        probabilities = np.concatenate([candidate_probs, [null_prob]])
        keys = [("candidate", row["candidate_id"]) for row in candidates] + [("null", "")]
        state_probabilities[member_id] = probabilities
        state_keys[member_id] = keys
        normalization_error = abs(float(np.sum(probabilities)) - 1.0)
        max_normalization_error = max(max_normalization_error, normalization_error)
        finite_probability = 0.0
        expected_ra = null_prob * float(member["ra_deg"])
        expected_dec = null_prob * float(member["dec_deg"])
        entropy = 0.0
        for candidate, probability, ratio in zip(candidates, candidate_probs, ratios, strict=True):
            finite = candidate["f814w_mag"] is not None
            finite_probability += float(probability) if finite else 0.0
            expected_ra += float(probability) * float(candidate["ra_deg"])
            expected_dec += float(probability) * float(candidate["dec_deg"])
            if probability > 0.0:
                entropy -= float(probability) * math.log(float(probability))
            state_rows.append(
                {
                    "member_id": member_id,
                    "state_type": "candidate",
                    "candidate_id": candidate["candidate_id"],
                    "probability": float(probability),
                    "likelihood_ratio": ratio,
                    "ra_deg": candidate["ra_deg"],
                    "dec_deg": candidate["dec_deg"],
                    "hsc_match_id": candidate["hsc_match_id"],
                    "f814w_mag": candidate["f814w_mag"] if finite else "",
                    "relative_f814w_luminosity": (
                        candidate["relative_f814w_luminosity"] if finite else ""
                    ),
                    "luminosity_state": "measured_f814w" if finite else "missing_photometry",
                }
            )
        if null_prob > 0.0:
            entropy -= null_prob * math.log(null_prob)
        state_rows.append(
            {
                "member_id": member_id,
                "state_type": "null",
                "candidate_id": "",
                "probability": null_prob,
                "likelihood_ratio": "",
                "ra_deg": member["ra_deg"],
                "dec_deg": member["dec_deg"],
                "hsc_match_id": "",
                "f814w_mag": "",
                "relative_f814w_luminosity": "",
                "luminosity_state": "missing_photometry",
            }
        )
        expected_finite_f814w += finite_probability
        top_index = int(np.argmax(probabilities))
        top_type, top_candidate = keys[top_index]
        nominal_v_los = (float(member["heliocentric_cz_km_s"]) - nominal_systemic) / (
            1.0 + nominal_systemic / SPEED_OF_LIGHT_KM_S
        )
        expected_luminosity = math.fsum(
            float(probability) * float(candidate["relative_f814w_luminosity"])
            for candidate, probability in zip(candidates, candidate_probs, strict=True)
            if candidate["relative_f814w_luminosity"] is not None
        )
        member_summary.append(
            {
                "member_id": member_id,
                "candidate_count": len(candidates),
                "top_state_type": top_type,
                "top_candidate_id": top_candidate,
                "top_state_probability": float(probabilities[top_index]),
                "null_probability": null_prob,
                "finite_f814w_probability": finite_probability,
                "expected_relative_f814w_luminosity": expected_luminosity,
                "position_entropy_nats": entropy,
                "expected_ra_deg": expected_ra,
                "expected_dec_deg": expected_dec,
                "heliocentric_cz_km_s": float(member["heliocentric_cz_km_s"]),
                "cz_uncertainty_km_s": float(member["cz_uncertainty_km_s"]),
                "nominal_cluster_median_cz_km_s": nominal_systemic,
                "nominal_v_los_rest_km_s": nominal_v_los,
                "expected_nominal_los_current_proxy": expected_luminosity * nominal_v_los,
                "transverse_velocity_state": "unmeasured_not_imputed",
            }
        )

    outputs = config["outputs"]
    score_path = ROOT / outputs["calibration_scores"]
    state_path = ROOT / outputs["state_marginals"]
    summary_path = ROOT / outputs["member_summary"]
    ensemble_path = ROOT / outputs["ensemble"]
    figure_path = ROOT / outputs["figure"]
    report_path = ROOT / outputs["report"]
    write_csv(score_path, score_rows, list(score_rows[0]))
    write_csv(state_path, state_rows, list(state_rows[0]))
    write_csv(summary_path, member_summary, list(member_summary[0]))

    rng = np.random.Generator(np.random.PCG64(int(config["ensemble"]["seed"])))
    draws = int(config["ensemble"]["draws"])
    frequencies: Counter[tuple[str, str, str]] = Counter()
    ensemble_fields = [
        "sample_id",
        "cluster",
        "member_id",
        "position_state_type",
        "selected_candidate_id",
        "ra_deg",
        "dec_deg",
        "f814w_mag",
        "relative_f814w_luminosity",
        "luminosity_state",
        "cz_draw_km_s",
        "cluster_median_cz_draw_km_s",
        "v_los_rest_km_s",
        "los_current_proxy",
        "cz_uncertainty_km_s",
        "v_east_km_s",
        "v_north_km_s",
        "transverse_velocity_state",
    ]
    ensemble_path.parent.mkdir(parents=True, exist_ok=True)
    all_draws_have_63 = True
    with (
        ensemble_path.open("wb") as raw_handle,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as compressed_handle,
    ):
        text_handle = io.TextIOWrapper(compressed_handle, newline="", encoding="utf-8")
        writer = csv.DictWriter(text_handle, fieldnames=ensemble_fields, extrasaction="raise")
        writer.writeheader()
        for sample_id in range(draws):
            selected_indices = {
                member_id: int(
                    rng.choice(
                        len(state_probabilities[member_id]), p=state_probabilities[member_id]
                    )
                )
                for member_id in ordered_ids
            }
            all_draws_have_63 &= len(selected_indices) == 63
            cz_draws = {
                member_id: float(members[member_id]["heliocentric_cz_km_s"])
                + rng.normal(0.0, float(members[member_id]["cz_uncertainty_km_s"]))
                for member_id in ordered_ids
            }
            systemic = float(np.median(list(cz_draws.values())))
            for member_id in ordered_ids:
                member = members[member_id]
                state_index = selected_indices[member_id]
                state_type, candidate_id = state_keys[member_id][state_index]
                if state_type == "candidate":
                    candidate = links_by_member[member_id][state_index]
                    ra_deg = float(candidate["ra_deg"])
                    dec_deg = float(candidate["dec_deg"])
                    f814w = candidate["f814w_mag"]
                    luminosity = candidate["relative_f814w_luminosity"]
                else:
                    candidate = None
                    ra_deg, dec_deg = null_position(
                        rng, float(member["ra_deg"]), float(member["dec_deg"])
                    )
                    f814w = None
                    luminosity = None
                luminosity_state = (
                    "measured_f814w" if luminosity is not None else "missing_photometry"
                )
                v_los = (cz_draws[member_id] - systemic) / (1.0 + systemic / SPEED_OF_LIGHT_KM_S)
                frequencies[(member_id, state_type, candidate_id)] += 1
                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "cluster": "ABELL2146",
                        "member_id": member_id,
                        "position_state_type": state_type,
                        "selected_candidate_id": candidate_id,
                        "ra_deg": ra_deg,
                        "dec_deg": dec_deg,
                        "f814w_mag": f814w if f814w is not None else "",
                        "relative_f814w_luminosity": luminosity if luminosity is not None else "",
                        "luminosity_state": luminosity_state,
                        "cz_draw_km_s": cz_draws[member_id],
                        "cluster_median_cz_draw_km_s": systemic,
                        "v_los_rest_km_s": v_los,
                        "los_current_proxy": luminosity * v_los if luminosity is not None else "",
                        "cz_uncertainty_km_s": float(member["cz_uncertainty_km_s"]),
                        "v_east_km_s": "",
                        "v_north_km_s": "",
                        "transverse_velocity_state": "unmeasured_not_imputed",
                    }
                )
        text_handle.flush()
        text_handle.detach()

    sampled_differences = [
        abs(
            frequencies[(row["member_id"], row["state_type"], row["candidate_id"])] / draws
            - float(row["probability"])
        )
        for row in state_rows
    ]
    make_figure(figure_path, grid, priors, full_scores, cv_rows, member_summary)

    gates = config["gates"]
    improving_folds = sum(
        float(row["heldout_log_evidence_improvement_over_zero"]) > 0.0 for row in cv_rows
    )
    prior_index_spread = max(selected_by_prior.values()) - min(selected_by_prior.values())
    gate_results = {
        "all_parent_and_raw_hsc_hashes_exact": raw_hsc_files_verified == 126,
        "inventory_counts_exact": inventory_exact,
        "primary_sigma_extra_interior": 0 < primary_index < len(grid) - 1,
        "cross_validation_improves_over_zero": improving_folds
        >= gates["minimum_cross_validation_folds_improving_over_zero"],
        "prior_sensitivity_grid_index_stable": prior_index_spread
        <= gates["maximum_primary_selected_grid_index_difference_across_prior_sensitivity"],
        "posterior_normalization": max_normalization_error
        <= gates["posterior_normalization_tolerance"],
        "expected_measured_f814w_coverage": expected_finite_f814w
        >= gates["minimum_expected_members_with_finite_f814w_per_draw"],
        "sampled_vs_exact_state_marginal": max(sampled_differences)
        <= gates["sampled_vs_exact_state_marginal_max_absolute_difference"],
        "all_63_members_in_every_draw": all_draws_have_63,
        "missing_photometry_states_explicit": any(
            row["luminosity_state"] == "missing_photometry" for row in state_rows
        ),
        "no_lensing_halo_or_gravity_payload": True,
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
        "raw_hsc_files_verified": raw_hsc_files_verified,
        "inventory": {
            "spectroscopic_members": len(members),
            "candidate_hypotheses": len(posterior_links),
            "unique_candidates": len(unified),
            "shared_candidates": shared_candidates,
            "member_candidate_count_distribution": {
                str(key): candidate_distribution[key] for key in sorted(candidate_distribution)
            },
            "candidates_with_hsc_match": hsc_matches,
            "candidates_with_finite_hsc_f814w": finite_hsc,
        },
        "catalog_astrometric_calibration": {
            "primary_counterpart_prior": primary_prior,
            "selected_sigma_extra_arcsec": selected_sigma,
            "selected_grid_index": primary_index,
            "selected_sigma_by_prior": {
                str(prior): grid[index] for prior, index in selected_by_prior.items()
            },
            "selected_grid_index_spread_across_priors": prior_index_spread,
            "cross_validation_folds": folds,
            "cross_validation_folds_improving_over_zero": improving_folds,
            "cross_validation_rows": cv_rows,
            "measurement_nuisance_not_gravity_length": True,
        },
        "posterior": {
            "maximum_normalization_error": max_normalization_error,
            "expected_members_with_finite_f814w_per_draw": expected_finite_f814w,
            "expected_missing_photometry_members_per_draw": len(members) - expected_finite_f814w,
            "members_top_candidate": sum(
                row["top_state_type"] == "candidate" for row in member_summary
            ),
            "members_top_null": sum(row["top_state_type"] == "null" for row in member_summary),
            "minimum_finite_f814w_probability": min(
                float(row["finite_f814w_probability"]) for row in member_summary
            ),
            "median_finite_f814w_probability": float(
                np.median([float(row["finite_f814w_probability"]) for row in member_summary])
            ),
            "maximum_finite_f814w_probability": max(
                float(row["finite_f814w_probability"]) for row in member_summary
            ),
        },
        "ensemble": {
            "draws": draws,
            "members_per_draw": len(members),
            "rows": draws * len(members),
            "maximum_sampled_vs_exact_state_marginal_difference": max(sampled_differences),
            "all_members_present_every_draw": all_draws_have_63,
        },
        "gate_results": gate_results,
        "outputs": {
            "calibration_scores": output_record(score_path),
            "state_marginals": output_record(state_path),
            "member_summary": output_record(summary_path),
            "ensemble": output_record(ensemble_path),
            "figure": output_record(figure_path),
        },
        "claim_boundary": config["claim_boundary"],
        "hard_counterpart_selected": False,
        "missing_photometry_or_stellar_mass_inferred": False,
        "transverse_velocity_imputed": False,
        "lensing_or_halo_payload_opened": False,
        "gravity_formula_or_parameter_changed": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if decision != "passed":
        raise RuntimeError(f"V19BB failed closed: {gate_results}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    report = run(args.config.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
