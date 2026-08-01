"""Construct and audit empirical CLASH PSFs for the RX J2129 light-profile fit."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.ndimage import shift


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/r1_rxj2129_hst_psf_protocol.json"


def _resolve(path: str) -> Path:
    return ROOT / path


def _read_catalog(path: Path) -> pd.DataFrame:
    header = next(
        line[2:].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("# CLASHID")
    )
    return pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        names=header.split(),
        low_memory=False,
    )


def _select_candidates(catalog: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rule = config["candidate_selection"]
    numeric = [
        "PointS",
        "s2n",
        "photoflag",
        "F125W_WFC3_PHOTOZ",
        "x",
        "y",
        "fwhm",
    ]
    converted = catalog.copy()
    for column in numeric:
        converted[column] = pd.to_numeric(converted[column], errors="coerce")
    selected = converted[
        (converted["PointS"] >= rule["PointS_minimum"])
        & (converted["s2n"] >= rule["catalog_s2n_minimum"])
        & (converted["photoflag"] == rule["photoflag_required"])
        & (
            converted["F125W_WFC3_PHOTOZ"]
            >= rule["f125w_ab_magnitude_minimum"]
        )
        & (
            converted["F125W_WFC3_PHOTOZ"]
            <= rule["f125w_ab_magnitude_maximum"]
        )
    ].copy()
    return selected.sort_values("CLASHID").reset_index(drop=True)


def _radius_grid(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.indices((size, size), dtype=float)
    center = 0.5 * (size - 1)
    radius = np.hypot(xx - center, yy - center)
    return yy, xx, radius


def _prepare_stamp(
    image: np.ndarray,
    weight: np.ndarray,
    x_catalog: float,
    y_catalog: float,
    settings: dict[str, Any],
) -> tuple[np.ndarray, dict[str, float]]:
    size = int(settings["stamp_size_pixels"])
    half = size // 2
    x_zero = x_catalog - 1.0
    y_zero = y_catalog - 1.0
    x_center = int(round(x_zero))
    y_center = int(round(y_zero))
    if (
        x_center - half < 0
        or y_center - half < 0
        or x_center + half >= image.shape[1]
        or y_center + half >= image.shape[0]
    ):
        raise ValueError("PSF candidate stamp extends outside the image")
    stamp = np.asarray(
        image[
            y_center - half : y_center + half + 1,
            x_center - half : x_center + half + 1,
        ],
        dtype=float,
    )
    stamp_weight = np.asarray(
        weight[
            y_center - half : y_center + half + 1,
            x_center - half : x_center + half + 1,
        ],
        dtype=float,
    )
    yy, xx = np.indices(stamp.shape, dtype=float)
    expected_x = half + (x_zero - x_center)
    expected_y = half + (y_zero - y_center)
    expected_radius = np.hypot(xx - expected_x, yy - expected_y)
    inner_bg, outer_bg = settings["background_annulus_pixels"]
    background_mask = (
        (expected_radius >= inner_bg)
        & (expected_radius <= outer_bg)
        & np.isfinite(stamp)
        & np.isfinite(stamp_weight)
        & (stamp_weight > 0)
    )
    if int(background_mask.sum()) < 100:
        raise ValueError("Too few valid background pixels for PSF candidate")
    background = float(np.median(stamp[background_mask]))
    signal = np.where(
        np.isfinite(stamp) & np.isfinite(stamp_weight) & (stamp_weight > 0),
        np.clip(stamp - background, 0.0, None),
        0.0,
    )
    centroid_mask = expected_radius <= settings["centroid_search_radius_pixels"]
    centroid_flux = float(signal[centroid_mask].sum())
    if centroid_flux <= 0:
        raise ValueError("Non-positive PSF candidate centroid flux")
    centroid_x = float((signal[centroid_mask] * xx[centroid_mask]).sum() / centroid_flux)
    centroid_y = float((signal[centroid_mask] * yy[centroid_mask]).sum() / centroid_flux)
    centroid_shift = float(np.hypot(centroid_x - expected_x, centroid_y - expected_y))
    recentered = shift(
        signal,
        shift=(half - centroid_y, half - centroid_x),
        order=int(settings["recentering_interpolation_order"]),
        mode="constant",
        cval=0.0,
        prefilter=True,
    )
    recentered = np.clip(recentered, 0.0, None)
    _, _, radius = _radius_grid(size)
    aperture = radius <= settings["normalization_radius_pixels"]
    normalization = float(recentered[aperture].sum())
    if normalization <= 0:
        raise ValueError("Non-positive PSF normalization")
    normalized = recentered / normalization
    moment_mask = radius <= 5.0
    moment_sum = float(normalized[moment_mask].sum())
    radial_second_moment = float(
        (normalized[moment_mask] * radius[moment_mask] ** 2).sum() / moment_sum
    )
    moment_fwhm = 2.355 * np.sqrt(radial_second_moment / 2.0)
    metrics = {
        "background_image_units": background,
        "centroid_x_stamp_pixels": centroid_x,
        "centroid_y_stamp_pixels": centroid_y,
        "catalog_to_centroid_shift_pixels": centroid_shift,
        "moment_fwhm_pixels": float(moment_fwhm),
        "encircled_energy_r3": float(normalized[radius <= 3.0].sum()),
        "positive_weight_fraction": float((stamp_weight > 0).mean()),
        "peak_normalized": float(normalized.max()),
    }
    return normalized, metrics


def _combine_and_score(
    stamps: list[np.ndarray], config: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    settings = config["stamp_and_normalization"]
    gates = config["quality_gates"]
    size = stamps[0].shape[0]
    _, _, radius = _radius_grid(size)
    aperture = radius <= settings["normalization_radius_pixels"]
    stack = np.stack(stamps)
    combined = np.median(stack, axis=0)
    combined = np.clip(combined, 0.0, None)
    combined /= combined[aperture].sum()

    pairwise_l1 = [
        float(np.abs(stack[first] - stack[second])[aperture].sum())
        for first, second in combinations(range(len(stamps)), 2)
    ]
    encircled = [float(stamp[radius <= 3.0].sum()) for stamp in stamps]
    leave_one_out_l1 = []
    for omitted in range(len(stamps)):
        retained = np.delete(stack, omitted, axis=0)
        trial = np.median(retained, axis=0)
        trial = np.clip(trial, 0.0, None)
        trial /= trial[aperture].sum()
        leave_one_out_l1.append(float(np.abs(trial - combined)[aperture].sum()))

    scores = {
        "pairwise_l1_within_r10": pairwise_l1,
        "maximum_pairwise_l1_within_r10": max(pairwise_l1),
        "encircled_energy_r3": encircled,
        "encircled_energy_r3_spread": max(encircled) - min(encircled),
        "leave_one_out_l1_within_r10": leave_one_out_l1,
        "maximum_leave_one_out_l1_within_r10": max(leave_one_out_l1),
    }
    scores["pairwise_l1_gate_pass"] = bool(
        scores["maximum_pairwise_l1_within_r10"]
        <= gates["maximum_pairwise_l1_difference_within_r10"]
    )
    scores["encircled_energy_gate_pass"] = bool(
        scores["encircled_energy_r3_spread"]
        <= gates["maximum_pairwise_encircled_energy_r3_spread"]
    )
    scores["leave_one_out_gate_pass"] = bool(
        scores["maximum_leave_one_out_l1_within_r10"]
        <= gates["maximum_leave_one_out_l1_shift_within_r10"]
    )
    return combined, scores


def _audit_filter(
    label: str,
    science_path: Path,
    weight_path: Path,
    candidates: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[np.ndarray], pd.DataFrame, dict[str, Any]]:
    with fits.open(science_path, memmap=True) as science_hdul, fits.open(
        weight_path, memmap=True
    ) as weight_hdul:
        image = science_hdul[0].data
        weight = weight_hdul[0].data
        if image.shape != weight.shape:
            raise ValueError(f"{label} science and weight shapes differ")
        rows: list[dict[str, Any]] = []
        stamps: list[np.ndarray] = []
        for candidate in candidates.itertuples(index=False):
            stamp, metrics = _prepare_stamp(
                image,
                weight,
                float(candidate.x),
                float(candidate.y),
                config["stamp_and_normalization"],
            )
            stamps.append(stamp)
            rows.append(
                {
                    "filter": label,
                    "clash_id": candidate.CLASHID,
                    "catalog_x_one_indexed": float(candidate.x),
                    "catalog_y_one_indexed": float(candidate.y),
                    "catalog_s2n": float(candidate.s2n),
                    "catalog_f125w_ab": float(candidate.F125W_WFC3_PHOTOZ),
                    "catalog_fwhm_pixels": float(candidate.fwhm),
                    **metrics,
                }
            )
        units = science_hdul[0].header.get("BUNIT", "unknown")

    combined, scores = _combine_and_score(stamps, config)
    ledger = pd.DataFrame(rows)
    gates = config["quality_gates"]
    ledger["centroid_gate_pass"] = (
        ledger["catalog_to_centroid_shift_pixels"]
        <= gates["maximum_centroid_shift_pixels"]
    )
    ledger["fwhm_gate_pass"] = ledger["moment_fwhm_pixels"].between(
        gates["minimum_moment_fwhm_pixels"], gates["maximum_moment_fwhm_pixels"]
    )
    per_star_pass = bool(ledger[["centroid_gate_pass", "fwhm_gate_pass"]].all().all())
    report = {
        "filter": label,
        "science_units": units,
        "candidate_count": len(stamps),
        "per_star_gate_pass": per_star_pass,
        **scores,
    }
    report["filter_gate_pass"] = bool(
        per_star_pass
        and scores["pairwise_l1_gate_pass"]
        and scores["encircled_energy_gate_pass"]
        and scores["leave_one_out_gate_pass"]
    )
    return combined, stamps, ledger, report


def _plot(
    path: Path,
    labels: list[str],
    star_ids: list[str],
    stamp_sets: list[list[np.ndarray]],
    combined: list[np.ndarray],
) -> None:
    columns = len(star_ids) + 1
    fig, axes = plt.subplots(len(labels), columns, figsize=(3.0 * columns, 5.8))
    for row, label in enumerate(labels):
        images = stamp_sets[row] + [combined[row]]
        titles = star_ids + ["median PSF"]
        common_maximum = max(float(np.max(image)) for image in images)
        for column, (image, title) in enumerate(zip(images, titles, strict=True)):
            axis = axes[row, column]
            axis.imshow(
                np.arcsinh(image / max(common_maximum, 1e-12) * 50.0),
                origin="lower",
                cmap="magma",
            )
            axis.set_title(title, fontsize=8)
            axis.set_xticks([])
            axis.set_yticks([])
            if column == 0:
                axis.set_ylabel(label)
    fig.suptitle("RX J2129 empirical CLASH PSF audit (asinh stretch)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def audit(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["authorization"]["gravity_response_fit"]:
        raise ValueError("PSF protocol cannot authorize a gravity response fit")
    catalog = _read_catalog(_resolve(config["inputs"]["catalog"]))
    candidates = _select_candidates(catalog, config)
    minimum = config["candidate_selection"]["minimum_candidates"]
    if len(candidates) < minimum:
        raise ValueError(f"Only {len(candidates)} PSF candidates; protocol requires {minimum}")

    specifications = [
        (
            "F125W",
            _resolve(config["inputs"]["f125w_science"]),
            _resolve(config["inputs"]["f125w_weight"]),
        ),
        (
            "F814W",
            _resolve(config["inputs"]["f814w_science"]),
            _resolve(config["inputs"]["f814w_weight"]),
        ),
    ]
    combined_psfs: list[np.ndarray] = []
    stamp_sets: list[list[np.ndarray]] = []
    ledgers: list[pd.DataFrame] = []
    filter_reports: list[dict[str, Any]] = []
    for label, science, weight in specifications:
        combined, stamps, ledger, filter_report = _audit_filter(
            label, science, weight, candidates, config
        )
        combined_psfs.append(combined)
        stamp_sets.append(stamps)
        ledgers.append(ledger)
        filter_reports.append(filter_report)

    outputs = config["outputs"]
    npz_path = _resolve(outputs["psf_npz"])
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        f125w=combined_psfs[0],
        f814w=combined_psfs[1],
        f125w_stars=np.stack(stamp_sets[0]),
        f814w_stars=np.stack(stamp_sets[1]),
        pixel_scale_arcsec=np.asarray(0.065),
        star_ids=candidates["CLASHID"].astype(str).to_numpy(),
    )
    ledger = pd.concat(ledgers, ignore_index=True)
    ledger_path = _resolve(outputs["star_ledger"])
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(ledger_path, index=False)
    _plot(
        _resolve(outputs["diagnostic"]),
        [item[0] for item in specifications],
        candidates["CLASHID"].astype(str).tolist(),
        stamp_sets,
        combined_psfs,
    )

    both_pass = bool(all(item["filter_gate_pass"] for item in filter_reports))
    report = {
        "protocol_version": config["protocol_version"],
        "status": (
            "empirical_psf_gate_pass_bcg_icl_protocol_may_proceed"
            if both_pass
            else "empirical_psf_gate_failed_external_or_synthetic_psf_required"
        ),
        "disclosure": config["disclosure"],
        "candidate_ids": candidates["CLASHID"].astype(str).tolist(),
        "candidate_count": int(len(candidates)),
        "filters": {item["filter"]: item for item in filter_reports},
        "both_filters_gate_pass": both_pass,
        "bcg_icl_decomposition_authorized": bool(
            both_pass
            and config["authorization"]["bcg_icl_decomposition_if_psf_gate_passes"]
        ),
        "gravity_response_fit_authorized": False,
        "strict_r1_ready": False,
        "outputs": outputs,
        "failure_response": config["failure_response"],
    }
    report_path = _resolve(outputs["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.config), indent=2))


if __name__ == "__main__":
    main()
