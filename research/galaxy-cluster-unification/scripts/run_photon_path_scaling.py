#!/usr/bin/env python3
"""Test whether SPARC rotation residuals grow with photon path length."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.data import load_curves  # noqa: E402
from voidscreen.photon_joint_forward import stable_source_folds  # noqa: E402
from voidscreen.photon_path_scaling import (  # noqa: E402
    baryonic_speed,
    path_feature,
    rar_speed,
    weighted_fit,
)

IDENTIFIABILITY_CONDITION_MAX = 1.0e5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extended_metadata(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(
            {
                "galaxy": line[0:11].strip(),
                "hubble_type": int(line[12:14]),
                "distance_method": int(line[28:29]),
                "effective_surface_brightness": float(line[62:70]),
                "disk_central_surface_brightness": float(line[77:85]),
            }
        )
    return pd.DataFrame(rows)


def robust_uncertainty(values: np.ndarray, errors: np.ndarray, floor: float = 5.0) -> float:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    count = len(values)
    return max(
        floor,
        float(np.median(errors)) / math.sqrt(count),
        1.4826 * mad / math.sqrt(count),
    )


def build_galaxy_summary(protocol: dict) -> pd.DataFrame:
    settings = protocol["sample"]
    inputs = protocol["input"]
    data_dir = ROOT / inputs["SPARC_directory"]
    metadata = extended_metadata(data_dir / "table1.dat").set_index("galaxy")
    rows = []
    for curve in load_curves(data_dir):
        meta = curve.metadata
        if (
            meta.quality > int(settings["maximum_quality_flag"])
            or meta.inclination_deg < float(settings["minimum_inclination_deg"])
        ):
            continue
        vbar = baryonic_speed(
            curve.velocity_gas_kms,
            curve.velocity_disk_unit_ml_kms,
            curve.velocity_bulge_unit_ml_kms,
            disk_mass_to_light=float(inputs["fixed_disk_mass_to_light"]),
            bulge_mass_to_light=float(inputs["fixed_bulge_mass_to_light"]),
        )
        vrar, gbar = rar_speed(
            curve.radius_kpc,
            vbar,
            acceleration_scale_m_s2=float(inputs["RAR_acceleration_scale_m_s2"]),
        )
        keep = (
            np.isfinite(vrar)
            & (vbar > 0.0)
            & (curve.radius_kpc >= float(settings["minimum_radius_in_disk_scale_lengths"]) * meta.disk_scale_kpc)
            & (gbar <= float(settings["maximum_baryonic_acceleration_m_s2"]))
            & (curve.velocity_error_kms > 0.0)
        )
        if int(np.sum(keep)) < int(settings["minimum_selected_points_per_galaxy"]):
            continue
        rar_residual = curve.velocity_observed_kms[keep] - vrar[keep]
        baryon_deficit = curve.velocity_observed_kms[keep] - vbar[keep]
        extra = metadata.loc[meta.name]
        rows.append(
            {
                "galaxy": meta.name,
                "distance_mpc": meta.distance_mpc,
                "distance_fractional_error": meta.distance_error_mpc / meta.distance_mpc,
                "distance_method": int(extra["distance_method"]),
                "quality": meta.quality,
                "inclination_deg": meta.inclination_deg,
                "hubble_type": int(extra["hubble_type"]),
                "effective_surface_brightness": float(extra["effective_surface_brightness"]),
                "disk_central_surface_brightness": float(extra["disk_central_surface_brightness"]),
                "selected_points": int(np.sum(keep)),
                "median_radius_kpc": float(np.median(curve.radius_kpc[keep])),
                "median_gbar_m_s2": float(np.median(gbar[keep])),
                "median_vbar_km_s": float(np.median(vbar[keep])),
                "median_vobs_km_s": float(np.median(curve.velocity_observed_kms[keep])),
                "median_baryon_deficit_km_s": float(np.median(baryon_deficit)),
                "median_RAR_residual_km_s": float(np.median(rar_residual)),
                "RAR_residual_uncertainty_km_s": robust_uncertainty(
                    rar_residual, curve.velocity_error_kms[keep]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("galaxy").reset_index(drop=True)


def fit_feature(frame: pd.DataFrame, feature: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    feature = np.asarray(feature, dtype=float)
    design = np.column_stack((np.ones(len(feature)), feature))
    if np.linalg.cond(design) > IDENTIFIABILITY_CONDITION_MAX:
        residual = frame["median_RAR_residual_km_s"].to_numpy(float)
        sigma = frame["RAR_residual_uncertainty_km_s"].to_numpy(float)
        weight = 1.0 / np.square(sigma)
        intercept = float(np.sum(weight * residual) / np.sum(weight))
        return (
            np.asarray([intercept, 0.0]),
            np.asarray([[1.0 / np.sum(weight), 0.0], [0.0, math.nan]]),
        )
    return weighted_fit(
        feature,
        frame["median_RAR_residual_km_s"].to_numpy(float),
        frame["RAR_residual_uncertainty_km_s"].to_numpy(float),
    )


def cross_validate_feature(
    frame: pd.DataFrame,
    feature: np.ndarray,
    *,
    folds: int,
    seed: int,
    label: str,
) -> pd.DataFrame:
    assignment = stable_source_folds(frame["galaxy"], folds, seed)
    rows = []
    for fold in range(folds):
        train = assignment != fold
        test = assignment == fold
        values, _ = fit_feature(frame[train], feature[train])
        prediction = values[0] + values[1] * feature[test]
        heldout = frame[test]
        for source, observed, sigma, predicted in zip(
            heldout["galaxy"],
            heldout["median_RAR_residual_km_s"],
            heldout["RAR_residual_uncertainty_km_s"],
            prediction,
            strict=True,
        ):
            rows.append(
                {
                    "galaxy": source,
                    "fold": fold,
                    "model": label,
                    "observed_km_s": observed,
                    "prediction_km_s": predicted,
                    "residual_km_s": observed - predicted,
                    "sigma_km_s": sigma,
                    "training_intercept_km_s": values[0],
                    "training_amplitude_km_s": values[1],
                }
            )
    return pd.DataFrame(rows)


def cv_metric(frame: pd.DataFrame) -> dict:
    residual = frame["residual_km_s"].to_numpy(float)
    sigma = frame["sigma_km_s"].to_numpy(float)
    return {
        "galaxies": int(len(frame)),
        "RMSE_km_s": float(np.sqrt(np.mean(np.square(residual)))),
        "MAE_km_s": float(np.mean(np.abs(residual))),
        "weighted_RMS_sigma": float(np.sqrt(np.mean(np.square(residual / sigma)))),
    }


def bootstrap_amplitude(
    frame: pd.DataFrame,
    feature: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for draw in range(draws):
        selected = rng.integers(0, len(frame), len(frame))
        values, _ = fit_feature(frame.iloc[selected], feature[selected])
        rows.append(
            {"draw": draw, "intercept_km_s": values[0], "amplitude_km_s": values[1]}
        )
    return pd.DataFrame(rows)


def make_figure(
    galaxies: pd.DataFrame,
    predictions: pd.DataFrame,
    scan: pd.DataFrame,
    log_values: np.ndarray,
    log_interval: list[float],
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    axes[0].errorbar(
        galaxies["distance_mpc"],
        galaxies["median_RAR_residual_km_s"],
        yerr=galaxies["RAR_residual_uncertainty_km_s"],
        fmt="o",
        alpha=0.55,
        markersize=4,
    )
    distance = np.geomspace(galaxies["distance_mpc"].min(), galaxies["distance_mpc"].max(), 200)
    axes[0].plot(
        distance,
        log_values[0] + log_values[1] * np.log(distance / 10.0),
        color="#D95F02",
        label=f"log-path A={log_values[1]:.1f}",
    )
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_xscale("log")
    axes[0].set(
        xlabel="galaxy distance (Mpc)",
        ylabel="outer fixed-RAR residual (km/s)",
        title="Does the anomaly accumulate?",
    )
    axes[0].legend(frameon=False, fontsize=8)

    metric_rows = []
    for model, block in predictions.groupby("model"):
        if model in {"null", "unsaturated_log_path"}:
            metric_rows.append((model, cv_metric(block)["weighted_RMS_sigma"]))
    axes[1].bar(
        [name.replace("_", "\n") for name, _ in metric_rows],
        [value for _, value in metric_rows],
    )
    axes[1].set(
        ylabel="held-out weighted RMS (sigma)",
        title="Distance law validation",
    )
    axes[1].tick_params(axis="x", labelsize=8)

    saturation = scan[scan["family"] == "saturating"].sort_values("scale")
    axes[2].plot(
        saturation["scale"],
        saturation["heldout_weighted_RMS_sigma"],
        marker="o",
    )
    axes[2].axhline(
        scan.loc[scan["model"] == "null", "heldout_weighted_RMS_sigma"].iloc[0],
        color="black",
        linestyle="--",
        label="distance-independent null",
    )
    axes[2].set_xscale("log")
    axes[2].set(
        xlabel="saturation length L (Mpc)",
        ylabel="held-out weighted RMS (sigma)",
        title="Could early saturation survive?",
    )
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=190)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs" / "photon_path_scaling_protocol.json",
    )
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_SPARC_distance_scoring":
        raise RuntimeError("protocol was not frozen before SPARC scoring")
    galaxies = build_galaxy_summary(protocol)
    folds = int(protocol["uncertainty"]["folds"])
    seed = int(protocol["uncertainty"]["fold_seed"])

    feature_specs = [("null", "null", None)]
    feature_specs.append(("unsaturated_log_path", "log", None))
    for exponent in protocol["models"]["power_path_exponents"]:
        feature_specs.append((f"power_q_{exponent:g}", "power", float(exponent)))
    for length in protocol["models"]["saturation_lengths_Mpc"]:
        feature_specs.append((f"saturating_L_{length:g}", "saturating", float(length)))

    predictions = []
    scan_rows = []
    for label, kind, scale in feature_specs:
        feature = (
            np.zeros(len(galaxies))
            if kind == "null"
            else path_feature(galaxies["distance_mpc"], kind=kind, scale=scale)
        )
        values, covariance = fit_feature(galaxies, feature)
        cv = cross_validate_feature(
            galaxies, feature, folds=folds, seed=seed, label=label
        )
        predictions.append(cv)
        metrics = cv_metric(cv)
        design_condition = float(
            np.linalg.cond(np.column_stack((np.ones(len(feature)), feature)))
        )
        scan_rows.append(
            {
                "model": label,
                "family": kind,
                "scale": scale,
                "intercept_km_s": values[0],
                "amplitude_km_s": values[1],
                "amplitude_formal_error_km_s": float(np.sqrt(covariance[1, 1]))
                if kind != "null"
                else math.nan,
                "design_condition_number": design_condition,
                "identifiable": design_condition <= IDENTIFIABILITY_CONDITION_MAX,
                "heldout_RMSE_km_s": metrics["RMSE_km_s"],
                "heldout_weighted_RMS_sigma": metrics["weighted_RMS_sigma"],
            }
        )
    predictions_frame = pd.concat(predictions, ignore_index=True)
    scan = pd.DataFrame(scan_rows)

    log_feature = path_feature(galaxies["distance_mpc"], kind="log")
    log_values, log_covariance = fit_feature(galaxies, log_feature)
    bootstrap = bootstrap_amplitude(
        galaxies,
        log_feature,
        draws=int(protocol["uncertainty"]["complete_galaxy_bootstrap_draws"]),
        seed=int(protocol["uncertainty"]["bootstrap_seed"]),
    )
    log_interval = [
        float(bootstrap["amplitude_km_s"].quantile(0.025)),
        float(bootstrap["amplitude_km_s"].quantile(0.975)),
    ]
    decade_growth = float(log_values[1] * math.log(10.0))
    decade_interval = [value * math.log(10.0) for value in log_interval]

    direct = galaxies[galaxies["distance_method"].isin([2, 3, 5])]
    direct_feature = path_feature(direct["distance_mpc"], kind="log")
    direct_values, direct_covariance = fit_feature(direct, direct_feature)
    primary_metric = cv_metric(
        predictions_frame[predictions_frame["model"] == "unsaturated_log_path"]
    )
    null_metric = cv_metric(predictions_frame[predictions_frame["model"] == "null"])
    gate = protocol["advance_gate"]
    improves = primary_metric["weighted_RMS_sigma"] < null_metric["weighted_RMS_sigma"]
    positive = decade_growth > 0.0
    reaches = (
        decade_interval[1]
        >= float(gate["minimum_growth_over_distance_decade_km_s"])
    )
    direct_positive = direct_values[1] * math.log(10.0) > 0.0
    survives = bool(improves and positive and reaches and direct_positive)

    saturation = scan[scan["family"] == "saturating"]
    identifiable_saturation = saturation[saturation["identifiable"]]
    best_saturation = identifiable_saturation.loc[
        identifiable_saturation["heldout_weighted_RMS_sigma"].idxmin()
    ]
    report = {
        "report_version": protocol["protocol_version"],
        "status": "completed SPARC photon path-scaling test",
        "protocol": {
            "path": str(protocol_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(protocol_path),
        },
        "sample": {
            "galaxies": int(len(galaxies)),
            "distance_range_Mpc": [
                float(galaxies["distance_mpc"].min()),
                float(galaxies["distance_mpc"].max()),
            ],
            "median_selected_points": float(galaxies["selected_points"].median()),
            "cuts": protocol["sample"],
        },
        "effective_metric": protocol["effective_metric"],
        "unsaturated_log_path": {
            "intercept_km_s": float(log_values[0]),
            "amplitude_km_s_per_ln_D_over_10": float(log_values[1]),
            "formal_standard_error_km_s": float(np.sqrt(log_covariance[1, 1])),
            "bootstrap_95_interval_km_s": log_interval,
            "growth_per_distance_decade_km_s": decade_growth,
            "growth_per_decade_bootstrap_95_interval_km_s": decade_interval,
            "bootstrap_probability_positive": float(
                np.mean(bootstrap["amplitude_km_s"] > 0.0)
            ),
            "cross_validated_metrics": primary_metric,
        },
        "distance_independent_null_cross_validated_metrics": null_metric,
        "direct_distance_control": {
            "methods": [2, 3, 5],
            "galaxies": int(len(direct)),
            "growth_per_distance_decade_km_s": float(
                direct_values[1] * math.log(10.0)
            ),
            "formal_standard_error_per_decade_km_s": float(
                np.sqrt(direct_covariance[1, 1]) * math.log(10.0)
            ),
        },
        "best_saturating_scan": {
            "length_Mpc": float(best_saturation["scale"]),
            "amplitude_km_s": float(best_saturation["amplitude_km_s"]),
            "heldout_weighted_RMS_sigma": float(
                best_saturation["heldout_weighted_RMS_sigma"]
            ),
            "identifiability_warning": "Lengths below the nearest sample distance behave almost like a constant and cannot be distinguished from the intercept.",
        },
        "unidentifiable_saturation_lengths_Mpc": list(
            map(
                float,
                saturation.loc[~saturation["identifiable"], "scale"],
            )
        ),
        "advance_gate": {
            "heldout_weighted_RMS_improves": bool(improves),
            "growth_per_decade_positive": bool(positive),
            "bootstrap_upper_reaches_30_km_s": bool(reaches),
            "direct_distance_subset_positive": bool(direct_positive),
            "unsaturated_path_illusion_survives": survives,
        },
        "interpretation_boundary": protocol["interpretation_boundary"],
    }

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    galaxies.to_csv(ROOT / protocol["outputs"]["galaxy_summary"], index=False)
    predictions_frame.to_csv(
        ROOT / protocol["outputs"]["cross_validated_predictions"], index=False
    )
    scan.to_csv(ROOT / protocol["outputs"]["model_scan"], index=False)
    bootstrap.to_csv(ROOT / protocol["outputs"]["bootstrap"], index=False)
    make_figure(
        galaxies,
        predictions_frame,
        scan,
        log_values,
        log_interval,
        ROOT / protocol["outputs"]["figure"],
    )
    (ROOT / protocol["outputs"]["report"]).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Photon path-scaling test",
        "",
        f"SPARC galaxies in the outer low-acceleration sample: **{len(galaxies)}**.",
        "",
        f"Growth over a factor of ten in distance: **{decade_growth:.3f} km/s** "
        f"(95% bootstrap {decade_interval[0]:.3f} to {decade_interval[1]:.3f}).",
        "",
        "| model | held-out weighted RMS (sigma) |",
        "|---|---:|",
        f"| distance-independent null | {null_metric['weighted_RMS_sigma']:.4f} |",
        f"| unsaturated log path | {primary_metric['weighted_RMS_sigma']:.4f} |",
        "",
        f"Unsaturated path illusion survives: **{survives}**.",
        "",
        "Early saturation remains unidentifiable if it occurs before the nearest galaxies.",
    ]
    (ROOT / protocol["outputs"]["summary"]).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
