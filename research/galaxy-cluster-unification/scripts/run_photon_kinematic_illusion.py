#!/usr/bin/env python3
"""Compare astrometric and spectroscopic rotation from the same masers."""

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
from astropy import units as u
from astropy.coordinates import SkyCoord

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.photon_kinematics import (  # noqa: E402
    circular_speed_from_channels,
    galactocentric_geometry,
    lsr_to_heliocentric_velocity,
    solar_galactocentric_velocity,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_catalog(path: Path, expected_rows: int) -> pd.DataFrame:
    raw = pd.read_csv(path, sep="\t", comment="#", dtype=str)
    raw["recno_numeric"] = pd.to_numeric(raw["recno"], errors="coerce")
    raw = raw[raw["recno_numeric"].notna()].copy()
    if len(raw) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} catalog rows, found {len(raw)}")
    numeric = [
        "plx",
        "e_plx",
        "pmE",
        "e_pmE",
        "pmN",
        "e_pmN",
        "VLSR",
        "e_VLSR",
    ]
    for column in numeric:
        raw[column] = pd.to_numeric(raw[column], errors="raise")
    raw["Arm"] = raw["Arm"].str.strip()
    coordinate = SkyCoord(
        raw["RAJ2000"].to_numpy(),
        raw["DEJ2000"].to_numpy(),
        unit=(u.hourangle, u.deg),
        pm_ra_cosdec=raw["pmE"].to_numpy(float) * u.mas / u.yr,
        pm_dec=raw["pmN"].to_numpy(float) * u.mas / u.yr,
        frame="icrs",
    ).galactic
    raw["l_deg"] = coordinate.l.deg
    raw["b_deg"] = coordinate.b.deg
    raw["pm_l_cosb_mas_yr"] = coordinate.pm_l_cosb.to_value(u.mas / u.yr)
    raw["pm_b_mas_yr"] = coordinate.pm_b.to_value(u.mas / u.yr)

    east_basis = SkyCoord(
        raw["RAJ2000"].to_numpy(),
        raw["DEJ2000"].to_numpy(),
        unit=(u.hourangle, u.deg),
        pm_ra_cosdec=np.ones(len(raw)) * u.mas / u.yr,
        pm_dec=np.zeros(len(raw)) * u.mas / u.yr,
        frame="icrs",
    ).galactic
    north_basis = SkyCoord(
        raw["RAJ2000"].to_numpy(),
        raw["DEJ2000"].to_numpy(),
        unit=(u.hourangle, u.deg),
        pm_ra_cosdec=np.zeros(len(raw)) * u.mas / u.yr,
        pm_dec=np.ones(len(raw)) * u.mas / u.yr,
        frame="icrs",
    ).galactic
    raw["pm_l_from_east"] = east_basis.pm_l_cosb.to_value(u.mas / u.yr)
    raw["pm_l_from_north"] = north_basis.pm_l_cosb.to_value(u.mas / u.yr)
    return raw.reset_index(drop=True)


def source_estimates(
    raw: pd.DataFrame,
    protocol: dict,
    *,
    peculiar_sigma: float | None = None,
) -> pd.DataFrame:
    constants = protocol["galactic_constants"]
    sample = protocol["primary_sample"]
    draws = int(protocol["uncertainty"]["monte_carlo_draws_per_source"])
    seed = int(protocol["uncertainty"]["monte_carlo_seed"])
    rng = np.random.default_rng(seed)
    r0 = float(constants["R0_kpc"])
    theta0 = float(constants["Theta0_km_s"])
    solar = solar_galactocentric_velocity(
        theta0, constants["solar_peculiar_UVW_km_s"]
    )
    standard_lsr = constants["catalog_LSR_standard_UVW_km_s"]
    conversion = float(constants["proper_motion_conversion_km_s_per_masyr_kpc"])
    peculiar = (
        float(sample["peculiar_velocity_sigma_per_channel_km_s"])
        if peculiar_sigma is None
        else float(peculiar_sigma)
    )
    rows = []
    for _, source in raw.iterrows():
        longitude = math.radians(float(source["l_deg"]))
        latitude = math.radians(float(source["b_deg"]))
        plx = float(source["plx"])
        distance = 1.0 / plx
        geometry = galactocentric_geometry(
            longitude, latitude, distance, solar_radius_kpc=r0
        )
        pm_l = float(source["pm_l_cosb_mas_yr"])
        v_longitude = conversion * pm_l * distance
        v_helio = lsr_to_heliocentric_velocity(
            float(source["VLSR"]),
            longitude,
            latitude,
            standard_lsr,
        )
        theta_pm, theta_rv = circular_speed_from_channels(
            geometry,
            transverse_longitude_velocity_km_s=v_longitude,
            heliocentric_radial_velocity_km_s=v_helio,
            solar_velocity_km_s=solar,
        )

        parallax_draw = rng.normal(plx, float(source["e_plx"]), draws)
        invalid = parallax_draw <= 0.0
        while np.any(invalid):
            parallax_draw[invalid] = rng.normal(
                plx, float(source["e_plx"]), int(np.sum(invalid))
            )
            invalid = parallax_draw <= 0.0
        distance_draw = 1.0 / parallax_draw
        pm_east = rng.normal(
            float(source["pmE"]), float(source["e_pmE"]), draws
        )
        pm_north = rng.normal(
            float(source["pmN"]), float(source["e_pmN"]), draws
        )
        pm_l_draw = (
            float(source["pm_l_from_east"]) * pm_east
            + float(source["pm_l_from_north"]) * pm_north
        )
        v_longitude_draw = (
            conversion * pm_l_draw * distance_draw
            + rng.normal(0.0, peculiar, draws)
        )
        v_lsr_draw = rng.normal(
            float(source["VLSR"]), float(source["e_VLSR"]), draws
        )
        v_helio_draw = lsr_to_heliocentric_velocity(
            v_lsr_draw, longitude, latitude, standard_lsr
        ) + rng.normal(0.0, peculiar, draws)
        geometry_draw = galactocentric_geometry(
            longitude, latitude, distance_draw, solar_radius_kpc=r0
        )
        theta_pm_draw, theta_rv_draw = circular_speed_from_channels(
            geometry_draw,
            transverse_longitude_velocity_km_s=v_longitude_draw,
            heliocentric_radial_velocity_km_s=v_helio_draw,
            solar_velocity_km_s=solar,
        )
        delta_draw = theta_rv_draw - theta_pm_draw
        rows.append(
            {
                "system": source["Name"].strip(),
                "arm": source["Arm"],
                "l_deg": float(source["l_deg"]),
                "b_deg": float(source["b_deg"]),
                "parallax_mas": plx,
                "parallax_error_mas": float(source["e_plx"]),
                "fractional_parallax_error": float(source["e_plx"]) / plx,
                "distance_kpc": distance,
                "radius_kpc": float(geometry["radius_kpc"]),
                "z_kpc": float(geometry["z_kpc"]),
                "radial_projection": float(geometry["radial_projection"]),
                "longitude_projection": float(geometry["longitude_projection"]),
                "v_longitude_km_s": float(v_longitude),
                "v_helio_radial_km_s": float(v_helio),
                "v_longitude_mc_median_km_s": float(
                    np.median(v_longitude_draw)
                ),
                "v_helio_radial_mc_median_km_s": float(
                    np.median(v_helio_draw)
                ),
                "v_longitude_mc_sigma_km_s": float(
                    np.std(v_longitude_draw, ddof=1)
                ),
                "v_helio_radial_mc_sigma_km_s": float(
                    np.std(v_helio_draw, ddof=1)
                ),
                "theta_pm_km_s": float(theta_pm),
                "theta_rv_km_s": float(theta_rv),
                "delta_theta_km_s": float(theta_rv - theta_pm),
                "theta_pm_mc_median_km_s": float(np.median(theta_pm_draw)),
                "theta_rv_mc_median_km_s": float(np.median(theta_rv_draw)),
                "delta_mc_median_km_s": float(np.median(delta_draw)),
                "delta_mc_sigma_km_s": float(np.std(delta_draw, ddof=1)),
                "delta_mc_p025_km_s": float(np.quantile(delta_draw, 0.025)),
                "delta_mc_p975_km_s": float(np.quantile(delta_draw, 0.975)),
                "peculiar_sigma_km_s": peculiar,
            }
        )
    return pd.DataFrame(rows)


def select_sample(
    frame: pd.DataFrame,
    protocol: dict,
    *,
    fractional_parallax_max: float | None = None,
    projection_minimum: float | None = None,
) -> pd.DataFrame:
    settings = protocol["primary_sample"]
    parallax_max = (
        float(settings["maximum_fractional_parallax_error"])
        if fractional_parallax_max is None
        else float(fractional_parallax_max)
    )
    projection = (
        float(settings["minimum_absolute_channel_projection"])
        if projection_minimum is None
        else float(projection_minimum)
    )
    keep = (
        (frame["fractional_parallax_error"] <= parallax_max)
        & (frame["radius_kpc"] >= float(settings["minimum_galactocentric_radius_kpc"]))
        & (frame["radius_kpc"] <= float(settings["maximum_galactocentric_radius_kpc"]))
        & (np.abs(frame["z_kpc"]) <= float(settings["maximum_absolute_height_kpc"]))
        & (~frame["arm"].isin(settings["excluded_arm_labels"]))
        & (np.abs(frame["radial_projection"]) >= projection)
        & (np.abs(frame["longitude_projection"]) >= projection)
        & np.isfinite(frame["delta_mc_sigma_km_s"])
        & (frame["delta_mc_sigma_km_s"] > 0.0)
    )
    return frame[keep].copy().reset_index(drop=True)


def stable_folds(names, folds: int, seed: int) -> np.ndarray:
    values = []
    for name in names:
        digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
        values.append(int.from_bytes(digest[:8], "big") % folds)
    return np.asarray(values, dtype=int)


def low_acceleration_gate(radius_kpc: np.ndarray, constants: dict) -> np.ndarray:
    radius_m = np.asarray(radius_kpc, dtype=float) * 3.085677581491367e19
    theta = float(constants["Theta0_km_s"]) * 1000.0
    acceleration = theta**2 / radius_m
    a_star = float(constants["a_star_m_s2"])
    return a_star / (a_star + acceleration)


def fit_amplitude(frame: pd.DataFrame, feature: np.ndarray) -> tuple[float, float]:
    delta = frame["delta_mc_median_km_s"].to_numpy(float)
    sigma = frame["delta_mc_sigma_km_s"].to_numpy(float)
    feature = np.asarray(feature, dtype=float)
    weight = 1.0 / np.square(sigma)
    denominator = float(np.sum(weight * np.square(feature)))
    amplitude = float(np.sum(weight * feature * delta) / denominator)
    uncertainty = float(1.0 / np.sqrt(denominator))
    return amplitude, uncertainty


def cross_validate(frame: pd.DataFrame, protocol: dict):
    folds = int(protocol["uncertainty"]["folds"])
    frame = frame.copy()
    frame["fold"] = stable_folds(
        frame["system"],
        folds,
        int(protocol["uncertainty"]["fold_seed"]),
    )
    predictions = []
    fits = []
    for fold in range(folds):
        training = frame[frame["fold"] != fold]
        heldout = frame[frame["fold"] == fold]
        features = {
            "null": (
                np.zeros(len(training)),
                np.zeros(len(heldout)),
                0.0,
                math.nan,
            ),
            "frequency_constant": (
                np.ones(len(training)),
                np.ones(len(heldout)),
                None,
                None,
            ),
            "frequency_low_acceleration": (
                low_acceleration_gate(
                    training["radius_kpc"].to_numpy(float),
                    protocol["galactic_constants"],
                ),
                low_acceleration_gate(
                    heldout["radius_kpc"].to_numpy(float),
                    protocol["galactic_constants"],
                ),
                None,
                None,
            ),
        }
        for model, (train_feature, test_feature, amplitude, uncertainty) in features.items():
            if amplitude is None:
                amplitude, uncertainty = fit_amplitude(training, train_feature)
            prediction = amplitude * test_feature
            block = heldout.copy()
            block["model"] = model
            block["prediction_km_s"] = prediction
            block["residual_km_s"] = (
                block["delta_mc_median_km_s"] - block["prediction_km_s"]
            )
            predictions.append(block)
            fits.append(
                {
                    "fold": fold,
                    "model": model,
                    "amplitude_km_s": amplitude,
                    "training_standard_error_km_s": uncertainty,
                    "training_sources": len(training),
                    "heldout_sources": len(heldout),
                }
            )
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(fits)


def model_metrics(frame: pd.DataFrame) -> dict:
    residual = frame["residual_km_s"].to_numpy(float)
    sigma = frame["delta_mc_sigma_km_s"].to_numpy(float)
    return {
        "sources": int(len(frame)),
        "RMSE_km_s": float(np.sqrt(np.mean(np.square(residual)))),
        "MAE_km_s": float(np.mean(np.abs(residual))),
        "weighted_RMS_sigma": float(np.sqrt(np.mean(np.square(residual / sigma)))),
        "mean_residual_km_s": float(np.mean(residual)),
    }


def bootstrap_amplitude(
    frame: pd.DataFrame,
    feature: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict:
    feature = np.asarray(feature, dtype=float)
    rng = np.random.default_rng(seed)
    values = np.empty(draws)
    for index in range(draws):
        sampled = rng.integers(0, len(frame), len(frame))
        values[index] = fit_amplitude(frame.iloc[sampled], feature[sampled])[0]
    observed, analytic = fit_amplitude(frame, feature)
    return {
        "amplitude_km_s": observed,
        "analytic_standard_error_km_s": analytic,
        "bootstrap_95_interval_km_s": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ],
        "probability_positive": float(np.mean(values > 0.0)),
        "draws": draws,
    }


def radial_trend(frame: pd.DataFrame, r0: float) -> dict:
    x = np.log(frame["radius_kpc"].to_numpy(float) / r0)
    y = frame["delta_mc_median_km_s"].to_numpy(float)
    sigma = frame["delta_mc_sigma_km_s"].to_numpy(float)
    design = np.column_stack((np.ones(len(frame)), x))
    weight = 1.0 / np.square(sigma)
    normal = design.T @ (weight[:, np.newaxis] * design)
    covariance = np.linalg.inv(normal)
    values = covariance @ design.T @ (weight * y)
    return {
        "A0_km_s": float(values[0]),
        "A1_km_s_per_ln_R_R0": float(values[1]),
        "A0_standard_error_km_s": float(np.sqrt(covariance[0, 0])),
        "A1_standard_error_km_s_per_ln_R_R0": float(np.sqrt(covariance[1, 1])),
    }


def sensitivity_table(all_estimates: pd.DataFrame, protocol: dict) -> pd.DataFrame:
    rows = []
    settings = protocol["sensitivity_checks"]
    for peculiar in settings["peculiar_velocity_sigmas_km_s"]:
        if peculiar == protocol["primary_sample"]["peculiar_velocity_sigma_per_channel_km_s"]:
            estimates = all_estimates
        else:
            raw_path = ROOT / protocol["data"]["catalog"]
            raw = load_catalog(raw_path, int(protocol["data"]["expected_rows"]))
            estimates = source_estimates(raw, protocol, peculiar_sigma=float(peculiar))
        for parallax in settings["maximum_fractional_parallax_errors"]:
            for projection in settings["minimum_absolute_channel_projections"]:
                sample = select_sample(
                    estimates,
                    protocol,
                    fractional_parallax_max=float(parallax),
                    projection_minimum=float(projection),
                )
                if len(sample) < 5:
                    continue
                amplitude, standard_error = fit_amplitude(
                    sample, np.ones(len(sample))
                )
                rows.append(
                    {
                        "peculiar_sigma_km_s": peculiar,
                        "maximum_fractional_parallax_error": parallax,
                        "minimum_absolute_channel_projection": projection,
                        "sources": len(sample),
                        "constant_amplitude_km_s": amplitude,
                        "analytic_standard_error_km_s": standard_error,
                    }
                )
    return pd.DataFrame(rows)


def make_figure(
    sample: pd.DataFrame,
    predictions: pd.DataFrame,
    fits: pd.DataFrame,
    report: dict,
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    axes[0].errorbar(
        sample["theta_pm_mc_median_km_s"],
        sample["theta_rv_mc_median_km_s"],
        xerr=sample["delta_mc_sigma_km_s"] / np.sqrt(2.0),
        yerr=sample["delta_mc_sigma_km_s"] / np.sqrt(2.0),
        fmt="o",
        alpha=0.58,
        markersize=4,
    )
    bounds = [
        min(axes[0].get_xlim()[0], axes[0].get_ylim()[0]),
        max(axes[0].get_xlim()[1], axes[0].get_ylim()[1]),
    ]
    axes[0].plot(bounds, bounds, color="black", linestyle="--")
    axes[0].set(
        xlabel="proper-motion circular speed (km/s)",
        ylabel="Doppler circular speed (km/s)",
        title="Same-source channel comparison",
    )

    axes[1].errorbar(
        sample["radius_kpc"],
        sample["delta_mc_median_km_s"],
        yerr=sample["delta_mc_sigma_km_s"],
        fmt="o",
        alpha=0.6,
        markersize=4,
    )
    amplitude = report["full_sample_amplitudes"]["frequency_constant"][
        "amplitude_km_s"
    ]
    interval = report["full_sample_amplitudes"]["frequency_constant"][
        "bootstrap_95_interval_km_s"
    ]
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].axhline(amplitude, color="#D95F02", label=f"fit A={amplitude:.1f}")
    axes[1].axhspan(interval[0], interval[1], color="#D95F02", alpha=0.18)
    axes[1].axhline(
        report["advance_gate"]["minimum_required_positive_amplitude_km_s"],
        color="#2E8B57",
        linestyle=":",
        label="minimum illusion target",
    )
    axes[1].set(
        xlabel="Galactocentric radius (kpc)",
        ylabel=r"$\Theta_{\rm Doppler}-\Theta_{\rm PM}$ (km/s)",
        title="Frequency-only anomaly",
    )
    axes[1].legend(frameon=False, fontsize=8)

    metric_rows = []
    for model, block in predictions.groupby("model", sort=True):
        metric_rows.append((model, model_metrics(block)["RMSE_km_s"]))
    axes[2].bar(
        [name.replace("_", "\n") for name, _ in metric_rows],
        [value for _, value in metric_rows],
    )
    axes[2].set(ylabel="held-out RMSE (km/s)", title="Complete-source validation")
    axes[2].tick_params(axis="x", labelsize=8)
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
        default=ROOT / "configs" / "photon_kinematic_illusion_protocol.json",
    )
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_maser_channel_comparison":
        raise RuntimeError("protocol was not frozen before maser scoring")
    catalog_path = ROOT / protocol["data"]["catalog"]
    provenance_path = ROOT / protocol["data"]["provenance"]
    raw = load_catalog(catalog_path, int(protocol["data"]["expected_rows"]))
    all_estimates = source_estimates(raw, protocol)
    sample = select_sample(all_estimates, protocol)
    if len(sample) < 10:
        raise RuntimeError("primary geometry cuts left too few masers")
    predictions, fold_fits = cross_validate(sample, protocol)
    metrics = {
        model: model_metrics(block)
        for model, block in predictions.groupby("model", sort=True)
    }
    uncertainty = protocol["uncertainty"]
    full_amplitudes = {
        "frequency_constant": bootstrap_amplitude(
            sample,
            np.ones(len(sample)),
            draws=int(uncertainty["complete_source_bootstrap_draws"]),
            seed=int(uncertainty["bootstrap_seed"]),
        ),
        "frequency_low_acceleration": bootstrap_amplitude(
            sample,
            low_acceleration_gate(
                sample["radius_kpc"].to_numpy(float),
                protocol["galactic_constants"],
            ),
            draws=int(uncertainty["complete_source_bootstrap_draws"]),
            seed=int(uncertainty["bootstrap_seed"]) + 1,
        ),
    }
    sensitivity = sensitivity_table(all_estimates, protocol)
    gate = protocol["frequency_only_advance_gate"]
    constant = full_amplitudes["frequency_constant"]
    cv_improves = (
        metrics["frequency_constant"]["RMSE_km_s"] < metrics["null"]["RMSE_km_s"]
    )
    interval_reaches = (
        constant["bootstrap_95_interval_km_s"][1]
        >= float(gate["minimum_required_positive_amplitude_km_s"])
    )
    positive = constant["amplitude_km_s"] > 0.0
    survives = bool(cv_improves and interval_reaches and positive)
    report = {
        "report_version": protocol["protocol_version"],
        "status": "completed same-source photon frequency-illusion test",
        "protocol": {
            "path": str(protocol_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(protocol_path),
        },
        "inputs": {
            "catalog": str(catalog_path.relative_to(ROOT)).replace("\\", "/"),
            "catalog_sha256": sha256(catalog_path),
            "provenance_sha256": sha256(provenance_path),
            "catalog_rows": len(raw),
        },
        "new_physics_framework": protocol["new_physics_framework"],
        "primary_sample": {
            "sources": len(sample),
            "radius_range_kpc": [
                float(sample["radius_kpc"].min()),
                float(sample["radius_kpc"].max()),
            ],
            "median_delta_theta_km_s": float(
                sample["delta_mc_median_km_s"].median()
            ),
            "median_delta_uncertainty_km_s": float(
                sample["delta_mc_sigma_km_s"].median()
            ),
            "cuts": protocol["primary_sample"],
        },
        "cross_validated_metrics": metrics,
        "full_sample_amplitudes": full_amplitudes,
        "radial_trend_diagnostic": radial_trend(
            sample, float(protocol["galactic_constants"]["R0_kpc"])
        ),
        "parameter_stability": {
            model: {
                "fold_amplitudes_km_s": list(
                    map(float, block.sort_values("fold")["amplitude_km_s"])
                ),
                "minimum_km_s": float(block["amplitude_km_s"].min()),
                "maximum_km_s": float(block["amplitude_km_s"].max()),
            }
            for model, block in fold_fits.groupby("model", sort=True)
        },
        "advance_gate": {
            "minimum_required_positive_amplitude_km_s": float(
                gate["minimum_required_positive_amplitude_km_s"]
            ),
            "cross_validated_improvement": cv_improves,
            "bootstrap_interval_reaches_required_amplitude": interval_reaches,
            "fitted_sign_positive": positive,
            "frequency_only_illusion_survives": survives,
        },
        "broader_optical_metric_status": {
            "remains_logically_open": True,
            "condition": protocol["observable_models"]["broader_optical_metric"],
            "next_discriminator": "Test annual-parallax ellipse versus secular-motion Jacobians and multi-frequency same-source astrometry; a static common angular remapping cancels from mu/parallax.",
        },
        "sensitivity_ranges": {
            "amplitude_min_km_s": float(sensitivity["constant_amplitude_km_s"].min()),
            "amplitude_max_km_s": float(sensitivity["constant_amplitude_km_s"].max()),
            "rows": len(sensitivity),
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    all_estimates["primary_sample"] = all_estimates["system"].isin(sample["system"])
    all_estimates.to_csv(ROOT / protocol["outputs"]["source_estimates"], index=False)
    predictions.to_csv(
        ROOT / protocol["outputs"]["cross_validated_predictions"], index=False
    )
    fold_fits.to_csv(ROOT / protocol["outputs"]["fold_fits"], index=False)
    sensitivity.to_csv(ROOT / protocol["outputs"]["sensitivity"], index=False)
    make_figure(
        sample,
        predictions,
        fold_fits,
        report,
        ROOT / protocol["outputs"]["figure"],
    )
    (ROOT / protocol["outputs"]["report"]).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Photon kinematic illusion",
        "",
        f"Primary same-source masers: **{len(sample)}**.",
        "",
        "| model | held-out RMSE (km/s) |",
        "|---|---:|",
    ]
    for model in ("null", "frequency_constant", "frequency_low_acceleration"):
        lines.append(f"| {model} | {metrics[model]['RMSE_km_s']:.3f} |")
    lines.extend(
        [
            "",
            f"Constant spectral amplitude: **{constant['amplitude_km_s']:.3f} km/s** "
            f"(95% bootstrap {constant['bootstrap_95_interval_km_s'][0]:.3f} to "
            f"{constant['bootstrap_95_interval_km_s'][1]:.3f}).",
            "",
            f"Frequency-only illusion survives: **{survives}**.",
            "",
            "The broader time-dependent/direction-dependent optical-metric branch remains open.",
        ]
    )
    (ROOT / protocol["outputs"]["summary"]).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
