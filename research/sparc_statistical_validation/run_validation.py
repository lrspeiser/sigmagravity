"""Preregistered galaxy-grouped SPARC statistical validation.

This module deliberately reimplements only the submitted baseline equations so
the research audit stays isolated from the manuscript and production regression.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"
ROTMOD_DIR = ROOT / "data" / "Rotmod_LTG"
TABLE1_PATH = ROOT / "data" / "sparc" / "Table1_SPARC.dat"
RDISK_PATH = ROOT / "data" / "sparc" / "sparc_true_rdisk.csv"

C_M_S = 2.998e8
H0_SI = 2.27e-18
KPC_M = 3.086e19
G_DAGGER = C_M_S * H0_SI / (4.0 * math.sqrt(math.pi))
A0 = math.exp(1.0 / (2.0 * math.pi))
A0_MOND = 1.2e-10
SEED = 20260718
BOOTSTRAPS = 20_000
SIGN_FLIPS = 50_000


@dataclass(frozen=True)
class Metadata:
    name: str
    distance_mpc: float
    distance_error_mpc: float
    inclination_deg: float
    inclination_error_deg: float
    rdisk_kpc: float
    quality: int


@dataclass(frozen=True)
class RawCurve:
    name: str
    radius_kpc: np.ndarray
    velocity_observed_kms: np.ndarray
    velocity_error_kms: np.ndarray
    velocity_gas_kms: np.ndarray
    velocity_disk_unit_ml_kms: np.ndarray
    velocity_bulge_unit_ml_kms: np.ndarray
    metadata: Metadata


def h_function(g_newton: np.ndarray | float) -> np.ndarray:
    g = np.maximum(np.asarray(g_newton, dtype=float), 1e-15)
    return np.sqrt(G_DAGGER / g) * G_DAGGER / (G_DAGGER + g)


def predict_sigma(
    radius_kpc: np.ndarray,
    velocity_bar_kms: np.ndarray,
    sigma_kms: float = 20.0,
    amplitude_scale: float = 1.0,
    rdisk_kpc: float | None = None,
    l0_kpc: float | None = None,
    n_exponent: float | None = None,
) -> np.ndarray:
    """Submitted fixed-point model; the final three arguments are inert by design."""
    del rdisk_kpc, l0_kpc, n_exponent
    radius_m = np.asarray(radius_kpc, dtype=float) * KPC_M
    velocity_bar = np.asarray(velocity_bar_kms, dtype=float)
    g_bar = (velocity_bar * 1000.0) ** 2 / radius_m
    h_value = h_function(g_bar)
    velocity = velocity_bar.copy()
    for _ in range(50):
        coherence = velocity**2 / (velocity**2 + sigma_kms**2)
        enhancement = 1.0 + (A0 * amplitude_scale) * coherence * h_value
        velocity_new = velocity_bar * np.sqrt(enhancement)
        if np.max(np.abs(velocity_new - velocity)) < 1e-6:
            break
        velocity = velocity_new
    return velocity


def predict_acceleration_only(
    radius_kpc: np.ndarray,
    velocity_bar_kms: np.ndarray,
    amplitude_scale: float = 1.0,
) -> np.ndarray:
    radius_m = np.asarray(radius_kpc, dtype=float) * KPC_M
    velocity_bar = np.asarray(velocity_bar_kms, dtype=float)
    g_bar = (velocity_bar * 1000.0) ** 2 / radius_m
    return velocity_bar * np.sqrt(1.0 + A0 * amplitude_scale * h_function(g_bar))


def predict_mond(radius_kpc: np.ndarray, velocity_bar_kms: np.ndarray) -> np.ndarray:
    radius_m = np.asarray(radius_kpc, dtype=float) * KPC_M
    velocity_bar = np.asarray(velocity_bar_kms, dtype=float)
    g_bar = (velocity_bar * 1000.0) ** 2 / radius_m
    x = g_bar / A0_MOND
    nu = 1.0 / (1.0 - np.exp(-np.sqrt(np.maximum(x, 1e-10))))
    return velocity_bar * np.sqrt(nu)


def parse_table1(path: Path = TABLE1_PATH) -> dict[str, Metadata]:
    rows: dict[str, Metadata] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            name = line[0:11].strip()
            rows[name] = Metadata(
                name=name,
                distance_mpc=float(line[15:21]),
                distance_error_mpc=float(line[22:27]),
                inclination_deg=float(line[30:34]),
                inclination_error_deg=float(line[35:39]),
                rdisk_kpc=float(line[71:76]),
                quality=int(line[112:115]),
            )
    return rows


def load_raw_curves(
    rotmod_dir: Path = ROTMOD_DIR,
    table1_path: Path = TABLE1_PATH,
) -> list[RawCurve]:
    metadata = parse_table1(table1_path)
    curves: list[RawCurve] = []
    for path in sorted(rotmod_dir.glob("*_rotmod.dat")):
        name = path.stem.replace("_rotmod", "")
        if name not in metadata:
            raise KeyError(f"No SPARC Table 1 metadata for {name}")
        rows: list[list[float]] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) < 6:
                    continue
                try:
                    rows.append([float(value) for value in parts[:6]])
                except ValueError:
                    continue
        if len(rows) < 5:
            continue
        values = np.asarray(rows, dtype=float)
        curves.append(
            RawCurve(
                name=name,
                radius_kpc=values[:, 0],
                velocity_observed_kms=values[:, 1],
                velocity_error_kms=values[:, 2],
                velocity_gas_kms=values[:, 3],
                velocity_disk_unit_ml_kms=values[:, 4],
                velocity_bulge_unit_ml_kms=values[:, 5],
                metadata=metadata[name],
            )
        )
    return curves


def prepare_curve(
    curve: RawCurve,
    ml_disk: float,
    ml_bulge: float,
    distance_scale: float,
    inclination_offset_deg: float,
) -> dict[str, np.ndarray] | None:
    component_scale = math.sqrt(distance_scale)
    radius = curve.radius_kpc * distance_scale
    gas = curve.velocity_gas_kms * component_scale
    disk = curve.velocity_disk_unit_ml_kms * component_scale * math.sqrt(ml_disk)
    bulge = curve.velocity_bulge_unit_ml_kms * component_scale * math.sqrt(ml_bulge)
    velocity_bar_squared = np.sign(gas) * gas**2 + disk**2 + bulge**2
    velocity_bar = np.sqrt(np.abs(velocity_bar_squared)) * np.sign(velocity_bar_squared)

    inclination_new = float(
        np.clip(curve.metadata.inclination_deg + inclination_offset_deg, 10.0, 90.0)
    )
    inclination_factor = math.sin(math.radians(curve.metadata.inclination_deg)) / math.sin(
        math.radians(inclination_new)
    )
    velocity_observed = curve.velocity_observed_kms * inclination_factor
    velocity_error = curve.velocity_error_kms * inclination_factor

    valid = (velocity_bar > 0) & (radius > 0) & (velocity_observed > 0)
    if int(np.sum(valid)) < 5:
        return None
    radius = radius[valid]
    velocity_bar = velocity_bar[valid]
    velocity_observed = velocity_observed[valid]
    velocity_error = velocity_error[valid]
    bulge = bulge[valid]
    velocity_bar_squared = velocity_bar**2

    bulge_fraction = bulge**2 / np.maximum(velocity_bar_squared, 1e-10)
    disk_mask = bulge_fraction < 0.3
    if int(np.sum(disk_mask)) < 3:
        return None
    return {
        "radius": radius[disk_mask],
        "velocity_bar": velocity_bar[disk_mask],
        "velocity_observed": velocity_observed[disk_mask],
        "velocity_error": velocity_error[disk_mask],
        "n_valid": np.asarray([len(radius)]),
        "n_excluded_bulge": np.asarray([int(np.sum(~disk_mask))]),
    }


def evaluate_configuration(
    curves: Iterable[RawCurve],
    ml_disk: float = 0.5,
    ml_bulge: float = 0.7,
    distance_scale: float = 1.0,
    inclination_offset_deg: float = 0.0,
    amplitude_scale: float = 1.0,
    sigma_kms: float = 20.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    galaxy_rows: list[dict[str, float | int | str]] = []
    residuals_sigma: list[np.ndarray] = []
    residuals_acceleration: list[np.ndarray] = []
    residuals_mond: list[np.ndarray] = []
    total_valid = 0
    total_excluded = 0

    for curve in curves:
        prepared = prepare_curve(
            curve, ml_disk, ml_bulge, distance_scale, inclination_offset_deg
        )
        if prepared is None:
            continue
        radius = prepared["radius"]
        velocity_bar = prepared["velocity_bar"]
        observed = prepared["velocity_observed"]
        error = np.maximum(prepared["velocity_error"], 1e-12)
        sigma_prediction = predict_sigma(
            radius,
            velocity_bar,
            sigma_kms=sigma_kms,
            amplitude_scale=amplitude_scale,
            rdisk_kpc=curve.metadata.rdisk_kpc,
        )
        acceleration_prediction = predict_acceleration_only(
            radius, velocity_bar, amplitude_scale=amplitude_scale
        )
        mond_prediction = predict_mond(radius, velocity_bar)
        sigma_residual = sigma_prediction - observed
        acceleration_residual = acceleration_prediction - observed
        mond_residual = mond_prediction - observed
        residuals_sigma.append(sigma_residual)
        residuals_acceleration.append(acceleration_residual)
        residuals_mond.append(mond_residual)
        total_valid += int(prepared["n_valid"][0])
        total_excluded += int(prepared["n_excluded_bulge"][0])
        rms_sigma = float(np.sqrt(np.mean(sigma_residual**2)))
        rms_acceleration = float(np.sqrt(np.mean(acceleration_residual**2)))
        rms_mond = float(np.sqrt(np.mean(mond_residual**2)))
        galaxy_rows.append(
            {
                "name": curve.name,
                "quality": curve.metadata.quality,
                "distance_mpc": curve.metadata.distance_mpc,
                "distance_error_mpc": curve.metadata.distance_error_mpc,
                "inclination_deg": curve.metadata.inclination_deg,
                "inclination_error_deg": curve.metadata.inclination_error_deg,
                "rdisk_kpc": curve.metadata.rdisk_kpc,
                "n_disk_points": len(radius),
                "n_bulge_points_excluded": int(prepared["n_excluded_bulge"][0]),
                "rms_sigma_kms": rms_sigma,
                "rms_acceleration_only_kms": rms_acceleration,
                "rms_mond_kms": rms_mond,
                "delta_sigma_minus_mond_kms": rms_sigma - rms_mond,
                "delta_sigma_minus_acceleration_kms": rms_sigma - rms_acceleration,
                "chi2_per_point_sigma": float(np.mean((sigma_residual / error) ** 2)),
                "chi2_per_point_acceleration_only": float(
                    np.mean((acceleration_residual / error) ** 2)
                ),
                "chi2_per_point_mond": float(np.mean((mond_residual / error) ** 2)),
            }
        )

    frame = pd.DataFrame(galaxy_rows).sort_values("name").reset_index(drop=True)
    sigma_all = np.concatenate(residuals_sigma)
    acceleration_all = np.concatenate(residuals_acceleration)
    mond_all = np.concatenate(residuals_mond)
    summary = {
        "n_galaxies": int(len(frame)),
        "n_disk_points": int(sum(frame["n_disk_points"])),
        "n_valid_points_before_bulge_cut": total_valid,
        "n_bulge_points_excluded": total_excluded,
        "mean_rms_sigma_kms": float(frame["rms_sigma_kms"].mean()),
        "mean_rms_acceleration_only_kms": float(
            frame["rms_acceleration_only_kms"].mean()
        ),
        "mean_rms_mond_kms": float(frame["rms_mond_kms"].mean()),
        "median_delta_sigma_minus_mond_kms": float(
            frame["delta_sigma_minus_mond_kms"].median()
        ),
        "mean_delta_sigma_minus_mond_kms": float(
            frame["delta_sigma_minus_mond_kms"].mean()
        ),
        "sigma_win_fraction": float(
            np.mean(frame["delta_sigma_minus_mond_kms"].to_numpy() < 0)
        ),
        "pooled_rmse_sigma_kms": float(np.sqrt(np.mean(sigma_all**2))),
        "pooled_rmse_acceleration_only_kms": float(
            np.sqrt(np.mean(acceleration_all**2))
        ),
        "pooled_rmse_mond_kms": float(np.sqrt(np.mean(mond_all**2))),
    }
    return frame, summary


def bootstrap_paired(
    contrast: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> dict[str, object]:
    samples = contrast[bootstrap_indices]
    boot_means = np.mean(samples, axis=1)
    boot_medians = np.median(samples, axis=1)
    boot_wins = np.mean(samples < 0, axis=1)
    wins = int(np.sum(contrast < 0))
    losses = int(np.sum(contrast > 0))
    exact_p = float(binomtest(wins, wins + losses, p=0.5).pvalue) if wins + losses else 1.0
    return {
        "n": int(len(contrast)),
        "mean": float(np.mean(contrast)),
        "mean_ci95": [float(x) for x in np.percentile(boot_means, [2.5, 97.5])],
        "median": float(np.median(contrast)),
        "median_ci95": [float(x) for x in np.percentile(boot_medians, [2.5, 97.5])],
        "win_fraction": float(np.mean(contrast < 0)),
        "win_fraction_ci95": [float(x) for x in np.percentile(boot_wins, [2.5, 97.5])],
        "wins": wins,
        "losses": losses,
        "ties": int(np.sum(contrast == 0)),
        "exact_binomial_p_two_sided": exact_p,
    }


def sign_flip_pvalue(
    contrast: np.ndarray,
    n_flips: int = SIGN_FLIPS,
    seed: int = SEED + 1,
) -> float:
    rng = np.random.default_rng(seed)
    observed = abs(float(np.mean(contrast)))
    extreme = 0
    completed = 0
    batch_size = 2_000
    while completed < n_flips:
        current = min(batch_size, n_flips - completed)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(current, len(contrast)))
        null_means = np.mean(signs * contrast, axis=1)
        extreme += int(np.sum(np.abs(null_means) >= observed))
        completed += current
    return float((extreme + 1) / (n_flips + 1))


def nuisance_grid(curves: list[RawCurve]) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for ml_pair, distance_scale, inclination_offset, amplitude_scale in itertools.product(
        [(0.3, 0.5), (0.5, 0.7), (0.7, 0.9)],
        [0.9, 1.0, 1.1],
        [-5.0, 0.0, 5.0],
        [0.9, 1.0, 1.1],
    ):
        _, summary = evaluate_configuration(
            curves,
            ml_disk=ml_pair[0],
            ml_bulge=ml_pair[1],
            distance_scale=distance_scale,
            inclination_offset_deg=inclination_offset,
            amplitude_scale=amplitude_scale,
        )
        rows.append(
            {
                "ml_disk": ml_pair[0],
                "ml_bulge": ml_pair[1],
                "distance_scale": distance_scale,
                "inclination_offset_deg": inclination_offset,
                "amplitude_scale": amplitude_scale,
                **summary,
            }
        )
    return pd.DataFrame(rows)


def sigma_sensitivity(curves: list[RawCurve]) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for sigma_kms in [10.0, 20.0, 30.0, 50.0]:
        _, summary = evaluate_configuration(curves, sigma_kms=sigma_kms)
        rows.append({"coherence_sigma_kms": sigma_kms, **summary})
    return pd.DataFrame(rows)


def quality_strata(primary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for quality, frame in primary.groupby("quality"):
        contrast = frame["delta_sigma_minus_mond_kms"].to_numpy()
        rows.append(
            {
                "quality": int(quality),
                "n_galaxies": int(len(frame)),
                "mean_rms_sigma_kms": float(frame["rms_sigma_kms"].mean()),
                "mean_rms_mond_kms": float(frame["rms_mond_kms"].mean()),
                "mean_delta_sigma_minus_mond_kms": float(np.mean(contrast)),
                "median_delta_sigma_minus_mond_kms": float(np.median(contrast)),
                "sigma_win_fraction": float(np.mean(contrast < 0)),
            }
        )
    return pd.DataFrame(rows)


def make_figure(primary: pd.DataFrame, nuisance: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    scatter = axes[0].scatter(
        primary["rms_mond_kms"],
        primary["rms_sigma_kms"],
        c=primary["quality"],
        cmap="viridis_r",
        alpha=0.8,
        edgecolor="none",
    )
    maximum = float(
        max(primary["rms_mond_kms"].max(), primary["rms_sigma_kms"].max())
    )
    axes[0].plot([0, maximum], [0, maximum], "k--", linewidth=1)
    axes[0].set(xlabel="MOND RMS [km/s]", ylabel="Σ RMS [km/s]", title="Paired galaxies")
    colorbar = figure.colorbar(scatter, ax=axes[0], fraction=0.046, pad=0.04)
    colorbar.set_label("SPARC quality")

    delta = primary["delta_sigma_minus_mond_kms"]
    axes[1].hist(delta, bins=25, color="#4c78a8", alpha=0.85)
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].axvline(delta.mean(), color="#d62728", linewidth=2, label="mean")
    axes[1].set(
        xlabel="Σ RMS − MOND RMS [km/s]",
        ylabel="Galaxies",
        title="Primary paired contrast",
    )
    axes[1].legend()

    axes[2].hist(
        nuisance["mean_delta_sigma_minus_mond_kms"],
        bins=18,
        color="#f58518",
        alpha=0.85,
    )
    axes[2].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[2].set(
        xlabel="Mean paired contrast [km/s]",
        ylabel="Nuisance configurations",
        title="81 frozen global diagnostics",
    )
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    curves = load_raw_curves()
    primary, primary_summary = evaluate_configuration(curves)
    primary.to_csv(RESULTS / "per_galaxy_primary.csv", index=False)

    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(primary), size=(BOOTSTRAPS, len(primary)))
    sigma_mond = primary["delta_sigma_minus_mond_kms"].to_numpy()
    sigma_acceleration = primary["delta_sigma_minus_acceleration_kms"].to_numpy()
    bootstrap = {
        "seed": SEED,
        "bootstrap_resamples": BOOTSTRAPS,
        "sign_flip_resamples": SIGN_FLIPS,
        "sigma_minus_mond": bootstrap_paired(sigma_mond, indices),
        "sigma_minus_acceleration_only": bootstrap_paired(sigma_acceleration, indices),
    }
    bootstrap["sigma_minus_mond"]["sign_flip_p_two_sided"] = sign_flip_pvalue(
        sigma_mond
    )
    bootstrap["sigma_minus_acceleration_only"][
        "sign_flip_p_two_sided"
    ] = sign_flip_pvalue(sigma_acceleration, seed=SEED + 2)
    (RESULTS / "bootstrap_summary.json").write_text(
        json.dumps(bootstrap, indent=2) + "\n", encoding="utf-8"
    )

    nuisance = nuisance_grid(curves)
    nuisance.to_csv(RESULTS / "nuisance_grid.csv", index=False)
    sensitivity = sigma_sensitivity(curves)
    sensitivity.to_csv(RESULTS / "coherence_sigma_sensitivity.csv", index=False)
    strata = quality_strata(primary)
    strata.to_csv(RESULTS / "quality_strata.csv", index=False)

    sm = bootstrap["sigma_minus_mond"]
    sa = bootstrap["sigma_minus_acceleration_only"]
    if sm["mean_ci95"][1] < 0 and sm["win_fraction_ci95"][0] > 0.5:
        comparison_verdict = "sigma_superior"
    elif sm["mean_ci95"][0] > 0 and sm["win_fraction_ci95"][1] < 0.5:
        comparison_verdict = "mond_superior"
    else:
        comparison_verdict = "comparable"
    coherence_improves = bool(sa["mean_ci95"][1] < 0)
    nuisance_signs = np.sign(nuisance["mean_delta_sigma_minus_mond_kms"].to_numpy())
    central_sign = float(np.sign(primary_summary["mean_delta_sigma_minus_mond_kms"]))
    sign_stability = float(np.mean(nuisance_signs == central_sign))
    sensitivity_sign_stable = bool(
        np.all(
            np.sign(sensitivity["mean_delta_sigma_minus_mond_kms"].to_numpy())
            == central_sign
        )
    )
    rdisk_catalog = pd.read_csv(RDISK_PATH)
    rdisk_matches = int(primary["name"].isin(rdisk_catalog["Name"]).sum())
    decision = {
        "design_frozen_before_results": True,
        "central_summary": primary_summary,
        "sigma_vs_mond_verdict": comparison_verdict,
        "submitted_coherence_improves_acceleration_only": coherence_improves,
        "coherence_caveat": (
            "The submitted C is computed from model-predicted velocity and is not an "
            "independently measured kinematic variable."
        ),
        "nuisance_grid_cases": int(len(nuisance)),
        "nuisance_mean_delta_range_kms": [
            float(nuisance["mean_delta_sigma_minus_mond_kms"].min()),
            float(nuisance["mean_delta_sigma_minus_mond_kms"].max()),
        ],
        "nuisance_sign_stability_fraction": sign_stability,
        "coherence_sigma_sign_stable": sensitivity_sign_stable,
        "rdisk_catalog_matches": rdisk_matches,
        "rdisk_l0_n_structurally_inert": True,
        "bootstrap": bootstrap,
    }
    (RESULTS / "decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    make_figure(primary, nuisance, RESULTS / "sparc_paired_diagnostics.png")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
