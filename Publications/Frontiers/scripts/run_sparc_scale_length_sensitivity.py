"""Reproduce the SPARC scale-length and weighting sensitivities for revision.

This script is deliberately isolated from the manuscript builder. It evaluates
fixed formulas only, writes machine-readable outputs, and does not tune any
model parameter or modify manuscript sources.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


HERE = Path(__file__).resolve().parent
FRONTIERS = HERE.parent
ROOT = FRONTIERS.parents[1]
OUTPUT = FRONTIERS / "analysis" / "sparc_scale_length"
ROTMOD = ROOT / "data" / "Rotmod_LTG"
TABLE1 = ROOT / "data" / "sparc" / "Table1_SPARC.dat"

C_M_S = 2.998e8
H0_SI = 2.27e-18
KPC_M = 3.086e19
G_DAGGER = C_M_S * H0_SI / (4.0 * math.sqrt(math.pi))
A0 = math.exp(1.0 / (2.0 * math.pi))
A0_MOND = 1.2e-10
SEED = 20260725
BOOTSTRAPS = 20_000
SIGN_FLIPS = 50_000
PERMUTATIONS = 2_000


@dataclass(frozen=True)
class Curve:
    name: str
    quality: int
    rdisk_kpc: float
    radius_kpc: np.ndarray
    observed_kms: np.ndarray
    error_kms: np.ndarray
    gas_kms: np.ndarray
    disk_unit_ml_kms: np.ndarray
    bulge_unit_ml_kms: np.ndarray


def h_function(g_newton: np.ndarray) -> np.ndarray:
    g_value = np.maximum(np.asarray(g_newton, dtype=float), 1e-15)
    return np.sqrt(G_DAGGER / g_value) * G_DAGGER / (G_DAGGER + g_value)


def predict_locked(radius_kpc: np.ndarray, velocity_bar_kms: np.ndarray) -> np.ndarray:
    radius_m = np.asarray(radius_kpc, dtype=float) * KPC_M
    velocity_bar = np.asarray(velocity_bar_kms, dtype=float)
    g_bar = (velocity_bar * 1000.0) ** 2 / radius_m
    h_value = h_function(g_bar)
    velocity = velocity_bar.copy()
    for _ in range(50):
        bounded = velocity**2 / (velocity**2 + 20.0**2)
        updated = velocity_bar * np.sqrt(1.0 + A0 * bounded * h_value)
        if np.max(np.abs(updated - velocity)) < 1e-6:
            return updated
        velocity = updated
    raise RuntimeError("Locked SPARC fixed point did not converge")


def window(radius_kpc: np.ndarray, rdisk_kpc: float) -> np.ndarray:
    if not np.isfinite(rdisk_kpc) or rdisk_kpc <= 0:
        raise ValueError("A positive finite catalog disk scale length is required")
    radius = np.asarray(radius_kpc, dtype=float)
    return radius / (rdisk_kpc / (2.0 * math.pi) + radius)


def predict_window(
    radius_kpc: np.ndarray,
    velocity_bar_kms: np.ndarray,
    rdisk_kpc: float,
) -> np.ndarray:
    radius = np.asarray(radius_kpc, dtype=float)
    velocity_bar = np.asarray(velocity_bar_kms, dtype=float)
    g_bar = (velocity_bar * 1000.0) ** 2 / (radius * KPC_M)
    return velocity_bar * np.sqrt(
        1.0 + A0 * window(radius, rdisk_kpc) * h_function(g_bar)
    )


def predict_acceleration_only(
    radius_kpc: np.ndarray, velocity_bar_kms: np.ndarray
) -> np.ndarray:
    radius = np.asarray(radius_kpc, dtype=float)
    velocity_bar = np.asarray(velocity_bar_kms, dtype=float)
    g_bar = (velocity_bar * 1000.0) ** 2 / (radius * KPC_M)
    return velocity_bar * np.sqrt(1.0 + A0 * h_function(g_bar))


def predict_mond(radius_kpc: np.ndarray, velocity_bar_kms: np.ndarray) -> np.ndarray:
    radius = np.asarray(radius_kpc, dtype=float)
    velocity_bar = np.asarray(velocity_bar_kms, dtype=float)
    g_bar = (velocity_bar * 1000.0) ** 2 / (radius * KPC_M)
    x_value = np.maximum(g_bar / A0_MOND, 1e-10)
    nu_value = 1.0 / (1.0 - np.exp(-np.sqrt(x_value)))
    return velocity_bar * np.sqrt(nu_value)


def load_metadata() -> dict[str, tuple[int, float]]:
    metadata: dict[str, tuple[int, float]] = {}
    with TABLE1.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            name = line[0:11].strip()
            metadata[name] = (int(line[112:115]), float(line[71:76]))
    if len(metadata) != 175:
        raise RuntimeError(f"Expected 175 SPARC metadata rows, found {len(metadata)}")
    return metadata


def load_curves() -> list[Curve]:
    metadata = load_metadata()
    curves: list[Curve] = []
    for path in sorted(ROTMOD.glob("*_rotmod.dat")):
        name = path.stem.replace("_rotmod", "")
        values = np.loadtxt(path, comments="#", ndmin=2)
        if len(values) < 5:
            continue
        quality, rdisk = metadata[name]
        curves.append(
            Curve(
                name=name,
                quality=quality,
                rdisk_kpc=rdisk,
                radius_kpc=values[:, 0],
                observed_kms=values[:, 1],
                error_kms=values[:, 2],
                gas_kms=values[:, 3],
                disk_unit_ml_kms=values[:, 4],
                bulge_unit_ml_kms=values[:, 5],
            )
        )
    if len(curves) != 171:
        raise RuntimeError(f"Expected 171 usable SPARC curves, found {len(curves)}")
    return curves


def prepare_curve(
    curve: Curve, bulge_threshold: float | None
) -> dict[str, np.ndarray | float] | None:
    gas = curve.gas_kms
    disk = curve.disk_unit_ml_kms * math.sqrt(0.5)
    bulge = curve.bulge_unit_ml_kms * math.sqrt(0.7)
    velocity_bar_squared = np.sign(gas) * gas**2 + disk**2 + bulge**2
    velocity_bar = np.sqrt(np.abs(velocity_bar_squared)) * np.sign(
        velocity_bar_squared
    )
    valid = (
        (curve.radius_kpc > 0)
        & (curve.observed_kms > 0)
        & (velocity_bar > 0)
        & (curve.error_kms > 0)
    )
    if int(np.sum(valid)) < 5:
        return None
    radius = curve.radius_kpc[valid]
    observed = curve.observed_kms[valid]
    error = curve.error_kms[valid]
    velocity_bar = velocity_bar[valid]
    bulge = bulge[valid]
    heuristic_rdisk = float(radius[len(radius) // 3])

    if bulge_threshold is not None:
        bulge_fraction = bulge**2 / np.maximum(velocity_bar**2, 1e-10)
        keep = bulge_fraction < bulge_threshold
        if int(np.sum(keep)) < 3:
            return None
        radius = radius[keep]
        observed = observed[keep]
        error = error[keep]
        velocity_bar = velocity_bar[keep]

    return {
        "radius": radius,
        "observed": observed,
        "error": error,
        "velocity_bar": velocity_bar,
        "heuristic_rdisk": heuristic_rdisk,
    }


def rms(residual: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(residual, dtype=float) ** 2)))


def weighted_rms(residual: np.ndarray, error: np.ndarray) -> float:
    weights = 1.0 / np.asarray(error, dtype=float) ** 2
    return float(np.sqrt(np.sum(weights * residual**2) / np.sum(weights)))


def evaluate(
    curves: list[Curve],
    bulge_threshold: float | None,
    rdisk_overrides: dict[str, float] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for curve in curves:
        prepared = prepare_curve(curve, bulge_threshold)
        if prepared is None:
            continue
        radius = np.asarray(prepared["radius"], dtype=float)
        observed = np.asarray(prepared["observed"], dtype=float)
        error = np.asarray(prepared["error"], dtype=float)
        velocity_bar = np.asarray(prepared["velocity_bar"], dtype=float)
        rdisk = (
            curve.rdisk_kpc
            if rdisk_overrides is None
            else float(rdisk_overrides[curve.name])
        )
        predictions = {
            "locked": predict_locked(radius, velocity_bar),
            "window_catalog": predict_window(radius, velocity_bar, rdisk),
            "window_heuristic": predict_window(
                radius, velocity_bar, float(prepared["heuristic_rdisk"])
            ),
            "acceleration_only": predict_acceleration_only(radius, velocity_bar),
            "mond": predict_mond(radius, velocity_bar),
        }
        row: dict[str, float | int | str] = {
            "name": curve.name,
            "quality": curve.quality,
            "n_points": len(radius),
            "rdisk_catalog_kpc": curve.rdisk_kpc,
            "rdisk_used_kpc": rdisk,
            "rdisk_heuristic_kpc": float(prepared["heuristic_rdisk"]),
        }
        for model, prediction in predictions.items():
            residual = prediction - observed
            row[f"rms_{model}_kms"] = rms(residual)
            row[f"wrms_{model}_kms"] = weighted_rms(residual, error)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("name").reset_index(drop=True)


def bootstrap_summary(
    contrast: np.ndarray, bootstrap_indices: np.ndarray, seed_offset: int
) -> dict[str, object]:
    values = np.asarray(contrast, dtype=float)
    samples = values[bootstrap_indices]
    boot_means = np.mean(samples, axis=1)
    boot_wins = np.mean(samples < 0, axis=1)
    wins = int(np.sum(values < 0))
    losses = int(np.sum(values > 0))

    rng = np.random.default_rng(SEED + seed_offset)
    observed = abs(float(np.mean(values)))
    extreme = 0
    completed = 0
    while completed < SIGN_FLIPS:
        count = min(2_000, SIGN_FLIPS - completed)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(count, len(values)))
        extreme += int(np.sum(np.abs(np.mean(signs * values, axis=1)) >= observed))
        completed += count

    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "mean_ci95": [float(x) for x in np.percentile(boot_means, [2.5, 97.5])],
        "win_fraction": float(np.mean(values < 0)),
        "win_fraction_ci95": [
            float(x) for x in np.percentile(boot_wins, [2.5, 97.5])
        ],
        "wins": wins,
        "losses": losses,
        "exact_binomial_p_two_sided": float(
            binomtest(wins, wins + losses, p=0.5).pvalue
        ),
        "sign_flip_p_two_sided": float((extreme + 1) / (SIGN_FLIPS + 1)),
    }


def permutation_test(
    curves: list[Curve], primary: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object]]:
    names = primary["name"].tolist()
    actual_rdisk = primary["rdisk_catalog_kpc"].to_numpy(dtype=float)
    actual_mean = float(primary["rms_window_catalog_kms"].mean())
    curve_by_name = {curve.name: curve for curve in curves}
    prepared = {
        name: prepare_curve(curve_by_name[name], 0.30)
        for name in names
    }
    rng = np.random.default_rng(SEED + 100)
    rows: list[dict[str, float | int]] = []
    for permutation_index in range(PERMUTATIONS):
        assigned = rng.permutation(actual_rdisk)
        galaxy_rms: list[float] = []
        for name, rdisk in zip(names, assigned, strict=True):
            item = prepared[name]
            assert item is not None
            radius = np.asarray(item["radius"], dtype=float)
            velocity_bar = np.asarray(item["velocity_bar"], dtype=float)
            observed = np.asarray(item["observed"], dtype=float)
            galaxy_rms.append(
                rms(predict_window(radius, velocity_bar, float(rdisk)) - observed)
            )
        rows.append(
            {
                "permutation": permutation_index,
                "mean_galaxy_rms_kms": float(np.mean(galaxy_rms)),
            }
        )
    frame = pd.DataFrame(rows)
    random_values = frame["mean_galaxy_rms_kms"].to_numpy(dtype=float)
    return frame, {
        "actual_mean_galaxy_rms_kms": actual_mean,
        "random_mean_galaxy_rms_kms": float(np.mean(random_values)),
        "random_ci95_kms": [
            float(x) for x in np.percentile(random_values, [2.5, 97.5])
        ],
        "one_sided_p_actual_better": float(
            (1 + np.sum(random_values <= actual_mean)) / (PERMUTATIONS + 1)
        ),
    }


def model_means(frame: pd.DataFrame) -> dict[str, float]:
    return {
        column.removeprefix("rms_").removesuffix("_kms"): float(frame[column].mean())
        for column in frame.columns
        if column.startswith("rms_")
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    curves = load_curves()
    primary = evaluate(curves, 0.30)
    if len(primary) != 164 or int(primary["n_points"].sum()) != 2745:
        raise RuntimeError("The locked 30% SPARC sample changed")
    primary.to_csv(OUTPUT / "per_galaxy_primary.csv", index=False)

    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(primary), size=(BOOTSTRAPS, len(primary)))
    comparisons = {
        "locked_minus_mond": primary["rms_locked_kms"].to_numpy()
        - primary["rms_mond_kms"].to_numpy(),
        "locked_minus_acceleration_only": primary["rms_locked_kms"].to_numpy()
        - primary["rms_acceleration_only_kms"].to_numpy(),
        "window_catalog_minus_locked": primary[
            "rms_window_catalog_kms"
        ].to_numpy()
        - primary["rms_locked_kms"].to_numpy(),
        "window_catalog_minus_acceleration_only": primary[
            "rms_window_catalog_kms"
        ].to_numpy()
        - primary["rms_acceleration_only_kms"].to_numpy(),
        "window_catalog_minus_mond": primary[
            "rms_window_catalog_kms"
        ].to_numpy()
        - primary["rms_mond_kms"].to_numpy(),
        "weighted_locked_minus_mond": primary["wrms_locked_kms"].to_numpy()
        - primary["wrms_mond_kms"].to_numpy(),
    }
    comparison_summary = {
        label: bootstrap_summary(values, indices, offset)
        for offset, (label, values) in enumerate(comparisons.items(), start=1)
    }

    threshold_rows: list[dict[str, float | int | str]] = []
    threshold_frames: dict[str, pd.DataFrame] = {}
    for label, threshold in (
        ("20_percent", 0.20),
        ("30_percent_primary", 0.30),
        ("40_percent", 0.40),
        ("all_valid_points", None),
    ):
        frame = evaluate(curves, threshold)
        threshold_frames[label] = frame
        threshold_rows.append(
            {
                "sample": label,
                "bulge_threshold": (
                    float("nan") if threshold is None else float(threshold)
                ),
                "n_galaxies": len(frame),
                "n_points": int(frame["n_points"].sum()),
                "mean_rms_locked_kms": float(frame["rms_locked_kms"].mean()),
                "mean_rms_window_catalog_kms": float(
                    frame["rms_window_catalog_kms"].mean()
                ),
                "mean_rms_mond_kms": float(frame["rms_mond_kms"].mean()),
                "mean_locked_minus_mond_kms": float(
                    (frame["rms_locked_kms"] - frame["rms_mond_kms"]).mean()
                ),
                "locked_win_fraction_vs_mond": float(
                    np.mean(frame["rms_locked_kms"] < frame["rms_mond_kms"])
                ),
            }
        )
    pd.DataFrame(threshold_rows).to_csv(
        OUTPUT / "bulge_threshold_sensitivity.csv", index=False
    )

    permutation_frame, permutation_summary = permutation_test(curves, primary)
    permutation_frame.to_csv(OUTPUT / "rdisk_permutation.csv", index=False)
    median_rdisk = float(primary["rdisk_catalog_kpc"].median())
    fixed = evaluate(
        curves,
        0.30,
        rdisk_overrides={name: median_rdisk for name in primary["name"]},
    )

    summary = {
        "design": {
            "seed": SEED,
            "bootstrap_resamples": BOOTSTRAPS,
            "sign_flip_resamples": SIGN_FLIPS,
            "rdisk_permutations": PERMUTATIONS,
            "parameters_fitted": 0,
        },
        "primary_sample": {
            "n_galaxies": len(primary),
            "n_points": int(primary["n_points"].sum()),
            "mean_rms_kms": model_means(primary),
            "mean_weighted_rms_locked_kms": float(
                primary["wrms_locked_kms"].mean()
            ),
            "mean_weighted_rms_mond_kms": float(
                primary["wrms_mond_kms"].mean()
            ),
        },
        "comparisons": comparison_summary,
        "rdisk_permutation": permutation_summary,
        "fixed_median_rdisk": {
            "rdisk_kpc": median_rdisk,
            "mean_window_rms_kms": float(fixed["rms_window_catalog_kms"].mean()),
        },
        "all_valid_points": {
            "n_galaxies": len(threshold_frames["all_valid_points"]),
            "n_points": int(
                threshold_frames["all_valid_points"]["n_points"].sum()
            ),
            "mean_rms_kms": model_means(threshold_frames["all_valid_points"]),
        },
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
