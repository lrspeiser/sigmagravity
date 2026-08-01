#!/usr/bin/env python3
"""Diagnose where the independently refit SPARC survivor differs from RAR."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def strict_json(value):
    if isinstance(value, dict):
        return {key: strict_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [strict_json(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def rms(values) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(np.square(array))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "results" / "sparc_independent_nuisance_refit",
    )
    args = parser.parse_args()

    points = pd.read_csv(args.results / "point_predictions.csv")
    fits = pd.read_csv(args.results / "galaxy_fits.csv")
    candidate = points[
        (points["model"] == "RAR_sharp_coherence_gated_RG")
        & (points["scenario"] == "primary")
        & (points["split"] == "outer_holdout")
    ].copy()
    rar = points[
        (points["model"] == "fixed_RAR")
        & (points["scenario"] == "invariant")
        & (points["split"] == "outer_holdout")
    ].copy()
    keys = ["galaxy", "radius_catalog_kpc"]
    joined = candidate.merge(
        rar[
            keys
            + [
                "velocity_predicted_km_s",
                "velocity_observed_adjusted_km_s",
                "velocity_error_total_km_s",
            ]
        ],
        on=keys,
        validate="one_to_one",
        suffixes=("_candidate", "_RAR"),
    )
    joined["candidate_residual_km_s"] = (
        joined["velocity_predicted_km_s_candidate"]
        - joined["velocity_observed_adjusted_km_s_candidate"]
    )
    joined["RAR_residual_km_s"] = (
        joined["velocity_predicted_km_s_RAR"]
        - joined["velocity_observed_adjusted_km_s_RAR"]
    )
    joined["RAR_same_candidate_nuisance_residual_km_s"] = (
        joined["velocity_RAR_same_nuisance_km_s"]
        - joined["velocity_observed_adjusted_km_s_candidate"]
    )
    joined["formula_velocity_delta_km_s"] = (
        joined["velocity_predicted_km_s_candidate"]
        - joined["velocity_RAR_same_nuisance_km_s"]
    )
    joined["coherence_active"] = joined["coherence"] < 0.999

    per_galaxy = (
        joined.assign(
            candidate_residual_squared=np.square(joined["candidate_residual_km_s"]),
            RAR_residual_squared=np.square(joined["RAR_residual_km_s"]),
        )
        .groupby("galaxy")
        .agg(
            outer_points=("radius_catalog_kpc", "size"),
            active_outer_points=("coherence_active", "sum"),
            candidate_MSE_km2_s2=("candidate_residual_squared", "mean"),
            RAR_MSE_km2_s2=("RAR_residual_squared", "mean"),
            median_coherence=("coherence", "median"),
            p95_formula_velocity_delta_km_s=(
                "formula_velocity_delta_km_s",
                lambda values: float(np.percentile(values, 95.0)),
            ),
        )
        .reset_index()
    )
    per_galaxy["candidate_RMSE_km_s"] = np.sqrt(
        per_galaxy["candidate_MSE_km2_s2"]
    )
    per_galaxy["RAR_RMSE_km_s"] = np.sqrt(per_galaxy["RAR_MSE_km2_s2"])
    per_galaxy["delta_MSE_candidate_minus_RAR_km2_s2"] = (
        per_galaxy["candidate_MSE_km2_s2"] - per_galaxy["RAR_MSE_km2_s2"]
    )
    tolerance = 1.0e-10
    per_galaxy["comparison"] = np.where(
        per_galaxy["delta_MSE_candidate_minus_RAR_km2_s2"] < -tolerance,
        "candidate_better",
        np.where(
            per_galaxy["delta_MSE_candidate_minus_RAR_km2_s2"] > tolerance,
            "candidate_worse",
            "identical",
        ),
    )

    active = joined[joined["coherence_active"]].copy()
    active_galaxies = set(active["galaxy"])
    candidate_fits = fits[
        (fits["model"] == "RAR_sharp_coherence_gated_RG")
        & (fits["scenario"] == "primary")
    ]
    rar_fits = fits[
        (fits["model"] == "fixed_RAR") & (fits["scenario"] == "invariant")
    ]
    fit_join = candidate_fits.merge(
        rar_fits,
        on="galaxy",
        validate="one_to_one",
        suffixes=("_candidate", "_RAR"),
    )
    fit_join = fit_join[fit_join["galaxy"].isin(active_galaxies)]
    nuisance_shifts = {
        column: float(
            np.median(
                np.abs(
                    fit_join[f"{column}_candidate"].to_numpy(dtype=float)
                    - fit_join[f"{column}_RAR"].to_numpy(dtype=float)
                )
            )
        )
        for column in [
            "disk_mass_to_light",
            "bulge_mass_to_light",
            "distance_scale",
            "inclination_adjusted_deg",
        ]
    }
    active_comparison = per_galaxy[per_galaxy["active_outer_points"] > 0]
    diagnostics = {
        "status": "completed survivor activation and outlier diagnostic",
        "outer_sample": {
            "points": len(joined),
            "galaxies": int(joined["galaxy"].nunique()),
        },
        "coherence_active_subset": {
            "points": len(active),
            "point_fraction": float(len(active) / len(joined)),
            "galaxies": int(active["galaxy"].nunique()),
            "candidate_RMSE_km_s": rms(active["candidate_residual_km_s"]),
            "independently_refit_RAR_RMSE_km_s": rms(active["RAR_residual_km_s"]),
            "RAR_same_candidate_nuisance_RMSE_km_s": rms(
                active["RAR_same_candidate_nuisance_residual_km_s"]
            ),
            "formula_velocity_delta_median_km_s": float(
                active["formula_velocity_delta_km_s"].median()
            ),
            "formula_velocity_delta_p95_km_s": float(
                np.percentile(active["formula_velocity_delta_km_s"], 95.0)
            ),
            "formula_velocity_delta_max_km_s": float(
                active["formula_velocity_delta_km_s"].max()
            ),
            "active_galaxies_candidate_better": int(
                (active_comparison["comparison"] == "candidate_better").sum()
            ),
            "active_galaxies_candidate_worse": int(
                (active_comparison["comparison"] == "candidate_worse").sum()
            ),
            "median_absolute_nuisance_shift_candidate_vs_RAR": nuisance_shifts,
        },
        "all_galaxy_comparison": {
            "candidate_better": int(
                (per_galaxy["comparison"] == "candidate_better").sum()
            ),
            "candidate_worse": int(
                (per_galaxy["comparison"] == "candidate_worse").sum()
            ),
            "identical": int((per_galaxy["comparison"] == "identical").sum()),
            "largest_improvements": per_galaxy.nsmallest(
                5, "delta_MSE_candidate_minus_RAR_km2_s2"
            )[
                [
                    "galaxy",
                    "candidate_RMSE_km_s",
                    "RAR_RMSE_km_s",
                    "delta_MSE_candidate_minus_RAR_km2_s2",
                ]
            ].to_dict(orient="records"),
            "largest_regressions": per_galaxy.nlargest(
                5, "delta_MSE_candidate_minus_RAR_km2_s2"
            )[
                [
                    "galaxy",
                    "candidate_RMSE_km_s",
                    "RAR_RMSE_km_s",
                    "delta_MSE_candidate_minus_RAR_km2_s2",
                ]
            ].to_dict(orient="records"),
        },
        "interpretation": (
            "Aggregate competitiveness is partly produced by exact RAR recovery in inactive "
            "galaxies. The active subset is the discriminating sample and must be replicated."
        ),
    }

    per_galaxy.sort_values(
        "delta_MSE_candidate_minus_RAR_km2_s2"
    ).to_csv(args.results / "per_galaxy_outer_comparison.csv", index=False)
    (args.results / "activation_diagnostics.json").write_text(
        json.dumps(strict_json(diagnostics), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))
    inactive = per_galaxy["active_outer_points"] == 0
    axes[0].scatter(
        per_galaxy.loc[inactive, "RAR_RMSE_km_s"],
        per_galaxy.loc[inactive, "candidate_RMSE_km_s"],
        s=18,
        alpha=0.45,
        color="#9ca3af",
        label="gate inactive",
    )
    axes[0].scatter(
        per_galaxy.loc[~inactive, "RAR_RMSE_km_s"],
        per_galaxy.loc[~inactive, "candidate_RMSE_km_s"],
        s=28,
        alpha=0.8,
        color="#7c3aed",
        label="gate active",
    )
    limit = float(
        max(
            per_galaxy["RAR_RMSE_km_s"].max(),
            per_galaxy["candidate_RMSE_km_s"].max(),
        )
        * 1.05
    )
    axes[0].plot([0.0, limit], [0.0, limit], color="black", linewidth=1.0)
    axes[0].set(xlim=(0.0, limit), ylim=(0.0, limit))
    axes[0].set_xlabel("Fixed RAR outer RMSE (km/s)")
    axes[0].set_ylabel("Candidate outer RMSE (km/s)")
    axes[0].set_title("Per-galaxy radial prediction")
    axes[0].legend(frameon=False)

    scatter = axes[1].scatter(
        active["coherence"],
        active["formula_velocity_delta_km_s"],
        c=np.log10(active["local_density_g_cm3"]),
        s=20,
        alpha=0.75,
        cmap="viridis",
    )
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_xlabel("Local coherence C")
    axes[1].set_ylabel("Candidate − RAR at same nuisance (km/s)")
    axes[1].set_title("Added channel where gate is active")
    colorbar = figure.colorbar(scatter, ax=axes[1])
    colorbar.set_label("log10 local density (g/cm³)")
    figure.tight_layout()
    figure.savefig(args.results / "activation_diagnostics.png", dpi=180)
    plt.close(figure)
    print(json.dumps(strict_json(diagnostics), indent=2))


if __name__ == "__main__":
    main()
