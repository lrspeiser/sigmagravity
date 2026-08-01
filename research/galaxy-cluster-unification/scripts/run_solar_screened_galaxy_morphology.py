#!/usr/bin/env python3
"""Score the cluster-selected Solar-screened tail across SPARC galaxy types."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

from run_sparc_independent_nuisance_refit import (
    bounds_for,
    build_frame,
    fit_one_variant,
    optimizer_settings,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_ORDER = [
    "solar_screened_isothermal",
    "fixed_RAR",
    "simple_MOND",
    "NFW",
]
TYPE_COLUMNS = [
    "stellar_structure",
    "hubble_family",
    "surface_brightness_family",
    "baryonic_mass_family",
    "gas_fraction_family",
    "inclination_family",
    "outer_rotation_shape",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def classify_galaxies(frame: pd.DataFrame, morphology: pd.DataFrame) -> pd.DataFrame:
    names = set(frame["galaxy"].unique())
    selected = morphology[morphology["galaxy"].isin(names)].copy()
    if len(selected) != len(names):
        raise ValueError("morphology catalog does not cover the frozen galaxy sample")
    numeric = [
        "hubble_type",
        "inclination_deg",
        "disk_central_surface_brightness",
        "baryonic_mass_solar",
        "stellar_bulge_fraction",
        "baryonic_bulge_fraction",
        "gas_fraction",
        "disk_scale_kpc",
    ]
    selected[numeric] = selected[numeric].apply(pd.to_numeric, errors="coerce")
    b_to_t = selected["stellar_bulge_fraction"]
    selected["stellar_structure"] = np.select(
        [b_to_t <= 0.05, b_to_t < 0.30],
        ["disk_dominated", "mixed_disk_bulge"],
        default="bulge_dominated",
    )
    hubble = selected["hubble_type"]
    selected["hubble_family"] = np.select(
        [hubble <= 3, hubble <= 6],
        ["early_S0_to_Sb", "spiral_Sbc_to_Scd"],
        default="late_Sd_to_BCD",
    )
    brightness = selected["disk_central_surface_brightness"]
    selected["surface_brightness_family"] = np.select(
        [brightness < 100.0, brightness <= 500.0],
        ["low_surface_brightness", "intermediate_surface_brightness"],
        default="high_surface_brightness",
    )
    mass = selected["baryonic_mass_solar"]
    selected["baryonic_mass_family"] = np.select(
        [mass < 1.0e9, mass < 1.0e11],
        ["dwarf_mass", "intermediate_mass"],
        default="giant_mass",
    )
    gas = selected["gas_fraction"]
    selected["gas_fraction_family"] = np.select(
        [gas < 0.2, gas < 0.5],
        ["gas_poor", "mixed_gas"],
        default="gas_rich",
    )
    inclination = selected["inclination_deg"]
    selected["inclination_family"] = np.select(
        [inclination < 50.0, inclination < 70.0],
        ["moderate_inclination", "intermediate_inclination"],
        default="edge_on",
    )

    shape_rows = []
    for galaxy, block in frame.groupby("galaxy", sort=True):
        ordered = block.sort_values("radius_catalog_kpc")
        count = max(3, int(math.ceil(len(ordered) / 3.0)))
        outer = ordered.tail(count)
        radius = outer["radius_catalog_kpc"].to_numpy(dtype=float)
        velocity = outer["velocity_observed_catalog_kms"].to_numpy(dtype=float)
        valid = (radius > 0.0) & (velocity > 0.0)
        slope = float(np.polyfit(np.log(radius[valid]), np.log(velocity[valid]), 1)[0])
        if slope < -0.10:
            label = "declining"
        elif slope > 0.10:
            label = "rising"
        else:
            label = "approximately_flat"
        shape_rows.append(
            {
                "galaxy": galaxy,
                "outer_log_velocity_slope": slope,
                "outer_rotation_shape": label,
                "outer_shape_points": int(np.sum(valid)),
            }
        )
    selected = selected.merge(pd.DataFrame(shape_rows), on="galaxy", validate="one_to_one")
    columns = [
        "galaxy",
        "hubble_type",
        "inclination_deg",
        "disk_scale_kpc",
        "disk_central_surface_brightness",
        "baryonic_mass_solar",
        "stellar_bulge_fraction",
        "baryonic_bulge_fraction",
        "gas_fraction",
        "outer_log_velocity_slope",
        "outer_shape_points",
        *TYPE_COLUMNS,
    ]
    result = selected[columns].sort_values("galaxy").reset_index(drop=True)
    if result[TYPE_COLUMNS].isna().any().any():
        raise ValueError("at least one frozen morphology classification is missing")
    return result


def score(block: pd.DataFrame) -> dict:
    if block.empty:
        raise ValueError("cannot score an empty block")
    residual = (
        block["velocity_predicted_km_s"]
        - block["velocity_observed_adjusted_km_s"]
    ).to_numpy(dtype=float)
    sigma = block["velocity_error_total_km_s"].to_numpy(dtype=float)
    with_mse = block.assign(residual_squared=np.square(residual))
    per_galaxy_mse = with_mse.groupby("galaxy")["residual_squared"].mean()
    return {
        "galaxies": int(block["galaxy"].nunique()),
        "points": int(len(block)),
        "RMSE_km_s": float(np.sqrt(np.mean(np.square(residual)))),
        "equal_galaxy_RMSE_km_s": float(np.sqrt(per_galaxy_mse.mean())),
        "MAE_km_s": float(np.mean(np.abs(residual))),
        "mean_residual_km_s": float(np.mean(residual)),
        "median_residual_km_s": float(np.median(residual)),
        "chi2_per_point": float(np.mean(np.square(residual / sigma))),
    }


def type_scores(points: pd.DataFrame) -> pd.DataFrame:
    outer = points[points["split"] == "outer_holdout"].copy()
    rows = []
    for model in MODEL_ORDER:
        block = outer[outer["model"] == model]
        rows.append({"dimension": "all", "bin": "all", "model": model, **score(block)})
        for dimension in TYPE_COLUMNS:
            for label, group in block.groupby(dimension, sort=True):
                rows.append(
                    {
                        "dimension": dimension,
                        "bin": label,
                        "model": model,
                        **score(group),
                    }
                )
    scores = pd.DataFrame(rows)
    rar = scores[scores["model"] == "fixed_RAR"][
        ["dimension", "bin", "RMSE_km_s", "equal_galaxy_RMSE_km_s"]
    ].rename(
        columns={
            "RMSE_km_s": "fixed_RAR_RMSE_km_s",
            "equal_galaxy_RMSE_km_s": "fixed_RAR_equal_galaxy_RMSE_km_s",
        }
    )
    scores = scores.merge(rar, on=["dimension", "bin"], validate="many_to_one")
    scores["RMSE_ratio_to_fixed_RAR"] = (
        scores["RMSE_km_s"] / scores["fixed_RAR_RMSE_km_s"]
    )
    scores["equal_galaxy_RMSE_ratio_to_fixed_RAR"] = (
        scores["equal_galaxy_RMSE_km_s"]
        / scores["fixed_RAR_equal_galaxy_RMSE_km_s"]
    )
    scores["model_order"] = scores["model"].map({name: i for i, name in enumerate(MODEL_ORDER)})
    return scores.sort_values(["dimension", "bin", "model_order"]).drop(
        columns="model_order"
    )


def per_galaxy_scores(points: pd.DataFrame, classes: pd.DataFrame) -> pd.DataFrame:
    outer = points[points["split"] == "outer_holdout"].copy()
    outer["residual_squared"] = np.square(
        outer["velocity_predicted_km_s"] - outer["velocity_observed_adjusted_km_s"]
    )
    outer["observed_squared"] = np.square(outer["velocity_observed_adjusted_km_s"])
    grouped = (
        outer.groupby(["model", "galaxy"], sort=True)
        .agg(
            outer_points=("residual_squared", "size"),
            MSE_km2_s2=("residual_squared", "mean"),
            observed_mean_square_km2_s2=("observed_squared", "mean"),
        )
        .reset_index()
    )
    grouped["RMSE_km_s"] = np.sqrt(grouped["MSE_km2_s2"])
    grouped["fractional_RMSE_percent"] = 100.0 * np.sqrt(
        grouped["MSE_km2_s2"] / grouped["observed_mean_square_km2_s2"]
    )
    wide = grouped.pivot(index="galaxy", columns="model", values="MSE_km2_s2")
    comparison = pd.DataFrame(
        {
            "galaxy": wide.index,
            "delta_MSE_screened_minus_RAR_km2_s2": (
                wide["solar_screened_isothermal"] - wide["fixed_RAR"]
            ).to_numpy(),
            "delta_MSE_screened_minus_MOND_km2_s2": (
                wide["solar_screened_isothermal"] - wide["simple_MOND"]
            ).to_numpy(),
        }
    )
    result = grouped.merge(classes, on="galaxy", validate="many_to_one")
    result = result.merge(comparison, on="galaxy", validate="many_to_one")
    return result.sort_values(["model", "galaxy"]).reset_index(drop=True)


def correlation_diagnostics(per_galaxy: pd.DataFrame) -> dict:
    screened = per_galaxy[per_galaxy["model"] == "solar_screened_isothermal"].copy()
    diagnostics = {}
    for feature in [
        "stellar_bulge_fraction",
        "baryonic_bulge_fraction",
        "disk_central_surface_brightness",
        "baryonic_mass_solar",
        "gas_fraction",
        "outer_log_velocity_slope",
    ]:
        values = screened[feature].to_numpy(dtype=float)
        target = screened["fractional_RMSE_percent"].to_numpy(dtype=float)
        rho, p_value = spearmanr(values, target)
        diagnostics[feature] = {
            "spearman_rho_with_fractional_RMSE": float(rho),
            "two_sided_p_value": float(p_value),
        }
    bulge_rank = rankdata(screened["stellar_bulge_fraction"].to_numpy(dtype=float))
    error_rank = rankdata(screened["fractional_RMSE_percent"].to_numpy(dtype=float))
    mass_rank = rankdata(np.log10(screened["baryonic_mass_solar"].to_numpy(dtype=float)))
    design = np.column_stack([np.ones(len(screened)), mass_rank])
    bulge_residual = bulge_rank - design @ np.linalg.lstsq(
        design, bulge_rank, rcond=None
    )[0]
    error_residual = error_rank - design @ np.linalg.lstsq(
        design, error_rank, rcond=None
    )[0]
    partial_r, partial_p = pearsonr(bulge_residual, error_residual)
    diagnostics["stellar_bulge_fraction_controlling_log_baryonic_mass"] = {
        "partial_spearman_rho_with_fractional_RMSE": float(partial_r),
        "two_sided_p_value": float(partial_p),
        "method": "Pearson correlation of rank residuals after linear removal of ranked log10 baryonic mass",
    }
    return diagnostics


def analytic_mass_scaling(law: dict) -> dict:
    gravitational_constant = 6.67430e-11
    solar_mass_kg = 1.98847e30
    kpc_m = 3.085677581491367e19
    a0 = float(law["a0_m_s2"])
    reference_radius_m = float(law["reference_radius_kpc"]) * kpc_m
    parameter = float(law["lambda"])
    matched_mass_solar = (
        a0
        * reference_radius_m**2
        / (gravitational_constant * parameter**2)
        / solar_mass_kg
    )
    comparison = []
    for mass_solar in (1.0e7, 1.0e8, 1.0e9, 1.0e10, 1.0e11, 1.0e12):
        mond_matching_parameter = reference_radius_m * math.sqrt(
            a0 / (gravitational_constant * mass_solar * solar_mass_kg)
        )
        comparison.append(
            {
                "baryonic_mass_solar": mass_solar,
                "lambda_needed_to_match_deep_MOND_speed": mond_matching_parameter,
            }
        )
    return {
        "screen_open_limit": "v_tail^2=lambda*G*Mbar/rstar",
        "deep_MOND_comparison": "v_MOND^2=sqrt(G*Mbar*a0)",
        "implied_lambda_scaling_to_match_MOND": "lambda=rstar*sqrt(a0/(G*Mbar)), proportional to Mbar^-1/2",
        "baryonic_mass_matched_by_selected_lambda_solar": matched_mass_solar,
        "mass_table": comparison,
        "interpretation": (
            "A constant lambda makes v_tail^4 proportional to Mbar^2, whereas the "
            "deep-MOND/baryonic-Tully-Fisher scaling is proportional to Mbar."
        ),
    }


def plot_summary(scores: pd.DataFrame, per_galaxy: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14.0, 9.0))
    panels = [
        ("stellar_structure", "Stellar structure"),
        ("hubble_family", "Hubble family"),
        ("surface_brightness_family", "Disk surface brightness"),
    ]
    colors = {
        "solar_screened_isothermal": "#7c3aed",
        "fixed_RAR": "#059669",
        "simple_MOND": "#2563eb",
        "NFW": "#9ca3af",
    }
    labels = {
        "solar_screened_isothermal": "screened tail",
        "fixed_RAR": "fixed RAR",
        "simple_MOND": "simple MOND",
        "NFW": "inner-fit NFW",
    }
    for axis, (dimension, title) in zip(axes.flat[:3], panels, strict=True):
        selected = scores[scores["dimension"] == dimension]
        bins = list(dict.fromkeys(selected["bin"]))
        x = np.arange(len(bins))
        width = 0.19
        for index, model in enumerate(MODEL_ORDER):
            values = (
                selected[selected["model"] == model]
                .set_index("bin")
                .loc[bins, "RMSE_km_s"]
                .to_numpy()
            )
            axis.bar(
                x + (index - 1.5) * width,
                values,
                width,
                color=colors[model],
                label=labels[model],
            )
        axis.set_xticks(x, [name.replace("_", "\n") for name in bins])
        axis.set_ylabel("Outer-point RMSE (km/s)")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8)

    screened = per_galaxy[per_galaxy["model"] == "solar_screened_isothermal"]
    scatter = axes[1, 1].scatter(
        screened["stellar_bulge_fraction"],
        screened["fractional_RMSE_percent"],
        c=np.log10(screened["baryonic_mass_solar"]),
        cmap="viridis",
        s=28,
        alpha=0.8,
    )
    axes[1, 1].set_xlabel("Stellar bulge fraction B/T")
    axes[1, 1].set_ylabel("Screened-tail outer fractional RMSE (%)")
    axes[1, 1].set_title("Per-galaxy morphology residual")
    colorbar = figure.colorbar(scatter, ax=axes[1, 1])
    colorbar.set_label("log10 baryonic mass (solar masses)")
    figure.suptitle("Locked cluster-selected screened law across SPARC galaxy types", y=1.01)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs" / "solar_screened_galaxy_morphology_protocol.json",
    )
    parser.add_argument(
        "--base-protocol",
        type=Path,
        default=ROOT / "configs" / "sparc_independent_nuisance_refit_protocol.json",
    )
    parser.add_argument(
        "--morphology",
        type=Path,
        default=ROOT / "data" / "derived" / "nbp0_sparc_morphology.csv",
    )
    parser.add_argument("--sparc", type=Path, default=ROOT / "data" / "raw" / "sparc")
    parser.add_argument(
        "--baseline-results",
        type=Path,
        default=ROOT / "results" / "sparc_independent_nuisance_refit",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "solar_screened_galaxy_morphology",
    )
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    base_protocol = json.loads(args.base_protocol.read_text(encoding="utf-8"))
    morphology = pd.read_csv(args.morphology)
    frame = build_frame(base_protocol, args.sparc, args.morphology)
    classes = classify_galaxies(frame, morphology)
    settings = optimizer_settings(base_protocol)
    law = protocol["law"]
    settings.update(
        {
            "screened_tail_parameter": float(law["lambda"]),
            "screened_tail_reference_radius_kpc": float(law["reference_radius_kpc"]),
            "screened_tail_a0_m_s2": float(law["a0_m_s2"]),
        }
    )
    print("fitting locked screened tail on inner radii", flush=True)
    screened_points, screened_fits = fit_one_variant(
        frame,
        output_model="solar_screened_isothermal",
        model="solar_screened_isothermal",
        scenario="cluster_selected_lambda_10p5",
        protocol=base_protocol,
        settings=settings,
        candidate_parameters=np.asarray([], dtype=float),
        density_geometry=None,
    )

    baseline_points_path = args.baseline_results / "point_predictions.csv"
    baseline_fits_path = args.baseline_results / "galaxy_fits.csv"
    baseline_points = pd.read_csv(baseline_points_path)
    baseline_points = baseline_points[
        (
            baseline_points["model"].isin(["fixed_RAR", "simple_MOND", "NFW"])
        )
        & (baseline_points["scenario"] == "invariant")
    ].copy()
    baseline_fits = pd.read_csv(baseline_fits_path)
    baseline_fits = baseline_fits[
        (baseline_fits["model"].isin(["fixed_RAR", "simple_MOND", "NFW"]))
        & (baseline_fits["scenario"] == "invariant")
    ].copy()
    points = pd.concat([screened_points, baseline_points], ignore_index=True)
    if set(points["model"].unique()) != set(MODEL_ORDER):
        raise ValueError("comparison model set is incomplete")
    points = points.merge(classes[["galaxy", *TYPE_COLUMNS]], on="galaxy", validate="many_to_one")
    fits = pd.concat([screened_fits, baseline_fits], ignore_index=True)

    scores = type_scores(points)
    galaxy_scores = per_galaxy_scores(points, classes)
    correlations = correlation_diagnostics(galaxy_scores)
    screened_fit = screened_fits
    tolerance = 1.0e-4
    bounds = bounds_for("solar_screened_isothermal", base_protocol)
    at_boundary = []
    for row in screened_fit.itertuples(index=False):
        theta = [row.disk_log_shift, row.bulge_log_shift, row.distance_z, row.inclination_z]
        at_boundary.append(
            any(
                abs(value - low) <= tolerance * max(1.0, abs(low))
                or abs(value - high) <= tolerance * max(1.0, abs(high))
                for value, (low, high) in zip(theta, bounds, strict=True)
            )
        )
    fit_diagnostics = {
        "finite_fit_fraction": float(screened_fit["finite_fit"].mean()),
        "optimizer_success_fraction": float(screened_fit["optimizer_success"].mean()),
        "nuisance_any_boundary_fraction": float(np.mean(at_boundary)),
        "median_optimizer_evaluations": float(screened_fit["evaluations"].median()),
    }

    overall = scores[scores["dimension"] == "all"].set_index("model")
    structure = scores[
        (scores["dimension"] == "stellar_structure")
        & (scores["model"] == "solar_screened_isothermal")
    ]
    gates = protocol["advance_gates"]
    gate_audit = {
        "all_outer_RMSE": bool(
            overall.loc["solar_screened_isothermal", "RMSE_ratio_to_fixed_RAR"]
            <= float(gates["all_outer_RMSE_relative_to_fixed_RAR_max"])
        ),
        "each_structural_bin_RMSE": bool(
            structure["RMSE_ratio_to_fixed_RAR"].max()
            <= float(gates["each_structural_bin_RMSE_relative_to_fixed_RAR_max"])
        ),
        "finite_fits": bool(
            fit_diagnostics["finite_fit_fraction"]
            >= float(gates["finite_fit_fraction_min"])
        ),
        "optimizer_success": bool(
            fit_diagnostics["optimizer_success_fraction"]
            >= float(gates["optimizer_success_fraction_min"])
        ),
        "nuisance_boundaries": bool(
            fit_diagnostics["nuisance_any_boundary_fraction"]
            <= float(gates["nuisance_any_boundary_fraction_max"])
        ),
    }
    gate_audit["passes_all"] = all(gate_audit.values())

    screened_galaxy = galaxy_scores[
        galaxy_scores["model"] == "solar_screened_isothermal"
    ]
    comparison = {
        "galaxies_better_than_fixed_RAR": int(
            (screened_galaxy["delta_MSE_screened_minus_RAR_km2_s2"] < 0.0).sum()
        ),
        "galaxies_worse_than_fixed_RAR": int(
            (screened_galaxy["delta_MSE_screened_minus_RAR_km2_s2"] > 0.0).sum()
        ),
        "galaxies_better_than_simple_MOND": int(
            (screened_galaxy["delta_MSE_screened_minus_MOND_km2_s2"] < 0.0).sum()
        ),
        "galaxies_worse_than_simple_MOND": int(
            (screened_galaxy["delta_MSE_screened_minus_MOND_km2_s2"] > 0.0).sum()
        ),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    points.to_csv(args.output / "point_predictions.csv", index=False)
    screened_fits.to_csv(args.output / "screened_tail_galaxy_fits.csv", index=False)
    classes.to_csv(args.output / "morphology_assignments.csv", index=False)
    scores.to_csv(args.output / "type_scores.csv", index=False)
    galaxy_scores.to_csv(args.output / "per_galaxy_scores.csv", index=False)
    plot_summary(scores, galaxy_scores, args.output / "galaxy_type_assessment.png")

    report = {
        "report_version": "SOLAR-SCREENED-GALAXY-MORPHOLOGY-0.1.0",
        "status": "completed locked cluster-to-galaxy morphology transfer",
        "inputs": {
            "protocol_sha256": sha256(args.protocol),
            "base_protocol_sha256": sha256(args.base_protocol),
            "morphology_sha256": sha256(args.morphology),
            "baseline_points_sha256": sha256(baseline_points_path),
            "baseline_fits_sha256": sha256(baseline_fits_path),
        },
        "law": law,
        "sample": {
            "galaxies": int(frame["galaxy"].nunique()),
            "inner_train_points": int((frame["split"] == "inner_train").sum()),
            "outer_holdout_points": int((frame["split"] == "outer_holdout").sum()),
            "bin_counts": {
                dimension: {
                    str(name): int(count)
                    for name, count in classes[dimension].value_counts().sort_index().items()
                }
                for dimension in TYPE_COLUMNS
            },
        },
        "overall_outer_scores": {
            model: strict_json(overall.loc[model].to_dict()) for model in MODEL_ORDER
        },
        "screened_tail_type_scores": strict_json(
            scores[scores["model"] == "solar_screened_isothermal"].to_dict(orient="records")
        ),
        "fit_diagnostics": fit_diagnostics,
        "per_galaxy_comparison": comparison,
        "morphology_correlations": correlations,
        "analytic_mass_scaling": analytic_mass_scaling(law),
        "gate_audit": gate_audit,
        "interpretation": {
            "transfer_result": (
                "passes the frozen galaxy morphology gates"
                if gate_audit["passes_all"]
                else "fails one or more frozen galaxy morphology gates"
            ),
            "geometry_result": (
                "The measured in-plane baryonic force includes separate gas, disk, and bulge templates. "
                "The added screened tail remains spherical and depends on total source mass, so vertical "
                "disk thickness and nonspherical bulge response are not independently predicted."
            ),
            "solar_boundary": (
                "Mercury passed a first-order supplementary-precession diagnostic; raw multi-planet "
                "ephemeris compatibility has not been established."
            ),
            "dark_matter_boundary": (
                "NFW is only an inner-fit, outer-extrapolation control with two halo parameters per galaxy."
            ),
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    (args.output / "report.json").write_text(
        json.dumps(strict_json(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            strict_json(
                {
                    "overall_outer_scores": report["overall_outer_scores"],
                    "per_galaxy_comparison": comparison,
                    "gate_audit": gate_audit,
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
