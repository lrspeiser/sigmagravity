#!/usr/bin/env python3
"""Run controlled reopened Sigma/RG variations across three data regimes."""

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
from scipy.optimize import differential_evolution, minimize
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_cpr0_accept_clash_bridge import (  # noqa: E402
    assign_group_folds,
    domain_metrics,
    equal_group_domain_mse,
)
from run_rxj2129_raw_theory_lensing import (  # noqa: E402
    RawLens,
    score,
    spec_for,
)
from run_unbounded_running_multicluster_raw import (  # noqa: E402
    aggregate_system_scores,
    load_anchors,
    load_system_images,
    predictive_split,
    system_protocol,
)
from voidscreen.raw_lensing import (  # noqa: E402
    RadialDeflectionField,
    loglog_interpolate_with_tails,
    spherical_deflection_radians,
)
from voidscreen.reopened_hybrids import (  # noqa: E402
    KPC_M,
    apply_channel_gate_memory_to_response,
    apply_radial_memory_to_response,
    mercury_precession_mas_per_century,
    screened_hybrid_profile_response,
    screened_hybrid_response,
    solar_system_diagnostics,
    tidal_shape_property,
)
from voidscreen.tensor_completion import (  # noqa: E402
    axisymmetric_tidal_eigenvalues,
    spherical_tidal_eigenvalues,
)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expand_variants(protocol: dict) -> list[dict]:
    variants = []
    for family, specification in protocol["sensitivity_families"].items():
        fixed_name = specification["fixed_name"]
        for value in specification["values"]:
            settings = dict(protocol.get("common_variant", {}))
            settings.update(specification["base_variant"])
            settings[fixed_name] = float(value)
            variants.append(
                {
                    "name": f"{family}:{fixed_name}={float(value):g}",
                    "family": family,
                    "fixed_name": fixed_name,
                    "fixed_value": float(value),
                    "settings": settings,
                }
            )
    if len({row["name"] for row in variants}) != len(variants):
        raise RuntimeError("variant names must be unique")
    return variants


def response(frame: pd.DataFrame, parameters, variant: dict, constants: dict) -> np.ndarray:
    local = screened_hybrid_response(
        np.power(10.0, frame["log_gbar"].to_numpy(float)),
        frame["local_density_g_cm3"].to_numpy(float),
        frame["radius_kpc"].to_numpy(float),
        parameters,
        variant["settings"],
        g_reference_m_s2=float(constants["g_reference_m_s2"]),
        g_dagger_m_s2=float(constants["g_dagger_m_s2"]),
        acceleration_screen_m_s2=float(constants["acceleration_screen_m_s2"]),
    )
    groups = list(
        frame.reset_index(drop=True)
        .groupby(["domain", "system"], sort=False)
        .indices.values()
    )
    return grouped_radial_memory_enhancement(
        local,
        frame["radius_kpc"].to_numpy(float),
        groups,
        variant["settings"],
    )


def grouped_radial_memory_enhancement(
    local: dict[str, np.ndarray],
    radius_kpc: np.ndarray,
    groups,
    settings: dict,
) -> np.ndarray:
    """Apply one profile-memory/diffusion setting to local responses."""

    strength = float(settings.get("radial_memory_strength", 0.0))
    gate_memory_strength = float(
        settings.get("channel_gate_memory_strength", 0.0)
    )
    diffusion_strength = float(settings.get("radial_diffusion_strength", 0.0))
    if (
        strength == 0.0
        and gate_memory_strength == 0.0
        and diffusion_strength == 0.0
    ):
        return local["enhancement"]
    enhancement = np.asarray(local["enhancement"], dtype=float).copy()
    for positions in groups:
        positions = np.asarray(positions, dtype=int)
        if len(positions) <= 1:
            continue
        group_local = {
            name: np.asarray(values)[positions]
            for name, values in local.items()
        }
        transformed = apply_channel_gate_memory_to_response(
            group_local, radius_kpc[positions], settings
        )
        transformed = apply_radial_memory_to_response(
            transformed, radius_kpc[positions], settings
        )
        enhancement[positions] = transformed["enhancement"]
    if np.any(~np.isfinite(enhancement)) or np.any(enhancement < 1.0):
        raise ValueError("grouped profile response produced invalid enhancement")
    return enhancement


def predict_log_acceleration(
    frame: pd.DataFrame, parameters, variant: dict, constants: dict
) -> np.ndarray:
    return frame["log_gbar"].to_numpy(float) + np.log10(
        response(frame, parameters, variant, constants)
    )


def equal_domain_point_weights(frame: pd.DataFrame) -> np.ndarray:
    """Precompute weights exactly equivalent to equal_group_domain_mse."""

    domain_systems = frame.groupby("domain")["system"].transform("nunique")
    system_points = frame.groupby(["domain", "system"])["system"].transform(
        "size"
    )
    return (
        1.0
        / float(frame.domain.nunique())
        / domain_systems.to_numpy(float)
        / system_points.to_numpy(float)
    )


def fit_bridge(
    frame: pd.DataFrame,
    variant: dict,
    protocol: dict,
    seed: int,
) -> np.ndarray:
    bounds = list(map(tuple, protocol["universal_parameters"]["bounds"]))
    constants = protocol["shared_constants"]
    gbar = np.power(10.0, frame["log_gbar"].to_numpy(float))
    density = frame["local_density_g_cm3"].to_numpy(float)
    radius = frame["radius_kpc"].to_numpy(float)
    observed = frame["log_gobs"].to_numpy(float)
    weights = equal_domain_point_weights(frame)
    groups = list(
        frame.reset_index(drop=True)
        .groupby(["domain", "system"], sort=False)
        .indices.values()
    )

    def objective(values):
        try:
            local = screened_hybrid_response(
                gbar,
                density,
                radius,
                values,
                variant["settings"],
                g_reference_m_s2=float(constants["g_reference_m_s2"]),
                g_dagger_m_s2=float(constants["g_dagger_m_s2"]),
                acceleration_screen_m_s2=float(
                    constants["acceleration_screen_m_s2"]
                ),
            )
            enhancement = grouped_radial_memory_enhancement(
                local, radius, groups, variant["settings"]
            )
            residual = np.log10(gbar * enhancement) - observed
            return float(np.sum(weights * np.square(residual)))
        except (FloatingPointError, OverflowError, ValueError):
            return 1.0e100

    global_fit = differential_evolution(
        objective,
        bounds,
        seed=seed,
        maxiter=int(protocol["optimization"]["differential_evolution_maxiter"]),
        popsize=int(protocol["optimization"]["differential_evolution_popsize"]),
        polish=False,
        workers=1,
        tol=1.0e-9,
    )
    local_fit = minimize(
        objective,
        global_fit.x,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 5000, "ftol": 1.0e-14, "gtol": 1.0e-9},
    )
    return np.asarray(local_fit.x if local_fit.success else global_fit.x, dtype=float)


def cross_validate_bridge(
    frame: pd.DataFrame, variant: dict, protocol: dict, variant_index: int
) -> tuple[np.ndarray, list[list[float]], np.ndarray]:
    prediction = np.full(len(frame), np.nan)
    fold_parameters = []
    folds = int(protocol["optimization"]["bridge_folds"])
    seed = int(protocol["optimization"]["bridge_seed"]) + 1000 * variant_index
    for fold in range(folds):
        training = frame[frame.fold != fold]
        heldout = frame[frame.fold == fold]
        parameters = fit_bridge(training, variant, protocol, seed + fold)
        prediction[heldout.index] = predict_log_acceleration(
            heldout, parameters, variant, protocol["shared_constants"]
        )
        fold_parameters.append(parameters.tolist())
    full_parameters = fit_bridge(frame, variant, protocol, seed + 100)
    if np.any(~np.isfinite(prediction)):
        raise RuntimeError(f"{variant['name']} left bridge predictions missing")
    return prediction, fold_parameters, full_parameters


def at_boundary(parameters, protocol: dict) -> dict[str, bool]:
    tolerance = float(protocol["optimization"]["boundary_fraction_tolerance"])
    output = {}
    for name, value, bounds in zip(
        protocol["universal_parameters"]["names"],
        parameters,
        protocol["universal_parameters"]["bounds"],
        strict=True,
    ):
        lower, upper = map(float, bounds)
        width = upper - lower
        output[name] = bool(
            value <= lower + tolerance * width
            or value >= upper - tolerance * width
        )
    return output


def sparc_scores(
    sparc: pd.DataFrame,
    parameters,
    variant: dict,
    constants: dict,
    tidal_geometry: dict | None = None,
) -> tuple[dict, pd.DataFrame]:
    gate_values = None
    gate_property = str(variant["settings"].get("channel_gate_property", ""))
    if gate_property.startswith("tidal_") and gate_property != "tidal_curvature":
        method = (tidal_geometry or {}).get(
            "sparc_method", "axisymmetric_midplane_density_closure"
        )
        if method == "spherical_density_closure":
            eigenvalues = spherical_tidal_eigenvalues(
                sparc["g_bar_m_s2"].to_numpy(float),
                sparc["radius_adjusted_kpc"].to_numpy(float),
                sparc["local_density_g_cm3"].to_numpy(float),
            )
        elif method == "axisymmetric_midplane_density_closure":
            eigenvalues = np.full((len(sparc), 3), np.nan, dtype=float)
            for _, indices in sparc.groupby("galaxy").groups.items():
                block = sparc.loc[indices].sort_values("radius_adjusted_kpc")
                eigenvalues[block.index] = axisymmetric_tidal_eigenvalues(
                    block["g_bar_m_s2"].to_numpy(float),
                    block["radius_adjusted_kpc"].to_numpy(float),
                    block["local_density_g_cm3"].to_numpy(float),
                )
        else:
            raise ValueError(f"unknown SPARC tidal geometry method {method}")
        if np.any(~np.isfinite(eigenvalues)):
            raise RuntimeError("SPARC tidal gate values remain missing")
        gate_values = tidal_shape_property(eigenvalues, gate_property)
    local = screened_hybrid_response(
        sparc["g_bar_m_s2"].to_numpy(float),
        sparc["local_density_g_cm3"].to_numpy(float),
        sparc["radius_adjusted_kpc"].to_numpy(float),
        parameters,
        variant["settings"],
        g_reference_m_s2=float(constants["g_reference_m_s2"]),
        g_dagger_m_s2=float(constants["g_dagger_m_s2"]),
        acceleration_screen_m_s2=float(constants["acceleration_screen_m_s2"]),
        channel_gate_property_values=gate_values,
    )
    groups = list(
        sparc.reset_index(drop=True)
        .groupby("galaxy", sort=False)
        .indices.values()
    )
    enhancement = grouped_radial_memory_enhancement(
        local,
        sparc["radius_adjusted_kpc"].to_numpy(float),
        groups,
        variant["settings"],
    )
    prediction = np.sqrt(
        sparc["g_bar_m_s2"].to_numpy(float)
        * enhancement
        * sparc["radius_adjusted_kpc"].to_numpy(float)
        * KPC_M
    ) / 1000.0
    local_prediction = np.sqrt(
        sparc["g_bar_m_s2"].to_numpy(float)
        * local["enhancement"]
        * sparc["radius_adjusted_kpc"].to_numpy(float)
        * KPC_M
    ) / 1000.0
    observed = sparc["velocity_observed_adjusted_kms"].to_numpy(float)
    uncertainty = sparc["velocity_error_total_kms"].to_numpy(float)
    rar = sparc["velocity_predicted_kms"].to_numpy(float)
    residual = prediction - observed
    rar_residual = rar - observed
    table = sparc[
        [
            "galaxy",
            "galaxy_index",
            "radius_adjusted_kpc",
            "velocity_observed_adjusted_kms",
            "velocity_error_total_kms",
            "g_bar_m_s2",
            "local_density_g_cm3",
        ]
    ].copy()
    table.insert(0, "variant", variant["name"])
    table["predicted_velocity_km_s"] = prediction
    table["local_without_memory_velocity_km_s"] = local_prediction
    table["radial_memory_velocity_change_km_s"] = (
        prediction - local_prediction
    )
    table["fixed_RAR_velocity_km_s"] = rar
    table["residual_km_s"] = residual
    per_galaxy = table.assign(
        squared_residual=np.square(residual),
        rar_squared_residual=np.square(rar_residual),
    ).groupby("galaxy", sort=False).agg(
        mse=("squared_residual", "mean"),
        rar_mse=("rar_squared_residual", "mean"),
    )
    return {
        "galaxies": int(sparc.galaxy.nunique()),
        "points": len(sparc),
        "RMSE_km_s": float(np.sqrt(np.mean(np.square(residual)))),
        "MAE_km_s": float(np.mean(np.abs(residual))),
        "chi2_per_point": float(np.mean(np.square(residual / uncertainty))),
        "mean_standardized_residual": float(np.mean(residual / uncertainty)),
        "median_extra_velocity_vs_RAR_km_s": float(np.median(prediction - rar)),
        "p95_absolute_extra_velocity_vs_RAR_km_s": float(
            np.percentile(np.abs(prediction - rar), 95.0)
        ),
        "galaxies_beating_fixed_RAR": int((per_galaxy.mse < per_galaxy.rar_mse).sum()),
    }, table


def build_hybrid_field(
    variant: dict,
    parameters,
    baryonic_anchors: pd.DataFrame,
    density_anchors: pd.DataFrame,
    protocol: dict,
    local_protocol: dict,
) -> tuple[RadialDeflectionField, pd.DataFrame]:
    raw = protocol["raw_lensing"]
    cutoff = float(raw["isolated_tail_cutoff_kpc"])
    radius = np.geomspace(0.1, cutoff, int(raw["radial_grid_points"]))
    baryon_radius = baryonic_anchors.radius_kpc.to_numpy(float)
    baryon_acceleration = np.power(
        10.0, baryonic_anchors.log_gbar.to_numpy(float)
    )
    gbar = loglog_interpolate_with_tails(
        radius, baryon_radius, baryon_acceleration, outer_slope=-2.0
    )
    density_radius = density_anchors.radius_kpc.to_numpy(float)
    density_values = density_anchors.local_density_g_cm3.to_numpy(float)
    density = loglog_interpolate_with_tails(
        radius, density_radius, density_values
    )
    hybrid = screened_hybrid_profile_response(
        gbar,
        density,
        radius,
        parameters,
        variant["settings"],
        g_reference_m_s2=float(protocol["shared_constants"]["g_reference_m_s2"]),
        g_dagger_m_s2=float(protocol["shared_constants"]["g_dagger_m_s2"]),
        acceleration_screen_m_s2=float(
            protocol["shared_constants"]["acceleration_screen_m_s2"]
        ),
    )
    enhancement = hybrid["enhancement"]
    acceleration = gbar * enhancement

    def lookup(target):
        return np.exp(
            np.interp(np.log(target), np.log(radius), np.log(acceleration))
        )

    impact_arcsec = np.geomspace(
        0.05, 500.0, int(raw["impact_grid_points"])
    )
    impact_kpc = impact_arcsec * float(
        local_protocol["cosmology_and_coordinates"][
            "angular_scale_kpc_per_arcsec"
        ]
    )
    alpha = spherical_deflection_radians(
        impact_kpc,
        lookup,
        maximum_radius_kpc=cutoff,
        integration_points=int(raw["line_of_sight_integration_points"]),
    )
    field = RadialDeflectionField(impact_arcsec, alpha)
    sample_index = np.unique(
        np.linspace(0, len(radius) - 1, 180).astype(int)
    )
    table = pd.DataFrame(
        {
            "variant": variant["name"],
            "radius_kpc": radius[sample_index],
            "gbar_m_s2": gbar[sample_index],
            "density_g_cm3": density[sample_index],
            "enhancement": enhancement[sample_index],
            "local_without_memory_enhancement": (
                1.0 + hybrid["local_fractional_excess"][sample_index]
            ),
            "radial_memory_average": hybrid[
                "radial_memory_average"
            ][sample_index],
            "predicted_acceleration_m_s2": acceleration[sample_index],
        }
    )
    return field, table


def run_raw_lensing(
    variants: list[dict],
    full_parameters: dict[str, np.ndarray],
    bridge: pd.DataFrame,
    protocol: dict,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_protocol_path = ROOT / protocol["inputs"]["raw_lensing_protocol"]
    base = json.loads(raw_protocol_path.read_text(encoding="utf-8"))
    base["optimization"]["maximum_function_evaluations"] = int(
        protocol["raw_lensing"]["maximum_function_evaluations"]
    )
    catalog = pd.read_csv(ROOT / protocol["inputs"]["raw_image_catalog"])
    tian = pd.read_csv(
        ROOT / protocol["inputs"]["tian_baryonic_profiles"],
        sep=r"\s+",
        names=[
            "system",
            "radius_kpc",
            "log_gbar",
            "log_gobs",
            "err_log_gbar",
            "err_log_gobs",
        ],
    )
    requested = set(protocol["raw_lensing"]["systems"])
    systems = [row for row in base["systems"] if row["system"] in requested]
    if len(systems) != len(requested):
        raise RuntimeError("raw-lensing system selection did not resolve")
    scores_by_system = {}
    predictions = []
    geometry_rows = []
    profiles = []
    for system_index, system in enumerate(systems):
        print(f"raw lensing system={system['label']}", flush=True)
        local = system_protocol(base, system)
        local["optimization"]["maximum_function_evaluations"] = int(
            protocol["raw_lensing"]["maximum_function_evaluations"]
        )
        images = load_system_images(catalog, system)
        training, heldout = predictive_split(images)
        baryonic = load_anchors(tian, system["label"])
        density = bridge[
            bridge.domain.eq("cluster") & bridge.system.eq(system["label"])
        ].sort_values("radius_kpc")
        if len(density) < 3:
            raise RuntimeError(f"missing density anchors for {system['label']}")
        fields = {}
        for variant in variants:
            fields[variant["name"]], profile = build_hybrid_field(
                variant,
                full_parameters[variant["name"]],
                baryonic,
                density,
                protocol,
                local,
            )
            profile.insert(0, "system", system["system"])
            profiles.append(profile)
        lens = RawLens(local, fields)
        scores_by_system[system["system"]] = {}
        previous = None
        for variant_index, variant in enumerate(variants):
            name = variant["name"]
            print(f"  formula={name}", flush=True)
            fitted = lens.fit(
                name,
                training,
                starts=int(protocol["raw_lensing"]["geometry_multi_starts"]),
                seed=(
                    int(protocol["optimization"]["bridge_seed"])
                    + 1000 * system_index
                    + variant_index
                ),
                initial_override=previous,
            )
            previous = fitted["result"].x
            train_prediction = lens.exact_predictions(
                name,
                fitted["result"].x,
                fitted["sources"],
                training,
                stage="training",
            )
            heldout_prediction = lens.exact_predictions(
                name,
                fitted["result"].x,
                fitted["sources"],
                heldout,
                stage="heldout",
            )
            for table in (train_prediction, heldout_prediction):
                table.insert(0, "system", system["system"])
                predictions.append(table)
            scores_by_system[system["system"]][name] = {
                "training": score(
                    train_prediction,
                    lens.sigma,
                    free_parameters=len(fitted["result"].x),
                ),
                "heldout": score(heldout_prediction, lens.sigma),
            }
            geometry_rows.append(
                {
                    "system": system["system"],
                    "variant": name,
                    **dict(
                        zip(
                            spec_for(name).labels,
                            fitted["result"].x,
                            strict=True,
                        )
                    ),
                }
            )
    aggregate = {}
    for variant in variants:
        rows = [
            scores_by_system[system["system"]][variant["name"]]["heldout"]
            for system in systems
        ]
        aggregate[variant["name"]] = aggregate_system_scores(rows)
    return (
        {"aggregate": aggregate, "per_system": scores_by_system},
        pd.concat(predictions, ignore_index=True),
        pd.DataFrame(geometry_rows),
        pd.concat(profiles, ignore_index=True),
    )


def family_impacts(scores: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "bridge_RMSE_dex",
        "SPARC_outer_RMSE_km_s",
        "raw_lensing_RMS_arcsec",
        "solar_maximum_fractional_change",
        "Mercury_precession_mas_per_century",
    ]
    rows = []
    for family, block in scores.groupby("family", sort=False):
        ordered = block.sort_values("fixed_value")
        for metric in metrics:
            valid = ordered[["fixed_value", metric]].replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            correlation = (
                float(spearmanr(valid.fixed_value, valid[metric]).statistic)
                if len(valid) >= 3
                else math.nan
            )
            rows.append(
                {
                    "family": family,
                    "fixed_name": ordered.fixed_name.iloc[0],
                    "metric": metric,
                    "minimum": float(valid[metric].min()) if len(valid) else math.nan,
                    "maximum": float(valid[metric].max()) if len(valid) else math.nan,
                    "absolute_span": (
                        float(valid[metric].max() - valid[metric].min())
                        if len(valid)
                        else math.nan
                    ),
                    "spearman_r": correlation,
                }
            )
    return pd.DataFrame(rows)


def make_figure(scores: pd.DataFrame, impacts: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True)
    families = scores.family.unique()
    palette = plt.get_cmap("tab20")(
        np.linspace(0.0, 1.0, max(len(families), 2))
    )
    colors = {
        family: color
        for family, color in zip(families, palette, strict=True)
    }
    for family, block in scores.groupby("family", sort=False):
        ordered = block.sort_values("fixed_value")
        axes[0].plot(
            ordered.SPARC_outer_RMSE_km_s,
            ordered.raw_lensing_RMS_arcsec,
            "o-",
            color=colors[family],
            label=family,
        )
        axes[1].plot(
            ordered.bridge_RMSE_dex,
            ordered.SPARC_outer_RMSE_km_s,
            "o-",
            color=colors[family],
            label=family,
        )
    axes[0].set(
        xlabel="SPARC outer RMSE (km/s)",
        ylabel="4-cluster raw held-out RMS (arcsec)",
        title="Raw lensing versus galaxy transfer",
    )
    axes[1].set(
        xlabel="BCG+CLASH held-out RMSE (dex)",
        ylabel="SPARC outer RMSE (km/s)",
        title="Derived bridge versus galaxy transfer",
    )
    span = impacts.pivot(index="family", columns="metric", values="absolute_span")
    plotted = span[
        [
            "bridge_RMSE_dex",
            "SPARC_outer_RMSE_km_s",
            "raw_lensing_RMS_arcsec",
        ]
    ].copy()
    plotted = plotted / plotted.max(axis=0)
    plotted.plot.bar(ax=axes[2])
    axes[2].set(
        ylabel="span / largest family span",
        title="Which formula change moves each test most?",
    )
    axes[2].tick_params(axis="x", rotation=30)
    for axis in axes[:2]:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    axes[2].legend(fontsize=7)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/reopened_hybrid_sensitivity_protocol.json",
        help="Protocol path relative to the research root",
    )
    arguments = parser.parse_args()
    config_path = ROOT / arguments.config
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_reopened_hybrid_scores":
        raise RuntimeError("reopened-hybrid protocol was not frozen before scoring")
    variants = expand_variants(protocol)
    bridge_path = ROOT / protocol["inputs"]["bridge_sample"]
    bridge = pd.read_csv(bridge_path)
    bridge = assign_group_folds(
        bridge.drop(columns=["fold"], errors="ignore"),
        int(protocol["optimization"]["bridge_folds"]),
        int(protocol["optimization"]["bridge_seed"]),
    )
    sparc_path = ROOT / protocol["inputs"]["sparc_outer_sample"]
    sparc = pd.read_csv(sparc_path)
    if len(sparc) != 968 or sparc.galaxy.nunique() != 131:
        raise RuntimeError("SPARC outer sample changed")

    results = {}
    full_parameters = {}
    bridge_tables = []
    sparc_tables = []
    for variant_index, variant in enumerate(variants):
        print(f"bridge CV {variant['name']}", flush=True)
        heldout, fold_parameters, parameters = cross_validate_bridge(
            bridge, variant, protocol, variant_index
        )
        full_parameters[variant["name"]] = parameters
        bridge_metric = domain_metrics(bridge, heldout)
        bridge_table = bridge[
            ["domain", "system", "radius_kpc", "log_gbar", "log_gobs", "fold"]
        ].copy()
        bridge_table.insert(0, "variant", variant["name"])
        bridge_table["predicted_log_gobs"] = heldout
        bridge_table["residual_dex"] = heldout - bridge_table.log_gobs
        bridge_tables.append(bridge_table)
        print(f"SPARC transfer {variant['name']}", flush=True)
        sparc_metric, sparc_table = sparc_scores(
            sparc,
            parameters,
            variant,
            protocol["shared_constants"],
            protocol.get("tidal_geometry"),
        )
        sparc_tables.append(sparc_table)
        solar = solar_system_diagnostics(
            parameters,
            variant["settings"],
            cassini_fractional_limit=float(
                protocol["solar_tests"]["cassini_fractional_force_proxy_limit"]
            ),
            interplanetary_density_g_cm3=float(
                protocol["shared_constants"]["interplanetary_density_g_cm3"]
            ),
            acceleration_screen_m_s2=float(
                protocol["shared_constants"]["acceleration_screen_m_s2"]
            ),
        )
        precession = mercury_precession_mas_per_century(
            parameters,
            variant["settings"],
            interplanetary_density_g_cm3=float(
                protocol["shared_constants"]["interplanetary_density_g_cm3"]
            ),
            acceleration_screen_m_s2=float(
                protocol["shared_constants"]["acceleration_screen_m_s2"]
            ),
        )
        results[variant["name"]] = {
            "variant": variant,
            "bridge": bridge_metric,
            "SPARC": sparc_metric,
            "solar": solar,
            "Mercury_precession_mas_per_century": precession,
            "full_fit_parameters": dict(
                zip(
                    protocol["universal_parameters"]["names"],
                    map(float, parameters),
                    strict=True,
                )
            ),
            "fold_parameters": fold_parameters,
            "full_fit_at_boundary": at_boundary(parameters, protocol),
        }

    baseline_name = protocol.get(
        "baseline_variant_name", "interaction:interaction_eta=0"
    )
    baseline = next(row for row in variants if row["name"] == baseline_name)
    baseline_parameters = full_parameters[baseline["name"]]
    for variant in variants:
        locked_bridge = predict_log_acceleration(
            bridge,
            baseline_parameters,
            variant,
            protocol["shared_constants"],
        )
        locked_sparc, _ = sparc_scores(
            sparc,
            baseline_parameters,
            variant,
            protocol["shared_constants"],
            protocol.get("tidal_geometry"),
        )
        results[variant["name"]]["locked_baseline_parameter_sensitivity"] = {
            "bridge": domain_metrics(bridge, locked_bridge),
            "SPARC": locked_sparc,
        }

    print("begin raw four-cluster lensing transfer", flush=True)
    raw, raw_predictions, raw_geometry, raw_profiles = run_raw_lensing(
        variants, full_parameters, bridge, protocol
    )
    for variant in variants:
        results[variant["name"]]["raw_lensing"] = raw["aggregate"][
            variant["name"]
        ]

    raw_reference_path = ROOT / protocol["inputs"]["raw_reference_report"]
    raw_reference = json.loads(raw_reference_path.read_text(encoding="utf-8"))
    references = {
        "SPARC_fixed_RAR_outer_RMSE_km_s": float(
            np.sqrt(
                np.mean(
                    np.square(
                        sparc.velocity_predicted_kms.to_numpy(float)
                        - sparc.velocity_observed_adjusted_kms.to_numpy(float)
                    )
                )
            )
        ),
        "raw_baryons_RMS_arcsec": raw_reference["primary_aggregate"][
            "baryons_GR"
        ]["equal_system_radial_RMS_arcsec"],
        "raw_simple_MOND_RMS_arcsec": raw_reference["primary_aggregate"][
            "fixed_simple_MOND"
        ]["equal_system_radial_RMS_arcsec"],
        "raw_compact_halo_RMS_arcsec": raw_reference["primary_aggregate"][
            "GR_plus_cluster_halo"
        ]["equal_system_radial_RMS_arcsec"],
    }
    solar_gates = protocol["solar_tests"]
    rows = []
    for variant in variants:
        result = results[variant["name"]]
        raw_score = result["raw_lensing"]["equal_system_radial_RMS_arcsec"]
        row = {
            "variant": variant["name"],
            "family": variant["family"],
            "fixed_name": variant["fixed_name"],
            "fixed_value": variant["fixed_value"],
            "bridge_RMSE_dex": result["bridge"]["equal_domain_RMSE_dex"],
            "BCG_RMSE_dex": result["bridge"]["BCG"]["equal_system_RMSE_dex"],
            "CLASH_RMSE_dex": result["bridge"]["cluster"][
                "equal_system_RMSE_dex"
            ],
            "SPARC_outer_RMSE_km_s": result["SPARC"]["RMSE_km_s"],
            "SPARC_galaxies_beating_RAR": result["SPARC"][
                "galaxies_beating_fixed_RAR"
            ],
            "raw_lensing_RMS_arcsec": raw_score,
            "raw_all_roots_converged": result["raw_lensing"][
                "all_roots_converged"
            ],
            "solar_maximum_fractional_change": result["solar"][
                "maximum_fractional_change_limb_to_Saturn"
            ],
            "Mercury_precession_mas_per_century": result[
                "Mercury_precession_mas_per_century"
            ],
            "Cassini_proxy_pass": result["solar"]["Cassini_proxy_pass"],
            "Earth_pass": abs(
                result["solar"]["Earth_orbit_fractional_change"]
            )
            <= float(solar_gates["earth_orbit_fractional_change_max"]),
            "Mercury_pass": abs(
                result["Mercury_precession_mas_per_century"]
            )
            <= float(
                solar_gates[
                    "mercury_supplementary_precession_absolute_max_mas_per_century"
                ]
            ),
            "any_universal_parameter_at_boundary": any(
                result["full_fit_at_boundary"].values()
            ),
        }
        row["cross_domain_reference_ratio"] = max(
            row["SPARC_outer_RMSE_km_s"]
            / references["SPARC_fixed_RAR_outer_RMSE_km_s"],
            row["raw_lensing_RMS_arcsec"]
            / references["raw_compact_halo_RMS_arcsec"],
        )
        rows.append(row)
    scores = pd.DataFrame(rows)
    impact_scores = scores.copy()
    impact_scores.loc[
        ~impact_scores.raw_all_roots_converged.astype(bool),
        "raw_lensing_RMS_arcsec",
    ] = np.nan
    impacts = family_impacts(impact_scores)
    for index, row in scores.iterrows():
        eligible = scores[
            scores.Cassini_proxy_pass
            & scores.Earth_pass
            & scores.Mercury_pass
            & scores.raw_all_roots_converged
        ]
        dominated = (
            (eligible.bridge_RMSE_dex <= row.bridge_RMSE_dex)
            & (eligible.SPARC_outer_RMSE_km_s <= row.SPARC_outer_RMSE_km_s)
            & (eligible.raw_lensing_RMS_arcsec <= row.raw_lensing_RMS_arcsec)
            & (
                (eligible.bridge_RMSE_dex < row.bridge_RMSE_dex)
                | (eligible.SPARC_outer_RMSE_km_s < row.SPARC_outer_RMSE_km_s)
                | (eligible.raw_lensing_RMS_arcsec < row.raw_lensing_RMS_arcsec)
            )
        ).any()
        scores.loc[index, "cross_domain_Pareto"] = bool(
            row.Cassini_proxy_pass
            and row.Earth_pass
            and row.Mercury_pass
            and row.raw_all_roots_converged
            and not dominated
        )

    ranking_all = scores.sort_values(
        ["cross_domain_reference_ratio", "bridge_RMSE_dex"]
    )
    solar_valid = scores[
        scores.Cassini_proxy_pass
        & scores.Earth_pass
        & scores.Mercury_pass
        & scores.raw_all_roots_converged
    ]
    ranking_valid = solar_valid.sort_values(
        ["cross_domain_reference_ratio", "bridge_RMSE_dex"]
    )
    impact_ranking = (
        impacts.groupby("family", as_index=False)
        .agg(total_normalized_span=("absolute_span", "sum"))
        .sort_values("total_normalized_span", ascending=False)
    )
    output = (ROOT / protocol["outputs"]["report"]).parent
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(ROOT / protocol["outputs"]["scores"], index=False)
    impacts.to_csv(ROOT / protocol["outputs"]["family_impacts"], index=False)
    pd.concat(bridge_tables, ignore_index=True).to_csv(
        ROOT / protocol["outputs"]["bridge_predictions"], index=False
    )
    pd.concat(sparc_tables, ignore_index=True).to_csv(
        ROOT / protocol["outputs"]["sparc_predictions"], index=False
    )
    raw_predictions.to_csv(
        ROOT / protocol["outputs"]["raw_predictions"], index=False
    )
    raw_geometry.to_csv(
        ROOT / protocol["outputs"]["raw_geometry"], index=False
    )
    raw_profiles.to_csv(output / "raw_radial_profiles.csv", index=False)
    make_figure(scores, impacts, ROOT / protocol["outputs"]["figure"])

    report = {
        "report_version": protocol["protocol_version"],
        "status": "completed reopened hybrid cross-domain sensitivity study",
        "protocol": {
            "path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(config_path),
        },
        "input_hashes": {
            "bridge_sample": sha256(bridge_path),
            "SPARC_outer_sample": sha256(sparc_path),
            "raw_image_catalog": sha256(
                ROOT / protocol["inputs"]["raw_image_catalog"]
            ),
            "Tian_baryonic_profiles": sha256(
                ROOT / protocol["inputs"]["tian_baryonic_profiles"]
            ),
        },
        "coverage": {
            "variants": len(variants),
            "bridge_rows": len(bridge),
            "bridge_systems": int(bridge.system.nunique()),
            "SPARC_galaxies": int(sparc.galaxy.nunique()),
            "SPARC_outer_points": len(sparc),
            "raw_lensing_systems": len(protocol["raw_lensing"]["systems"]),
            "raw_heldout_images": int(
                raw_predictions.stage.eq("heldout").sum() / len(variants)
            ),
        },
        "baseline_for_locked_parameter_sensitivity": baseline["name"],
        "references": references,
        "ranking_all_scored": ranking_all.variant.tolist(),
        "ranking_solar_valid_complete_raw": ranking_valid.variant.tolist(),
        "Pareto_front": scores[
            scores.cross_domain_Pareto.astype(bool)
        ].variant.tolist(),
        "family_impacts": impacts.to_dict(orient="records"),
        "results": results,
        "raw_per_system": raw["per_system"],
        "claim_boundary": protocol["claim_boundary"],
    }
    (ROOT / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )

    top = ranking_valid.iloc[0] if len(ranking_valid) else ranking_all.iloc[0]
    top_scope = (
        "Best fully evaluable Solar-valid cross-domain compromise"
        if len(ranking_valid)
        else "Best numerical compromise; no formula passed every Solar/raw check"
    )
    largest_by_metric = (
        impacts.loc[impacts.groupby("metric").absolute_span.idxmax()]
        .set_index("metric")["family"]
        .to_dict()
    )
    lines = [
        "# Reopened Sigma/RG hybrid sensitivity",
        "",
        f"Tested **{len(variants)}** controlled formula variations on 64 bridge systems, "
        "131 SPARC galaxies, four raw strong-lensing clusters, and Solar-System proxies.",
        "",
        f"## {top_scope}",
        "",
        f"`{top.variant}`",
        "",
        f"- BCG+CLASH bridge RMSE: {top.bridge_RMSE_dex:.4f} dex",
        f"- SPARC outer RMSE: {top.SPARC_outer_RMSE_km_s:.3f} km/s "
        f"(RAR reference {references['SPARC_fixed_RAR_outer_RMSE_km_s']:.3f})",
        f"- Four-cluster raw held-out RMS: {top.raw_lensing_RMS_arcsec:.3f} arcsec "
        f"(compact-halo reference {references['raw_compact_halo_RMS_arcsec']:.3f})",
        f"- Mercury supplementary precession proxy: "
        f"{top.Mercury_precession_mas_per_century:.3e} mas/century",
        "",
        "## Most impactful controlled change by metric",
        "",
    ]
    for metric, family in largest_by_metric.items():
        lines.append(f"- {metric}: **{family}**")
    lines += [
        "",
        "These rankings identify leverage, not confirmation. Derived CLASH accelerations, "
        "the zero-slip raw-lensing closure, and the Solar force-fraction proxy remain model-dependent.",
    ]
    (ROOT / protocol["outputs"]["summary"]).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "variants": len(variants),
                "best": top.variant,
                "best_scope": top_scope,
                "Pareto_front": report["Pareto_front"],
                "largest_impact_by_metric": largest_by_metric,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
