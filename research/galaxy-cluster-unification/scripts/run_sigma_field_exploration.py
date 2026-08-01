#!/usr/bin/env python3
"""Run the frozen exploratory screened Sigma-field tests."""

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
sys.path.insert(0, str(ROOT / "scripts"))

from run_rxj2129_raw_theory_lensing import RawLens, load_images, score
from voidscreen.axisymmetric_permittivity import (
    AxisymmetricGrid,
    acceleration_components,
    double_exponential_density,
    hernquist_density,
    midplane_inward_acceleration,
    solve_axisymmetric_potential,
)
from voidscreen.phenomenology import fixed_rar_enhancement
from voidscreen.raw_lensing import (
    RadialDeflectionField,
    loglog_interpolate_with_tails,
    spherical_deflection_radians,
)
from voidscreen.sigma_field import (
    KPC_CM,
    MSUN_G,
    MSUN_KPC3_TO_G_CM3,
    geometric_radial_faces,
    hernquist_density_g_cm3,
    hernquist_enclosed_mass_solar,
    radial_cell_centers,
    sigma_permittivity,
    solve_axisymmetric_sigma,
    solve_spherical_sigma,
)


G_SI = 6.67430e-11
KPC_M = KPC_CM / 100.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def log_slope(radius, values, low: float, high: float) -> float:
    radius = np.asarray(radius, dtype=float)
    values = np.asarray(values, dtype=float)
    use = (radius >= low) & (radius <= high) & (values > 0.0) & np.isfinite(values)
    if use.sum() < 3:
        return math.nan
    return float(np.polyfit(np.log10(radius[use]), np.log10(values[use]), 1)[0])


def interpolate_log_radius(radius, values, targets):
    return np.interp(np.log(targets), np.log(radius), values)


def load_rxj_profiles() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    tian = pd.read_csv(
        ROOT / "data" / "raw" / "clash_tian2020" / "fig2.dat",
        sep=r"\s+",
        names=["system", "radius_kpc", "log_gbar", "log_gobs", "err_gbar", "err_gobs"],
    )
    tian = tian[tian["system"] == "RXJ2129"].sort_values("radius_kpc").reset_index(drop=True)
    sample = pd.read_csv(ROOT / "results" / "rar_sharp_coherence_rg_sweep" / "sample.csv")
    density = (
        sample[(sample["domain"] == "cluster") & (sample["system"] == "RXJ2129")]
        .sort_values("radius_kpc")
        .drop_duplicates("radius_kpc")
    )
    density_radius = density["radius_kpc"].to_numpy(float)
    density_values = density["local_density_g_cm3"].to_numpy(float)
    value_600 = loglog_interpolate_with_tails(
        [600.0], density_radius, density_values
    )[0]
    density_radius = np.r_[density_radius, 600.0]
    density_values = np.r_[density_values, value_600]
    return tian, density_radius, density_values


def galaxy_solution(protocol: dict, rho_s: float, length: float):
    settings = protocol["tests"]["spherical_galaxy"]
    faces = geometric_radial_faces(560, 0.01, 1000.0)
    radius = radial_cell_centers(faces)
    density = hernquist_density_g_cm3(
        radius, float(settings["mass_solar"]), float(settings["scale_kpc"])
    )
    solution = solve_spherical_sigma(
        faces, density, rho_s_g_cm3=rho_s, length_kpc=length, outer_sigma=1.0
    )
    enclosed = hernquist_enclosed_mass_solar(
        radius, float(settings["mass_solar"]), float(settings["scale_kpc"])
    )
    gbar = G_SI * enclosed * (MSUN_G / 1000.0) / np.square(radius * KPC_M)
    return radius, density, solution, gbar


def cluster_solution(rho_s: float, length: float):
    tian, density_radius, density_values = load_rxj_profiles()
    faces = geometric_radial_faces(620, 0.05, 10000.0)
    radius = radial_cell_centers(faces)
    density = loglog_interpolate_with_tails(
        radius, density_radius, density_values
    )
    solution = solve_spherical_sigma(
        faces, density, rho_s_g_cm3=rho_s, length_kpc=length, outer_sigma=1.0
    )
    return tian, density_radius, density_values, radius, density, solution


def run_parameter_grid(protocol: dict) -> tuple[pd.DataFrame, dict, dict]:
    grid = protocol["universal_parameter_grid"]
    galaxy_cache = {}
    cluster_cache = {}
    records = []
    gdagger = 1.2e-10
    galaxy_settings = protocol["tests"]["spherical_galaxy"]
    low, high = map(float, galaxy_settings["score_radii_kpc"])
    far_low, far_high = map(float, galaxy_settings["far_slope_radii_kpc"])

    for log_rho in grid["log10_rho_s_g_cm3"]:
        rho_s = 10.0 ** float(log_rho)
        for length in grid["L_Sigma_kpc"]:
            key = (float(log_rho), float(length))
            galaxy_cache[key] = galaxy_solution(protocol, rho_s, float(length))
            cluster_cache[key] = cluster_solution(rho_s, float(length))
            galaxy_radius, _, galaxy_sigma_solution, galaxy_gbar = galaxy_cache[key]
            tian, _, _, cluster_radius, _, cluster_sigma_solution = cluster_cache[key]
            target_radius = tian["radius_kpc"].to_numpy(float)
            sigma_cluster = interpolate_log_radius(
                cluster_radius, cluster_sigma_solution.field, target_radius
            )

            for eta in grid["eta"]:
                eta = float(eta)
                galaxy_epsilon = sigma_permittivity(galaxy_sigma_solution.field, eta)
                galaxy_g = galaxy_gbar / galaxy_epsilon
                rar_g = galaxy_gbar * fixed_rar_enhancement(galaxy_gbar, gdagger)
                use = (galaxy_radius >= low) & (galaxy_radius <= high)
                rar_rmse = float(
                    np.sqrt(np.mean(np.square(np.log10(galaxy_g[use] / rar_g[use]))))
                )
                velocity = np.sqrt(galaxy_g * galaxy_radius * KPC_M) / 1000.0
                cluster_epsilon = sigma_permittivity(sigma_cluster, eta)
                cluster_log_prediction = tian["log_gbar"].to_numpy(float) - np.log10(
                    cluster_epsilon
                )
                cluster_residual = cluster_log_prediction - tian["log_gobs"].to_numpy(float)
                cluster_rmse = float(np.sqrt(np.mean(np.square(cluster_residual))))
                records.append(
                    {
                        "eta": eta,
                        "log10_rho_s_g_cm3": float(log_rho),
                        "L_Sigma_kpc": float(length),
                        "galaxy_RAR_RMSE_dex_5_50kpc": rar_rmse,
                        "galaxy_velocity_log_slope_10_50kpc": log_slope(
                            galaxy_radius, velocity, 10.0, 50.0
                        ),
                        "galaxy_velocity_log_slope_100_250kpc": log_slope(
                            galaxy_radius, velocity, far_low, far_high
                        ),
                        "galaxy_enhancement_at_20kpc": float(
                            interpolate_log_radius(
                                galaxy_radius, 1.0 / galaxy_epsilon, np.asarray([20.0])
                            )[0]
                        ),
                        "RXJ2129_derived_field_RMSE_dex": cluster_rmse,
                        "RXJ2129_mean_log_residual_dex": float(np.mean(cluster_residual)),
                        "RXJ2129_enhancement_at_100kpc": float(1.0 / cluster_epsilon[1]),
                        "joint_descriptive_score_dex": float(
                            np.sqrt(0.5 * (rar_rmse**2 + cluster_rmse**2))
                        ),
                        "galaxy_sigma_converged": galaxy_sigma_solution.converged,
                        "cluster_sigma_converged": cluster_sigma_solution.converged,
                    }
                )
    table = pd.DataFrame(records).sort_values("joint_descriptive_score_dex").reset_index(drop=True)
    return table, galaxy_cache, cluster_cache


def run_void_buildup(eta: float, ratios) -> pd.DataFrame:
    records = []
    for ratio in ratios:
        faces = geometric_radial_faces(180, ratio * 1.0e-5, ratio)
        solution = solve_spherical_sigma(
            faces,
            np.zeros(len(faces) - 1),
            rho_s_g_cm3=1.0,
            length_kpc=1.0,
            outer_sigma=0.0,
        )
        records.append(
            {
                "cavity_radius_over_L": ratio,
                "center_sigma": float(solution.field[0]),
                "center_permittivity": float(sigma_permittivity([solution.field[0]], eta)[0]),
                "center_gravity_multiplier_if_spherical_source_present": float(
                    1.0 / sigma_permittivity([solution.field[0]], eta)[0]
                ),
                "converged": solution.converged,
            }
        )
    return pd.DataFrame(records)


def run_axisymmetric(best: pd.Series, protocol: dict):
    settings = protocol["tests"]["disk_and_solar_environment"]
    grid = AxisymmetricGrid(96, 96, 60.0, 30.0)
    disk = double_exponential_density(
        grid,
        mass=float(settings["disk_mass_solar"]),
        radial_scale=float(settings["disk_scale_kpc"]),
        vertical_scale=float(settings["disk_half_thickness_kpc"]),
    )
    bulge = hernquist_density(
        grid,
        mass=float(settings["bulge_mass_solar"]),
        scale_radius=float(settings["bulge_scale_kpc"]),
    )
    density_msun_kpc3 = disk + bulge
    density_g_cm3 = density_msun_kpc3 * MSUN_KPC3_TO_G_CM3
    sigma_solution = solve_axisymmetric_sigma(
        grid,
        density_g_cm3,
        rho_s_g_cm3=10.0 ** float(best["log10_rho_s_g_cm3"]),
        length=float(best["L_Sigma_kpc"]),
        outer_sigma=1.0,
    )
    epsilon = sigma_permittivity(sigma_solution.field, float(best["eta"]))
    newtonian = solve_axisymmetric_potential(
        grid, density_msun_kpc3, np.ones_like(epsilon), far_permittivity=1.0
    )
    modified = solve_axisymmetric_potential(
        grid,
        density_msun_kpc3,
        epsilon,
        far_permittivity=1.0 - float(best["eta"]),
    )
    radial = grid.radial_centers
    mid_newtonian = midplane_inward_acceleration(grid, newtonian)
    mid_modified = midplane_inward_acceleration(grid, modified)
    enhancement = np.divide(
        mid_modified,
        mid_newtonian,
        out=np.full_like(mid_modified, np.nan),
        where=mid_newtonian > 0.0,
    )
    g_r_n, g_z_n = acceleration_components(grid, newtonian)
    g_r_m, g_z_m = acceleration_components(grid, modified)
    solar_i = int(np.argmin(np.abs(radial - float(settings["solar_radius_kpc"]))))
    probe_i = int(np.argmin(np.abs(radial - float(settings["probe_R_kpc"]))))
    probe_j = int(
        np.argmin(np.abs(grid.vertical_centers - float(settings["probe_z_kpc"])))
    )
    newtonian_direction = abs(g_z_n[probe_i, probe_j]) / abs(g_r_n[probe_i, probe_j])
    modified_direction = abs(g_z_m[probe_i, probe_j]) / abs(g_r_m[probe_i, probe_j])
    summary = {
        "sigma_solver_converged": sigma_solution.converged,
        "sigma_solver_iterations": sigma_solution.iterations,
        "solar_radius_grid_kpc": float(radial[solar_i]),
        "Sigma_at_solar_midplane": float(sigma_solution.field[solar_i, 0]),
        "epsilon_at_solar_midplane": float(epsilon[solar_i, 0]),
        "galactic_midplane_force_enhancement_at_solar_radius": float(enhancement[solar_i]),
        "probe_R_kpc": float(radial[probe_i]),
        "probe_z_kpc": float(grid.vertical_centers[probe_j]),
        "newtonian_vertical_to_radial_force_ratio": float(newtonian_direction),
        "modified_vertical_to_radial_force_ratio": float(modified_direction),
        "directional_focusing_ratio": float(modified_direction / newtonian_direction),
    }
    profile = pd.DataFrame(
        {
            "radius_kpc": radial,
            "midplane_Sigma": sigma_solution.field[:, 0],
            "midplane_density_g_cm3": density_g_cm3[:, 0],
            "midplane_epsilon": epsilon[:, 0],
            "midplane_force_enhancement": enhancement,
        }
    )
    return grid, density_g_cm3, sigma_solution.field, epsilon, profile, summary


def build_best_radial_profiles(best: pd.Series, protocol: dict, galaxy_cache, cluster_cache):
    key = (float(best["log10_rho_s_g_cm3"]), float(best["L_Sigma_kpc"]))
    galaxy_radius, galaxy_density, galaxy_solution_value, galaxy_gbar = galaxy_cache[key]
    galaxy_epsilon = sigma_permittivity(galaxy_solution_value.field, float(best["eta"]))
    galaxy_g = galaxy_gbar / galaxy_epsilon
    galaxy_rar = galaxy_gbar * fixed_rar_enhancement(galaxy_gbar, 1.2e-10)
    galaxy = pd.DataFrame(
        {
            "domain": "galaxy_archetype",
            "radius_kpc": galaxy_radius,
            "density_g_cm3": galaxy_density,
            "Sigma": galaxy_solution_value.field,
            "epsilon": galaxy_epsilon,
            "gbar_m_s2": galaxy_gbar,
            "gSigma_m_s2": galaxy_g,
            "comparison_g_m_s2": galaxy_rar,
        }
    )
    tian, _, _, cluster_radius, cluster_density, cluster_solution_value = cluster_cache[key]
    gbar_cluster = loglog_interpolate_with_tails(
        cluster_radius,
        tian["radius_kpc"].to_numpy(float),
        np.power(10.0, tian["log_gbar"].to_numpy(float)),
        outer_slope=-2.0,
    )
    cluster_epsilon = sigma_permittivity(cluster_solution_value.field, float(best["eta"]))
    cluster = pd.DataFrame(
        {
            "domain": "RXJ2129",
            "radius_kpc": cluster_radius,
            "density_g_cm3": cluster_density,
            "Sigma": cluster_solution_value.field,
            "epsilon": cluster_epsilon,
            "gbar_m_s2": gbar_cluster,
            "gSigma_m_s2": gbar_cluster / cluster_epsilon,
            "comparison_g_m_s2": loglog_interpolate_with_tails(
                cluster_radius,
                tian["radius_kpc"].to_numpy(float),
                np.power(10.0, tian["log_gobs"].to_numpy(float)),
            ),
        }
    )
    return pd.concat([galaxy, cluster], ignore_index=True)


def run_diagnostic_lensing(best: pd.Series, protocol: dict, radial_profiles: pd.DataFrame):
    raw_protocol = json.loads(
        (ROOT / "configs" / "rxj2129_raw_theory_lensing_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    cluster = radial_profiles[radial_profiles["domain"] == "RXJ2129"].copy()
    radius = cluster["radius_kpc"].to_numpy(float)
    acceleration = cluster["gSigma_m_s2"].to_numpy(float)

    def lookup(target):
        return np.exp(np.interp(np.log(target), np.log(radius), np.log(acceleration)))

    impact_arcsec = np.geomspace(0.05, 500.0, 700)
    scale_kpc_arcsec = float(
        raw_protocol["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"]
    )
    physical_alpha = spherical_deflection_radians(
        impact_arcsec * scale_kpc_arcsec,
        lookup,
        maximum_radius_kpc=float(radius.max()),
        integration_points=800,
    )
    field = RadialDeflectionField(impact_arcsec, physical_alpha)
    lens = RawLens(raw_protocol, {"locked_universal_candidate": field})
    images = load_images(raw_protocol)
    heldout_ids = set(raw_protocol["predictive_split"]["heldout"])
    heldout = images[images["image_id"].isin(heldout_ids)].copy()
    training = images[~images["image_id"].isin(heldout_ids)].copy()
    fit = lens.fit(
        "locked_universal_candidate",
        training,
        starts=8,
        seed=20260729,
    )
    training_prediction = lens.exact_predictions(
        "locked_universal_candidate",
        fit["result"].x,
        fit["sources"],
        training,
        stage="training",
    )
    heldout_prediction = lens.exact_predictions(
        "locked_universal_candidate",
        fit["result"].x,
        fit["sources"],
        heldout,
        stage="heldout",
    )
    predictions = pd.concat([training_prediction, heldout_prediction], ignore_index=True)
    summary = {
        "status": "spent-holdout zero-slip diagnostic",
        "training": score(training_prediction, lens.sigma, free_parameters=20),
        "heldout": score(heldout_prediction, lens.sigma),
        "geometry_parameters": {
            label: float(value)
            for label, value in zip(
                (
                    "axis_ratio_q",
                    "position_angle_phi_radian",
                    "center_x_arcsec",
                    "center_y_arcsec",
                    "external_shear_gamma1",
                    "external_shear_gamma2",
                ),
                fit["result"].x,
            )
        },
        "gravity_or_lensing_amplitudes_fit_to_images": 0,
    }
    return predictions, summary


def make_figure(
    best: pd.Series,
    profiles: pd.DataFrame,
    void: pd.DataFrame,
    axis_grid: AxisymmetricGrid,
    axis_sigma: np.ndarray,
    axis_profile: pd.DataFrame,
    lens_predictions: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), constrained_layout=True)
    galaxy = profiles[profiles["domain"] == "galaxy_archetype"]
    ax = axes[0, 0]
    radius = galaxy["radius_kpc"].to_numpy(float)
    velocity_sigma = np.sqrt(galaxy["gSigma_m_s2"] * radius * KPC_M) / 1000.0
    velocity_rar = np.sqrt(galaxy["comparison_g_m_s2"] * radius * KPC_M) / 1000.0
    velocity_bar = np.sqrt(galaxy["gbar_m_s2"] * radius * KPC_M) / 1000.0
    ax.semilogx(radius, velocity_bar, label="Newtonian baryons", color="grey")
    ax.semilogx(radius, velocity_rar, label="RAR comparison", color="#D95F02")
    ax.semilogx(radius, velocity_sigma, label="Sigma field", color="#1874CD")
    ax.set_xlim(1.0, 300.0)
    ax.set_xlabel("radius (kpc)")
    ax.set_ylabel("circular speed (km/s)")
    ax.set_title("1. Spherical galaxy scaling")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

    cluster = profiles[profiles["domain"] == "RXJ2129"]
    ax = axes[0, 1]
    ax.loglog(cluster["radius_kpc"], cluster["gbar_m_s2"], color="grey", label="baryons")
    ax.loglog(cluster["radius_kpc"], cluster["gSigma_m_s2"], color="#1874CD", label="Sigma")
    tian, _, _ = load_rxj_profiles()
    ax.errorbar(
        tian["radius_kpc"],
        np.power(10.0, tian["log_gobs"]),
        yerr=np.power(10.0, tian["log_gobs"]) * np.log(10.0) * tian["err_gobs"],
        fmt="o",
        color="black",
        label="derived target",
    )
    ax.set_xlim(10.0, 1000.0)
    ax.set_xlabel("radius (kpc)")
    ax.set_ylabel("acceleration (m/s²)")
    ax.set_title("2. RX J2129 transfer")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    ax.semilogx(void["cavity_radius_over_L"], void["center_sigma"], marker="o")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(r"cavity radius / $L_\Sigma$")
    ax.set_ylabel(r"central $\Sigma$")
    ax.set_title("4. Finite void buildup")
    ax.grid(alpha=0.2)

    ax = axes[1, 0]
    image = ax.pcolormesh(
        axis_grid.radial_centers,
        axis_grid.vertical_centers,
        axis_sigma.T,
        shading="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    fig.colorbar(image, ax=ax, label=r"$\Sigma$")
    ax.set_xlim(0.0, 30.0)
    ax.set_ylim(0.0, 12.0)
    ax.set_xlabel("R (kpc)")
    ax.set_ylabel("z (kpc)")
    ax.set_title("3 & 5. Disk screening geometry")

    ax = axes[1, 1]
    ax.semilogx(
        axis_profile["radius_kpc"],
        axis_profile["midplane_force_enhancement"],
        color="#1874CD",
    )
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_xlim(1.0, 40.0)
    ax.set_xlabel("midplane radius (kpc)")
    ax.set_ylabel("modified / Newtonian force")
    ax.set_title("Disk midplane response")
    ax.grid(alpha=0.2)

    ax = axes[1, 2]
    heldout = lens_predictions[lens_predictions["stage"] == "heldout"]
    ax.scatter(
        heldout["observed_x_arcsec"],
        heldout["observed_y_arcsec"],
        color="black",
        label="observed",
    )
    converged = heldout["root_converged"].astype(bool)
    ax.scatter(
        heldout.loc[converged, "predicted_x_arcsec"],
        heldout.loc[converged, "predicted_y_arcsec"],
        marker="x",
        color="#1874CD",
        label="Sigma prediction",
    )
    for row in heldout[converged].itertuples(index=False):
        ax.plot(
            [row.observed_x_arcsec, row.predicted_x_arcsec],
            [row.observed_y_arcsec, row.predicted_y_arcsec],
            color="#1874CD",
            alpha=0.5,
        )
    ax.set_aspect("equal")
    ax.set_xlabel("east offset (arcsec)")
    ax.set_ylabel("north offset (arcsec)")
    ax.set_title("6. Zero-slip lensing diagnostic")
    ax.legend(fontsize=8)

    fig.suptitle(
        "Screened Sigma-field exploration: "
        f"eta={best['eta']:.2f}, log rho_s={best['log10_rho_s_g_cm3']:.1f}, "
        f"L={best['L_Sigma_kpc']:.1f} kpc",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


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
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", default=ROOT / "configs" / "sigma_field_exploration_protocol.json", type=Path
    )
    args = parser.parse_args()
    config_path = args.protocol.resolve()
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / "results" / "sigma_field_exploration"
    output.mkdir(parents=True, exist_ok=True)

    parameter_grid, galaxy_cache, cluster_cache = run_parameter_grid(protocol)
    best = parameter_grid.iloc[0]
    print("Best descriptive grid row:", best.to_dict(), flush=True)
    radial_profiles = build_best_radial_profiles(best, protocol, galaxy_cache, cluster_cache)
    void = run_void_buildup(
        float(best["eta"]), protocol["tests"]["void_buildup"]["cavity_radius_over_L"]
    )
    axis_grid, _, axis_sigma, _, axis_profile, axis_summary = run_axisymmetric(best, protocol)
    print("Axisymmetric summary:", axis_summary, flush=True)
    lens_predictions, lens_summary = run_diagnostic_lensing(best, protocol, radial_profiles)
    print("Lensing summary:", lens_summary, flush=True)

    parameter_grid.to_csv(output / "parameter_grid.csv", index=False)
    radial_profiles.to_csv(output / "radial_profiles.csv", index=False)
    void.to_csv(output / "void_buildup.csv", index=False)
    axis_profile.to_csv(output / "axisymmetric_profiles.csv", index=False)
    lens_predictions.to_csv(output / "raw_lensing_predictions.csv", index=False)
    make_figure(
        best,
        radial_profiles,
        void,
        axis_grid,
        axis_sigma,
        axis_profile,
        lens_predictions,
        output / "sigma_field_exploration.png",
    )

    converged_grid = parameter_grid[
        parameter_grid["galaxy_sigma_converged"]
        & parameter_grid["cluster_sigma_converged"]
    ]
    galaxy_best = converged_grid.loc[
        converged_grid["galaxy_RAR_RMSE_dex_5_50kpc"].idxmin()
    ]
    cluster_best = converged_grid.loc[
        converged_grid["RXJ2129_derived_field_RMSE_dex"].idxmin()
    ]
    flattest = converged_grid.loc[
        converged_grid["galaxy_velocity_log_slope_10_50kpc"].abs().idxmin()
    ]
    report = {
        "report_version": "SIGMA-FIELD-EXPLORATION-0.1",
        "status": "completed permissive exploratory tests",
        "protocol": {
            "path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(config_path),
            "freeze_status": protocol["status"],
        },
        "equations": protocol["equations"],
        "grid_rows": len(parameter_grid),
        "grid_convergence": {
            "both_spherical_solutions_converged": len(converged_grid),
            "not_both_converged": len(parameter_grid) - len(converged_grid),
        },
        "best_descriptive_grid_row": best.to_dict(),
        "top_10_grid_rows": parameter_grid.head(10).to_dict(orient="records"),
        "parameter_tradeoff": {
            "best_galaxy_only_row": galaxy_best.to_dict(),
            "best_cluster_derived_target_only_row": cluster_best.to_dict(),
            "flattest_10_to_50_kpc_row": flattest.to_dict(),
        },
        "void_buildup": void.to_dict(orient="records"),
        "void_linear_stability": {
            "prediction": "Sigma=0 loses stability when cavity radius / L_Sigma exceeds pi",
            "derivation": "lowest spherical Dirichlet Laplacian eigenvalue is (pi/R)^2",
        },
        "axisymmetric_disk_and_solar_environment": axis_summary,
        "raw_lensing_diagnostic": lens_summary,
        "scope": {
            "RAR_or_MOND_term_inserted": False,
            "class_or_coherence_label_inserted": False,
            "cluster_specific_gravity_parameter": False,
            "relativistic_completion": False,
            "lensing_closure": "same-potential zero-slip diagnostic",
            "RXJ2129_independent_validation": False,
        },
        "outputs": {
            **protocol["outputs"],
            "raw_lensing_predictions": "results/sigma_field_exploration/raw_lensing_predictions.csv",
        },
    }
    (output / "report.json").write_text(
        json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
