#!/usr/bin/env python3
"""Commission real 2D galaxy maps and 3D field solves on project-spent DDO154."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.data import load_curves
from voidscreen.field_solvers import (
    cell_coordinates,
    simple_mond_acceleration,
    solve_aqual,
    solve_newtonian,
    solve_qumond,
    surface_density_to_volume,
)
from voidscreen.galaxy_maps import (
    deproject_to_disk_grid,
    hi_moment0_surface_density,
    integrated_hi_mass_solar,
    normalize_surface_density_mass,
    optical_morphology_map,
    weighted_disk_geometry,
)

DEFAULT_CONFIG = ROOT / "configs" / "p0635_ddo154_map_commissioning.json"
DEFAULT_RAW = ROOT / "data" / "raw" / "p0635_commissioning_ddo154"
DEFAULT_OUTPUT = ROOT / "results" / "p0635_ddo154_map_commissioning"
SPARC = ROOT / "data" / "raw" / "sparc"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def optical_detector_center(header, detector_center_pixel: float) -> tuple[float, float]:
    """Map the original detector center into a cropped IRAF image (zero based)."""

    x = header["CRPIX1"] + (detector_center_pixel - header["CRVAL1"]) / header["CDELT1"] - 1.0
    y = header["CRPIX2"] + (detector_center_pixel - header["CRVAL2"]) / header["CDELT2"] - 1.0
    return float(x), float(y)


def radial_circular_speed(solution, axis_kpc: np.ndarray) -> pd.DataFrame:
    x, y, _ = cell_coordinates(solution.potential.shape, float(np.diff(axis_kpc)[0]))
    middle = (solution.potential.shape[2] - 1) // 2
    x2 = x[:, :, middle]
    y2 = y[:, :, middle]
    radius = np.hypot(x2, y2)
    ax = solution.acceleration[0][:, :, middle]
    ay = solution.acceleration[1][:, :, middle]
    inward = np.zeros_like(radius)
    nonzero = radius > 0.0
    inward[nonzero] = -(ax[nonzero] * x2[nonzero] + ay[nonzero] * y2[nonzero]) / radius[nonzero]
    spacing = float(np.diff(axis_kpc)[0])
    rows = []
    for radial in np.arange(spacing, float(axis_kpc[-1]) - spacing / 2.0, spacing):
        ring = (radius >= radial - spacing / 2.0) & (radius < radial + spacing / 2.0)
        values = inward[ring]
        values = values[np.isfinite(values)]
        if values.size < 8:
            continue
        mean_inward = float(np.mean(values))
        rows.append(
            {
                "radius_kpc": float(radial),
                "inward_acceleration_km2_s2_kpc": mean_inward,
                "circular_speed_km_s": float(np.sqrt(max(radial * mean_inward, 0.0))),
                "azimuthal_scatter_fraction": float(
                    np.std(values) / max(abs(mean_inward), np.finfo(float).tiny)
                ),
                "ring_cells": int(values.size),
            }
        )
    return pd.DataFrame(rows)


def score_curve(radius, predicted, observed_radius, observed, uncertainty) -> dict[str, float | int]:
    valid = (observed_radius >= np.min(radius)) & (observed_radius <= np.max(radius))
    model = np.interp(observed_radius[valid], radius, predicted)
    residual = model - observed[valid]
    return {
        "points": int(np.count_nonzero(valid)),
        "RMSE_km_s": float(np.sqrt(np.mean(np.square(residual)))),
        "weighted_RMSE_km_s": float(
            np.sqrt(np.sum(np.square(residual / uncertainty[valid])) / np.sum(1.0 / uncertainty[valid] ** 2))
        ),
        "mean_bias_km_s": float(np.mean(residual)),
    }


def build_maps(config: dict, raw: Path):
    galaxy = config["galaxy"]
    grid = config["grid"]
    cells = int(grid["cells_per_axis"])
    spacing = float(grid["spacing_kpc"])
    axis = (np.arange(cells, dtype=float) - (cells - 1.0) / 2.0) * spacing

    hi_path = raw / "DDO154_NA_X0_P_R.FITS"
    hi_header = fits.getheader(hi_path)
    hi_moment0 = np.squeeze(fits.getdata(hi_path)).astype(float)
    gas_sky_faceon_pc2 = hi_moment0_surface_density(
        hi_moment0,
        hi_header,
        inclination_deg=float(galaxy["inclination_deg"]),
        helium_factor=float(galaxy["helium_factor"]),
    )
    gas_geometry = weighted_disk_geometry(
        gas_sky_faceon_pc2,
        inclination_deg=float(galaxy["inclination_deg"]),
    )
    gas_disk = deproject_to_disk_grid(
        gas_sky_faceon_pc2,
        gas_geometry,
        sky_pixel_scale_arcsec=abs(float(hi_header["CDELT1"])) * 3600.0,
        distance_mpc=float(galaxy["distance_mpc"]),
        disk_axis_kpc=axis,
        surface_is_face_on=True,
    ) * 1e6

    optical = config["optical_shape"]
    optical_path = raw / optical["source"]
    optical_header = fits.getheader(optical_path)
    optical_counts = fits.getdata(optical_path).astype(float)
    detector_hint = optical_detector_center(
        optical_header, float(optical["detector_center_pixel"])
    )
    optical_shape, optical_diagnostics = optical_morphology_map(
        optical_counts,
        center_hint=detector_hint,
        maximum_radius_pixel=float(optical["maximum_radius_pixel"]),
        cap_quantile=float(optical["foreground_cap_quantile"]),
        smoothing_sigma_pixel=float(optical["smoothing_sigma_pixel"]),
    )
    stellar_geometry = weighted_disk_geometry(
        optical_shape,
        inclination_deg=float(galaxy["inclination_deg"]),
        center_hint=detector_hint,
        maximum_radius_pixel=float(optical["maximum_radius_pixel"]),
        quantile_floor=0.75,
    )
    stellar_shape = deproject_to_disk_grid(
        optical_shape,
        stellar_geometry,
        sky_pixel_scale_arcsec=float(optical["pixel_scale_arcsec"]),
        distance_mpc=float(galaxy["distance_mpc"]),
        disk_axis_kpc=axis,
        surface_is_face_on=False,
    )
    stellar_disk = normalize_surface_density_mass(
        stellar_shape,
        pixel_size_kpc=spacing,
        total_mass_solar=float(galaxy["stellar_mass_solar_commissioning_only"]),
    )
    return {
        "axis_kpc": axis,
        "gas_surface_density_solar_kpc2": gas_disk,
        "stellar_surface_density_solar_kpc2": stellar_disk,
        "hi_moment0": hi_moment0,
        "gas_geometry": gas_geometry,
        "stellar_geometry": stellar_geometry,
        "optical_diagnostics": optical_diagnostics,
        "raw_hi_mass_solar": integrated_hi_mass_solar(
            hi_moment0, hi_header, distance_mpc=float(galaxy["distance_mpc"])
        ),
    }


def plot_maps(maps: dict, output: Path) -> None:
    axis = maps["axis_kpc"]
    gas = maps["gas_surface_density_solar_kpc2"] / 1e6
    stars = maps["stellar_surface_density_solar_kpc2"] / 1e6
    total = gas + stars
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True, sharey=True)
    for axis_plot, image, title in zip(
        axes,
        (gas, stars, total),
        ("H I + helium", "stellar commissioning shape", "total baryons"),
        strict=True,
    ):
        positive = image[image > 0.0]
        lower = max(float(np.quantile(positive, 0.05)), 1e-4)
        shown = axis_plot.imshow(
            image.T,
            origin="lower",
            extent=(axis[0], axis[-1], axis[0], axis[-1]),
            norm=LogNorm(vmin=lower, vmax=float(np.max(image))),
            cmap="magma",
        )
        axis_plot.set_title(title)
        axis_plot.set_xlabel("disk x (kpc)")
        figure.colorbar(shown, ax=axis_plot, label="solar masses / pc^2", shrink=0.83)
    axes[0].set_ylabel("disk y (kpc)")
    figure.suptitle("DDO154 real baryonic maps; no LITTLE THINGS velocity product used")
    figure.tight_layout()
    figure.savefig(output / "baryonic_maps.png", dpi=180)
    plt.close(figure)


def plot_curves(curves: pd.DataFrame, sparc_curve, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 5.2))
    axis.errorbar(
        sparc_curve.radius_kpc,
        sparc_curve.velocity_observed_kms,
        yerr=sparc_curve.velocity_error_kms,
        fmt="o",
        color="black",
        label="spent SPARC DDO154 observations",
    )
    sparc_baryon = np.sqrt(
        np.square(sparc_curve.velocity_gas_kms)
        + 0.5 * np.square(sparc_curve.velocity_disk_unit_ml_kms)
    )
    axis.plot(sparc_curve.radius_kpc, sparc_baryon, "--", label="SPARC radial baryons")
    styles = {
        "newtonian_3d_map": ("#4477AA", "Newtonian 3D map"),
        "algebraic_simple_mond": ("#999933", "algebraic simple MOND"),
        "QUMOND_3d_map": ("#228833", "QUMOND 3D map"),
        "AQUAL_3d_map": ("#CC6677", "AQUAL 3D map"),
    }
    for law, (color, label) in styles.items():
        subset = curves.loc[curves["law"].eq(law)]
        axis.plot(subset["radius_kpc"], subset["circular_speed_km_s"], color=color, label=label)
    axis.set_xlabel("radius (kpc)")
    axis.set_ylabel("circular speed (km/s)")
    axis.set_xlim(0.0, 8.0)
    axis.set_ylim(bottom=0.0)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    axis.set_title("DDO154 commissioning: real 2D baryons through field equations")
    figure.tight_layout()
    figure.savefig(output / "rotation_curve_comparison.png", dpi=180)
    plt.close(figure)


def run(config: dict, raw: Path, output: Path) -> dict:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    maps = build_maps(config, raw)
    axis = maps["axis_kpc"]
    spacing = float(config["grid"]["spacing_kpc"])
    gas = maps["gas_surface_density_solar_kpc2"]
    stars = maps["stellar_surface_density_solar_kpc2"]
    density = surface_density_to_volume(
        gas,
        axis,
        scale_height=float(config["grid"]["gas_scale_height_kpc"]),
    ) + surface_density_to_volume(
        stars,
        axis,
        scale_height=float(config["grid"]["stellar_scale_height_kpc"]),
    )
    constants = config["field_laws"]
    gravity = float(constants["gravitational_constant_kpc_km2_s2_per_solar_mass"])
    a0 = float(constants["a0_km2_s2_per_kpc"])
    newtonian = solve_newtonian(density, spacing, gravitational_constant=gravity)
    qumond = solve_qumond(density, spacing, a0=a0, gravitational_constant=gravity)
    aqual = solve_aqual(
        density,
        spacing,
        a0=a0,
        gravitational_constant=gravity,
        residual_tolerance=1e-5,
        maximum_nonlinear_iterations=100,
        damping=0.5,
    )
    curve_frames = []
    for law, solution in (
        ("newtonian_3d_map", newtonian),
        ("QUMOND_3d_map", qumond),
        ("AQUAL_3d_map", aqual),
    ):
        frame = radial_circular_speed(solution, axis)
        frame.insert(0, "law", law)
        curve_frames.append(frame)
    algebraic = curve_frames[0].copy()
    algebraic["law"] = "algebraic_simple_mond"
    g_newton = algebraic["inward_acceleration_km2_s2_kpc"].to_numpy()
    g_mond = simple_mond_acceleration(g_newton, a0)
    algebraic["inward_acceleration_km2_s2_kpc"] = g_mond
    algebraic["circular_speed_km_s"] = np.sqrt(
        algebraic["radius_kpc"].to_numpy() * g_mond
    )
    curve_frames.append(algebraic)
    curves = pd.concat(curve_frames, ignore_index=True)

    sparc_curve = next(curve for curve in load_curves(SPARC) if curve.metadata.name == "DDO154")
    scores = {}
    for law, frame in curves.groupby("law", sort=False):
        scores[law] = score_curve(
            frame["radius_kpc"].to_numpy(),
            frame["circular_speed_km_s"].to_numpy(),
            sparc_curve.radius_kpc,
            sparc_curve.velocity_observed_kms,
            sparc_curve.velocity_error_kms,
        )

    gas_mass_grid = float(np.sum(gas) * spacing**2)
    stellar_mass_grid = float(np.sum(stars) * spacing**2)
    raw_hi_mass = float(maps["raw_hi_mass_solar"])
    expected_gas_mass = raw_hi_mass * float(config["galaxy"]["helium_factor"])
    fields = {
        "newtonian": newtonian,
        "QUMOND": qumond,
        "AQUAL": aqual,
    }
    report = {
        "status": "commissioned" if aqual.converged else "solver_failure",
        "galaxy": config["galaxy"]["id"],
        "data_boundary": config["data_boundary"],
        "mass_inventory_solar": {
            "raw_HI": raw_hi_mass,
            "raw_HI_plus_helium": expected_gas_mass,
            "gridded_HI_plus_helium": gas_mass_grid,
            "gas_grid_fraction_of_raw": gas_mass_grid / expected_gas_mass,
            "gridded_stars": stellar_mass_grid,
            "gridded_total_baryons": gas_mass_grid + stellar_mass_grid,
            "gas_fraction": gas_mass_grid / (gas_mass_grid + stellar_mass_grid),
        },
        "geometry": {
            "gas_center_pixel": [
                maps["gas_geometry"].center_x_pixel,
                maps["gas_geometry"].center_y_pixel,
            ],
            "gas_position_angle_pixel_deg": maps["gas_geometry"].position_angle_pixel_deg,
            "stellar_center_pixel": [
                maps["stellar_geometry"].center_x_pixel,
                maps["stellar_geometry"].center_y_pixel,
            ],
            "stellar_position_angle_pixel_deg": maps[
                "stellar_geometry"
            ].position_angle_pixel_deg,
            "optical_preprocessing": maps["optical_diagnostics"],
        },
        "field_solvers": {
            law: {
                "converged": solution.converged,
                "normalized_residual_RMS": solution.normalized_residual_rms,
                "nonlinear_iterations": solution.nonlinear_iterations,
            }
            for law, solution in fields.items()
        },
        "spent_DDO154_rotation_scores": scores,
        "three_dimensional_effects": {
            "QUMOND_RMSE_change_vs_algebraic_km_s": scores["QUMOND_3d_map"]["RMSE_km_s"]
            - scores["algebraic_simple_mond"]["RMSE_km_s"],
            "AQUAL_RMSE_change_vs_algebraic_km_s": scores["AQUAL_3d_map"]["RMSE_km_s"]
            - scores["algebraic_simple_mond"]["RMSE_km_s"],
        },
        "runtime_seconds": float(time.perf_counter() - started),
        "claim_boundary": config["claim_boundary"],
    }
    curves.to_csv(output / "field_rotation_curves.csv", index=False)
    np.savez_compressed(
        output / "baryonic_maps.npz",
        axis_kpc=axis,
        gas_surface_density_solar_kpc2=gas,
        stellar_surface_density_solar_kpc2=stars,
        total_surface_density_solar_kpc2=gas + stars,
    )
    plot_maps(maps, output)
    plot_curves(curves, sparc_curve, output)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "commissioning_on_project_spent_galaxy":
        raise RuntimeError("P0635 commissioning config is not frozen")
    raw = args.raw.resolve()
    provenance = []
    for product in config["raw_products"]:
        path = raw / product["filename"]
        if path.stat().st_size != product["bytes"] or sha256(path) != product["sha256"]:
            raise RuntimeError(f"raw product failed provenance check: {path}")
        provenance.append({**product, "verified": True})
    report = run(config, raw, args.output.resolve())
    report["protocol_version"] = config["protocol_version"]
    report["config_sha256"] = sha256(config_path)
    report["raw_provenance"] = provenance
    (args.output.resolve() / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "mass_inventory_solar": report["mass_inventory_solar"],
                "field_solvers": report["field_solvers"],
                "spent_DDO154_rotation_scores": report["spent_DDO154_rotation_scores"],
            },
            indent=2,
        )
    )
    if report["status"] != "commissioned":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
