#!/usr/bin/env python3
"""Construct registered physical baryonic maps for the sealed galaxy sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import astropy.units as u
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.galaxy_maps import (
    aips_clean_beam_degrees,
    hi_moment0_surface_density,
    integrated_hi_mass_solar,
    normalize_surface_density_mass,
    optical_morphology_map,
    reproject_wcs_to_disk_grid,
    resolved_map_morphology,
    sky_pixels_to_disk_coordinates,
    weighted_radius_quantile,
)

DEFAULT_CONFIG = ROOT / "configs" / "p0639_registered_baryonic_maps.json"
DEFAULT_OUTPUT = ROOT / "results" / "p0639_registered_baryonic_maps"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_fitted_wcs(path: Path) -> WCS:
    payload = json.loads(path.read_text(encoding="utf-8"))
    header = fits.Header()
    for keyword, value in payload["wcs_header"].items():
        header[keyword] = value
    return WCS(header).celestial


def outer_cell_fraction(image: np.ndarray) -> float:
    edge = np.zeros(image.shape, dtype=bool)
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    return float(np.sum(image[edge]) / np.sum(image))


def map_for_target(
    target: dict, metadata: pd.Series, config: dict, output: Path
) -> tuple[dict, dict[str, np.ndarray]]:
    galaxy = target["id"]
    maps_config = json.loads((ROOT / config["parent_maps"]).read_text(encoding="utf-8"))
    raw = ROOT / maps_config["raw_directory"] / galaxy
    hi_path = raw / target["hi_filename"]
    v_path = raw / f"{target['optical_prefix']}v.fits"
    hi_header = fits.getheader(hi_path)
    hi_wcs = WCS(hi_header).celestial
    hi = np.squeeze(fits.getdata(hi_path)).astype(float)
    optical = np.squeeze(fits.getdata(v_path)).astype(float)
    optical_wcs = load_fitted_wcs(
        ROOT / "results" / "p0638_gaia_astrometric_registration" / "wcs" / f"{galaxy}.json"
    )
    center = SkyCoord(
        str(metadata["photometric_center_ra_j2000"]),
        str(metadata["photometric_center_dec_j2000"]),
        unit=(u.hourangle, u.deg),
    )
    distance = float(metadata["distance_mpc"])
    inclination = float(metadata["derived_photometric_inclination_deg"])
    position_angle = float(metadata["photometric_pa_deg"])
    positive_y, positive_x = np.nonzero(hi > 0.0)
    hi_major, hi_minor = sky_pixels_to_disk_coordinates(
        positive_x,
        positive_y,
        hi_wcs,
        center=center,
        position_angle_deg=position_angle,
        inclination_deg=inclination,
        distance_mpc=distance,
    )
    hi_radius = weighted_radius_quantile(
        hi_major,
        hi_minor,
        hi[positive_y, positive_x],
        float(config["grid"]["hi_weighted_radius_quantile"]),
    )
    optical_radius = (
        float(metadata["integrated_aperture_radius_arcmin"])
        * 60.0
        / 206264.80624709636
        * distance
        * 1000.0
    )
    half_extent = max(
        float(config["grid"]["minimum_half_extent_kpc"]),
        float(config["grid"]["half_extent_padding_factor"])
        * max(hi_radius, optical_radius),
    )
    beam = aips_clean_beam_degrees(hi_header)
    beam_arcsec = np.sqrt(beam[0] * beam[1]) * 3600.0
    beam_kpc = beam_arcsec / 206264.80624709636 * distance * 1000.0
    requested_cells = int(
        np.ceil(
            2.0
            * half_extent
            * float(config["grid"]["target_cells_per_radio_beam"])
            / beam_kpc
        )
        + 1
    )
    cells = int(
        np.clip(
            requested_cells,
            int(config["grid"]["minimum_cells_per_axis"]),
            int(config["grid"]["maximum_cells_per_axis"]),
        )
    )
    if cells % 2 == 0:
        cells = min(cells + 1, int(config["grid"]["maximum_cells_per_axis"]))
    axis = np.linspace(-half_extent, half_extent, cells)
    spacing = float(axis[1] - axis[0])
    gas_faceon_pc2 = hi_moment0_surface_density(
        hi,
        hi_header,
        inclination_deg=inclination,
        helium_factor=float(config["gas"]["helium_factor"]),
    )
    gas_unscaled = (
        reproject_wcs_to_disk_grid(
            gas_faceon_pc2,
            hi_wcs,
            center=center,
            position_angle_deg=position_angle,
            inclination_deg=inclination,
            distance_mpc=distance,
            disk_axis_kpc=axis,
            interpolation_order=int(config["grid"]["interpolation_order"]),
        )
        * 1e6
    )
    expected_gas_mass = integrated_hi_mass_solar(
        hi, hi_header, distance_mpc=distance
    ) * float(config["gas"]["helium_factor"])
    unscaled_gas_mass = float(np.sum(gas_unscaled) * spacing**2)
    gas = normalize_surface_density_mass(
        gas_unscaled, pixel_size_kpc=spacing, total_mass_solar=expected_gas_mass
    )
    center_x, center_y = optical_wcs.world_to_pixel(center)
    optical_shape, optical_diagnostics = optical_morphology_map(
        optical,
        center_hint=(center_x, center_y),
        maximum_radius_pixel=(
            float(metadata["integrated_aperture_radius_arcmin"])
            * 60.0
            / float(metadata["optical_pixel_scale_arcsec"])
        ),
        border_fraction=float(config["stars"]["sky_border_fraction"]),
        cap_quantile=float(config["stars"]["foreground_cap_quantile"]),
        cap_sigma_above_sky=float(config["stars"]["foreground_cap_sigma_above_sky"]),
        smoothing_sigma_pixel=(
            float(config["stars"]["smoothing_sigma_arcsec"])
            / float(metadata["optical_pixel_scale_arcsec"])
        ),
    )
    stellar_shape = reproject_wcs_to_disk_grid(
        optical_shape,
        optical_wcs,
        center=center,
        position_angle_deg=position_angle,
        inclination_deg=inclination,
        distance_mpc=distance,
        disk_axis_kpc=axis,
        interpolation_order=int(config["grid"]["interpolation_order"]),
    )
    stellar_mass = float(metadata["nominal_stellar_mass_solar"])
    stars = normalize_surface_density_mass(
        stellar_shape, pixel_size_kpc=spacing, total_mass_solar=stellar_mass
    )
    total = gas + stars
    xx, yy = np.meshgrid(axis, axis, indexing="ij")

    def centroid(image: np.ndarray) -> tuple[float, float]:
        mass = float(np.sum(image))
        return float(np.sum(image * xx) / mass), float(np.sum(image * yy) / mass)

    gas_center = centroid(gas)
    star_center = centroid(stars)
    cells_per_beam = beam_kpc / spacing
    morphology = resolved_map_morphology(
        gas,
        disk_axis_kpc=axis,
        smoothing_sigma_pixel=max(
            beam_kpc / spacing * float(config["morphology"]["clumpiness_smoothing_beam_sigma"]),
            0.5,
        ),
    )
    maps = {"axis_kpc": axis, "gas": gas, "stars": stars, "total": total}
    np.savez_compressed(output / "maps" / f"{galaxy}.npz", **maps)
    row = {
        "galaxy": galaxy,
        "distance_mpc": distance,
        "inclination_deg": inclination,
        "position_angle_deg": position_angle,
        "cells_per_axis": cells,
        "half_extent_kpc": half_extent,
        "spacing_kpc": spacing,
        "hi_r995_kpc": hi_radius,
        "optical_aperture_radius_kpc": optical_radius,
        "beam_geometric_mean_arcsec": beam_arcsec,
        "beam_geometric_mean_kpc": beam_kpc,
        "cells_per_radio_beam": cells_per_beam,
        "expected_gas_mass_solar": expected_gas_mass,
        "unscaled_registered_gas_mass_solar": unscaled_gas_mass,
        "pre_normalization_gas_mass_fraction": unscaled_gas_mass / expected_gas_mass,
        "gas_mass_solar": float(np.sum(gas) * spacing**2),
        "stellar_mass_solar": float(np.sum(stars) * spacing**2),
        "total_baryonic_mass_solar": float(np.sum(total) * spacing**2),
        "gas_fraction": float(np.sum(gas) / np.sum(total)),
        "outer_cell_gas_mass_fraction": outer_cell_fraction(gas),
        "outer_cell_stellar_mass_fraction": outer_cell_fraction(stars),
        "gas_centroid_x_kpc": gas_center[0],
        "gas_centroid_y_kpc": gas_center[1],
        "stellar_centroid_x_kpc": star_center[0],
        "stellar_centroid_y_kpc": star_center[1],
        "gas_star_centroid_offset_kpc": float(
            np.hypot(gas_center[0] - star_center[0], gas_center[1] - star_center[1])
        ),
        **morphology,
        **{f"optical_{key}": value for key, value in optical_diagnostics.items()},
        "hi_sha256": sha256(hi_path),
        "v_sha256": sha256(v_path),
    }
    return row, maps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    maps_config = json.loads((ROOT / config["parent_maps"]).read_text(encoding="utf-8"))
    metadata = pd.read_csv(
        ROOT / "results" / "p0637_little_things_photometric_metadata" / "photometric_inputs.csv"
    ).set_index("galaxy")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "maps").mkdir(exist_ok=True)
    rows = []
    map_sets = []
    for target in maps_config["targets"]:
        row, maps = map_for_target(target, metadata.loc[target["id"]], config, output)
        rows.append(row)
        map_sets.append((target["id"], maps))
        print(
            f"{target['id']}: gas={row['gas_mass_solar']:.3e}, "
            f"stars={row['stellar_mass_solar']:.3e}, "
            f"offset={row['gas_star_centroid_offset_kpc']:.3f} kpc"
        )
    frame = pd.DataFrame(rows)
    acceptance = config["acceptance"]
    mass_error = np.maximum(
        np.abs(frame["gas_mass_solar"] / frame["expected_gas_mass_solar"] - 1.0),
        np.abs(
            frame["stellar_mass_solar"]
            / metadata.loc[frame["galaxy"], "nominal_stellar_mass_solar"].to_numpy()
            - 1.0
        ),
    )
    frame["all_gates_pass"] = (
        frame["pre_normalization_gas_mass_fraction"].between(
            float(acceptance["minimum_pre_normalization_gas_mass_fraction"]),
            float(acceptance["maximum_pre_normalization_gas_mass_fraction"]),
        )
        & (
            frame["outer_cell_gas_mass_fraction"]
            <= float(acceptance["maximum_outer_cell_gas_mass_fraction"])
        )
        & (
            frame["outer_cell_stellar_mass_fraction"]
            <= float(acceptance["maximum_outer_cell_stellar_mass_fraction"])
        )
        & (
            frame["cells_per_radio_beam"]
            >= float(acceptance["minimum_cells_per_radio_beam"])
        )
        & (mass_error <= float(acceptance["maximum_mass_normalization_relative_error"]))
    )
    frame.to_csv(output / "map_audit.csv", index=False)
    figure, axes = plt.subplots(13, 3, figsize=(9.5, 35), squeeze=False)
    for row_index, (galaxy, maps) in enumerate(map_sets):
        axis = maps["axis_kpc"]
        for column, (key, title) in enumerate(
            [("gas", "H I + He"), ("stars", "stars"), ("total", "total baryons")]
        ):
            image = maps[key] / 1e6
            positive = image[image > 0.0]
            lower = max(float(np.quantile(positive, 0.02)), 1e-5)
            shown = axes[row_index, column].imshow(
                image.T,
                origin="lower",
                extent=(axis[0], axis[-1], axis[0], axis[-1]),
                cmap="magma",
                norm=LogNorm(vmin=lower, vmax=float(np.max(image))),
            )
            axes[row_index, column].set_title(f"{galaxy}: {title}")
            axes[row_index, column].set_aspect("equal")
            figure.colorbar(shown, ax=axes[row_index, column], shrink=0.72)
        axes[row_index, 0].set_ylabel("disk y (kpc)")
    for axis_plot in axes[-1, :]:
        axis_plot.set_xlabel("disk x (kpc)")
    figure.suptitle("P0639 registered baryonic maps (solar masses / pc^2)", y=0.999)
    figure.tight_layout()
    figure.savefig(output / "registered_baryonic_map_atlas.png", dpi=150)
    plt.close(figure)
    report = {
        "status": "pass" if len(frame) == 13 and frame["all_gates_pass"].all() else "fail",
        "protocol_version": config["protocol_version"],
        "targets": len(frame),
        "all_gates_pass": bool(frame["all_gates_pass"].all()),
        "minimum_pre_normalization_gas_mass_fraction": float(
            frame["pre_normalization_gas_mass_fraction"].min()
        ),
        "maximum_pre_normalization_gas_mass_fraction": float(
            frame["pre_normalization_gas_mass_fraction"].max()
        ),
        "maximum_outer_cell_gas_mass_fraction": float(
            frame["outer_cell_gas_mass_fraction"].max()
        ),
        "maximum_outer_cell_stellar_mass_fraction": float(
            frame["outer_cell_stellar_mass_fraction"].max()
        ),
        "minimum_cells_per_radio_beam": float(frame["cells_per_radio_beam"].min()),
        "maximum_mass_normalization_relative_error": float(np.max(mass_error)),
        "gas_star_centroid_offset_kpc_range": [
            float(frame["gas_star_centroid_offset_kpc"].min()),
            float(frame["gas_star_centroid_offset_kpc"].max()),
        ],
        "config_sha256": sha256(config_path),
        "sealed_target_observables_opened": False,
        "per_object_gravity_parameters": 0,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (output / "SUMMARY.md").write_text(
        f"""# P0639 registered baryonic maps

- Status: **{report['status'].upper()}**
- Complete physical maps: {report['targets']} / 13
- Pre-normalization gas-mass fraction: {report['minimum_pre_normalization_gas_mass_fraction']:.4f} to {report['maximum_pre_normalization_gas_mass_fraction']:.4f}
- Maximum gas mass on an outer grid cell: {report['maximum_outer_cell_gas_mass_fraction']:.5f}
- Maximum stellar mass on an outer grid cell: {report['maximum_outer_cell_stellar_mass_fraction']:.5f}
- Minimum surface-map cells per radio beam: {report['minimum_cells_per_radio_beam']:.3f}
- Gas-star centroid offsets: {report['gas_star_centroid_offset_kpc_range'][0]:.3f} to {report['gas_star_centroid_offset_kpc_range'][1]:.3f} kpc
- Sealed kinematics opened: `{str(report['sealed_target_observables_opened']).lower()}`
- Per-object gravity parameters: `{report['per_object_gravity_parameters']}`

The maps retain measured two-dimensional gas and stellar morphology on one
registered face-on physical grid. They are ready for blind field predictions.
""",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
