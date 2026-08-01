#!/usr/bin/env python3
"""Build physical HST+Chandra baryon maps without opening lensing constraints."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.cluster_baryon_maps import (
    chandra_rate_map,
    f160_stellar_mass_msun,
    gas_surface_density,
    member_light_center,
    physical_grid,
    stellar_surface_density,
    strict_f160_members,
    surface_moments,
)
from voidscreen.gravity_arc_tomography import read_relics_catalog

DEFAULT_CONFIG = ROOT / "configs" / "p0641_registered_cluster_baryon_maps.json"
DEFAULT_OUTPUT = ROOT / "results" / "p0641_registered_cluster_baryon_maps"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def one(directory: Path, pattern: str) -> Path:
    paths = list(directory.glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(f"expected one {pattern} in {directory}, found {len(paths)}")
    return paths[0]


def angular_difference_180(first: float, second: float) -> float:
    return float(abs((first - second + 90.0) % 180.0 - 90.0))


def normalized_overlap(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> float:
    left = np.asarray(first[mask], dtype=float)
    right = np.asarray(second[mask], dtype=float)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0.0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["status"] != "frozen_baryonic_map_rules_before_lensing_unseal":
        raise RuntimeError("P0641 map protocol is not frozen")
    if any(config["blind_state"].values()):
        raise RuntimeError("P0641 blind-state declaration is not fully false")
    parent_path = ROOT / config["parent_acquisition"]
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    targets = {item["id"]: item for item in parent["targets"]}
    gas_rules = {item["id"]: item for item in config["gas_normalizations"]}
    if set(targets) != set(gas_rules):
        raise RuntimeError("cluster and gas-normalization inventories differ")

    output = args.output.resolve()
    maps_output = output / "maps"
    maps_output.mkdir(parents=True, exist_ok=True)
    raw = ROOT / config["raw_directory"]
    grid_rule = config["map_rules"]
    population = config["stellar_population"]
    rows = []
    xray_rows = []
    figure, axes = plt.subplots(
        len(targets),
        3,
        figsize=(12, 4 * len(targets)),
        constrained_layout=True,
        squeeze=False,
    )
    for row_index, system in enumerate(targets.values()):
        system_id = system["id"]
        directory = raw / system_id
        hst = directory / "hst"
        catalog_path = one(hst, "*_cat.txt")
        image_path = one(hst, "*_f160w_v1_drz.fits")
        segmentation_path = one(hst, "*_segm.fits")
        catalog = read_relics_catalog(catalog_path)
        selected, flux = strict_f160_members(catalog, float(system["redshift"]))
        nominal_member_mass = f160_stellar_mass_msun(
            flux[selected],
            redshift=float(system["redshift"]),
            mass_to_light_solar=float(population["nominal_mass_to_light_solar"]),
            solar_absolute_ab_magnitude=float(population["solar_absolute_ab_magnitude"]),
        )
        center = member_light_center(catalog, selected, nominal_member_mass)
        grid = physical_grid(
            center,
            half_extent_kpc=float(grid_rule["grid_half_extent_kpc"]),
            size=int(grid_rule["grid_size"]),
            redshift=float(system["redshift"]),
        )
        stellar, stellar_report = stellar_surface_density(
            catalog_path,
            image_path,
            segmentation_path,
            grid=grid,
            redshift=float(system["redshift"]),
            mass_to_light_solar=float(population["nominal_mass_to_light_solar"]),
            solar_absolute_ab_magnitude=float(population["solar_absolute_ab_magnitude"]),
        )
        low_ml, _nominal_ml, high_ml = population["mass_to_light_sensitivity_solar"]
        stellar_low = stellar * float(low_ml) / float(population["nominal_mass_to_light_solar"])
        stellar_high = stellar * float(high_ml) / float(population["nominal_mass_to_light_solar"])

        rate, exposure, observation_rows = chandra_rate_map(
            sorted((directory / "chandra").glob("*.fits.gz")), grid
        )
        gas_rule = gas_rules[system_id]
        gas, gas_report = gas_surface_density(
            rate,
            exposure,
            grid=grid,
            aperture_kpc=float(gas_rule["aperture_kpc"]),
            gas_mass_msun=float(gas_rule["gas_mass_msun"]),
            morphology_exponent=0.5,
            smoothing_kpc=float(grid_rule["xray_smoothing_kpc"]),
            winsor_quantile=float(grid_rule["xray_point_source_winsor_quantile"]),
            background_mad_sigma=float(grid_rule["xray_background_mad_sigma"]),
        )
        gas_shape_low, _ = gas_surface_density(
            rate,
            exposure,
            grid=grid,
            aperture_kpc=float(gas_rule["aperture_kpc"]),
            gas_mass_msun=float(gas_rule["gas_mass_msun"]),
            morphology_exponent=0.4,
            smoothing_kpc=float(grid_rule["xray_smoothing_kpc"]),
            winsor_quantile=float(grid_rule["xray_point_source_winsor_quantile"]),
            background_mad_sigma=float(grid_rule["xray_background_mad_sigma"]),
        )
        gas_shape_high, _ = gas_surface_density(
            rate,
            exposure,
            grid=grid,
            aperture_kpc=float(gas_rule["aperture_kpc"]),
            gas_mass_msun=float(gas_rule["gas_mass_msun"]),
            morphology_exponent=0.6,
            smoothing_kpc=float(grid_rule["xray_smoothing_kpc"]),
            winsor_quantile=float(grid_rule["xray_point_source_winsor_quantile"]),
            background_mad_sigma=float(grid_rule["xray_background_mad_sigma"]),
        )
        gas_low = gas * (
            float(gas_rule["gas_mass_msun"]) - float(gas_rule["gas_mass_sigma_msun"])
        ) / float(gas_rule["gas_mass_msun"])
        gas_high = gas * (
            float(gas_rule["gas_mass_msun"]) + float(gas_rule["gas_mass_sigma_msun"])
        ) / float(gas_rule["gas_mass_msun"])
        baryon = stellar + gas
        baryon_low = stellar_low + gas_low
        baryon_high = stellar_high + gas_high
        stellar_moments = surface_moments(stellar, grid)
        gas_moments = surface_moments(gas, grid)
        baryon_moments = surface_moments(baryon, grid)
        centroid_offset = float(
            np.hypot(
                stellar_moments["centroid_x_kpc"] - gas_moments["centroid_x_kpc"],
                stellar_moments["centroid_y_kpc"] - gas_moments["centroid_y_kpc"],
            )
        )
        central_mask = np.hypot(grid.x_kpc, grid.y_kpc) <= min(
            500.0, float(gas_rule["aperture_kpc"])
        )
        overlap = normalized_overlap(stellar, gas, central_mask)
        map_path = maps_output / f"{system_id}_baryons.npz"
        np.savez_compressed(
            map_path,
            axis_kpc=grid.axis_kpc.astype(np.float32),
            stellar_surface_density_msun_kpc2=stellar.astype(np.float32),
            stellar_surface_density_low_msun_kpc2=stellar_low.astype(np.float32),
            stellar_surface_density_high_msun_kpc2=stellar_high.astype(np.float32),
            gas_surface_density_msun_kpc2=gas.astype(np.float32),
            gas_surface_density_low_msun_kpc2=gas_low.astype(np.float32),
            gas_surface_density_high_msun_kpc2=gas_high.astype(np.float32),
            gas_shape_exponent_0p4_msun_kpc2=gas_shape_low.astype(np.float32),
            gas_shape_exponent_0p6_msun_kpc2=gas_shape_high.astype(np.float32),
            baryon_surface_density_msun_kpc2=baryon.astype(np.float32),
            baryon_surface_density_low_msun_kpc2=baryon_low.astype(np.float32),
            baryon_surface_density_high_msun_kpc2=baryon_high.astype(np.float32),
            chandra_rate=rate.astype(np.float32),
            chandra_exposure_s=exposure.astype(np.float32),
            center_ra_deg=np.float64(center.ra.deg),
            center_dec_deg=np.float64(center.dec.deg),
            redshift=np.float64(system["redshift"]),
        )
        row = {
            "system": system_id,
            "redshift": float(system["redshift"]),
            "center_ra_deg": float(center.ra.deg),
            "center_dec_deg": float(center.dec.deg),
            "grid_size": int(grid_rule["grid_size"]),
            "cell_kpc": grid.cell_kpc,
            "gas_aperture_kpc": float(gas_rule["aperture_kpc"]),
            "gas_mass_sigma_msun": float(gas_rule["gas_mass_sigma_msun"]),
            **stellar_report,
            **gas_report,
            "baryon_mass_msun": baryon_moments["mass_msun"],
            "gas_to_stellar_mass_ratio": gas_moments["mass_msun"]
            / stellar_moments["mass_msun"],
            "gas_stellar_centroid_offset_kpc": centroid_offset,
            "gas_stellar_axis_angle_difference_deg": angular_difference_180(
                gas_moments["position_angle_deg"], stellar_moments["position_angle_deg"]
            ),
            "central_gas_stellar_cosine_overlap": overlap,
            **{f"stellar_{key}": value for key, value in stellar_moments.items()},
            **{f"gas_{key}": value for key, value in gas_moments.items()},
            **{f"baryon_{key}": value for key, value in baryon_moments.items()},
            "map_path": map_path.relative_to(ROOT).as_posix(),
            "map_sha256": sha256(map_path),
        }
        rows.append(row)
        for observation in observation_rows:
            xray_rows.append({"system": system_id, **observation})

        extent = [
            -float(grid_rule["grid_half_extent_kpc"]),
            float(grid_rule["grid_half_extent_kpc"]),
            -float(grid_rule["grid_half_extent_kpc"]),
            float(grid_rule["grid_half_extent_kpc"]),
        ]
        for axis, data, title in zip(
            axes[row_index],
            (stellar, gas, baryon),
            ("member stars", "X-ray gas", "all baryons"),
            strict=True,
        ):
            positive = data[data > 0.0]
            floor = float(np.quantile(positive, 0.05)) if positive.size else 1.0
            image = axis.imshow(
                np.log10(data + floor),
                origin="lower",
                extent=extent,
                cmap="magma",
                interpolation="nearest",
            )
            axis.set(
                title=f"{system_id}: {title}",
                xlim=(-700.0, 700.0),
                ylim=(-700.0, 700.0),
                aspect="equal",
                xlabel="east offset (kpc)",
                ylabel="north offset (kpc)",
            )
            figure.colorbar(image, ax=axis, shrink=0.75, label="log10 surface mass")
        print(f"{system_id}: map complete", flush=True)

    systems = pd.DataFrame(rows)
    observations = pd.DataFrame(xray_rows)
    systems.to_csv(output / "systems.csv", index=False)
    observations.to_csv(output / "chandra_reprojection.csv", index=False)
    figure.savefig(output / "baryon_map_atlas.png", dpi=160)
    plt.close(figure)
    gates = {
        "four_cluster_maps": len(systems) == 4,
        "stellar_mass_recovery_relative_error_max_1e_10": bool(
            (np.abs(systems["stellar_mass_recovery_fraction"] - 1.0) <= 1.0e-10).all()
        ),
        "gas_mass_recovery_relative_error_max_1e_10": bool(
            (np.abs(systems["gas_mass_recovery_fraction"] - 1.0) <= 1.0e-10).all()
        ),
        "minimum_90_members": bool((systems["selected_members"] >= 90).all()),
        "all_chandra_observations_reprojected": len(observations) == 19,
        "all_maps_finite": bool(
            systems[
                [
                    "baryon_mass_msun",
                    "gas_stellar_centroid_offset_kpc",
                    "gas_stellar_axis_angle_difference_deg",
                    "central_gas_stellar_cosine_overlap",
                ]
            ]
            .apply(np.isfinite)
            .all()
            .all()
        ),
        "blind_state_untouched": not any(config["blind_state"].values()),
        "zero_per_cluster_gravity_parameters": not config["blind_state"][
            "per_cluster_gravity_parameter"
        ],
    }
    status = "ready" if all(gates.values()) else "map_failure"
    report = {
        "report_version": config["protocol_version"],
        "status": status,
        "config_sha256": sha256(config_path),
        "parent_acquisition_sha256": sha256(parent_path),
        "gates": gates,
        "blind_state": config["blind_state"],
        "systems": rows,
        "ranges": {
            "stellar_mass_msun": [
                float(systems["stellar_mass_msun"].min()),
                float(systems["stellar_mass_msun"].max()),
            ],
            "gas_stellar_centroid_offset_kpc": [
                float(systems["gas_stellar_centroid_offset_kpc"].min()),
                float(systems["gas_stellar_centroid_offset_kpc"].max()),
            ],
            "central_gas_stellar_cosine_overlap": [
                float(systems["central_gas_stellar_cosine_overlap"].min()),
                float(systems["central_gas_stellar_cosine_overlap"].max()),
            ],
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = f"""# P0641 registered cluster baryon maps

- Status: **{status.upper()}**
- Physical maps: {len(systems)}
- Chandra observations reprojected: {len(observations)}
- Selected F160W members per cluster: {int(systems.selected_members.min())}-{int(systems.selected_members.max())}
- Stellar mass range: {systems.stellar_mass_msun.min():.3e}-{systems.stellar_mass_msun.max():.3e} Msun
- Gas/stellar centroid offsets: {systems.gas_stellar_centroid_offset_kpc.min():.1f}-{systems.gas_stellar_centroid_offset_kpc.max():.1f} kpc
- Central gas/star morphology overlap: {systems.central_gas_stellar_cosine_overlap.min():.3f}-{systems.central_gas_stellar_cosine_overlap.max():.3f}
- Lensing constraints opened: `false`
- Per-cluster gravity parameters: `zero`

Every map is built from registered HST member pixels and Chandra morphology,
with externally measured gas normalization. The maps and their low/high
baryonic uncertainty variants are ready for equation-blind prediction.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(json.dumps({"status": status, "gates": gates, "ranges": report["ranges"]}, indent=2))
    if status != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
