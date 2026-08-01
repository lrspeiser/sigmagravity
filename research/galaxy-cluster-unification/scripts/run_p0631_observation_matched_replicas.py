#!/usr/bin/env python3
"""Run P0631 observation-matched galaxy replicas and virtual-telescope checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.data import load_curves  # noqa: E402
from voidscreen.galaxy_replica import (  # noqa: E402
    generate_replica_particles,
    load_replica_seed,
    render_observed_replica,
    score_replica,
    valid_rotation_mask,
)
from voidscreen.synthetic_universe import stable_hash_partition  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_any_P0631_replica_score":
        raise RuntimeError("P0631 protocol status is not frozen")
    return protocol


def retained_names(protocol: dict) -> list[str]:
    settings = protocol["sample"]
    names: list[str] = []
    for curve in load_curves(ROOT / protocol["inputs"]["sparc_directory"]):
        baryonic_v2 = (
            np.sign(curve.velocity_gas_kms) * curve.velocity_gas_kms**2
            + 0.5 * curve.velocity_disk_unit_ml_kms**2
            + 0.7 * curve.velocity_bulge_unit_ml_kms**2
        )
        valid = (
            np.isfinite(curve.radius_kpc)
            & np.isfinite(curve.velocity_observed_kms)
            & np.isfinite(curve.velocity_error_kms)
            & np.isfinite(baryonic_v2)
            & (curve.radius_kpc > 0.0)
            & (curve.velocity_observed_kms > 0.0)
            & (curve.velocity_error_kms > 0.0)
            & (baryonic_v2 > 0.0)
        )
        if curve.metadata.quality > int(settings["quality_max"]):
            continue
        if curve.metadata.inclination_deg < float(settings["minimum_inclination_deg"]):
            continue
        if int(valid.sum()) < int(settings["minimum_rotation_points"]):
            continue
        names.append(curve.metadata.name)
    if len(names) != int(settings["expected_galaxies"]):
        raise RuntimeError(f"Expected {settings['expected_galaxies']} galaxies, found {len(names)}")
    return sorted(names)


def summaries(frame: pd.DataFrame) -> dict:
    result = {}
    for split in ["all", "train", "development", "holdout"]:
        subset = frame if split == "all" else frame.loc[frame.split.eq(split)]
        result[split] = {
            "galaxies": int(len(subset)),
            "median_light_rmse_dex": float(subset.light_rmse_dex.median()),
            "p90_light_rmse_dex": float(subset.light_rmse_dex.quantile(0.9)),
            "median_angular_photometry_rmse_dex": float(
                subset.angular_photometry_rmse_dex.median()
            ),
            "median_rotation_rmse_km_s": float(subset.rotation_rmse_km_s.median()),
            "p90_rotation_rmse_km_s": float(subset.rotation_rmse_km_s.quantile(0.9)),
            "median_pixelized_light_rmse_dex": float(
                subset.pixelized_light_rmse_dex.median()
            ),
            "median_pixelized_rotation_rmse_km_s": float(
                subset.pixelized_rotation_rmse_km_s.median()
            ),
            "median_abs_total_light_fractional_error": float(
                subset.total_light_fractional_error.abs().median()
            ),
        }
    return result


def plot_representative(seed, rendered, output: Path) -> None:
    coordinates = rendered.x_kpc[0]
    extent = [coordinates[0], coordinates[-1], coordinates[0], coordinates[-1]]
    total = rendered.total_lsun_pc2
    positive = total[total > 0.0]
    light_floor = max(float(np.quantile(positive, 0.02)), float(positive.max()) * 1.0e-5)
    vmax_velocity = float(np.nanmax(np.abs(rendered.line_of_sight_velocity_km_s)))
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 8.6), constrained_layout=True)

    image = axes[0, 0].imshow(
        total,
        origin="lower",
        extent=extent,
        cmap="magma",
        norm=LogNorm(vmin=light_floor, vmax=float(positive.max())),
        interpolation="nearest",
    )
    axes[0, 0].set(title="Generated 3.6 μm light", xlabel="x [kpc]", ylabel="y [kpc]")
    figure.colorbar(image, ax=axes[0, 0], label=r"$L_\odot\,pc^{-2}$")

    velocity = axes[0, 1].imshow(
        rendered.line_of_sight_velocity_km_s,
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-vmax_velocity, vcenter=0.0, vmax=vmax_velocity),
        interpolation="nearest",
    )
    axes[0, 1].set(title="Generated line-of-sight velocity", xlabel="x [kpc]", ylabel="y [kpc]")
    figure.colorbar(velocity, ax=axes[0, 1], label="km/s")

    angular = seed.angular_photometry
    angular_radius = angular.radius_arcsec * seed.distance_mpc * 1000.0 / 206265.0
    angular_density = np.power(
        10.0,
        -0.4 * (angular.surface_brightness_mag_arcsec2 - 3.24 - 21.572),
    )
    axes[1, 0].scatter(angular_radius, angular_density, s=10, c="black", label="SPARC photometry")
    axes[1, 0].plot(
        seed.light.radius_kpc,
        seed.light.disk_lsun_pc2 + seed.light.bulge_lsun_pc2,
        color="#d95f02",
        lw=1.8,
        label="generated radial profile",
    )
    axes[1, 0].set(
        xscale="log",
        yscale="log",
        xlabel="radius [kpc]",
        ylabel=r"surface light [$L_\odot\,pc^{-2}$]",
        title="Actual photometry vs replica",
    )
    axes[1, 0].legend(fontsize=8)

    mask = valid_rotation_mask(seed)
    radius = seed.rotation.radius_kpc[mask]
    observed = seed.rotation.velocity_observed_kms[mask]
    error = seed.rotation.velocity_error_kms[mask]
    axes[1, 1].errorbar(radius, observed, yerr=error, fmt="o", ms=3, color="black", label="SPARC observed")
    axes[1, 1].plot(radius, observed, color="#1b9e77", lw=1.8, label="replica-mode input")
    axes[1, 1].set(
        xlabel="radius [kpc]",
        ylabel="circular speed [km/s]",
        title="Actual rotation vs replica",
    )
    axes[1, 1].legend(fontsize=8)
    figure.suptitle(
        f"{seed.name}: observation-matched replica (T={seed.hubble_type}, "
        f"i={seed.inclination_deg:.0f}°, B/T={seed.bulge_fraction:.2f})",
        fontsize=13,
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs" / "p0631_observation_matched_replica_protocol.json",
    )
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    output = ROOT / protocol["outputs"]["directory"]
    representative_output = output / "representatives"
    output.mkdir(parents=True, exist_ok=True)
    representative_output.mkdir(parents=True, exist_ok=True)

    names = retained_names(protocol)
    settings = protocol["sample"]
    splits = stable_hash_partition(
        names,
        salt=settings["split_salt"],
        train_fraction=float(settings["train_fraction"]),
        development_fraction=float(settings["development_fraction"]),
    )
    actual_counts = Counter(splits.values())
    expected_counts = Counter(settings["expected_split_counts"])
    if actual_counts != expected_counts:
        raise RuntimeError(f"Split ledger changed: {dict(actual_counts)}")

    inputs = protocol["inputs"]
    renderer = protocol["renderer"]
    scores: list[dict] = []
    seeds = {}
    rendered_representatives = {}
    deterministic = True
    particle_checks = []
    for index, name in enumerate(names, start=1):
        seed = load_replica_seed(
            name,
            ROOT / inputs["sparc_directory"],
            ROOT / inputs["photometric_profiles"],
            ROOT / inputs["bulge_disk_decompositions"],
        )
        seeds[name] = seed
        pixels = int(renderer["grid_pixels_all_galaxies"])
        rendered = render_observed_replica(
            seed,
            pixels=pixels,
            extent_multiplier=float(renderer["extent_multiplier"]),
            intrinsic_disk_axis_ratio=float(renderer["intrinsic_disk_axis_ratio"]),
        )
        row = {
            "galaxy": name,
            "split": splits[name],
            "hubble_type": seed.hubble_type,
            "inclination_deg": seed.inclination_deg,
            "quality": seed.quality,
            "distance_mpc": seed.distance_mpc,
            **score_replica(seed, rendered),
        }
        scores.append(row)
        if name in renderer["representative_galaxies"]:
            high_resolution = render_observed_replica(
                seed,
                pixels=int(renderer["grid_pixels_representatives"]),
                extent_multiplier=float(renderer["extent_multiplier"]),
                intrinsic_disk_axis_ratio=float(renderer["intrinsic_disk_axis_ratio"]),
            )
            rendered_representatives[name] = high_resolution
            replay = render_observed_replica(
                seed,
                pixels=int(renderer["grid_pixels_representatives"]),
                extent_multiplier=float(renderer["extent_multiplier"]),
                intrinsic_disk_axis_ratio=float(renderer["intrinsic_disk_axis_ratio"]),
            )
            deterministic &= np.array_equal(high_resolution.total_lsun_pc2, replay.total_lsun_pc2)
            deterministic &= np.array_equal(
                high_resolution.line_of_sight_velocity_km_s,
                replay.line_of_sight_velocity_km_s,
                equal_nan=True,
            )
            particles = generate_replica_particles(
                seed, particle_count=int(renderer["particle_count_representatives"])
            )
            replay_particles = generate_replica_particles(
                seed, particle_count=int(renderer["particle_count_representatives"])
            )
            deterministic &= particles.fingerprint == replay_particles.fingerprint
            particle_checks.append(
                {
                    "galaxy": name,
                    "particles": int(len(particles.positions_kpc)),
                    "luminosity_fractional_error": float(
                        particles.luminosities_lsun.sum() / seed.light.total_lsun - 1.0
                    ),
                    "fingerprint": particles.fingerprint,
                }
            )
            np.savez_compressed(
                representative_output / f"{name}_replica.npz",
                coordinate_kpc=high_resolution.x_kpc[0],
                light_lsun_pc2=high_resolution.total_lsun_pc2,
                line_of_sight_velocity_km_s=high_resolution.line_of_sight_velocity_km_s,
                particle_positions_kpc=particles.positions_kpc,
                particle_velocities_km_s=particles.velocities_km_s,
                particle_luminosities_lsun=particles.luminosities_lsun,
                particle_components=particles.components,
            )
            plot_representative(
                seed, high_resolution, representative_output / f"{name}_replica.png"
            )
        if index % 25 == 0 or index == len(names):
            print(f"Rendered {index}/{len(names)} galaxies")

    frame = pd.DataFrame(scores).sort_values("galaxy").reset_index(drop=True)
    frame.to_csv(output / protocol["outputs"]["scores"], index=False)
    catalog_columns = [
        "galaxy",
        "split",
        "hubble_type",
        "inclination_deg",
        "quality",
        "distance_mpc",
        "input_luminosity_lsun",
        "bulge_fraction",
        "apparent_axis_ratio",
        "light_knots",
        "angular_photometry_knots",
        "rotation_knots",
    ]
    frame[catalog_columns].to_csv(output / protocol["outputs"]["catalog"], index=False)
    pd.DataFrame(particle_checks).to_csv(output / "particle_checks.csv", index=False)

    aggregate = summaries(frame)
    gates = protocol["predeclared_replica_gates"]
    gate_results = {
        "median_major_axis_light_rmse": aggregate["all"]["median_light_rmse_dex"]
        <= float(gates["median_major_axis_light_rmse_dex_max"]),
        "p90_major_axis_light_rmse": aggregate["all"]["p90_light_rmse_dex"]
        <= float(gates["p90_major_axis_light_rmse_dex_max"]),
        "median_rotation_rmse": aggregate["all"]["median_rotation_rmse_km_s"]
        <= float(gates["median_rotation_rmse_km_s_max"]),
        "p90_rotation_rmse": aggregate["all"]["p90_rotation_rmse_km_s"]
        <= float(gates["p90_rotation_rmse_km_s_max"]),
        "median_total_light": aggregate["all"]["median_abs_total_light_fractional_error"]
        <= float(gates["median_total_light_fractional_error_max"]),
        "deterministic_replay": deterministic,
    }
    report = {
        "protocol_id": protocol["protocol_id"],
        "replica_gate_pass": bool(all(gate_results.values())),
        "gate_results": gate_results,
        "split_counts": dict(actual_counts),
        "aggregate": aggregate,
        "particle_checks": particle_checks,
        "provenance": {
            "protocol_sha256": sha256(args.protocol),
            "sparc_replica_provenance_sha256": sha256(ROOT / inputs["provenance"]),
        },
        "interpretation": {
            "what_pass_means": "The generator and virtual telescope can reproduce the supplied radial light and rotation observables.",
            "what_pass_does_not_mean": "The observed rotation curve is an input in replica mode, so this is not evidence that any gravity law predicts it.",
            "blind_test_boundary": "Blind physics mode must call render_replica with theory-predicted velocities and may not use observed velocity or lensing targets.",
            "morphology_limit": "The current SPARC products constrain radial photometry, not observed pixel-level bars, arms, warps, or lopsidedness.",
        },
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )

    representatives = list(renderer["representative_galaxies"])
    overview, axes = plt.subplots(2, len(representatives), figsize=(15, 7.2), constrained_layout=True)
    for column, name in enumerate(representatives):
        rendered = rendered_representatives[name]
        coordinate = rendered.x_kpc[0]
        extent = [coordinate[0], coordinate[-1], coordinate[0], coordinate[-1]]
        total = rendered.total_lsun_pc2
        positive = total[total > 0.0]
        floor = max(float(np.quantile(positive, 0.02)), float(positive.max()) * 1.0e-5)
        axes[0, column].imshow(
            total,
            origin="lower",
            extent=extent,
            cmap="magma",
            norm=LogNorm(vmin=floor, vmax=float(positive.max())),
        )
        vmax = float(np.nanmax(np.abs(rendered.line_of_sight_velocity_km_s)))
        axes[1, column].imshow(
            rendered.line_of_sight_velocity_km_s,
            origin="lower",
            extent=extent,
            cmap="coolwarm",
            norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
        )
        axes[0, column].set_title(name)
        axes[0, column].set_ylabel("light" if column == 0 else "")
        axes[1, column].set_ylabel("LOS velocity" if column == 0 else "")
        axes[1, column].set_xlabel("x [kpc]")
        for axis in axes[:, column]:
            axis.set_xticks([])
            axis.set_yticks([])
    overview.suptitle("P0631 observation-matched replicas: dwarf → disk → bulge-heavy → edge-on")
    overview.savefig(output / protocol["outputs"]["overview_figure"], dpi=180)
    plt.close(overview)

    summary = [
        "# P0631 observation-matched galaxy replicas",
        "",
        f"**Replica gate: {'PASS' if report['replica_gate_pass'] else 'FAIL'}**",
        "",
        f"- Galaxies: {len(frame)} ({actual_counts['train']} train / {actual_counts['development']} development / {actual_counts['holdout']} holdout).",
        f"- Median angular-photometry reconstruction: {aggregate['all']['median_angular_photometry_rmse_dex']:.6f} dex.",
        f"- Median continuous rotation reconstruction: {aggregate['all']['median_rotation_rmse_km_s']:.6f} km/s.",
        f"- Median finite-grid rotation loss: {aggregate['all']['median_pixelized_rotation_rmse_km_s']:.3f} km/s.",
        f"- Median absolute total-light integration error: {100.0 * aggregate['all']['median_abs_total_light_fractional_error']:.2f}%.",
        f"- Deterministic replay: {deterministic}.",
        "",
        "## Meaning",
        "",
        "This establishes that the simulator can generate an axisymmetric galaxy whose radial 3.6 μm light profile, projected inclination, and velocity field reproduce the supplied SPARC observables. Rotation is supplied in replica mode, so this is a reconstruction test—not a successful gravity prediction.",
        "",
        "For a gravity test, the identical light seed is retained but the observed velocity is removed. `render_replica` then requires an explicit theory-predicted circular-speed curve. The theory is scored only against the hidden observed curve.",
        "",
        "## Current observational limit",
        "",
        "The downloaded SPARC products contain radial profiles rather than raw two-dimensional 3.6 μm and H I images. The simulator therefore does not yet claim to reproduce observed bars, spiral arms, warps, gas clumps, or lopsidedness. Resolved survey cutouts and velocity cubes are the next data layer for that test.",
    ]
    (output / protocol["outputs"]["summary"]).write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps(json_safe(report), indent=2))


if __name__ == "__main__":
    main()
