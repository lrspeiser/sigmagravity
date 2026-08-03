"""Fuse SINGS and AllWISE development maps without using kinematic targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "src"))

import run_p0739_spiral_baryonic_registration as base  # noqa: E402
from voidscreen.sparc_morphology import parse_sparc_metadata  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/p0741_fused_spiral_baryonic_registration_development.json"
DEFAULT_OUTPUT = ROOT / "results/p0741_fused_spiral_baryonic_registration_development"
P0738_RAW = ROOT / "data/raw/p0738_things_sings_resolved"
P0739_RESULT = ROOT / "results/p0739_spiral_baryonic_registration_development"
P0740_RAW = ROOT / "data/raw/p0740_allwise_w1_supplement"
P0740_RESULT = ROOT / "results/p0740_allwise_w1_coverage_supplement"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def border_mask(valid: np.ndarray, fraction: float = 0.15) -> np.ndarray:
    border_y = max(1, int(round(fraction * valid.shape[0])))
    border_x = max(1, int(round(fraction * valid.shape[1])))
    border = np.zeros(valid.shape, dtype=bool)
    border[:border_y, :] = True
    border[-border_y:, :] = True
    border[:, :border_x] = True
    border[:, -border_x:] = True
    result = valid & border
    if int(result.sum()) < 64:
        result = valid
    return result


def beam_match(image: np.ndarray, input_fwhm_kpc: float, target_fwhm_kpc: float, spacing: float) -> np.ndarray:
    additional = math.sqrt(max(target_fwhm_kpc**2 - input_fwhm_kpc**2, 0.0))
    sigma_cells = additional / 2.355 / spacing
    return gaussian_filter(np.nan_to_num(image, nan=0.0), sigma=sigma_cells, mode="constant")


def robust_fractional_spread(values: list[float], reference: float) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value) and value > 0.0])
    if finite.size < 2:
        return math.inf
    return float(1.4826 * np.median(np.abs(finite - np.median(finite))) / reference)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    p0739_result = ROOT / config["parents"].get(
        "p0739ResultPath", "results/p0739_spiral_baryonic_registration_development"
    )
    p0740_result = ROOT / config["parents"].get(
        "p0740ResultPath", "results/p0740_allwise_w1_coverage_supplement"
    )
    p0739_report = read_json(p0739_result / "report.json")
    p0740_report = read_json(p0740_result / "manifest.json")
    expected_p0739_result = config["parents"].get("p0739ResultSha256")
    expected_p0739_config = config["parents"].get("p0739ConfigSha256")
    if expected_p0739_result and p0739_report["reportSha256"] != expected_p0739_result:
        raise ValueError("P0739 parent result hash mismatch")
    if expected_p0739_config and p0739_report["configSha256"] != expected_p0739_config:
        raise ValueError("P0739 parent config hash mismatch")
    if p0740_report["reportSha256"] != config["parents"]["p0740ResultSha256"]:
        raise ValueError("P0740 parent hash mismatch")

    wise_manifest = pd.read_csv(p0740_result / "file_manifest.csv")
    p0739_audit = pd.read_csv(p0739_result / "map_audit.csv").set_index("galaxy")
    metadata = parse_sparc_metadata(base.SPARC_TABLE).set_index("galaxy")
    coordinates = base.load_coordinates()
    args.output.mkdir(parents=True, exist_ok=True)
    maps_directory = args.output / "maps"
    maps_directory.mkdir(exist_ok=True)

    records: list[dict[str, Any]] = []
    atlas: list[dict[str, Any]] = []
    wise_arrays_opened = 0
    for galaxy in config["systems"]:
        audit = p0739_audit.loc[galaxy]
        split = str(audit.split)
        if split not in set(config.get("eligibleSplits", ["development"])):
            raise ValueError(f"array split is outside this frozen config: {galaxy} ({split})")
        meta = metadata.loc[galaxy]
        center: SkyCoord = coordinates[galaxy]
        distance_mpc = float(audit.distance_mpc)
        inclination_deg = float(audit.inclination_deg)
        position_angle_deg = float(audit.photometric_position_angle_deg)
        effective_radius_kpc = float(meta.effective_radius_kpc)
        hi_radius_kpc = float(meta.HI_radius_kpc)
        stellar_mass = float(audit.target_stellar_mass_solar)

        p0739_map_path = p0739_result / "maps" / f"{galaxy}.npz"
        expected_map_hash = next(
            row["sha256"] for row in p0739_report["mapFiles"] if row["galaxy"] == galaxy
        )
        if file_sha256(p0739_map_path) != expected_map_hash:
            raise ValueError(f"P0739 map hash mismatch for {galaxy}")
        with np.load(p0739_map_path) as prior:
            axis = np.asarray(prior["axis_kpc"], dtype=float)
            gas = np.asarray(prior["gas"], dtype=float)
        spacing = float(axis[1] - axis[0])
        xx, yy = np.meshgrid(axis, axis)
        radius_plane = np.hypot(xx, yy)
        beam_kpc = float(audit.things_beam_kpc)
        kpc_per_arcsec = distance_mpc * 1000.0 / 206265.0

        source_dir = P0738_RAW / galaxy
        irac_path = next(source_dir.glob("*v7.phot.1.fits"))
        irac_weight_path = next(source_dir.glob("*_wt.fits"))
        with fits.open(irac_path, memmap=True) as hdus:
            irac = np.asarray(hdus[0].data, dtype=float).squeeze()
            irac_header = hdus[0].header.copy()
        with fits.open(irac_weight_path, memmap=True) as hdus:
            irac_weight = np.asarray(hdus[0].data, dtype=float).squeeze()
        irac_wcs = base.celestial_wcs(irac_header, ignore_inconsistent_sip=True)
        irac_center = tuple(float(value) for value in irac_wcs.world_to_pixel(center))
        irac_radius = base.approximate_sky_radius_kpc(
            irac.shape, irac_center, base.local_pixel_scale_arcsec(irac_wcs), distance_mpc
        )
        irac_valid = np.isfinite(irac) & np.isfinite(irac_weight) & (irac_weight > 0.0)
        irac_background_mask, _ = base.outer_background_mask(
            irac_valid,
            irac_radius,
            max(1.2 * hi_radius_kpc, 6.0 * effective_radius_kpc),
        )
        irac_background, irac_noise = base.robust_location_scale(irac[irac_background_mask])
        irac_subtracted = np.where(irac_valid, irac - irac_background, 0.0)
        irac_clean, irac_foreground_count, _ = base.compact_foreground_suppression(
            irac_subtracted,
            irac_valid,
            irac_radius,
            effective_radius_kpc,
            irac_background_mask,
        )
        irac_clean = np.clip(irac_clean, 0.0, None)
        irac_x, irac_y = base.plane_world_pixels(
            axis, center, position_angle_deg, inclination_deg, distance_mpc, irac_wcs
        )
        sings_sample = base.sample_plane(irac_clean, irac_x, irac_y)
        sings_weight = base.sample_plane(irac_weight, irac_x, irac_y)
        sings_footprint = np.isfinite(sings_sample) & np.isfinite(sings_weight) & (sings_weight > 0.0)
        sings_native = np.where(sings_footprint, sings_sample, 0.0)
        sings_matched = beam_match(sings_native, 1.7 * kpc_per_arcsec, beam_kpc, spacing)

        numerator = np.zeros_like(sings_matched)
        denominator = np.zeros_like(sings_matched)
        tile_backgrounds: list[float] = []
        galaxy_files = wise_manifest[
            (wise_manifest.galaxy == galaxy) & (wise_manifest["product"] == "intensity")
        ]
        for file_row in galaxy_files.itertuples(index=False):
            coadd = file_row.coadd_id
            intensity_path = ROOT / file_row.relative_path
            uncertainty_row = wise_manifest[
                (wise_manifest.galaxy == galaxy)
                & (wise_manifest.coadd_id == coadd)
                & (wise_manifest["product"] == "uncertainty")
            ].iloc[0]
            uncertainty_path = ROOT / uncertainty_row.relative_path
            if file_sha256(intensity_path) != file_row.sha256:
                raise ValueError(f"WISE intensity hash mismatch for {galaxy} {coadd}")
            if file_sha256(uncertainty_path) != uncertainty_row.sha256:
                raise ValueError(f"WISE uncertainty hash mismatch for {galaxy} {coadd}")
            with fits.open(intensity_path, memmap=True) as hdus:
                intensity = np.asarray(hdus[0].data, dtype=float).squeeze()
                wise_header = hdus[0].header.copy()
            with fits.open(uncertainty_path, memmap=True) as hdus:
                uncertainty = np.asarray(hdus[0].data, dtype=float).squeeze()
            wise_arrays_opened += 2
            wise_wcs = base.celestial_wcs(wise_header)
            valid = np.isfinite(intensity) & np.isfinite(uncertainty) & (uncertainty > 0.0)
            background_mask = border_mask(valid)
            background, _ = base.robust_location_scale(intensity[background_mask])
            tile_backgrounds.append(background)
            wise_signal = np.where(valid, intensity - background, np.nan)
            wise_x, wise_y = base.plane_world_pixels(
                axis, center, position_angle_deg, inclination_deg, distance_mpc, wise_wcs
            )
            sampled_signal = base.sample_plane(wise_signal, wise_x, wise_y)
            sampled_uncertainty = base.sample_plane(uncertainty, wise_x, wise_y)
            sampled_valid = (
                np.isfinite(sampled_signal)
                & np.isfinite(sampled_uncertainty)
                & (sampled_uncertainty > 0.0)
            )
            weights = np.where(sampled_valid, 1.0 / np.square(sampled_uncertainty), 0.0)
            numerator += np.where(sampled_valid, sampled_signal * weights, 0.0)
            denominator += weights

        wise_footprint = denominator > 0.0
        wise_mosaic = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=wise_footprint,
        )
        wise_uncertainty = np.divide(
            1.0,
            np.sqrt(denominator),
            out=np.full_like(denominator, np.nan),
            where=wise_footprint,
        )
        wise_background_mask, _ = base.outer_background_mask(
            wise_footprint, radius_plane, max(1.2 * hi_radius_kpc, 6.0 * effective_radius_kpc)
        )
        wise_clean, wise_foreground_count, _ = base.compact_foreground_suppression(
            wise_mosaic,
            wise_footprint,
            radius_plane,
            effective_radius_kpc,
            wise_background_mask,
        )
        wise_clean = np.clip(wise_clean, 0.0, None)
        wise_matched = beam_match(wise_clean, 6.1 * kpc_per_arcsec, beam_kpc, spacing)

        wise_noise = float(np.nanmedian(wise_uncertainty[wise_footprint]))
        overlap_base = (
            sings_footprint
            & wise_footprint
            & (radius_plane >= 0.2 * effective_radius_kpc)
            & (radius_plane <= 5.0 * effective_radius_kpc)
            & (sings_matched > 3.0 * irac_noise)
            & (wise_matched > 3.0 * wise_noise)
        )
        sings_cap = float(np.quantile(sings_matched[overlap_base], 0.995))
        wise_cap = float(np.quantile(wise_matched[overlap_base], 0.995))
        overlap = overlap_base & (sings_matched <= sings_cap) & (wise_matched <= wise_cap)
        ratios = sings_matched[overlap] / wise_matched[overlap]
        scale = float(np.median(ratios))
        correlation = float(spearmanr(sings_matched[overlap], wise_matched[overlap]).statistic)
        quadrant_scales: list[float] = []
        for quadrant in ((xx >= 0) & (yy >= 0), (xx < 0) & (yy >= 0), (xx < 0) & (yy < 0), (xx >= 0) & (yy < 0)):
            selected = overlap & quadrant
            quadrant_scales.append(
                float(np.median(sings_matched[selected] / wise_matched[selected]))
                if int(selected.sum()) >= 50
                else math.nan
            )
        quadrant_spread = robust_fractional_spread(quadrant_scales, scale)

        scaled_wise = scale * wise_matched
        fused = np.where(sings_footprint, sings_matched, scaled_wise)
        fused_footprint = sings_footprint | wise_footprint
        fused = np.where(fused_footprint, fused, 0.0)
        stars = base.normalized_surface(fused, stellar_mass, spacing)
        total = gas + stars
        inside_hi = radius_plane <= hi_radius_kpc
        coverage = float(np.mean(fused_footprint[inside_hi]))
        outside_sings = inside_hi & ~sings_footprint
        outside_fill_fraction = float(
            np.mean((scaled_wise > 0.0)[outside_sings]) if np.any(outside_sings) else 1.0
        )
        gas_mass_actual = float(np.sum(gas) * spacing**2)
        stellar_mass_actual = float(np.sum(stars) * spacing**2)

        output_path = maps_directory / f"{galaxy}.npz"
        with output_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                axis_kpc=axis,
                gas=gas,
                stars=stars,
                total=total,
                sings_beam_matched=sings_matched,
                wise_beam_matched_scaled=scaled_wise,
                fused_stellar_shape=fused,
                sings_footprint=sings_footprint.astype(np.uint8),
                wise_footprint=wise_footprint.astype(np.uint8),
                fused_footprint=fused_footprint.astype(np.uint8),
            )
        record = {
            "galaxy": galaxy,
            "split": split,
            "hubble_type": int(meta.hubble_type),
            "distance_mpc": distance_mpc,
            "inclination_deg": inclination_deg,
            "photometric_position_angle_deg": position_angle_deg,
            "cells_per_axis": len(axis),
            "half_extent_kpc": float(axis[-1]),
            "spacing_kpc": spacing,
            "things_beam_kpc": beam_kpc,
            "wise_coadds": len(galaxy_files),
            "wise_arrays_opened": 2 * len(galaxy_files),
            "wise_tile_background_median_dn": float(np.median(tile_backgrounds)),
            "wise_foreground_pixels_replaced": wise_foreground_count,
            "sings_foreground_pixels_replaced": irac_foreground_count,
            "cross_calibration_pixels": int(overlap.sum()),
            "wise_to_sings_scale": scale,
            "sings_wise_spearman": correlation,
            "quadrant_scale_q1": quadrant_scales[0],
            "quadrant_scale_q2": quadrant_scales[1],
            "quadrant_scale_q3": quadrant_scales[2],
            "quadrant_scale_q4": quadrant_scales[3],
            "quadrant_scale_robust_fractional_spread": quadrant_spread,
            "sings_footprint_fraction_inside_hi_r995": float(np.mean(sings_footprint[inside_hi])),
            "fused_footprint_fraction_inside_hi_r995": coverage,
            "positive_wise_fill_fraction_outside_sings_inside_hi": outside_fill_fraction,
            "target_gas_mass_solar": float(audit.target_gas_mass_solar),
            "gas_mass_solar": gas_mass_actual,
            "gas_mass_relative_error": abs(gas_mass_actual - float(audit.target_gas_mass_solar))
            / float(audit.target_gas_mass_solar),
            "target_stellar_mass_solar": stellar_mass,
            "stellar_mass_solar": stellar_mass_actual,
            "stellar_mass_relative_error": abs(stellar_mass_actual - stellar_mass) / stellar_mass,
            "outer_cell_mass_fraction": base.edge_mass_fraction(total),
            "map_sha256": file_sha256(output_path),
            "validation_arrays_opened": (2 + 2 * len(galaxy_files)) if split == "validation" else 0,
            "holdout_arrays_opened": (2 + 2 * len(galaxy_files)) if split == "holdout" else 0,
            "velocity_or_dispersion_arrays_opened": 0,
            "gravity_parameters": 0,
        }
        records.append(record)
        atlas.append(
            {
                "galaxy": galaxy,
                "axis": axis,
                "sings": sings_matched,
                "wise": scaled_wise,
                "stars": stars,
                "total": total,
            }
        )
        print(
            f"{galaxy}: SINGS {100*np.mean(sings_footprint[inside_hi]):.1f}% -> "
            f"fused {100*coverage:.1f}%, rho={correlation:.3f}, scale spread={quadrant_spread:.3f}"
        )

    frame = pd.DataFrame(records)
    frame.to_csv(args.output / "map_audit.csv", index=False)
    gates = config["engineeringGates"]
    checks = {
        "requiredSystems": len(frame) == int(gates["requiredSystems"]),
        "requiredWiseCoadds": int(frame.wise_coadds.sum()) == int(gates["requiredWiseCoadds"]),
        "requiredValidationArraysOpened": int(frame.validation_arrays_opened.sum())
        == int(gates["requiredValidationArraysOpened"]),
        "requiredHoldoutArraysOpened": int(frame.holdout_arrays_opened.sum())
        == int(gates["requiredHoldoutArraysOpened"]),
        "requiredVelocityOrDispersionArraysOpened": int(
            frame.velocity_or_dispersion_arrays_opened.sum()
        )
        == int(gates["requiredVelocityOrDispersionArraysOpened"]),
        "maximumGravityParameters": int(frame.gravity_parameters.max())
        <= int(gates["maximumGravityParameters"]),
        "minimumFiniteStellarFootprintFractionInsideHiR995": float(
            frame.fused_footprint_fraction_inside_hi_r995.min()
        )
        >= float(gates["minimumFiniteStellarFootprintFractionInsideHiR995"]),
        "minimumCrossCalibrationPixels": int(frame.cross_calibration_pixels.min())
        >= int(gates["minimumCrossCalibrationPixels"]),
        "minimumPositiveSingsWiseSpearmanCorrelation": float(frame.sings_wise_spearman.min())
        >= float(gates["minimumPositiveSingsWiseSpearmanCorrelation"]),
        "maximumQuadrantScaleRobustFractionalSpread": float(
            frame.quadrant_scale_robust_fractional_spread.max()
        )
        <= float(gates["maximumQuadrantScaleRobustFractionalSpread"]),
        "maximumGasMassRelativeError": float(frame.gas_mass_relative_error.max())
        <= float(gates["maximumGasMassRelativeError"]),
        "maximumStellarMassRelativeError": float(frame.stellar_mass_relative_error.max())
        <= float(gates["maximumStellarMassRelativeError"]),
        "maximumOuterCellMassFraction": float(frame.outer_cell_mass_fraction.max())
        <= float(gates["maximumOuterCellMassFraction"]),
        "allOutputBundlesContentHashed": bool(frame.map_sha256.str.len().eq(64).all()),
    }
    status = "pass" if all(checks.values()) else "fail"

    fig, axes = plt.subplots(len(atlas), 4, figsize=(14, 3.1 * len(atlas)), constrained_layout=True)
    for row_index, item in enumerate(atlas):
        for column, (key, label) in enumerate(
            (("sings", "SINGS core"), ("wise", "scaled WISE"), ("stars", "fused stellar mass"), ("total", "total baryons"))
        ):
            values = item[key]
            positive = values[values > 0.0]
            vmax = float(np.quantile(positive, 0.995))
            ax = axes[row_index, column]
            image = ax.imshow(
                np.log10(np.clip(values, vmax * 1.0e-5, None)),
                origin="lower",
                extent=[item["axis"][0], item["axis"][-1], item["axis"][0], item["axis"][-1]],
                cmap="magma",
                vmin=math.log10(vmax) - 5.0,
                vmax=math.log10(vmax),
            )
            ax.set_title(f"{item['galaxy']} - {label}")
            ax.set_xlabel("x (kpc)")
            ax.set_ylabel("y (kpc)")
            fig.colorbar(image, ax=ax, fraction=0.046)
    atlas_name = config.get("atlasFileName", "fused_development_baryonic_map_atlas.png")
    fig.savefig(args.output / atlas_name, dpi=180)
    plt.close(fig)

    report_core = {
        "schemaVersion": config.get(
            "resultSchemaVersion", "sigma-p0741-fused-spiral-baryonic-registration-result/1"
        ),
        "stage": config.get("stage", "P0741"),
        "status": status,
        "configSha256": hashlib.sha256(config_bytes).hexdigest(),
        "p0739ResultSha256": p0739_report["reportSha256"],
        "p0740ResultSha256": p0740_report["reportSha256"],
        "systems": len(frame),
        "splitsOpened": sorted(frame.split.unique().tolist()),
        "wiseImageArraysOpened": wise_arrays_opened,
        "validationArraysOpened": int(frame.validation_arrays_opened.sum()),
        "holdoutArraysOpened": int(frame.holdout_arrays_opened.sum()),
        "velocityOrDispersionArraysOpened": int(frame.velocity_or_dispersion_arrays_opened.sum()),
        "gravityParameters": int(frame.gravity_parameters.sum()),
        "checks": checks,
        "aggregate": {
            "minimumSingsFootprintFractionInsideHiR995": float(
                frame.sings_footprint_fraction_inside_hi_r995.min()
            ),
            "minimumFusedFootprintFractionInsideHiR995": float(
                frame.fused_footprint_fraction_inside_hi_r995.min()
            ),
            "minimumSingsWiseSpearman": float(frame.sings_wise_spearman.min()),
            "maximumQuadrantScaleRobustFractionalSpread": float(
                frame.quadrant_scale_robust_fractional_spread.max()
            ),
        },
        "mapFiles": [
            {"galaxy": row.galaxy, "path": f"maps/{row.galaxy}.npz", "sha256": row.map_sha256}
            for row in frame.itertuples(index=False)
        ],
        "claimBoundary": config["claimBoundary"],
    }
    report = {**report_core, "reportSha256": canonical_sha256(report_core)}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    split_label = ", ".join(sorted(frame.split.unique().tolist()))
    summary = f"""# {config.get('stage', 'P0741')} fused baryonic registration

Status: **{status.upper()}**

- Systems: {len(frame)} ({split_label})
- WISE arrays opened: {wise_arrays_opened}
- Validation image arrays opened: {int(frame.validation_arrays_opened.sum())}
- Holdout image arrays opened: {int(frame.holdout_arrays_opened.sum())}
- Velocity or dispersion arrays opened: 0
- Gravity parameters: 0
- Minimum SINGS-only footprint inside H I R995: {100*frame.sings_footprint_fraction_inside_hi_r995.min():.2f}%
- Minimum fused footprint inside H I R995: {100*frame.fused_footprint_fraction_inside_hi_r995.min():.2f}%
- Minimum SINGS/WISE morphology Spearman correlation: {frame.sings_wise_spearman.min():.3f}
- Maximum quadrant calibration spread: {100*frame.quadrant_scale_robust_fractional_spread.max():.2f}%
- Report SHA-256: `{report['reportSha256']}`

This is an observational registration result, not a gravity-formula score.
"""
    (args.output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(json.dumps({"status": status, "checks": checks, "reportSha256": report["reportSha256"]}))
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
