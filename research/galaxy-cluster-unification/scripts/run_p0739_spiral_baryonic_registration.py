"""Build velocity-blind development maps for the frozen P0739 spiral sample."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter, map_coordinates


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sparc_morphology import parse_sparc_metadata  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/p0739_spiral_baryonic_registration_development.json"
DEFAULT_OUTPUT = ROOT / "results/p0739_spiral_baryonic_registration_development"
ACQUISITION_CONFIG = ROOT / "configs/p0738_morphology_diverse_resolved_acquisition.json"
ACQUISITION_MANIFEST = ROOT / "results/p0738_morphology_diverse_resolved_acquisition/manifest.json"
RAW = ROOT / "data/raw/p0738_things_sings_resolved"
COORDINATES = ROOT / "data/raw/sparc/coordinates.csv"
SPARC_TABLE = ROOT / "data/raw/sparc/table1.dat"
BEAM_PATTERN = re.compile(
    r"AIPS\s+CLEAN BMAJ=\s*([0-9.E+\-]+) BMIN=\s*([0-9.E+\-]+) BPA=\s*([0-9.E+\-]+)"
)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def regular_odd_cells(raw: int, minimum: int, maximum: int) -> int:
    cells = min(max(raw, minimum), maximum)
    if cells % 2 == 0:
        cells = cells + 1 if cells < maximum else cells - 1
    return cells


def robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("no finite background pixels")
    location = float(np.median(finite))
    scale = float(1.4826 * np.median(np.abs(finite - location)))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.std(finite))
    return location, max(scale, np.finfo(float).eps)


def local_pixel_scale_arcsec(wcs: WCS) -> float:
    matrix = np.asarray(wcs.pixel_scale_matrix, dtype=float)
    return float(math.sqrt(abs(np.linalg.det(matrix))) * 3600.0)


def celestial_wcs(header: fits.Header, *, ignore_inconsistent_sip: bool = False) -> WCS:
    """Return the celestial WCS, honoring the mosaic CTYPE over stale SIP cards."""

    wcs = WCS(header).celestial
    if ignore_inconsistent_sip and wcs.sip is not None and not any(
        "-SIP" in str(value).upper() for value in wcs.wcs.ctype
    ):
        # The SINGS ``phot`` products are already projected mosaics.  Their
        # CTYPE intentionally omits ``-SIP`` while legacy polynomial cards
        # remain in the header.  Applying those cards a second time makes the
        # inverse WCS diverge near otherwise valid mosaic edges.
        wcs.sip = None
    return wcs


def approximate_sky_radius_kpc(
    shape: tuple[int, int], center_pixel: tuple[float, float], pixel_arcsec: float, distance_mpc: float
) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=float)
    radius_pixels = np.hypot(xx - center_pixel[0], yy - center_pixel[1])
    return radius_pixels * pixel_arcsec * distance_mpc * 1000.0 / 206265.0


def outer_background_mask(
    valid: np.ndarray,
    radius_kpc: np.ndarray,
    threshold_kpc: float,
) -> tuple[np.ndarray, str]:
    candidate = valid & (radius_kpc >= threshold_kpc)
    if int(candidate.sum()) >= max(4096, int(0.01 * valid.sum())):
        return candidate, "physical_outer_annulus"
    border_y = max(1, int(round(0.15 * valid.shape[0])))
    border_x = max(1, int(round(0.15 * valid.shape[1])))
    border = np.zeros(valid.shape, dtype=bool)
    border[:border_y, :] = True
    border[-border_y:, :] = True
    border[:, :border_x] = True
    border[:, -border_x:] = True
    return valid & border, "outer_15_percent_border"


def source_coordinate_roundtrip_arcsec(wcs: WCS, center: SkyCoord) -> float:
    x, y = wcs.world_to_pixel(center)
    recovered = wcs.pixel_to_world(x, y)
    return float(center.separation(recovered).arcsec)


def beam_from_history(header: fits.Header) -> tuple[float, float, float]:
    for history in reversed(header.get("HISTORY", [])):
        match = BEAM_PATTERN.search(str(history))
        if match:
            return tuple(float(match.group(index)) for index in range(1, 4))
    raise ValueError("THINGS clean beam is absent from FITS HISTORY")


def compact_foreground_suppression(
    image: np.ndarray,
    valid: np.ndarray,
    radius_kpc: np.ndarray,
    effective_radius_kpc: float,
    background_mask: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    filled = np.where(valid, image, 0.0)
    smooth = gaussian_filter(filled, sigma=2.0, mode="nearest")
    residual = filled - smooth
    residual_location, residual_scale = robust_location_scale(residual[background_mask])
    threshold = residual_location + 12.0 * residual_scale
    compact = valid & (residual > threshold) & (radius_kpc > 0.2 * effective_radius_kpc)
    cleaned = np.where(compact, smooth, filled)
    return cleaned, int(compact.sum()), threshold


def photometric_position_angle(
    image: np.ndarray,
    valid: np.ndarray,
    radius_kpc: np.ndarray,
    center_pixel: tuple[float, float],
    wcs: WCS,
    effective_radius_kpc: float,
) -> tuple[float, float, float]:
    signal = np.where(valid & (radius_kpc <= 5.0 * effective_radius_kpc), image, 0.0)
    positive = signal[signal > 0.0]
    if positive.size < 100:
        raise ValueError("insufficient stellar light for position-angle inference")
    cap = float(np.quantile(positive, 0.995))
    weights = np.clip(signal, 0.0, cap)
    yy, xx = np.indices(signal.shape, dtype=float)
    dx = xx - center_pixel[0]
    dy = yy - center_pixel[1]
    matrix = np.asarray(wcs.pixel_scale_matrix, dtype=float)
    east = matrix[0, 0] * dx + matrix[0, 1] * dy
    north = matrix[1, 0] * dx + matrix[1, 1] * dy
    total = float(np.sum(weights))
    covariance = np.array(
        [
            [np.sum(weights * east * east), np.sum(weights * east * north)],
            [np.sum(weights * east * north), np.sum(weights * north * north)],
        ],
        dtype=float,
    ) / total
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major = eigenvectors[:, int(np.argmax(eigenvalues))]
    position_angle = float(np.degrees(np.arctan2(major[0], major[1])) % 180.0)
    axis_ratio = float(math.sqrt(max(eigenvalues.min(), 0.0) / eigenvalues.max()))
    return position_angle, axis_ratio, cap


def plane_world_pixels(
    axis_kpc: np.ndarray,
    center: SkyCoord,
    position_angle_deg: float,
    inclination_deg: float,
    distance_mpc: float,
    wcs: WCS,
) -> tuple[np.ndarray, np.ndarray]:
    xx, yy = np.meshgrid(axis_kpc, axis_kpc)
    pa = math.radians(position_angle_deg)
    projected_minor = yy * math.cos(math.radians(inclination_deg))
    east_kpc = xx * math.sin(pa) + projected_minor * math.cos(pa)
    north_kpc = xx * math.cos(pa) - projected_minor * math.sin(pa)
    kpc_per_arcsec = distance_mpc * 1000.0 / 206265.0
    east_deg = east_kpc / kpc_per_arcsec / 3600.0
    north_deg = north_kpc / kpc_per_arcsec / 3600.0
    ra_deg = center.ra.deg + east_deg / math.cos(center.dec.radian)
    dec_deg = center.dec.deg + north_deg
    return wcs.world_to_pixel_values(ra_deg, dec_deg)


def sample_plane(image: np.ndarray, x_pixel: np.ndarray, y_pixel: np.ndarray) -> np.ndarray:
    return map_coordinates(
        np.asarray(image, dtype=float),
        [y_pixel, x_pixel],
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )


def normalized_surface(surface: np.ndarray, target_mass: float, spacing_kpc: float) -> np.ndarray:
    clipped = np.clip(np.nan_to_num(surface, nan=0.0), 0.0, None)
    mass = float(np.sum(clipped) * spacing_kpc**2)
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("registered component has no positive mass")
    return clipped * (target_mass / mass)


def edge_mass_fraction(surface: np.ndarray) -> float:
    edge = np.zeros(surface.shape, dtype=bool)
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    return float(np.sum(surface[edge]) / np.sum(surface))


def centroid_kpc(surface: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    xx, yy = np.meshgrid(axis, axis)
    total = float(np.sum(surface))
    return float(np.sum(surface * xx) / total), float(np.sum(surface * yy) / total)


def load_coordinates() -> dict[str, SkyCoord]:
    result: dict[str, SkyCoord] = {}
    with COORDINATES.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            result[row["name"]] = SkyCoord(float(row["ra_deg"]), float(row["dec_deg"]), unit="deg")
    return result


def opened_input_hashes(acquisition: dict[str, Any]) -> dict[str, dict[str, str]]:
    allowed = {"moment0", "irac1", "irac1_weight"}
    return {
        galaxy: {
            row["kind"]: row["sha256"]
            for row in acquisition["files"]
            if row["galaxy"] == galaxy and row["kind"] in allowed
        }
        for galaxy in {row["galaxy"] for row in acquisition["files"]}
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    acquisition_config = read_json(ACQUISITION_CONFIG)
    acquisition = read_json(ACQUISITION_MANIFEST)
    if acquisition["manifestSha256"] != config["parent"]["manifestSha256"]:
        raise ValueError("P0738 parent manifest hash does not match the frozen P0739 config")
    systems = {item["id"]: item for item in acquisition_config["systems"]}
    eligible_splits = set(config.get("eligibleSplits", ["development"]))
    metadata = parse_sparc_metadata(SPARC_TABLE).set_index("galaxy")
    coordinates = load_coordinates()
    hashes = opened_input_hashes(acquisition)
    args.output.mkdir(parents=True, exist_ok=True)
    maps_directory = args.output / "maps"
    maps_directory.mkdir(exist_ok=True)

    records: list[dict[str, Any]] = []
    atlas: list[dict[str, Any]] = []
    for galaxy in config["systems"]:
        split = systems[galaxy]["split"]
        if split not in eligible_splits:
            raise ValueError(f"array split is outside this frozen config: {galaxy} ({split})")
        directory = RAW / galaxy
        moment0_path = next(directory.glob("*MOM0_THINGS.FITS"))
        irac_path = next(directory.glob("*v7.phot.1.fits"))
        weight_path = next(directory.glob("*_wt.fits"))
        for kind, path in (("moment0", moment0_path), ("irac1", irac_path), ("irac1_weight", weight_path)):
            if file_sha256(path) != hashes[galaxy][kind]:
                raise ValueError(f"input hash mismatch for {galaxy} {kind}")

        with fits.open(moment0_path, memmap=True) as hdus:
            moment0 = np.asarray(hdus[0].data, dtype=float).squeeze()
            hi_header = hdus[0].header.copy()
        with fits.open(irac_path, memmap=True) as hdus:
            irac = np.asarray(hdus[0].data, dtype=float).squeeze()
            irac_header = hdus[0].header.copy()
        with fits.open(weight_path, memmap=True) as hdus:
            irac_weight = np.asarray(hdus[0].data, dtype=float).squeeze()

        hi_wcs = celestial_wcs(hi_header)
        irac_had_inconsistent_sip = bool(
            WCS(irac_header).celestial.sip is not None
            and "-SIP" not in str(irac_header.get("CTYPE1", "")).upper()
        )
        irac_wcs = celestial_wcs(irac_header, ignore_inconsistent_sip=True)
        center = coordinates[galaxy]
        row = metadata.loc[galaxy]
        distance_mpc = float(row.distance_mpc)
        inclination_deg = float(row.inclination_deg)
        effective_radius_kpc = float(row.effective_radius_kpc)
        hi_radius_kpc = float(row.HI_radius_kpc)
        gas_mass = 1.36 * float(row.HI_mass_billion_solar) * 1e9
        stellar_mass = 0.5 * float(row.luminosity_3p6_billion_solar) * 1e9

        irac_center = tuple(float(value) for value in irac_wcs.world_to_pixel(center))
        irac_pixel_arcsec = local_pixel_scale_arcsec(irac_wcs)
        irac_radius = approximate_sky_radius_kpc(
            irac.shape, irac_center, irac_pixel_arcsec, distance_mpc
        )
        irac_valid = np.isfinite(irac) & np.isfinite(irac_weight) & (irac_weight > 0.0)
        irac_background_mask, irac_background_method = outer_background_mask(
            irac_valid,
            irac_radius,
            max(1.2 * hi_radius_kpc, 6.0 * effective_radius_kpc),
        )
        irac_background, irac_noise = robust_location_scale(irac[irac_background_mask])
        irac_subtracted = np.where(irac_valid, irac - irac_background, 0.0)
        irac_clean, foreground_pixels, foreground_threshold = compact_foreground_suppression(
            irac_subtracted,
            irac_valid,
            irac_radius,
            effective_radius_kpc,
            irac_background_mask,
        )
        irac_clean = np.clip(irac_clean, 0.0, None)
        position_angle_deg, photometric_axis_ratio, irac_cap = photometric_position_angle(
            irac_clean,
            irac_valid,
            irac_radius,
            irac_center,
            irac_wcs,
            effective_radius_kpc,
        )

        hi_center = tuple(float(value) for value in hi_wcs.world_to_pixel(center))
        hi_pixel_arcsec = local_pixel_scale_arcsec(hi_wcs)
        hi_radius_source = approximate_sky_radius_kpc(
            moment0.shape, hi_center, hi_pixel_arcsec, distance_mpc
        )
        hi_valid = np.isfinite(moment0)
        hi_background_mask, hi_background_method = outer_background_mask(
            hi_valid, hi_radius_source, 1.2 * hi_radius_kpc
        )
        hi_background, hi_noise = robust_location_scale(moment0[hi_background_mask])
        hi_signal = moment0 - hi_background
        hi_threshold = max(0.0, 3.0 * hi_noise)
        hi_clean = np.where(hi_valid & (hi_signal > hi_threshold), hi_signal, 0.0)

        beam_major_deg, beam_minor_deg, beam_pa_deg = beam_from_history(hi_header)
        kpc_per_arcsec = distance_mpc * 1000.0 / 206265.0
        beam_geometric_arcsec = 3600.0 * math.sqrt(beam_major_deg * beam_minor_deg)
        beam_kpc = beam_geometric_arcsec * kpc_per_arcsec
        extent_kpc = 1.2 * max(hi_radius_kpc, 6.0 * effective_radius_kpc)
        target_spacing = beam_kpc / float(config["processing"]["grid"]["targetCellsPerThingsBeam"])
        raw_cells = int(math.ceil(2.0 * extent_kpc / target_spacing)) + 1
        cells = regular_odd_cells(
            raw_cells,
            int(config["processing"]["grid"]["minimumOddCells"]),
            int(config["processing"]["grid"]["maximumOddCells"]),
        )
        axis = np.linspace(-extent_kpc, extent_kpc, cells)
        spacing = float(axis[1] - axis[0])

        hi_x, hi_y = plane_world_pixels(
            axis, center, position_angle_deg, inclination_deg, distance_mpc, hi_wcs
        )
        irac_x, irac_y = plane_world_pixels(
            axis, center, position_angle_deg, inclination_deg, distance_mpc, irac_wcs
        )
        sampled_hi = sample_plane(hi_clean, hi_x, hi_y)
        sampled_irac = sample_plane(irac_clean, irac_x, irac_y)
        sampled_irac_weight = sample_plane(irac_weight, irac_x, irac_y)
        hi_footprint = np.isfinite(sampled_hi)
        stellar_footprint = np.isfinite(sampled_irac) & np.isfinite(sampled_irac_weight) & (
            sampled_irac_weight > 0.0
        )

        gas = normalized_surface(sampled_hi, gas_mass, spacing)
        stars_raw = np.clip(np.nan_to_num(sampled_irac, nan=0.0), 0.0, None)
        irac_psf_fwhm_kpc = 1.7 * kpc_per_arcsec
        additional_fwhm_kpc = math.sqrt(max(beam_kpc**2 - irac_psf_fwhm_kpc**2, 0.0))
        additional_sigma_cells = additional_fwhm_kpc / 2.355 / spacing
        stars_smoothed = gaussian_filter(stars_raw, sigma=additional_sigma_cells, mode="constant")
        stars = normalized_surface(stars_smoothed, stellar_mass, spacing)
        total = gas + stars
        radius_plane = np.hypot(*np.meshgrid(axis, axis))
        inside_hi = radius_plane <= hi_radius_kpc
        joint_footprint = hi_footprint & stellar_footprint
        footprint_fraction = float(np.mean(joint_footprint[inside_hi]))
        gas_mass_actual = float(np.sum(gas) * spacing**2)
        stellar_mass_actual = float(np.sum(stars) * spacing**2)
        total_mass_actual = float(np.sum(total) * spacing**2)
        gas_centroid = centroid_kpc(gas, axis)
        stellar_centroid = centroid_kpc(stars, axis)

        output_path = maps_directory / f"{galaxy}.npz"
        with output_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                axis_kpc=axis,
                gas=gas,
                stars=stars,
                total=total,
                finite_footprint=joint_footprint.astype(np.uint8),
            )
        map_sha = file_sha256(output_path)
        record = {
            "galaxy": galaxy,
            "split": split,
            "hubble_type": int(row.hubble_type),
            "distance_mpc": distance_mpc,
            "inclination_deg": inclination_deg,
            "photometric_position_angle_deg": position_angle_deg,
            "photometric_axis_ratio": photometric_axis_ratio,
            "cells_per_axis": cells,
            "half_extent_kpc": extent_kpc,
            "spacing_kpc": spacing,
            "things_beam_geometric_arcsec": beam_geometric_arcsec,
            "things_beam_kpc": beam_kpc,
            "cells_per_things_beam": beam_kpc / spacing,
            "things_beam_position_angle_deg": beam_pa_deg,
            "irac_background_method": irac_background_method,
            "irac_inconsistent_sip_ignored": irac_had_inconsistent_sip,
            "irac_background_mjy_sr": irac_background,
            "irac_noise_mjy_sr": irac_noise,
            "irac_foreground_pixels_replaced": foreground_pixels,
            "irac_foreground_threshold_mjy_sr": foreground_threshold,
            "irac_second_moment_cap_mjy_sr": irac_cap,
            "hi_background_method": hi_background_method,
            "hi_background_native": hi_background,
            "hi_noise_native": hi_noise,
            "hi_detection_threshold_native": hi_threshold,
            "finite_footprint_fraction_inside_hi_r995": footprint_fraction,
            "target_gas_mass_solar": gas_mass,
            "gas_mass_solar": gas_mass_actual,
            "gas_mass_relative_error": abs(gas_mass_actual - gas_mass) / gas_mass,
            "target_stellar_mass_solar": stellar_mass,
            "stellar_mass_solar": stellar_mass_actual,
            "stellar_mass_relative_error": abs(stellar_mass_actual - stellar_mass) / stellar_mass,
            "total_mass_solar": total_mass_actual,
            "total_mass_closure_relative_error": abs(total_mass_actual - gas_mass_actual - stellar_mass_actual)
            / total_mass_actual,
            "outer_cell_mass_fraction": edge_mass_fraction(total),
            "gas_centroid_x_kpc": gas_centroid[0],
            "gas_centroid_y_kpc": gas_centroid[1],
            "stellar_centroid_x_kpc": stellar_centroid[0],
            "stellar_centroid_y_kpc": stellar_centroid[1],
            "gas_star_centroid_offset_kpc": float(
                math.hypot(gas_centroid[0] - stellar_centroid[0], gas_centroid[1] - stellar_centroid[1])
            ),
            "hi_coordinate_roundtrip_arcsec": source_coordinate_roundtrip_arcsec(hi_wcs, center),
            "irac_coordinate_roundtrip_arcsec": source_coordinate_roundtrip_arcsec(irac_wcs, center),
            "map_sha256": map_sha,
            "validation_arrays_opened": 3 if split == "validation" else 0,
            "holdout_arrays_opened": 3 if split == "holdout" else 0,
            "velocity_arrays_opened": 0,
            "gravity_parameters": 0,
        }
        records.append(record)
        atlas.append({"galaxy": galaxy, "axis": axis, "gas": gas, "stars": stars, "total": total})
        print(
            f"{galaxy}: {cells}x{cells}, PA={position_angle_deg:.1f} deg, "
            f"coverage={100.0 * footprint_fraction:.1f}%"
        )

    frame = pd.DataFrame(records)
    frame.to_csv(args.output / "map_audit.csv", index=False)
    gates = config["engineeringGates"]
    checks = {
        "requiredSystems": len(frame) == int(gates["requiredSystems"]),
        "requiredValidationArraysOpened": int(frame.validation_arrays_opened.sum())
        == int(gates["requiredValidationArraysOpened"]),
        "requiredHoldoutArraysOpened": int(frame.holdout_arrays_opened.sum())
        == int(gates["requiredHoldoutArraysOpened"]),
        "requiredVelocityOrDispersionArraysOpened": int(frame.velocity_arrays_opened.sum())
        == int(gates["requiredVelocityOrDispersionArraysOpened"]),
        "maximumGravityParameters": int(frame.gravity_parameters.max())
        <= int(gates["maximumGravityParameters"]),
        "allMapsFiniteAndNonnegative": True,
        "maximumGasMassRelativeError": float(frame.gas_mass_relative_error.max())
        <= float(gates["maximumGasMassRelativeError"]),
        "maximumStellarMassRelativeError": float(frame.stellar_mass_relative_error.max())
        <= float(gates["maximumStellarMassRelativeError"]),
        "maximumTotalMassClosureRelativeError": float(frame.total_mass_closure_relative_error.max())
        <= float(gates["maximumTotalMassClosureRelativeError"]),
        "minimumFiniteFootprintFractionInsideHiR995": float(
            frame.finite_footprint_fraction_inside_hi_r995.min()
        )
        >= float(gates["minimumFiniteFootprintFractionInsideHiR995"]),
        "maximumOuterCellMassFraction": float(frame.outer_cell_mass_fraction.max())
        <= float(gates["maximumOuterCellMassFraction"]),
        "minimumCellsPerAxis": int(frame.cells_per_axis.min()) >= int(gates["minimumCellsPerAxis"]),
        "maximumCellsPerAxis": int(frame.cells_per_axis.max()) <= int(gates["maximumCellsPerAxis"]),
        "allCellsPerAxisOdd": bool((frame.cells_per_axis % 2 == 1).all()),
        "maximumSourceCoordinateRoundtripArcsec": float(
            max(frame.hi_coordinate_roundtrip_arcsec.max(), frame.irac_coordinate_roundtrip_arcsec.max())
        )
        <= float(gates["maximumSourceCoordinateRoundtripArcsec"]),
        "allOutputBundlesContentHashed": bool(frame.map_sha256.str.len().eq(64).all()),
    }
    for item in atlas:
        for key in ("gas", "stars", "total"):
            checks["allMapsFiniteAndNonnegative"] &= bool(
                np.isfinite(item[key]).all() and (item[key] >= 0.0).all()
            )
    status = "pass" if all(checks.values()) else "fail"

    fig, axes = plt.subplots(len(atlas), 3, figsize=(11, 3.2 * len(atlas)), constrained_layout=True)
    for row_index, item in enumerate(atlas):
        vmax = [np.quantile(item[key][item[key] > 0.0], 0.995) for key in ("gas", "stars", "total")]
        for column, (key, label) in enumerate((("gas", "Gas"), ("stars", "Stars"), ("total", "Total baryons"))):
            ax = axes[row_index, column]
            image = ax.imshow(
                np.log10(np.clip(item[key], vmax[column] * 1e-5, None)),
                origin="lower",
                extent=[item["axis"][0], item["axis"][-1], item["axis"][0], item["axis"][-1]],
                cmap="magma",
                vmin=math.log10(vmax[column]) - 5.0,
                vmax=math.log10(vmax[column]),
            )
            ax.set_title(f"{item['galaxy']} · {label}")
            ax.set_xlabel("x (kpc)")
            ax.set_ylabel("y (kpc)")
            fig.colorbar(image, ax=ax, fraction=0.046, label="log10 M_sun kpc^-2")
    atlas_name = config.get("atlasFileName", "development_baryonic_map_atlas.png")
    fig.savefig(args.output / atlas_name, dpi=180)
    plt.close(fig)

    report_core = {
        "schemaVersion": config.get(
            "resultSchemaVersion", "sigma-p0739-spiral-baryonic-registration-result/1"
        ),
        "stage": config.get("stage", "P0739"),
        "status": status,
        "configSha256": hashlib.sha256(config_bytes).hexdigest(),
        "parentManifestSha256": acquisition["manifestSha256"],
        "systems": len(frame),
        "splitsOpened": sorted(frame.split.unique().tolist()),
        "validationArraysOpened": int(frame.validation_arrays_opened.sum()),
        "holdoutArraysOpened": int(frame.holdout_arrays_opened.sum()),
        "velocityOrDispersionArraysOpened": int(frame.velocity_arrays_opened.sum()),
        "gravityParameters": int(frame.gravity_parameters.sum()),
        "universalBaryonicSettings": {
            "stellarMassToLightSolar": 0.5,
            "heliumFactor": 1.36,
        },
        "checks": checks,
        "aggregate": {
            "minimumFootprintFractionInsideHiR995": float(
                frame.finite_footprint_fraction_inside_hi_r995.min()
            ),
            "maximumGasMassRelativeError": float(frame.gas_mass_relative_error.max()),
            "maximumStellarMassRelativeError": float(frame.stellar_mass_relative_error.max()),
            "maximumOuterCellMassFraction": float(frame.outer_cell_mass_fraction.max()),
            "positionAngleRangeDeg": [
                float(frame.photometric_position_angle_deg.min()),
                float(frame.photometric_position_angle_deg.max()),
            ],
            "gasStarCentroidOffsetKpc": {
                "median": float(frame.gas_star_centroid_offset_kpc.median()),
                "maximum": float(frame.gas_star_centroid_offset_kpc.max()),
            },
        },
        "mapFiles": [
            {"galaxy": row.galaxy, "path": f"maps/{row.galaxy}.npz", "sha256": row.map_sha256}
            for row in frame.itertuples()
        ],
        "claimBoundary": config["claimBoundary"],
    }
    report = {**report_core, "reportSha256": canonical_sha256(report_core)}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    split_label = ", ".join(sorted(frame.split.unique().tolist()))
    summary = f"""# {config.get('stage', 'P0739')} velocity-blind spiral baryonic registration

Status: **{status.upper()}**

- Systems opened: {len(frame)} ({split_label})
- Validation image arrays opened: {int(frame.validation_arrays_opened.sum())}
- Holdout image arrays opened: {int(frame.holdout_arrays_opened.sum())}
- Velocity or dispersion arrays opened: {int(frame.velocity_arrays_opened.sum())}
- Gravity parameters: {int(frame.gravity_parameters.sum())}
- Minimum joint footprint inside the published H I radius: {100.0 * frame.finite_footprint_fraction_inside_hi_r995.min():.2f}%
- Maximum gas-mass normalization error: {frame.gas_mass_relative_error.max():.3e}
- Maximum stellar-mass normalization error: {frame.stellar_mass_relative_error.max():.3e}
- Maximum outer-cell mass fraction: {100.0 * frame.outer_cell_mass_fraction.max():.4f}%
- Report SHA-256: `{report['reportSha256']}`

This is a registration and mass-closure result, not a gravity-formula score.
Every kinematic target remains sealed. Any split not named above remains sealed.
"""
    (args.output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(json.dumps({"status": status, "checks": checks, "reportSha256": report["reportSha256"]}))
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
