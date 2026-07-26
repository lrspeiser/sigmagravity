from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, Supergalactic
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator

from voidscreen.data import parse_table1

CF4_H0_KMS_MPC = 74.6
CF4_H100 = CF4_H0_KMS_MPC / 100.0
MPC_M = 3.085677581491367e22


@dataclass(frozen=True)
class DensityGridSpec:
    key: str
    filename: str
    box_size_hmpc: float
    shape: tuple[int, int, int]

    @property
    def voxel_size_hmpc(self) -> float:
        return self.box_size_hmpc / self.shape[0]


GRID_SPECS = (
    DensityGridSpec(
        key="grouped_64",
        filename="CF4gp_new_64-z008_delta.fits",
        box_size_hmpc=500.0,
        shape=(64, 64, 64),
    ),
    DensityGridSpec(
        key="ungrouped_64",
        filename="CF4_new_64-z008_delta.fits",
        box_size_hmpc=500.0,
        shape=(64, 64, 64),
    ),
    DensityGridSpec(
        key="ungrouped_128",
        filename="CF4_new_128-z008_delta.fits",
        box_size_hmpc=1000.0,
        shape=(128, 128, 128),
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_density_grid(cf4_dir: Path, spec: DensityGridSpec) -> np.ndarray:
    path = Path(cf4_dir) / spec.filename
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/download_cosmicflows4.ps1 first.")
    grid = np.asarray(fits.getdata(path), dtype=np.float64)
    if grid.shape != spec.shape:
        raise ValueError(f"Expected {spec.shape} in {path}, found {grid.shape}")
    if not np.isfinite(grid).all():
        raise ValueError(f"Non-finite density values in {path}")
    return grid


def supergalactic_cartesian_hmpc(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    distance_mpc: np.ndarray,
    *,
    h100: float = CF4_H100,
) -> np.ndarray:
    """Convert ICRS positions to observer-centered SGX/SGY/SGZ in h100^-1 Mpc."""
    sky = SkyCoord(
        ra=np.asarray(ra_deg, dtype=float) * u.deg,
        dec=np.asarray(dec_deg, dtype=float) * u.deg,
        distance=np.asarray(distance_mpc, dtype=float) * u.Mpc,
        frame="icrs",
    )
    supergalactic = sky.transform_to(Supergalactic())
    physical_mpc = np.column_stack(
        [
            supergalactic.cartesian.x.to_value(u.Mpc),
            supergalactic.cartesian.y.to_value(u.Mpc),
            supergalactic.cartesian.z.to_value(u.Mpc),
        ]
    )
    return physical_mpc * h100


def sample_density_grid(
    grid: np.ndarray,
    points_hmpc: np.ndarray,
    *,
    box_size_hmpc: float,
) -> np.ndarray:
    """Trilinearly sample a centered CF4 grid whose NumPy axes are SGX, SGY, SGZ."""
    if grid.ndim != 3 or len(set(grid.shape)) != 1:
        raise ValueError(f"Expected a cubic 3-D grid, found {grid.shape}")
    points = np.asarray(points_hmpc, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_hmpc must have shape (n, 3)")
    voxel_size = box_size_hmpc / grid.shape[0]
    centers = -box_size_hmpc / 2.0 + (np.arange(grid.shape[0]) + 0.5) * voxel_size
    interpolator = RegularGridInterpolator(
        (centers, centers, centers),
        grid,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    return np.asarray(interpolator(points), dtype=np.float64)


def density_acceleration_and_tidal_fields(
    density_contrast: np.ndarray,
    *,
    box_size_hmpc: float,
    h0_km_s_mpc: float = CF4_H0_KMS_MPC,
    omega_m: float = 0.3,
    padding_factor: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the present-day linear peculiar field from a density-contrast cube.

    The zero-padded FFT solves ``div(g) = -3/2 Omega_m H0^2 delta`` with the
    mean mode removed. Returned acceleration has units m/s^2 and the symmetric
    tidal tensor has units s^-2. The CF4 coordinates are h100^-1 Mpc, so the
    physical FFT spacing divides those coordinates by h100.
    """
    delta = np.asarray(density_contrast, dtype=np.float64)
    if delta.ndim != 3 or len(set(delta.shape)) != 1:
        raise ValueError(f"Expected a cubic 3-D density grid, found {delta.shape}")
    if not np.isfinite(delta).all():
        raise ValueError("Density grid contains non-finite values")
    if padding_factor < 1:
        raise ValueError("padding_factor must be at least one")
    if not 0.0 < omega_m <= 1.0:
        raise ValueError("omega_m must lie in (0, 1]")

    side = delta.shape[0]
    padded_side = side * padding_factor
    start = (padded_side - side) // 2
    stop = start + side
    padded = np.zeros((padded_side,) * 3, dtype=np.float64)
    padded[start:stop, start:stop, start:stop] = delta - float(delta.mean())

    h100 = h0_km_s_mpc / 100.0
    physical_voxel_m = box_size_hmpc / side / h100 * MPC_M
    frequencies = 2.0 * np.pi * np.fft.fftfreq(padded_side, d=physical_voxel_m)
    kx, ky, kz = np.meshgrid(frequencies, frequencies, frequencies, indexing="ij")
    wavevectors = (kx, ky, kz)
    k_squared = kx**2 + ky**2 + kz**2
    nonzero = k_squared > 0.0
    inverse_k_squared = np.zeros_like(k_squared)
    inverse_k_squared[nonzero] = 1.0 / k_squared[nonzero]

    h0_s = h0_km_s_mpc * 1000.0 / MPC_M
    poisson_coefficient = 1.5 * omega_m * h0_s**2
    delta_fourier = np.fft.fftn(padded)
    center = (slice(start, stop),) * 3

    acceleration = np.empty((3, side, side, side), dtype=np.float64)
    for axis, wavevector in enumerate(wavevectors):
        field_fourier = 1j * poisson_coefficient * wavevector * inverse_k_squared * delta_fourier
        acceleration[axis] = np.fft.ifftn(field_fourier).real[center]

    tidal = np.empty((3, 3, side, side, side), dtype=np.float64)
    for first, first_wavevector in enumerate(wavevectors):
        for second in range(first, 3):
            field_fourier = (
                -poisson_coefficient
                * first_wavevector
                * wavevectors[second]
                * inverse_k_squared
                * delta_fourier
            )
            values = np.fft.ifftn(field_fourier).real[center]
            tidal[first, second] = values
            tidal[second, first] = values
    return acceleration, tidal


def build_cf4_environment_table(sparc_dir: Path, cf4_dir: Path) -> pd.DataFrame:
    sparc_dir = Path(sparc_dir)
    coordinates = pd.read_csv(sparc_dir / "coordinates.csv")
    required = {"name", "ra_deg", "dec_deg"}
    if not required.issubset(coordinates.columns):
        raise ValueError(f"SPARC coordinates must contain {sorted(required)}")
    if coordinates["name"].duplicated().any():
        raise ValueError("SPARC coordinates contain duplicate galaxy names")

    metadata = parse_table1(sparc_dir / "table1.dat")
    if set(coordinates["name"]) != set(metadata):
        raise ValueError("SPARC coordinate names do not exactly match the metadata table")
    coordinates["distance_mpc"] = coordinates["name"].map(
        {name: row.distance_mpc for name, row in metadata.items()}
    )
    points = supergalactic_cartesian_hmpc(
        coordinates["ra_deg"].to_numpy(),
        coordinates["dec_deg"].to_numpy(),
        coordinates["distance_mpc"].to_numpy(),
    )

    output = pd.DataFrame(
        {
            "galaxy": coordinates["name"].astype(str),
            "ra_deg": coordinates["ra_deg"].astype(float),
            "dec_deg": coordinates["dec_deg"].astype(float),
            "distance_mpc": coordinates["distance_mpc"].astype(float),
            "sgx_hmpc": points[:, 0],
            "sgy_hmpc": points[:, 1],
            "sgz_hmpc": points[:, 2],
        }
    )
    for spec in GRID_SPECS:
        grid = load_density_grid(cf4_dir, spec)
        values = sample_density_grid(grid, points, box_size_hmpc=spec.box_size_hmpc)
        if not np.isfinite(values).all():
            missing = output.loc[~np.isfinite(values), "galaxy"].tolist()
            raise ValueError(f"Galaxies outside {spec.key} interpolation volume: {missing}")
        output[f"delta_{spec.key}"] = values
        output[f"void_score_{spec.key}"] = -values

    # The grouped reconstruction is primary. Larger scores mean lower reconstructed density.
    output.insert(1, "void_score", output["void_score_grouped_64"])
    return output.sort_values("galaxy", kind="stable").reset_index(drop=True)


def validate_catalog_coordinates(catalog_path: Path) -> dict[str, float | int]:
    """Check the CF4 fixed-width coordinate columns against Astropy's frame transform."""
    rows: list[tuple[float, ...]] = []
    with gzip.open(catalog_path, "rt", encoding="ascii") as handle:
        for line in handle:
            try:
                rows.append(
                    (
                        float(line[83:91]),
                        float(line[92:100]),
                        float(line[119:127]),
                        float(line[128:136]),
                        float(line[137:143]),
                        float(line[144:150]),
                        float(line[151:157]),
                    )
                )
            except ValueError as exc:
                raise ValueError("Malformed coordinate row in CF4 group catalog") from exc
    values = np.asarray(rows, dtype=np.float64)
    sky = SkyCoord(ra=values[:, 0] * u.deg, dec=values[:, 1] * u.deg, frame="icrs")
    transformed = sky.transform_to(Supergalactic())
    longitude_residual = (transformed.sgl.to_value(u.deg) - values[:, 2] + 180.0) % 360.0 - 180.0
    latitude_residual = transformed.sgb.to_value(u.deg) - values[:, 3]

    xyz = values[:, 4:7]
    # Very local catalog velocities can be negative and reverse the rounded XYZ vector.
    # Above 200 km/s, the tabulated Cartesian direction is stable enough for this check.
    valid_xyz = np.linalg.norm(xyz, axis=1) > 200.0
    observed_unit = xyz[valid_xyz] / np.linalg.norm(xyz[valid_xyz], axis=1, keepdims=True)
    sgl = np.deg2rad(values[valid_xyz, 2])
    sgb = np.deg2rad(values[valid_xyz, 3])
    catalog_angle_unit = np.column_stack(
        [np.cos(sgb) * np.cos(sgl), np.cos(sgb) * np.sin(sgl), np.sin(sgb)]
    )
    separation = np.rad2deg(
        np.arccos(np.clip(np.sum(observed_unit * catalog_angle_unit, axis=1), -1.0, 1.0))
    )
    return {
        "rows": int(values.shape[0]),
        "max_astropy_sgl_residual_deg": float(np.max(np.abs(longitude_residual))),
        "max_astropy_sgb_residual_deg": float(np.max(np.abs(latitude_residual))),
        "xyz_direction_rows_above_200_km_s": int(valid_xyz.sum()),
        "median_xyz_direction_residual_deg": float(np.median(separation)),
        "max_xyz_direction_residual_deg": float(np.max(separation)),
    }


def write_environment_products(
    table: pd.DataFrame,
    output_csv: Path,
    report_json: Path,
    *,
    cf4_dir: Path,
) -> None:
    output_csv = Path(output_csv)
    report_json = Path(report_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_csv, index=False, float_format="%.10g")

    score_columns = [column for column in table if column.startswith("void_score_")]
    report = {
        "status": "frozen independent environment table",
        "rows": len(table),
        "primary_score": "void_score = -delta_grouped_64",
        "h0_km_s_mpc": CF4_H0_KMS_MPC,
        "coordinate_frame": "observer-centered ICRS to Supergalactic Cartesian",
        "coordinate_units": "h100^-1 Mpc",
        "interpolation": "trilinear at voxel centers",
        "fits_convention": {
            "published_fits_axes": ["SGZ", "SGY", "SGX"],
            "numpy_array_axes": ["SGX", "SGY", "SGZ"],
            "source": "https://projets.ip2i.in2p3.fr/cosmicflows/",
        },
        "guardrail": "No SPARC velocity, residual, or fitted model quantity was used.",
        "grid_files": [
            {
                "key": spec.key,
                "path": spec.filename,
                "shape": list(spec.shape),
                "box_size_hmpc": spec.box_size_hmpc,
                "voxel_size_hmpc": spec.voxel_size_hmpc,
                "sha256": _sha256(Path(cf4_dir) / spec.filename),
            }
            for spec in GRID_SPECS
        ],
        "score_summary": {
            column: {
                "minimum": float(table[column].min()),
                "maximum": float(table[column].max()),
                "mean": float(table[column].mean()),
                "standard_deviation": float(table[column].std(ddof=0)),
            }
            for column in score_columns
        },
        "spearman_score_correlations": table[score_columns].corr(method="spearman").to_dict(),
        "catalog_coordinate_validation": validate_catalog_coordinates(
            Path(cf4_dir) / "CF4_table4_groups.dat.gz"
        ),
    }
    report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
