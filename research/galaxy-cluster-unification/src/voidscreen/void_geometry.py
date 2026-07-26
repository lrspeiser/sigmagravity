from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

from voidscreen.data import parse_table1

LOCAL_VOID_H = 0.681
LOCAL_VOID_BOX_CENTER_HMPC = np.asarray([340.5, 340.5, 340.5], dtype=np.float64)
LOCAL_VOID_OVERLAP_THRESHOLD = 0.37


def icrs_to_local_void_box_hmpc(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    distance_mpc: np.ndarray,
    *,
    h: float = LOCAL_VOID_H,
) -> np.ndarray:
    """Map observer-centered ICRS positions into the catalog's Cartesian box."""
    ra = np.deg2rad(np.asarray(ra_deg, dtype=np.float64))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))
    radius = np.asarray(distance_mpc, dtype=np.float64) * h
    displacement = np.column_stack(
        (
            radius * np.cos(dec) * np.cos(ra),
            radius * np.cos(dec) * np.sin(ra),
            radius * np.sin(dec),
        )
    )
    return displacement + LOCAL_VOID_BOX_CENTER_HMPC


def _cloud_grid(path: Path) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    cloud = np.load(path)
    if cloud.shape != (32**3, 4):
        raise ValueError(f"Expected a (32768, 4) Voronoi cloud in {path}, found {cloud.shape}")
    grid = np.asarray(cloud, dtype=np.float64).reshape(32, 32, 32, 4)
    # Published flattening order reshapes as (y, x, z); transpose the scalar
    # field so all public functions consistently accept Cartesian (x, y, z).
    axes = (
        grid[0, :, 0, 0],
        grid[:, 0, 0, 1],
        grid[0, 0, :, 2],
    )
    for axis in axes:
        spacing = np.diff(axis)
        if not np.all(spacing > 0.0) or not np.allclose(spacing, spacing[0], rtol=1e-10):
            raise ValueError(f"Non-regular Voronoi-cloud coordinates in {path}")
    overlap = np.transpose(grid[:, :, :, 3], (1, 0, 2))
    if np.min(overlap) < 0.0 or np.max(overlap) > 1.0:
        raise ValueError(f"Voronoi overlap lies outside [0, 1] in {path}")
    return axes, overlap


def sample_cloud_membership_and_wall_distance(
    path: Path,
    points_hmpc: np.ndarray,
    *,
    threshold: float = LOCAL_VOID_OVERLAP_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample cloud overlap and half-voxel-corrected interior wall distance."""
    axes, overlap = _cloud_grid(path)
    points = np.asarray(points_hmpc, dtype=np.float64)
    overlap_interpolator = RegularGridInterpolator(
        axes, overlap, bounds_error=False, fill_value=0.0
    )
    sampled_overlap = np.asarray(overlap_interpolator(points), dtype=np.float64)

    inside_grid = overlap > threshold
    spacing = tuple(float(axis[1] - axis[0]) for axis in axes)
    distance_at_centers = distance_transform_edt(inside_grid, sampling=spacing)
    half_voxel = 0.5 * min(spacing)
    corrected_distance = np.maximum(distance_at_centers - half_voxel, 0.0)
    distance_interpolator = RegularGridInterpolator(
        axes, corrected_distance, bounds_error=False, fill_value=0.0
    )
    sampled_distance = np.asarray(distance_interpolator(points), dtype=np.float64)
    sampled_distance[sampled_overlap <= threshold] = 0.0
    return sampled_overlap, sampled_distance


def build_local_void_wall_table(sparc_dir: Path, catalog_dir: Path) -> pd.DataFrame:
    """Create the frozen W1 score without using any rotation-curve velocities."""
    sparc_dir = Path(sparc_dir)
    catalog_dir = Path(catalog_dir)
    coordinates = pd.read_csv(sparc_dir / "coordinates.csv")
    metadata = parse_table1(sparc_dir / "table1.dat")
    if set(coordinates["name"]) != set(metadata):
        raise ValueError("SPARC coordinate names do not exactly match the metadata table")
    coordinates["distance_mpc"] = coordinates["name"].map(
        {name: row.distance_mpc for name, row in metadata.items()}
    )
    points = icrs_to_local_void_box_hmpc(
        coordinates["ra_deg"].to_numpy(),
        coordinates["dec_deg"].to_numpy(),
        coordinates["distance_mpc"].to_numpy(),
    )

    catalog = pd.read_csv(catalog_dir / "voids_catalog.csv")
    if len(catalog) != 100:
        raise ValueError(f"Expected 100 Local Voids rows, found {len(catalog)}")
    best_overlap = np.zeros(len(coordinates), dtype=np.float64)
    best_wall_distance = np.zeros(len(coordinates), dtype=np.float64)
    best_void_index = np.full(len(coordinates), -1, dtype=np.int64)
    for void_index in range(len(catalog)):
        path = catalog_dir / "VoronoiClouds" / f"Voronoi_cloud_void_{void_index}_N32.npy"
        overlap, wall_distance = sample_cloud_membership_and_wall_distance(path, points)
        replace = (overlap > LOCAL_VOID_OVERLAP_THRESHOLD) & (overlap > best_overlap)
        best_overlap[replace] = overlap[replace]
        best_wall_distance[replace] = wall_distance[replace]
        best_void_index[replace] = void_index

    assigned = best_void_index >= 0
    effective_radius = np.full(len(coordinates), np.nan, dtype=np.float64)
    center_distance = np.full(len(coordinates), np.nan, dtype=np.float64)
    if np.any(assigned):
        assigned_catalog = catalog.iloc[best_void_index[assigned]]
        effective_radius[assigned] = assigned_catalog["mean radius (Mpc/h)"].to_numpy()
        centers = assigned_catalog[
            ["center x (Mpc/h)", "center y (Mpc/h)", "center z (Mpc/h)"]
        ].to_numpy(dtype=np.float64)
        center_distance[assigned] = np.linalg.norm(points[assigned] - centers, axis=1)
    wall_score = np.zeros(len(coordinates), dtype=np.float64)
    wall_score[assigned] = np.minimum(
        best_wall_distance[assigned] / effective_radius[assigned], 1.0
    )

    return (
        pd.DataFrame(
            {
                "galaxy": coordinates["name"].astype(str),
                "void_score": wall_score,
                "void_wall_score": wall_score,
                "inside_catalog_void": assigned,
                "void_index": best_void_index,
                "voronoi_overlap": best_overlap,
                "wall_distance_hmpc": best_wall_distance,
                "void_effective_radius_hmpc": effective_radius,
                "center_distance_hmpc": center_distance,
                "center_distance_over_radius": center_distance / effective_radius,
                "catalog_x_hmpc": points[:, 0],
                "catalog_y_hmpc": points[:, 1],
                "catalog_z_hmpc": points[:, 2],
            }
        )
        .sort_values("galaxy", kind="stable")
        .reset_index(drop=True)
    )


def write_local_void_products(
    table: pd.DataFrame,
    output_csv: Path,
    report_json: Path,
    *,
    catalog_dir: Path,
) -> None:
    output_csv = Path(output_csv)
    report_json = Path(report_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_csv, index=False, float_format="%.10g")
    provenance_path = Path(catalog_dir) / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8-sig"))
    inside = table["inside_catalog_void"].astype(bool)
    report = {
        "status": "frozen external void-wall environment table",
        "rows": len(table),
        "primary_score": "void_score = void_wall_score",
        "guardrail": "Uses only SPARC sky position/distance and the external Local Voids geometry; no rotation velocity or residual.",
        "catalog": {
            "paper": provenance["paper"],
            "repository": provenance["repository"],
            "commit": provenance["commit"],
            "provenance_sha256": hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
        },
        "geometry": {
            "coordinate_h": LOCAL_VOID_H,
            "box_center_hmpc": LOCAL_VOID_BOX_CENTER_HMPC.tolist(),
            "inside_rule": f"trilinear Voronoi overlap > {LOCAL_VOID_OVERLAP_THRESHOLD}",
            "overlap_assignment": "largest overlap; exact tie retains lower void index",
            "wall_distance": "Euclidean distance transform minus half voxel, trilinearly sampled, clipped to zero inside",
            "normalization": "published mean effective radius",
        },
        "coverage": {
            "inside_count": int(inside.sum()),
            "outside_count": int((~inside).sum()),
            "inside_fraction": float(inside.mean()),
            "unique_assigned_voids": int(table.loc[inside, "void_index"].nunique()),
        },
        "score_summary": {
            "minimum": float(table["void_wall_score"].min()),
            "median_all": float(table["void_wall_score"].median()),
            "median_inside": (
                float(table.loc[inside, "void_wall_score"].median()) if inside.any() else None
            ),
            "maximum": float(table["void_wall_score"].max()),
        },
    }
    report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
