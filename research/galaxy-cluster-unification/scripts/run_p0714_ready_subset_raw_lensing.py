#!/usr/bin/env python3
"""Run descriptive raw-lensing diagnostics on the P0713-ready subset.

P0713 found only two of four preregistered clusters ready, so this program is
not a P0633 validation score.  It preserves the frozen P0708 prediction as
written and separately reports a post-unseal coordinate-axis repair.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
READINESS = ROOT / "results/p0713_external_cluster_readiness_audit"
PREDICTIONS = ROOT / "results/p0708_external_prediction_lock/clusters"
BARYON_MAPS = ROOT / "results/p0641_registered_cluster_baryon_maps/maps"
COMPARATORS = ROOT / "data/raw/p0633_relics_lensing_comparators"
OUTPUT = ROOT / "results/p0714_ready_subset_raw_lensing"
SALT = "P0633-RAW-LENS-FAMILY-V1"
MODELS = [
    "P0707_Weyl_frozen_axis_contract",
    "P0707_Weyl_axis_repaired_exploratory",
    "baryon_only_GR",
    "AQUAL_simple_mu_diagnostic",
    "QUMOND_simple_nu_diagnostic",
    "glafic_v2_compact_halo",
]


def distance_ratio(lens_redshift: float, source_redshift: float) -> float:
    if source_redshift <= lens_redshift:
        raise ValueError("source must be behind the lens")
    return float(
        Planck18.angular_diameter_distance_z1z2(lens_redshift, source_redshift)
        / Planck18.angular_diameter_distance(source_redshift)
    )


def one(directory: Path, pattern: str) -> Path:
    matches = list(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern} in {directory}, found {len(matches)}")
    return matches[0]


@dataclass
class DeflectionField:
    cluster: str
    lens_redshift: float
    half_extent_arcsec: float

    def alpha(self, east_arcsec, north_arcsec, source_redshift: float) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def beta(self, east_arcsec, north_arcsec, source_redshift: float) -> tuple[np.ndarray, np.ndarray]:
        east = np.asarray(east_arcsec, dtype=float)
        north = np.asarray(north_arcsec, dtype=float)
        alpha_east, alpha_north = self.alpha(east, north, source_redshift)
        return east - alpha_east, north - alpha_north

    def jacobian(self, east_arcsec, north_arcsec, source_redshift: float, step: float = 0.08) -> np.ndarray:
        east = np.atleast_1d(np.asarray(east_arcsec, dtype=float))
        north = np.atleast_1d(np.asarray(north_arcsec, dtype=float))
        bxp, byp = self.beta(east + step, north, source_redshift)
        bxm, bym = self.beta(east - step, north, source_redshift)
        cxp, cyp = self.beta(east, north + step, source_redshift)
        cxm, cym = self.beta(east, north - step, source_redshift)
        matrices = np.empty((len(east), 2, 2), dtype=float)
        matrices[:, 0, 0] = (bxp - bxm) / (2.0 * step)
        matrices[:, 1, 0] = (byp - bym) / (2.0 * step)
        matrices[:, 0, 1] = (cxp - cxm) / (2.0 * step)
        matrices[:, 1, 1] = (cyp - cym) / (2.0 * step)
        return matrices


class FrozenGridField(DeflectionField):
    def __init__(self, cluster: str, lens_redshift: float, component: str, *, axis_repaired: bool):
        path = PREDICTIONS / f"{cluster}_physical_deflections.npz"
        with np.load(path) as data:
            axis_kpc = data["axis_kpc"].astype(float)
            alpha_0 = data[f"alpha_x_{component}_arcsec"].astype(float)
            alpha_1 = data[f"alpha_y_{component}_arcsec"].astype(float)
        self.axis_repaired = axis_repaired
        self.kpc_per_arcsec = float(Planck18.kpc_proper_per_arcmin(lens_redshift).value / 60.0)
        self.interp_0 = RegularGridInterpolator((axis_kpc, axis_kpc), alpha_0, bounds_error=False, fill_value=np.nan)
        self.interp_1 = RegularGridInterpolator((axis_kpc, axis_kpc), alpha_1, bounds_error=False, fill_value=np.nan)
        super().__init__(cluster, lens_redshift, float(np.max(np.abs(axis_kpc))) / self.kpc_per_arcsec)

    def alpha(self, east_arcsec, north_arcsec, source_redshift: float) -> tuple[np.ndarray, np.ndarray]:
        east = np.asarray(east_arcsec, dtype=float)
        north = np.asarray(north_arcsec, dtype=float)
        if self.axis_repaired:
            points = np.column_stack([(north * self.kpc_per_arcsec).ravel(), (east * self.kpc_per_arcsec).ravel()])
            alpha_east = self.interp_1(points).reshape(east.shape)
            alpha_north = self.interp_0(points).reshape(east.shape)
        else:
            points = np.column_stack([(east * self.kpc_per_arcsec).ravel(), (north * self.kpc_per_arcsec).ravel()])
            alpha_east = self.interp_0(points).reshape(east.shape)
            alpha_north = self.interp_1(points).reshape(east.shape)
        scale = distance_ratio(self.lens_redshift, source_redshift)
        return scale * alpha_east, scale * alpha_north


class GlaficField(DeflectionField):
    def __init__(self, cluster: str, lens_redshift: float, baryon_center: SkyCoord):
        directory = COMPARATORS / cluster / "glafic" / "v2"
        x_path = one(directory, "*_x-arcsec-deflect.fits")
        y_path = one(directory, "*_y-arcsec-deflect.fits")
        with fits.open(x_path, memmap=True) as handle:
            x_map = np.asarray(handle[0].data, dtype=float)
            header = handle[0].header
        with fits.open(y_path, memmap=True) as handle:
            y_map = np.asarray(handle[0].data, dtype=float)
        pixel_arcsec = abs(float(header["CDELT1"])) * 3600.0
        east_axis = -(np.arange(x_map.shape[1]) - (float(header["CRPIX1"]) - 1.0)) * pixel_arcsec
        north_axis = (np.arange(x_map.shape[0]) - (float(header["CRPIX2"]) - 1.0)) * pixel_arcsec
        reference = SkyCoord(float(header["CRVAL1"]) * u.deg, float(header["CRVAL2"]) * u.deg)
        center_east, center_north = reference.spherical_offsets_to(baryon_center)
        self.center_east_from_reference = center_east.to_value(u.arcsec)
        self.center_north_from_reference = center_north.to_value(u.arcsec)
        # GLAFIC image x increases west because CDELT1 is negative; convert to east.
        self.interp_east = RegularGridInterpolator((north_axis, east_axis[::-1]), (-x_map)[:, ::-1], bounds_error=False, fill_value=np.nan)
        self.interp_north = RegularGridInterpolator((north_axis, east_axis[::-1]), y_map[:, ::-1], bounds_error=False, fill_value=np.nan)
        half = min(float(np.max(np.abs(east_axis))), float(np.max(np.abs(north_axis))))
        super().__init__(cluster, lens_redshift, half)

    def alpha(self, east_arcsec, north_arcsec, source_redshift: float) -> tuple[np.ndarray, np.ndarray]:
        east = np.asarray(east_arcsec, dtype=float)
        north = np.asarray(north_arcsec, dtype=float)
        points = np.column_stack(
            [
                (north + self.center_north_from_reference).ravel(),
                (east + self.center_east_from_reference).ravel(),
            ]
        )
        scale = distance_ratio(self.lens_redshift, source_redshift)
        return (
            scale * self.interp_east(points).reshape(east.shape),
            scale * self.interp_north(points).reshape(east.shape),
        )


def profiled_source(field: DeflectionField, images: pd.DataFrame, source_redshift: float) -> np.ndarray:
    beta_east, beta_north = field.beta(images.east_arcsec.to_numpy(), images.north_arcsec.to_numpy(), source_redshift)
    return np.asarray([np.mean(beta_east), np.mean(beta_north)], dtype=float)


def solve_one_root(field: DeflectionField, source: np.ndarray, source_redshift: float, start: np.ndarray, bound: float) -> tuple[np.ndarray | None, float]:
    def equation(theta: np.ndarray) -> np.ndarray:
        beta_east, beta_north = field.beta(np.asarray([theta[0]]), np.asarray([theta[1]]), source_redshift)
        return np.asarray([beta_east[0] - source[0], beta_north[0] - source[1]])

    start = np.clip(np.asarray(start, dtype=float), -bound, bound)
    result = least_squares(
        equation,
        start,
        bounds=([-bound, -bound], [bound, bound]),
        max_nfev=200,
        ftol=1.0e-11,
        xtol=1.0e-11,
        gtol=1.0e-11,
    )
    closure = float(np.linalg.norm(equation(result.x)))
    if not result.success or not np.all(np.isfinite(result.x)) or closure > 2.0e-3:
        return None, closure
    return np.asarray(result.x, dtype=float), closure


def deduplicate_roots(roots: list[np.ndarray], tolerance: float = 0.20) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for candidate in roots:
        if not unique or min(float(np.linalg.norm(candidate - root)) for root in unique) > tolerance:
            unique.append(candidate)
    return unique


def find_roots(field: DeflectionField, source: np.ndarray, source_redshift: float, images: pd.DataFrame, bound: float) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(-bound, bound, 161)
    east, north = np.meshgrid(grid, grid, indexing="xy")
    beta_east, beta_north = field.beta(east, north, source_redshift)
    first = beta_east - source[0]
    second = beta_north - source[1]
    crossing = (
        (np.minimum.reduce([first[:-1, :-1], first[1:, :-1], first[:-1, 1:], first[1:, 1:]]) <= 0.0)
        & (np.maximum.reduce([first[:-1, :-1], first[1:, :-1], first[:-1, 1:], first[1:, 1:]]) >= 0.0)
        & (np.minimum.reduce([second[:-1, :-1], second[1:, :-1], second[:-1, 1:], second[1:, 1:]]) <= 0.0)
        & (np.maximum.reduce([second[:-1, :-1], second[1:, :-1], second[:-1, 1:], second[1:, 1:]]) >= 0.0)
    )
    iy, ix = np.nonzero(crossing)
    starts = [np.asarray([0.5 * (grid[x] + grid[x + 1]), 0.5 * (grid[y] + grid[y + 1])]) for y, x in zip(iy, ix, strict=True)]
    starts.extend(images[["east_arcsec", "north_arcsec"]].to_numpy(float))
    roots: list[np.ndarray] = []
    closures: list[float] = []
    for start in starts:
        root_value, closure = solve_one_root(field, source, source_redshift, start, bound)
        if root_value is not None:
            roots.append(root_value)
            closures.append(closure)
    unique = deduplicate_roots(roots)
    if not unique:
        return np.empty((0, 2)), np.empty(0)
    root_array = np.asarray(unique)
    jacobian = field.jacobian(root_array[:, 0], root_array[:, 1], source_redshift)
    determinant = np.linalg.det(jacobian)
    magnification = 1.0 / np.maximum(np.abs(determinant), 1.0e-12)
    return root_array, magnification


def assignment(images: pd.DataFrame, roots: np.ndarray) -> tuple[np.ndarray, float, int]:
    observed = images[["east_arcsec", "north_arcsec"]].to_numpy(float)
    if len(roots) == 0:
        return np.empty((0, 2), dtype=int), float("inf"), 0
    cost = np.linalg.norm(observed[:, None, :] - roots[None, :, :], axis=2)
    observed_index, root_index = linear_sum_assignment(cost)
    pairs = np.column_stack([observed_index, root_index])
    matched = len(pairs)
    rms = float(np.sqrt(np.mean(np.square(cost[observed_index, root_index])))) if matched == len(observed) else float("inf")
    return pairs, rms, matched


def curve_points(field: DeflectionField, source_redshift: float, bound: float) -> np.ndarray:
    grid = np.linspace(-bound, bound, 161)
    east, north = np.meshgrid(grid, grid, indexing="xy")
    step = float(grid[1] - grid[0])
    beta_east, beta_north = field.beta(east, north, source_redshift)
    dbx_dn, dbx_de = np.gradient(beta_east, step, step, edge_order=2)
    dby_dn, dby_de = np.gradient(beta_north, step, step, edge_order=2)
    determinant = dbx_de * dby_dn - dbx_dn * dby_de
    sign_change = (
        np.minimum.reduce([determinant[:-1, :-1], determinant[1:, :-1], determinant[:-1, 1:], determinant[1:, 1:]]) <= 0.0
    ) & (
        np.maximum.reduce([determinant[:-1, :-1], determinant[1:, :-1], determinant[:-1, 1:], determinant[1:, 1:]]) >= 0.0
    )
    iy, ix = np.nonzero(sign_change)
    return np.column_stack([0.5 * (grid[ix] + grid[ix + 1]), 0.5 * (grid[iy] + grid[iy + 1])])


def symmetric_p95_distance(first: np.ndarray, second: np.ndarray) -> float | None:
    if len(first) == 0 or len(second) == 0:
        return None
    left = cKDTree(first).query(second)[0]
    right = cKDTree(second).query(first)[0]
    return float(max(np.quantile(left, 0.95), np.quantile(right, 0.95)))


def family_partition(cluster: str, family_ids: list[str]) -> dict[str, str]:
    ranked = sorted(family_ids, key=lambda family: hashlib.sha256(f"{SALT}|{cluster}|{family}".encode()).hexdigest())
    holdout_count = max(1, math.ceil(0.30 * len(ranked)))
    return {family: ("holdout" if family in ranked[:holdout_count] else "fit") for family in family_ids}


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def main() -> None:
    readiness = json.loads((READINESS / "report.json").read_text(encoding="utf-8"))
    if readiness["status"] != "fail_data_readiness" or readiness["ready_clusters"] != 2:
        raise RuntimeError("P0714 expects the frozen P0713 two-cluster ready subset")
    ready = [row["cluster"] for row in readiness["cluster_rows"] if row["ready"]]
    catalog = pd.read_csv(READINESS / "parsed_image_catalog.csv")
    catalog = catalog[catalog.secure_image.astype(str).str.lower().eq("true") & catalog.cluster.isin(ready)].copy()

    family_records: list[dict[str, object]] = []
    root_records: list[dict[str, object]] = []
    curve_records: list[dict[str, object]] = []
    cluster_model_records: list[dict[str, object]] = []

    for cluster in ready:
        with np.load(BARYON_MAPS / f"{cluster}_baryons.npz") as data:
            center = SkyCoord(float(data["center_ra_deg"]) * u.deg, float(data["center_dec_deg"]) * u.deg)
            lens_redshift = float(data["redshift"])
        block = catalog[catalog.cluster == cluster].copy()
        sky = SkyCoord(block.ra_deg.to_numpy() * u.deg, block.dec_deg.to_numpy() * u.deg)
        east, north = center.spherical_offsets_to(sky)
        block["east_arcsec"] = east.to_value(u.arcsec)
        block["north_arcsec"] = north.to_value(u.arcsec)
        partition = family_partition(cluster, sorted(block.family_id.astype(str).unique()))
        fields: dict[str, DeflectionField] = {
            "P0707_Weyl_frozen_axis_contract": FrozenGridField(cluster, lens_redshift, "P0707_Weyl", axis_repaired=False),
            "P0707_Weyl_axis_repaired_exploratory": FrozenGridField(cluster, lens_redshift, "P0707_Weyl", axis_repaired=True),
            "baryon_only_GR": FrozenGridField(cluster, lens_redshift, "baryon_only_GR", axis_repaired=False),
            "AQUAL_simple_mu_diagnostic": FrozenGridField(cluster, lens_redshift, "AQUAL_simple_mu_diagnostic", axis_repaired=False),
            "QUMOND_simple_nu_diagnostic": FrozenGridField(cluster, lens_redshift, "QUMOND_simple_nu_diagnostic", axis_repaired=False),
            "glafic_v2_compact_halo": GlaficField(cluster, lens_redshift, center),
        }
        common_bound = min(field.half_extent_arcsec for field in fields.values()) * 0.965
        print(f"P0714 {cluster}: {len(block)} images, bound={common_bound:.1f} arcsec", flush=True)

        for family_id, family_images in block.groupby(block.family_id.astype(str), sort=True):
            source_redshift = float(family_images.adopted_catalog_redshift.median())
            model_cache: dict[str, dict[str, object]] = {}
            for model in MODELS:
                field = fields[model]
                source = profiled_source(field, family_images, source_redshift)
                roots, magnification = find_roots(field, source, source_redshift, family_images, common_bound)
                pairs, rms, matched = assignment(family_images, roots)
                model_cache[model] = {
                    "source": source,
                    "roots": roots,
                    "magnification": magnification,
                    "pairs": pairs,
                    "rms": rms,
                    "matched": matched,
                }

            comparator = model_cache["glafic_v2_compact_halo"]
            if comparator["matched"] == len(family_images):
                matched_magnification = comparator["magnification"][comparator["pairs"][:, 1]]
                threshold = float(np.min(matched_magnification))
            else:
                threshold = 0.0
            glafic_curve = curve_points(fields["glafic_v2_compact_halo"], source_redshift, common_bound)

            for model in MODELS:
                cached = model_cache[model]
                roots = cached["roots"]
                magnification = cached["magnification"]
                retained = roots[magnification >= threshold]
                _retained_pairs, _retained_rms, retained_matched = assignment(family_images, retained)
                all_pairs = cached["pairs"]
                all_rms = cached["rms"]
                all_matched = cached["matched"]
                topology_correct = retained_matched == len(family_images) and len(retained) == len(family_images)
                family_records.append(
                    {
                        "cluster": cluster,
                        "partition": partition[family_id],
                        "family_id": family_id,
                        "source_redshift": source_redshift,
                        "model": model,
                        "observed_images": len(family_images),
                        "global_roots": len(roots),
                        "magnification_threshold_from_glafic": threshold,
                        "retained_roots": len(retained),
                        "matched_images": all_matched,
                        "image_RMS_arcsec": all_rms,
                        "topology_retained_matched_images": retained_matched,
                        "topology_correct": topology_correct,
                        "source_east_arcsec": cached["source"][0],
                        "source_north_arcsec": cached["source"][1],
                    }
                )
                observed = family_images[["image_id", "east_arcsec", "north_arcsec"]].reset_index(drop=True)
                for observed_index, root_index in all_pairs:
                    root = roots[root_index]
                    root_records.append(
                        {
                            "cluster": cluster,
                            "family_id": family_id,
                            "model": model,
                            "image_id": observed.loc[observed_index, "image_id"],
                            "observed_east_arcsec": observed.loc[observed_index, "east_arcsec"],
                            "observed_north_arcsec": observed.loc[observed_index, "north_arcsec"],
                            "root_east_arcsec": root[0],
                            "root_north_arcsec": root[1],
                            "residual_arcsec": float(np.linalg.norm(root - observed.loc[observed_index, ["east_arcsec", "north_arcsec"]].to_numpy(float))),
                        }
                    )
                model_curve = curve_points(fields[model], source_redshift, common_bound)
                observed_points = family_images[["east_arcsec", "north_arcsec"]].to_numpy(float)
                observed_curve_distance = float(np.mean(cKDTree(model_curve).query(observed_points)[0])) if len(model_curve) else None
                curve_records.append(
                    {
                        "cluster": cluster,
                        "family_id": family_id,
                        "partition": partition[family_id],
                        "model": model,
                        "curve_points": len(model_curve),
                        "mean_observed_image_to_curve_arcsec": observed_curve_distance,
                        "symmetric_p95_distance_to_glafic_arcsec": symmetric_p95_distance(model_curve, glafic_curve),
                        "independently_observable_gate": False,
                        "reason": "catalog has no arc orientation or parity",
                    }
                )

        family_frame = pd.DataFrame.from_records(family_records)
        current = family_frame[family_frame.cluster == cluster]
        for model in MODELS:
            model_rows = current[(current.model == model) & (current.partition == "holdout")]
            complete = bool((model_rows.matched_images == model_rows.observed_images).all())
            total_images = int(model_rows.observed_images.sum())
            matched_images = int(model_rows.matched_images.sum())
            if complete:
                rms = float(np.sqrt(np.average(np.square(model_rows.image_RMS_arcsec), weights=model_rows.observed_images)))
            else:
                rms = float("inf")
            cluster_model_records.append(
                {
                    "cluster": cluster,
                    "model": model,
                    "heldout_families": len(model_rows),
                    "heldout_images": total_images,
                    "matched_images": matched_images,
                    "root_convergence_fraction": matched_images / total_images,
                    "heldout_image_RMS_arcsec": rms,
                    "all_heldout_topologies_correct": bool(model_rows.topology_correct.all()),
                }
            )

    families = pd.DataFrame.from_records(family_records)
    roots = pd.DataFrame.from_records(root_records)
    curves = pd.DataFrame.from_records(curve_records)
    scores = pd.DataFrame.from_records(cluster_model_records)
    for cluster in ready:
        indexed = scores[scores.cluster == cluster].set_index("model")
        halo = float(indexed.loc["glafic_v2_compact_halo", "heldout_image_RMS_arcsec"])
        for model in MODELS:
            value = float(indexed.loc[model, "heldout_image_RMS_arcsec"])
            scores.loc[(scores.cluster == cluster) & (scores.model == model), "RMS_ratio_to_glafic"] = value / halo if np.isfinite(value) and np.isfinite(halo) and halo > 0.0 else np.nan

    candidate = scores[scores.model == "P0707_Weyl_frozen_axis_contract"]
    candidate_complete = bool((candidate.root_convergence_fraction == 1.0).all())
    candidate_topology = bool(candidate.all_heldout_topologies_correct.all())
    ratios = candidate.RMS_ratio_to_glafic.dropna()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    families.to_csv(OUTPUT / "family_model_scores.csv", index=False)
    roots.to_csv(OUTPUT / "matched_roots.csv", index=False)
    curves.to_csv(OUTPUT / "critical_curve_diagnostics.csv", index=False)
    scores.to_csv(OUTPUT / "cluster_model_scores.csv", index=False)

    figure, axes = plt.subplots(1, len(ready), figsize=(7 * len(ready), 5.2), constrained_layout=True, squeeze=False)
    colors = ["#1f77b4", "#17becf", "#777777", "#d95f02", "#e6ab02", "#7b3294"]
    for axis, cluster in zip(axes[0], ready, strict=True):
        block = scores[scores.cluster == cluster].set_index("model").loc[MODELS]
        values = block.heldout_image_RMS_arcsec.to_numpy(float)
        shown = np.where(np.isfinite(values), values, max(100.0, np.nanmax(values[np.isfinite(values)]) * 2.0))
        bars = axis.bar(np.arange(len(MODELS)), shown, color=colors)
        for bar, finite in zip(bars, np.isfinite(values), strict=True):
            if not finite:
                bar.set_hatch("//")
                axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), "incomplete", rotation=90, ha="center", va="top", color="white", fontsize=8)
        axis.set_yscale("log")
        axis.set_xticks(np.arange(len(MODELS)), [name.replace("_", "\n") for name in MODELS], rotation=35, ha="right", fontsize=8)
        axis.set_ylabel("heldout image-plane RMS (arcsec)")
        axis.set_title(f"{cluster} (descriptive ready subset)")
    figure.savefig(OUTPUT / "ready_subset_lensing_scores.png", dpi=160)
    plt.close(figure)

    report = {
        "stage": "P0714",
        "status": "completed_exploratory_ready_subset",
        "validation_status": "not_a_P0633_validation_score_due_to_P0713_data_readiness_failure",
        "ready_clusters": ready,
        "selected_clusters_not_replaced": True,
        "models": MODELS,
        "frozen_candidate_all_heldout_roots_converged": candidate_complete,
        "frozen_candidate_all_heldout_topologies_correct": candidate_topology,
        "frozen_candidate_RMS_ratios_to_glafic": [float(value) for value in ratios],
        "critical_curve_gate": "not_independently_observable_missing_arc_orientation_or_parity",
        "coordinate_audit": {
            "frozen_contract": "P0708 treated array axis 0/component 0 as east/x and axis 1/component 1 as north/y.",
            "registered_map_storage": "P0641 arrays are image-row north then image-column east.",
            "axis_repair_status": "post-unseal exploratory diagnostic only; not validation",
        },
        "claim_boundary": [
            "Only AS295 and PLCKG287 passed the frozen raw-constraint readiness gate.",
            "The compact-halo comparator is the same predeclared glafic v2 method for both clusters.",
            "AQUAL and QUMOND photon maps remain phenomenological non-relativistic diagnostics.",
            "The critical-curve comparison is model-to-model because the catalogs do not publish arc parity/orientation.",
            "No lens center, shear, ellipticity, mass sheet, radial amplitude, or gravity parameter was fitted.",
        ],
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    displayed = scores.copy()
    displayed["heldout_image_RMS_arcsec"] = displayed.heldout_image_RMS_arcsec.map(finite_or_none)
    lines = [
        "# P0714 descriptive raw lensing on the ready subset",
        "",
        "This is not the preregistered four-cluster validation: P0713 found only two ready clusters.",
        "",
        "| Cluster | Model | Root fraction | RMS (arcsec) | RMS / glafic | Topology |",
        "|---|---|---:|---:|---:|:---:|",
    ]
    for row in displayed.itertuples(index=False):
        rms = "incomplete" if row.heldout_image_RMS_arcsec is None or not np.isfinite(row.heldout_image_RMS_arcsec) else f"{row.heldout_image_RMS_arcsec:.3f}"
        ratio = "n/a" if not np.isfinite(row.RMS_ratio_to_glafic) else f"{row.RMS_ratio_to_glafic:.3f}"
        lines.append(f"| {row.cluster} | {row.model} | {row.root_convergence_fraction:.3f} | {rms} | {ratio} | {'yes' if row.all_heldout_topologies_correct else 'no'} |")
    lines.extend(
        [
            "",
            "The axis-repaired candidate is a post-unseal software diagnostic and cannot replace the frozen score.",
            "The critical-curve gate is unobservable from these position-only catalogs because arc orientation/parity is absent.",
        ]
    )
    summary = "\n".join(lines) + "\n"
    (OUTPUT / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
