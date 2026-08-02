#!/usr/bin/env python3
"""Validate the coordinate-safe lens engine on archived real cluster maps.

This is a software/measurement validation stage on the already-spent P0714
sample.  It does not rescore the frozen physical hypothesis.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sky_lensing import (
    GridSkyDeflectionField,
    assign_observed_roots,
    critical_curve_points,
    find_lens_roots,
    lens_invariants,
    profiled_source,
    symmetric_percentile_distance,
)

READINESS = ROOT / "results/p0713_external_cluster_readiness_audit"
PREDICTIONS = ROOT / "results/p0708_external_prediction_lock/clusters"
BARYON_MAPS = ROOT / "results/p0641_registered_cluster_baryon_maps/maps"
COMPARATORS = ROOT / "data/raw/p0633_relics_lensing_comparators"
P0714 = ROOT / "results/p0714_ready_subset_raw_lensing"
OUTPUT = ROOT / "results/p0715_sky_lensing_engine_validation"
GRID_DENSITIES = (81, 161, 241)
MODEL_NAMES = {
    "candidate_axis_repaired": "P0707_Weyl_axis_repaired_exploratory",
    "glafic_v2_compact_halo": "glafic_v2_compact_halo",
}


def distance_ratio(lens_redshift: float, source_redshift: float) -> float:
    if source_redshift <= lens_redshift:
        raise ValueError("source must be behind lens")
    return float(
        Planck18.angular_diameter_distance_z1z2(lens_redshift, source_redshift)
        / Planck18.angular_diameter_distance(source_redshift)
    )


def one(directory: Path, pattern: str) -> Path:
    matches = list(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern} in {directory}, found {len(matches)}")
    return matches[0]


def frozen_sky_field(
    cluster: str,
    lens_redshift: float,
    component: str,
) -> GridSkyDeflectionField:
    """Read the P0708 array using its post-unseal, explicit sky convention."""
    path = PREDICTIONS / f"{cluster}_physical_deflections.npz"
    with np.load(path) as data:
        axis_kpc = data["axis_kpc"].astype(float)
        # P0708 component 0 followed array axis 0 (north); component 1 followed
        # array axis 1 (east).  The names x/y were therefore semantically wrong.
        alpha_north = data[f"alpha_x_{component}_arcsec"].astype(float)
        alpha_east = data[f"alpha_y_{component}_arcsec"].astype(float)
    kpc_per_arcsec = float(
        Planck18.kpc_proper_per_arcmin(lens_redshift).value / 60.0
    )
    axis_arcsec = axis_kpc / kpc_per_arcsec
    return GridSkyDeflectionField(
        north_axis_arcsec=axis_arcsec,
        east_axis_arcsec=axis_arcsec,
        alpha_east_ratio_one_arcsec=alpha_east,
        alpha_north_ratio_one_arcsec=alpha_north,
        distance_ratio=lambda source_redshift: distance_ratio(
            lens_redshift, source_redshift
        ),
    )


def glafic_comparator(
    cluster: str,
    lens_redshift: float,
    baryon_center: SkyCoord,
) -> GridSkyDeflectionField:
    directory = COMPARATORS / cluster / "glafic" / "v2"
    x_path = one(directory, "*_x-arcsec-deflect.fits")
    y_path = one(directory, "*_y-arcsec-deflect.fits")
    with fits.open(x_path, memmap=True) as handle:
        x_map = np.asarray(handle[0].data, dtype=float)
        header = handle[0].header
    with fits.open(y_path, memmap=True) as handle:
        y_map = np.asarray(handle[0].data, dtype=float)
    pixel_arcsec = abs(float(header["CDELT1"])) * 3600.0
    east_from_reference = -(
        np.arange(x_map.shape[1]) - (float(header["CRPIX1"]) - 1.0)
    ) * pixel_arcsec
    north_from_reference = (
        np.arange(x_map.shape[0]) - (float(header["CRPIX2"]) - 1.0)
    ) * pixel_arcsec
    reference = SkyCoord(
        float(header["CRVAL1"]) * u.deg,
        float(header["CRVAL2"]) * u.deg,
    )
    center_east, center_north = reference.spherical_offsets_to(baryon_center)
    east_from_baryon = east_from_reference - center_east.to_value(u.arcsec)
    north_from_baryon = north_from_reference - center_north.to_value(u.arcsec)
    # GLAFIC image x increases west; reverse the grid and vector sign to east.
    return GridSkyDeflectionField(
        north_axis_arcsec=north_from_baryon,
        east_axis_arcsec=east_from_baryon[::-1],
        alpha_east_ratio_one_arcsec=(-x_map)[:, ::-1],
        alpha_north_ratio_one_arcsec=y_map[:, ::-1],
        distance_ratio=lambda source_redshift: distance_ratio(
            lens_redshift, source_redshift
        ),
    )


def finite_difference(first: float, second: float) -> float:
    if np.isinf(first) and np.isinf(second):
        return 0.0
    return abs(float(first) - float(second))


def main() -> None:
    readiness = json.loads((READINESS / "report.json").read_text(encoding="utf-8"))
    if readiness["status"] != "fail_data_readiness":
        raise RuntimeError("P0715 requires the spent P0713/P0714 archive")
    ready = [row["cluster"] for row in readiness["cluster_rows"] if row["ready"]]
    if ready != ["AS295", "PLCKG287"]:
        raise RuntimeError("ready subset changed")
    catalog = pd.read_csv(READINESS / "parsed_image_catalog.csv")
    catalog = catalog[
        catalog.secure_image.astype(str).str.lower().eq("true")
        & catalog.cluster.isin(ready)
    ].copy()
    archived = pd.read_csv(P0714 / "family_model_scores.csv")

    root_records: list[dict[str, object]] = []
    invariant_records: list[dict[str, object]] = []
    curve_records: list[dict[str, object]] = []
    conformance_records: list[dict[str, object]] = []

    for cluster in ready:
        with np.load(BARYON_MAPS / f"{cluster}_baryons.npz") as data:
            center = SkyCoord(
                float(data["center_ra_deg"]) * u.deg,
                float(data["center_dec_deg"]) * u.deg,
            )
            lens_redshift = float(data["redshift"])
        block = catalog[catalog.cluster == cluster].copy()
        coordinates = SkyCoord(
            block.ra_deg.to_numpy(float) * u.deg,
            block.dec_deg.to_numpy(float) * u.deg,
        )
        east, north = center.spherical_offsets_to(coordinates)
        block["east_arcsec"] = east.to_value(u.arcsec)
        block["north_arcsec"] = north.to_value(u.arcsec)
        fields = {
            "candidate_axis_repaired": frozen_sky_field(
                cluster, lens_redshift, "P0707_Weyl"
            ),
            "glafic_v2_compact_halo": glafic_comparator(
                cluster, lens_redshift, center
            ),
        }
        common_bound = min(field.half_extent_arcsec for field in fields.values()) * 0.965
        print(
            f"P0715 {cluster}: {len(block)} images, bound={common_bound:.1f} arcsec",
            flush=True,
        )
        for family_id, images in block.groupby(block.family_id.astype(str), sort=True):
            observed = images[["east_arcsec", "north_arcsec"]].to_numpy(float)
            source_redshift = float(images.adopted_catalog_redshift.median())
            family_curves: dict[str, np.ndarray] = {}
            for model, field in fields.items():
                source = profiled_source(field, observed, source_redshift)
                invariants = lens_invariants(
                    field,
                    observed[:, 0],
                    observed[:, 1],
                    source_redshift,
                )
                for image, kappa, gamma, rotation, determinant, eigen_min, eigen_max in zip(
                    images.image_id.astype(str),
                    invariants.convergence,
                    invariants.shear_magnitude,
                    invariants.rotation,
                    invariants.determinant,
                    invariants.minimum_eigenvalue,
                    invariants.maximum_eigenvalue,
                    strict=True,
                ):
                    invariant_records.append(
                        {
                            "cluster": cluster,
                            "family_id": family_id,
                            "image_id": image,
                            "model": model,
                            "convergence": kappa,
                            "shear_magnitude": gamma,
                            "rotation": rotation,
                            "jacobian_determinant": determinant,
                            "minimum_eigenvalue": eigen_min,
                            "maximum_eigenvalue": eigen_max,
                        }
                    )
                by_density: dict[int, tuple[object, object]] = {}
                for grid_points in GRID_DENSITIES:
                    roots = find_lens_roots(
                        field,
                        source,
                        source_redshift,
                        bound_arcsec=common_bound,
                        observed_starts_arcsec=observed,
                        grid_points=grid_points,
                    )
                    assignment = assign_observed_roots(observed, roots.roots_arcsec)
                    by_density[grid_points] = (roots, assignment)
                    root_records.append(
                        {
                            "cluster": cluster,
                            "family_id": family_id,
                            "model": model,
                            "grid_points": grid_points,
                            "source_redshift": source_redshift,
                            "source_east_arcsec": source[0],
                            "source_north_arcsec": source[1],
                            "root_count": len(roots.roots_arcsec),
                            "matched_images": assignment.matched_images,
                            "image_RMS_arcsec": assignment.rms_arcsec,
                            "maximum_closure_arcsec": (
                                float(np.max(roots.closure_arcsec))
                                if len(roots.closure_arcsec)
                                else np.nan
                            ),
                        }
                    )
                # Re-run the exact legacy seeding rule only for provenance
                # conformance; the density sweep above uses the hardened engine.
                legacy_roots = find_lens_roots(
                    field,
                    source,
                    source_redshift,
                    bound_arcsec=common_bound,
                    observed_starts_arcsec=observed,
                    grid_points=161,
                    include_residual_minima=False,
                    supplemental_grid_points=(),
                )
                legacy_assignment = assign_observed_roots(
                    observed, legacy_roots.roots_arcsec
                )
                old = archived[
                    (archived.cluster == cluster)
                    & (archived.family_id.astype(str) == str(family_id))
                    & (archived.model == MODEL_NAMES[model])
                ]
                if len(old) != 1:
                    raise RuntimeError(f"missing archived row {cluster}/{family_id}/{model}")
                old_row = old.iloc[0]
                conformance_records.append(
                    {
                        "cluster": cluster,
                        "family_id": family_id,
                        "model": model,
                        "archived_root_count": int(old_row.global_roots),
                        "current_root_count": len(legacy_roots.roots_arcsec),
                        "root_count_match": (
                            int(old_row.global_roots) == len(legacy_roots.roots_arcsec)
                        ),
                        "source_position_difference_arcsec": float(
                            np.linalg.norm(
                                source
                                - old_row[
                                    ["source_east_arcsec", "source_north_arcsec"]
                                ].to_numpy(float)
                            )
                        ),
                        "RMS_difference_arcsec": finite_difference(
                            legacy_assignment.rms_arcsec,
                            float(old_row.image_RMS_arcsec),
                        ),
                        "root_count_stable_81_161_241": len(
                            {
                                len(by_density[density][0].roots_arcsec)
                                for density in GRID_DENSITIES
                            }
                        )
                        == 1,
                    }
                )
                family_curves[model] = critical_curve_points(
                    field,
                    source_redshift,
                    bound_arcsec=common_bound,
                    grid_points=241,
                )
            comparator_curve = family_curves["glafic_v2_compact_halo"]
            for model, points in family_curves.items():
                curve_records.append(
                    {
                        "cluster": cluster,
                        "family_id": family_id,
                        "model": model,
                        "critical_cells": len(points),
                        "median_critical_radius_arcsec": (
                            float(np.median(np.hypot(points[:, 0], points[:, 1])))
                            if len(points)
                            else np.nan
                        ),
                        "symmetric_p95_to_glafic_arcsec": symmetric_percentile_distance(
                            points, comparator_curve
                        ),
                    }
                )

    roots = pd.DataFrame.from_records(root_records)
    invariants = pd.DataFrame.from_records(invariant_records)
    curves = pd.DataFrame.from_records(curve_records)
    conformance = pd.DataFrame.from_records(conformance_records)
    root_matches = bool(conformance.root_count_match.all())
    source_matches = bool(
        conformance.source_position_difference_arcsec.max() < 1.0e-9
    )
    rms_matches = bool(conformance.RMS_difference_arcsec.max() < 1.0e-6)
    stable_fraction = float(conformance.root_count_stable_81_161_241.mean())
    comparator_stability = float(
        conformance.loc[
            conformance.model == "glafic_v2_compact_halo",
            "root_count_stable_81_161_241",
        ].mean()
    )
    passed = root_matches and source_matches and rms_matches and comparator_stability == 1.0

    OUTPUT.mkdir(parents=True, exist_ok=True)
    roots.to_csv(OUTPUT / "root_density_sweep.csv", index=False)
    invariants.to_csv(OUTPUT / "observed_arc_invariants.csv", index=False)
    curves.to_csv(OUTPUT / "critical_curve_measurements.csv", index=False)
    conformance.to_csv(OUTPUT / "p0714_conformance.csv", index=False)
    report = {
        "stage": "P0715",
        "status": "pass" if passed else "fail",
        "evaluation_kind": "software_measurement_validation_on_spent_cluster_sample",
        "formula_rescored": False,
        "coordinate_contract": {
            "array_axis_0": "north",
            "array_axis_1": "east",
            "vector_component_0": "alpha_east",
            "vector_component_1": "alpha_north",
        },
        "production_root_search_floor": {
            "supplemental_grid_points": [81, 161, 241],
            "reason": "the archived 65x65 maps contain complementary narrow basins at the three grid phases",
        },
        "analytic_tests": [
            "north/east asymmetric interpolation",
            "affine convergence/shear/Jacobian",
            "SIS two-image roots at 81/161/241 grids",
            "SIS Einstein critical ring",
            "source profiling and image assignment",
            "north/east photon integration wrapper",
        ],
        "ready_clusters": ready,
        "families": int(conformance[["cluster", "family_id"]].drop_duplicates().shape[0]),
        "models": list(MODEL_NAMES),
        "root_count_conformance_to_P0714": root_matches,
        "maximum_source_position_difference_arcsec": float(
            conformance.source_position_difference_arcsec.max()
        ),
        "maximum_RMS_difference_arcsec": float(
            conformance.RMS_difference_arcsec.max()
        ),
        "all_model_root_count_stability_fraction": stable_fraction,
        "glafic_root_count_stability_fraction": comparator_stability,
        "claim_boundary": [
            "P0715 validates software and coordinate semantics; it does not rehabilitate P0714 as a blind validation.",
            "The candidate map is the disclosed post-unseal axis repair, never the frozen P0708 score.",
            "Critical curves remain model-to-model diagnostics because the catalogs lack observed parity/orientation.",
        ],
    }
    (OUTPUT / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    summary = "\n".join(
        [
            "# P0715 coordinate-safe sky-lensing engine validation",
            "",
            f"Status: **{report['status'].upper()}**",
            "",
            f"- P0714 root-count conformance: {root_matches}",
            f"- Maximum source-position difference: {report['maximum_source_position_difference_arcsec']:.3e} arcsec",
            f"- Maximum RMS difference: {report['maximum_RMS_difference_arcsec']:.3e} arcsec",
            f"- Compact-halo root-count stability across 81/161/241 grids: {comparator_stability:.3f}",
            f"- All-model root-count stability: {stable_fraction:.3f}",
            "",
            "This validates the measurement engine, not the candidate formula or the spent cluster sample.",
        ]
    ) + "\n"
    (OUTPUT / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
