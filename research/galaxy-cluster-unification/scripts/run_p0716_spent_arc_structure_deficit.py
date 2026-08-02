#!/usr/bin/env python3
"""Measure which local lens structures the P0707 field lacks at real arcs.

The P0713/P0714 sample is already spent.  This stage is diagnostic: it compares
coordinate-corrected baryon-derived maps with the archived compact-halo
comparator without fitting or claiming new validation.
"""

from __future__ import annotations

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0715_sky_lensing_engine_validation import (
    BARYON_MAPS,
    READINESS,
    frozen_sky_field,
    glafic_comparator,
)

from voidscreen.sky_lensing import (
    critical_curve_points,
    lens_invariants,
    symmetric_percentile_distance,
)

OUTPUT = ROOT / "results/p0716_spent_arc_structure_deficit"
MODELS = {
    "P0707_Weyl_axis_repaired": "P0707_Weyl",
    "baryon_only_GR_axis_repaired": "baryon_only_GR",
    "AQUAL_simple_mu_axis_repaired": "AQUAL_simple_mu_diagnostic",
    "QUMOND_simple_nu_axis_repaired": "QUMOND_simple_nu_diagnostic",
}
COMPARATOR = "glafic_v2_compact_halo"


def shear_angle_difference_degrees(
    first_gamma_1: np.ndarray,
    first_gamma_2: np.ndarray,
    second_gamma_1: np.ndarray,
    second_gamma_2: np.ndarray,
) -> np.ndarray:
    """Return the acute spin-2 shear-axis difference in degrees."""
    first = 0.5 * np.arctan2(first_gamma_2, first_gamma_1)
    second = 0.5 * np.arctan2(second_gamma_2, second_gamma_1)
    difference = 0.5 * np.arctan2(
        np.sin(2.0 * (first - second)),
        np.cos(2.0 * (first - second)),
    )
    return np.abs(np.degrees(difference))


def correlation(first: pd.Series, second: pd.Series) -> float:
    if len(first) < 3 or np.std(first) == 0.0 or np.std(second) == 0.0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def main() -> None:
    readiness = json.loads((READINESS / "report.json").read_text(encoding="utf-8"))
    ready = [row["cluster"] for row in readiness["cluster_rows"] if row["ready"]]
    if ready != ["AS295", "PLCKG287"]:
        raise RuntimeError("P0716 requires the spent P0714 ready subset")
    catalog = pd.read_csv(READINESS / "parsed_image_catalog.csv")
    catalog = catalog[
        catalog.secure_image.astype(str).str.lower().eq("true")
        & catalog.cluster.isin(ready)
    ].copy()
    arc_records: list[dict[str, object]] = []
    curve_records: list[dict[str, object]] = []

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
            model: frozen_sky_field(cluster, lens_redshift, component)
            for model, component in MODELS.items()
        }
        fields[COMPARATOR] = glafic_comparator(cluster, lens_redshift, center)
        common_bound = min(field.half_extent_arcsec for field in fields.values()) * 0.965

        for family_id, images in block.groupby(block.family_id.astype(str), sort=True):
            source_redshift = float(images.adopted_catalog_redshift.median())
            observed_east = images.east_arcsec.to_numpy(float)
            observed_north = images.north_arcsec.to_numpy(float)
            family_curves: dict[str, np.ndarray] = {}
            for model, field in fields.items():
                invariants = lens_invariants(
                    field,
                    observed_east,
                    observed_north,
                    source_redshift,
                )
                shear_angle = 0.5 * np.degrees(
                    np.arctan2(invariants.shear_2, invariants.shear_1)
                )
                for index, image in enumerate(images.itertuples(index=False)):
                    arc_records.append(
                        {
                            "cluster": cluster,
                            "family_id": family_id,
                            "image_id": str(image.image_id),
                            "model": model,
                            "source_redshift": source_redshift,
                            "east_arcsec": observed_east[index],
                            "north_arcsec": observed_north[index],
                            "convergence": invariants.convergence[index],
                            "shear_1": invariants.shear_1[index],
                            "shear_2": invariants.shear_2[index],
                            "shear_magnitude": invariants.shear_magnitude[index],
                            "shear_angle_deg": shear_angle[index],
                            "rotation": invariants.rotation[index],
                            "jacobian_determinant": invariants.determinant[index],
                            "minimum_eigenvalue": invariants.minimum_eigenvalue[index],
                            "maximum_eigenvalue": invariants.maximum_eigenvalue[index],
                        }
                    )
                family_curves[model] = critical_curve_points(
                    field,
                    source_redshift,
                    bound_arcsec=common_bound,
                    grid_points=241,
                )
            halo_curve = family_curves[COMPARATOR]
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
                            points, halo_curve
                        ),
                    }
                )

    arcs = pd.DataFrame.from_records(arc_records)
    curves = pd.DataFrame.from_records(curve_records)
    keys = ["cluster", "family_id", "image_id"]
    halo = arcs[arcs.model == COMPARATOR].set_index(keys)
    deficit_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []
    for model in MODELS:
        candidate = arcs[arcs.model == model].set_index(keys)
        joined = candidate.join(halo, lsuffix="_model", rsuffix="_halo")
        angle_difference = shear_angle_difference_degrees(
            joined.shear_1_model.to_numpy(),
            joined.shear_2_model.to_numpy(),
            joined.shear_1_halo.to_numpy(),
            joined.shear_2_halo.to_numpy(),
        )
        joined["shear_angle_difference_deg"] = angle_difference
        joined["kappa_needed"] = joined.convergence_halo - joined.convergence_model
        joined["shear_magnitude_needed"] = (
            joined.shear_magnitude_halo - joined.shear_magnitude_model
        )
        joined["minimum_eigenvalue_gap"] = (
            joined.minimum_eigenvalue_model - joined.minimum_eigenvalue_halo
        )
        joined["eigenvalue_decomposition_residual"] = (
            joined.minimum_eigenvalue_gap
            - joined.kappa_needed
            - joined.shear_magnitude_needed
        )
        joined["determinant_sign_match"] = (
            np.sign(joined.jacobian_determinant_model)
            == np.sign(joined.jacobian_determinant_halo)
        )
        joined = joined.reset_index()
        joined["model"] = model
        deficit_records.extend(joined.to_dict(orient="records"))
        for cluster in ["ALL", *ready]:
            subset = joined if cluster == "ALL" else joined[joined.cluster == cluster]
            summary_records.append(
                {
                    "cluster": cluster,
                    "model": model,
                    "arcs": len(subset),
                    "median_model_convergence": float(
                        subset.convergence_model.median()
                    ),
                    "median_halo_convergence": float(subset.convergence_halo.median()),
                    "median_kappa_needed": float(subset.kappa_needed.median()),
                    "convergence_correlation": correlation(
                        subset.convergence_model, subset.convergence_halo
                    ),
                    "median_model_shear": float(subset.shear_magnitude_model.median()),
                    "median_halo_shear": float(subset.shear_magnitude_halo.median()),
                    "median_shear_needed": float(
                        subset.shear_magnitude_needed.median()
                    ),
                    "shear_magnitude_correlation": correlation(
                        subset.shear_magnitude_model, subset.shear_magnitude_halo
                    ),
                    "median_shear_angle_difference_deg": float(
                        subset.shear_angle_difference_deg.median()
                    ),
                    "median_model_minimum_eigenvalue": float(
                        subset.minimum_eigenvalue_model.median()
                    ),
                    "median_halo_minimum_eigenvalue": float(
                        subset.minimum_eigenvalue_halo.median()
                    ),
                    "median_minimum_eigenvalue_gap": float(
                        subset.minimum_eigenvalue_gap.median()
                    ),
                    "fraction_model_near_critical_abs_lambda_min_lt_0p2": float(
                        (subset.minimum_eigenvalue_model.abs() < 0.2).mean()
                    ),
                    "fraction_halo_near_critical_abs_lambda_min_lt_0p2": float(
                        (subset.minimum_eigenvalue_halo.abs() < 0.2).mean()
                    ),
                    "determinant_sign_match_fraction": float(
                        subset.determinant_sign_match.mean()
                    ),
                    "maximum_eigenvalue_decomposition_residual": float(
                        subset.eigenvalue_decomposition_residual.abs().max()
                    ),
                }
            )

    deficits = pd.DataFrame.from_records(deficit_records)
    summaries = pd.DataFrame.from_records(summary_records)
    candidate_summary = summaries[
        (summaries.cluster == "ALL")
        & (summaries.model == "P0707_Weyl_axis_repaired")
    ].iloc[0]
    candidate_by_cluster = summaries[
        (summaries.cluster != "ALL")
        & (summaries.model == "P0707_Weyl_axis_repaired")
    ].set_index("cluster")
    candidate_curves = curves[curves.model == "P0707_Weyl_axis_repaired"]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    arcs.to_csv(OUTPUT / "all_model_arc_invariants.csv", index=False)
    deficits.to_csv(OUTPUT / "arc_structure_deficits.csv", index=False)
    summaries.to_csv(OUTPUT / "structure_deficit_summary.csv", index=False)
    curves.to_csv(OUTPUT / "critical_curve_deficits.csv", index=False)

    candidate = deficits[deficits.model == "P0707_Weyl_axis_repaired"]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    colors = candidate.cluster.map({"AS295": "#1f77b4", "PLCKG287": "#d95f02"})
    axes[0].scatter(candidate.convergence_halo, candidate.convergence_model, c=colors)
    axes[0].plot([0, 1.5], [0, 1.5], "k--", linewidth=1)
    axes[0].set(xlabel="compact-halo convergence", ylabel="P0707 convergence")
    axes[1].scatter(
        candidate.shear_magnitude_halo,
        candidate.shear_magnitude_model,
        c=colors,
    )
    axes[1].plot([0, 0.8], [0, 0.8], "k--", linewidth=1)
    axes[1].set(xlabel="compact-halo shear", ylabel="P0707 shear")
    contribution = candidate.groupby("cluster")[[
        "kappa_needed",
        "shear_magnitude_needed",
    ]].median()
    contribution.plot.bar(ax=axes[2], color=["#7570b3", "#e7298a"])
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set(ylabel="median missing contribution", xlabel="")
    figure.savefig(OUTPUT / "arc_structure_deficits.png", dpi=180)
    plt.close(figure)

    report = {
        "stage": "P0716",
        "status": "completed_spent_structure_diagnostic",
        "evaluation_kind": "post_unseal_no_fit_structure_deficit",
        "formula_fitted": False,
        "sample_is_spent": True,
        "coordinate_contract": "north rows, east columns, explicit east/north vectors",
        "candidate_overall": {
            "median_kappa_needed": float(candidate_summary.median_kappa_needed),
            "median_shear_needed": float(candidate_summary.median_shear_needed),
            "median_minimum_eigenvalue_gap": float(
                candidate_summary.median_minimum_eigenvalue_gap
            ),
            "convergence_correlation_to_halo": float(
                candidate_summary.convergence_correlation
            ),
            "shear_correlation_to_halo": float(
                candidate_summary.shear_magnitude_correlation
            ),
            "median_shear_axis_error_deg": float(
                candidate_summary.median_shear_angle_difference_deg
            ),
            "candidate_near_critical_arc_fraction": float(
                candidate_summary.fraction_model_near_critical_abs_lambda_min_lt_0p2
            ),
            "halo_near_critical_arc_fraction": float(
                candidate_summary.fraction_halo_near_critical_abs_lambda_min_lt_0p2
            ),
            "determinant_sign_match_fraction": float(
                candidate_summary.determinant_sign_match_fraction
            ),
            "median_critical_curve_p95_distance_arcsec": float(
                candidate_curves.symmetric_p95_to_glafic_arcsec.median()
            ),
        },
        "cluster_contrast": {
            cluster: {
                "median_kappa_needed": float(row.median_kappa_needed),
                "median_shear_needed": float(row.median_shear_needed),
                "median_minimum_eigenvalue_gap": float(
                    row.median_minimum_eigenvalue_gap
                ),
            }
            for cluster, row in candidate_by_cluster.iterrows()
        },
        "main_inference": (
            "The missing topology is a local Hessian problem, not only an amplitude problem: "
            "P0707 lacks both convergence and correctly oriented shear near the arcs, with a "
            "strong cluster-dependent split between those contributions."
        ),
        "next_formula_requirement": (
            "A viable baryon-derived correction must alter the two-dimensional Hessian of the "
            "Weyl potential and transfer across clusters; a universal radial rescaling is insufficient."
        ),
        "claim_boundary": [
            "The glafic map is a compact-halo reconstruction, not direct ground truth at every pixel.",
            "The same image catalogs helped constrain the comparator; this stage is diagnostic, not validation.",
            "AQUAL and QUMOND photon maps remain nonrelativistic diagnostics.",
        ],
    }
    (OUTPUT / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    summary = "\n".join(
        [
            "# P0716 spent arc-structure deficit",
            "",
            "The robust one-root result is primarily a missing local-Hessian problem.",
            "",
            f"- Median missing convergence: {report['candidate_overall']['median_kappa_needed']:.3f}",
            f"- Median missing shear: {report['candidate_overall']['median_shear_needed']:.3f}",
            f"- Median minimum-eigenvalue gap: {report['candidate_overall']['median_minimum_eigenvalue_gap']:.3f}",
            f"- Candidate versus halo shear correlation: {report['candidate_overall']['shear_correlation_to_halo']:.3f}",
            f"- Median shear-axis error: {report['candidate_overall']['median_shear_axis_error_deg']:.1f} degrees",
            f"- Candidate/halo near-critical arc fractions: {report['candidate_overall']['candidate_near_critical_arc_fraction']:.3f} / {report['candidate_overall']['halo_near_critical_arc_fraction']:.3f}",
            "",
            "This is a post-unseal diagnostic, not a new validation score.",
        ]
    ) + "\n"
    (OUTPUT / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
