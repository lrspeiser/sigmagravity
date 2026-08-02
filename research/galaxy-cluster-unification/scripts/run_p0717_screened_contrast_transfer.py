#!/usr/bin/env python3
"""Test one-parameter, Solar-screened Weyl-potential contrasts across clusters.

Every fit uses one whole spent cluster and transfers the same scalar to the
other cluster.  This remains exploratory because both clusters were unsealed
before P0717; the purpose is to reject or prioritize formulas, not validate one.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
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
    LinearCombinationSkyDeflectionField,
    assign_observed_roots,
    find_lens_roots,
    lens_invariants,
    profiled_source,
    ray_shoot,
)

OUTPUT = ROOT / "results/p0717_screened_contrast_transfer"
A0_M_S2 = 1.2e-10
SOLAR_G_M3_S2 = 1.32712440018e20
SOLAR_RADIUS_M = 6.957e8
MERCURY_SEMIMAJOR_M = 5.790905e10
SOLAR_PPN_LIMIT = 2.3e-5

# Engineering rejection gates recorded before the complete P0717 transfer run.
MAXIMUM_Q_DISAGREEMENT_FRACTION = 0.25
MINIMUM_ROOT_CONVERGENCE_FRACTION = 0.75
MINIMUM_FINITE_FAMILY_FRACTION = 0.50
MINIMUM_TOPOLOGY_CORRECT_FRACTION = 0.25
MAXIMUM_MEDIAN_RMS_RATIO_TO_HALO = 3.0


@dataclass(frozen=True)
class ClusterData:
    cluster: str
    lens_redshift: float
    images: pd.DataFrame
    fields: dict[str, object]
    common_bound_arcsec: float


@dataclass(frozen=True)
class FormulaDefinition:
    name: str
    base: str
    contrast_positive: str
    contrast_negative: str
    fit_objective: str


FORMULAS = (
    FormulaDefinition(
        "AQUAL_contrast_source_fit",
        "N",
        "A",
        "N",
        "source_scatter",
    ),
    FormulaDefinition(
        "AQUAL_contrast_hessian_fit",
        "N",
        "A",
        "N",
        "local_hessian",
    ),
    FormulaDefinition(
        "QUMOND_contrast_source_fit",
        "N",
        "Q",
        "N",
        "source_scatter",
    ),
    FormulaDefinition(
        "P0707_plus_AQUAL_contrast_source_fit",
        "C",
        "A",
        "N",
        "source_scatter",
    ),
)


def load_clusters() -> dict[str, ClusterData]:
    readiness = json.loads((READINESS / "report.json").read_text(encoding="utf-8"))
    ready = [row["cluster"] for row in readiness["cluster_rows"] if row["ready"]]
    if ready != ["AS295", "PLCKG287"]:
        raise RuntimeError("P0717 requires the spent P0714 ready subset")
    catalog = pd.read_csv(READINESS / "parsed_image_catalog.csv")
    catalog = catalog[
        catalog.secure_image.astype(str).str.lower().eq("true")
        & catalog.cluster.isin(ready)
    ].copy()
    result: dict[str, ClusterData] = {}
    for cluster in ready:
        with np.load(BARYON_MAPS / f"{cluster}_baryons.npz") as data:
            center = SkyCoord(
                float(data["center_ra_deg"]) * u.deg,
                float(data["center_dec_deg"]) * u.deg,
            )
            lens_redshift = float(data["redshift"])
        images = catalog[catalog.cluster == cluster].copy()
        coordinates = SkyCoord(
            images.ra_deg.to_numpy(float) * u.deg,
            images.dec_deg.to_numpy(float) * u.deg,
        )
        east, north = center.spherical_offsets_to(coordinates)
        images["east_arcsec"] = east.to_value(u.arcsec)
        images["north_arcsec"] = north.to_value(u.arcsec)
        fields = {
            "C": frozen_sky_field(cluster, lens_redshift, "P0707_Weyl"),
            "N": frozen_sky_field(cluster, lens_redshift, "baryon_only_GR"),
            "A": frozen_sky_field(
                cluster, lens_redshift, "AQUAL_simple_mu_diagnostic"
            ),
            "Q": frozen_sky_field(
                cluster, lens_redshift, "QUMOND_simple_nu_diagnostic"
            ),
            "H": glafic_comparator(cluster, lens_redshift, center),
        }
        result[cluster] = ClusterData(
            cluster=cluster,
            lens_redshift=lens_redshift,
            images=images,
            fields=fields,
            common_bound_arcsec=min(
                field.half_extent_arcsec for field in fields.values()
            )
            * 0.965,
        )
    return result


def difference_field(data: ClusterData, positive: str, negative: str):
    return LinearCombinationSkyDeflectionField(
        (data.fields[positive], data.fields[negative]),
        (1.0, -1.0),
    )


def formula_field(data: ClusterData, formula: FormulaDefinition, q: float):
    return LinearCombinationSkyDeflectionField(
        (
            data.fields[formula.base],
            data.fields[formula.contrast_positive],
            data.fields[formula.contrast_negative],
        ),
        (1.0, float(q), -float(q)),
    )


def fit_source_scatter_q(data: ClusterData, formula: FormulaDefinition) -> float:
    numerator = 0.0
    denominator = 0.0
    base = data.fields[formula.base]
    contrast = difference_field(
        data, formula.contrast_positive, formula.contrast_negative
    )
    for _family, images in data.images.groupby(
        data.images.family_id.astype(str), sort=True
    ):
        theta = images[["east_arcsec", "north_arcsec"]].to_numpy(float)
        source_redshift = float(images.adopted_catalog_redshift.median())
        base_east, base_north = base.alpha(
            theta[:, 0], theta[:, 1], source_redshift
        )
        delta_east, delta_north = contrast.alpha(
            theta[:, 0], theta[:, 1], source_redshift
        )
        uncorrected = theta - np.column_stack([base_east, base_north])
        delta = np.column_stack([delta_east, delta_north])
        centered_uncorrected = uncorrected - np.mean(uncorrected, axis=0)
        centered_delta = delta - np.mean(delta, axis=0)
        numerator += float(np.sum(centered_delta * centered_uncorrected))
        denominator += float(np.sum(centered_delta * centered_delta))
    if denominator <= 0.0:
        raise RuntimeError("source-scatter contrast is degenerate")
    return numerator / denominator


def fit_hessian_q(data: ClusterData, formula: FormulaDefinition) -> float:
    numerator = 0.0
    denominator = 0.0
    base = data.fields[formula.base]
    contrast = difference_field(
        data, formula.contrast_positive, formula.contrast_negative
    )
    for _family, images in data.images.groupby(
        data.images.family_id.astype(str), sort=True
    ):
        east = images.east_arcsec.to_numpy(float)
        north = images.north_arcsec.to_numpy(float)
        source_redshift = float(images.adopted_catalog_redshift.median())
        base_invariants = lens_invariants(base, east, north, source_redshift)
        contrast_invariants = lens_invariants(
            contrast, east, north, source_redshift
        )
        halo_invariants = lens_invariants(
            data.fields["H"], east, north, source_redshift
        )
        base_vector = np.column_stack(
            [
                base_invariants.convergence,
                base_invariants.shear_1,
                base_invariants.shear_2,
            ]
        )
        contrast_vector = np.column_stack(
            [
                contrast_invariants.convergence,
                contrast_invariants.shear_1,
                contrast_invariants.shear_2,
            ]
        )
        halo_vector = np.column_stack(
            [
                halo_invariants.convergence,
                halo_invariants.shear_1,
                halo_invariants.shear_2,
            ]
        )
        numerator += float(np.sum(contrast_vector * (halo_vector - base_vector)))
        denominator += float(np.sum(contrast_vector * contrast_vector))
    if denominator <= 0.0:
        raise RuntimeError("Hessian contrast is degenerate")
    return numerator / denominator


def fit_q(data: ClusterData, formula: FormulaDefinition) -> float:
    if formula.fit_objective == "source_scatter":
        return fit_source_scatter_q(data, formula)
    if formula.fit_objective == "local_hessian":
        return fit_hessian_q(data, formula)
    raise ValueError(f"unknown fit objective {formula.fit_objective}")


def source_plane_rms(field, images: pd.DataFrame) -> float:
    total = 0.0
    degrees = 0
    for _family, family_images in images.groupby(
        images.family_id.astype(str), sort=True
    ):
        theta = family_images[["east_arcsec", "north_arcsec"]].to_numpy(float)
        source_redshift = float(family_images.adopted_catalog_redshift.median())
        beta_east, beta_north = ray_shoot(
            field, theta[:, 0], theta[:, 1], source_redshift
        )
        beta = np.column_stack([beta_east, beta_north])
        centered = beta - np.mean(beta, axis=0)
        total += float(np.sum(centered * centered))
        degrees += 2 * (len(theta) - 1)
    return math.sqrt(total / degrees)


def finite_ratio(numerator: float, denominator: float) -> float | None:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0.0:
        return None
    return float(numerator / denominator)


def solar_fractional_slip(q: float, radius_m: float) -> float:
    g_newton = SOLAR_G_M3_S2 / radius_m**2
    radical = math.sqrt(g_newton**2 + 4.0 * g_newton * A0_M_S2)
    aqual_excess = (
        2.0 * g_newton * A0_M_S2 / (radical + g_newton)
    )
    # Phi = 2W - Psi and Psi -> Phi_N in the high-acceleration limit.
    return 2.0 * abs(float(q)) * aqual_excess / g_newton


def main() -> None:
    clusters = load_clusters()
    fit_records: list[dict[str, object]] = []
    family_records: list[dict[str, object]] = []
    folds = (("AS295", "PLCKG287"), ("PLCKG287", "AS295"))

    for train_cluster, test_cluster in folds:
        train = clusters[train_cluster]
        test = clusters[test_cluster]
        fitted: dict[str, tuple[FormulaDefinition, float, object]] = {}
        for formula in FORMULAS:
            q = fit_q(train, formula)
            field = formula_field(test, formula, q)
            fitted[formula.name] = (formula, q, field)
            fit_records.append(
                {
                    "train_cluster": train_cluster,
                    "test_cluster": test_cluster,
                    "formula": formula.name,
                    "fit_objective": formula.fit_objective,
                    "q": q,
                    "train_source_plane_RMS_arcsec": source_plane_rms(
                        formula_field(train, formula, q), train.images
                    ),
                    "test_source_plane_RMS_arcsec": source_plane_rms(
                        field, test.images
                    ),
                    "test_halo_source_plane_RMS_arcsec": source_plane_rms(
                        test.fields["H"], test.images
                    ),
                    "solar_limb_PPN_slip_proxy": solar_fractional_slip(
                        q, SOLAR_RADIUS_M
                    ),
                    "mercury_PPN_slip_proxy": solar_fractional_slip(
                        q, MERCURY_SEMIMAJOR_M
                    ),
                }
            )
        evaluated_fields = {
            "P0707_baseline": test.fields["C"],
            **{name: item[2] for name, item in fitted.items()},
        }
        print(
            f"P0717 train={train_cluster} test={test_cluster} "
            f"families={test.images.family_id.nunique()}",
            flush=True,
        )
        for family_id, images in test.images.groupby(
            test.images.family_id.astype(str), sort=True
        ):
            observed = images[["east_arcsec", "north_arcsec"]].to_numpy(float)
            source_redshift = float(images.adopted_catalog_redshift.median())
            halo_source = profiled_source(test.fields["H"], observed, source_redshift)
            halo_roots = find_lens_roots(
                test.fields["H"],
                halo_source,
                source_redshift,
                bound_arcsec=test.common_bound_arcsec,
                observed_starts_arcsec=observed,
            )
            halo_assignment = assign_observed_roots(
                observed, halo_roots.roots_arcsec
            )
            if halo_assignment.complete:
                threshold = float(
                    np.min(
                        halo_roots.absolute_magnification[
                            halo_assignment.pairs[:, 1]
                        ]
                    )
                )
            else:
                threshold = 0.0
            family_records.append(
                {
                    "train_cluster": train_cluster,
                    "test_cluster": test_cluster,
                    "family_id": family_id,
                    "model": "glafic_v2_compact_halo",
                    "observed_images": len(observed),
                    "global_roots": len(halo_roots.roots_arcsec),
                    "matched_images": halo_assignment.matched_images,
                    "image_RMS_arcsec": halo_assignment.rms_arcsec,
                    "magnification_threshold": threshold,
                    "retained_roots": len(halo_roots.roots_arcsec),
                    "topology_correct": halo_assignment.complete
                    and len(halo_roots.roots_arcsec) == len(observed),
                }
            )
            for name, field in evaluated_fields.items():
                source = profiled_source(field, observed, source_redshift)
                roots = find_lens_roots(
                    field,
                    source,
                    source_redshift,
                    bound_arcsec=test.common_bound_arcsec,
                    observed_starts_arcsec=observed,
                )
                assignment = assign_observed_roots(observed, roots.roots_arcsec)
                retained = roots.roots_arcsec[
                    roots.absolute_magnification >= threshold
                ]
                retained_assignment = assign_observed_roots(observed, retained)
                family_records.append(
                    {
                        "train_cluster": train_cluster,
                        "test_cluster": test_cluster,
                        "family_id": family_id,
                        "model": name,
                        "observed_images": len(observed),
                        "global_roots": len(roots.roots_arcsec),
                        "matched_images": assignment.matched_images,
                        "image_RMS_arcsec": assignment.rms_arcsec,
                        "magnification_threshold": threshold,
                        "retained_roots": len(retained),
                        "topology_correct": retained_assignment.complete
                        and len(retained) == len(observed),
                    }
                )

    fits = pd.DataFrame.from_records(fit_records)
    families = pd.DataFrame.from_records(family_records)
    score_records: list[dict[str, object]] = []
    for (train, test, model), block in families.groupby(
        ["train_cluster", "test_cluster", "model"], sort=True
    ):
        halo = families[
            (families.train_cluster == train)
            & (families.test_cluster == test)
            & (families.model == "glafic_v2_compact_halo")
        ].set_index("family_id")
        ratios = []
        for row in block.itertuples(index=False):
            halo_rms = float(halo.loc[row.family_id, "image_RMS_arcsec"])
            ratio = finite_ratio(float(row.image_RMS_arcsec), halo_rms)
            if ratio is not None:
                ratios.append(ratio)
        total_images = int(block.observed_images.sum())
        score_records.append(
            {
                "train_cluster": train,
                "test_cluster": test,
                "model": model,
                "families": len(block),
                "root_convergence_fraction": float(
                    block.matched_images.sum() / total_images
                ),
                "finite_family_fraction": float(
                    np.isfinite(block.image_RMS_arcsec).mean()
                ),
                "median_finite_RMS_arcsec": float(
                    block.loc[
                        np.isfinite(block.image_RMS_arcsec), "image_RMS_arcsec"
                    ].median()
                ),
                "median_RMS_ratio_to_halo": (
                    float(np.median(ratios)) if ratios else np.nan
                ),
                "topology_correct_fraction": float(block.topology_correct.mean()),
            }
        )
    scores = pd.DataFrame.from_records(score_records)

    q_agreement_records = []
    for formula, block in fits.groupby("formula", sort=True):
        q_values = block.q.to_numpy(float)
        disagreement = float(
            abs(q_values[0] - q_values[1]) / max(np.mean(np.abs(q_values)), 1.0e-12)
        )
        q_agreement_records.append(
            {
                "formula": formula,
                "q_AS295_fit": float(block.loc[block.train_cluster == "AS295", "q"].iloc[0]),
                "q_PLCKG287_fit": float(
                    block.loc[block.train_cluster == "PLCKG287", "q"].iloc[0]
                ),
                "q_disagreement_fraction": disagreement,
                "q_transfer_gate": disagreement <= MAXIMUM_Q_DISAGREEMENT_FRACTION,
                "solar_gate": bool(
                    (block.solar_limb_PPN_slip_proxy < SOLAR_PPN_LIMIT).all()
                ),
            }
        )
    q_agreement = pd.DataFrame.from_records(q_agreement_records)

    transfer_models = [formula.name for formula in FORMULAS]
    gate_records = []
    for model in transfer_models:
        model_scores = scores[scores.model == model]
        q_row = q_agreement[q_agreement.formula == model].iloc[0]
        gates = {
            "q_transfer": bool(q_row.q_transfer_gate),
            "solar": bool(q_row.solar_gate),
            "root_convergence": bool(
                (model_scores.root_convergence_fraction >= MINIMUM_ROOT_CONVERGENCE_FRACTION).all()
            ),
            "finite_families": bool(
                (model_scores.finite_family_fraction >= MINIMUM_FINITE_FAMILY_FRACTION).all()
            ),
            "topology": bool(
                (model_scores.topology_correct_fraction >= MINIMUM_TOPOLOGY_CORRECT_FRACTION).all()
            ),
            "RMS_ratio": bool(
                (
                    model_scores.median_RMS_ratio_to_halo
                    <= MAXIMUM_MEDIAN_RMS_RATIO_TO_HALO
                ).all()
            ),
        }
        gate_records.append(
            {
                "model": model,
                **gates,
                "all_gates": all(gates.values()),
            }
        )
    gates = pd.DataFrame.from_records(gate_records)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    fits.to_csv(OUTPUT / "cross_cluster_parameter_fits.csv", index=False)
    families.to_csv(OUTPUT / "raw_family_transfer_scores.csv", index=False)
    scores.to_csv(OUTPUT / "raw_cluster_transfer_scores.csv", index=False)
    q_agreement.to_csv(OUTPUT / "parameter_transfer_gates.csv", index=False)
    gates.to_csv(OUTPUT / "formula_rejection_gates.csv", index=False)

    plot = scores[scores.model.isin(["P0707_baseline", *transfer_models])].copy()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for test_cluster, marker in [("AS295", "o"), ("PLCKG287", "s")]:
        block = plot[plot.test_cluster == test_cluster]
        axes[0].scatter(
            block.root_convergence_fraction,
            block.topology_correct_fraction,
            label=test_cluster,
            marker=marker,
            s=60,
        )
        axes[1].scatter(
            block.finite_family_fraction,
            block.median_RMS_ratio_to_halo,
            label=test_cluster,
            marker=marker,
            s=60,
        )
    axes[0].axvline(MINIMUM_ROOT_CONVERGENCE_FRACTION, color="black", linestyle="--")
    axes[0].axhline(MINIMUM_TOPOLOGY_CORRECT_FRACTION, color="black", linestyle="--")
    axes[0].set(xlabel="root convergence", ylabel="topology-correct families")
    axes[1].axvline(MINIMUM_FINITE_FAMILY_FRACTION, color="black", linestyle="--")
    axes[1].axhline(MAXIMUM_MEDIAN_RMS_RATIO_TO_HALO, color="black", linestyle="--")
    axes[1].set(xlabel="finite-RMS families", ylabel="median RMS / compact halo")
    axes[0].legend()
    figure.savefig(OUTPUT / "screened_contrast_transfer.png", dpi=180)
    plt.close(figure)

    survivors = gates.loc[gates.all_gates, "model"].tolist()
    best_root = (
        scores[scores.model.isin(transfer_models)]
        .groupby("model")
        .root_convergence_fraction.mean()
        .sort_values(ascending=False)
    )
    report = {
        "stage": "P0717",
        "status": "pass" if survivors else "fail_no_formula_passed_all_transfer_gates",
        "evaluation_kind": "spent_two_cluster_cross_transfer",
        "sample_is_spent": True,
        "per_family_gravity_parameters": 0,
        "per_cluster_gravity_parameters_at_test_time": 0,
        "formula": "W = Phi_base + q (Phi_nonlinear - Phi_N)",
        "matter_potential": "Psi remains the frozen RAR/coherent matter law",
        "thresholds": {
            "maximum_q_disagreement_fraction": MAXIMUM_Q_DISAGREEMENT_FRACTION,
            "minimum_root_convergence_fraction": MINIMUM_ROOT_CONVERGENCE_FRACTION,
            "minimum_finite_family_fraction": MINIMUM_FINITE_FAMILY_FRACTION,
            "minimum_topology_correct_fraction": MINIMUM_TOPOLOGY_CORRECT_FRACTION,
            "maximum_median_RMS_ratio_to_halo": MAXIMUM_MEDIAN_RMS_RATIO_TO_HALO,
            "solar_PPN_limit": SOLAR_PPN_LIMIT,
        },
        "survivors": survivors,
        "best_mean_root_convergence_model": str(best_root.index[0]),
        "best_mean_root_convergence_fraction": float(best_root.iloc[0]),
        "parameter_transfer": q_agreement.to_dict(orient="records"),
        "solar_interpretation": (
            "The AQUAL/QUMOND contrast tends to zero as a0/g in high acceleration; "
            "Psi is unchanged, so the current Mercury dynamics proxy remains exactly unchanged."
        ),
        "claim_boundary": [
            "P0717 was designed after inspecting both clusters and is not blind validation.",
            "The compact-halo comparator is constrained by the same arcs and is not pixelwise truth.",
            "Passing the Solar proxy is not a substitute for a covariant PPN derivation.",
            "A successful algebraic Weyl potential still requires a field equation and stability proof.",
        ],
    }
    (OUTPUT / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# P0717 screened-contrast cross-cluster transfer",
        "",
        f"Status: **{report['status']}**",
        "",
        (
            f"Best mean root convergence: {report['best_mean_root_convergence_model']} "
            f"({report['best_mean_root_convergence_fraction']:.3f})."
        ),
        "",
        "| Formula | q transfer | Solar | Roots | Finite RMS | Topology | RMS ratio | All |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for row in gates.itertuples(index=False):
        values = [
            row.q_transfer,
            row.solar,
            row.root_convergence,
            row.finite_families,
            row.topology,
            row.RMS_ratio,
            row.all_gates,
        ]
        lines.append(
            f"| {row.model} | " + " | ".join("yes" if value else "no" for value in values) + " |"
        )
    lines.extend(
        [
            "",
            "This is a rejection/prioritization test on a spent sample, not validation.",
        ]
    )
    summary = "\n".join(lines) + "\n"
    (OUTPUT / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
