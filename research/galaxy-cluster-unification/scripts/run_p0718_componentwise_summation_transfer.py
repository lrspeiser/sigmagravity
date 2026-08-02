#!/usr/bin/env python3
"""Test nonlinear-before-summation member fields with raw cluster lensing."""

from __future__ import annotations

import json
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
from astropy.cosmology import Planck18

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0715_sky_lensing_engine_validation import (
    BARYON_MAPS,
    distance_ratio,
)
from run_p0717_screened_contrast_transfer import (
    ClusterData,
    load_clusters,
    source_plane_rms,
)

from voidscreen.cluster_baryon_maps import (
    f160_stellar_mass_msun,
    strict_f160_members,
)
from voidscreen.componentwise_mond_lensing import (
    componentwise_simple_mond_excess_deflection,
)
from voidscreen.gravity_arc_tomography import read_relics_catalog
from voidscreen.sky_lensing import (
    GridSkyDeflectionField,
    LinearCombinationSkyDeflectionField,
    assign_observed_roots,
    find_lens_roots,
    profiled_source,
)
from voidscreen.thin_lens import thin_lens_deflection_from_surface_density

OUTPUT = ROOT / "results/p0718_componentwise_summation_transfer"
P0717 = ROOT / "results/p0717_screened_contrast_transfer"
MAP_CONFIG = ROOT / "configs/p0641_registered_cluster_baryon_maps.json"
RAW_BARYONS = ROOT / "data/raw/p0633_relics_baryons"
NOMINAL_SOFTENING_KPC = 3.515625
MINIMUM_ROOT_CONVERGENCE_FRACTION = 0.75
MINIMUM_FINITE_FAMILY_FRACTION = 0.50
MINIMUM_TOPOLOGY_CORRECT_FRACTION = 0.25
MAXIMUM_MEDIAN_RMS_RATIO_TO_HALO = 3.0
MAXIMUM_Q_DISAGREEMENT_FRACTION = 0.25


@dataclass(frozen=True)
class Members:
    east_arcsec: np.ndarray
    north_arcsec: np.ndarray
    mass_msun: np.ndarray
    kpc_per_arcsec: float
    center: SkyCoord


def load_members(data: ClusterData) -> Members:
    config = json.loads(MAP_CONFIG.read_text(encoding="utf-8"))
    population = config["stellar_population"]
    with np.load(BARYON_MAPS / f"{data.cluster}_baryons.npz") as maps:
        center = SkyCoord(
            float(maps["center_ra_deg"]) * u.deg,
            float(maps["center_dec_deg"]) * u.deg,
        )
    catalog_path = next((RAW_BARYONS / data.cluster / "hst").glob("*_cat.txt"))
    catalog = read_relics_catalog(catalog_path)
    selected, flux = strict_f160_members(catalog, data.lens_redshift)
    mass = f160_stellar_mass_msun(
        flux[selected],
        redshift=data.lens_redshift,
        mass_to_light_solar=float(population["nominal_mass_to_light_solar"]),
        solar_absolute_ab_magnitude=float(population["solar_absolute_ab_magnitude"]),
    )
    coordinates = SkyCoord(
        catalog.loc[selected, "RA"].to_numpy(float) * u.deg,
        catalog.loc[selected, "Dec"].to_numpy(float) * u.deg,
    )
    east, north = center.spherical_offsets_to(coordinates)
    scale = float(
        Planck18.kpc_proper_per_arcmin(data.lens_redshift).value / 60.0
    )
    return Members(
        east.to_value(u.arcsec),
        north.to_value(u.arcsec),
        mass,
        scale,
        center,
    )


def component_field(
    data: ClusterData,
    members: Members,
    *,
    softening_kpc: float,
    mass_scale: float,
) -> GridSkyDeflectionField:
    axis_kpc = np.linspace(-450.0, 450.0, 257)
    axis_arcsec = axis_kpc / members.kpc_per_arcsec
    east, north = np.meshgrid(axis_arcsec, axis_arcsec, indexing="xy")
    alpha_east, alpha_north = componentwise_simple_mond_excess_deflection(
        east,
        north,
        members.east_arcsec,
        members.north_arcsec,
        members.mass_msun * float(mass_scale),
        kpc_per_arcsec=members.kpc_per_arcsec,
        distance_ratio=1.0,
        softening_kpc=float(softening_kpc),
    )
    return GridSkyDeflectionField(
        north_axis_arcsec=axis_arcsec,
        east_axis_arcsec=axis_arcsec,
        alpha_east_ratio_one_arcsec=alpha_east,
        alpha_north_ratio_one_arcsec=alpha_north,
        distance_ratio=lambda source_redshift: distance_ratio(
            data.lens_redshift, source_redshift
        ),
    )


def smooth_contrast(data: ClusterData):
    return LinearCombinationSkyDeflectionField(
        (data.fields["A"], data.fields["N"]), (1.0, -1.0)
    )


def combined_contrast(data: ClusterData, component):
    return LinearCombinationSkyDeflectionField(
        (smooth_contrast(data), component), (1.0, 1.0)
    )


def fit_q(data: ClusterData, contrast) -> float:
    numerator = 0.0
    denominator = 0.0
    for _family, images in data.images.groupby(
        data.images.family_id.astype(str), sort=True
    ):
        observed = images[["east_arcsec", "north_arcsec"]].to_numpy(float)
        source_redshift = float(images.adopted_catalog_redshift.median())
        alpha_east, alpha_north = data.fields["N"].alpha(
            observed[:, 0], observed[:, 1], source_redshift
        )
        delta_east, delta_north = contrast.alpha(
            observed[:, 0], observed[:, 1], source_redshift
        )
        target = observed - np.column_stack([alpha_east, alpha_north])
        delta = np.column_stack([delta_east, delta_north])
        target -= np.mean(target, axis=0)
        delta -= np.mean(delta, axis=0)
        numerator += float(np.sum(target * delta))
        denominator += float(np.sum(delta * delta))
    return numerator / denominator


def candidate_field(data: ClusterData, contrast, q: float):
    return LinearCombinationSkyDeflectionField(
        (data.fields["N"], contrast), (1.0, float(q))
    )


def high_resolution_newtonian_field(
    data: ClusterData,
    members: Members,
) -> GridSkyDeflectionField:
    with np.load(BARYON_MAPS / f"{data.cluster}_baryons.npz") as maps:
        axis = maps["axis_kpc"].astype(float)
        selected = np.abs(axis) <= 450.0001
        cropped_axis = axis[selected]
        surface = maps["baryon_surface_density_msun_kpc2"][
            np.ix_(selected, selected)
        ].astype(float)
    deflection = thin_lens_deflection_from_surface_density(
        surface, float(cropped_axis[1] - cropped_axis[0])
    )
    return GridSkyDeflectionField(
        north_axis_arcsec=cropped_axis / members.kpc_per_arcsec,
        east_axis_arcsec=cropped_axis / members.kpc_per_arcsec,
        alpha_east_ratio_one_arcsec=deflection.alpha_east_arcsec,
        alpha_north_ratio_one_arcsec=deflection.alpha_north_arcsec,
        distance_ratio=lambda source_redshift: distance_ratio(
            data.lens_redshift, source_redshift
        ),
    )


def resolution_audit(
    data: ClusterData,
    members: Members,
) -> dict[str, float | str]:
    high = high_resolution_newtonian_field(data, members)
    differences = []
    fractional = []
    for _family, images in data.images.groupby(
        data.images.family_id.astype(str), sort=True
    ):
        observed = images[["east_arcsec", "north_arcsec"]].to_numpy(float)
        source_redshift = float(images.adopted_catalog_redshift.median())
        high_east, high_north = high.alpha(
            observed[:, 0], observed[:, 1], source_redshift
        )
        low_east, low_north = data.fields["N"].alpha(
            observed[:, 0], observed[:, 1], source_redshift
        )
        difference = np.hypot(high_east - low_east, high_north - low_north)
        low_magnitude = np.hypot(low_east, low_north)
        differences.extend(difference)
        fractional.extend(difference / np.maximum(low_magnitude, 1.0e-12))
    contrast = smooth_contrast(data)
    # Replace only the Newtonian resolution while retaining the same nonlinear contrast.
    numerator = 0.0
    denominator = 0.0
    for _family, images in data.images.groupby(
        data.images.family_id.astype(str), sort=True
    ):
        observed = images[["east_arcsec", "north_arcsec"]].to_numpy(float)
        source_redshift = float(images.adopted_catalog_redshift.median())
        high_east, high_north = high.alpha(
            observed[:, 0], observed[:, 1], source_redshift
        )
        delta_east, delta_north = contrast.alpha(
            observed[:, 0], observed[:, 1], source_redshift
        )
        target = observed - np.column_stack([high_east, high_north])
        delta = np.column_stack([delta_east, delta_north])
        target -= np.mean(target, axis=0)
        delta -= np.mean(delta, axis=0)
        numerator += float(np.sum(target * delta))
        denominator += float(np.sum(delta * delta))
    q = numerator / denominator
    field = LinearCombinationSkyDeflectionField((high, contrast), (1.0, q))
    return {
        "cluster": data.cluster,
        "median_high_minus_low_arc_deflection_arcsec": float(np.median(differences)),
        "maximum_high_minus_low_arc_deflection_arcsec": float(np.max(differences)),
        "median_fractional_arc_deflection_change": float(np.median(fractional)),
        "same_cluster_fitted_q": q,
        "same_cluster_source_plane_RMS_arcsec": source_plane_rms(field, data.images),
    }


def main() -> None:
    clusters = load_clusters()
    members = {cluster: load_members(data) for cluster, data in clusters.items()}
    component_fields: dict[tuple[str, float, float], object] = {}
    sensitivity_records: list[dict[str, object]] = []
    sensitivity_specs = [
        ("softening_half", 0.5 * NOMINAL_SOFTENING_KPC, 1.0),
        ("nominal", NOMINAL_SOFTENING_KPC, 1.0),
        ("softening_double", 2.0 * NOMINAL_SOFTENING_KPC, 1.0),
        ("stellar_mass_low", NOMINAL_SOFTENING_KPC, 0.5 / 0.8),
        ("stellar_mass_high", NOMINAL_SOFTENING_KPC, 1.1 / 0.8),
    ]
    for variant, softening, mass_scale in sensitivity_specs:
        for cluster, data in clusters.items():
            key = (cluster, softening, mass_scale)
            if key not in component_fields:
                component_fields[key] = component_field(
                    data,
                    members[cluster],
                    softening_kpc=softening,
                    mass_scale=mass_scale,
                )
            contrast = combined_contrast(data, component_fields[key])
            q = fit_q(data, contrast)
            field = candidate_field(data, contrast, q)
            sensitivity_records.append(
                {
                    "variant": variant,
                    "cluster": cluster,
                    "softening_kpc": softening,
                    "stellar_mass_scale": mass_scale,
                    "fitted_q": q,
                    "same_cluster_source_plane_RMS_arcsec": source_plane_rms(
                        field, data.images
                    ),
                }
            )
    sensitivities = pd.DataFrame.from_records(sensitivity_records)
    nominal = sensitivities[sensitivities.variant == "nominal"].set_index("cluster")
    nominal_component = {
        cluster: component_fields[(cluster, NOMINAL_SOFTENING_KPC, 1.0)]
        for cluster in clusters
    }
    q_by_train = nominal.fitted_q.to_dict()
    q_values = np.asarray(list(q_by_train.values()), dtype=float)
    q_disagreement = float(
        abs(q_values[0] - q_values[1]) / np.mean(np.abs(q_values))
    )

    raw_reference = pd.read_csv(P0717 / "raw_family_transfer_scores.csv")
    family_records: list[dict[str, object]] = []
    for train_cluster, test_cluster in (("AS295", "PLCKG287"), ("PLCKG287", "AS295")):
        data = clusters[test_cluster]
        q = float(q_by_train[train_cluster])
        contrast = combined_contrast(data, nominal_component[test_cluster])
        field = candidate_field(data, contrast, q)
        print(
            f"P0718 train={train_cluster} test={test_cluster} q={q:.6f}",
            flush=True,
        )
        for family_id, images in data.images.groupby(
            data.images.family_id.astype(str), sort=True
        ):
            observed = images[["east_arcsec", "north_arcsec"]].to_numpy(float)
            source_redshift = float(images.adopted_catalog_redshift.median())
            source = profiled_source(field, observed, source_redshift)
            roots = find_lens_roots(
                field,
                source,
                source_redshift,
                bound_arcsec=data.common_bound_arcsec,
                observed_starts_arcsec=observed,
            )
            assignment = assign_observed_roots(observed, roots.roots_arcsec)
            reference = raw_reference[
                (raw_reference.train_cluster == train_cluster)
                & (raw_reference.test_cluster == test_cluster)
                & (raw_reference.family_id.astype(str) == str(family_id))
                & (raw_reference.model == "glafic_v2_compact_halo")
            ]
            if len(reference) != 1:
                raise RuntimeError("missing P0717 halo reference")
            halo = reference.iloc[0]
            threshold = float(halo.magnification_threshold)
            retained = roots.roots_arcsec[
                roots.absolute_magnification >= threshold
            ]
            retained_assignment = assign_observed_roots(observed, retained)
            ratio = (
                float(assignment.rms_arcsec / halo.image_RMS_arcsec)
                if np.isfinite(assignment.rms_arcsec)
                and np.isfinite(halo.image_RMS_arcsec)
                and float(halo.image_RMS_arcsec) > 0.0
                else np.nan
            )
            family_records.append(
                {
                    "train_cluster": train_cluster,
                    "test_cluster": test_cluster,
                    "family_id": family_id,
                    "q": q,
                    "observed_images": len(observed),
                    "global_roots": len(roots.roots_arcsec),
                    "matched_images": assignment.matched_images,
                    "image_RMS_arcsec": assignment.rms_arcsec,
                    "RMS_ratio_to_halo": ratio,
                    "magnification_threshold": threshold,
                    "retained_roots": len(retained),
                    "topology_correct": retained_assignment.complete
                    and len(retained) == len(observed),
                }
            )
    families = pd.DataFrame.from_records(family_records)
    score_records = []
    for (train, test), block in families.groupby(
        ["train_cluster", "test_cluster"], sort=True
    ):
        total_images = int(block.observed_images.sum())
        finite = np.isfinite(block.image_RMS_arcsec)
        ratios = block.RMS_ratio_to_halo.dropna()
        score_records.append(
            {
                "train_cluster": train,
                "test_cluster": test,
                "root_convergence_fraction": float(
                    block.matched_images.sum() / total_images
                ),
                "finite_family_fraction": float(finite.mean()),
                "median_finite_RMS_arcsec": float(
                    block.loc[finite, "image_RMS_arcsec"].median()
                ),
                "median_RMS_ratio_to_halo": (
                    float(ratios.median()) if len(ratios) else np.nan
                ),
                "topology_correct_fraction": float(block.topology_correct.mean()),
            }
        )
    scores = pd.DataFrame.from_records(score_records)
    resolution = pd.DataFrame.from_records(
        [resolution_audit(data, members[cluster]) for cluster, data in clusters.items()]
    )

    gates = {
        "q_transfer": q_disagreement <= MAXIMUM_Q_DISAGREEMENT_FRACTION,
        "root_convergence": bool(
            (scores.root_convergence_fraction >= MINIMUM_ROOT_CONVERGENCE_FRACTION).all()
        ),
        "finite_families": bool(
            (scores.finite_family_fraction >= MINIMUM_FINITE_FAMILY_FRACTION).all()
        ),
        "topology": bool(
            (scores.topology_correct_fraction >= MINIMUM_TOPOLOGY_CORRECT_FRACTION).all()
        ),
        "RMS_ratio": bool(
            (scores.median_RMS_ratio_to_halo <= MAXIMUM_MEDIAN_RMS_RATIO_TO_HALO).all()
        ),
    }
    passed = all(gates.values())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sensitivities.to_csv(OUTPUT / "member_input_sensitivities.csv", index=False)
    resolution.to_csv(OUTPUT / "resolution_audit.csv", index=False)
    families.to_csv(OUTPUT / "raw_family_transfer_scores.csv", index=False)
    scores.to_csv(OUTPUT / "raw_cluster_transfer_scores.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    for cluster, block in sensitivities.groupby("cluster"):
        axes[0].plot(
            block.variant,
            block.fitted_q,
            marker="o",
            label=cluster,
        )
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].set(ylabel="same-cluster fitted q", xlabel="member input sensitivity")
    axes[0].legend()
    axes[1].scatter(
        scores.root_convergence_fraction,
        scores.median_RMS_ratio_to_halo,
        s=80,
    )
    for row in scores.itertuples(index=False):
        axes[1].annotate(row.test_cluster, (row.root_convergence_fraction, row.median_RMS_ratio_to_halo))
    axes[1].axvline(MINIMUM_ROOT_CONVERGENCE_FRACTION, color="black", linestyle="--")
    axes[1].axhline(MAXIMUM_MEDIAN_RMS_RATIO_TO_HALO, color="black", linestyle="--")
    axes[1].set(xlabel="root convergence", ylabel="median RMS / compact halo")
    figure.savefig(OUTPUT / "componentwise_transfer.png", dpi=180)
    plt.close(figure)

    report = {
        "stage": "P0718",
        "status": "pass" if passed else "fail_raw_transfer_gates",
        "evaluation_kind": "spent_componentwise_nonlinear_before_summation_transfer",
        "sample_is_spent": True,
        "formula": "W = Phi_N + q[(Phi_AQUAL,total-Phi_N,total) + sum_i(Phi_AQUAL,i-Phi_N,i)]",
        "members": {
            cluster: {
                "count": len(value.mass_msun),
                "stellar_mass_msun": float(np.sum(value.mass_msun)),
            }
            for cluster, value in members.items()
        },
        "q_by_training_cluster": {key: float(value) for key, value in q_by_train.items()},
        "q_disagreement_fraction": q_disagreement,
        "gates": gates,
        "all_gates": passed,
        "scores": scores.to_dict(orient="records"),
        "resolution_finding": (
            "A 257x257 thin-lens reconstruction changes arc deflections at the sub-arcsecond to arcsecond level but does not by itself repair the cross-cluster source mapping."
        ),
        "interpretation": (
            "Nonlinear-before-summation is the strongest root-completeness improvement in this sequence, especially on PLCKG287, but the asymmetric transfer and large image RMS reject this implementation as a universal raw-lensing solution."
        ),
        "claim_boundary": [
            "This ordering is a new diagnostic ansatz, not AQUAL or QUMOND.",
            "The member catalog and stellar masses were frozen from baryonic photometry, but both lensing clusters were already unsealed.",
            "The component model treats members as softened spherical sources and does not model three-dimensional depth or intracluster light.",
            "Galaxy dynamics remain unchanged because only the Weyl potential is altered; a covariant field equation is still absent.",
        ],
    }
    (OUTPUT / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    summary = "\n".join(
        [
            "# P0718 componentwise nonlinear summation transfer",
            "",
            f"Status: **{report['status']}**",
            "",
            f"The two fitted q values differ by {q_disagreement:.1%}.",
            "",
            "| Train to test | Root completeness | Finite families | Topology | Median RMS / halo |",
            "|---|---:|---:|---:|---:|",
            *[
                f"| {row.train_cluster} to {row.test_cluster} | {row.root_convergence_fraction:.3f} | {row.finite_family_fraction:.3f} | {row.topology_correct_fraction:.3f} | {row.median_RMS_ratio_to_halo:.3f} |"
                for row in scores.itertuples(index=False)
            ],
            "",
            "The ordering hypothesis improves multiplicity but fails universal image-position accuracy.",
        ]
    ) + "\n"
    (OUTPUT / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
