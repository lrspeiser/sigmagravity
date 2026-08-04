from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_p0714_ready_subset_raw_lensing as raw_lensing

from voidscreen.sigma_nonmetricity import (
    simple_nu,
    trace_action_derivative,
    trace_action_primitive,
    trace_nonmetricity,
    trace_split_spherical_accelerations,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SigmaV2WeylField(raw_lensing.DeflectionField):
    """Weyl deflection fixed by W=(Psi_QUMOND+Phi_Newton)/2."""

    def __init__(
        self,
        newtonian: raw_lensing.DeflectionField,
        qumond: raw_lensing.DeflectionField,
    ) -> None:
        if newtonian.cluster != qumond.cluster or newtonian.lens_redshift != qumond.lens_redshift:
            raise ValueError("the component fields must describe the same lens")
        self.newtonian = newtonian
        self.qumond = qumond
        super().__init__(
            newtonian.cluster,
            newtonian.lens_redshift,
            min(newtonian.half_extent_arcsec, qumond.half_extent_arcsec),
        )

    def alpha(
        self,
        east_arcsec,
        north_arcsec,
        source_redshift: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        newtonian_east, newtonian_north = self.newtonian.alpha(
            east_arcsec, north_arcsec, source_redshift
        )
        qumond_east, qumond_north = self.qumond.alpha(
            east_arcsec, north_arcsec, source_redshift
        )
        return (
            0.5 * (newtonian_east + qumond_east),
            0.5 * (newtonian_north + qumond_north),
        )


def raw_cluster_audit(config: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    inputs = config["spent_empirical_inputs"]
    readiness = json.loads((ROOT / inputs["readiness_report"]).read_text(encoding="utf-8"))
    if readiness["ready_clusters"] != 2:
        raise RuntimeError("Sigma v2 expects the frozen two-cluster P0713 ready subset")
    ready = [row["cluster"] for row in readiness["cluster_rows"] if row["ready"]]
    catalog = pd.read_csv(ROOT / inputs["image_catalog"])
    catalog = catalog[
        catalog.secure_image.astype(str).str.lower().eq("true")
        & catalog.cluster.isin(ready)
    ].copy()

    family_records: list[dict[str, object]] = []
    cluster_records: list[dict[str, object]] = []
    for cluster in ready:
        with np.load(ROOT / inputs["baryon_maps"] / f"{cluster}_baryons.npz") as data:
            center = SkyCoord(
                float(data["center_ra_deg"]) * u.deg,
                float(data["center_dec_deg"]) * u.deg,
            )
            lens_redshift = float(data["redshift"])
        block = catalog[catalog.cluster == cluster].copy()
        sky = SkyCoord(block.ra_deg.to_numpy() * u.deg, block.dec_deg.to_numpy() * u.deg)
        east, north = center.spherical_offsets_to(sky)
        block["east_arcsec"] = east.to_value(u.arcsec)
        block["north_arcsec"] = north.to_value(u.arcsec)
        partition = raw_lensing.family_partition(
            cluster, sorted(block.family_id.astype(str).unique())
        )

        # P0641 stores array rows north and columns east.  This fresh theory
        # audit uses the repaired coordinate contract rather than inheriting
        # the intentionally frozen P0708 axis mistake.
        newtonian = raw_lensing.FrozenGridField(
            cluster, lens_redshift, "baryon_only_GR", axis_repaired=True
        )
        qumond = raw_lensing.FrozenGridField(
            cluster, lens_redshift, "QUMOND_simple_nu_diagnostic", axis_repaired=True
        )
        candidate = SigmaV2WeylField(newtonian, qumond)
        halo = raw_lensing.GlaficField(cluster, lens_redshift, center)
        bound = min(candidate.half_extent_arcsec, halo.half_extent_arcsec) * 0.965

        for family_id, images in block.groupby(block.family_id.astype(str), sort=True):
            source_redshift = float(images.adopted_catalog_redshift.median())
            candidate_source = raw_lensing.profiled_source(candidate, images, source_redshift)
            candidate_roots, candidate_magnification = raw_lensing.find_roots(
                candidate, candidate_source, source_redshift, images, bound
            )
            _candidate_pairs, candidate_rms, candidate_matched = raw_lensing.assignment(
                images, candidate_roots
            )

            halo_source = raw_lensing.profiled_source(halo, images, source_redshift)
            halo_roots, halo_magnification = raw_lensing.find_roots(
                halo, halo_source, source_redshift, images, bound
            )
            halo_pairs, halo_rms, halo_matched = raw_lensing.assignment(images, halo_roots)
            if halo_matched == len(images):
                threshold = float(np.min(halo_magnification[halo_pairs[:, 1]]))
            else:
                threshold = 0.0
            retained = candidate_roots[candidate_magnification >= threshold]
            _, _, retained_matched = raw_lensing.assignment(images, retained)
            topology_correct = retained_matched == len(images) and len(retained) == len(images)
            family_records.append(
                {
                    "cluster": cluster,
                    "partition": partition[family_id],
                    "family_id": family_id,
                    "source_redshift": source_redshift,
                    "observed_images": len(images),
                    "global_roots": len(candidate_roots),
                    "retained_roots": len(retained),
                    "matched_images": candidate_matched,
                    "image_RMS_arcsec": candidate_rms,
                    "topology_correct": topology_correct,
                    "halo_global_roots": len(halo_roots),
                    "halo_matched_images": halo_matched,
                    "halo_image_RMS_arcsec": halo_rms,
                }
            )

        family_frame = pd.DataFrame.from_records(family_records)
        holdout = family_frame[
            (family_frame.cluster == cluster) & (family_frame.partition == "holdout")
        ]
        total_images = int(holdout.observed_images.sum())
        matched_images = int(holdout.matched_images.sum())
        complete = bool((holdout.matched_images == holdout.observed_images).all())
        if complete:
            rms = float(
                np.sqrt(
                    np.average(
                        np.square(holdout.image_RMS_arcsec), weights=holdout.observed_images
                    )
                )
            )
        else:
            rms = float("inf")
        cluster_records.append(
            {
                "cluster": cluster,
                "heldout_families": len(holdout),
                "heldout_images": total_images,
                "matched_images": matched_images,
                "root_convergence_fraction": matched_images / total_images,
                "heldout_image_RMS_arcsec": rms,
                "all_heldout_topologies_correct": bool(holdout.topology_correct.all()),
            }
        )
    return pd.DataFrame.from_records(family_records), pd.DataFrame.from_records(cluster_records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the Sigma v2 trace-nonmetricity action.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v2_trace_nonmetricity_cycle.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v2_trace_nonmetricity_cycle",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    gates = config["gates"]
    a_sigma = float(config["parameters"]["a_sigma_m_s2"])

    rng = np.random.default_rng(8201)
    grad_psi = rng.normal(size=(8192, 3))
    grad_phi = rng.normal(size=(8192, 3))
    trace_expected = 4.0 * np.sum(np.square(grad_phi), axis=1)
    trace_error = float(np.max(np.abs(trace_nonmetricity(grad_psi, grad_phi) - trace_expected)))

    y_squared = np.geomspace(1e-10, 1e8, 6000)
    step = 1e-5
    numerical_derivative = (
        trace_action_primitive(y_squared * np.exp(step))
        - trace_action_primitive(y_squared * np.exp(-step))
    ) / (2.0 * step * y_squared)
    analytic_derivative = trace_action_derivative(y_squared)
    derivative_error = float(
        np.max(
            np.abs(numerical_derivative - analytic_derivative)
            / np.maximum(np.abs(analytic_derivative), 1e-30)
        )
    )

    deep_gbar = float(gates["deep_gbar_over_a_sigma"]) * a_sigma
    deep = trace_split_spherical_accelerations(deep_gbar, a_sigma)
    deep_target = np.sqrt(deep_gbar * a_sigma)
    deep_matter_error = abs(float(deep["matter_psi"]) / deep_target - 1.0)
    high_gbar = float(gates["high_gbar_over_a_sigma"]) * a_sigma
    high = trace_split_spherical_accelerations(high_gbar, a_sigma)
    high_matter_correction = float(high["matter_psi"]) / high_gbar - 1.0
    deep_lensing_to_matter = float(deep["photon_weyl"] / deep["matter_psi"])

    galaxy_path = ROOT / config["spent_empirical_inputs"]["galaxy_report"]
    galaxy = json.loads(galaxy_path.read_text(encoding="utf-8"))
    scores = galaxy["sample_RMSE_km_s"]
    galaxy_rmse = float(scores[config["spent_empirical_inputs"]["galaxy_comparator"]])
    best_mond = min(float(scores["AQUAL_simple_mu_3D"]), float(scores["QUMOND_simple_nu_3D"]))
    galaxy_ratio = galaxy_rmse / best_mond

    families, clusters = raw_cluster_audit(config)
    minimum_root_fraction = float(clusters.root_convergence_fraction.min())
    all_topologies_correct = bool(clusters.all_heldout_topologies_correct.all())

    mathematical_pass = bool(
        trace_error <= gates["trace_identity_max_absolute_error"]
        and derivative_error <= gates["action_derivative_max_relative_error"]
        and deep_matter_error <= gates["deep_matter_relative_error_max"]
        and high_matter_correction <= gates["high_matter_fractional_correction_max"]
        and np.all(simple_nu(np.geomspace(1e-12, 1e12, 10000)) > 0.0)
    )
    galaxy_pass = bool(galaxy_ratio <= gates["galaxy_RMSE_ratio_to_best_fixed_MOND_max"])
    raw_cluster_pass = bool(
        minimum_root_fraction >= gates["raw_cluster_root_convergence_fraction_min"]
        and all_topologies_correct
    )
    known_reduction = True
    advances = bool(mathematical_pass and galaxy_pass and raw_cluster_pass and not known_reduction)

    args.output.mkdir(parents=True, exist_ok=True)
    families.to_csv(args.output / "family_scores.csv", index=False)
    clusters.to_csv(args.output / "cluster_scores.csv", index=False)
    report = {
        "status": "completed Sigma v2 trace-nonmetricity action audit",
        "model_id": config["model_id"],
        "input_hashes": {
            "config": sha256(args.config),
            "galaxy_report": sha256(galaxy_path),
            "readiness_report": sha256(
                ROOT / config["spent_empirical_inputs"]["readiness_report"]
            ),
            "image_catalog": sha256(ROOT / config["spent_empirical_inputs"]["image_catalog"]),
        },
        "postulate_audit": {
            "material_sources": ["baryonic stress-energy"],
            "physical_matter_metrics": 1,
            "global_physical_parameters": 1,
            "per_object_gravity_parameters": 0,
            "lensing_only_parameters": 0,
            "freely_initialized_halo_state": False,
        },
        "weak_field_derivation": {
            "trace_identity_max_abs_error": trace_error,
            "action_derivative_max_relative_error": derivative_error,
            "spatial_equation": "Laplacian(Phi)=4 pi G rho_b",
            "time_equation": "Laplacian(Psi)=div[nu_simple(|grad Phi|/a_sigma) grad Phi]",
            "massive_potential": "Psi",
            "photon_Weyl_potential": "W=(Psi+Phi)/2",
            "known_reduction": "simple QUMOND matter dynamics with a half-QUMOND, half-Newtonian Weyl potential",
            "TT_trace_invariant_at_linear_order": 0.0,
            "TT_wave_speed": "c at linear order because the added trace invariant vanishes for a transverse-traceless perturbation",
            "health_boundary": "scalar/vector Hamiltonian health remains unproved; the raw empirical failure retires the action before that stage",
        },
        "limits": {
            "deep_matter_relative_error": deep_matter_error,
            "deep_photon_to_matter_acceleration_ratio": deep_lensing_to_matter,
            "high_matter_fractional_correction": high_matter_correction,
        },
        "spent_observation_mapping": {
            "external_dwarf_galaxy_RMSE_km_s": galaxy_rmse,
            "best_fixed_MOND_RMSE_km_s": best_mond,
            "galaxy_RMSE_ratio": galaxy_ratio,
            "galaxy_gate_pass": galaxy_pass,
            "raw_cluster_coordinate_contract": "registered array rows north and columns east",
            "raw_cluster_count": len(clusters),
            "raw_cluster_minimum_root_convergence_fraction": minimum_root_fraction,
            "raw_cluster_all_topologies_correct": all_topologies_correct,
            "raw_cluster_gate_pass": raw_cluster_pass,
        },
        "gate_results": {
            "mathematical_and_limit_checks": mathematical_pass,
            "galaxy": galaxy_pass,
            "raw_cluster_lensing": raw_cluster_pass,
            "novel_weak_field_response": not known_reduction,
        },
        "advances": advances,
        "decision": (
            "advance"
            if advances
            else "retire as a galaxy-cluster unifier: the action derives QUMOND matter dynamics but its fixed Weyl average does not recover the observed raw cluster topology"
        ),
        "next_mechanism_requirement": (
            "A third action must contain a baryon-forced trace-free/tidal state, not another local scalar of first metric derivatives. "
            "It must generate shear orientation and extra critical structure without a freely initialized halo profile."
        ),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["gate_results"], indent=2, sort_keys=True))
    print(clusters.to_string(index=False))
    print(report["decision"])


if __name__ == "__main__":
    main()
