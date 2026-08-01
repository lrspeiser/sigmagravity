#!/usr/bin/env python3
"""Assemble the preregistered A1689 signed and symmetrized covariance products."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/r1_a1689_gmos_final_covariance_assembly_protocol.json"


def covariance(samples: np.ndarray) -> np.ndarray:
    return np.cov(np.asarray(samples, dtype=float), rowvar=False, ddof=1)


def pairwise_gls_transform(covariance_9: np.ndarray) -> np.ndarray:
    transform = np.zeros((5, 9), dtype=float)
    transform[0, 4] = 1.0
    for radial_index, pair in enumerate(((3, 5), (2, 6), (1, 7), (0, 8)), start=1):
        pair_covariance = covariance_9[np.ix_(pair, pair)]
        inverse = np.linalg.pinv(pair_covariance, hermitian=True)
        raw_weight = inverse @ np.ones(2)
        weight = raw_weight / raw_weight.sum()
        transform[radial_index, list(pair)] = weight
    return transform


def finite_psd(matrix: np.ndarray, relative_tolerance: float) -> tuple[bool, float, float]:
    symmetric = (matrix + matrix.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(symmetric)
    scale = float(max(np.max(np.abs(eigenvalues)), np.finfo(float).tiny))
    minimum = float(np.min(eigenvalues))
    return bool(np.isfinite(matrix).all() and minimum >= -relative_tolerance * scale), minimum, scale


def main() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    parent = json.loads((ROOT / cfg["inputs"]["parent_protocol"]).read_text(encoding="utf-8"))
    baseline_report = json.loads((ROOT / cfg["inputs"]["baseline_report"]).read_text(encoding="utf-8"))
    bootstrap_report = json.loads((ROOT / cfg["inputs"]["bootstrap_report"]).read_text(encoding="utf-8"))
    systematic_report = json.loads((ROOT / cfg["inputs"]["systematic_report"]).read_text(encoding="utf-8"))
    baseline = pd.read_csv(ROOT / cfg["inputs"]["baseline_profile"]).sort_values("signed_bin")
    grid = pd.read_csv(ROOT / cfg["inputs"]["systematic_grid"])
    lens = pd.read_csv(ROOT / cfg["inputs"]["lens_images"])
    boot = np.load(ROOT / cfg["inputs"]["bootstrap_covariance"])

    if len(baseline) != 9 or baseline["signed_bin"].tolist() != list(range(1, 10)):
        raise RuntimeError("Frozen baseline does not contain signed bins 1..9")
    complete_grid = grid.loc[grid["status"] == "success"].copy()
    velocity_grid = complete_grid.pivot(index="run", columns="signed_bin", values="velocity_km_s")
    sigma_grid = complete_grid.pivot(index="run", columns="signed_bin", values="sigma_km_s")
    complete_runs = velocity_grid.dropna().index.intersection(sigma_grid.dropna().index)
    velocity_grid = velocity_grid.loc[complete_runs, range(1, 10)]
    sigma_grid = sigma_grid.loc[complete_runs, range(1, 10)]

    velocity = baseline["velocity_km_s"].to_numpy(dtype=float)
    sigma = baseline["sigma_km_s"].to_numpy(dtype=float)
    baseline_joint = np.concatenate((velocity, sigma))
    velocity_bootstrap = np.asarray(boot["velocity_bootstrap"], dtype=float)
    sigma_bootstrap = np.asarray(boot["sigma_bootstrap"], dtype=float)
    complete_replicates = np.asarray(boot["complete_replicates"], dtype=int)
    bootstrap_joint_samples = np.column_stack((velocity_bootstrap, sigma_bootstrap))
    bootstrap_joint_covariance = covariance(bootstrap_joint_samples)

    systematic_joint = np.column_stack((velocity_grid.to_numpy(), sigma_grid.to_numpy()))
    systematic_delta = systematic_joint - baseline_joint[None, :]
    systematic_joint_covariance = systematic_delta.T @ systematic_delta / len(systematic_delta)
    total_joint_covariance = bootstrap_joint_covariance + systematic_joint_covariance
    total_joint_covariance = (total_joint_covariance + total_joint_covariance.T) / 2.0

    first = sigma_bootstrap[complete_replicates <= 100]
    second = sigma_bootstrap[complete_replicates > 100]
    full_sigma_covariance = covariance(sigma_bootstrap)
    half_covariances = [covariance(first), covariance(second)]
    full_diagonal = np.diag(full_sigma_covariance)
    full_error = np.sqrt(full_diagonal)
    diagonal_changes = np.asarray([
        np.abs(np.diag(item) - full_diagonal) / full_diagonal for item in half_covariances
    ])
    error_changes = np.asarray([
        np.abs(np.sqrt(np.diag(item)) - full_error) / full_error for item in half_covariances
    ])
    max_diagonal_change = float(np.max(diagonal_changes))
    max_error_change = float(np.max(error_changes))

    velocity_transform = pairwise_gls_transform(total_joint_covariance[:9, :9])
    sigma_transform = pairwise_gls_transform(total_joint_covariance[9:, 9:])
    joint_transform = np.zeros((10, 18), dtype=float)
    joint_transform[:5, :9] = velocity_transform
    joint_transform[5:, 9:] = sigma_transform
    radial_joint = joint_transform @ baseline_joint
    radial_covariance = joint_transform @ total_joint_covariance @ joint_transform.T
    radial_covariance = (radial_covariance + radial_covariance.T) / 2.0
    radial_velocity = radial_joint[:5]
    radial_sigma = radial_joint[5:]
    radial_sigma_error = np.sqrt(np.diag(radial_covariance)[5:])
    fractional_sigma_error = radial_sigma_error / radial_sigma

    tolerance = cfg["gates"]["covariance_psd_relative_eigenvalue_tolerance"]
    signed_psd, signed_min_eigenvalue, signed_eigen_scale = finite_psd(total_joint_covariance, tolerance)
    radial_psd, radial_min_eigenvalue, radial_eigen_scale = finite_psd(radial_covariance, tolerance)
    radial_edges = np.asarray(cfg["assembly"]["radial_edges_arcsec"], dtype=float)
    radial_midpoint = radial_edges.mean(axis=1)
    kpc_per_arcsec = float(parent["spatial_extraction"]["kpc_per_arcsec"])
    finite_radial = np.isfinite(radial_sigma) & np.isfinite(radial_sigma_error) & (radial_sigma > 0)
    outer_support_arcsec = float(np.max(radial_edges[finite_radial, 1])) if finite_radial.any() else 0.0
    outer_support_kpc = outer_support_arcsec * kpc_per_arcsec
    anchored = lens["family_independent_redshift_anchor"].astype(bool)
    lens_inside = lens.loc[anchored & (lens["radius_kpc"] <= outer_support_kpc)].copy()
    distinct_lens_radii = int(lens_inside["radius_kpc"].nunique())
    distinct_lens_families = int(lens_inside["family_id"].nunique())

    profile = pd.DataFrame({
        "radial_bin": np.arange(1, 6),
        "lower_arcsec": radial_edges[:, 0],
        "upper_arcsec": radial_edges[:, 1],
        "radius_arcsec": radial_midpoint,
        "lower_kpc": radial_edges[:, 0] * kpc_per_arcsec,
        "upper_kpc": radial_edges[:, 1] * kpc_per_arcsec,
        "radius_kpc": radial_midpoint * kpc_per_arcsec,
        "velocity_km_s": radial_velocity,
        "velocity_error_km_s": np.sqrt(np.diag(radial_covariance)[:5]),
        "sigma_km_s": radial_sigma,
        "sigma_error_km_s": radial_sigma_error,
        "fractional_sigma_error": fractional_sigma_error,
        "finite_retained": finite_radial,
    })

    thresholds = cfg["gates"]
    checks = {
        "bootstrap_complete": len(sigma_bootstrap) >= thresholds["minimum_complete_bootstrap_replicates"],
        "bootstrap_parent_gate": bootstrap_report["gates"]["P3d_bootstrap_covariance_gate_passed"] is True,
        "systematic_parent_gate": systematic_report["gates"]["P3e_systematic_shift_gate_passed"] is True,
        "bootstrap_half_covariance_diagonal_stability": max_diagonal_change <= thresholds["maximum_covariance_diagonal_fractional_change_between_bootstrap_halves"],
        "bootstrap_half_sigma_error_stability": max_error_change <= thresholds["maximum_sigma_error_fractional_change_between_bootstrap_halves"],
        "signed_total_covariance_psd": signed_psd,
        "radial_total_covariance_psd": radial_psd,
        "minimum_finite_radial_bins": int(finite_radial.sum()) >= thresholds["minimum_finite_symmetrized_radial_bins"],
        "fractional_sigma_uncertainty": bool((fractional_sigma_error[finite_radial] <= thresholds["maximum_fractional_sigma_uncertainty_each_retained_radial_bin"]).all()),
        "baseline_side_consistency": baseline_report["gates"]["P3c_baseline_internal_consistency_gate_passed"] is True,
        "realized_lens_overlap": distinct_lens_radii >= thresholds["minimum_distinct_lens_radii_inside_realized_support"],
    }
    gate = all(checks.values())

    outputs = {key: str(value) for key, value in cfg["outputs"].items()}
    profile_path = ROOT / outputs["profile"]
    covariance_path = ROOT / outputs["covariance"]
    report_path = ROOT / outputs["report"]
    diagnostic_path = ROOT / outputs["diagnostic"]
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(profile_path, index=False)
    np.savez_compressed(
        covariance_path,
        signed_baseline_joint=baseline_joint,
        signed_bootstrap_joint_covariance=bootstrap_joint_covariance,
        signed_systematic_joint_covariance=systematic_joint_covariance,
        signed_total_joint_covariance=total_joint_covariance,
        velocity_symmetrization_transform=velocity_transform,
        sigma_symmetrization_transform=sigma_transform,
        radial_joint=radial_joint,
        radial_total_joint_covariance=radial_covariance,
        complete_replicates=complete_replicates,
        bootstrap_half_covariance_diagonal_fractional_changes=diagonal_changes,
        bootstrap_half_sigma_error_fractional_changes=error_changes,
    )

    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.errorbar(profile["radius_kpc"], profile["sigma_km_s"], yerr=profile["sigma_error_km_s"],
                  marker="o", capsize=3, color="#205493", label="GMOS stellar dispersion")
    if len(lens_inside):
        y0 = max(0.0, float(np.min(profile["sigma_km_s"] - profile["sigma_error_km_s"])) - 18.0)
        axis.scatter(lens_inside["radius_kpc"], np.full(len(lens_inside), y0), marker="|", s=130,
                     color="#c44e52", label="independent lens image radius")
    axis.set(xlabel="Projected radius (kpc)", ylabel=r"Stellar $\sigma_{los}$ (km s$^{-1}$)",
             title="Abell 1689: covariance-propagated dynamics and lens-radius overlap")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(diagnostic_path, dpi=180)
    plt.close(figure)

    report = {
        "report_version": "R1B1-A1689-GMOS-final-profile-0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "assembly_protocol_status": cfg["status"],
        "selection_blind": True,
        "baseline_kinematics_known_when_assembly_was_frozen": True,
        "bootstrap_or_systematic_outcomes_known_when_assembly_was_frozen": False,
        "bootstrap": {
            "complete_replicates": int(len(sigma_bootstrap)),
            "first_half_complete_replicates": int(len(first)),
            "second_half_complete_replicates": int(len(second)),
            "maximum_covariance_diagonal_fractional_change": max_diagonal_change,
            "maximum_sigma_error_fractional_change": max_error_change,
        },
        "systematics": {
            "complete_grid_runs_used": int(len(complete_runs)),
            "covariance_definition": cfg["assembly"]["systematic_covariance"],
        },
        "covariance": {
            "signed_joint_shape": list(total_joint_covariance.shape),
            "radial_joint_shape": list(radial_covariance.shape),
            "signed_minimum_eigenvalue": signed_min_eigenvalue,
            "signed_eigenvalue_scale": signed_eigen_scale,
            "radial_minimum_eigenvalue": radial_min_eigenvalue,
            "radial_eigenvalue_scale": radial_eigen_scale,
        },
        "profile": {
            "finite_signed_bins": int(np.isfinite(sigma).sum()),
            "finite_symmetrized_radial_bins": int(finite_radial.sum()),
            "maximum_fractional_sigma_uncertainty": float(np.max(fractional_sigma_error[finite_radial])),
            "realized_outer_support_arcsec": outer_support_arcsec,
            "realized_outer_support_kpc": outer_support_kpc,
            "radial_sigma_km_s": radial_sigma.tolist(),
            "radial_sigma_error_km_s": radial_sigma_error.tolist(),
        },
        "lens_overlap": {
            "independently_redshift_anchored_images_inside_realized_support": int(len(lens_inside)),
            "distinct_image_radii_inside_realized_support": distinct_lens_radii,
            "distinct_families_inside_realized_support": distinct_lens_families,
            "image_ids": lens_inside["image_id"].astype(str).tolist(),
        },
        "checks": checks,
        "gates": {
            "P3_profile_covariance_gate_passed": bool(gate),
            "A1689_numerical_dynamics_profile_promoted": bool(gate),
            "gravity_response_fit_authorized": False,
            "weyl_response_fit_authorized": False,
        },
        "authorization": {
            "record_numerical_dynamics_profile": bool(gate),
            "derive_dynamical_gravity_response": False,
            "infer_weyl_response": False,
            "fit_lens_mass_model": False,
            "fit_new_force_or_action": False,
        },
        "outputs": outputs,
        "limitations": [
            "This is a stellar line-of-sight kinematic profile, not a dynamical gravitational acceleration; a tracer-light profile, anisotropy model, and baryonic mass profile are still required.",
            "The lens rows provide a realized radial-overlap count only. Published image-position covariance is absent, so no Weyl-potential or lens-mass response is inferred.",
            "No void, antigravity, dark-matter, gravity-law, or cluster-mass residual entered selection, extraction, covariance assembly, or gating."
        ],
        "next_action": (
            "Preserve this numerical dynamics profile, acquire a source-traceable tracer-light/baryonic profile and raw-HST astrometric covariance, then freeze separate Jeans and lens likelihoods before either response is inferred."
            if gate else
            "Keep A1689 geometry-only for the premise audit and diagnose the failed frozen P3 check without changing bins, covariance construction, or thresholds."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
