"""Execute the bounded derivation sprint and freeze machine-readable outputs."""

from __future__ import annotations

import json
import platform
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sigma_sprint.auxiliary_field import (
    cluster_coherence_gate,
    fit_coupled_action_field,
    fit_density_field,
    leave_one_cluster_out_coupled_action,
    leave_one_cluster_out_density_field,
)
from sigma_sprint.baseline_audit import sparc_scale_length_audit, static_baseline_audit
from sigma_sprint.cluster_audit import audit_tian, grouped_bootstrap_posteriors
from sigma_sprint.counterrotation import counterrotation_readiness
from sigma_sprint.datasets import (
    download_mistele2025,
    download_tian2020,
    fox_overlap_names,
    load_mistele_profiles,
    load_tian2020,
)
from sigma_sprint.identifiability import dependency_audit
from sigma_sprint.mistele_crosscheck import crosscheck_mistele
from sigma_sprint.model import DEFAULT_G_DAGGER, q_potential, q_z
from sigma_sprint.qumond_fft import compare_axisymmetric_disk, run_representative_disks


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value)}")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, default=_json_default, allow_nan=False),
        encoding="utf-8",
    )


def main():
    audit_root = Path(__file__).resolve().parent
    repo_root = audit_root.parents[1]
    data_root = audit_root / "data"
    result_root = audit_root / "results"
    data_root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)
    write_json(
        result_root / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__,
            "astropy": __import__("astropy").__version__,
            "requests": __import__("requests").__version__,
        },
    )

    tian_manifest = download_tian2020(data_root)
    mistele_manifest = download_mistele2025(data_root)
    tian = load_tian2020(data_root / "tian2020" / "fig2.dat")
    fox_names = fox_overlap_names(repo_root / "data" / "clusters" / "fox2022_unique_clusters.csv")

    cluster_summary, submitted_rows, loco_rows = audit_tian(tian, fox_names)
    submitted_rows.to_csv(result_root / "tian_submitted_residuals.csv", index=False)
    loco_rows.to_csv(result_root / "tian_loco_residuals.csv", index=False)
    write_json(result_root / "tian_cluster_audit.json", cluster_summary)
    split_definitions = pd.DataFrame(
        {
            "fold": range(tian["cluster"].nunique()),
            "held_out_cluster": sorted(tian["cluster"].unique()),
            "split_type": "leave_one_cluster_out",
        }
    )
    split_definitions.to_csv(result_root / "cluster_split_definitions.csv", index=False)
    posterior = grouped_bootstrap_posteriors(tian, draws=500)
    posterior.to_csv(result_root / "cluster_bootstrap_posteriors.csv", index=False)

    density_fit = fit_density_field(tian)
    density_loco_rows, density_loco = leave_one_cluster_out_density_field(tian)
    density_loco_rows.to_csv(result_root / "density_field_loco_residuals.csv", index=False)
    coupled_fit = fit_coupled_action_field(tian)
    coupled_loco_rows, coupled_loco = leave_one_cluster_out_coupled_action(tian)
    coupled_loco_rows.to_csv(result_root / "coupled_action_loco_residuals.csv", index=False)
    write_json(
        result_root / "auxiliary_field_audit.json",
        {
            "full_sample_fit": density_fit,
            "leave_one_cluster_out": density_loco,
            "coupled_full_action_fit": coupled_fit,
            "coupled_full_action_leave_one_cluster_out": coupled_loco,
            "coherence_gate": cluster_coherence_gate(tian),
            "boundary_warning": (
                "The density source is reconstructed from only 3-5 g_bar radii per cluster; "
                "the outer zero-source continuation and natural boundary are diagnostic choices."
            ),
        },
    )

    profiles = load_mistele_profiles(data_root / "mistele2025")
    mistele_summary, mistele_rows, mistele_cluster_rows = crosscheck_mistele(tian, profiles)
    mistele_rows.to_csv(result_root / "mistele_tian_reconstruction_residuals.csv", index=False)
    mistele_cluster_rows.to_csv(result_root / "mistele_cluster_covariance_diagnostics.csv", index=False)
    write_json(result_root / "mistele_crosscheck.json", mistele_summary)

    static = static_baseline_audit(repo_root / "scripts" / "run_regression_extended.py")
    scale_summary, scale_rows = sparc_scale_length_audit(
        repo_root / "data" / "sparc" / "sparc_true_rdisk.csv",
        repo_root / "data" / "Rotmod_LTG",
    )
    scale_rows.to_csv(result_root / "sparc_scale_length_audit.csv", index=False)
    write_json(
        result_root / "canonical_baseline_audit.json",
        {"static_checks": static, "scale_lengths": scale_summary},
    )

    qumond_summaries, qumond_rows = run_representative_disks(
        repo_root / "data" / "sparc" / "sparc_true_rdisk.csv", grid_size=65, B=1.0
    )
    qumond_rows.to_csv(result_root / "qumond_axisymmetric_residuals.csv", index=False)
    median_disk = next(
        summary for summary in qumond_summaries if summary["surface_density_class"] == "median"
    )
    _, coarse_rows = compare_axisymmetric_disk(
        median_disk["baryonic_mass_msun"],
        median_disk["Rdisk_kpc"],
        B=1.0,
        grid_size=49,
        galaxy=median_disk["galaxy"],
    )
    fine_rows = qumond_rows[qumond_rows["galaxy"] == median_disk["galaxy"]].copy()
    coarse_exact = np.interp(
        fine_rows["radius_over_Rdisk"],
        coarse_rows["radius_over_Rdisk"],
        coarse_rows["g_qumond_exact"],
    )
    convergence = fine_rows[["galaxy", "radius_over_Rdisk", "g_qumond_exact"]].copy()
    convergence["g_qumond_exact_grid49_interpolated"] = coarse_exact
    convergence["grid49_to_grid65_relative_difference"] = (
        coarse_exact / convergence["g_qumond_exact"] - 1.0
    )
    convergence.to_csv(result_root / "qumond_grid_convergence.csv", index=False)
    convergence_summary = {
        "galaxy": median_disk["galaxy"],
        "grids": [49, 65],
        "median_absolute_fractional_difference": float(
            convergence["grid49_to_grid65_relative_difference"].abs().median()
        ),
        "maximum_absolute_fractional_difference": float(
            convergence["grid49_to_grid65_relative_difference"].abs().max()
        ),
    }
    write_json(
        result_root / "qumond_axisymmetric_summary.json",
        {
            "solver": "three-dimensional padded periodic FFT, axisymmetric exponential-sech^2 disk",
            "warning": (
                "These are analytic reconstructions selected from SPARC catalog masses and true "
                "Rdisk, not exact gas/bulge density maps."
            ),
            "grid_convergence": convergence_summary,
            "galaxies": qumond_summaries,
        },
    )

    counter_summary, matches, smd_before, smd_after, map_manifest = counterrotation_readiness(
        repo_root / "data" / "stellar_corgi" / "bevacqua2022_counter_rotating.tsv",
        repo_root / "data" / "manga_dynpop" / "SDSSDR17_MaNGA_JAM.fits",
    )
    matches.to_csv(result_root / "counterrotation_matched_controls.csv", index=False)
    smd_before.to_csv(result_root / "counterrotation_smd_before.csv", index=False)
    smd_after.to_csv(result_root / "counterrotation_smd_after.csv", index=False)
    map_manifest.to_csv(result_root / "counterrotation_required_map_manifest.csv", index=False)
    write_json(result_root / "counterrotation_readiness.json", counter_summary)

    z_test = np.logspace(-8, 8, 1000)
    numerical_qz = (
        q_potential(z_test * np.exp(1e-6), 1.7)
        - q_potential(z_test * np.exp(-1e-6), 1.7)
    ) / (2e-6 * z_test)
    derivative_absolute_error = np.max(np.abs(numerical_qz - q_z(z_test, 1.7)))
    derivative_relative_error = np.max(
        np.abs(numerical_qz - q_z(z_test, 1.7)) / np.abs(q_z(z_test, 1.7))
    )
    identifiability = dependency_audit()
    write_json(result_root / "identifiability_audit.json", identifiability)

    constant_cv = cluster_summary["leave_one_cluster_out"]["constant_B"]
    radial_cv = cluster_summary["leave_one_cluster_out"]["radial_B_diagnostic"]
    density_cv = density_loco
    coupled_cv = coupled_loco
    best_action_field_rms = min(density_cv["rms_dex"], coupled_cv["rms_dex"])
    action_field_outperforms_constant = best_action_field_rms < constant_cv["rms_dex"]
    independent_coherence_available = counter_summary["primary_direct_test_gate_passed"]
    decision = {
        "canonical_B_equals_A_times_C": True,
        "qumond_action_derivative_max_absolute_error": float(derivative_absolute_error),
        "qumond_action_derivative_max_relative_error": float(derivative_relative_error),
        "nonrelativistic_action_embedding_passed": bool(derivative_relative_error < 1e-6),
        "independent_coherence_dataset_available": independent_coherence_available,
        "independent_coherence_gate_passed": False,
        "cluster_operational_coherence_conflicts_with_submitted_assumption": True,
        "submitted_path_length_validation_passed": False,
        "constant_B_loco_rms_dex": constant_cv["rms_dex"],
        "radial_diagnostic_loco_rms_dex": radial_cv["rms_dex"],
        "density_action_field_loco_rms_dex": density_cv["rms_dex"],
        "coupled_action_field_loco_rms_dex": coupled_cv["rms_dex"],
        "best_action_field_loco_rms_dex": best_action_field_rms,
        "density_action_field_outperforms_constant_B": action_field_outperforms_constant,
        "derive_A_of_L_gate_passed": False,
        "go_no_go_derived_coherence_and_path_length": "NO-GO",
        "go_no_go_honest_acceleration_phenomenology": "GO",
        "reason": (
            "The QUMOND embedding is mathematically viable, but B is not independently "
            "derived: cluster C_kin conflicts with C=1, direct galaxy coherence maps are "
            "absent, and the submitted fixed path length fails the disjoint CLASH profile test."
        ),
        "g_dagger_SI": DEFAULT_G_DAGGER,
    }
    write_json(result_root / "decision_gates.json", decision)
    write_json(
        result_root / "sprint_summary.json",
        {
            "datasets": {"tian2020": tian_manifest, "mistele2025": mistele_manifest},
            "cluster": cluster_summary,
            "mistele_crosscheck": mistele_summary,
            "baseline": {"static_checks": static, "scale_lengths": scale_summary},
            "qumond": qumond_summaries,
            "counterrotation": counter_summary,
            "decision": decision,
        },
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
