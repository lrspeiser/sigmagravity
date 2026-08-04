from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_nonmetricity import (
    dimensionless_action_invariant,
    regular_isolated_branch_has_zero_slip,
    slip_nonmetricity,
    standard_action_primitive,
    standard_mu,
    standard_mu_spherical_acceleration,
    stegr_nonmetricity,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the Sigma v1 nonmetricity action.")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "sigma_v1_nonmetricity_cycle.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "sigma_v1_nonmetricity_cycle"
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    gates = config["gates"]
    a_sigma = float(config["parameters"]["a_sigma_m_s2"])

    rng = np.random.default_rng(7301)
    grad_psi = rng.normal(size=(8192, 3))
    grad_phi = rng.normal(size=(8192, 3))
    expected_q = 4.0 * np.sum(grad_psi * grad_phi, axis=1) - 2.0 * np.sum(
        np.square(grad_phi), axis=1
    )
    expected_slip = np.sum(np.square(grad_psi - grad_phi), axis=1)
    q_error = float(np.max(np.abs(stegr_nonmetricity(grad_psi, grad_phi) - expected_q)))
    slip_error = float(np.max(np.abs(slip_nonmetricity(grad_psi, grad_phi) - expected_slip)))

    equal_gradient = rng.normal(size=(4096, 3)) * a_sigma
    equal_x = dimensionless_action_invariant(equal_gradient, equal_gradient, a_sigma)
    equal_x_error = float(
        np.max(np.abs(equal_x - np.sum(np.square(equal_gradient), axis=1) / a_sigma**2))
    )

    x = np.geomspace(1e-10, 1e10, 4000)
    step = 1e-5
    derivative = (
        standard_action_primitive(x * np.exp(step))
        - standard_action_primitive(x * np.exp(-step))
    ) / (2.0 * step * x)
    analytic_mu = standard_mu(x)
    derivative_relative_error = float(
        np.max(np.abs(derivative - analytic_mu) / np.maximum(analytic_mu, 1e-30))
    )

    deep_gbar = gates["deep_gbar_over_a_sigma"] * a_sigma
    deep_g = float(standard_mu_spherical_acceleration(deep_gbar, a_sigma))
    deep_target = float(np.sqrt(deep_gbar * a_sigma))
    deep_error = abs(deep_g / deep_target - 1.0)
    high_gbar = gates["high_gbar_over_a_sigma"] * a_sigma
    high_g = float(standard_mu_spherical_acceleration(high_gbar, a_sigma))
    high_correction = high_g / high_gbar - 1.0
    mu_grid = standard_mu(np.geomspace(1e-20, 1e20, 20000))
    zero_slip = regular_isolated_branch_has_zero_slip(float(np.min(mu_grid)))

    galaxy_path = ROOT / config["spent_empirical_inputs"]["galaxy_report"]
    cluster_path = ROOT / config["spent_empirical_inputs"]["raw_cluster_scores"]
    galaxy = json.loads(galaxy_path.read_text(encoding="utf-8"))
    galaxy_scores = galaxy["sample_RMSE_km_s"]
    aqual_galaxy = float(galaxy_scores[config["spent_empirical_inputs"]["mapped_weak_field_comparator"]])
    best_mond = min(
        float(galaxy_scores["AQUAL_simple_mu_3D"]),
        float(galaxy_scores["QUMOND_simple_nu_3D"]),
    )
    galaxy_ratio = aqual_galaxy / best_mond

    cluster = pd.read_csv(cluster_path)
    aqual_cluster = cluster[
        cluster["model"] == config["spent_empirical_inputs"]["mapped_raw_lens_comparator"]
    ].copy()
    if len(aqual_cluster) == 0:
        raise RuntimeError("the mapped AQUAL raw-lens rows were not found")
    minimum_root_fraction = float(aqual_cluster["root_convergence_fraction"].min())
    all_topologies_correct = bool(aqual_cluster["all_heldout_topologies_correct"].all())

    mathematical_pass = bool(
        max(q_error, slip_error, equal_x_error)
        <= gates["invariant_identity_max_absolute_error"]
        and derivative_relative_error <= gates["action_derivative_max_relative_error"]
        and deep_error <= gates["deep_relative_error_max"]
        and high_correction <= gates["high_fractional_correction_max"]
        and np.all(mu_grid > 0.0)
        and zero_slip
    )
    galaxy_pass = bool(galaxy_ratio <= gates["galaxy_RMSE_ratio_to_best_fixed_MOND_max"])
    raw_cluster_pass = bool(
        minimum_root_fraction >= gates["raw_cluster_root_convergence_fraction_min"]
        and all_topologies_correct
    )
    advances = bool(mathematical_pass and galaxy_pass and raw_cluster_pass and not zero_slip)

    args.output.mkdir(parents=True, exist_ok=True)
    aqual_cluster.to_csv(args.output / "inherited_raw_cluster_scores.csv", index=False)
    report = {
        "status": "completed Sigma v1 pure-nonmetricity action audit",
        "model_id": config["model_id"],
        "input_hashes": {
            "config": _sha256(args.config),
            "galaxy_report": _sha256(galaxy_path),
            "raw_cluster_scores": _sha256(cluster_path),
        },
        "postulate_audit": {
            "material_sources": ["baryonic stress-energy"],
            "physical_matter_metrics": 1,
            "global_physical_parameters": config["parameters"]["global_physical_parameter_count"],
            "per_object_gravity_parameters": 0,
            "lensing_only_parameters": 0,
            "separate_sigma_matter_stress_tensor": False,
            "interpretation": "nonlinear gravitational self-interaction of nonmetricity",
        },
        "weak_field_derivation": {
            "stegr_invariant_identity_max_abs_error": q_error,
            "slip_invariant_identity_max_abs_error": slip_error,
            "equal_potential_X_identity_max_abs_error": equal_x_error,
            "action_derivative_relative_error": derivative_relative_error,
            "mu_positive": bool(np.all(mu_grid > 0.0)),
            "regular_isolated_branch_has_zero_slip": zero_slip,
            "massive_potential": "Psi",
            "photon_Weyl_potential": "(Psi+Phi)/2=Psi on the regular isolated branch",
            "reduced_equation": "div[mu(|grad Psi|/a_sigma) grad Psi]=4 pi G rho_b",
            "reduced_theory": "standard-mu AQUAL",
        },
        "limits": {
            "deep_relative_error": deep_error,
            "deep_relative_error_max": gates["deep_relative_error_max"],
            "high_fractional_correction": high_correction,
            "high_fractional_correction_max": gates["high_fractional_correction_max"],
        },
        "spent_observation_mapping": {
            "justification": "the executable weak-field equations are exactly the frozen standard-mu AQUAL comparator",
            "external_dwarf_galaxy_RMSE_km_s": aqual_galaxy,
            "best_fixed_MOND_RMSE_km_s": best_mond,
            "galaxy_RMSE_ratio": galaxy_ratio,
            "galaxy_gate_pass": galaxy_pass,
            "raw_cluster_rows": len(aqual_cluster),
            "raw_cluster_minimum_root_convergence_fraction": minimum_root_fraction,
            "raw_cluster_all_topologies_correct": all_topologies_correct,
            "raw_cluster_gate_pass": raw_cluster_pass,
        },
        "gate_results": {
            "mathematical_and_limit_checks": mathematical_pass,
            "galaxy": galaxy_pass,
            "raw_cluster_lensing": raw_cluster_pass,
            "independent_lensing_response": not zero_slip,
        },
        "advances": advances,
        "decision": (
            "advance"
            if advances
            else "retire as a galaxy-cluster unifier: the healthy isolated branch is exactly no-slip AQUAL and inherits its raw cluster topology failure"
        ),
        "next_mechanism_requirement": (
            "An additional baryon-predictable vector/tensor or nonlocal state must generate anisotropic stress and the missing convergence/shear. "
            "Another regular scalar function of the same first-derivative nonmetricity invariant cannot do so."
        ),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["gate_results"], indent=2, sort_keys=True))
    print(report["decision"])


if __name__ == "__main__":
    main()
