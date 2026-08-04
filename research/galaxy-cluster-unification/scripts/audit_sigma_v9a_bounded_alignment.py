from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v9a_bounded_alignment import audit_v9a_bounded_alignment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the no-data Sigma v9A bounded alignment mechanism."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v9a_bounded_alignment_gate.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v9a_bounded_alignment_gate",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    selected = config["selected_values"]
    robustness = config["robustness_values"]
    audit = audit_v9a_bounded_alignment(
        k_b=float(selected["K_B"]),
        eta=float(selected["eta_sigma"]),
        rank_surface_eta_values=np.asarray(
            robustness["rank_surface_eta_values"], dtype=float
        ),
        y_values=np.asarray(robustness["Y_over_a_squared"], dtype=float),
        z_values=np.asarray(robustness["Z_over_a_squared"], dtype=float),
        cosine_values=np.asarray(robustness["cosine"], dtype=float),
        random_samples=int(robustness["random_samples"]),
        random_seed=int(robustness["random_seed"]),
        fixed_mond_mean_fraction=float(
            selected[
                "fixed_MOND_mean_predicted_fraction_from_existing_development_report"
            ]
        ),
    )
    report = {
        "status": "completed Sigma v9A bounded alignment theory-only gate",
        "candidate": config["candidate"],
        "protocol_version": config["protocol_version"],
        "covariant_definitions": config["covariant_definitions"],
        "variants": config["variants"],
        "physical_parameters": config["physical_parameters"],
        **audit,
        "decision": "retire_exact_v9A_as_the_standalone_unification_completion",
        "reason": (
            "The direct one-sided term changes the six-variable static principal "
            "inertia at a finite aether acceleration for every tested nonzero eta. "
            "Adding the minimal Z saturation removes that failure throughout the "
            "declared necessary scan, but cannot solve the target mechanism: the "
            "Gram determinant and both first-variation fluxes vanish exactly whenever "
            "the scalar and aether gradients are aligned. Every spherical baryonic "
            "configuration therefore remains exactly the fixed AeST/MOND solution, "
            "which already supplies only 0.318 of the existing cluster target on "
            "average. Eta cannot repair an identically zero source."
        ),
        "scope": (
            "This retires the exact Gram-only v9A interaction as the standalone "
            "galaxy-cluster completion. The saturated term remains a mathematically "
            "interesting topology perturbation, but it has not passed a full Dirac, "
            "characteristic, PPN, or source-uniqueness audit and is not authorized as "
            "an extra fitted term. No new observational array or holdout was opened."
        ),
        "prior_art_boundary": (
            "AeST, scalar-vector-tensor actions, aether-acceleration invariants, and "
            "scalar/aether derivative cross-couplings are prior-art categories. The "
            "targeted search did not establish that this exact rational Gram function "
            "was published, but algebraic novelty would not rescue its mechanism-null "
            "result and no novelty claim is made."
        ),
        "next_gate": (
            "Do not add another angular gate to AeST. Select a baryon-forced carrier "
            "that has a nonzero spherical monopole and independently transports "
            "orientation/shear, with bounded first-order dynamics or an exact "
            "degeneracy identity, before opening any observational holdout."
        ),
        "requirements": config["requirements"],
        "decision_rule": config["decision_rule"],
        "data_policy": config["data_policy"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
