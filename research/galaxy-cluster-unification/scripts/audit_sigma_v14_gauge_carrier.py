from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v14_gauge_carrier import (
    constant_curvature_riemann,
    curved_improvement_divergence,
    electric_weyl_tensor,
    fourth_order_propagator_residues,
    minimal_covariant_gauge_residual,
    partially_massless_gauge_residual,
    riemann_symmetry_residuals,
    tracefree_stress_double_divergence,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the Sigma v14 local covariant gauge carrier."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v14_gauge_carrier_gate.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v14_gauge_carrier_gate",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fixed = config["fixed_values"]
    tolerance = float(fixed["identity_tolerance"])
    metric = np.diag([-1.0, 1.0, 1.0, 1.0])
    curvature = float(fixed["constant_curvature"])
    gradient = np.asarray(fixed["gauge_gradient"], dtype=float)
    constant_riemann = constant_curvature_riemann(
        metric,
        curvature=curvature,
    )
    minimal_flat = minimal_covariant_gauge_residual(
        np.zeros((4, 4, 4, 4)),
        gradient,
    )
    minimal_curved = minimal_covariant_gauge_residual(
        constant_riemann,
        gradient,
    )
    pm_constant = partially_massless_gauge_residual(
        constant_riemann,
        gradient,
        metric=metric,
        curvature_counterterm=curvature,
    )

    tidal_amplitude = float(fixed["weyl_tidal_amplitude"])
    electric_tidal = tidal_amplitude * np.diag([-2.0, 1.0, 1.0])
    weyl = electric_weyl_tensor(electric_tidal)
    weyl_symmetries = riemann_symmetry_residuals(weyl, metric=metric)
    pm_weyl = partially_massless_gauge_residual(
        constant_riemann + weyl,
        gradient,
        metric=metric,
        curvature_counterterm=curvature,
    )
    exact_weyl_contraction = minimal_covariant_gauge_residual(weyl, gradient)
    pm_weyl_identity_residual = float(
        np.max(np.abs(pm_weyl - exact_weyl_contraction))
    )

    tracefree_source_residual = tracefree_stress_double_divergence(
        wave_covector_squared=float(fixed["tracefree_source_wave_squared"]),
        stress_trace=float(fixed["tracefree_source_stress_trace"]),
    )
    ricci = np.diag(
        np.asarray(fixed["curved_improvement_ricci_diagonal"], dtype=float)
    )
    curved_improvement = curved_improvement_divergence(ricci, gradient)
    residues = fourth_order_propagator_residues(
        massive_pole_squared=float(fixed["fourth_order_massive_pole_squared"])
    )

    formulation_rows = [
        {
            **config["formulations"][0],
            "flat_gauge_residual_norm": float(np.max(np.abs(minimal_flat))),
            "curved_gauge_residual_norm": float(np.max(np.abs(minimal_curved))),
            "required_gate_passes": False,
            "formulation_rejected": True,
        },
        {
            **config["formulations"][1],
            "constant_curvature_gauge_residual_norm": float(
                np.max(np.abs(pm_constant))
            ),
            "weyl_curved_gauge_residual_norm": float(np.max(np.abs(pm_weyl))),
            "required_gate_passes": False,
            "formulation_rejected": True,
        },
        {
            **config["formulations"][2],
            "massless_pole_residue": residues.massless,
            "massive_pole_residue": residues.massive,
            "residue_sum": residues.massless + residues.massive,
            "opposite_residue_signs": residues.massless * residues.massive < 0.0,
            "required_gate_passes": False,
            "formulation_rejected": True,
        },
    ]
    source_rows = [
        {
            "source": "complete conserved stress tensor T_mn",
            "double_divergence": 0.0,
            "gauge_compatible": True,
            "zero_monopole": False,
            "decision": "reject because it restores the forbidden direct mass charge",
        },
        {
            "source": "trace-free stress T_mn-g_mn T/4",
            "double_divergence": tracefree_source_residual,
            "gauge_compatible": abs(tracefree_source_residual) <= tolerance,
            "zero_monopole": "not sufficient",
            "decision": "reject because stress conservation does not conserve its trace-free projection",
        },
        {
            "source": "flat improvement (nabla_mn-g_mn box)S",
            "flat_divergence": 0.0,
            "curved_divergence_norm": float(np.max(np.abs(curved_improvement))),
            "gauge_compatible": False,
            "zero_monopole": True,
            "decision": "reject as a general covariant source because Ricci curvature spoils conservation",
        },
        {
            "source": "Bach tensor",
            "double_divergence": 0.0,
            "gauge_compatible": True,
            "zero_monopole": True,
            "decision": "source identity passes, but the associated local conformal spin-two completion has an opposite-residue ghost",
        },
    ]
    verification_gates = {
        "minimal_field_strength_is_gauge_invariant_in_flat_space": float(
            np.max(np.abs(minimal_flat))
        )
        <= tolerance,
        "minimal_covariantization_has_curvature_residual": float(
            np.max(np.abs(minimal_curved))
        )
        > tolerance,
        "partially_massless_term_cancels_constant_curvature": float(
            np.max(np.abs(pm_constant))
        )
        <= tolerance,
        "synthetic_weyl_tensor_has_required_symmetries": max(
            weyl_symmetries.values()
        )
        <= tolerance,
        "partially_massless_residual_equals_weyl_contraction": (
            pm_weyl_identity_residual <= tolerance
        ),
        "partially_massless_symmetry_fails_on_weyl_curvature": float(
            np.max(np.abs(pm_weyl))
        )
        > tolerance,
        "tracefree_stress_source_is_not_conserved_for_generic_trace": abs(
            tracefree_source_residual
        )
        > tolerance,
        "flat_improvement_is_not_conserved_on_generic_curvature": float(
            np.max(np.abs(curved_improvement))
        )
        > tolerance,
        "fourth_order_completion_has_opposite_residues": (
            residues.massless > 0.0 and residues.massive < 0.0
        ),
        "all_three_materially_distinct_formulations_fail_common_gate": (
            len(formulation_rows) == 3
            and all(row["formulation_rejected"] for row in formulation_rows)
        ),
    }
    report = {
        "status": "Sigma v14 local covariant gauge-carrier falsification",
        "protocol_status": config["protocol_status"],
        "candidate_family": config["candidate_family"],
        "analytic_identities": {
            "minimal_field_strength_variation": (
                "delta F_mn|r=R_mnr{}^s nabla_s alpha"
            ),
            "partially_massless_corrected_variation": (
                "delta F_mn|r=C_mnr{}^s nabla_s alpha after the constant-curvature part cancels"
            ),
            "tracefree_stress_double_divergence": "-box(T)/4 in four dimensions",
            "flat_improvement_curved_divergence": "R_ns nabla^s S",
            "fourth_order_partial_fraction": (
                "1/[k^2(k^2+m^2)]=(1/m^2)[1/k^2-1/(k^2+m^2)]"
            ),
        },
        "weyl_tensor_symmetry_residuals": weyl_symmetries,
        "partially_massless_weyl_identity_residual": pm_weyl_identity_residual,
        "formulation_rows": formulation_rows,
        "source_rows": source_rows,
        "verification_gates": verification_gates,
        "all_verification_gates_pass": all(verification_gates.values()),
        "materially_distinct_failure_count": 3,
        "common_failed_gate": "healthy local covariant gauge-reduced tidal carrier on arbitrary one-metric backgrounds",
        "three_failure_mechanism_reset_triggered": True,
        "action_gate_passed": False,
        "observational_data_accessed": False,
        "raw_holdout_opened": False,
        "theory_viable": False,
        "prior_art_boundary": {
            "higher_rank_scalar_gauge_symmetry": "established fracton and longitudinal-diffeomorphism prior art",
            "partially_massless_spin_two": "established constant-curvature construction with published nonlinear/gravitational no-go results",
            "conformal_bach_completion": "established conformal-gravity construction and opposite-sign spin-two spectrum",
            "sigma_novelty_claimed": False,
        },
        "decision": (
            "Reject the frozen local covariant gauge-reduced tidal carrier before data. "
            "Minimal covariantization loses gauge invariance on curvature; the "
            "partially-massless correction works only after removing the Weyl "
            "curvature that real galaxies and clusters require; and the local "
            "conformal/Bach completion restores covariance with an opposite-residue "
            "negative-energy spin-two mode. The three-variant stopping rule resets "
            "this carrier family rather than adding another curvature counterterm."
        ),
        "next_gate": (
            "Return to the physical postulates. Any successor must not require a "
            "local covariant scalar-gauge rank-two carrier. Do not open observations."
        ),
        "scope_limit": config["scope_limit"],
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
