from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v13c_khronon_completion import (
    KhrononCompletionParameters,
    dimensionless_khronon_static_function,
    effective_adm_lambda,
    khronon_completion_row,
    scalar_shift_block,
    static_susceptibility,
    temporal_excess_curvature,
    traceless_tensor_modifier_contraction,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the Sigma v13C minimal khronon completion."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v13c_khronon_completion.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v13c_khronon_completion",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fixed = config["fixed_values"]
    selected = KhrononCompletionParameters(
        epsilon=float(fixed["selected_epsilon"]),
        completion_weight=float(fixed["selected_completion_weight"]),
        trace_counterterm=float(fixed["selected_trace_counterterm"]),
    ).validated()
    ratios = np.concatenate(
        (
            [0.0],
            np.geomspace(
                float(fixed["acceleration_ratio_minimum"]),
                float(fixed["acceleration_ratio_maximum"]),
                int(fixed["acceleration_ratio_samples"]),
            ),
        )
    )

    selected_rows = [
        khronon_completion_row(float(ratio), parameters=selected)
        for ratio in fixed["sentinel_acceleration_ratios"]
    ]
    selected_lambda = effective_adm_lambda(ratios, parameters=selected)
    selected_ghost = (selected_lambda > 1.0 / 3.0) & (selected_lambda < 1.0)
    selected_kinetic = np.asarray(
        [
            khronon_completion_row(float(ratio), parameters=selected)[
                "analytic_reduced_kinetic_coefficient"
            ]
            for ratio in ratios
        ],
        dtype=float,
    )

    identity_ratios = np.geomspace(1.0e-5, 1.0e5, 401)
    steps = 1.0e-5 * identity_ratios
    upper = dimensionless_khronon_static_function(
        identity_ratios + steps,
        epsilon=selected.epsilon,
    )
    lower = dimensionless_khronon_static_function(
        identity_ratios - steps,
        epsilon=selected.epsilon,
    )
    numerical_susceptibility = (
        (upper - lower) / (2.0 * steps) / (2.0 * identity_ratios)
    )
    exact_susceptibility = static_susceptibility(
        identity_ratios,
        epsilon=selected.epsilon,
    )
    maximum_static_identity_residual = float(
        np.max(np.abs(numerical_susceptibility - exact_susceptibility))
    )

    schur_rows = [scalar_shift_block(value) for value in (-10.0, 0.0, 0.5, 0.9, 1.1, 10.0)]
    maximum_schur_identity_residual = float(
        max(abs(float(row["schur_identity_residual"])) for row in schur_rows)
    )

    weight_rows = []
    for weight_value in fixed["completion_weight_scan"]:
        weight = float(weight_value)
        params = KhrononCompletionParameters(
            epsilon=selected.epsilon,
            completion_weight=weight,
            trace_counterterm=0.0,
        ).validated()
        onset = max(
            0.0,
            0.75 * weight * (1.0 - selected.epsilon) - selected.epsilon,
        )
        witness = max(
            float(fixed["high_field_acceleration_ratio"]),
            10.0 * weight,
        )
        witness_row = khronon_completion_row(witness, parameters=params)
        weight_rows.append(
            {
                "completion_weight": weight,
                "analytic_ghost_onset_acceleration_ratio": onset,
                "high_field_witness_acceleration_ratio": witness,
                "high_field_witness_adm_lambda": float(
                    witness_row["adm_lambda"]
                ),
                "high_field_witness_reduced_kinetic": float(
                    witness_row["analytic_reduced_kinetic_coefficient"]
                ),
                "positive_weight_has_high_field_ghost": bool(
                    witness_row["in_standard_ghost_interval"]
                ),
            }
        )

    maximum_excess = float(
        temporal_excess_curvature(0.0, parameters=selected)
    )
    minimum_counterterm_to_avoid_crossing = 0.5 * maximum_excess
    counterterm_rows = []
    for counterterm_value in fixed["trace_counterterm_scan"]:
        counterterm = float(counterterm_value)
        params = KhrononCompletionParameters(
            epsilon=selected.epsilon,
            completion_weight=selected.completion_weight,
            trace_counterterm=counterterm,
        ).validated()
        low_lambda = float(effective_adm_lambda(0.0, parameters=params))
        asymptotic_lambda = 1.0 + counterterm
        crosses_ghost_interval = bool(
            counterterm > -2.0 / 3.0
            and counterterm < minimum_counterterm_to_avoid_crossing
        )
        high_field_gr_limit = bool(
            abs(asymptotic_lambda - 1.0)
            <= float(fixed["high_field_lambda_deviation_tolerance"])
        )
        counterterm_rows.append(
            {
                "trace_counterterm": counterterm,
                "lambda_at_zero_acceleration": low_lambda,
                "lambda_at_infinite_acceleration": asymptotic_lambda,
                "analytically_crosses_ghost_interval": crosses_ghost_interval,
                "preserves_high_field_gr_lambda": high_field_gr_limit,
                "rescues_kinetic_and_high_field_gr_together": bool(
                    not crosses_ghost_interval and high_field_gr_limit
                ),
            }
        )

    high_field_ratio = float(fixed["high_field_acceleration_ratio"])
    high_field_row = khronon_completion_row(
        high_field_ratio,
        parameters=selected,
    )
    plus_tensor = np.diag([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    cross_tensor = np.asarray(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    ) / np.sqrt(2.0)
    tensor_contractions = {
        "plus": traceless_tensor_modifier_contraction(
            plus_tensor,
            trace_curvature=maximum_excess,
        ),
        "cross": traceless_tensor_modifier_contraction(
            cross_tensor,
            trace_curvature=maximum_excess,
        ),
    }

    identity_tolerance = float(fixed["identity_tolerance"])
    static_derivative_tolerance = float(fixed["static_derivative_tolerance"])
    verification_gates = {
        "static_f_derivative_reproduces_selected_mu": (
            maximum_static_identity_residual <= static_derivative_tolerance
        ),
        "direct_shift_elimination_matches_closed_form": (
            maximum_schur_identity_residual <= identity_tolerance
        ),
        "positive_weights_all_have_high_field_ghost_witness": all(
            row["positive_weight_has_high_field_ghost"] for row in weight_rows
        ),
        "no_constant_counterterm_preserves_gr_and_avoids_ghost": not any(
            row["rescues_kinetic_and_high_field_gr_together"]
            for row in counterterm_rows
        ),
        "selected_high_field_force_limit_passes": float(
            high_field_row["static_fractional_extra_force"]
        )
        <= float(fixed["high_field_fractional_force_tolerance"]),
        "selected_high_field_scalar_kinetic_is_negative": float(
            high_field_row["analytic_reduced_kinetic_coefficient"]
        )
        < 0.0,
        "trace_modifier_leaves_tt_quadratic_kinetic_unchanged": all(
            abs(float(value)) <= identity_tolerance
            for value in tensor_contractions.values()
        ),
        "candidate_parameter_count_within_budget": int(
            fixed["candidate_physical_constants"]
        )
        <= int(fixed["maximum_physical_constants"]),
    }
    physical_advancement_gates = {
        "manifestly_covariant_khronon_action_written": True,
        "single_physical_metric_for_matter_and_light": True,
        "static_aqual_response_retained": bool(
            verification_gates["static_f_derivative_reproduces_selected_mu"]
        ),
        "local_tt_tensor_cone_unchanged": bool(
            verification_gates[
                "trace_modifier_leaves_tt_quadratic_kinetic_unchanged"
            ]
        ),
        "positive_reduced_scalar_kinetic_all_required_regimes": False,
        "bounded_physical_hamiltonian": False,
        "solar_ppn_gate": False,
        "raw_lensing_gate": False,
    }
    report = {
        "status": "Sigma v13C minimal khronon completion falsification",
        "protocol_status": config["protocol_status"],
        "candidate": config["candidate"],
        "selected_parameters": {
            "epsilon": selected.epsilon,
            "completion_weight": selected.completion_weight,
            "trace_counterterm": selected.trace_counterterm,
            "physical_constant_count": int(
                fixed["candidate_physical_constants"]
            ),
        },
        "covariant_action": {
            "normal": "u_mu=-nabla_mu T/sqrt(-g^ab nabla_a T nabla_b T)",
            "invariants": "Theta=nabla_mu u^mu; a_mu=u^nu nabla_nu u_mu",
            "carrier": "L_C=p Theta-H_13B(p,sqrt(a_mu a^mu))",
            "modifier": "Delta L=L_C-(Theta^2-a_mu a^mu)/2",
            "matter_coupling": "S_m[g,psi]; the same g_mu_nu governs massive matter and photons",
        },
        "analytic_failure": {
            "temporal_excess_curvature": "delta=w(1-epsilon)/(epsilon+a/a_sigma)>0",
            "effective_adm_lambda": "lambda=1+c_trace-delta/2",
            "shift_reduced_scalar_kinetic": "K_red=2(1-3lambda)/(1-lambda)",
            "ghost_interval": "1/3 < lambda < 1",
            "selected_no_counterterm_ghost_onset_acceleration_ratio": max(
                0.0,
                0.75
                * selected.completion_weight
                * (1.0 - selected.epsilon)
                - selected.epsilon,
            ),
            "minimum_constant_counterterm_avoiding_any_crossing": (
                minimum_counterterm_to_avoid_crossing
            ),
            "constant_counterterm_no_ghost_domains": (
                "c_trace<=-2/3 or c_trace>=delta_max/2"
            ),
            "why_counterterm_is_not_rescue": (
                "Avoiding the continuous ghost crossing requires a constant "
                "trace counterterm at least delta_max/2, but the same constant "
                "leaves lambda(infinity)=1+c_trace and therefore does not "
                "recover the GR high-field kinetic structure."
            ),
        },
        "high_field_sentinel": high_field_row,
        "selected_sentinel_rows": selected_rows,
        "selected_grid_summary": {
            "ghost_row_count": int(np.count_nonzero(selected_ghost)),
            "total_row_count": int(ratios.size),
            "minimum_reduced_scalar_kinetic": float(
                np.nanmin(selected_kinetic)
            ),
            "maximum_reduced_scalar_kinetic": float(
                np.nanmax(selected_kinetic)
            ),
        },
        "completion_weight_rows": weight_rows,
        "trace_counterterm_rows": counterterm_rows,
        "independent_identities": {
            "maximum_static_function_derivative_residual": (
                maximum_static_identity_residual
            ),
            "maximum_direct_schur_identity_residual": (
                maximum_schur_identity_residual
            ),
            "tensor_modifier_contractions": tensor_contractions,
        },
        "verification_gates": verification_gates,
        "all_verification_gates_pass": bool(all(verification_gates.values())),
        "physical_advancement_gates": physical_advancement_gates,
        "all_physical_advancement_gates_pass": bool(
            all(physical_advancement_gates.values())
        ),
        "v13c_formulation_rejected": True,
        "v13b_reduced_convexity_result_retained": True,
        "v13b_selected_for_more_same_mechanism_covariantization": False,
        "post_v12_reset_total_material_formulation_failure_count": 3,
        "bounded_hamiltonian_same_gate_failure_count": 3,
        "three_failure_mechanism_reset_triggered": True,
        "observational_data_accessed": False,
        "raw_holdout_opened": False,
        "theory_viable": False,
        "prior_art_boundary": {
            "established_static_lane": (
                "Blanchet-Marsat khronon acceleration actions already derive "
                "MOND-like dynamics and equal weak-field metric potentials."
            ),
            "tested_distinction": (
                "v13C tests the v13B convex Legendre pair as a temporal trace "
                "completion of that established static lane."
            ),
            "novelty_claimed": False,
        },
        "decision": (
            "Reject the minimal v13C one-metric khronon trace completion before "
            "data. Its static response and local TT cone are correct, but the "
            "same positive temporal curvature places the shift-reduced scalar "
            "graviton in the negative-kinetic interval at every sufficiently "
            "high finite acceleration. Positive completion weights only move "
            "the crossing. A constant trace counterterm large enough to avoid "
            "it destroys the GR high-field limit. This is the third materially "
            "distinct bounded-Hamiltonian failure after the v12 reset, so stop "
            "repairing this preferred-foliation clock/trace mechanism."
        ),
        "next_gate": (
            "Issue the three-failure mechanism reset. Return to the physical "
            "postulates and choose a carrier whose covariant kinetic variable "
            "is not the gravitational ADM trace, rather than adding another "
            "clock coefficient or interpolation parameter."
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
