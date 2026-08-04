"""Aligned finite-wave-vector Dirac subgate for Sigma v12A."""

from __future__ import annotations

import numpy as np

from voidscreen.sigma_v12a_same_clock_dhost import same_clock_activation


def aligned_hessian_invariants(
    scalar_clock: float,
    normal_hessian: float,
    extrinsic_curvature_trace: float,
    spatial_clock_gradient: np.ndarray,
) -> dict[str, float]:
    """Return ``L3,L4,L5`` on the aligned zero-spatial-gradient branch.

    With ``q=nabla_n phi``, ``V=phi_nn``, ``u_i=D_i q`` and vanishing
    spatial first derivative of ``phi``, the Hessian components give

    ``L3=-q^2 V^2-q^3 K V``,
    ``L4=-q^2 V^2+q^2 |u|^2`` and ``L5=q^4 V^2``.
    """

    q_clock = float(scalar_clock)
    v_star = float(normal_hessian)
    trace = float(extrinsic_curvature_trace)
    gradient = np.asarray(spatial_clock_gradient, dtype=float)
    if (
        gradient.shape != (3,)
        or np.any(~np.isfinite(gradient))
        or not np.all(np.isfinite([q_clock, v_star, trace]))
    ):
        raise ValueError("aligned Hessian inputs must be finite")
    gradient_squared = float(gradient @ gradient)
    return {
        "L3": -(q_clock**2) * v_star**2 - q_clock**3 * trace * v_star,
        "L4": -(q_clock**2) * v_star**2 + q_clock**2 * gradient_squared,
        "L5": q_clock**4 * v_star**2,
        "gradient_squared": gradient_squared,
    }


def normalized_aligned_coefficients(
    scalar_clock_ratio: float,
    *,
    background_clock_ratio: float,
    orientation_strength: float,
) -> dict[str, float]:
    """Return the normalized v12A ``A3,A4`` and clock-gradient coefficient.

    Bars remove the common factors ``F0/q_sigma^4``.  If
    ``r=q/q_sigma`` and ``x=-r^2``, then the coefficient multiplying
    ``(F0/q_sigma^2)|Dq|^2`` is ``C_bar=r^2 A4_bar``.
    """

    clock = float(scalar_clock_ratio)
    background = float(background_clock_ratio)
    strength = float(orientation_strength)
    if not np.all(np.isfinite([clock, background, strength])):
        raise ValueError("aligned coefficient inputs must be finite")
    x_value = -(clock**2)
    x_zero = -(background**2)
    activation = float(
        same_clock_activation(
            np.asarray(x_value),
            background_kinetic_ratio=x_zero,
        )
    )
    a3 = strength * activation
    z_value = x_value**2 * a3
    a4 = -a3 - x_value**2 * a3**2 / 8.0
    gradient_coefficient = clock**2 * a4
    return {
        "clock_ratio": clock,
        "x": x_value,
        "x0": x_zero,
        "activation": activation,
        "A3_bar": a3,
        "A4_bar": a4,
        "z": z_value,
        "clock_gradient_coefficient_bar": gradient_coefficient,
    }


def activation_weight_bound(
    scalar_kinetic_ratio: float,
    *,
    background_kinetic_ratio: float,
) -> dict[str, float | bool]:
    """Evaluate ``B=x^2 A`` and its analytic global upper bound.

    For ``d=x-x0``, the v12A activation obeys

    ``B <= |x|/sqrt(1+d^2) <= sqrt(1+x0^2)``.
    """

    x_value = float(scalar_kinetic_ratio)
    x_zero = float(background_kinetic_ratio)
    if not np.all(np.isfinite([x_value, x_zero])):
        raise ValueError("activation-bound inputs must be finite")
    activation = float(
        same_clock_activation(
            np.asarray(x_value),
            background_kinetic_ratio=x_zero,
        )
    )
    weighted = x_value**2 * activation
    global_bound = float(np.sqrt(1.0 + x_zero**2))
    return {
        "weighted_activation": weighted,
        "global_upper_bound": global_bound,
        "bound_satisfied": bool(
            weighted <= global_bound + 5.0e-15 * max(1.0, weighted, global_bound)
        ),
    }


def negative_strength_sufficient_condition(
    *,
    orientation_strength: float,
    background_kinetic_ratio: float,
) -> dict[str, float | bool]:
    """Return a sufficient all-``X`` condition for ``A4>=0``.

    Since ``A4_bar=-A3_bar(1+z/8)`` with
    ``z=lambda_D x^2 A``, it is non-negative whenever ``-8<=z<=0``.
    The analytic activation bound therefore gives

    ``-8/sqrt(1+x0^2) <= lambda_D <= 0``.
    """

    strength = float(orientation_strength)
    x_zero = float(background_kinetic_ratio)
    if not np.all(np.isfinite([strength, x_zero])):
        raise ValueError("strength-condition inputs must be finite")
    lower = -8.0 / np.sqrt(1.0 + x_zero**2)
    return {
        "minimum_orientation_strength": float(lower),
        "maximum_orientation_strength": 0.0,
        "actual_orientation_strength": strength,
        "sufficient_condition_satisfied": bool(lower <= strength <= 0.0),
        "interaction_nonzero": bool(strength != 0.0),
    }


def normalized_primary_secondary_symbol(
    scalar_clock_ratio: float,
    wave_number_ratio: float,
    *,
    background_clock_ratio: float,
    orientation_strength: float,
    k_2: float,
) -> dict[str, float | bool]:
    """Return the *unprojected clock block* used in the superseded audit.

    Around a constant, zero-momentum aligned auxiliary background, the reduced null
    coordinate has

    ``L=L_AeST(q)+C(q)|Dq|^2``.

    The expression ``-(4 K2+2 C_bar k_bar^2)`` omits the spatial-metric part
    of the Class-Ia null direction.  It is retained only to reproduce and
    diagnose the earlier incomplete sign argument; it is not ``Delta``.
    """

    wave_number = float(wave_number_ratio)
    clock_curvature = float(k_2)
    if (
        not np.all(np.isfinite([wave_number, clock_curvature]))
        or wave_number < 0.0
        or clock_curvature <= 0.0
    ):
        raise ValueError("require a finite non-negative wave number and K2>0")
    coefficients = normalized_aligned_coefficients(
        scalar_clock_ratio,
        background_clock_ratio=background_clock_ratio,
        orientation_strength=orientation_strength,
    )
    core = (
        4.0 * clock_curvature
        + 2.0 * coefficients["clock_gradient_coefficient_bar"] * wave_number**2
    )
    return {
        **coefficients,
        "wave_number_ratio": wave_number,
        "positive_core": core,
        "incomplete_clock_block_over_F0": -core,
        "projection_complete": False,
        "symbol_nonzero": bool(core != 0.0),
    }


def critical_wave_number_ratio(
    scalar_clock_ratio: float,
    *,
    background_clock_ratio: float,
    orientation_strength: float,
    k_2: float,
) -> float | None:
    """Return the zero of the superseded unprojected clock block."""

    coefficients = normalized_aligned_coefficients(
        scalar_clock_ratio,
        background_clock_ratio=background_clock_ratio,
        orientation_strength=orientation_strength,
    )
    gradient_coefficient = coefficients["clock_gradient_coefficient_bar"]
    if gradient_coefficient >= 0.0:
        return None
    clock_curvature = float(k_2)
    if not np.isfinite(clock_curvature) or clock_curvature <= 0.0:
        raise ValueError("K2 must be finite and positive")
    return float(np.sqrt(-2.0 * clock_curvature / gradient_coefficient))


def normalized_full_aligned_symbol(
    scalar_clock_ratio: float,
    wave_number_ratio: float,
    *,
    background_clock_ratio: float,
    orientation_strength: float,
    k_b: float,
    k_2: float,
) -> dict[str, float | bool]:
    """Return the complete aligned null-projected Dirac symbol.

    The primary null direction moves the conformal spatial metric together
    with the auxiliary clock.  After constructing that primary, the regular
    timelike unitary gauge identifies ``delta N/N=-delta r/r``; the published
    gauge-fixed Dirac determinant contains the same primary-secondary factor.
    For ``delta zeta=eta_bar delta r`` with
    ``eta_bar=-r^3 A3_bar/4``, the normalized gradient terms are

    ``K_B/r^2 + r^2 A4_bar - 4 eta_bar/r + 2 eta_bar^2``.

    The last three terms cancel exactly by the Class-Ia identity, leaving
    only the positive AeST Maxwell coefficient ``K_B/r^2``.
    """

    clock = float(scalar_clock_ratio)
    wave_number = float(wave_number_ratio)
    vector_coupling = float(k_b)
    clock_curvature = float(k_2)
    if (
        not np.all(np.isfinite([clock, wave_number, vector_coupling, clock_curvature]))
        or clock == 0.0
        or wave_number < 0.0
        or vector_coupling <= 0.0
        or clock_curvature <= 0.0
    ):
        raise ValueError("require q!=0, k>=0, K_B>0, and K2>0")
    coefficients = normalized_aligned_coefficients(
        clock,
        background_clock_ratio=background_clock_ratio,
        orientation_strength=orientation_strength,
    )
    a3 = coefficients["A3_bar"]
    null_metric_ratio = -(clock**3) * a3 / 4.0
    dhost_clock = coefficients["clock_gradient_coefficient_bar"]
    einstein_cross = -4.0 * null_metric_ratio / clock
    einstein_metric = 2.0 * null_metric_ratio**2
    class_ia_sum = dhost_clock + einstein_cross + einstein_metric
    class_ia_scale = max(
        1.0,
        abs(dhost_clock),
        abs(einstein_cross),
        abs(einstein_metric),
    )
    cancellation_residual = abs(class_ia_sum) / class_ia_scale
    aest_maxwell = vector_coupling / clock**2
    full_gradient = aest_maxwell + class_ia_sum
    core = 4.0 * clock_curvature + 2.0 * full_gradient * wave_number**2
    return {
        **coefficients,
        "wave_number_ratio": wave_number,
        "null_metric_ratio_delta_zeta_over_delta_r": null_metric_ratio,
        "gradient_terms": {
            "dhost_clock": dhost_clock,
            "einstein_clock_metric_cross": einstein_cross,
            "einstein_metric": einstein_metric,
            "class_ia_sum": class_ia_sum,
            "class_ia_normalized_cancellation_residual": cancellation_residual,
            "aest_maxwell": aest_maxwell,
            "full_null_projected": full_gradient,
        },
        "positive_core": core,
        "Delta_over_F0": -core,
        "projection_complete_on_aligned_branch": True,
        "symbol_nonzero": bool(core != 0.0),
    }


def audit_v12a_aligned_finite_k(
    *,
    k_b: float,
    k_2: float,
    background_clock_ratio: float,
    selected_positive_strength: float,
    surviving_negative_strength: float,
    counterexample_clock_ratio: float,
    random_trials: int,
    logarithmic_clock_limit: float,
    logarithmic_wave_limit: float,
    random_seed: int,
) -> dict[str, object]:
    """Correct the clock-only audit with the complete aligned null projection."""

    if random_trials < 1 or logarithmic_clock_limit <= 0.0 or logarithmic_wave_limit <= 0.0:
        raise ValueError("aligned finite-k audit coverage must be positive")
    if selected_positive_strength <= 0.0 or surviving_negative_strength >= 0.0:
        raise ValueError("the audit requires positive and negative sign sentinels")

    critical_wave = critical_wave_number_ratio(
        counterexample_clock_ratio,
        background_clock_ratio=background_clock_ratio,
        orientation_strength=selected_positive_strength,
        k_2=k_2,
    )
    if critical_wave is None:
        raise AssertionError("the positive-sign counterexample did not produce a root")
    counterexample = normalized_primary_secondary_symbol(
        counterexample_clock_ratio,
        critical_wave,
        background_clock_ratio=background_clock_ratio,
        orientation_strength=selected_positive_strength,
        k_2=k_2,
    )

    corrected_counterexample = normalized_full_aligned_symbol(
        counterexample_clock_ratio,
        critical_wave,
        background_clock_ratio=background_clock_ratio,
        orientation_strength=selected_positive_strength,
        k_b=k_b,
        k_2=k_2,
    )

    rng = np.random.default_rng(int(random_seed))
    minimum_positive_core = np.inf
    minimum_negative_core = np.inf
    maximum_cancellation_residual = 0.0
    worst_positive_row: dict[str, object] | None = None
    worst_negative_row: dict[str, object] | None = None
    for trial in range(int(random_trials)):
        sign = -1.0 if rng.random() < 0.5 else 1.0
        clock = sign * 10.0 ** rng.uniform(
            -float(logarithmic_clock_limit),
            float(logarithmic_clock_limit),
        )
        if trial % 10 == 0:
            clock = float(background_clock_ratio)
        wave = 10.0 ** rng.uniform(
            -float(logarithmic_wave_limit),
            float(logarithmic_wave_limit),
        )
        for branch, strength in (
            ("positive", selected_positive_strength),
            ("negative", surviving_negative_strength),
        ):
            row = normalized_full_aligned_symbol(
                clock,
                wave,
                background_clock_ratio=background_clock_ratio,
                orientation_strength=strength,
                k_b=k_b,
                k_2=k_2,
            )
            maximum_cancellation_residual = max(
                maximum_cancellation_residual,
                float(row["gradient_terms"]["class_ia_normalized_cancellation_residual"]),
            )
            if branch == "positive" and float(row["positive_core"]) < minimum_positive_core:
                minimum_positive_core = float(row["positive_core"])
                worst_positive_row = {"trial": trial, **row}
            if branch == "negative" and float(row["positive_core"]) < minimum_negative_core:
                minimum_negative_core = float(row["positive_core"])
                worst_negative_row = {"trial": trial, **row}

    counterexample_scale = max(
        1.0,
        4.0 * float(k_2),
        abs(2.0 * float(counterexample["clock_gradient_coefficient_bar"]) * critical_wave**2),
    )
    normalized_counterexample_residual = (
        abs(float(counterexample["positive_core"])) / counterexample_scale
    )
    gates = {
        "superseded_clock_block_zero_reproduced": (
            np.isfinite(critical_wave) and normalized_counterexample_residual < 1.0e-12
        ),
        "clock_block_explicitly_marked_incomplete": not counterexample["projection_complete"],
        "class_ia_gradient_cancellation": maximum_cancellation_residual < 1.0e-12,
        "old_counterexample_removed_by_full_projection": float(
            corrected_counterexample["positive_core"]
        )
        > 0.0,
        "positive_sign_full_symbol_positive_in_scan": minimum_positive_core > 0.0,
        "negative_sign_full_symbol_positive_in_scan": minimum_negative_core > 0.0,
        "aest_maxwell_gradient_positive": float(
            corrected_counterexample["gradient_terms"]["aest_maxwell"]
        )
        > 0.0,
    }
    return {
        "candidate": "Sigma v12A same-AeST-clock luminal DHOST geometry",
        "aligned_reduction": {
            "L3": "-q^2 V_*^2-q^3 K V_*",
            "L4": "-q^2 V_*^2+q^2 |Dq|^2",
            "L5": "q^4 V_*^2",
            "null_metric_direction": "delta zeta=-(r^3 A3_bar/4) delta r",
            "class_ia_cancellation": "r^2 A4_bar-4 eta_bar/r+2 eta_bar^2=0",
            "full_gradient_coefficient": "K_B/r^2",
            "symbol": "Delta/F0=-(4K2+2 K_B (k_bar/r)^2)",
        },
        "superseded_clock_only_result": {
            "orientation_strength": float(selected_positive_strength),
            "prior_decision": "falsified_by_finite_wave_vector_constraint_rank_zero",
            "correction": "invalid_incomplete_projection_omitted_metric_null_direction",
            "counterexample_clock_ratio": float(counterexample_clock_ratio),
            "critical_wave_number_ratio": critical_wave,
            "normalized_symbol_residual": normalized_counterexample_residual,
            "row": counterexample,
        },
        "corrected_old_counterexample": corrected_counterexample,
        "full_aligned_scan": {
            "positive_orientation_strength": float(selected_positive_strength),
            "negative_orientation_strength": float(surviving_negative_strength),
            "random_trials": int(random_trials),
            "signed_log10_clock_limit": float(logarithmic_clock_limit),
            "log10_wave_number_limit": float(logarithmic_wave_limit),
            "minimum_positive_sign_symbol_core": minimum_positive_core,
            "minimum_negative_sign_symbol_core": minimum_negative_core,
            "maximum_class_ia_cancellation_residual": maximum_cancellation_residual,
            "minimum_positive_sign_row": worst_positive_row,
            "minimum_negative_sign_row": worst_negative_row,
        },
        "gates": {name: bool(value) for name, value in gates.items()},
        "prior_positive_sign_falsification_valid": False,
        "selected_positive_row_survives_aligned_gate": bool(all(gates.values())),
        "negative_row_survives_aligned_gate": bool(all(gates.values())),
        "orientation_sign_constrained_by_aligned_gate": False,
        "full_tilted_anisotropic_symbol_derived": False,
        "complete_delta_eff_proven_invertible": False,
        "physical_degree_count_proven_unchanged": False,
        "theory_viable": False,
        "observational_data_accessed": False,
        "raw_holdout_opened": False,
    }
