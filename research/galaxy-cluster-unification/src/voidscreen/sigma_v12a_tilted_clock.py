"""Tilted AeST clock susceptibility entering the v12A Dirac bracket."""

from __future__ import annotations

import numpy as np


def simple_aest_interpolation_derivatives(y_value: float) -> tuple[float, float]:
    """Return ``(f_y,f_yy)`` for the fixed simple AeST interpolation.

    ``f_y=sqrt(y)/(1+sqrt(y))`` and
    ``f_yy=1/[2 sqrt(y)(1+sqrt(y))^2]``.  The second derivative diverges at
    ``y=0``, but it only enters the susceptibility multiplied by ``(Y')^2``;
    that product has a regular zero limit on the projected-gradient axis.
    """

    y_ratio = float(y_value)
    if not np.isfinite(y_ratio) or y_ratio < 0.0:
        raise ValueError("the AeST interpolation argument must be finite and non-negative")
    if y_ratio == 0.0:
        return 0.0, np.inf
    root = np.sqrt(y_ratio)
    return float(root / (1.0 + root)), float(1.0 / (2.0 * root * (1.0 + root) ** 2))


def tilted_clock_kinematics(
    normal_clock: float,
    spatial_gradient: np.ndarray,
    aether_spatial: np.ndarray,
) -> dict[str, float]:
    """Return ``Q,Y,dY/dq,d2Y/dq2`` in reduced AeST ADM variables."""

    q_clock = float(normal_clock)
    gradient = np.asarray(spatial_gradient, dtype=float)
    aether = np.asarray(aether_spatial, dtype=float)
    if (
        gradient.shape != (3,)
        or aether.shape != (3,)
        or np.any(~np.isfinite(gradient))
        or np.any(~np.isfinite(aether))
        or not np.isfinite(q_clock)
    ):
        raise ValueError("tilted clock inputs must be finite three-vectors and a scalar")
    aether_squared = float(aether @ aether)
    aether_norm = float(np.sqrt(aether_squared))
    chi = float(np.sqrt(1.0 + aether_squared))

    # Directly evaluating Y=-q^2+|s|^2+Q^2 loses all precision on the
    # physically important Y=0 projected-gradient axis.  Resolve the spatial
    # gradient parallel and perpendicular to the aether instead.  The exact
    # Lorentz-boost identities are
    #
    #   Y=(|A| q+chi s_parallel)^2+|s_perp|^2,
    #   Y_q=2 |A| (|A| q+chi s_parallel),
    #   Q=(q+|A| (|A| q+chi s_parallel))/chi.
    #
    # This form is a sum of squares and makes Y>=0 and the projected Cauchy
    # identity numerically manifest over the twelve-decade stress scan.
    if aether_norm == 0.0:
        parallel_gradient = 0.0
        perpendicular_gradient = gradient.copy()
    else:
        aether_direction = aether / aether_norm
        parallel_gradient = float(aether_direction @ gradient)
        perpendicular_gradient = gradient - parallel_gradient * aether_direction
    boosted_spatial_gradient = aether_norm * q_clock + chi * parallel_gradient
    boost_scale = max(
        1.0,
        abs(aether_norm * q_clock),
        abs(chi * parallel_gradient),
    )
    roundoff_limit = 64.0 * np.finfo(float).eps
    if abs(boosted_spatial_gradient) <= roundoff_limit * boost_scale:
        boosted_spatial_gradient = 0.0
    perpendicular_norm = float(np.linalg.norm(perpendicular_gradient))
    if perpendicular_norm <= roundoff_limit * max(
        1.0,
        float(np.linalg.norm(gradient)),
    ):
        perpendicular_gradient = np.zeros(3, dtype=float)
    perpendicular_squared = float(perpendicular_gradient @ perpendicular_gradient)
    y_invariant = float(boosted_spatial_gradient**2 + perpendicular_squared)
    q_invariant = float((q_clock + aether_norm * boosted_spatial_gradient) / chi)
    first_y = 2.0 * aether_norm * boosted_spatial_gradient
    second_y = 2.0 * aether_squared
    cauchy_residual = aether_squared * perpendicular_squared
    identity_left = aether_squared * y_invariant - 0.25 * first_y**2
    identity_scale = max(
        1.0,
        abs(aether_squared * y_invariant),
        abs(0.25 * first_y**2),
        abs(cauchy_residual),
    )
    identity_relative_error = abs(identity_left - cauchy_residual) / identity_scale
    return {
        "chi": chi,
        "aether_squared": aether_squared,
        "Q": q_invariant,
        "Y": y_invariant,
        "dY_dq": first_y,
        "d2Y_dq2": second_y,
        "projected_cauchy_residual": cauchy_residual,
        "projected_cauchy_identity_relative_error": identity_relative_error,
    }


def tilted_aest_clock_susceptibility(
    normal_clock: float,
    spatial_gradient: np.ndarray,
    aether_spatial: np.ndarray,
    *,
    a_sigma: float,
    k_b: float,
    k_2: float,
) -> dict[str, float | bool]:
    """Return ``d2 L_AeST/dq2`` after eliminating ``mu,nu``.

    For vanishing derivative of the aether background at the evaluation point,

    ``L=-(2-KB)Y-(2-KB)a_sigma^2 f(Y/a_sigma^2)+2K2(Q-Q0)^2``.

    The ``J.B`` term is affine in the clock and does not enter this Hessian.
    """

    acceleration = float(a_sigma)
    vector_coupling = float(k_b)
    clock_curvature = float(k_2)
    if (
        not np.all(np.isfinite([acceleration, vector_coupling, clock_curvature]))
        or acceleration <= 0.0
        or not 0.0 < vector_coupling < 2.0
        or clock_curvature <= 0.0
    ):
        raise ValueError("require a_sigma>0, 0<K_B<2, and K2>0")
    kinematics = tilted_clock_kinematics(
        normal_clock,
        spatial_gradient,
        aether_spatial,
    )
    y_ratio = kinematics["Y"] / acceleration**2
    f_y, _ = simple_aest_interpolation_derivatives(y_ratio)
    if y_ratio == 0.0:
        interpolation_curvature = 0.0
    else:
        root = np.sqrt(y_ratio)
        directional_fraction = min(
            1.0,
            max(
                0.0,
                0.25 * kinematics["dY_dq"] ** 2 / (kinematics["aether_squared"] * kinematics["Y"])
                if kinematics["aether_squared"] > 0.0
                else 0.0,
            ),
        )
        # Algebraically identical to f_yy*Y_q^2/a_sigma^2, but regular at
        # Y=0 and free of the infinity-times-zero cancellation.
        interpolation_curvature = float(
            2.0 * kinematics["aether_squared"] * directional_fraction * root / (1.0 + root) ** 2
        )
    density_coefficient = 2.0 - vector_coupling
    susceptibility = 4.0 * clock_curvature * kinematics["chi"] ** 2 - density_coefficient * (
        (1.0 + f_y) * kinematics["d2Y_dq2"] + interpolation_curvature
    )
    # Cauchy gives (Y')^2 <= 4 |A|^2 Y.  The fixed interpolation then gives
    # f_yy (Y')^2/a_sigma^2 <= 2 |A|^2 r/(1+r)^2 <= |A|^2/2.
    lower_bound = (
        4.0 * clock_curvature * (1.0 + kinematics["aether_squared"])
        - 4.5 * density_coefficient * kinematics["aether_squared"]
    )
    relative_bound_violation = max(0.0, lower_bound - susceptibility) / max(
        1.0,
        abs(lower_bound),
        abs(susceptibility),
    )
    return {
        **kinematics,
        "y_ratio": y_ratio,
        "f_y": f_y,
        "interpolation_curvature_term": interpolation_curvature,
        "clock_susceptibility": susceptibility,
        "analytic_lower_bound": lower_bound,
        "relative_lower_bound_violation": relative_bound_violation,
        "bound_satisfied": bool(relative_bound_violation <= 5.0e-15),
        "susceptibility_positive": bool(susceptibility > 0.0),
    }


def global_tilt_parameter_condition(*, k_b: float, k_2: float) -> dict[str, float | bool]:
    """Return the sufficient all-tilt positivity condition.

    The lower bound is

    ``4 K2 + [4 K2-(9/2)(2-KB)] |A|^2``.

    It is positive for every finite tilt when ``K2 >= 9(2-KB)/8``.
    """

    vector_coupling = float(k_b)
    clock_curvature = float(k_2)
    if (
        not np.all(np.isfinite([vector_coupling, clock_curvature]))
        or not 0.0 < vector_coupling < 2.0
        or clock_curvature <= 0.0
    ):
        raise ValueError("require 0<K_B<2 and K2>0")
    threshold = 9.0 * (2.0 - vector_coupling) / 8.0
    tilt_coefficient = 4.0 * clock_curvature - 4.5 * (2.0 - vector_coupling)
    return {
        "minimum_k2": threshold,
        "actual_k2": clock_curvature,
        "constant_margin": 4.0 * clock_curvature,
        "tilt_squared_margin": tilt_coefficient,
        "globally_positive_sufficient_condition": bool(clock_curvature >= threshold),
    }


def audit_v12a_tilted_clock(
    *,
    a_sigma: float,
    k_b: float,
    k_2: float,
    random_trials: int,
    logarithmic_amplitude_limit: float,
    random_seed: int,
) -> dict[str, object]:
    """Stress-test the exact tilted AeST contribution to ``Delta_eff``."""

    if random_trials < 1 or logarithmic_amplitude_limit <= 0.0:
        raise ValueError("the tilted clock audit requires positive scan coverage")
    rng = np.random.default_rng(int(random_seed))
    minimum_susceptibility = np.inf
    minimum_bound = np.inf
    minimum_cauchy_residual = np.inf
    maximum_cauchy_identity_relative_error = 0.0
    maximum_bound_violation = 0.0
    maximum_relative_bound_violation = 0.0
    maximum_tilt = 0.0
    maximum_gradient = 0.0
    worst: dict[str, object] | None = None

    def signed_log_vector() -> np.ndarray:
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        magnitude = 10.0 ** rng.uniform(
            -float(logarithmic_amplitude_limit),
            float(logarithmic_amplitude_limit),
        )
        return direction * magnitude

    for trial in range(int(random_trials)):
        aether = signed_log_vector()
        gradient = signed_log_vector()
        q_clock = float(
            rng.choice(np.asarray([-1.0, 1.0]))
            * 10.0
            ** rng.uniform(
                -float(logarithmic_amplitude_limit),
                float(logarithmic_amplitude_limit),
            )
        )
        if trial % 5 == 0:
            # Exact projected-gradient axis Y=0: B_mu is parallel to the aether.
            chi = np.sqrt(1.0 + float(aether @ aether))
            gradient = -q_clock * aether / chi
        row = tilted_aest_clock_susceptibility(
            q_clock,
            gradient,
            aether,
            a_sigma=float(a_sigma),
            k_b=float(k_b),
            k_2=float(k_2),
        )
        violation = max(
            0.0, float(row["analytic_lower_bound"]) - float(row["clock_susceptibility"])
        )
        maximum_bound_violation = max(maximum_bound_violation, violation)
        maximum_relative_bound_violation = max(
            maximum_relative_bound_violation,
            float(row["relative_lower_bound_violation"]),
        )
        minimum_cauchy_residual = min(
            minimum_cauchy_residual,
            float(row["projected_cauchy_residual"]),
        )
        maximum_cauchy_identity_relative_error = max(
            maximum_cauchy_identity_relative_error,
            float(row["projected_cauchy_identity_relative_error"]),
        )
        minimum_bound = min(minimum_bound, float(row["analytic_lower_bound"]))
        if float(row["clock_susceptibility"]) < minimum_susceptibility:
            minimum_susceptibility = float(row["clock_susceptibility"])
            worst = {
                "trial": trial,
                "normal_clock": q_clock,
                "spatial_gradient": gradient.tolist(),
                "aether_spatial": aether.tolist(),
                **row,
            }
        maximum_tilt = max(maximum_tilt, float(np.linalg.norm(aether)))
        maximum_gradient = max(maximum_gradient, float(np.linalg.norm(gradient)))

    condition = global_tilt_parameter_condition(k_b=float(k_b), k_2=float(k_2))
    gates = {
        "selected_parameters_satisfy_global_bound": condition[
            "globally_positive_sufficient_condition"
        ],
        "all_sampled_susceptibilities_positive": minimum_susceptibility > 0.0,
        "all_sampled_analytic_bounds_positive": minimum_bound > 0.0,
        "projected_cauchy_inequality": minimum_cauchy_residual >= 0.0,
        "projected_cauchy_identity": maximum_cauchy_identity_relative_error < 1.0e-12,
        "analytic_lower_bound_respected": maximum_relative_bound_violation < 1.0e-12,
    }
    return {
        "candidate": "Sigma v12A same-AeST-clock luminal DHOST geometry",
        "exact_reduced_aest_result": {
            "Q": "chi q+A.s",
            "Y": "-q^2+s^2+Q^2",
            "susceptibility": "d2L_AeST/dq2=4K2 chi^2-(2-KB)[(1+f_y)Y_qq+f_yy Y_q^2/a_sigma^2]",
            "cauchy_identity": "|A|^2 Y-(Y_q/2)^2=|A|^2|s|^2-(A.s)^2>=0",
            "global_lower_bound": "4K2+[4K2-(9/2)(2-KB)]|A|^2",
            "sufficient_parameter_condition": "K2>=9(2-KB)/8",
            "selected_parameter_condition": condition,
        },
        "random_scan": {
            "trials": int(random_trials),
            "signed_log10_amplitude_limit": float(logarithmic_amplitude_limit),
            "maximum_aether_spatial_norm": maximum_tilt,
            "maximum_scalar_spatial_gradient_norm": maximum_gradient,
            "minimum_clock_susceptibility": minimum_susceptibility,
            "minimum_analytic_lower_bound": minimum_bound,
            "minimum_projected_cauchy_residual": minimum_cauchy_residual,
            "maximum_projected_cauchy_identity_relative_error": maximum_cauchy_identity_relative_error,
            "maximum_lower_bound_violation": maximum_bound_violation,
            "maximum_relative_lower_bound_violation": maximum_relative_bound_violation,
            "minimum_susceptibility_row": worst,
        },
        "gates": {name: bool(value) for name, value in gates.items()},
        "tilted_reduced_aest_susceptibility_globally_positive": bool(all(gates.values())),
        "dhost_spatial_operator_included": False,
        "complete_delta_eff_proven_invertible": False,
        "complete_dirac_chain_derived": False,
        "physical_degree_count_proven_unchanged": False,
        "theory_viable": False,
        "observational_data_accessed": False,
        "raw_holdout_opened": False,
    }
