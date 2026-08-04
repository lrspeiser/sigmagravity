"""Homogeneous aligned-clock Dirac branch for Sigma v12A."""

from __future__ import annotations

import numpy as np

from voidscreen.sigma_degenerate_action import luminal_class_ia_coefficients
from voidscreen.sigma_v12a_same_clock_dhost import same_clock_activation


def homogeneous_dhost_invariants(
    scalar_clock: float,
    scalar_normal_hessian: float,
    extrinsic_curvature_trace: float,
) -> dict[str, float]:
    """Return ``L3,L4,L5`` for ``grad_i(phi)=0`` and aligned aether.

    In a local orthonormal ADM frame with ``q=nabla_n phi``, ``V=phi_nn`` and
    trace ``K``, the nonzero Hessian components give

    ``box(phi)=-V-q K`` and ``phi^mu phi_munu phi^nu=q^2 V``.
    """

    q_clock = float(scalar_clock)
    v_star = float(scalar_normal_hessian)
    trace = float(extrinsic_curvature_trace)
    if not np.all(np.isfinite([q_clock, v_star, trace])):
        raise ValueError("homogeneous invariant inputs must be finite")
    return {
        "L3": -(q_clock**2) * v_star**2 - q_clock**3 * trace * v_star,
        "L4": -(q_clock**2) * v_star**2,
        "L5": q_clock**4 * v_star**2,
    }


def homogeneous_kinetic_coefficients(
    scalar_clock: float,
    *,
    f0: float,
    a3: float,
) -> dict[str, float]:
    """Return ``kappa,b,a`` in ``kappa K^2+2bVK+aV^2``.

    The pure-trace Einstein kinetic coefficient is ``kappa=-2 F0/3``.  The
    fixed Class-Ia ``L3-L5`` combination gives ``b=-q^3 A3/2`` and exactly
    ``a=b^2/kappa``.
    """

    q_clock = float(scalar_clock)
    einstein = float(f0)
    coefficient = float(a3)
    if not np.all(np.isfinite([q_clock, einstein, coefficient])) or einstein <= 0.0:
        raise ValueError("F0 must be positive and homogeneous coefficients finite")
    dependent = luminal_class_ia_coefficients(
        np.asarray(q_clock * -q_clock),
        np.asarray(einstein),
        np.asarray(0.0),
        np.asarray(coefficient),
    )
    a4 = float(dependent["A4"])
    a5 = float(dependent["A5"])
    kappa = -2.0 * einstein / 3.0
    mixing = -0.5 * q_clock**3 * coefficient
    scalar_direct = -(q_clock**2) * coefficient - q_clock**2 * a4 + q_clock**4 * a5
    scalar_schur = mixing**2 / kappa
    return {
        "kappa": kappa,
        "mixing_b": mixing,
        "scalar_a_direct": scalar_direct,
        "scalar_a_schur": scalar_schur,
        "degeneracy_residual": scalar_direct - scalar_schur,
        "A3": coefficient,
        "A4": a4,
        "A5": a5,
    }


def homogeneous_momenta(
    scalar_normal_hessian: float,
    extrinsic_curvature_trace: float,
    *,
    kappa: float,
    mixing_b: float,
) -> tuple[float, float]:
    """Return momenta ``(p_q,pi_K)`` for the degenerate two-velocity block."""

    v_star = float(scalar_normal_hessian)
    trace = float(extrinsic_curvature_trace)
    metric = float(kappa)
    mixing = float(mixing_b)
    if not np.all(np.isfinite([v_star, trace, metric, mixing])) or metric == 0.0:
        raise ValueError("homogeneous momentum inputs must be finite with nonzero kappa")
    scalar = mixing**2 / metric
    p_clock = 2.0 * (scalar * v_star + mixing * trace)
    p_metric = 2.0 * (mixing * v_star + metric * trace)
    return p_clock, p_metric


def homogeneous_primary(p_clock: float, p_metric: float, *, kappa: float, mixing_b: float) -> float:
    """Evaluate ``Psi=p_q-(b/kappa) pi_K``."""

    values = np.asarray([p_clock, p_metric, kappa, mixing_b], dtype=float)
    if np.any(~np.isfinite(values)) or float(kappa) == 0.0:
        raise ValueError("homogeneous primary inputs must be finite with nonzero kappa")
    return float(p_clock - float(mixing_b) * p_metric / float(kappa))


def homogeneous_reduced_hamiltonian(
    p_metric: float,
    scalar_clock: float,
    p_phi: float,
    *,
    kappa: float,
    k_2: float,
    background_clock: float,
) -> float:
    """Return the aligned reduced Hamiltonian excluding constraints.

    The homogeneous AeST Lagrangian is ``L_AeST=2 K2(q-Q0)^2`` in the
    published convention ``K(Q)=-F(0,Q)/2``.  The degenerate DHOST mixing drops
    out after the Legendre transform.
    """

    momentum = float(p_metric)
    q_clock = float(scalar_clock)
    scalar_momentum = float(p_phi)
    metric = float(kappa)
    curvature = float(k_2)
    q_zero = float(background_clock)
    if (
        not np.all(np.isfinite([momentum, q_clock, scalar_momentum, metric, curvature, q_zero]))
        or metric == 0.0
        or curvature <= 0.0
    ):
        raise ValueError("homogeneous Hamiltonian inputs are outside the regular branch")
    aest_lagrangian = 2.0 * curvature * (q_clock - q_zero) ** 2
    return float(momentum**2 / (4.0 * metric) + scalar_momentum * q_clock - aest_lagrangian)


def homogeneous_secondary(
    scalar_clock: float,
    p_phi: float,
    *,
    k_2: float,
    background_clock: float,
) -> float:
    """Return ``Omega=-p_phi+d L_AeST/dq`` on the aligned branch."""

    q_clock = float(scalar_clock)
    scalar_momentum = float(p_phi)
    curvature = float(k_2)
    q_zero = float(background_clock)
    if not np.all(np.isfinite([q_clock, scalar_momentum, curvature, q_zero])) or curvature <= 0.0:
        raise ValueError("homogeneous secondary inputs require finite positive K2")
    return float(-scalar_momentum + 4.0 * curvature * (q_clock - q_zero))


def homogeneous_primary_secondary_bracket(*, k_2: float) -> float:
    """Return ``{Psi,Omega}=-d^2 L_AeST/dq^2=-4 K2``."""

    curvature = float(k_2)
    if not np.isfinite(curvature) or curvature <= 0.0:
        raise ValueError("K2 must be finite and positive")
    return -4.0 * curvature


def audit_v12a_homogeneous_dirac(
    *,
    f0: float,
    k_2: float,
    orientation_strength: float,
    background_clock: float,
    clock_scan_minimum: float,
    clock_scan_maximum: float,
    clock_scan_points: int,
    random_velocity_trials: int,
    random_seed: int,
) -> dict[str, object]:
    """Audit the exact homogeneous aligned v12A primary-secondary pair."""

    if clock_scan_points < 3 or random_velocity_trials < 1:
        raise ValueError("homogeneous audit scans require at least three/one points")
    if clock_scan_minimum >= clock_scan_maximum:
        raise ValueError("clock scan bounds must be ordered")
    clocks = np.linspace(
        float(clock_scan_minimum),
        float(clock_scan_maximum),
        int(clock_scan_points),
    )
    q_zero = float(background_clock)
    background_x = -(q_zero**2)
    activation = same_clock_activation(-(clocks**2), background_kinetic_ratio=background_x)
    a3_values = float(orientation_strength) * activation
    maximum_degeneracy_residual = 0.0
    maximum_primary_residual = 0.0
    maximum_hamiltonian_velocity_residual = 0.0
    rng = np.random.default_rng(int(random_seed))
    for q_clock, a3 in zip(clocks, a3_values, strict=True):
        coefficients = homogeneous_kinetic_coefficients(
            float(q_clock),
            f0=float(f0),
            a3=float(a3),
        )
        scale = max(
            1.0,
            abs(coefficients["scalar_a_direct"]),
            abs(coefficients["scalar_a_schur"]),
        )
        maximum_degeneracy_residual = max(
            maximum_degeneracy_residual,
            abs(coefficients["degeneracy_residual"]) / scale,
        )
        for _ in range(int(random_velocity_trials)):
            v_star = float(rng.normal())
            trace = float(rng.normal())
            p_clock, p_metric = homogeneous_momenta(
                v_star,
                trace,
                kappa=coefficients["kappa"],
                mixing_b=coefficients["mixing_b"],
            )
            maximum_primary_residual = max(
                maximum_primary_residual,
                abs(
                    homogeneous_primary(
                        p_clock,
                        p_metric,
                        kappa=coefficients["kappa"],
                        mixing_b=coefficients["mixing_b"],
                    )
                ),
            )
            kinetic = (
                coefficients["kappa"] * trace**2
                + 2.0 * coefficients["mixing_b"] * v_star * trace
                + coefficients["scalar_a_schur"] * v_star**2
            )
            legendre = p_clock * v_star + p_metric * trace - kinetic
            reduced = p_metric**2 / (4.0 * coefficients["kappa"])
            maximum_hamiltonian_velocity_residual = max(
                maximum_hamiltonian_velocity_residual,
                abs(legendre - reduced),
            )

    background_coefficients = homogeneous_kinetic_coefficients(
        q_zero,
        f0=float(f0),
        a3=float(
            orientation_strength
            * same_clock_activation(
                -(q_zero**2),
                background_kinetic_ratio=background_x,
            )
        ),
    )
    bracket = homogeneous_primary_secondary_bracket(k_2=float(k_2))
    gates = {
        "homogeneous_class_ia_identity": maximum_degeneracy_residual < 1.0e-12,
        "canonical_primary_identity": maximum_primary_residual < 1.0e-11,
        "reduced_hamiltonian_independent_of_dhost_velocity": (
            maximum_hamiltonian_velocity_residual < 1.0e-10
        ),
        "background_activation_exactly_zero": abs(background_coefficients["A3"]) < 1.0e-15,
        "background_primary_reduces_to_p_q": abs(background_coefficients["mixing_b"]) < 1.0e-15,
        "background_secondary_bracket_nonzero": abs(bracket) > 1.0e-12,
    }
    return {
        "candidate": "Sigma v12A same-AeST-clock luminal DHOST geometry",
        "branch": "homogeneous scalar, aether aligned with ADM normal, zero spatial scalar gradient",
        "homogeneous_reduction": {
            "invariants": {
                "L3": "-q^2 V_*^2-q^3 K V_*",
                "L4": "-q^2 V_*^2",
                "L5": "q^4 V_*^2",
            },
            "metric_trace_coefficient": "kappa=-2 F0/3",
            "mixing": "b=-q^3 A3/2",
            "scalar_velocity_coefficient": "a=b^2/kappa=-3 q^6 A3^2/(8 F0)",
            "primary": "Psi=p_q-(b/kappa) pi_K",
            "reduced_hamiltonian": "H=pi_K^2/(4 kappa)+p_phi q-L_AeST(q)",
            "secondary": "Omega=-p_phi+dL_AeST/dq",
            "primary_secondary_bracket": "{Psi,Omega}=-d^2L_AeST/dq^2=-4 K2",
        },
        "scan": {
            "clock_minimum": float(clock_scan_minimum),
            "clock_maximum": float(clock_scan_maximum),
            "clock_points": int(clock_scan_points),
            "velocity_trials_per_clock": int(random_velocity_trials),
            "maximum_normalized_degeneracy_residual": maximum_degeneracy_residual,
            "maximum_primary_residual": maximum_primary_residual,
            "maximum_hamiltonian_velocity_residual": maximum_hamiltonian_velocity_residual,
        },
        "background": {
            "clock": q_zero,
            "A3": background_coefficients["A3"],
            "mixing_b": background_coefficients["mixing_b"],
            "primary_secondary_bracket": bracket,
            "absolute_primary_secondary_bracket": abs(bracket),
        },
        "gates": {name: bool(value) for name, value in gates.items()},
        "homogeneous_aligned_dirac_pair_regular": bool(all(gates.values())),
        "arbitrary_gradient_tilt_regular": False,
        "complete_dirac_chain_derived": False,
        "physical_degree_count_proven_unchanged": False,
        "theory_viable": False,
        "observational_data_accessed": False,
        "raw_holdout_opened": False,
    }
