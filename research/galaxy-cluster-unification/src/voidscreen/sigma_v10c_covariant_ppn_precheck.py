"""Covariant coefficient and PPN-applicability precheck for Sigma v10C.

The selected v10C counterterm changes the electric/magnetic balance of the
AeST aether.  On the unit-aether constraint,

``B_mn B^mn = F_mn F^mn + 2 J_m J^m``

for ``J_n=A^m F_mn`` and the fully spatial projection ``B_mn``.  Therefore

``-K_B F^2/2 + K_B(1-u) B^2/2``

is exactly

``-K_B u F^2/2 + K_B(1-u) J^2``.

In standard Einstein-aether notation this maps to
``(c1,c2,c3,c4)=(K_B u,0,-K_B u,K_B(1-u))``.  The pure
Einstein-aether PPN formulas are only a diagnostic proxy: v10C also contains
the AeST scalar and a dynamical spatial tensor, and the mapped proxy has
``c123=0``, where the pure-aether alpha2 formula is singular.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


def _finite_positive(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _unit_interval(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 < result <= 1.0:
        raise ValueError(f"{name} must lie in (0, 1]")
    return result


@dataclass(frozen=True)
class ElectricMagneticInvariants:
    field_strength_squared: float
    magnetic_squared: float
    acceleration_squared: float
    decomposition_residual: float


def electric_magnetic_invariants(
    covariant_metric: Array,
    aether_contravariant: Array,
    field_strength_covariant: Array,
) -> ElectricMagneticInvariants:
    """Return the unit-aether electric/magnetic decomposition invariants."""

    metric = np.asarray(covariant_metric, dtype=float)
    aether = np.asarray(aether_contravariant, dtype=float)
    field = np.asarray(field_strength_covariant, dtype=float)
    if metric.shape != (4, 4) or field.shape != (4, 4):
        raise ValueError("metric and field strength must have shape (4, 4)")
    if aether.shape != (4,):
        raise ValueError("aether must have shape (4,)")
    if np.any(~np.isfinite(metric)) or np.any(~np.isfinite(aether)) or np.any(
        ~np.isfinite(field)
    ):
        raise ValueError("all tensor entries must be finite")
    if not np.allclose(metric, metric.T, rtol=0.0, atol=1.0e-12):
        raise ValueError("metric must be symmetric")
    if not np.allclose(field, -field.T, rtol=0.0, atol=1.0e-12):
        raise ValueError("field strength must be antisymmetric")
    inverse = np.linalg.inv(metric)
    norm = float(aether @ metric @ aether)
    if not np.isclose(norm, -1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("aether must have unit timelike norm -1")

    aether_covariant = metric @ aether
    mixed_projector = np.eye(4) + np.outer(aether_covariant, aether)
    magnetic = mixed_projector @ field @ mixed_projector.T
    acceleration_covariant = np.einsum("m,mn->n", aether, field)
    raised_field = inverse @ field @ inverse
    raised_magnetic = inverse @ magnetic @ inverse
    field_squared = float(np.einsum("mn,mn->", field, raised_field))
    magnetic_squared = float(np.einsum("mn,mn->", magnetic, raised_magnetic))
    acceleration_squared = float(acceleration_covariant @ inverse @ acceleration_covariant)
    residual = field_squared - (magnetic_squared - 2.0 * acceleration_squared)
    return ElectricMagneticInvariants(
        field_strength_squared=field_squared,
        magnetic_squared=magnetic_squared,
        acceleration_squared=acceleration_squared,
        decomposition_residual=float(residual),
    )


def mapped_einstein_aether_coefficients(*, k_b: float, u: float) -> dict[str, float]:
    """Map the v10C aether kinetic terms to standard ``c_i`` coefficients."""

    stiffness = _finite_positive(k_b, name="k_b")
    speed = _unit_interval(u, name="u")
    c1 = stiffness * speed
    c2 = 0.0
    c3 = -stiffness * speed
    c4 = stiffness * (1.0 - speed)
    return {
        "c1": c1,
        "c2": c2,
        "c3": c3,
        "c4": c4,
        "c13": c1 + c3,
        "c14": c1 + c4,
        "c123": c1 + c2 + c3,
        "pure_aether_vector_speed_squared": c1 / (c1 + c4),
    }


def pure_einstein_aether_alpha1(*, c1: float, c3: float, c4: float) -> float:
    """Return the Foster--Jacobson pure Einstein-aether ``alpha1`` value."""

    values = np.asarray([c1, c3, c4], dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError("c1, c3, and c4 must be finite")
    denominator = 2.0 * c1 - c1**2 + c3**2
    if np.isclose(denominator, 0.0, rtol=0.0, atol=1.0e-15):
        raise ValueError("the pure-aether alpha1 denominator vanishes")
    return float(-8.0 * (c3**2 + c1 * c4) / denominator)


def aether_action_density_pair(
    invariants: ElectricMagneticInvariants,
    *,
    k_b: float,
    u: float,
) -> dict[str, float]:
    """Compare the selected and coefficient-mapped aether Lagrangians."""

    stiffness = _finite_positive(k_b, name="k_b")
    speed = _unit_interval(u, name="u")
    selected = (
        -0.5 * stiffness * invariants.field_strength_squared
        + 0.5 * stiffness * (1.0 - speed) * invariants.magnetic_squared
    )
    mapped = (
        -0.5 * stiffness * speed * invariants.field_strength_squared
        + stiffness * (1.0 - speed) * invariants.acceleration_squared
    )
    return {
        "selected_density": float(selected),
        "mapped_density": float(mapped),
        "residual": float(selected - mapped),
    }


def audit_v10c_covariant_ppn_precheck(*, k_b: float, u: float) -> dict[str, object]:
    """Run the exact coefficient map and delimit the valid PPN conclusion."""

    metric = np.diag([-1.0, 1.0, 1.0, 1.0])
    aether = np.array([1.0, 0.0, 0.0, 0.0])
    field = np.array(
        [
            [0.0, 0.7, -0.2, 0.4],
            [-0.7, 0.0, 0.3, -0.5],
            [0.2, -0.3, 0.0, 0.6],
            [-0.4, 0.5, -0.6, 0.0],
        ]
    )
    invariants = electric_magnetic_invariants(metric, aether, field)
    densities = aether_action_density_pair(invariants, k_b=k_b, u=u)
    mapped = mapped_einstein_aether_coefficients(k_b=k_b, u=u)
    base = mapped_einstein_aether_coefficients(k_b=k_b, u=1.0)
    mapped_alpha1 = pure_einstein_aether_alpha1(
        c1=mapped["c1"], c3=mapped["c3"], c4=mapped["c4"]
    )
    base_alpha1 = pure_einstein_aether_alpha1(
        c1=base["c1"], c3=base["c3"], c4=base["c4"]
    )
    exact_gates = {
        "unit_aether_electric_magnetic_identity": abs(invariants.decomposition_residual)
        < 1.0e-12,
        "selected_and_mapped_action_densities_equal": abs(densities["residual"])
        < 1.0e-12,
        "c13_remains_zero": abs(mapped["c13"]) < 1.0e-12,
        "c14_remains_KB": np.isclose(mapped["c14"], k_b, rtol=0.0, atol=1.0e-12),
        "mapped_vector_speed_is_u": np.isclose(
            mapped["pure_aether_vector_speed_squared"], u, rtol=0.0, atol=1.0e-12
        ),
        "pure_aether_alpha1_proxy_unchanged_from_base": np.isclose(
            mapped_alpha1, base_alpha1, rtol=0.0, atol=1.0e-12
        ),
        "pure_aether_alpha2_formula_is_inapplicable": abs(mapped["c123"]) < 1.0e-12,
    }
    return {
        "invariants": {
            "F_squared": invariants.field_strength_squared,
            "B_squared": invariants.magnetic_squared,
            "J_squared": invariants.acceleration_squared,
            "identity_residual": invariants.decomposition_residual,
        },
        "action_density_map": densities,
        "mapped_coefficients": mapped,
        "base_maxwell_coefficients": base,
        "pure_einstein_aether_proxy": {
            "base_alpha1": base_alpha1,
            "v10c_alpha1": mapped_alpha1,
            "alpha1_closed_form": "-4 K_B",
            "alpha2_applicable": False,
            "reason": "c123=0 removes the pure-aether longitudinal gradient; AeST scalar and P dynamics are absent from that theory",
        },
        "first_order_interaction": {
            "second_derivative_form": "beta P^mn nabla_m J_n",
            "boundary_equivalent_form": "-beta (nabla_m P^mn) J_n",
            "aether_euler_term_modulo_spatial_constraint": "beta nabla_r(A^r C^s)-beta C^n nabla^s A_n, C^n=nabla_m P^mn",
            "derivative_order": "first derivatives in the action; at most second derivatives in Euler equations",
        },
        "exact_gates": {name: bool(value) for name, value in exact_gates.items()},
        "all_exact_precheck_gates_pass": bool(all(exact_gates.values())),
        "full_AeST_plus_P_PPN_derived": False,
        "counterterm_independently_retired": False,
    }
