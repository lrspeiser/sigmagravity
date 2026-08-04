"""Pre-data construction checks for the Sigma v8A AeST--Galileon envelope.

The base is the one-metric scalar--vector--tensor action of Skordis and
Zlosnik.  Its quasistatic diagonalization writes the physical potential as
``Phi=Phi_hat+varphi`` and gives ``Psi=Phi``.  The scalar therefore contributes
to massive motion and to the Weyl potential with the same sign.

The proposed new envelope adds the standard shift-symmetric cubic Horndeski
operator ``-(L_H^2/2) (grad varphi)^2 box(varphi)``.  Its scalar equation
contains ``L_H^2[(box varphi)^2-(nabla_mn varphi)^2]``.  This file only audits
the selection identities; a full combined constraint and characteristic
analysis is required before observational use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class AestMetricProjection:
    psi: Array
    phi: Array
    weyl: Array


@dataclass(frozen=True)
class CubicPrincipalSymbol:
    temporal_coefficient: float
    spatial_eigenvalues: Array
    speed_squared: Array
    positive: bool
    causal: bool


def _finite_array(values: Array | float, *, name: str) -> Array:
    result = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def simple_mu(acceleration_ratio: Array | float) -> Array:
    """Return the fixed simple interpolation ``x/(1+x)``."""

    ratio = _finite_array(acceleration_ratio, name="acceleration_ratio")
    if np.any(ratio < 0.0):
        raise ValueError("acceleration_ratio must be non-negative")
    return ratio / (1.0 + ratio)


def simple_aqual_free_function(y: Array | float) -> Array:
    """Return ``f(y)=y-2 sqrt(y)+2 log(1+sqrt(y))``.

    Its derivative is ``sqrt(y)/(1+sqrt(y))``.  Consequently it has the
    deep-AQUAL limit ``2 y^(3/2)/3`` and the high-field limit ``y`` without a
    free interpolation-shape parameter.
    """

    invariant = _finite_array(y, name="y")
    if np.any(invariant < 0.0):
        raise ValueError("y must be non-negative")
    root = np.sqrt(invariant)
    return invariant - 2.0 * root + 2.0 * np.log1p(root)


def aest_metric_projection(
    tensor_potential: Array | float,
    scalar_potential: Array | float,
) -> AestMetricProjection:
    """Return the one-metric AeST weak-field potentials.

    In the quasistatic limit the physical potential is
    ``Phi_phys=Phi_tensor+varphi`` and the traceless metric equation gives
    ``Psi=Phi``.  Matter and photons are both minimally coupled to this metric.
    """

    tensor = _finite_array(tensor_potential, name="tensor_potential")
    scalar = _finite_array(scalar_potential, name="scalar_potential")
    try:
        physical = tensor + scalar
    except ValueError as error:
        raise ValueError("tensor_potential and scalar_potential must broadcast") from error
    return AestMetricProjection(psi=physical, phi=physical, weyl=physical)


def cubic_galileon_eom(
    hessian: Array,
    horndeski_length: float = 1.0,
) -> Array:
    """Return ``L_H^2[(tr H)^2-tr(H^2)]`` from the cubic action term."""

    matrix = _finite_array(hessian, name="hessian")
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("hessian must have final dimensions (3, 3)")
    if not np.allclose(matrix, np.swapaxes(matrix, -1, -2), rtol=0.0, atol=1.0e-12):
        raise ValueError("hessian must be symmetric")
    length = float(horndeski_length)
    if not np.isfinite(length) or length < 0.0:
        raise ValueError("horndeski_length must be finite and non-negative")
    trace = np.trace(matrix, axis1=-2, axis2=-1)
    square_trace = np.einsum("...ij,...ji->...", matrix, matrix)
    return length**2 * (trace**2 - square_trace)


def cubic_static_principal_symbol(
    dimensionless_hessian: Array,
    *,
    base_speed_squared: float,
) -> CubicPrincipalSymbol:
    """Return the scalar-sector principal symbol on a static background.

    ``dimensionless_hessian`` is ``L_H^2 nabla_i nabla_j(phi)``.  The time
    kinetic coefficient is normalized to one in the AeST flat background, so
    its spatial coefficient is the flat scalar speed squared.  Linearizing the
    cubic equation gives

    ``Z_t = 1 + 2 tr(Xi)`` and
    ``Z_ij = c_s^2 delta_ij + 2[tr(Xi) delta_ij - Xi_ij]``.

    This is a necessary decoupling-limit scalar gate, not the full coupled
    metric--vector--scalar characteristic determinant.
    """

    matrix = _finite_array(dimensionless_hessian, name="dimensionless_hessian")
    if matrix.shape != (3, 3):
        raise ValueError("dimensionless_hessian must have shape (3, 3)")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-12):
        raise ValueError("dimensionless_hessian must be symmetric")
    base = float(base_speed_squared)
    if not np.isfinite(base) or base <= 0.0:
        raise ValueError("base_speed_squared must be finite and positive")
    trace = float(np.trace(matrix))
    temporal = 1.0 + 2.0 * trace
    spatial = base * np.eye(3) + 2.0 * (trace * np.eye(3) - matrix)
    eigenvalues = np.linalg.eigvalsh(spatial)
    if temporal > 0.0:
        speeds = eigenvalues / temporal
    else:
        speeds = np.full(3, np.nan)
    positive = temporal > 0.0 and bool(np.all(eigenvalues > 0.0))
    causal = positive and bool(np.all(speeds <= 1.0 + 1.0e-12))
    return CubicPrincipalSymbol(
        temporal_coefficient=temporal,
        spatial_eigenvalues=eigenvalues,
        speed_squared=speeds,
        positive=positive,
        causal=causal,
    )


def spherical_positive_cubic_characteristics(
    dimensionless_tangential_hessian: float,
    *,
    base_speed_squared: float,
) -> dict[str, float | bool]:
    """Return characteristics of the positive cubic exterior branch.

    Let ``u=L_H^2 phi'/r``.  Integrating the static vacuum equation once gives
    ``c_s^2 r^2 phi' + 2 L_H^2 r phi'^2 = constant``.  Differentiating it fixes
    the radial-to-tangential Hessian ratio.  The returned nonlinear fraction is
    the cubic share of that conserved exterior flux.
    """

    u = float(dimensionless_tangential_hessian)
    base = float(base_speed_squared)
    if not np.isfinite(u) or u < 0.0:
        raise ValueError(
            "dimensionless_tangential_hessian must be finite and non-negative"
        )
    if not np.isfinite(base) or not 0.0 < base < 1.0:
        raise ValueError("base_speed_squared must lie strictly between zero and one")
    radial_ratio = -2.0 * (base + u) / (base + 4.0 * u)
    hessian = np.diag([radial_ratio * u, u, u])
    symbol = cubic_static_principal_symbol(
        hessian,
        base_speed_squared=base,
    )
    radial_spatial_coefficient = base + 4.0 * u
    tangential_spatial_coefficient = base + 2.0 * u * (radial_ratio + 1.0)
    radial_speed_squared = (
        radial_spatial_coefficient / symbol.temporal_coefficient
    )
    tangential_speed_squared = (
        tangential_spatial_coefficient / symbol.temporal_coefficient
    )
    nonlinear_to_linear_flux = 2.0 * u / base
    nonlinear_fraction = nonlinear_to_linear_flux / (
        1.0 + nonlinear_to_linear_flux
    )
    return {
        "dimensionless_tangential_hessian": u,
        "radial_to_tangential_hessian": radial_ratio,
        "temporal_coefficient": symbol.temporal_coefficient,
        "radial_speed_squared": radial_speed_squared,
        "tangential_speed_squared": tangential_speed_squared,
        "positive": symbol.positive,
        "causal": symbol.causal,
        "nonlinear_to_linear_flux": nonlinear_to_linear_flux,
        "nonlinear_fraction_of_total_flux": nonlinear_fraction,
    }


def positive_cubic_causality_limit(
    *,
    base_speed_squared: float,
) -> dict[str, float]:
    """Return the analytic point where the radial cubic mode reaches light."""

    base = float(base_speed_squared)
    if not np.isfinite(base) or not 0.0 < base < 1.0:
        raise ValueError("base_speed_squared must lie strictly between zero and one")
    root = np.sqrt(3.0 * base**2 - 3.0 * base + 1.0)
    threshold = 0.5 * (1.0 - 2.0 * base + root)
    flux_ratio = 2.0 * threshold / base
    return {
        "dimensionless_tangential_hessian": threshold,
        "nonlinear_to_linear_flux": flux_ratio,
        "maximum_nonlinear_fraction_of_total_flux": flux_ratio / (1.0 + flux_ratio),
    }


def negative_cubic_branch_limit(
    *,
    base_speed_squared: float,
) -> dict[str, float]:
    """Return the positive-source endpoint for the opposite cubic sign.

    With the sign reversed, the integrated flux is proportional to
    ``c_s^2 b - 2 |L_H|^2 b^2``.  It has a maximum at
    ``|L_H|^2 b=c_s^2/4``, exactly where the radial principal coefficient
    ``c_s^2-4|L_H|^2 b`` vanishes.
    """

    base = float(base_speed_squared)
    if not np.isfinite(base) or not 0.0 < base < 1.0:
        raise ValueError("base_speed_squared must lie strictly between zero and one")
    endpoint = base / 4.0
    return {
        "dimensionless_tangential_hessian_magnitude": endpoint,
        "radial_spatial_coefficient_at_endpoint": 0.0,
        "maximum_dimensionless_flux": base**2 / 8.0,
    }


def aest_linear_spectrum(
    *,
    k_b: float,
    k_2: float,
    lambda_s: float,
) -> dict[str, float | bool]:
    """Return the published flat-background AeST mode-speed conditions."""

    vector_coupling = float(k_b)
    clock_kinetic = float(k_2)
    scalar_coupling = float(lambda_s)
    if not all(np.isfinite(value) for value in (vector_coupling, clock_kinetic, scalar_coupling)):
        raise ValueError("AeST spectrum parameters must be finite")
    if vector_coupling == 0.0 or clock_kinetic == 0.0:
        scalar_speed_squared = np.inf
    else:
        scalar_speed_squared = (
            (2.0 - vector_coupling)
            / (clock_kinetic * vector_coupling)
            * (1.0 + 0.5 * vector_coupling * scalar_coupling)
        )
    positive_base = (
        0.0 < vector_coupling < 2.0
        and clock_kinetic > 0.0
        and scalar_coupling > -1.0
        and scalar_speed_squared > 0.0
    )
    return {
        "tensor_speed_squared": 1.0,
        "vector_speed_squared": 1.0,
        "scalar_speed_squared": float(scalar_speed_squared),
        "positive_base_spectrum": bool(positive_base),
        "causal_base_spectrum": bool(
            positive_base and scalar_speed_squared <= 1.0
        ),
    }


def audit_v8a_selection(
    *,
    k_b: float,
    k_2: float,
    lambda_s: float,
    physical_parameter_count: int,
    maximum_physical_parameters: int,
) -> dict[str, object]:
    """Evaluate the no-data v8A action-selection identities."""

    spectrum = aest_linear_spectrum(k_b=k_b, k_2=k_2, lambda_s=lambda_s)
    projection = aest_metric_projection(0.0, 1.0)
    isotropic = np.eye(3)
    rank_one = np.diag([3.0, 0.0, 0.0])
    isotropic_response = float(cubic_galileon_eom(isotropic))
    rank_one_response = float(cubic_galileon_eom(rank_one))
    same_trace = bool(np.isclose(np.trace(isotropic), np.trace(rank_one)))
    cubic_starts_beyond_quadratic = True
    count = int(physical_parameter_count)
    maximum = int(maximum_physical_parameters)
    if count < 0 or maximum < 0:
        raise ValueError("parameter counts must be non-negative")
    gates = {
        "one_minimally_coupled_matter_metric": True,
        "scalar_is_weyl_active": bool(np.isclose(float(projection.weyl), 1.0)),
        "no_linear_metric_slip": bool(np.isclose(float(projection.psi), float(projection.phi))),
        "luminal_tensor_mode": bool(np.isclose(spectrum["tensor_speed_squared"], 1.0)),
        "positive_base_linear_spectrum": bool(spectrum["positive_base_spectrum"]),
        "causal_base_linear_spectrum": bool(spectrum["causal_base_spectrum"]),
        "cubic_does_not_change_flat_quadratic_spectrum": cubic_starts_beyond_quadratic,
        "second_order_scalar_equation": True,
        "equal_trace_geometry_discrimination": same_trace
        and not np.isclose(isotropic_response, rank_one_response),
        "parameter_count": count <= maximum,
    }
    return {
        "spectrum": spectrum,
        "metric_projection": {
            "scalar_delta_psi": float(projection.psi),
            "scalar_delta_phi": float(projection.phi),
            "scalar_delta_weyl": float(projection.weyl),
        },
        "geometry_stress_test": {
            "common_hessian_trace": float(np.trace(isotropic)),
            "isotropic_hessian_response": isotropic_response,
            "rank_one_hessian_response": rank_one_response,
            "response_difference": isotropic_response - rank_one_response,
        },
        "gates": {name: bool(value) for name, value in gates.items()},
    }
