"""Covariant constant-gradient quadratic density for Sigma v12A.

The earlier local audit used scalar-unitary gauge, ``d_mu phi=(q,0,0,0)``.
This module evaluates the same AeST plus luminal-Class-Ia action for an
arbitrary constant timelike scalar covector.  It provides the prerequisite for
testing candidate time covectors related by physical-metric Lorentz boosts.

The AeST density is written in manifest four-dimensional form,

``-K_B F^2/2 + 2(2-K_B) J.dphi -(2-K_B)Y
  -(2-K_B)a_sigma^2 f(Y/a_sigma^2)+2K2(Q-Q0)^2``,

where ``Q=U.dphi``, ``Y=X+Q^2``, and ``J^mu=U^nu nabla_nu U^mu``.  Unit-aether
normalization is exact through the ADM spatial covector parametrization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from scipy import linalg, optimize

from voidscreen.sigma_v12a_tilted_principal import (
    AETHER_SLICE,
    FIELD_COUNT,
    _activation,
    _fierz_pauli_density,
    _metric_derivatives,
    _simple_aest_f,
    _spatial_metric_perturbation,
)

SPATIAL_GAUGE_BASE_INDICES = (6, 8, 9)


@dataclass(frozen=True)
class GeneralCovectorBackground:
    """Constant local background in a general physical-metric frame."""

    scalar_covector: tuple[float, float, float, float]
    aether_spatial_covector: tuple[float, float, float]
    background_clock_ratio: float = 1.0
    orientation_strength: float = 1.0
    k_b: float = 1.0
    k_2: float = 2.0
    a_sigma: float = 1.0
    f_0: float = 1.0

    def validated(self) -> GeneralCovectorBackground:
        scalar = np.asarray(self.scalar_covector, dtype=float)
        aether = np.asarray(self.aether_spatial_covector, dtype=float)
        parameters = np.asarray(
            [
                self.background_clock_ratio,
                self.orientation_strength,
                self.k_b,
                self.k_2,
                self.a_sigma,
                self.f_0,
            ],
            dtype=float,
        )
        if scalar.shape != (4,) or aether.shape != (3,):
            raise ValueError("scalar and aether backgrounds must have shapes (4,) and (3,)")
        if np.any(~np.isfinite(scalar)) or np.any(~np.isfinite(aether)):
            raise ValueError("scalar and aether backgrounds must be finite")
        if np.any(~np.isfinite(parameters)):
            raise ValueError("general-covector parameters must be finite")
        scalar_norm = -scalar[0] ** 2 + float(scalar[1:] @ scalar[1:])
        if scalar_norm >= 0.0 or scalar[0] == 0.0:
            raise ValueError("the constant scalar covector must be timelike")
        if self.background_clock_ratio == 0.0:
            raise ValueError("background_clock_ratio must be nonzero")
        if not 0.0 < self.k_b < 2.0:
            raise ValueError("the selected AeST row requires 0<K_B<2")
        if self.k_2 <= 0.0 or self.a_sigma <= 0.0 or self.f_0 <= 0.0:
            raise ValueError("K2, a_sigma, and F0 must be positive")
        return self


def general_covector_background_invariants(
    background: GeneralCovectorBackground,
) -> dict[str, float]:
    """Return the flat-frame invariants and constant aether equation residual.

    On a constant flat background the scalar equation is automatic because the
    action is shift symmetric.  The projected aether equation is either
    satisfied when the scalar and aether clocks are aligned (``Y=0``), or when
    ``dL/dQ=0``.  Its invariant norm is ``sqrt(Y) |dL/dQ|``.
    """

    background.validated()
    scalar = np.asarray(background.scalar_covector, dtype=float)
    aether_spatial = np.asarray(background.aether_spatial_covector, dtype=float)
    aether = np.concatenate(
        ([np.sqrt(1.0 + float(aether_spatial @ aether_spatial))], aether_spatial)
    )
    scalar_norm = -scalar[0] ** 2 + float(scalar[1:] @ scalar[1:])
    aether_norm = -aether[0] ** 2 + float(aether[1:] @ aether[1:])
    q_value = float(aether @ scalar)
    y_value = scalar_norm + q_value**2
    roundoff_scale = max(1.0, abs(scalar_norm), q_value**2)
    if y_value < -1.0e-12 * roundoff_scale:
        raise ValueError("Y=X+Q^2 must be nonnegative for timelike clocks")
    y_value = max(0.0, y_value)
    root_y = np.sqrt(y_value) / float(background.a_sigma)
    f_prime = root_y / (1.0 + root_y)
    d_l_d_q = (
        -2.0 * (2.0 - float(background.k_b)) * q_value * (1.0 + f_prime)
        + 4.0
        * float(background.k_2)
        * (q_value - float(background.background_clock_ratio))
    )
    return {
        "scalar_norm_x": float(scalar_norm),
        "aether_norm": float(aether_norm),
        "aether_clock_q": q_value,
        "projected_scalar_norm_y": float(y_value),
        "aest_f_prime": float(f_prime),
        "d_l_d_q": float(d_l_d_q),
        "projected_aether_eom_residual": float(np.sqrt(y_value) * abs(d_l_d_q)),
    }


def solve_tilted_constant_branch_scalar_clock(
    *,
    tilt_magnitude: float,
    background_clock_ratio: float = 1.0,
    k_b: float = 1.0,
    k_2: float = 2.0,
    a_sigma: float = 1.0,
) -> float:
    """Solve the nonaligned constant-background aether equation for ``p_0``."""

    tilt = float(tilt_magnitude)
    if not np.isfinite(tilt) or tilt <= 0.0:
        raise ValueError("tilt_magnitude must be finite and positive")
    template = GeneralCovectorBackground(
        scalar_covector=(1.0, 0.0, 0.0, 0.0),
        aether_spatial_covector=(0.0, 0.0, tilt),
        background_clock_ratio=float(background_clock_ratio),
        k_b=float(k_b),
        k_2=float(k_2),
        a_sigma=float(a_sigma),
    ).validated()

    def equation(clock: float) -> float:
        row = GeneralCovectorBackground(
            scalar_covector=(float(clock), 0.0, 0.0, 0.0),
            aether_spatial_covector=template.aether_spatial_covector,
            background_clock_ratio=template.background_clock_ratio,
            k_b=template.k_b,
            k_2=template.k_2,
            a_sigma=template.a_sigma,
            f_0=template.f_0,
        )
        return general_covector_background_invariants(row)["d_l_d_q"]

    lower = np.finfo(float).eps
    upper = max(2.0, 2.0 * abs(float(background_clock_ratio)))
    lower_value = equation(lower)
    upper_value = equation(upper)
    while lower_value * upper_value > 0.0 and upper < 1.0e8:
        upper *= 2.0
        upper_value = equation(upper)
    if lower_value * upper_value > 0.0:
        raise ValueError("no positive tilted constant branch was bracketed")
    return float(optimize.brentq(equation, lower, upper, xtol=1.0e-14, rtol=1.0e-14))


def _metric_and_connection(
    fields: torch.Tensor,
    time_derivatives: torch.Tensor,
    spatial_derivatives: torch.Tensor,
) -> dict[str, torch.Tensor]:
    lapse = 1.0 + fields[0]
    shift = fields[1:4]
    spatial_metric = torch.eye(3, dtype=fields.dtype, device=fields.device)
    spatial_metric = spatial_metric + _spatial_metric_perturbation(fields)
    inverse_spatial_metric = torch.linalg.inv(spatial_metric)
    shift_covariant = spatial_metric @ shift
    metric = torch.zeros((4, 4), dtype=fields.dtype, device=fields.device)
    metric[0, 0] = -(lapse**2) + shift @ shift_covariant
    metric[0, 1:] = shift_covariant
    metric[1:, 0] = shift_covariant
    metric[1:, 1:] = spatial_metric
    inverse_metric = torch.linalg.inv(metric)

    field_derivatives = torch.zeros((4, FIELD_COUNT), dtype=fields.dtype)
    field_derivatives[0] = time_derivatives
    field_derivatives[3] = spatial_derivatives
    spatial_metric_derivatives = torch.zeros((4, 3, 3), dtype=fields.dtype)
    metric_derivatives = torch.zeros((4, 4, 4), dtype=fields.dtype)
    for derivative_index in range(4):
        derivative_fields = field_derivatives[derivative_index]
        derivative_lapse = derivative_fields[0]
        derivative_shift = derivative_fields[1:4]
        derivative_spatial_metric = _spatial_metric_perturbation(derivative_fields)
        spatial_metric_derivatives[derivative_index] = derivative_spatial_metric
        derivative_shift_covariant = (
            derivative_spatial_metric @ shift + spatial_metric @ derivative_shift
        )
        metric_derivatives[derivative_index, 0, 0] = (
            -2.0 * lapse * derivative_lapse
            + derivative_shift @ shift_covariant
            + shift @ derivative_shift_covariant
        )
        metric_derivatives[derivative_index, 0, 1:] = derivative_shift_covariant
        metric_derivatives[derivative_index, 1:, 0] = derivative_shift_covariant
        metric_derivatives[derivative_index, 1:, 1:] = derivative_spatial_metric
    connection = torch.zeros((4, 4, 4), dtype=fields.dtype)
    for rho in range(4):
        for mu in range(4):
            for nu in range(4):
                connection[rho, mu, nu] = 0.5 * sum(
                    inverse_metric[rho, sigma]
                    * (
                        metric_derivatives[mu, nu, sigma]
                        + metric_derivatives[nu, mu, sigma]
                        - metric_derivatives[sigma, mu, nu]
                    )
                    for sigma in range(4)
                )
    return {
        "lapse": lapse,
        "shift": shift,
        "spatial_metric": spatial_metric,
        "inverse_spatial_metric": inverse_spatial_metric,
        "inverse_metric": inverse_metric,
        "field_derivatives": field_derivatives,
        "spatial_metric_derivatives": spatial_metric_derivatives,
        "metric_derivatives": metric_derivatives,
        "connection": connection,
    }


def _aether_geometry(
    fields: torch.Tensor,
    geometry: dict[str, torch.Tensor],
    *,
    background: GeneralCovectorBackground,
) -> dict[str, torch.Tensor]:
    lapse = geometry["lapse"]
    shift = geometry["shift"]
    inverse_spatial_metric = geometry["inverse_spatial_metric"]
    field_derivatives = geometry["field_derivatives"]
    spatial_metric_derivatives = geometry["spatial_metric_derivatives"]
    connection = geometry["connection"]
    aether_background = torch.tensor(
        background.aether_spatial_covector,
        dtype=fields.dtype,
        device=fields.device,
    )
    aether_covariant_spatial = aether_background + fields[AETHER_SLICE]
    aether_spatial_up = inverse_spatial_metric @ aether_covariant_spatial
    chi = torch.sqrt(1.0 + aether_covariant_spatial @ aether_spatial_up)
    inverse_spatial_metric_derivatives = -torch.einsum(
        "ma,lab,bn->lmn",
        inverse_spatial_metric,
        spatial_metric_derivatives,
        inverse_spatial_metric,
    )
    chi_derivatives = torch.zeros(4, dtype=fields.dtype)
    aether_spatial_up_derivatives = torch.zeros((4, 3), dtype=fields.dtype)
    for derivative_index in range(4):
        derivative_aether = field_derivatives[derivative_index, AETHER_SLICE]
        aether_spatial_up_derivatives[derivative_index] = (
            inverse_spatial_metric_derivatives[derivative_index]
            @ aether_covariant_spatial
            + inverse_spatial_metric @ derivative_aether
        )
        chi_derivatives[derivative_index] = (
            torch.einsum(
                "mn,m,n->",
                inverse_spatial_metric_derivatives[derivative_index],
                aether_covariant_spatial,
                aether_covariant_spatial,
            )
            + 2.0 * (aether_spatial_up @ derivative_aether)
        ) / (2.0 * chi)

    aether_contravariant = torch.zeros(4, dtype=fields.dtype)
    aether_contravariant[0] = chi / lapse
    aether_contravariant[1:] = aether_spatial_up - chi * shift / lapse
    aether_covariant = torch.zeros(4, dtype=fields.dtype)
    aether_covariant[0] = -lapse * chi + shift @ aether_covariant_spatial
    aether_covariant[1:] = aether_covariant_spatial

    aether_contravariant_derivatives = torch.zeros((4, 4), dtype=fields.dtype)
    aether_covariant_derivatives = torch.zeros((4, 4), dtype=fields.dtype)
    for derivative_index in range(4):
        derivative_fields = field_derivatives[derivative_index]
        derivative_lapse = derivative_fields[0]
        derivative_shift = derivative_fields[1:4]
        derivative_aether = derivative_fields[AETHER_SLICE]
        derivative_chi = chi_derivatives[derivative_index]
        aether_contravariant_derivatives[derivative_index, 0] = (
            derivative_chi / lapse - chi * derivative_lapse / lapse**2
        )
        aether_contravariant_derivatives[derivative_index, 1:] = (
            aether_spatial_up_derivatives[derivative_index]
            - derivative_chi * shift / lapse
            - chi * derivative_shift / lapse
            + chi * shift * derivative_lapse / lapse**2
        )
        aether_covariant_derivatives[derivative_index, 0] = (
            -derivative_lapse * chi
            - lapse * derivative_chi
            + derivative_shift @ aether_covariant_spatial
            + shift @ derivative_aether
        )
        aether_covariant_derivatives[derivative_index, 1:] = derivative_aether

    covariant_derivative_up = torch.zeros((4, 4), dtype=fields.dtype)
    for derivative_index in range(4):
        for component in range(4):
            covariant_derivative_up[derivative_index, component] = (
                aether_contravariant_derivatives[derivative_index, component]
                + sum(
                    connection[component, derivative_index, rho]
                    * aether_contravariant[rho]
                    for rho in range(4)
                )
            )
    acceleration = torch.einsum(
        "n,nm->m",
        aether_contravariant,
        covariant_derivative_up,
    )
    field_strength = (
        aether_covariant_derivatives - aether_covariant_derivatives.T
    )
    return {
        "contravariant": aether_contravariant,
        "covariant": aether_covariant,
        "acceleration": acceleration,
        "field_strength": field_strength,
    }


def _general_covariant_density(
    fields: torch.Tensor,
    time_derivatives: torch.Tensor,
    spatial_derivatives: torch.Tensor,
    *,
    background: GeneralCovectorBackground,
) -> torch.Tensor:
    background.validated()
    geometry = _metric_and_connection(fields, time_derivatives, spatial_derivatives)
    inverse_metric = geometry["inverse_metric"]
    spatial_metric = geometry["spatial_metric"]
    connection = geometry["connection"]
    aether = _aether_geometry(fields, geometry, background=background)
    scalar_covector = torch.tensor(
        background.scalar_covector,
        dtype=fields.dtype,
        device=fields.device,
    )
    scalar_contravector = inverse_metric @ scalar_covector
    x_value = scalar_covector @ scalar_contravector
    q_value = aether["contravariant"] @ scalar_covector
    y_value = x_value + q_value**2

    field_strength = aether["field_strength"]
    field_strength_squared = torch.einsum(
        "mn,ma,nb,ab->",
        field_strength,
        inverse_metric,
        inverse_metric,
        field_strength,
    )
    maxwell = -0.5 * float(background.k_b) * field_strength_squared
    aether_scalar = 2.0 * (2.0 - float(background.k_b)) * (
        aether["acceleration"] @ scalar_covector
    )
    aest_scalar = (
        -(2.0 - float(background.k_b)) * y_value
        - (2.0 - float(background.k_b))
        * float(background.a_sigma) ** 2
        * _simple_aest_f(y_value / float(background.a_sigma) ** 2)
        + 2.0
        * float(background.k_2)
        * (q_value - float(background.background_clock_ratio)) ** 2
    )

    scalar_hessian = -torch.einsum("rmn,r->mn", connection, scalar_covector)
    box_scalar = torch.einsum("mn,mn->", inverse_metric, scalar_hessian)
    scalar_hessian_inner = scalar_contravector @ scalar_hessian @ scalar_contravector
    scalar_hessian_up = inverse_metric @ scalar_hessian @ inverse_metric
    l_3 = box_scalar * scalar_hessian_inner
    l_4 = scalar_contravector @ scalar_hessian @ scalar_hessian_up @ scalar_covector
    l_5 = scalar_hessian_inner**2
    x_zero = -(float(background.background_clock_ratio) ** 2)
    a_3 = (
        float(background.orientation_strength)
        * float(background.f_0)
        * _activation(x_value, x_zero)
    )
    a_4 = -a_3 - x_value**2 * a_3**2 / (8.0 * float(background.f_0))
    a_5 = x_value * a_3**2 / (2.0 * float(background.f_0))
    dhost = a_3 * l_3 + a_4 * l_4 + a_5 * l_5

    determinant_factor = geometry["lapse"] * torch.sqrt(torch.linalg.det(spatial_metric))
    return _fierz_pauli_density(
        _metric_derivatives(time_derivatives, spatial_derivatives),
        f_0=background.f_0,
    ) + determinant_factor * (maxwell + aether_scalar + aest_scalar + dhost)


def general_covector_mode_lagrangian_hessian(
    background: GeneralCovectorBackground,
    *,
    wave_number_ratio: float,
) -> dict[str, np.ndarray | float]:
    """Return ``K,A,B`` for a general scalar covector and a wave on axis 3."""

    background.validated()
    wave_number = float(wave_number_ratio)
    if not np.isfinite(wave_number) or wave_number <= 0.0:
        raise ValueError("wave_number_ratio must be finite and positive")
    size = 2 * FIELD_COUNT

    def averaged_lagrangian(joint: torch.Tensor) -> torch.Tensor:
        velocities = joint[:size]
        amplitudes = joint[size:]
        velocity_cos = velocities[:FIELD_COUNT]
        velocity_sin = velocities[FIELD_COUNT:]
        amplitude_cos = amplitudes[:FIELD_COUNT]
        amplitude_sin = amplitudes[FIELD_COUNT:]
        derivative_cos = wave_number * amplitude_sin
        derivative_sin = -wave_number * amplitude_cos
        return 0.5 * (
            _general_covariant_density(
                amplitude_cos,
                velocity_cos,
                derivative_cos,
                background=background,
            )
            + _general_covariant_density(
                amplitude_sin,
                velocity_sin,
                derivative_sin,
                background=background,
            )
        )

    origin = torch.zeros(2 * size, requires_grad=True)
    hessian = torch.autograd.functional.hessian(
        averaged_lagrangian,
        origin,
        vectorize=True,
    ).detach().cpu().numpy()
    hessian = 0.5 * (hessian + hessian.T)
    return {
        "K": hessian[:size, :size],
        "A": hessian[:size, size:],
        "B": hessian[size:, size:],
        "wave_number_ratio": wave_number,
    }


def unitary_hessian_parity(
    *,
    scalar_clock_ratio: float,
    aether_parallel: float,
    aether_perpendicular: float,
    orientation_strength: float,
    wave_number_ratio: float,
    k_b: float = 1.0,
    k_2: float = 2.0,
) -> dict[str, float]:
    """Compare the covariant density with the established unitary ADM form."""

    from voidscreen.sigma_v12a_tilted_principal import (  # local import avoids a cycle
        TiltedPrincipalBackground,
        mode_lagrangian_hessian,
    )

    unitary_background = TiltedPrincipalBackground(
        scalar_clock_ratio=float(scalar_clock_ratio),
        aether_parallel=float(aether_parallel),
        aether_perpendicular=float(aether_perpendicular),
        orientation_strength=float(orientation_strength),
        k_b=float(k_b),
        k_2=float(k_2),
    )
    covariant_background = GeneralCovectorBackground(
        scalar_covector=(float(scalar_clock_ratio), 0.0, 0.0, 0.0),
        aether_spatial_covector=(
            float(aether_perpendicular),
            0.0,
            float(aether_parallel),
        ),
        orientation_strength=float(orientation_strength),
        k_b=float(k_b),
        k_2=float(k_2),
    )
    reference = mode_lagrangian_hessian(
        unitary_background,
        wave_number_ratio=wave_number_ratio,
    )
    covariant = general_covector_mode_lagrangian_hessian(
        covariant_background,
        wave_number_ratio=wave_number_ratio,
    )
    output = {}
    for name in ("K", "A", "B"):
        reference_matrix = np.asarray(reference[name], dtype=float)
        covariant_matrix = np.asarray(covariant[name], dtype=float)
        scale = max(1.0, float(np.max(np.abs(reference_matrix))))
        output[f"maximum_normalized_{name}_residual"] = float(
            np.max(np.abs(covariant_matrix - reference_matrix)) / scale
        )
    return output


def lorentz_boost_contravariant(
    vector: np.ndarray | tuple[float, float, float, float],
    velocity: np.ndarray | tuple[float, float, float],
) -> np.ndarray:
    """Return contravariant components in a frame boosted by ``velocity``."""

    value = np.asarray(vector, dtype=float)
    boost = np.asarray(velocity, dtype=float)
    if value.shape != (4,) or boost.shape != (3,):
        raise ValueError("Lorentz vectors and velocities must have shapes (4,) and (3,)")
    speed_squared = float(boost @ boost)
    if not np.isfinite(speed_squared) or speed_squared >= 1.0:
        raise ValueError("boost velocity must be finite and subluminal")
    if speed_squared == 0.0:
        return value.copy()
    gamma = 1.0 / np.sqrt(1.0 - speed_squared)
    spatial_dot = float(boost @ value[1:])
    output = np.empty(4, dtype=float)
    output[0] = gamma * (value[0] - spatial_dot)
    output[1:] = value[1:] + (
        (gamma - 1.0) * spatial_dot / speed_squared - gamma * value[0]
    ) * boost
    return output


def lorentz_boost_covector(
    covector: np.ndarray | tuple[float, float, float, float],
    velocity: np.ndarray | tuple[float, float, float],
) -> np.ndarray:
    """Return covariant components in a frame boosted by ``velocity``."""

    value = np.asarray(covector, dtype=float)
    boost = np.asarray(velocity, dtype=float)
    if value.shape != (4,) or boost.shape != (3,):
        raise ValueError("Lorentz covectors and velocities must have shapes (4,) and (3,)")
    speed_squared = float(boost @ boost)
    if not np.isfinite(speed_squared) or speed_squared >= 1.0:
        raise ValueError("boost velocity must be finite and subluminal")
    if speed_squared == 0.0:
        return value.copy()
    gamma = 1.0 / np.sqrt(1.0 - speed_squared)
    spatial_dot = float(boost @ value[1:])
    output = np.empty(4, dtype=float)
    output[0] = gamma * (value[0] + spatial_dot)
    output[1:] = value[1:] + (
        (gamma - 1.0) * spatial_dot / speed_squared + gamma * value[0]
    ) * boost
    return output


def boosted_unitary_background(
    *,
    scalar_clock_ratio: float,
    aether_spatial_covector: tuple[float, float, float],
    boost_velocity: tuple[float, float, float],
    background_clock_ratio: float = 1.0,
    orientation_strength: float = 1.0,
    k_b: float = 1.0,
    k_2: float = 2.0,
    a_sigma: float = 1.0,
    f_0: float = 1.0,
) -> GeneralCovectorBackground:
    """Boost a scalar-unitary constant background into a general frame."""

    spatial = np.asarray(aether_spatial_covector, dtype=float)
    if spatial.shape != (3,):
        raise ValueError("aether_spatial_covector must have shape (3,)")
    aether = np.concatenate(([np.sqrt(1.0 + float(spatial @ spatial))], spatial))
    scalar = np.asarray([float(scalar_clock_ratio), 0.0, 0.0, 0.0])
    boosted_aether = lorentz_boost_contravariant(aether, boost_velocity)
    boosted_scalar = lorentz_boost_covector(scalar, boost_velocity)
    return GeneralCovectorBackground(
        scalar_covector=tuple(float(value) for value in boosted_scalar),
        aether_spatial_covector=tuple(float(value) for value in boosted_aether[1:]),
        background_clock_ratio=float(background_clock_ratio),
        orientation_strength=float(orientation_strength),
        k_b=float(k_b),
        k_2=float(k_2),
        a_sigma=float(a_sigma),
        f_0=float(f_0),
    ).validated()


def lorentz_invariant_background_residuals(
    reference: GeneralCovectorBackground,
    transformed: GeneralCovectorBackground,
) -> dict[str, float]:
    """Compare the four scalar/aether invariants of two background frames."""

    left = general_covector_background_invariants(reference)
    right = general_covector_background_invariants(transformed)
    keys = (
        "scalar_norm_x",
        "aether_norm",
        "aether_clock_q",
        "projected_scalar_norm_y",
    )
    return {
        f"absolute_{key}_residual": abs(float(left[key]) - float(right[key]))
        for key in keys
    }


def rotate_background_to_wave_axis(
    background: GeneralCovectorBackground,
    wave_direction: np.ndarray | tuple[float, float, float],
) -> GeneralCovectorBackground:
    """Rotate spatial components so the supplied wave direction becomes axis 3."""

    direction = np.asarray(wave_direction, dtype=float)
    if direction.shape != (3,) or np.any(~np.isfinite(direction)):
        raise ValueError("wave_direction must be a finite three-vector")
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        raise ValueError("wave_direction must be nonzero")
    e_3 = direction / norm
    reference = np.asarray([1.0, 0.0, 0.0])
    if abs(float(reference @ e_3)) > 0.9:
        reference = np.asarray([0.0, 1.0, 0.0])
    e_1 = reference - float(reference @ e_3) * e_3
    e_1 = e_1 / np.linalg.norm(e_1)
    e_2 = np.cross(e_3, e_1)
    basis = np.stack((e_1, e_2, e_3))
    scalar = np.asarray(background.scalar_covector, dtype=float)
    aether = np.asarray(background.aether_spatial_covector, dtype=float)
    rotated_scalar = np.concatenate(([scalar[0]], basis @ scalar[1:]))
    rotated_aether = basis @ aether
    return GeneralCovectorBackground(
        scalar_covector=tuple(float(value) for value in rotated_scalar),
        aether_spatial_covector=tuple(float(value) for value in rotated_aether),
        background_clock_ratio=background.background_clock_ratio,
        orientation_strength=background.orientation_strength,
        k_b=background.k_b,
        k_2=background.k_2,
        a_sigma=background.a_sigma,
        f_0=background.f_0,
    ).validated()


def general_covector_characteristic_eigensystem(
    background: GeneralCovectorBackground,
    *,
    wave_number_ratio: float,
    infinity_tolerance: float = 1.0e-6,
) -> dict[str, object]:
    """Return the homogeneous gauge-fixed characteristic eigensystem."""

    tolerance = float(infinity_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("infinity_tolerance must be finite and positive")
    mode = general_covector_mode_lagrangian_hessian(
        background,
        wave_number_ratio=wave_number_ratio,
    )
    full_kinetic = np.asarray(mode["K"], dtype=float)
    full_mixing = np.asarray(mode["A"], dtype=float)
    full_potential = np.asarray(mode["B"], dtype=float)
    gauge_indices = (
        *SPATIAL_GAUGE_BASE_INDICES,
        *(index + FIELD_COUNT for index in SPATIAL_GAUGE_BASE_INDICES),
    )
    retained = np.asarray(
        [index for index in range(2 * FIELD_COUNT) if index not in set(gauge_indices)],
        dtype=int,
    )
    kinetic = full_kinetic[np.ix_(retained, retained)]
    gyroscopic = (full_mixing - full_mixing.T)[np.ix_(retained, retained)]
    potential = full_potential[np.ix_(retained, retained)]
    size = retained.size
    zero = np.zeros((size, size), dtype=float)
    identity = np.eye(size, dtype=float)
    left = np.block([[zero, identity], [potential, -gyroscopic]])
    right = np.block([[identity, zero], [zero, kinetic]])
    homogeneous, eigenvectors = linalg.eig(
        left,
        right,
        homogeneous_eigvals=True,
    )
    alpha, beta = homogeneous
    finite_mask = np.abs(beta) > tolerance * np.maximum(1.0, np.abs(alpha))
    return {
        "K": kinetic,
        "C": gyroscopic,
        "B": potential,
        "finite_roots": alpha[finite_mask] / beta[finite_mask],
        "finite_eigenvectors": eigenvectors[:, finite_mask],
        "finite_generalized_root_count": int(np.count_nonzero(finite_mask)),
        "infinite_generalized_root_count": int(np.count_nonzero(~finite_mask)),
        "minimum_finite_homogeneous_beta_margin": float(
            np.min(
                np.abs(beta[finite_mask])
                / (tolerance * np.maximum(1.0, np.abs(alpha[finite_mask])))
            )
        ),
        "maximum_infinite_homogeneous_beta_margin": float(
            np.max(
                np.abs(beta[~finite_mask])
                / (tolerance * np.maximum(1.0, np.abs(alpha[~finite_mask])))
            )
        ),
    }


def general_covector_characteristic_row(
    background: GeneralCovectorBackground,
    *,
    wave_number_ratio: float,
    principal_growth_threshold: float = 1.0e-2,
    metric_cone_frequency_tolerance: float = 1.0e-2,
) -> dict[str, object]:
    """Return compact principal growth, cone, and coordinate-energy checks."""

    wave_number = float(wave_number_ratio)
    growth_threshold = float(principal_growth_threshold)
    cone_tolerance = float(metric_cone_frequency_tolerance)
    if any(
        not np.isfinite(value) or value <= 0.0
        for value in (wave_number, growth_threshold, cone_tolerance)
    ):
        raise ValueError("wave number and characteristic thresholds must be positive")
    eigensystem = general_covector_characteristic_eigensystem(
        background,
        wave_number_ratio=wave_number,
    )
    roots = np.asarray(eigensystem["finite_roots"], dtype=complex)
    eigenvectors = np.asarray(eigensystem["finite_eigenvectors"], dtype=complex)
    kinetic = np.asarray(eigensystem["K"], dtype=float)
    gyroscopic = np.asarray(eigensystem["C"], dtype=float)
    potential = np.asarray(eigensystem["B"], dtype=float)
    growth = np.abs(roots.real) / wave_number
    frequency = np.abs(roots.imag) / wave_number
    energies: list[float] = []
    oscillatory_characteristic_norms: list[float] = []
    maximum_polynomial_residual = 0.0
    matrix_scale = (
        np.linalg.norm(kinetic),
        np.linalg.norm(gyroscopic),
        np.linalg.norm(potential),
    )
    for root, vector, growth_fraction, frequency_fraction in zip(
        roots,
        eigenvectors.T,
        growth,
        frequency,
        strict=True,
    ):
        amplitude = vector[: kinetic.shape[0]]
        amplitude = amplitude / np.linalg.norm(amplitude)
        polynomial = root**2 * kinetic + root * gyroscopic - potential
        scale = max(
            1.0,
            abs(root) ** 2 * matrix_scale[0]
            + abs(root) * matrix_scale[1]
            + matrix_scale[2],
        )
        maximum_polynomial_residual = max(
            maximum_polynomial_residual,
            float(np.linalg.norm(polynomial @ amplitude) / scale),
        )
        if growth_fraction < 2.0e-3 and frequency_fraction > 1.0e-3:
            omega = abs(float(root.imag))
            oscillatory_characteristic_norms.append(
                1.0 - float(frequency_fraction) ** 2
            )
            energies.append(
                0.25
                * float(
                    np.real(
                        np.vdot(
                            amplitude,
                            (omega**2 * kinetic - potential) @ amplitude,
                        )
                    )
                )
            )
    maximum_growth = float(np.max(growth))
    maximum_frequency = float(np.max(frequency))
    minimum_energy = float(min(energies)) if energies else None
    minimum_characteristic_norm = (
        float(min(oscillatory_characteristic_norms))
        if oscillatory_characteristic_norms
        else None
    )
    timelike_norm_threshold = -(
        2.0 * cone_tolerance + cone_tolerance**2
    )
    timelike_root_count = sum(
        value < timelike_norm_threshold for value in oscillatory_characteristic_norms
    )
    return {
        "background": asdict(background),
        "background_invariants": general_covector_background_invariants(background),
        "wave_number_ratio": wave_number,
        "finite_generalized_root_count": eigensystem[
            "finite_generalized_root_count"
        ],
        "infinite_generalized_root_count": eigensystem[
            "infinite_generalized_root_count"
        ],
        "maximum_normalized_exponential_growth": maximum_growth,
        "maximum_absolute_frequency_over_metric_light": maximum_frequency,
        "minimum_normalized_oscillatory_characteristic_covector_norm": (
            minimum_characteristic_norm
        ),
        "metric_timelike_oscillatory_characteristic_root_count": int(
            timelike_root_count
        ),
        "minimum_oscillatory_mode_energy": minimum_energy,
        "maximum_polynomial_residual": maximum_polynomial_residual,
        "scalar_unitary_or_boosted_time_hyperbolic": maximum_growth
        < growth_threshold,
        "frequencies_inside_metric_cone": timelike_root_count == 0,
        "identified_oscillatory_energies_nonnegative": minimum_energy is not None
        and minimum_energy >= -1.0e-8,
    }
