"""Local tilted-background principal-system audit for Sigma v12A.

The scalar gradient is assumed timelike, so a regular local unitary gauge can
set ``phi=q t``.  The relative orientation data are not lost: the unit aether
has an arbitrary spatial tilt and the Fourier wave vector is independently
oriented.  A passive spatial rotation puts the wave vector on the third axis
and the aether in the first/third plane; no metric component is gauge-fixed
inside the action.

This module constructs the exact quadratic local action with Torch automatic
differentiation, keeps the three spatial-diffeomorphism primaries separate,
and returns the finite-dimensional DHOST primary-secondary Dirac block.  It is
intentionally a local constant-background calculation; background Hessians
and curvature require a later extension.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

torch.set_default_dtype(torch.float64)


@dataclass(frozen=True)
class TiltedPrincipalBackground:
    """Dimensionless constant background used by the local symbol."""

    scalar_clock_ratio: float
    aether_parallel: float
    aether_perpendicular: float
    background_clock_ratio: float = 1.0
    orientation_strength: float = 1.0
    k_b: float = 1.0
    k_2: float = 2.0
    a_sigma: float = 1.0
    f_0: float = 1.0

    def validated(self) -> TiltedPrincipalBackground:
        values = np.asarray(
            [
                self.scalar_clock_ratio,
                self.aether_parallel,
                self.aether_perpendicular,
                self.background_clock_ratio,
                self.orientation_strength,
                self.k_b,
                self.k_2,
                self.a_sigma,
                self.f_0,
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(values)):
            raise ValueError("tilted-principal background values must be finite")
        if self.scalar_clock_ratio == 0.0 or self.background_clock_ratio == 0.0:
            raise ValueError("scalar clock ratios must be nonzero")
        if self.k_b <= 0.0 or self.k_2 <= 0.0 or self.a_sigma <= 0.0 or self.f_0 <= 0.0:
            raise ValueError("K_B, K2, a_sigma, and F0 must be positive")
        if abs(self.k_b) >= 2.0:
            raise ValueError("the selected AeST interpolation requires 0<K_B<2")
        return self


# Thirteen local ADM amplitudes:
# delta N,delta N^1,delta N^2,delta N^3,
# h11,h12,h13,h22,h23,h33,delta A_1,delta A_2,delta A_3.
FIELD_COUNT = 13
SHIFT_BASE_INDICES = (1, 2, 3)
AETHER_SLICE = slice(10, 13)


def _metric_perturbation(fields: torch.Tensor) -> torch.Tensor:
    """Return the linear four-metric perturbation from local ADM fields."""

    h = torch.zeros((4, 4), dtype=fields.dtype, device=fields.device)
    h[0, 0] = -2.0 * fields[0]
    h[0, 1] = h[1, 0] = fields[1]
    h[0, 2] = h[2, 0] = fields[2]
    h[0, 3] = h[3, 0] = fields[3]
    h[1, 1] = fields[4]
    h[1, 2] = h[2, 1] = fields[5]
    h[1, 3] = h[3, 1] = fields[6]
    h[2, 2] = fields[7]
    h[2, 3] = h[3, 2] = fields[8]
    h[3, 3] = fields[9]
    return h


def _spatial_metric_perturbation(fields: torch.Tensor) -> torch.Tensor:
    h = torch.zeros((3, 3), dtype=fields.dtype, device=fields.device)
    h[0, 0] = fields[4]
    h[0, 1] = h[1, 0] = fields[5]
    h[0, 2] = h[2, 0] = fields[6]
    h[1, 1] = fields[7]
    h[1, 2] = h[2, 1] = fields[8]
    h[2, 2] = fields[9]
    return h


def _metric_derivatives(
    time_derivatives: torch.Tensor,
    spatial_derivatives: torch.Tensor,
) -> torch.Tensor:
    derivatives = torch.zeros(
        (4, 4, 4),
        dtype=time_derivatives.dtype,
        device=time_derivatives.device,
    )
    derivatives[0] = _metric_perturbation(time_derivatives)
    derivatives[3] = _metric_perturbation(spatial_derivatives)
    return derivatives


def _fierz_pauli_density(metric_derivatives: torch.Tensor, *, f_0: float) -> torch.Tensor:
    """Quadratic Einstein-Hilbert density with the ``F0 R`` normalization."""

    eta = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0], dtype=metric_derivatives.dtype))
    derivative_up_metric = torch.einsum(
        "ma,lab,nb->lmn",
        eta,
        metric_derivatives,
        eta,
    )
    term_1 = -0.5 * torch.einsum(
        "ls,lmn,smn->",
        eta,
        metric_derivatives,
        derivative_up_metric,
    )
    divergence_up = torch.stack(
        [sum(derivative_up_metric[mu, mu, nu] for mu in range(4)) for nu in range(4)]
    )
    divergence_lower = torch.stack(
        [
            sum(
                eta[lam, sigma] * metric_derivatives[sigma, lam, nu]
                for lam in range(4)
                for sigma in range(4)
            )
            for nu in range(4)
        ]
    )
    trace_derivative = torch.stack(
        [torch.einsum("mn,mn->", eta, metric_derivatives[lam]) for lam in range(4)]
    )
    term_2 = divergence_up @ divergence_lower
    term_3 = -(divergence_up @ trace_derivative)
    term_4 = 0.5 * torch.einsum("ls,l,s->", eta, trace_derivative, trace_derivative)
    return 0.5 * float(f_0) * (term_1 + term_2 + term_3 + term_4)


def _simple_aest_f(y_value: torch.Tensor) -> torch.Tensor:
    root = torch.sqrt(y_value)
    return y_value - 2.0 * root + 2.0 * torch.log1p(root)


def _activation(x_value: torch.Tensor, x_zero: float) -> torch.Tensor:
    difference = x_value - float(x_zero)
    return difference**2 / ((1.0 + difference**2) ** 1.5 * torch.sqrt(1.0 + x_value**2))


def _local_covariant_density(
    fields: torch.Tensor,
    time_derivatives: torch.Tensor,
    spatial_derivatives: torch.Tensor,
    *,
    background: TiltedPrincipalBackground,
) -> torch.Tensor:
    """Return the local v12A ADM density through terms needed by its Hessian."""

    background.validated()
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

    field_derivatives = torch.zeros((4, FIELD_COUNT), dtype=fields.dtype, device=fields.device)
    field_derivatives[0] = time_derivatives
    field_derivatives[3] = spatial_derivatives
    spatial_metric_derivatives = torch.zeros(
        (4, 3, 3), dtype=fields.dtype, device=fields.device
    )
    metric_derivatives = torch.zeros((4, 4, 4), dtype=fields.dtype, device=fields.device)
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
    connection = torch.zeros((4, 4, 4), dtype=fields.dtype, device=fields.device)
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

    spatial_connection = torch.zeros((3, 3, 3), dtype=fields.dtype, device=fields.device)
    for rho in range(3):
        for mu in range(3):
            for nu in range(3):
                spatial_connection[rho, mu, nu] = 0.5 * sum(
                    inverse_spatial_metric[rho, sigma]
                    * (
                        spatial_metric_derivatives[mu + 1, nu, sigma]
                        + spatial_metric_derivatives[nu + 1, mu, sigma]
                        - spatial_metric_derivatives[sigma + 1, mu, nu]
                    )
                    for sigma in range(3)
                )

    aether_spatial_background = torch.tensor(
        [
            float(background.aether_perpendicular),
            0.0,
            float(background.aether_parallel),
        ],
        dtype=fields.dtype,
        device=fields.device,
    )
    aether_spatial = aether_spatial_background + fields[AETHER_SLICE]
    aether_spatial_up = inverse_spatial_metric @ aether_spatial
    chi = torch.sqrt(1.0 + aether_spatial @ aether_spatial_up)
    inverse_spatial_metric_derivatives = -torch.einsum(
        "ma,lab,bn->lmn",
        inverse_spatial_metric,
        spatial_metric_derivatives,
        inverse_spatial_metric,
    )
    chi_derivatives = torch.zeros(4, dtype=fields.dtype, device=fields.device)
    for derivative_index in range(4):
        derivative_aether = field_derivatives[derivative_index, AETHER_SLICE]
        chi_derivatives[derivative_index] = (
            torch.einsum(
                "mn,m,n->",
                inverse_spatial_metric_derivatives[derivative_index],
                aether_spatial,
                aether_spatial,
            )
            + 2.0 * (aether_spatial_up @ derivative_aether)
        ) / (2.0 * chi)

    # Exact AeST electric and magnetic fields in the ADM variables of the
    # published Hamiltonian formulation.
    spatial_aether_derivatives = field_derivatives[1:, AETHER_SLICE]
    field_strength_spatial = spatial_aether_derivatives - spatial_aether_derivatives.T
    levi_civita_symbol = torch.zeros((3, 3, 3), dtype=fields.dtype, device=fields.device)
    levi_civita_symbol[0, 1, 2] = levi_civita_symbol[1, 2, 0] = 1.0
    levi_civita_symbol[2, 0, 1] = 1.0
    levi_civita_symbol[0, 2, 1] = levi_civita_symbol[2, 1, 0] = -1.0
    levi_civita_symbol[1, 0, 2] = -1.0
    sqrt_spatial_determinant = torch.sqrt(torch.linalg.det(spatial_metric))
    magnetic_up = 0.5 * torch.einsum(
        "kij,ij->k", levi_civita_symbol / sqrt_spatial_determinant, field_strength_spatial
    )
    magnetic_squared = magnetic_up @ spatial_metric @ magnetic_up

    spatial_scalar_derivative = torch.zeros(3, dtype=fields.dtype, device=fields.device)
    for spatial_index in range(3):
        coordinate_index = spatial_index + 1
        derivative_lapse = field_derivatives[coordinate_index, 0]
        derivative_shift = field_derivatives[coordinate_index, 1:4]
        derivative_aether = field_derivatives[coordinate_index, AETHER_SLICE]
        spatial_scalar_derivative[spatial_index] = (
            derivative_lapse * chi
            + lapse * chi_derivatives[coordinate_index]
            - derivative_shift @ aether_spatial
            - shift @ derivative_aether
        )
    electric_covariant = (
        field_derivatives[0, AETHER_SLICE] + spatial_scalar_derivative
    ) / lapse
    electric_covariant = electric_covariant + torch.einsum(
        "ijk,j,k->i",
        levi_civita_symbol * sqrt_spatial_determinant,
        shift,
        magnetic_up,
    ) / lapse
    electric_squared = electric_covariant @ inverse_spatial_metric @ electric_covariant
    maxwell = float(background.k_b) * (electric_squared - magnetic_squared)

    scalar_clock = float(background.scalar_clock_ratio)
    sigma = scalar_clock / lapse
    q_value = chi * sigma
    y_value = (aether_spatial @ aether_spatial_up) * sigma**2
    aether_scalar = 2.0 * (2.0 - float(background.k_b)) * sigma * (
        aether_spatial_up @ electric_covariant
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

    # Covariant DHOST terms evaluated from the exact ADM four-metric.
    scalar_covector = torch.tensor(
        [scalar_clock, 0.0, 0.0, 0.0], dtype=fields.dtype, device=fields.device
    )
    scalar_contravector = inverse_metric @ scalar_covector
    x_value = scalar_covector @ scalar_contravector
    scalar_hessian = -scalar_clock * connection[0]
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

    determinant_factor = lapse * sqrt_spatial_determinant
    return (
        _fierz_pauli_density(_metric_derivatives(time_derivatives, spatial_derivatives), f_0=background.f_0)
        + determinant_factor * (maxwell + aether_scalar + aest_scalar + dhost)
    )


def mode_lagrangian_hessian(
    background: TiltedPrincipalBackground,
    *,
    wave_number_ratio: float,
) -> dict[str, np.ndarray | float]:
    """Return ``K,A,B`` for the real sine/cosine Fourier mode.

    The averaged quadratic mode is

    ``L=1/2 v.T K v + v.T A q + 1/2 q.T B q``.
    """

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
            _local_covariant_density(
                amplitude_cos,
                velocity_cos,
                derivative_cos,
                background=background,
            )
            + _local_covariant_density(
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


def reduced_dirac_block(
    background: TiltedPrincipalBackground,
    *,
    wave_number_ratio: float,
    relative_tolerance: float = 1.0e-9,
) -> dict[str, object]:
    """Return the two-phase DHOST Dirac block before spatial gauge fixing.

    The full kinetic nullspace has eight real Fourier directions: six are the
    sine/cosine shift primaries associated with spatial diffeomorphisms and
    two form the Class-Ia DHOST primary.  The latter are isolated from the
    non-shift kinetic block; deleting metric components before this step would
    spuriously lift the Class-Ia null direction.
    """

    mode = mode_lagrangian_hessian(
        background,
        wave_number_ratio=wave_number_ratio,
    )
    kinetic = np.asarray(mode["K"], dtype=float)
    mixing = np.asarray(mode["A"], dtype=float)
    potential_lagrangian = np.asarray(mode["B"], dtype=float)
    shift_indices = np.asarray(
        [*SHIFT_BASE_INDICES, *(index + FIELD_COUNT for index in SHIFT_BASE_INDICES)],
        dtype=int,
    )
    all_indices = np.arange(2 * FIELD_COUNT)
    dynamic_indices = np.asarray(
        [index for index in all_indices if index not in set(shift_indices)],
        dtype=int,
    )
    kinetic_eigenvalues, kinetic_eigenvectors = np.linalg.eigh(kinetic)
    kinetic_scale = max(1.0, float(np.max(np.abs(kinetic_eigenvalues))))
    null_threshold = float(relative_tolerance) * kinetic_scale
    null_mask = np.abs(kinetic_eigenvalues) <= null_threshold
    full_null_basis = kinetic_eigenvectors[:, null_mask]

    nonshift_kinetic = kinetic[np.ix_(dynamic_indices, dynamic_indices)]
    nonshift_eigenvalues, nonshift_eigenvectors = np.linalg.eigh(nonshift_kinetic)
    nonshift_scale = max(1.0, float(np.max(np.abs(nonshift_eigenvalues))))
    nonshift_null_mask = np.abs(nonshift_eigenvalues) <= (
        float(relative_tolerance) * nonshift_scale
    )
    reduced_dhost_basis = nonshift_eigenvectors[:, nonshift_null_mask]
    dhost_null_basis = np.zeros((2 * FIELD_COUNT, reduced_dhost_basis.shape[1]))
    dhost_null_basis[dynamic_indices] = reduced_dhost_basis
    shift_null_basis = np.zeros((2 * FIELD_COUNT, len(shift_indices)))
    shift_null_basis[shift_indices, np.arange(len(shift_indices))] = 1.0

    expected_null_structure = bool(
        full_null_basis.shape[1] == 8 and dhost_null_basis.shape[1] == 2
    )
    if not expected_null_structure:
        return {
            "background": background.__dict__,
            "wave_number_ratio": float(wave_number_ratio),
            "kinetic_eigenvalues": kinetic_eigenvalues.tolist(),
            "kinetic_nullity": int(full_null_basis.shape[1]),
            "nonshift_kinetic_eigenvalues": nonshift_eigenvalues.tolist(),
            "dhost_nullity": int(dhost_null_basis.shape[1]),
            "expected_eight_total_two_dhost_nullity": False,
            "dirac_block": None,
            "dirac_eigenvalues": None,
            "dirac_invertible": False,
        }

    inverse_eigenvalues = np.zeros_like(kinetic_eigenvalues)
    inverse_eigenvalues[~null_mask] = 1.0 / kinetic_eigenvalues[~null_mask]
    kinetic_pseudoinverse = (
        kinetic_eigenvectors * inverse_eigenvalues[np.newaxis, :]
    ) @ kinetic_eigenvectors.T
    antisymmetric_mixing = mixing.T - mixing
    hamiltonian_potential = -potential_lagrangian
    secondary_operator = (
        hamiltonian_potential
        - antisymmetric_mixing @ kinetic_pseudoinverse @ antisymmetric_mixing
    )
    secondary_operator = 0.5 * (secondary_operator + secondary_operator.T)
    clock_selector = np.zeros((2 * FIELD_COUNT, 2))
    clock_selector[0, 0] = 1.0
    clock_selector[FIELD_COUNT, 1] = 1.0
    scalar_clock = float(background.scalar_clock_ratio)
    clock_map = -scalar_clock * clock_selector.T @ dhost_null_basis
    clock_map_singular_values = np.linalg.svd(clock_map, compute_uv=False)
    if clock_map_singular_values[-1] <= float(relative_tolerance):
        return {
            "background": background.__dict__,
            "wave_number_ratio": float(wave_number_ratio),
            "kinetic_eigenvalues": kinetic_eigenvalues.tolist(),
            "kinetic_nullity": int(full_null_basis.shape[1]),
            "nonshift_kinetic_eigenvalues": nonshift_eigenvalues.tolist(),
            "dhost_nullity": int(dhost_null_basis.shape[1]),
            "expected_eight_total_two_dhost_nullity": True,
            "clock_normalization_regular": False,
            "dirac_block": None,
            "dirac_eigenvalues": None,
            "dirac_invertible": False,
        }
    clock_normalized_basis = dhost_null_basis @ np.linalg.inv(clock_map)
    dirac_block = clock_normalized_basis.T @ secondary_operator @ clock_normalized_basis
    dirac_block = 0.5 * (dirac_block + dirac_block.T)
    dirac_eigenvalues = np.linalg.eigvalsh(dirac_block)
    dirac_scale = max(1.0, float(np.max(np.abs(dirac_eigenvalues))))
    dirac_threshold = float(relative_tolerance) * dirac_scale
    dhost_primary_bracket = (
        clock_normalized_basis.T @ antisymmetric_mixing @ clock_normalized_basis
    )
    shift_dhost_primary_cross = (
        shift_null_basis.T @ antisymmetric_mixing @ clock_normalized_basis
    )
    shift_dhost_secondary_cross = (
        shift_null_basis.T @ secondary_operator @ clock_normalized_basis
    )
    conformal_indices = np.asarray([4, 7, 9], dtype=int)
    conformal_components = 0.5 * clock_normalized_basis[conformal_indices, 0]
    x_value = -(scalar_clock**2)
    x_zero = -(float(background.background_clock_ratio) ** 2)
    difference = x_value - x_zero
    activation = difference**2 / (
        (1.0 + difference**2) ** 1.5 * np.sqrt(1.0 + x_value**2)
    )
    a_3_bar = float(background.orientation_strength) * activation
    expected_conformal_ratio = -(scalar_clock**3) * a_3_bar / 4.0
    conformal_scale = max(1.0, abs(expected_conformal_ratio))
    conformal_residual = float(
        np.max(np.abs(conformal_components - expected_conformal_ratio)) / conformal_scale
    )
    return {
        "background": background.__dict__,
        "wave_number_ratio": float(wave_number_ratio),
        "kinetic_eigenvalues": kinetic_eigenvalues.tolist(),
        "kinetic_nullity": int(full_null_basis.shape[1]),
        "nonshift_kinetic_eigenvalues": nonshift_eigenvalues.tolist(),
        "dhost_nullity": int(dhost_null_basis.shape[1]),
        "expected_eight_total_two_dhost_nullity": True,
        "clock_normalization_regular": True,
        "clock_map_condition_number": float(
            clock_map_singular_values[0] / clock_map_singular_values[-1]
        ),
        "clock_normalized_null_residual": float(
            np.linalg.norm(kinetic @ clock_normalized_basis) / kinetic_scale
        ),
        "null_conformal_components_delta_zeta_over_delta_r": conformal_components.tolist(),
        "expected_null_conformal_ratio": float(expected_conformal_ratio),
        "null_conformal_ratio_residual": conformal_residual,
        "dhost_primary_self_bracket_norm": float(np.linalg.norm(dhost_primary_bracket)),
        "shift_dhost_primary_cross_norm": float(np.linalg.norm(shift_dhost_primary_cross)),
        "shift_dhost_secondary_cross_norm": float(np.linalg.norm(shift_dhost_secondary_cross)),
        "dirac_block": dirac_block.tolist(),
        "dirac_eigenvalues": dirac_eigenvalues.tolist(),
        "dirac_normalized_minimum_absolute_eigenvalue": float(
            np.min(np.abs(dirac_eigenvalues)) / dirac_scale
        ),
        "dirac_invertible": bool(np.min(np.abs(dirac_eigenvalues)) > dirac_threshold),
    }


def audit_v12a_tilted_principal(
    *,
    k_b: float,
    k_2: float,
    background_clock_ratio: float,
    positive_orientation_strength: float,
    negative_orientation_strength: float,
    random_trials: int,
    logarithmic_clock_limit: float,
    logarithmic_tilt_limit: float,
    wave_number_sentinels: tuple[float, float, float],
    wave_invariance_trials: int,
    aligned_limit_tilt: float,
    random_seed: int,
) -> dict[str, object]:
    """Audit the constant-background tilted DHOST Dirac block."""

    if random_trials < 1 or wave_invariance_trials < 1:
        raise ValueError("random and wave-invariance trial counts must be positive")
    if logarithmic_clock_limit <= 0.0 or logarithmic_tilt_limit <= 0.0:
        raise ValueError("logarithmic scan limits must be positive")
    if positive_orientation_strength <= 0.0 or negative_orientation_strength >= 0.0:
        raise ValueError("positive and negative orientation sentinels are required")
    if aligned_limit_tilt <= 0.0:
        raise ValueError("the differentiable aligned-limit sentinel must be positive")
    wave_numbers = tuple(float(value) for value in wave_number_sentinels)
    if len(wave_numbers) != 3 or any(value <= 0.0 for value in wave_numbers):
        raise ValueError("three positive wave-number sentinels are required")

    rng = np.random.default_rng(int(random_seed))
    strengths = {
        "positive": float(positive_orientation_strength),
        "negative": float(negative_orientation_strength),
    }
    closest_rows: dict[str, dict[str, object] | None] = {name: None for name in strengths}
    closest_magnitudes = {name: np.inf for name in strengths}
    maximum_null_residual = 0.0
    maximum_conformal_residual = 0.0
    maximum_primary_bracket = 0.0
    maximum_primary_cross = 0.0
    maximum_secondary_cross = 0.0
    maximum_wave_residual = 0.0
    null_structure_failures = 0
    sign_or_rank_failures = 0

    for trial in range(int(random_trials)):
        clock_sign = -1.0 if rng.random() < 0.5 else 1.0
        scalar_clock = clock_sign * 10.0 ** rng.uniform(
            -float(logarithmic_clock_limit), float(logarithmic_clock_limit)
        )
        tilt_magnitude = 10.0 ** rng.uniform(
            -float(logarithmic_tilt_limit), float(logarithmic_tilt_limit)
        )
        angle = rng.uniform(-np.pi, np.pi)
        parallel = tilt_magnitude * np.cos(angle)
        perpendicular = tilt_magnitude * np.sin(angle)
        for branch, strength in strengths.items():
            background = TiltedPrincipalBackground(
                scalar_clock_ratio=float(scalar_clock),
                aether_parallel=float(parallel),
                aether_perpendicular=float(perpendicular),
                background_clock_ratio=float(background_clock_ratio),
                orientation_strength=float(strength),
                k_b=float(k_b),
                k_2=float(k_2),
            )
            row = reduced_dirac_block(background, wave_number_ratio=wave_numbers[1])
            if not row["expected_eight_total_two_dhost_nullity"]:
                null_structure_failures += 1
                continue
            eigenvalues = np.asarray(row["dirac_eigenvalues"], dtype=float)
            if (not row["dirac_invertible"]) or np.any(eigenvalues >= 0.0):
                sign_or_rank_failures += 1
            magnitude = float(np.min(np.abs(eigenvalues)))
            if magnitude < closest_magnitudes[branch]:
                closest_magnitudes[branch] = magnitude
                closest_rows[branch] = {"trial": trial, **row}
            maximum_null_residual = max(
                maximum_null_residual, float(row["clock_normalized_null_residual"])
            )
            maximum_conformal_residual = max(
                maximum_conformal_residual, float(row["null_conformal_ratio_residual"])
            )
            maximum_primary_bracket = max(
                maximum_primary_bracket, float(row["dhost_primary_self_bracket_norm"])
            )
            maximum_primary_cross = max(
                maximum_primary_cross, float(row["shift_dhost_primary_cross_norm"])
            )
            maximum_secondary_cross = max(
                maximum_secondary_cross, float(row["shift_dhost_secondary_cross_norm"])
            )

            if trial < int(wave_invariance_trials):
                sentinel_eigenvalues = []
                for wave_number in (wave_numbers[0], wave_numbers[2]):
                    sentinel = reduced_dirac_block(
                        background,
                        wave_number_ratio=wave_number,
                    )
                    sentinel_eigenvalues.append(
                        np.asarray(sentinel["dirac_eigenvalues"], dtype=float)
                    )
                comparison_scale = max(
                    1.0,
                    float(np.max(np.abs(eigenvalues))),
                    *(float(np.max(np.abs(values))) for values in sentinel_eigenvalues),
                )
                for values in sentinel_eigenvalues:
                    maximum_wave_residual = max(
                        maximum_wave_residual,
                        float(np.max(np.abs(values - eigenvalues)) / comparison_scale),
                    )

    aligned_rows = {}
    aligned_target = -4.0 * float(k_2)
    maximum_aligned_residual = 0.0
    for branch, strength in strengths.items():
        background = TiltedPrincipalBackground(
            scalar_clock_ratio=float(background_clock_ratio),
            aether_parallel=0.0,
            aether_perpendicular=float(aligned_limit_tilt),
            background_clock_ratio=float(background_clock_ratio),
            orientation_strength=float(strength),
            k_b=float(k_b),
            k_2=float(k_2),
        )
        rows = [
            reduced_dirac_block(background, wave_number_ratio=wave_number)
            for wave_number in wave_numbers
        ]
        continuum_equivalent = [
            (2.0 * np.asarray(row["dirac_eigenvalues"], dtype=float)).tolist()
            for row in rows
        ]
        aligned_residual = max(
            float(
                np.max(
                    np.abs(np.asarray(values, dtype=float) - aligned_target)
                    / max(1.0, abs(aligned_target))
                )
            )
            for values in continuum_equivalent
        )
        maximum_aligned_residual = max(maximum_aligned_residual, aligned_residual)
        aligned_rows[branch] = {
            "orientation_strength": strength,
            "tilt_sentinel": float(aligned_limit_tilt),
            "wave_number_sentinels": list(wave_numbers),
            "continuum_equivalent_dirac_eigenvalues": continuum_equivalent,
            "target_minus_4K2": aligned_target,
            "maximum_relative_residual": aligned_residual,
        }

    gates = {
        "expected_eight_total_two_dhost_nullity": null_structure_failures == 0,
        "clock_normalized_null_direction": maximum_null_residual < 1.0e-9,
        "class_ia_conformal_ratio": maximum_conformal_residual < 1.0e-9,
        "dhost_primary_self_bracket_zero": maximum_primary_bracket < 1.0e-9,
        "spatial_diffeomorphism_primary_decouples": maximum_primary_cross < 1.0e-9,
        "spatial_diffeomorphism_secondary_decouples": maximum_secondary_cross < 1.0e-9,
        "constant_background_dirac_block_negative_nonzero": sign_or_rank_failures == 0,
        "wave_number_independence_after_dynamical_aether_schur": maximum_wave_residual
        < 1.0e-9,
        "aligned_limit_equals_minus_4K2": maximum_aligned_residual < 1.0e-6,
    }
    return {
        "candidate": "Sigma v12A same-AeST-clock luminal DHOST geometry",
        "calculation": {
            "gauge": "regular local unitary gauge for timelike scalar gradient",
            "retained_fields_per_phase": (
                "delta N, three shifts, six spatial-metric components, three aether components"
            ),
            "fourier_representation": "real sine/cosine pair",
            "kinetic_null_structure": (
                "six shift primaries plus two clock-normalized Class-Ia primaries"
            ),
            "dirac_formula": "Z^T[-B-(A^T-A)K^+(A^T-A)]Z",
            "aether_maxwell_result": (
                "the apparent lapse-gradient term cancels against the dynamical aether Schur term"
            ),
        },
        "scan": {
            "random_trials": int(random_trials),
            "signed_log10_clock_limit": float(logarithmic_clock_limit),
            "log10_aether_tilt_limit": float(logarithmic_tilt_limit),
            "wave_number_sentinels": list(wave_numbers),
            "wave_invariance_trials_per_sign": int(wave_invariance_trials),
            "closest_positive_row": closest_rows["positive"],
            "closest_negative_row": closest_rows["negative"],
            "maximum_clock_normalized_null_residual": maximum_null_residual,
            "maximum_class_ia_conformal_residual": maximum_conformal_residual,
            "maximum_dhost_primary_self_bracket_norm": maximum_primary_bracket,
            "maximum_shift_dhost_primary_cross_norm": maximum_primary_cross,
            "maximum_shift_dhost_secondary_cross_norm": maximum_secondary_cross,
            "maximum_wave_number_dependence_residual": maximum_wave_residual,
            "null_structure_failures": null_structure_failures,
            "sign_or_rank_failures": sign_or_rank_failures,
        },
        "aligned_limit": aligned_rows,
        "gates": {name: bool(value) for name, value in gates.items()},
        "previous_aligned_maxwell_stabilization_valid": False,
        "aligned_sign_conclusion_changed": False,
        "positive_sign_survives_constant_background_dirac_gate": bool(all(gates.values())),
        "negative_sign_survives_constant_background_dirac_gate": bool(all(gates.values())),
        "constant_background_delta_eff_proven_invertible": bool(all(gates.values())),
        "nonconstant_background_delta_eff_proven_invertible": False,
        "complete_physical_characteristic_matrix_scored": False,
        "physical_degree_count_proven_unchanged": False,
        "theory_viable": False,
        "observational_data_accessed": False,
        "raw_holdout_opened": False,
    }
