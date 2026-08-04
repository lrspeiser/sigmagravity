"""Theory-only audit tools for the Sigma v9A alignment interaction.

The candidate keeps the one-metric AeST base and uses only first derivatives.
With the aether projector ``q_mn=g_mn+A_m A_n`` define

``S_m=q_m^n grad_n(phi)``, ``J_m=A^n nabla_n A_m``,
``Y=S.S``, ``Z=J.J`` and ``U=S.J``.

The Gram determinant ``N=Y Z-U^2`` is non-negative in the aether rest space
and measures squared gradient misalignment.  Two variants are audited:

* ``one_sided``: ``-4 eta a^2 N/(a^2+Y)^2``;
* ``saturated``: ``-4 eta a^4 N/[(a^2+Y)^2(a^2+Z)]``.

The first is the direct bounded-in-Y proposal.  The second is its minimal
bounded-in-both-gradients repair.  This module deliberately stops at a local
static principal-symbol and mechanism gate.  It is not a full covariant Dirac,
PPN, or observational calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from scipy.optimize import brentq

Array = np.ndarray
Variant = Literal["one_sided", "saturated"]
TORCH_DTYPE = torch.float64


@dataclass(frozen=True)
class AlignmentInvariants:
    y: float
    z: float
    u: float
    gram: float
    sine_squared: float


@dataclass(frozen=True)
class AlignmentFluxes:
    scalar_gradient_flux: Array
    aether_acceleration_flux: Array


def _finite_vector(values: Array, *, name: str) -> Array:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (3,) or np.any(~np.isfinite(vector)):
        raise ValueError(f"{name} must contain three finite values")
    return vector


def _positive(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _nonnegative(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _variant(value: str) -> Variant:
    if value not in {"one_sided", "saturated"}:
        raise ValueError("variant must be 'one_sided' or 'saturated'")
    return value  # type: ignore[return-value]


def alignment_invariants(
    scalar_gradient: Array,
    aether_acceleration: Array,
) -> AlignmentInvariants:
    """Return the Euclidean aether-rest-space invariants.

    Roundoff can make the Gram determinant infinitesimally negative.  Values
    within 64 machine epsilons of ``Y Z`` are clipped to zero; a larger
    violation is treated as an implementation error.
    """

    scalar = _finite_vector(scalar_gradient, name="scalar_gradient")
    aether = _finite_vector(aether_acceleration, name="aether_acceleration")
    y = float(scalar @ scalar)
    z = float(aether @ aether)
    u = float(scalar @ aether)
    raw_gram = y * z - u**2
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, y * z)
    if raw_gram < -tolerance:
        raise ValueError("Gram determinant violates Cauchy-Schwarz")
    gram = max(0.0, raw_gram)
    sine_squared = 0.0 if y == 0.0 or z == 0.0 else gram / (y * z)
    return AlignmentInvariants(y=y, z=z, u=u, gram=gram, sine_squared=sine_squared)


def alignment_interaction_density(
    scalar_gradient: Array,
    aether_acceleration: Array,
    *,
    a_sigma: float,
    eta: float,
    variant: Variant = "one_sided",
) -> float:
    """Return the local v9A interaction density in inverse-length squared units."""

    scale = _positive(a_sigma, name="a_sigma")
    coupling = _nonnegative(eta, name="eta")
    selected = _variant(variant)
    invariants = alignment_invariants(scalar_gradient, aether_acceleration)
    dy = scale**2 + invariants.y
    density = -4.0 * coupling * scale**2 * invariants.gram / dy**2
    if selected == "saturated":
        density *= scale**2 / (scale**2 + invariants.z)
    return float(density)


def alignment_fluxes(
    scalar_gradient: Array,
    aether_acceleration: Array,
    *,
    a_sigma: float,
    eta: float,
    variant: Variant = "one_sided",
) -> AlignmentFluxes:
    """Return exact derivatives of the interaction with respect to S and J."""

    scalar = _finite_vector(scalar_gradient, name="scalar_gradient")
    aether = _finite_vector(aether_acceleration, name="aether_acceleration")
    scale = _positive(a_sigma, name="a_sigma")
    coupling = _nonnegative(eta, name="eta")
    selected = _variant(variant)
    inv = alignment_invariants(scalar, aether)
    dy = scale**2 + inv.y
    dz = scale**2 + inv.z
    scalar_numerator = inv.z * scalar - inv.u * aether
    aether_numerator = inv.y * aether - inv.u * scalar
    if selected == "one_sided":
        scalar_flux = -8.0 * coupling * scale**2 * (
            scalar_numerator / dy**2 - 2.0 * inv.gram * scalar / dy**3
        )
        aether_flux = -8.0 * coupling * scale**2 * aether_numerator / dy**2
    else:
        scalar_flux = -8.0 * coupling * scale**4 * (
            scalar_numerator / (dy**2 * dz)
            - 2.0 * inv.gram * scalar / (dy**3 * dz)
        )
        aether_flux = -8.0 * coupling * scale**4 * (
            aether_numerator / (dy**2 * dz)
            - inv.gram * aether / (dy**2 * dz**2)
        )
    return AlignmentFluxes(
        scalar_gradient_flux=np.asarray(scalar_flux, dtype=float),
        aether_acceleration_flux=np.asarray(aether_flux, dtype=float),
    )


def aether_rest_vector_kinetic_eigenvalues(
    y_over_a_squared: float,
    *,
    k_b: float,
    eta: float,
) -> dict[str, float]:
    """Return the exact small-J kinetic coefficients parallel/perpendicular to S.

    Both variants have the same quadratic limit at ``J=0``.  The Lagrangian
    coefficients (one half of the Hessian eigenvalues) are
    ``K_B`` and ``K_B-4 eta y/(1+y)^2``.
    """

    y = _nonnegative(y_over_a_squared, name="y_over_a_squared")
    vector = _positive(k_b, name="k_b")
    coupling = _nonnegative(eta, name="eta")
    activation = 4.0 * y / (1.0 + y) ** 2
    return {
        "activation": float(activation),
        "parallel": vector,
        "perpendicular": float(vector - coupling * activation),
    }


def maximum_perpendicular_amplification(*, k_b: float, eta: float) -> float:
    """Return the linear small-J response bound at the transition ``Y=a^2``."""

    vector = _positive(k_b, name="k_b")
    coupling = _nonnegative(eta, name="eta")
    if coupling >= vector:
        return np.inf
    return float(vector / (vector - coupling))


def angle_required_for_amplification(
    target_amplification: float,
    *,
    k_b: float,
    eta: float,
) -> dict[str, float | bool | None]:
    """Return the best-case misalignment angle needed for a target response.

    This assumes ``Y=a_sigma^2`` and the small-J limit, so it is an optimistic
    lower bound on the angle.  A spherical/aligned field has zero enhancement.
    """

    target = _positive(target_amplification, name="target_amplification")
    vector = _positive(k_b, name="k_b")
    coupling = _positive(eta, name="eta")
    if target < 1.0:
        raise ValueError("target_amplification must be at least one")
    sine_squared = vector * (1.0 - 1.0 / target) / coupling
    reachable = sine_squared <= 1.0 + 1.0e-14
    angle = None
    if reachable:
        angle = float(np.degrees(np.arcsin(np.sqrt(np.clip(sine_squared, 0.0, 1.0)))))
    return {
        "target_amplification": target,
        "minimum_sine_squared": float(sine_squared),
        "reachable": bool(reachable),
        "minimum_angle_degrees": angle,
    }


def _aqual_free_function_torch(y: torch.Tensor) -> torch.Tensor:
    root = torch.sqrt(y)
    return y - 2.0 * root + 2.0 * torch.log1p(root)


def _static_lagrangian_torch(
    state: torch.Tensor,
    *,
    k_b: float,
    eta: float,
    variant: Variant,
) -> torch.Tensor:
    aether = state[:3]
    scalar = state[3:]
    y = scalar @ scalar
    z = aether @ aether
    u = scalar @ aether
    gram = y * z - u**2
    base = (
        k_b * z
        + 2.0 * (2.0 - k_b) * u
        - (2.0 - k_b) * (y + _aqual_free_function_torch(y))
    )
    interaction = -4.0 * eta * gram / (1.0 + y) ** 2
    if variant == "saturated":
        interaction = interaction / (1.0 + z)
    return base + interaction


def static_lagrangian(
    state: Array,
    *,
    k_b: float,
    eta: float,
    variant: Variant,
) -> float:
    """Evaluate the dimensionless aether-rest quasistatic density.

    The state order is ``J_x,J_y,J_z,S_x,S_y,S_z`` and all gradients are in
    units of ``a_sigma``.  ``|S|`` must be nonzero because the selected simple
    AQUAL free function is nonanalytic exactly at the origin.
    """

    values = np.asarray(state, dtype=float)
    if values.shape != (6,) or np.any(~np.isfinite(values)):
        raise ValueError("state must contain six finite values")
    if np.linalg.norm(values[3:]) == 0.0:
        raise ValueError("the static Hessian evaluator requires nonzero |S|")
    vector = _positive(k_b, name="k_b")
    if vector >= 2.0:
        raise ValueError("k_b must lie in (0, 2)")
    coupling = _nonnegative(eta, name="eta")
    selected = _variant(variant)
    tensor = torch.tensor(values, dtype=TORCH_DTYPE)
    return float(
        _static_lagrangian_torch(
            tensor,
            k_b=vector,
            eta=coupling,
            variant=selected,
        ).detach()
    )


def static_principal_hessian(
    state: Array,
    *,
    k_b: float,
    eta: float,
    variant: Variant,
) -> Array:
    """Return the exact six-by-six local static principal matrix."""

    values = np.asarray(state, dtype=float)
    static_lagrangian(values, k_b=k_b, eta=eta, variant=variant)
    tensor = torch.tensor(values, dtype=TORCH_DTYPE, requires_grad=True)
    hessian = torch.autograd.functional.hessian(
        lambda item: _static_lagrangian_torch(
            item,
            k_b=float(k_b),
            eta=float(eta),
            variant=_variant(variant),
        ),
        tensor,
        vectorize=True,
    )
    return hessian.detach().cpu().numpy()


def centered_finite_difference_static_hessian(
    state: Array,
    *,
    k_b: float,
    eta: float,
    variant: Variant,
    step: float = 2.0e-4,
) -> Array:
    """Return an independent centered finite-difference Hessian."""

    center = np.asarray(state, dtype=float)
    static_lagrangian(center, k_b=k_b, eta=eta, variant=variant)
    delta = _positive(step, name="step")
    result = np.empty((6, 6), dtype=float)

    def evaluate(values: Array) -> float:
        return static_lagrangian(values, k_b=k_b, eta=eta, variant=variant)

    center_value = evaluate(center)
    for row in range(6):
        dr = np.zeros(6)
        dr[row] = delta
        result[row, row] = (
            evaluate(center + dr) - 2.0 * center_value + evaluate(center - dr)
        ) / delta**2
        for column in range(row):
            dc = np.zeros(6)
            dc[column] = delta
            result[row, column] = result[column, row] = (
                evaluate(center + dr + dc)
                - evaluate(center + dr - dc)
                - evaluate(center - dr + dc)
                + evaluate(center - dr - dc)
            ) / (4.0 * delta**2)
    return result


def _state_from_invariants(y: float, z: float, cosine: float) -> Array:
    transverse = np.sqrt(max(0.0, 1.0 - cosine**2))
    return np.array(
        [np.sqrt(z) * cosine, np.sqrt(z) * transverse, 0.0, np.sqrt(y), 0.0, 0.0]
    )


def _inertia(matrix: Array, *, relative_tolerance: float = 1.0e-9) -> tuple[int, int, int]:
    eigenvalues = np.linalg.eigvalsh(matrix)
    tolerance = relative_tolerance * max(1.0, float(np.max(np.abs(eigenvalues))))
    return (
        int(np.sum(eigenvalues < -tolerance)),
        int(np.sum(np.abs(eigenvalues) <= tolerance)),
        int(np.sum(eigenvalues > tolerance)),
    )


def find_one_sided_rank_surface(
    *,
    eta: float,
    k_b: float = 1.0,
    lower_z: float = 1.0e-8,
    upper_z: float = 1.0e4,
) -> dict[str, object]:
    """Find the first determinant zero at ``Y=1`` and ``S.J=0``."""

    coupling = _positive(eta, name="eta")
    lower = _positive(lower_z, name="lower_z")
    upper = _positive(upper_z, name="upper_z")
    if lower >= upper:
        raise ValueError("lower_z must be smaller than upper_z")

    def determinant(z: float) -> float:
        matrix = static_principal_hessian(
            _state_from_invariants(1.0, z, 0.0),
            k_b=k_b,
            eta=coupling,
            variant="one_sided",
        )
        return float(np.linalg.det(matrix))

    grid = np.geomspace(lower, upper, 400)
    bracket: tuple[float, float] | None = None
    previous_z = float(grid[0])
    previous_value = determinant(previous_z)
    for current_z in grid[1:]:
        current = float(current_z)
        value = determinant(current)
        if previous_value * value < 0.0:
            bracket = (previous_z, current)
            break
        previous_z = current
        previous_value = value
    if bracket is None:
        raise ValueError("no determinant sign change found in the requested interval")
    root = float(brentq(determinant, *bracket, xtol=1.0e-11))
    below = static_principal_hessian(
        _state_from_invariants(1.0, root * (1.0 - 1.0e-6), 0.0),
        k_b=k_b,
        eta=coupling,
        variant="one_sided",
    )
    at_root = static_principal_hessian(
        _state_from_invariants(1.0, root, 0.0),
        k_b=k_b,
        eta=coupling,
        variant="one_sided",
    )
    above = static_principal_hessian(
        _state_from_invariants(1.0, root * (1.0 + 1.0e-6), 0.0),
        k_b=k_b,
        eta=coupling,
        variant="one_sided",
    )
    eigenvalues, eigenvectors = np.linalg.eigh(at_root)
    null_index = int(np.argmin(np.abs(eigenvalues)))
    null = eigenvectors[:, null_index]
    return {
        "eta": coupling,
        "Y_over_a_squared": 1.0,
        "Z_over_a_squared": root,
        "J_over_a": float(np.sqrt(root)),
        "minimum_absolute_eigenvalue": float(np.min(np.abs(eigenvalues))),
        "inertia_below": _inertia(below),
        "inertia_above": _inertia(above),
        "null_mode_aether_power": float(null[:3] @ null[:3]),
        "null_mode_scalar_power": float(null[3:] @ null[3:]),
    }


def scan_saturated_static_principal_symbol(
    *,
    eta: float,
    k_b: float,
    y_values: Array,
    z_values: Array,
    cosine_values: Array,
    random_samples: int,
    random_seed: int,
) -> dict[str, object]:
    """Scan whether the saturated repair preserves the AeST static inertia."""

    coupling = _nonnegative(eta, name="eta")
    vector = _positive(k_b, name="k_b")
    ys = np.asarray(y_values, dtype=float)
    zs = np.asarray(z_values, dtype=float)
    cosines = np.asarray(cosine_values, dtype=float)
    if (
        ys.ndim != 1
        or zs.ndim != 1
        or cosines.ndim != 1
        or not ys.size
        or not zs.size
        or not cosines.size
        or np.any(~np.isfinite(ys))
        or np.any(ys <= 0.0)
        or np.any(~np.isfinite(zs))
        or np.any(zs < 0.0)
        or np.any(~np.isfinite(cosines))
        or np.any(np.abs(cosines) > 1.0)
    ):
        raise ValueError("invalid deterministic scan axes")
    samples = int(random_samples)
    if samples < 0:
        raise ValueError("random_samples must be non-negative")
    records: list[tuple[float, float, float]] = [
        (float(y), float(z), float(cosine))
        for y in ys
        for z in zs
        for cosine in cosines
    ]
    rng = np.random.default_rng(int(random_seed))
    if samples:
        log_y = rng.uniform(np.log10(np.min(ys)), np.log10(np.max(ys)), samples)
        positive_z = zs[zs > 0.0]
        minimum_z = float(np.min(positive_z)) if positive_z.size else 1.0e-8
        maximum_z = float(np.max(zs)) if np.max(zs) > 0.0 else 1.0
        log_z = rng.uniform(np.log10(minimum_z), np.log10(maximum_z), samples)
        random_cosine = rng.uniform(-1.0, 1.0, samples)
        records.extend(
            zip(10.0**log_y, 10.0**log_z, random_cosine, strict=True)
        )
    minimum_singular = np.inf
    location: dict[str, float] = {}
    inertia_changes = 0
    worst_determinant_ratio = np.inf
    for y, z, cosine in records:
        state = _state_from_invariants(y, z, cosine)
        base = static_principal_hessian(
            state,
            k_b=vector,
            eta=0.0,
            variant="saturated",
        )
        combined = static_principal_hessian(
            state,
            k_b=vector,
            eta=coupling,
            variant="saturated",
        )
        singular = float(np.linalg.svd(combined, compute_uv=False)[-1])
        if singular < minimum_singular:
            minimum_singular = singular
            location = {"Y_over_a_squared": y, "Z_over_a_squared": z, "cosine": cosine}
        if _inertia(base) != _inertia(combined):
            inertia_changes += 1
        sign_base, log_base = np.linalg.slogdet(base)
        sign_combined, log_combined = np.linalg.slogdet(combined)
        ratio = float(sign_combined / sign_base * np.exp(log_combined - log_base))
        worst_determinant_ratio = min(worst_determinant_ratio, ratio)
    return {
        "points_scanned": len(records),
        "minimum_singular_value": float(minimum_singular),
        "minimum_location": location,
        "inertia_changes": inertia_changes,
        "minimum_combined_to_base_determinant_ratio": float(worst_determinant_ratio),
        "necessary_static_gate_passed": bool(
            inertia_changes == 0 and minimum_singular > 1.0e-6
        ),
    }


def _softened_field(
    points: Array,
    positions: Array,
    masses: Array,
    *,
    radial_power: float,
    softening: float,
    mass_power: float,
) -> Array:
    result = np.zeros_like(points)
    for position, mass in zip(positions, masses, strict=True):
        displacement = points - position
        radius_squared = np.sum(displacement**2, axis=1) + softening**2
        result += (
            mass**mass_power
            * displacement
            / radius_squared[:, None] ** (radial_power / 2.0)
        )
    return result


def synthetic_multisource_misalignment(
    *,
    grid_size: int = 101,
    extent: float = 6.0,
    separation: float = 4.0,
    mass_ratio: float = 0.3,
) -> dict[str, float]:
    """Return a rotation-invariant two-source geometry diagnostic.

    ``J`` and ``S`` use deliberately different source weights and smoothing.
    The construction is not an AeST solution; it asks only how much angular
    activation a separated-source geometry can provide.  A single source is
    an exact zero control for any radial kernels.
    """

    size = int(grid_size)
    if size < 11:
        raise ValueError("grid_size must be at least 11")
    limit = _positive(extent, name="extent")
    distance = _positive(separation, name="separation")
    ratio = _positive(mass_ratio, name="mass_ratio")
    axis = np.linspace(-limit, limit, size)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    positions = np.array([[-distance / 2.0, 0.0, 0.0], [distance / 2.0, 0.0, 0.0]])
    masses = np.array([1.0, ratio])
    aether = _softened_field(
        points,
        positions,
        masses,
        radial_power=3.0,
        softening=0.35,
        mass_power=1.0,
    )
    scalar = _softened_field(
        points,
        positions,
        masses,
        radial_power=2.5,
        softening=1.0,
        mass_power=0.5,
    )
    y_inv = np.einsum("ij,ij->i", scalar, scalar)
    z_inv = np.einsum("ij,ij->i", aether, aether)
    u_inv = np.einsum("ij,ij->i", scalar, aether)
    denominator = y_inv * z_inv
    valid = denominator > 1.0e-16
    sine_squared = np.clip(
        (denominator[valid] - u_inv[valid] ** 2) / denominator[valid],
        0.0,
        1.0,
    )
    weights = np.sqrt(z_inv[valid])
    order = np.argsort(sine_squared)
    cumulative = np.cumsum(weights[order])

    def weighted_quantile(probability: float) -> float:
        index = int(np.searchsorted(cumulative, probability * cumulative[-1]))
        return float(sine_squared[order[min(index, order.size - 1)]])

    single_position = np.zeros((1, 3))
    single_mass = np.ones(1)
    single_aether = _softened_field(
        points,
        single_position,
        single_mass,
        radial_power=3.0,
        softening=0.35,
        mass_power=1.0,
    )
    single_scalar = _softened_field(
        points,
        single_position,
        single_mass,
        radial_power=2.5,
        softening=1.0,
        mass_power=0.5,
    )
    sy = np.einsum("ij,ij->i", single_scalar, single_scalar)
    sz = np.einsum("ij,ij->i", single_aether, single_aether)
    su = np.einsum("ij,ij->i", single_scalar, single_aether)
    single_valid = sy * sz > 1.0e-16
    single_sine = np.abs((sy[single_valid] * sz[single_valid] - su[single_valid] ** 2) / (sy[single_valid] * sz[single_valid]))
    return {
        "single_source_maximum_sine_squared": float(np.max(single_sine)),
        "two_source_maximum_sine_squared": float(np.max(sine_squared)),
        "two_source_field_weighted_mean_sine_squared": float(
            np.average(sine_squared, weights=weights)
        ),
        "two_source_field_weighted_median_sine_squared": weighted_quantile(0.5),
        "two_source_field_weighted_p90_sine_squared": weighted_quantile(0.9),
        "two_source_field_weighted_fraction_above_half": float(
            np.average(sine_squared > 0.5, weights=weights)
        ),
    }


def audit_v9a_bounded_alignment(
    *,
    k_b: float,
    eta: float,
    rank_surface_eta_values: Array,
    y_values: Array,
    z_values: Array,
    cosine_values: Array,
    random_samples: int,
    random_seed: int,
    fixed_mond_mean_fraction: float,
) -> dict[str, object]:
    """Run the preregistered no-observation v9A gate."""

    vector = _positive(k_b, name="k_b")
    coupling = _positive(eta, name="eta")
    mean_fraction = _positive(fixed_mond_mean_fraction, name="fixed_mond_mean_fraction")
    rank_surfaces = [
        find_one_sided_rank_surface(eta=float(value), k_b=vector)
        for value in np.asarray(rank_surface_eta_values, dtype=float)
    ]
    saturated_scan = scan_saturated_static_principal_symbol(
        eta=coupling,
        k_b=vector,
        y_values=y_values,
        z_values=z_values,
        cosine_values=cosine_values,
        random_samples=random_samples,
        random_seed=random_seed,
    )
    transition_kinetic = aether_rest_vector_kinetic_eigenvalues(
        1.0,
        k_b=vector,
        eta=coupling,
    )
    target = 1.0 / mean_fraction
    seventy_five_percent_target = 1.0 + 0.75 * (target - 1.0)
    aligned_scalar = np.array([1.0, 0.0, 0.0])
    aligned_aether = np.array([2.0, 0.0, 0.0])
    aligned_checks: dict[str, object] = {}
    for selected in ("one_sided", "saturated"):
        fluxes = alignment_fluxes(
            aligned_scalar,
            aligned_aether,
            a_sigma=1.0,
            eta=coupling,
            variant=selected,  # type: ignore[arg-type]
        )
        aligned_checks[selected] = {
            "density": alignment_interaction_density(
                aligned_scalar,
                aligned_aether,
                a_sigma=1.0,
                eta=coupling,
                variant=selected,  # type: ignore[arg-type]
            ),
            "maximum_absolute_flux": float(
                max(
                    np.max(np.abs(fluxes.scalar_gradient_flux)),
                    np.max(np.abs(fluxes.aether_acceleration_flux)),
                )
            ),
        }
    synthetic = synthetic_multisource_misalignment()
    return {
        "selected_values": {"K_B": vector, "eta_sigma": coupling},
        "flat_quadratic_spectrum_unchanged": True,
        "first_derivatives_only": True,
        "transition_vector_kinetic": transition_kinetic,
        "vector_kinetic_positive_all_Y": bool(coupling < vector),
        "one_sided_rank_surfaces": rank_surfaces,
        "one_sided_globally_regular": False,
        "saturated_static_scan": saturated_scan,
        "aligned_spherical_checks": aligned_checks,
        "exact_spherical_background_correction": 0.0,
        "maximum_perpendicular_amplification": maximum_perpendicular_amplification(
            k_b=vector,
            eta=coupling,
        ),
        "known_development_amplitude_target": target,
        "seventy_five_percent_gap_target": seventy_five_percent_target,
        "angle_for_full_target": angle_required_for_amplification(
            target,
            k_b=vector,
            eta=coupling,
        ),
        "angle_for_seventy_five_percent_gap": angle_required_for_amplification(
            seventy_five_percent_target,
            k_b=vector,
            eta=coupling,
        ),
        "synthetic_geometry": synthetic,
        "one_sided_decision": "retire_on_finite_static_principal_rank_surface",
        "saturated_decision": "retire_as_standalone_unification_completion_on_exact_spherical_null",
        "observational_data_accessed": False,
        "new_holdout_opened": False,
    }
