"""Global Legendre-rank falsification of the Sigma v8B completion."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.optimize import brentq

from .sigma_v8b_tilted_adm import (
    aether_spatial_norm,
    homogeneous_canonical_point,
    homogeneous_kinetic_point,
)


def asymptotic_schur_coefficient(
    spatial_norm_squared: float,
    *,
    k_b: float = 1.0,
) -> float:
    """Return ``u.T A^-1 u`` for the leading completion mixing.

    The local kinetic Hessian at zero background extrinsic/electric velocity
    has block form ``[[A, b], [b.T, d]]``.  At large scalar velocity,
    ``b=-C sigma^2 u+O(sigma)``.  A positive result makes the scalar Schur
    complement diverge as ``-C^2 sigma^4``.
    """

    x = float(spatial_norm_squared)
    coupling = float(k_b)
    if not np.isfinite(x) or x < 0.0:
        raise ValueError("spatial_norm_squared must be finite and non-negative")
    if not np.isfinite(coupling) or not 0.0 < coupling < 2.0:
        raise ValueError("K_B must be finite and lie in (0, 2)")
    numerator = (
        (32.0 - 20.0 * coupling) * x**3
        + (32.0 - 67.0 * coupling) * x**2
        + (8.0 - 78.0 * coupling) * x
        - 27.0 * coupling
    )
    return numerator / (4.0 * coupling)


def leading_mixing_vector(spatial_norm_squared: float) -> np.ndarray:
    """Return the leading nine-component metric--aether mixing vector."""

    x = float(spatial_norm_squared)
    if not np.isfinite(x) or x < 0.0:
        raise ValueError("spatial_norm_squared must be finite and non-negative")
    spatial_norm = np.sqrt(x)
    return np.array(
        (
            3.0 + 5.0 * x,
            3.0 + 4.0 * x + x**2,
            3.0 + 4.0 * x + x**2,
            0.0,
            0.0,
            0.0,
            2.0 * spatial_norm * (1.0 + 2.0 * x),
            0.0,
            0.0,
        ),
        dtype=float,
    )


def critical_aether_tilt(*, k_b: float = 1.0) -> dict[str, float]:
    """Return the first positive tilt at which the asymptotic no-go activates."""

    coupling = float(k_b)
    if not np.isfinite(coupling) or not 0.0 < coupling < 1.6:
        raise ValueError("this positive-root branch requires K_B in (0, 1.6)")
    lower = 0.0
    upper = 1.0
    while asymptotic_schur_coefficient(upper, k_b=coupling) <= 0.0:
        upper *= 2.0
        if upper > 1.0e8:
            raise RuntimeError("failed to bracket the asymptotic rank coefficient root")
    root = float(
        brentq(
            lambda value: asymptotic_schur_coefficient(value, k_b=coupling),
            lower,
            upper,
            xtol=1.0e-13,
        )
    )
    velocity = float(np.sqrt(root / (1.0 + root)))
    return {
        "K_B": coupling,
        "critical_spatial_norm_squared": root,
        "critical_aether_velocity": velocity,
        "coefficient_at_root": asymptotic_schur_coefficient(root, k_b=coupling),
    }


def derived_causal_alpha(
    *,
    k_b: float,
    k_2: float = 2.0,
    lambda_s: float = 1.0,
) -> dict[str, float]:
    """Return the published AeST scalar speed and the derived v8B alpha."""

    vector_coupling = float(k_b)
    clock = float(k_2)
    scalar_coupling = float(lambda_s)
    if not all(
        np.isfinite(value)
        for value in (vector_coupling, clock, scalar_coupling)
    ):
        raise ValueError("AeST mode parameters must be finite")
    if not 0.0 < vector_coupling < 2.0 or clock <= 0.0:
        raise ValueError("K_B must lie in (0, 2) and K_2 must be positive")
    speed_squared = (
        (2.0 - vector_coupling)
        / (clock * vector_coupling)
        * (1.0 + 0.5 * vector_coupling * scalar_coupling)
    )
    if not 0.0 < speed_squared < 1.0:
        raise ValueError("the derived completion requires scalar speed squared in (0, 1)")
    alpha = 1.0 / (3.0 * speed_squared * (1.0 - speed_squared))
    return {
        "scalar_speed_squared": float(speed_squared),
        "alpha": float(alpha),
    }


def isotropic_extrinsic_rank_surface(
    *,
    k_b: float,
    aether_velocity: float = 0.5,
    q_over_q0: float = 1.2,
    a_sigma_over_q0: float = 1.0,
    ell_h_q0: float = 1.0,
) -> dict[str, object]:
    """Locate the affine determinant zero along isotropic extrinsic curvature."""

    mode = derived_causal_alpha(k_b=k_b)
    alpha = mode["alpha"]
    spatial_norm = aether_spatial_norm(aether_velocity)
    chi = np.sqrt(1.0 + spatial_norm**2)

    def background(trace: float) -> np.ndarray:
        values = np.zeros(9, dtype=float)
        values[:3] = trace / 3.0
        return values

    def point(trace: float):
        return homogeneous_kinetic_point(
            aether_velocity=aether_velocity,
            q_over_q0=q_over_q0,
            a_sigma_over_q0=a_sigma_over_q0,
            ell_h_q0=ell_h_q0,
            k_b=k_b,
            alpha=alpha,
            background_kinematics=background(trace),
        )

    determinant_zero = point(0.0).determinant_ratio
    determinant_one = point(1.0).determinant_ratio
    determinant_two = point(2.0).determinant_ratio
    slope = determinant_one - determinant_zero
    if abs(slope) <= 1.0e-14:
        raise ValueError("extrinsic-curvature determinant slope is numerically zero")
    root = float(-determinant_zero / slope)
    root_point = point(root)
    eigenvalues, eigenvectors = np.linalg.eigh(root_point.combined_hessian)
    index = int(np.argmin(np.abs(eigenvalues)))
    null_mode = eigenvectors[:, index]
    offset = 0.02 * max(1.0, abs(root))
    before = point(root - offset)
    after = point(root + offset)
    velocities = np.concatenate(
        (background(root), np.array((q_over_q0 / chi,)))
    )
    canonical = homogeneous_canonical_point(
        velocities,
        aether_velocity=aether_velocity,
        a_sigma_over_q0=a_sigma_over_q0,
        ell_h_q0=ell_h_q0,
        k_b=k_b,
        alpha=alpha,
    )
    return {
        "K_B": float(k_b),
        **mode,
        "aether_velocity": float(aether_velocity),
        "q_over_q0": float(q_over_q0),
        "isotropic_extrinsic_trace_over_q0": root,
        "determinant_ratio_at_root": root_point.determinant_ratio,
        "minimum_singular_value_at_root": float(
            root_point.combined_singular_values[-1]
        ),
        "determinant_affine_residual_at_two": float(
            determinant_two - (2.0 * determinant_one - determinant_zero)
        ),
        "null_eigenvalue": float(eigenvalues[index]),
        "null_mode_sector_power": {
            "metric": float(np.sum(null_mode[:6] ** 2)),
            "aether": float(np.sum(null_mode[6:9] ** 2)),
            "scalar": float(null_mode[9] ** 2),
        },
        "inertia_below_surface": before.combined_inertia,
        "inertia_above_surface": after.combined_inertia,
        "lagrangian_at_surface": canonical.lagrangian,
        "canonical_energy_at_surface": canonical.canonical_energy,
        "all_momenta_finite": bool(np.all(np.isfinite(canonical.momenta))),
    }


def find_first_rank_surface(
    *,
    aether_velocity: float,
    a_sigma_over_q0: float,
    ell_h_q0: float,
    k_b: float = 1.0,
    k_2: float = 2.0,
    alpha: float = 16.0 / 9.0,
    initial_q_ratio: float = 1.0,
    maximum_q_ratio: float = 1.0e4,
) -> float:
    """Find the first determinant sign change above the clock minimum."""

    lower = float(initial_q_ratio)
    maximum = float(maximum_q_ratio)
    if not np.isfinite(lower) or lower <= 0.0 or not np.isfinite(maximum):
        raise ValueError("Q search limits must be finite and positive")
    if maximum <= lower:
        raise ValueError("maximum_q_ratio must exceed initial_q_ratio")

    def determinant_ratio(q_ratio: float) -> float:
        return homogeneous_kinetic_point(
            aether_velocity=aether_velocity,
            q_over_q0=q_ratio,
            a_sigma_over_q0=a_sigma_over_q0,
            ell_h_q0=ell_h_q0,
            k_b=k_b,
            k_2=k_2,
            alpha=alpha,
        ).determinant_ratio

    lower_value = determinant_ratio(lower)
    upper = lower * 1.25
    while upper <= maximum:
        upper_value = determinant_ratio(upper)
        if lower_value * upper_value < 0.0:
            return float(
                brentq(determinant_ratio, lower, upper, xtol=1.0e-11)
            )
        lower = upper
        lower_value = upper_value
        upper *= 1.25
    raise ValueError("no rank surface found inside the requested Q range")


def rank_surface_mode_diagnostics(
    *,
    aether_velocity: float = 0.97,
    q_over_q0: float | None = None,
    a_sigma_over_q0: float = 1.0,
    ell_h_q0: float = 1.0,
) -> dict[str, object]:
    """Characterize the null eigenvector and finite canonical crossing energy."""

    q_ratio = (
        find_first_rank_surface(
            aether_velocity=aether_velocity,
            a_sigma_over_q0=a_sigma_over_q0,
            ell_h_q0=ell_h_q0,
        )
        if q_over_q0 is None
        else float(q_over_q0)
    )
    point = homogeneous_kinetic_point(
        aether_velocity=aether_velocity,
        q_over_q0=q_ratio,
        a_sigma_over_q0=a_sigma_over_q0,
        ell_h_q0=ell_h_q0,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(point.combined_hessian)
    index = int(np.argmin(np.abs(eigenvalues)))
    mode = eigenvectors[:, index]
    conformal = np.zeros(10, dtype=float)
    conformal[:3] = 1.0 / np.sqrt(3.0)
    traceless_axis = np.zeros(10, dtype=float)
    traceless_axis[:3] = np.array((2.0, -1.0, -1.0)) / np.sqrt(6.0)
    spatial_norm = aether_spatial_norm(aether_velocity)
    chi = np.sqrt(1.0 + spatial_norm**2)
    velocities = np.zeros(10, dtype=float)
    velocities[-1] = q_ratio / chi
    canonical = homogeneous_canonical_point(
        velocities,
        aether_velocity=aether_velocity,
        a_sigma_over_q0=a_sigma_over_q0,
        ell_h_q0=ell_h_q0,
    )
    labels = (
        "K_xx",
        "K_yy",
        "K_zz",
        "K_xy",
        "K_xz",
        "K_yz",
        "E_x",
        "E_y",
        "E_z",
        "sigma",
    )
    return {
        "aether_velocity": float(aether_velocity),
        "spatial_norm_squared": float(spatial_norm**2),
        "q_over_q0": q_ratio,
        "null_eigenvalue": float(eigenvalues[index]),
        "minimum_singular_value": float(point.combined_singular_values[-1]),
        "mode_components": {
            label: float(component) for label, component in zip(labels, mode, strict=True)
        },
        "sector_power": {
            "metric": float(np.sum(mode[:6] ** 2)),
            "aether": float(np.sum(mode[6:9] ** 2)),
            "scalar": float(mode[9] ** 2),
        },
        "absolute_conformal_overlap": float(abs(mode @ conformal)),
        "absolute_axis_traceless_overlap": float(abs(mode @ traceless_axis)),
        "lagrangian_at_zero_K_E": canonical.lagrangian,
        "canonical_energy_at_zero_K_E": canonical.canonical_energy,
        "all_canonical_momenta_finite": bool(np.all(np.isfinite(canonical.momenta))),
    }


def audit_global_rank_falsification(
    *,
    ell_h_q0_values: Iterable[float],
    a_sigma_over_q0_values: Iterable[float],
    k_b_escape_values: Iterable[float] = (1.0, 1.6, 1.7, 1.8, 1.95),
    selected_aether_velocity: float = 0.97,
    k_b: float = 1.0,
    alpha: float = 16.0 / 9.0,
) -> dict[str, object]:
    """Apply the analytic and numerical global-rank kill gate to v8B."""

    lengths = tuple(float(value) for value in ell_h_q0_values)
    acceleration_scales = tuple(float(value) for value in a_sigma_over_q0_values)
    escape_couplings = tuple(float(value) for value in k_b_escape_values)
    if not lengths or any(value <= 0.0 or not np.isfinite(value) for value in lengths):
        raise ValueError("ell_h_q0_values must contain positive finite values")
    if not acceleration_scales or any(
        value <= 0.0 or not np.isfinite(value) for value in acceleration_scales
    ):
        raise ValueError("a_sigma_over_q0_values must contain positive finite values")
    if not escape_couplings or any(
        not 0.0 < value < 2.0 or not np.isfinite(value)
        for value in escape_couplings
    ):
        raise ValueError("k_b_escape_values must lie in the finite open interval (0, 2)")
    critical = critical_aether_tilt(k_b=k_b)
    x = aether_spatial_norm(selected_aether_velocity) ** 2
    coefficient = asymptotic_schur_coefficient(x, k_b=k_b)
    length_roots = {
        f"{value:g}": find_first_rank_surface(
            aether_velocity=selected_aether_velocity,
            a_sigma_over_q0=1.0,
            ell_h_q0=value,
            k_b=k_b,
            alpha=alpha,
        )
        for value in lengths
    }
    acceleration_roots = {
        f"{value:g}": find_first_rank_surface(
            aether_velocity=selected_aether_velocity,
            a_sigma_over_q0=value,
            ell_h_q0=1.0,
            k_b=k_b,
            alpha=alpha,
        )
        for value in acceleration_scales
    }
    reference_q = length_roots.get("1")
    if reference_q is None:
        reference_q = find_first_rank_surface(
            aether_velocity=selected_aether_velocity,
            a_sigma_over_q0=1.0,
            ell_h_q0=1.0,
            k_b=k_b,
            alpha=alpha,
        )
    mode = rank_surface_mode_diagnostics(
        aether_velocity=selected_aether_velocity,
        q_over_q0=reference_q,
        a_sigma_over_q0=1.0,
        ell_h_q0=1.0,
    )
    before = homogeneous_kinetic_point(
        aether_velocity=selected_aether_velocity,
        q_over_q0=0.98 * reference_q,
        a_sigma_over_q0=1.0,
        ell_h_q0=1.0,
    )
    after = homogeneous_kinetic_point(
        aether_velocity=selected_aether_velocity,
        q_over_q0=1.02 * reference_q,
        a_sigma_over_q0=1.0,
        ell_h_q0=1.0,
    )
    extrinsic_surfaces = {
        f"{value:g}": isotropic_extrinsic_rank_surface(k_b=value)
        for value in escape_couplings
    }
    completed = {
        "asymptotic_rank_coefficient_positive_at_selected_tilt": bool(
            coefficient > 0.0
        ),
        "critical_tilt_is_finite_and_subluminal": bool(
            0.0 < critical["critical_aether_velocity"] < 1.0
        ),
        "every_nonzero_tested_completion_length_has_finite_rank_surface": bool(
            all(np.isfinite(tuple(length_roots.values())))
        ),
        "rank_surface_is_insensitive_to_a_sigma_escape": bool(
            all(np.isfinite(tuple(acceleration_roots.values())))
        ),
        "null_mode_mixes_metric_and_aether_sectors": bool(
            mode["sector_power"]["metric"] > 0.1
            and mode["sector_power"]["aether"] > 0.1
        ),
        "crossing_energy_and_momenta_are_finite": bool(
            np.isfinite(mode["canonical_energy_at_zero_K_E"])
            and mode["all_canonical_momenta_finite"]
        ),
        "rank_crossing_adds_raw_negative_direction": bool(
            before.combined_inertia == (1, 0, 9)
            and after.combined_inertia == (2, 0, 8)
        ),
        "high_K_B_escape_closed_by_finite_extrinsic_curvature_surface": bool(
            all(
                np.isfinite(item["isotropic_extrinsic_trace_over_q0"])
                and abs(item["determinant_ratio_at_root"]) < 1.0e-8
                and abs(item["determinant_affine_residual_at_two"]) < 1.0e-8
                and item["all_momenta_finite"]
                for item in extrinsic_surfaces.values()
            )
        ),
    }
    return {
        "asymptotic_schur_coefficient_formula": (
            "[(32-20*K_B)x^3+(32-67*K_B)x^2+(8-78*K_B)x-27*K_B]/(4*K_B)"
        ),
        "selected_tilt_coefficient": coefficient,
        "critical_tilt": critical,
        "rank_surfaces_by_L_H_Q0": length_roots,
        "rank_surfaces_by_a_sigma_over_Q0": acceleration_roots,
        "isotropic_extrinsic_rank_surfaces_by_K_B": extrinsic_surfaces,
        "reference_surface_mode": mode,
        "inertia_before_surface": before.combined_inertia,
        "inertia_after_surface": after.combined_inertia,
        "completed_falsification_subgates": completed,
        "all_falsification_subgates_pass": bool(all(completed.values())),
        "global_legendre_map_regular": False,
        "candidate_retired": bool(all(completed.values())),
        "raw_holdout_opened": False,
    }
