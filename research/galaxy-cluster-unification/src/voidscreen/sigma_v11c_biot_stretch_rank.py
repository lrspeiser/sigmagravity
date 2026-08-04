"""Exact tilted-background kinetic audit for the Sigma v11C Biot triad."""

from __future__ import annotations

import numpy as np


def _validated_inputs(
    *,
    aether_tilt: float,
    shear_speed_squared: float,
    bulk_weight: float,
    transverse_stretch: float,
    axial_stretch: float,
) -> tuple[float, float, float, float, float]:
    values = np.asarray(
        [
            aether_tilt,
            shear_speed_squared,
            bulk_weight,
            transverse_stretch,
            axial_stretch,
        ],
        dtype=float,
    )
    if (
        np.any(~np.isfinite(values))
        or not 0.0 < abs(values[0]) < 1.0
        or values[1] <= 0.0
        or values[2] <= 1.0 / 3.0
        or values[3] <= 0.0
        or values[4] <= 0.0
    ):
        raise ValueError("v11C inputs are outside the finite GL+(3) domain")
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def biot_energy_from_singular_values(
    singular_values: np.ndarray,
    *,
    shear_speed_squared: float,
    bulk_weight: float,
) -> float:
    """Return s[dev(U-I)^2+b tr(U-I)^2] for positive stretches."""

    sigma = np.asarray(singular_values, dtype=float)
    shear = float(shear_speed_squared)
    bulk = float(bulk_weight)
    if (
        sigma.shape != (3,)
        or np.any(~np.isfinite(sigma))
        or np.any(sigma <= 0.0)
        or not np.isfinite(shear)
        or not np.isfinite(bulk)
        or shear <= 0.0
        or bulk <= 1.0 / 3.0
    ):
        raise ValueError("Biot energy requires three positive finite stretches")
    strain = sigma - 1.0
    trace = float(np.sum(strain))
    deviator = strain - trace / 3.0
    return float(shear * (deviator @ deviator + bulk * trace**2))


def shear_path_invariants(
    velocity_perturbation: float,
    *,
    aether_tilt: float,
    transverse_stretch: float,
    axial_stretch: float,
) -> dict[str, float]:
    """Exact invariants of D=[[e,gamma*v*w,0],[0,e,0],[0,0,M]]."""

    w = float(velocity_perturbation)
    v = float(aether_tilt)
    e = float(transverse_stretch)
    axial = float(axial_stretch)
    if (
        not np.all(np.isfinite([w, v, e, axial]))
        or not 0.0 < abs(v) < 1.0
        or e <= 0.0
        or axial <= 0.0
    ):
        raise ValueError("shear-path inputs are outside the finite GL+(3) domain")
    gamma = 1.0 / np.sqrt(1.0 - v**2)
    off_diagonal = gamma * v * w
    nuclear_norm = axial + np.sqrt(4.0 * e**2 + off_diagonal**2)
    frobenius_squared = 2.0 * e**2 + axial**2 + off_diagonal**2
    return {
        "gamma": float(gamma),
        "off_diagonal": float(off_diagonal),
        "nuclear_norm": float(nuclear_norm),
        "frobenius_squared": float(frobenius_squared),
        "determinant": float(e**2 * axial),
    }


def tilted_biot_lagrangian(
    velocity_perturbation: float,
    *,
    aether_tilt: float,
    shear_speed_squared: float,
    bulk_weight: float,
    transverse_stretch: float,
    axial_stretch: float,
) -> float:
    """Exact Q^2/2-W_Biot along the decisive material-shear path."""

    v, shear, bulk, e, axial = _validated_inputs(
        aether_tilt=aether_tilt,
        shear_speed_squared=shear_speed_squared,
        bulk_weight=bulk_weight,
        transverse_stretch=transverse_stretch,
        axial_stretch=axial_stretch,
    )
    w = float(velocity_perturbation)
    if not np.isfinite(w):
        raise ValueError("velocity perturbation must be finite")
    invariants = shear_path_invariants(
        w,
        aether_tilt=v,
        transverse_stretch=e,
        axial_stretch=axial,
    )
    trace_u = invariants["nuclear_norm"]
    sum_strain_squared = invariants["frobenius_squared"] - 2.0 * trace_u + 3.0
    trace_strain = trace_u - 3.0
    energy = shear * (sum_strain_squared + (bulk - 1.0 / 3.0) * trace_strain**2)
    return float(0.5 * invariants["gamma"] ** 2 * w**2 - energy)


def rank_one_biot_stiffness(
    *,
    shear_speed_squared: float,
    bulk_weight: float,
    transverse_stretch: float,
    axial_stretch: float,
) -> float:
    """D^2 W[e1 tensor e2,e1 tensor e2] at diag(e,e,M)."""

    shear = float(shear_speed_squared)
    bulk = float(bulk_weight)
    e = float(transverse_stretch)
    axial = float(axial_stretch)
    if (
        not np.all(np.isfinite([shear, bulk, e, axial]))
        or shear <= 0.0
        or bulk <= 1.0 / 3.0
        or e <= 0.0
        or axial <= 0.0
    ):
        raise ValueError("rank-one stiffness inputs are outside their domain")
    trace_u = axial + 2.0 * e
    bulk_offset = bulk - 1.0 / 3.0
    nuclear_coefficient = -2.0 + 2.0 * bulk_offset * (trace_u - 3.0)
    return float(shear * (2.0 + nuclear_coefficient / (2.0 * e)))


def tilted_biot_hessian(
    *,
    aether_tilt: float,
    shear_speed_squared: float,
    bulk_weight: float,
    transverse_stretch: float,
    axial_stretch: float,
) -> float:
    """Exact coordinate-velocity Hessian at w=0."""

    v, shear, bulk, e, axial = _validated_inputs(
        aether_tilt=aether_tilt,
        shear_speed_squared=shear_speed_squared,
        bulk_weight=bulk_weight,
        transverse_stretch=transverse_stretch,
        axial_stretch=axial_stretch,
    )
    gamma_squared = 1.0 / (1.0 - v**2)
    stiffness = rank_one_biot_stiffness(
        shear_speed_squared=shear,
        bulk_weight=bulk,
        transverse_stretch=e,
        axial_stretch=axial,
    )
    return float(gamma_squared * (1.0 - v**2 * stiffness))


def critical_axial_stretch(
    *,
    aether_tilt: float,
    shear_speed_squared: float,
    bulk_weight: float,
    transverse_stretch: float,
) -> float:
    """Solve H=0 exactly for the positive axial stretch M."""

    v, shear, bulk, e, _ = _validated_inputs(
        aether_tilt=aether_tilt,
        shear_speed_squared=shear_speed_squared,
        bulk_weight=bulk_weight,
        transverse_stretch=transverse_stretch,
        axial_stretch=1.0,
    )
    bulk_offset = bulk - 1.0 / 3.0
    critical = 3.0 - 2.0 * e + (1.0 + e * (1.0 / (shear * v**2) - 2.0)) / bulk_offset
    if critical <= 0.0 or not np.isfinite(critical):
        raise RuntimeError("no positive finite axial rank surface")
    return float(critical)


def vacuum_longitudinal_hessian(
    *,
    aether_tilt: float,
    shear_speed_squared: float,
    bulk_weight: float,
) -> float:
    """The healthy one-stretch channel that motivated the v11C completion."""

    v, shear, bulk, _, _ = _validated_inputs(
        aether_tilt=aether_tilt,
        shear_speed_squared=shear_speed_squared,
        bulk_weight=bulk_weight,
        transverse_stretch=1.0,
        axial_stretch=1.0,
    )
    gamma_squared = 1.0 / (1.0 - v**2)
    longitudinal_speed_squared = 2.0 * shear * (2.0 / 3.0 + bulk)
    return float(gamma_squared * (1.0 - v**2 * longitudinal_speed_squared))


def audit_v11c_biot_stretch_rank(
    *,
    aether_tilt: float,
    shear_speed_squared: float,
    bulk_weight: float,
    transverse_stretch: float,
    counterexample_axial_stretch: float,
    finite_difference_step: float,
) -> dict[str, object]:
    fixed = {
        "aether_tilt": float(aether_tilt),
        "shear_speed_squared": float(shear_speed_squared),
        "bulk_weight": float(bulk_weight),
        "transverse_stretch": float(transverse_stretch),
    }
    critical = critical_axial_stretch(**fixed)
    ratios = np.asarray([0.99, 1.0, 1.01])
    hessians = np.asarray(
        [tilted_biot_hessian(axial_stretch=ratio * critical, **fixed) for ratio in ratios]
    )
    counterexample_hessian = tilted_biot_hessian(
        axial_stretch=float(counterexample_axial_stretch), **fixed
    )
    stiffness = rank_one_biot_stiffness(
        shear_speed_squared=float(shear_speed_squared),
        bulk_weight=float(bulk_weight),
        transverse_stretch=float(transverse_stretch),
        axial_stretch=float(counterexample_axial_stretch),
    )
    step = float(finite_difference_step)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("finite-difference step must be positive and finite")
    lagrangian_values = np.asarray(
        [
            tilted_biot_lagrangian(
                offset,
                axial_stretch=float(counterexample_axial_stretch),
                **fixed,
            )
            for offset in (-step, 0.0, step)
        ]
    )
    fd_hessian = float(
        (lagrangian_values[0] - 2.0 * lagrangian_values[1] + lagrangian_values[2]) / step**2
    )
    counterexample_invariants = shear_path_invariants(
        0.0,
        aether_tilt=float(aether_tilt),
        transverse_stretch=float(transverse_stretch),
        axial_stretch=float(counterexample_axial_stretch),
    )
    longitudinal_hessian = vacuum_longitudinal_hessian(
        aether_tilt=float(aether_tilt),
        shear_speed_squared=float(shear_speed_squared),
        bulk_weight=float(bulk_weight),
    )
    gates = {
        "simple_longitudinal_repair_positive": longitudinal_hessian > 0.0,
        "critical_configuration_finite": bool(
            np.all(np.isfinite([critical, *hessians, *lagrangian_values]))
        ),
        "critical_map_orientation_preserving": bool(
            float(transverse_stretch) ** 2 * critical > 0.0
        ),
        "critical_material_flow_timelike": True,
        "below_surface_positive": hessians[0] > 0.0,
        "critical_surface_zero": abs(hessians[1]) < 1.0e-12,
        "above_surface_negative": hessians[2] < 0.0,
        "analytic_finite_difference_agree": bool(abs(fd_hessian - counterexample_hessian) < 1.0e-5),
        "globally_positive_material_velocity_hessian": counterexample_hessian > 0.0,
    }
    return {
        "background": {
            "metric": "Minkowski",
            "aether": "A^mu=gamma(1,v,0,0)",
            "material_map": "X^1=e gamma(x-vt), X^2=e y, X^3=M z",
            "perturbation": "X^2 -> X^2+w t",
            "aether_rest_velocity": "Q(w)=gamma w e_2; Q(0)=0",
            "aether_spatial_gradient": "D(w)=[[e,gamma v w,0],[0,e,0],[0,0,M]]",
        },
        "action_identity": {
            "stretch": "U=sqrt(D^T D)",
            "biot_strain": "S=U-I",
            "lagrangian": "L=Q_I Q^I/2-s[S_TF:S_TF+b(tr S)^2]",
            "linear_spectrum": "c_shear^2=s=3/11; c_longitudinal^2=2s(2/3+b)=3/4",
        },
        "analytic_identity": {
            "trace_stretch": "tr U=M+sqrt(4e^2+(gamma v w)^2)",
            "rank_one_stiffness": "K=s[2+{-2+2(b-1/3)(M+2e-3)}/(2e)]",
            "coordinate_hessian": "H=gamma^2[1-v^2 K]",
            "rank_surface": "M*=3-2e+[1+e(1/(s v^2)-2)]/(b-1/3)",
        },
        "simple_channel": {
            "vacuum_longitudinal_hessian": longitudinal_hessian,
            "interpretation": "Biot stretch repairs the v11B one-dimensional quartic, but not mixed shear on anisotropic finite backgrounds.",
        },
        "critical_configuration": {
            "transverse_stretch": float(transverse_stretch),
            "axial_stretch": critical,
            "determinant_D": float(transverse_stretch) ** 2 * critical,
            "material_speed_relative_to_aether": 0.0,
            "lagrangian_at_zero_velocity": tilted_biot_lagrangian(
                0.0, axial_stretch=critical, **fixed
            ),
        },
        "crossing": [
            {
                "axial_stretch_over_critical": float(ratio),
                "axial_stretch": float(ratio * critical),
                "hessian": float(hessian),
                "material_speed_relative_to_aether": 0.0,
            }
            for ratio, hessian in zip(ratios, hessians, strict=True)
        ],
        "counterexample": {
            "transverse_stretch": float(transverse_stretch),
            "axial_stretch": float(counterexample_axial_stretch),
            "determinant_D": counterexample_invariants["determinant"],
            "rank_one_biot_stiffness": stiffness,
            "maximum_stiffness_allowed_at_tilt": 1.0 / float(aether_tilt) ** 2,
            "analytic_hessian": counterexample_hessian,
            "finite_difference_hessian": fd_hessian,
            "lagrangian_at_zero_velocity": float(lagrangian_values[1]),
            "material_speed_relative_to_aether": 0.0,
        },
        "gates": {name: bool(value) for name, value in gates.items()},
        "all_nonlinear_rank_gates_pass": bool(all(gates.values())),
        "third_post_reset_same_gate_failure": bool(
            not gates["globally_positive_material_velocity_hessian"]
        ),
        "mechanism_reset_required": bool(not gates["globally_positive_material_velocity_hessian"]),
        "observational_data_accessed": False,
        "raw_holdout_opened": False,
    }
