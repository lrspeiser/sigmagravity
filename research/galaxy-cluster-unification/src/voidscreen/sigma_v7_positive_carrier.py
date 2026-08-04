"""Construction-level checks for a positive local spin-2 Sigma carrier.

The module audits the universal linear limit shared by a massless graviton and
a positive-norm Fierz--Pauli massive spin-2 mode.  It deliberately does not
open or fit astronomical data.  Around a proportional flat background the
same spectrum is obtained from ghost-free Hassan--Rosen bimetric gravity.

For a conserved nonrelativistic source, a massive spin-2 exchange contributes
``4 a / 3`` to the dynamical potential and ``2 a / 3`` to the spatial
potential, where ``a`` is the non-negative pole residue.  The Weyl/lensing
potential therefore receives ``a``.  These fixed ratios make it possible to
test Solar-System slip and large-scale usefulness without choosing an object.
"""

from __future__ import annotations

import numpy as np


def positive_spin2_spectrum() -> dict[str, object]:
    """Return the linear spectrum and manifest kinetic signs of the carrier."""

    return {
        "massless_spin2_degrees_of_freedom": 2,
        "massive_spin2_degrees_of_freedom": 5,
        "total_degrees_of_freedom": 7,
        "kinetic_eigenvalues": np.ones(7, dtype=float),
        "negative_kinetic_directions": 0,
        "null_kinetic_directions": 0,
    }


def yukawa_force_kernel(radius_over_range: np.ndarray | float) -> np.ndarray:
    """Return ``(1+x) exp(-x)``, the point-source Yukawa force kernel."""

    ratio = np.asarray(radius_over_range, dtype=float)
    if np.any(~np.isfinite(ratio)) or np.any(ratio < 0.0):
        raise ValueError("radius_over_range must be finite and non-negative")
    return (1.0 + ratio) * np.exp(-ratio)


def spin2_force_enhancements(
    radius_over_range: np.ndarray | float,
    pole_residue: float,
) -> dict[str, np.ndarray]:
    """Return dynamics, spatial-potential, and lensing force enhancements.

    ``pole_residue`` is the square of the relative coupling of the healthy
    massive mode to conserved matter, so it cannot be negative.
    """

    residue = float(pole_residue)
    if not np.isfinite(residue) or residue < 0.0:
        raise ValueError("pole_residue must be finite and non-negative")
    kernel = yukawa_force_kernel(radius_over_range)
    return {
        "dynamics": 1.0 + (4.0 / 3.0) * residue * kernel,
        "spatial": 1.0 + (2.0 / 3.0) * residue * kernel,
        "lensing": 1.0 + residue * kernel,
    }


def ppn_gamma(pole_residue: np.ndarray | float) -> np.ndarray:
    """Return the unsuppressed short-range PPN slip ``Phi/Psi``."""

    residue = np.asarray(pole_residue, dtype=float)
    if np.any(~np.isfinite(residue)) or np.any(residue < 0.0):
        raise ValueError("pole_residue must be finite and non-negative")
    return (1.0 + (2.0 / 3.0) * residue) / (
        1.0 + (4.0 / 3.0) * residue
    )


def maximum_residue_from_ppn(maximum_gamma_minus_one: float) -> float:
    """Invert the exact short-range PPN bound for a positive massive residue."""

    bound = float(maximum_gamma_minus_one)
    if not np.isfinite(bound) or not 0.0 < bound < 0.5:
        raise ValueError("maximum_gamma_minus_one must lie between zero and one half")
    return 3.0 * bound / (2.0 - 4.0 * bound)


def maximum_residue_from_high_field_force(
    maximum_fractional_extra_force: float,
) -> float:
    """Return the positive residue allowed by the unscreened high-field gate."""

    bound = float(maximum_fractional_extra_force)
    if not np.isfinite(bound) or bound <= 0.0:
        raise ValueError("maximum_fractional_extra_force must be finite and positive")
    return 0.75 * bound


def locally_calibrated_dynamical_ratio(
    radius_over_range: np.ndarray | float,
    pole_residue: float,
) -> np.ndarray:
    """Return force relative to Newton's constant calibrated at ``r/L -> 0``."""

    residue = float(pole_residue)
    enhancement = spin2_force_enhancements(radius_over_range, residue)["dynamics"]
    return enhancement / (1.0 + (4.0 / 3.0) * residue)


def audit_linear_positive_carrier(
    *,
    ppn_bound: float,
    high_field_force_bound: float,
    required_lensing_enhancement: float,
    radius_over_range: np.ndarray,
) -> dict[str, object]:
    """Evaluate the preregistered linear positive-carrier gates."""

    required = float(required_lensing_enhancement)
    if not np.isfinite(required) or required <= 1.0:
        raise ValueError("required_lensing_enhancement must exceed one")
    ratio = np.asarray(radius_over_range, dtype=float)
    kernel = yukawa_force_kernel(ratio)
    spectrum = positive_spin2_spectrum()
    ppn_limit = maximum_residue_from_ppn(ppn_bound)
    force_limit = maximum_residue_from_high_field_force(high_field_force_bound)
    allowed_residue = min(ppn_limit, force_limit)
    maximum_lensing = float(np.max(1.0 + allowed_residue * kernel))
    local_ratio = locally_calibrated_dynamical_ratio(ratio, allowed_residue)
    kernel_differences = np.diff(kernel)
    gates = {
        "positive_local_kinetic_spectrum": spectrum["negative_kinetic_directions"] == 0,
        "fierz_pauli_constraint_count": spectrum["total_degrees_of_freedom"] == 7,
        "solar_ppn_gamma": abs(float(ppn_gamma(allowed_residue)) - 1.0) <= ppn_bound,
        "high_field_extra_force": (4.0 / 3.0) * allowed_residue
        <= high_field_force_bound,
        "large_scale_lensing_amplitude": maximum_lensing
        >= required_lensing_enhancement,
        "turns_on_with_distance": bool(np.any(kernel_differences > 1.0e-14)),
    }
    return {
        "spectrum": spectrum,
        "maximum_residue_from_ppn": ppn_limit,
        "maximum_residue_from_high_field_force": force_limit,
        "maximum_jointly_allowed_residue": allowed_residue,
        "maximum_lensing_enhancement": maximum_lensing,
        "minimum_locally_calibrated_far_force_ratio": float(np.min(local_ratio)),
        "yukawa_force_kernel_monotone_nonincreasing": bool(
            np.all(kernel_differences <= 1.0e-14)
        ),
        "gates": {name: bool(value) for name, value in gates.items()},
    }
