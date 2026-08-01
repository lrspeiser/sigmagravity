"""Universal one-parameter radial acceleration laws for cluster lens tests."""

from __future__ import annotations

import numpy as np

from voidscreen.phenomenology import fixed_rar_enhancement


FAMILIES = {
    "constant_boost",
    "mass_isothermal_tail",
    "screened_mass_isothermal_tail",
    "mond_isothermal_tail",
    "rar_acceleration_scale",
    "scaled_rar_extra",
    "constant_acceleration",
    "harmonic_tide",
    "rar_transition_power",
}


def predict_one_parameter_acceleration(
    family: str,
    gbar_m_s2,
    radius_kpc,
    parameter: float,
    *,
    a0_m_s2: float = 1.2e-10,
    reference_radius_kpc: float = 200.0,
    gbar_at_reference_m_s2: float | None = None,
) -> np.ndarray:
    """Return one universal radial law.

    ``gbar_at_reference_m_s2`` is an observed baryonic-profile input rather
    than a fitted cluster parameter. It is required only by the mass-normalized
    isothermal tail.
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown one-parameter family {family}")
    gbar = np.asarray(gbar_m_s2, dtype=np.float64)
    radius = np.asarray(radius_kpc, dtype=np.float64)
    gbar, radius = np.broadcast_arrays(gbar, radius)
    value = float(parameter)
    if (
        np.any(~np.isfinite(gbar))
        or np.any(gbar <= 0.0)
        or np.any(~np.isfinite(radius))
        or np.any(radius <= 0.0)
        or not np.isfinite(value)
        or not np.isfinite(a0_m_s2)
        or a0_m_s2 <= 0.0
        or not np.isfinite(reference_radius_kpc)
        or reference_radius_kpc <= 0.0
    ):
        raise ValueError("accelerations, radii, scales, and parameter must be finite")

    if family == "constant_boost":
        if value <= 0.0:
            raise ValueError("constant boost must be positive")
        predicted = value * gbar
    elif family == "mass_isothermal_tail":
        if value < 0.0:
            raise ValueError("tail amplitude must be nonnegative")
        if gbar_at_reference_m_s2 is None:
            raise ValueError("mass-isothermal tail requires reference acceleration")
        reference = float(gbar_at_reference_m_s2)
        if not np.isfinite(reference) or reference <= 0.0:
            raise ValueError("reference acceleration must be finite and positive")
        predicted = gbar + value * reference * reference_radius_kpc / radius
    elif family == "screened_mass_isothermal_tail":
        if value < 0.0:
            raise ValueError("screened-tail amplitude must be nonnegative")
        if gbar_at_reference_m_s2 is None:
            raise ValueError("screened mass-isothermal tail requires reference acceleration")
        reference = float(gbar_at_reference_m_s2)
        if not np.isfinite(reference) or reference <= 0.0:
            raise ValueError("reference acceleration must be finite and positive")
        high_acceleration_screen = a0_m_s2 / (a0_m_s2 + gbar)
        predicted = (
            gbar
            + value
            * reference
            * reference_radius_kpc
            / radius
            * high_acceleration_screen
        )
    elif family == "mond_isothermal_tail":
        if value < 0.0:
            raise ValueError("MOND-tail amplitude must be nonnegative")
        predicted = gbar + value * np.sqrt(a0_m_s2 * gbar)
    elif family == "rar_acceleration_scale":
        if value <= 0.0:
            raise ValueError("RAR scale multiplier must be positive")
        predicted = gbar * fixed_rar_enhancement(gbar, value * a0_m_s2)
    elif family == "scaled_rar_extra":
        if value < 0.0:
            raise ValueError("RAR extra-force scale must be nonnegative")
        rar = gbar * fixed_rar_enhancement(gbar, a0_m_s2)
        predicted = gbar + value * (rar - gbar)
    elif family == "constant_acceleration":
        if value < 0.0:
            raise ValueError("constant acceleration amplitude must be nonnegative")
        predicted = gbar + value * a0_m_s2
    elif family == "harmonic_tide":
        if value < 0.0:
            raise ValueError("harmonic amplitude must be nonnegative")
        predicted = gbar + value * a0_m_s2 * radius / reference_radius_kpc
    else:
        if value <= 0.0:
            raise ValueError("transition power must be positive")
        predicted = gbar * np.power(
            1.0 + np.power(a0_m_s2 / gbar, value),
            1.0 / (2.0 * value),
        )

    if np.any(~np.isfinite(predicted)) or np.any(predicted <= 0.0):
        raise ValueError("one-parameter law produced an invalid acceleration")
    return predicted
