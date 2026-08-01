"""Parsing and interpolation for the ACCEPT Chandra profile table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ACCEPT_COLUMNS = (
    "name",
    "rin_mpc",
    "rout_mpc",
    "nelec_cm3",
    "neerr_cm3",
    "Kitpl_keV_cm2",
    "Kflat_keV_cm2",
    "Kerr_keV_cm2",
    "Pitpl_dyne_cm2",
    "Pflat_dyne_cm2",
    "Perr_dyne_cm2",
    "Mgrav_msun",
    "Merr_msun",
    "Tx_keV",
    "Txerr_keV",
    "Lambda_erg_cm3_s",
    "tcool52_gyr",
    "t52err_gyr",
    "tcool32_gyr",
    "t32err_gyr",
)


def load_accept_profiles(path: str | Path) -> pd.DataFrame:
    """Load the whitespace table and add its annular midpoint in kpc."""
    frame = pd.read_csv(
        Path(path),
        sep=r"\s+",
        comment="#",
        header=None,
        names=ACCEPT_COLUMNS,
    )
    if frame.empty:
        raise ValueError("ACCEPT table is empty")
    # The public snapshot has four ABELL_2384 shells with a literal NaN
    # electron density and 88 missing gravitating-mass estimates.  Only the
    # measured density fields are required here; discard unusable density
    # shells while retaining optional NaNs in unrelated columns.
    required = ["rin_mpc", "rout_mpc", "nelec_cm3", "neerr_cm3"]
    frame = frame.dropna(subset=required).reset_index(drop=True)
    values = frame[required].to_numpy(dtype=float)
    if frame.empty or np.any(~np.isfinite(values)):
        raise ValueError("ACCEPT table has no finite density rows")
    if (
        np.any(frame["rin_mpc"] < 0.0)
        or np.any(frame["rout_mpc"] <= frame["rin_mpc"])
        or np.any(frame["nelec_cm3"] <= 0.0)
        or np.any(frame["neerr_cm3"] < 0.0)
    ):
        raise ValueError("ACCEPT radii and electron densities are invalid")
    frame = frame.copy()
    frame["radius_kpc"] = 500.0 * (frame["rin_mpc"] + frame["rout_mpc"])
    return frame


def interpolate_electron_density_cm3(
    profile: pd.DataFrame, radius_kpc
) -> np.ndarray:
    """Log-log interpolate n_e at measured radii without extrapolation."""
    radius = np.asarray(radius_kpc, dtype=float)
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("radius_kpc must be finite and positive")
    ordered = profile.sort_values("radius_kpc")
    measured_radius = ordered["radius_kpc"].to_numpy(dtype=float)
    density = ordered["nelec_cm3"].to_numpy(dtype=float)
    if len(measured_radius) < 2 or np.any(np.diff(measured_radius) <= 0.0):
        raise ValueError("profile needs at least two distinct radial points")
    if np.any(radius < measured_radius[0]) or np.any(radius > measured_radius[-1]):
        raise ValueError("refusing to extrapolate beyond ACCEPT radial coverage")
    return np.power(
        10.0,
        np.interp(np.log10(radius), np.log10(measured_radius), np.log10(density)),
    )
