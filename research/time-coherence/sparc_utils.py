"""
Utility functions for loading SPARC data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np


def load_sparc_summary(path: str | Path) -> pd.DataFrame:
    """
    Load SPARC summary CSV file.
    
    Parameters
    ----------
    path : str or Path
        Path to SPARC summary CSV
        
    Returns
    -------
    pd.DataFrame
        DataFrame with galaxy properties
    """
    df = pd.read_csv(path)
    # Rename / enforce columns as needed
    # expects at least: galaxy_name, sigma_velocity, R_disk, (optional) morph_class
    if "R_disk" not in df.columns and "Ropt" in df.columns:
        # example mapping; adjust to your schema
        df = df.rename(columns={"Ropt": "R_disk"})
    return df


def load_rotmod(path: str | Path) -> pd.DataFrame:
    """
    Minimal SPARC rotmod loader.
    
    Parameters
    ----------
    path : str or Path
        Path to rotmod file
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: R_kpc, V_obs, V_gr, V_gas, V_disk, V_bul
    """
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        usecols=[0, 1, 3, 4, 5],
        names=["R_kpc", "V_obs", "V_gas", "V_disk", "V_bul"],
        engine="python",
    )
    V_gr = np.sqrt(
        np.clip(
            df["V_gas"].to_numpy() ** 2
            + df["V_disk"].to_numpy() ** 2
            + df["V_bul"].to_numpy() ** 2,
            0.0,
            None,
        )
    )
    df["V_gr"] = V_gr
    return df[["R_kpc", "V_obs", "V_gr", "V_gas", "V_disk", "V_bul"]]


def rms_velocity(velocity_residuals: np.ndarray) -> float:
    """
    Compute RMS velocity from residuals.
    
    Parameters
    ----------
    velocity_residuals : np.ndarray
        V_obs - V_model residuals in km/s
        
    Returns
    -------
    float
        RMS velocity in km/s
    """
    residuals = np.asarray(velocity_residuals, dtype=float)
    mask = np.isfinite(residuals)
    if not np.any(mask):
        return np.nan
    return float(np.sqrt(np.mean(residuals[mask] ** 2)))


