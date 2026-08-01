"""Forward model for same-source astrometric and spectroscopic velocities."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def stable_source_folds(names, folds: int, seed: int) -> np.ndarray:
    """Assign complete sources reproducibly to folds."""
    values = []
    for name in names:
        digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
        values.append(int.from_bytes(digest[:8], "big") % folds)
    return np.asarray(values, dtype=int)


def photon_features(
    radius_kpc: np.ndarray,
    *,
    model: str,
    r0_kpc: float,
    theta_reference_km_s: float,
    a_star_m_s2: float,
) -> tuple[np.ndarray, list[str]]:
    """Return effective circular-speed features for a photon model."""
    radius = np.asarray(radius_kpc, dtype=float)
    if model == "null":
        return np.empty((len(radius), 0)), []
    if model == "frequency_constant":
        return np.ones((len(radius), 1)), ["photon_A_km_s"]
    if model == "frequency_low_acceleration":
        radius_m = radius * 3.085677581491367e19
        acceleration = (theta_reference_km_s * 1000.0) ** 2 / radius_m
        gate = a_star_m_s2 / (a_star_m_s2 + acceleration)
        return gate[:, np.newaxis], ["photon_A_km_s"]
    if model == "frequency_radial_log_post_primary":
        return (
            np.column_stack((np.ones(len(radius)), np.log(radius / r0_kpc))),
            ["photon_A0_km_s", "photon_A1_km_s_per_ln_R_R0"],
        )
    raise ValueError(f"unknown photon model: {model}")


def channel_design(
    frame: pd.DataFrame,
    *,
    rotation_order: int,
    photon_model: str,
    r0_kpc: float,
    theta_reference_km_s: float,
    a_star_m_s2: float,
    fixed_solar_z_km_s: float,
    velocity_error_floor_km_s: float,
) -> dict[str, np.ndarray | list[str]]:
    """Stack longitude and radial measurements into one linear forward model."""
    if rotation_order < 0:
        raise ValueError("rotation_order must be non-negative")
    radius = frame["radius_kpc"].to_numpy(float)
    x = (radius - r0_kpc) / r0_kpc
    curve = np.column_stack([x**power for power in range(rotation_order + 1)])
    curve_names = [
        "theta_R0_km_s"
        if power == 0
        else f"theta_curve_c{power}_km_s"
        for power in range(rotation_order + 1)
    ]

    longitude = np.deg2rad(frame["l_deg"].to_numpy(float))
    latitude = np.deg2rad(frame["b_deg"].to_numpy(float))
    cos_b = np.cos(latitude)
    e_l_x = np.sin(longitude)
    e_l_y = np.cos(longitude)
    e_r_x = -cos_b * np.cos(longitude)
    e_r_y = cos_b * np.sin(longitude)
    e_r_z = np.sin(latitude)
    p_l = frame["longitude_projection"].to_numpy(float)
    p_r = frame["radial_projection"].to_numpy(float)

    photons, photon_names = photon_features(
        radius,
        model=photon_model,
        r0_kpc=r0_kpc,
        theta_reference_km_s=theta_reference_km_s,
        a_star_m_s2=a_star_m_s2,
    )
    parameter_names = curve_names + ["solar_vx_km_s", "solar_vy_km_s"] + photon_names
    zeros = np.zeros_like(photons)
    proper_design = np.column_stack(
        (p_l[:, np.newaxis] * curve, -e_l_x, -e_l_y, zeros)
    )
    radial_design = np.column_stack(
        (p_r[:, np.newaxis] * curve, -e_r_x, -e_r_y, p_r[:, np.newaxis] * photons)
    )
    design = np.vstack((proper_design, radial_design))

    proper_velocity = frame["v_longitude_mc_median_km_s"].to_numpy(float)
    radial_velocity = (
        frame["v_helio_radial_mc_median_km_s"].to_numpy(float)
        + e_r_z * fixed_solar_z_km_s
    )
    observed = np.concatenate((proper_velocity, radial_velocity))
    proper_sigma = np.maximum(
        frame["v_longitude_mc_sigma_km_s"].to_numpy(float),
        velocity_error_floor_km_s,
    )
    radial_sigma = np.maximum(
        frame["v_helio_radial_mc_sigma_km_s"].to_numpy(float),
        velocity_error_floor_km_s,
    )
    sigma = np.concatenate((proper_sigma, radial_sigma))
    channel = np.asarray(["proper_motion"] * len(frame) + ["radial_velocity"] * len(frame))
    source = np.concatenate(
        (frame["system"].to_numpy(str), frame["system"].to_numpy(str))
    )
    return {
        "design": design,
        "observed": observed,
        "sigma": sigma,
        "channel": channel,
        "source": source,
        "parameter_names": parameter_names,
        "radial_solar_z_correction": np.concatenate(
            (np.zeros(len(frame)), e_r_z * fixed_solar_z_km_s)
        ),
    }


def fit_weighted_linear(
    design: np.ndarray, observed: np.ndarray, sigma: np.ndarray
) -> dict[str, np.ndarray | float]:
    """Fit a weighted linear model and return formal covariance."""
    design = np.asarray(design, dtype=float)
    observed = np.asarray(observed, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    whitened_design = design / sigma[:, np.newaxis]
    whitened_observed = observed / sigma
    values, _, rank, singular = np.linalg.lstsq(
        whitened_design, whitened_observed, rcond=None
    )
    if rank != design.shape[1]:
        raise np.linalg.LinAlgError(
            f"rank-deficient design: rank {rank}, parameters {design.shape[1]}"
        )
    normal = whitened_design.T @ whitened_design
    covariance = np.linalg.inv(normal)
    prediction = design @ values
    residual = observed - prediction
    return {
        "values": values,
        "covariance": covariance,
        "prediction": prediction,
        "residual": residual,
        "chi2": float(np.sum(np.square(residual / sigma))),
        "rank": float(rank),
        "condition_number": float(singular[0] / singular[-1]),
    }
