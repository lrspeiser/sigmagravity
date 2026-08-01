"""Independent inner-radius nuisance fits for fixed SPARC force laws."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .data import KPC_M
from .path_completion import (
    MASS_PATH_MODELS,
    PATH_MODELS,
    mass_path_completion_profile,
    path_completion_profile,
)
from .phenomenology import (
    fixed_rar_enhancement,
    response_enhancement,
    simple_mond_enhancement,
)
from .tensor_completion import (
    TENSOR_MODELS,
    axisymmetric_tidal_eigenvalues,
    predict_tensor_acceleration,
)
from .unbounded_running import (
    RUNNING_MODELS,
    TENSOR_RUNNING_MODELS,
    VARIABLE_EXPONENT_DENSITY_MODELS,
    VARIABLE_EXPONENT_MODELS,
    predict_running_acceleration,
)
from .vector_completion import predict_completion_acceleration

M_SUN_G = 1.988409870698051e33
PC_CM = 3.085677581491367e18
MSUN_PC3_TO_G_CM3 = M_SUN_G / PC_CM**3
G_KPC_KM2_S2_PER_MSUN = 4.300917270036279e-6


@dataclass(frozen=True)
class FitResult:
    theta: np.ndarray
    objective: float
    success: bool
    finite: bool
    starts: int
    evaluations: int
    message: str


def hernquist_density_msun_pc3(total_mass_msun, radius_kpc, scale_kpc) -> np.ndarray:
    mass, radius, scale = np.broadcast_arrays(
        np.asarray(total_mass_msun, dtype=float),
        np.asarray(radius_kpc, dtype=float),
        np.asarray(scale_kpc, dtype=float),
    )
    valid = (
        np.isfinite(mass)
        & np.isfinite(radius)
        & np.isfinite(scale)
        & (mass > 0.0)
        & (radius > 0.0)
        & (scale > 0.0)
    )
    density = np.zeros_like(radius)
    density[valid] = (
        mass[valid]
        * scale[valid]
        / (2.0 * math.pi * radius[valid] * np.power(radius[valid] + scale[valid], 3))
        / 1.0e9
    )
    return density


def _mass_function(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    return np.log1p(value) - value / (1.0 + value)


def effective_prediction(
    frame: pd.DataFrame,
    theta: np.ndarray,
    *,
    model: str,
    settings: Mapping[str, float],
    candidate_parameters: np.ndarray | None = None,
    density_geometry: Mapping[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate one fixed force law after applying one galaxy's nuisances."""
    values = np.asarray(theta, dtype=float)
    if values.shape not in {(4,), (6,)}:
        raise ValueError("theta must contain four nuisances or four nuisances plus NFW halo")
    if model == "nfw" and values.shape != (6,):
        raise ValueError("NFW requires V200 and concentration")
    if model != "nfw" and values.shape != (4,):
        raise ValueError("non-NFW models require four nuisance values")

    disk_ml = float(settings["disk_mass_to_light_prior"]) * math.exp(values[0])
    bulge_ml = float(settings["bulge_mass_to_light_prior"]) * math.exp(values[1])
    distance_fractional_error = float(frame["distance_fractional_error"].iloc[0])
    distance_scale = math.exp(float(np.clip(values[2] * distance_fractional_error, -1.5, 1.5)))
    inclination_catalog = float(frame["inclination_catalog_deg"].iloc[0])
    inclination_error = float(frame["inclination_error_deg"].iloc[0])
    inclination_adjusted = float(
        np.clip(inclination_catalog + values[3] * inclination_error, 10.0, 89.5)
    )
    inclination_factor = math.sin(math.radians(inclination_catalog)) / math.sin(
        math.radians(inclination_adjusted)
    )

    radius_catalog = frame["radius_catalog_kpc"].to_numpy(dtype=float)
    radius_adjusted = radius_catalog * distance_scale
    gas_velocity = frame["gas_velocity_component_km_s"].to_numpy(dtype=float)
    disk_velocity = frame["disk_velocity_unit_ml_km_s"].to_numpy(dtype=float)
    bulge_velocity = frame["bulge_velocity_unit_ml_km_s"].to_numpy(dtype=float)
    gas_v2 = np.sign(gas_velocity) * np.square(gas_velocity)
    baryonic_v2 = distance_scale * (
        gas_v2 + disk_ml * np.square(disk_velocity) + bulge_ml * np.square(bulge_velocity)
    )
    baryonic_v2 = np.maximum(baryonic_v2, 1.0e-8)
    g_bar = baryonic_v2 * 1.0e6 / (radius_adjusted * KPC_M)
    rar_enhancement = fixed_rar_enhancement(
        g_bar, float(settings["rar_acceleration_m_s2"])
    )
    rar_same_nuisance = np.sqrt(
        np.maximum(
            g_bar * rar_enhancement * radius_adjusted * KPC_M / 1.0e6,
            1.0e-20,
        )
    )

    density = np.full(len(frame), np.nan)
    coherence = np.full(len(frame), np.nan)
    tensor_tidal_norm = np.full(len(frame), np.nan)
    tensor_projected_availability = np.full(len(frame), np.nan)
    tensor_projected_completion = np.full(len(frame), np.nan)
    running_coordinate = np.full(len(frame), np.nan)
    running_enhancement = np.full(len(frame), np.nan)
    running_directional_availability = np.full(len(frame), np.nan)
    running_effective_exponent = np.full(len(frame), np.nan)
    source_baryonic_mass_solar = np.full(len(frame), np.nan)
    screened_tail_factor = np.full(len(frame), np.nan)
    screened_tail_acceleration_m_s2 = np.full(len(frame), np.nan)
    geometric_models = {
        "candidate",
        "vector_completion",
        "vector_completion_coherence",
    } | TENSOR_MODELS | RUNNING_MODELS
    if model in geometric_models:
        if candidate_parameters is None:
            raise ValueError(f"{model} evaluation requires fixed universal parameters")
        if (
            model
            in (
                {"candidate"}
                | TENSOR_MODELS
                | TENSOR_RUNNING_MODELS
                | VARIABLE_EXPONENT_DENSITY_MODELS
            )
            and density_geometry is None
        ):
            raise ValueError(f"{model} evaluation requires fixed density geometry")
        positive_force = (
            np.square(gas_velocity)
            + disk_ml * np.square(disk_velocity)
            + bulge_ml * np.square(bulge_velocity)
        )
        bulge_force = bulge_ml * np.square(bulge_velocity)
        coherence = np.clip(
            1.0
            - np.divide(
                bulge_force,
                positive_force,
                out=np.zeros_like(bulge_force),
                where=positive_force > 0.0,
            ),
            0.0,
            1.0,
        )

    if (
        model == "candidate"
        or model in TENSOR_MODELS
        or model in TENSOR_RUNNING_MODELS
        or model in VARIABLE_EXPONENT_DENSITY_MODELS
    ):
        disk_half_thickness_pc = (
            float(density_geometry["disk_hz_over_Rdisk"])
            * float(frame["disk_scale_kpc"].iloc[0])
            * distance_scale
            * 1000.0
        )
        disk_density = (
            disk_ml
            * frame["disk_surface_brightness"].to_numpy(dtype=float)
            / (2.0 * disk_half_thickness_pc)
        )
        bulge_mass = (
            float(frame["bulge_luminosity_fit_solar"].fillna(0.0).iloc[0])
            * bulge_ml
            * distance_scale**2
        )
        bulge_scale = (
            float(frame["bulge_scale_fit_kpc"].fillna(0.0).iloc[0]) * distance_scale
        )
        bulge_density = hernquist_density_msun_pc3(
            bulge_mass, radius_adjusted, bulge_scale
        )
        gas_density = np.zeros(len(frame))
        r_hi = float(frame["HI_radius_kpc"].iloc[0])
        if math.isfinite(r_hi) and r_hi > 0.0:
            gas_scale = (
                r_hi
                / float(density_geometry["gas_RHI_divisor"])
                * distance_scale
            )
            gas_mass = (
                1.33
                * float(frame["HI_mass_billion_solar"].iloc[0])
                * 1.0e9
                * distance_scale**2
            )
            gas_surface = (
                gas_mass
                / (2.0 * math.pi * gas_scale**2)
                * np.exp(-radius_adjusted / gas_scale)
            )
            gas_density = (
                gas_surface
                / 1.0e6
                / (
                    2.0
                    * float(density_geometry["gas_hz_over_Rgas"])
                    * gas_scale
                    * 1000.0
                )
            )
        density = np.maximum(
            (disk_density + bulge_density + gas_density) * MSUN_PC3_TO_G_CM3,
            1.0e-35,
        )
        if model == "candidate":
            enhancement = response_enhancement(
                "RAR_sharp_coherence_gated_RG",
                g_bar,
                density,
                radius_adjusted,
                candidate_parameters,
                rar_acceleration_m_s2=float(settings["rar_acceleration_m_s2"]),
                coherence=coherence,
                coherence_gate_power=float(settings["coherence_gate_power"]),
            )
            predicted_v2 = g_bar * enhancement * radius_adjusted * KPC_M / 1.0e6
        elif model in TENSOR_MODELS:
            tidal_eigenvalues = axisymmetric_tidal_eigenvalues(
                g_bar,
                radius_adjusted,
                density,
            )
            completed = predict_tensor_acceleration(
                g_bar,
                tidal_eigenvalues,
                model,
                candidate_parameters,
                direction_components=(1.0, 0.0, 0.0),
            )
            predicted_v2 = (
                completed["predicted_acceleration_m_s2"]
                * radius_adjusted
                * KPC_M
                / 1.0e6
            )
            tensor_tidal_norm = completed["tidal_norm_s2"]
            tensor_projected_availability = completed["projected_availability"]
            tensor_projected_completion = completed["projected_completion_fraction"]
        elif model in TENSOR_RUNNING_MODELS:
            tidal_eigenvalues = axisymmetric_tidal_eigenvalues(
                g_bar,
                radius_adjusted,
                density,
            )
            completed = predict_running_acceleration(
                g_bar,
                radius_adjusted,
                model,
                candidate_parameters,
                tidal_eigenvalues_s2=tidal_eigenvalues,
            )
            predicted_v2 = (
                completed["predicted_acceleration_m_s2"]
                * radius_adjusted
                * KPC_M
                / 1.0e6
            )
            tensor_tidal_norm = np.linalg.norm(tidal_eigenvalues, axis=-1)
            running_coordinate = completed["running_coordinate"]
            running_enhancement = completed["enhancement_relative_to_local_G"]
            running_directional_availability = completed["directional_availability"]
        else:
            completed = predict_running_acceleration(
                g_bar,
                radius_adjusted,
                model,
                candidate_parameters,
                local_density_g_cm3=density,
            )
            predicted_v2 = (
                completed["predicted_acceleration_m_s2"]
                * radius_adjusted
                * KPC_M
                / 1.0e6
            )
            running_coordinate = completed["running_coordinate"]
            running_enhancement = completed["enhancement_relative_to_local_G"]
            running_directional_availability = completed["directional_availability"]
            running_effective_exponent = completed["effective_exponent"]
    elif model in {"vector_completion", "vector_completion_coherence"}:
        if candidate_parameters is None:
            raise ValueError("vector completion requires fixed universal parameters")
        completed = predict_completion_acceleration(
            g_bar,
            radius_adjusted,
            candidate_parameters,
            coherence=coherence if model == "vector_completion_coherence" else None,
        )
        predicted_v2 = (
            completed["predicted_acceleration_m_s2"]
            * radius_adjusted
            * KPC_M
            / 1.0e6
        )
    elif model in PATH_MODELS:
        if candidate_parameters is None:
            raise ValueError("path completion requires fixed universal parameters")
        order = np.argsort(radius_adjusted, kind="stable")
        completed_sorted = path_completion_profile(
            radius_adjusted[order],
            g_bar[order],
            model,
            candidate_parameters,
        )
        completed_acceleration = np.empty_like(g_bar)
        completed_acceleration[order] = completed_sorted["predicted_acceleration_m_s2"]
        predicted_v2 = completed_acceleration * radius_adjusted * KPC_M / 1.0e6
    elif model in MASS_PATH_MODELS:
        if candidate_parameters is None:
            raise ValueError("mass path completion requires fixed universal parameters")
        order = np.argsort(radius_adjusted, kind="stable")
        completed_sorted = mass_path_completion_profile(
            radius_adjusted[order],
            g_bar[order],
            model,
            candidate_parameters,
        )
        completed_acceleration = np.empty_like(g_bar)
        completed_acceleration[order] = completed_sorted["predicted_acceleration_m_s2"]
        predicted_v2 = completed_acceleration * radius_adjusted * KPC_M / 1.0e6
    elif model in RUNNING_MODELS:
        if candidate_parameters is None:
            raise ValueError("running-G model requires fixed universal parameters")
        completed = predict_running_acceleration(
            g_bar,
            radius_adjusted,
            model,
            candidate_parameters,
        )
        predicted_v2 = (
            completed["predicted_acceleration_m_s2"]
            * radius_adjusted
            * KPC_M
            / 1.0e6
        )
        running_coordinate = completed["running_coordinate"]
        running_enhancement = completed["enhancement_relative_to_local_G"]
        running_directional_availability = completed["directional_availability"]
        if model in VARIABLE_EXPONENT_MODELS:
            running_effective_exponent = completed["effective_exponent"]
    elif model == "solar_screened_isothermal":
        required = {
            "screened_tail_parameter",
            "screened_tail_reference_radius_kpc",
            "screened_tail_a0_m_s2",
        }
        missing = required.difference(settings)
        if missing:
            raise ValueError(f"screened-tail settings missing {sorted(missing)}")
        disk_luminosity = float(frame["disk_luminosity_fit_solar"].iloc[0])
        bulge_luminosity = float(
            frame["bulge_luminosity_fit_solar"].fillna(0.0).iloc[0]
        )
        gas_mass = 1.33 * float(frame["HI_mass_billion_solar"].iloc[0]) * 1.0e9
        source_mass = distance_scale**2 * (
            gas_mass + disk_ml * disk_luminosity + bulge_ml * bulge_luminosity
        )
        if not math.isfinite(source_mass) or source_mass <= 0.0:
            raise ValueError("screened-tail source mass must be finite and positive")
        parameter = float(settings["screened_tail_parameter"])
        reference_radius = float(settings["screened_tail_reference_radius_kpc"])
        screen_a0 = float(settings["screened_tail_a0_m_s2"])
        if parameter < 0.0 or reference_radius <= 0.0 or screen_a0 <= 0.0:
            raise ValueError("screened-tail constants must be physical")
        screen = screen_a0 / (screen_a0 + g_bar)
        extra_v2 = (
            parameter
            * G_KPC_KM2_S2_PER_MSUN
            * source_mass
            / reference_radius
            * screen
        )
        predicted_v2 = baryonic_v2 + extra_v2
        source_baryonic_mass_solar.fill(source_mass)
        screened_tail_factor = screen
        screened_tail_acceleration_m_s2 = (
            extra_v2 * 1.0e6 / (radius_adjusted * KPC_M)
        )
    elif model == "rar":
        predicted_v2 = np.square(rar_same_nuisance)
    elif model == "simple_mond":
        enhancement = simple_mond_enhancement(
            g_bar, float(settings["mond_acceleration_m_s2"])
        )
        predicted_v2 = g_bar * enhancement * radius_adjusted * KPC_M / 1.0e6
    elif model == "nfw":
        v200 = math.exp(values[4])
        concentration = math.exp(values[5])
        r200_kpc = v200 / (10.0 * float(settings["hubble_km_s_mpc"]) / 1000.0)
        x = np.maximum(radius_adjusted / r200_kpc, 1.0e-8)
        halo_v2 = (
            v200**2
            * _mass_function(concentration * x)
            / np.maximum(x * _mass_function(np.asarray(concentration)), 1.0e-30)
        )
        predicted_v2 = baryonic_v2 + halo_v2
    else:
        raise ValueError(f"unknown model {model}")

    predicted = np.sqrt(np.maximum(predicted_v2, 1.0e-20))
    observed_adjusted = (
        frame["velocity_observed_catalog_kms"].to_numpy(dtype=float) * inclination_factor
    )
    error_adjusted = (
        frame["velocity_error_catalog_kms"].to_numpy(dtype=float) * inclination_factor
    )
    sigma = np.sqrt(
        np.square(error_adjusted) + float(settings["velocity_error_floor_km_s"]) ** 2
    )
    return {
        "velocity_predicted_km_s": predicted,
        "velocity_RAR_same_nuisance_km_s": rar_same_nuisance,
        "velocity_observed_adjusted_km_s": observed_adjusted,
        "velocity_error_total_km_s": sigma,
        "velocity_predicted_catalog_km_s": predicted / inclination_factor,
        "g_bar_m_s2": g_bar,
        "radius_adjusted_kpc": radius_adjusted,
        "local_density_g_cm3": density,
        "coherence": coherence,
        "tensor_tidal_norm_s2": tensor_tidal_norm,
        "tensor_projected_availability": tensor_projected_availability,
        "tensor_projected_completion_fraction": tensor_projected_completion,
        "running_coordinate": running_coordinate,
        "running_enhancement_relative_to_local_G": running_enhancement,
        "running_directional_availability": running_directional_availability,
        "running_effective_exponent": running_effective_exponent,
        "source_baryonic_mass_solar": source_baryonic_mass_solar,
        "screened_tail_factor": screened_tail_factor,
        "screened_tail_acceleration_m_s2": screened_tail_acceleration_m_s2,
        "disk_mass_to_light": np.full(len(frame), disk_ml),
        "bulge_mass_to_light": np.full(len(frame), bulge_ml),
        "distance_scale": np.full(len(frame), distance_scale),
        "inclination_adjusted_deg": np.full(len(frame), inclination_adjusted),
    }


def negative_log_posterior(
    theta: np.ndarray,
    frame: pd.DataFrame,
    *,
    model: str,
    settings: Mapping[str, float],
    candidate_parameters: np.ndarray | None = None,
    density_geometry: Mapping[str, float] | None = None,
) -> float:
    prediction = effective_prediction(
        frame,
        theta,
        model=model,
        settings=settings,
        candidate_parameters=candidate_parameters,
        density_geometry=density_geometry,
    )
    residual = (
        prediction["velocity_predicted_km_s"]
        - prediction["velocity_observed_adjusted_km_s"]
    )
    sigma = prediction["velocity_error_total_km_s"]
    likelihood = 0.5 * np.sum(np.square(residual / sigma) + 2.0 * np.log(sigma))
    prior = 0.5 * (
        (theta[0] / float(settings["log_mass_to_light_prior_sigma"])) ** 2
        + (theta[1] / float(settings["log_mass_to_light_prior_sigma"])) ** 2
        + theta[2] ** 2
        + theta[3] ** 2
    )
    if model == "nfw":
        prior += 0.5 * (
            (
                (theta[4] - math.log(float(settings["nfw_v200_prior_km_s"])))
                / float(settings["nfw_log_v200_sigma"])
            )
            ** 2
            + (
                (theta[5] - math.log(float(settings["nfw_concentration_prior"])))
                / float(settings["nfw_log_concentration_sigma"])
            )
            ** 2
        )
    result = float(likelihood + prior)
    return result if math.isfinite(result) else 1.0e100


def fit_galaxy(
    training_frame: pd.DataFrame,
    *,
    model: str,
    settings: Mapping[str, float],
    starts: list[np.ndarray],
    bounds: list[tuple[float, float]],
    candidate_parameters: np.ndarray | None = None,
    density_geometry: Mapping[str, float] | None = None,
    max_iterations: int = 1000,
) -> FitResult:
    if training_frame.empty:
        raise ValueError("training_frame cannot be empty")
    best = None
    evaluations = 0
    messages = []
    objective = partial(
        negative_log_posterior,
        frame=training_frame,
        model=model,
        settings=settings,
        candidate_parameters=candidate_parameters,
        density_geometry=density_geometry,
    )
    for start in starts:
        result = minimize(
            objective,
            np.asarray(start, dtype=float),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(max_iterations), "ftol": 1.0e-12, "gtol": 1.0e-8},
        )
        evaluations += int(result.nfev)
        messages.append(str(result.message))
        if np.isfinite(result.fun) and (best is None or result.fun < best.fun):
            best = result
    if best is None:
        dimension = 6 if model == "nfw" else 4
        return FitResult(
            theta=np.full(dimension, np.nan),
            objective=math.inf,
            success=False,
            finite=False,
            starts=len(starts),
            evaluations=evaluations,
            message="; ".join(messages),
        )
    return FitResult(
        theta=np.asarray(best.x, dtype=float),
        objective=float(best.fun),
        success=bool(best.success),
        finite=bool(np.isfinite(best.fun) and np.isfinite(best.x).all()),
        starts=len(starts),
        evaluations=evaluations,
        message=str(best.message),
    )
