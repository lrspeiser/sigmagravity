from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit

from .constitutive import simple_mu_acceleration, standard_mu_acceleration
from .data import KPC_M
from .unified import A0_M_S2

H7A_MODEL_NAME = "H7a_simple_mu_potential"
H7S_MODEL_NAME = "H7s_standard_mu_potential"


@dataclass(frozen=True)
class TheoryFitResult:
    vector: np.ndarray
    parameters: dict[str, float]
    chi2: float
    success: bool
    starts: int


def h7a_bounds() -> tuple[np.ndarray, np.ndarray]:
    return np.asarray([0.0, -9.0, 0.1]), np.asarray([2.0, -4.0, 2.0])


def h7a_parameters(vector) -> dict[str, float]:
    values = np.asarray(vector, dtype=float)
    if values.shape != (3,):
        raise ValueError("H7a requires log10(F), log10(chi_t), and w_dex")
    return {
        "F": float(10.0 ** values[0]),
        "chi_t": float(10.0 ** values[1]),
        "w_dex": float(values[2]),
    }


def h7a_acceleration(gbar_m_s2, chi, vector) -> np.ndarray:
    """Quasistatic acceleration for the action-derived simple-mu closure."""
    gbar = np.asarray(gbar_m_s2, dtype=float)
    chi_values = np.asarray(chi, dtype=float)
    if gbar.shape != chi_values.shape:
        raise ValueError("gbar and chi must have matching shapes")
    if np.any(gbar <= 0.0) or np.any(chi_values <= 0.0):
        raise ValueError("gbar and chi must be positive")
    parameters = h7a_parameters(vector)
    activation = expit(
        (np.log10(chi_values) - np.log10(parameters["chi_t"]))
        / parameters["w_dex"]
    )
    a_x = A0_M_S2 * np.exp(np.log(parameters["F"]) * activation)
    return simple_mu_acceleration(gbar, a_x)


def h7s_acceleration(gbar_m_s2, chi, vector) -> np.ndarray:
    """Quasistatic acceleration for the action-derived standard-mu closure."""
    gbar = np.asarray(gbar_m_s2, dtype=float)
    chi_values = np.asarray(chi, dtype=float)
    if gbar.shape != chi_values.shape:
        raise ValueError("gbar and chi must have matching shapes")
    if np.any(gbar <= 0.0) or np.any(chi_values <= 0.0):
        raise ValueError("gbar and chi must be positive")
    parameters = h7a_parameters(vector)
    activation = expit(
        (np.log10(chi_values) - np.log10(parameters["chi_t"]))
        / parameters["w_dex"]
    )
    a_x = A0_M_S2 * np.exp(np.log(parameters["F"]) * activation)
    return standard_mu_acceleration(gbar, a_x)


def _cluster_slope(
    frame: pd.DataFrame, vector, *, acceleration_function=h7a_acceleration
) -> np.ndarray:
    epsilon = 1e-5
    gbar = frame["gbar_m_s2"].to_numpy(dtype=float)
    chi = frame["chi"].to_numpy(dtype=float)
    upper = acceleration_function(gbar * np.exp(epsilon), chi, vector)
    lower = acceleration_function(gbar * np.exp(-epsilon), chi, vector)
    return (np.log(upper) - np.log(lower)) / (2.0 * epsilon)


def h7a_standardized_residuals(
    vector,
    galaxy_frame: pd.DataFrame,
    cluster_frame: pd.DataFrame,
    *,
    velocity_error_floor_km_s: float = 5.0,
    cluster_intrinsic_scatter_dex: float = 0.0,
) -> np.ndarray:
    pieces = []
    if len(galaxy_frame):
        predicted = h7a_acceleration(
            galaxy_frame["gbar_m_s2"], galaxy_frame["chi"], vector
        )
        velocity = np.sqrt(
            predicted * galaxy_frame["radius_kpc"].to_numpy(dtype=float) * KPC_M
        ) / 1000.0
        sigma = np.sqrt(
            np.square(galaxy_frame["velocity_error_km_s"].to_numpy(dtype=float))
            + velocity_error_floor_km_s**2
        )
        pieces.append(
            (velocity - galaxy_frame["observed_velocity_km_s"].to_numpy(dtype=float))
            / sigma
        )
    if len(cluster_frame):
        predicted = h7a_acceleration(
            cluster_frame["gbar_m_s2"], cluster_frame["chi"], vector
        )
        slope = _cluster_slope(cluster_frame, vector)
        sigma = np.sqrt(
            np.square(cluster_frame["err_log_gtot"].to_numpy(dtype=float))
            + np.square(slope * cluster_frame["err_log_gbar"].to_numpy(dtype=float))
            + cluster_intrinsic_scatter_dex**2
        )
        pieces.append(
            (np.log10(predicted) - cluster_frame["log_gtot"].to_numpy(dtype=float))
            / sigma
        )
    if not pieces:
        return np.asarray([], dtype=float)
    return np.concatenate(pieces)


def fit_h7a(
    galaxy_train: pd.DataFrame,
    cluster_train: pd.DataFrame,
    *,
    starts: int = 16,
    seed: int = 20260726,
) -> TheoryFitResult:
    if starts < 1:
        raise ValueError("starts must be positive")
    lower, upper = h7a_bounds()
    rng = np.random.default_rng(seed)
    initial = [0.5 * (lower + upper)]
    initial.extend(rng.uniform(lower, upper) for _ in range(starts - 1))
    best = None
    for guess in initial:
        result = least_squares(
            lambda values: h7a_standardized_residuals(
                values, galaxy_train, cluster_train
            ),
            guess,
            bounds=(lower, upper),
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
            max_nfev=3000,
        )
        chi2 = float(result.fun @ result.fun)
        if best is None or chi2 < best[0]:
            best = (chi2, result)
    assert best is not None
    chi2, result = best
    return TheoryFitResult(
        vector=result.x.copy(),
        parameters=h7a_parameters(result.x),
        chi2=chi2,
        success=bool(result.success),
        starts=starts,
    )


def h7s_standardized_residuals(
    vector,
    galaxy_frame: pd.DataFrame,
    cluster_frame: pd.DataFrame,
    *,
    velocity_error_floor_km_s: float = 5.0,
    cluster_intrinsic_scatter_dex: float = 0.0,
) -> np.ndarray:
    pieces = []
    if len(galaxy_frame):
        predicted = h7s_acceleration(
            galaxy_frame["gbar_m_s2"], galaxy_frame["chi"], vector
        )
        velocity = np.sqrt(
            predicted * galaxy_frame["radius_kpc"].to_numpy(dtype=float) * KPC_M
        ) / 1000.0
        sigma = np.sqrt(
            np.square(galaxy_frame["velocity_error_km_s"].to_numpy(dtype=float))
            + velocity_error_floor_km_s**2
        )
        pieces.append(
            (velocity - galaxy_frame["observed_velocity_km_s"].to_numpy(dtype=float))
            / sigma
        )
    if len(cluster_frame):
        predicted = h7s_acceleration(
            cluster_frame["gbar_m_s2"], cluster_frame["chi"], vector
        )
        slope = _cluster_slope(
            cluster_frame, vector, acceleration_function=h7s_acceleration
        )
        sigma = np.sqrt(
            np.square(cluster_frame["err_log_gtot"].to_numpy(dtype=float))
            + np.square(slope * cluster_frame["err_log_gbar"].to_numpy(dtype=float))
            + cluster_intrinsic_scatter_dex**2
        )
        pieces.append(
            (np.log10(predicted) - cluster_frame["log_gtot"].to_numpy(dtype=float))
            / sigma
        )
    if not pieces:
        return np.asarray([], dtype=float)
    return np.concatenate(pieces)


def fit_h7s(
    galaxy_train: pd.DataFrame,
    cluster_train: pd.DataFrame,
    *,
    starts: int = 16,
    seed: int = 20260726,
) -> TheoryFitResult:
    if starts < 1:
        raise ValueError("starts must be positive")
    lower, upper = h7a_bounds()
    rng = np.random.default_rng(seed)
    initial = [0.5 * (lower + upper)]
    initial.extend(rng.uniform(lower, upper) for _ in range(starts - 1))
    best = None
    for guess in initial:
        result = least_squares(
            lambda values: h7s_standardized_residuals(
                values, galaxy_train, cluster_train
            ),
            guess,
            bounds=(lower, upper),
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
            max_nfev=3000,
        )
        chi2 = float(result.fun @ result.fun)
        if best is None or chi2 < best[0]:
            best = (chi2, result)
    assert best is not None
    chi2, result = best
    return TheoryFitResult(
        vector=result.x.copy(),
        parameters=h7a_parameters(result.x),
        chi2=chi2,
        success=bool(result.success),
        starts=starts,
    )


def h7a_prediction_frame(
    vector,
    frame: pd.DataFrame,
    *,
    velocity_error_floor_km_s: float = 5.0,
    cluster_intrinsic_scatter_dex: float = 0.0,
) -> pd.DataFrame:
    if frame["domain"].nunique() != 1:
        raise ValueError("prediction frame must contain exactly one domain")
    output = frame.copy()
    domain = str(output["domain"].iloc[0])
    predicted = h7a_acceleration(output["gbar_m_s2"], output["chi"], vector)
    output["model"] = H7A_MODEL_NAME
    output["predicted_g_m_s2"] = predicted
    if domain == "galaxy":
        output["predicted_velocity_km_s"] = np.sqrt(
            predicted * output["radius_kpc"].to_numpy(dtype=float) * KPC_M
        ) / 1000.0
        output["sigma"] = np.sqrt(
            np.square(output["velocity_error_km_s"].to_numpy(dtype=float))
            + velocity_error_floor_km_s**2
        )
        output["residual"] = (
            output["predicted_velocity_km_s"]
            - output["observed_velocity_km_s"].to_numpy(dtype=float)
        )
    else:
        output["predicted_log_gtot"] = np.log10(predicted)
        slope = _cluster_slope(output, vector)
        output["sigma"] = np.sqrt(
            np.square(output["err_log_gtot"].to_numpy(dtype=float))
            + np.square(slope * output["err_log_gbar"].to_numpy(dtype=float))
            + cluster_intrinsic_scatter_dex**2
        )
        output["residual"] = output["predicted_log_gtot"] - output["log_gtot"]
    output["standardized_residual"] = output["residual"] / output["sigma"]
    output["chi2_term"] = np.square(output["standardized_residual"])
    return output


def h7s_prediction_frame(
    vector,
    frame: pd.DataFrame,
    *,
    velocity_error_floor_km_s: float = 5.0,
    cluster_intrinsic_scatter_dex: float = 0.0,
) -> pd.DataFrame:
    if frame["domain"].nunique() != 1:
        raise ValueError("prediction frame must contain exactly one domain")
    output = frame.copy()
    domain = str(output["domain"].iloc[0])
    predicted = h7s_acceleration(output["gbar_m_s2"], output["chi"], vector)
    output["model"] = H7S_MODEL_NAME
    output["predicted_g_m_s2"] = predicted
    if domain == "galaxy":
        output["predicted_velocity_km_s"] = np.sqrt(
            predicted * output["radius_kpc"].to_numpy(dtype=float) * KPC_M
        ) / 1000.0
        output["sigma"] = np.sqrt(
            np.square(output["velocity_error_km_s"].to_numpy(dtype=float))
            + velocity_error_floor_km_s**2
        )
        output["residual"] = (
            output["predicted_velocity_km_s"]
            - output["observed_velocity_km_s"].to_numpy(dtype=float)
        )
    else:
        output["predicted_log_gtot"] = np.log10(predicted)
        slope = _cluster_slope(output, vector, acceleration_function=h7s_acceleration)
        output["sigma"] = np.sqrt(
            np.square(output["err_log_gtot"].to_numpy(dtype=float))
            + np.square(slope * output["err_log_gbar"].to_numpy(dtype=float))
            + cluster_intrinsic_scatter_dex**2
        )
        output["residual"] = output["predicted_log_gtot"] - output["log_gtot"]
    output["standardized_residual"] = output["residual"] / output["sigma"]
    output["chi2_term"] = np.square(output["standardized_residual"])
    return output
