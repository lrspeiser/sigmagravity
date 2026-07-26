from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit

from .data import KPC_M, pack_dataset

A0_M_S2 = 1.2e-10
C_M_S = 299_792_458.0
G_SI = 6.67430e-11
M_SUN_KG = 1.98847e30
MODEL_NAMES = (
    "fixed_rar",
    "joint_a0",
    "U0_emond_like",
    "U1_coherence_length",
    "domain_oracle",
)


@dataclass(frozen=True)
class FitResult:
    model: str
    vector: np.ndarray
    parameters: dict[str, float]
    chi2: float
    success: bool
    starts: int


def baryonic_potential_profile(radius_kpc, gbar_m_s2) -> np.ndarray:
    """Potential depth with zero at infinity and a declared point-mass tail."""
    radius = np.asarray(radius_kpc, dtype=float)
    gbar = np.asarray(gbar_m_s2, dtype=float)
    if radius.ndim != 1 or gbar.ndim != 1 or radius.shape != gbar.shape:
        raise ValueError("radius and gbar must be matching one-dimensional arrays")
    if len(radius) < 2:
        raise ValueError("a potential profile requires at least two radii")
    if not np.all(np.isfinite(radius)) or not np.all(np.isfinite(gbar)):
        raise ValueError("radius and gbar must be finite")
    if np.any(radius <= 0.0) or np.any(gbar <= 0.0):
        raise ValueError("radius and gbar must be positive")

    order = np.argsort(radius, kind="stable")
    sorted_radius_m = radius[order] * KPC_M
    sorted_gbar = gbar[order]
    if np.any(np.diff(sorted_radius_m) <= 0.0):
        raise ValueError("radii within a system must be unique")
    segments = 0.5 * (sorted_gbar[:-1] + sorted_gbar[1:]) * np.diff(sorted_radius_m)
    inward_integral = np.zeros_like(sorted_radius_m)
    inward_integral[:-1] = np.cumsum(segments[::-1])[::-1]
    point_mass_tail = sorted_gbar[-1] * sorted_radius_m[-1]
    sorted_potential = inward_integral + point_mass_tail
    potential = np.empty_like(sorted_potential)
    potential[order] = sorted_potential
    return potential


def rar_acceleration(gbar_m_s2, a_eff_m_s2=A0_M_S2) -> np.ndarray:
    gbar = np.asarray(gbar_m_s2, dtype=float)
    a_eff = np.asarray(a_eff_m_s2, dtype=float)
    if np.any(gbar <= 0.0) or np.any(a_eff <= 0.0):
        raise ValueError("gbar and a_eff must be positive")
    root = np.sqrt(gbar / a_eff)
    denominator = np.maximum(-np.expm1(-root), np.finfo(float).tiny)
    return gbar / denominator


def model_bounds(model: str) -> tuple[np.ndarray, np.ndarray]:
    bounds = {
        "fixed_rar": ([], []),
        "joint_a0": ([-12.0], [-8.0]),
        "U0_emond_like": ([0.0, -9.0, 0.1], [2.0, -4.0, 2.0]),
        "U1_coherence_length": ([0.0, 0.05], [3.0, 2.0]),
        "domain_oracle": ([-12.0], [-8.0]),
    }
    if model not in bounds:
        raise ValueError(f"unknown unified model: {model}")
    lower, upper = bounds[model]
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def physical_parameters(model: str, vector) -> dict[str, float]:
    values = np.asarray(vector, dtype=float)
    if model == "fixed_rar":
        return {"a0_m_s2": A0_M_S2}
    if model == "joint_a0":
        return {"a_joint_m_s2": float(10.0 ** values[0])}
    if model == "U0_emond_like":
        return {
            "F": float(10.0 ** values[0]),
            "chi_t": float(10.0 ** values[1]),
            "w_dex": float(values[2]),
        }
    if model == "U1_coherence_length":
        return {"ell_c_kpc": float(10.0 ** values[0]), "q": float(values[1])}
    if model == "domain_oracle":
        return {"cluster_a0_m_s2": float(10.0 ** values[0]), "galaxy_a0_m_s2": A0_M_S2}
    raise ValueError(f"unknown unified model: {model}")


def predict_acceleration(
    model: str,
    gbar_m_s2,
    chi,
    ell_bar_kpc,
    vector=(),
    *,
    domain: str,
) -> np.ndarray:
    gbar = np.asarray(gbar_m_s2, dtype=float)
    chi_values = np.asarray(chi, dtype=float)
    ell_values = np.asarray(ell_bar_kpc, dtype=float)
    parameters = physical_parameters(model, vector)
    if model == "fixed_rar":
        a_eff = A0_M_S2
    elif model == "joint_a0":
        a_eff = parameters["a_joint_m_s2"]
    elif model == "U0_emond_like":
        activation = expit(
            (np.log10(chi_values) - np.log10(parameters["chi_t"]))
            / parameters["w_dex"]
        )
        a_eff = A0_M_S2 * np.exp(np.log(parameters["F"]) * activation)
    elif model == "U1_coherence_length":
        a_eff = A0_M_S2 * (
            1.0 + np.power(ell_values / parameters["ell_c_kpc"], parameters["q"])
        )
    elif model == "domain_oracle":
        if domain not in {"galaxy", "cluster"}:
            raise ValueError(f"unknown domain: {domain}")
        a_eff = A0_M_S2 if domain == "galaxy" else parameters["cluster_a0_m_s2"]
    else:
        raise ValueError(f"unknown unified model: {model}")
    return rar_acceleration(gbar, a_eff)


def _add_field_geometry(frame: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for _, group in frame.groupby("system", sort=True):
        selected = group.sort_values("radius_kpc", kind="stable").copy()
        potential = baryonic_potential_profile(selected["radius_kpc"], selected["gbar_m_s2"])
        selected["phi_bar_m2_s2"] = potential
        selected["chi"] = potential / (C_M_S**2)
        selected["ell_bar_kpc"] = potential / selected["gbar_m_s2"].to_numpy() / KPC_M
        pieces.append(selected)
    return pd.concat(pieces, ignore_index=True)


def load_sparc_acceleration_frame(data_dir: Path) -> pd.DataFrame:
    packed = pack_dataset(Path(data_dir))
    gas_v2 = np.sign(packed.velocity_gas_kms) * packed.velocity_gas_kms**2
    baryonic_v2 = (
        gas_v2
        + 0.5 * packed.velocity_disk_unit_ml_kms**2
        + 0.7 * packed.velocity_bulge_unit_ml_kms**2
    )
    baryonic_v2 = np.maximum(baryonic_v2, 1e-8)
    radius_m = packed.radius_kpc * KPC_M
    frame = pd.DataFrame(
        {
            "domain": "galaxy",
            "system": np.asarray(packed.galaxy_names, dtype=object)[packed.galaxy_index],
            "radius_kpc": packed.radius_kpc,
            "gbar_m_s2": baryonic_v2 * 1e6 / radius_m,
            "observed_velocity_km_s": packed.velocity_observed_kms,
            "velocity_error_km_s": packed.velocity_error_kms,
        }
    )
    frame["observed_g_m_s2"] = (
        frame["observed_velocity_km_s"].to_numpy() ** 2 * 1e6 / radius_m
    )
    return _add_field_geometry(frame)


def load_clash_acceleration_frame(path: Path) -> pd.DataFrame:
    columns = [
        "system",
        "radius_kpc",
        "log_gbar",
        "log_gtot",
        "err_log_gbar",
        "err_log_gtot",
    ]
    frame = pd.read_csv(path, sep=r"\s+", names=columns, comment="#")
    for column in columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if len(frame) != 84 or frame["system"].nunique() != 20:
        raise ValueError("expected 84 rows and 20 systems in the Tian CLASH catalog")
    frame.insert(0, "domain", "cluster")
    frame["gbar_m_s2"] = np.power(10.0, frame["log_gbar"])
    frame["observed_g_m_s2"] = np.power(10.0, frame["log_gtot"])
    return _add_field_geometry(frame)


def assign_system_folds(frame: pd.DataFrame, *, folds: int, seed: int) -> pd.DataFrame:
    if folds < 2:
        raise ValueError("at least two folds are required")
    names = np.asarray(sorted(frame["system"].astype(str).unique()), dtype=object)
    permutation = np.random.default_rng(seed).permutation(names)
    assignment = {str(name): int(index % folds) for index, name in enumerate(permutation)}
    output = frame.copy()
    output["fold"] = output["system"].map(assignment).astype(int)
    return output


def _cluster_model_slope(model: str, frame: pd.DataFrame, vector) -> np.ndarray:
    epsilon = 1e-5
    gbar = frame["gbar_m_s2"].to_numpy(dtype=float)
    common = {
        "model": model,
        "chi": frame["chi"].to_numpy(dtype=float),
        "ell_bar_kpc": frame["ell_bar_kpc"].to_numpy(dtype=float),
        "vector": vector,
        "domain": "cluster",
    }
    upper = predict_acceleration(gbar_m_s2=gbar * np.exp(epsilon), **common)
    lower = predict_acceleration(gbar_m_s2=gbar * np.exp(-epsilon), **common)
    return (np.log(upper) - np.log(lower)) / (2.0 * epsilon)


def standardized_residuals(
    model: str,
    vector,
    galaxy_frame: pd.DataFrame,
    cluster_frame: pd.DataFrame,
    *,
    velocity_error_floor_km_s: float = 5.0,
    cluster_intrinsic_scatter_dex: float = 0.0,
) -> np.ndarray:
    pieces = []
    if len(galaxy_frame):
        galaxy_prediction = predict_acceleration(
            model,
            galaxy_frame["gbar_m_s2"],
            galaxy_frame["chi"],
            galaxy_frame["ell_bar_kpc"],
            vector,
            domain="galaxy",
        )
        velocity_prediction = np.sqrt(
            galaxy_prediction * galaxy_frame["radius_kpc"].to_numpy() * KPC_M
        ) / 1000.0
        velocity_sigma = np.sqrt(
            galaxy_frame["velocity_error_km_s"].to_numpy() ** 2
            + velocity_error_floor_km_s**2
        )
        pieces.append(
            (velocity_prediction - galaxy_frame["observed_velocity_km_s"].to_numpy())
            / velocity_sigma
        )
    if len(cluster_frame):
        cluster_prediction = predict_acceleration(
            model,
            cluster_frame["gbar_m_s2"],
            cluster_frame["chi"],
            cluster_frame["ell_bar_kpc"],
            vector,
            domain="cluster",
        )
        slope = _cluster_model_slope(model, cluster_frame, vector)
        sigma = np.sqrt(
            cluster_frame["err_log_gtot"].to_numpy() ** 2
            + (slope * cluster_frame["err_log_gbar"].to_numpy()) ** 2
            + cluster_intrinsic_scatter_dex**2
        )
        pieces.append((np.log10(cluster_prediction) - cluster_frame["log_gtot"].to_numpy()) / sigma)
    if not pieces:
        return np.asarray([], dtype=float)
    return np.concatenate(pieces)


def fit_unified_model(
    model: str,
    galaxy_train: pd.DataFrame,
    cluster_train: pd.DataFrame,
    *,
    starts: int = 16,
    seed: int = 20260726,
) -> FitResult:
    lower, upper = model_bounds(model)
    if len(lower) == 0:
        residual = standardized_residuals(model, [], galaxy_train, cluster_train)
        return FitResult(model, np.asarray([]), physical_parameters(model, []), float(residual @ residual), True, 0)
    if starts < 1:
        raise ValueError("starts must be positive")
    rng = np.random.default_rng(seed)
    initial = [0.5 * (lower + upper)]
    initial.extend(rng.uniform(lower, upper) for _ in range(starts - 1))
    best = None
    for guess in initial:
        result = least_squares(
            lambda values: standardized_residuals(
                model, values, galaxy_train, cluster_train
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
    return FitResult(
        model=model,
        vector=result.x.copy(),
        parameters=physical_parameters(model, result.x),
        chi2=chi2,
        success=bool(result.success),
        starts=starts,
    )


def prediction_frame(
    model: str,
    vector,
    frame: pd.DataFrame,
    *,
    velocity_error_floor_km_s: float = 5.0,
    cluster_intrinsic_scatter_dex: float = 0.0,
) -> pd.DataFrame:
    if frame["domain"].nunique() != 1:
        raise ValueError("prediction_frame expects one domain at a time")
    domain = str(frame["domain"].iloc[0])
    output = frame.copy()
    predicted = predict_acceleration(
        model,
        output["gbar_m_s2"],
        output["chi"],
        output["ell_bar_kpc"],
        vector,
        domain=domain,
    )
    output["model"] = model
    output["predicted_g_m_s2"] = predicted
    if domain == "galaxy":
        output["predicted_velocity_km_s"] = np.sqrt(
            predicted * output["radius_kpc"].to_numpy() * KPC_M
        ) / 1000.0
        output["sigma"] = np.sqrt(
            output["velocity_error_km_s"].to_numpy() ** 2 + velocity_error_floor_km_s**2
        )
        output["residual"] = (
            output["predicted_velocity_km_s"] - output["observed_velocity_km_s"]
        )
    else:
        output["predicted_log_gtot"] = np.log10(predicted)
        slope = _cluster_model_slope(model, output, vector)
        output["sigma"] = np.sqrt(
            output["err_log_gtot"].to_numpy() ** 2
            + (slope * output["err_log_gbar"].to_numpy()) ** 2
            + cluster_intrinsic_scatter_dex**2
        )
        output["residual"] = output["predicted_log_gtot"] - output["log_gtot"]
        radius_m = output["radius_kpc"].to_numpy() * KPC_M
        output["predicted_lensing_mass_msun"] = predicted * radius_m**2 / G_SI / M_SUN_KG
        output["observed_lensing_mass_msun"] = (
            output["observed_g_m_s2"].to_numpy() * radius_m**2 / G_SI / M_SUN_KG
        )
    output["standardized_residual"] = output["residual"] / output["sigma"]
    output["chi2_term"] = output["standardized_residual"] ** 2
    return output
