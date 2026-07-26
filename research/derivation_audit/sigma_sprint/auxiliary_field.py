"""Action-derived auxiliary-order-field diagnostics for CLASH profiles.

The deliberately simple candidate order-field action is

    S_B = integral [ell^2 |grad B|^2 / 2 + B^2 / 2 - beta s B] d^3x,

When combined with the QUMOND action, variation of B also supplies the
mandatory source ``eta Q_B``, where eta contains the unknown order-field
stiffness.  The full diagnostic equation is therefore

    B - ell^2 laplacian(B) = beta s_rho + eta Q_B + gamma C_kin.

This is a bounded diagnostic, not a claimed microscopic derivation.  All
couplings and the correlation length are global; no system-type switches or
per-cluster parameters are permitted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from .cluster_audit import _prediction_log, residual_sigma, score_prediction
from .model import DEFAULT_G_DAGGER, G_SI, q_B

KPC_M = 3.085677581491367e19
MSUN_KG = 1.98847e30


def baryon_density_proxy(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Estimate spherical rho_b from independent g_bar(r), with no g_tot use."""
    ordered = group.sort_values("radius_kpc")
    radius = ordered["radius_kpc"].to_numpy(dtype=float)
    radius_m = radius * KPC_M
    mass_kg = ordered["gbar"].to_numpy(dtype=float) * radius_m**2 / G_SI
    mass_kg = np.maximum.accumulate(mass_kg)
    derivative = np.gradient(mass_kg, radius_m, edge_order=1)
    density_si = np.maximum(derivative / (4.0 * np.pi * radius_m**2), 0.0)
    density = density_si * KPC_M**3 / MSUN_KG
    return radius, density


def density_reference(frame: pd.DataFrame) -> float:
    values = []
    for _, group in frame.groupby("cluster"):
        _, density = baryon_density_proxy(group)
        values.extend(density[density > 0])
    if not values:
        raise ValueError("no positive baryon-density estimates")
    return float(np.median(values))


def gravitational_source_proxy(group: pd.DataFrame) -> np.ndarray:
    """Return Q_B=4[z^(1/4)-atan(z^(1/4))] from independent g_bar."""
    z = (group["gbar"].to_numpy(dtype=float) / DEFAULT_G_DAGGER) ** 2
    return q_B(z)


def gravitational_source_reference(frame: pd.DataFrame) -> float:
    values = np.concatenate(
        [gravitational_source_proxy(group) for _, group in frame.groupby("cluster")]
    )
    positive = values[values > 0]
    if not len(positive):
        raise ValueError("no positive Q_B source values")
    return float(np.median(positive))


def solve_spherical_helmholtz(
    sample_radii_kpc,
    source_radii_kpc,
    source_values,
    ell_kpc: float,
    *,
    grid_size: int = 160,
) -> np.ndarray:
    """Solve (1-ell^2 radial_laplacian)u=s with regular/Neumann boundaries."""
    sample = np.asarray(sample_radii_kpc, dtype=float)
    source_r = np.asarray(source_radii_kpc, dtype=float)
    source = np.asarray(source_values, dtype=float)
    if ell_kpc <= 0 or np.any(sample < 0) or np.any(source_r <= 0):
        raise ValueError("ell and source radii must be positive")
    outer = max(float(np.max(sample)), float(np.max(source_r))) * 1.5
    radii = np.linspace(0.0, outer, grid_size)
    dr = radii[1] - radii[0]
    interpolated = np.interp(radii, source_r, source, left=source[0], right=0.0)
    operator = lil_matrix((grid_size, grid_size), dtype=float)
    alpha = (ell_kpc / dr) ** 2
    operator[0, 0] = 1.0 + 6.0 * alpha
    operator[0, 1] = -6.0 * alpha
    for index in range(1, grid_size - 1):
        r = radii[index]
        lower_laplace = 1.0 / dr**2 - 1.0 / (r * dr)
        center_laplace = -2.0 / dr**2
        upper_laplace = 1.0 / dr**2 + 1.0 / (r * dr)
        operator[index, index - 1] = -(ell_kpc**2) * lower_laplace
        operator[index, index] = 1.0 - (ell_kpc**2) * center_laplace
        operator[index, index + 1] = -(ell_kpc**2) * upper_laplace
    # Natural zero-flux boundary from the action.
    operator[-1, -2] = -1.0
    operator[-1, -1] = 1.0
    interpolated[-1] = 0.0
    solution = spsolve(operator.tocsr(), interpolated)
    return np.interp(sample, radii, solution)


def predict_density_field_B(
    frame: pd.DataFrame, beta: float, ell_kpc: float, rho_reference: float
) -> np.ndarray:
    output = pd.Series(index=frame.index, dtype=float)
    for _, group in frame.groupby("cluster"):
        radius, density = baryon_density_proxy(group)
        source = density / rho_reference
        unit_field = solve_spherical_helmholtz(
            group["radius_kpc"].to_numpy(), radius, source, ell_kpc
        )
        output.loc[group.index] = beta * unit_field
    return output.loc[frame.index].to_numpy()


def predict_coupled_action_B(
    frame: pd.DataFrame,
    beta_density: float,
    eta_gravity: float,
    ell_kpc: float,
    rho_reference: float,
    qB_reference: float,
) -> np.ndarray:
    """Solve the B equation including its mandatory QUMOND-action source."""
    output = pd.Series(index=frame.index, dtype=float)
    for _, group in frame.groupby("cluster"):
        radius, density = baryon_density_proxy(group)
        ordered = group.sort_values("radius_kpc")
        q_source = gravitational_source_proxy(ordered) / qB_reference
        density_unit = solve_spherical_helmholtz(
            ordered["radius_kpc"].to_numpy(),
            radius,
            density / rho_reference,
            ell_kpc,
        )
        gravity_unit = solve_spherical_helmholtz(
            ordered["radius_kpc"].to_numpy(),
            ordered["radius_kpc"].to_numpy(),
            q_source,
            ell_kpc,
        )
        output.loc[ordered.index] = beta_density * density_unit + eta_gravity * gravity_unit
    return output.loc[frame.index].to_numpy()


def fit_density_field(frame: pd.DataFrame, rho_reference: float | None = None) -> dict:
    reference = density_reference(frame) if rho_reference is None else float(rho_reference)

    def objective(parameters):
        beta, ell = np.exp(parameters)
        B = predict_density_field_B(frame, beta, ell, reference)
        residual = _prediction_log(frame, B) - frame["log_gtot"].to_numpy()
        return residual / residual_sigma(frame, B)

    fit = least_squares(
        objective,
        np.log([5.0, 100.0]),
        bounds=(np.log([1e-3, 1.0]), np.log([1e3, 2000.0])),
        max_nfev=80,
    )
    beta, ell = np.exp(fit.x)
    B = predict_density_field_B(frame, beta, ell, reference)
    score, _ = score_prediction(frame, B)
    return {
        "beta": float(beta),
        "ell_kpc": float(ell),
        "rho_reference_msun_kpc3": reference,
        "chi2": score.chi2,
        "dof": max(1, len(frame) - 2),
        "rms_dex": score.rms_dex,
        "median_abs_dex": score.median_abs_dex,
    }


def fit_coupled_action_field(
    frame: pd.DataFrame,
    rho_reference: float | None = None,
    qB_reference: float | None = None,
) -> dict:
    rho_ref = density_reference(frame) if rho_reference is None else float(rho_reference)
    q_ref = (
        gravitational_source_reference(frame) if qB_reference is None else float(qB_reference)
    )

    def objective(parameters):
        beta, eta, ell = np.exp(parameters)
        B = predict_coupled_action_B(frame, beta, eta, ell, rho_ref, q_ref)
        residual = _prediction_log(frame, B) - frame["log_gtot"].to_numpy()
        return residual / residual_sigma(frame, B)

    fit = least_squares(
        objective,
        np.log([0.1, 5.0, 100.0]),
        bounds=(np.log([1e-6, 1e-6, 1.0]), np.log([1e3, 1e3, 2000.0])),
        max_nfev=100,
    )
    beta, eta, ell = np.exp(fit.x)
    B = predict_coupled_action_B(frame, beta, eta, ell, rho_ref, q_ref)
    score, _ = score_prediction(frame, B)
    return {
        "beta_density": float(beta),
        "eta_QB": float(eta),
        "ell_kpc": float(ell),
        "rho_reference_msun_kpc3": rho_ref,
        "qB_reference": q_ref,
        "chi2": score.chi2,
        "dof": max(1, len(frame) - 3),
        "rms_dex": score.rms_dex,
        "median_abs_dex": score.median_abs_dex,
    }


def leave_one_cluster_out_density_field(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    fits = []
    for held_out in sorted(frame["cluster"].unique()):
        train = frame[frame["cluster"] != held_out]
        test = frame[frame["cluster"] == held_out]
        reference = density_reference(train)
        fit = fit_density_field(train, reference)
        B = predict_density_field_B(test, fit["beta"], fit["ell_kpc"], reference)
        _, scored = score_prediction(test, B)
        scored["model"] = "density_helmholtz"
        scored["held_out_cluster"] = held_out
        rows.append(scored)
        fits.append({"held_out_cluster": held_out, **fit})
    predictions = pd.concat(rows, ignore_index=True)
    residual = predictions["residual_dex"].to_numpy()
    return predictions, {
        "model": "density_helmholtz",
        "action": "1/2 ell^2 |grad B|^2 + 1/2 B^2 - beta rho_b B",
        "n_points": int(len(predictions)),
        "n_clusters": int(predictions["cluster"].nunique()),
        "rms_dex": float(np.sqrt(np.mean(residual**2))),
        "median_abs_dex": float(np.median(np.abs(residual))),
        "mean_residual_dex": float(np.mean(residual)),
        "fits": fits,
    }


def leave_one_cluster_out_coupled_action(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    fits = []
    for held_out in sorted(frame["cluster"].unique()):
        train = frame[frame["cluster"] != held_out]
        test = frame[frame["cluster"] == held_out]
        rho_ref = density_reference(train)
        q_ref = gravitational_source_reference(train)
        fit = fit_coupled_action_field(train, rho_ref, q_ref)
        B = predict_coupled_action_B(
            test,
            fit["beta_density"],
            fit["eta_QB"],
            fit["ell_kpc"],
            rho_ref,
            q_ref,
        )
        _, scored = score_prediction(test, B)
        scored["model"] = "coupled_action_density_plus_QB"
        scored["held_out_cluster"] = held_out
        rows.append(scored)
        fits.append({"held_out_cluster": held_out, **fit})
    predictions = pd.concat(rows, ignore_index=True)
    residual = predictions["residual_dex"].to_numpy()
    return predictions, {
        "model": "coupled_action_density_plus_QB",
        "action_equation": (
            "B - ell^2 laplacian(B) = beta rho_b/rho_ref + eta Q_B/QB_ref; C_kin=0"
        ),
        "n_points": int(len(predictions)),
        "n_clusters": int(predictions["cluster"].nunique()),
        "rms_dex": float(np.sqrt(np.mean(residual**2))),
        "median_abs_dex": float(np.median(np.abs(residual))),
        "mean_residual_dex": float(np.mean(residual)),
        "fits": fits,
    }


def cluster_coherence_gate(frame: pd.DataFrame) -> dict:
    """Evaluate the operational C_kin implication for dispersion support."""
    B = np.zeros(len(frame))
    score, rows = score_prediction(frame, B)
    return {
        "assumed_streaming_velocity": 0.0,
        "operational_C_kin": 0.0,
        "submitted_cluster_C": 1.0,
        "newtonian_limit_rms_dex": score.rms_dex,
        "newtonian_limit_mean_residual_dex": float(rows["residual_dex"].mean()),
        "conclusion": (
            "A dispersion-supported cluster has C_kin approximately zero, so B=A*C "
            "removes the enhancement. This conflicts with the submitted C=1 assignment."
        ),
        "density_plus_coherence_identifiable": False,
        "identifiability_reason": (
            "A cluster sample with C_kin fixed near zero has a zero Jacobian column for "
            "the coherence coupling."
        ),
    }


def euler_residual_spherical(radii_kpc, B, source, ell_kpc, beta) -> np.ndarray:
    """Finite-difference Euler residual used in action-variation tests."""
    r = np.asarray(radii_kpc, dtype=float)
    field = np.asarray(B, dtype=float)
    src = np.asarray(source, dtype=float)
    derivative = np.gradient(field, r, edge_order=2)
    second_derivative = np.gradient(derivative, r, edge_order=2)
    laplacian = second_derivative + 2.0 * derivative / r
    return field - ell_kpc**2 * laplacian - beta * src
