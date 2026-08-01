#!/usr/bin/env python3
"""Run the frozen raw-image RX J2129 gravity-law comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Util import param_util
from scipy.optimize import least_squares, root

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.phenomenology import (
    fixed_rar_enhancement,
    response_enhancement,
    simple_mond_enhancement,
)
from voidscreen.raw_lensing import (
    RadialDeflectionField,
    finite_ratio_or_none,
    loglog_interpolate_with_tails,
    pseudo_elliptical_deflection,
    shear_deflection,
    spherical_deflection_radians,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    labels: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    initial: np.ndarray


FIXED_LABELS = (
    "axis_ratio_q",
    "position_angle_phi_radian",
    "center_x_arcsec",
    "center_y_arcsec",
    "external_shear_gamma1",
    "external_shear_gamma2",
)
FIXED_LOWER = np.array([0.55, -np.pi / 2, -3.0, -3.0, -0.25, -0.25])
FIXED_UPPER = np.array([1.0, np.pi / 2, 3.0, 3.0, 0.25, 0.25])
FIXED_INITIAL = np.array([0.8, 0.0, 0.0, 0.0, 0.0, 0.0])
HALO_LABELS = (
    "theta_E_ref_arcsec",
    "axis_ratio_q",
    "position_angle_phi_radian",
    "log10_core_arcsec",
    "center_x_arcsec",
    "center_y_arcsec",
    "external_shear_gamma1",
    "external_shear_gamma2",
)
HALO_LOWER = np.array([1.0, 0.4, -np.pi / 2, -1.3, -5.0, -5.0, -0.25, -0.25])
HALO_UPPER = np.array([40.0, 1.0, np.pi / 2, 1.4, 5.0, 5.0, 0.25, 0.25])
HALO_INITIAL = np.array([15.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def spec_for(model: str) -> ModelSpec:
    if model == "GR_plus_cluster_halo":
        return ModelSpec(model, HALO_LABELS, HALO_LOWER, HALO_UPPER, HALO_INITIAL)
    return ModelSpec(model, FIXED_LABELS, FIXED_LOWER, FIXED_UPPER, FIXED_INITIAL)


def load_images(protocol: dict) -> pd.DataFrame:
    settings = protocol["raw_lensing_inputs"]
    catalog = pd.read_csv(ROOT / settings["image_catalog"])
    selected = catalog[
        (catalog["system"] == settings["system_exact_name"])
        & catalog["metric_neutral_likelihood_row"].astype(bool)
    ].copy()
    selected["image_id"] = selected["image_id"].astype(str)
    selected["source_family"] = selected["family_id"].astype(int)
    selected["source_redshift"] = selected["spectroscopic_redshift"].astype(float)
    geometry = protocol["cosmology_and_coordinates"]
    cosine = np.cos(np.deg2rad(float(geometry["center_dec_deg"])))
    selected["x_arcsec"] = (
        (selected["ra_deg"].astype(float) - float(geometry["center_ra_deg"]))
        * 3600.0
        * cosine
    )
    selected["y_arcsec"] = (
        selected["dec_deg"].astype(float) - float(geometry["center_dec_deg"])
    ) * 3600.0
    selected["radius_arcsec"] = np.hypot(selected["x_arcsec"], selected["y_arcsec"])
    selected = selected.sort_values(["source_family", "image_id"]).reset_index(drop=True)
    if len(selected) != settings["images"]:
        raise RuntimeError("raw image count changed")
    if selected["source_family"].nunique() != settings["source_families"]:
        raise RuntimeError("source-family count changed")
    sigma = selected["position_sigma_axis_1_arcsec"].to_numpy(float)
    if not np.allclose(sigma, float(settings["position_sigma_arcsec_per_coordinate"])):
        raise RuntimeError("coordinate-error model changed")
    return selected


def load_baryonic_anchors(protocol: dict) -> pd.DataFrame:
    settings = protocol["baryonic_inputs"]
    tian = pd.read_csv(
        ROOT / settings["radial_profile"],
        sep=r"\s+",
        names=["system", "radius_kpc", "log_gbar", "log_gobs", "err_gbar", "err_gobs"],
    )
    tian = tian[tian["system"] == settings["system_label"]].copy()
    sample = pd.read_csv(ROOT / settings["density_profile"])
    density = (
        sample[(sample["domain"] == "cluster") & (sample["system"] == settings["system_label"])]
        .sort_values("radius_kpc")
        .drop_duplicates("radius_kpc")[["radius_kpc", "local_density_g_cm3"]]
    )
    expected = np.asarray(settings["baryonic_acceleration_radii_kpc"], dtype=float)
    density_expected = np.asarray(settings["local_density_radii_kpc"], dtype=float)
    if not np.allclose(tian["radius_kpc"].to_numpy(float), expected):
        raise RuntimeError("baryonic anchor radii changed")
    if not np.allclose(density["radius_kpc"].to_numpy(float), density_expected):
        raise RuntimeError("density anchor radii changed")
    tian["local_density_g_cm3"] = loglog_interpolate_with_tails(
        tian["radius_kpc"].to_numpy(float),
        density["radius_kpc"].to_numpy(float),
        density["local_density_g_cm3"].to_numpy(float),
    )
    return tian.sort_values("radius_kpc").reset_index(drop=True)


def acceleration_curve(
    model: str,
    radius_kpc: np.ndarray,
    anchors: pd.DataFrame,
    protocol: dict,
    *,
    gbar_shift_dex: float = 0.0,
    density_shift_dex: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    anchor_radius = anchors["radius_kpc"].to_numpy(float)
    anchor_gbar = np.power(10.0, anchors["log_gbar"].to_numpy(float) + gbar_shift_dex)
    anchor_density = (
        anchors["local_density_g_cm3"].to_numpy(float) * np.power(10.0, density_shift_dex)
    )
    gbar = loglog_interpolate_with_tails(
        radius_kpc, anchor_radius, anchor_gbar, outer_slope=-2.0
    )
    density = loglog_interpolate_with_tails(radius_kpc, anchor_radius, anchor_density)
    radial = protocol["radial_models"]
    if model == "baryons_GR" or model == "GR_plus_cluster_halo":
        enhancement = np.ones_like(gbar)
    elif model == "fixed_simple_MOND":
        enhancement = simple_mond_enhancement(
            gbar, float(radial[model]["a0_m_s2"])
        )
    elif model == "cluster_retuned_RAR_diagnostic":
        enhancement = fixed_rar_enhancement(
            gbar, float(radial[model]["g_dagger_m_s2"])
        )
    elif model == "locked_universal_candidate":
        settings = radial[model]
        enhancement = response_enhancement(
            "RAR_sharp_coherence_gated_RG",
            gbar,
            density,
            radius_kpc,
            [
                settings["epsilon_0"],
                settings["log10_rho_c_g_cm3"],
                settings["Q"],
            ],
            rar_acceleration_m_s2=float(settings["g_dagger_m_s2"]),
            coherence=float(settings["cluster_coherence"]),
            coherence_gate_power=float(settings["coherence_gate_power"]),
        )
    else:
        raise ValueError(model)
    return gbar, density, gbar * enhancement


def build_field(
    model: str,
    anchors: pd.DataFrame,
    protocol: dict,
    *,
    gbar_shift_dex: float = 0.0,
    density_shift_dex: float = 0.0,
) -> tuple[RadialDeflectionField, pd.DataFrame]:
    radius_grid = np.geomspace(0.1, 1.0e6, 4096)
    gbar, density, acceleration = acceleration_curve(
        model,
        radius_grid,
        anchors,
        protocol,
        gbar_shift_dex=gbar_shift_dex,
        density_shift_dex=density_shift_dex,
    )

    def lookup(radius):
        return np.exp(np.interp(np.log(radius), np.log(radius_grid), np.log(acceleration)))

    impact_arcsec = np.geomspace(0.05, 500.0, 700)
    impact_kpc = (
        impact_arcsec
        * float(protocol["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
    )
    physical_alpha = spherical_deflection_radians(
        impact_kpc,
        lookup,
        maximum_radius_kpc=1.0e6,
        integration_points=800,
    )
    field = RadialDeflectionField(impact_arcsec, physical_alpha)
    sampled_radius = np.geomspace(1.0, 1000.0, 240)
    sampled_gbar, sampled_density, sampled_g = acceleration_curve(
        model,
        sampled_radius,
        anchors,
        protocol,
        gbar_shift_dex=gbar_shift_dex,
        density_shift_dex=density_shift_dex,
    )
    sampled_alpha = field.reduced_alpha_arcsec(
        sampled_radius
        / float(protocol["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"]),
        1.0,
    )
    table = pd.DataFrame(
        {
            "model": model,
            "gbar_shift_dex": gbar_shift_dex,
            "density_shift_dex": density_shift_dex,
            "radius_kpc": sampled_radius,
            "gbar_m_s2": sampled_gbar,
            "local_density_g_cm3": sampled_density,
            "effective_acceleration_m_s2": sampled_g,
            "physical_deflection_arcsec_before_distance_ratio": sampled_alpha,
        }
    )
    return field, table


class RawLens:
    def __init__(self, protocol: dict, fields: dict[str, RadialDeflectionField]):
        self.protocol = protocol
        self.fields = fields
        geometry = protocol["cosmology_and_coordinates"]
        self.z_lens = float(geometry["lens_redshift"])
        self.z_ref = float(geometry["reference_source_redshift"])
        self.cosmo = FlatLambdaCDM(
            H0=float(geometry["H0_km_s_Mpc"]), Om0=float(geometry["Omega_m"])
        )
        self.distance_ratio_ref = self.distance_ratio(self.z_ref)
        self.sigma = float(
            protocol["raw_lensing_inputs"]["position_sigma_arcsec_per_coordinate"]
        )
        self._nie = LensModel(lens_model_list=["NIE"])

    def distance_ratio(self, redshift: float) -> float:
        source = self.cosmo.angular_diameter_distance(redshift)
        lens_source = self.cosmo.angular_diameter_distance_z1z2(self.z_lens, redshift)
        return float((lens_source / source).value)

    def alpha(
        self,
        model: str,
        parameters: np.ndarray,
        x_arcsec,
        y_arcsec,
        source_redshift: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x_arcsec, dtype=float)
        y = np.asarray(y_arcsec, dtype=float)
        ratio = self.distance_ratio(source_redshift)
        scale = ratio / self.distance_ratio_ref
        if model == "GR_plus_cluster_halo":
            baryon = self.fields["baryons_GR"]
            radial = lambda radius: baryon.reduced_alpha_arcsec(radius, ratio)
            base_x, base_y = pseudo_elliptical_deflection(
                x,
                y,
                radial,
                axis_ratio=1.0,
                phi_radian=0.0,
                center_x_arcsec=0.0,
                center_y_arcsec=0.0,
            )
            theta_e, q, phi, log_core, cx, cy, gamma1, gamma2 = parameters
            e1, e2 = param_util.phi_q2_ellipticity(phi=phi, q=q)
            halo_x, halo_y = self._nie.alpha(
                x,
                y,
                [
                    {
                        "theta_E": theta_e * scale,
                        "e1": e1,
                        "e2": e2,
                        "s_scale": 10.0**log_core,
                        "center_x": cx,
                        "center_y": cy,
                    }
                ],
            )
            shear_x, shear_y = shear_deflection(x, y, gamma1 * scale, gamma2 * scale)
            return base_x + halo_x + shear_x, base_y + halo_y + shear_y

        q, phi, cx, cy, gamma1, gamma2 = parameters
        field = self.fields[model]
        radial = lambda radius: field.reduced_alpha_arcsec(radius, ratio)
        base_x, base_y = pseudo_elliptical_deflection(
            x,
            y,
            radial,
            axis_ratio=q,
            phi_radian=phi,
            center_x_arcsec=cx,
            center_y_arcsec=cy,
        )
        shear_x, shear_y = shear_deflection(x, y, gamma1 * scale, gamma2 * scale)
        return base_x + shear_x, base_y + shear_y

    def ray_shooting(
        self,
        model: str,
        parameters: np.ndarray,
        x_arcsec,
        y_arcsec,
        source_redshift: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        alpha_x, alpha_y = self.alpha(
            model, parameters, x_arcsec, y_arcsec, source_redshift
        )
        return np.asarray(x_arcsec) - alpha_x, np.asarray(y_arcsec) - alpha_y

    def jacobian(
        self,
        model: str,
        parameters: np.ndarray,
        x_arcsec,
        y_arcsec,
        source_redshift: float,
        step: float = 2.0e-4,
    ) -> np.ndarray:
        x = np.atleast_1d(np.asarray(x_arcsec, dtype=float))
        y = np.atleast_1d(np.asarray(y_arcsec, dtype=float))
        bxp, byp = self.ray_shooting(model, parameters, x + step, y, source_redshift)
        bxm, bym = self.ray_shooting(model, parameters, x - step, y, source_redshift)
        c_xp, c_yp = self.ray_shooting(model, parameters, x, y + step, source_redshift)
        c_xm, c_ym = self.ray_shooting(model, parameters, x, y - step, source_redshift)
        matrices = np.empty((len(x), 2, 2), dtype=float)
        matrices[:, 0, 0] = (bxp - bxm) / (2.0 * step)
        matrices[:, 1, 0] = (byp - bym) / (2.0 * step)
        matrices[:, 0, 1] = (c_xp - c_xm) / (2.0 * step)
        matrices[:, 1, 1] = (c_yp - c_ym) / (2.0 * step)
        return matrices

    def profiled_residuals(
        self,
        model: str,
        parameters: np.ndarray,
        rows: pd.DataFrame,
    ) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        residuals = []
        sources = {}
        for family, group in rows.groupby("source_family", sort=True):
            x = group["x_arcsec"].to_numpy(float)
            y = group["y_arcsec"].to_numpy(float)
            redshift = float(group["source_redshift"].median())
            beta_x, beta_y = self.ray_shooting(
                model, parameters, x, y, redshift
            )
            beta = np.column_stack([beta_x, beta_y])
            matrices = self.jacobian(model, parameters, x, y, redshift)
            inverse = np.asarray([np.linalg.pinv(item, rcond=1.0e-9) for item in matrices])
            weights = np.einsum("nji,njk->nik", inverse, inverse)
            normal = weights.sum(axis=0)
            rhs = np.einsum("nij,nj->i", weights, beta)
            source = np.linalg.pinv(normal, rcond=1.0e-9) @ rhs
            sources[int(family)] = source
            delta_beta = beta - source
            delta_theta = np.einsum("nij,nj->ni", inverse, delta_beta)
            residuals.extend((delta_theta / self.sigma).ravel())
        return np.asarray(residuals), sources

    @staticmethod
    def prior_residuals(model: str, parameters: np.ndarray) -> np.ndarray:
        if model == "GR_plus_cluster_halo":
            p = parameters
            return np.asarray(
                [
                    (p[0] - 15.0) / 15.0,
                    (p[1] - 0.7) / 0.3,
                    p[3] / 1.0,
                    p[4] / 3.0,
                    p[5] / 3.0,
                    p[6] / 0.1,
                    p[7] / 0.1,
                ]
            )
        p = parameters
        return np.asarray(
            [
                (p[0] - 0.8) / 0.2,
                p[2] / 1.5,
                p[3] / 1.5,
                p[4] / 0.1,
                p[5] / 0.1,
            ]
        )

    def objective(self, model: str, parameters: np.ndarray, rows: pd.DataFrame) -> np.ndarray:
        data, _ = self.profiled_residuals(model, parameters, rows)
        return np.r_[data, self.prior_residuals(model, parameters)]

    def fit(
        self,
        model: str,
        rows: pd.DataFrame,
        *,
        starts: int,
        seed: int,
        initial_override: np.ndarray | None = None,
    ) -> dict:
        spec = spec_for(model)
        rng = np.random.default_rng(seed)
        initial = spec.initial if initial_override is None else initial_override
        candidates = [np.clip(initial, spec.lower + 1.0e-6, spec.upper - 1.0e-6)]
        span = spec.upper - spec.lower
        for _ in range(starts - 1):
            candidates.append(
                np.clip(
                    spec.initial + rng.normal(0.0, 0.20, len(spec.initial)) * span,
                    spec.lower + 1.0e-6,
                    spec.upper - 1.0e-6,
                )
            )
        best = None
        for index, start in enumerate(candidates, start=1):
            result = least_squares(
                lambda p: self.objective(model, p, rows),
                start,
                bounds=(spec.lower, spec.upper),
                jac="2-point",
                diff_step=2.0e-3,
                x_scale=span,
                max_nfev=int(self.protocol["optimization"]["maximum_function_evaluations"]),
                ftol=1.0e-10,
                xtol=1.0e-10,
                gtol=1.0e-10,
            )
            if best is None or result.cost < best.cost:
                best = result
            print(
                f"{model} start {index:02d}/{starts}: cost={result.cost:.5f}; "
                f"best={best.cost:.5f}",
                flush=True,
            )
        data, sources = self.profiled_residuals(model, best.x, rows)
        return {
            "result": best,
            "sources": sources,
            "optimization_radial_RMS_arcsec": float(
                np.sqrt(np.mean(np.sum((data.reshape(-1, 2) * self.sigma) ** 2, axis=1)))
            ),
        }

    def exact_predictions(
        self,
        model: str,
        parameters: np.ndarray,
        sources: dict[int, np.ndarray],
        rows: pd.DataFrame,
        *,
        stage: str,
    ) -> pd.DataFrame:
        records = []
        for row in rows.itertuples(index=False):
            source = sources[int(row.source_family)]
            observed = np.array([row.x_arcsec, row.y_arcsec], dtype=float)
            redshift = float(row.source_redshift)

            def equation(theta):
                bx, by = self.ray_shooting(
                    model,
                    parameters,
                    np.array([theta[0]]),
                    np.array([theta[1]]),
                    redshift,
                )
                return np.array([bx[0] - source[0], by[0] - source[1]])

            def derivative(theta):
                return self.jacobian(
                    model,
                    parameters,
                    np.array([theta[0]]),
                    np.array([theta[1]]),
                    redshift,
                )[0]

            solution = root(equation, observed, jac=derivative, method="hybr", tol=1.0e-10)
            predicted = np.asarray(solution.x, dtype=float)
            closure = float(np.linalg.norm(equation(predicted)))
            converged = bool(solution.success and closure <= 1.0e-6 and np.all(np.isfinite(predicted)))
            delta = predicted - observed if converged else np.array([np.nan, np.nan])
            records.append(
                {
                    "stage": stage,
                    "model": model,
                    "image_id": row.image_id,
                    "source_family": int(row.source_family),
                    "source_redshift": redshift,
                    "observed_x_arcsec": observed[0],
                    "observed_y_arcsec": observed[1],
                    "predicted_x_arcsec": predicted[0] if converged else np.nan,
                    "predicted_y_arcsec": predicted[1] if converged else np.nan,
                    "delta_x_arcsec": delta[0],
                    "delta_y_arcsec": delta[1],
                    "radial_residual_arcsec": float(np.linalg.norm(delta)),
                    "root_converged": converged,
                    "source_plane_closure_arcsec": closure,
                    "source_x_arcsec": source[0],
                    "source_y_arcsec": source[1],
                }
            )
        return pd.DataFrame(records)


def score(predictions: pd.DataFrame, sigma: float, free_parameters: int = 0) -> dict:
    converged = predictions["root_converged"].astype(bool).to_numpy()
    residual = predictions[["delta_x_arcsec", "delta_y_arcsec"]].to_numpy(float)
    if bool(converged.all()):
        radial_rms = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
        coordinate_rms = float(np.sqrt(np.mean(residual**2)))
        chi2 = float(np.sum(np.square(residual / sigma)))
        dof = max(1, residual.size - free_parameters)
    else:
        radial_rms = coordinate_rms = chi2 = float("inf")
        dof = max(1, residual.size - free_parameters)
    return {
        "images": int(len(predictions)),
        "converged_roots": int(converged.sum()),
        "all_roots_converged": bool(converged.all()),
        "exact_radial_RMS_arcsec": radial_rms,
        "exact_coordinate_RMS_arcsec": coordinate_rms,
        "coordinate_chi2": chi2,
        "degrees_of_freedom": int(dof),
        "reduced_chi2": float(chi2 / dof),
        "maximum_radial_residual_arcsec": float(
            predictions["radial_residual_arcsec"].max()
        ),
    }


def near_bound(model: str, parameters: np.ndarray, fraction: float = 0.01) -> dict[str, bool]:
    spec = spec_for(model)
    span = spec.upper - spec.lower
    distance = np.minimum(parameters - spec.lower, spec.upper - parameters) / span
    return {label: bool(value <= fraction) for label, value in zip(spec.labels, distance)}


def make_plot(
    images: pd.DataFrame,
    predictions: pd.DataFrame,
    scores: dict,
    profiles: pd.DataFrame,
    output: Path,
) -> None:
    colors = {
        "baryons_GR": "#777777",
        "fixed_simple_MOND": "#D95F02",
        "locked_universal_candidate": "#1874CD",
        "cluster_retuned_RAR_diagnostic": "#2E8B57",
        "GR_plus_cluster_halo": "#7B3294",
    }
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    ax = axes[0]
    for model, block in profiles.groupby("model", sort=False):
        ax.loglog(
            block["radius_kpc"],
            block["effective_acceleration_m_s2"],
            color=colors[model],
            label=model.replace("_", " "),
        )
    ax.axvspan(
        images["radius_arcsec"].min() * 3.741653570564318,
        images["radius_arcsec"].max() * 3.741653570564318,
        color="black",
        alpha=0.08,
        label="raw image radii",
    )
    ax.set_xlabel("3D radius (kpc)")
    ax.set_ylabel(r"Effective acceleration (m s$^{-2}$)")
    ax.set_title("Frozen radial fields")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)

    ax = axes[1]
    heldout = predictions[predictions["stage"] == "heldout"]
    heldout_ids = heldout["image_id"].drop_duplicates()
    heldout_observed = images[images["image_id"].isin(heldout_ids)]
    ax.scatter(
        heldout_observed["x_arcsec"],
        heldout_observed["y_arcsec"],
        color="black",
        s=30,
        label="observed heldout",
    )
    for model in ("fixed_simple_MOND", "locked_universal_candidate", "GR_plus_cluster_halo"):
        block = heldout[heldout["model"] == model]
        ax.scatter(
            block["predicted_x_arcsec"],
            block["predicted_y_arcsec"],
            color=colors[model],
            marker="x",
            s=45,
            label=f"{model.replace('_', ' ')} heldout",
        )
        for row in block.itertuples(index=False):
            if np.isfinite(row.predicted_x_arcsec):
                ax.plot(
                    [row.observed_x_arcsec, row.predicted_x_arcsec],
                    [row.observed_y_arcsec, row.predicted_y_arcsec],
                    color=colors[model],
                    alpha=0.35,
                    linewidth=0.8,
                )
    ax.set_aspect("equal")
    ax.set_xlabel("east offset (arcsec)")
    ax.set_ylabel("north offset (arcsec)")
    ax.set_title("Exact heldout image roots")
    ax.legend(fontsize=6)

    ax = axes[2]
    models = list(colors)
    values = [scores[model]["heldout"]["exact_radial_RMS_arcsec"] for model in models]
    finite = [value is not None and np.isfinite(float(value)) for value in values]
    display_values = [float(value) if valid else 0.0 for value, valid in zip(values, finite)]
    ax.barh(
        [model.replace("_", " ") for model in models],
        display_values,
        color=[colors[model] for model in models],
    )
    finite_maximum = max(display_values)
    for index, (model, valid) in enumerate(zip(models, finite)):
        if not valid:
            recovered = scores[model]["heldout"]["converged_roots"]
            total = scores[model]["heldout"]["images"]
            ax.text(
                0.02 * finite_maximum,
                index,
                f"FAILED: {recovered}/{total} roots",
                va="center",
                color=colors[model],
                fontweight="bold",
            )
    ax.axvline(0.5, color="crimson", linestyle="--", label="candidate 0.5 arcsec gate")
    ax.set_xlabel("Exact heldout radial RMS (arcsec)")
    ax.set_title("Raw predictive comparison")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190)
    plt.close(fig)


def run_sensitivity(
    model: str,
    anchors: pd.DataFrame,
    protocol: dict,
    images: pd.DataFrame,
    training: pd.DataFrame,
    heldout: pd.DataFrame,
    baseline_fields: dict[str, RadialDeflectionField],
    baseline_parameters: np.ndarray,
    *,
    gbar_shift_dex: float,
    density_shift_dex: float,
    starts: int,
    seed: int,
) -> dict:
    field, _ = build_field(
        model,
        anchors,
        protocol,
        gbar_shift_dex=gbar_shift_dex,
        density_shift_dex=density_shift_dex,
    )
    fields = dict(baseline_fields)
    fields[model] = field
    lens = RawLens(protocol, fields)
    fit = lens.fit(
        model,
        training,
        starts=starts,
        seed=seed,
        initial_override=baseline_parameters,
    )
    prediction = lens.exact_predictions(
        model, fit["result"].x, fit["sources"], heldout, stage="sensitivity_heldout"
    )
    return {
        "model": model,
        "gbar_shift_dex": gbar_shift_dex,
        "density_shift_dex": density_shift_dex,
        "fit_success": bool(fit["result"].success),
        "heldout_score": score(prediction, lens.sigma),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "rxj2129_raw_theory_lensing_protocol.json",
    )
    parser.add_argument("--starts", type=int, default=None)
    parser.add_argument("--skip-sensitivities", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if not protocol["status"].startswith("frozen_before_any_candidate"):
        raise RuntimeError("protocol must be frozen before scoring")

    images = load_images(protocol)
    heldout_ids = set(protocol["predictive_split"]["heldout"])
    heldout = images[images["image_id"].isin(heldout_ids)].copy()
    training = images[~images["image_id"].isin(heldout_ids)].copy()
    if len(training) != 15 or len(heldout) != 7:
        raise RuntimeError("predictive split changed")
    if set(training["source_family"]) != set(heldout["source_family"]):
        raise RuntimeError("every heldout family must be represented in training")

    anchors = load_baryonic_anchors(protocol)
    models = [
        "baryons_GR",
        "fixed_simple_MOND",
        "locked_universal_candidate",
        "cluster_retuned_RAR_diagnostic",
        "GR_plus_cluster_halo",
    ]
    fields = {}
    profile_tables = []
    for model in models:
        field_model = "baryons_GR" if model == "GR_plus_cluster_halo" else model
        if field_model not in fields:
            fields[field_model], table = build_field(field_model, anchors, protocol)
            profile_tables.append(table)
    profiles = pd.concat(profile_tables, ignore_index=True)
    lens = RawLens(protocol, fields)
    starts = args.starts or int(protocol["optimization"]["multi_starts"])
    seed = int(protocol["optimization"]["random_seed"])

    scores = {}
    predictions = []
    parameters = []
    training_fits = {}
    for offset, model in enumerate(models):
        fit = lens.fit(model, training, starts=starts, seed=seed + offset)
        training_fits[model] = fit
        spec = spec_for(model)
        parameters += [
            {
                "stage": "training",
                "model": model,
                "parameter": label,
                "value": float(value),
                "near_bound": near_bound(model, fit["result"].x)[label],
            }
            for label, value in zip(spec.labels, fit["result"].x)
        ]
        training_prediction = lens.exact_predictions(
            model,
            fit["result"].x,
            fit["sources"],
            training,
            stage="training",
        )
        heldout_prediction = lens.exact_predictions(
            model,
            fit["result"].x,
            fit["sources"],
            heldout,
            stage="heldout",
        )
        predictions += [training_prediction, heldout_prediction]
        scores[model] = {
            "training": score(
                training_prediction,
                lens.sigma,
                free_parameters=len(spec.labels) + 14,
            ),
            "heldout": score(heldout_prediction, lens.sigma),
            "optimizer_success": bool(fit["result"].success),
            "optimizer_message": str(fit["result"].message),
            "best_total_cost": float(fit["result"].cost),
            "optimization_radial_RMS_arcsec": fit["optimization_radial_RMS_arcsec"],
            "geometry_parameter_near_bound": near_bound(model, fit["result"].x),
        }

    for offset, model in enumerate(models):
        training_fit = training_fits[model]
        fit = lens.fit(
            model,
            images,
            starts=starts,
            seed=seed + 100 + offset,
            initial_override=training_fit["result"].x,
        )
        spec = spec_for(model)
        parameters += [
            {
                "stage": "all_images_descriptive",
                "model": model,
                "parameter": label,
                "value": float(value),
                "near_bound": near_bound(model, fit["result"].x)[label],
            }
            for label, value in zip(spec.labels, fit["result"].x)
        ]
        prediction = lens.exact_predictions(
            model,
            fit["result"].x,
            fit["sources"],
            images,
            stage="all_images_descriptive",
        )
        predictions.append(prediction)
        scores[model]["all_images_descriptive"] = score(
            prediction, lens.sigma, free_parameters=len(spec.labels) + 14
        )

    sensitivities = []
    if not args.skip_sensitivities:
        sensitivity_starts = max(4, starts // 2)
        index = 0
        for model in ("fixed_simple_MOND", "locked_universal_candidate"):
            for shift in protocol["frozen_sensitivities"]["global_log10_gbar_shifts_dex"]:
                sensitivities.append(
                    run_sensitivity(
                        model,
                        anchors,
                        protocol,
                        images,
                        training,
                        heldout,
                        fields,
                        training_fits[model]["result"].x,
                        gbar_shift_dex=float(shift),
                        density_shift_dex=0.0,
                        starts=sensitivity_starts,
                        seed=seed + 200 + index,
                    )
                )
                index += 1
        for shift in protocol["frozen_sensitivities"]["candidate_log10_density_shifts_dex"]:
            sensitivities.append(
                run_sensitivity(
                    "locked_universal_candidate",
                    anchors,
                    protocol,
                    images,
                    training,
                    heldout,
                    fields,
                    training_fits["locked_universal_candidate"]["result"].x,
                    gbar_shift_dex=0.0,
                    density_shift_dex=float(shift),
                    starts=sensitivity_starts,
                    seed=seed + 200 + index,
                )
            )
            index += 1

    output = ROOT / "results" / "rxj2129_raw_theory_lensing"
    output.mkdir(parents=True, exist_ok=True)
    prediction_table = pd.concat(predictions, ignore_index=True)
    prediction_table.to_csv(output / "image_predictions.csv", index=False)
    pd.DataFrame(parameters).to_csv(output / "fitted_parameters.csv", index=False)
    profiles.to_csv(output / "radial_deflection_profiles.csv", index=False)
    make_plot(images, prediction_table, scores, profiles, output / "raw_lensing_comparison.png")

    gates = protocol["advance_gates"]
    candidate = scores["locked_universal_candidate"]
    mond = scores["fixed_simple_MOND"]
    halo = scores["GR_plus_cluster_halo"]
    candidate_bounds = candidate["geometry_parameter_near_bound"]
    gate_audit = {
        "candidate_all_heldout_roots_converged": candidate["heldout"][
            "all_roots_converged"
        ],
        "candidate_heldout_RMS_below_absolute_gate": bool(
            candidate["heldout"]["exact_radial_RMS_arcsec"]
            <= gates["candidate_maximum_heldout_radial_RMS_arcsec"]
        ),
        "candidate_heldout_RMS_lower_than_fixed_simple_MOND": bool(
            candidate["heldout"]["exact_radial_RMS_arcsec"]
            < mond["heldout"]["exact_radial_RMS_arcsec"]
        ),
        "candidate_to_compact_halo_heldout_RMS_ratio": float(
            candidate["heldout"]["exact_radial_RMS_arcsec"]
            / halo["heldout"]["exact_radial_RMS_arcsec"]
        ),
        "candidate_within_compact_halo_ratio_gate": bool(
            candidate["heldout"]["exact_radial_RMS_arcsec"]
            / halo["heldout"]["exact_radial_RMS_arcsec"]
            <= gates["candidate_to_compact_halo_heldout_RMS_ratio_max"]
        ),
        "candidate_no_fitted_lensing_amplitude_or_slip": True,
        "candidate_no_geometry_parameter_near_bound": not any(candidate_bounds.values()),
    }
    gate_audit["passes_all"] = bool(
        gate_audit["candidate_all_heldout_roots_converged"]
        and gate_audit["candidate_heldout_RMS_below_absolute_gate"]
        and gate_audit["candidate_heldout_RMS_lower_than_fixed_simple_MOND"]
        and gate_audit["candidate_within_compact_halo_ratio_gate"]
        and gate_audit["candidate_no_fitted_lensing_amplitude_or_slip"]
        and gate_audit["candidate_no_geometry_parameter_near_bound"]
    )
    report = {
        "report_version": "RXJ2129-RAW-THEORY-LENSING-0.1.1",
        "status": "executed_raw_image_position_pilot",
        "protocol": {
            "path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(config_path),
            "freeze_status": protocol["status"],
        },
        "inputs": {
            "raw_image_catalog": protocol["raw_lensing_inputs"],
            "baryonic_profile": protocol["baryonic_inputs"],
            "training_images": len(training),
            "heldout_images": len(heldout),
            "heldout_ids": sorted(heldout_ids),
            "observed_impact_radius_range_kpc": [
                float(images["radius_arcsec"].min() * 3.741653570564318),
                float(images["radius_arcsec"].max() * 3.741653570564318),
            ],
        },
        "photon_closure": protocol["photon_closure"],
        "model_scores": scores,
        "frozen_baryon_and_density_sensitivities": sensitivities,
        "comparisons": {
            "candidate_vs_fixed_simple_MOND_heldout_RMS_ratio": finite_ratio_or_none(
                candidate["heldout"]["exact_radial_RMS_arcsec"],
                mond["heldout"]["exact_radial_RMS_arcsec"],
            ),
            "candidate_vs_compact_halo_heldout_RMS_ratio": gate_audit[
                "candidate_to_compact_halo_heldout_RMS_ratio"
            ],
            "candidate_vs_cluster_retuned_RAR_heldout_RMS_ratio": float(
                candidate["heldout"]["exact_radial_RMS_arcsec"]
                / scores["cluster_retuned_RAR_diagnostic"]["heldout"][
                    "exact_radial_RMS_arcsec"
                ]
            ),
        },
        "parameter_accounting": {
            "candidate_RXJ2129_fitted_gravity_or_lensing_amplitudes": 0,
            "candidate_structural_geometry_nuisances": len(FIXED_LABELS),
            "source_position_nuisances_each_model": 14,
            "compact_halo_object_specific_parameters": 6,
            "compact_halo_external_shear_nuisances": 2,
            "published_RXJ2129_reference_model_halos": 71,
        },
        "advance_gate_audit": gate_audit,
        "claim_boundary": protocol["claim_boundary"],
        "strict_interpretation": {
            "raw_lens_observables_used": True,
            "NFW_deprojected_gobs_used": False,
            "raw_image_positions_used_in_candidate_parameter_selection": False,
            "RXJ2129_derived_field_used_in_prior_candidate_calibration": True,
            "independent_cluster_validation": False,
            "complete_native_baryonic_likelihood_used": False,
            "covariant_candidate_action_used": False,
            "single_cluster_pilot": True,
            "publication_grade_raw_lensing_claim": False,
        },
        "outputs": protocol["outputs"],
    }
    (output / "report.json").write_text(
        json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"scores": scores, "gate_audit": gate_audit}, indent=2))


if __name__ == "__main__":
    main()
