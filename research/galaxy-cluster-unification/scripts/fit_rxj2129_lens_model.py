#!/usr/bin/env python3
"""Fit the frozen independent RX J2129 strong-lens nuisance models."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.constants import G, c
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Util import param_util
from scipy.optimize import least_squares, root


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "r1_rxj2129_lens_model_protocol.json"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    labels: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    initial: np.ndarray


BASE_LABELS = (
    "theta_E_ref_arcsec",
    "axis_ratio_q",
    "position_angle_phi_radian",
    "log10_core_arcsec",
    "center_x_arcsec",
    "center_y_arcsec",
    "gamma1",
    "gamma2",
    "log10_central_mass_msun",
    "log10_central_Rs_arcsec",
)
BASE_LOWER = np.array([5.0, 0.3, -np.pi / 2, -1.3, -5.0, -5.0, -0.2, -0.2, 11.0, 0.2])
BASE_UPPER = np.array([35.0, 1.0, np.pi / 2, 1.3, 5.0, 5.0, 0.2, 0.2, 12.5, 1.4])
BASE_INITIAL = np.array(
    [15.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.log10(5.81e11), np.log10(22.8114 / 3.741653570564318)]
)
MEMBER_LABELS = (
    "log10_total_to_stellar_member_mass_ratio",
    "log10_pseudo_jaffe_cut_to_core_ratio",
)


def model_spec(name: str) -> ModelSpec:
    if name == "model_A":
        return ModelSpec(name, BASE_LABELS, BASE_LOWER, BASE_UPPER, BASE_INITIAL)
    if name == "model_B":
        return ModelSpec(
            name,
            BASE_LABELS + MEMBER_LABELS,
            np.r_[BASE_LOWER, [0.0, 0.5]],
            np.r_[BASE_UPPER, [2.0, 1.7]],
            np.r_[BASE_INITIAL, [1.0, 1.0]],
        )
    raise ValueError(name)


class FrozenLens:
    def __init__(self, cfg: dict, images: pd.DataFrame, members: pd.DataFrame):
        geometry = cfg["cosmology_and_coordinates"]
        self.cfg = cfg
        self.images = images
        self.members = members
        self.z_lens = float(geometry["lens_redshift"])
        self.z_ref = float(geometry["reference_source_redshift"])
        self.sigma_image = float(cfg["likelihood"]["sigma_arcsec_per_coordinate"])
        self.cosmo = FlatLambdaCDM(
            H0=float(geometry["H0_km_s_Mpc"]),
            Om0=float(geometry["Omega_m"]),
        )
        self.dd_mpc = float(self.cosmo.angular_diameter_distance(self.z_lens).value)
        self.sigma_crit_ref = self.sigma_crit(self.z_ref)
        self.distance_ratio_ref = self.distance_ratio(self.z_ref)
        self.center_ra = float(geometry["center_ra_deg"])
        self.center_dec = float(geometry["center_dec_deg"])
        cos_dec = np.cos(np.deg2rad(self.center_dec))
        self.member_x = (members["ra_deg"].to_numpy(float) - self.center_ra) * 3600 * cos_dec
        self.member_y = (members["dec_deg"].to_numpy(float) - self.center_dec) * 3600
        a = members["a"].to_numpy(float)
        b = members["b"].to_numpy(float)
        raw_core = np.sqrt(a * b) * 0.065
        finite = raw_core[np.isfinite(raw_core) & (raw_core > 0)]
        replacement = float(np.median(finite)) if len(finite) else 0.13
        self.member_core = np.maximum(0.065, np.where(np.isfinite(raw_core), raw_core, replacement))
        self.member_stellar_mass = members["stellar_mass_msun"].to_numpy(float)
        self.member_probability = members["membership_probability"].to_numpy(float)
        self.member_ids = members["clash_id"].astype(str).to_numpy()
        self.area_per_arcsec2_mpc2 = (self.dd_mpc * np.deg2rad(1 / 3600)) ** 2
        self._models: dict[tuple[str, float], LensModel] = {}

    def distance_ratio(self, z_source: float) -> float:
        ds = self.cosmo.angular_diameter_distance(z_source)
        dds = self.cosmo.angular_diameter_distance_z1z2(self.z_lens, z_source)
        return float((dds / ds).value)

    def sigma_crit(self, z_source: float) -> float:
        dd = self.cosmo.angular_diameter_distance(self.z_lens)
        ds = self.cosmo.angular_diameter_distance(z_source)
        dds = self.cosmo.angular_diameter_distance_z1z2(self.z_lens, z_source)
        return float((c**2 / (4 * np.pi * G) * ds / (dd * dds)).to(u.Msun / u.Mpc**2).value)

    def lens_model(self, kind: str, z_source: float) -> LensModel:
        key = (kind, round(float(z_source), 6))
        if key not in self._models:
            profiles = ["NIE", "HERNQUIST", "SHEAR"]
            if kind == "model_B":
                profiles += ["PJAFFE"] * len(self.members)
            self._models[key] = LensModel(lens_model_list=profiles)
        return self._models[key]

    def kwargs_lens(
        self,
        kind: str,
        params: np.ndarray,
        member_weights: np.ndarray | None = None,
        z_source: float | None = None,
    ) -> list[dict]:
        theta_e, q, phi, log_core, cx, cy, g1, g2, log_mass, log_rs = params[:10]
        e1, e2 = param_util.phi_q2_ellipticity(phi=phi, q=q)
        rs_arcsec = 10**log_rs
        rs_mpc = rs_arcsec * 3.741653570564318 / 1000
        rho_s = 10**log_mass / (2 * np.pi * rs_mpc**3)
        sigma0_bcg = rho_s * rs_mpc / self.sigma_crit_ref
        scale = 1.0 if z_source is None else self.distance_ratio(z_source) / self.distance_ratio_ref
        kwargs: list[dict] = [
            {
                "theta_E": theta_e * scale,
                "e1": e1,
                "e2": e2,
                "s_scale": 10**log_core,
                "center_x": cx,
                "center_y": cy,
            },
            {
                "sigma0": sigma0_bcg * scale,
                "Rs": rs_arcsec,
                "center_x": 0.0,
                "center_y": 0.0,
            },
            {"gamma1": g1 * scale, "gamma2": g2 * scale, "ra_0": 0.0, "dec_0": 0.0},
        ]
        if kind == "model_B":
            mass_ratio = 10 ** params[10]
            cut_ratio = 10 ** params[11]
            weights = self.member_probability if member_weights is None else member_weights
            masses = self.member_stellar_mass * np.asarray(weights) * mass_ratio
            cut = self.member_core * cut_ratio
            denom = (
                2
                * np.pi
                * self.member_core
                * cut
                * self.sigma_crit_ref
                * self.area_per_arcsec2_mpc2
            )
            sigma0 = masses / denom
            kwargs += [
                {
                    "sigma0": float(s0 * scale),
                    "Ra": float(ra),
                    "Rs": float(rs),
                    "center_x": float(x),
                    "center_y": float(y),
                }
                for s0, ra, rs, x, y in zip(
                    sigma0, self.member_core, cut, self.member_x, self.member_y
                )
            ]
        return kwargs

    @staticmethod
    def jacobian_matrices(model: LensModel, x: np.ndarray, y: np.ndarray, kwargs: list[dict]):
        f_xx, f_xy, f_yx, f_yy = model.hessian(x, y, kwargs)
        matrices = np.empty((len(np.atleast_1d(x)), 2, 2), dtype=float)
        matrices[:, 0, 0] = 1 - np.asarray(f_xx)
        matrices[:, 0, 1] = -np.asarray(f_xy)
        matrices[:, 1, 0] = -np.asarray(f_yx)
        matrices[:, 1, 1] = 1 - np.asarray(f_yy)
        return matrices

    def profiled_residuals(
        self,
        kind: str,
        params: np.ndarray,
        rows: pd.DataFrame,
        member_weights: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        residuals: list[float] = []
        sources: dict[int, np.ndarray] = {}
        for family, group in rows.groupby("source_family", sort=True):
            x = group["delta_ra_east_arcsec"].to_numpy(float)
            y = group["delta_dec_north_arcsec"].to_numpy(float)
            z = float(group["source_redshift"].median())
            model = self.lens_model(kind, z)
            kwargs = self.kwargs_lens(kind, params, member_weights, z_source=z)
            bx, by = model.ray_shooting(x, y, kwargs)
            beta = np.c_[bx, by]
            matrices = self.jacobian_matrices(model, x, y, kwargs)
            inv_a = np.array([np.linalg.pinv(a, rcond=1e-10) for a in matrices])
            weights = np.einsum("nji,njk->nik", inv_a, inv_a)
            normal = weights.sum(axis=0)
            rhs = np.einsum("nij,nj->i", weights, beta)
            source = np.linalg.pinv(normal, rcond=1e-10) @ rhs
            sources[int(family)] = source
            delta_beta = beta - source
            delta_theta = np.einsum("nij,nj->ni", inv_a, delta_beta)
            residuals.extend((delta_theta / self.sigma_image).ravel())
        return np.asarray(residuals), sources

    @staticmethod
    def prior_residuals(kind: str, p: np.ndarray) -> np.ndarray:
        priors = [
            (p[0] - 15.0) / 15.0,
            (p[1] - 0.7) / 0.3,
            p[3] / 1.0,
            p[4] / 3.0,
            p[5] / 3.0,
            p[6] / 0.1,
            p[7] / 0.1,
            (p[8] - np.log10(5.81e11)) / 0.30,
            (p[9] - np.log10(22.8114 / 3.741653570564318)) / 0.30,
        ]
        if kind == "model_B":
            priors += [(p[10] - 1.0) / 0.7, (p[11] - 1.0) / 0.4]
        return np.asarray(priors)

    def objective(self, kind: str, p: np.ndarray, rows: pd.DataFrame) -> np.ndarray:
        data, _ = self.profiled_residuals(kind, p, rows)
        return np.r_[data, self.prior_residuals(kind, p)]

    def fit(self, kind: str, rows: pd.DataFrame, starts: int, seed: int) -> dict:
        spec = model_spec(kind)
        rng = np.random.default_rng(seed)
        candidates = [spec.initial.copy()]
        span = spec.upper - spec.lower
        for _ in range(starts - 1):
            jitter = rng.normal(0, 0.22, len(spec.initial)) * span
            trial = np.clip(spec.initial + jitter, spec.lower + 1e-6, spec.upper - 1e-6)
            candidates.append(trial)
        best = None
        for index, start in enumerate(candidates, start=1):
            result = least_squares(
                lambda p: self.objective(kind, p, rows),
                start,
                bounds=(spec.lower, spec.upper),
                jac="3-point",
                diff_step=1e-3,
                x_scale=span,
                max_nfev=2400,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
            )
            if best is None or result.cost < best.cost:
                best = result
            print(
                f"{kind} start {index:02d}/{starts}: cost={result.cost:.6f}; "
                f"best={best.cost:.6f}; nfev={result.nfev}",
                flush=True,
            )
        assert best is not None
        data_residual, sources = self.profiled_residuals(kind, best.x, rows)
        return {
            "result": best,
            "sources": sources,
            "optimization_coordinate_rms_arcsec": float(
                np.sqrt(np.mean((data_residual * self.sigma_image) ** 2))
            ),
            "optimization_radial_rms_arcsec": float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            (data_residual.reshape(-1, 2) * self.sigma_image) ** 2, axis=1
                        )
                    )
                )
            ),
        }

    def exact_predictions(
        self,
        kind: str,
        params: np.ndarray,
        sources: dict[int, np.ndarray],
        rows: pd.DataFrame,
        member_weights: np.ndarray | None = None,
        stage: str = "final",
    ) -> pd.DataFrame:
        records = []
        for row in rows.itertuples(index=False):
            family = int(row.source_family)
            source = sources[family]
            z_source = float(row.source_redshift)
            model = self.lens_model(kind, z_source)
            kwargs = self.kwargs_lens(
                kind, params, member_weights, z_source=z_source
            )

            def equation(theta):
                bx, by = model.ray_shooting(theta[0], theta[1], kwargs)
                return np.array([bx - source[0], by - source[1]])

            def jacobian(theta):
                return self.jacobian_matrices(
                    model, np.array([theta[0]]), np.array([theta[1]]), kwargs
                )[0]

            observed = np.array([row.delta_ra_east_arcsec, row.delta_dec_north_arcsec])
            solution = root(equation, observed, jac=jacobian, method="hybr", tol=1e-11)
            predicted = np.asarray(solution.x, dtype=float)
            closure = float(np.linalg.norm(equation(predicted)))
            converged = bool(solution.success and closure <= 1e-7 and np.all(np.isfinite(predicted)))
            delta = predicted - observed if converged else np.array([np.nan, np.nan])
            records.append(
                {
                    "stage": stage,
                    "model": kind,
                    "image_id": row.image_id,
                    "source_family": family,
                    "source_redshift": float(row.source_redshift),
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


def score_predictions(predictions: pd.DataFrame, sigma: float, free_parameters: int) -> dict:
    valid = predictions["root_converged"].astype(bool).to_numpy()
    residual = predictions[["delta_x_arcsec", "delta_y_arcsec"]].to_numpy(float)
    all_converged = bool(valid.all())
    if all_converged:
        radial_rms = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
        coordinate_rms = float(np.sqrt(np.mean(residual**2)))
        chi2 = float(np.sum((residual / sigma) ** 2))
        dof = max(1, residual.size - free_parameters)
        reduced_chi2 = chi2 / dof
        standardized_mean = float(
            np.max(np.abs(np.mean(residual, axis=0)) / (sigma / np.sqrt(len(residual))))
        )
    else:
        radial_rms = coordinate_rms = chi2 = reduced_chi2 = standardized_mean = float("inf")
    return {
        "images": int(len(predictions)),
        "converged_roots": int(valid.sum()),
        "all_roots_converged": all_converged,
        "exact_radial_rms_arcsec": radial_rms,
        "exact_coordinate_rms_arcsec": coordinate_rms,
        "exact_coordinate_chi2": chi2,
        "degrees_of_freedom": int(max(1, residual.size - free_parameters)),
        "exact_coordinate_reduced_chi2": reduced_chi2,
        "maximum_absolute_standardized_coordinate_mean": standardized_mean,
        "maximum_source_plane_root_closure_arcsec": float(
            predictions["source_plane_closure_arcsec"].max()
        ),
    }


def full_laplace_covariance(
    lens: FrozenLens,
    kind: str,
    params: np.ndarray,
    sources: dict[int, np.ndarray],
    rows: pd.DataFrame,
) -> tuple[list[str], np.ndarray, dict]:
    spec = model_spec(kind)
    families = sorted(sources)
    labels = list(spec.labels)
    for family in families:
        labels += [f"source_{family}_x_arcsec", f"source_{family}_y_arcsec"]
    vector = np.r_[params, np.concatenate([sources[f] for f in families])]
    steps = np.maximum(1e-6, np.abs(vector) * 1e-5)

    def residual(v):
        p = v[: len(params)]
        source_values = v[len(params) :].reshape(-1, 2)
        source_map = {family: source_values[i] for i, family in enumerate(families)}
        data = []
        for row in rows.itertuples(index=False):
            z_source = float(row.source_redshift)
            model = lens.lens_model(kind, z_source)
            kwargs = lens.kwargs_lens(kind, p, z_source=z_source)
            x = np.array([row.delta_ra_east_arcsec])
            y = np.array([row.delta_dec_north_arcsec])
            bx, by = model.ray_shooting(x, y, kwargs)
            delta_beta = np.array([bx[0], by[0]]) - source_map[int(row.source_family)]
            a = lens.jacobian_matrices(model, x, y, kwargs)[0]
            data.extend((np.linalg.pinv(a, rcond=1e-10) @ delta_beta) / lens.sigma_image)
        return np.r_[data, lens.prior_residuals(kind, p)]

    base = residual(vector)
    jac = np.empty((len(base), len(vector)))
    for j, step in enumerate(steps):
        plus = vector.copy()
        minus = vector.copy()
        plus[j] += step
        minus[j] -= step
        jac[:, j] = (residual(plus) - residual(minus)) / (2 * step)
    precision = jac.T @ jac
    covariance = np.linalg.pinv(precision, rcond=1e-10)
    covariance = (covariance + covariance.T) / 2
    eig = np.linalg.eigvalsh(covariance)
    diagnostics = {
        "dimension": int(len(vector)),
        "jacobian_rank": int(np.linalg.matrix_rank(jac)),
        "precision_condition_number": float(np.linalg.cond(precision)),
        "minimum_covariance_eigenvalue": float(eig.min()),
        "maximum_covariance_eigenvalue": float(eig.max()),
        "symmetric": bool(np.allclose(covariance, covariance.T, atol=1e-10, rtol=1e-10)),
        "positive_semidefinite": bool(eig.min() >= -1e-9 * max(1.0, eig.max())),
        "method": "local observed-Hessian/Laplace covariance including frozen weak priors",
    }
    return labels, covariance, diagnostics


def near_bound(spec: ModelSpec, params: np.ndarray, fraction: float = 0.01) -> np.ndarray:
    span = spec.upper - spec.lower
    distance = np.minimum(params - spec.lower, spec.upper - params) / span
    return distance <= fraction


def json_safe(value):
    """Replace non-finite optimizer diagnostics with JSON null without hiding failures."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def parameter_records(stage: str, kind: str, params: np.ndarray) -> list[dict]:
    spec = model_spec(kind)
    close = near_bound(spec, params)
    return [
        {
            "stage": stage,
            "model": kind,
            "parameter": label,
            "value": value,
            "lower_bound": lower,
            "upper_bound": upper,
            "within_one_percent_of_bound": bool(bound),
        }
        for label, value, lower, upper, bound in zip(
            spec.labels, params, spec.lower, spec.upper, close
        )
    ]


def make_diagnostic(images: pd.DataFrame, predictions: pd.DataFrame, output: Path):
    fig, (ax, axr) = plt.subplots(1, 2, figsize=(12, 5))
    for family, group in images.groupby("source_family"):
        ax.scatter(
            group["delta_ra_east_arcsec"],
            group["delta_dec_north_arcsec"],
            label=f"family {int(family)}",
            s=35,
        )
    for row in predictions.itertuples(index=False):
        if row.root_converged:
            ax.plot(
                [row.observed_x_arcsec, row.predicted_x_arcsec],
                [row.observed_y_arcsec, row.predicted_y_arcsec],
                color="black",
                alpha=0.55,
                linewidth=1,
            )
            ax.scatter(row.predicted_x_arcsec, row.predicted_y_arcsec, marker="x", color="black")
    circle = plt.Circle((0, 0), 5, fill=False, linestyle="--", color="0.5")
    ax.add_patch(circle)
    ax.set_aspect("equal")
    ax.set_xlabel("east offset (arcsec)")
    ax.set_ylabel("north offset (arcsec)")
    ax.set_title("Observed points and exact model roots")
    ax.legend(fontsize=7, ncol=2)
    finite = predictions[np.isfinite(predictions["radial_residual_arcsec"])]
    axr.bar(finite["image_id"].astype(str), finite["radial_residual_arcsec"])
    axr.axhline(1.0, color="crimson", linestyle="--", label="1.0 arcsec gate")
    axr.set_ylabel("exact radial residual (arcsec)")
    axr.set_xlabel("image")
    axr.tick_params(axis="x", rotation=75)
    axr.set_title("All-image exact residuals")
    axr.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--starts", type=int, default=None, help="Override frozen starts for testing")
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if cfg["status"] != "frozen_before_independent_image_residual_evaluation":
        raise RuntimeError("Refusing to execute an unfrozen or already-mutated protocol")

    images = pd.read_csv(ROOT / cfg["inputs"]["image_ledger"])
    images = images[images["likelihood_included"].astype(bool)].copy()
    images["image_id"] = images["image_id"].astype(str)
    images = images.sort_values(["source_family", "image_id"]).reset_index(drop=True)
    training = images[images["image_id"].str.endswith(".1") | images["image_id"].str.endswith(".2")]
    heldout = images[images["image_id"].str.endswith(".3")]
    if len(images) != 21 or len(training) != 14 or len(heldout) != 7:
        raise RuntimeError("Frozen image-count or blind-split invariant failed")

    members = pd.read_csv(ROOT / cfg["inputs"]["member_likelihood"])
    if len(members) != 66:
        raise RuntimeError("Frozen member count changed")
    lens = FrozenLens(cfg, images, members)
    starts = args.starts or int(cfg["optimization"]["multi_starts_each_model"])
    seed = int(cfg["optimization"]["random_seed"])

    train_fits = {}
    heldout_predictions = []
    parameter_rows = []
    for offset, kind in enumerate(("model_A", "model_B")):
        fit = lens.fit(kind, training, starts, seed + offset)
        train_fits[kind] = fit
        parameter_rows += parameter_records("training", kind, fit["result"].x)
        prediction = lens.exact_predictions(
            kind,
            fit["result"].x,
            fit["sources"],
            heldout,
            stage="heldout",
        )
        heldout_predictions.append(prediction)
        fit["heldout_score"] = score_predictions(
            prediction, lens.sigma_image, 0
        )
        print(
            f"{kind} heldout exact radial RMS: "
            f"{fit['heldout_score']['exact_radial_rms_arcsec']:.6f} arcsec",
            flush=True,
        )

    rms_a = train_fits["model_A"]["heldout_score"]["exact_radial_rms_arcsec"]
    rms_b = train_fits["model_B"]["heldout_score"]["exact_radial_rms_arcsec"]
    improvement = (rms_a - rms_b) / rms_a if np.isfinite(rms_a) and rms_a > 0 else -np.inf
    b_all_roots = train_fits["model_B"]["heldout_score"]["all_roots_converged"]
    select_b = bool(b_all_roots and improvement >= 0.10)
    selected = "model_B" if select_b else "model_A"
    print(
        f"Selected {selected}; model_B heldout improvement={improvement:.6f}", flush=True
    )

    final_fit = lens.fit(selected, images, starts, seed + 100)
    parameter_rows += parameter_records("all_images", selected, final_fit["result"].x)
    final_predictions = lens.exact_predictions(
        selected,
        final_fit["result"].x,
        final_fit["sources"],
        images,
        stage="all_images",
    )
    n_full = len(final_fit["result"].x) + 14
    final_score = score_predictions(final_predictions, lens.sigma_image, n_full)

    labels, covariance, covariance_diagnostics = full_laplace_covariance(
        lens, selected, final_fit["result"].x, final_fit["sources"], images
    )
    spec = model_spec(selected)
    bound_flags = near_bound(spec, final_fit["result"].x)

    member_sensitivity = {
        "applicable": selected == "model_B",
        "vectors": 0,
        "exact_radial_rms_p16_arcsec": None,
        "exact_radial_rms_median_arcsec": None,
        "exact_radial_rms_p84_arcsec": None,
        "all_vectors_all_roots_converged": None,
        "gate_pass": True,
    }
    if selected == "model_B":
        bootstrap = np.load(ROOT / cfg["inputs"]["member_probability_bootstrap"])
        if not np.array_equal(bootstrap["clash_ids"].astype(str), lens.member_ids):
            raise RuntimeError("Member bootstrap order no longer matches the likelihood ledger")
        indices = np.linspace(0, len(bootstrap["membership_probability"]) - 1, 64).astype(int)
        rms_values = []
        converged_values = []
        for index in indices:
            weights = bootstrap["membership_probability"][index]
            prediction = lens.exact_predictions(
                selected,
                final_fit["result"].x,
                final_fit["sources"],
                images,
                member_weights=weights,
                stage=f"member_bootstrap_{index}",
            )
            score = score_predictions(prediction, lens.sigma_image, n_full)
            rms_values.append(score["exact_radial_rms_arcsec"])
            converged_values.append(score["all_roots_converged"])
        quantiles = np.percentile(rms_values, [16, 50, 84])
        member_sensitivity = {
            "applicable": True,
            "vectors": 64,
            "preindexed_bootstrap_indices": indices.tolist(),
            "exact_radial_rms_p16_arcsec": float(quantiles[0]),
            "exact_radial_rms_median_arcsec": float(quantiles[1]),
            "exact_radial_rms_p84_arcsec": float(quantiles[2]),
            "all_vectors_all_roots_converged": bool(all(converged_values)),
            "gate_pass": bool(all(converged_values) and quantiles[2] <= 1.25),
        }

    thresholds = cfg["model_selection_and_advance_gates"]
    checks = {
        "selected_model_all_exact_roots_converged": bool(
            final_score["converged_roots"] == thresholds["selected_model_all_image_exact_root_count"]
        ),
        "selected_model_all_image_exact_radial_rms": bool(
            final_score["exact_radial_rms_arcsec"]
            <= thresholds["selected_model_maximum_all_image_exact_radial_rms_arcsec"]
        ),
        "selected_model_exact_coordinate_reduced_chi2": bool(
            final_score["exact_coordinate_reduced_chi2"]
            <= thresholds["selected_model_maximum_exact_coordinate_reduced_chi2"]
        ),
        "selected_model_standardized_coordinate_mean": bool(
            final_score["maximum_absolute_standardized_coordinate_mean"]
            <= thresholds["selected_model_maximum_absolute_standardized_source_closure_mean"]
        ),
        "no_lens_parameter_near_bound": bool(not bound_flags.any()),
        "covariance_symmetric_positive_semidefinite": bool(
            covariance_diagnostics["symmetric"] and covariance_diagnostics["positive_semidefinite"]
        ),
        "member_bootstrap_sensitivity": bool(member_sensitivity["gate_pass"]),
        "published_residual_or_mass_parameters_used": False,
        "gravity_response_or_new_force_fit": False,
    }
    advance = bool(
        checks["selected_model_all_exact_roots_converged"]
        and checks["selected_model_all_image_exact_radial_rms"]
        and checks["selected_model_exact_coordinate_reduced_chi2"]
        and checks["selected_model_standardized_coordinate_mean"]
        and checks["no_lens_parameter_near_bound"]
        and checks["covariance_symmetric_positive_semidefinite"]
        and checks["member_bootstrap_sensitivity"]
    )

    output_params = ROOT / cfg["outputs"]["parameter_table"]
    output_predictions = ROOT / cfg["outputs"]["image_predictions"]
    output_covariance = ROOT / cfg["outputs"]["parameter_covariance"]
    output_report = ROOT / cfg["outputs"]["report"]
    output_diagnostic = ROOT / cfg["outputs"]["diagnostic"]
    for path in (output_params, output_predictions, output_covariance, output_report):
        path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(parameter_rows).to_csv(output_params, index=False)
    pd.concat(heldout_predictions + [final_predictions], ignore_index=True).to_csv(
        output_predictions, index=False
    )
    pd.DataFrame(covariance, index=labels, columns=labels).to_csv(output_covariance)
    make_diagnostic(images, final_predictions, output_diagnostic)

    report = {
        "protocol_version": cfg["protocol_version"],
        "status": (
            "all_image_engineering_gate_passed_heldout_predictive_claim_not_authorized"
            if advance
            else "independent_lens_mass_family_inadequate_or_unidentified"
        ),
        "published_gr_mass_map_read": False,
        "published_best_fit_mass_parameters_used": False,
        "published_model_residual_used": False,
        "new_force_or_action_fit": False,
        "counts": {
            "images": len(images),
            "source_families": int(images["source_family"].nunique()),
            "training_images": len(training),
            "heldout_images": len(heldout),
            "member_candidates": len(members),
        },
        "training_fits": {
            kind: {
                "optimizer_success": bool(fit["result"].success),
                "optimizer_message": str(fit["result"].message),
                "multi_starts": starts,
                "best_total_cost": float(fit["result"].cost),
                "optimization_coordinate_rms_arcsec": fit["optimization_coordinate_rms_arcsec"],
                "optimization_radial_rms_arcsec": fit["optimization_radial_rms_arcsec"],
                "heldout_exact_score": fit["heldout_score"],
            }
            for kind, fit in train_fits.items()
        },
        "model_selection": {
            "selected_model": selected,
            "model_B_heldout_radial_rms_improvement_fraction": float(improvement),
            "minimum_improvement_fraction": 0.10,
            "model_B_all_heldout_roots_converged": bool(b_all_roots),
            "model_B_selected": select_b,
            "heldout_numeric_adequacy_threshold_predeclared": False,
            "predictive_validation_claim_authorized": False,
            "protocol_defect": "The frozen protocol used heldout images for nested-model selection but omitted a numeric heldout adequacy threshold. The heldout scores are therefore reported but cannot authorize a predictive or Weyl-response claim.",
        },
        "all_image_refit": {
            "optimizer_success": bool(final_fit["result"].success),
            "optimizer_message": str(final_fit["result"].message),
            "multi_starts": starts,
            "best_total_cost": float(final_fit["result"].cost),
            "optimization_coordinate_rms_arcsec": final_fit["optimization_coordinate_rms_arcsec"],
            "optimization_radial_rms_arcsec": final_fit["optimization_radial_rms_arcsec"],
            "exact_score": final_score,
            "lens_parameters": {
                label: float(value) for label, value in zip(spec.labels, final_fit["result"].x)
            },
            "lens_parameter_near_bound": {
                label: bool(value) for label, value in zip(spec.labels, bound_flags)
            },
            "profiled_source_positions_arcsec": {
                str(family): {"x": float(value[0]), "y": float(value[1])}
                for family, value in final_fit["sources"].items()
            },
        },
        "member_membership_sensitivity": member_sensitivity,
        "laplace_covariance": covariance_diagnostics,
        "advance_checks": checks,
        "independent_lens_engineering_gate_pass": advance,
        "heldout_predictive_closure_established": False,
        "weyl_response_reconstruction_authorized": False,
        "strict_r1_ready": False,
        "limitations": [
            "The exact score is conditional on the published image labels and does not model missing-image survey completeness.",
            "The central stellar term is a broad total-light nuisance because the BCG/ICL split was not identifiable.",
            "The hot-gas radial likelihood remains missing and is not silently replaced by a smooth lens component.",
            "The local Laplace covariance is not a replacement for a sampled, multimodal lens posterior.",
            "The frozen protocol omitted a heldout adequacy threshold; its seven-image holdout can select between the two nested models but cannot establish predictive closure.",
            "Passing this control would validate an observable-to-Weyl reconstruction route, not a new gravity theory.",
        ],
        "outputs": cfg["outputs"],
        "next_action": "Retain the all-image engineering fit and failed member-layer generalization as diagnostics. Correct the heldout-adequacy gate only on a fresh blind system or split; do not promote RX J2129 to a Weyl-response pilot or expand its lens family on the spent holdout.",
    }
    output_report.write_text(
        json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"selected_model": selected, "final_score": final_score, "advance": advance}, indent=2))


if __name__ == "__main__":
    main()
