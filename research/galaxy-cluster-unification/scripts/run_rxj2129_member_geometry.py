#!/usr/bin/env python3
"""Test RX J2129 member-galaxy geometry at fixed radial mass profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_rxj2129_raw_theory_lensing import (  # noqa: E402
    RawLens,
    load_images,
    score,
)
from voidscreen.member_lensing import (  # noqa: E402
    member_geometry_delta_deflection,
    point_mass_einstein_radius_squared_arcsec2,
)
from voidscreen.raw_lensing import (  # noqa: E402
    RadialDeflectionField,
    spherical_deflection_radians,
)


MODEL = "locked_universal_candidate"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def build_action_field(settings: dict, raw_protocol: dict) -> RadialDeflectionField:
    profiles = pd.read_csv(ROOT / settings["radial_profile"])
    cluster = profiles[
        profiles["model"].eq(settings["model"])
        & profiles["domain"].eq(settings["domain"])
    ].sort_values("radius_kpc")
    if cluster.empty:
        raise RuntimeError("selected action-model radial profile is empty")
    radius = cluster["radius_kpc"].to_numpy(float)
    acceleration = cluster["gpred_m_s2"].to_numpy(float)

    def lookup(target):
        return np.exp(np.interp(np.log(target), np.log(radius), np.log(acceleration)))

    impact_arcsec = np.geomspace(0.05, 500.0, 700)
    scale = float(raw_protocol["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
    physical_alpha = spherical_deflection_radians(
        impact_arcsec * scale,
        lookup,
        maximum_radius_kpc=float(radius.max()),
        integration_points=800,
    )
    return RadialDeflectionField(impact_arcsec, physical_alpha)


def load_members(settings: dict, raw_protocol: dict) -> pd.DataFrame:
    members = pd.read_csv(ROOT / settings["path"]).copy()
    geometry = raw_protocol["cosmology_and_coordinates"]
    cosine = np.cos(np.deg2rad(float(geometry["center_dec_deg"])))
    members["x_arcsec"] = (
        (members["ra_deg"].astype(float) - float(geometry["center_ra_deg"]))
        * 3600.0
        * cosine
    )
    members["y_arcsec"] = (
        members["dec_deg"].astype(float) - float(geometry["center_dec_deg"])
    ) * 3600.0
    members["radius_arcsec_recomputed"] = np.hypot(
        members["x_arcsec"], members["y_arcsec"]
    )
    members["expected_stellar_mass_msun"] = (
        members["stellar_mass_msun"].astype(float)
        * members["membership_probability"].astype(float)
    )
    catalog_size = (
        float(settings["catalog_shape_pixel_scale_arcsec"])
        * np.sqrt(members["a"].astype(float) * members["b"].astype(float))
    )
    members["base_softening_arcsec"] = np.maximum(
        float(settings["softening_floor_arcsec"]), catalog_size
    )
    required = [
        "x_arcsec",
        "y_arcsec",
        "expected_stellar_mass_msun",
        "base_softening_arcsec",
    ]
    if not np.isfinite(members[required].to_numpy(float)).all():
        raise RuntimeError("member catalog contains non-finite lens inputs")
    return members


class MemberGeometryLens(RawLens):
    """Raw lens plus a zero-radial-average member-geometry perturbation."""

    def __init__(
        self,
        protocol: dict,
        field: RadialDeflectionField,
        members: pd.DataFrame,
        *,
        layout_x_arcsec: np.ndarray,
        layout_y_arcsec: np.ndarray,
        mass_scale: float,
        softening_scale: float,
    ):
        super().__init__(protocol, {MODEL: field})
        self.member_x = np.asarray(layout_x_arcsec, dtype=float)
        self.member_y = np.asarray(layout_y_arcsec, dtype=float)
        self.member_mass = (
            members["expected_stellar_mass_msun"].to_numpy(float) * float(mass_scale)
        )
        self.member_softening = (
            members["base_softening_arcsec"].to_numpy(float)
            * float(softening_scale)
        )
        if not (
            self.member_x.shape
            == self.member_y.shape
            == self.member_mass.shape
            == self.member_softening.shape
        ):
            raise ValueError("member layout and property vectors must match")
        self.lens_angular_distance_m = float(
            self.cosmo.angular_diameter_distance(self.z_lens).to_value("m")
        )

    def alpha(
        self,
        model: str,
        parameters: np.ndarray,
        x_arcsec,
        y_arcsec,
        source_redshift: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        base_x, base_y = super().alpha(
            model, parameters, x_arcsec, y_arcsec, source_redshift
        )
        strength = point_mass_einstein_radius_squared_arcsec2(
            self.member_mass,
            lens_angular_distance_m=self.lens_angular_distance_m,
            distance_ratio=self.distance_ratio(source_redshift),
        )
        delta_x, delta_y = member_geometry_delta_deflection(
            x_arcsec,
            y_arcsec,
            self.member_x,
            self.member_y,
            strength,
            self.member_softening,
        )
        return base_x + delta_x, base_y + delta_y


def split_images(images: pd.DataFrame, raw_protocol: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    heldout_ids = set(raw_protocol["predictive_split"]["heldout"])
    heldout = images[images["image_id"].isin(heldout_ids)].copy()
    training = images[~images["image_id"].isin(heldout_ids)].copy()
    if len(training) != 15 or len(heldout) != 7:
        raise RuntimeError("predictive split changed")
    return training, heldout


def predictions_at_fixed_geometry(
    lens: RawLens,
    parameters: np.ndarray,
    training: pd.DataFrame,
    heldout: pd.DataFrame,
    *,
    label: str,
) -> tuple[pd.DataFrame, dict]:
    _, sources = lens.profiled_residuals(MODEL, parameters, training)
    training_prediction = lens.exact_predictions(
        MODEL, parameters, sources, training, stage="training"
    )
    heldout_prediction = lens.exact_predictions(
        MODEL, parameters, sources, heldout, stage="heldout"
    )
    prediction = pd.concat([training_prediction, heldout_prediction], ignore_index=True)
    prediction["variant"] = label
    scores = {
        "training": score(training_prediction, lens.sigma, free_parameters=20),
        "heldout": score(heldout_prediction, lens.sigma),
    }
    return prediction, scores


def fit_layout(
    lens: RawLens,
    training: pd.DataFrame,
    heldout: pd.DataFrame,
    *,
    label: str,
    starts: int,
    seed: int,
    initial: np.ndarray,
) -> tuple[pd.DataFrame, dict, dict]:
    fit = lens.fit(
        MODEL,
        training,
        starts=starts,
        seed=seed,
        initial_override=initial,
    )
    training_prediction = lens.exact_predictions(
        MODEL, fit["result"].x, fit["sources"], training, stage="training"
    )
    heldout_prediction = lens.exact_predictions(
        MODEL, fit["result"].x, fit["sources"], heldout, stage="heldout"
    )
    prediction = pd.concat([training_prediction, heldout_prediction], ignore_index=True)
    prediction["variant"] = label
    scores = {
        "training": score(training_prediction, lens.sigma, free_parameters=20),
        "heldout": score(heldout_prediction, lens.sigma),
    }
    fit_summary = {
        "parameters": fit["result"].x.copy(),
        "optimizer_success": bool(fit["result"].success),
        "optimizer_cost": float(fit["result"].cost),
    }
    return prediction, scores, fit_summary


def randomized_layout(members: pd.DataFrame, rng: np.random.Generator):
    radius = members["radius_arcsec_recomputed"].to_numpy(float)
    angle = rng.uniform(-np.pi, np.pi, len(members))
    return radius * np.cos(angle), radius * np.sin(angle)


def rms_value(scores: dict) -> float:
    value = scores["heldout"]["exact_radial_RMS_arcsec"]
    return float(value) if value is not None else float("inf")


def score_row(
    label: str,
    scores: dict,
    *,
    baseline_rms: float,
    mass_scale: float,
    softening_scale: float,
    geometry_refit: bool,
) -> dict:
    heldout_rms = rms_value(scores)
    return {
        "variant": label,
        "mass_scale": mass_scale,
        "softening_scale": softening_scale,
        "geometry_refit": geometry_refit,
        "training_RMS_arcsec": scores["training"]["exact_radial_RMS_arcsec"],
        "heldout_RMS_arcsec": heldout_rms,
        "heldout_converged_roots": scores["heldout"]["converged_roots"],
        "heldout_maximum_residual_arcsec": scores["heldout"][
            "maximum_radial_residual_arcsec"
        ],
        "fractional_heldout_improvement_vs_baseline": (
            (baseline_rms - heldout_rms) / baseline_rms
            if np.isfinite(heldout_rms)
            else -np.inf
        ),
    }


def image_diagnostics(
    baseline: pd.DataFrame,
    actual: pd.DataFrame,
    heldout: pd.DataFrame,
    members: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    baseline = baseline[baseline["stage"] == "heldout"].set_index("image_id")
    actual = actual[actual["stage"] == "heldout"].set_index("image_id")
    member_xy = members[["x_arcsec", "y_arcsec"]].to_numpy(float)
    rows = []
    for observed in heldout.itertuples(index=False):
        key = str(observed.image_id)
        before = baseline.loc[key]
        after = actual.loc[key]
        point = np.array([observed.x_arcsec, observed.y_arcsec], dtype=float)
        distances = np.linalg.norm(member_xy - point, axis=1)
        nearest_index = int(np.argmin(distances))
        nearest = members.iloc[nearest_index]
        needed = point - np.array(
            [before.predicted_x_arcsec, before.predicted_y_arcsec], dtype=float
        )
        achieved = np.array(
            [after.predicted_x_arcsec, after.predicted_y_arcsec], dtype=float
        ) - np.array([before.predicted_x_arcsec, before.predicted_y_arcsec], dtype=float)
        denominator = float(np.linalg.norm(needed) * np.linalg.norm(achieved))
        alignment = float(np.dot(needed, achieved) / denominator) if denominator > 0.0 else np.nan
        rows.append(
            {
                "image_id": key,
                "source_family": int(observed.source_family),
                "nearest_member_id": nearest["clash_id"],
                "nearest_member_distance_arcsec": float(distances[nearest_index]),
                "nearest_member_expected_stellar_mass_msun": float(
                    nearest["expected_stellar_mass_msun"]
                ),
                "baseline_residual_arcsec": float(before.radial_residual_arcsec),
                "actual_layout_residual_arcsec": float(after.radial_residual_arcsec),
                "residual_improvement_arcsec": float(
                    before.radial_residual_arcsec - after.radial_residual_arcsec
                ),
                "prediction_shift_arcsec": float(np.linalg.norm(achieved)),
                "shift_alignment_with_needed_correction": alignment,
            }
        )
    table = pd.DataFrame(rows)
    correlation = spearmanr(
        table["nearest_member_distance_arcsec"],
        table["residual_improvement_arcsec"],
    )
    summary = {
        "images_improved": int((table["residual_improvement_arcsec"] > 0.0).sum()),
        "images_worsened": int((table["residual_improvement_arcsec"] < 0.0).sum()),
        "median_prediction_shift_arcsec": float(table["prediction_shift_arcsec"].median()),
        "maximum_prediction_shift_arcsec": float(table["prediction_shift_arcsec"].max()),
        "nearest_distance_vs_improvement_spearman_r": float(correlation.statistic),
        "nearest_distance_vs_improvement_spearman_p": float(correlation.pvalue),
    }
    return table, summary


def empirical_p(values: np.ndarray, observed: float) -> float:
    finite_or_inf = np.asarray(values, dtype=float)
    return float((1 + np.count_nonzero(finite_or_inf <= observed)) / (len(values) + 1))


def distribution_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return {
        "draws": int(len(values)),
        "finite_scores": int(len(finite)),
        "median_RMS_arcsec": float(np.median(finite)) if len(finite) else None,
        "p16_RMS_arcsec": float(np.percentile(finite, 16.0)) if len(finite) else None,
        "p84_RMS_arcsec": float(np.percentile(finite, 84.0)) if len(finite) else None,
        "minimum_RMS_arcsec": float(np.min(finite)) if len(finite) else None,
        "maximum_RMS_arcsec": float(np.max(finite)) if len(finite) else None,
    }


def make_figure(
    members: pd.DataFrame,
    images: pd.DataFrame,
    baseline_prediction: pd.DataFrame,
    actual_prediction: pd.DataFrame,
    randomizations: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    ax = axes[0, 0]
    size = 12.0 + 28.0 * np.clip(
        np.log10(np.maximum(members["expected_stellar_mass_msun"], 1.0)) - 8.0,
        0.0,
        3.5,
    )
    ax.scatter(members["x_arcsec"], members["y_arcsec"], s=size, alpha=0.45, label="members")
    ax.scatter(
        images["x_arcsec"],
        images["y_arcsec"],
        marker="*",
        s=80,
        color="black",
        label="images",
    )
    ax.set(
        title="Measured member geometry",
        xlabel="east offset (arcsec)",
        ylabel="north offset (arcsec)",
    )
    ax.set_aspect("equal")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    observed = images.set_index("image_id")
    colors = {"smooth baseline": "#7570b3", "resolved members": "#d95f02"}
    for label, prediction in [
        ("smooth baseline", baseline_prediction),
        ("resolved members", actual_prediction),
    ]:
        held = prediction[prediction["stage"] == "heldout"]
        ox = np.array([observed.loc[key, "x_arcsec"] for key in held["image_id"]])
        oy = np.array([observed.loc[key, "y_arcsec"] for key in held["image_id"]])
        ax.quiver(
            ox,
            oy,
            held["predicted_x_arcsec"].to_numpy(float) - ox,
            held["predicted_y_arcsec"].to_numpy(float) - oy,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=colors[label],
            alpha=0.8,
            label=label,
        )
    ax.scatter(
        images[images["image_id"].isin(observed.index)]["x_arcsec"],
        images[images["image_id"].isin(observed.index)]["y_arcsec"],
        s=18,
        color="black",
    )
    ax.set(
        title="Heldout residual vectors",
        xlabel="east offset (arcsec)",
        ylabel="north offset (arcsec)",
    )
    ax.set_aspect("equal")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for mode, color in [("fixed_geometry", "#1b9e77"), ("refitted_geometry", "#377eb8")]:
        values = randomizations[randomizations["mode"] == mode]["heldout_RMS_arcsec"]
        finite = values[np.isfinite(values)]
        if len(finite):
            ax.hist(finite, bins=18, alpha=0.55, color=color, label=mode.replace("_", " "))
    actual_rms = float(
        np.sqrt(
            np.mean(
                actual_prediction.loc[
                    actual_prediction["stage"] == "heldout", "radial_residual_arcsec"
                ].to_numpy(float)
                ** 2
            )
        )
    )
    ax.axvline(actual_rms, color="#d95f02", linewidth=2.0, label="actual layout")
    ax.set(title="Randomized-angle controls", xlabel="heldout radial RMS (arcsec)", ylabel="draws")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.scatter(
        diagnostics["nearest_member_distance_arcsec"],
        diagnostics["residual_improvement_arcsec"],
        c=diagnostics["nearest_member_expected_stellar_mass_msun"],
        cmap="viridis",
        s=55,
    )
    for row in diagnostics.itertuples(index=False):
        ax.annotate(
            row.image_id,
            (row.nearest_member_distance_arcsec, row.residual_improvement_arcsec),
            fontsize=7,
        )
    ax.set(
        title="Which heldout images improve?",
        xlabel="distance to nearest member (arcsec)",
        ylabel="baseline residual minus new residual (arcsec)",
    )
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs" / "rxj2129_member_geometry_protocol.json",
    )
    parser.add_argument("--fixed-randomizations", type=int, default=None)
    parser.add_argument("--refit-randomizations", type=int, default=None)
    args = parser.parse_args()
    config_path = args.protocol.resolve()
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_member_geometry_scores":
        raise RuntimeError("member-geometry protocol was not frozen before scoring")
    raw_path = ROOT / protocol["base_model"]["raw_lensing_protocol"]
    raw_protocol = json.loads(raw_path.read_text(encoding="utf-8"))
    field = build_action_field(protocol["base_model"], raw_protocol)
    members = load_members(protocol["member_catalog"], raw_protocol)
    images = load_images(raw_protocol)
    training, heldout = split_images(images, raw_protocol)
    output = ROOT / "results" / "rxj2129_member_geometry"
    output.mkdir(parents=True, exist_ok=True)

    fit_settings = protocol["fits"]
    fit_seed = int(fit_settings["fit_seed"])
    baseline_lens = RawLens(raw_protocol, {MODEL: field})
    baseline_fit = baseline_lens.fit(
        MODEL,
        training,
        starts=int(fit_settings["baseline_starts"]),
        seed=fit_seed,
    )
    baseline_prediction = pd.concat(
        [
            baseline_lens.exact_predictions(
                MODEL,
                baseline_fit["result"].x,
                baseline_fit["sources"],
                training,
                stage="training",
            ),
            baseline_lens.exact_predictions(
                MODEL,
                baseline_fit["result"].x,
                baseline_fit["sources"],
                heldout,
                stage="heldout",
            ),
        ],
        ignore_index=True,
    )
    baseline_prediction["variant"] = "smooth_baseline"
    baseline_scores = {
        "training": score(
            baseline_prediction[baseline_prediction["stage"] == "training"],
            baseline_lens.sigma,
            free_parameters=20,
        ),
        "heldout": score(
            baseline_prediction[baseline_prediction["stage"] == "heldout"],
            baseline_lens.sigma,
        ),
    }
    baseline_rms = rms_value(baseline_scores)
    expected_baseline = float(protocol["base_model"]["expected_baseline_heldout_RMS_arcsec"])
    if abs(baseline_rms - expected_baseline) > float(
        protocol["base_model"]["baseline_tolerance_arcsec"]
    ):
        raise RuntimeError(
            f"baseline reproduction failed: measured {baseline_rms}, expected {expected_baseline}"
        )

    member_x = members["x_arcsec"].to_numpy(float)
    member_y = members["y_arcsec"].to_numpy(float)
    central_lens = MemberGeometryLens(
        raw_protocol,
        field,
        members,
        layout_x_arcsec=member_x,
        layout_y_arcsec=member_y,
        mass_scale=1.0,
        softening_scale=1.0,
    )
    actual_fixed_prediction, actual_fixed_scores = predictions_at_fixed_geometry(
        central_lens,
        baseline_fit["result"].x,
        training,
        heldout,
        label="central_catalog_fixed_geometry",
    )
    actual_matched_prediction, actual_matched_scores, _ = fit_layout(
        central_lens,
        training,
        heldout,
        label="central_catalog_matched_optimizer",
        starts=int(fit_settings["randomized_layout_starts"]),
        seed=fit_seed + 1,
        initial=baseline_fit["result"].x,
    )
    actual_prediction, actual_scores, actual_fit = fit_layout(
        central_lens,
        training,
        heldout,
        label="central_catalog",
        starts=int(fit_settings["actual_layout_starts"]),
        seed=fit_seed + 1,
        initial=baseline_fit["result"].x,
    )

    predictions = [
        baseline_prediction,
        actual_fixed_prediction,
        actual_matched_prediction,
        actual_prediction,
    ]
    variant_rows = [
        score_row(
            "smooth_baseline",
            baseline_scores,
            baseline_rms=baseline_rms,
            mass_scale=0.0,
            softening_scale=1.0,
            geometry_refit=True,
        ),
        score_row(
            "central_catalog_fixed_geometry",
            actual_fixed_scores,
            baseline_rms=baseline_rms,
            mass_scale=1.0,
            softening_scale=1.0,
            geometry_refit=False,
        ),
        score_row(
            "central_catalog_matched_optimizer",
            actual_matched_scores,
            baseline_rms=baseline_rms,
            mass_scale=1.0,
            softening_scale=1.0,
            geometry_refit=True,
        ),
        score_row(
            "central_catalog",
            actual_scores,
            baseline_rms=baseline_rms,
            mass_scale=1.0,
            softening_scale=1.0,
            geometry_refit=True,
        ),
    ]
    sensitivity_scores = {}
    for index, sensitivity in enumerate(protocol["predeclared_sensitivities"]):
        label = sensitivity["label"]
        if label == "central_catalog":
            sensitivity_scores[label] = actual_scores
            continue
        lens = MemberGeometryLens(
            raw_protocol,
            field,
            members,
            layout_x_arcsec=member_x,
            layout_y_arcsec=member_y,
            mass_scale=float(sensitivity["mass_scale"]),
            softening_scale=float(sensitivity["softening_scale"]),
        )
        prediction, scores, _ = fit_layout(
            lens,
            training,
            heldout,
            label=label,
            starts=int(fit_settings["sensitivity_starts"]),
            seed=fit_seed + 10 + index,
            initial=actual_fit["parameters"],
        )
        predictions.append(prediction)
        sensitivity_scores[label] = scores
        variant_rows.append(
            score_row(
                label,
                scores,
                baseline_rms=baseline_rms,
                mass_scale=float(sensitivity["mass_scale"]),
                softening_scale=float(sensitivity["softening_scale"]),
                geometry_refit=True,
            )
        )

    random_settings = protocol["geometry_control"]
    fixed_draws = (
        int(args.fixed_randomizations)
        if args.fixed_randomizations is not None
        else int(random_settings["fixed_geometry_randomizations"])
    )
    refit_draws = (
        int(args.refit_randomizations)
        if args.refit_randomizations is not None
        else int(random_settings["refitted_geometry_randomizations"])
    )
    rng = np.random.default_rng(int(random_settings["random_seed"]))
    randomized_rows = []
    saved_layouts = [randomized_layout(members, rng) for _ in range(max(fixed_draws, refit_draws))]
    for index in range(fixed_draws):
        random_x, random_y = saved_layouts[index]
        lens = MemberGeometryLens(
            raw_protocol,
            field,
            members,
            layout_x_arcsec=random_x,
            layout_y_arcsec=random_y,
            mass_scale=1.0,
            softening_scale=1.0,
        )
        _, scores = predictions_at_fixed_geometry(
            lens,
            baseline_fit["result"].x,
            training,
            heldout,
            label=f"random_fixed_{index:03d}",
        )
        randomized_rows.append(
            {
                "mode": "fixed_geometry",
                "draw": index,
                "heldout_RMS_arcsec": rms_value(scores),
                "heldout_converged_roots": scores["heldout"]["converged_roots"],
            }
        )
    for index in range(refit_draws):
        random_x, random_y = saved_layouts[index]
        lens = MemberGeometryLens(
            raw_protocol,
            field,
            members,
            layout_x_arcsec=random_x,
            layout_y_arcsec=random_y,
            mass_scale=1.0,
            softening_scale=1.0,
        )
        _, scores, _ = fit_layout(
            lens,
            training,
            heldout,
            label=f"random_refit_{index:03d}",
            starts=int(fit_settings["randomized_layout_starts"]),
            seed=fit_seed + 1000 + index,
            initial=baseline_fit["result"].x,
        )
        randomized_rows.append(
            {
                "mode": "refitted_geometry",
                "draw": index,
                "heldout_RMS_arcsec": rms_value(scores),
                "heldout_converged_roots": scores["heldout"]["converged_roots"],
            }
        )

    prediction_table = pd.concat(predictions, ignore_index=True)
    variant_table = pd.DataFrame(variant_rows)
    randomization_table = pd.DataFrame(randomized_rows)
    diagnostics, diagnostic_summary = image_diagnostics(
        baseline_prediction, actual_prediction, heldout, members
    )
    actual_rms = rms_value(actual_scores)
    actual_fixed_rms = rms_value(actual_fixed_scores)
    actual_matched_rms = rms_value(actual_matched_scores)
    fixed_values = randomization_table[
        randomization_table["mode"] == "fixed_geometry"
    ]["heldout_RMS_arcsec"].to_numpy(float)
    refit_values = randomization_table[
        randomization_table["mode"] == "refitted_geometry"
    ]["heldout_RMS_arcsec"].to_numpy(float)
    fixed_p = empirical_p(fixed_values, actual_fixed_rms)
    refit_p = empirical_p(refit_values, actual_matched_rms)
    fractional_improvement = (baseline_rms - actual_rms) / baseline_rms
    gates = protocol["interpretation_gates"]
    meaningful = bool(
        fractional_improvement
        >= float(gates["meaningful_fractional_heldout_RMS_improvement"])
    )
    strong = bool(actual_rms <= float(gates["strong_absolute_heldout_RMS_arcsec"]))
    arrangement_specific = bool(refit_p <= float(gates["randomization_empirical_p_max"]))

    if meaningful and arrangement_specific:
        primary_concept = (
            "The measured member arrangement carries lensing information beyond the "
            "radial profile; "
            "a two-dimensional coupled Sigma solve is justified."
        )
        equation_decision = (
            "Do not add a new term yet. First let the existing Sigma equation respond "
            "to the measured "
            "two-dimensional baryon field."
        )
    elif meaningful:
        primary_concept = (
            "Clumpiness helps, but the real member angles are not special relative to "
            "randomized layouts; the effect is generic substructure rather than evidence "
            "for conflicting observed pulls."
        )
        equation_decision = (
            "Upgrade the two-dimensional mass model before changing the Sigma equation."
        )
    else:
        primary_concept = (
            "Redistributing the catalogued stellar mass into the measured galaxy positions "
            "is insufficient "
            "to repair the raw cluster lensing residuals."
        )
        equation_decision = (
            "Do not encode galaxy count or conflicting directions as a new Sigma term on "
            "this evidence. The next discriminating test must be a full nonlinear "
            "two-dimensional Sigma solve or a "
            "relativistic photon-coupling test."
        )

    used_columns = [
        "clash_id",
        "ra_deg",
        "dec_deg",
        "x_arcsec",
        "y_arcsec",
        "radius_arcsec_recomputed",
        "membership_probability",
        "stellar_mass_msun",
        "expected_stellar_mass_msun",
        "a",
        "b",
        "base_softening_arcsec",
    ]
    members[used_columns].to_csv(output / "member_catalog_used.csv", index=False)
    prediction_table.to_csv(output / "image_predictions.csv", index=False)
    variant_table.to_csv(output / "variant_scores.csv", index=False)
    randomization_table.to_csv(output / "randomization_scores.csv", index=False)
    diagnostics.to_csv(output / "image_diagnostics.csv", index=False)
    make_figure(
        members,
        images,
        baseline_prediction,
        actual_prediction,
        randomization_table,
        diagnostics,
        output / "member_geometry_test.png",
    )

    report = {
        "report_version": protocol["protocol_version"],
        "status": "completed fixed-radial-profile member-geometry test",
        "protocol": {
            "path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(config_path),
            "freeze_status": protocol["status"],
        },
        "inputs": {
            "raw_lensing_protocol": {
                "path": str(raw_path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(raw_path),
            },
            "radial_profile": protocol["base_model"]["radial_profile"],
            "radial_profile_sha256": sha256(ROOT / protocol["base_model"]["radial_profile"]),
            "member_catalog": protocol["member_catalog"]["path"],
            "member_catalog_sha256": sha256(ROOT / protocol["member_catalog"]["path"]),
            "members": int(len(members)),
            "expected_member_stellar_mass_msun": float(
                members["expected_stellar_mass_msun"].sum()
            ),
        },
        "controlled_change": {
            "operation": (
                "resolved member deflection minus analytic azimuthal average of the same members"
            ),
            "net_added_member_mass_msun": 0.0,
            "azimuthally_averaged_radial_deflection_change": 0.0,
            "gravity_or_lensing_amplitudes_fit_to_images": 0,
            "geometry_nuisances_refit": 6,
        },
        "headline_scores": {
            "smooth_baseline_heldout_RMS_arcsec": baseline_rms,
            "actual_layout_fixed_geometry_heldout_RMS_arcsec": actual_fixed_rms,
            "actual_layout_matched_optimizer_heldout_RMS_arcsec": actual_matched_rms,
            "actual_layout_refitted_geometry_heldout_RMS_arcsec": actual_rms,
            "fractional_improvement_after_full_geometry_refit": fractional_improvement,
            "actual_layout_below_1_arcsec": strong,
            "fixed_geometry_randomization_empirical_p": fixed_p,
            "refitted_geometry_randomization_empirical_p": refit_p,
        },
        "randomization_controls": {
            "fixed_geometry": distribution_summary(fixed_values),
            "refitted_geometry": distribution_summary(refit_values),
        },
        "per_image_diagnostics": diagnostic_summary,
        "predeclared_sensitivities": {
            row["variant"]: row
            for row in variant_rows
            if row["variant"]
            not in {
                "smooth_baseline",
                "central_catalog_fixed_geometry",
                "central_catalog_matched_optimizer",
            }
        },
        "interpretation_gates": {
            **gates,
            "meaningful_improvement_passed": meaningful,
            "strong_absolute_score_passed": strong,
            "observed_arrangement_specificity_passed": arrangement_specific,
        },
        "concepts_learned": {
            "primary": primary_concept,
            "equation_decision": equation_decision,
            "what_this_tests": (
                "Whether directional baryonic pulls omitted by a radial model explain "
                "raw image-position errors."
            ),
            "what_this_does_not_test": (
                "Nonlinear overlap of Sigma fields, three-dimensional member depths, "
                "gas-map asymmetry, or a covariant light-bending law."
            ),
        },
        "scope": protocol["scope"],
        "outputs": protocol["outputs"],
    }
    (output / "report.json").write_text(
        json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_safe(report["headline_scores"]), indent=2), flush=True)
    print(primary_concept, flush=True)


if __name__ == "__main__":
    main()
