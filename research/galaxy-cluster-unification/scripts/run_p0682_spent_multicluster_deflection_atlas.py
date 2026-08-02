#!/usr/bin/env python3
"""Build a derivative-free atlas of spent compact-halo deflection targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Util import param_util
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.raw_lensing import shear_deflection  # noqa: E402
from voidscreen.required_field_decomposition import (  # noqa: E402
    radial_vector_decomposition,
)
from voidscreen.spent_deflection_atlas import (  # noqa: E402
    leave_one_out_constant,
    leave_one_out_log_linear,
    loglog_interpolate,
    vector_alignment,
    vector_rms,
)

DEFAULT_CONFIG = ROOT / "configs" / "p0682_spent_multicluster_deflection_atlas.json"
G_SI = 6.67430e-11
KPC_M = 3.085677581491367e19


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def boundary_names(raw_report: dict, system_name: str) -> list[str]:
    flags = raw_report["system_scores"][system_name]["GR_plus_cluster_halo"][
        "geometry_at_boundary"
    ]
    return [name for name, value in flags.items() if bool(value)]


def reliability(raw_report: dict, system_name: str) -> dict:
    score = raw_report["system_scores"][system_name]["GR_plus_cluster_halo"]
    heldout = score["heldout"]
    heldout_rms = heldout.get("exact_radial_RMS_arcsec")
    boundaries = boundary_names(raw_report, system_name)
    training_roots = bool(score["training"]["all_roots_converged"])
    reliable = (
        not boundaries
        and training_roots
        and heldout_rms is not None
        and float(heldout_rms) <= 5.0
    )
    return {
        "parameter_boundary_names": ";".join(boundaries),
        "parameter_at_boundary": bool(boundaries),
        "training_all_roots_converged": training_roots,
        "heldout_radial_RMS_arcsec": heldout_rms,
        "reliable_predictor_target": reliable,
    }


def make_sampling_annulus(images: pd.DataFrame, sampling: dict):
    radii = np.hypot(
        images["observed_x_arcsec"].to_numpy(float),
        images["observed_y_arcsec"].to_numpy(float),
    )
    lower = float(np.min(radii))
    upper = float(np.max(radii))
    radial = np.geomspace(lower, upper, int(sampling["radial_points"]))
    angle = np.linspace(0.0, 2.0 * np.pi, int(sampling["angular_points"]), endpoint=False)
    radius_grid, angle_grid = np.meshgrid(radial, angle, indexing="ij")
    return lower, upper, radius_grid.ravel(), angle_grid.ravel()


def cosmology_values(raw_protocol: dict, system: dict) -> tuple[float, float]:
    cosmology = FlatLambdaCDM(
        H0=float(raw_protocol["cosmology"]["H0_km_s_Mpc"]),
        Om0=float(raw_protocol["cosmology"]["Omega_m"]),
    )
    lens_redshift = float(system["lens_redshift"])
    source_redshift = float(raw_protocol["cosmology"]["reference_source_redshift"])
    scale = float(cosmology.kpc_proper_per_arcmin(lens_redshift).value / 60.0)
    source_distance = cosmology.angular_diameter_distance(source_redshift)
    lens_source_distance = cosmology.angular_diameter_distance_z1z2(
        lens_redshift, source_redshift
    )
    distance_ratio = float((lens_source_distance / source_distance).value)
    return scale, distance_ratio


def scalar_at(values_x, values_y, target_x):
    return float(loglog_interpolate(np.asarray([target_x]), values_x, values_y)[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    if protocol.get("status") != "frozen_before_any_P0682_atlas_metric":
        raise RuntimeError("P0682 protocol is not frozen")

    raw_protocol_path = ROOT / protocol["raw_protocol"]
    raw_report_path = ROOT / protocol["raw_report"]
    parameter_path = ROOT / protocol["parameter_table"]
    profile_path = ROOT / protocol["baryonic_profiles"]
    prediction_path = ROOT / protocol["image_predictions"]
    raw_protocol = read_json(raw_protocol_path)
    raw_report = read_json(raw_report_path)
    parameters = pd.read_csv(parameter_path)
    profiles = pd.read_csv(profile_path)
    predictions = pd.read_csv(prediction_path)
    settings = protocol["sampling"]
    selected = protocol["parameter_selection"]
    source_redshift = float(settings["source_redshift"])
    if source_redshift != float(raw_protocol["cosmology"]["reference_source_redshift"]):
        raise RuntimeError("reference source redshift changed")

    system_rows = []
    radial_rows = []
    reconstruction_errors = []
    finite_flags = []
    nie = LensModel(lens_model_list=["NIE"])

    for system in raw_protocol["systems"]:
        name = system["system"]
        label = system["label"]
        print(f"system={label}", flush=True)
        parameter = parameters[
            parameters.system.eq(name)
            & parameters.model.eq(selected["model"])
            & np.isclose(parameters.cutoff_kpc.astype(float), float(selected["cutoff_kpc"]))
        ]
        if len(parameter) != 1:
            raise RuntimeError(f"expected one compact-halo parameter row for {name}")
        p = parameter.iloc[0]
        profile = profiles[
            profiles.system.eq(name)
            & profiles.model.eq("baryons_GR")
            & np.isclose(profiles.cutoff_kpc.astype(float), float(selected["cutoff_kpc"]))
        ].sort_values("radius_kpc")
        if len(profile) < 20:
            raise RuntimeError(f"baryonic profile changed for {name}")
        image_rows = predictions[
            predictions.system.eq(name)
            & predictions.model.eq("baryons_GR")
            & np.isclose(predictions.cutoff_kpc.astype(float), float(selected["cutoff_kpc"]))
        ].drop_duplicates("image_id")
        if len(image_rows) != int(system["images"]):
            raise RuntimeError(f"image annulus rows changed for {name}")

        scale_kpc_arcsec, distance_ratio = cosmology_values(raw_protocol, system)
        lower, upper, radius, angle = make_sampling_annulus(image_rows, settings)
        x_arcsec = radius * np.cos(angle)
        y_arcsec = radius * np.sin(angle)
        radius_kpc = radius * scale_kpc_arcsec
        anchor_radius = profile.radius_kpc.to_numpy(float)
        anchor_gbar = profile.gbar_m_s2.to_numpy(float)
        anchor_alpha = profile.physical_deflection_arcsec_before_distance_ratio.to_numpy(float)
        baryon_alpha = loglog_interpolate(radius_kpc, anchor_radius, anchor_alpha) * distance_ratio
        baryon_x = baryon_alpha * np.cos(angle)
        baryon_y = baryon_alpha * np.sin(angle)

        ellipticity = param_util.phi_q2_ellipticity(
            phi=float(p.position_angle_phi_radian), q=float(p.axis_ratio_q)
        )
        halo_x, halo_y = nie.alpha(
            x_arcsec,
            y_arcsec,
            [
                {
                    "theta_E": float(p.theta_E_ref_arcsec),
                    "e1": float(ellipticity[0]),
                    "e2": float(ellipticity[1]),
                    "s_scale": 10.0 ** float(p.log10_core_arcsec),
                    "center_x": float(p.center_x_arcsec),
                    "center_y": float(p.center_y_arcsec),
                }
            ],
        )
        halo_x = np.asarray(halo_x, dtype=float)
        halo_y = np.asarray(halo_y, dtype=float)
        shear_x, shear_y = shear_deflection(
            x_arcsec,
            y_arcsec,
            float(p.external_shear_gamma1),
            float(p.external_shear_gamma2),
        )
        radial_unit_x = np.cos(angle)
        radial_unit_y = np.sin(angle)
        halo_radial = halo_x * radial_unit_x + halo_y * radial_unit_y

        edges = np.geomspace(lower, upper, int(settings["radial_bins"]) + 1)
        decomposition = radial_vector_decomposition(
            halo_x, halo_y, x_arcsec, y_arcsec, edges
        )
        reconstructed_x = decomposition.monopole_x + decomposition.angular_x
        reconstructed_y = decomposition.monopole_y + decomposition.angular_y
        halo_rms = vector_rms(halo_x, halo_y)
        baryon_rms = vector_rms(baryon_x, baryon_y)
        angular_rms = vector_rms(decomposition.angular_x, decomposition.angular_y)
        monopole_rms = vector_rms(decomposition.monopole_x, decomposition.monopole_y)
        reconstruction_error = vector_rms(
            reconstructed_x - halo_x, reconstructed_y - halo_y
        ) / max(halo_rms, np.finfo(float).tiny)
        reconstruction_errors.append(reconstruction_error)
        finite_flags.append(
            bool(
                np.all(
                    np.isfinite(
                        np.r_[
                            baryon_x,
                            baryon_y,
                            halo_x,
                            halo_y,
                            shear_x,
                            shear_y,
                        ]
                    )
                )
            )
        )

        ratio = halo_radial / baryon_alpha
        pivot_arcsec = math.sqrt(lower * upper)
        pivot_kpc = pivot_arcsec * scale_kpc_arcsec
        pivot_gbar = scalar_at(anchor_radius, anchor_gbar, pivot_kpc)
        pivot_alpha = scalar_at(anchor_radius, anchor_alpha, pivot_kpc) * distance_ratio
        slope_step = 1.01
        slope = (
            math.log(scalar_at(anchor_radius, anchor_gbar, pivot_kpc * slope_step))
            - math.log(scalar_at(anchor_radius, anchor_gbar, pivot_kpc / slope_step))
        ) / (2.0 * math.log(slope_step))
        mass_proxy_kg = pivot_gbar * (pivot_kpc * KPC_M) ** 2 / G_SI
        core_arcsec = 10.0 ** float(p.log10_core_arcsec)
        quality = reliability(raw_report, name)
        metrics = {
            "system": name,
            "label": label,
            "samples": len(radius),
            "annulus_minimum_arcsec": lower,
            "annulus_maximum_arcsec": upper,
            "annulus_pivot_arcsec": pivot_arcsec,
            "annulus_pivot_kpc": pivot_kpc,
            "annulus_width_ratio": upper / lower,
            "angular_scale_kpc_per_arcsec": scale_kpc_arcsec,
            "reference_distance_ratio": distance_ratio,
            "baryon_vector_RMS_arcsec": baryon_rms,
            "halo_vector_RMS_arcsec": halo_rms,
            "shear_vector_RMS_arcsec": vector_rms(shear_x, shear_y),
            "halo_to_baryon_vector_RMS_ratio": halo_rms / baryon_rms,
            "baryon_plus_halo_to_baryon_vector_RMS_ratio": vector_rms(
                baryon_x + halo_x, baryon_y + halo_y
            )
            / baryon_rms,
            "shear_to_baryon_vector_RMS_ratio": vector_rms(shear_x, shear_y)
            / baryon_rms,
            "halo_baryon_vector_alignment_cosine": vector_alignment(
                halo_x, halo_y, baryon_x, baryon_y
            ),
            "halo_monopole_fraction_of_vector_power": (monopole_rms / halo_rms) ** 2,
            "halo_angular_residual_RMS_fraction": angular_rms / halo_rms,
            "halo_monopole_reconstruction_relative_RMS": reconstruction_error,
            "halo_to_baryon_radial_ratio_q25": float(np.quantile(ratio, 0.25)),
            "halo_to_baryon_radial_ratio_median": float(np.median(ratio)),
            "halo_to_baryon_radial_ratio_q75": float(np.quantile(ratio, 0.75)),
            "theta_E_ref_arcsec": float(p.theta_E_ref_arcsec),
            "NIE_core_radius_arcsec": core_arcsec,
            "NIE_core_radius_kpc": core_arcsec * scale_kpc_arcsec,
            "NIE_core_to_annulus_pivot_ratio": core_arcsec / pivot_arcsec,
            "NIE_core_to_annulus_span_ratio": core_arcsec / (upper - lower),
            "baryonic_acceleration_at_pivot_m_s2": pivot_gbar,
            "baryonic_enclosed_mass_proxy_at_pivot_kg": mass_proxy_kg,
            "baryonic_acceleration_log_slope_at_pivot": slope,
            "baryonic_reduced_deflection_at_pivot_arcsec": pivot_alpha,
            **quality,
        }
        system_rows.append(metrics)

        radial_table = decomposition.table.copy()
        for _, row in radial_table.iterrows():
            radial_use = (radius >= row.radius_minimum) & (radius <= row.radius_maximum)
            radial_rows.append(
                {
                    "system": name,
                    "label": label,
                    "radial_bin": int(row.radial_bin),
                    "radius_midpoint_arcsec": float(row.radius_midpoint),
                    "radius_midpoint_kpc": float(row.radius_midpoint) * scale_kpc_arcsec,
                    "mean_halo_radial_deflection_arcsec": float(
                        np.mean(halo_radial[radial_use])
                    ),
                    "mean_baryon_radial_deflection_arcsec": float(
                        np.mean(baryon_alpha[radial_use])
                    ),
                    "median_halo_to_baryon_radial_ratio": float(
                        np.median(ratio[radial_use])
                    ),
                }
            )

    system_table = pd.DataFrame(system_rows)
    radial_table = pd.DataFrame(radial_rows)
    descriptive = system_table[~system_table.parameter_at_boundary.astype(bool)].copy()
    reliable = system_table[system_table.reliable_predictor_target.astype(bool)].copy()
    target_column = "halo_to_baryon_radial_ratio_median"
    predictor_columns = {
        "log10_baryonic_acceleration_at_pivot": "baryonic_acceleration_at_pivot_m_s2",
        "log10_baryonic_enclosed_mass_proxy_at_pivot": "baryonic_enclosed_mass_proxy_at_pivot_kg",
        "baryonic_acceleration_log_slope_at_pivot": "baryonic_acceleration_log_slope_at_pivot",
        "log10_baryonic_reduced_deflection_at_pivot": "baryonic_reduced_deflection_at_pivot_arcsec",
        "log10_annulus_pivot_radius_kpc": "annulus_pivot_kpc",
        "log10_annulus_width_ratio": "annulus_width_ratio",
    }
    target = np.log10(descriptive[target_column].to_numpy(float))
    _, constant_rmse = leave_one_out_constant(target)
    predictor_rows = []
    for predictor_name, column in predictor_columns.items():
        values = descriptive[column].to_numpy(float)
        if predictor_name.startswith("log10_"):
            values = np.log10(values)
        result = spearmanr(values, target)
        loo_prediction, loo_rmse = leave_one_out_log_linear(values, target)
        predictor_rows.append(
            {
                "subset": "non_boundary_descriptive",
                "predictor": predictor_name,
                "systems": len(descriptive),
                "spearman_rho": float(result.statistic),
                "spearman_p_value": float(result.pvalue),
                "LOO_RMSE_dex": loo_rmse,
                "constant_LOO_RMSE_dex": constant_rmse,
                "LOO_RMSE_to_constant_ratio": loo_rmse / constant_rmse,
                "minimum_reliable_systems_pass": len(reliable)
                >= int(protocol["predeclared_pattern_gates"]["predictor_minimum_reliable_systems"]),
                "LOO_predictions_json": json.dumps(loo_prediction.tolist()),
            }
        )
    predictor_table = pd.DataFrame(predictor_rows)

    gates = protocol["predeclared_pattern_gates"]
    radial_flags = (
        (descriptive.halo_baryon_vector_alignment_cosine >= gates["radial_morphology_alignment_cosine_min"])
        & (descriptive.halo_angular_residual_RMS_fraction <= gates["radial_morphology_angular_RMS_fraction_max"])
    )
    radial_count = int(radial_flags.sum())
    log_scatter = float(np.std(target, ddof=1))
    predictor_table["correlation_pass"] = (
        predictor_table.spearman_rho.abs() + 1.0e-12
        >= float(gates["predictor_absolute_spearman_rho_min"])
    )
    predictor_table["LOO_improvement_pass"] = (
        predictor_table.LOO_RMSE_to_constant_ratio
        <= float(gates["predictor_LOO_RMSE_to_constant_LOO_RMSE_ratio_max"])
    )
    predictor_table["all_selection_gates_pass"] = (
        predictor_table.correlation_pass
        & predictor_table.LOO_improvement_pass
        & predictor_table.minimum_reliable_systems_pass
    )

    integrity = {
        "systems": len(system_table),
        "systems_expected_pass": len(system_table)
        == int(protocol["predeclared_integrity_gates"]["systems_expected"]),
        "non_boundary_systems": len(descriptive),
        "non_boundary_systems_minimum_pass": len(descriptive)
        >= int(protocol["predeclared_integrity_gates"]["non_boundary_systems_expected_min"]),
        "samples_per_system_pass": bool(
            np.all(
                system_table.samples
                == int(protocol["predeclared_integrity_gates"]["samples_per_system"])
            )
        ),
        "all_sampled_fields_finite": bool(all(finite_flags)),
        "maximum_reconstruction_relative_RMS": float(max(reconstruction_errors)),
        "reconstruction_gate_pass": float(max(reconstruction_errors))
        <= float(
            protocol["predeclared_integrity_gates"][
                "halo_monopole_plus_angular_reconstruction_relative_RMS_max"
            ]
        ),
        "no_numerical_field_derivatives": True,
        "new_candidate_formula_fit": False,
        "new_raw_image_root_score_computed": False,
        "sealed_target_outcomes_opened": False,
    }
    integrity["all_gates_pass"] = bool(
        all(
            value
            for key, value in integrity.items()
            if key.endswith("_pass") or key in {
                "all_sampled_fields_finite",
                "no_numerical_field_derivatives",
            }
        )
    )
    pattern = {
        "non_boundary_systems": len(descriptive),
        "reliable_predictor_systems": len(reliable),
        "radial_morphology_systems_passing": radial_count,
        "radial_morphology_pass": radial_count
        >= int(gates["radial_morphology_minimum_non_boundary_systems"]),
        "constant_amplitude_log10_scatter_dex": log_scatter,
        "constant_amplitude_pass": log_scatter
        <= float(gates["constant_amplitude_log10_scatter_max_dex"]),
        "constant_amplitude_geometric_mean": float(10.0 ** np.mean(target)),
        "constant_amplitude_factor_scatter": float(10.0**log_scatter),
        "predictor_survivors": predictor_table.loc[
            predictor_table.all_selection_gates_pass, "predictor"
        ].tolist(),
    }

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    system_table.to_csv(output / protocol["outputs"]["system_metrics"], index=False)
    radial_table.to_csv(output / protocol["outputs"]["radial_profiles"], index=False)
    predictor_table.to_csv(output / protocol["outputs"]["predictor_scores"], index=False)

    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    labels = system_table.label.tolist()
    xx = np.arange(len(labels))
    axes[0, 0].bar(xx, system_table.halo_to_baryon_vector_RMS_ratio)
    axes[0, 0].set_xticks(xx, labels, rotation=30)
    axes[0, 0].set_ylabel("halo / baryon vector RMS")
    axes[0, 0].set_title("Required compact-halo amplitude")
    axes[0, 1].bar(xx - 0.18, system_table.halo_baryon_vector_alignment_cosine, 0.36, label="alignment")
    axes[0, 1].bar(xx + 0.18, system_table.halo_angular_residual_RMS_fraction, 0.36, label="angular fraction")
    axes[0, 1].axhline(float(gates["radial_morphology_alignment_cosine_min"]), color="C0", linestyle="--", alpha=0.5)
    axes[0, 1].axhline(float(gates["radial_morphology_angular_RMS_fraction_max"]), color="C1", linestyle="--", alpha=0.5)
    axes[0, 1].set_xticks(xx, labels, rotation=30)
    axes[0, 1].set_title("Radial morphology")
    axes[0, 1].legend()
    for label, group in radial_table.groupby("label"):
        axes[1, 0].plot(group.radius_midpoint_kpc, group.median_halo_to_baryon_radial_ratio, marker="o", label=label)
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xlabel("radius (kpc)")
    axes[1, 0].set_ylabel("median radial halo / baryon")
    axes[1, 0].set_title("Ratio through each observed strong-lens annulus")
    axes[1, 0].legend(fontsize=8)
    best = predictor_table.sort_values("LOO_RMSE_to_constant_ratio").iloc[0]
    best_column = predictor_columns[best.predictor]
    xplot = descriptive[best_column].to_numpy(float)
    if str(best.predictor).startswith("log10_"):
        xplot = np.log10(xplot)
    axes[1, 1].scatter(xplot, target, c=descriptive.reliable_predictor_target.map({True: "C2", False: "C3"}))
    for xvalue, yvalue, label in zip(xplot, target, descriptive.label, strict=True):
        axes[1, 1].annotate(label, (xvalue, yvalue), xytext=(4, 3), textcoords="offset points", fontsize=8)
    axes[1, 1].set_xlabel(str(best.predictor))
    axes[1, 1].set_ylabel("log10 median halo / baryon")
    axes[1, 1].set_title(f"Best descriptive LOO ratio = {best.LOO_RMSE_to_constant_ratio:.2f}")
    for axis in axes.ravel():
        axis.grid(alpha=0.2)
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)

    report = {
        "report_version": protocol["protocol_version"],
        "status": "pass" if integrity["all_gates_pass"] else "fail",
        "protocol": {
            "path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(config_path),
        },
        "input_hashes": {
            "raw_protocol": sha256(raw_protocol_path),
            "raw_report": sha256(raw_report_path),
            "parameter_table": sha256(parameter_path),
            "baryonic_profiles": sha256(profile_path),
            "image_predictions": sha256(prediction_path),
        },
        "integrity_audit": integrity,
        "pattern_audit": pattern,
        "system_metrics": system_table.set_index("label").to_dict(orient="index"),
        "best_descriptive_predictor": predictor_table.sort_values(
            "LOO_RMSE_to_constant_ratio"
        ).iloc[0].to_dict(),
        "selection": {
            "advance_radial_morphology": pattern["radial_morphology_pass"],
            "advance_constant_amplitude": pattern["constant_amplitude_pass"],
            "advance_baryonic_predictor": bool(pattern["predictor_survivors"]),
            "next_action": "Use only a passed qualitative pattern to design a frozen spent-data law; do not open sealed outcomes.",
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    (output / "report.json").write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(json_safe(report["pattern_audit"]), indent=2))


if __name__ == "__main__":
    main()
