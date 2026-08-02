#!/usr/bin/env python3
"""Decompose the spent RX J2129 compact-halo field on the absolute field grid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Util import param_util

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_adaptive_route_raw_rxj2129 import json_safe
from run_p0660_exact_tensor_activation_audit import sha256
from run_rxj2129_raw_theory_lensing import RawLens

from voidscreen.compound_activation_3d import exact_compound_path_activation_3d
from voidscreen.metric_lensing_3d import KPC_M
from voidscreen.raw_lensing import shear_deflection
from voidscreen.required_field_decomposition import (
    angular_harmonics,
    convergence_and_jacobian_determinant,
    positive_weight_radius_quantile,
    predictor_correlations,
    radial_vector_decomposition,
    sign_change_cells,
    vector_rms,
)

DEFAULT_CONFIG = (
    ROOT / "configs" / "p0678_spent_rxj2129_required_field_decomposition.json"
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rms(values, mask=None) -> float:
    data = np.asarray(values, dtype=float)
    selected = np.ones(data.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    return float(np.sqrt(np.mean(data[selected] ** 2)))


def los_mass_weighted(values, density):
    numerator = np.sum(np.asarray(values, dtype=float) * density, axis=2)
    denominator = np.sum(density, axis=2)
    return numerator / np.maximum(denominator, np.finfo(float).tiny)


def vector_alignment(first_x, first_y, second_x, second_y, mask) -> float:
    selected = np.asarray(mask, dtype=bool)
    numerator = float(
        np.sum(first_x[selected] * second_x[selected] + first_y[selected] * second_y[selected])
    )
    first_norm = float(np.sum(first_x[selected] ** 2 + first_y[selected] ** 2))
    second_norm = float(np.sum(second_x[selected] ** 2 + second_y[selected] ** 2))
    return numerator / max(np.sqrt(first_norm * second_norm), np.finfo(float).tiny)


def parameter_map(table: pd.DataFrame, selection: dict) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = table[
        table.stage.eq(selection["stage"]) & table.model.eq(selection["model"])
    ].copy()
    values = {
        name: float(rows.loc[rows.parameter.eq(name), "value"].iloc[0])
        for name in selection["parameters"]
    }
    return rows, values


def decompose_with_center(
    halo_x,
    halo_y,
    x_arcsec,
    y_arcsec,
    lower_arcsec,
    upper_arcsec,
    bins,
    center_label,
    center_x,
    center_y,
    modes,
):
    edges = np.linspace(lower_arcsec, upper_arcsec, int(bins) + 1)
    result = radial_vector_decomposition(
        halo_x,
        halo_y,
        x_arcsec,
        y_arcsec,
        edges,
        center_x=center_x,
        center_y=center_y,
    )
    radius = np.hypot(x_arcsec - center_x, y_arcsec - center_y)
    mask = (radius >= lower_arcsec) & (radius <= upper_arcsec)
    harmonics = angular_harmonics(
        result.radial_component,
        x_arcsec,
        y_arcsec,
        mask,
        modes,
        center_x=center_x,
        center_y=center_y,
    )
    result.table.insert(0, "center", center_label)
    harmonics.insert(0, "center", center_label)
    return result, mask, harmonics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    if protocol.get("status") != "frozen_before_any_P0678_required_field_metric":
        raise RuntimeError("P0678 protocol is not frozen")
    failure_parent = read_json(ROOT / protocol["failure_parent"])
    raw_failure = read_json(ROOT / protocol["raw_failure_reference"])
    raw = read_json(ROOT / protocol["raw_lensing_protocol"])

    field_path = ROOT / protocol["absolute_field_input"]
    with np.load(field_path) as data:
        axis_kpc = data["axis_kpc"].astype(float)
        scalar_physical_x = data["scalar_alpha_x_physical_arcsec"].astype(float)
        scalar_physical_y = data["scalar_alpha_y_physical_arcsec"].astype(float)
    cube_path = ROOT / protocol["baryon_cube_input"]
    with np.load(cube_path) as data:
        cube_axis_kpc = data["axis_kpc"].astype(float)
        stars = data["stellar_volume_density_kg_m3"].astype(float)
        gas = data["gas_volume_density_kg_m3"].astype(float)
        a0 = float(data["a0_m_s2"])
    if not np.array_equal(axis_kpc, cube_axis_kpc):
        raise RuntimeError("P0674 field and P0670 cube axes differ")

    scale = float(raw["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
    reference_redshift = float(protocol["common_grid"]["reference_source_redshift"])
    lens = RawLens(raw, {})
    ratio_ref = lens.distance_ratio(reference_redshift)
    axis_arcsec = axis_kpc / scale
    x_arcsec, y_arcsec = np.meshgrid(axis_arcsec, axis_arcsec, indexing="ij")
    radius_kpc = np.hypot(x_arcsec, y_arcsec) * scale
    spacing_arcsec = float(axis_arcsec[1] - axis_arcsec[0])

    parameter_path = ROOT / protocol["compact_halo_parameter_table"]
    parameter_table = pd.read_csv(parameter_path)
    selected_rows, parameters = parameter_map(
        parameter_table,
        protocol["compact_halo_selection"],
    )
    theta_e = parameters["theta_E_ref_arcsec"]
    q = parameters["axis_ratio_q"]
    phi = parameters["position_angle_phi_radian"]
    core = 10.0 ** parameters["log10_core_arcsec"]
    center_x = parameters["center_x_arcsec"]
    center_y = parameters["center_y_arcsec"]
    gamma1 = parameters["external_shear_gamma1"]
    gamma2 = parameters["external_shear_gamma2"]
    e1, e2 = param_util.phi_q2_ellipticity(phi=phi, q=q)
    nie = LensModel(lens_model_list=["NIE"])
    halo_flat_x, halo_flat_y = nie.alpha(
        x_arcsec.ravel(),
        y_arcsec.ravel(),
        [
            {
                "theta_E": theta_e,
                "e1": e1,
                "e2": e2,
                "s_scale": core,
                "center_x": center_x,
                "center_y": center_y,
            }
        ],
    )
    halo_x = np.asarray(halo_flat_x).reshape(x_arcsec.shape)
    halo_y = np.asarray(halo_flat_y).reshape(y_arcsec.shape)
    shear_x, shear_y = shear_deflection(x_arcsec, y_arcsec, gamma1, gamma2)
    scalar_x = ratio_ref * scalar_physical_x
    scalar_y = ratio_ref * scalar_physical_y
    scalar_halo_x = scalar_x + halo_x
    scalar_halo_y = scalar_y + halo_y
    target_x = scalar_halo_x + shear_x
    target_y = scalar_halo_y + shear_y

    lower_kpc, upper_kpc = (
        float(value) for value in protocol["common_grid"]["strong_lens_radius_kpc"]
    )
    lower_arcsec = lower_kpc / scale
    upper_arcsec = upper_kpc / scale
    decomposition = protocol["decomposition"]
    primary, annulus, primary_harmonics = decompose_with_center(
        halo_x,
        halo_y,
        x_arcsec,
        y_arcsec,
        lower_arcsec,
        upper_arcsec,
        decomposition["radial_bins"],
        "baryonic_center",
        0.0,
        0.0,
        decomposition["harmonics"],
    )
    sensitivity, halo_annulus, sensitivity_harmonics = decompose_with_center(
        halo_x,
        halo_y,
        x_arcsec,
        y_arcsec,
        lower_arcsec,
        upper_arcsec,
        decomposition["radial_bins"],
        "halo_fit_center",
        center_x,
        center_y,
        decomposition["harmonics"],
    )
    reconstructed_x = primary.monopole_x + primary.angular_x
    reconstructed_y = primary.monopole_y + primary.angular_y
    reconstruction_error = vector_rms(
        reconstructed_x - halo_x,
        reconstructed_y - halo_y,
        annulus,
    ) / max(vector_rms(halo_x, halo_y, annulus), np.finfo(float).tiny)

    scalar_kappa, scalar_curl, scalar_det = convergence_and_jacobian_determinant(
        scalar_x,
        scalar_y,
        spacing_arcsec,
    )
    halo_kappa, halo_curl, _ = convergence_and_jacobian_determinant(
        halo_x,
        halo_y,
        spacing_arcsec,
    )
    shear_kappa, shear_curl, _ = convergence_and_jacobian_determinant(
        shear_x,
        shear_y,
        spacing_arcsec,
    )
    _, _, scalar_halo_det = convergence_and_jacobian_determinant(
        scalar_halo_x,
        scalar_halo_y,
        spacing_arcsec,
    )
    target_kappa, target_curl, target_det = convergence_and_jacobian_determinant(
        target_x,
        target_y,
        spacing_arcsec,
    )
    interior = np.zeros_like(annulus)
    interior[2:-2, 2:-2] = True
    derivative_mask = interior & annulus
    halo_curl_normalized = rms(halo_curl, derivative_mask) / max(
        rms(2.0 * halo_kappa, derivative_mask),
        np.finfo(float).tiny,
    )
    shear_normalization = max(abs(gamma1), abs(gamma2), np.finfo(float).tiny)
    shear_derivative_residual = max(
        rms(2.0 * shear_kappa, derivative_mask),
        rms(shear_curl, derivative_mask),
    ) / shear_normalization

    spacing_m = float(axis_kpc[1] - axis_kpc[0]) * KPC_M
    activation = exact_compound_path_activation_3d(
        stars,
        gas,
        spacing_m,
        a0=a0,
        coherence_length=10.0 * KPC_M,
        coherence_power=2.0,
    )
    density = stars + gas
    dz_m = spacing_m
    star_surface = np.sum(stars, axis=2) * dz_m
    gas_surface = np.sum(gas, axis=2) * dz_m
    total_surface = star_surface + gas_surface
    gas_fraction = gas_surface / np.maximum(total_surface, np.finfo(float).tiny)
    newtonian_magnitude = np.sqrt(
        sum(component * component for component in activation.local.total_field.acceleration)
    )
    predictors = {
        "stellar_surface_density": star_surface,
        "gas_surface_density": gas_surface,
        "total_surface_density": total_surface,
        "gas_fraction": gas_fraction,
        "compound_sigma_LOS_mass_weighted": los_mass_weighted(
            activation.sigma,
            density,
        ),
        "Newtonian_acceleration_LOS_mass_weighted": los_mass_weighted(
            newtonian_magnitude,
            density,
        ),
        "tidal_trace_length_LOS_mass_weighted": los_mass_weighted(
            activation.local.trace_length / KPC_M,
            density,
        ),
        "component_mismatch_LOS_mass_weighted": los_mass_weighted(
            activation.local.transverse_mismatch,
            density,
        ),
        "absolute_scalar_deflection_magnitude": np.hypot(scalar_x, scalar_y),
    }
    scalar_magnitude = np.hypot(scalar_x, scalar_y)
    halo_magnitude = np.hypot(halo_x, halo_y)
    angular_magnitude = np.hypot(primary.angular_x, primary.angular_y)
    targets = {
        "required_deflection_magnitude": halo_magnitude,
        "required_positive_convergence": np.maximum(halo_kappa, 0.0),
        "required_deflection_to_scalar_ratio": halo_magnitude
        / np.maximum(scalar_magnitude, np.finfo(float).tiny),
        "required_angular_residual_magnitude": angular_magnitude,
    }
    correlations = predictor_correlations(predictors, targets, annulus)

    radial_tables = []
    for center_label, result, center_mask in (
        ("baryonic_center", primary, annulus),
        ("halo_fit_center", sensitivity, halo_annulus),
    ):
        table = result.table.copy()
        table["radius_minimum_kpc"] = table.radius_minimum * scale
        table["radius_maximum_kpc"] = table.radius_maximum * scale
        table["radius_midpoint_kpc"] = table.radius_midpoint * scale
        for index, row in table.iterrows():
            lower = float(row.radius_minimum)
            upper = float(row.radius_maximum)
            cx = 0.0 if center_label == "baryonic_center" else center_x
            cy = 0.0 if center_label == "baryonic_center" else center_y
            radius = np.hypot(x_arcsec - cx, y_arcsec - cy)
            use = (radius >= lower) & (radius <= upper) & center_mask
            table.loc[index, "mean_scalar_magnitude_arcsec"] = float(
                np.mean(scalar_magnitude[use])
            )
            table.loc[index, "mean_halo_magnitude_arcsec"] = float(
                np.mean(halo_magnitude[use])
            )
            table.loc[index, "mean_halo_kappa"] = float(np.mean(halo_kappa[use]))
            table.loc[index, "halo_to_scalar_magnitude_ratio"] = float(
                np.mean(halo_magnitude[use])
                / max(float(np.mean(scalar_magnitude[use])), np.finfo(float).tiny)
            )
        radial_tables.append(table)
    radial_profile = pd.concat(radial_tables, ignore_index=True)
    harmonics = pd.concat(
        [primary_harmonics, sensitivity_harmonics],
        ignore_index=True,
    )

    scalar_rms = vector_rms(scalar_x, scalar_y, annulus)
    halo_rms = vector_rms(halo_x, halo_y, annulus)
    shear_rms = vector_rms(shear_x, shear_y, annulus)
    target_rms = vector_rms(target_x, target_y, annulus)
    monopole_rms = vector_rms(primary.monopole_x, primary.monopole_y, annulus)
    angular_rms = vector_rms(primary.angular_x, primary.angular_y, annulus)
    top_required = (
        correlations[correlations.target.eq("required_deflection_magnitude")]
        .assign(abs_rho=lambda frame: frame.spearman_rho.abs())
        .sort_values("abs_rho", ascending=False)
        .iloc[0]
    )
    top_ratio = (
        correlations[correlations.target.eq("required_deflection_to_scalar_ratio")]
        .assign(abs_rho=lambda frame: frame.spearman_rho.abs())
        .sort_values("abs_rho", ascending=False)
        .iloc[0]
    )
    metrics = {
        "reference_distance_ratio_Dds_over_Ds": ratio_ref,
        "compact_halo_theta_E_reference_arcsec": theta_e,
        "compact_halo_theta_E_reference_kpc": theta_e * scale,
        "compact_halo_core_arcsec": core,
        "compact_halo_core_kpc": core * scale,
        "compact_halo_axis_ratio": q,
        "scalar_strong_lens_RMS_reduced_arcsec": scalar_rms,
        "compact_halo_strong_lens_RMS_arcsec": halo_rms,
        "external_shear_strong_lens_RMS_arcsec": shear_rms,
        "scalar_plus_halo_plus_shear_RMS_arcsec": target_rms,
        "compact_halo_to_scalar_RMS_ratio": halo_rms
        / max(scalar_rms, np.finfo(float).tiny),
        "target_to_scalar_RMS_ratio": target_rms
        / max(scalar_rms, np.finfo(float).tiny),
        "halo_monopole_RMS_fraction": monopole_rms
        / max(halo_rms, np.finfo(float).tiny),
        "halo_angular_residual_RMS_fraction": angular_rms
        / max(halo_rms, np.finfo(float).tiny),
        "halo_scalar_vector_alignment_cosine": vector_alignment(
            halo_x,
            halo_y,
            scalar_x,
            scalar_y,
            annulus,
        ),
        "halo_positive_kappa_R50_kpc": positive_weight_radius_quantile(
            radius_kpc,
            halo_kappa,
            0.5,
        ),
        "halo_positive_kappa_R80_kpc": positive_weight_radius_quantile(
            radius_kpc,
            halo_kappa,
            0.8,
        ),
        "scalar_critical_sign_change_cells": sign_change_cells(scalar_det),
        "scalar_plus_halo_critical_sign_change_cells": sign_change_cells(
            scalar_halo_det
        ),
        "scalar_plus_halo_plus_shear_critical_sign_change_cells": sign_change_cells(
            target_det
        ),
        "halo_monopole_plus_angular_reconstruction_relative_RMS": reconstruction_error,
        "compact_halo_normalized_curl_RMS": halo_curl_normalized,
        "external_shear_normalized_derivative_residual": shear_derivative_residual,
        "top_required_magnitude_baryonic_predictor": str(top_required.predictor),
        "top_required_magnitude_baryonic_predictor_spearman_rho": float(
            top_required.spearman_rho
        ),
        "top_required_ratio_baryonic_predictor": str(top_ratio.predictor),
        "top_required_ratio_baryonic_predictor_spearman_rho": float(
            top_ratio.spearman_rho
        ),
    }

    gates = protocol["predeclared_integrity_gates"]
    failed_topology = raw_failure["topology"]["compound_absolute_P0673"]
    finite = bool(
        all(
            np.all(np.isfinite(values))
            for values in (
                scalar_x,
                scalar_y,
                halo_x,
                halo_y,
                shear_x,
                shear_y,
                primary.monopole_x,
                primary.monopole_y,
                primary.angular_x,
                primary.angular_y,
                halo_kappa,
                target_det,
            )
        )
        and correlations.spearman_rho.notna().all()
    )
    gate_results = {
        "P0677_failed": failure_parent["status"] == gates["P0677_status"],
        "P0675_missing_multiplicity": int(
            failed_topology["missing_multiplicity_families"]
        )
        == int(gates["P0675_compound_missing_multiplicity_families"]),
        "P0675_no_critical_curves": int(
            failed_topology["critical_curve_present_families"]
        )
        == int(gates["P0675_compound_critical_curve_present_families"]),
        "parameter_rows": len(selected_rows)
        == int(gates["compact_halo_training_parameter_rows"]),
        "parameter_stage": selected_rows.stage.eq(
            gates["compact_halo_parameter_stage"]
        ).all(),
        "parameter_model": selected_rows.model.eq(
            gates["compact_halo_parameter_model"]
        ).all(),
        "common_grid": scalar_x.shape
        == (int(gates["common_grid_cells_per_axis"]),) * 2,
        "finite": finite is bool(gates["all_fields_and_decompositions_finite"]),
        "exact_reconstruction": reconstruction_error
        <= float(gates["halo_monopole_plus_angular_reconstruction_relative_RMS_max"]),
        "halo_curl": halo_curl_normalized
        <= float(gates["compact_halo_normalized_curl_RMS_max"]),
        "shear_derivatives": shear_derivative_residual
        <= float(gates["external_shear_normalized_divergence_and_curl_RMS_max"]),
        "radial_bins": int(
            np.sum(
                (radial_profile.center.eq("baryonic_center"))
                & (radial_profile.samples > 0)
            )
        )
        == int(gates["radial_bins_with_samples"]),
        "predictor_coverage": correlations.predictor.nunique()
        >= int(gates["minimum_baryonic_predictors_reported"]),
        "no_candidate_fit": not bool(gates["new_candidate_formula_fit"]),
        "no_raw_root_score": not bool(gates["new_raw_image_root_score_computed"]),
        "sealed_targets_untouched": not bool(gates["sealed_target_outcomes_opened"]),
    }
    all_pass = bool(all(gate_results.values()))

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    common_field_path = output / protocol["outputs"]["common_field"]
    np.savez_compressed(
        common_field_path,
        axis_kpc=axis_kpc,
        scalar_alpha_x_reference_arcsec=scalar_x,
        scalar_alpha_y_reference_arcsec=scalar_y,
        compact_halo_alpha_x_reference_arcsec=halo_x,
        compact_halo_alpha_y_reference_arcsec=halo_y,
        external_shear_alpha_x_reference_arcsec=shear_x,
        external_shear_alpha_y_reference_arcsec=shear_y,
        halo_monopole_x_reference_arcsec=primary.monopole_x,
        halo_monopole_y_reference_arcsec=primary.monopole_y,
        halo_angular_x_reference_arcsec=primary.angular_x,
        halo_angular_y_reference_arcsec=primary.angular_y,
        scalar_kappa_reference=scalar_kappa,
        halo_kappa_reference=halo_kappa,
        target_kappa_reference=target_kappa,
        scalar_jacobian_determinant=scalar_det,
        target_jacobian_determinant=target_det,
        scalar_curl_reference=scalar_curl,
        target_curl_reference=target_curl,
    )
    radial_profile.to_csv(output / protocol["outputs"]["radial_profile"], index=False)
    correlations.to_csv(
        output / protocol["outputs"]["predictor_correlations"],
        index=False,
    )
    harmonics.to_csv(output / protocol["outputs"]["harmonics"], index=False)
    report = {
        "report_version": "P0678-SPENT-RXJ2129-REQUIRED-FIELD-DECOMPOSITION-RESULTS-1.0.0",
        "status": "pass" if all_pass else "fail",
        "all_integrity_gates_pass": all_pass,
        "candidate_formula_advanced": False,
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__).resolve()),
        "decomposition_source_sha256": sha256(
            ROOT / "src/voidscreen/required_field_decomposition.py"
        ),
        "absolute_field_sha256": sha256(field_path),
        "baryon_cube_sha256": sha256(cube_path),
        "compact_halo_parameters_sha256": sha256(parameter_path),
        "common_field_sha256": sha256(common_field_path),
        "metrics": metrics,
        "gate_results": gate_results,
        "new_candidate_formula_fit": False,
        "new_raw_image_root_score_computed": False,
        "sealed_P0633_kinematics_opened": False,
        "sealed_P0640_lensing_constraints_opened": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    (output / "report.json").write_text(
        json.dumps(json_safe(report), indent=2),
        encoding="utf-8",
    )

    figure, axes = plt.subplots(2, 3, figsize=(13.5, 8.5))
    extent = [axis_kpc[0], axis_kpc[-1], axis_kpc[0], axis_kpc[-1]]
    panels = (
        (scalar_magnitude, "absolute scalar (reduced)", "arcsec", "viridis"),
        (halo_magnitude, "required compact-halo field", "arcsec", "viridis"),
        (
            np.hypot(primary.monopole_x, primary.monopole_y),
            "halo radial monopole",
            "arcsec",
            "viridis",
        ),
        (angular_magnitude, "halo angular residual", "arcsec", "viridis"),
        (halo_kappa, "halo effective convergence", "kappa", "coolwarm"),
        (np.sign(target_det), "target Jacobian sign", "sign", "coolwarm"),
    )
    for axis, (values, title, label, cmap) in zip(axes.ravel(), panels, strict=True):
        image = axis.imshow(values.T, origin="lower", extent=extent, cmap=cmap)
        axis.set(title=title, xlabel="x (kpc)", ylabel="y (kpc)")
        figure.colorbar(image, ax=axis, shrink=0.75, label=label)
    figure.tight_layout()
    figure.savefig(output / "p0678_required_field_decomposition.png", dpi=180)
    plt.close(figure)

    failed = [name for name, passed in gate_results.items() if not passed]
    summary = f"""# P0678 spent RX J2129 required-field decomposition

- Status: **{'PASS' if all_pass else 'FAIL'}** integrity audit.
- Scalar / compact-halo / shear strong-lens RMS: **{scalar_rms:.4g} / {halo_rms:.4g} / {shear_rms:.4g} arcsec** at z={reference_redshift:g}.
- Compact-halo/scalar RMS ratio: **{metrics['compact_halo_to_scalar_RMS_ratio']:.4g}x**.
- Halo monopole / angular-residual RMS fractions: **{metrics['halo_monopole_RMS_fraction']:.4g} / {metrics['halo_angular_residual_RMS_fraction']:.4g}**.
- Scalar / scalar+halo / scalar+halo+shear critical sign-change cells: **{metrics['scalar_critical_sign_change_cells']} / {metrics['scalar_plus_halo_critical_sign_change_cells']} / {metrics['scalar_plus_halo_plus_shear_critical_sign_change_cells']}**.
- Halo positive-kappa R50 / R80: **{metrics['halo_positive_kappa_R50_kpc']:.4g} / {metrics['halo_positive_kappa_R80_kpc']:.4g} kpc**.
- Strongest baryonic correlate of required magnitude: **{metrics['top_required_magnitude_baryonic_predictor']} (rho={metrics['top_required_magnitude_baryonic_predictor_spearman_rho']:+.3f})**.
- Strongest baryonic correlate of required halo/scalar ratio: **{metrics['top_required_ratio_baryonic_predictor']} (rho={metrics['top_required_ratio_baryonic_predictor_spearman_rho']:+.3f})**.
- Failed frozen integrity gates: **{', '.join(failed) if failed else 'none'}**.
- New formula/root score/sealed outcome: **no / no / no**.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
