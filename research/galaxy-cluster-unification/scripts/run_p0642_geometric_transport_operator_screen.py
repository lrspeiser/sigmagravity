#!/usr/bin/env python3
"""Screen baryon-only geometry operators without opening sealed targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.geometric_transport import (
    G_SI,
    M_SUN_KG,
    PathGeometry,
    aperture_weighted_statistics,
    component_cancellation,
    high_acceleration_screen,
    hybrid_geometry,
    normalized_discrete_curl,
    resample_surface_density,
    spectral_poisson_acceleration_2d,
    streamline_incoherence,
    tensor_source_2d,
    thin_sheet_newtonian_field,
)

DEFAULT_PROTOCOL = ROOT / "configs" / "p0642_geometric_transport_operator_screen.json"
DEFAULT_OUTPUT = ROOT / "results" / "p0642_geometric_transport_operator_screen"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gaussian_component(
    axis: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    scale: float,
    mass_msun: float,
) -> np.ndarray:
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    shape = np.exp(-0.5 * ((xx - center_x) ** 2 + (yy - center_y) ** 2) / scale**2)
    cell = float(np.median(np.diff(axis)))
    return shape * float(mass_msun) / (float(np.sum(shape)) * cell**2)


def geometry_bundle(stars, gas, cell_kpc, protocol) -> tuple[dict, object, PathGeometry]:
    definitions = protocol["definitions"]
    controls = protocol["synthetic_controls"]
    star_field = thin_sheet_newtonian_field(stars, cell_kpc)
    gas_field = thin_sheet_newtonian_field(gas, cell_kpc)
    total = np.asarray(stars, dtype=float) + np.asarray(gas, dtype=float)
    total_field = thin_sheet_newtonian_field(total, cell_kpc)
    path = streamline_incoherence(
        total_field,
        cell_kpc,
        beta=float(definitions["beta"]),
        trace_steps=int(controls["trace_steps"]),
        trace_length_floor_cells=float(controls["trace_length_floor_cells"]),
        trace_length_cap_cells=float(controls["trace_length_cap_cells"]),
    )
    cancellation = component_cancellation(star_field, gas_field)
    fields = {
        "path_incoherence": path.incoherence,
        "component_cancellation": cancellation,
        "hybrid": hybrid_geometry(path.incoherence, cancellation),
    }
    return fields, total_field, path


def summarize_case(case, domain, stars, gas, cell_kpc, protocol):
    fields, total_field, path = geometry_bundle(stars, gas, cell_kpc, protocol)
    total = stars + gas
    screen = high_acceleration_screen(
        total_field.magnitude_m_s2, float(protocol["definitions"]["a0_m_s2"])
    )
    rows = []
    for name, geometry in fields.items():
        raw = aperture_weighted_statistics(
            geometry, total, total_field.magnitude_m_s2, cell_kpc
        )
        activation = aperture_weighted_statistics(
            screen * geometry, total, total_field.magnitude_m_s2, cell_kpc
        )
        rows.append(
            {
                "case": case,
                "domain": domain,
                "variant": name,
                "cell_kpc": float(cell_kpc),
                **{f"geometry_{key}": value for key, value in raw.items()},
                **{f"activation_{key}": value for key, value in activation.items()},
                "median_trace_length_kpc": float(np.median(path.trace_length_kpc)),
                "median_screen": float(np.median(screen)),
            }
        )
    return rows, fields, total_field, path


def synthetic_suite(protocol):
    settings = protocol["synthetic_controls"]
    cells = int(settings["grid_cells"])
    half = float(settings["half_width"])
    axis = np.linspace(-half, half, cells)
    cell = float(axis[1] - axis[0])
    scale = float(settings["radial_plummer_scale"])
    radial_stars = gaussian_component(
        axis, center_x=0.0, center_y=0.0, scale=scale, mass_msun=6.0e10
    )
    radial_gas = gaussian_component(
        axis, center_x=0.0, center_y=0.0, scale=1.5 * scale, mass_msun=4.0e10
    )
    separation = float(settings["binary_separation"])
    binary_scale = float(settings["binary_component_scale"])
    binary_stars = gaussian_component(
        axis,
        center_x=-0.5 * separation,
        center_y=0.0,
        scale=binary_scale,
        mass_msun=5.0e10,
    )
    binary_gas = gaussian_component(
        axis,
        center_x=0.5 * separation,
        center_y=0.0,
        scale=binary_scale,
        mass_msun=5.0e10,
    )
    offset = float(settings["offset_component_separation"])
    offset_stars = gaussian_component(
        axis, center_x=-0.5 * offset, center_y=0.0, scale=1.0, mass_msun=3.0e10
    )
    offset_gas = gaussian_component(
        axis, center_x=0.5 * offset, center_y=0.0, scale=1.8, mass_msun=7.0e10
    )
    cases = {
        "radial_cocentered": (radial_stars, radial_gas),
        "equal_binary": (binary_stars, binary_gas),
        "offset_components": (offset_stars, offset_gas),
    }
    rows, products = [], {}
    for name, components in cases.items():
        case_rows, fields, total_field, path = summarize_case(
            name, "synthetic", *components, cell, protocol
        )
        rows.extend(case_rows)
        products[name] = (components, fields, total_field, path)

    rotated = (np.rot90(binary_stars), np.rot90(binary_gas))
    rotated_rows, _, _, _ = summarize_case(
        "equal_binary_rotated_90", "synthetic_control", *rotated, cell, protocol
    )
    shifted = (
        ndimage.shift(binary_stars, (17, -13), order=1, mode="constant", cval=0.0),
        ndimage.shift(binary_gas, (17, -13), order=1, mode="constant", cval=0.0),
    )
    shifted_rows, _, _, _ = summarize_case(
        "equal_binary_translated", "synthetic_control", *shifted, cell, protocol
    )
    rows.extend(rotated_rows + shifted_rows)
    return pd.DataFrame(rows), products


def observed_suite(protocol):
    inputs = protocol["observational_inputs"]
    rows = []
    for path in sorted((ROOT / inputs["galaxy_map_directory"]).glob("*.npz")):
        with np.load(path) as data:
            axis = data["axis_kpc"].astype(float)
            stars = data["stars"].astype(float)
            gas = data["gas"].astype(float)
        case_rows, _, _, _ = summarize_case(
            path.stem, "sealed_galaxy_baryons_only", stars, gas, float(axis[1] - axis[0]), protocol
        )
        rows.extend(case_rows)
    target = int(inputs["cluster_downsample_cells"])
    for path in sorted((ROOT / inputs["cluster_map_directory"]).glob("*.npz")):
        with np.load(path) as data:
            axis = data["axis_kpc"].astype(float)
            stars_original = data["stellar_surface_density_msun_kpc2"].astype(float)
            gas_original = data["gas_surface_density_msun_kpc2"].astype(float)
        stars = resample_surface_density(stars_original, target)
        gas = resample_surface_density(gas_original, target)
        cell = float((axis[-1] - axis[0]) / (target - 1))
        case_rows, _, _, _ = summarize_case(
            path.stem.replace("_baryons", ""),
            "sealed_cluster_baryons_only",
            stars,
            gas,
            cell,
            protocol,
        )
        rows.extend(case_rows)
    return pd.DataFrame(rows)


def relative_error(value, reference):
    return float(abs(float(value) - float(reference)) / max(abs(float(reference)), 1e-15))


def evaluate(protocol, synthetic, observed, products):
    gates = protocol["predeclared_numerical_gates"]
    lookup = synthetic.set_index(["case", "variant"])
    radial_path = float(lookup.loc[("radial_cocentered", "path_incoherence"), "geometry_weighted_mean"])
    radial_cancel = float(
        lookup.loc[("radial_cocentered", "component_cancellation"), "geometry_weighted_mean"]
    )
    binary_path = float(lookup.loc[("equal_binary", "path_incoherence"), "geometry_weighted_mean"])
    offset_cancel = float(
        lookup.loc[("offset_components", "component_cancellation"), "geometry_weighted_mean"]
    )
    baseline = synthetic[synthetic.case.eq("equal_binary")].set_index("variant")
    rotated = synthetic[synthetic.case.eq("equal_binary_rotated_90")].set_index("variant")
    translated = synthetic[synthetic.case.eq("equal_binary_translated")].set_index("variant")
    rotation_error = max(
        relative_error(rotated.loc[name, "geometry_weighted_mean"], baseline.loc[name, "geometry_weighted_mean"])
        for name in protocol["operator_variants"]
    )
    translation_error = max(
        relative_error(translated.loc[name, "geometry_weighted_mean"], baseline.loc[name, "geometry_weighted_mean"])
        for name in protocol["operator_variants"]
    )
    all_values = []
    for _, fields, _, _ in products.values():
        all_values.extend(fields.values())
    bounded = bool(all(np.min(field) >= 0.0 and np.max(field) <= 1.0 for field in all_values))
    solar_g = G_SI * M_SUN_KG / (149_597_870_700.0**2)
    solar_coefficient = 5.0 * float(protocol["definitions"]["a0_m_s2"]) / (
        float(protocol["definitions"]["a0_m_s2"]) + solar_g
    )
    _, binary_fields, binary_total, binary_trace = products["equal_binary"]
    source = tensor_source_2d(
        binary_total,
        binary_trace,
        binary_fields["hybrid"],
        float(synthetic.cell_kpc.iloc[0]),
        a0_m_s2=float(protocol["definitions"]["a0_m_s2"]),
        geometric_strength=1.0,
    )
    _, ax, ay = spectral_poisson_acceleration_2d(source, float(synthetic.cell_kpc.iloc[0]))
    curl = normalized_discrete_curl(ax, ay, float(synthetic.cell_kpc.iloc[0]))
    gate_results = {
        "radial_path_null": radial_path <= gates["radial_path_incoherence_weighted_mean_max"],
        "radial_component_null": radial_cancel <= gates["radial_component_cancellation_weighted_mean_max"],
        "binary_path_activation": binary_path / max(radial_path, 1e-15)
        >= gates["binary_path_incoherence_gain_over_radial_min"],
        "offset_component_activation": offset_cancel
        >= gates["offset_component_cancellation_weighted_mean_min"],
        "rotation_covariance": rotation_error
        <= gates["rotation_covariance_weighted_mean_relative_error_max"],
        "translation_covariance": translation_error
        <= gates["translation_covariance_weighted_mean_relative_error_max"],
        "bounded": bounded is bool(gates["all_activation_values_bounded_0_1"]),
        "solar_screen": solar_coefficient
        <= gates["solar_1au_geometric_coefficient_at_lambda_5_max"],
        "sealed_targets_untouched": not bool(inputs_opened := protocol["observational_inputs"]["sealed_target_outcomes_opened"]),
    }
    domain_summary = (
        observed.groupby(["domain", "variant"], as_index=False)
        .agg(
            systems=("case", "nunique"),
            median_activation=("activation_weighted_mean", "median"),
            minimum_activation=("activation_weighted_mean", "min"),
            maximum_activation=("activation_weighted_mean", "max"),
        )
    )
    galaxy = domain_summary[domain_summary.domain.eq("sealed_galaxy_baryons_only")].set_index("variant")
    cluster = domain_summary[domain_summary.domain.eq("sealed_cluster_baryons_only")].set_index("variant")
    ratios = {}
    applicable = {
        "path_incoherence": gate_results["radial_path_null"] and gate_results["binary_path_activation"],
        "component_cancellation": gate_results["radial_component_null"] and gate_results["offset_component_activation"],
        "hybrid": all(
            gate_results[name]
            for name in (
                "radial_path_null",
                "radial_component_null",
                "binary_path_activation",
                "offset_component_activation",
            )
        ),
    }
    for name in protocol["operator_variants"]:
        ratios[name] = float(
            cluster.loc[name, "median_activation"]
            / max(float(galaxy.loc[name, "median_activation"]), 1e-15)
        )
    eligible = [name for name in protocol["operator_variants"] if applicable[name]]
    provisional = max(eligible, key=lambda name: ratios[name]) if eligible else None
    universal_gates = all(
        gate_results[name]
        for name in (
            "rotation_covariance",
            "translation_covariance",
            "bounded",
            "solar_screen",
            "sealed_targets_untouched",
        )
    )
    selected = provisional if universal_gates else None
    return {
        "status": "pass" if all(gate_results.values()) and selected else "fail",
        "all_numerical_gates_pass": bool(all(gate_results.values())),
        "gate_results": gate_results,
        "synthetic_metrics": {
            "radial_path_weighted_mean": radial_path,
            "radial_component_cancellation_weighted_mean": radial_cancel,
            "binary_path_weighted_mean": binary_path,
            "binary_path_gain_over_radial": binary_path / max(radial_path, 1e-15),
            "offset_component_cancellation_weighted_mean": offset_cancel,
            "rotation_covariance_relative_error": rotation_error,
            "translation_covariance_relative_error": translation_error,
            "solar_1au_lambda5_max_coefficient": solar_coefficient,
            "derived_field_normalized_curl": curl,
        },
        "variant_applicable_gate_pass": applicable,
        "cluster_to_galaxy_median_activation_ratio": ratios,
        "selected_operator": selected,
        "provisional_operator_before_universal_gates": provisional,
        "selection_used_sealed_target_outcomes": bool(inputs_opened),
        "domain_summary": domain_summary.to_dict(orient="records"),
    }, domain_summary


def make_figure(synthetic, observed, products, output):
    figure, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    for axis, name in zip(axes[0], ("radial_cocentered", "equal_binary", "offset_components"), strict=True):
        components, fields, _, _ = products[name]
        image = axis.imshow(fields["hybrid"], origin="lower", cmap="magma", vmin=0.0, vmax=0.5)
        axis.contour(components[0] + components[1], levels=5, colors="cyan", linewidths=0.5)
        axis.set_title(name.replace("_", " "))
        axis.set_xticks([])
        axis.set_yticks([])
    figure.colorbar(image, ax=axes[0].tolist(), shrink=0.8, label="hybrid geometry C")
    syn = synthetic[synthetic.domain.eq("synthetic")]
    pivot = syn.pivot(index="case", columns="variant", values="geometry_weighted_mean")
    pivot.plot.bar(ax=axes[1, 0], logy=True)
    axes[1, 0].set_ylabel("geometry weighted mean")
    axes[1, 0].set_title("Synthetic activation")
    obs = observed.pivot(index="case", columns="variant", values="activation_weighted_mean")
    obs.plot.bar(ax=axes[1, 1], logy=True)
    axes[1, 1].set_ylabel("screened activation S C")
    axes[1, 1].set_title("All registered baryon maps")
    axes[1, 1].legend(fontsize=7)
    grouped = observed.groupby(["domain", "variant"]).activation_weighted_mean.median().unstack(0)
    grouped.plot.bar(ax=axes[1, 2], logy=True)
    axes[1, 2].set_ylabel("median screened activation")
    axes[1, 2].set_title("Galaxy versus cluster")
    for axis in axes[1]:
        axis.tick_params(axis="x", labelrotation=35)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("P0642 baryon-only geometric operator screen")
    figure.subplots_adjust(left=0.06, right=0.98, bottom=0.19, top=0.92, wspace=0.32, hspace=0.35)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_any_P0642_operator_score":
        raise RuntimeError("P0642 protocol is not frozen")
    synthetic, products = synthetic_suite(protocol)
    observed = observed_suite(protocol)
    report, domain_summary = evaluate(protocol, synthetic, observed, products)
    report.update(
        {
            "protocol_version": protocol["protocol_version"],
            "protocol_sha256": sha256(protocol_path),
            "source_sha256": sha256(ROOT / "src" / "voidscreen" / "geometric_transport.py"),
            "coverage": {
                "synthetic_cases": int(synthetic.case.nunique()),
                "registered_galaxies": int(observed[observed.domain.str.contains("galaxy")].case.nunique()),
                "registered_clusters": int(observed[observed.domain.str.contains("cluster")].case.nunique()),
                "per_object_gravity_parameters": 0,
            },
            "environment": {"python": platform.python_version(), "numpy": np.__version__},
            "sealed_P0633_kinematics_opened": False,
            "sealed_P0640_lensing_constraints_opened": False,
            "claim_boundary": protocol["claim_boundary"],
        }
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    synthetic.to_csv(output / "synthetic_operator_scores.csv", index=False)
    observed.to_csv(output / "registered_map_operator_scores.csv", index=False)
    domain_summary.to_csv(output / "domain_summary.csv", index=False)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    make_figure(synthetic, observed, products, output / "geometric_operator_screen.png")
    selected = report["selected_operator"] or "NONE"
    ratios = report["cluster_to_galaxy_median_activation_ratio"]
    summary = f"""# P0642 baryon-only geometric operator screen

- Numerical status: **{report['status'].upper()}** ({sum(report['gate_results'].values())}/{len(report['gate_results'])} gates).
- Selected operator before any target outcome was opened: **{selected}**.
- Cluster/galaxy median screened-activation ratios: path {ratios['path_incoherence']:.4g}, component cancellation {ratios['component_cancellation']:.4g}, hybrid {ratios['hybrid']:.4g}.
- Radial path null: {report['synthetic_metrics']['radial_path_weighted_mean']:.6g}; binary gain: {report['synthetic_metrics']['binary_path_gain_over_radial']:.4g}x.
- Co-centered component null: {report['synthetic_metrics']['radial_component_cancellation_weighted_mean']:.6g}; offset activation: {report['synthetic_metrics']['offset_component_cancellation_weighted_mean']:.6g}.
- Solar 1 AU worst-case coefficient for lambda=5: {report['synthetic_metrics']['solar_1au_lambda5_max_coefficient']:.3e}.
- Registered maps used: {report['coverage']['registered_galaxies']} galaxies and {report['coverage']['registered_clusters']} clusters; target velocities and lens constraints opened: **no**.

This screen selects a measurable geometry variable, not a successful gravity
theory.  The universal strength must next be ablated on already-spent systems,
frozen, and only then evaluated once on the sealed validation set.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(json.dumps({"status": report["status"], "selected": selected, "ratios": ratios}, indent=2))


if __name__ == "__main__":
    main()
