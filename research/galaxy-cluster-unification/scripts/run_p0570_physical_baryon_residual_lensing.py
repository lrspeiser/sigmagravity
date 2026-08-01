#!/usr/bin/env python3
"""Test measured noncircular baryon deflection on four raw cluster lenses."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_member_tidal_metric import (  # noqa: E402
    MemberTidalLens,
    build_contexts,
    fit_context,
    model_name,
)
from run_p0557_baryon_proxy_tidal import compressed_map_catalog, json_safe  # noqa: E402
from run_p0559_accept_projected_gas_tidal import (  # noqa: E402
    physical_catalogs,
    prepare_registered_maps,
)
from run_unbounded_running_multicluster_raw import aggregate_system_scores  # noqa: E402
from voidscreen.tidal_metric import TidalCorrectionField  # noqa: E402


G_SI = 6.67430e-11
C_SI = 299_792_458.0
M_SUN_KG = 1.98847e30
KPC_M = 3.085677581491367e19
ARCSEC_PER_RADIAN = 206_264.80624709636


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def physical_residual_field(
    catalog: pd.DataFrame,
    total_mass_msun: float,
    scale_kpc_per_arcsec: float,
    extent_scale: float,
    protocol: dict,
) -> TidalCorrectionField:
    numerical = protocol["numerics"]
    size = int(numerical["field_pixels_per_axis"])
    half = float(numerical["field_half_width_arcsec"])
    spacing = 2.0 * half / size
    axis = -half + spacing * np.arange(size)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    weights = catalog.normalized_light_weight.to_numpy(float)
    weights /= np.sum(weights)
    source_x = catalog.x_arcsec.to_numpy(float)
    source_y = catalog.y_arcsec.to_numpy(float)
    center_x = float(np.sum(weights * source_x))
    center_y = float(np.sum(weights * source_y))
    source_x = center_x + float(extent_scale) * (source_x - center_x)
    source_y = center_y + float(extent_scale) * (source_y - center_y)
    masses = weights * float(total_mass_msun)
    soft2 = float(numerical["point_softening_arcsec"]) ** 2
    coefficient_per_msun = (
        4.0
        * G_SI
        * M_SUN_KG
        / (C_SI**2 * float(scale_kpc_per_arcsec) * KPC_M)
        * ARCSEC_PER_RADIAN
    )
    full_x = np.zeros_like(xx)
    full_y = np.zeros_like(yy)
    full_potential = np.zeros_like(xx)
    for sx, sy, mass in zip(source_x, source_y, masses, strict=True):
        dx = xx - sx
        dy = yy - sy
        distance2 = dx * dx + dy * dy + soft2
        inverse = 1.0 / distance2
        coefficient = coefficient_per_msun * mass
        full_x += coefficient * dx * inverse
        full_y += coefficient * dy * inverse
        full_potential += 0.5 * coefficient * np.log(distance2)

    # Subtract the exact azimuthal (m=0) potential of every softened point
    # source, then take one discrete gradient.  Constructing both residual
    # components from the same scalar potential preserves curl-freedom on the
    # grid.  For A=r^2+a^2+s^2 and B=2ra,
    # <log(A-B cos(phi))> = log((A+sqrt(A^2-B^2))/2).
    radius = np.hypot(xx, yy)
    circular_potential = np.zeros_like(xx)
    for sx, sy, mass in zip(source_x, source_y, masses, strict=True):
        source_radius = math.hypot(float(sx), float(sy))
        coefficient = coefficient_per_msun * mass
        a_term = radius * radius + source_radius * source_radius + soft2
        b_term = 2.0 * radius * source_radius
        discriminant = np.maximum(a_term * a_term - b_term * b_term, 0.0)
        circular_potential += 0.5 * coefficient * np.log(
            0.5 * (a_term + np.sqrt(discriminant))
        )
    residual_potential = full_potential - circular_potential
    residual_y, residual_x = np.gradient(residual_potential, spacing, spacing)
    d_y_dx = np.gradient(residual_y, spacing, axis=1)
    d_x_dy = np.gradient(residual_x, spacing, axis=0)
    d_x_dx = np.gradient(residual_x, spacing, axis=1)
    d_y_dy = np.gradient(residual_y, spacing, axis=0)
    interior = radius <= 200.0
    curl = d_y_dx - d_x_dy
    divergence = d_x_dx + d_y_dy
    normalized_curl = float(
        np.sqrt(np.mean(curl[interior] ** 2))
        / max(np.sqrt(np.mean(divergence[interior] ** 2)), np.finfo(float).tiny)
    )
    full_rms = float(np.sqrt(np.mean((full_x[interior] ** 2 + full_y[interior] ** 2))))
    residual_rms = float(
        np.sqrt(np.mean((residual_x[interior] ** 2 + residual_y[interior] ** 2)))
    )
    audit = {
        "grid_spacing_arcsec": spacing,
        "total_mass_msun": float(total_mass_msun),
        "extent_scale": float(extent_scale),
        "source_centroid_x_arcsec": center_x,
        "source_centroid_y_arcsec": center_y,
        "full_deflection_RMS_arcsec": full_rms,
        "residual_deflection_RMS_arcsec": residual_rms,
        "residual_to_full_RMS_fraction": residual_rms / max(full_rms, np.finfo(float).tiny),
        "maximum_residual_deflection_arcsec": float(np.max(np.hypot(residual_x[interior], residual_y[interior]))),
        "normalized_curl_RMS": normalized_curl,
        "maximum_Q_eigenvalue": 0.0,
        "RMS_Q_eigenvalue": 0.0,
        "maximum_edge_Q_eigenvalue": 0.0,
        "maximum_solver_edge_Q_eigenvalue": 0.0,
        "maximum_abs_circular_cross_mean": 0.0,
        "circular_mean_subtracted": True,
        "circular_subtraction_method": "exact_softened_point_m0_potential_then_grid_gradient",
        "correction_RMS_arcsec_at_distance_ratio_one": residual_rms,
        "correction_maximum_arcsec_at_distance_ratio_one": float(np.max(np.hypot(residual_x, residual_y))),
    }
    zeros = np.zeros_like(residual_x)
    return TidalCorrectionField(axis, residual_x, residual_y, zeros, zeros, audit)


def source_plane_rms(lens, coupling, parameters, sources, rows):
    residuals = []
    name = model_name(coupling)
    for family, group in rows.groupby("source_family", sort=True):
        x = group.x_arcsec.to_numpy(float)
        y = group.y_arcsec.to_numpy(float)
        redshift = float(group.source_redshift.median())
        beta_x, beta_y = lens.ray_shooting(name, parameters, x, y, redshift)
        source = sources[int(family)]
        residuals.append(np.column_stack([beta_x - source[0], beta_y - source[1]]))
    residual = np.vstack(residuals)
    return float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))


def circular_null_audit(protocol):
    catalog = pd.DataFrame(
        {
            "x_arcsec": [0.0],
            "y_arcsec": [0.0],
            "normalized_light_weight": [1.0],
        }
    )
    field = physical_residual_field(catalog, 1.0e14, 5.0, 1.0, protocol)
    return {
        "residual_to_full_RMS_fraction": field.audit["residual_to_full_RMS_fraction"],
        "normalized_curl_RMS": field.audit["normalized_curl_RMS"],
    }


def main():
    protocol_path = ROOT / "configs/p0570_physical_baryon_residual_lensing_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_any_physical_residual_lens_score":
        raise RuntimeError("P0570 protocol is not frozen")
    p0559 = json.loads((ROOT / protocol["inputs"]["p0559_protocol"]).read_text(encoding="utf-8"))
    p0557 = json.loads((ROOT / p0559["inputs"]["p0557_protocol"]).read_text(encoding="utf-8"))
    member = json.loads((ROOT / p0559["inputs"]["member_tidal_protocol"]).read_text(encoding="utf-8"))
    member["optimization"]["maximum_function_evaluations"] = int(
        p0559["optimization"]["maximum_function_evaluations"]
    )
    contexts, _, _ = build_contexts(
        member, softening_kpc=float(p0559["locked_field"]["softening_kpc"])
    )
    registered = prepare_registered_maps(p0559, contexts)
    physical, physical_audits = physical_catalogs(p0559, contexts, registered)
    physical_audits = physical_audits.set_index("system_label")
    block = int(p0557["proxy_maps"]["compression_block_pixels"])
    component_catalogs = {}
    component_masses = {}
    for context in contexts:
        label = context.system["label"]
        star = compressed_map_catalog(
            registered[label]["axis"],
            registered[label]["star"],
            block_pixels=block,
            transform="linear",
        )
        star_mass = float(physical_audits.loc[label, "stellar_mass_assigned_to_map_msun"])
        gas_mass = float(physical_audits.loc[label, "projected_ACCEPT_gas_mass_on_map_msun"])
        component_catalogs[label] = {
            "registered_starlight": star,
            "accept_gas_spherical": physical[label][("accept_absolute", 0.0, False)],
            "accept_gas_sqrt_morphology": physical[label][("accept_absolute", 0.5, False)],
            "stars_plus_gas_spherical": physical[label][("accept_absolute", 0.0, True)],
            "stars_plus_gas_sqrt_morphology": physical[label][("accept_absolute", 0.5, True)],
        }
        component_masses[label] = {
            "registered_starlight": star_mass,
            "accept_gas_spherical": gas_mass,
            "accept_gas_sqrt_morphology": gas_mass,
            "stars_plus_gas_spherical": star_mass + gas_mass,
            "stars_plus_gas_sqrt_morphology": star_mass + gas_mass,
        }
    fields = {}
    field_rows = []
    for context in contexts:
        label = context.system["label"]
        scale = float(
            context.local_protocol["cosmology_and_coordinates"][
                "angular_scale_kpc_per_arcsec"
            ]
        )
        for component in protocol["components"]:
            for extent in protocol["factorial"]["extent_scale"]:
                print(f"field {label} {component} extent={extent}", flush=True)
                field = physical_residual_field(
                    component_catalogs[label][component],
                    component_masses[label][component],
                    scale,
                    float(extent),
                    protocol,
                )
                fields[(label, component, float(extent))] = field
                field_rows.append(
                    {
                        "system_label": label,
                        "component": component,
                        **field.audit,
                    }
                )
    null_audit = circular_null_audit(protocol)
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(field_rows).to_csv(output / protocol["outputs"]["field_audits"], index=False)

    baseline_fits = {}
    for index, context in enumerate(contexts):
        print(f"zero exact fit {context.system['label']}", flush=True)
        baseline_fits[context.system["label"]] = fit_context(
            context,
            0.0,
            starts=6,
            seed=20260810 + index,
        )
    selection_labels = set(protocol["validation"]["selection_systems"])
    screen_rows = []
    for component in protocol["components"]:
        for extent in map(float, protocol["factorial"]["extent_scale"]):
            for coupling in map(float, protocol["factorial"]["response_q"]):
                per_system = []
                for context in contexts:
                    label = context.system["label"]
                    if label not in selection_labels:
                        continue
                    field = fields[(label, component, extent)]
                    candidate = replace(context, correction=field)
                    lens = MemberTidalLens(candidate.local_protocol, candidate.fields, field, coupling)
                    baseline = baseline_fits[label]
                    value = source_plane_rms(
                        lens,
                        coupling,
                        baseline["fit"]["result"].x,
                        baseline["fit"]["sources"],
                        candidate.heldout,
                    )
                    per_system.append(value)
                    screen_rows.append(
                        {
                            "row_type": "system",
                            "component": component,
                            "extent_scale": extent,
                            "response_q": coupling,
                            "system_label": label,
                            "source_plane_RMS_arcsec": value,
                        }
                    )
                screen_rows.append(
                    {
                        "row_type": "aggregate",
                        "component": component,
                        "extent_scale": extent,
                        "response_q": coupling,
                        "system_label": "selection",
                        "source_plane_RMS_arcsec": float(np.sqrt(np.mean(np.square(per_system)))),
                    }
                )
    screen = pd.DataFrame(screen_rows)
    screen.to_csv(output / protocol["outputs"]["screen_scores"], index=False)
    aggregate_screen = screen[screen.row_type.eq("aggregate")].sort_values("source_plane_RMS_arcsec")
    selected = aggregate_screen.iloc[0]
    selected_component = str(selected.component)
    selected_extent = float(selected.extent_scale)
    selected_q = float(selected.response_q)

    exact_rows = []
    prediction_tables = []
    exact_fits = {}
    for context_index, context in enumerate(contexts):
        label = context.system["label"]
        roles = [("zero", 0.0, context)]
        candidate = replace(
            context,
            correction=fields[(label, selected_component, selected_extent)],
        )
        roles.append(("selected", selected_q, candidate))
        for role_index, (role, coupling, local_context) in enumerate(roles):
            if role == "zero":
                fitted = baseline_fits[label]
            else:
                starts = 3 if label in selection_labels else 6
                print(f"selected exact fit {label} q={coupling:g}", flush=True)
                fitted = fit_context(
                    local_context,
                    coupling,
                    starts=starts,
                    seed=20260900 + context_index * 10 + role_index,
                )
            exact_fits[(label, role)] = fitted
            for table in (fitted["training_predictions"], fitted["heldout_predictions"]):
                copy = table.copy()
                copy.insert(3, "role", role)
                copy.insert(4, "component", selected_component if role == "selected" else "zero")
                prediction_tables.append(copy)
            exact_rows.append(
                {
                    "row_type": "system",
                    "role": role,
                    "system_label": label,
                    "subset": "selection" if label in selection_labels else "validation",
                    "component": selected_component if role == "selected" else "zero",
                    "extent_scale": selected_extent if role == "selected" else 1.0,
                    "response_q": coupling,
                    "training_exact_RMS_arcsec": fitted["training"]["exact_radial_RMS_arcsec"],
                    "heldout_exact_RMS_arcsec": fitted["heldout"]["exact_radial_RMS_arcsec"],
                    "all_training_roots": fitted["training"]["all_roots_converged"],
                    "all_heldout_roots": fitted["heldout"]["all_roots_converged"],
                }
            )
    exact = pd.DataFrame(exact_rows)
    for subset, labels in (
        ("selection", selection_labels),
        ("validation", set(protocol["validation"]["validation_systems"])),
        ("all_four", set(protocol["validation"]["selection_systems"] + protocol["validation"]["validation_systems"])),
    ):
        for role in ("zero", "selected"):
            block = exact[exact.system_label.isin(labels) & exact.role.eq(role)]
            exact = pd.concat(
                [
                    exact,
                    pd.DataFrame(
                        [
                            {
                                "row_type": "aggregate",
                                "role": role,
                                "system_label": subset,
                                "subset": subset,
                                "component": selected_component if role == "selected" else "zero",
                                "extent_scale": selected_extent if role == "selected" else 1.0,
                                "response_q": selected_q if role == "selected" else 0.0,
                                "training_exact_RMS_arcsec": float(np.sqrt(np.mean(block.training_exact_RMS_arcsec**2))),
                                "heldout_exact_RMS_arcsec": float(np.sqrt(np.mean(block.heldout_exact_RMS_arcsec**2))),
                                "all_training_roots": bool(block.all_training_roots.all()),
                                "all_heldout_roots": bool(block.all_heldout_roots.all()),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
    exact.to_csv(output / protocol["outputs"]["exact_scores"], index=False)
    pd.concat(prediction_tables, ignore_index=True).to_csv(
        output / protocol["outputs"]["predictions"], index=False
    )

    impact_rows = []
    aggregate = screen[screen.row_type.eq("aggregate")]
    for coordinate in ("component", "extent_scale", "response_q"):
        means = aggregate.groupby(coordinate).source_plane_RMS_arcsec.mean()
        impact_rows.append(
            {
                "coordinate": coordinate,
                "minimum_level": str(means.idxmin()),
                "minimum_mean_RMS_arcsec": float(means.min()),
                "maximum_mean_RMS_arcsec": float(means.max()),
                "main_effect_span_arcsec": float(means.max() - means.min()),
                "relative_span": float((means.max() - means.min()) / means.mean()),
            }
        )
    impacts = pd.DataFrame(impact_rows).sort_values("main_effect_span_arcsec", ascending=False)
    impacts.to_csv(output / protocol["outputs"]["parameter_impacts"], index=False)
    validation_zero = exact[(exact.row_type.eq("aggregate")) & exact.system_label.eq("validation") & exact.role.eq("zero")].iloc[0]
    validation_selected = exact[(exact.row_type.eq("aggregate")) & exact.system_label.eq("validation") & exact.role.eq("selected")].iloc[0]
    selection_selected = exact[(exact.row_type.eq("aggregate")) & exact.system_label.eq("selection") & exact.role.eq("selected")].iloc[0]
    improvement = 1.0 - float(validation_selected.heldout_exact_RMS_arcsec) / float(validation_zero.heldout_exact_RMS_arcsec)
    metric_report = json.loads((ROOT / protocol["inputs"]["metric_slip_report"]).read_text(encoding="utf-8"))
    compact = float(metric_report["comparators"]["compact_halo_validation"]["equal_system_radial_RMS_arcsec"])
    ratio = float(validation_selected.heldout_exact_RMS_arcsec) / compact
    max_curl = float(pd.DataFrame(field_rows).normalized_curl_RMS.max())
    gates = protocol["advance_gates"]
    report = {
        "report_version": "P0570-PHYSICAL-BARYON-RESIDUAL-LENSING-RESULTS-0.1.0",
        "status": "complete_physical_residual_raw_lensing_test",
        "protocol": {"path": str(protocol_path.relative_to(ROOT)), "sha256": sha256(protocol_path)},
        "coverage": {"systems": len(contexts), "screen_candidates": len(aggregate_screen), "exact_system_fits": int(len(exact[exact.row_type.eq("system")])), "components": len(protocol["components"])},
        "selected": {"component": selected_component, "extent_scale": selected_extent, "response_q": selected_q, "selection_source_plane_RMS_arcsec": float(selected.source_plane_RMS_arcsec)},
        "validation": {"selection_selected_all_roots": bool(selection_selected.all_training_roots and selection_selected.all_heldout_roots), "zero_heldout_RMS_arcsec": float(validation_zero.heldout_exact_RMS_arcsec), "selected_heldout_RMS_arcsec": float(validation_selected.heldout_exact_RMS_arcsec), "improvement_fraction": improvement, "selected_all_roots": bool(validation_selected.all_training_roots and validation_selected.all_heldout_roots), "compact_halo_RMS_arcsec": compact, "selected_to_compact_ratio": ratio},
        "parameter_impacts": json_safe(impacts.to_dict(orient="records")),
        "numerical": {"maximum_normalized_curl_RMS": max_curl, "circular_point_mass_residual_fraction": null_audit["residual_to_full_RMS_fraction"], "circular_point_mass_normalized_curl_RMS": null_audit["normalized_curl_RMS"]},
        "cross_domain": {"SPARC_outer_RMSE_km_s": protocol["cross_domain"]["fixed_RAR_outer_RMSE_km_s"], "solar_fractional_change": 0.0, "axisymmetric_null": True, "interpretation": "preserved only as an angular closure on the locked scalar parent"},
        "gates": {
            "exact_selection_roots_pass": bool(selection_selected.all_training_roots and selection_selected.all_heldout_roots),
            "validation_all_roots": bool(validation_selected.all_training_roots and validation_selected.all_heldout_roots),
            "validation_improvement_pass": bool(improvement >= float(gates["validation_improvement_vs_zero_fraction_min"])),
            "compact_halo_ratio_pass": bool(ratio <= float(gates["validation_to_compact_halo_RMS_ratio_max"])),
            "curl_pass": bool(max_curl <= float(gates["maximum_normalized_curl_RMS"])),
            "circular_null_pass": bool(null_audit["residual_to_full_RMS_fraction"] <= float(protocol["numerics"]["maximum_circular_source_residual_fraction"])),
            "formula_promoted": False,
        },
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# P0570 physical baryon residual lensing",
        "",
        f"Selected `{selected_component}`, extent scale **{selected_extent:g}**, response `q={selected_q:g}`.",
        f"Validation exact heldout RMS: **{validation_selected.heldout_exact_RMS_arcsec:.3f} arcsec** versus **{validation_zero.heldout_exact_RMS_arcsec:.3f}** at zero.",
        f"Improvement: **{100*improvement:.3f}%**; compact-halo ratio: **{ratio:.3f}**.",
        f"Exact selection roots preserved: **{bool(selection_selected.all_training_roots and selection_selected.all_heldout_roots)}**.",
        f"Largest screen coordinate: **{impacts.iloc[0].coordinate}**.",
        f"SPARC/Solar are unchanged only through the explicit axisymmetric angular null.",
    ]
    (output / protocol["outputs"]["summary"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    top = aggregate_screen.head(12).sort_values("source_plane_RMS_arcsec")
    labels = [f"{row.component}\ne={row.extent_scale:g}, q={row.response_q:g}" for row in top.itertuples(index=False)]
    axes[0].barh(labels, top.source_plane_RMS_arcsec)
    axes[0].invert_yaxis(); axes[0].set_xlabel("selection source-plane RMS (arcsec)"); axes[0].tick_params(axis="y", labelsize=6)
    system_exact = exact[exact.row_type.eq("system")].pivot(index="system_label", columns="role", values="heldout_exact_RMS_arcsec")
    x = np.arange(len(system_exact))
    axes[1].bar(x - 0.18, system_exact.zero, 0.36, label="zero")
    axes[1].bar(x + 0.18, system_exact.selected, 0.36, label="selected")
    axes[1].set_xticks(x, system_exact.index, rotation=30, ha="right"); axes[1].set_ylabel("heldout exact RMS (arcsec)"); axes[1].legend()
    axes[2].barh(impacts.coordinate, impacts.main_effect_span_arcsec)
    axes[2].set_xlabel("screen main-effect span (arcsec)")
    fig.suptitle("P0570 measured noncircular baryon deflection")
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)
    print(json.dumps(report["selected"], indent=2), flush=True)
    print(json.dumps(report["validation"], indent=2), flush=True)
    print(json.dumps(report["gates"], indent=2), flush=True)


if __name__ == "__main__":
    main()
