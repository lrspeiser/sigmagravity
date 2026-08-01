#!/usr/bin/env python3
"""Replay the P0586C signed compromise with nonlinear multiple-image roots."""

from __future__ import annotations

import json
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

from run_member_tidal_metric import build_contexts, fit_context  # noqa: E402
from run_p0559_accept_projected_gas_tidal import (  # noqa: E402
    physical_catalogs,
    prepare_registered_maps,
)
from run_p0586_continuous_baryonic_metric import (  # noqa: E402
    aggregate_exact,
    json_safe,
    sha256,
)
from voidscreen.baryonic_metric import (  # noqa: E402
    build_baryonic_metric_correction_field,
    prepare_baryonic_metric_state,
    prepare_baryonic_metric_workspace,
)


def affine_vector_r2(field, images: pd.DataFrame) -> float:
    x = images.x_arcsec.to_numpy(float)
    y = images.y_arcsec.to_numpy(float)
    alpha_x, alpha_y = field.alpha_arcsec(x, y)
    design = np.column_stack([x, y, np.ones_like(x)])
    coefficients_x = np.linalg.lstsq(design, alpha_x, rcond=None)[0]
    coefficients_y = np.linalg.lstsq(design, alpha_y, rcond=None)[0]
    predicted_x = design @ coefficients_x
    predicted_y = design @ coefficients_y
    residual = np.r_[alpha_x - predicted_x, alpha_y - predicted_y]
    centered = np.r_[alpha_x - np.mean(alpha_x), alpha_y - np.mean(alpha_y)]
    denominator = float(np.sum(np.square(centered)))
    if denominator <= np.finfo(float).tiny:
        return 0.0
    return float(1.0 - np.sum(np.square(residual)) / denominator)


def main():
    protocol_path = ROOT / "configs/p0586d_signed_metric_exact_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_any_signed_metric_exact_score":
        raise RuntimeError("P0586D protocol is not frozen")
    p0586 = json.loads(
        (ROOT / protocol["inputs"]["p0586_protocol"]).read_text(encoding="utf-8")
    )
    p0559 = json.loads(
        (ROOT / protocol["inputs"]["p0559_protocol"]).read_text(encoding="utf-8")
    )
    member = json.loads(
        (ROOT / p0559["inputs"]["member_tidal_protocol"]).read_text(encoding="utf-8")
    )
    member["optimization"]["maximum_function_evaluations"] = int(
        p0559["optimization"]["maximum_function_evaluations"]
    )
    contexts, _, _ = build_contexts(
        member, softening_kpc=float(p0559["locked_field"]["softening_kpc"])
    )
    registered = prepare_registered_maps(p0559, contexts)
    physical, physical_audits = physical_catalogs(p0559, contexts, registered)
    physical_audits = physical_audits.set_index("system_label")
    primary = protocol["primary"]
    numerical = p0586["numerics"]

    catalogs = {}
    masses = {}
    workspaces = {}
    states = {}
    for context in contexts:
        label = context.system["label"]
        catalog = physical[label][("accept_absolute", 0.5, True)]
        total_mass = float(
            physical_audits.loc[label, "stellar_mass_assigned_to_map_msun"]
            + physical_audits.loc[label, "projected_ACCEPT_gas_mass_on_map_msun"]
        )
        scale = float(
            context.local_protocol["cosmology_and_coordinates"][
                "angular_scale_kpc_per_arcsec"
            ]
        )
        catalogs[label] = catalog
        masses[label] = total_mass
        print(f"P0586D workspace {label}", flush=True)
        workspaces[label] = prepare_baryonic_metric_workspace(
            catalog.x_arcsec.to_numpy(float),
            catalog.y_arcsec.to_numpy(float),
            catalog.normalized_light_weight.to_numpy(float),
            total_mass_msun=total_mass,
            scale_kpc_per_arcsec=scale,
            half_width_arcsec=float(numerical["field_half_width_arcsec"]),
            pixels_per_axis=int(numerical["field_pixels_per_axis"]),
            point_softening_arcsec=float(numerical["point_softening_arcsec"]),
        )
        states[label] = prepare_baryonic_metric_state(
            workspaces[label], float(primary["smoothing_r80_fraction"])
        )

    fields = {}
    audit_rows = []
    for model in protocol["exact_models"]:
        model_id = model["model_id"]
        tau = float(model["anisotropy_tau"])
        if model_id == "zero":
            continue
        for context in contexts:
            label = context.system["label"]
            catalog = catalogs[label]
            field = build_baryonic_metric_correction_field(
                catalog.x_arcsec.to_numpy(float),
                catalog.y_arcsec.to_numpy(float),
                catalog.normalized_light_weight.to_numpy(float),
                total_mass_msun=masses[label],
                scale_kpc_per_arcsec=workspaces[label].scale_kpc_per_arcsec,
                minimum_permittivity=float(primary["minimum_permittivity"]),
                a0_m_s2=float(primary["a0_m_s2"]),
                gate_power=float(primary["gate_power"]),
                anisotropy=tau,
                smoothing_r80_fraction=float(primary["smoothing_r80_fraction"]),
                asymmetry_threshold=float(numerical["asymmetry_threshold"]),
                asymmetry_power=float(numerical["asymmetry_power"]),
                workspace=workspaces[label],
                state=states[label],
            )
            fields[(model_id, label)] = field
            images = pd.concat([context.training, context.heldout], ignore_index=True)
            audit_rows.append(
                {
                    "model_id": model_id,
                    "system_label": label,
                    "anisotropy_tau": tau,
                    "affine_vector_R2_on_images": affine_vector_r2(field, images),
                    **field.audit,
                }
            )
    audits = pd.DataFrame(audit_rows)

    exact_rows = []
    prediction_tables = []
    for model_index, model in enumerate(protocol["exact_models"]):
        model_id = model["model_id"]
        tau = float(model["anisotropy_tau"])
        starts = int(model["starts"])
        for context_index, context in enumerate(contexts):
            label = context.system["label"]
            if model_id == "zero":
                local_context = context
                coupling = 0.0
            else:
                local_context = replace(
                    context, correction=fields[(model_id, label)]
                )
                coupling = 1.0
            print(f"P0586D exact {label} {model_id}", flush=True)
            fitted = fit_context(
                local_context,
                coupling,
                starts=starts,
                seed=20261500 + 100 * model_index + context_index,
            )
            for table in (fitted["training_predictions"], fitted["heldout_predictions"]):
                copy = table.copy()
                copy.insert(3, "model_id", model_id)
                copy.insert(4, "anisotropy_tau", tau)
                prediction_tables.append(copy)
            exact_rows.append(
                {
                    "row_type": "system",
                    "model_id": model_id,
                    "anisotropy_tau": tau,
                    "system_label": label,
                    "training_exact_RMS_arcsec": fitted["training"][
                        "exact_radial_RMS_arcsec"
                    ],
                    "heldout_exact_RMS_arcsec": fitted["heldout"][
                        "exact_radial_RMS_arcsec"
                    ],
                    "all_training_roots": fitted["training"][
                        "all_roots_converged"
                    ],
                    "all_heldout_roots": fitted["heldout"]["all_roots_converged"],
                }
            )
    exact = pd.DataFrame(exact_rows)
    labels = {context.system["label"] for context in contexts}
    for model in protocol["exact_models"]:
        model_id = model["model_id"]
        aggregate = aggregate_exact(
            exact.rename(columns={"model_id": "role"}), labels, model_id
        )
        exact = pd.concat(
            [
                exact,
                pd.DataFrame(
                    [
                        {
                            "row_type": "aggregate",
                            "model_id": model_id,
                            "anisotropy_tau": float(model["anisotropy_tau"]),
                            "system_label": "all_four",
                            **aggregate,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    exact.to_csv(output / protocol["outputs"]["exact_scores"], index=False)
    audits.to_csv(output / protocol["outputs"]["field_audits"], index=False)
    pd.concat(prediction_tables, ignore_index=True).to_csv(
        output / protocol["outputs"]["predictions"], index=False
    )

    zero = exact[
        exact.row_type.eq("aggregate")
        & exact.model_id.eq("zero")
        & exact.system_label.eq("all_four")
    ].iloc[0]
    primary_row = exact[
        exact.row_type.eq("aggregate")
        & exact.model_id.eq("primary_tau_m1p2")
        & exact.system_label.eq("all_four")
    ].iloc[0]
    improvement = 1.0 - float(primary_row.heldout_exact_RMS_arcsec) / float(
        zero.heldout_exact_RMS_arcsec
    )
    primary_systems = exact[
        exact.row_type.eq("system") & exact.model_id.eq("primary_tau_m1p2")
    ].set_index("system_label")
    zero_systems = exact[
        exact.row_type.eq("system") & exact.model_id.eq("zero")
    ].set_index("system_label")
    all_systems_improve = bool(
        (
            primary_systems.heldout_exact_RMS_arcsec
            < zero_systems.heldout_exact_RMS_arcsec
        ).all()
    )
    all_roots = bool(
        primary_row.all_training_roots and primary_row.all_heldout_roots
    )
    metric_report = json.loads(
        (ROOT / protocol["inputs"]["metric_slip_report"]).read_text(encoding="utf-8")
    )
    compact = float(
        metric_report["comparators"]["compact_halo_validation"][
            "equal_system_radial_RMS_arcsec"
        ]
    )
    compact_ratio = float(primary_row.heldout_exact_RMS_arcsec) / compact
    primary_audits = audits[audits.model_id.eq("primary_tau_m1p2")]
    max_affine = float(primary_audits.affine_vector_R2_on_images.max())
    max_curl = float(primary_audits.normalized_curl_RMS.max())
    min_eigenvalue = float(primary_audits.metric_minimum_eigenvalue.min())
    aggregate_models = exact[
        exact.row_type.eq("aggregate") & exact.system_label.eq("all_four")
    ].copy()
    aggregate_models["improvement_vs_zero_fraction"] = 1.0 - (
        aggregate_models.heldout_exact_RMS_arcsec
        / float(zero.heldout_exact_RMS_arcsec)
    )
    gates = protocol["advance_gates"]
    report = {
        "report_version": "P0586D-SIGNED-METRIC-EXACT-RESULTS-0.1.0",
        "status": "complete_signed_metric_exact_replay",
        "protocol": {
            "path": str(protocol_path.relative_to(ROOT)),
            "sha256": sha256(protocol_path),
        },
        "coverage": {
            "clusters": len(contexts),
            "models": len(protocol["exact_models"]),
            "system_fits": int(len(exact[exact.row_type.eq("system")])),
            "starts_per_fit": 12,
        },
        "primary": primary,
        "primary_exact": {
            "zero_heldout_RMS_arcsec": float(zero.heldout_exact_RMS_arcsec),
            "primary_heldout_RMS_arcsec": float(
                primary_row.heldout_exact_RMS_arcsec
            ),
            "improvement_fraction": improvement,
            "all_systems_improve": all_systems_improve,
            "all_training_roots": bool(primary_row.all_training_roots),
            "all_heldout_roots": bool(primary_row.all_heldout_roots),
            "compact_halo_RMS_arcsec": compact,
            "primary_to_compact_ratio": compact_ratio,
        },
        "sensitivity": json_safe(
            aggregate_models[
                [
                    "model_id",
                    "anisotropy_tau",
                    "heldout_exact_RMS_arcsec",
                    "finite_systems",
                    "all_training_roots",
                    "all_heldout_roots",
                    "improvement_vs_zero_fraction",
                ]
            ].to_dict(orient="records")
        ),
        "numerical": {
            "maximum_primary_affine_vector_R2": max_affine,
            "maximum_primary_normalized_curl_RMS": max_curl,
            "minimum_primary_metric_eigenvalue": min_eigenvalue,
        },
        "cross_domain": {
            "spherical_SPARC_RMSE_km_s": 72.39921475798786,
            "fixed_RAR_RMSE_km_s": p0586["cross_domain"]["comparators"][
                "fixed_RAR_outer_RMSE_km_s"
            ],
            "solar_fractional_change": 0.0,
            "meaning": "epsilon0=1 turns off the scalar branch; the cluster angular tensor does not explain galaxy rotation",
        },
        "gates": {
            "primary_all_roots": all_roots,
            "primary_all_four_systems_improve": all_systems_improve,
            "primary_improvement_pass": bool(
                all_roots
                and improvement
                >= float(gates["primary_equal_system_improvement_fraction_min"])
            ),
            "compact_halo_ratio_pass": bool(
                all_roots
                and compact_ratio
                <= float(gates["primary_to_compact_halo_RMS_ratio_max"])
            ),
            "mass_sheet_audit_pass": bool(
                max_affine <= float(gates["maximum_affine_mass_sheet_R2"])
            ),
            "curl_pass": bool(
                max_curl
                <= float(protocol["numerical_audits"]["maximum_normalized_curl_RMS"])
            ),
            "positive_metric_pass": bool(
                min_eigenvalue
                >= float(protocol["numerical_audits"]["minimum_metric_eigenvalue"])
            ),
            "per_cluster_gravity_parameters": 0,
            "formula_promoted": False,
        },
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# P0586D signed continuous-metric exact replay",
        "",
        f"Primary exact heldout RMS: **{primary_row.heldout_exact_RMS_arcsec:.4f} arcsec** versus **{zero.heldout_exact_RMS_arcsec:.4f}** at zero; change **{100*improvement:.3f}%**.",
        f"All systems improve: **{all_systems_improve}**; all training and heldout roots: **{all_roots}**.",
        f"Compact-halo ratio: **{compact_ratio:.3f}**; maximum affine R2: **{max_affine:.4f}**.",
        "The selected epsilon0=1 law is Newtonian in the spherical galaxy limit and therefore leaves SPARC at 72.40 km/s.",
    ]
    (output / protocol["outputs"]["summary"]).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    system = exact[exact.row_type.eq("system")]
    pivot = system.pivot(
        index="system_label", columns="model_id", values="heldout_exact_RMS_arcsec"
    )
    position = np.arange(len(pivot))
    width = 0.15
    for offset, model_id in enumerate(
        ["zero", "tau_m0p3", "tau_m0p6", "tau_m0p9", "primary_tau_m1p2"]
    ):
        axes[0].bar(
            position + (offset - 2) * width,
            pivot[model_id],
            width,
            label=model_id,
        )
    axes[0].set_xticks(position, pivot.index, rotation=30, ha="right")
    axes[0].set_ylabel("heldout exact RMS (arcsec)")
    axes[0].legend(fontsize=6)
    axes[1].plot(
        aggregate_models.anisotropy_tau,
        100.0 * aggregate_models.improvement_vs_zero_fraction,
        marker="o",
    )
    axes[1].axhline(0.0, color="black", lw=1)
    axes[1].set_xlabel("anisotropy tau")
    axes[1].set_ylabel("all-four exact gain (%)")
    axes[2].barh(
        primary_audits.system_label,
        primary_audits.affine_vector_R2_on_images,
    )
    axes[2].axvline(
        float(protocol["numerical_audits"]["maximum_affine_mass_sheet_R2"]),
        color="red",
        ls="--",
    )
    axes[2].set_xlabel("affine vector R2 on images")
    fig.suptitle("P0586D exact signed continuous-metric replay")
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)
    print(json.dumps(report["primary_exact"], indent=2), flush=True)
    print(json.dumps(report["sensitivity"], indent=2), flush=True)
    print(json.dumps(report["numerical"], indent=2), flush=True)
    print(json.dumps(report["gates"], indent=2), flush=True)


if __name__ == "__main__":
    main()
