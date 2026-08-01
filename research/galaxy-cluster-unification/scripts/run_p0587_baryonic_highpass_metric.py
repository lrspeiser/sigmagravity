#!/usr/bin/env python3
"""Test a baryon-defined high-pass projection of the P0586D tidal metric."""

from __future__ import annotations

import itertools
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

from run_member_tidal_metric import MemberTidalLens, build_contexts, fit_context  # noqa: E402
from run_p0559_accept_projected_gas_tidal import (  # noqa: E402
    physical_catalogs,
    prepare_registered_maps,
)
from run_p0570_physical_baryon_residual_lensing import source_plane_rms  # noqa: E402
from run_p0586_continuous_baryonic_metric import (  # noqa: E402
    aggregate_exact,
    json_safe,
    sha256,
)
from run_p0586d_signed_metric_exact import affine_vector_r2  # noqa: E402
from voidscreen.baryonic_metric import (  # noqa: E402
    build_baryonic_metric_correction_field,
    prepare_baryonic_metric_state,
    prepare_baryonic_metric_workspace,
    remove_baryonic_affine_modes,
)


def highpass_id(mode, aperture, removal):
    return f"{mode}_a{aperture:.2f}_f{removal:.1f}".replace(".", "d")


def main():
    protocol_path = ROOT / "configs/p0587_baryonic_highpass_metric_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_any_baryonic_highpass_metric_score":
        raise RuntimeError("P0587 protocol is not frozen")
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
    base = protocol["locked_base_metric"]
    primary = protocol["highpass_formula"]["primary"]
    factorial = protocol["factorial"]
    numerical = p0586["numerics"]

    catalogs = {}
    masses = {}
    workspaces = {}
    states = {}
    raw_fields = {}
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
        print(f"P0587 workspace {label}", flush=True)
        workspace = prepare_baryonic_metric_workspace(
            catalog.x_arcsec.to_numpy(float),
            catalog.y_arcsec.to_numpy(float),
            catalog.normalized_light_weight.to_numpy(float),
            total_mass_msun=total_mass,
            scale_kpc_per_arcsec=scale,
            half_width_arcsec=float(numerical["field_half_width_arcsec"]),
            pixels_per_axis=int(numerical["field_pixels_per_axis"]),
            point_softening_arcsec=float(numerical["point_softening_arcsec"]),
        )
        state = prepare_baryonic_metric_state(
            workspace, float(base["smoothing_r80_fraction"])
        )
        workspaces[label] = workspace
        states[label] = state
        raw_fields[label] = build_baryonic_metric_correction_field(
            catalog.x_arcsec.to_numpy(float),
            catalog.y_arcsec.to_numpy(float),
            catalog.normalized_light_weight.to_numpy(float),
            total_mass_msun=total_mass,
            scale_kpc_per_arcsec=scale,
            minimum_permittivity=float(base["minimum_permittivity"]),
            a0_m_s2=float(base["a0_m_s2"]),
            gate_power=float(base["gate_power"]),
            anisotropy=float(base["anisotropy_tau"]),
            smoothing_r80_fraction=float(base["smoothing_r80_fraction"]),
            asymmetry_threshold=float(numerical["asymmetry_threshold"]),
            asymmetry_power=float(numerical["asymmetry_power"]),
            workspace=workspace,
            state=state,
        )

    candidates = [("raw_metric", None, math.nan, 0.0)]
    candidates.extend(
        (
            highpass_id(mode, aperture, removal),
            mode,
            float(aperture),
            float(removal),
        )
        for mode, aperture, removal in itertools.product(
            factorial["modes"],
            map(float, factorial["aperture_r80_fraction"]),
            map(float, factorial["removal_fraction"]),
        )
    )
    if len(candidates) != int(factorial["candidate_count"]):
        raise RuntimeError("P0587 candidate count differs from the protocol")
    fields = {}
    for candidate_id, mode, aperture, removal in candidates:
        for context in contexts:
            label = context.system["label"]
            if candidate_id == "raw_metric":
                fields[(candidate_id, label)] = raw_fields[label]
            else:
                fields[(candidate_id, label)] = remove_baryonic_affine_modes(
                    raw_fields[label],
                    aperture_r80_fraction=aperture,
                    removal_fraction=removal,
                    mode=mode,
                    taper_outer_factor=float(primary["taper_outer_factor"]),
                )

    baseline_fits = {}
    zero_screen = {}
    for index, context in enumerate(contexts):
        label = context.system["label"]
        print(f"P0587 zero exact fit {label}", flush=True)
        fitted = fit_context(
            context,
            0.0,
            starts=12,
            seed=20261600 + index,
        )
        baseline_fits[label] = fitted
        zero_lens = MemberTidalLens(
            context.local_protocol, context.fields, context.correction, 0.0
        )
        zero_screen[label] = source_plane_rms(
            zero_lens,
            0.0,
            fitted["fit"]["result"].x,
            fitted["fit"]["sources"],
            context.heldout,
        )

    screen_rows = []
    audit_rows = []
    for candidate_id, mode, aperture, removal in candidates:
        for context in contexts:
            label = context.system["label"]
            field = fields[(candidate_id, label)]
            fitted = baseline_fits[label]
            lens = MemberTidalLens(context.local_protocol, context.fields, field, 1.0)
            score = source_plane_rms(
                lens,
                1.0,
                fitted["fit"]["result"].x,
                fitted["fit"]["sources"],
                context.heldout,
            )
            images = pd.concat([context.training, context.heldout], ignore_index=True)
            image_r2 = affine_vector_r2(field, images)
            screen_rows.append(
                {
                    "row_type": "system",
                    "candidate_id": candidate_id,
                    "mode": mode or "raw",
                    "aperture_r80_fraction": aperture,
                    "removal_fraction": removal,
                    "system_label": label,
                    "source_plane_RMS_arcsec": score,
                    "zero_source_plane_RMS_arcsec": zero_screen[label],
                    "improvement_fraction": 1.0 - score / zero_screen[label],
                    "affine_vector_R2_on_images": image_r2,
                }
            )
            audit_rows.append(
                {
                    "candidate_id": candidate_id,
                    "mode": mode or "raw",
                    "aperture_r80_fraction": aperture,
                    "removal_fraction": removal,
                    "system_label": label,
                    "affine_vector_R2_on_images": image_r2,
                    **field.audit,
                }
            )
        print(f"P0587 screen {candidate_id}", flush=True)
    screen = pd.DataFrame(screen_rows)
    audits = pd.DataFrame(audit_rows)
    aggregate_rows = []
    for candidate_id, block in screen.groupby("candidate_id", sort=False):
        first = block.iloc[0]
        aggregate_rows.append(
            {
                "row_type": "aggregate",
                "candidate_id": candidate_id,
                "mode": first["mode"],
                "aperture_r80_fraction": first.aperture_r80_fraction,
                "removal_fraction": float(first.removal_fraction),
                "system_label": "all_four",
                "source_plane_RMS_arcsec": float(
                    np.sqrt(np.mean(np.square(block.source_plane_RMS_arcsec)))
                ),
                "zero_source_plane_RMS_arcsec": float(
                    np.sqrt(np.mean(np.square(block.zero_source_plane_RMS_arcsec)))
                ),
                "improvement_fraction": float(
                    1.0
                    - np.sqrt(np.mean(np.square(block.source_plane_RMS_arcsec)))
                    / np.sqrt(np.mean(np.square(block.zero_source_plane_RMS_arcsec)))
                ),
                "affine_vector_R2_on_images": float(
                    block.affine_vector_R2_on_images.max()
                ),
            }
        )
    screen = pd.concat([screen, pd.DataFrame(aggregate_rows)], ignore_index=True)

    highpass_aggregate = screen[
        screen.row_type.eq("aggregate") & ~screen.candidate_id.eq("raw_metric")
    ]
    impact_rows = []
    for coordinate in ("mode", "aperture_r80_fraction", "removal_fraction"):
        means = highpass_aggregate.groupby(coordinate).source_plane_RMS_arcsec.mean()
        impact_rows.append(
            {
                "coordinate": coordinate,
                "best_main_effect_level": str(means.idxmin()),
                "main_effect_span_arcsec": float(means.max() - means.min()),
            }
        )
    impacts = pd.DataFrame(impact_rows).sort_values(
        "main_effect_span_arcsec", ascending=False
    )

    primary_id = highpass_id(
        primary["mode"],
        float(primary["aperture_r80_fraction"]),
        float(primary["removal_fraction"]),
    )
    exact_models = [
        ("zero", None, 0.0),
        ("raw_metric", "raw_metric", 1.0),
        ("highpass_primary", primary_id, 1.0),
    ]
    exact_rows = []
    prediction_tables = []
    for model_index, (model_id, candidate_id, coupling) in enumerate(exact_models):
        for context_index, context in enumerate(contexts):
            label = context.system["label"]
            local_context = (
                context
                if model_id == "zero"
                else replace(context, correction=fields[(candidate_id, label)])
            )
            print(f"P0587 exact {label} {model_id}", flush=True)
            fitted = fit_context(
                local_context,
                coupling,
                starts=12,
                seed=20261700 + 100 * model_index + context_index,
            )
            for table in (fitted["training_predictions"], fitted["heldout_predictions"]):
                copy = table.copy()
                copy.insert(3, "model_id", model_id)
                prediction_tables.append(copy)
            exact_rows.append(
                {
                    "row_type": "system",
                    "model_id": model_id,
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
    for model_id, _, _ in exact_models:
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
    screen.to_csv(output / protocol["outputs"]["screen_scores"], index=False)
    audits.to_csv(output / protocol["outputs"]["field_audits"], index=False)
    impacts.to_csv(output / protocol["outputs"]["parameter_impacts"], index=False)
    exact.to_csv(output / protocol["outputs"]["exact_scores"], index=False)
    pd.concat(prediction_tables, ignore_index=True).to_csv(
        output / protocol["outputs"]["predictions"], index=False
    )

    aggregate_exact_rows = exact[
        exact.row_type.eq("aggregate") & exact.system_label.eq("all_four")
    ].set_index("model_id")
    zero = aggregate_exact_rows.loc["zero"]
    raw = aggregate_exact_rows.loc["raw_metric"]
    highpass = aggregate_exact_rows.loc["highpass_primary"]
    gain_zero = 1.0 - float(highpass.heldout_exact_RMS_arcsec) / float(
        zero.heldout_exact_RMS_arcsec
    )
    gain_raw = 1.0 - float(highpass.heldout_exact_RMS_arcsec) / float(
        raw.heldout_exact_RMS_arcsec
    )
    highpass_systems = exact[
        exact.row_type.eq("system") & exact.model_id.eq("highpass_primary")
    ].set_index("system_label")
    zero_systems = exact[
        exact.row_type.eq("system") & exact.model_id.eq("zero")
    ].set_index("system_label")
    all_systems_improve = bool(
        (
            highpass_systems.heldout_exact_RMS_arcsec
            < zero_systems.heldout_exact_RMS_arcsec
        ).all()
    )
    all_roots = bool(highpass.all_training_roots and highpass.all_heldout_roots)
    primary_audits = audits[audits.candidate_id.eq(primary_id)]
    max_affine = float(primary_audits.affine_vector_R2_on_images.max())
    max_curl = float(primary_audits.normalized_curl_RMS.max())
    min_eigenvalue = float(primary_audits.metric_minimum_eigenvalue.min())
    metric_report = json.loads(
        (ROOT / protocol["inputs"]["metric_slip_report"]).read_text(encoding="utf-8")
    )
    compact = float(
        metric_report["comparators"]["compact_halo_validation"][
            "equal_system_radial_RMS_arcsec"
        ]
    )
    compact_ratio = float(highpass.heldout_exact_RMS_arcsec) / compact
    gates = protocol["advance_gates"]
    report = {
        "report_version": "P0587-BARYONIC-HIGHPASS-METRIC-RESULTS-0.1.0",
        "status": "complete_baryonic_highpass_metric_test",
        "protocol": {
            "path": str(protocol_path.relative_to(ROOT)),
            "sha256": sha256(protocol_path),
        },
        "coverage": {
            "clusters": len(contexts),
            "screen_candidates": len(candidates),
            "screen_system_fields": len(audits),
            "exact_models": len(exact_models),
            "exact_system_fits": int(len(exact[exact.row_type.eq("system")])),
        },
        "primary": {"candidate_id": primary_id, **primary},
        "screen": {
            "raw_metric": json_safe(
                screen[
                    screen.row_type.eq("aggregate")
                    & screen.candidate_id.eq("raw_metric")
                ].iloc[0].to_dict()
            ),
            "highpass_primary": json_safe(
                screen[
                    screen.row_type.eq("aggregate")
                    & screen.candidate_id.eq(primary_id)
                ].iloc[0].to_dict()
            ),
            "diagnostic_best": json_safe(
                highpass_aggregate.sort_values("source_plane_RMS_arcsec").iloc[0].to_dict()
            ),
        },
        "exact": {
            "zero_RMS_arcsec": float(zero.heldout_exact_RMS_arcsec),
            "raw_metric_RMS_arcsec": float(raw.heldout_exact_RMS_arcsec),
            "highpass_primary_RMS_arcsec": float(highpass.heldout_exact_RMS_arcsec),
            "highpass_improvement_vs_zero_fraction": gain_zero,
            "highpass_improvement_vs_raw_fraction": gain_raw,
            "highpass_all_systems_improve": all_systems_improve,
            "highpass_all_training_roots": bool(highpass.all_training_roots),
            "highpass_all_heldout_roots": bool(highpass.all_heldout_roots),
            "compact_halo_RMS_arcsec": compact,
            "highpass_to_compact_ratio": compact_ratio,
        },
        "parameter_impacts": json_safe(impacts.to_dict(orient="records")),
        "numerical": {
            "maximum_primary_affine_R2_on_images": max_affine,
            "maximum_primary_normalized_curl_RMS": max_curl,
            "minimum_base_metric_eigenvalue": min_eigenvalue,
            "primary_field_audits": json_safe(
                primary_audits.to_dict(orient="records")
            ),
        },
        "cross_domain": {
            "spherical_SPARC_RMSE_km_s": 72.39921475798786,
            "solar_fractional_change": 0.0,
        },
        "gates": {
            "primary_all_roots": all_roots,
            "primary_all_four_systems_improve": all_systems_improve,
            "primary_improvement_vs_zero_pass": bool(
                all_roots
                and gain_zero
                >= float(gates["primary_improvement_vs_zero_fraction_min"])
            ),
            "primary_improvement_vs_raw_pass": bool(
                all_roots
                and gain_raw
                >= float(gates["primary_improvement_vs_raw_metric_fraction_min"])
            ),
            "compact_halo_ratio_pass": bool(
                all_roots
                and compact_ratio
                <= float(gates["primary_to_compact_halo_RMS_ratio_max"])
            ),
            "affine_audit_pass": bool(
                max_affine <= float(gates["maximum_affine_R2_on_images"])
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
        "# P0587 baryon-defined high-pass metric",
        "",
        f"Exact RMS: zero **{zero.heldout_exact_RMS_arcsec:.4f}**, raw metric **{raw.heldout_exact_RMS_arcsec:.4f}**, high-pass primary **{highpass.heldout_exact_RMS_arcsec:.4f} arcsec**.",
        f"High-pass gain versus zero: **{100*gain_zero:.3f}%**; versus raw metric: **{100*gain_raw:.3f}%**.",
        f"All roots: **{all_roots}**; all systems improve: **{all_systems_improve}**; maximum image-sampled affine R2: **{max_affine:.4f}**.",
    ]
    (output / protocol["outputs"]["summary"]).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    aggregate_screen = screen[screen.row_type.eq("aggregate")].sort_values(
        "source_plane_RMS_arcsec"
    )
    axes[0].barh(
        aggregate_screen.candidate_id,
        aggregate_screen.source_plane_RMS_arcsec,
    )
    axes[0].invert_yaxis()
    axes[0].set_xlabel("fixed-geometry all-four RMS (arcsec)")
    axes[0].tick_params(axis="y", labelsize=6)
    system = exact[exact.row_type.eq("system")].pivot(
        index="system_label", columns="model_id", values="heldout_exact_RMS_arcsec"
    )
    position = np.arange(len(system))
    axes[1].bar(position - 0.22, system.zero, 0.22, label="zero")
    axes[1].bar(position, system.raw_metric, 0.22, label="raw")
    axes[1].bar(position + 0.22, system.highpass_primary, 0.22, label="high-pass")
    axes[1].set_xticks(position, system.index, rotation=30, ha="right")
    axes[1].set_ylabel("heldout exact RMS (arcsec)")
    axes[1].legend()
    axes[2].barh(
        primary_audits.system_label,
        primary_audits.affine_vector_R2_on_images,
    )
    axes[2].axvline(
        float(protocol["numerical_audits"]["maximum_affine_R2_on_images"]),
        color="red",
        ls="--",
    )
    axes[2].set_xlabel("affine R2 on observed images")
    fig.suptitle("P0587 baryon-defined high-pass tidal metric")
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)
    print(json.dumps(report["screen"], indent=2), flush=True)
    print(json.dumps(report["exact"], indent=2), flush=True)
    print(json.dumps(report["numerical"], indent=2), flush=True)
    print(json.dumps(report["gates"], indent=2), flush=True)


if __name__ == "__main__":
    main()
