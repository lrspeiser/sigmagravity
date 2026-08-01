#!/usr/bin/env python3
"""Extend the high-impact P0586 anisotropy and baryonic-reach boundaries."""

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
    candidate_id,
    equal_system_velocity_rmse,
    galaxy_prediction,
    json_safe,
    sha256,
)
from voidscreen.baryonic_metric import (  # noqa: E402
    build_baryonic_metric_correction_field,
    prepare_baryonic_metric_state,
    prepare_baryonic_metric_workspace,
)


def aggregate_source_plane(frame, labels):
    block = frame[frame.system_label.isin(labels)]
    return float(np.sqrt(np.mean(np.square(block.source_plane_RMS_arcsec))))


def main():
    protocol_path = ROOT / "configs/p0586b_metric_boundary_response_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_any_metric_boundary_extension_score":
        raise RuntimeError("P0586B protocol is not frozen")
    p0586 = json.loads(
        (ROOT / protocol["inputs"]["p0586_protocol"]).read_text(encoding="utf-8")
    )
    p0559 = json.loads(
        (ROOT / protocol["inputs"]["p0559_protocol"]).read_text(encoding="utf-8")
    )
    p0557 = json.loads(
        (ROOT / p0559["inputs"]["p0557_protocol"]).read_text(encoding="utf-8")
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
    selection_labels = set(protocol["validation"]["selection_systems"])
    validation_labels = set(protocol["validation"]["validation_systems"])
    locked = protocol["locked_from_p0586"]
    factorial = protocol["factorial"]
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
        print(f"P0586B workspace {label}", flush=True)
        workspaces[label] = prepare_baryonic_metric_workspace(
            catalog.x_arcsec.to_numpy(float),
            catalog.y_arcsec.to_numpy(float),
            catalog.normalized_light_weight.to_numpy(float),
            total_mass_msun=total_mass,
            scale_kpc_per_arcsec=scale,
            half_width_arcsec=float(locked["field_half_width_arcsec"]),
            pixels_per_axis=int(locked["field_pixels_per_axis"]),
            point_softening_arcsec=float(locked["point_softening_arcsec"]),
        )
        for width in map(float, factorial["smoothing_r80_fraction"]):
            states[(label, width)] = prepare_baryonic_metric_state(
                workspaces[label], width
            )

    baseline_fits = {}
    zero_screen = {}
    for index, context in enumerate(contexts):
        label = context.system["label"]
        print(f"P0586B zero exact fit {label}", flush=True)
        fitted = fit_context(
            context,
            0.0,
            starts=8,
            seed=20261200 + index,
        )
        baseline_fits[label] = fitted
        lens = MemberTidalLens(
            context.local_protocol, context.fields, context.correction, 0.0
        )
        zero_screen[label] = source_plane_rms(
            lens,
            0.0,
            fitted["fit"]["result"].x,
            fitted["fit"]["sources"],
            context.heldout,
        )

    grid = list(
        itertools.product(
            map(float, factorial["minimum_permittivity"]),
            map(float, factorial["anisotropy_tau"]),
            map(float, factorial["smoothing_r80_fraction"]),
        )
    )
    if len(grid) != int(factorial["candidate_count"]):
        raise RuntimeError("P0586B candidate count differs from the frozen protocol")
    rows = []
    for candidate_index, (epsilon, tau, width) in enumerate(grid):
        cid = candidate_id(
            epsilon,
            float(locked["a0_m_s2"]),
            float(locked["gate_power"]),
            tau,
            width,
        )
        for context in contexts:
            label = context.system["label"]
            catalog = catalogs[label]
            field = build_baryonic_metric_correction_field(
                catalog.x_arcsec.to_numpy(float),
                catalog.y_arcsec.to_numpy(float),
                catalog.normalized_light_weight.to_numpy(float),
                total_mass_msun=masses[label],
                scale_kpc_per_arcsec=workspaces[label].scale_kpc_per_arcsec,
                minimum_permittivity=epsilon,
                a0_m_s2=float(locked["a0_m_s2"]),
                gate_power=float(locked["gate_power"]),
                anisotropy=tau,
                smoothing_r80_fraction=width,
                asymmetry_threshold=float(numerical["asymmetry_threshold"]),
                asymmetry_power=float(numerical["asymmetry_power"]),
                workspace=workspaces[label],
                state=states[(label, width)],
            )
            lens = MemberTidalLens(context.local_protocol, context.fields, field, 1.0)
            fitted = baseline_fits[label]
            value = source_plane_rms(
                lens,
                1.0,
                fitted["fit"]["result"].x,
                fitted["fit"]["sources"],
                context.heldout,
            )
            rows.append(
                {
                    "candidate_id": cid,
                    "system_label": label,
                    "minimum_permittivity": epsilon,
                    "anisotropy_tau": tau,
                    "smoothing_r80_fraction": width,
                    "source_plane_RMS_arcsec": value,
                    "zero_source_plane_RMS_arcsec": zero_screen[label],
                    "improvement_fraction": 1.0 - value / zero_screen[label],
                    "correction_RMS_arcsec": field.audit[
                        "correction_RMS_arcsec_at_distance_ratio_one"
                    ],
                    "minimum_metric_eigenvalue": field.audit[
                        "metric_minimum_eigenvalue"
                    ],
                }
            )
        if (candidate_index + 1) % 14 == 0:
            print(f"P0586B screen {candidate_index + 1}/{len(grid)}", flush=True)
    screen = pd.DataFrame(rows)
    candidate_rows = []
    for cid, block in screen.groupby("candidate_id", sort=False):
        first = block.iloc[0]
        candidate_rows.append(
            {
                "candidate_id": cid,
                "minimum_permittivity": float(first.minimum_permittivity),
                "anisotropy_tau": float(first.anisotropy_tau),
                "smoothing_r80_fraction": float(first.smoothing_r80_fraction),
                "selection_RMS_arcsec": aggregate_source_plane(
                    block, selection_labels
                ),
                "validation_RMS_arcsec": aggregate_source_plane(
                    block, validation_labels
                ),
                "all_four_RMS_arcsec": aggregate_source_plane(
                    block, selection_labels | validation_labels
                ),
                "systems_improved": int((block.improvement_fraction > 0.0).sum()),
                "all_four_improve": bool((block.improvement_fraction > 0.0).all()),
                "minimum_metric_eigenvalue": float(
                    block.minimum_metric_eigenvalue.min()
                ),
            }
        )
    candidates = pd.DataFrame(candidate_rows)
    selected = candidates.sort_values("selection_RMS_arcsec").iloc[0]
    selected_parameters = {
        "candidate_id": str(selected.candidate_id),
        "minimum_permittivity": float(selected.minimum_permittivity),
        "anisotropy_tau": float(selected.anisotropy_tau),
        "smoothing_r80_fraction": float(selected.smoothing_r80_fraction),
        "a0_m_s2": float(locked["a0_m_s2"]),
        "gate_power": float(locked["gate_power"]),
        "selection_RMS_arcsec": float(selected.selection_RMS_arcsec),
        "validation_fixed_geometry_RMS_arcsec": float(
            selected.validation_RMS_arcsec
        ),
        "systems_improved_fixed_geometry": int(selected.systems_improved),
    }
    optima = (
        screen.sort_values("source_plane_RMS_arcsec")
        .groupby("system_label", as_index=False)
        .first()
    )

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    screen.to_csv(output / protocol["outputs"]["screen_scores"], index=False)
    candidates.to_csv(output / protocol["outputs"]["candidate_scores"], index=False)
    optima.to_csv(output / protocol["outputs"]["per_system_optima"], index=False)

    selected_fields = {}
    for context in contexts:
        label = context.system["label"]
        catalog = catalogs[label]
        selected_fields[label] = build_baryonic_metric_correction_field(
            catalog.x_arcsec.to_numpy(float),
            catalog.y_arcsec.to_numpy(float),
            catalog.normalized_light_weight.to_numpy(float),
            total_mass_msun=masses[label],
            scale_kpc_per_arcsec=workspaces[label].scale_kpc_per_arcsec,
            minimum_permittivity=selected_parameters["minimum_permittivity"],
            a0_m_s2=selected_parameters["a0_m_s2"],
            gate_power=selected_parameters["gate_power"],
            anisotropy=selected_parameters["anisotropy_tau"],
            smoothing_r80_fraction=selected_parameters[
                "smoothing_r80_fraction"
            ],
            asymmetry_threshold=float(numerical["asymmetry_threshold"]),
            asymmetry_power=float(numerical["asymmetry_power"]),
            workspace=workspaces[label],
            state=states[(label, selected_parameters["smoothing_r80_fraction"])],
        )

    exact_rows = []
    prediction_tables = []
    for index, context in enumerate(contexts):
        label = context.system["label"]
        for role in ("zero", "selected"):
            if role == "zero":
                fitted = baseline_fits[label]
            else:
                print(f"P0586B selected exact fit {label}", flush=True)
                fitted = fit_context(
                    replace(context, correction=selected_fields[label]),
                    1.0,
                    starts=8,
                    seed=20261300 + index,
                )
            for table in (fitted["training_predictions"], fitted["heldout_predictions"]):
                copy = table.copy()
                copy.insert(3, "role", role)
                prediction_tables.append(copy)
            exact_rows.append(
                {
                    "row_type": "system",
                    "role": role,
                    "system_label": label,
                    "subset": "selection" if label in selection_labels else "validation",
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
    for subset, labels in (
        ("selection", selection_labels),
        ("validation", validation_labels),
        ("all_four", selection_labels | validation_labels),
    ):
        for role in ("zero", "selected"):
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
                                **aggregate_exact(exact, labels, role),
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
    validation_zero = exact[
        exact.row_type.eq("aggregate")
        & exact.system_label.eq("validation")
        & exact.role.eq("zero")
    ].iloc[0]
    validation_selected = exact[
        exact.row_type.eq("aggregate")
        & exact.system_label.eq("validation")
        & exact.role.eq("selected")
    ].iloc[0]
    exact_improvement = 1.0 - float(
        validation_selected.heldout_exact_RMS_arcsec
    ) / float(validation_zero.heldout_exact_RMS_arcsec)
    selected_all_roots = bool(
        validation_selected.all_training_roots
        and validation_selected.all_heldout_roots
    )
    tau_values = list(map(float, factorial["anisotropy_tau"]))
    width_values = list(map(float, factorial["smoothing_r80_fraction"]))
    interior = bool(
        selected_parameters["anisotropy_tau"] not in {min(tau_values), max(tau_values)}
        and selected_parameters["smoothing_r80_fraction"]
        not in {min(width_values), max(width_values)}
    )
    all_four_candidates = candidates[candidates.all_four_improve]

    sparc = pd.read_csv(ROOT / protocol["inputs"]["SPARC_points"])
    sparc = sparc[
        sparc.model.eq("fixed_RAR")
        & sparc.scenario.eq("invariant")
        & sparc.split.eq("outer_holdout")
    ].copy()
    velocity = galaxy_prediction(
        sparc,
        selected_parameters["minimum_permittivity"],
        selected_parameters["a0_m_s2"],
        selected_parameters["gate_power"],
    )
    sparc_rmse, sparc_equal = equal_system_velocity_rmse(sparc, velocity)
    report = {
        "report_version": "P0586B-METRIC-BOUNDARY-RESPONSE-RESULTS-0.1.0",
        "status": "complete_metric_boundary_response",
        "protocol": {
            "path": str(protocol_path.relative_to(ROOT)),
            "sha256": sha256(protocol_path),
        },
        "coverage": {
            "clusters": len(contexts),
            "candidates": len(candidates),
            "system_scores": len(screen),
            "exact_fits": int(len(exact[exact.row_type.eq("system")])),
        },
        "selected": selected_parameters,
        "zero_fixed_geometry_RMS": {
            "selection": float(
                np.sqrt(np.mean([zero_screen[label] ** 2 for label in selection_labels]))
            ),
            "validation": float(
                np.sqrt(np.mean([zero_screen[label] ** 2 for label in validation_labels]))
            ),
            "all_four": float(np.sqrt(np.mean(np.square(list(zero_screen.values()))))),
        },
        "per_system_optima": json_safe(optima.to_dict(orient="records")),
        "all_four_fixed_geometry": {
            "candidates_improving_all_four": int(len(all_four_candidates)),
            "best_if_any": json_safe(
                all_four_candidates.sort_values("all_four_RMS_arcsec").head(1).to_dict(
                    orient="records"
                )
            ),
        },
        "exact_validation": {
            "zero_heldout_RMS_arcsec": float(validation_zero.heldout_exact_RMS_arcsec),
            "selected_heldout_RMS_arcsec": float(
                validation_selected.heldout_exact_RMS_arcsec
            ),
            "improvement_fraction": exact_improvement,
            "selected_all_roots": selected_all_roots,
        },
        "cross_domain": {
            "selected_spherical_SPARC_RMSE_km_s": sparc_rmse,
            "selected_spherical_SPARC_equal_system_RMSE_km_s": sparc_equal,
            "fixed_RAR_RMSE_km_s": p0586["cross_domain"]["comparators"][
                "fixed_RAR_outer_RMSE_km_s"
            ],
        },
        "gates": {
            "interior_selection_optimum": interior,
            "all_four_fixed_geometry_improve": bool(len(all_four_candidates) > 0),
            "validation_all_roots": selected_all_roots,
            "validation_improvement_pass": bool(
                selected_all_roots
                and exact_improvement
                >= float(
                    protocol["advance_gates"][
                        "validation_improvement_vs_zero_fraction_min"
                    ]
                )
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
        "# P0586B continuous-metric boundary response",
        "",
        f"Selected `{selected_parameters['candidate_id']}` on the original two selection clusters.",
        f"It improves **{selected_parameters['systems_improved_fixed_geometry']}/4** fixed-geometry cluster scores; **{len(all_four_candidates)}** candidates improve all four.",
        f"Validation exact RMS: **{validation_selected.heldout_exact_RMS_arcsec:.4f} arcsec** versus **{validation_zero.heldout_exact_RMS_arcsec:.4f}** at zero; change **{100*exact_improvement:.3f}%**.",
        f"The selection optimum is interior in both extended coordinates: **{interior}**.",
        f"Selected spherical SPARC RMSE: **{sparc_rmse:.3f} km/s**.",
    ]
    (output / protocol["outputs"]["summary"]).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    pivot = candidates[
        candidates.minimum_permittivity.eq(selected.minimum_permittivity)
    ].pivot(
        index="smoothing_r80_fraction",
        columns="anisotropy_tau",
        values="selection_RMS_arcsec",
    )
    image = axes[0].imshow(pivot, origin="lower", aspect="auto", cmap="viridis")
    axes[0].set_xticks(range(len(pivot.columns)), pivot.columns)
    axes[0].set_yticks(range(len(pivot.index)), pivot.index)
    axes[0].set_xlabel("anisotropy tau")
    axes[0].set_ylabel("smoothing / R80")
    axes[0].set_title("selection response")
    fig.colorbar(image, ax=axes[0], label="RMS (arcsec)")
    for label, block in screen.groupby("system_label"):
        curve = block[
            block.minimum_permittivity.eq(selected.minimum_permittivity)
            & block.smoothing_r80_fraction.eq(
                selected.smoothing_r80_fraction
            )
        ].sort_values("anisotropy_tau")
        axes[1].plot(
            curve.anisotropy_tau,
            100.0 * curve.improvement_fraction,
            marker="o",
            label=label,
        )
    axes[1].axhline(0.0, color="black", lw=1)
    axes[1].set_xlabel("anisotropy tau")
    axes[1].set_ylabel("fixed-geometry gain (%)")
    axes[1].legend(fontsize=7)
    system_exact = exact[exact.row_type.eq("system")].pivot(
        index="system_label", columns="role", values="heldout_exact_RMS_arcsec"
    )
    position = np.arange(len(system_exact))
    axes[2].bar(position - 0.18, system_exact.zero, 0.36, label="zero")
    axes[2].bar(position + 0.18, system_exact.selected, 0.36, label="metric")
    axes[2].set_xticks(position, system_exact.index, rotation=30, ha="right")
    axes[2].set_ylabel("heldout exact RMS (arcsec)")
    axes[2].legend()
    fig.suptitle("P0586B extended continuous-metric response")
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)
    print(json.dumps(report["selected"], indent=2), flush=True)
    print(json.dumps(report["all_four_fixed_geometry"], indent=2), flush=True)
    print(json.dumps(report["exact_validation"], indent=2), flush=True)
    print(json.dumps(report["gates"], indent=2), flush=True)


if __name__ == "__main__":
    main()
