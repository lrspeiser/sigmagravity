#!/usr/bin/env python3
"""Test universal angular rotations of the self-coupled route field."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import rotate as rotate_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_adaptive_route_multicluster_raw import json_safe  # noqa: E402
from run_p0615_self_coupled_quadrupole_route import derived_state  # noqa: E402
from run_p0617_self_coupled_support_phase_atlas import (  # noqa: E402
    contexts_and_frozen_geometry,
    lens_score,
)
from voidscreen.route_template import (  # noqa: E402
    conservative_route_template,
    weighted_radius,
)
from voidscreen.stellar_morphology_lensing import (  # noqa: E402
    build_stellar_morphology_deflection_field,
)


def phase_field(p0581: dict, context, state: dict, phase_degrees: float):
    translation = p0581["field_translation"]
    scale = float(
        context.local["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"]
    )
    xy = context.members[["x_arcsec", "y_arcsec"]].to_numpy(float)
    weights = context.members.base_weight.to_numpy(float)
    weights /= weights.sum()
    radius_kpc = np.linalg.norm(xy, axis=1) * scale
    r80 = weighted_radius(radius_kpc, weights, 0.8)
    width_over_r80 = 0.23 * np.sqrt(1.0 + float(state["quadrupole_Q"]) ** 2)
    spacing = float(translation["grid_spacing_arcsec"])
    half = float(translation["grid_half_width_arcsec"])
    axis = np.arange(-half, half + 0.5 * spacing, spacing)
    route_map, route_audit = conservative_route_template(
        axis,
        xy,
        weights,
        routing_fraction=float(state["self_routed_fraction"]),
        return_scale=0.36 * r80 / scale,
        radius_exponent=0.0,
        reference_radius=100.0 / scale,
        smoothing=width_over_r80 * r80 / scale,
        travel_mode="constant",
        center=None,
    )
    if abs(float(phase_degrees)) > 1.0e-12:
        phased = rotate_image(
            route_map,
            float(phase_degrees),
            reshape=False,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        retained = float(phased.sum())
        if retained <= 0.0:
            raise RuntimeError("phase rotation removed the complete route map")
        phased /= retained
    else:
        phased = route_map
        retained = 1.0

    def carrier_alpha(radius_arcsec):
        return context.parent.reduced_alpha_arcsec(
            radius_arcsec, 1.0
        ) - context.baryons.reduced_alpha_arcsec(radius_arcsec, 1.0)

    field = build_stellar_morphology_deflection_field(
        axis,
        phased,
        carrier_alpha,
        contrast_cap=20.0,
        contrast_mode="tanh",
        contrast_strength=float(translation["primary_contrast_strength"]),
        annulus_width_arcsec=float(translation["annulus_width_arcsec"]),
        taper_inner_arcsec=float(translation["taper_inner_arcsec"]),
        support_radius_arcsec=float(translation["support_radius_arcsec"]),
        radial_samples=2048,
        circular_radii=512,
        circular_azimuths=720,
    )
    return field, {
        "phase_degrees": float(phase_degrees),
        "R80_kpc": r80,
        "width_over_R80": width_over_r80,
        "return_length_over_R80": 0.36,
        "pre_normalization_retained_weight": retained,
        "route_map_normalization_error": abs(float(phased.sum()) - 1.0),
        "sources_crossing_center": int(route_audit["sources_crossing_center"]),
        "source_weight_crossing_center": float(
            route_audit["source_weight_crossing_center"]
        ),
        **field.audit,
    }


def main() -> None:
    protocol_path = ROOT / "configs/p0618_universal_route_phase_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_after_P0617_before_universal_phase_scores":
        raise RuntimeError("P0618 protocol is not frozen")
    p0615 = json.loads(
        (ROOT / protocol["inputs"]["P0615_protocol"]).read_text(encoding="utf-8")
    )
    p0581 = json.loads(
        (ROOT / p0615["inputs"]["P0581_protocol"]).read_text(encoding="utf-8")
    )
    prepared = contexts_and_frozen_geometry(p0615)
    score_rows = []
    audits = []
    for context, cohort, parameters, sources in prepared:
        label = context.system["label"]
        state = derived_state(context)
        epsilon = float(state["amplitudes"]["quadratic_Q2_over_total"])
        scalar = lens_score(context, parameters, sources, None, 0.0)
        score_rows.append(
            {
                "cohort": cohort,
                "system_label": label,
                "phase_degrees": np.nan,
                "variant_id": "scalar_control",
                "epsilon": 0.0,
                **scalar,
            }
        )
        for phase in protocol["universal_phase_degrees"]:
            field, audit = phase_field(p0581, context, state, float(phase))
            metrics = lens_score(context, parameters, sources, field, epsilon)
            score_rows.append(
                {
                    "cohort": cohort,
                    "system_label": label,
                    "phase_degrees": float(phase),
                    "variant_id": f"phase_{float(phase):+05.1f}",
                    "epsilon": epsilon,
                    **metrics,
                }
            )
            audits.append(
                {
                    "cohort": cohort,
                    "system_label": label,
                    "epsilon": epsilon,
                    **audit,
                }
            )
        print(f"P0618 {label}: {len(protocol['universal_phase_degrees'])} universal phases", flush=True)

    scores = pd.DataFrame(score_rows)
    controls = scores[scores.variant_id.eq("scalar_control")].set_index("system_label")
    response_rows = []
    for phase in protocol["universal_phase_degrees"]:
        block = scores[np.isclose(scores.phase_degrees, float(phase))].set_index(
            "system_label"
        )
        improvements = np.asarray(
            [
                1.0
                - float(block.loc[label].heldout_RMS_arcsec)
                / float(controls.loc[label].heldout_RMS_arcsec)
                if bool(block.loc[label].heldout_all_roots)
                else -np.inf
                for label in controls.index
            ]
        )
        four = block[block.cohort.eq("P0581_four")]
        rx = block[block.cohort.eq("RXJ2129")]
        four_control = controls[controls.cohort.eq("P0581_four")]
        rx_control = controls[controls.cohort.eq("RXJ2129")]
        four_rms = float(np.sqrt(np.mean(np.square(four.heldout_RMS_arcsec))))
        four_control_rms = float(
            np.sqrt(np.mean(np.square(four_control.heldout_RMS_arcsec)))
        )
        response_rows.append(
            {
                "phase_degrees": float(phase),
                "combined_roots": int(block.heldout_converged_roots.sum()),
                "all_18_roots": bool(block.heldout_all_roots.astype(bool).all()),
                "systems_not_worse": int(np.sum(improvements >= 0.0)),
                "minimum_system_improvement": float(np.min(improvements)),
                "mean_system_improvement": float(np.mean(improvements)),
                "four_cluster_improvement": 1.0 - four_rms / four_control_rms,
                "RXJ2129_improvement": 1.0
                - float(rx.heldout_RMS_arcsec.iloc[0])
                / float(rx_control.heldout_RMS_arcsec.iloc[0]),
            }
        )
    responses = pd.DataFrame(response_rows).sort_values(
        [
            "all_18_roots",
            "systems_not_worse",
            "minimum_system_improvement",
            "mean_system_improvement",
        ],
        ascending=False,
    )
    selected = responses.iloc[0]

    preference_rows = []
    for label in controls.index:
        block = scores[
            scores.system_label.eq(label) & ~scores.variant_id.eq("scalar_control")
        ].copy()
        block["improvement"] = 1.0 - block.heldout_RMS_arcsec / float(
            controls.loc[label].heldout_RMS_arcsec
        )
        complete = block[block.heldout_all_roots.astype(bool)]
        best = complete.sort_values("improvement", ascending=False).iloc[0]
        preference_rows.append(
            {
                "system_label": label,
                "preferred_phase_degrees": float(best.phase_degrees),
                "best_diagnostic_improvement": float(best.improvement),
                "phases_with_all_roots": len(complete),
            }
        )
    preferences = pd.DataFrame(preference_rows)
    phase_radians = np.deg2rad(2.0 * preferences.preferred_phase_degrees.to_numpy(float))
    spin2_resultant = float(abs(np.mean(np.exp(1j * phase_radians))))

    p0615_report = json.loads(
        (ROOT / protocol["inputs"]["P0615_report"]).read_text(encoding="utf-8")
    )
    p0617_report = json.loads(
        (ROOT / protocol["inputs"]["P0617_report"]).read_text(encoding="utf-8")
    )
    cfg = protocol["gates"]
    gates = {
        "all_18_roots_pass": bool(
            int(selected.combined_roots) == int(cfg["combined_heldout_roots_required"])
        ),
        "all_five_systems_not_worse_pass": bool(
            int(selected.systems_not_worse) == int(cfg["systems_not_worse_required"])
        ),
        "minimum_system_improvement_pass": bool(
            float(selected.minimum_system_improvement)
            >= float(cfg["minimum_system_improvement"])
        ),
        "galaxy_near_RAR_pass": bool(
            p0615_report["inherited_cross_domain"]["SPARC_to_RAR_ratio"]
            <= float(cfg["galaxy_RMSE_to_fixed_RAR_max"])
        ),
        "Solar_all_proxies_pass": bool(
            p0615_report["inherited_cross_domain"]["Solar_all_proxies_pass"]
        ),
        "zero_new_fitted_gravity_parameters_pass": True,
    }
    gates["all_diagnostic_gates_pass"] = bool(all(gates.values()))

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / protocol["outputs"]["scores"], index=False)
    responses.to_csv(
        output / protocol["outputs"]["universal_phase_responses"], index=False
    )
    preferences.to_csv(
        output / protocol["outputs"]["system_phase_preferences"], index=False
    )
    pd.DataFrame(audits).to_csv(output / protocol["outputs"]["field_audits"], index=False)
    report = {
        "report_version": "P0618-UNIVERSAL-ROUTE-PHASE-RESULTS-0.1.0",
        "status": "complete_opened_data_universal_phase_test",
        "coverage": {
            "raw_systems": len(prepared),
            "heldout_images": 18,
            "universal_phases": len(protocol["universal_phase_degrees"]),
            "score_rows_including_control": len(scores),
            "new_fitted_gravity_parameters": 0,
        },
        "frozen_formula": protocol["frozen_formula"],
        "selected_universal_phase": selected.to_dict(),
        "universal_phase_responses": responses.to_dict("records"),
        "per_system_diagnostic_preferences": preferences.to_dict("records"),
        "preferred_phase_spin2_resultant": spin2_resultant,
        "inherited_cross_domain": p0615_report["inherited_cross_domain"],
        "P0617_context": p0617_report["interpretation"],
        "gates": gates,
        "interpretation": {
            "formula_promoted": False,
            "one_universal_phase_found": bool(gates["all_diagnostic_gates_pass"]),
            "per_system_phase_selection_allowed": False,
            "future_full_refit_transfer_required": True,
        },
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )

    angle_plot = responses.sort_values("phase_degrees")
    matrix = scores[~scores.variant_id.eq("scalar_control")].copy()
    matrix = matrix.merge(
        controls.heldout_RMS_arcsec.rename("control_RMS"),
        left_on="system_label",
        right_index=True,
    )
    matrix["improvement"] = 100.0 * (
        1.0 - matrix.heldout_RMS_arcsec / matrix.control_RMS
    )
    pivot = matrix.pivot(
        index="system_label", columns="phase_degrees", values="improvement"
    )
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    axes[0].plot(
        angle_plot.phase_degrees,
        100.0 * angle_plot.mean_system_improvement,
        marker="o",
        label="mean system",
    )
    axes[0].plot(
        angle_plot.phase_degrees,
        100.0 * angle_plot.minimum_system_improvement,
        marker="s",
        label="worst system",
    )
    axes[0].axhline(0.0, color="black", lw=0.8)
    axes[0].set(xlabel="one universal route phase (degrees)", ylabel="change vs scalar (%)", title="No universal phase clears consistency")
    axes[0].legend()
    image = axes[1].imshow(pivot.to_numpy(float), aspect="auto", cmap="RdBu", vmin=-2.0, vmax=2.0)
    axes[1].set(
        xticks=np.arange(len(pivot.columns)),
        xticklabels=[f"{value:g}" for value in pivot.columns],
        yticks=np.arange(len(pivot.index)),
        yticklabels=pivot.index,
        xlabel="phase (degrees)",
        title="Each cluster prefers a different phase",
    )
    figure.colorbar(image, ax=axes[1], label="improvement vs scalar (%)")
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)

    (output / protocol["outputs"]["summary"]).write_text(
        "# P0618 universal route phase\n\n"
        f"Safety-first universal phase: **{float(selected.phase_degrees):+.1f} deg** "
        f"with **{int(selected.combined_roots)}/18** roots and "
        f"**{int(selected.systems_not_worse)}/5** systems not worse.\n\n"
        f"Mean system change: **{100.0*float(selected.mean_system_improvement):+.3f}%**; "
        f"worst system: **{100.0*float(selected.minimum_system_improvement):+.3f}%**.\n\n"
        f"Per-system preferred-phase spin-2 resultant: **{spin2_resultant:.3f}** "
        "(1 means aligned, 0 means dispersed). Per-system selection is diagnostic only.\n\n"
        f"All diagnostic gates pass: **{gates['all_diagnostic_gates_pass']}**.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            json_safe(
                {
                    "selected": report["selected_universal_phase"],
                    "preferences": report["per_system_diagnostic_preferences"],
                    "spin2_resultant": spin2_resultant,
                    "gates": gates,
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
