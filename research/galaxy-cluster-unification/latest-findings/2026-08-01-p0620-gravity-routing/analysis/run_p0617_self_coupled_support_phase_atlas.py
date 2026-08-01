#!/usr/bin/env python3
"""Scan parameter-free support and center-crossing laws at frozen geometry."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_adaptive_route_multicluster_raw import (  # noqa: E402
    MODEL,
    build_contexts,
    json_safe,
)
from run_clash_stellar_morphology_response import MorphologyLens  # noqa: E402
from run_p0581_locked_endpoint_exact_root import endpoint_field  # noqa: E402
from run_p0582_smooth_endpoint_saturation import source_positions  # noqa: E402
from run_p0615_self_coupled_quadrupole_route import (  # noqa: E402
    derived_state,
    rxj_context,
)
from run_rxj2129_raw_theory_lensing import FIXED_LABELS, score  # noqa: E402


def factor(name: str, q: float) -> float:
    values = {
        "fixed": 1.0,
        "minus_linear": 1.0 - q,
        "minus_rational": 1.0 / (1.0 + q),
        "minus_even": np.sqrt(max(1.0 - q**2, np.finfo(float).tiny)),
        "plus_even": np.sqrt(1.0 + q**2),
        "plus_linear": 1.0 + q,
    }
    return float(values[name])


def support_field(p0581: dict, context, state: dict, variant: dict):
    q = float(state["quadrupole_Q"])
    width = 0.23 * factor(str(variant["width_law"]), q)
    length = 0.36 * factor(str(variant["return_law"]), q)
    spec = {
        "route_fraction_multiplier": float(state["self_routed_fraction"]),
        "return_length_over_R80": length,
        "width_over_R80": width,
        "gate_mode": "none",
        "contrast_mode": "tanh",
        "contrast_cap": 20.0,
        "travel_mode": str(variant["travel_mode"]),
        "variant": str(variant["variant_id"]),
    }
    field, audit = endpoint_field(p0581, context, spec)
    return field, {**audit, "width_factor": width / 0.23, "return_factor": length / 0.36}


def lens_score(context, parameters, sources, field, epsilon: float) -> dict:
    lens = MorphologyLens(
        context.local,
        {MODEL: context.parent},
        parent=MODEL,
        morphology=field,
        fraction=float(epsilon),
    )
    predictions = lens.exact_predictions(
        MODEL,
        parameters,
        sources,
        context.heldout,
        stage="heldout",
    )
    metrics = score(predictions, lens.sigma)
    return {
        "heldout_images": len(context.heldout),
        "heldout_converged_roots": int(metrics["converged_roots"]),
        "heldout_all_roots": bool(metrics["all_roots_converged"]),
        "heldout_RMS_arcsec": float(metrics["exact_radial_RMS_arcsec"]),
    }


def contexts_and_frozen_geometry(p0615: dict):
    four_protocol = json.loads(
        (ROOT / p0615["inputs"]["four_cluster_protocol"]).read_text(encoding="utf-8")
    )
    contexts, _, _ = build_contexts(four_protocol)
    geometry = pd.read_csv(ROOT / p0615["inputs"]["P0581_geometry"])
    prior = pd.read_csv(ROOT / p0615["inputs"]["P0581_predictions"])
    prepared = []
    for context in contexts:
        label = context.system["label"]
        row = geometry[
            geometry.system_label.eq(label) & geometry.variant.eq("K0338_primary")
        ].iloc[0]
        prepared.append(
            (
                context,
                "P0581_four",
                np.asarray([float(row[name]) for name in FIXED_LABELS]),
                source_positions(prior, label),
            )
        )
    rx_context, rx_parameters, rx_sources = rxj_context(p0615)
    prepared.append((rx_context, "RXJ2129", rx_parameters, rx_sources))
    return prepared


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cohort, variant), block in scores.groupby(["cohort", "variant_id"], sort=False):
        complete = block[block.heldout_all_roots.astype(bool)]
        values = complete.heldout_RMS_arcsec.to_numpy(float)
        rows.append(
            {
                "cohort": cohort,
                "variant_id": variant,
                "systems": len(block),
                "complete_systems": len(complete),
                "heldout_converged_roots": int(block.heldout_converged_roots.sum()),
                "all_roots": bool(len(complete) == len(block)),
                "equal_complete_system_RMS_arcsec": (
                    float(np.sqrt(np.mean(np.square(values)))) if len(values) else np.inf
                ),
            }
        )
    result = pd.DataFrame(rows)
    pivot = result.pivot(
        index="variant_id", columns="cohort", values="equal_complete_system_RMS_arcsec"
    )
    for cohort in result.cohort.unique():
        control = float(pivot.loc["scalar_control", cohort])
        mask = result.cohort.eq(cohort)
        result.loc[mask, "improvement_vs_scalar"] = 1.0 - (
            result.loc[mask, "equal_complete_system_RMS_arcsec"] / control
        )
    return result


def fixed_route_residual_alignment(p0615: dict, scores: pd.DataFrame) -> pd.DataFrame:
    """Measure whether the frozen route pushes predictions with or against residuals."""
    predictions = pd.read_csv(
        ROOT / p0615["outputs"]["directory"] / p0615["outputs"]["predictions"]
    )
    controls = scores[scores.variant_id.eq("scalar_control")].set_index("system_label")
    routes = scores[scores.variant_id.eq("fixed_support")].set_index("system_label")
    rows = []
    for label in controls.index:
        block = predictions[predictions.system_label.eq(label)]
        scalar = block[block.variant.eq("scalar_control")].set_index("image_id")
        route = block[block.variant.eq("quadratic_Q2_over_total")].set_index(
            "image_id"
        ).reindex(scalar.index)
        valid = scalar.root_converged.astype(bool) & route.root_converged.astype(bool)
        scalar = scalar[valid]
        route = route[valid]
        residual = scalar[["delta_x_arcsec", "delta_y_arcsec"]].to_numpy(float)
        shift = route[["predicted_x_arcsec", "predicted_y_arcsec"]].to_numpy(float) - scalar[
            ["predicted_x_arcsec", "predicted_y_arcsec"]
        ].to_numpy(float)
        dots = np.sum(residual * shift, axis=1)
        norms = np.linalg.norm(residual, axis=1) * np.linalg.norm(shift, axis=1)
        improvement = 1.0 - float(routes.loc[label].heldout_RMS_arcsec) / float(
            controls.loc[label].heldout_RMS_arcsec
        )
        rows.append(
            {
                "system_label": label,
                "images": len(dots),
                "mean_residual_dot_route_shift_arcsec2": float(np.mean(dots)),
                "weighted_alignment_cosine": float(
                    np.sum(dots) / np.maximum(np.sum(norms), 1.0e-99)
                ),
                "images_with_first_order_improving_shift": int(np.sum(dots < 0.0)),
                "route_shift_RMS_arcsec": float(
                    np.sqrt(np.mean(np.sum(np.square(shift), axis=1)))
                ),
                "fixed_route_improvement": improvement,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    protocol_path = ROOT / "configs/p0617_self_coupled_support_phase_atlas_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_after_P0616_before_support_phase_scores":
        raise RuntimeError("P0617 protocol is not frozen")
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
                "variant_id": "scalar_control",
                "family": "control",
                "quadrupole_Q": float(state["quadrupole_Q"]),
                "epsilon": 0.0,
                **scalar,
            }
        )
        for variant in protocol["variants"]:
            field, audit = support_field(p0581, context, state, variant)
            metrics = lens_score(context, parameters, sources, field, epsilon)
            score_rows.append(
                {
                    "cohort": cohort,
                    "system_label": label,
                    "variant_id": variant["variant_id"],
                    "family": variant["family"],
                    "quadrupole_Q": float(state["quadrupole_Q"]),
                    "epsilon": epsilon,
                    "width_over_R80": float(audit["width_over_R80"]),
                    "return_length_over_R80": float(audit["return_length_over_R80"]),
                    "travel_mode": variant["travel_mode"],
                    **metrics,
                }
            )
            audits.append(
                {
                    "cohort": cohort,
                    "system_label": label,
                    "variant_id": variant["variant_id"],
                    "family": variant["family"],
                    "epsilon": epsilon,
                    **audit,
                }
            )
        print(f"P0617 {label}: {len(protocol['variants'])} support/phase laws", flush=True)

    scores = pd.DataFrame(score_rows)
    summaries = summarize(scores)
    controls = scores[scores.variant_id.eq("scalar_control")].set_index("system_label")
    summary_index = summaries.set_index(["cohort", "variant_id"])
    response_rows = []
    for variant in protocol["variants"]:
        variant_id = variant["variant_id"]
        block = scores[scores.variant_id.eq(variant_id)].set_index("system_label")
        system_improvements = []
        for label in controls.index:
            control = controls.loc[label]
            candidate = block.loc[label]
            if bool(candidate.heldout_all_roots) and np.isfinite(candidate.heldout_RMS_arcsec):
                system_improvements.append(
                    1.0 - float(candidate.heldout_RMS_arcsec) / float(control.heldout_RMS_arcsec)
                )
            else:
                system_improvements.append(-np.inf)
        four = summary_index.loc[("P0581_four", variant_id)]
        rx = summary_index.loc[("RXJ2129", variant_id)]
        complete = block[block.heldout_all_roots.astype(bool)]
        candidate_values = complete.heldout_RMS_arcsec.to_numpy(float)
        response_rows.append(
            {
                "variant_id": variant_id,
                "family": variant["family"],
                "width_law": variant["width_law"],
                "return_law": variant["return_law"],
                "travel_mode": variant["travel_mode"],
                "combined_roots": int(four.heldout_converged_roots + rx.heldout_converged_roots),
                "all_18_roots": bool(four.all_roots and rx.all_roots),
                "systems_not_worse": int(np.sum(np.asarray(system_improvements) >= 0.0)),
                "minimum_system_improvement": float(np.min(system_improvements)),
                "mean_system_improvement": float(np.mean(system_improvements)),
                "four_cluster_improvement": float(four.improvement_vs_scalar),
                "RXJ2129_improvement": float(rx.improvement_vs_scalar),
                "equal_complete_five_system_RMS_arcsec": (
                    float(np.sqrt(np.mean(np.square(candidate_values))))
                    if len(complete) == len(controls)
                    else np.inf
                ),
            }
        )
    responses = pd.DataFrame(response_rows)
    responses = responses.sort_values(
        [
            "all_18_roots",
            "systems_not_worse",
            "minimum_system_improvement",
            "mean_system_improvement",
        ],
        ascending=False,
    )
    selected = responses.iloc[0]
    baseline = responses[responses.variant_id.eq("fixed_support")].iloc[0]
    family_rows = []
    for family, block in responses.groupby("family", sort=False):
        finite = block.replace([np.inf, -np.inf], np.nan)
        family_rows.append(
            {
                "family": family,
                "variants": len(block),
                "minimum_combined_roots": int(block.combined_roots.min()),
                "maximum_combined_roots": int(block.combined_roots.max()),
                "mean_system_improvement_span": float(
                    finite.mean_system_improvement.max() - finite.mean_system_improvement.min()
                ),
                "four_cluster_improvement_span": float(
                    finite.four_cluster_improvement.max() - finite.four_cluster_improvement.min()
                ),
                "RXJ2129_improvement_span": float(
                    finite.RXJ2129_improvement.max() - finite.RXJ2129_improvement.min()
                ),
                "best_variant": str(block.iloc[0].variant_id),
            }
        )
    family_impacts = pd.DataFrame(family_rows).sort_values(
        "mean_system_improvement_span", ascending=False
    )
    alignment = fixed_route_residual_alignment(p0615, scores)
    alignment_sign_matches = bool(
        np.all(
            (alignment.weighted_alignment_cosine < 0.0)
            == (alignment.fixed_route_improvement > 0.0)
        )
    )

    p0615_report = json.loads(
        (ROOT / protocol["inputs"]["P0615_report"]).read_text(encoding="utf-8")
    )
    p0616_report = json.loads(
        (ROOT / protocol["inputs"]["P0616_report"]).read_text(encoding="utf-8")
    )
    cfg = protocol["gates"]
    gates = {
        "all_18_roots_pass": bool(
            int(selected.combined_roots) == int(cfg["combined_heldout_roots_required"])
        ),
        "all_five_systems_not_worse_pass": bool(
            int(selected.systems_not_worse) == int(cfg["systems_not_worse_required"])
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
    summaries.to_csv(output / protocol["outputs"]["summary_scores"], index=False)
    responses.to_csv(output / protocol["outputs"]["variant_responses"], index=False)
    family_impacts.to_csv(output / protocol["outputs"]["family_impacts"], index=False)
    alignment.to_csv(output / protocol["outputs"]["residual_alignment"], index=False)
    pd.DataFrame(audits).to_csv(output / protocol["outputs"]["field_audits"], index=False)
    report = {
        "report_version": "P0617-SELF-COUPLED-SUPPORT-PHASE-ATLAS-RESULTS-0.1.0",
        "status": "complete_opened_data_support_phase_atlas",
        "coverage": {
            "raw_systems": len(prepared),
            "heldout_images": 18,
            "support_phase_variants": len(protocol["variants"]),
            "score_rows_including_control": len(scores),
            "new_fitted_gravity_parameters": 0,
        },
        "frozen_formula": protocol["frozen_formula"],
        "selected_diagnostic": selected.to_dict(),
        "fixed_support_response": baseline.to_dict(),
        "family_impacts": family_impacts.to_dict("records"),
        "variant_responses": responses.to_dict("records"),
        "fixed_route_residual_alignment": alignment.to_dict("records"),
        "inherited_cross_domain": p0615_report["inherited_cross_domain"],
        "prior_transfer_warning": p0616_report["interpretation"],
        "gates": gates,
        "interpretation": {
            "formula_promoted": False,
            "opened_data_diagnostic_only": True,
            "future_full_refit_transfer_required": True,
            "residual_alignment_sign_matches_all_five_responses": alignment_sign_matches,
            "phase_lesson": "Support changes scale the effect, but whether a route helps is set by the angular alignment of its predicted shift with the pre-existing lens residual.",
        },
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )

    plot = responses.sort_values("mean_system_improvement")
    colors = np.where(plot.all_18_roots, "#1b9e77", "#d95f02")
    figure, axes = plt.subplots(1, 2, figsize=(14, 7.2), constrained_layout=True)
    axes[0].barh(plot.variant_id, 100.0 * plot.mean_system_improvement, color=colors)
    axes[0].axvline(0.0, color="black", lw=0.8)
    axes[0].set(xlabel="mean five-system change vs scalar (%)", title="Support/phase response")
    impact_plot = family_impacts.sort_values("mean_system_improvement_span")
    axes[1].barh(
        impact_plot.family,
        100.0 * impact_plot.mean_system_improvement_span,
        color="#7570b3",
    )
    axes[1].set(xlabel="within-family response span (percentage points)", title="Which coordinate matters most?")
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)

    (output / protocol["outputs"]["summary"]).write_text(
        "# P0617 self-coupled support/phase atlas\n\n"
        f"Safety-first diagnostic: **{selected.variant_id}** with "
        f"**{int(selected.combined_roots)}/18** roots and "
        f"**{int(selected.systems_not_worse)}/5** systems not worse.\n\n"
        f"Mean system change: **{100.0*float(selected.mean_system_improvement):+.3f}%**; "
        f"four-cluster change: **{100.0*float(selected.four_cluster_improvement):+.3f}%**; "
        f"RXJ2129 change: **{100.0*float(selected.RXJ2129_improvement):+.3f}%**.\n\n"
        f"All diagnostic gates pass: **{gates['all_diagnostic_gates_pass']}**. "
        "This opened-data atlas cannot promote a formula.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            json_safe(
                {
                    "selected": report["selected_diagnostic"],
                    "families": report["family_impacts"],
                    "gates": gates,
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
