#!/usr/bin/env python3
"""Test parameter-free route amplitudes derived from scalar excess and shape."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

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
    decorate_predictions,
    json_safe,
)
from run_adaptive_route_raw_rxj2129 import baryon_field, load_sources  # noqa: E402
from run_arc_invariant_absolute_lensing import raw_field  # noqa: E402
from run_arc_invariant_pareto_refinement import build_specs  # noqa: E402
from run_clash_stellar_morphology_response import MorphologyLens  # noqa: E402
from run_p0581_locked_endpoint_exact_root import endpoint_field  # noqa: E402
from run_p0582_smooth_endpoint_saturation import source_positions  # noqa: E402
from run_rxj2129_member_geometry import split_images  # noqa: E402
from run_rxj2129_raw_theory_lensing import (  # noqa: E402
    FIXED_LABELS,
    load_baryonic_anchors,
    load_images,
    score,
)
from voidscreen.baryonic_metric import weighted_morphology  # noqa: E402
from voidscreen.route_template import weighted_radius  # noqa: E402


LAWS = [
    "scalar_control",
    "linear_Q_over_total",
    "quadratic_Q2_over_total",
    "quadratic_Q2_routed",
    "quadratic_Q2_strong_screen",
    "quartic_Q4_over_total",
]


def derived_state(context) -> dict:
    xy = context.members[["x_arcsec", "y_arcsec"]].to_numpy(float)
    weights = context.members.base_weight.to_numpy(float)
    weights /= weights.sum()
    scale = float(
        context.local["cosmology_and_coordinates"][
            "angular_scale_kpc_per_arcsec"
        ]
    )
    radius_kpc = np.linalg.norm(xy, axis=1) * scale
    r80_kpc = weighted_radius(radius_kpc, weights, 0.8)
    r80_arcsec = r80_kpc / scale
    parent = float(context.parent.reduced_alpha_arcsec(r80_arcsec, 1.0))
    baryons = float(context.baryons.reduced_alpha_arcsec(r80_arcsec, 1.0))
    delta = max(parent / baryons - 1.0, 0.0)
    if delta <= 0.0:
        raise RuntimeError("P0554 has no positive excess at R80")
    morphology = weighted_morphology(xy[:, 0], xy[:, 1], weights)
    q = float(morphology["quadrupole_asymmetry"])
    total = 1.0 + delta
    amplitudes = {
        "scalar_control": 0.0,
        "linear_Q_over_total": q / total,
        "quadratic_Q2_over_total": q**2 / total,
        "quadratic_Q2_routed": q**2 * delta / total**2,
        "quadratic_Q2_strong_screen": q**2 / total**2,
        "quartic_Q4_over_total": q**4 / total,
    }
    return {
        "R80_kpc": r80_kpc,
        "parent_alpha_R80_arcsec": parent,
        "baryonic_alpha_R80_arcsec": baryons,
        "Delta80": delta,
        "self_routed_fraction": delta / total,
        "quadrupole_Q": q,
        **{f"epsilon_{key}": value for key, value in amplitudes.items()},
        "amplitudes": amplitudes,
    }


def build_self_field(p0581: dict, context, state: dict):
    spec = {
        "route_fraction_multiplier": float(state["self_routed_fraction"]),
        "return_length_over_R80": 0.36,
        "width_over_R80": 0.23,
        "gate_mode": "none",
        "contrast_mode": "tanh",
        "contrast_cap": 20.0,
        "variant": "self_coupled_endpoint_field",
    }
    return endpoint_field(p0581, context, spec)


def score_fixed_context(
    context,
    cohort: str,
    parameters: np.ndarray,
    sources: dict[int, np.ndarray],
    field,
    state: dict,
) -> tuple[list[dict], list[pd.DataFrame]]:
    rows = []
    predictions = []
    for law in LAWS:
        epsilon = float(state["amplitudes"][law])
        lens = MorphologyLens(
            context.local,
            {MODEL: context.parent},
            parent=MODEL,
            morphology=field,
            fraction=epsilon,
        )
        predicted = lens.exact_predictions(
            MODEL,
            parameters,
            sources,
            context.heldout,
            stage="heldout",
        )
        metrics = score(predicted, lens.sigma)
        rows.append(
            {
                "cohort": cohort,
                "system_label": context.system["label"],
                "law": law,
                "epsilon": epsilon,
                "heldout_images": len(context.heldout),
                "heldout_converged_roots": metrics["converged_roots"],
                "heldout_all_roots": metrics["all_roots_converged"],
                "heldout_RMS_arcsec": metrics["exact_radial_RMS_arcsec"],
            }
        )
        decorated = decorate_predictions(predicted, context, law)
        decorated["cohort"] = cohort
        decorated["epsilon"] = epsilon
        predictions.append(decorated)
    return rows, predictions


def rxj_context(protocol: dict):
    p0583 = json.loads(
        (ROOT / protocol["inputs"]["P0583_protocol"]).read_text(encoding="utf-8")
    )
    rxj_protocol = json.loads(
        (ROOT / p0583["inputs"]["rxj_route_protocol"]).read_text(encoding="utf-8")
    )
    raw_protocol = json.loads(
        (ROOT / rxj_protocol["raw_inputs"]["raw_protocol"]).read_text(
            encoding="utf-8"
        )
    )
    images = load_images(raw_protocol)
    _, heldout = split_images(images, raw_protocol)
    members = load_sources(rxj_protocol, raw_protocol)
    anchors = load_baryonic_anchors(raw_protocol)
    parent_protocol = json.loads(
        (ROOT / rxj_protocol["raw_inputs"]["parent_protocol"]).read_text(
            encoding="utf-8"
        )
    )
    parent_scores = pd.read_csv(ROOT / rxj_protocol["raw_inputs"]["parent_scores"])
    parent_id = str(rxj_protocol["raw_inputs"]["parent_candidate"])
    parent_row = parent_scores[parent_scores.candidate_id.eq(parent_id)].iloc[0]
    specs = {item["candidate_id"]: item for item in build_specs(parent_protocol)}
    parent, _ = raw_field(
        specs[parent_id],
        float(parent_row.universal_q),
        anchors,
        raw_protocol,
        1.2e-10,
    )
    baryons = baryon_field(anchors, raw_protocol)
    system = {"label": "RXJ2129", "system": "RX J2129.7+0005"}
    context = SimpleNamespace(
        system=system,
        local=raw_protocol,
        members=members,
        parent=parent,
        baryons=baryons,
        heldout=heldout,
    )
    parameter_frame = pd.read_csv(ROOT / protocol["inputs"]["P0583_parameters"])
    scalar_parameters = parameter_frame[
        parameter_frame.variant.eq("scalar_baseline")
    ].set_index("parameter")
    parameters = np.asarray(
        [float(scalar_parameters.loc[name, "value"]) for name in FIXED_LABELS]
    )
    prior = pd.read_csv(ROOT / protocol["inputs"]["P0583_predictions"])
    scalar = prior[prior.variant.eq("scalar_baseline")]
    sources = {
        int(family): block[["source_x_arcsec", "source_y_arcsec"]]
        .iloc[0]
        .to_numpy(float)
        for family, block in scalar.groupby("source_family")
    }
    return context, parameters, sources


def aggregate_scores(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cohort, law), block in scores.groupby(["cohort", "law"], sort=False):
        complete = block[block.heldout_all_roots.astype(bool)]
        values = complete.heldout_RMS_arcsec.to_numpy(float)
        rows.append(
            {
                "cohort": cohort,
                "law": law,
                "systems": len(block),
                "complete_systems": len(complete),
                "heldout_images": int(block.heldout_images.sum()),
                "heldout_converged_roots": int(block.heldout_converged_roots.sum()),
                "all_roots": bool(len(complete) == len(block)),
                "equal_complete_system_RMS_arcsec": (
                    float(np.sqrt(np.mean(np.square(values))))
                    if len(values)
                    else float("inf")
                ),
                "median_epsilon": float(block.epsilon.median()),
                "minimum_epsilon": float(block.epsilon.min()),
                "maximum_epsilon": float(block.epsilon.max()),
            }
        )
    result = pd.DataFrame(rows)
    pivot = result.pivot(index="law", columns="cohort", values="equal_complete_system_RMS_arcsec")
    for cohort in result.cohort.unique():
        control = float(pivot.loc["scalar_control", cohort])
        mask = result.cohort.eq(cohort)
        result.loc[mask, "improvement_vs_frozen_scalar"] = 1.0 - (
            result.loc[mask, "equal_complete_system_RMS_arcsec"] / control
        )
    return result


def main() -> None:
    protocol_path = ROOT / "configs/p0615_self_coupled_quadrupole_route_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_after_P0614_before_self_coupled_scores":
        raise RuntimeError("P0615 protocol is not frozen")
    p0581 = json.loads(
        (ROOT / protocol["inputs"]["P0581_protocol"]).read_text(encoding="utf-8")
    )
    four_protocol = json.loads(
        (ROOT / protocol["inputs"]["four_cluster_protocol"]).read_text(
            encoding="utf-8"
        )
    )
    contexts, _, _ = build_contexts(four_protocol)
    geometry = pd.read_csv(ROOT / protocol["inputs"]["P0581_geometry"])
    prior = pd.read_csv(ROOT / protocol["inputs"]["P0581_predictions"])
    score_rows = []
    prediction_frames = []
    state_rows = []
    audit_rows = []
    for context in contexts:
        label = context.system["label"]
        row = geometry[
            geometry.system_label.eq(label) & geometry.variant.eq("K0338_primary")
        ].iloc[0]
        parameters = np.asarray([float(row[name]) for name in FIXED_LABELS])
        sources = source_positions(prior, label)
        state = derived_state(context)
        field, audit = build_self_field(p0581, context, state)
        rows, predictions = score_fixed_context(
            context, "P0581_four", parameters, sources, field, state
        )
        score_rows.extend(rows)
        prediction_frames.extend(predictions)
        state_rows.append(
            {
                "cohort": "P0581_four",
                "system_label": label,
                **{key: value for key, value in state.items() if key != "amplitudes"},
            }
        )
        audit_rows.append({"cohort": "P0581_four", "system_label": label, **audit})
        print(f"P0615 {label}: {len(LAWS)} self-coupled laws", flush=True)

    rx_context, rx_parameters, rx_sources = rxj_context(protocol)
    rx_state = derived_state(rx_context)
    rx_field, rx_audit = build_self_field(p0581, rx_context, rx_state)
    rows, predictions = score_fixed_context(
        rx_context,
        "RXJ2129",
        rx_parameters,
        rx_sources,
        rx_field,
        rx_state,
    )
    score_rows.extend(rows)
    prediction_frames.extend(predictions)
    state_rows.append(
        {
            "cohort": "RXJ2129",
            "system_label": "RXJ2129",
            **{key: value for key, value in rx_state.items() if key != "amplitudes"},
        }
    )
    audit_rows.append({"cohort": "RXJ2129", "system_label": "RXJ2129", **rx_audit})
    print(f"P0615 RXJ2129: {len(LAWS)} self-coupled laws", flush=True)

    scores = pd.DataFrame(score_rows)
    summaries = aggregate_scores(scores)
    four = summaries[summaries.cohort.eq("P0581_four")].set_index("law")
    rx = summaries[summaries.cohort.eq("RXJ2129")].set_index("law")
    combined_rows = []
    for law in LAWS:
        combined_rows.append(
            {
                "law": law,
                "combined_roots": int(four.loc[law, "heldout_converged_roots"])
                + int(rx.loc[law, "heldout_converged_roots"]),
                "all_18_roots": bool(four.loc[law, "all_roots"] and rx.loc[law, "all_roots"]),
                "four_cluster_improvement": float(
                    four.loc[law, "improvement_vs_frozen_scalar"]
                ),
                "RXJ2129_improvement": float(rx.loc[law, "improvement_vs_frozen_scalar"]),
            }
        )
    combined = pd.DataFrame(combined_rows).sort_values(
        ["all_18_roots", "combined_roots", "four_cluster_improvement", "RXJ2129_improvement"],
        ascending=False,
    )
    diagnostic = combined.iloc[0]

    p0554_report = json.loads(
        (ROOT / protocol["inputs"]["P0554_report"]).read_text(encoding="utf-8")
    )
    p0554 = next(
        row
        for row in p0554_report["shortlist_selected_without_raw_scores"]
        if row["candidate_id"] == "P0554"
    )
    sparc = json.loads(
        (ROOT / protocol["inputs"]["SPARC_report"]).read_text(encoding="utf-8")
    )
    rar = float(sparc["scores"]["fixed_RAR:invariant"]["outer_holdout"]["RMSE_km_s"])
    gates_cfg = protocol["gates"]
    gates = {
        "four_cluster_all_roots_pass": bool(
            int(four.loc[diagnostic.law, "heldout_converged_roots"])
            == int(gates_cfg["four_cluster_heldout_roots_required"])
        ),
        "RXJ2129_all_roots_pass": bool(
            int(rx.loc[diagnostic.law, "heldout_converged_roots"])
            == int(gates_cfg["RXJ2129_heldout_roots_required"])
        ),
        "four_cluster_improvement_pass": bool(
            float(diagnostic.four_cluster_improvement)
            >= float(gates_cfg["four_cluster_matched_improvement_min"])
        ),
        "RXJ2129_improvement_pass": bool(
            float(diagnostic.RXJ2129_improvement)
            >= float(gates_cfg["RXJ2129_improvement_min"])
        ),
        "galaxy_near_RAR_pass": bool(
            float(p0554["cross_galaxy_outer_RMSE_km_s"])
            <= float(gates_cfg["galaxy_RMSE_to_fixed_RAR_max"]) * rar
        ),
        "Solar_all_proxies_pass": bool(p0554["all_solar_proxies_pass"]),
        "zero_new_fitted_gravity_parameters_pass": True,
    }
    gates["all_diagnostic_gates_pass"] = bool(all(gates.values()))

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(state_rows).to_csv(output / protocol["outputs"]["system_states"], index=False)
    scores.to_csv(output / protocol["outputs"]["scores"], index=False)
    summaries.to_csv(output / protocol["outputs"]["summary_scores"], index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        output / protocol["outputs"]["predictions"], index=False
    )
    pd.DataFrame(audit_rows).to_csv(output / protocol["outputs"]["field_audits"], index=False)
    report = {
        "report_version": "P0615-SELF-COUPLED-QUADRUPOLE-ROUTE-RESULTS-0.1.0",
        "status": "complete_opened_data_parameter_free_response_test",
        "coverage": {
            "raw_systems": 5,
            "heldout_images": 18,
            "amplitude_laws_including_control": len(LAWS),
            "new_fitted_gravity_parameters": 0,
        },
        "derived_quantities": protocol["derived_quantities"],
        "system_states": pd.DataFrame(state_rows).to_dict("records"),
        "summary_scores": summaries.to_dict("records"),
        "combined_diagnostic": combined.to_dict("records"),
        "diagnostic_winner": diagnostic.to_dict(),
        "inherited_cross_domain": {
            "SPARC_outer_RMSE_km_s": float(p0554["cross_galaxy_outer_RMSE_km_s"]),
            "fixed_RAR_outer_RMSE_km_s": rar,
            "SPARC_to_RAR_ratio": float(p0554["cross_galaxy_outer_RMSE_km_s"]) / rar,
            "Solar_all_proxies_pass": bool(p0554["all_solar_proxies_pass"]),
            "route_layer_axisymmetric_and_point_source_change": 0.0,
        },
        "gates": gates,
        "interpretation": {
            "formula_promoted": False,
            "parameter_reduction_achieved": True,
            "quadratic_shape_has_first_principle_motivation": "Q^2 is the lowest even rotational scalar built from the spin-2 baryonic quadrupole.",
            "future_transfer_required": True,
        },
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    pivot = summaries.pivot(index="law", columns="cohort", values="improvement_vs_frozen_scalar")
    root_pivot = summaries.pivot(index="law", columns="cohort", values="heldout_converged_roots")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.3), constrained_layout=True)
    pivot.plot.barh(ax=axes[0])
    axes[0].axvline(0.0, color="black", lw=0.8)
    axes[0].set(xlabel="improvement at frozen geometry", title="Self-coupled amplitude response")
    root_pivot.plot.barh(ax=axes[1])
    axes[1].set(xlabel="converged held-out roots", title="11 four-cluster + 7 RXJ2129")
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)
    (output / protocol["outputs"]["summary"]).write_text(
        "# P0615 self-coupled quadrupole route\n\n"
        f"Diagnostic winner: **{diagnostic.law}** with **{int(diagnostic.combined_roots)}/18** roots.\n\n"
        f"Four-cluster change: **{100*float(diagnostic.four_cluster_improvement):+.3f}%**; "
        f"RXJ2129 change: **{100*float(diagnostic.RXJ2129_improvement):+.3f}%**.\n\n"
        f"All diagnostic gates pass: **{gates['all_diagnostic_gates_pass']}**. This opened-data result cannot promote the law.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            json_safe(
                {
                    "states": report["system_states"],
                    "combined": report["combined_diagnostic"],
                    "gates": gates,
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
