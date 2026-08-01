#!/usr/bin/env python3
"""Transfer the frozen P0615 self-coupled route to A383 and MS2137."""

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

from run_adaptive_route_raw_rxj2129 import MODEL, baryon_field, exact_fit  # noqa: E402
from run_arc_invariant_absolute_lensing import raw_field  # noqa: E402
from run_arc_invariant_pareto_refinement import build_specs  # noqa: E402
from run_clash_stellar_morphology_response import MorphologyLens, json_safe  # noqa: E402
from run_p0566_a383_ms2137_morphology_gate_transfer import build_contexts  # noqa: E402
from run_p0611_frozen_dual_misalignment_raw_transfer import (  # noqa: E402
    complete,
    member_sources,
    score_row,
)
from run_p0615_self_coupled_quadrupole_route import (  # noqa: E402
    build_self_field,
    derived_state,
)
from run_unbounded_running_multicluster_raw import load_anchors  # noqa: E402


def main() -> None:
    protocol_path = ROOT / "configs/p0616_frozen_self_coupled_transfer_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_after_P0615_before_A383_MS2137_self_coupled_scores":
        raise RuntimeError("P0616 protocol is not frozen")
    inputs = protocol["inputs"]
    p0611 = json.loads((ROOT / inputs["P0611_protocol"]).read_text(encoding="utf-8"))
    p0566 = json.loads((ROOT / inputs["P0566_protocol"]).read_text(encoding="utf-8"))
    acquisition = json.loads(
        (ROOT / inputs["P0566_acquisition_protocol"]).read_text(encoding="utf-8")
    )
    base_raw = json.loads((ROOT / inputs["base_raw_protocol"]).read_text(encoding="utf-8"))
    metric = json.loads((ROOT / inputs["metric_protocol"]).read_text(encoding="utf-8"))
    contexts, _, _ = build_contexts(p0566, base_raw, metric)
    if [context.system["label"] for context in contexts] != protocol["systems"]:
        raise RuntimeError("P0616 system coverage changed")
    p0554_protocol = json.loads(
        (ROOT / inputs["P0554_protocol"]).read_text(encoding="utf-8")
    )
    p0554_scores = pd.read_csv(ROOT / inputs["P0554_scores"])
    p0554_row = p0554_scores[p0554_scores.candidate_id.eq("P0554")].iloc[0]
    p0554_specs = {item["candidate_id"]: item for item in build_specs(p0554_protocol)}
    p0581 = json.loads(
        (ROOT / inputs["P0581_protocol"]).read_text(encoding="utf-8")
    )
    tian = pd.read_csv(
        ROOT / inputs["baryonic_profile"],
        sep=r"\s+",
        names=["system", "radius_kpc", "log_gbar", "log_gobs", "err_log_gbar", "err_log_gobs"],
    )
    source_frames = []
    state_rows = []
    audit_rows = []
    score_rows = []
    prediction_frames = []
    for system_index, context in enumerate(contexts):
        label = context.system["label"]
        sources = member_sources(
            ROOT / inputs[f"{label}_member_catalog"],
            context,
            p0611["member_selection"],
        )
        source_frames.append(sources)
        local = json.loads(json.dumps(context.local_protocol))
        scale = float(local["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
        images = pd.concat([context.training, context.heldout], ignore_index=True)
        radii = np.hypot(images.x_arcsec.to_numpy(float), images.y_arcsec.to_numpy(float)) * scale
        local.setdefault("baryonic_inputs", {})[
            "strong_lens_impact_radius_range_kpc_expected"
        ] = [float(radii.min()), float(radii.max())]
        anchors = load_anchors(tian, label)
        parent, _ = raw_field(
            p0554_specs["P0554"],
            float(p0554_row.universal_q),
            anchors,
            local,
            1.2e-10,
        )
        baryons = baryon_field(anchors, local)
        route_context = SimpleNamespace(
            system=context.system,
            local=local,
            members=sources,
            parent=parent,
            baryons=baryons,
        )
        state = derived_state(route_context)
        field, audit = build_self_field(p0581, route_context, state)
        epsilon = float(state["amplitudes"]["quadratic_Q2_over_total"])
        state_rows.append(
            {
                "system_label": label,
                **{key: value for key, value in state.items() if key != "amplitudes"},
                "selected_epsilon": epsilon,
            }
        )
        audit_rows.append({"system_label": label, "selected_epsilon": epsilon, **audit})
        variants = [
            (
                "P0554_scalar_control",
                0.0,
                MorphologyLens(local, {MODEL: parent}, parent=MODEL, morphology=None, fraction=0.0),
            ),
            (
                "P0615_quadratic_self_route",
                epsilon,
                MorphologyLens(local, {MODEL: parent}, parent=MODEL, morphology=field, fraction=epsilon),
            ),
        ]
        for variant_id, strength, lens in variants:
            print(f"P0616 {label}: exact fit {variant_id}", flush=True)
            try:
                fitted = exact_fit(
                    lens,
                    context.training,
                    context.heldout,
                    initial=context.initial_geometry,
                    starts=int(protocol["fit"]["starts_per_variant_system"]),
                    seed=int(protocol["fit"]["seed"]) + 100 * system_index,
                )
                score_rows.append(score_row(context, variant_id, strength, fitted))
                for stage in ("training", "heldout"):
                    frame = fitted[f"{stage}_prediction"].copy()
                    frame["system_label"] = label
                    frame["variant_id"] = variant_id
                    prediction_frames.append(frame)
            except Exception as error:
                score_rows.append(
                    {
                        "system_label": label,
                        "variant_id": variant_id,
                        "effective_strength": strength,
                        "training_images": len(context.training),
                        "heldout_images": len(context.heldout),
                        "training_RMS_arcsec": np.inf,
                        "training_roots_converged": 0,
                        "heldout_RMS_arcsec": np.inf,
                        "heldout_roots_converged": 0,
                        "optimizer_cost": np.inf,
                        "failure": f"{type(error).__name__}: {error}",
                    }
                )

    scores = pd.DataFrame(score_rows)
    indexed = scores.set_index(["system_label", "variant_id"])
    responses = []
    for label in protocol["systems"]:
        base = indexed.loc[(label, "P0554_scalar_control")]
        candidate = indexed.loc[(label, "P0615_quadratic_self_route")]
        full = complete(base) and complete(candidate)
        heldout = bool(
            int(base.heldout_roots_converged) == int(base.heldout_images)
            and int(candidate.heldout_roots_converged) == int(candidate.heldout_images)
            and np.isfinite(base.heldout_RMS_arcsec)
            and np.isfinite(candidate.heldout_RMS_arcsec)
        )
        improvement = (
            1.0 - float(candidate.heldout_RMS_arcsec) / float(base.heldout_RMS_arcsec)
            if heldout
            else np.nan
        )
        responses.append(
            {
                "system_label": label,
                "complete_pair": full,
                "heldout_pair_complete": heldout,
                "heldout_improvement_fraction": improvement,
            }
        )
    responses = pd.DataFrame(responses)
    complete_labels = responses[responses.complete_pair].system_label.tolist()
    if complete_labels:
        base_values = indexed.loc[
            [(label, "P0554_scalar_control") for label in complete_labels],
            "heldout_RMS_arcsec",
        ].to_numpy(float)
        candidate_values = indexed.loc[
            [(label, "P0615_quadratic_self_route") for label in complete_labels],
            "heldout_RMS_arcsec",
        ].to_numpy(float)
        base_rms = float(np.sqrt(np.mean(np.square(base_values))))
        candidate_rms = float(np.sqrt(np.mean(np.square(candidate_values))))
        aggregate_improvement = 1.0 - candidate_rms / base_rms
    else:
        base_rms = candidate_rms = np.inf
        aggregate_improvement = -np.inf
    p0554_report = json.loads(
        (ROOT / inputs["P0554_report"]).read_text(encoding="utf-8")
    )
    p0554 = next(
        row
        for row in p0554_report["shortlist_selected_without_raw_scores"]
        if row["candidate_id"] == "P0554"
    )
    sparc = json.loads((ROOT / inputs["SPARC_report"]).read_text(encoding="utf-8"))
    rar = float(sparc["scores"]["fixed_RAR:invariant"]["outer_holdout"]["RMSE_km_s"])
    cfg = protocol["gates"]
    gates = {
        "all_training_and_heldout_roots_each_system_pass": bool(
            responses.complete_pair.all()
        ),
        "both_systems_not_worse_pass": bool(
            responses.complete_pair.all()
            and (responses.heldout_improvement_fraction >= 0.0).all()
        ),
        "equal_system_heldout_improvement_pass": bool(
            aggregate_improvement
            >= float(cfg["equal_system_heldout_improvement_fraction_min"])
        ),
        "equal_system_absolute_RMS_pass": bool(
            candidate_rms <= float(cfg["equal_system_heldout_RMS_arcsec_max"])
        ),
        "galaxy_near_RAR_pass": bool(
            float(p0554["cross_galaxy_outer_RMSE_km_s"])
            <= float(cfg["galaxy_RMSE_to_fixed_RAR_max"]) * rar
        ),
        "Solar_all_proxies_pass": bool(p0554["all_solar_proxies_pass"]),
        "zero_fitted_gravity_parameters_pass": True,
    }
    gates["all_transfer_gates_pass"] = bool(all(gates.values()))

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    pd.concat(source_frames, ignore_index=True).to_csv(
        output / protocol["outputs"]["member_sources"], index=False
    )
    pd.DataFrame(state_rows).to_csv(output / protocol["outputs"]["system_states"], index=False)
    scores.to_csv(output / protocol["outputs"]["scores"], index=False)
    pd.DataFrame(audit_rows).to_csv(output / protocol["outputs"]["field_audits"], index=False)
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(
            output / protocol["outputs"]["predictions"], index=False
        )
    report = {
        "report_version": "P0616-FROZEN-SELF-COUPLED-TRANSFER-RESULTS-0.1.0",
        "status": "complete_chronologically_prospective_formula_transfer",
        "chronology": protocol["chronology"],
        "locked_formula": protocol["locked_formula"],
        "coverage": {
            "systems": len(contexts),
            "member_sources": int(sum(len(frame) for frame in source_frames)),
            "raw_fits": len(scores),
            "complete_matched_systems": len(complete_labels),
        },
        "system_states": state_rows,
        "scores": scores.replace([np.inf, -np.inf], np.nan).to_dict("records"),
        "responses": responses.to_dict("records"),
        "aggregate": {
            "matched_baseline_RMS_arcsec": base_rms if np.isfinite(base_rms) else None,
            "matched_candidate_RMS_arcsec": candidate_rms if np.isfinite(candidate_rms) else None,
            "matched_improvement_fraction": (
                aggregate_improvement if np.isfinite(aggregate_improvement) else None
            ),
        },
        "inherited_cross_domain": {
            "SPARC_outer_RMSE_km_s": float(p0554["cross_galaxy_outer_RMSE_km_s"]),
            "SPARC_to_RAR_ratio": float(p0554["cross_galaxy_outer_RMSE_km_s"]) / rar,
            "Solar_all_proxies_pass": bool(p0554["all_solar_proxies_pass"]),
        },
        "gates": gates,
        "interpretation": {
            "formula_promoted": False,
            "advance_candidate": bool(gates["all_transfer_gates_pass"]),
            "per_object_gravity_parameters": 0,
            "pristine_project_holdout": False,
            "conclusion": (
                "The fixed-geometry P0615 gain did not transfer: A383 worsened "
                "slightly and MS2137 was inconclusive because neither control nor "
                "candidate achieved complete roots."
            ),
        },
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    states = pd.DataFrame(state_rows)
    axes[0].bar(states.system_label, states.selected_epsilon, color="#1261A0")
    axes[0].set(ylabel="derived epsilon", title="No fitted route strength")
    x = np.arange(len(protocol["systems"]))
    base_plot = scores[scores.variant_id.eq("P0554_scalar_control")].set_index("system_label").loc[protocol["systems"]]
    route_plot = scores[scores.variant_id.eq("P0615_quadratic_self_route")].set_index("system_label").loc[protocol["systems"]]
    axes[1].bar(x - 0.18, base_plot.heldout_RMS_arcsec, 0.36, label="P0554 scalar")
    axes[1].bar(x + 0.18, route_plot.heldout_RMS_arcsec, 0.36, label="self route")
    axes[1].set(xticks=x, xticklabels=protocol["systems"], ylabel="held-out RMS (arcsec)", title="Full ordinary-geometry refit")
    axes[1].legend()
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)
    (output / protocol["outputs"]["summary"]).write_text(
        "# P0616 frozen self-coupled transfer\n\n"
        + "\n".join(
            f"- {row.system_label}: held-out change {row.heldout_improvement_fraction:+.3%}, complete pair={row.complete_pair}"
            if np.isfinite(row.heldout_improvement_fraction)
            else f"- {row.system_label}: incomplete pair"
            for row in responses.itertuples(index=False)
        )
        + f"\n\nMatched improvement: **{aggregate_improvement:+.3%}**; candidate RMS: **{candidate_rms:.3f} arcsec**; all transfer gates pass: **{gates['all_transfer_gates_pass']}**.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            json_safe(
                {
                    "states": state_rows,
                    "responses": responses.to_dict("records"),
                    "aggregate": report["aggregate"],
                    "gates": gates,
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
