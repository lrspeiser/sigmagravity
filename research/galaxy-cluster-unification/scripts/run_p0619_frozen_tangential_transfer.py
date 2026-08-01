#!/usr/bin/env python3
"""Transfer the frozen +90-degree self-coupled route to A383 and MS2137."""

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
from run_p0566_a383_ms2137_morphology_gate_transfer import build_contexts  # noqa: E402
from run_p0611_frozen_dual_misalignment_raw_transfer import (  # noqa: E402
    complete,
    member_sources,
    score_row,
)
from run_p0615_self_coupled_quadrupole_route import derived_state  # noqa: E402
from run_p0618_universal_route_phase import phase_field  # noqa: E402
from run_unbounded_running_multicluster_raw import load_anchors  # noqa: E402
from run_clash_stellar_morphology_response import MorphologyLens, json_safe  # noqa: E402


def main() -> None:
    protocol_path = ROOT / "configs/p0619_frozen_tangential_transfer_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_after_P0618_before_A383_MS2137_tangential_scores":
        raise RuntimeError("P0619 protocol is not frozen")
    p0616 = json.loads(
        (ROOT / protocol["inputs"]["P0616_protocol"]).read_text(encoding="utf-8")
    )
    inputs = p0616["inputs"]
    p0611 = json.loads((ROOT / inputs["P0611_protocol"]).read_text(encoding="utf-8"))
    p0566 = json.loads((ROOT / inputs["P0566_protocol"]).read_text(encoding="utf-8"))
    base_raw = json.loads((ROOT / inputs["base_raw_protocol"]).read_text(encoding="utf-8"))
    metric = json.loads((ROOT / inputs["metric_protocol"]).read_text(encoding="utf-8"))
    contexts, _, _ = build_contexts(p0566, base_raw, metric)
    if [context.system["label"] for context in contexts] != protocol["systems"]:
        raise RuntimeError("P0619 system coverage changed")
    p0554_protocol = json.loads(
        (ROOT / inputs["P0554_protocol"]).read_text(encoding="utf-8")
    )
    p0554_scores = pd.read_csv(ROOT / inputs["P0554_scores"])
    p0554_row = p0554_scores[p0554_scores.candidate_id.eq("P0554")].iloc[0]
    p0554_specs = {item["candidate_id"]: item for item in build_specs(p0554_protocol)}
    p0581 = json.loads((ROOT / inputs["P0581_protocol"]).read_text(encoding="utf-8"))
    tian = pd.read_csv(
        ROOT / inputs["baryonic_profile"],
        sep=r"\s+",
        names=["system", "radius_kpc", "log_gbar", "log_gobs", "err_log_gbar", "err_log_gobs"],
    )
    prior_scores = pd.read_csv(ROOT / protocol["inputs"]["P0616_scores"])
    baseline_scores = prior_scores[
        prior_scores.variant_id.eq("P0554_scalar_control")
    ].copy()
    radial_scores = prior_scores[
        prior_scores.variant_id.eq("P0615_quadratic_self_route")
    ].set_index("system_label")

    state_rows = []
    audit_rows = []
    candidate_rows = []
    prediction_frames = []
    for system_index, context in enumerate(contexts):
        label = context.system["label"]
        sources = member_sources(
            ROOT / inputs[f"{label}_member_catalog"],
            context,
            p0611["member_selection"],
        )
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
        epsilon = float(state["amplitudes"]["quadratic_Q2_over_total"])
        field, audit = phase_field(p0581, route_context, state, 90.0)
        state_rows.append(
            {
                "system_label": label,
                **{key: value for key, value in state.items() if key != "amplitudes"},
                "selected_epsilon": epsilon,
                "universal_phase_degrees": 90.0,
            }
        )
        audit_rows.append({"system_label": label, "selected_epsilon": epsilon, **audit})
        lens = MorphologyLens(
            local,
            {MODEL: parent},
            parent=MODEL,
            morphology=field,
            fraction=epsilon,
        )
        print(f"P0619 {label}: exact fit +90-degree route", flush=True)
        try:
            fitted = exact_fit(
                lens,
                context.training,
                context.heldout,
                initial=context.initial_geometry,
                starts=int(protocol["fit"]["starts_per_candidate_system"]),
                seed=int(protocol["fit"]["seed"]) + 100 * system_index,
            )
            candidate_rows.append(
                score_row(context, "P0619_tangential_self_route", epsilon, fitted)
            )
            for stage in ("training", "heldout"):
                frame = fitted[f"{stage}_prediction"].copy()
                frame["system_label"] = label
                frame["variant_id"] = "P0619_tangential_self_route"
                prediction_frames.append(frame)
        except Exception as error:
            candidate_rows.append(
                {
                    "system_label": label,
                    "variant_id": "P0619_tangential_self_route",
                    "effective_strength": epsilon,
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

    scores = pd.concat([baseline_scores, pd.DataFrame(candidate_rows)], ignore_index=True)
    indexed = scores.set_index(["system_label", "variant_id"])
    responses = []
    for label in protocol["systems"]:
        base = indexed.loc[(label, "P0554_scalar_control")]
        candidate = indexed.loc[(label, "P0619_tangential_self_route")]
        full = complete(base) and complete(candidate)
        improvement = (
            1.0 - float(candidate.heldout_RMS_arcsec) / float(base.heldout_RMS_arcsec)
            if full
            else np.nan
        )
        radial = radial_scores.loc[label]
        radial_improvement = (
            1.0 - float(radial.heldout_RMS_arcsec) / float(base.heldout_RMS_arcsec)
            if complete(base) and complete(radial)
            else np.nan
        )
        responses.append(
            {
                "system_label": label,
                "complete_pair": full,
                "tangential_improvement_fraction": improvement,
                "prior_radial_improvement_fraction": radial_improvement,
                "tangential_minus_radial_improvement": (
                    improvement - radial_improvement
                    if np.isfinite(improvement) and np.isfinite(radial_improvement)
                    else np.nan
                ),
            }
        )
    responses = pd.DataFrame(responses)
    a383 = responses[responses.system_label.eq("A383")].iloc[0]
    a383_candidate = indexed.loc[("A383", "P0619_tangential_self_route")]
    p0554_report = json.loads((ROOT / inputs["P0554_report"]).read_text(encoding="utf-8"))
    p0554 = next(
        row
        for row in p0554_report["shortlist_selected_without_raw_scores"]
        if row["candidate_id"] == "P0554"
    )
    sparc = json.loads((ROOT / inputs["SPARC_report"]).read_text(encoding="utf-8"))
    rar = float(sparc["scores"]["fixed_RAR:invariant"]["outer_holdout"]["RMSE_km_s"])
    cfg = protocol["gates"]
    gates = {
        "A383_all_roots_pass": bool(complete(a383_candidate)),
        "A383_nonworsening_pass": bool(
            np.isfinite(a383.tangential_improvement_fraction)
            and float(a383.tangential_improvement_fraction) >= 0.0
        ),
        "A383_absolute_RMS_pass": bool(
            float(a383_candidate.heldout_RMS_arcsec)
            <= float(cfg["A383_heldout_RMS_arcsec_max"])
        ),
        "galaxy_near_RAR_pass": bool(
            float(p0554["cross_galaxy_outer_RMSE_km_s"])
            <= float(cfg["galaxy_RMSE_to_fixed_RAR_max"]) * rar
        ),
        "Solar_all_proxies_pass": bool(p0554["all_solar_proxies_pass"]),
        "zero_new_fitted_gravity_parameters_pass": True,
    }
    gates["all_transfer_gates_pass"] = bool(all(gates.values()))

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(state_rows).to_csv(output / protocol["outputs"]["system_states"], index=False)
    scores.to_csv(output / protocol["outputs"]["scores"], index=False)
    pd.DataFrame(audit_rows).to_csv(output / protocol["outputs"]["field_audits"], index=False)
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(
            output / protocol["outputs"]["predictions"], index=False
        )
    report = {
        "report_version": "P0619-FROZEN-TANGENTIAL-TRANSFER-RESULTS-0.1.0",
        "status": "complete_chronologically_prospective_tangential_transfer",
        "chronology": protocol["chronology"],
        "locked_formula": protocol["locked_formula"],
        "coverage": {
            "systems": len(contexts),
            "candidate_refits": len(candidate_rows),
            "complete_matched_systems": int(responses.complete_pair.sum()),
        },
        "system_states": state_rows,
        "responses": responses.to_dict("records"),
        "scores": scores.replace([np.inf, -np.inf], np.nan).to_dict("records"),
        "inherited_cross_domain": {
            "SPARC_outer_RMSE_km_s": float(p0554["cross_galaxy_outer_RMSE_km_s"]),
            "SPARC_to_RAR_ratio": float(p0554["cross_galaxy_outer_RMSE_km_s"]) / rar,
            "Solar_all_proxies_pass": bool(p0554["all_solar_proxies_pass"]),
        },
        "gates": gates,
        "interpretation": {
            "formula_promoted": False,
            "advance_candidate": bool(gates["all_transfer_gates_pass"]),
            "MS2137_comparison_inconclusive_if_control_incomplete": True,
            "per_object_gravity_parameters": 0,
        },
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )

    plot = scores.pivot(index="system_label", columns="variant_id", values="heldout_RMS_arcsec")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True)
    states = pd.DataFrame(state_rows)
    axes[0].bar(states.system_label, states.selected_epsilon, color="#6a3d9a")
    axes[0].set(ylabel="derived epsilon", title="Frozen +90-degree route")
    finite = plot.replace([np.inf, -np.inf], np.nan)
    finite.plot.bar(ax=axes[1], color=["#1f78b4", "#e31a1c"])
    axes[1].set(ylabel="held-out RMS (arcsec)", title="Paired full geometry refit")
    axes[1].tick_params(axis="x", rotation=0)
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)
    (output / protocol["outputs"]["summary"]).write_text(
        "# P0619 frozen tangential transfer\n\n"
        + "\n".join(
            f"- {row.system_label}: tangential change {row.tangential_improvement_fraction:+.3%}; prior radial {row.prior_radial_improvement_fraction:+.3%}; complete pair={row.complete_pair}"
            if np.isfinite(row.tangential_improvement_fraction)
            else f"- {row.system_label}: incomplete matched pair"
            for row in responses.itertuples(index=False)
        )
        + f"\n\nA383 candidate RMS: **{float(a383_candidate.heldout_RMS_arcsec):.3f} arcsec**; all transfer gates pass: **{gates['all_transfer_gates_pass']}**.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            json_safe(
                {
                    "responses": responses.to_dict("records"),
                    "gates": gates,
                    "A383_candidate_RMS_arcsec": float(
                        a383_candidate.heldout_RMS_arcsec
                    ),
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
