#!/usr/bin/env python3
"""Measure the transition from coherent local routing to explicit branches."""

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

from run_adaptive_route_multicluster_raw import load_member_sources, route_fraction  # noqa: E402
from run_adaptive_route_raw_rxj2129 import baryon_field  # noqa: E402
from run_arc_invariant_absolute_lensing import raw_field  # noqa: E402
from run_clash_stellar_morphology_response import MorphologyLens  # noqa: E402
from run_p0554_baryonic_network_rxj1347_holdout import pair_split  # noqa: E402
from run_p0554_baryonic_network_screen import build_network_field  # noqa: E402
from run_p0554_local_cross_domain_sensitivity import A0, json_safe, raw_contexts, sha256  # noqa: E402
from run_p0554_route_localization_screen import build_directional_field, geometry_for, linearized_residuals, rms  # noqa: E402
from run_p0554_route_softness_interaction import build_variants, load_route_sources  # noqa: E402
from run_rxj2129_raw_theory_lensing import FIXED_LABELS, score as raw_score  # noqa: E402
from run_unbounded_running_multicluster_raw import load_anchors, load_system_images, system_protocol  # noqa: E402
from voidscreen.stellar_morphology_lensing import blend_morphology_deflection_fields  # noqa: E402


def local_variant():
    return {
        "local_mix": 1.0,
        "softening_kpc": 200.0,
        "neighbor_weight_power": 1.0,
        "distance_power": 2.0,
        "symmetric_bend_degrees": 0.0,
        "center_mode": "light_centroid",
    }


def field_pair(interaction, route_protocol, context, sources, parent, radial, baryons, network_spec):
    local, local_audit = build_directional_field(
        interaction, route_protocol, context, sources, parent, radial, baryons, local_variant()
    )
    network, network_audit = build_network_field(
        interaction, route_protocol, context, sources, parent, radial, baryons, network_spec
    )
    return local, network, local_audit, network_audit


def main():
    config_path = ROOT / "configs/p0554_route_coherence_transition_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    interaction = json.loads((ROOT / protocol["inputs"]["interaction_protocol"]).read_text(encoding="utf-8"))
    route_protocol = json.loads((ROOT / protocol["inputs"]["route_protocol"]).read_text(encoding="utf-8"))
    parent = next(item for item in build_variants(interaction) if item.variant_id == "combined_parent")
    fractions = [float(value) for value in protocol["branch_fractions"]]
    network_spec = dict(protocol["network"])
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)

    contexts = raw_contexts(interaction)
    route_sources, route_protocols = load_route_sources(interaction, contexts)
    geometry_frame = pd.read_csv(ROOT / protocol["inputs"]["fixed_geometry"])
    discovery_rows, audit_rows = [], []
    for context in contexts:
        geometry = geometry_for(geometry_frame, context.label)
        radial, _ = raw_field(parent.spec, parent.q, context.anchors, context.local, A0)
        baryons = baryon_field(context.anchors, context.local)
        adaptive = route_fraction(parent.candidate, route_sources[context.label], context.local)
        strength = 0.30 * float(adaptive["routing_fraction"] ** parent.route_power)
        local, network, local_audit, network_audit = field_pair(
            interaction,
            route_protocols[context.label],
            context,
            route_sources[context.label],
            parent,
            radial,
            baryons,
            network_spec,
        )
        audit_rows.extend([
            {"scope": "discovery", "system_label": context.label, "field": "local", **local_audit},
            {"scope": "discovery", "system_label": context.label, "field": "network", **network_audit},
        ])
        for fraction in fractions:
            morphology = blend_morphology_deflection_fields(local, network, fraction)
            lens = MorphologyLens(context.local, {parent.variant_id: radial}, parent=parent.variant_id, morphology=morphology, fraction=strength)
            _, sources = lens.profiled_residuals(parent.variant_id, geometry, context.training)
            heldout = linearized_residuals(lens, parent.variant_id, geometry, sources, context.heldout, "heldout")
            discovery_rows.append({
                "system_label": context.label,
                "branch_fraction": fraction,
                "heldout_images": len(heldout),
                "heldout_linearized_RMS_arcsec": rms(heldout.linearized_radial_residual_arcsec),
            })
    discovery_system = pd.DataFrame(discovery_rows)
    discovery_scores = []
    for fraction, block in discovery_system.groupby("branch_fraction"):
        discovery_scores.append({
            "branch_fraction": float(fraction),
            "equal_system_RMS_arcsec": rms(block.heldout_linearized_RMS_arcsec),
            "systems_better_than_coherent": 0,
        })
    discovery_scores = pd.DataFrame(discovery_scores).sort_values("branch_fraction")
    coherent_system = discovery_system[discovery_system.branch_fraction.eq(0.0)].set_index("system_label")
    for index, row in discovery_scores.iterrows():
        block = discovery_system[discovery_system.branch_fraction.eq(row.branch_fraction)].set_index("system_label")
        discovery_scores.loc[index, "systems_better_than_coherent"] = int((block.heldout_linearized_RMS_arcsec < coherent_system.heldout_linearized_RMS_arcsec).sum())
    coherent_rms = float(discovery_scores[discovery_scores.branch_fraction.eq(0.0)].equal_system_RMS_arcsec.iloc[0])
    discovery_scores["change_fraction_vs_coherent"] = discovery_scores.equal_system_RMS_arcsec / coherent_rms - 1.0
    selected = discovery_scores[discovery_scores.branch_fraction.gt(0.0)].sort_values("equal_system_RMS_arcsec").iloc[0]
    selected_fraction = float(selected.branch_fraction)
    discovery_scores.to_csv(output / protocol["outputs"]["discovery_scores"], index=False)
    discovery_system.to_csv(output / protocol["outputs"]["discovery_system_scores"], index=False)

    raw_protocol = json.loads((ROOT / protocol["inputs"]["raw_protocol"]).read_text(encoding="utf-8"))
    system = next(row for row in raw_protocol["systems"] if row["label"] == "RXJ1347")
    local_protocol = system_protocol(raw_protocol, system)
    catalog = pd.read_csv(ROOT / protocol["inputs"]["image_catalog"])
    images = load_system_images(catalog, system)
    training, heldout = pair_split(images)
    tian = pd.read_csv(ROOT / protocol["inputs"]["baryonic_profile"], sep=r"\s+", names=["system","radius_kpc","log_gbar","log_gobs","err_log_gbar","err_log_gobs"])
    anchors = load_anchors(tian, "RXJ1347")
    radial, _ = raw_field(parent.spec, parent.q, anchors, local_protocol, A0)
    baryons = baryon_field(anchors, local_protocol)
    members = load_member_sources(ROOT / protocol["inputs"]["RXJ1347_member_catalog"], system, local_protocol, route_protocol["member_sources"])
    context = SimpleNamespace(label="RXJ1347", system=system["system"], local=local_protocol, anchors=anchors)
    adaptive = route_fraction(parent.candidate, members, local_protocol)
    strength = 0.30 * float(adaptive["routing_fraction"] ** parent.route_power)
    local, network, local_audit, network_audit = field_pair(interaction, route_protocol, context, members, parent, radial, baryons, network_spec)
    audit_rows.extend([
        {"scope": "holdout", "system_label": "RXJ1347", "field": "local", **local_audit},
        {"scope": "holdout", "system_label": "RXJ1347", "field": "network", **network_audit},
    ])
    geometry_row = pd.read_csv(ROOT / protocol["inputs"]["RXJ1347_geometry"]).iloc[0]
    geometry = geometry_row[list(FIXED_LABELS)].to_numpy(float)
    holdout_rows, prediction_frames = [], []
    for fraction in fractions:
        morphology = blend_morphology_deflection_fields(local, network, fraction)
        lens = MorphologyLens(local_protocol, {parent.variant_id: radial}, parent=parent.variant_id, morphology=morphology, fraction=strength)
        _, sources = lens.profiled_residuals(parent.variant_id, geometry, training)
        exact = lens.exact_predictions(parent.variant_id, geometry, sources, heldout, stage="heldout")
        score = raw_score(exact, lens.sigma)
        holdout_rows.append({
            "branch_fraction": fraction,
            "heldout_exact_RMS_arcsec": score["exact_radial_RMS_arcsec"],
            "heldout_roots_converged": score["converged_roots"],
            "heldout_all_roots": score["all_roots_converged"],
        })
        exact.insert(0, "branch_fraction", fraction)
        prediction_frames.append(exact)
    holdout_scores = pd.DataFrame(holdout_rows)
    holdout_scores.to_csv(output / protocol["outputs"]["holdout_scores"], index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(output / protocol["outputs"]["holdout_predictions"], index=False)
    pd.DataFrame(audit_rows).to_csv(output / protocol["outputs"]["field_audits"], index=False)

    holdout_index = holdout_scores.set_index("branch_fraction")
    selected_holdout = holdout_index.loc[selected_fraction]
    coherent_holdout = holdout_index.loc[0.0]
    gate = bool(selected_holdout.heldout_all_roots and selected_holdout.heldout_exact_RMS_arcsec < coherent_holdout.heldout_exact_RMS_arcsec)
    parent_report = json.loads((ROOT / protocol["inputs"]["network_screen_report"]).read_text(encoding="utf-8"))
    report = {
        "report_version": "P0554-ROUTE-COHERENCE-TRANSITION-RESULTS-0.1.0",
        "status": "complete",
        "protocol": {"path": str(config_path.relative_to(ROOT)), "sha256": sha256(config_path)},
        "coverage": {"discovery_systems": 5, "branch_fractions": len(fractions), "discovery_formula_scores": len(discovery_system), "holdout_images": len(heldout)},
        "discovery_scores": discovery_scores.to_dict("records"),
        "selected_positive_branch_fraction": selected_fraction,
        "selected_discovery_change_fraction_vs_coherent": float(selected.change_fraction_vs_coherent),
        "holdout_scores": holdout_scores.to_dict("records"),
        "selected_holdout_change_fraction_vs_coherent": float(selected_holdout.heldout_exact_RMS_arcsec / coherent_holdout.heldout_exact_RMS_arcsec - 1.0),
        "transfer_gate_passed": gate,
        "cross_domain_preservation": parent_report["cross_domain_preservation"],
        "verdict": {"positive_branch_fraction_beats_fully_coherent_on_holdout": gate, "no_formula_promoted": True},
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].plot(discovery_scores.branch_fraction, discovery_scores.equal_system_RMS_arcsec, marker="o")
    axes[0].axvline(selected_fraction, color="tab:orange", linestyle="--")
    axes[0].set(xlabel="explicit branch fraction zeta", ylabel="five-cluster RMS (arcsec)", title="Spent discovery coherence transition")
    axes[1].plot(holdout_scores.branch_fraction, holdout_scores.heldout_exact_RMS_arcsec, marker="o", color="tab:green")
    axes[1].axvline(selected_fraction, color="tab:orange", linestyle="--", label="discovery selected")
    axes[1].set(xlabel="explicit branch fraction zeta", ylabel="RXJ1347 exact RMS (arcsec)", title="Frozen formula holdout")
    axes[1].legend()
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)
    summary = (
        "# Route coherence transition\n\n"
        f"The lowest discovery RMS among positive branch fractions occurs at zeta={selected_fraction:g}, changing RMS by {100*float(selected.change_fraction_vs_coherent):+.3f}% versus the coherent vector. "
        f"On RXJ1347 it changes exact heldout RMS by {100*report['selected_holdout_change_fraction_vs_coherent']:+.3f}%. Transfer gate: {gate}.\n"
    )
    (output / protocol["outputs"]["summary"]).write_text(summary, encoding="utf-8")
    print(json.dumps(json_safe(report), indent=2), flush=True)


if __name__ == "__main__":
    main()
