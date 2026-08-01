#!/usr/bin/env python3
"""Transfer the frozen baryonic-network kernel to RXJ1347."""

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
from run_p0554_baryonic_network_screen import build_network_field  # noqa: E402
from run_p0554_local_cross_domain_sensitivity import A0, json_safe, sha256  # noqa: E402
from run_p0554_route_localization_screen import build_directional_field, linearized_residuals  # noqa: E402
from run_p0554_route_softness_interaction import build_variants  # noqa: E402
from run_rxj2129_raw_theory_lensing import FIXED_LABELS, near_bound, score as raw_score  # noqa: E402
from run_unbounded_running_multicluster_raw import (  # noqa: E402
    load_anchors,
    load_system_images,
    system_protocol,
)


def pair_split(images):
    training, heldout = [], []
    for _, group in images.groupby("source_family", sort=True):
        ordered = group.sort_values("image_id")
        if len(ordered) != 2:
            raise RuntimeError("RXJ1347 pair split expected exactly two images per family")
        training.append(ordered.iloc[[0]])
        heldout.append(ordered.iloc[[1]])
    return pd.concat(training, ignore_index=True), pd.concat(heldout, ignore_index=True)


def initial_geometry(path):
    frame = pd.read_csv(path)
    row = frame[(frame.system == "RX J1347.5-1145") & (frame.model == "baryons_GR")].iloc[0]
    return row[list(FIXED_LABELS)].to_numpy(float)


def main():
    config_path = ROOT / "configs/p0554_baryonic_network_rxj1347_holdout_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    screen_path = ROOT / protocol["inputs"]["network_screen_protocol"]
    screen = json.loads((ROOT / protocol["inputs"]["network_screen_report"]).read_text(encoding="utf-8"))
    if screen["protocol"]["sha256"] != sha256(screen_path):
        raise RuntimeError("network screen report does not match its protocol")
    if screen["descriptive_holdout_transfer_variant"] != protocol["candidate"]["variant_id"]:
        raise RuntimeError("frozen holdout candidate differs from the discovery selection")

    interaction = json.loads((ROOT / protocol["inputs"]["interaction_protocol"]).read_text(encoding="utf-8"))
    route_protocol = json.loads((ROOT / protocol["inputs"]["route_protocol"]).read_text(encoding="utf-8"))
    raw_protocol = json.loads((ROOT / protocol["inputs"]["raw_protocol"]).read_text(encoding="utf-8"))
    system = next(row for row in raw_protocol["systems"] if row["label"] == protocol["evaluation"]["label"])
    local = system_protocol(raw_protocol, system)
    local["optimization"]["maximum_function_evaluations"] = int(protocol["evaluation"]["maximum_function_evaluations"])
    catalog = pd.read_csv(ROOT / protocol["inputs"]["image_catalog"])
    images = load_system_images(catalog, system)
    training, heldout = pair_split(images)
    tian = pd.read_csv(
        ROOT / protocol["inputs"]["baryonic_profile"],
        sep=r"\s+",
        names=["system", "radius_kpc", "log_gbar", "log_gobs", "err_log_gbar", "err_log_gobs"],
    )
    anchors = load_anchors(tian, system["label"])
    parent = next(item for item in build_variants(interaction) if item.variant_id == "combined_parent")
    radial, _ = raw_field(parent.spec, parent.q, anchors, local, A0)
    baryons = baryon_field(anchors, local)
    members = load_member_sources(
        ROOT / protocol["inputs"]["member_catalog"],
        system,
        local,
        route_protocol["member_sources"],
    )
    context = SimpleNamespace(label=system["label"], system=system["system"], local=local, anchors=anchors)
    adaptive = route_fraction(parent.candidate, members, local)
    strength = float(protocol["candidate"]["eta"]) * float(adaptive["routing_fraction"] ** protocol["candidate"]["route_power"])

    baseline_lens = MorphologyLens(local, {parent.variant_id: radial}, parent=parent.variant_id, morphology=None, fraction=0.0)
    fit = baseline_lens.fit(
        parent.variant_id,
        images,
        starts=int(protocol["evaluation"]["baseline_fit_starts"]),
        seed=int(protocol["evaluation"]["random_seed"]),
        initial_override=initial_geometry(ROOT / protocol["inputs"]["initial_geometry"]),
    )
    geometry = fit["result"].x
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{
            "system": system["system"],
            **dict(zip(FIXED_LABELS, geometry, strict=True)),
            "optimizer_cost": float(fit["result"].cost),
            "geometry_at_boundary": any(near_bound(parent.variant_id, geometry).values()),
        }]
    ).to_csv(output / protocol["outputs"]["baseline_geometry"], index=False)

    network_variant = dict(protocol["candidate"])
    network, network_audit = build_network_field(
        interaction, route_protocol, context, members, parent, radial, baryons, network_variant
    )
    local_vector, local_audit = build_directional_field(
        interaction,
        route_protocol,
        context,
        members,
        parent,
        radial,
        baryons,
        {
            "local_mix": 1.0,
            "softening_kpc": 200.0,
            "neighbor_weight_power": 1.0,
            "distance_power": 2.0,
            "symmetric_bend_degrees": 0.0,
            "center_mode": "light_centroid",
        },
    )
    variants = [
        ("eta_000", None, 0.0),
        ("local_vector_s200", local_vector, strength),
        (protocol["candidate"]["variant_id"], network, strength),
    ]
    rows, predictions = [], []
    for variant_id, morphology, fraction in variants:
        lens = MorphologyLens(local, {parent.variant_id: radial}, parent=parent.variant_id, morphology=morphology, fraction=fraction)
        _, sources = lens.profiled_residuals(parent.variant_id, geometry, training)
        linear = linearized_residuals(lens, parent.variant_id, geometry, sources, heldout, "heldout")
        exact = lens.exact_predictions(parent.variant_id, geometry, sources, heldout, stage="heldout")
        metrics = raw_score(exact, lens.sigma)
        rows.append(
            {
                "variant_id": variant_id,
                "training_images": len(training),
                "heldout_images": len(heldout),
                "heldout_exact_RMS_arcsec": metrics["exact_radial_RMS_arcsec"],
                "heldout_roots_converged": metrics["converged_roots"],
                "heldout_all_roots": metrics["all_roots_converged"],
                "heldout_linearized_RMS_arcsec": float(np.sqrt(np.mean(np.square(linear.linearized_radial_residual_arcsec)))),
                "angular_strength": fraction,
            }
        )
        exact.insert(0, "variant_id", variant_id)
        predictions.append(exact)
    scores = pd.DataFrame(rows)
    scores.to_csv(output / protocol["outputs"]["scores"], index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / protocol["outputs"]["predictions"], index=False)
    pd.DataFrame(
        [
            {"variant_id": "local_vector_s200", **local_audit},
            {"variant_id": protocol["candidate"]["variant_id"], **network_audit},
        ]
    ).to_csv(output / protocol["outputs"]["field_audits"], index=False)
    indexed = scores.set_index("variant_id")
    candidate_id = protocol["candidate"]["variant_id"]
    candidate = indexed.loc[candidate_id]
    gate = bool(
        candidate.heldout_all_roots
        and candidate.heldout_exact_RMS_arcsec < indexed.loc["eta_000", "heldout_exact_RMS_arcsec"]
        and candidate.heldout_exact_RMS_arcsec < indexed.loc["local_vector_s200", "heldout_exact_RMS_arcsec"]
    )
    report = {
        "report_version": "P0554-BARYONIC-NETWORK-RXJ1347-HOLDOUT-RESULTS-0.1.0",
        "status": "complete",
        "protocol": {"path": str(config_path.relative_to(ROOT)), "sha256": sha256(config_path)},
        "coverage": {
            "images": len(images),
            "families": int(images.source_family.nunique()),
            "pair_training_images": len(training),
            "pair_heldout_images": len(heldout),
            "member_sources": len(members),
            "baseline_geometry_fit_starts": int(protocol["evaluation"]["baseline_fit_starts"]),
        },
        "baseline_geometry": {
            "optimizer_cost": float(fit["result"].cost),
            "at_boundary": any(near_bound(parent.variant_id, geometry).values()),
        },
        "scores": scores.to_dict("records"),
        "candidate_improvement_fraction_vs_eta0": float(1.0 - candidate.heldout_exact_RMS_arcsec / indexed.loc["eta_000", "heldout_exact_RMS_arcsec"]),
        "candidate_improvement_fraction_vs_local_vector": float(1.0 - candidate.heldout_exact_RMS_arcsec / indexed.loc["local_vector_s200", "heldout_exact_RMS_arcsec"]),
        "transfer_gate_passed": gate,
        "cross_domain_preservation": screen["cross_domain_preservation"],
        "verdict": {"network_generalizes_better_than_both_references": gate, "no_formula_promoted": True},
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.bar(scores.variant_id, scores.heldout_exact_RMS_arcsec, color=["0.5", "tab:blue", "tab:orange"])
    ax.set(ylabel="three pair-heldout roots RMS (arcsec)", title="RXJ1347 frozen formula transfer")
    ax.tick_params(axis="x", rotation=15)
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)
    summary = (
        "# RXJ1347 baryonic-network holdout\n\n"
        f"The frozen network changes the exact three-image pair-heldout RMS by {100*report['candidate_improvement_fraction_vs_eta0']:+.3f}% versus eta=0 and "
        f"{100*report['candidate_improvement_fraction_vs_local_vector']:+.3f}% versus the local-vector reference. Transfer gate: {gate}.\n"
    )
    (output / protocol["outputs"]["summary"]).write_text(summary, encoding="utf-8")
    print(json.dumps(json_safe(report), indent=2), flush=True)


if __name__ == "__main__":
    main()
