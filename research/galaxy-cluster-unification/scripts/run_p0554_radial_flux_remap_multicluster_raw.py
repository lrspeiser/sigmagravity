#!/usr/bin/env python3
"""Transfer every frozen radial-remap setting to five raw lens systems."""

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

from run_p0554_local_cross_domain_sensitivity import json_safe, raw_contexts, rms, sha256  # noqa: E402
from run_p0554_radial_flux_remap import candidates, raw_radial_field  # noqa: E402
from run_rxj2129_raw_theory_lensing import RawLens, score as raw_score  # noqa: E402


def evaluate_context(context, variants, spec, q):
    fields, profiles = {}, []
    for index, item in enumerate(variants, start=1):
        print(f"{context.label}: field {index}/{len(variants)} {item['candidate_id']}", flush=True)
        field, profile = raw_radial_field(
            spec,
            q,
            context.anchors,
            context.local,
            route_fraction=float(item["route_fraction"]),
            radial_scale=float(item["radial_scale"]),
            candidate_id=str(item["candidate_id"]),
        )
        fields[item["candidate_id"]] = field
        profile.insert(0, "system_label", context.label)
        profiles.append(profile)
    lens = RawLens(context.local, fields)
    rows, predictions = [], []
    for item in variants:
        model = item["candidate_id"]
        _, sources = lens.profiled_residuals(model, context.geometry, context.training)
        train = lens.exact_predictions(model, context.geometry, sources, context.training, stage="training")
        held = lens.exact_predictions(model, context.geometry, sources, context.heldout, stage="heldout")
        train_metrics = raw_score(train, lens.sigma)
        held_metrics = raw_score(held, lens.sigma)
        rows.append(
            {
                "system": context.system,
                "system_label": context.label,
                "candidate_id": model,
                "route_fraction": float(item["route_fraction"]),
                "radial_scale": float(item["radial_scale"]),
                "training_images": len(context.training),
                "training_RMS_arcsec": train_metrics["exact_radial_RMS_arcsec"],
                "training_all_roots": train_metrics["all_roots_converged"],
                "heldout_images": len(context.heldout),
                "heldout_RMS_arcsec": held_metrics["exact_radial_RMS_arcsec"],
                "heldout_roots_converged": held_metrics["converged_roots"],
                "heldout_all_roots": held_metrics["all_roots_converged"],
            }
        )
        for frame in (train, held):
            local = frame.copy()
            local.insert(0, "system", context.system)
            local.insert(1, "system_label", context.label)
            local.insert(2, "candidate_id", model)
            predictions.append(local)
    return rows, predictions, profiles


def aggregate(system_scores, variants):
    parent = system_scores[system_scores.candidate_id.eq("parent")].set_index("system_label")
    rows = []
    for item in variants:
        block = system_scores[system_scores.candidate_id.eq(item["candidate_id"])].set_index("system_label")
        common = [
            label
            for label in parent.index
            if bool(parent.loc[label, "heldout_all_roots"])
            and bool(block.loc[label, "heldout_all_roots"])
            and np.isfinite(float(parent.loc[label, "heldout_RMS_arcsec"]))
            and np.isfinite(float(block.loc[label, "heldout_RMS_arcsec"]))
        ]
        parent_rms = rms(parent.loc[common, "heldout_RMS_arcsec"]) if common else np.nan
        candidate_rms = rms(block.loc[common, "heldout_RMS_arcsec"]) if common else np.nan
        rows.append(
            {
                **item,
                "complete_systems": int(block.heldout_all_roots.astype(bool).sum()),
                "converged_roots": int(block.heldout_roots_converged.sum()),
                "heldout_images": int(block.heldout_images.sum()),
                "matched_complete_systems": len(common),
                "matched_labels": "+".join(common),
                "matched_parent_RMS_arcsec": parent_rms,
                "matched_candidate_RMS_arcsec": candidate_rms,
                "matched_gain_vs_parent": None if not common else float(1.0 - candidate_rms / parent_rms),
            }
        )
    return pd.DataFrame(rows)


def make_figure(system_scores, candidate_scores, selected_id, output):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    ax = axes[0]
    for label, block in system_scores.groupby("system_label", sort=True):
        merged = block.copy()
        x = merged.route_fraction * (merged.radial_scale - 1.0)
        baseline = merged[merged.candidate_id.eq("parent")].heldout_RMS_arcsec.iloc[0]
        ax.scatter(x, merged.heldout_RMS_arcsec / baseline, s=15, alpha=0.55, label=label)
    ax.axhline(1.0, color="black", ls="--")
    ax.set(xlabel="effective radial displacement f(lambda-1)", ylabel="heldout RMS / parent", title="System-specific raw-lens response")
    ax.legend(fontsize=7, ncol=2)
    ax = axes[1]
    complete = candidate_scores[candidate_scores.matched_complete_systems.eq(candidate_scores.matched_complete_systems.max())]
    x = complete.route_fraction * (complete.radial_scale - 1.0)
    ax.scatter(x, 100 * complete.matched_gain_vs_parent, c=complete.radial_scale, cmap="coolwarm", alpha=0.8)
    selected = candidate_scores[candidate_scores.candidate_id.eq(selected_id)].iloc[0]
    ax.scatter(selected.route_fraction * (selected.radial_scale - 1.0), 100 * selected.matched_gain_vs_parent, marker="*", s=220, c="gold", edgecolor="black")
    ax.axhline(0.0, color="black", ls="--")
    ax.set(xlabel="effective radial displacement f(lambda-1)", ylabel="matched raw-lens gain (%)", title="Five-system aggregate")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    config_path = ROOT / "configs/p0554_radial_flux_remap_multicluster_raw_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    radial_path = ROOT / protocol["inputs"]["radial_protocol"]
    radial = json.loads(radial_path.read_text(encoding="utf-8"))
    prior = json.loads((ROOT / protocol["inputs"]["radial_report"]).read_text(encoding="utf-8"))
    if prior["protocol"]["sha256"] != sha256(radial_path):
        raise RuntimeError("prior radial report does not match its frozen protocol")
    selected_id = prior["selected"]["candidate_id"]
    spec = dict(radial["parent"])
    q = float(spec.pop("universal_q"))
    spec.pop("candidate")
    spec["candidate_id"] = "P0554_radial_parent"
    variants = candidates(radial)
    context_protocol = json.loads((ROOT / protocol["inputs"]["context_protocol"]).read_text(encoding="utf-8"))
    contexts = raw_contexts(context_protocol)
    geometry = pd.read_csv(ROOT / protocol["inputs"]["fixed_geometry"])
    for context in contexts:
        row = geometry[(geometry.system_label.eq(context.label)) & (geometry.variant_id.eq("lensing_softness_098"))].iloc[0]
        context.geometry = row[["axis_ratio_q", "position_angle_phi_radian", "center_x_arcsec", "center_y_arcsec", "external_shear_gamma1", "external_shear_gamma2"]].to_numpy(float)
    rows, predictions, profiles = [], [], []
    for context in contexts:
        local_rows, local_predictions, local_profiles = evaluate_context(context, variants, spec, q)
        rows.extend(local_rows); predictions.extend(local_predictions); profiles.extend(local_profiles)
    system_scores = pd.DataFrame(rows)
    candidate_scores = aggregate(system_scores, variants)
    scalar = pd.read_csv(ROOT / protocol["inputs"]["radial_scores"])[["candidate_id", "discovery_galaxy_gain", "discovery_cluster_gain", "RXJ1347_raw_gain"]]
    candidate_scores = candidate_scores.merge(scalar, on="candidate_id", validate="one_to_one")
    candidate_scores["improves_all_three_prior_domains"] = (candidate_scores.discovery_galaxy_gain > 0) & (candidate_scores.discovery_cluster_gain > 0) & (candidate_scores.RXJ1347_raw_gain > 0)
    candidate_scores["improves_prior_galaxy_cluster_and_five_raw"] = (candidate_scores.discovery_galaxy_gain > 0) & (candidate_scores.discovery_cluster_gain > 0) & (candidate_scores.matched_gain_vs_parent > 0)
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    system_scores.to_csv(output / protocol["outputs"]["system_scores"], index=False)
    candidate_scores.to_csv(output / protocol["outputs"]["candidate_scores"], index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / protocol["outputs"]["predictions"], index=False)
    pd.concat(profiles, ignore_index=True).to_csv(output / protocol["outputs"]["profiles"], index=False)
    parent = candidate_scores[candidate_scores.candidate_id.eq("parent")].iloc[0]
    selected = candidate_scores[candidate_scores.candidate_id.eq(selected_id)].iloc[0]
    ranked = candidate_scores[candidate_scores.matched_complete_systems.eq(parent.matched_complete_systems)].sort_values("matched_candidate_RMS_arcsec")
    preference_rows = []
    for label, block in system_scores.groupby("system_label", sort=True):
        parent_row = block[block.candidate_id.eq("parent")].iloc[0]
        inward = block[(block.route_fraction > 0) & (block.radial_scale < 1) & block.heldout_all_roots.astype(bool)]
        outward = block[(block.route_fraction > 0) & (block.radial_scale > 1) & block.heldout_all_roots.astype(bool)]
        best_in = inward.sort_values("heldout_RMS_arcsec").iloc[0] if not inward.empty else None
        best_out = outward.sort_values("heldout_RMS_arcsec").iloc[0] if not outward.empty else None
        preference_rows.append({
            "system_label": label,
            "parent_RMS_arcsec": float(parent_row.heldout_RMS_arcsec),
            "parent_all_roots": bool(parent_row.heldout_all_roots),
            "best_inward_id": None if best_in is None else best_in.candidate_id,
            "best_inward_RMS_arcsec": None if best_in is None else float(best_in.heldout_RMS_arcsec),
            "best_outward_id": None if best_out is None else best_out.candidate_id,
            "best_outward_RMS_arcsec": None if best_out is None else float(best_out.heldout_RMS_arcsec),
            "preferred_direction": "unavailable" if best_in is None or best_out is None else ("inward" if best_in.heldout_RMS_arcsec < best_out.heldout_RMS_arcsec else "outward"),
        })
    report = {
        "report_version": "P0554-RADIAL-FLUX-REMAP-MULTICLUSTER-RAW-RESULTS-0.1.0",
        "status": "complete",
        "protocol": {"path": str(config_path.relative_to(ROOT)), "sha256": sha256(config_path)},
        "coverage": {"candidates": len(candidate_scores), "systems": len(contexts), "heldout_images": int(system_scores[system_scores.candidate_id.eq("parent")].heldout_images.sum())},
        "parent": parent.to_dict(),
        "frozen_selected": selected.to_dict(),
        "best_matched_complete_candidate": ranked.iloc[0].to_dict(),
        "system_direction_preferences": preference_rows,
        "counts": {
            "nonparent_five_raw_improvers": int(((candidate_scores.candidate_id != "parent") & (candidate_scores.matched_gain_vs_parent > 0)).sum()),
            "galaxy_derived_cluster_and_five_raw_improvers": int(((candidate_scores.candidate_id != "parent") & candidate_scores.improves_prior_galaxy_cluster_and_five_raw).sum()),
            "galaxy_derived_cluster_RXJ1347_and_five_raw_improvers": int(((candidate_scores.candidate_id != "parent") & candidate_scores.improves_all_three_prior_domains & (candidate_scores.matched_gain_vs_parent > 0)).sum()),
        },
        "verdict": {"universal_radial_remap_supported": False, "no_formula_promoted": True},
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8")
    make_figure(system_scores, candidate_scores, selected_id, output / protocol["outputs"]["figure"])
    summary = (
        "# P0554 radial remap: five raw clusters\n\n"
        f"The frozen scalar-selected candidate changes the matched raw-lens RMS by {100*float(selected.matched_gain_vs_parent):+.3f}%. "
        f"The best complete-grid candidate is `{ranked.iloc[0].candidate_id}` at {100*float(ranked.iloc[0].matched_gain_vs_parent):+.3f}%. "
        f"No formula is promoted; system-level sign preferences are reported in `report.json`.\n"
    )
    (output / protocol["outputs"]["summary"]).write_text(summary, encoding="utf-8")
    print(json.dumps(json_safe(report), indent=2), flush=True)


if __name__ == "__main__":
    main()
