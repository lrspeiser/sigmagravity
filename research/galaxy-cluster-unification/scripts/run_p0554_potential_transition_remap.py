#!/usr/bin/env python3
"""Test a one-parameter potential-driven change in radial route direction."""

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

from run_arc_invariant_absolute_lensing import prepare_clusters, prepare_galaxies, response_for_frame, response_parameters  # noqa: E402
from run_p0554_baryonic_network_rxj1347_holdout import pair_split  # noqa: E402
from run_p0554_local_cross_domain_sensitivity import A0, RawContext, json_safe, raw_contexts, rms, sha256  # noqa: E402
from run_p0554_radial_flux_remap import cluster_metrics, galaxy_metrics, object_remainder, raw_radial_field, remap_profiles  # noqa: E402
from run_rxj2129_raw_theory_lensing import FIXED_LABELS, RawLens, score as raw_score  # noqa: E402
from run_unbounded_running_multicluster_raw import load_anchors, load_system_images, system_protocol  # noqa: E402
from voidscreen.data import KPC_M  # noqa: E402
from voidscreen.radial_route import potential_transition_scale, remapped_solar_diagnostics  # noqa: E402


def variant_id(amplitude):
    if amplitude == 0.0:
        return "parent_A0000"
    sign = "p" if amplitude > 0 else "m"
    return f"A{sign}{int(round(abs(amplitude) * 1000)):04d}"


def variants(protocol):
    return [{"candidate_id": variant_id(float(value)), "amplitude": float(value)} for value in protocol["amplitude_grid"]]


def scalar_screen(protocol, radial, spec, q):
    parent_protocol = json.loads((ROOT / radial["inputs"]["parent_protocol"]).read_text(encoding="utf-8"))
    galaxy, _ = prepare_galaxies(parent_protocol, A0)
    cluster, _ = prepare_clusters(parent_protocol)
    galaxy = galaxy.reset_index(drop=True); cluster = cluster.reset_index(drop=True)
    galaxy["partition"] = np.where(galaxy.galaxy.map(object_remainder).eq(0), "formula_holdout", "discovery")
    cluster["partition"] = np.where(cluster.system.map(object_remainder).eq(0), "formula_holdout", "discovery")
    galaxy_response = response_for_frame(galaxy, spec, q=q, a0=A0, radius_column="radius_adjusted_kpc", gbar_column="g_bar_m_s2")
    cluster_response = response_for_frame(cluster, spec, q=q, a0=A0, radius_column="radius_kpc", gbar_column="gbar_m_s2")
    outer = galaxy.split.eq("outer_holdout").to_numpy()
    score_rows, partition_rows = [], []
    for item in variants(protocol):
        amplitude = item["amplitude"]
        galaxy_scale = potential_transition_scale(galaxy.potential_depth.to_numpy(float), log_scale_amplitude=amplitude, pivot=protocol["formula"]["fixed_pivot"], sharpness=protocol["formula"]["fixed_sharpness"])
        cluster_scale = potential_transition_scale(cluster.potential_depth.to_numpy(float), log_scale_amplitude=amplitude, pivot=protocol["formula"]["fixed_pivot"], sharpness=protocol["formula"]["fixed_sharpness"])
        gdyn, _ = remap_profiles(galaxy, galaxy_response, system_column="galaxy", radius_column="radius_adjusted_kpc", gbar_column="g_bar_m_s2", route_fraction=1.0, radial_scale=galaxy_scale)
        _, clens = remap_profiles(cluster, cluster_response, system_column="system", radius_column="radius_kpc", gbar_column="gbar_m_s2", route_fraction=1.0, radial_scale=cluster_scale)
        velocity = np.sqrt(np.maximum(gdyn * galaxy.radius_adjusted_kpc.to_numpy(float) * KPC_M / 1.0e6, 0.0))
        solar = remapped_solar_diagnostics(response_parameters=response_parameters(spec, q=q, a0=A0), route_fraction=1.0, potential_log_scale_amplitude=amplitude, potential_pivot=protocol["formula"]["fixed_pivot"], potential_sharpness=protocol["formula"]["fixed_sharpness"])
        score_rows.append({**item, **solar, "all_solar_proxies_pass": bool(solar["Cassini_proxy_pass"] and solar["Earth_proxy_pass"] and solar["Mercury_proxy_pass"])})
        for split in ("discovery", "formula_holdout"):
            gm = outer & galaxy.partition.eq(split).to_numpy(); cm = cluster.partition.eq(split).to_numpy()
            for domain, metric in (("galaxy", galaxy_metrics(galaxy.loc[gm], velocity[gm])), ("cluster", cluster_metrics(cluster.loc[cm], clens[cm]))):
                partition_rows.append({"candidate_id": item["candidate_id"], "amplitude": amplitude, "partition": split, "domain": domain, **metric})
    scores = pd.DataFrame(score_rows); partitions = pd.DataFrame(partition_rows)
    parent = partitions[partitions.candidate_id.eq("parent_A0000")].set_index(["partition", "domain"])
    partitions["parent_equal_object_RMSE"] = [parent.loc[(row.partition, row.domain), "equal_object_RMSE"] for row in partitions.itertuples()]
    partitions["gain_vs_parent"] = 1.0 - partitions.equal_object_RMSE / partitions.parent_equal_object_RMSE
    discovery = partitions[partitions.partition.eq("discovery")].pivot(index="candidate_id", columns="domain", values="gain_vs_parent")
    scores = scores.merge(discovery.rename(columns={"galaxy": "discovery_galaxy_gain", "cluster": "discovery_cluster_gain"}), left_on="candidate_id", right_index=True, validate="one_to_one")
    scores["joint_discovery_gain"] = scores[["discovery_galaxy_gain", "discovery_cluster_gain"]].min(axis=1)
    scores["mean_discovery_gain"] = scores[["discovery_galaxy_gain", "discovery_cluster_gain"]].mean(axis=1)
    eligible = scores[scores.all_solar_proxies_pass & scores.candidate_id.ne("parent_A0000")].copy()
    eligible["absolute_amplitude"] = eligible.amplitude.abs()
    selected = eligible.sort_values(["joint_discovery_gain", "mean_discovery_gain", "absolute_amplitude"], ascending=[False, False, True], kind="stable").iloc[0].to_dict()
    scores["selection_role"] = np.where(scores.candidate_id.eq(selected["candidate_id"]), "frozen_transfer_candidate", np.where(scores.candidate_id.eq("parent_A0000"), "parent", "screened"))
    return scores, partitions, selected


def build_raw_contexts(protocol):
    context_protocol = json.loads((ROOT / protocol["inputs"]["context_protocol"]).read_text(encoding="utf-8"))
    contexts = raw_contexts(context_protocol)
    geometry = pd.read_csv(ROOT / protocol["inputs"]["fixed_geometry"])
    for context in contexts:
        row = geometry[(geometry.system_label.eq(context.label)) & (geometry.variant_id.eq("lensing_softness_098"))].iloc[0]
        context.geometry = row[list(FIXED_LABELS)].to_numpy(float)
    raw_protocol = json.loads((ROOT / protocol["inputs"]["raw_protocol"]).read_text(encoding="utf-8"))
    system = next(item for item in raw_protocol["systems"] if item["label"] == "RXJ1347")
    local = system_protocol(raw_protocol, system)
    catalog = pd.read_csv(ROOT / protocol["inputs"]["image_catalog"])
    images = load_system_images(catalog, system)
    training, heldout = pair_split(images)
    tian = pd.read_csv(ROOT / protocol["inputs"]["baryonic_profile"], sep=r"\s+", names=["system", "radius_kpc", "log_gbar", "log_gobs", "err_log_gbar", "err_log_gobs"])
    geometry_rx = pd.read_csv(ROOT / protocol["inputs"]["RXJ1347_geometry"]).iloc[0][list(FIXED_LABELS)].to_numpy(float)
    contexts.append(RawContext("RXJ1347", system["system"], "RXJ1347", local, training, heldout, load_anchors(tian, "RXJ1347"), geometry_rx))
    return contexts


def raw_screen(protocol, spec, q, items):
    rows, predictions, profiles = [], [], []
    for context in build_raw_contexts(protocol):
        fields = {}
        for item in items:
            print(f"{context.label}: {item['candidate_id']}", flush=True)
            field, profile = raw_radial_field(spec, q, context.anchors, context.local, route_fraction=1.0, radial_scale=1.0, candidate_id=item["candidate_id"], potential_log_scale_amplitude=item["amplitude"], potential_pivot=protocol["formula"]["fixed_pivot"], potential_sharpness=protocol["formula"]["fixed_sharpness"])
            fields[item["candidate_id"]] = field
            profile.insert(0, "system_label", context.label); profiles.append(profile)
        lens = RawLens(context.local, fields)
        for item in items:
            model = item["candidate_id"]
            _, sources = lens.profiled_residuals(model, context.geometry, context.training)
            held = lens.exact_predictions(model, context.geometry, sources, context.heldout, stage="heldout")
            metric = raw_score(held, lens.sigma)
            rows.append({"system_label": context.label, **item, "heldout_images": len(context.heldout), "heldout_RMS_arcsec": metric["exact_radial_RMS_arcsec"], "heldout_roots_converged": metric["converged_roots"], "heldout_all_roots": metric["all_roots_converged"]})
            held.insert(0, "system_label", context.label); held.insert(1, "candidate_id", model); predictions.append(held)
    system_scores = pd.DataFrame(rows)
    parent = system_scores[system_scores.candidate_id.eq("parent_A0000")].set_index("system_label")
    aggregate = []
    for item in items:
        block = system_scores[system_scores.candidate_id.eq(item["candidate_id"])].set_index("system_label")
        common = [label for label in parent.index if bool(parent.loc[label, "heldout_all_roots"]) and bool(block.loc[label, "heldout_all_roots"]) and np.isfinite(parent.loc[label, "heldout_RMS_arcsec"]) and np.isfinite(block.loc[label, "heldout_RMS_arcsec"])]
        pr = rms(parent.loc[common, "heldout_RMS_arcsec"]) if common else np.nan; cr = rms(block.loc[common, "heldout_RMS_arcsec"]) if common else np.nan
        aggregate.append({**item, "complete_systems": int(block.heldout_all_roots.astype(bool).sum()), "converged_roots": int(block.heldout_roots_converged.sum()), "heldout_images": int(block.heldout_images.sum()), "matched_complete_systems": len(common), "matched_labels": "+".join(common), "matched_parent_RMS_arcsec": pr, "matched_candidate_RMS_arcsec": cr, "matched_gain_vs_parent": None if not common else float(1.0-cr/pr)})
    return system_scores, pd.DataFrame(aggregate), pd.concat(predictions, ignore_index=True), pd.concat(profiles, ignore_index=True)


def main():
    config_path = ROOT / "configs/p0554_potential_transition_remap_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    radial = json.loads((ROOT / protocol["inputs"]["radial_protocol"]).read_text(encoding="utf-8"))
    spec = dict(radial["parent"]); q = float(spec.pop("universal_q")); spec.pop("candidate"); spec["candidate_id"] = "P0554_potential_transition"
    scores, partitions, selected = scalar_screen(protocol, radial, spec, q)
    items = variants(protocol)
    raw_system, raw_candidates, raw_predictions, raw_profiles = raw_screen(protocol, spec, q, items)
    raw_candidates = raw_candidates.merge(scores[["candidate_id", "discovery_galaxy_gain", "discovery_cluster_gain"]], on="candidate_id", validate="one_to_one")
    output = ROOT / protocol["outputs"]["directory"]; output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / protocol["outputs"]["scalar_scores"], index=False); partitions.to_csv(output / protocol["outputs"]["partition_scores"], index=False); raw_system.to_csv(output / protocol["outputs"]["raw_system_scores"], index=False); raw_candidates.to_csv(output / protocol["outputs"]["raw_candidate_scores"], index=False); raw_predictions.to_csv(output / protocol["outputs"]["raw_predictions"], index=False); raw_profiles.to_csv(output / protocol["outputs"]["raw_profiles"], index=False)
    selected_id = selected["candidate_id"]
    selected_part = partitions[partitions.candidate_id.eq(selected_id)]
    holdout = {row.domain: float(row.gain_vs_parent) for row in selected_part[selected_part.partition.eq("formula_holdout")].itertuples()}
    raw_selected = raw_candidates[raw_candidates.candidate_id.eq(selected_id)].iloc[0]
    per_system = raw_system[raw_system.candidate_id.isin(["parent_A0000", selected_id])].pivot(index="system_label", columns="candidate_id", values="heldout_RMS_arcsec")
    transfer = bool(holdout["galaxy"] > 0 and holdout["cluster"] > 0 and raw_selected.matched_gain_vs_parent > 0 and raw_selected.complete_systems >= raw_candidates[raw_candidates.candidate_id.eq("parent_A0000")].complete_systems.iloc[0])
    report = {"report_version": "P0554-POTENTIAL-TRANSITION-REMAP-RESULTS-0.1.0", "status": "complete", "protocol": {"path": str(config_path.relative_to(ROOT)), "sha256": sha256(config_path)}, "coverage": {"amplitudes": len(scores), "raw_systems": int(raw_system.system_label.nunique()), "raw_heldout_images": int(raw_system[raw_system.candidate_id.eq("parent_A0000")].heldout_images.sum())}, "selected": {"candidate_id": selected_id, "amplitude": float(selected["amplitude"]), "discovery_galaxy_gain": float(selected["discovery_galaxy_gain"]), "discovery_cluster_gain": float(selected["discovery_cluster_gain"]), "formula_holdout_gains": holdout, "raw_matched_gain": float(raw_selected.matched_gain_vs_parent), "raw_complete_systems": int(raw_selected.complete_systems), "solar_pass": bool(selected["all_solar_proxies_pass"]), "Mercury_precession_mas_per_century": float(selected["Mercury_precession_mas_per_century"])}, "selected_raw_system_RMS": per_system.reset_index().to_dict("records"), "best_raw_candidate": raw_candidates.sort_values("matched_candidate_RMS_arcsec").iloc[0].to_dict(), "counts": {"nonparent_discovery_joint_improvers": int(((scores.candidate_id != "parent_A0000") & (scores.discovery_galaxy_gain > 0) & (scores.discovery_cluster_gain > 0)).sum()), "nonparent_discovery_and_raw_improvers": int(((raw_candidates.candidate_id != "parent_A0000") & (raw_candidates.discovery_galaxy_gain > 0) & (raw_candidates.discovery_cluster_gain > 0) & (raw_candidates.matched_gain_vs_parent > 0)).sum())}, "transfer_gate_passed": transfer, "verdict": {"one_parameter_potential_transition_promoted": transfer, "no_formula_promoted": not transfer}, "claim_limits": protocol["claim_limits"]}
    (output / protocol["outputs"]["report"]).write_text(json.dumps(json_safe(report), indent=2)+"\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes[0].plot(scores.amplitude, 100*scores.discovery_galaxy_gain, "o-", label="galaxy"); axes[0].plot(scores.amplitude, 100*scores.discovery_cluster_gain, "o-", label="derived cluster"); axes[0].axhline(0,color="black",ls="--"); axes[0].legend(); axes[0].set(xlabel="A",ylabel="discovery gain (%)",title="Potential transition")
    for label, block in raw_system.groupby("system_label"):
        parent_rms=block[block.candidate_id.eq("parent_A0000")].heldout_RMS_arcsec.iloc[0]
        if not np.isfinite(parent_rms) or parent_rms <= 0.0:
            continue
        axes[1].plot(block.amplitude, block.heldout_RMS_arcsec/parent_rms, "o-", ms=3, alpha=.6, label=label)
    axes[1].axhline(1,color="black",ls="--"); axes[1].set_yscale("log"); axes[1].set(xlabel="A",ylabel="raw RMS / parent (log scale)",title="Five parent-complete raw clusters"); axes[1].legend(fontsize=6,ncol=2)
    axes[2].plot(raw_candidates.amplitude,100*raw_candidates.matched_gain_vs_parent,"o-"); axes[2].axhline(0,color="black",ls="--"); axes[2].set(xlabel="A",ylabel="matched raw gain (%)",title="Raw aggregate")
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180); plt.close(fig)
    summary=("# Potential-transition radial remap\n\n" f"The discovery rule selected `{selected_id}` (A={selected['amplitude']}). Discovery gains: galaxy {100*selected['discovery_galaxy_gain']:+.3f}%, cluster {100*selected['discovery_cluster_gain']:+.3f}%. Formula-holdout gains: galaxy {100*holdout['galaxy']:+.3f}%, cluster {100*holdout['cluster']:+.3f}%. Six-system raw gain: {100*raw_selected.matched_gain_vs_parent:+.3f}%. Transfer gate: {transfer}.\n")
    (output / protocol["outputs"]["summary"]).write_text(summary, encoding="utf-8")
    print(json.dumps(json_safe(report), indent=2), flush=True)


if __name__ == "__main__":
    main()
