#!/usr/bin/env python3
"""Factorial compensation test for three high-impact P0554 coordinates."""

from __future__ import annotations

import itertools
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
from run_p0554_local_cross_domain_sensitivity import A0, json_safe, rms, sha256  # noqa: E402
from run_p0554_potential_transition_remap import build_raw_contexts  # noqa: E402
from run_p0554_radial_flux_remap import cluster_metrics, galaxy_metrics, object_remainder, raw_radial_field  # noqa: E402
from run_rxj2129_raw_theory_lensing import RawLens, score as raw_score  # noqa: E402
from voidscreen.data import KPC_M  # noqa: E402
from voidscreen.radial_route import remapped_solar_diagnostics  # noqa: E402


FACTORS = ["mass_radius_delta", "extent_leak", "screen_exponent"]


def build_variants(protocol, base_spec):
    rows = []
    values = [protocol["grid"][name] for name in FACTORS]
    for mass_delta, extent_leak, screen_exponent in itertools.product(*values):
        candidate_id = f"m{int(round(100*mass_delta)):02d}_e{int(round(100*extent_leak)):02d}_n{int(round(100*screen_exponent)):03d}"
        spec = dict(base_spec)
        spec.update({"mass_radius_delta": float(mass_delta), "extent_leak": float(extent_leak), "screen_exponent": float(screen_exponent), "candidate_id": candidate_id})
        rows.append({"candidate_id": candidate_id, "mass_radius_delta": float(mass_delta), "extent_leak": float(extent_leak), "screen_exponent": float(screen_exponent), "spec": spec})
    if len(rows) != int(protocol["grid"]["candidates"]) or len({row["candidate_id"] for row in rows}) != len(rows):
        raise RuntimeError("factorial grid coverage or identifiers changed")
    return rows


def scalar_discovery(protocol, radial, variants, q):
    parent_protocol = json.loads((ROOT / radial["inputs"]["parent_protocol"]).read_text(encoding="utf-8"))
    galaxy, _ = prepare_galaxies(parent_protocol, A0); cluster, _ = prepare_clusters(parent_protocol)
    galaxy = galaxy.reset_index(drop=True); cluster = cluster.reset_index(drop=True)
    galaxy["partition"] = np.where(galaxy.galaxy.map(object_remainder).eq(0), "formula_holdout", "discovery")
    cluster["partition"] = np.where(cluster.system.map(object_remainder).eq(0), "formula_holdout", "discovery")
    outer = galaxy.split.eq("outer_holdout").to_numpy()
    rows, cache = [], {}
    for index, item in enumerate(variants, start=1):
        print(f"scalar {index}/{len(variants)} {item['candidate_id']}", flush=True)
        response_g = response_for_frame(galaxy, item["spec"], q=q, a0=A0, radius_column="radius_adjusted_kpc", gbar_column="g_bar_m_s2")
        dynamic = galaxy.g_bar_m_s2.to_numpy(float) * response_g["dynamical_enhancement"]
        velocity = np.sqrt(np.maximum(dynamic * galaxy.radius_adjusted_kpc.to_numpy(float) * KPC_M / 1.0e6, 0.0))
        response_c = response_for_frame(cluster, item["spec"], q=q, a0=A0, radius_column="radius_kpc", gbar_column="gbar_m_s2")
        lensing = cluster.gbar_m_s2.to_numpy(float) * response_c["lensing_enhancement"]
        gm = outer & galaxy.partition.eq("discovery").to_numpy(); cm = cluster.partition.eq("discovery").to_numpy()
        g_metric = galaxy_metrics(galaxy.loc[gm], velocity[gm]); c_metric = cluster_metrics(cluster.loc[cm], lensing[cm])
        solar = remapped_solar_diagnostics(response_parameters=response_parameters(item["spec"], q=q, a0=A0), route_fraction=0.0, radial_scale=1.0)
        rows.append({key: item[key] for key in ["candidate_id", *FACTORS]} | {"discovery_galaxy_RMSE": g_metric["equal_object_RMSE"], "discovery_cluster_RMSE": c_metric["equal_object_RMSE"], **solar, "all_solar_proxies_pass": bool(solar["Cassini_proxy_pass"] and solar["Earth_proxy_pass"] and solar["Mercury_proxy_pass"])})
        cache[item["candidate_id"]] = (velocity, lensing)
    return pd.DataFrame(rows), cache, galaxy, cluster, outer


def evaluate_raw_contexts(contexts, variants, q):
    rows, predictions = [], []
    for context in contexts:
        fields = {}
        for index, item in enumerate(variants, start=1):
            print(f"{context.label}: field {index}/{len(variants)} {item['candidate_id']}", flush=True)
            field, _ = raw_radial_field(item["spec"], q, context.anchors, context.local, route_fraction=0.0, radial_scale=1.0, candidate_id=item["candidate_id"])
            fields[item["candidate_id"]] = field
        lens = RawLens(context.local, fields)
        for item in variants:
            model = item["candidate_id"]
            _, sources = lens.profiled_residuals(model, context.geometry, context.training)
            held = lens.exact_predictions(model, context.geometry, sources, context.heldout, stage="heldout")
            metric = raw_score(held, lens.sigma)
            rows.append({"system_label": context.label, "candidate_id": model, **{name: item[name] for name in FACTORS}, "heldout_images": len(context.heldout), "heldout_RMS_arcsec": metric["exact_radial_RMS_arcsec"], "heldout_roots_converged": metric["converged_roots"], "heldout_all_roots": metric["all_roots_converged"]})
            held.insert(0, "system_label", context.label); held.insert(1, "candidate_id", model); predictions.append(held)
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)


def raw_aggregate(raw, labels, parent_id):
    block = raw[raw.system_label.isin(labels)]
    parent = block[block.candidate_id.eq(parent_id)].set_index("system_label")
    rows = []
    for candidate_id, local in block.groupby("candidate_id", sort=False):
        indexed = local.set_index("system_label")
        complete = [label for label in labels if bool(indexed.loc[label, "heldout_all_roots"])]
        common = [label for label in labels if label in complete and bool(parent.loc[label, "heldout_all_roots"]) and np.isfinite(indexed.loc[label, "heldout_RMS_arcsec"]) and np.isfinite(parent.loc[label, "heldout_RMS_arcsec"])]
        candidate_rms = rms(indexed.loc[complete, "heldout_RMS_arcsec"]) if complete else np.nan
        parent_rms = rms(parent.loc[common, "heldout_RMS_arcsec"]) if common else np.nan
        matched_rms = rms(indexed.loc[common, "heldout_RMS_arcsec"]) if common else np.nan
        rows.append({"candidate_id": candidate_id, "complete_systems": len(complete), "converged_roots": int(indexed.heldout_roots_converged.sum()), "heldout_images": int(indexed.heldout_images.sum()), "complete_equal_system_RMS": candidate_rms, "matched_complete_systems": len(common), "matched_labels": "+".join(common), "matched_parent_RMS": parent_rms, "matched_candidate_RMS": matched_rms, "matched_gain_vs_parent": np.nan if not common else float(1.0-matched_rms/parent_rms), "recovered_parent_incomplete": "+".join(label for label in complete if not bool(parent.loc[label, "heldout_all_roots"])), "lost_parent_complete": "+".join(label for label in parent.index if bool(parent.loc[label, "heldout_all_roots"]) and label not in complete)})
    return pd.DataFrame(rows)


def partition_scores(cache, variants, galaxy, cluster, outer, parent_id):
    rows = []
    for item in variants:
        velocity, lensing = cache[item["candidate_id"]]
        for split in ("discovery", "formula_holdout"):
            gm = outer & galaxy.partition.eq(split).to_numpy(); cm = cluster.partition.eq(split).to_numpy()
            for domain, metric in (("galaxy", galaxy_metrics(galaxy.loc[gm], velocity[gm])), ("cluster", cluster_metrics(cluster.loc[cm], lensing[cm]))):
                rows.append({"candidate_id": item["candidate_id"], **{name: item[name] for name in FACTORS}, "partition": split, "domain": domain, **metric})
    frame = pd.DataFrame(rows)
    parent = frame[frame.candidate_id.eq(parent_id)].set_index(["partition", "domain"])
    frame["parent_equal_object_RMSE"] = [parent.loc[(row.partition, row.domain), "equal_object_RMSE"] for row in frame.itertuples()]
    frame["gain_vs_parent"] = 1.0-frame.equal_object_RMSE/frame.parent_equal_object_RMSE
    return frame


def factor_impacts(scores, metrics):
    rows = []
    for metric in metrics:
        grand = float(scores[metric].mean())
        main_means = {factor: scores.groupby(factor)[metric].mean() for factor in FACTORS}
        for factor in FACTORS:
            values = main_means[factor]
            rows.append({"metric": metric, "effect_type": "main", "effect": factor, "span": float(values.max()-values.min()), "best_level": float(values.idxmax()), "worst_level": float(values.idxmin())})
        for first, second in itertools.combinations(FACTORS, 2):
            cells = scores.groupby([first, second])[metric].mean()
            residuals = []
            for (a, b), value in cells.items():
                residuals.append(float(value-main_means[first].loc[a]-main_means[second].loc[b]+grand))
            rows.append({"metric": metric, "effect_type": "pair_interaction", "effect": f"{first}*{second}", "span": float(max(residuals)-min(residuals)), "best_level": np.nan, "worst_level": np.nan})
    return pd.DataFrame(rows)


def main():
    config_path = ROOT / "configs/p0554_mass_extent_screen_factorial_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    radial = json.loads((ROOT / protocol["inputs"]["radial_parent_protocol"]).read_text(encoding="utf-8"))
    base_spec = dict(radial["parent"]); q = float(base_spec.pop("universal_q")); base_spec.pop("candidate")
    variants = build_variants(protocol, base_spec); parent_id = "m00_e00_n100"
    scalar, cache, galaxy, cluster, outer = scalar_discovery(protocol, radial, variants, q)
    contexts = build_raw_contexts(protocol)
    discovery_labels = protocol["selection"]["raw_discovery"]
    holdout_labels = protocol["transfer"]["raw_formula_holdout"]
    discovery_contexts = [context for context in contexts if context.label in discovery_labels]
    holdout_contexts = [context for context in contexts if context.label in holdout_labels]
    raw_discovery, pred_discovery = evaluate_raw_contexts(discovery_contexts, variants, q)
    raw_discovery_agg = raw_aggregate(raw_discovery, discovery_labels, parent_id).rename(columns={"matched_gain_vs_parent": "discovery_raw_gain", "complete_systems": "discovery_raw_complete_systems", "converged_roots": "discovery_raw_converged_roots", "heldout_images": "discovery_raw_heldout_images", "matched_candidate_RMS": "discovery_raw_RMS"})
    scalar = scalar.merge(raw_discovery_agg[["candidate_id", "discovery_raw_gain", "discovery_raw_complete_systems", "discovery_raw_converged_roots", "discovery_raw_heldout_images", "discovery_raw_RMS"]], on="candidate_id", validate="one_to_one")
    parent = scalar[scalar.candidate_id.eq(parent_id)].iloc[0]
    scalar["discovery_galaxy_gain"] = 1.0-scalar.discovery_galaxy_RMSE/parent.discovery_galaxy_RMSE
    scalar["discovery_cluster_gain"] = 1.0-scalar.discovery_cluster_RMSE/parent.discovery_cluster_RMSE
    scalar["joint_discovery_gain"] = scalar[["discovery_galaxy_gain", "discovery_cluster_gain", "discovery_raw_gain"]].min(axis=1)
    scalar["mean_discovery_gain"] = scalar[["discovery_galaxy_gain", "discovery_cluster_gain", "discovery_raw_gain"]].mean(axis=1)
    scalar["normalized_step"] = scalar.mass_radius_delta/0.01 + scalar.extent_leak/0.02 + (scalar.screen_exponent-1.0)/0.05
    eligible = scalar[(scalar.candidate_id != parent_id) & scalar.all_solar_proxies_pass & scalar.discovery_raw_complete_systems.eq(len(discovery_labels))].sort_values(["joint_discovery_gain", "mean_discovery_gain", "normalized_step"], ascending=[False, False, True], kind="stable")
    if eligible.empty:
        raise RuntimeError("no non-parent candidate passes Solar and discovery-root gates")
    selected = eligible.iloc[0].to_dict(); selected_id = selected["candidate_id"]
    scalar["selection_role"] = np.where(scalar.candidate_id.eq(selected_id), "frozen_transfer_candidate", np.where(scalar.candidate_id.eq(parent_id), "parent", "screened"))
    # Selection is frozen above before any formula-holdout score is computed.
    partitions = partition_scores(cache, variants, galaxy, cluster, outer, parent_id)
    raw_holdout, pred_holdout = evaluate_raw_contexts(holdout_contexts, variants, q)
    raw_holdout_agg = raw_aggregate(raw_holdout, holdout_labels, parent_id).add_prefix("holdout_").rename(columns={"holdout_candidate_id": "candidate_id"})
    scalar = scalar.merge(raw_holdout_agg, on="candidate_id", validate="one_to_one")
    raw_all = pd.concat([raw_discovery, raw_holdout], ignore_index=True)
    predictions = pd.concat([pred_discovery, pred_holdout], ignore_index=True)
    holdout_pivot = partitions[partitions.partition.eq("formula_holdout")].pivot(index="candidate_id", columns="domain", values="gain_vs_parent")
    scalar = scalar.merge(holdout_pivot.rename(columns={"galaxy": "holdout_galaxy_gain", "cluster": "holdout_cluster_gain"}), left_on="candidate_id", right_index=True, validate="one_to_one")
    impacts = factor_impacts(scalar, ["discovery_galaxy_gain", "discovery_cluster_gain", "discovery_raw_gain", "holdout_galaxy_gain", "holdout_cluster_gain", "holdout_matched_gain_vs_parent", "Mercury_precession_mas_per_century"])
    output = ROOT / protocol["outputs"]["directory"]; output.mkdir(parents=True, exist_ok=True)
    scalar.to_csv(output/protocol["outputs"]["candidate_scores"], index=False); partitions.to_csv(output/protocol["outputs"]["partition_scores"], index=False); raw_all.to_csv(output/protocol["outputs"]["raw_system_scores"], index=False); predictions.to_csv(output/protocol["outputs"]["raw_predictions"], index=False); impacts.to_csv(output/protocol["outputs"]["factor_impacts"], index=False)
    chosen = scalar[scalar.candidate_id.eq(selected_id)].iloc[0]
    chosen_part = partitions[partitions.candidate_id.eq(selected_id) & partitions.partition.eq("formula_holdout")]
    holdout_gains = {row.domain: float(row.gain_vs_parent) for row in chosen_part.itertuples()}
    selected_raw_systems = raw_holdout[raw_holdout.candidate_id.isin([parent_id, selected_id])].pivot(index="system_label", columns="candidate_id", values=["heldout_RMS_arcsec", "heldout_roots_converged"]).reset_index()
    top_effects = {metric: block.sort_values("span", ascending=False).head(4).to_dict("records") for metric, block in impacts.groupby("metric")}
    transfer_gate = bool(holdout_gains["galaxy"] > 0 and holdout_gains["cluster"] > 0 and chosen.holdout_matched_gain_vs_parent > 0 and not chosen.holdout_lost_parent_complete)
    report = {"report_version": "P0554-MASS-EXTENT-SCREEN-FACTORIAL-RESULTS-0.1.0", "status": "complete", "protocol": {"path": str(config_path.relative_to(ROOT)), "sha256": sha256(config_path)}, "coverage": {"candidates": len(scalar), "SPARC_galaxies": int(galaxy.galaxy.nunique()), "CLASH_systems": int(cluster.system.nunique()), "raw_discovery_systems": len(discovery_contexts), "raw_holdout_systems": len(holdout_contexts), "raw_heldout_images": int(raw_all[raw_all.candidate_id.eq(parent_id)].heldout_images.sum())}, "parent": scalar[scalar.candidate_id.eq(parent_id)].iloc[0].to_dict(), "selected": {"candidate_id": selected_id, **{name: float(chosen[name]) for name in FACTORS}, "discovery_gains": {"galaxy": float(chosen.discovery_galaxy_gain), "cluster": float(chosen.discovery_cluster_gain), "raw": float(chosen.discovery_raw_gain)}, "formula_holdout_gains": {"galaxy": holdout_gains["galaxy"], "cluster": holdout_gains["cluster"], "raw_matched": float(chosen.holdout_matched_gain_vs_parent)}, "raw_holdout_complete_systems": int(chosen.holdout_complete_systems), "raw_holdout_converged_roots": int(chosen.holdout_converged_roots), "raw_holdout_recovered_parent_incomplete": chosen.holdout_recovered_parent_incomplete, "raw_holdout_lost_parent_complete": chosen.holdout_lost_parent_complete, "solar_pass": bool(chosen.all_solar_proxies_pass), "Mercury_precession_mas_per_century": float(chosen.Mercury_precession_mas_per_century)}, "selected_raw_system_comparison": selected_raw_systems.to_dict("records"), "counts": {"nonparent_positive_all_discovery_domains": int(((scalar.candidate_id != parent_id) & (scalar.discovery_galaxy_gain > 0) & (scalar.discovery_cluster_gain > 0) & (scalar.discovery_raw_gain > 0)).sum()), "nonparent_positive_all_discovery_and_holdout_domains": int(((scalar.candidate_id != parent_id) & (scalar.discovery_galaxy_gain > 0) & (scalar.discovery_cluster_gain > 0) & (scalar.discovery_raw_gain > 0) & (scalar.holdout_galaxy_gain > 0) & (scalar.holdout_cluster_gain > 0) & (scalar.holdout_matched_gain_vs_parent > 0)).sum()), "solar_safe": int(scalar.all_solar_proxies_pass.sum()), "discovery_root_complete": int(scalar.discovery_raw_complete_systems.eq(len(discovery_labels)).sum())}, "top_factor_effects": top_effects, "transfer_gate_passed": transfer_gate, "verdict": {"compensated_interaction_supported": transfer_gate, "no_formula_promoted": not transfer_gate}, "claim_limits": protocol["claim_limits"]}
    (output/protocol["outputs"]["report"]).write_text(json.dumps(json_safe(report), indent=2)+"\n", encoding="utf-8")
    fig, axes = plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
    for exponent, block in scalar.groupby("screen_exponent"):
        axes[0].scatter(100*block.discovery_galaxy_gain,100*block.discovery_cluster_gain,s=20+500*block.extent_leak,alpha=.7,label=f"n={exponent:g}")
    axes[0].axhline(0,color="black",lw=.8); axes[0].axvline(0,color="black",lw=.8); axes[0].set(xlabel="galaxy discovery gain (%)",ylabel="cluster discovery gain (%)",title="Scalar compensation surface"); axes[0].legend(fontsize=7)
    axes[1].scatter(100*scalar.discovery_raw_gain,100*scalar.joint_discovery_gain,c=scalar.mass_radius_delta,cmap="viridis"); axes[1].axhline(0,color="black",lw=.8); axes[1].axvline(0,color="black",lw=.8); axes[1].set(xlabel="raw discovery gain (%)",ylabel="worst discovery gain (%)",title="Three-domain selection")
    chosen_impacts=impacts[impacts.metric.isin(["discovery_galaxy_gain","discovery_cluster_gain","discovery_raw_gain"])].sort_values("span").tail(9); axes[2].barh(chosen_impacts.metric.str.replace("discovery_","")+":"+chosen_impacts.effect,100*chosen_impacts.span); axes[2].set(xlabel="gain span (percentage points)",title="Largest factorial effects")
    fig.savefig(output/protocol["outputs"]["figure"],dpi=180); plt.close(fig)
    summary=("# P0554 mass × extent × screen factorial\n\n" f"Selected `{selected_id}`: mass delta={chosen.mass_radius_delta}, extent leak={chosen.extent_leak}, screen exponent={chosen.screen_exponent}. Discovery gains were galaxy {100*chosen.discovery_galaxy_gain:+.3f}%, cluster {100*chosen.discovery_cluster_gain:+.3f}%, raw {100*chosen.discovery_raw_gain:+.3f}%. Formula-holdout gains were galaxy {100*holdout_gains['galaxy']:+.3f}%, cluster {100*holdout_gains['cluster']:+.3f}%, raw matched {100*chosen.holdout_matched_gain_vs_parent:+.3f}%. Transfer gate: {transfer_gate}.\n")
    (output/protocol["outputs"]["summary"]).write_text(summary,encoding="utf-8")
    print(json.dumps(json_safe(report),indent=2),flush=True)


if __name__ == "__main__":
    main()
