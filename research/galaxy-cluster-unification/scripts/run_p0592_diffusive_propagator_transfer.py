#!/usr/bin/env python3
"""Transfer a universal baryon-size-scaled diffusive endpoint across clusters."""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_gravity_arc_tomography import shape_metrics  # noqa: E402
from run_p0567_baryon_flux_tensor_backtrack import deposit_baryons, lens_source_map  # noqa: E402
from run_p0568_baryon_only_tensor_forward import build_contexts  # noqa: E402


def baryon_r80(data) -> float:
    weight = np.asarray(data.weights, dtype=float)
    weight /= weight.sum()
    positions = np.asarray(data.positions, dtype=float)
    center = np.sum(weight[:, None] * positions, axis=0)
    radius = np.hypot(positions[:, 0] - center[0], positions[:, 1] - center[1])
    order = np.argsort(radius)
    return float(radius[order[min(np.searchsorted(np.cumsum(weight[order]), 0.8), len(order) - 1)]])


def normalize_inside(values, aperture):
    result = np.maximum(np.asarray(values, dtype=float), 0.0).copy()
    result[~aperture] = 0.0
    result /= np.sum(result)
    return result


def main() -> None:
    protocol_path = ROOT / "configs/p0592_diffusive_propagator_transfer_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    p0568 = json.loads((ROOT / protocol["data"]["p0568_protocol"]).read_text(encoding="utf-8"))
    p0567 = json.loads((ROOT / protocol["data"]["p0567_protocol"]).read_text(encoding="utf-8"))
    contexts = build_contexts(p0568, p0567)
    development = set(protocol["data"]["development_systems"])
    holdouts = set(protocol["data"]["holdout_systems"])
    spacing = float(protocol["preprocessing"]["grid_spacing_kpc"])
    cache = {}
    rows = []
    r80_by_system = {}
    for context in contexts:
        label = context.data.label
        cohort = "development" if label in development else "holdout"
        r80 = baryon_r80(context.data)
        r80_by_system[label] = r80
        local = normalize_inside(
            deposit_baryons(context.data, protocol["preprocessing"]["base_baryon_smoothing_kpc"]),
            context.aperture,
        )
        cache[(label, "local")] = local
        rows.append({"system": label, "cohort": cohort, "candidate_id": "local", "q_R80": 0.0, "routed_fraction": 0.0, "R80_kpc": r80, "diffusion_length_kpc": 0.0, **shape_metrics(local, context.target, context.aperture)})
        for q_value, fraction in itertools.product(protocol["factorial"]["q_R80"], protocol["factorial"]["routed_fraction"]):
            length = float(q_value) * r80
            endpoint = normalize_inside(gaussian_filter(local, length / spacing, mode="constant"), context.aperture)
            prediction = normalize_inside((1.0 - float(fraction)) * local + float(fraction) * endpoint, context.aperture)
            candidate_id = f"q{q_value:g}__f{fraction:g}"
            cache[(label, candidate_id)] = prediction
            rows.append({"system": label, "cohort": cohort, "candidate_id": candidate_id, "q_R80": q_value, "routed_fraction": fraction, "R80_kpc": r80, "diffusion_length_kpc": length, **shape_metrics(prediction, context.target, context.aperture)})
    system_scores = pd.DataFrame(rows)
    candidates = (
        system_scores.groupby(["candidate_id", "q_R80", "routed_fraction"], dropna=False)
        .apply(
            lambda block: pd.Series(
                {
                    "development_mean_jsd": block.loc[block.cohort == "development", "jensen_shannon"].mean(),
                    "holdout_mean_jsd": block.loc[block.cohort == "holdout", "jensen_shannon"].mean(),
                    "development_mean_pearson": block.loc[block.cohort == "development", "pearson"].mean(),
                    "holdout_mean_pearson": block.loc[block.cohort == "holdout", "pearson"].mean(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    best = candidates[candidates.candidate_id != "local"].sort_values(["development_mean_jsd", "q_R80", "routed_fraction"]).iloc[0]
    local_summary = candidates[candidates.candidate_id == "local"].iloc[0]
    selected_rows = system_scores[system_scores.candidate_id == best.candidate_id]
    local_rows = system_scores[system_scores.candidate_id == "local"]
    paired = selected_rows.merge(local_rows[["system", "jensen_shannon"]], on="system", suffixes=("_selected", "_local"))
    holdout_paired = paired[paired.cohort == "holdout"]
    holdout_gain = 1.0 - float(best.holdout_mean_jsd) / float(local_summary.holdout_mean_jsd)
    holdout_improved = int(np.count_nonzero(holdout_paired.jensen_shannon_selected < holdout_paired.jensen_shannon_local))

    uncertainty_rows, glafic_rows = [], []
    for context in contexts:
        label = context.data.label
        cohort = "development" if label in development else "holdout"
        chosen = cache[(label, best.candidate_id)]
        local = cache[(label, "local")]
        for realization, raw in enumerate(context.data.range_maps):
            target = lens_source_map(raw, context.data.radius, spacing, 20.0, (250.0, 300.0))
            selected_metric = shape_metrics(chosen, target, context.aperture)
            local_metric = shape_metrics(local, target, context.aperture)
            uncertainty_rows.append({"system": label, "cohort": cohort, "realization": realization, "selected_jsd": selected_metric["jensen_shannon"], "local_jsd": local_metric["jensen_shannon"], "selected_improves": selected_metric["jensen_shannon"] < local_metric["jensen_shannon"]})
        for name, prediction in (("selected", chosen), ("local", local)):
            glafic_rows.append({"system": label, "cohort": cohort, "prediction": name, **shape_metrics(prediction, context.glafic_target, context.aperture)})
    uncertainty = pd.DataFrame(uncertainty_rows)
    glafic = pd.DataFrame(glafic_rows)
    holdout_uncertainty_fraction = float(uncertainty[uncertainty.cohort == "holdout"].selected_improves.mean())
    glafic_means = glafic[glafic.cohort == "holdout"].groupby("prediction").jensen_shannon.mean()
    glafic_gain = 1.0 - float(glafic_means.selected) / float(glafic_means.local)

    peaks = pd.read_csv(ROOT / protocol["data"]["p0567_peaks"])
    peak_rows = []
    for context in contexts:
        label = context.data.label
        if label not in holdouts:
            continue
        selected_peaks = peaks[(peaks.system == label) & (peaks.method == "lenstool_ensemble")]
        positions = np.asarray(context.data.positions, dtype=float)
        weights = np.asarray(context.data.weights, dtype=float)
        length = float(best.q_R80) * r80_by_system[label]
        for peak in selected_peaks.itertuples():
            distance = np.hypot(positions[:, 0] - peak.peak_x_kpc, positions[:, 1] - peak.peak_y_kpc)
            probability = weights * np.exp(-0.5 * np.square(distance / max(length, np.finfo(float).tiny)))
            probability /= probability.sum()
            top = int(np.argmax(probability))
            peak_rows.append({"system": label, "peak_rank": peak.peak_rank, "peak_x_kpc": peak.peak_x_kpc, "peak_y_kpc": peak.peak_y_kpc, "diffusion_length_kpc": length, "top_origin_index": top, "top_origin_probability": float(probability[top]), "origin_x_kpc": float(positions[top, 0]), "origin_y_kpc": float(positions[top, 1]), "origin_distance_kpc": float(distance[top])})
    backtracks = pd.DataFrame(peak_rows)
    gates_cfg = protocol["advance_gates"]
    gates = {
        "holdout_mean_improvement_fraction": holdout_gain,
        "holdout_mean_improvement_pass": bool(holdout_gain >= gates_cfg["holdout_mean_jsd_improvement_fraction_min"]),
        "holdout_systems_improved": holdout_improved,
        "holdout_system_count_pass": bool(holdout_improved >= gates_cfg["holdout_systems_improved_min"]),
        "glafic_holdout_improvement_fraction": glafic_gain,
        "glafic_holdout_pass": bool(glafic_gain >= gates_cfg["glafic_holdout_improvement_fraction_min"]),
        "holdout_uncertainty_realizations_improved_fraction": holdout_uncertainty_fraction,
        "uncertainty_pass": bool(holdout_uncertainty_fraction >= gates_cfg["holdout_uncertainty_realizations_improved_fraction_min"]),
    }
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output / protocol["outputs"]["candidate_scores"], index=False)
    system_scores.to_csv(output / protocol["outputs"]["system_scores"], index=False)
    uncertainty.to_csv(output / protocol["outputs"]["uncertainty"], index=False)
    glafic.to_csv(output / protocol["outputs"]["glafic_scores"], index=False)
    backtracks.to_csv(output / protocol["outputs"]["backtracked_peaks"], index=False)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    holdout_plot = paired[paired.cohort == "holdout"]
    axes[0].scatter(holdout_plot.jensen_shannon_local, holdout_plot.jensen_shannon_selected)
    limit = max(holdout_plot.jensen_shannon_local.max(), holdout_plot.jensen_shannon_selected.max()) * 1.05
    axes[0].plot([0, limit], [0, limit], ls="--", color="black")
    for row in holdout_plot.itertuples():
        axes[0].annotate(row.system, (row.jensen_shannon_local, row.jensen_shannon_selected), fontsize=8)
    axes[0].set(xlabel="local member-light JSD", ylabel="diffusive JSD", title="three cluster holdouts")
    top = candidates[candidates.candidate_id != "local"].sort_values("development_mean_jsd").head(12)
    axes[1].scatter(top.q_R80, top.development_mean_jsd, c=top.routed_fraction, cmap="viridis")
    axes[1].set(xlabel="universal q = ell/R80", ylabel="development mean JSD", title="best development candidates")
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)
    report = {
        "report_version": "P0592-DIFFUSIVE-PROPAGATOR-TRANSFER-RESULTS-0.1.0",
        "status": "complete_ten_cluster_transfer",
        "coverage": {"clusters": len(contexts), "development": len(development), "holdout": len(holdouts), "candidates": int(len(candidates) - 1), "lenstool_realizations": len(uncertainty)},
        "locked_candidate": best.to_dict(),
        "local_control": local_summary.to_dict(),
        "holdout": {"mean_improvement_fraction": holdout_gain, "systems_improved": holdout_improved, "systems": 3, "glafic_improvement_fraction": glafic_gain, "uncertainty_realizations_improved_fraction": holdout_uncertainty_fraction},
        "backtracking": {"holdout_peaks": len(backtracks), "median_top_origin_probability": float(backtracks.top_origin_probability.median()) if len(backtracks) else None, "median_origin_distance_kpc": float(backtracks.origin_distance_kpc.median()) if len(backtracks) else None},
        "gates": gates,
        "all_advance_gates_pass": bool(all(value for key, value in gates.items() if key.endswith("_pass"))),
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / protocol["outputs"]["summary"]).write_text(f"# P0592 diffusive propagator transfer\n\nDevelopment selected q={best.q_R80:g}, f={best.routed_fraction:g}. Holdout mean JSD changed from {local_summary.holdout_mean_jsd:.5f} to {best.holdout_mean_jsd:.5f} ({holdout_gain:+.2%}); {holdout_improved}/3 systems improved. GLAFIC holdout change: {glafic_gain:+.2%}; Lenstool realization improvement fraction: {holdout_uncertainty_fraction:.1%}.\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
