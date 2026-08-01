#!/usr/bin/env python3
"""Resolve the final P0568 low-density tidal width/coupling interaction."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_gravity_arc_tomography import shape_metrics  # noqa: E402
from run_p0567_baryon_flux_tensor_backtrack import deposit_baryons  # noqa: E402
from run_p0568_baryon_only_tensor_forward import (  # noqa: E402
    build_contexts,
    correction_map,
    cross_domain_rows,
    json_safe,
    prediction,
    tensor_map,
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    protocol_path = ROOT / "configs/p0568c_width_coupling_interaction_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_after_p0568b_boundary_hit_before_extended_interaction_scores":
        raise RuntimeError("P0568C protocol is not frozen")
    parent = json.loads((ROOT / protocol["parent_protocol"]).read_text(encoding="utf-8"))
    p0567 = json.loads((ROOT / parent["inputs"]["p0567_protocol"]).read_text(encoding="utf-8"))
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    contexts = build_contexts(parent, p0567)
    development = set(parent["validation"]["development_systems"])
    family = protocol["family"]
    widths = [float(value) for value in protocol["source_smoothing_kpc"]]
    couplings = [float(value) for value in protocol["coupling_t"]]
    spacing = float(parent["grids"]["grid_spacing_kpc"])
    rows = []
    local_rows = []
    for context in contexts:
        cohort = "development" if context.data.label in development else "holdout"
        for width in [75.0, 100.0, 125.0, 150.0, 175.0, 200.0]:
            source = deposit_baryons(context.data, width)
            metrics = shape_metrics(source, context.target, context.aperture)
            local_rows.append({"system": context.data.label, "cohort": cohort, "width": width, **metrics})
        for width in widths:
            source = deposit_baryons(context.data, width)
            tensor, derivatives = tensor_map(context, source, family, parent)
            correction = correction_map(source, tensor, derivatives, spacing)
            for coupling in couplings:
                predicted, negative = prediction(source, correction, coupling, context.aperture)
                metrics = shape_metrics(predicted, context.target, context.aperture)
                rows.append({"system": context.data.label, "cohort": cohort, "source_width_kpc": width, "coupling_t": coupling, "negative_raw_fraction": negative, **metrics})
        print(f"interaction {context.data.label}", flush=True)
    scores = pd.DataFrame(rows)
    local = pd.DataFrame(local_rows)
    scores.to_csv(output / protocol["outputs"]["scores"], index=False)
    local.to_csv(output / "local_scores.csv", index=False)
    candidates = (
        scores.groupby(["source_width_kpc", "coupling_t"])
        .apply(lambda block: pd.Series({
            "development_mean_JS": block.loc[block.cohort.eq("development"), "jensen_shannon"].mean(),
            "holdout_mean_JS": block.loc[block.cohort.eq("holdout"), "jensen_shannon"].mean(),
            "development_mean_Pearson": block.loc[block.cohort.eq("development"), "pearson"].mean(),
            "holdout_mean_Pearson": block.loc[block.cohort.eq("holdout"), "pearson"].mean(),
            "maximum_negative_raw_fraction": block.negative_raw_fraction.max(),
        }), include_groups=False)
        .reset_index()
    )
    candidates.to_csv(output / protocol["outputs"]["candidates"], index=False)
    selected = candidates.sort_values("development_mean_JS").iloc[0].copy()
    boundary = bool(float(selected.coupling_t) in {min(couplings), max(couplings)})
    # Find the local width whose ten-system JS vector most resembles each tensor candidate.
    vector_rows = []
    local_pivot = local.pivot(index="system", columns="width", values="jensen_shannon")
    for candidate in candidates.itertuples(index=False):
        vector = scores[(scores.source_width_kpc.eq(candidate.source_width_kpc)) & (scores.coupling_t.eq(candidate.coupling_t))].set_index("system").jensen_shannon
        distances = {float(width): float(np.sqrt(np.mean(np.square(vector.loc[local_pivot.index] - local_pivot[width])))) for width in local_pivot.columns}
        best_width = min(distances, key=distances.get)
        vector_rows.append({"source_width_kpc": candidate.source_width_kpc, "coupling_t": candidate.coupling_t, "closest_local_width_kpc": best_width, "system_score_vector_RMS_distance": distances[best_width]})
    degeneracy = pd.DataFrame(vector_rows)
    degeneracy.to_csv(output / "local_width_degeneracy.csv", index=False)
    selected_deg = degeneracy[(degeneracy.source_width_kpc.eq(selected.source_width_kpc)) & (degeneracy.coupling_t.eq(selected.coupling_t))].iloc[0]
    sparc = pd.read_csv(ROOT / parent["inputs"]["SPARC_points"])
    sparc = sparc[(sparc.model.eq("fixed_RAR")) & sparc.scenario.eq("invariant") & sparc.split.eq("outer_holdout")].copy()
    winner = pd.DataFrame([{"family": family, "source_width_kpc": selected.source_width_kpc, "coupling_t": selected.coupling_t}])
    cross = cross_domain_rows(parent, winner, sparc).iloc[0]
    transfer_stable = bool(float(selected.holdout_mean_JS) <= float(protocol["diagnostics"]["transfer"].split()[-1]))
    report = {
        "report_version": "P0568C-WIDTH-COUPLING-INTERACTION-RESULTS-0.1.0",
        "status": "complete_interaction_resolution",
        "protocol": {"path": str(protocol_path.relative_to(ROOT)), "sha256": sha256(protocol_path)},
        "selected": json_safe(selected.to_dict()),
        "stability": {"coupling_at_grid_boundary": boundary, "transfer_better_than_p0568_original": transfer_stable, "closest_local_width_kpc": float(selected_deg.closest_local_width_kpc), "system_score_vector_RMS_distance_to_closest_local": float(selected_deg.system_score_vector_RMS_distance)},
        "cross_domain": json_safe(cross.to_dict()),
        "verdict": {"stable_universal_tensor_lead": bool((not boundary) and transfer_stable), "SPARC_pass": bool(cross.SPARC_pass), "solar_pass": bool(cross.Cassini_pass and cross.Earth_pass and cross.Mercury_pass), "promoted": False},
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8")
    lines = [
        "# P0568C width/coupling interaction",
        "",
        f"Development selected width **{selected.source_width_kpc:.0f} kpc**, `t={selected.coupling_t:+.3f}`.",
        f"Descriptive holdout JS: **{selected.holdout_mean_JS:.5f}**.",
        f"Closest smoothing-only system-score pattern: **{selected_deg.closest_local_width_kpc:.0f} kpc**.",
        f"Stable universal tensor lead: **{report['verdict']['stable_universal_tensor_lead']}**.",
        f"SPARC passes: **{bool(cross.SPARC_pass)}**; Solar passes: **{report['verdict']['solar_pass']}**.",
    ]
    (output / protocol["outputs"]["summary"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    table = candidates.pivot(index="coupling_t", columns="source_width_kpc", values="development_mean_JS")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    image = axes[0].imshow(table.to_numpy(), origin="lower", aspect="auto", cmap="viridis")
    axes[0].set_xticks(range(len(table.columns)), [f"{v:g}" for v in table.columns])
    axes[0].set_yticks(range(len(table.index)), [f"{v:+.3f}" for v in table.index])
    axes[0].set_xlabel("width kpc"); axes[0].set_ylabel("coupling t"); axes[0].set_title("development JS")
    fig.colorbar(image, ax=axes[0])
    for width, block in candidates.groupby("source_width_kpc"):
        axes[1].plot(block.coupling_t, block.development_mean_JS, "o-", label=f"{width:g} kpc")
    axes[1].axvline(float(selected.coupling_t), color="black", ls="--")
    axes[1].set_xlabel("coupling t"); axes[1].set_ylabel("development JS"); axes[1].legend(fontsize=7)
    axes[2].axis("off")
    axes[2].text(0.02, 0.95, f"selected width={selected.source_width_kpc:.0f} kpc\nt={selected.coupling_t:+.3f}\ndev JS={selected.development_mean_JS:.5f}\nholdout JS={selected.holdout_mean_JS:.5f}\nclosest local={selected_deg.closest_local_width_kpc:.0f} kpc\nSPARC RMSE={cross.SPARC_outer_RMSE_km_s:.1f} km/s", va="top", family="monospace", fontsize=11)
    fig.suptitle("P0568C width/coupling interaction")
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)
    print(json.dumps(report["selected"], indent=2), flush=True)
    print(json.dumps(report["stability"], indent=2), flush=True)
    print(json.dumps(report["verdict"], indent=2), flush=True)


if __name__ == "__main__":
    main()
