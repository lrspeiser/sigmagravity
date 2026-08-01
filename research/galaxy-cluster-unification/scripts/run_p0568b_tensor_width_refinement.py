#!/usr/bin/env python3
"""Resolve the P0568 source-width boundary before interpreting its tensor lead."""

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_figure(candidates, selected, local, output):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), constrained_layout=True)
    families = ["tidal_low_density", "isotropic_low_density"]
    for axis, family in zip(axes[:2], families):
        block = candidates[candidates.family.eq(family)]
        table = block.pivot(index="coupling_t", columns="source_width_kpc", values="development_mean_JS")
        image = axis.imshow(table.to_numpy(), origin="lower", aspect="auto", cmap="viridis")
        axis.set_xticks(range(len(table.columns)), [f"{value:g}" for value in table.columns])
        axis.set_yticks(range(len(table.index)), [f"{value:+.3f}" for value in table.index])
        axis.set_xlabel("smoothing width (kpc)")
        axis.set_ylabel("coupling t")
        axis.set_title(f"{family}\ndevelopment JS")
        fig.colorbar(image, ax=axis, shrink=0.8)
    axes[2].plot(local.source_width_kpc, local.development_mean_JS, "o-", label="development")
    axes[2].plot(local.source_width_kpc, local.holdout_mean_JS, "o-", label="descriptive holdout")
    axes[2].axvline(float(selected.source_width_kpc), color="black", ls="--")
    axes[2].set_xlabel("local smoothing width (kpc)")
    axes[2].set_ylabel("mean JS")
    axes[2].set_title("smoothing-only response")
    axes[2].legend(fontsize=8)
    text = (
        f"Selected {selected.family}\n"
        f"width={selected.source_width_kpc:.0f} kpc\n"
        f"t={selected.coupling_t:+.3f}\n"
        f"development JS={selected.development_mean_JS:.5f}\n"
        f"holdout JS={selected.holdout_mean_JS:.5f}\n"
        f"gain vs refined local={100*selected.improvement_vs_local_development:.2f}%"
    )
    axes[3].axis("off")
    axes[3].text(0.02, 0.95, text, va="top", family="monospace", fontsize=11)
    fig.suptitle("P0568B tensor/width refinement")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    protocol_path = ROOT / "configs/p0568b_tensor_width_refinement_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_after_p0568_before_extended_width_or_refined_coupling_scores":
        raise RuntimeError("P0568B protocol is not frozen")
    parent_protocol_path = ROOT / protocol["parent"]["protocol"]
    parent = json.loads(parent_protocol_path.read_text(encoding="utf-8"))
    p0567 = json.loads((ROOT / parent["inputs"]["p0567_protocol"]).read_text(encoding="utf-8"))
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    contexts = build_contexts(parent, p0567)
    development = set(parent["validation"]["development_systems"])
    families = protocol["frozen_refinement"]["families"]
    widths = [float(value) for value in protocol["frozen_refinement"]["source_smoothing_kpc"]]
    couplings = [float(value) for value in protocol["frozen_refinement"]["coupling_t"]]
    spacing = float(parent["grids"]["grid_spacing_kpc"])
    rows = []
    prediction_cache = {}
    for context in contexts:
        label = context.data.label
        cohort = "development" if label in development else "holdout"
        for width in widths:
            source = deposit_baryons(context.data, width)
            local_id = f"local_w{int(width):03d}"
            metrics = shape_metrics(source, context.target, context.aperture)
            rows.append({"system": label, "cohort": cohort, "candidate_id": local_id, "family": "local_identity", "source_width_kpc": width, "coupling_t": 0.0, **metrics})
            prediction_cache[(label, local_id)] = source
            for family in families:
                tensor, derivatives = tensor_map(context, source, family, parent)
                correction = correction_map(source, tensor, derivatives, spacing)
                for coupling in couplings:
                    candidate_id = f"{family}_w{int(width):03d}_t{coupling:+.3f}"
                    predicted, _ = prediction(source, correction, coupling, context.aperture)
                    metrics = shape_metrics(predicted, context.target, context.aperture)
                    rows.append({"system": label, "cohort": cohort, "candidate_id": candidate_id, "family": family, "source_width_kpc": width, "coupling_t": coupling, **metrics})
                    prediction_cache[(label, candidate_id)] = predicted
        print(f"refined {label}", flush=True)
    scores = pd.DataFrame(rows)
    scores.to_csv(output / protocol["outputs"]["scores"], index=False)
    candidates = (
        scores.groupby(["candidate_id", "family", "source_width_kpc", "coupling_t"])
        .apply(
            lambda block: pd.Series(
                {
                    "development_mean_JS": block.loc[block.cohort.eq("development"), "jensen_shannon"].mean(),
                    "holdout_mean_JS": block.loc[block.cohort.eq("holdout"), "jensen_shannon"].mean(),
                    "development_mean_Pearson": block.loc[block.cohort.eq("development"), "pearson"].mean(),
                    "holdout_mean_Pearson": block.loc[block.cohort.eq("holdout"), "pearson"].mean(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    candidates.to_csv(output / protocol["outputs"]["candidates"], index=False)
    local = candidates[candidates.family.eq("local_identity")].sort_values("source_width_kpc")
    best_local = local.sort_values("development_mean_JS").iloc[0]
    tensors = candidates[candidates.family.isin(families)]
    selected = tensors.sort_values("development_mean_JS").iloc[0].copy()
    selected["improvement_vs_local_development"] = 1.0 - float(selected.development_mean_JS) / float(best_local.development_mean_JS)
    selected["improvement_vs_local_holdout"] = 1.0 - float(selected.holdout_mean_JS) / float(best_local.holdout_mean_JS)
    glafic_rows = []
    for context in contexts:
        cohort = "development" if context.data.label in development else "holdout"
        for role, row in [("selected_tensor", selected), ("refined_local", best_local)]:
            predicted = prediction_cache[(context.data.label, row.candidate_id)]
            metrics = shape_metrics(predicted, context.glafic_target, context.aperture)
            glafic_rows.append({"system": context.data.label, "cohort": cohort, "role": role, **metrics})
    glafic = pd.DataFrame(glafic_rows)
    glafic.to_csv(output / "glafic_scores.csv", index=False)
    sparc = pd.read_csv(ROOT / parent["inputs"]["SPARC_points"])
    sparc = sparc[(sparc.model.eq("fixed_RAR")) & sparc.scenario.eq("invariant") & sparc.split.eq("outer_holdout")].copy()
    winner_frame = pd.DataFrame([selected.to_dict()])
    cross = cross_domain_rows(parent, winner_frame, sparc).iloc[0]
    smoothing_span = float(local.development_mean_JS.max() - local.development_mean_JS.min())
    best_family_block = tensors[tensors.family.eq(selected.family)]
    amplitude_profile = best_family_block.groupby("coupling_t").development_mean_JS.min()
    coupling_span = float(amplitude_profile.max() - amplitude_profile.min())
    smoothing_dominates = bool(selected.improvement_vs_local_development < 0.01 or abs(float(selected.coupling_t)) < 1e-12)
    report = {
        "report_version": "P0568B-TENSOR-WIDTH-REFINEMENT-RESULTS-0.1.0",
        "status": "complete_boundary_refinement",
        "protocol": {"path": str(protocol_path.relative_to(ROOT)), "sha256": sha256(protocol_path)},
        "coverage": {"clusters": len(contexts), "candidates": len(candidates), "system_scores": len(scores)},
        "selected": json_safe(selected.to_dict()),
        "refined_local": json_safe(best_local.to_dict()),
        "parameter_impacts": {"local_width_development_JS_span": smoothing_span, "selected_family_coupling_development_JS_span": coupling_span, "smoothing_dominates_by_frozen_rule": smoothing_dominates},
        "glafic_mean_JS": json_safe(glafic.groupby(["cohort", "role"]).jensen_shannon.mean().unstack().to_dict()),
        "cross_domain": json_safe(cross.to_dict()),
        "verdict": {"tensor_lead_survives_one_percent_gate": bool(selected.improvement_vs_local_development >= 0.01), "SPARC_pass": bool(cross.SPARC_pass), "solar_pass": bool(cross.Cassini_pass and cross.Earth_pass and cross.Mercury_pass), "promoted": False},
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8")
    lines = [
        "# P0568B tensor/width refinement",
        "",
        f"Selected `{selected.family}` at {selected.source_width_kpc:.0f} kpc and `t={selected.coupling_t:+.3f}`.",
        f"Development improvement versus the refined local-width null: **{100*selected.improvement_vs_local_development:.3f}%**.",
        f"Descriptive transfer improvement: **{100*selected.improvement_vs_local_holdout:.3f}%**.",
        f"Smoothing dominates by the frozen rule: **{smoothing_dominates}**.",
        f"SPARC passes: **{bool(cross.SPARC_pass)}**; Solar passes: **{bool(cross.Cassini_pass and cross.Earth_pass and cross.Mercury_pass)}**.",
    ]
    (output / protocol["outputs"]["summary"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    make_figure(candidates, selected, local, output / protocol["outputs"]["figure"])
    print(json.dumps(report["selected"], indent=2), flush=True)
    print(json.dumps(report["parameter_impacts"], indent=2), flush=True)
    print(json.dumps(report["verdict"], indent=2), flush=True)


if __name__ == "__main__":
    main()
