#!/usr/bin/env python3
"""Compare R80-scaled diffusion with a selected fixed physical blur."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_gravity_arc_tomography import shape_metrics  # noqa: E402
from run_p0567_baryon_flux_tensor_backtrack import deposit_baryons  # noqa: E402
from run_p0568_baryon_only_tensor_forward import build_contexts  # noqa: E402
from run_p0592_diffusive_propagator_transfer import baryon_r80, normalize_inside  # noqa: E402


def main() -> None:
    protocol_path = ROOT / "configs/p0592b_diffusion_scale_null_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    base = json.loads((ROOT / protocol["base_protocol"]).read_text(encoding="utf-8"))
    p0568 = json.loads((ROOT / base["data"]["p0568_protocol"]).read_text(encoding="utf-8"))
    p0567 = json.loads((ROOT / base["data"]["p0567_protocol"]).read_text(encoding="utf-8"))
    contexts = build_contexts(p0568, p0567)
    development = set(base["data"]["development_systems"])
    spacing = float(base["preprocessing"]["grid_spacing_kpc"])
    adaptive = {}
    fixed = {}
    fixed_rows = []
    for context in contexts:
        label = context.data.label
        cohort = "development" if label in development else "holdout"
        local = normalize_inside(deposit_baryons(context.data, base["preprocessing"]["base_baryon_smoothing_kpc"]), context.aperture)
        length = float(protocol["disclosed_P0592_choice"]["q_R80"]) * baryon_r80(context.data)
        adaptive[label] = normalize_inside(gaussian_filter(local, length / spacing, mode="constant"), context.aperture)
        for width in protocol["fixed_width_null_kpc"]:
            prediction = normalize_inside(deposit_baryons(context.data, width), context.aperture)
            fixed[(label, float(width))] = prediction
            fixed_rows.append({"system": label, "cohort": cohort, "width_kpc": width, **shape_metrics(prediction, context.target, context.aperture)})
    fixed_systems = pd.DataFrame(fixed_rows)
    fixed_candidates = fixed_systems.groupby("width_kpc").apply(
        lambda block: pd.Series({"development_mean_jsd": block.loc[block.cohort == "development", "jensen_shannon"].mean(), "holdout_mean_jsd": block.loc[block.cohort == "holdout", "jensen_shannon"].mean()}),
        include_groups=False,
    ).reset_index()
    best_fixed = fixed_candidates.sort_values(["development_mean_jsd", "width_kpc"]).iloc[0]
    map_rows = []
    for context in contexts:
        label = context.data.label
        cohort = "development" if label in development else "holdout"
        for name, prediction in (("adaptive_R80", adaptive[label]), ("fixed_width", fixed[(label, float(best_fixed.width_kpc))])):
            map_rows.append({"system": label, "cohort": cohort, "target_method": "lenstool_ensemble", "prediction": name, **shape_metrics(prediction, context.target, context.aperture)})
            map_rows.append({"system": label, "cohort": cohort, "target_method": "glafic", "prediction": name, **shape_metrics(prediction, context.glafic_target, context.aperture)})
    scores = pd.DataFrame(map_rows)
    means = scores.groupby(["cohort", "target_method", "prediction"]).jensen_shannon.mean().unstack()
    dev_advantage = float((means.loc[("development", "lenstool_ensemble"), "fixed_width"] - means.loc[("development", "lenstool_ensemble"), "adaptive_R80"]) / means.loc[("development", "lenstool_ensemble"), "fixed_width"])
    holdout_advantage = float((means.loc[("holdout", "lenstool_ensemble"), "fixed_width"] - means.loc[("holdout", "lenstool_ensemble"), "adaptive_R80"]) / means.loc[("holdout", "lenstool_ensemble"), "fixed_width"])
    glafic_advantage = float((means.loc[("holdout", "glafic"), "fixed_width"] - means.loc[("holdout", "glafic"), "adaptive_R80"]) / means.loc[("holdout", "glafic"), "fixed_width"])
    pivot = scores[(scores.cohort == "holdout") & (scores.target_method == "lenstool_ensemble")].pivot(index="system", columns="prediction", values="jensen_shannon")
    systems_improved = int(np.count_nonzero(pivot.adaptive_R80 < pivot.fixed_width))
    gates = {"adaptive_development_improvement_fraction": dev_advantage, "adaptive_development_pass": bool(dev_advantage >= protocol["gates"]["adaptive_development_improvement_fraction_min"]), "adaptive_holdout_improvement_fraction": holdout_advantage, "adaptive_holdout_pass": bool(holdout_advantage >= protocol["gates"]["adaptive_holdout_improvement_fraction_min"]), "adaptive_holdout_systems_improved": systems_improved, "adaptive_system_count_pass": bool(systems_improved >= 2), "adaptive_glafic_holdout_improvement_fraction": glafic_advantage, "adaptive_glafic_pass": bool(glafic_advantage >= protocol["gates"]["adaptive_glafic_holdout_improvement_fraction_min"])}
    all_pass = all(value for key, value in gates.items() if key.endswith("_pass"))
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    fixed_candidates.to_csv(output / protocol["outputs"]["fixed_candidates"], index=False)
    scores.to_csv(output / protocol["outputs"]["map_scores"], index=False)
    report = {"report_version": "P0592B-DIFFUSION-SCALE-NULL-RESULTS-0.1.0", "status": "complete_fixed_width_null", "locked_adaptive": protocol["disclosed_P0592_choice"], "locked_fixed_width": best_fixed.to_dict(), "mean_jsd": {"adaptive_development_lenstool": float(means.loc[("development", "lenstool_ensemble"), "adaptive_R80"]), "fixed_development_lenstool": float(means.loc[("development", "lenstool_ensemble"), "fixed_width"]), "adaptive_holdout_lenstool": float(means.loc[("holdout", "lenstool_ensemble"), "adaptive_R80"]), "fixed_holdout_lenstool": float(means.loc[("holdout", "lenstool_ensemble"), "fixed_width"]), "adaptive_holdout_glafic": float(means.loc[("holdout", "glafic"), "adaptive_R80"]), "fixed_holdout_glafic": float(means.loc[("holdout", "glafic"), "fixed_width"])}, "gates": gates, "conclusion": "adaptive_diffusion_scale_beats_fixed_blur" if all_pass else "diffusion_broadening_not_specific_to_R80_scale", "claim_limits": protocol["claim_limits"]}
    (output / protocol["outputs"]["report"]).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / protocol["outputs"]["summary"]).write_text(f"# P0592B diffusion-scale null\n\nDevelopment chose a fixed width of {best_fixed.width_kpc:g} kpc. Adaptive minus fixed relative JSD changes: {dev_advantage:+.2%} development, {holdout_advantage:+.2%} Lenstool holdout, {glafic_advantage:+.2%} GLAFIC holdout. Adaptive improved {systems_improved}/3 holdout systems. Conclusion: {report['conclusion']}.\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
