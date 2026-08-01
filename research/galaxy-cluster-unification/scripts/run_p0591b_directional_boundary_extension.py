#!/usr/bin/env python3
"""Extend the two active P0591 boundaries and compare with smoothing."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0590_gravity_return_backtrack import coordinate_grid, jensen_shannon_divergence, load_reprojected_map, source_surface  # noqa: E402
from run_p0591_directional_return import baryonic_major_axis, normalized_prediction  # noqa: E402
from voidscreen.gravity_return import normalized_directional_ring_kernel, normalized_ring_kernel  # noqa: E402


def main() -> None:
    path = ROOT / "configs/p0591b_directional_boundary_extension_protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    base = json.loads((ROOT / protocol["base_protocol"]).read_text(encoding="utf-8"))
    sources = pd.read_csv(ROOT / base["source"]["nominal_baryons"])
    angle, _, _ = baryonic_major_axis(sources)
    axis, xx, yy, mask = coordinate_grid(base)
    local = source_surface(sources, axis, base["source"]["source_smoothing_arcsec"])
    local[~mask] = 0.0
    local /= local.sum()
    dev_models = [m for m in base["target_maps"] if m["role"] == "development"]
    holdout_models = [m for m in base["target_maps"] if m["role"] == "method_holdout"]
    development = {m["model_id"]: load_reprojected_map(m, base, xx, yy, mask) for m in dev_models}
    rows, cache = [], {}
    for radius, width_fraction, concentration, fraction in itertools.product(
        protocol["factorial"]["return_radius_arcsec"], protocol["factorial"]["width_fraction"], protocol["factorial"]["directional_concentration"], protocol["factorial"]["routed_fraction"]
    ):
        width = radius * width_fraction
        kernel = normalized_directional_ring_kernel(axis, return_radius_arcsec=radius, width_arcsec=width, major_axis_deg=angle, directional_concentration=concentration)
        prediction, _ = normalized_prediction(local, kernel, fraction, mask)
        key = (radius, width_fraction, concentration, fraction)
        cache[key] = prediction
        scores = [jensen_shannon_divergence(prediction, target, mask) for target in development.values()]
        rows.append({"return_radius_arcsec": radius, "width_fraction": width_fraction, "width_arcsec": width, "directional_concentration": concentration, "routed_fraction": fraction, "development_mean_jsd": float(np.mean(scores))})
    candidates = pd.DataFrame(rows).sort_values(["development_mean_jsd", "return_radius_arcsec", "width_fraction", "directional_concentration", "routed_fraction"])
    best = candidates.iloc[0]
    best_prediction = cache[(best.return_radius_arcsec, best.width_fraction, best.directional_concentration, best.routed_fraction)]
    control = protocol["gaussian_control"]
    gaussian_kernel = normalized_ring_kernel(axis, return_radius_arcsec=control["return_radius_arcsec"], width_arcsec=control["width_arcsec"])
    gaussian_prediction, _ = normalized_prediction(local, gaussian_kernel, control["routed_fraction"], mask)
    holdouts = {m["model_id"]: load_reprojected_map(m, base, xx, yy, mask) for m in holdout_models}
    targets = {**development, **holdouts}
    roles = {m["model_id"]: m["role"] for m in base["target_maps"]}
    rows = []
    for model_id, target in targets.items():
        for name, prediction in (("directional_extension", best_prediction), ("gaussian_smoothing", gaussian_prediction)):
            rows.append({"model_id": model_id, "role": roles[model_id], "prediction": name, "jsd": jensen_shannon_divergence(prediction, target, mask)})
    scores = pd.DataFrame(rows)
    means = scores.groupby(["role", "prediction"]).jsd.mean().unstack()
    development_advantage = float((means.loc["development", "gaussian_smoothing"] - means.loc["development", "directional_extension"]) / means.loc["development", "gaussian_smoothing"])
    holdout_advantage = float((means.loc["method_holdout", "gaussian_smoothing"] - means.loc["method_holdout", "directional_extension"]) / means.loc["method_holdout", "gaussian_smoothing"])
    pivot = scores.pivot(index="model_id", columns="prediction", values="jsd")
    holdout_ids = list(holdouts)
    both = bool(np.all(pivot.loc[holdout_ids, "directional_extension"] < pivot.loc[holdout_ids, "gaussian_smoothing"]))
    gates = {"development_improvement_over_gaussian_fraction": development_advantage, "development_pass": bool(development_advantage >= protocol["gates"]["development_improvement_over_gaussian_fraction_min"]), "holdout_improvement_over_gaussian_fraction": holdout_advantage, "holdout_pass": bool(holdout_advantage >= protocol["gates"]["holdout_improvement_over_gaussian_fraction_min"]), "both_holdouts_better_than_gaussian": both}
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output / protocol["outputs"]["candidate_scores"], index=False)
    scores.to_csv(output / protocol["outputs"]["map_scores"], index=False)
    passed = gates["development_pass"] and gates["holdout_pass"] and both
    report = {"report_version": "P0591B-DIRECTIONAL-BOUNDARY-RESULTS-0.1.0", "status": "complete_reused_map_boundary_diagnostic", "baryonic_major_axis_deg": angle, "locked_candidate": best.to_dict(), "mean_jsd": means.to_dict(), "gates": gates, "conclusion": "boundary_extension_beats_smoothing" if passed else "global_directional_return_family_closed_against_smoothing_on_MACS0416", "claim_limits": protocol["claim_limits"]}
    (output / protocol["outputs"]["report"]).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / protocol["outputs"]["summary"]).write_text(f"# P0591B directional boundary extension\n\nBest extended candidate: L={best.return_radius_arcsec:g} arcsec, w={best.width_arcsec:g} arcsec, kappa={best.directional_concentration:g}, f={best.routed_fraction:g}. Relative to Gaussian smoothing: {development_advantage:+.2%} development and {holdout_advantage:+.2%} method holdout. Conclusion: {report['conclusion']}.\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
