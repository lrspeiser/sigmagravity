#!/usr/bin/env python3
"""Determine whether P0590 needs a displaced return ring or only smoothing."""

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

from run_p0590_gravity_return_backtrack import (  # noqa: E402
    coordinate_grid,
    jensen_shannon_divergence,
    load_reprojected_map,
    source_surface,
)
from voidscreen.gravity_return import normalized_ring_kernel, routed_arrival_map  # noqa: E402


def choose_family(candidates: pd.DataFrame, family: str) -> pd.Series:
    if family == "general_return":
        subset = candidates
    elif family == "gaussian_smoothing_null":
        subset = candidates[candidates.return_radius_arcsec == 0.0]
    elif family == "strict_arc_return":
        subset = candidates[
            (candidates.return_radius_arcsec >= 8.0)
            & (candidates.width_arcsec <= 0.5 * candidates.return_radius_arcsec)
        ]
    else:
        raise ValueError(family)
    return subset.sort_values(
        ["development_mean_jsd", "return_radius_arcsec", "width_arcsec", "routed_fraction"]
    ).iloc[0]


def main() -> None:
    audit_path = ROOT / "configs/p0590b_ring_necessity_audit_protocol.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    base = json.loads((ROOT / audit["base_protocol"]).read_text(encoding="utf-8"))
    sources = pd.read_csv(ROOT / base["source"]["nominal_baryons"])
    axis, xx, yy, mask = coordinate_grid(base)
    local = source_surface(sources, axis, base["source"]["source_smoothing_arcsec"])
    local[~mask] = 0.0
    local /= local.sum()
    development_models = [model for model in base["target_maps"] if model["role"] == "development"]
    holdout_models = [model for model in base["target_maps"] if model["role"] == "method_holdout"]
    development = {
        model["model_id"]: load_reprojected_map(model, base, xx, yy, mask)
        for model in development_models
    }
    prediction_cache = {}
    rows = []
    for radius, width, fraction in itertools.product(
        audit["factorial"]["return_radius_arcsec"],
        audit["factorial"]["width_arcsec"],
        audit["factorial"]["routed_fraction"],
    ):
        kernel = normalized_ring_kernel(axis, return_radius_arcsec=radius, width_arcsec=width)
        prediction, _ = routed_arrival_map(local, kernel, routed_fraction=fraction)
        prediction[~mask] = 0.0
        prediction /= prediction.sum()
        key = (float(radius), float(width), float(fraction))
        prediction_cache[key] = prediction
        scores = {model_id: jensen_shannon_divergence(prediction, target, mask) for model_id, target in development.items()}
        rows.append(
            {
                "return_radius_arcsec": radius,
                "width_arcsec": width,
                "routed_fraction": fraction,
                "development_mean_jsd": float(np.mean(list(scores.values()))),
                **{f"{model_id}_jsd": score for model_id, score in scores.items()},
            }
        )
    candidates = pd.DataFrame(rows)
    family_rows = []
    locked = {}
    for family in audit["families"]:
        selected = choose_family(candidates, family)
        key = (float(selected.return_radius_arcsec), float(selected.width_arcsec), float(selected.routed_fraction))
        locked[family] = prediction_cache[key]
        family_rows.append({"family": family, **selected.to_dict()})
    families = pd.DataFrame(family_rows)

    # Open method holdouts only after all three family representatives are locked.
    holdouts = {
        model["model_id"]: load_reprojected_map(model, base, xx, yy, mask)
        for model in holdout_models
    }
    targets = {**development, **holdouts}
    roles = {model["model_id"]: model["role"] for model in base["target_maps"]}
    map_rows = []
    for model_id, target in targets.items():
        for family, prediction in locked.items():
            map_rows.append(
                {
                    "model_id": model_id,
                    "role": roles[model_id],
                    "family": family,
                    "jsd": jensen_shannon_divergence(prediction, target, mask),
                }
            )
    map_scores = pd.DataFrame(map_rows)
    holdout_means = map_scores[map_scores.role == "method_holdout"].groupby("family").jsd.mean()
    development_means = families.set_index("family").development_mean_jsd
    general = families.set_index("family").loc["general_return"]
    gaussian = families.set_index("family").loc["gaussian_smoothing_null"]
    general_development_advantage = float(
        (development_means.gaussian_smoothing_null - development_means.general_return)
        / development_means.gaussian_smoothing_null
    )
    general_holdout_advantage = float(
        (holdout_means.gaussian_smoothing_null - holdout_means.general_return)
        / holdout_means.gaussian_smoothing_null
    )
    strict_map = map_scores[map_scores.family == "strict_arc_return"].set_index("model_id")
    gaussian_map = map_scores[map_scores.family == "gaussian_smoothing_null"].set_index("model_id")
    holdout_ids = list(holdouts)
    strict_both_better = bool(np.all(strict_map.loc[holdout_ids].jsd < gaussian_map.loc[holdout_ids].jsd))
    gates = {
        "general_nonzero_return_radius": bool(general.return_radius_arcsec > 0.0),
        "general_development_improvement_over_gaussian_fraction": general_development_advantage,
        "general_development_improvement_pass": bool(general_development_advantage >= audit["ring_necessity_gates"]["general_development_improvement_over_gaussian_fraction_min"]),
        "general_holdout_improvement_over_gaussian_fraction": general_holdout_advantage,
        "general_holdout_improvement_pass": bool(general_holdout_advantage >= audit["ring_necessity_gates"]["general_holdout_improvement_over_gaussian_fraction_min"]),
        "strict_arc_both_holdouts_better_than_gaussian": strict_both_better,
    }

    output_dir = ROOT / audit["outputs"]["directory"]
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates.sort_values("development_mean_jsd").to_csv(output_dir / audit["outputs"]["candidate_scores"], index=False)
    families.to_csv(output_dir / audit["outputs"]["family_scores"], index=False)
    map_scores.to_csv(output_dir / audit["outputs"]["map_scores"], index=False)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    extent = [axis[0], axis[-1], axis[0], axis[-1]]
    for panel, family in zip(axes, audit["families"], strict=True):
        panel.imshow(locked[family], origin="lower", extent=extent, cmap="viridis")
        selected = families.set_index("family").loc[family]
        panel.set_title(f"{family}\nL={selected.return_radius_arcsec:g}, w={selected.width_arcsec:g}, f={selected.routed_fraction:g}")
        panel.set(xlim=(-50, 50), ylim=(-50, 50), xlabel="west arcsec", ylabel="north arcsec")
        panel.set_aspect("equal")
    figure.savefig(output_dir / audit["outputs"]["figure"], dpi=180)
    plt.close(figure)
    report = {
        "report_version": "P0590B-RING-NECESSITY-AUDIT-RESULTS-0.1.0",
        "status": "complete_reused_map_diagnostic",
        "selected_families": families.set_index("family").to_dict("index"),
        "holdout_mean_jsd": holdout_means.to_dict(),
        "gates": gates,
        "conclusion": (
            "nonzero_return_radius_is_morphologically_required"
            if all([gates["general_nonzero_return_radius"], gates["general_development_improvement_pass"], gates["general_holdout_improvement_pass"], gates["strict_arc_both_holdouts_better_than_gaussian"]])
            else "P0590_improvement_does_not_isolate_arc_return_from_smoothing"
        ),
        "claim_limits": audit["claim_limits"],
    }
    (output_dir / audit["outputs"]["report"]).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output_dir / audit["outputs"]["summary"]).write_text(
        "# P0590B ring-necessity audit\n\n"
        f"Best general return: L={general.return_radius_arcsec:g} arcsec, w={general.width_arcsec:g} arcsec, f={general.routed_fraction:g}. Best Gaussian null: w={gaussian.width_arcsec:g} arcsec, f={gaussian.routed_fraction:g}.\n\n"
        f"The general model's advantage over Gaussian smoothing is {general_development_advantage:+.2%} on development and {general_holdout_advantage:+.2%} on the method holdouts. Conclusion: {report['conclusion']}.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
