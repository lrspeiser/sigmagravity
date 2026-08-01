#!/usr/bin/env python3
"""Test a baryon-axis directional return kernel against the smoothing null."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0590_gravity_return_backtrack import (  # noqa: E402
    coordinate_grid,
    find_peaks,
    jensen_shannon_divergence,
    load_reprojected_map,
    positive_dark_residual,
    source_surface,
)
from voidscreen.gravity_return import (  # noqa: E402
    normalized_directional_ring_kernel,
    normalized_ring_kernel,
    routed_arrival_map,
    source_origin_probabilities,
)


def baryonic_major_axis(sources: pd.DataFrame) -> tuple[float, float, float]:
    weight = np.asarray(sources.mass_msun, dtype=float)
    weight /= weight.sum()
    cx = float(np.sum(weight * sources.x_arcsec))
    cy = float(np.sum(weight * sources.y_arcsec))
    dx = np.asarray(sources.x_arcsec) - cx
    dy = np.asarray(sources.y_arcsec) - cy
    covariance = np.array(
        [
            [np.sum(weight * dx * dx), np.sum(weight * dx * dy)],
            [np.sum(weight * dx * dy), np.sum(weight * dy * dy)],
        ]
    )
    values, vectors = np.linalg.eigh(covariance)
    major = vectors[:, int(np.argmax(values))]
    return math.degrees(math.atan2(major[1], major[0])), cx, cy


def normalized_prediction(local, kernel, fraction, mask):
    prediction, arrival = routed_arrival_map(local, kernel, routed_fraction=fraction)
    prediction[~mask] = 0.0
    arrival[~mask] = 0.0
    return prediction / prediction.sum(), arrival / arrival.sum()


def main() -> None:
    protocol_path = ROOT / "configs/p0591_directional_return_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    base = json.loads((ROOT / protocol["base_protocol"]).read_text(encoding="utf-8"))
    sources = pd.read_csv(ROOT / base["source"]["nominal_baryons"])
    major_axis, center_x, center_y = baryonic_major_axis(sources)
    axis, xx, yy, mask = coordinate_grid(base)
    local = source_surface(sources, axis, base["source"]["source_smoothing_arcsec"])
    local[~mask] = 0.0
    local /= local.sum()
    development_models = [m for m in base["target_maps"] if m["role"] == "development"]
    holdout_models = [m for m in base["target_maps"] if m["role"] == "method_holdout"]
    development = {m["model_id"]: load_reprojected_map(m, base, xx, yy, mask) for m in development_models}
    rows = []
    cache = {}
    for radius, width_fraction, concentration, fraction in itertools.product(
        protocol["frozen_factorial"]["return_radius_arcsec"],
        protocol["frozen_factorial"]["width_fraction"],
        protocol["frozen_factorial"]["directional_concentration"],
        protocol["frozen_factorial"]["routed_fraction"],
    ):
        width = float(width_fraction) * float(radius)
        kernel = normalized_directional_ring_kernel(
            axis,
            return_radius_arcsec=radius,
            width_arcsec=width,
            major_axis_deg=major_axis,
            directional_concentration=concentration,
        )
        prediction, arrival = normalized_prediction(local, kernel, fraction, mask)
        key = (float(radius), float(width_fraction), float(concentration), float(fraction))
        cache[key] = (prediction, arrival)
        scores = {model_id: jensen_shannon_divergence(prediction, target, mask) for model_id, target in development.items()}
        rows.append(
            {
                "return_radius_arcsec": radius,
                "width_fraction": width_fraction,
                "width_arcsec": width,
                "directional_concentration": concentration,
                "routed_fraction": fraction,
                "development_mean_jsd": float(np.mean(list(scores.values()))),
                **{f"{model_id}_jsd": score for model_id, score in scores.items()},
            }
        )
    candidates = pd.DataFrame(rows).sort_values(
        ["development_mean_jsd", "return_radius_arcsec", "width_fraction", "directional_concentration", "routed_fraction"]
    )
    best = candidates.iloc[0]
    best_key = (float(best.return_radius_arcsec), float(best.width_fraction), float(best.directional_concentration), float(best.routed_fraction))
    best_prediction, best_arrival = cache[best_key]
    rotated_kernel = normalized_directional_ring_kernel(
        axis,
        return_radius_arcsec=best.return_radius_arcsec,
        width_arcsec=best.width_arcsec,
        major_axis_deg=major_axis + protocol["controls"]["axis_rotation_degrees"],
        directional_concentration=best.directional_concentration,
    )
    rotated_prediction, _ = normalized_prediction(local, rotated_kernel, best.routed_fraction, mask)
    gaussian_control = protocol["controls"]["gaussian_smoothing"]
    gaussian_kernel = normalized_ring_kernel(
        axis,
        return_radius_arcsec=gaussian_control["return_radius_arcsec"],
        width_arcsec=gaussian_control["width_arcsec"],
    )
    gaussian_prediction, _ = normalized_prediction(local, gaussian_kernel, gaussian_control["routed_fraction"], mask)
    isotropic_control = protocol["controls"]["strict_isotropic_ring"]
    isotropic_kernel = normalized_ring_kernel(
        axis,
        return_radius_arcsec=isotropic_control["return_radius_arcsec"],
        width_arcsec=isotropic_control["width_arcsec"],
    )
    isotropic_prediction, _ = normalized_prediction(local, isotropic_kernel, isotropic_control["routed_fraction"], mask)

    holdouts = {m["model_id"]: load_reprojected_map(m, base, xx, yy, mask) for m in holdout_models}
    targets = {**development, **holdouts}
    roles = {m["model_id"]: m["role"] for m in base["target_maps"]}
    predictions = {
        "directional_return": best_prediction,
        "gaussian_smoothing": gaussian_prediction,
        "strict_isotropic_ring": isotropic_prediction,
        "axis_rotated_45deg": rotated_prediction,
    }
    map_rows = []
    backtrack_rows = []
    for model_id, target in targets.items():
        for model_name, prediction in predictions.items():
            map_rows.append(
                {
                    "model_id": model_id,
                    "role": roles[model_id],
                    "prediction": model_name,
                    "jsd": jensen_shannon_divergence(prediction, target, mask),
                }
            )
        residual, _ = positive_dark_residual(target, local, mask)
        for rank, (peak_x, peak_y, _) in enumerate(find_peaks(residual, axis, mask, 5, 15.0), start=1):
            probability = source_origin_probabilities(
                sources.x_arcsec,
                sources.y_arcsec,
                sources.mass_msun,
                destination_x_arcsec=peak_x,
                destination_y_arcsec=peak_y,
                return_radius_arcsec=best.return_radius_arcsec,
                width_arcsec=best.width_arcsec,
                major_axis_deg=major_axis,
                directional_concentration=best.directional_concentration,
            )
            top = int(np.argmax(probability))
            backtrack_rows.append(
                {
                    "model_id": model_id,
                    "role": roles[model_id],
                    "peak_rank": rank,
                    "peak_x_arcsec": peak_x,
                    "peak_y_arcsec": peak_y,
                    "top_source_component": sources.iloc[top].component,
                    "top_source_id": sources.iloc[top].source_id,
                    "top_source_probability": float(probability[top]),
                    "origin_x_arcsec": float(sources.iloc[top].x_arcsec),
                    "origin_y_arcsec": float(sources.iloc[top].y_arcsec),
                }
            )
    map_scores = pd.DataFrame(map_rows)
    backtracks = pd.DataFrame(backtrack_rows)
    dev_means = map_scores[map_scores.role == "development"].groupby("prediction").jsd.mean()
    holdout_means = map_scores[map_scores.role == "method_holdout"].groupby("prediction").jsd.mean()
    dev_advantage = float((dev_means.gaussian_smoothing - dev_means.directional_return) / dev_means.gaussian_smoothing)
    holdout_advantage = float((holdout_means.gaussian_smoothing - holdout_means.directional_return) / holdout_means.gaussian_smoothing)
    holdout_ids = list(holdouts)
    pivot = map_scores.pivot(index="model_id", columns="prediction", values="jsd")
    both_better = bool(np.all(pivot.loc[holdout_ids, "directional_return"] < pivot.loc[holdout_ids, "gaussian_smoothing"]))
    axis_better = bool(holdout_means.directional_return < holdout_means.axis_rotated_45deg)
    gates = {
        "development_improvement_over_gaussian_fraction": dev_advantage,
        "development_improvement_pass": bool(dev_advantage >= protocol["advance_gates"]["development_improvement_over_gaussian_fraction_min"]),
        "holdout_improvement_over_gaussian_fraction": holdout_advantage,
        "holdout_improvement_pass": bool(holdout_advantage >= protocol["advance_gates"]["holdout_improvement_over_gaussian_fraction_min"]),
        "both_holdouts_better_than_gaussian": both_better,
        "baryon_axis_better_than_45deg_rotated_on_holdout": axis_better,
    }
    output_dir = ROOT / protocol["outputs"]["directory"]
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_dir / protocol["outputs"]["candidate_scores"], index=False)
    map_scores.to_csv(output_dir / protocol["outputs"]["map_scores"], index=False)
    backtracks.to_csv(output_dir / protocol["outputs"]["backtracked_peaks"], index=False)
    figure, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    extent = [axis[0], axis[-1], axis[0], axis[-1]]
    for panel, (name, prediction) in zip(axes, predictions.items(), strict=True):
        panel.imshow(prediction, origin="lower", extent=extent, cmap="viridis")
        panel.set_title(name)
        panel.set(xlim=(-50, 50), ylim=(-50, 50), xlabel="west arcsec", ylabel="north arcsec")
        panel.set_aspect("equal")
    figure.savefig(output_dir / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)
    all_pass = all(
        [
            gates["development_improvement_pass"],
            gates["holdout_improvement_pass"],
            gates["both_holdouts_better_than_gaussian"],
            gates["baryon_axis_better_than_45deg_rotated_on_holdout"],
        ]
    )
    report = {
        "report_version": "P0591-DIRECTIONAL-RETURN-RESULTS-0.1.0",
        "status": "complete_reused_map_directional_diagnostic",
        "baryonic_axis": {"major_axis_deg": major_axis, "center_x_arcsec": center_x, "center_y_arcsec": center_y},
        "locked_candidate": best.to_dict(),
        "development_mean_jsd": dev_means.to_dict(),
        "holdout_mean_jsd": holdout_means.to_dict(),
        "gates": gates,
        "conclusion": "directional_arc_survives_smoothing_null" if all_pass else "directional_arc_not_separated_from_smoothing_null",
        "claim_limits": protocol["claim_limits"],
    }
    (output_dir / protocol["outputs"]["report"]).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output_dir / protocol["outputs"]["summary"]).write_text(
        "# P0591 directional return\n\n"
        f"The frozen baryonic major axis is {major_axis:.2f} deg. Development selected L={best.return_radius_arcsec:g} arcsec, w={best.width_arcsec:g} arcsec, kappa={best.directional_concentration:g}, f={best.routed_fraction:g}.\n\n"
        f"Relative to the Gaussian smoothing null, the directional return changes mean JSD by {dev_advantage:+.2%} on development and {holdout_advantage:+.2%} on method holdouts. Conclusion: {report['conclusion']}.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
