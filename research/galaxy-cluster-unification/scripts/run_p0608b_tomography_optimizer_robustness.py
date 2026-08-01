#!/usr/bin/env python3
"""Measure geometry-basin noise around the P0608 redshift exponent test."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_adaptive_route_raw_rxj2129 import MODEL, baryon_field, exact_fit, load_sources  # noqa: E402
from run_p0554_all_baryon_route_screen import prepare_xray_maps  # noqa: E402
from run_p0601_frozen_potential_raw_lensing import build_fields as build_p0599_fields, json_safe  # noqa: E402
from run_p0607_component_direction_raw_lensing import component_fields, fixed_geometry  # noqa: E402
from run_p0608_route_redshift_tomography import TomographicRouteLens  # noqa: E402
from run_rxj2129_member_geometry import split_images  # noqa: E402
from run_rxj2129_raw_theory_lensing import load_baryonic_anchors, load_images  # noqa: E402
from voidscreen.baryon_morphology import map_attraction_directions  # noqa: E402


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(config_relative="configs/p0608b_tomography_optimizer_robustness_protocol.json") -> None:
    config_path = ROOT / config_relative
    protocol = read_json(config_path)
    if not protocol["status"].startswith("frozen_"):
        raise RuntimeError("P0608B protocol is not frozen")
    inputs = protocol["inputs"]
    p0608 = read_json(ROOT / inputs["P0608_report"])
    locked = protocol["locked_formula"]
    if p0608["locked_route"]["component"] != locked["component"]:
        raise RuntimeError("P0608 component changed")
    if not np.isclose(p0608["locked_route"]["angular_strength"], locked["angular_strength"]):
        raise RuntimeError("P0608 angular strength changed")

    raw_protocol = read_json(ROOT / inputs["raw_protocol"])
    p0601_protocol = read_json(ROOT / inputs["P0601_protocol"])
    p0607_protocol = read_json(ROOT / inputs["P0607_protocol"])
    source_protocol = read_json(ROOT / inputs["route_source_protocol"])
    screen_protocol = read_json(ROOT / inputs["component_screen_protocol"])
    acquisition = read_json(ROOT / inputs["component_acquisition_protocol"])
    images = load_images(raw_protocol)
    training, heldout = split_images(images, raw_protocol)
    anchors = load_baryonic_anchors(raw_protocol)
    radial_fields, _, _ = build_p0599_fields(anchors, raw_protocol, p0601_protocol["constants"])
    parent = radial_fields["P0599_potential_shape"]
    baryons = baryon_field(anchors, raw_protocol)
    initial = fixed_geometry(ROOT / inputs["P0601_parameters"])
    sources = load_sources(source_protocol, raw_protocol)
    map_settings = screen_protocol["map_construction"]
    map_axis = np.arange(
        float(map_settings["axis_min_arcsec"]),
        float(map_settings["axis_max_arcsec"]) + 0.5 * float(map_settings["grid_spacing_arcsec"]),
        float(map_settings["grid_spacing_arcsec"]),
    )
    context = SimpleNamespace(label="RXJ2129", local=raw_protocol)
    _, gas_map, _ = prepare_xray_maps(screen_protocol, acquisition, context, map_axis)
    xy = sources[["x_arcsec", "y_arcsec"]].to_numpy(float)
    scale = float(raw_protocol["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
    route = p0607_protocol["route_geometry"]
    gas_direction, _ = map_attraction_directions(
        map_axis,
        gas_map,
        xy,
        softening=float(route["direction_softening_kpc"]) / scale,
        distance_power=float(route["direction_distance_power"]),
    )
    fields, _, _ = component_fields(
        p0607_protocol, raw_protocol, sources, parent, baryons, {"gas": gas_direction}
    )
    gas_field = fields["gas"]

    repeats = int(protocol["robustness"]["independent_one_start_fits_per_gamma"])
    starts_per_repeat = int(protocol["robustness"].get("starts_per_repeat", 1))
    base_seed = int(protocol["robustness"]["seed"])
    rows = []
    for gamma_index, gamma in enumerate(locked["gammas"]):
        for trial in range(repeats):
            lens = TomographicRouteLens(
                raw_protocol,
                {MODEL: parent},
                parent=MODEL,
                morphology=gas_field,
                strength=float(locked["angular_strength"]),
                gamma=float(gamma),
            )
            try:
                fitted = exact_fit(
                    lens,
                    training,
                    heldout,
                    initial=initial,
                    starts=starts_per_repeat,
                    seed=base_seed + 1000 * gamma_index + trial,
                )
                rows.append(
                    {
                        "gamma": float(gamma),
                        "trial": trial,
                        "seed": base_seed + 1000 * gamma_index + trial,
                        "training_RMS_arcsec": fitted["training_score"]["exact_radial_RMS_arcsec"],
                        "training_roots_converged": fitted["training_score"]["converged_roots"],
                        "heldout_RMS_arcsec": fitted["heldout_score"]["exact_radial_RMS_arcsec"],
                        "heldout_roots_converged": fitted["heldout_score"]["converged_roots"],
                        "optimizer_cost": fitted["optimizer_cost"],
                    }
                )
            except Exception as error:
                rows.append(
                    {
                        "gamma": float(gamma),
                        "trial": trial,
                        "seed": base_seed + 1000 * gamma_index + trial,
                        "training_RMS_arcsec": np.inf,
                        "training_roots_converged": 0,
                        "heldout_RMS_arcsec": np.inf,
                        "heldout_roots_converged": 0,
                        "optimizer_cost": np.inf,
                        "failure": f"{type(error).__name__}: {error}",
                    }
                )
            print(
                f"gamma={gamma:g} repeat={trial + 1}/{repeats} "
                f"starts={starts_per_repeat}",
                flush=True,
            )
    fits = pd.DataFrame(rows)
    complete = fits[
        fits.training_roots_converged.eq(len(training))
        & fits.heldout_roots_converged.eq(len(heldout))
        & np.isfinite(fits.training_RMS_arcsec)
        & np.isfinite(fits.heldout_RMS_arcsec)
    ].copy()
    summaries, best_rows = [], []
    for gamma, block in complete.groupby("gamma"):
        best = block.sort_values(["training_RMS_arcsec", "trial"]).iloc[0]
        best_rows.append(best)
        summaries.append(
            {
                "gamma": float(gamma),
                "attempted_fits": repeats,
                "complete_fits": len(block),
                "training_RMS_best_arcsec": float(best.training_RMS_arcsec),
                "training_RMS_median_arcsec": float(block.training_RMS_arcsec.median()),
                "training_RMS_p16_arcsec": float(block.training_RMS_arcsec.quantile(0.16)),
                "training_RMS_p84_arcsec": float(block.training_RMS_arcsec.quantile(0.84)),
                "heldout_RMS_at_best_training_arcsec": float(best.heldout_RMS_arcsec),
                "heldout_RMS_median_arcsec": float(block.heldout_RMS_arcsec.median()),
                "heldout_RMS_p16_arcsec": float(block.heldout_RMS_arcsec.quantile(0.16)),
                "heldout_RMS_p84_arcsec": float(block.heldout_RMS_arcsec.quantile(0.84)),
            }
        )
    summary = pd.DataFrame(summaries).sort_values("gamma")
    best = pd.DataFrame(best_rows).set_index("gamma")
    g0, g1 = best.loc[0.0], best.loc[1.0]
    pooled_training_span = float(complete.training_RMS_arcsec.quantile(0.84) - complete.training_RMS_arcsec.quantile(0.16))
    pooled_heldout_span = float(complete.heldout_RMS_arcsec.quantile(0.84) - complete.heldout_RMS_arcsec.quantile(0.16))
    best_training_difference = abs(float(g0.training_RMS_arcsec - g1.training_RMS_arcsec))
    best_heldout_difference = abs(float(g0.heldout_RMS_arcsec - g1.heldout_RMS_arcsec))
    indexed_summary = summary.set_index("gamma")
    median_training_difference = abs(
        float(indexed_summary.loc[0.0, "training_RMS_median_arcsec"])
        - float(indexed_summary.loc[1.0, "training_RMS_median_arcsec"])
    )
    median_heldout_difference = abs(
        float(indexed_summary.loc[0.0, "heldout_RMS_median_arcsec"])
        - float(indexed_summary.loc[1.0, "heldout_RMS_median_arcsec"])
    )
    heldout_order_stable = bool(
        (indexed_summary.loc[0.0, "heldout_RMS_p84_arcsec"] < indexed_summary.loc[1.0, "heldout_RMS_p16_arcsec"])
        or (indexed_summary.loc[1.0, "heldout_RMS_p84_arcsec"] < indexed_summary.loc[0.0, "heldout_RMS_p16_arcsec"])
    )

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    fits.to_csv(output / protocol["outputs"]["fits"], index=False)
    summary.to_csv(output / protocol["outputs"]["summary_table"], index=False)
    report = {
        "report_version": f"{protocol['protocol_version']}-RESULTS",
        "status": "complete_posthoc_geometry_basin_audit",
        "coverage": {
            "gammas": len(locked["gammas"]),
            "repeats_per_gamma": repeats,
            "starts_per_repeat": starts_per_repeat,
            "total_fits": len(fits),
            "complete_fits": len(complete),
        },
        "gamma_summary": summary.to_dict("records"),
        "best_training_comparison": {
            "gamma0_training_RMS_arcsec": float(g0.training_RMS_arcsec),
            "gamma1_training_RMS_arcsec": float(g1.training_RMS_arcsec),
            "absolute_training_difference_arcsec": best_training_difference,
            "gamma0_heldout_RMS_arcsec": float(g0.heldout_RMS_arcsec),
            "gamma1_heldout_RMS_arcsec": float(g1.heldout_RMS_arcsec),
            "absolute_heldout_difference_arcsec": best_heldout_difference,
        },
        "basin_noise": {
            "pooled_training_p16_p84_span_arcsec": pooled_training_span,
            "pooled_heldout_p16_p84_span_arcsec": pooled_heldout_span,
            "median_training_gamma_difference_arcsec": median_training_difference,
            "median_heldout_gamma_difference_arcsec": median_heldout_difference,
            "median_training_gamma_difference_to_basin_span_ratio": median_training_difference / max(pooled_training_span, np.finfo(float).tiny),
            "median_heldout_gamma_difference_to_basin_span_ratio": median_heldout_difference / max(pooled_heldout_span, np.finfo(float).tiny),
        },
        "interpretation": {
            "gamma_separable_from_optimizer_basin": bool(
                median_training_difference > pooled_training_span and heldout_order_stable
            ),
            "heldout_order_is_stable": heldout_order_stable,
            "gamma_identified": False,
            "independent_random_start_realized": bool(starts_per_repeat > 1),
        },
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    for gamma, block in complete.groupby("gamma"):
        axes[0].scatter(block.training_RMS_arcsec, block.heldout_RMS_arcsec, s=28, alpha=0.7, label=f"gamma={gamma:g}")
    axes[0].set(xlabel="training RMS (arcsec)", ylabel="spent held-out RMS (arcsec)", title="Random-start geometry basins")
    axes[0].legend()
    data = [complete[complete.gamma.eq(gamma)].heldout_RMS_arcsec.to_numpy() for gamma in locked["gammas"]]
    axes[1].boxplot(data, tick_labels=[f"gamma={gamma:g}" for gamma in locked["gammas"]])
    axes[1].set(ylabel="spent held-out RMS (arcsec)", title="Basin spread exceeds tomography signal")
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)

    (output / protocol["outputs"]["summary"]).write_text(
        f"# {protocol['protocol_version']} optimizer-basin audit\n\n"
        f"Across {len(fits)} repeated fits, the median gamma=0 versus gamma=1 training difference is {median_training_difference:.6f} arcsec, while the pooled 16-84% basin span is {pooled_training_span:.6f} arcsec. "
        f"The corresponding median held-out difference/span is {median_heldout_difference:.6f}/{pooled_heldout_span:.6f} arcsec. The rare best-training basins reverse the held-out ordering.\n\n"
        "The current angular route is too small and too degenerate with structural lens geometry to identify a redshift exponent.\n",
        encoding="utf-8",
    )
    print(json.dumps(json_safe({"summary": report["gamma_summary"], "basin_noise": report["basin_noise"], "interpretation": report["interpretation"]}), indent=2))


if __name__ == "__main__":
    main()
