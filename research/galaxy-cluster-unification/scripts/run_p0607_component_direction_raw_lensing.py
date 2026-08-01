#!/usr/bin/env python3
"""Test which observed baryonic component best directs a small angular route."""

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

from run_adaptive_route_raw_rxj2129 import (  # noqa: E402
    MODEL,
    baryon_field,
    exact_fit,
    load_sources,
)
from run_clash_stellar_morphology_response import MorphologyLens  # noqa: E402
from run_p0554_all_baryon_route_screen import (  # noqa: E402
    prepare_hst_map,
    prepare_xray_maps,
)
from run_p0601_frozen_potential_raw_lensing import (  # noqa: E402
    build_fields as build_p0599_fields,
    json_safe,
)
from run_rxj2129_member_geometry import split_images  # noqa: E402
from run_rxj2129_raw_theory_lensing import (  # noqa: E402
    FIXED_LABELS,
    load_baryonic_anchors,
    load_images,
    score,
)
from voidscreen.baryon_morphology import (  # noqa: E402
    blend_unit_directions,
    map_attraction_directions,
)
from voidscreen.route_template import (  # noqa: E402
    baryonic_route_directions,
    conservative_explicit_direction_route_template,
    weighted_radius,
)
from voidscreen.stellar_morphology_lensing import (  # noqa: E402
    build_stellar_morphology_deflection_field,
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fixed_geometry(path: Path) -> np.ndarray:
    parameters = pd.read_csv(path)
    block = parameters[parameters.model.eq("P0599_potential_shape")].set_index("parameter")
    return block.loc[list(FIXED_LABELS), "value"].to_numpy(float)


def component_fields(protocol, raw_protocol, sources, parent, baryons, directions):
    settings = protocol["route_geometry"]
    scale = float(raw_protocol["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
    xy = sources[["x_arcsec", "y_arcsec"]].to_numpy(float)
    weights = sources.base_weight.to_numpy(float) ** float(settings["source_weight_power"])
    weights /= np.sum(weights)
    radius_kpc = np.hypot(xy[:, 0], xy[:, 1]) * scale
    r80 = weighted_radius(radius_kpc, weights, 0.8)
    route_axis = np.arange(-255.5, 256.0, 1.0)

    def carrier_alpha(radius_arcsec):
        return parent.reduced_alpha_arcsec(radius_arcsec, 1.0) - baryons.reduced_alpha_arcsec(radius_arcsec, 1.0)

    fields, rows = {}, []
    for component_id, direction in directions.items():
        route_map, route_audit = conservative_explicit_direction_route_template(
            route_axis,
            xy,
            weights,
            direction,
            routing_fraction=float(settings["routing_fraction_inside_unit_template"]),
            return_scale=float(settings["length_over_R80"]) * r80 / scale,
            radius_exponent=float(settings["radius_exponent"]),
            reference_radius=r80 / scale,
            smoothing=float(settings["width_over_R80"]) * r80 / scale,
            center=None,
        )
        field = build_stellar_morphology_deflection_field(
            route_axis,
            route_map,
            carrier_alpha,
            contrast_cap=float(settings["contrast_cap"]),
            contrast_strength=1.0,
            annulus_width_arcsec=float(settings["annulus_width_arcsec"]),
            taper_inner_arcsec=float(settings["taper_inner_arcsec"]),
            support_radius_arcsec=float(settings["support_radius_arcsec"]),
            radial_samples=2048,
            circular_radii=512,
            circular_azimuths=720,
        )
        fields[component_id] = field
        rows.append(
            {
                "component_id": component_id,
                "R80_kpc": r80,
                "route_length_kpc": float(settings["length_over_R80"]) * r80,
                "route_width_kpc": float(settings["width_over_R80"]) * r80,
                "route_map_normalization_error": float(route_audit["normalization_error"]),
                **field.audit,
            }
        )
    return fields, pd.DataFrame(rows), {"R80_kpc": r80, "sources": len(sources)}


def evaluate_fixed(lens, training, heldout, parameters, label):
    try:
        _, profiled_sources = lens.profiled_residuals(MODEL, parameters, training)
        training_prediction = lens.exact_predictions(
            MODEL, parameters, profiled_sources, training, stage="training"
        )
        heldout_prediction = lens.exact_predictions(
            MODEL, parameters, profiled_sources, heldout, stage="heldout"
        )
        training_score = score(training_prediction, lens.sigma, free_parameters=14)
        heldout_score = score(heldout_prediction, lens.sigma)
        predictions = pd.concat([training_prediction, heldout_prediction], ignore_index=True)
        predictions["variant_id"] = label
        return {
            "training_RMS_arcsec": training_score["exact_radial_RMS_arcsec"],
            "training_roots_converged": training_score["converged_roots"],
            "heldout_RMS_arcsec": heldout_score["exact_radial_RMS_arcsec"],
            "heldout_roots_converged": heldout_score["converged_roots"],
        }, predictions
    except Exception as error:  # root topology failures are a scored outcome
        return {
            "training_RMS_arcsec": np.inf,
            "training_roots_converged": 0,
            "heldout_RMS_arcsec": np.inf,
            "heldout_roots_converged": 0,
            "failure": f"{type(error).__name__}: {error}",
        }, pd.DataFrame()


def selected_row(frame, sign):
    if sign == "positive":
        eligible = frame[(frame.angular_strength > 0.0) & frame.training_roots_converged.eq(15)]
    else:
        eligible = frame[(frame.angular_strength < 0.0) & frame.training_roots_converged.eq(15)]
    eligible = eligible[np.isfinite(eligible.training_RMS_arcsec)]
    if eligible.empty:
        raise RuntimeError(f"no complete {sign} component candidate")
    return eligible.sort_values(["training_RMS_arcsec", "component_id", "angular_strength"]).iloc[0]


def main() -> None:
    config_path = ROOT / "configs/p0607_component_direction_raw_lensing_protocol.json"
    protocol = read_json(config_path)
    if not protocol["status"].startswith("frozen_"):
        raise RuntimeError("P0607 protocol is not frozen")
    inputs = protocol["inputs"]
    adequacy = read_json(ROOT / inputs["component_input_audit"])
    if not adequacy["input_adequacy_pass"]:
        raise RuntimeError("registered component-map inputs did not pass their frozen audit")

    raw_protocol = read_json(ROOT / inputs["raw_protocol"])
    p0601_protocol = read_json(ROOT / inputs["P0601_protocol"])
    route_source_protocol = read_json(ROOT / inputs["route_source_protocol"])
    screen_protocol = read_json(ROOT / inputs["component_screen_protocol"])
    acquisition = read_json(ROOT / inputs["component_acquisition_protocol"])
    reused = read_json(ROOT / inputs["reused_hst_protocol"])
    images = load_images(raw_protocol)
    training, heldout = split_images(images, raw_protocol)
    anchors = load_baryonic_anchors(raw_protocol)
    radial_fields, _, radial_diagnostic = build_p0599_fields(
        anchors, raw_protocol, p0601_protocol["constants"]
    )
    parent = radial_fields["P0599_potential_shape"]
    baryons = baryon_field(anchors, raw_protocol)
    initial = fixed_geometry(ROOT / inputs["P0601_parameters"])
    sources = load_sources(route_source_protocol, raw_protocol)

    map_settings = screen_protocol["map_construction"]
    map_axis = np.arange(
        float(map_settings["axis_min_arcsec"]),
        float(map_settings["axis_max_arcsec"]) + 0.5 * float(map_settings["grid_spacing_arcsec"]),
        float(map_settings["grid_spacing_arcsec"]),
    )
    context = SimpleNamespace(label="RXJ2129", local=raw_protocol)
    star_map, star_audit = prepare_hst_map(
        screen_protocol, acquisition, reused, context, images, map_axis
    )
    _, gas_map, gas_audit = prepare_xray_maps(
        screen_protocol, acquisition, context, map_axis
    )
    xy = sources[["x_arcsec", "y_arcsec"]].to_numpy(float)
    weights = sources.base_weight.to_numpy(float)
    weights /= np.sum(weights)
    scale = float(raw_protocol["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
    route = protocol["route_geometry"]
    member, _, _ = baryonic_route_directions(
        xy,
        weights,
        local_mix=1.0,
        softening=float(route["direction_softening_kpc"]) / scale,
        distance_power=float(route["direction_distance_power"]),
    )
    star, star_direction_audit = map_attraction_directions(
        map_axis,
        star_map,
        xy,
        softening=float(route["direction_softening_kpc"]) / scale,
        distance_power=float(route["direction_distance_power"]),
    )
    gas, gas_direction_audit = map_attraction_directions(
        map_axis,
        gas_map,
        xy,
        softening=float(route["direction_softening_kpc"]) / scale,
        distance_power=float(route["direction_distance_power"]),
    )
    star_gas = blend_unit_directions(star, gas, 0.5)
    all_map = blend_unit_directions(star, gas, 0.5)
    member_all = blend_unit_directions(member, all_map, 0.5)
    directions = {
        "members": member,
        "stars": star,
        "gas": gas,
        "stars_gas_050": star_gas,
        "members_all_050": member_all,
    }
    fields, component_audits, realized = component_fields(
        protocol, raw_protocol, sources, parent, baryons, directions
    )
    component_audits["mean_alignment_with_members"] = [
        float(np.sum(weights * np.sum(directions[name] * member, axis=1)))
        for name in component_audits.component_id
    ]
    component_audits["mean_alignment_with_stars"] = [
        float(np.sum(weights * np.sum(directions[name] * star, axis=1)))
        for name in component_audits.component_id
    ]
    component_audits["mean_alignment_with_gas"] = [
        float(np.sum(weights * np.sum(directions[name] * gas, axis=1)))
        for name in component_audits.component_id
    ]

    screen_rows, prediction_frames = [], []
    baseline_lens = MorphologyLens(
        raw_protocol, {MODEL: parent}, parent=MODEL, morphology=None, fraction=0.0
    )
    baseline_metrics, baseline_predictions = evaluate_fixed(
        baseline_lens, training, heldout, initial, "no_route"
    )
    screen_rows.append(
        {"variant_id": "no_route", "component_id": "none", "angular_strength": 0.0, **baseline_metrics}
    )
    prediction_frames.append(baseline_predictions)
    for component_id, field in fields.items():
        for angular_strength in protocol["effective_angular_strength_grid"]:
            variant_id = f"{component_id}__s{angular_strength:+.4f}".replace("+", "p").replace("-", "m")
            lens = MorphologyLens(
                raw_protocol,
                {MODEL: parent},
                parent=MODEL,
                morphology=field,
                fraction=float(angular_strength),
            )
            metrics, predictions = evaluate_fixed(
                lens, training, heldout, initial, variant_id
            )
            screen_rows.append(
                {
                    "variant_id": variant_id,
                    "component_id": component_id,
                    "angular_strength": float(angular_strength),
                    **metrics,
                }
            )
            if not predictions.empty:
                prediction_frames.append(predictions)
            print(
                f"screen {variant_id}: train={metrics['training_RMS_arcsec']:.4g} "
                f"held={metrics['heldout_RMS_arcsec']:.4g}",
                flush=True,
            )
    screen = pd.DataFrame(screen_rows)
    positive = selected_row(screen, "positive")
    signed = selected_row(screen, "signed")
    global_training = screen[
        screen.training_roots_converged.eq(len(training)) & np.isfinite(screen.training_RMS_arcsec)
    ].sort_values(["training_RMS_arcsec", "component_id", "angular_strength"]).iloc[0]

    impact_rows = []
    for component_id, block in screen[screen.component_id.ne("none")].groupby("component_id"):
        complete = block[
            block.training_roots_converged.eq(len(training)) & np.isfinite(block.training_RMS_arcsec)
        ]
        best_positive = complete[complete.angular_strength > 0.0].sort_values("training_RMS_arcsec").iloc[0]
        best_signed = complete.sort_values("training_RMS_arcsec").iloc[0]
        impact_rows.append(
            {
                "component_id": component_id,
                "complete_variants": len(complete),
                "failed_variants": len(block) - len(complete),
                "training_RMS_span_arcsec": float(complete.training_RMS_arcsec.max() - complete.training_RMS_arcsec.min()),
                "heldout_RMS_span_arcsec": float(complete.heldout_RMS_arcsec.max() - complete.heldout_RMS_arcsec.min()),
                "best_positive_strength": float(best_positive.angular_strength),
                "best_positive_training_RMS_arcsec": float(best_positive.training_RMS_arcsec),
                "best_positive_heldout_RMS_arcsec": float(best_positive.heldout_RMS_arcsec),
                "best_signed_strength": float(best_signed.angular_strength),
                "best_signed_training_RMS_arcsec": float(best_signed.training_RMS_arcsec),
                "best_signed_heldout_RMS_arcsec": float(best_signed.heldout_RMS_arcsec),
            }
        )
    impacts = pd.DataFrame(impact_rows).sort_values("training_RMS_span_arcsec", ascending=False)

    refit_rows, refit_prediction_frames = [], []
    for offset, (role, selected) in enumerate((("positive_route", positive), ("opposite_sign_control", signed))):
        component_id = str(selected.component_id)
        strength = float(selected.angular_strength)
        lens = MorphologyLens(
            raw_protocol,
            {MODEL: parent},
            parent=MODEL,
            morphology=fields[component_id],
            fraction=strength,
        )
        fitted = exact_fit(
            lens,
            training,
            heldout,
            initial=initial,
            starts=int(protocol["selection"]["selected_exact_refit_starts"]),
            seed=int(protocol["selection"]["random_seed"]) + offset,
        )
        joined = pd.concat([fitted["training_prediction"], fitted["heldout_prediction"]], ignore_index=True)
        joined["role"] = role
        joined["component_id"] = component_id
        joined["angular_strength"] = strength
        refit_prediction_frames.append(joined)
        refit_rows.append(
            {
                "role": role,
                "component_id": component_id,
                "angular_strength": strength,
                "training_RMS_arcsec": fitted["training_score"]["exact_radial_RMS_arcsec"],
                "training_roots_converged": fitted["training_score"]["converged_roots"],
                "heldout_RMS_arcsec": fitted["heldout_score"]["exact_radial_RMS_arcsec"],
                "heldout_roots_converged": fitted["heldout_score"]["converged_roots"],
                "optimizer_cost": fitted["optimizer_cost"],
            }
        )
    previous = read_json(ROOT / inputs["P0601_report"])
    previous_row = next(row for row in previous["scores"] if row["model"] == "P0599_potential_shape")
    refit_rows.insert(
        0,
        {
            "role": "P0599_no_route_16_start_reference",
            "component_id": "none",
            "angular_strength": 0.0,
            "training_RMS_arcsec": previous_row["training_RMS_arcsec"],
            "training_roots_converged": previous_row["training_roots_converged"],
            "heldout_RMS_arcsec": previous_row["heldout_RMS_arcsec"],
            "heldout_roots_converged": previous_row["heldout_roots_converged"],
            "optimizer_cost": previous_row["optimizer_cost"],
        },
    )
    refits = pd.DataFrame(refit_rows)

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    component_audits.to_csv(output / protocol["outputs"]["component_audits"], index=False)
    screen.to_csv(output / protocol["outputs"]["screen_scores"], index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        output / protocol["outputs"]["screen_predictions"], index=False
    )
    impacts.to_csv(output / protocol["outputs"]["parameter_impacts"], index=False)
    refits.to_csv(output / protocol["outputs"]["refit_scores"], index=False)
    pd.concat(refit_prediction_frames, ignore_index=True).to_csv(
        output / protocol["outputs"]["refit_predictions"], index=False
    )

    refit_positive = refits[refits.role.eq("positive_route")].iloc[0]
    refit_signed = refits[refits.role.eq("opposite_sign_control")].iloc[0]
    report = {
        "report_version": "P0607-COMPONENT-DIRECTION-RAW-LENSING-RESULTS-0.1.0",
        "status": "complete_spent_component_direction_mechanism_test",
        "coverage": {
            "components": len(fields),
            "angular_strengths_per_component": len(protocol["effective_angular_strength_grid"]),
            "screen_variants_including_no_route": len(screen),
            "training_images": len(training),
            "spent_heldout_images": len(heldout),
            "hard_photometric_members": len(sources),
            "Chandra_exposure_ks": float(gas_audit["total_exposure_ks"]),
        },
        "realized_route": realized,
        "map_readiness": {
            "HST_positive_cells": int(star_audit["positive_cells"]),
            "Chandra_soft_events_on_grid": int(gas_audit["soft_events_on_grid"]),
            "absolute_component_mass_ready": False,
            "use_in_this_test": "direction only",
        },
        "direction_alignment": {
            "stars_vs_members": float(np.sum(weights * np.sum(star * member, axis=1))),
            "gas_vs_members": float(np.sum(weights * np.sum(gas * member, axis=1))),
            "gas_vs_stars": float(np.sum(weights * np.sum(gas * star, axis=1))),
            "star_map_centroid_arcsec": np.asarray(star_direction_audit["map_centroid"]).tolist(),
            "gas_map_centroid_arcsec": np.asarray(gas_direction_audit["map_centroid"]).tolist(),
        },
        "fixed_geometry_training_selection": {
            "global_including_no_route": global_training.to_dict(),
            "best_positive_route": positive.to_dict(),
            "best_opposite_sign_control": signed.to_dict(),
        },
        "exact_refits": refits.to_dict("records"),
        "parameter_impacts": impacts.to_dict("records"),
        "cross_domain_controls": protocol["cross_domain_controls"],
        "interpretation": {
            "positive_route_beats_no_route_training": bool(refit_positive.training_RMS_arcsec < previous_row["training_RMS_arcsec"]),
            "positive_route_beats_no_route_spent_heldout": bool(refit_positive.heldout_RMS_arcsec < previous_row["heldout_RMS_arcsec"]),
            "opposite_sign_beats_positive_training": bool(refit_signed.training_RMS_arcsec < refit_positive.training_RMS_arcsec),
            "opposite_sign_beats_positive_spent_heldout": bool(refit_signed.heldout_RMS_arcsec < refit_positive.heldout_RMS_arcsec),
            "component_mass_claimed": False,
            "hidden_arc_height_identified": False,
        },
        "radial_parent_diagnostic": radial_diagnostic,
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    extent = [map_axis[0], map_axis[-1], map_axis[0], map_axis[-1]]
    axes[0].imshow(np.log1p(star_map), origin="lower", extent=extent, cmap="magma")
    axes[0].contour(map_axis, map_axis, gas_map, levels=np.quantile(gas_map[gas_map > 0.0], [0.7, 0.9, 0.98]), colors="cyan", linewidths=0.8)
    axes[0].set(title="RX J2129: F160W + X-ray contours", xlabel="east offset (arcsec)", ylabel="north offset (arcsec)")
    for component_id, block in screen[screen.component_id.ne("none")].groupby("component_id"):
        ordered = block.sort_values("angular_strength")
        axes[1].plot(ordered.angular_strength, ordered.training_RMS_arcsec, marker="o", ms=3, label=component_id)
    axes[1].axhline(float(baseline_metrics["training_RMS_arcsec"]), color="black", ls="--", label="no route")
    axes[1].set(xlabel="effective angular strength", ylabel="fixed-geometry training RMS (arcsec)", title="Direction and sign response")
    axes[1].legend(fontsize=7)
    display = refits.copy()
    axes[2].barh(display.role, display.heldout_RMS_arcsec, color=["gray", "#1261A0", "#C44E52"])
    axes[2].set(xlabel="spent held-out exact-root RMS (arcsec)", title="Selected exact refits")
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)

    summary = (
        "# P0607 component-direction raw-lensing test\n\n"
        f"Training selected the positive route **{positive.component_id}** at "
        f"**s_theta={positive.angular_strength:g}**. After an 8-start geometry refit its "
        f"training/held-out RMS is **{refit_positive.training_RMS_arcsec:.4f}/"
        f"{refit_positive.heldout_RMS_arcsec:.4f} arcsec** versus the P0599 no-route "
        f"reference **{previous_row['training_RMS_arcsec']:.4f}/"
        f"{previous_row['heldout_RMS_arcsec']:.4f} arcsec**.\n\n"
        f"The opposite-sign control selected **{signed.component_id}**, "
        f"**s_theta={signed.angular_strength:g}**, and refit to "
        f"**{refit_signed.training_RMS_arcsec:.4f}/{refit_signed.heldout_RMS_arcsec:.4f} arcsec**.\n\n"
        "The component maps set direction only; no X-ray count rate is interpreted as gas mass. "
        "RX J2129 is spent, so this is a mechanism diagnostic rather than confirmation.\n"
    )
    (output / protocol["outputs"]["summary"]).write_text(summary, encoding="utf-8")
    print(json.dumps(json_safe({"selection": report["fixed_geometry_training_selection"], "exact_refits": report["exact_refits"], "interpretation": report["interpretation"]}), indent=2))


if __name__ == "__main__":
    main()
