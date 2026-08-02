#!/usr/bin/env python3
"""Ablate the accumulated tensor on spent RX J2129 raw lensing."""

from __future__ import annotations

import argparse
import hashlib
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

from run_adaptive_route_raw_rxj2129 import baryon_field
from run_clash_stellar_morphology_response import MorphologyLens
from run_p0554_all_baryon_route_screen import (
    prepare_hst_map,
    prepare_xray_maps,
)
from run_p0601_frozen_potential_raw_lensing import (
    build_fields as build_p0599_fields,
)
from run_p0601_frozen_potential_raw_lensing import (
    json_safe,
)
from run_p0607_component_direction_raw_lensing import (
    evaluate_fixed,
    exact_fit,
    fixed_geometry,
)
from run_rxj2129_member_geometry import split_images
from run_rxj2129_raw_theory_lensing import (
    load_baryonic_anchors,
    load_images,
)

from voidscreen.accumulated_lensing import (
    build_accumulated_transport_deflection_field,
    zero_pad_square_component_maps,
)
from voidscreen.raw_lensing import loglog_interpolate_with_tails

DEFAULT_CONFIG = ROOT / "configs" / "p0644_spent_rxj2129_accumulated_tensor.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make_field(protocol, raw_protocol, anchors, parent, baryons, images):
    inputs = protocol["inputs"]
    screen_protocol = read_json(ROOT / inputs["component_screen_protocol"])
    acquisition = read_json(ROOT / inputs["component_acquisition_protocol"])
    reused = read_json(ROOT / inputs["reused_hst_protocol"])
    settings = screen_protocol["map_construction"]
    axis = np.arange(
        float(settings["axis_min_arcsec"]),
        float(settings["axis_max_arcsec"]) + 0.5 * float(settings["grid_spacing_arcsec"]),
        float(settings["grid_spacing_arcsec"]),
    )
    context = SimpleNamespace(label="RXJ2129", local=raw_protocol)
    stars, star_audit = prepare_hst_map(
        screen_protocol, acquisition, reused, context, images, axis
    )
    _, gas, gas_audit = prepare_xray_maps(screen_protocol, acquisition, context, axis)
    candidate = protocol["candidate"]
    computational_padding_arcsec = float(
        candidate.get("computational_padding_arcsec", 0.0)
    )
    spacing_arcsec = float(settings["grid_spacing_arcsec"])
    padding_cells = round(computational_padding_arcsec / spacing_arcsec)
    if not np.isclose(padding_cells * spacing_arcsec, computational_padding_arcsec):
        raise ValueError("computational padding must be an integer number of map cells")
    axis, (stars, gas) = zero_pad_square_component_maps(
        axis, stars, gas, padding_cells=padding_cells
    )
    scale = float(raw_protocol["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
    anchor_radius = anchors.radius_kpc.to_numpy(float)
    anchor_gbar = np.power(10.0, anchors.log_gbar.to_numpy(float))

    def gbar(query_radius_kpc):
        return loglog_interpolate_with_tails(
            query_radius_kpc, anchor_radius, anchor_gbar, outer_slope=-2.0
        )

    def carrier(query_radius_arcsec):
        return parent.reduced_alpha_arcsec(query_radius_arcsec, 1.0) - baryons.reduced_alpha_arcsec(
            query_radius_arcsec, 1.0
        )

    field = build_accumulated_transport_deflection_field(
        axis,
        stars,
        gas,
        angular_scale_kpc_per_arcsec=scale,
        carrier_alpha_arcsec=carrier,
        radial_gbar_m_s2=gbar,
        stellar_mass_fraction=float(candidate["stellar_mass_fraction"]),
        gas_mass_fraction=float(candidate["gas_mass_fraction"]),
        coherence_length_kpc=float(candidate["coherence_length_kpc"]),
        accumulation_power=float(candidate["accumulation_power"]),
        a0_m_s2=float(candidate["a0_m_s2"]),
        common_smoothing_kpc=float(candidate.get("common_smoothing_kpc", 0.0)),
        mismatch_mode=str(candidate.get("mismatch_mode", "quadratic_cancellation")),
        closure=str(candidate.get("closure", "path_tensor")),
        transport_steps=int(candidate.get("transport_steps", 12)),
        taper_inner_arcsec=float(candidate["taper_inner_arcsec"]),
        support_radius_arcsec=float(candidate["support_radius_arcsec"]),
        computational_padding_arcsec=computational_padding_arcsec,
    )
    return field, {"hst": star_audit, "xray": gas_audit}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    if protocol.get("status") != "frozen_before_any_P0644_score":
        raise RuntimeError("P0644 protocol is not frozen")
    inputs = protocol["inputs"]
    adequacy = read_json(ROOT / inputs["component_input_audit"])
    if not adequacy["input_adequacy_pass"]:
        raise RuntimeError("component input audit failed")
    raw_protocol = read_json(ROOT / inputs["raw_protocol"])
    p0601_protocol = read_json(ROOT / inputs["P0601_protocol"])
    images = load_images(raw_protocol)
    training, heldout = split_images(images, raw_protocol)
    anchors = load_baryonic_anchors(raw_protocol)
    radial_fields, _, radial_diagnostic = build_p0599_fields(
        anchors, raw_protocol, p0601_protocol["constants"]
    )
    parent = radial_fields["P0599_potential_shape"]
    baryons = baryon_field(anchors, raw_protocol)
    initial = fixed_geometry(ROOT / inputs["P0601_parameters"])
    field, map_audits = make_field(
        protocol, raw_protocol, anchors, parent, baryons, images
    )

    screen_rows, prediction_frames = [], []
    for value in protocol["candidate"]["lambda_grid"]:
        strength = float(value)
        label = f"lambda_{strength:g}".replace(".", "p")
        lens = MorphologyLens(
            raw_protocol,
            {"P0599_potential_shape": parent},
            parent="P0599_potential_shape",
            morphology=field if strength != 0.0 else None,
            fraction=strength,
        )
        metrics, predictions = evaluate_fixed(
            lens, training, heldout, initial, label
        )
        screen_rows.append({"variant_id": label, "lambda": strength, **metrics})
        if not predictions.empty:
            prediction_frames.append(predictions)
        print(
            f"{label}: train={metrics['training_RMS_arcsec']:.6g} "
            f"held={metrics['heldout_RMS_arcsec']:.6g} "
            f"roots={metrics['training_roots_converged']}/{metrics['heldout_roots_converged']}",
            flush=True,
        )
    screen = pd.DataFrame(screen_rows)
    eligible = screen[
        screen.training_roots_converged.eq(len(training))
        & np.isfinite(screen.training_RMS_arcsec)
    ].sort_values(["training_RMS_arcsec", "lambda"])
    if eligible.empty:
        raise RuntimeError("no lambda retained all training roots")
    selected = eligible.iloc[0]
    selected_lambda = float(selected["lambda"])
    selected_lens = MorphologyLens(
        raw_protocol,
        {"P0599_potential_shape": parent},
        parent="P0599_potential_shape",
        morphology=field if selected_lambda != 0.0 else None,
        fraction=selected_lambda,
    )
    fitted = exact_fit(
        selected_lens,
        training,
        heldout,
        initial=initial,
        starts=int(protocol["selection"]["selected_exact_refit_starts"]),
        seed=int(protocol["selection"]["random_seed"]),
    )
    selected_predictions = pd.concat(
        [fitted["training_prediction"], fitted["heldout_prediction"]], ignore_index=True
    )
    selected_predictions["variant_id"] = "selected_exact_refit"
    selected_predictions["lambda"] = selected_lambda
    previous = read_json(ROOT / inputs["P0601_report"])
    baseline = next(row for row in previous["scores"] if row["model"] == "P0599_potential_shape")
    training_score = fitted["training_score"]
    heldout_score = fitted["heldout_score"]
    training_improvement = 1.0 - float(training_score["exact_radial_RMS_arcsec"]) / float(
        baseline["training_RMS_arcsec"]
    )
    heldout_worsening = float(heldout_score["exact_radial_RMS_arcsec"]) / float(
        baseline["heldout_RMS_arcsec"]
    ) - 1.0
    gates = protocol["predeclared_progression_gates"]
    gate_results = {
        "field_curl": float(field.audit["normalized_curl_RMS"])
        <= gates["field_normalized_curl_RMS_max"],
        "source_integral": float(field.audit["source_integral_fraction"])
        <= gates["field_source_integral_fraction_max"],
        "lambda_not_upper_endpoint": selected_lambda
        < max(float(value) for value in protocol["candidate"]["lambda_grid"]),
        "training_roots": int(training_score["converged_roots"]) == len(training),
        "heldout_roots": int(heldout_score["converged_roots"]) == len(heldout),
        "training_improvement": training_improvement
        >= gates["training_RMS_improvement_fraction_min"],
        "heldout_not_materially_worse": heldout_worsening
        <= gates["spent_heldout_RMS_worsening_fraction_max"],
        "fixed_screen_root_safety": int(
            np.sum(screen.training_roots_converged.lt(len(training)))
            + np.sum(screen.heldout_roots_converged.lt(len(heldout)))
        )
        <= gates["fixed_geometry_root_failures_allowed"],
    }
    report = {
        "report_version": "P0644-SPENT-RXJ2129-ACCUMULATED-TENSOR-RESULTS-1.0.0",
        "status": "pass" if all(gate_results.values()) else "fail",
        "all_progression_gates_pass": bool(all(gate_results.values())),
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(ROOT / "src" / "voidscreen" / "accumulated_lensing.py"),
        "coverage": {
            "cluster": "RX J2129.7+0005",
            "training_images": len(training),
            "spent_heldout_images": len(heldout),
            "lambda_rows": len(screen),
            "ordinary_geometry_parameters_refit": 6,
            "universal_gravity_parameters_screened": 1,
            "per_object_spatial_gravity_parameters": 0,
        },
        "field_audit": field.audit,
        "map_audits": map_audits,
        "radial_parent_diagnostic": radial_diagnostic,
        "screen_scores": screen.to_dict(orient="records"),
        "selection": {
            "selected_lambda": selected_lambda,
            "selected_on": "training exact-root RMS only",
            "spent_heldout_used_for_selection": False,
        },
        "exact_refit": {
            "training_RMS_arcsec": float(training_score["exact_radial_RMS_arcsec"]),
            "training_roots": int(training_score["converged_roots"]),
            "spent_heldout_RMS_arcsec": float(heldout_score["exact_radial_RMS_arcsec"]),
            "spent_heldout_roots": int(heldout_score["converged_roots"]),
            "optimizer_cost": float(fitted["optimizer_cost"]),
            "training_improvement_fraction_vs_P0599": training_improvement,
            "spent_heldout_worsening_fraction_vs_P0599": heldout_worsening,
        },
        "comparators": {
            "P0599_training_RMS_arcsec": baseline["training_RMS_arcsec"],
            "P0599_spent_heldout_RMS_arcsec": baseline["heldout_RMS_arcsec"],
            **previous["comparators"],
        },
        "gate_results": gate_results,
        "sealed_P0633_kinematics_opened": False,
        "sealed_P0640_lensing_constraints_opened": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    screen.to_csv(output / "lambda_screen.csv", index=False)
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(
            output / "fixed_geometry_predictions.csv", index=False
        )
    selected_predictions.to_csv(output / "selected_refit_predictions.csv", index=False)
    (output / "report.json").write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(screen["lambda"], screen.training_RMS_arcsec, "o-", label="training")
    axes[0].plot(screen["lambda"], screen.heldout_RMS_arcsec, "o-", label="spent heldout")
    axes[0].axvline(selected_lambda, color="black", linestyle="--", linewidth=1)
    axes[0].set(xlabel="universal lambda", ylabel="exact root RMS (arcsec)", title="Fixed geometry screen")
    axes[0].legend()
    axes[1].imshow(field.light_weight, origin="lower", cmap="magma")
    axes[1].quiver(
        np.arange(0, len(field.axis_arcsec), 24),
        np.arange(0, len(field.axis_arcsec), 24),
        field.raw_alpha_x_arcsec[::24, ::24],
        field.raw_alpha_y_arcsec[::24, ::24],
        color="cyan",
    )
    axes[1].set(title="Accumulated activation and unit deflection", xticks=[], yticks=[])
    figure.tight_layout()
    figure.savefig(output / "spent_tensor_screen.png", dpi=180)
    plt.close(figure)
    summary = f"""# P0644 spent RX J2129 accumulated tensor

- Status: **{report['status'].upper()}** ({sum(gate_results.values())}/{len(gate_results)} progression gates).
- Training-selected universal lambda: **{selected_lambda:g}**.
- Exact-refit training RMS: **{report['exact_refit']['training_RMS_arcsec']:.6g} arcsec** ({100*training_improvement:+.3f}% versus P0599).
- Spent-heldout RMS: **{report['exact_refit']['spent_heldout_RMS_arcsec']:.6g} arcsec** ({100*heldout_worsening:+.3f}% versus P0599).
- Root completion: {report['exact_refit']['training_roots']}/{len(training)} training and {report['exact_refit']['spent_heldout_roots']}/{len(heldout)} spent heldout.
- New validation targets opened: **no**.

This is a mechanism ablation on fully spent data, not validation. A passing
result would still require unchanged transfer to other spent clusters and a
frozen resolution audit before P0640 is opened.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(json.dumps({"status": report["status"], "selected_lambda": selected_lambda, "gates": gate_results}, indent=2))


if __name__ == "__main__":
    main()
