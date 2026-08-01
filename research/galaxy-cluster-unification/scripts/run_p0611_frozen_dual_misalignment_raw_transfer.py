#!/usr/bin/env python3
"""Prospectively transfer the frozen P0610 gate to A383 and MS2137."""

from __future__ import annotations

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

from run_adaptive_route_raw_rxj2129 import MODEL, baryon_field, exact_fit  # noqa: E402
from run_clash_stellar_morphology_response import MorphologyLens  # noqa: E402
from run_p0566_a383_ms2137_morphology_gate_transfer import (  # noqa: E402
    build_contexts,
    prepare_hst_map,
    prepare_xray_map,
)
from run_p0601_frozen_potential_raw_lensing import (  # noqa: E402
    build_fields as build_p0599_fields,
    json_safe,
)
from run_p0607_component_direction_raw_lensing import component_fields  # noqa: E402
from run_p0608_route_redshift_tomography import TomographicRouteLens  # noqa: E402
from run_unbounded_running_multicluster_raw import load_anchors  # noqa: E402
from voidscreen.baryon_morphology import map_attraction_directions  # noqa: E402
from voidscreen.route_template import baryonic_route_directions  # noqa: E402


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_clash_catalog(path: Path) -> pd.DataFrame:
    header = next(
        line[2:].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("##  id")
    )
    return pd.read_csv(path, sep=r"\s+", comment="#", names=header.split(), low_memory=False)


def member_sources(path: Path, context, settings: dict) -> pd.DataFrame:
    catalog = read_clash_catalog(path)
    geometry = context.local_protocol["cosmology_and_coordinates"]
    ra = pd.to_numeric(catalog.RA, errors="coerce").to_numpy(float)
    dec = pd.to_numeric(catalog.Dec, errors="coerce").to_numpy(float)
    magnitude = pd.to_numeric(catalog.f160w_mag, errors="coerce").to_numpy(float)
    stellarity = pd.to_numeric(catalog.stel, errors="coerce").to_numpy(float)
    flag = pd.to_numeric(catalog.flag5sig, errors="coerce").to_numpy(float)
    low = pd.to_numeric(catalog.zbmin, errors="coerce").to_numpy(float)
    high = pd.to_numeric(catalog.zbmax, errors="coerce").to_numpy(float)
    odds = pd.to_numeric(catalog.odds, errors="coerce").to_numpy(float)
    cosine = math.cos(math.radians(float(geometry["center_dec_deg"])))
    x = (ra - float(geometry["center_ra_deg"])) * 3600.0 * cosine
    y = (dec - float(geometry["center_dec_deg"])) * 3600.0
    radius = np.hypot(x, y) * float(geometry["angular_scale_kpc_per_arcsec"])
    z_lens = float(geometry["lens_redshift"])
    keep = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(magnitude)
        & (magnitude > 0.0)
        & (magnitude < 90.0)
        & np.isfinite(stellarity)
        & (stellarity < float(settings["stellarity_max"]))
        & (flag == 0.0)
        & np.isfinite(low)
        & np.isfinite(high)
        & (low <= z_lens)
        & (high >= z_lens)
        & np.isfinite(odds)
        & (odds >= float(settings["photoz_odds_min"]))
        & (radius <= float(settings["aperture_kpc"]))
    )
    result = pd.DataFrame(
        {
            "system_label": context.system["label"],
            "source_id": catalog.id.astype(str).to_numpy()[keep],
            "x_arcsec": x[keep],
            "y_arcsec": y[keep],
            "radius_kpc": radius[keep],
            "F160W_magnitude": magnitude[keep],
            "photoz_low": low[keep],
            "photoz_high": high[keep],
            "photoz_odds": odds[keep],
        }
    )
    result["base_weight"] = np.power(
        10.0, -0.4 * (result.F160W_magnitude - result.F160W_magnitude.min())
    )
    result["base_weight"] /= result.base_weight.sum()
    if len(result) < int(settings["minimum_members"]):
        raise RuntimeError(f"{context.system['label']} has too few frozen photometric members")
    return result.sort_values(["radius_kpc", "source_id"]).reset_index(drop=True)


def score_row(context, variant_id: str, strength: float, fitted: dict) -> dict:
    training = fitted["training_score"]
    heldout = fitted["heldout_score"]
    return {
        "system_label": context.system["label"],
        "variant_id": variant_id,
        "effective_strength": strength,
        "training_images": len(context.training),
        "heldout_images": len(context.heldout),
        "training_RMS_arcsec": training["exact_radial_RMS_arcsec"],
        "training_roots_converged": training["converged_roots"],
        "heldout_RMS_arcsec": heldout["exact_radial_RMS_arcsec"],
        "heldout_roots_converged": heldout["converged_roots"],
        "optimizer_cost": fitted["optimizer_cost"],
    }


def complete(row) -> bool:
    return bool(
        int(row.training_roots_converged) == int(row.training_images)
        and int(row.heldout_roots_converged) == int(row.heldout_images)
        and np.isfinite(row.training_RMS_arcsec)
        and np.isfinite(row.heldout_RMS_arcsec)
    )


def main() -> None:
    config_path = ROOT / "configs/p0611_frozen_dual_misalignment_raw_transfer_protocol.json"
    protocol = read_json(config_path)
    if not protocol["status"].startswith("frozen_"):
        raise RuntimeError("P0611 protocol is not frozen")
    inputs = protocol["inputs"]
    p0566 = read_json(ROOT / inputs["P0566_protocol"])
    acquisition = read_json(ROOT / inputs["P0566_acquisition_protocol"])
    base_raw = read_json(ROOT / inputs["base_raw_protocol"])
    metric = read_json(ROOT / inputs["metric_protocol"])
    p0601 = read_json(ROOT / inputs["P0601_protocol"])
    p0607 = read_json(ROOT / inputs["P0607_protocol"])
    contexts, _, all_images = build_contexts(p0566, base_raw, metric)
    if [context.system["label"] for context in contexts] != protocol["systems"]:
        raise RuntimeError("P0611 system order or coverage changed")

    map_settings = p0566["registered_map_construction"]
    map_axis = np.arange(
        float(map_settings["axis_min_arcsec"]),
        float(map_settings["axis_max_arcsec"]) + 0.5 * float(map_settings["grid_spacing_arcsec"]),
        float(map_settings["grid_spacing_arcsec"]),
    )
    tian = pd.read_csv(
        ROOT / p0566["inputs"]["baryonic_profile"],
        sep=r"\s+",
        names=["system", "radius_kpc", "log_gbar", "log_gobs", "err_log_gbar", "err_log_gobs"],
    )
    source_frames, direction_rows, field_rows, score_rows, prediction_frames = [], [], [], [], []
    gate = protocol["locked_gate"]
    route = p0607["route_geometry"]

    for system_index, context in enumerate(contexts):
        label = context.system["label"]
        print(f"P0611 {label}: construct frozen baryonic directions", flush=True)
        catalog_key = f"{label}_member_catalog"
        sources = member_sources(ROOT / inputs[catalog_key], context, protocol["member_selection"])
        source_frames.append(sources)
        star_map, star_audit = prepare_hst_map(
            p0566, acquisition, context, all_images[label], map_axis
        )
        gas_map, gas_audit = prepare_xray_map(p0566, acquisition, context, map_axis)
        xy = sources[["x_arcsec", "y_arcsec"]].to_numpy(float)
        weights = sources.base_weight.to_numpy(float)
        weights /= weights.sum()
        scale = float(context.local_protocol["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
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
        c_gm = float(np.sum(weights * np.sum(gas * member, axis=1)))
        c_gs = float(np.sum(weights * np.sum(gas * star, axis=1)))
        dual = float(np.sqrt(max(0.0, 1.0 - c_gm) * max(0.0, 1.0 - c_gs)))
        power = float(gate["power"])
        threshold = float(gate["threshold"])
        activation = float(dual**power / (dual**power + threshold**power))
        strength = float(gate["base_strength"]) * activation
        direction_rows.append(
            {
                "system_label": label,
                "members": len(sources),
                "mean_alignment_gas_member": c_gm,
                "mean_alignment_gas_star": c_gs,
                "dual_misalignment": dual,
                "candidate_gate_H": activation,
                "effective_strength": strength,
                "star_centroid_x_arcsec": float(star_direction_audit["map_centroid"][0]),
                "star_centroid_y_arcsec": float(star_direction_audit["map_centroid"][1]),
                "gas_centroid_x_arcsec": float(gas_direction_audit["map_centroid"][0]),
                "gas_centroid_y_arcsec": float(gas_direction_audit["map_centroid"][1]),
                "member_direction_finite": bool(np.isfinite(member).all()),
                "HST_positive_cells": int(star_audit["positive_cells"]),
                "Chandra_exposure_ks": float(gas_audit["total_exposure_ks"]),
                "Chandra_soft_events": int(gas_audit["soft_events_on_grid"]),
            }
        )

        local = json.loads(json.dumps(context.local_protocol))
        images = pd.concat([context.training, context.heldout], ignore_index=True)
        radii = np.hypot(images.x_arcsec.to_numpy(float), images.y_arcsec.to_numpy(float)) * scale
        local.setdefault("baryonic_inputs", {})["strong_lens_impact_radius_range_kpc_expected"] = [
            float(radii.min()),
            float(radii.max()),
        ]
        anchors = load_anchors(tian, label)
        radial_fields, _, radial_diagnostic = build_p0599_fields(
            anchors, local, p0601["constants"]
        )
        parent = radial_fields["P0599_potential_shape"]
        baryons = baryon_field(anchors, local)
        fields, audits, realized = component_fields(
            p0607, local, sources, parent, baryons, {"gas": gas}
        )
        field_rows.append(
            {
                "system_label": label,
                "P0599_shape_gate": radial_diagnostic["shape_gate"],
                "gate_activation": activation,
                "effective_strength": strength,
                "R80_kpc": realized["R80_kpc"],
                **audits.iloc[0].to_dict(),
            }
        )
        variants = [
            (
                "P0599_no_route",
                0.0,
                MorphologyLens(local, {MODEL: parent}, parent=MODEL, morphology=None, fraction=0.0),
            ),
            (
                "P0610_gated_gas_route",
                strength,
                TomographicRouteLens(
                    local,
                    {MODEL: parent},
                    parent=MODEL,
                    morphology=fields["gas"],
                    strength=strength,
                    gamma=float(gate["redshift_exponent"]),
                ),
            ),
        ]
        for variant_id, variant_strength, lens in variants:
            print(f"P0611 {label}: exact {variant_id}", flush=True)
            try:
                fitted = exact_fit(
                    lens,
                    context.training,
                    context.heldout,
                    initial=context.initial_geometry,
                    starts=int(protocol["raw_fit"]["optimization_starts_per_variant_cluster"]),
                    # Use identical random starts for the paired variants so an
                    # effectively zero gate cannot masquerade as a field effect.
                    seed=int(protocol["raw_fit"]["seed"]) + 100 * system_index,
                )
                score_rows.append(score_row(context, variant_id, variant_strength, fitted))
                for stage in ("training", "heldout"):
                    frame = fitted[f"{stage}_prediction"].copy()
                    frame["system_label"] = label
                    frame["variant_id"] = variant_id
                    prediction_frames.append(frame)
            except Exception as error:
                score_rows.append(
                    {
                        "system_label": label,
                        "variant_id": variant_id,
                        "effective_strength": variant_strength,
                        "training_images": len(context.training),
                        "heldout_images": len(context.heldout),
                        "training_RMS_arcsec": np.inf,
                        "training_roots_converged": 0,
                        "heldout_RMS_arcsec": np.inf,
                        "heldout_roots_converged": 0,
                        "optimizer_cost": np.inf,
                        "failure": f"{type(error).__name__}: {error}",
                    }
                )

    sources = pd.concat(source_frames, ignore_index=True)
    directions = pd.DataFrame(direction_rows)
    fields = pd.DataFrame(field_rows)
    scores = pd.DataFrame(score_rows)
    indexed = scores.set_index(["system_label", "variant_id"])
    response_rows = []
    for label in protocol["systems"]:
        baseline = indexed.loc[(label, "P0599_no_route")]
        candidate = indexed.loc[(label, "P0610_gated_gas_route")]
        valid = complete(baseline) and complete(candidate)
        heldout_valid = bool(
            int(baseline.heldout_roots_converged) == int(baseline.heldout_images)
            and int(candidate.heldout_roots_converged) == int(candidate.heldout_images)
            and np.isfinite(baseline.heldout_RMS_arcsec)
            and np.isfinite(candidate.heldout_RMS_arcsec)
        )
        heldout_diagnostic = (
            float(1.0 - candidate.heldout_RMS_arcsec / baseline.heldout_RMS_arcsec)
            if heldout_valid
            else np.nan
        )
        response_rows.append(
            {
                "system_label": label,
                "complete_pair": valid,
                "heldout_pair_complete": heldout_valid,
                "heldout_only_diagnostic_improvement_fraction": heldout_diagnostic,
                "heldout_improvement_fraction": heldout_diagnostic if valid else np.nan,
            }
        )
    responses = pd.DataFrame(response_rows).merge(
        directions[["system_label", "candidate_gate_H"]], on="system_label", how="left"
    )
    complete_labels = responses[responses.complete_pair].system_label.tolist()
    if complete_labels:
        base_values = indexed.loc[
            [(label, "P0599_no_route") for label in complete_labels], "heldout_RMS_arcsec"
        ].to_numpy(float)
        candidate_values = indexed.loc[
            [(label, "P0610_gated_gas_route") for label in complete_labels], "heldout_RMS_arcsec"
        ].to_numpy(float)
        base_rms = float(np.sqrt(np.mean(base_values**2)))
        candidate_rms = float(np.sqrt(np.mean(candidate_values**2)))
        improvement = 1.0 - candidate_rms / base_rms
    else:
        base_rms = candidate_rms = np.inf
        improvement = -np.inf
    order = responses.sort_values("candidate_gate_H")
    ordering_pass = bool(
        len(order) == 2
        and order.complete_pair.all()
        and order.heldout_improvement_fraction.iloc[-1]
        >= order.heldout_improvement_fraction.iloc[0]
    )
    gates = protocol["advance_gates"]
    gate_audit = {
        "all_training_and_heldout_roots_each_system_pass": bool(responses.complete_pair.all()),
        "activation_response_ordering_pass": ordering_pass,
        "both_systems_not_worse_pass": bool(
            responses.complete_pair.all()
            and (responses.heldout_improvement_fraction >= 0.0).all()
        ),
        "equal_system_heldout_improvement_fraction": improvement,
        "equal_system_heldout_improvement_pass": bool(
            improvement >= float(gates["equal_system_heldout_improvement_fraction_min"])
        ),
        "equal_system_candidate_RMS_arcsec": candidate_rms,
        "equal_system_absolute_RMS_pass": bool(
            candidate_rms <= float(gates["equal_system_heldout_RMS_arcsec_max"])
        ),
    }
    gate_audit["all_gates_pass"] = all(
        value for key, value in gate_audit.items() if key.endswith("_pass")
    )

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    sources.to_csv(output / protocol["outputs"]["member_sources"], index=False)
    directions.to_csv(output / protocol["outputs"]["direction_audits"], index=False)
    fields.to_csv(output / protocol["outputs"]["field_audits"], index=False)
    scores.to_csv(output / protocol["outputs"]["system_scores"], index=False)
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(
            output / protocol["outputs"]["predictions"], index=False
        )
    report = {
        "report_version": "P0611-FROZEN-DUAL-MISALIGNMENT-RAW-TRANSFER-RESULTS-0.1.0",
        "status": "complete_chronologically_prospective_project_spent_transfer",
        "chronology": protocol["chronology"],
        "coverage": {
            "systems": len(contexts),
            "member_sources": len(sources),
            "raw_variant_fits": len(scores),
            "complete_matched_systems": len(complete_labels),
        },
        "locked_gate": protocol["locked_gate"],
        "direction_audits": directions.to_dict("records"),
        "raw_scores": scores.to_dict("records"),
        "responses": responses.to_dict("records"),
        "aggregate": {
            "matched_baseline_RMS_arcsec": base_rms,
            "matched_candidate_RMS_arcsec": candidate_rms,
            "matched_improvement_fraction": improvement,
        },
        "gate_audit": gate_audit,
        "interpretation": {
            "P0610_gate_transfers_to_both_systems": bool(gate_audit["all_gates_pass"]),
            "can_count_as_pristine_project_holdout": False,
            "per_cluster_gravity_retuning": False,
        },
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), constrained_layout=True)
    axes[0].bar(directions.system_label, directions.candidate_gate_H, color="#1261A0")
    axes[0].set(ylabel="frozen gate H", title="Baryon-only activation")
    width = 0.35
    x = np.arange(len(protocol["systems"]))
    base_plot = scores[scores.variant_id.eq("P0599_no_route")].set_index("system_label").loc[protocol["systems"]]
    candidate_plot = scores[scores.variant_id.eq("P0610_gated_gas_route")].set_index("system_label").loc[protocol["systems"]]
    axes[1].bar(x - width / 2, base_plot.heldout_RMS_arcsec, width, label="no route", color="gray")
    axes[1].bar(x + width / 2, candidate_plot.heldout_RMS_arcsec, width, label="gated route", color="#55A868")
    axes[1].set(xticks=x, xticklabels=protocol["systems"], ylabel="held-out RMS (arcsec)", title="Raw multiple-image transfer")
    axes[1].legend(fontsize=8)
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)
    (output / protocol["outputs"]["summary"]).write_text(
        "# P0611 frozen dual-misalignment transfer\n\n"
        + "\n".join(
            f"- {row.system_label}: H={row.candidate_gate_H:.4f}, "
            + (
                f"valid held-out improvement={row.heldout_improvement_fraction:+.3%}"
                if np.isfinite(row.heldout_improvement_fraction)
                else f"invalid full score; held-out-only diagnostic={row.heldout_only_diagnostic_improvement_fraction:+.3%}"
            )
            for row in responses.itertuples(index=False)
        )
        + f"\n\nMatched equal-system improvement: **{improvement:+.3%}**; candidate RMS: **{candidate_rms:.3f} arcsec**; all gates pass: **{gate_audit['all_gates_pass']}**.\n",
        encoding="utf-8",
    )
    print(json.dumps(json_safe({"directions": report["direction_audits"], "responses": report["responses"], "aggregate": report["aggregate"], "gate": report["gate_audit"]}), indent=2))


if __name__ == "__main__":
    main()
