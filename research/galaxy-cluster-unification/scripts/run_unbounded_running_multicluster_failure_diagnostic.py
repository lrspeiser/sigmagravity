#!/usr/bin/env python3
"""Post-failure per-cluster amplitude diagnostic for locked running laws."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_rxj2129_raw_theory_lensing import RawLens, score  # noqa: E402
from run_unbounded_running_multicluster_raw import (  # noqa: E402
    aggregate_system_scores,
    build_field,
    json_safe,
    load_anchors,
    load_system_images,
    predictive_split,
    system_protocol,
)
from voidscreen.raw_lensing import RadialDeflectionField  # noqa: E402


def scaled_field(field: RadialDeflectionField, amplitude: float) -> RadialDeflectionField:
    return RadialDeflectionField(
        np.asarray(field.impact_arcsec, dtype=float),
        np.asarray(field.physical_deflection_radians, dtype=float) * float(amplitude),
    )


def main() -> None:
    config_path = ROOT / "configs/unbounded_running_multicluster_failure_diagnostic.json"
    diagnostic = json.loads(config_path.read_text(encoding="utf-8"))
    if diagnostic["status"] != "frozen_after_primary_failure_before_amplitude_scores":
        raise RuntimeError("failure diagnostic protocol was not frozen")
    protocol = json.loads((ROOT / diagnostic["inputs"]["primary_protocol"]).read_text())
    primary = json.loads((ROOT / diagnostic["inputs"]["primary_report"]).read_text())
    catalog = pd.read_csv(ROOT / protocol["inputs"]["image_catalog"])
    tian = pd.read_csv(
        ROOT / protocol["baryonic_profile"]["input"],
        sep=r"\s+",
        names=["system", "radius_kpc", "log_gbar", "log_gobs", "err_log_gbar", "err_log_gobs"],
    )
    systems = {item["system"]: item for item in protocol["systems"]}
    cutoff = float(protocol["photon_and_environment_closure"]["primary_isolated_tail_cutoff_kpc"])
    grid_rows = []
    predictions = []
    chosen = {}
    for system_index, system_name in enumerate(diagnostic["systems"]):
        system = systems[system_name]
        local = system_protocol(protocol, system)
        images = load_system_images(catalog, system)
        training, heldout = predictive_split(images)
        anchors = load_anchors(tian, system["label"])
        chosen[system_name] = {}
        for model_index, model in enumerate(diagnostic["candidate_models"]):
            base_field, _ = build_field(model, anchors, protocol, local, cutoff)
            candidates = []
            for amplitude_index, amplitude in enumerate(diagnostic["amplitude_grid"]):
                print(f"system={system['label']} model={model} amplitude={amplitude:g}", flush=True)
                lens = RawLens(local, {model: scaled_field(base_field, amplitude)})
                fitted = lens.fit(
                    model,
                    training,
                    starts=int(diagnostic["optimization"]["multi_starts_per_grid_value"]),
                    seed=int(diagnostic["optimization"]["random_seed"]) + 1000 * system_index + 100 * model_index + amplitude_index,
                )
                row = {
                    "system": system_name,
                    "model": model,
                    "amplitude": float(amplitude),
                    "training_objective_cost": float(fitted["result"].cost),
                    "parameters": fitted["result"].x,
                    "sources": fitted["sources"],
                }
                grid_rows.append({key: value for key, value in row.items() if key not in {"parameters", "sources"}})
                candidates.append(row)
            best = min(candidates, key=lambda row: row["training_objective_cost"])
            lens = RawLens(local, {model: scaled_field(base_field, best["amplitude"])})
            heldout_prediction = lens.exact_predictions(
                model,
                best["parameters"],
                best["sources"],
                heldout,
                stage="heldout_amplitude_diagnostic",
            )
            heldout_prediction.insert(0, "system", system_name)
            heldout_prediction.insert(1, "amplitude", best["amplitude"])
            predictions.append(heldout_prediction)
            chosen[system_name][model] = {
                "training_selected_amplitude": best["amplitude"],
                "training_objective_cost": best["training_objective_cost"],
                "heldout": score(heldout_prediction, lens.sigma),
            }

    aggregates = {}
    diagnoses = {}
    thresholds = diagnostic["diagnostic_thresholds"]
    for model in diagnostic["candidate_models"]:
        rows = [chosen[system][model]["heldout"] for system in diagnostic["systems"]]
        aggregate = aggregate_system_scores(rows)
        amplitudes = np.asarray([chosen[system][model]["training_selected_amplitude"] for system in diagnostic["systems"]], dtype=float)
        locked = primary["primary_aggregate"][model]["equal_system_radial_RMS_arcsec"]
        halo = primary["primary_aggregate"]["GR_plus_cluster_halo"]["equal_system_radial_RMS_arcsec"]
        fractional_range = float((amplitudes.max() - amplitudes.min()) / amplitudes.mean())
        aggregates[model] = aggregate
        diagnoses[model] = {
            "locked_RMS_arcsec": locked,
            "amplitude_diagnostic_RMS_arcsec": aggregate["equal_system_radial_RMS_arcsec"],
            "diagnostic_to_locked_RMS_ratio": aggregate["equal_system_radial_RMS_arcsec"] / locked,
            "normalization_rescue": bool(
                aggregate["all_roots_converged"]
                and aggregate["equal_system_radial_RMS_arcsec"] / locked
                <= thresholds["normalization_rescue_fraction_of_locked_RMS_max"]
            ),
            "selected_amplitudes": amplitudes,
            "amplitude_fractional_range": fractional_range,
            "amplitude_consistent_with_universal": fractional_range <= thresholds["amplitude_universality_fractional_range_max"],
            "diagnostic_to_compact_halo_RMS_ratio": aggregate["equal_system_radial_RMS_arcsec"] / halo,
            "within_compact_halo_ratio": aggregate["equal_system_radial_RMS_arcsec"] / halo <= thresholds["candidate_within_compact_halo_ratio_max"],
        }
    halo_adequacy = {
        system: primary["system_scores"][system]["GR_plus_cluster_halo"]["heldout"]["exact_radial_RMS_arcsec"] <= thresholds["compact_halo_geometry_adequate_RMS_arcsec_max"]
        for system in diagnostic["systems"]
    }
    report = {
        "report_version": diagnostic["protocol_version"],
        "status": "completed post-failure amplitude diagnostic",
        "claim_boundary": diagnostic["claim_boundary"],
        "protocol": {"path": str(config_path.relative_to(ROOT)).replace("\\", "/"), "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()},
        "chosen_by_system": chosen,
        "aggregate": aggregates,
        "diagnosis": diagnoses,
        "compact_halo_geometry_adequacy": halo_adequacy,
        "verdict": {
            "any_normalization_rescue": any(value["normalization_rescue"] for value in diagnoses.values()),
            "any_amplitude_universal": any(value["amplitude_consistent_with_universal"] for value in diagnoses.values()),
            "compact_halo_adequate_systems": [name for name, adequate in halo_adequacy.items() if adequate],
        },
    }
    output = (ROOT / diagnostic["outputs"]["report"]).parent
    output.mkdir(parents=True, exist_ok=True)
    (ROOT / diagnostic["outputs"]["report"]).write_text(json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(grid_rows).to_csv(ROOT / diagnostic["outputs"]["grid"], index=False)
    pd.concat(predictions, ignore_index=True).to_csv(ROOT / diagnostic["outputs"]["predictions"], index=False)
    lines = ["# Multi-cluster failure diagnostic", "", diagnostic["claim_boundary"], "", "| model | locked RMS | amplitude-diagnostic RMS | all roots | selected amplitudes | normalization rescue | universal amplitude |", "|---|---:|---:|---|---|---|---|"]
    for model in diagnostic["candidate_models"]:
        result = diagnoses[model]
        amplitudes = ", ".join(f"{value:g}" for value in result["selected_amplitudes"])
        lines.append(f"| {model} | {result['locked_RMS_arcsec']:.3f} | {result['amplitude_diagnostic_RMS_arcsec']:.3f} | {aggregates[model]['all_roots_converged']} | {amplitudes} | {result['normalization_rescue']} | {result['amplitude_consistent_with_universal']} |")
    lines += ["", "Compact one-halo geometry adequate systems: " + (", ".join(report["verdict"]["compact_halo_adequate_systems"]) or "none") + "."]
    (ROOT / diagnostic["outputs"]["summary"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((ROOT / diagnostic["outputs"]["summary"]).read_text())


if __name__ == "__main__":
    main()
