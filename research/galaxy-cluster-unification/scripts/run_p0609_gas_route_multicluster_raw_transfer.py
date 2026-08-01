#!/usr/bin/env python3
"""Transfer the locked RX J2129 gas route to four other raw CLASH lenses."""

from __future__ import annotations

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

from run_adaptive_route_raw_rxj2129 import MODEL, baryon_field, exact_fit  # noqa: E402
from run_clash_stellar_morphology_response import MorphologyLens  # noqa: E402
from run_p0554_all_baryon_route_screen import prepare_xray_maps  # noqa: E402
from run_p0554_local_cross_domain_sensitivity import raw_contexts  # noqa: E402
from run_p0554_route_localization_screen import geometry_for  # noqa: E402
from run_p0554_route_softness_interaction import load_route_sources  # noqa: E402
from run_p0601_frozen_potential_raw_lensing import build_fields as build_p0599_fields, json_safe  # noqa: E402
from run_p0607_component_direction_raw_lensing import component_fields  # noqa: E402
from run_p0608_route_redshift_tomography import TomographicRouteLens  # noqa: E402
from voidscreen.baryon_morphology import map_attraction_directions  # noqa: E402


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(scores, variants, systems):
    rows = []
    for variant in variants:
        block = scores[scores.variant_id.eq(variant)]
        complete = block[
            block.heldout_roots_converged.eq(block.heldout_images)
            & np.isfinite(block.heldout_RMS_arcsec)
        ]
        rows.append(
            {
                "variant_id": variant,
                "systems": len(systems),
                "complete_systems": len(complete),
                "all_systems_complete": len(complete) == len(systems),
                "equal_system_heldout_RMS_arcsec": float(np.sqrt(np.mean(np.square(complete.heldout_RMS_arcsec))))
                if len(complete)
                else np.inf,
                "median_system_heldout_RMS_arcsec": float(complete.heldout_RMS_arcsec.median())
                if len(complete)
                else np.inf,
            }
        )
    return pd.DataFrame(rows)


def matched_comparison(scores, candidate, systems):
    baseline = scores[scores.variant_id.eq("P0599_no_route")].set_index("system_label")
    route = scores[scores.variant_id.eq(candidate)].set_index("system_label")
    common = []
    for label in systems:
        if (
            int(baseline.loc[label, "heldout_roots_converged"]) == int(baseline.loc[label, "heldout_images"])
            and int(route.loc[label, "heldout_roots_converged"]) == int(route.loc[label, "heldout_images"])
            and np.isfinite(baseline.loc[label, "heldout_RMS_arcsec"])
            and np.isfinite(route.loc[label, "heldout_RMS_arcsec"])
        ):
            common.append(label)
    if not common:
        return {
            "variant_id": candidate,
            "matched_systems": 0,
            "matched_labels": [],
            "reference_RMS_arcsec": np.inf,
            "candidate_RMS_arcsec": np.inf,
            "fractional_improvement": -np.inf,
            "systems_improved": 0,
        }
    reference = float(np.sqrt(np.mean(np.square(baseline.loc[common, "heldout_RMS_arcsec"]))))
    value = float(np.sqrt(np.mean(np.square(route.loc[common, "heldout_RMS_arcsec"]))))
    return {
        "variant_id": candidate,
        "matched_systems": len(common),
        "matched_labels": common,
        "reference_RMS_arcsec": reference,
        "candidate_RMS_arcsec": value,
        "fractional_improvement": 1.0 - value / reference,
        "systems_improved": int(
            np.sum(route.loc[common, "heldout_RMS_arcsec"] < baseline.loc[common, "heldout_RMS_arcsec"])
        ),
    }


def main() -> None:
    config_path = ROOT / "configs/p0609_gas_route_multicluster_raw_transfer_protocol.json"
    protocol = read_json(config_path)
    if not protocol["status"].startswith("frozen_"):
        raise RuntimeError("P0609 protocol is not frozen")
    inputs = protocol["inputs"]
    adequacy = read_json(ROOT / inputs["component_input_audit"])
    if not adequacy["input_adequacy_pass"]:
        raise RuntimeError("registered component maps failed their frozen input audit")
    p0608 = read_json(ROOT / inputs["P0608_report"])
    locked = protocol["locked_route"]
    if p0608["locked_route"]["component"] != "gas" or not np.isclose(
        p0608["locked_route"]["angular_strength"], locked["angular_strength"]
    ):
        raise RuntimeError("locked RX J2129 gas route changed")

    p0601 = read_json(ROOT / inputs["P0601_protocol"])
    p0607 = read_json(ROOT / inputs["P0607_protocol"])
    interaction = read_json(ROOT / inputs["interaction_protocol"])
    screen_protocol = read_json(ROOT / inputs["component_screen_protocol"])
    acquisition = read_json(ROOT / inputs["component_acquisition_protocol"])
    all_contexts = raw_contexts(interaction)
    contexts = [context for context in all_contexts if context.label in protocol["systems"]]
    if [context.label for context in contexts] != protocol["systems"]:
        raise RuntimeError("raw context ordering or coverage changed")
    route_sources, _ = load_route_sources(interaction, all_contexts)
    geometry = pd.read_csv(ROOT / inputs["fixed_geometry"])
    settings = screen_protocol["map_construction"]
    map_axis = np.arange(
        float(settings["axis_min_arcsec"]),
        float(settings["axis_max_arcsec"]) + 0.5 * float(settings["grid_spacing_arcsec"]),
        float(settings["grid_spacing_arcsec"]),
    )

    score_rows, prediction_frames, field_rows = [], [], []
    for system_index, context in enumerate(contexts):
        print(f"system {context.label}", flush=True)
        local_for_p0599 = json.loads(json.dumps(context.local))
        image_rows = pd.concat([context.training, context.heldout], ignore_index=True)
        image_radius_kpc = np.hypot(
            image_rows.x_arcsec.to_numpy(float), image_rows.y_arcsec.to_numpy(float)
        ) * float(context.local["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
        local_for_p0599.setdefault("baryonic_inputs", {})[
            "strong_lens_impact_radius_range_kpc_expected"
        ] = [float(np.min(image_radius_kpc)), float(np.max(image_radius_kpc))]
        radial_fields, _, radial_diagnostic = build_p0599_fields(
            context.anchors, local_for_p0599, p0601["constants"]
        )
        parent = radial_fields["P0599_potential_shape"]
        baryons = baryon_field(context.anchors, context.local)
        sources = route_sources[context.label]
        _, gas_map, gas_audit = prepare_xray_maps(
            screen_protocol, acquisition, context, map_axis
        )
        xy = sources[["x_arcsec", "y_arcsec"]].to_numpy(float)
        scale = float(context.local["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
        route_settings = p0607["route_geometry"]
        gas_direction, gas_direction_audit = map_attraction_directions(
            map_axis,
            gas_map,
            xy,
            softening=float(route_settings["direction_softening_kpc"]) / scale,
            distance_power=float(route_settings["direction_distance_power"]),
        )
        fields, audits, realized = component_fields(
            p0607, context.local, sources, parent, baryons, {"gas": gas_direction}
        )
        gas_field = fields["gas"]
        field_rows.append(
            {
                "system_label": context.label,
                "members": len(sources),
                "Chandra_exposure_ks": float(gas_audit["total_exposure_ks"]),
                "gas_centroid_x_arcsec": float(gas_direction_audit["map_centroid"][0]),
                "gas_centroid_y_arcsec": float(gas_direction_audit["map_centroid"][1]),
                "R80_kpc": realized["R80_kpc"],
                "P0599_shape_gate": radial_diagnostic["shape_gate"],
                **audits.iloc[0].to_dict(),
            }
        )
        initial = geometry_for(geometry, context.label)
        for variant_index, variant in enumerate(protocol["variants"]):
            if not variant["route"]:
                lens = MorphologyLens(
                    context.local,
                    {MODEL: parent},
                    parent=MODEL,
                    morphology=None,
                    fraction=0.0,
                )
            else:
                lens = TomographicRouteLens(
                    context.local,
                    {MODEL: parent},
                    parent=MODEL,
                    morphology=gas_field,
                    strength=float(locked["angular_strength"]),
                    gamma=float(variant["gamma"]),
                )
            try:
                fitted = exact_fit(
                    lens,
                    context.training,
                    context.heldout,
                    initial=initial,
                    starts=int(protocol["fit"]["optimization_starts_per_variant_cluster"]),
                    seed=int(protocol["fit"]["seed"]) + 100 * system_index + variant_index,
                )
                joined = pd.concat(
                    [fitted["training_prediction"], fitted["heldout_prediction"]],
                    ignore_index=True,
                )
                joined["system_label"] = context.label
                joined["variant_id"] = variant["variant_id"]
                prediction_frames.append(joined)
                score_rows.append(
                    {
                        "system_label": context.label,
                        "variant_id": variant["variant_id"],
                        "gamma": float(variant.get("gamma", np.nan)),
                        "training_images": len(context.training),
                        "heldout_images": len(context.heldout),
                        "training_RMS_arcsec": fitted["training_score"]["exact_radial_RMS_arcsec"],
                        "training_roots_converged": fitted["training_score"]["converged_roots"],
                        "heldout_RMS_arcsec": fitted["heldout_score"]["exact_radial_RMS_arcsec"],
                        "heldout_roots_converged": fitted["heldout_score"]["converged_roots"],
                        "optimizer_cost": fitted["optimizer_cost"],
                    }
                )
            except Exception as error:
                score_rows.append(
                    {
                        "system_label": context.label,
                        "variant_id": variant["variant_id"],
                        "gamma": float(variant.get("gamma", np.nan)),
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
    scores = pd.DataFrame(score_rows)
    variants = [variant["variant_id"] for variant in protocol["variants"]]
    aggregates = aggregate(scores, variants, protocol["systems"])
    comparisons = [
        matched_comparison(scores, candidate, protocol["systems"])
        for candidate in ("gas_route_gamma0", "gas_route_gamma1")
    ]
    standard_comparison = next(row for row in comparisons if row["variant_id"] == "gas_route_gamma1")
    gates = protocol["transfer_gates"]
    standard_aggregate = aggregates[aggregates.variant_id.eq("gas_route_gamma1")].iloc[0]
    gate_audit = {
        "all_heldout_roots_pass": bool(standard_aggregate.all_systems_complete),
        "matched_improvement_fraction": standard_comparison["fractional_improvement"],
        "matched_improvement_pass": bool(
            standard_comparison["fractional_improvement"]
            >= float(gates["matched_equal_system_improvement_vs_no_route_min"])
        ),
        "systems_improved": standard_comparison["systems_improved"],
        "systems_improved_pass": bool(
            standard_comparison["systems_improved"] >= int(gates["systems_improved_min"])
        ),
        "absolute_RMS_arcsec": float(standard_aggregate.equal_system_heldout_RMS_arcsec),
        "absolute_RMS_pass": bool(
            float(standard_aggregate.equal_system_heldout_RMS_arcsec)
            <= float(gates["maximum_equal_system_heldout_RMS_arcsec"])
        ),
    }
    gate_audit["all_gates_pass"] = all(
        value for key, value in gate_audit.items() if key.endswith("_pass")
    )

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / protocol["outputs"]["system_scores"], index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        output / protocol["outputs"]["predictions"], index=False
    )
    pd.DataFrame(field_rows).to_csv(output / protocol["outputs"]["field_audits"], index=False)
    aggregates.to_csv(output / protocol["outputs"]["aggregate_scores"], index=False)
    report = {
        "report_version": "P0609-GAS-ROUTE-MULTICLUSTER-RAW-TRANSFER-RESULTS-0.1.0",
        "status": "complete_locked_four_cluster_spent_transfer",
        "coverage": {
            "systems": len(contexts),
            "variants": len(variants),
            "variant_system_refits": len(scores),
            "training_images": int(sum(len(context.training) for context in contexts)),
            "heldout_images": int(sum(len(context.heldout) for context in contexts)),
        },
        "locked_route": locked,
        "aggregate_scores": aggregates.to_dict("records"),
        "matched_comparisons": comparisons,
        "per_system": scores.to_dict("records"),
        "transfer_gate": gate_audit,
        "cross_domain_controls": protocol["cross_domain_controls"],
        "interpretation": {
            "standard_gas_route_transfers": bool(gate_audit["all_gates_pass"]),
            "gamma0_beats_gamma1_equal_system": bool(
                float(aggregates[aggregates.variant_id.eq("gas_route_gamma0")].equal_system_heldout_RMS_arcsec.iloc[0])
                < float(standard_aggregate.equal_system_heldout_RMS_arcsec)
            ),
            "absolute_gas_mass_claimed": False,
        },
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    width = 0.25
    x = np.arange(len(protocol["systems"]))
    for index, variant in enumerate(variants):
        block = scores[scores.variant_id.eq(variant)].set_index("system_label").loc[protocol["systems"]]
        axes[0].bar(x + (index - 1) * width, block.heldout_RMS_arcsec, width=width, label=variant)
    axes[0].set(xticks=x, xticklabels=protocol["systems"], ylabel="held-out RMS (arcsec)", title="Locked raw transfer by cluster")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend(fontsize=7)
    axes[1].barh(aggregates.variant_id, aggregates.equal_system_heldout_RMS_arcsec, color=["gray", "#1261A0", "#55A868"])
    axes[1].axvline(float(gates["maximum_equal_system_heldout_RMS_arcsec"]), color="black", ls="--", label="2 arcsec gate")
    axes[1].set(xlabel="equal-system held-out RMS (arcsec)", title="Four-cluster aggregate")
    axes[1].legend(fontsize=7)
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)

    (output / protocol["outputs"]["summary"]).write_text(
        "# P0609 locked gas-route raw transfer\n\n"
        f"The standard gamma=1 route has matched equal-system improvement {standard_comparison['fractional_improvement']:+.3%} versus P0599 without the route, improves {standard_comparison['systems_improved']}/{standard_comparison['matched_systems']} matched systems, and has aggregate RMS {standard_aggregate.equal_system_heldout_RMS_arcsec:.4f} arcsec.\n\n"
        f"Transfer gate passed: **{gate_audit['all_gates_pass']}**. These are spent clusters and the gas map supplies direction, not mass.\n",
        encoding="utf-8",
    )
    print(json.dumps(json_safe({"aggregate": report["aggregate_scores"], "matched": report["matched_comparisons"], "gate": report["transfer_gate"], "interpretation": report["interpretation"]}), indent=2))


if __name__ == "__main__":
    main()
