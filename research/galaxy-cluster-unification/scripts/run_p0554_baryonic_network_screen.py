#!/usr/bin/env python3
"""Screen explicit baryon-to-baryon propagation kernels on spent raw lenses."""

from __future__ import annotations

import argparse
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

from run_adaptive_route_multicluster_raw import route_fraction  # noqa: E402
from run_adaptive_route_raw_rxj2129 import baryon_field  # noqa: E402
from run_arc_invariant_absolute_lensing import raw_field  # noqa: E402
from run_clash_stellar_morphology_response import MorphologyLens  # noqa: E402
from run_p0554_local_cross_domain_sensitivity import A0, json_safe, raw_contexts, sha256  # noqa: E402
from run_p0554_route_localization_screen import (  # noqa: E402
    build_directional_field,
    geometry_for,
    linearized_residuals,
    rms,
)
from run_p0554_route_softness_interaction import build_variants, load_route_sources  # noqa: E402
from voidscreen.adaptive_route_kernel import (  # noqa: E402
    adaptive_route_parameters,
    transformed_source_weights,
)
from voidscreen.route_template import conservative_network_route_template, weighted_radius  # noqa: E402
from voidscreen.stellar_morphology_lensing import build_stellar_morphology_deflection_field  # noqa: E402


def expanded_variant(parent, item):
    result = dict(parent)
    for key in parent:
        if key in item:
            result[key] = item[key]
    result["variant_id"] = item["variant_id"]
    result["mode"] = item["mode"]
    result["coordinate"] = item["coordinate"]
    result["coordinate_value"] = item["coordinate_value"]
    return result


def build_network_field(interaction, route_protocol, context, sources, parent, radial, baryons, variant):
    scale = float(context.local["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
    xy = sources[["x_arcsec", "y_arcsec"]].to_numpy(float)
    weights = transformed_source_weights(
        sources.base_weight.to_numpy(float), float(parent.candidate.source_weight_power)
    )
    radius_kpc = np.hypot(xy[:, 0], xy[:, 1]) * scale
    r50 = weighted_radius(radius_kpc, weights, 0.5)
    r80 = weighted_radius(radius_kpc, weights, 0.8)
    concentration = r50 / r80
    adaptive = adaptive_route_parameters(
        r50_kpc=r50,
        concentration=concentration,
        source_weights=weights,
        feature=str(parent.candidate.feature),
        base_fraction=float(parent.candidate.base_fraction),
        extent_slope=float(parent.candidate.extent_slope),
        base_length_kpc=float(parent.candidate.base_length_kpc),
        length_power=float(parent.candidate.length_power),
        base_width_kpc=float(parent.candidate.base_width_kpc),
        width_power=float(parent.candidate.width_power),
        gate_power=float(parent.candidate.gate_power),
    )
    translation = route_protocol["route_to_deflection_translation"]
    axis = np.arange(-255.5, 256.0, float(translation["grid_spacing_arcsec"]))
    route_map, route_audit = conservative_network_route_template(
        axis,
        xy,
        weights,
        routing_fraction=adaptive["routing_fraction"],
        target_weight_power=float(variant["target_weight_power"]),
        distance_power=float(variant["distance_power"]),
        softening=float(variant["softening_kpc"]) / scale,
        link_scale=float(variant["link_scale_kpc"]) / scale,
        hop_fraction=float(variant["hop_fraction"]),
        smoothing=adaptive["width_kpc"] / scale,
        top_k=None if variant["top_k"] is None else int(variant["top_k"]),
    )

    def carrier_alpha(radius_arcsec):
        return radial.reduced_alpha_arcsec(radius_arcsec, 1.0) - baryons.reduced_alpha_arcsec(radius_arcsec, 1.0)

    field = build_stellar_morphology_deflection_field(
        axis,
        route_map,
        carrier_alpha,
        contrast_cap=float(interaction["route_parent"]["contrast_cap"]),
        contrast_strength=1.0,
        annulus_width_arcsec=float(translation["annulus_width_arcsec"]),
        taper_inner_arcsec=float(translation["taper_inner_arcsec"]),
        support_radius_arcsec=float(translation["support_radius_arcsec"]),
        radial_samples=2048,
        circular_radii=512,
        circular_azimuths=720,
    )
    return field, {
        **adaptive,
        "r50_kpc": r50,
        "r80_kpc": r80,
        "concentration_r50_over_r80": concentration,
        "network_branch_count": route_audit["branch_count"],
        "network_mean_route_length_kpc": route_audit["mean_route_length"] * scale,
        "network_rms_route_length_kpc": route_audit["rms_route_length"] * scale,
        "network_mean_source_entropy": route_audit["mean_source_transition_entropy"],
        "network_effective_receiving_targets": route_audit["effective_receiving_targets"],
        "network_largest_receiving_fraction": route_audit["largest_receiving_fraction"],
        "route_map_normalization_error": route_audit["normalization_error"],
        **field.audit,
    }


def summarize(protocol, config_path, output):
    systems = pd.read_csv(output / protocol["outputs"]["system_scores"])
    audits = pd.read_csv(output / protocol["outputs"]["field_audits"])
    baseline = systems[systems.variant_id.eq("eta_000")].set_index("system_label")
    local = systems[systems.variant_id.eq("local_vector_s200")].set_index("system_label")
    rows = []
    for variant_id, block in systems.groupby("variant_id", sort=False):
        indexed = block.set_index("system_label")
        value = rms(indexed.heldout_linearized_RMS_arcsec)
        base_value = rms(baseline.heldout_linearized_RMS_arcsec)
        local_value = rms(local.heldout_linearized_RMS_arcsec)
        changes = 1.0 - indexed.heldout_linearized_RMS_arcsec / baseline.heldout_linearized_RMS_arcsec
        metadata = block.iloc[0]
        rows.append(
            {
                "variant_id": variant_id,
                "mode": metadata["mode"],
                "coordinate": metadata["coordinate"],
                "coordinate_value": metadata["coordinate_value"],
                "equal_system_RMS_arcsec": value,
                "improvement_fraction_vs_eta0": 1.0 - value / base_value,
                "improvement_fraction_vs_local_vector": 1.0 - value / local_value,
                "systems_improved_vs_eta0": int(changes.gt(0.0).sum()),
                "minimum_system_improvement_fraction": float(changes.min()),
                "maximum_system_improvement_fraction": float(changes.max()),
            }
        )
    scores = pd.DataFrame(rows).sort_values("equal_system_RMS_arcsec")
    network = scores[scores["mode"].eq("network")]
    best = network.iloc[0]
    gate = protocol["evaluation"]
    shortlist = network[
        network.improvement_fraction_vs_eta0.ge(float(gate["exact_shortlist_minimum_improvement_fraction_vs_eta0"]))
        & network.systems_improved_vs_eta0.ge(int(gate["exact_shortlist_minimum_systems_improved"]))
        & network.minimum_system_improvement_fraction.ge(-float(gate["exact_shortlist_maximum_single_system_worsening_fraction"]))
    ].copy()
    impacts = []
    parent_rms = float(scores[scores.variant_id.eq("network_parent")].equal_system_RMS_arcsec.iloc[0])
    for coordinate, block in network[~network.coordinate.isin(["parent"])].groupby("coordinate"):
        combined = pd.concat(
            [block, scores[scores.variant_id.eq("network_parent")]], ignore_index=True
        ).drop_duplicates("variant_id")
        best_coordinate = combined.sort_values("equal_system_RMS_arcsec").iloc[0]
        impacts.append(
            {
                "coordinate": coordinate,
                "tested_values": "+".join(str(value) for value in sorted(block.coordinate_value.tolist())),
                "profile_RMS_span_arcsec": float(combined.equal_system_RMS_arcsec.max() - combined.equal_system_RMS_arcsec.min()),
                "best_variant_id": best_coordinate.variant_id,
                "best_value": best_coordinate.coordinate_value,
                "best_improvement_fraction_vs_network_parent": 1.0 - float(best_coordinate.equal_system_RMS_arcsec) / parent_rms,
            }
        )
    impacts = pd.DataFrame(impacts).sort_values("profile_RMS_span_arcsec", ascending=False)
    scores.to_csv(output / protocol["outputs"]["scores"], index=False)
    shortlist.to_csv(output / protocol["outputs"]["shortlist"], index=False)
    impacts.to_csv(output / protocol["outputs"]["coordinate_impacts"], index=False)

    cross = pd.read_csv(ROOT / "results/p0554_route_softness_interaction/variant_scores.csv").set_index("variant_id").loc["lensing_softness_098"]
    report = {
        "report_version": "P0554-BARYONIC-NETWORK-SCREEN-RESULTS-0.1.0",
        "status": "complete",
        "protocol": {"path": str(config_path.relative_to(ROOT)), "sha256": sha256(config_path)},
        "coverage": {
            "variants": len(scores),
            "network_variants": len(network),
            "systems": int(systems.system_label.nunique()),
            "variant_system_scores": len(systems),
            "route_fields": len(audits),
        },
        "references": {
            "eta0": scores[scores.variant_id.eq("eta_000")].iloc[0].to_dict(),
            "local_vector_s200": scores[scores.variant_id.eq("local_vector_s200")].iloc[0].to_dict(),
            "network_parent": scores[scores.variant_id.eq("network_parent")].iloc[0].to_dict(),
        },
        "best_network": best.to_dict(),
        "coordinate_impacts": impacts.to_dict("records"),
        "shortlist": shortlist.to_dict("records"),
        "descriptive_holdout_transfer_variant": str(best.variant_id),
        "cross_domain_preservation": {
            "galaxy_outer_RMSE_km_s": float(cross.galaxy_outer_RMSE_km_s),
            "CLASH_radial_RMSE_dex": float(cross.cluster_RMSE_dex),
            "Mercury_precession_mas_per_century": float(cross.Mercury_precession_mas_per_century),
            "all_solar_proxies_pass": bool(cross.all_solar_proxies_pass),
            "interpretation": "unchanged because this screen changes only a conservative zero-circular-monopole angular term",
        },
        "field_invariants": {
            "maximum_route_map_normalization_error": float(audits.route_map_normalization_error.max()),
            "maximum_annular_convergence_error": float(audits.maximum_annular_convergence_mean_fraction.max()),
            "maximum_normalized_curl_RMS": float(audits.normalized_curl_RMS.max()),
        },
        "verdict": {
            "best_network_beats_eta0": bool(best.improvement_fraction_vs_eta0 > 0.0),
            "best_network_beats_local_vector": bool(best.improvement_fraction_vs_local_vector > 0.0),
            "any_variant_meets_exact_shortlist": not shortlist.empty,
            "no_formula_promoted": True,
        },
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )

    ordered = scores.sort_values("improvement_fraction_vs_eta0")
    pivot = systems.pivot(index="variant_id", columns="system_label", values="heldout_linearized_RMS_arcsec")
    changes = 100.0 * (1.0 - pivot.div(pivot.loc["eta_000"], axis=1))
    changes = changes.loc[ordered.variant_id]
    fig, axes = plt.subplots(1, 2, figsize=(17, 7), constrained_layout=True)
    axes[0].barh(ordered.variant_id, 100.0 * ordered.improvement_fraction_vs_eta0, color=np.where(ordered.improvement_fraction_vs_eta0 > 0.0, "tab:green", "crimson"))
    axes[0].axvline(0.0, color="black", linewidth=0.8)
    axes[0].set(xlabel="five-cluster improvement versus eta=0 (%)", title="Explicit baryonic-network kernel screen")
    image = axes[1].imshow(changes, aspect="auto", cmap="RdYlGn", vmin=-2.0, vmax=2.0)
    axes[1].set(xticks=np.arange(len(changes.columns)), xticklabels=changes.columns, yticks=np.arange(len(changes.index)), yticklabels=changes.index, title="Per-cluster response (%)")
    axes[1].tick_params(axis="x", rotation=30)
    fig.colorbar(image, ax=axes[1], label="improvement versus eta=0 (%)")
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)
    summary = (
        "# P0554 baryonic-network screen\n\n"
        f"The best explicit network is `{best.variant_id}`, changing the five-cluster fixed-geometry RMS by "
        f"{100.0*best.improvement_fraction_vs_eta0:+.3f}% versus no route and "
        f"{100.0*best.improvement_fraction_vs_local_vector:+.3f}% versus the single-vector local route. "
        f"{len(shortlist)} variants pass the frozen exact-followup gate.\n"
    )
    (output / protocol["outputs"]["summary"]).write_text(summary, encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--postprocess-only", action="store_true")
    args = parser.parse_args()
    config_path = ROOT / "configs/p0554_baryonic_network_screen_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    if args.postprocess_only:
        print(json.dumps(json_safe(summarize(protocol, config_path, output)["verdict"]), indent=2))
        return

    interaction = json.loads((ROOT / protocol["inputs"]["interaction_protocol"]).read_text(encoding="utf-8"))
    parent = next(item for item in build_variants(interaction) if item.variant_id == "combined_parent")
    contexts = raw_contexts(interaction)
    route_sources, route_protocols = load_route_sources(interaction, contexts)
    geometry = pd.read_csv(ROOT / protocol["inputs"]["fixed_geometry"])
    variants = [expanded_variant(protocol["network_parent"], item) for item in protocol["variants"]]
    system_rows, image_frames, audit_rows = [], [], []
    for context in contexts:
        parameters = geometry_for(geometry, context.label)
        radial, _ = raw_field(parent.spec, parent.q, context.anchors, context.local, A0)
        baryons = baryon_field(context.anchors, context.local)
        adaptive = route_fraction(parent.candidate, route_sources[context.label], context.local)
        strength_parent = 0.30 * float(adaptive["routing_fraction"] ** parent.route_power)
        for variant in variants:
            print(f"{context.label}: {variant['variant_id']}", flush=True)
            if variant["mode"] == "none":
                morphology, audit, strength = None, None, 0.0
            elif variant["mode"] == "local_vector":
                morphology, audit = build_directional_field(
                    interaction,
                    route_protocols[context.label],
                    context,
                    route_sources[context.label],
                    parent,
                    radial,
                    baryons,
                    {
                        "local_mix": 1.0,
                        "softening_kpc": 200.0,
                        "neighbor_weight_power": 1.0,
                        "distance_power": 2.0,
                        "symmetric_bend_degrees": 0.0,
                        "center_mode": "light_centroid",
                    },
                )
                strength = strength_parent
            else:
                morphology, audit = build_network_field(
                    interaction,
                    route_protocols[context.label],
                    context,
                    route_sources[context.label],
                    parent,
                    radial,
                    baryons,
                    variant,
                )
                strength = strength_parent
            if audit is not None:
                audit_rows.append(
                    {
                        "system_label": context.label,
                        "variant_id": variant["variant_id"],
                        "mode": variant["mode"],
                        **{key: value for key, value in variant.items() if key not in {"variant_id", "mode"}},
                        **audit,
                    }
                )
            lens = MorphologyLens(context.local, {parent.variant_id: radial}, parent=parent.variant_id, morphology=morphology, fraction=strength)
            _, sources = lens.profiled_residuals(parent.variant_id, parameters, context.training)
            training = linearized_residuals(lens, parent.variant_id, parameters, sources, context.training, "training")
            heldout = linearized_residuals(lens, parent.variant_id, parameters, sources, context.heldout, "heldout")
            combined = pd.concat([training, heldout], ignore_index=True)
            combined.insert(0, "variant_id", variant["variant_id"])
            combined.insert(0, "system_label", context.label)
            image_frames.append(combined)
            system_rows.append(
                {
                    "system_label": context.label,
                    "variant_id": variant["variant_id"],
                    "mode": variant["mode"],
                    "coordinate": variant["coordinate"],
                    "coordinate_value": variant["coordinate_value"],
                    "training_images": len(training),
                    "training_linearized_RMS_arcsec": rms(training.linearized_radial_residual_arcsec),
                    "heldout_images": len(heldout),
                    "heldout_linearized_RMS_arcsec": rms(heldout.linearized_radial_residual_arcsec),
                    "applied_angular_strength": strength,
                }
            )
    pd.DataFrame(system_rows).to_csv(output / protocol["outputs"]["system_scores"], index=False)
    pd.concat(image_frames, ignore_index=True).to_csv(output / protocol["outputs"]["image_residuals"], index=False)
    pd.DataFrame(audit_rows).to_csv(output / protocol["outputs"]["field_audits"], index=False)
    report = summarize(protocol, config_path, output)
    print(json.dumps(json_safe({"best": report["best_network"], "coordinate_impacts": report["coordinate_impacts"], "verdict": report["verdict"]}), indent=2), flush=True)


if __name__ == "__main__":
    main()
