#!/usr/bin/env python3
"""Screen baryon-only route localization changes at fixed lens geometry."""

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
from run_p0554_route_softness_interaction import build_variants, load_route_sources  # noqa: E402
from run_rxj2129_raw_theory_lensing import FIXED_LABELS  # noqa: E402
from voidscreen.adaptive_route_kernel import (  # noqa: E402
    adaptive_route_parameters,
    transformed_source_weights,
)
from voidscreen.route_template import (  # noqa: E402
    conservative_directional_route_template,
    weighted_radius,
)
from voidscreen.stellar_morphology_lensing import (  # noqa: E402
    build_stellar_morphology_deflection_field,
)


def geometry_for(frame, system_label):
    row = frame[
        frame.system_label.eq(system_label) & frame.variant_id.eq("eta_000")
    ].iloc[0]
    return row[list(FIXED_LABELS)].to_numpy(float)


def build_directional_field(
    interaction,
    route_protocol,
    context,
    sources,
    parent,
    radial,
    baryons,
    variant,
):
    scale = float(context.local["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
    xy = sources[["x_arcsec", "y_arcsec"]].to_numpy(float)
    weights = transformed_source_weights(
        sources.base_weight.to_numpy(float), float(parent.candidate.source_weight_power)
    )
    neighbor_weights = transformed_source_weights(
        sources.base_weight.to_numpy(float),
        float(variant.get("neighbor_weight_power", parent.candidate.source_weight_power)),
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
    spacing = float(translation["grid_spacing_arcsec"])
    axis = np.arange(-255.5, 256.0, spacing)
    center = (
        np.zeros(2)
        if variant["center_mode"] == "cluster_origin"
        else None
    )
    route_map, route_audit = conservative_directional_route_template(
        axis,
        xy,
        weights,
        routing_fraction=adaptive["routing_fraction"],
        return_scale=adaptive["return_scale_kpc"] / scale,
        radius_exponent=float(translation["source_radius_exponent"]),
        reference_radius=float(translation["source_reference_radius_kpc"]) / scale,
        smoothing=adaptive["width_kpc"] / scale,
        local_mix=float(variant["local_mix"]),
        softening=float(variant["softening_kpc"]) / scale,
        distance_power=float(variant.get("distance_power", 2.0)),
        neighbor_weights=neighbor_weights,
        symmetric_bend_degrees=float(variant["symmetric_bend_degrees"]),
        center=center,
    )

    def carrier_alpha(radius_arcsec):
        return radial.reduced_alpha_arcsec(
            radius_arcsec, 1.0
        ) - baryons.reduced_alpha_arcsec(radius_arcsec, 1.0)

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
        "mean_global_local_alignment": route_audit["mean_global_local_alignment"],
        "route_map_normalization_error": route_audit["normalization_error"],
        **field.audit,
    }


def linearized_residuals(lens, model, parameters, sources, rows, stage):
    records = []
    for row in rows.itertuples(index=False):
        observed = np.asarray([row.x_arcsec, row.y_arcsec], dtype=float)
        redshift = float(row.source_redshift)
        beta_x, beta_y = lens.ray_shooting(
            model,
            parameters,
            np.asarray([observed[0]]),
            np.asarray([observed[1]]),
            redshift,
        )
        delta_beta = np.asarray([beta_x[0], beta_y[0]]) - sources[int(row.source_family)]
        jacobian = lens.jacobian(
            model,
            parameters,
            np.asarray([observed[0]]),
            np.asarray([observed[1]]),
            redshift,
        )[0]
        delta_theta = np.linalg.pinv(jacobian, rcond=1.0e-9) @ delta_beta
        records.append(
            {
                "stage": stage,
                "image_id": str(row.image_id),
                "source_family": int(row.source_family),
                "source_redshift": redshift,
                "observed_x_arcsec": observed[0],
                "observed_y_arcsec": observed[1],
                "linearized_delta_x_arcsec": float(delta_theta[0]),
                "linearized_delta_y_arcsec": float(delta_theta[1]),
                "linearized_radial_residual_arcsec": float(np.linalg.norm(delta_theta)),
                "source_plane_closure_arcsec": float(np.linalg.norm(delta_beta)),
            }
        )
    return pd.DataFrame(records)


def rms(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(np.square(values))))


def summarize(protocol, config_path, output):
    system_scores = pd.read_csv(output / protocol["outputs"]["system_scores"])
    audits = pd.read_csv(output / protocol["outputs"]["field_audits"])
    baseline = system_scores[system_scores.variant_id.eq("eta_000")].set_index("system_label")
    primary_labels = [
        label
        for label in protocol["evaluation"]["systems"]
        if label != protocol["evaluation"]["spent_selection_system"]
    ]
    rows = []
    for variant_id, block in system_scores.groupby("variant_id", sort=False):
        indexed = block.set_index("system_label")
        primary_rms = rms(indexed.loc[primary_labels, "heldout_linearized_RMS_arcsec"])
        primary_base = rms(baseline.loc[primary_labels, "heldout_linearized_RMS_arcsec"])
        all_rms = rms(indexed.heldout_linearized_RMS_arcsec)
        all_base = rms(baseline.heldout_linearized_RMS_arcsec)
        changes = 1.0 - (
            indexed.loc[primary_labels, "heldout_linearized_RMS_arcsec"]
            / baseline.loc[primary_labels, "heldout_linearized_RMS_arcsec"]
        )
        rows.append(
            {
                "variant_id": variant_id,
                "primary_other_four_equal_system_RMS_arcsec": primary_rms,
                "primary_improvement_fraction_vs_eta0": 1.0 - primary_rms / primary_base,
                "all_five_equal_system_RMS_arcsec": all_rms,
                "all_five_improvement_fraction_vs_eta0": 1.0 - all_rms / all_base,
                "primary_systems_improved": int(changes.gt(0).sum()),
                "primary_direction_consistency": float(changes.gt(0).mean()),
                "minimum_primary_system_improvement_fraction": float(changes.min()),
                "maximum_primary_system_improvement_fraction": float(changes.max()),
            }
        )
    scores = pd.DataFrame(rows).sort_values("primary_other_four_equal_system_RMS_arcsec")
    nonbaseline = scores[~scores.variant_id.eq("eta_000")]
    best = nonbaseline.iloc[0]
    most_consistent = nonbaseline.sort_values(
        ["primary_systems_improved", "primary_other_four_equal_system_RMS_arcsec"],
        ascending=[False, True],
    ).iloc[0]
    shortlist_ids = list(dict.fromkeys([best.variant_id, most_consistent.variant_id]))
    shortlist = scores[scores.variant_id.isin(shortlist_ids)].copy()
    shortlist["selection_reason"] = shortlist.variant_id.map(
        lambda value: "+".join(
            reason
            for condition, reason in (
                (value == best.variant_id, "lowest_primary_RMS"),
                (value == most_consistent.variant_id, "most_direction_consistent"),
            )
            if condition
        )
    )
    scores.to_csv(output / protocol["outputs"]["scores"], index=False)
    shortlist.to_csv(output / protocol["outputs"]["shortlist"], index=False)

    cross_domain = pd.read_csv(
        ROOT / "results/p0554_route_softness_interaction/variant_scores.csv"
    ).set_index("variant_id").loc["lensing_softness_098"]
    report = {
        "report_version": "P0554-ROUTE-LOCALIZATION-SCREEN-RESULTS-0.1.0",
        "status": "complete",
        "protocol": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": sha256(config_path),
        },
        "coverage": {
            "variants": len(scores),
            "systems": int(system_scores.system_label.nunique()),
            "variant_system_scores": len(system_scores),
            "images": int(protocol["evaluation"]["images"]),
            "source_families": int(protocol["evaluation"]["source_families"]),
            "route_fields": len(audits),
        },
        "descriptive_best_primary": best.to_dict(),
        "descriptive_most_direction_consistent": most_consistent.to_dict(),
        "shortlist": shortlist.to_dict("records"),
        "scores": scores.to_dict("records"),
        "cross_domain_preservation": {
            "galaxy_outer_RMSE_km_s": float(cross_domain.galaxy_outer_RMSE_km_s),
            "CLASH_radial_RMSE_dex": float(cross_domain.cluster_RMSE_dex),
            "Mercury_precession_mas_per_century": float(
                cross_domain.Mercury_precession_mas_per_century
            ),
            "all_solar_proxies_pass": bool(cross_domain.all_solar_proxies_pass),
            "interpretation": "identical for every localization because the angular correction has zero circular monopole and is absent in point-mass Solar controls",
        },
        "field_invariants": {
            "maximum_route_map_normalization_error": float(
                audits.route_map_normalization_error.max()
            ),
            "maximum_annular_convergence_error": float(
                audits.maximum_annular_convergence_mean_fraction.max()
            ),
            "maximum_normalized_curl_RMS": float(audits.normalized_curl_RMS.max()),
        },
        "verdict": {
            "any_nonbaseline_improves_primary_aggregate": bool(
                nonbaseline.primary_improvement_fraction_vs_eta0.gt(0).any()
            ),
            "any_nonbaseline_improves_at_least_three_primary_systems": bool(
                nonbaseline.primary_systems_improved.ge(3).any()
            ),
            "shortlist_requires_exact_refit_and_global_roots": True,
            "no_formula_promoted": True,
        },
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )

    ordered = scores.sort_values("primary_improvement_fraction_vs_eta0")
    pivot = system_scores.pivot_table(
        index="variant_id",
        columns="system_label",
        values="heldout_linearized_RMS_arcsec",
        aggfunc="first",
    )
    changes = 100.0 * (1.0 - pivot.div(pivot.loc["eta_000"], axis=1))
    changes = changes.loc[ordered.variant_id]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    axes[0].barh(
        ordered.variant_id,
        100.0 * ordered.primary_improvement_fraction_vs_eta0,
        color=np.where(ordered.primary_improvement_fraction_vs_eta0 > 0, "tab:green", "crimson"),
    )
    axes[0].axvline(0.0, color="black", linewidth=0.8)
    axes[0].set(xlabel="primary four-cluster improvement (%)", title="Route localization impact")
    image = axes[1].imshow(changes.to_numpy(float), aspect="auto", cmap="RdYlGn", vmin=-1.0, vmax=1.0)
    axes[1].set(
        xticks=np.arange(len(changes.columns)),
        xticklabels=changes.columns,
        yticks=np.arange(len(changes.index)),
        yticklabels=changes.index,
        title="Per-cluster direction of change (%)",
    )
    axes[1].tick_params(axis="x", rotation=30)
    fig.colorbar(image, ax=axes[1], label="improvement vs eta=0 (%)")
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)

    summary = f"""# P0554 route-localization screen

The frozen screen evaluated {len(scores)} baryon-only route geometries at the
topology-safe eta=0.30 amplitude. The lowest primary fixed-geometry local-response
RMS is `{best.variant_id}` at
{best.primary_other_four_equal_system_RMS_arcsec:.3f} arcsec
({100.0 * best.primary_improvement_fraction_vs_eta0:+.3f}% versus eta=0),
improving {int(best.primary_systems_improved)}/4 transfer systems.

The most direction-consistent shape is `{most_consistent.variant_id}`, improving
{int(most_consistent.primary_systems_improved)}/4 systems. Shortlisted shapes:
{', '.join(shortlist_ids)}. This local-response screen does not test nonlinear
root topology; no formula is promoted.
"""
    (output / protocol["outputs"]["summary"]).write_text(summary, encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--postprocess-only", action="store_true")
    args = parser.parse_args()
    config_path = ROOT / "configs/p0554_route_localization_screen_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if not protocol["status"].startswith("frozen_"):
        raise RuntimeError("route localization protocol is not frozen")
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    if args.postprocess_only:
        report = summarize(protocol, config_path, output)
        print(json.dumps(json_safe(report["verdict"]), indent=2), flush=True)
        return

    interaction = json.loads(
        (ROOT / protocol["inputs"]["interaction_protocol"]).read_text(encoding="utf-8")
    )
    parent = next(
        item for item in build_variants(interaction) if item.variant_id == "combined_parent"
    )
    contexts = raw_contexts(interaction)
    route_sources, route_protocols = load_route_sources(interaction, contexts)
    geometry = pd.read_csv(ROOT / protocol["inputs"]["fixed_geometry"])
    system_rows, image_frames, audit_rows = [], [], []
    for context in contexts:
        parameters = geometry_for(geometry, context.label)
        radial, _ = raw_field(parent.spec, parent.q, context.anchors, context.local, A0)
        baryons = baryon_field(context.anchors, context.local)
        adaptive = route_fraction(parent.candidate, route_sources[context.label], context.local)
        parent_strength = float(adaptive["routing_fraction"] ** parent.route_power)
        for variant in protocol["variants"]:
            variant_id = variant["variant_id"]
            print(f"{context.label}: {variant_id}", flush=True)
            if variant["mode"] == "none":
                morphology = None
                audit = None
                strength = 0.0
            else:
                morphology, audit = build_directional_field(
                    interaction,
                    route_protocols[context.label],
                    context,
                    route_sources[context.label],
                    parent,
                    radial,
                    baryons,
                    variant,
                )
                strength = 0.30 * parent_strength
                audit_rows.append(
                    {
                        "system_label": context.label,
                        "variant_id": variant_id,
                        "local_mix": float(variant["local_mix"]),
                        "softening_kpc": float(variant["softening_kpc"]),
                        "symmetric_bend_degrees": float(variant["symmetric_bend_degrees"]),
                        "center_mode": variant["center_mode"],
                        **audit,
                    }
                )
            lens = MorphologyLens(
                context.local,
                {parent.variant_id: radial},
                parent=parent.variant_id,
                morphology=morphology,
                fraction=strength,
            )
            _, sources = lens.profiled_residuals(
                parent.variant_id, parameters, context.training
            )
            training = linearized_residuals(
                lens, parent.variant_id, parameters, sources, context.training, "training"
            )
            heldout = linearized_residuals(
                lens, parent.variant_id, parameters, sources, context.heldout, "heldout"
            )
            combined = pd.concat([training, heldout], ignore_index=True)
            combined.insert(0, "variant_id", variant_id)
            combined.insert(0, "system_label", context.label)
            image_frames.append(combined)
            system_rows.append(
                {
                    "system_label": context.label,
                    "variant_id": variant_id,
                    "training_images": len(training),
                    "training_linearized_RMS_arcsec": rms(
                        training.linearized_radial_residual_arcsec
                    ),
                    "heldout_images": len(heldout),
                    "heldout_linearized_RMS_arcsec": rms(
                        heldout.linearized_radial_residual_arcsec
                    ),
                    "applied_angular_strength": strength,
                }
            )
    pd.DataFrame(system_rows).to_csv(
        output / protocol["outputs"]["system_scores"], index=False
    )
    pd.concat(image_frames, ignore_index=True).to_csv(
        output / protocol["outputs"]["image_residuals"], index=False
    )
    pd.DataFrame(audit_rows).to_csv(
        output / protocol["outputs"]["field_audits"], index=False
    )
    report = summarize(protocol, config_path, output)
    print(
        json.dumps(
            json_safe(
                {
                    "coverage": report["coverage"],
                    "best": report["descriptive_best_primary"],
                    "shortlist": report["shortlist"],
                    "verdict": report["verdict"],
                }
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
