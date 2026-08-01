#!/usr/bin/env python3
"""Screen gravity-route directions reconstructed from registered light and X-rays."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.io.fits import Header
from astropy.wcs import WCS
from scipy.ndimage import binary_dilation, gaussian_filter, map_coordinates


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_adaptive_route_multicluster_raw import route_fraction  # noqa: E402
from run_adaptive_route_raw_rxj2129 import baryon_field  # noqa: E402
from run_arc_invariant_absolute_lensing import raw_field  # noqa: E402
from run_clash_stellar_morphology_response import MorphologyLens  # noqa: E402
from run_p0554_local_cross_domain_sensitivity import A0, json_safe, raw_contexts, sha256  # noqa: E402
from run_p0554_route_localization_screen import geometry_for, linearized_residuals, rms  # noqa: E402
from run_p0554_route_softness_interaction import build_variants, load_route_sources  # noqa: E402
from voidscreen.adaptive_route_kernel import adaptive_route_parameters, transformed_source_weights  # noqa: E402
from voidscreen.baryon_morphology import blend_unit_directions, map_attraction_directions  # noqa: E402
from voidscreen.route_template import (  # noqa: E402
    baryonic_route_directions,
    conservative_explicit_direction_route_template,
    weighted_radius,
)
from voidscreen.stellar_morphology_lensing import build_stellar_morphology_deflection_field  # noqa: E402


def low_level_wcs(header) -> WCS:
    result = WCS(header)
    result.sip = None
    return result


def event_wcs(header) -> WCS:
    result = Header()
    result["NAXIS"] = 2
    for output, source in {
        "CTYPE1": "TCTYP11",
        "CTYPE2": "TCTYP12",
        "CRVAL1": "TCRVL11",
        "CRVAL2": "TCRVL12",
        "CRPIX1": "TCRPX11",
        "CRPIX2": "TCRPX12",
        "CDELT1": "TCDLT11",
        "CDELT2": "TCDLT12",
    }.items():
        result[output] = header[source]
    result["RADESYS"] = header.get("RADESYS", "ICRS")
    return WCS(result)


def hst_products(protocol, acquisition, reused, label):
    if label == "RXJ2129":
        system = acquisition["hst"]["new_system"]
        directory = ROOT / acquisition["outputs"]["hst_directory"]
    else:
        system = next(row for row in reused["systems"] if row["label"] == label)
        directory = ROOT / reused["outputs"]["directory"]
    return directory / system["science"]["filename"], directory / system["weight"]["filename"]


def prepare_hst_map(protocol, acquisition, reused, context, images, axis):
    settings = protocol["map_construction"]
    science_path, weight_path = hst_products(protocol, acquisition, reused, context.label)
    with fits.open(science_path, memmap=True) as science_hdul, fits.open(weight_path, memmap=True) as weight_hdul:
        header = science_hdul[0].header
        science = science_hdul[0].data
        weight = weight_hdul[0].data
        wcs = low_level_wcs(header)
        local_geometry = context.local["cosmology_and_coordinates"]
        center_ra = float(local_geometry["center_ra_deg"])
        center_dec = float(local_geometry["center_dec_deg"])
        center_x, center_y = wcs.wcs_world2pix([[center_ra, center_dec]], 0)[0]
        matrix = np.asarray([[header["CD1_1"], header["CD1_2"]], [header["CD2_1"], header["CD2_2"]]], dtype=float)
        native_scale = float(np.sqrt(abs(np.linalg.det(matrix))) * 3600.0)
        cut_radius = float(settings["hst_native_cut_radius_arcsec"])
        cut_pixels = int(np.ceil(cut_radius / native_scale))
        x0, x1 = int(round(center_x)) - cut_pixels, int(round(center_x)) + cut_pixels + 1
        y0, y1 = int(round(center_y)) - cut_pixels, int(round(center_y)) + cut_pixels + 1
        cut = np.asarray(science[y0:y1, x0:x1], dtype=np.float64)
        cut_weight = np.asarray(weight[y0:y1, x0:x1], dtype=np.float64)
        yy, xx = np.indices(cut.shape, dtype=float)
        x_arcsec = -(xx + x0 - center_x) * native_scale
        y_arcsec = (yy + y0 - center_y) * native_scale
        radius = np.hypot(x_arcsec, y_arcsec)
        valid = (cut_weight > 0.0) & np.isfinite(cut)
        bg_lo, bg_hi = settings["hst_background_annulus_arcsec"]
        outer = valid & (radius >= float(bg_lo)) & (radius <= float(bg_hi))
        background = float(np.median(cut[outer]))
        image_mask = np.zeros_like(valid)
        mask_radius = float(settings["known_image_mask_radius_arcsec"])
        for row in images.itertuples(index=False):
            image_mask |= (
                np.square(x_arcsec - float(row.x_arcsec))
                + np.square(y_arcsec - float(row.y_arcsec))
                <= mask_radius**2
            )
        usable = valid & ~image_mask
        fill = np.full_like(cut, background)
        bins = np.floor(radius / 0.5).astype(int)
        for index in range(int(np.ceil(cut_radius / 0.5)) + 1):
            target = bins == index
            source = target & usable
            if np.any(source):
                fill[target] = float(np.median(cut[source]))
        filled = cut.copy()
        filled[~usable] = fill[~usable]
        positive = np.maximum(filled - background, 0.0)
        target_x, target_y = np.meshgrid(axis, axis)
        cosine = math.cos(math.radians(center_dec))
        target_ra = center_ra + target_x / (3600.0 * cosine)
        target_dec = center_dec + target_y / 3600.0
        pixel_x, pixel_y = wcs.wcs_world2pix(target_ra, target_dec, 0)
        coordinates = np.vstack([(pixel_y - y0).ravel(), (pixel_x - x0).ravel()])
        sampled = map_coordinates(
            positive,
            coordinates,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        ).reshape(target_x.shape)
        sampled = np.maximum(sampled, 0.0)
    return sampled, {
        "system_label": context.label,
        "map_kind": "hst_f160w",
        "input_count": 2,
        "native_pixel_scale_arcsec": native_scale,
        "background": background,
        "masked_fraction": float(np.mean(image_mask)),
        "positive_cells": int(np.sum(sampled > 0.0)),
        "map_sum_before_normalization": float(np.sum(sampled)),
    }


def chandra_paths(acquisition, label):
    system = next(row for row in acquisition["chandra"]["systems"] if row["label"] == label)
    if system.get("source", "").startswith("reused:"):
        directory = ROOT / system["source"].split(":", 1)[1]
        paths = [next((directory / str(obsid) / "primary").glob("*evt2.fits.gz")) for obsid in system["obsids"]]
    else:
        directory = ROOT / acquisition["outputs"]["chandra_directory"] / label
        paths = [next((directory / str(obsid)).glob("*evt2.fits.gz")) for obsid in system["obsids"]]
    return paths


def prepare_xray_maps(protocol, acquisition, context, axis):
    settings = protocol["map_construction"]
    spacing = float(settings["grid_spacing_arcsec"])
    edges = np.r_[axis - 0.5 * spacing, axis[-1] + 0.5 * spacing]
    rate = np.zeros((len(axis), len(axis)), dtype=float)
    counts = np.zeros_like(rate)
    total_exposure = 0.0
    lo, hi = settings["xray_band_keV"]
    geometry = context.local["cosmology_and_coordinates"]
    for path in chandra_paths(acquisition, context.label):
        with fits.open(path, memmap=False) as hdul:
            events = hdul["EVENTS"]
            header = events.header
            wcs = event_wcs(header)
            center_x, center_y = wcs.all_world2pix([[geometry["center_ra_deg"], geometry["center_dec_deg"]]], 1)[0]
            pixel_arcsec = abs(float(header["TCDLT11"])) * 3600.0
            x = -(events.data["x"].astype(float) - center_x) * pixel_arcsec
            y = (events.data["y"].astype(float) - center_y) * pixel_arcsec
            energy = events.data["energy"].astype(float) / 1000.0
            use = (energy >= float(lo)) & (energy <= float(hi))
            image, _, _ = np.histogram2d(y[use], x[use], bins=(edges, edges))
            exposure = float(header["EXPOSURE"])
            rate += image / exposure
            counts += image
            total_exposure += exposure
    point = settings["xray_point_source_detection"]
    small_sigma = float(point["small_gaussian_sigma_arcsec"]) / spacing
    broad_sigma = float(point["broad_gaussian_sigma_arcsec"]) / spacing
    small_counts = gaussian_filter(counts, small_sigma, mode="nearest")
    broad_counts = gaussian_filter(counts, broad_sigma, mode="nearest")
    variance = np.maximum(broad_counts, 0.0) / (4.0 * np.pi * small_sigma**2) + 1.0e-6
    significance = (small_counts - broad_counts) / np.sqrt(variance)
    grid_x, grid_y = np.meshgrid(axis, axis)
    outside_core = np.hypot(grid_x, grid_y) >= float(point["protected_cluster_core_radius_arcsec"])
    seeds = (significance >= float(point["difference_significance_threshold"])) & outside_core
    dilation = int(np.ceil(float(point["mask_dilation_radius_arcsec"]) / spacing))
    yy, xx = np.indices((2 * dilation + 1, 2 * dilation + 1)) - dilation
    structure = np.square(xx) + np.square(yy) <= dilation**2
    point_mask = binary_dilation(seeds, structure=structure)
    broad_rate = gaussian_filter(rate, broad_sigma, mode="nearest")
    masked = rate.copy()
    masked[point_mask] = broad_rate[point_mask]
    return np.maximum(rate, 0.0), np.maximum(masked, 0.0), {
        "system_label": context.label,
        "map_kind": "chandra_soft_rate",
        "input_count": len(chandra_paths(acquisition, context.label)),
        "total_exposure_ks": total_exposure / 1000.0,
        "soft_events_on_grid": int(np.sum(counts)),
        "point_source_seed_cells": int(np.sum(seeds)),
        "point_source_masked_fraction": float(np.mean(point_mask)),
        "unmasked_map_sum": float(np.sum(rate)),
        "masked_map_sum": float(np.sum(masked)),
    }


def route_components(interaction, route_protocol, context, sources, parent, radial, baryons):
    scale = float(context.local["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
    xy = sources[["x_arcsec", "y_arcsec"]].to_numpy(float)
    weights = transformed_source_weights(sources.base_weight.to_numpy(float), float(parent.candidate.source_weight_power))
    radius_kpc = np.hypot(xy[:, 0], xy[:, 1]) * scale
    r50 = weighted_radius(radius_kpc, weights, 0.5)
    r80 = weighted_radius(radius_kpc, weights, 0.8)
    adaptive = adaptive_route_parameters(
        r50_kpc=r50,
        concentration=r50 / r80,
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
    return scale, xy, weights, adaptive


def field_from_directions(interaction, route_protocol, scale, xy, weights, adaptive, radial, baryons, directions):
    translation = route_protocol["route_to_deflection_translation"]
    spacing = float(translation["grid_spacing_arcsec"])
    route_axis = np.arange(-255.5, 256.0, spacing)
    route_map, route_audit = conservative_explicit_direction_route_template(
        route_axis,
        xy,
        weights,
        directions,
        routing_fraction=adaptive["routing_fraction"],
        return_scale=adaptive["return_scale_kpc"] / scale,
        radius_exponent=float(translation["source_radius_exponent"]),
        reference_radius=float(translation["source_reference_radius_kpc"]) / scale,
        smoothing=adaptive["width_kpc"] / scale,
        center=None,
    )

    def carrier_alpha(radius_arcsec):
        return radial.reduced_alpha_arcsec(radius_arcsec, 1.0) - baryons.reduced_alpha_arcsec(radius_arcsec, 1.0)

    field = build_stellar_morphology_deflection_field(
        route_axis,
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
    return field, route_audit, field.audit


def choose_direction(variant, member, star, gas_unmasked, gas_masked):
    kind = variant["direction"]
    fraction = float(variant.get("second_fraction", 0.0))
    if kind == "member":
        return member
    if kind == "member_star":
        return blend_unit_directions(member, star, fraction)
    if kind == "star":
        return star
    if kind == "gas_unmasked":
        return gas_unmasked
    if kind == "gas_masked":
        return gas_masked
    if kind == "star_gas":
        return blend_unit_directions(star, gas_masked, fraction)
    if kind == "member_all":
        all_baryon = blend_unit_directions(star, gas_masked, float(variant["gas_fraction_inside_all"]))
        return blend_unit_directions(member, all_baryon, fraction)
    raise ValueError(kind)


def summarize(protocol, config_path, output):
    systems = pd.read_csv(output / protocol["outputs"]["system_scores"])
    fields = pd.read_csv(output / protocol["outputs"]["field_audits"])
    baseline = systems[systems.variant_id.eq("eta_000")].set_index("system_label")
    member = systems[systems.variant_id.eq("member_parent")].set_index("system_label")
    primary = [label for label in protocol["evaluation"]["systems"] if label != protocol["evaluation"]["spent_selection_system"]]
    base_rms = rms(baseline.loc[primary, "heldout_linearized_RMS_arcsec"])
    member_rms = rms(member.loc[primary, "heldout_linearized_RMS_arcsec"])
    rows = []
    for variant_id, block in systems.groupby("variant_id", sort=False):
        indexed = block.set_index("system_label")
        value = rms(indexed.loc[primary, "heldout_linearized_RMS_arcsec"])
        vs_member = 1.0 - indexed.loc[primary, "heldout_linearized_RMS_arcsec"] / member.loc[primary, "heldout_linearized_RMS_arcsec"]
        rows.append({
            "variant_id": variant_id,
            "primary_equal_system_RMS_arcsec": value,
            "primary_improvement_fraction_vs_eta0": 1.0 - value / base_rms,
            "primary_improvement_fraction_vs_member_parent": 1.0 - value / member_rms,
            "primary_systems_improved_vs_member_parent": int(vs_member.gt(0).sum()),
            "minimum_primary_system_improvement_fraction_vs_member_parent": float(vs_member.min()),
            "spent_MACS1931_improvement_fraction_vs_member_parent": 1.0 - float(indexed.loc["MACS1931", "heldout_linearized_RMS_arcsec"]) / float(member.loc["MACS1931", "heldout_linearized_RMS_arcsec"]),
        })
    scores = pd.DataFrame(rows).sort_values("primary_equal_system_RMS_arcsec")
    gate = protocol["evaluation"]
    shortlist = scores[
        ~scores.variant_id.isin(["eta_000", "member_parent"])
        & scores.primary_improvement_fraction_vs_member_parent.ge(float(gate["exact_followup_minimum_improvement_fraction_vs_member_parent"]))
        & scores.primary_systems_improved_vs_member_parent.ge(int(gate["exact_followup_minimum_primary_systems_improved_vs_member_parent"]))
        & scores.minimum_primary_system_improvement_fraction_vs_member_parent.ge(-float(gate["exact_followup_maximum_single_system_worsening_fraction_vs_member_parent"]))
    ].copy()
    scores.to_csv(output / protocol["outputs"]["scores"], index=False)
    shortlist.to_csv(output / protocol["outputs"]["shortlist"], index=False)
    cross = pd.read_csv(ROOT / "results/p0554_route_softness_interaction/variant_scores.csv").set_index("variant_id").loc["lensing_softness_098"]
    best = scores.iloc[0]
    report = {
        "report_version": "P0554-ALL-BARYON-ROUTE-SCREEN-RESULTS-0.1.0",
        "status": "complete",
        "protocol": {"path": str(config_path.relative_to(ROOT)), "sha256": sha256(config_path)},
        "coverage": {
            "variants": int(scores.variant_id.nunique()),
            "systems": int(systems.system_label.nunique()),
            "variant_system_scores": len(systems),
            "route_fields": len(fields),
            "registered_proxy_maps": len(pd.read_csv(output / protocol["outputs"]["map_audits"])),
        },
        "best_primary": best.to_dict(),
        "scores": scores.to_dict("records"),
        "shortlist": shortlist.to_dict("records"),
        "cross_domain_preservation": {
            "galaxy_outer_RMSE_km_s": float(cross.galaxy_outer_RMSE_km_s),
            "CLASH_radial_RMSE_dex": float(cross.cluster_RMSE_dex),
            "Mercury_precession_mas_per_century": float(cross.Mercury_precession_mas_per_century),
            "all_solar_proxies_pass": bool(cross.all_solar_proxies_pass),
            "interpretation": "unchanged for every direction because only the zero-monopole angular route is modified",
        },
        "field_invariants": {
            "maximum_route_map_normalization_error": float(fields.route_map_normalization_error.max()),
            "maximum_annular_convergence_error": float(fields.maximum_annular_convergence_mean_fraction.max()),
            "maximum_normalized_curl_RMS": float(fields.normalized_curl_RMS.max()),
        },
        "verdict": {
            "any_map_direction_beats_member_parent": bool((scores[~scores.variant_id.isin(["eta_000", "member_parent"])].primary_improvement_fraction_vs_member_parent > 0).any()),
            "any_variant_meets_exact_followup_rule": not shortlist.empty,
            "no_formula_promoted": True,
        },
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    plot = scores.sort_values("primary_improvement_fraction_vs_member_parent")
    ax.barh(plot.variant_id, 100 * plot.primary_improvement_fraction_vs_member_parent)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set(xlabel="primary improvement versus member-only route (%)", title="Registered baryon-map direction screen")
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)
    summary = f"""# P0554 all-baryon route screen

The best primary fixed-geometry direction is `{best.variant_id}`, with
{100 * best.primary_improvement_fraction_vs_member_parent:+.3f}% change versus
the member-only route. {len(shortlist)} variants pass the frozen exact-followup
rule. X-ray brightness is used only as a morphology proxy; no gas mass or
formula is promoted.
"""
    (output / protocol["outputs"]["summary"]).write_text(summary, encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--postprocess-only", action="store_true")
    args = parser.parse_args()
    config_path = ROOT / "configs/p0554_all_baryon_route_screen_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if not protocol["status"].startswith("frozen_"):
        raise RuntimeError("all-baryon route protocol is not frozen")
    adequacy = json.loads((ROOT / protocol["inputs"]["input_audit"]).read_text(encoding="utf-8"))
    if not adequacy["input_adequacy_pass"]:
        raise RuntimeError("all-baryon route inputs did not pass adequacy")
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    if args.postprocess_only:
        report = summarize(protocol, config_path, output)
        print(json.dumps(json_safe(report["verdict"]), indent=2), flush=True)
        return
    acquisition = json.loads((ROOT / protocol["inputs"]["acquisition_protocol"]).read_text(encoding="utf-8"))
    reused = json.loads((ROOT / protocol["inputs"]["reused_hst_protocol"]).read_text(encoding="utf-8"))
    interaction = json.loads((ROOT / protocol["inputs"]["interaction_protocol"]).read_text(encoding="utf-8"))
    parent = next(item for item in build_variants(interaction) if item.variant_id == "combined_parent")
    contexts = raw_contexts(interaction)
    route_sources, route_protocols = load_route_sources(interaction, contexts)
    geometry = pd.read_csv(ROOT / protocol["inputs"]["fixed_geometry"])
    map_axis = np.arange(
        float(protocol["map_construction"]["axis_min_arcsec"]),
        float(protocol["map_construction"]["axis_max_arcsec"]) + 0.5 * float(protocol["map_construction"]["grid_spacing_arcsec"]),
        float(protocol["map_construction"]["grid_spacing_arcsec"]),
    )
    system_rows, field_rows, direction_rows, map_rows = [], [], [], []
    map_figure, map_axes = plt.subplots(len(contexts), 3, figsize=(12, 4 * len(contexts)), constrained_layout=True)
    for row_index, context in enumerate(contexts):
        print(f"maps: {context.label}", flush=True)
        parameters = geometry_for(geometry, context.label)
        radial, _ = raw_field(parent.spec, parent.q, context.anchors, context.local, A0)
        baryons = baryon_field(context.anchors, context.local)
        sources = route_sources[context.label]
        scale, xy, weights, adaptive = route_components(interaction, route_protocols[context.label], context, sources, parent, radial, baryons)
        known_images = pd.concat([context.training, context.heldout], ignore_index=True)
        star_map, star_audit = prepare_hst_map(protocol, acquisition, reused, context, known_images, map_axis)
        gas_unmasked_map, gas_masked_map, gas_audit = prepare_xray_maps(protocol, acquisition, context, map_axis)
        map_rows.extend([star_audit, gas_audit])
        member, _, _ = baryonic_route_directions(
            xy,
            weights,
            local_mix=1.0,
            softening=float(protocol["parent"]["direction_softening_kpc"]) / scale,
            distance_power=float(protocol["parent"]["distance_power"]),
        )
        star, star_direction_audit = map_attraction_directions(
            map_axis, star_map, xy,
            softening=float(protocol["parent"]["direction_softening_kpc"]) / scale,
            distance_power=float(protocol["parent"]["distance_power"]),
        )
        gas_unmasked, gas_unmasked_direction_audit = map_attraction_directions(
            map_axis, gas_unmasked_map, xy,
            softening=float(protocol["parent"]["direction_softening_kpc"]) / scale,
            distance_power=float(protocol["parent"]["distance_power"]),
        )
        gas_masked, gas_masked_direction_audit = map_attraction_directions(
            map_axis, gas_masked_map, xy,
            softening=float(protocol["parent"]["direction_softening_kpc"]) / scale,
            distance_power=float(protocol["parent"]["distance_power"]),
        )
        for name, directions, audit in (
            ("star", star, star_direction_audit),
            ("gas_unmasked", gas_unmasked, gas_unmasked_direction_audit),
            ("gas_masked", gas_masked, gas_masked_direction_audit),
        ):
            direction_rows.append({
                "system_label": context.label,
                "direction_kind": name,
                "mean_alignment_with_member": float(np.sum(weights * np.sum(directions * member, axis=1))),
                "mean_alignment_with_star": float(np.sum(weights * np.sum(directions * star, axis=1))),
                "mean_alignment_with_masked_gas": float(np.sum(weights * np.sum(directions * gas_masked, axis=1))),
                **{key: value for key, value in audit.items() if np.isscalar(value)},
            })
        strength = float(protocol["parent"]["eta"]) * float(adaptive["routing_fraction"] ** parent.route_power)
        for variant in protocol["variants"]:
            variant_id = variant["variant_id"]
            print(f"  {variant_id}", flush=True)
            if variant["direction"] == "none":
                morphology = None
                field_audit = {
                    "maximum_annular_convergence_mean_fraction": 0.0,
                    "normalized_curl_RMS": 0.0,
                }
                route_audit = {"normalization_error": 0.0}
                fraction = 0.0
            else:
                directions = choose_direction(variant, member, star, gas_unmasked, gas_masked)
                morphology, route_audit, field_audit = field_from_directions(
                    interaction, route_protocols[context.label], scale, xy, weights, adaptive, radial, baryons, directions
                )
                fraction = strength
            lens = MorphologyLens(context.local, {parent.variant_id: radial}, parent=parent.variant_id, morphology=morphology, fraction=fraction)
            _, profiled_sources = lens.profiled_residuals(parent.variant_id, parameters, context.training)
            heldout = linearized_residuals(lens, parent.variant_id, parameters, profiled_sources, context.heldout, "heldout")
            system_rows.append({
                "system_label": context.label,
                "variant_id": variant_id,
                "heldout_linearized_RMS_arcsec": rms(heldout.linearized_radial_residual_arcsec),
            })
            field_rows.append({
                "system_label": context.label,
                "variant_id": variant_id,
                "route_map_normalization_error": float(route_audit["normalization_error"]),
                **field_audit,
            })
        for column, (title, image) in enumerate((("F160W", star_map), ("X-ray raw", gas_unmasked_map), ("X-ray masked", gas_masked_map))):
            ax = map_axes[row_index, column]
            ax.imshow(np.log1p(image / max(float(np.nanmedian(image[image > 0])) if np.any(image > 0) else 1.0, np.finfo(float).tiny)), origin="lower", extent=[map_axis[0], map_axis[-1], map_axis[0], map_axis[-1]], cmap="magma")
            ax.set(title=f"{context.label} {title}", xlabel="east (arcsec)", ylabel="north (arcsec)")
    pd.DataFrame(map_rows).to_csv(output / protocol["outputs"]["map_audits"], index=False)
    pd.DataFrame(direction_rows).to_csv(output / protocol["outputs"]["direction_audits"], index=False)
    pd.DataFrame(field_rows).to_csv(output / protocol["outputs"]["field_audits"], index=False)
    pd.DataFrame(system_rows).to_csv(output / protocol["outputs"]["system_scores"], index=False)
    map_figure.savefig(output / protocol["outputs"]["map_figure"], dpi=160)
    plt.close(map_figure)
    report = summarize(protocol, config_path, output)
    print(json.dumps(json_safe({"coverage": report["coverage"], "best_primary": report["best_primary"], "shortlist": report["shortlist"], "verdict": report["verdict"]}), indent=2), flush=True)


if __name__ == "__main__":
    main()
