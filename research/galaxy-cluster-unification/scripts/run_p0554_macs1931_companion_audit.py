#!/usr/bin/env python3
"""Audit P0554's extra MACS1931 roots against the local CLASH F160W mosaic."""

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
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval
from astropy.wcs import WCS


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_adaptive_route_multicluster_raw import route_fraction  # noqa: E402
from run_adaptive_route_raw_rxj2129 import baryon_field, build_route_field  # noqa: E402
from run_arc_invariant_absolute_lensing import raw_field  # noqa: E402
from run_clash_stellar_morphology_response import MorphologyLens  # noqa: E402
from run_p0554_caustic_margin import geometry_for  # noqa: E402
from run_p0554_local_cross_domain_sensitivity import A0, json_safe, raw_contexts, sha256  # noqa: E402
from run_p0554_route_softness_interaction import build_variants, load_route_sources  # noqa: E402


def tangent_distance_arcsec(ra1, dec1, ra2, dec2, reference_dec):
    dx = (np.asarray(ra1) - np.asarray(ra2)) * 3600.0 * math.cos(
        math.radians(float(reference_dec))
    )
    dy = (np.asarray(dec1) - np.asarray(dec2)) * 3600.0
    return np.hypot(dx, dy)


def model_to_sky(x, y, astrometry):
    dec0 = float(astrometry["model_center_dec_deg"])
    ra0 = float(astrometry["model_center_ra_deg"])
    ra = ra0 + np.asarray(x, dtype=float) / (
        3600.0 * math.cos(math.radians(dec0))
    )
    dec = dec0 + np.asarray(y, dtype=float) / 3600.0
    return ra, dec


def load_drizzled_wcs(header):
    wcs = WCS(header).celestial
    # CLASH drizzled products are distortion-corrected, but residual SIP keywords
    # can remain without matching -SIP CTYPE labels. Use the declared linear WCS.
    wcs.sip = None
    return wcs


def pixel_scales_arcsec(wcs):
    return np.asarray(
        [
            abs(float(value.to_value("deg"))) if hasattr(value, "to_value") else abs(float(value))
            for value in wcs.proj_plane_pixel_scales()
        ]
    ) * 3600.0


def aperture_photometry(data, weight, wcs, ra, dec, settings):
    x, y = wcs.world_to_pixel_values(float(ra), float(dec))
    scales = pixel_scales_arcsec(wcs)
    pixel_scale = float(np.sqrt(np.prod(scales)))
    r_ap = float(settings["aperture_radius_arcsec"]) / pixel_scale
    r_in = float(settings["background_annulus_inner_arcsec"]) / pixel_scale
    r_out = float(settings["background_annulus_outer_arcsec"]) / pixel_scale
    margin = int(math.ceil(r_out + 2))
    x0, x1 = max(0, int(math.floor(x)) - margin), min(data.shape[1], int(math.floor(x)) + margin + 1)
    y0, y1 = max(0, int(math.floor(y)) - margin), min(data.shape[0], int(math.floor(y)) + margin + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.hypot(xx - x, yy - y)
    finite = np.isfinite(data[y0:y1, x0:x1]) & np.isfinite(weight[y0:y1, x0:x1])
    valid = finite & (weight[y0:y1, x0:x1] > 0.0)
    aperture_all = rr <= r_ap
    aperture = aperture_all & valid
    annulus = (rr >= r_in) & (rr <= r_out) & valid
    if not np.any(aperture_all) or not np.any(aperture) or annulus.sum() < 10:
        return {
            "pixel_x": float(x),
            "pixel_y": float(y),
            "pixel_scale_arcsec": pixel_scale,
            "valid_weight_fraction": 0.0,
            "aperture_flux": np.nan,
            "formal_flux_sigma": np.nan,
            "formal_snr": np.nan,
            "background_median": np.nan,
            "annulus_robust_sigma": np.nan,
            "peak_background_sigma": np.nan,
        }
    image = data[y0:y1, x0:x1]
    local_weight = weight[y0:y1, x0:x1]
    background = float(np.median(image[annulus]))
    annulus_mad = float(np.median(np.abs(image[annulus] - background)))
    robust_sigma = 1.4826 * annulus_mad
    flux = float(np.sum(image[aperture] - background))
    formal_variance = float(np.sum(1.0 / local_weight[aperture]))
    formal_sigma = math.sqrt(formal_variance) if formal_variance > 0 else np.nan
    peak = float(np.max(image[aperture] - background))
    return {
        "pixel_x": float(x),
        "pixel_y": float(y),
        "pixel_scale_arcsec": pixel_scale,
        "valid_weight_fraction": float(aperture.sum() / aperture_all.sum()),
        "aperture_flux": flux,
        "formal_flux_sigma": formal_sigma,
        "formal_snr": float(flux / formal_sigma) if formal_sigma > 0 else np.nan,
        "background_median": background,
        "annulus_robust_sigma": robust_sigma,
        "peak_background_sigma": float(peak / robust_sigma) if robust_sigma > 0 else np.nan,
    }


def assign_position_groups(frame, tolerance):
    centers = []
    group_ids = []
    for row in frame.itertuples(index=False):
        point = np.asarray([row.root_x_arcsec, row.root_y_arcsec], dtype=float)
        matches = [
            index
            for index, center in enumerate(centers)
            if np.linalg.norm(point - center["mean"]) <= float(tolerance)
        ]
        if matches:
            group = matches[0]
            centers[group]["points"].append(point)
            centers[group]["mean"] = np.mean(centers[group]["points"], axis=0)
        else:
            group = len(centers)
            centers.append({"points": [point], "mean": point.copy()})
        group_ids.append(group + 1)
    result = frame.copy()
    result["position_group"] = group_ids
    rows = []
    for group_id, block in result.groupby("position_group"):
        rows.append(
            {
                "position_group": int(group_id),
                "roles": ";".join(sorted(block.root_role.unique())),
                "variants": ";".join(sorted(block.variant_id.unique())),
                "formula_positions": len(block),
                "mean_x_arcsec": float(block.root_x_arcsec.mean()),
                "mean_y_arcsec": float(block.root_y_arcsec.mean()),
                "maximum_internal_separation_arcsec": float(
                    max(
                        tangent_distance_arcsec(
                            block.root_ra_deg,
                            block.root_dec_deg,
                            row.root_ra_deg,
                            row.root_dec_deg,
                            block.root_dec_deg.mean(),
                        ).max()
                        for row in block.itertuples(index=False)
                    )
                ),
            }
        )
    return result, pd.DataFrame(rows)


def nearest_catalog_fields(row, catalog, dec0, radius):
    distances = tangent_distance_arcsec(
        catalog.ra_deg.to_numpy(float),
        catalog.dec_deg.to_numpy(float),
        float(row.root_ra_deg),
        float(row.root_dec_deg),
        dec0,
    )
    nearest_index = int(np.argmin(distances))
    nearest = catalog.iloc[nearest_index]
    family = catalog[catalog.family_id.eq(2)]
    family_distances = tangent_distance_arcsec(
        family.ra_deg.to_numpy(float),
        family.dec_deg.to_numpy(float),
        float(row.root_ra_deg),
        float(row.root_dec_deg),
        dec0,
    )
    family_index = int(np.argmin(family_distances))
    family_nearest = family.iloc[family_index]
    return {
        "nearest_catalog_image": str(nearest.image_id),
        "nearest_catalog_family": int(nearest.family_id),
        "nearest_catalog_distance_arcsec": float(distances[nearest_index]),
        "nearest_family2_image": str(family_nearest.image_id),
        "nearest_family2_distance_arcsec": float(family_distances[family_index]),
        "catalogued_family2_match": bool(family_distances[family_index] <= float(radius)),
    }


def reconstruct_signed_determinants(protocol, selected_variants, roots):
    interaction = json.loads(
        (ROOT / protocol["inputs"]["interaction_protocol"]).read_text(encoding="utf-8")
    )
    variants = {item.variant_id: item for item in build_variants(interaction)}
    contexts = raw_contexts(interaction)
    context = next(item for item in contexts if item.label == protocol["selection"]["system"])
    sources, route_protocols = load_route_sources(interaction, contexts)
    geometry = pd.read_csv(ROOT / protocol["inputs"]["interaction_geometry"])
    baryons = baryon_field(context.anchors, context.local)
    radial_cache, route_cache = {}, {}
    all_images = pd.concat([context.training, context.heldout], ignore_index=True)
    family_id = int(protocol["selection"]["source_family"])
    redshift = float(all_images[all_images.source_family.eq(family_id)].source_redshift.iloc[0])
    signed = {}
    for variant_id in selected_variants:
        variant = variants[variant_id]
        radial_key = float(variant.spec["lensing_addition_softness"])
        if radial_key not in radial_cache:
            radial_cache[radial_key], _ = raw_field(
                variant.spec, variant.q, context.anchors, context.local, A0
            )
        radial = radial_cache[radial_key]
        angular = None
        angular_strength = 0.0
        if variant.route:
            adaptive = route_fraction(variant.candidate, sources[context.label], context.local)
            angular_strength = float(adaptive["routing_fraction"] ** variant.route_power)
            candidate_key = tuple(
                str(variant.candidate[key])
                for key in (
                    "feature",
                    "base_fraction",
                    "extent_slope",
                    "base_length_kpc",
                    "length_power",
                    "base_width_kpc",
                    "width_power",
                    "gate_power",
                    "source_weight_power",
                )
            )
            route_key = (radial_key, candidate_key)
            if route_key not in route_cache:
                route_cache[route_key] = build_route_field(
                    route_protocols[context.label],
                    context.local,
                    sources[context.label],
                    variant.candidate,
                    radial,
                    baryons,
                    contrast_cap=float(interaction["route_parent"]["contrast_cap"]),
                    contrast_strength=1.0,
                    centroid_mode=str(interaction["route_parent"]["centroid_mode"]),
                )[0]
            angular = route_cache[route_key]
        lens = MorphologyLens(
            context.local,
            {variant_id: radial},
            parent=variant_id,
            morphology=angular,
            fraction=angular_strength,
        )
        parameters = geometry_for(geometry, context.label, variant_id)
        block = roots[roots.variant_id.eq(variant_id)]
        jacobians = lens.jacobian(
            variant_id,
            parameters,
            block.root_x_arcsec.to_numpy(float),
            block.root_y_arcsec.to_numpy(float),
            redshift,
        )
        for row, jacobian in zip(block.itertuples(index=False), jacobians):
            determinant = float(np.linalg.det(jacobian))
            signed[(variant_id, int(row.root_index))] = {
                "signed_jacobian_determinant": determinant,
                "signed_magnification": float(1.0 / determinant),
                "parity": "positive" if determinant > 0 else "negative",
            }
    return signed


def display_normalization(cutout):
    finite = np.asarray(cutout)[np.isfinite(cutout)]
    interval = PercentileInterval(99.4)
    vmin, vmax = interval.get_limits(finite)
    return ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch(0.04))


def make_overview(data, wcs, audit, catalog, observed_2c, settings, output):
    scale = float(np.sqrt(np.prod(pixel_scales_arcsec(wcs))))
    size = int(math.ceil(2.0 * float(settings["overview_half_width_arcsec"]) / scale))
    center = (float(observed_2c.ra_deg), float(observed_2c.dec_deg))
    position = wcs.world_to_pixel_values(*center)
    cutout = Cutout2D(data, position, (size, size), wcs=wcs, mode="trim")
    fig = plt.figure(figsize=(11, 10), constrained_layout=True)
    ax = fig.add_subplot(111, projection=cutout.wcs)
    ax.imshow(cutout.data, origin="lower", cmap="gray", norm=display_normalization(cutout.data))
    transform = ax.get_transform("world")
    family = catalog[catalog.family_id.eq(2)]
    ax.scatter(
        family.ra_deg,
        family.dec_deg,
        transform=transform,
        facecolors="none",
        edgecolors="magenta",
        s=130,
        lw=1.8,
        label="published family 2",
    )
    for row in family.itertuples(index=False):
        ax.text(row.ra_deg, row.dec_deg, str(row.image_id), transform=transform, color="magenta", fontsize=9)
    anchor = audit[audit.root_role.eq("anchor")]
    companion = audit[audit.root_role.eq("companion")]
    ax.scatter(
        anchor.root_ra_deg,
        anchor.root_dec_deg,
        transform=transform,
        marker="o",
        facecolors="none",
        edgecolors="cyan",
        s=70,
        label="five-root anchor predictions",
    )
    ax.scatter(
        companion.root_ra_deg,
        companion.root_dec_deg,
        transform=transform,
        marker="x",
        color="gold",
        s=75,
        label="five-root companion predictions",
    )
    for row in companion.itertuples(index=False):
        ax.text(
            row.root_ra_deg,
            row.root_dec_deg,
            str(row.position_group),
            transform=transform,
            color="gold",
            fontsize=7,
        )
    ax.set_title("MACS1931 F160W: frozen P0554 extra-pair predictions near image 2c")
    ax.coords[0].set_axislabel("Right ascension")
    ax.coords[1].set_axislabel("Declination")
    ax.legend(loc="upper right", fontsize=9)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def make_cutouts(data, wcs, audit, observed_2c, settings, output):
    panels = [
        {
            "label": "observed family-2 image 2c",
            "ra": float(observed_2c.ra_deg),
            "dec": float(observed_2c.dec_deg),
        }
    ]
    for variant_id in settings["representative_variants"]:
        for role in ("anchor", "companion"):
            row = audit[audit.variant_id.eq(variant_id) & audit.root_role.eq(role)].iloc[0]
            panels.append(
                {
                    "label": f"{variant_id}\n{role}, group {int(row.position_group)}",
                    "ra": float(row.root_ra_deg),
                    "dec": float(row.root_dec_deg),
                }
            )
    ncols = 4
    nrows = int(math.ceil(len(panels) / ncols))
    fig = plt.figure(figsize=(15, 4.1 * nrows), constrained_layout=True)
    size = 2.0 * float(settings["cutout_half_width_arcsec"])
    for index, panel in enumerate(panels, start=1):
        position = wcs.world_to_pixel_values(panel["ra"], panel["dec"])
        scale = float(np.sqrt(np.prod(pixel_scales_arcsec(wcs))))
        pixels = int(math.ceil(size / scale))
        cutout = Cutout2D(data, position, (pixels, pixels), wcs=wcs, mode="trim")
        ax = fig.add_subplot(nrows, ncols, index, projection=cutout.wcs)
        ax.imshow(cutout.data, origin="lower", cmap="gray", norm=display_normalization(cutout.data))
        x, y = cutout.wcs.world_to_pixel_values(panel["ra"], panel["dec"])
        ax.scatter([x], [y], marker="+", color="red", s=90, lw=1.4)
        ax.set_title(panel["label"], fontsize=9)
        ax.coords[0].set_ticklabel_visible(False)
        ax.coords[1].set_ticklabel_visible(False)
        ax.coords[0].set_axislabel("")
        ax.coords[1].set_axislabel("")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    config_path = ROOT / "configs" / "p0554_macs1931_companion_audit_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if not protocol["status"].startswith("frozen_"):
        raise RuntimeError("companion audit protocol is not frozen")
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    roots = pd.read_csv(ROOT / protocol["inputs"]["global_roots"])
    assignments = pd.read_csv(ROOT / protocol["inputs"]["global_assignments"])
    catalog = pd.read_csv(ROOT / protocol["inputs"]["image_catalog"])
    catalog = catalog[catalog.system.str.contains("1931")].copy()
    observed_2c = catalog[catalog.image_id.eq("2c")].iloc[0]
    observed_xy = assignments[assignments.image_id.eq("2c")][
        ["observed_x_arcsec", "observed_y_arcsec"]
    ].drop_duplicates().iloc[0].to_numpy(float)
    counts = roots.groupby("variant_id").size()
    selected = sorted(counts[counts.eq(5)].index.tolist())
    signed = reconstruct_signed_determinants(protocol, selected, roots)

    rows = []
    for variant_id in selected:
        block = roots[roots.variant_id.eq(variant_id)].copy()
        distance = np.linalg.norm(
            block[["root_x_arcsec", "root_y_arcsec"]].to_numpy(float) - observed_xy[None, :],
            axis=1,
        )
        order = np.argsort(distance)[:2]
        for rank, row_index in enumerate(order, start=1):
            row = block.iloc[row_index]
            ra, dec = model_to_sky(
                row.root_x_arcsec,
                row.root_y_arcsec,
                protocol["astrometry"],
            )
            rows.append(
                {
                    "variant_id": variant_id,
                    "root_role": "anchor" if rank == 1 else "companion",
                    "root_index": int(row.root_index),
                    "root_x_arcsec": float(row.root_x_arcsec),
                    "root_y_arcsec": float(row.root_y_arcsec),
                    "root_ra_deg": float(ra),
                    "root_dec_deg": float(dec),
                    "distance_to_observed_2c_arcsec": float(distance[row_index]),
                    "archived_abs_jacobian_determinant": float(row.root_abs_determinant),
                    **signed[(variant_id, int(row.root_index))],
                }
            )
    audit = pd.DataFrame(rows)
    audit, groups = assign_position_groups(
        audit,
        protocol["display_and_grouping"]["root_grouping_radius_arcsec"],
    )

    with fits.open(ROOT / protocol["inputs"]["science_fits"], memmap=True) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32).copy()
        wcs = load_drizzled_wcs(hdul[0].header)
    with fits.open(ROOT / protocol["inputs"]["weight_fits"], memmap=True) as hdul:
        weight = np.asarray(hdul[0].data, dtype=np.float32).copy()

    reference_photometry = aperture_photometry(
        data,
        weight,
        wcs,
        observed_2c.ra_deg,
        observed_2c.dec_deg,
        protocol["photometry"],
    )
    catalog_radius = protocol["astrometry"]["published_catalog_match_radius_arcsec"]
    photometry_rows = []
    for row in audit.itertuples(index=False):
        measured = aperture_photometry(
            data,
            weight,
            wcs,
            row.root_ra_deg,
            row.root_dec_deg,
            protocol["photometry"],
        )
        photometry_rows.append(
            {
                **measured,
                **nearest_catalog_fields(
                    row,
                    catalog,
                    protocol["astrometry"]["model_center_dec_deg"],
                    catalog_radius,
                ),
            }
        )
    audit = pd.concat([audit.reset_index(drop=True), pd.DataFrame(photometry_rows)], axis=1)
    anchor_mu = (
        audit[audit.root_role.eq("anchor")]
        .set_index("variant_id")
        .signed_magnification.abs()
    )
    companion_mask = audit.root_role.eq("companion")
    audit["predicted_companion_to_anchor_flux_ratio"] = np.nan
    audit.loc[companion_mask, "predicted_companion_to_anchor_flux_ratio"] = audit.loc[
        companion_mask, "variant_id"
    ].map(
        audit[audit.root_role.eq("companion")]
        .set_index("variant_id")
        .signed_magnification.abs()
        / anchor_mu
    )
    audit["expected_companion_snr_from_2c"] = np.nan
    audit.loc[companion_mask, "expected_companion_snr_from_2c"] = (
        abs(float(reference_photometry["formal_snr"]))
        * audit.loc[companion_mask, "predicted_companion_to_anchor_flux_ratio"]
    )
    rules = protocol["photometry"]
    audit["formal_blank_rejection_candidate"] = False
    audit.loc[companion_mask, "formal_blank_rejection_candidate"] = (
        audit.loc[companion_mask, "predicted_companion_to_anchor_flux_ratio"]
        >= float(rules["minimum_predicted_companion_to_anchor_flux_ratio"])
    ) & (
        audit.loc[companion_mask, "expected_companion_snr_from_2c"]
        >= float(rules["strong_expected_companion_snr"])
    ) & (
        audit.loc[companion_mask, "formal_snr"] < float(rules["formal_detection_snr"])
    ) & (
        audit.loc[companion_mask, "valid_weight_fraction"]
        >= float(rules["minimum_valid_weight_fraction"])
    )
    audit["formal_source_at_position"] = (
        audit.formal_snr >= float(rules["formal_detection_snr"])
    ) & (
        audit.valid_weight_fraction >= float(rules["minimum_valid_weight_fraction"])
    )
    audit.to_csv(output / protocol["outputs"]["root_audit"], index=False)

    group_photometry = (
        audit.groupby("position_group", as_index=False)
        .agg(
            roles=("root_role", lambda values: ";".join(sorted(set(values)))),
            formula_positions=("variant_id", "size"),
            variants=("variant_id", lambda values: ";".join(sorted(set(values)))),
            mean_x_arcsec=("root_x_arcsec", "mean"),
            mean_y_arcsec=("root_y_arcsec", "mean"),
            mean_ra_deg=("root_ra_deg", "mean"),
            mean_dec_deg=("root_dec_deg", "mean"),
            median_formal_snr=("formal_snr", "median"),
            minimum_nearest_catalog_distance_arcsec=("nearest_catalog_distance_arcsec", "min"),
            any_catalogued_family2_match=("catalogued_family2_match", "any"),
            any_formal_source_at_position=("formal_source_at_position", "any"),
            any_formal_blank_rejection_candidate=("formal_blank_rejection_candidate", "any"),
        )
        .merge(
            groups[["position_group", "maximum_internal_separation_arcsec"]],
            on="position_group",
            validate="one_to_one",
        )
    )
    group_photometry.to_csv(output / protocol["outputs"]["position_groups"], index=False)

    make_overview(
        data,
        wcs,
        audit,
        catalog,
        observed_2c,
        protocol["display_and_grouping"],
        output / protocol["outputs"]["overview_figure"],
    )
    make_cutouts(
        data,
        wcs,
        audit,
        observed_2c,
        protocol["display_and_grouping"],
        output / protocol["outputs"]["cutout_figure"],
    )

    companions = audit[companion_mask].copy()
    opposite_parity = []
    for variant_id in selected:
        pair = audit[audit.variant_id.eq(variant_id)].set_index("root_role")
        opposite_parity.append(pair.loc["anchor", "parity"] != pair.loc["companion", "parity"])
    report = {
        "report_version": "P0554-MACS1931-COMPANION-AUDIT-RESULTS-0.1.0",
        "status": "pixel_audit_complete_visual_interpretation_required",
        "protocol": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": sha256(config_path),
        },
        "coverage": {
            "five_root_variants": len(selected),
            "near_2c_roots": len(audit),
            "companion_predictions": len(companions),
            "position_groups": int(group_photometry.position_group.nunique()),
            "published_MACS1931_images": len(catalog),
            "published_family2_images": int(catalog.family_id.eq(2).sum()),
        },
        "reference_image_2c": {
            "ra_deg": float(observed_2c.ra_deg),
            "dec_deg": float(observed_2c.dec_deg),
            **reference_photometry,
        },
        "model_predictions": {
            "opposite_parity_pairs": int(sum(opposite_parity)),
            "same_parity_pairs": int(len(opposite_parity) - sum(opposite_parity)),
            "companion_to_anchor_flux_ratio_range": [
                float(companions.predicted_companion_to_anchor_flux_ratio.min()),
                float(companions.predicted_companion_to_anchor_flux_ratio.max()),
            ],
            "companion_distance_from_2c_range_arcsec": [
                float(companions.distance_to_observed_2c_arcsec.min()),
                float(companions.distance_to_observed_2c_arcsec.max()),
            ],
        },
        "catalog_audit": {
            "companion_predictions_matching_published_family2": int(
                companions.catalogued_family2_match.astype(bool).sum()
            ),
            "unique_companion_position_groups_matching_published_family2": int(
                group_photometry[
                    group_photometry.roles.str.contains("companion")
                ].any_catalogued_family2_match.astype(bool).sum()
            ),
        },
        "formal_photometry": {
            "companion_predictions_with_valid_weight": int(
                (companions.valid_weight_fraction >= rules["minimum_valid_weight_fraction"]).sum()
            ),
            "companion_predictions_with_formal_source": int(
                companions.formal_source_at_position.astype(bool).sum()
            ),
            "formal_blank_rejection_candidates_before_visual_audit": int(
                companions.formal_blank_rejection_candidate.astype(bool).sum()
            ),
            "formal_companion_snr_range": [
                float(companions.formal_snr.min()),
                float(companions.formal_snr.max()),
            ],
            "expected_companion_snr_range": [
                float(companions.expected_companion_snr_from_2c.min()),
                float(companions.expected_companion_snr_from_2c.max()),
            ],
        },
        "verdict": {
            "published_catalog_confirms_extra_family2_image": bool(
                companions.catalogued_family2_match.astype(bool).any()
            ),
            "single_band_pixel_audit_is_final_identity_test": False,
            "visual_foreground_and_morphology_audit_required": True,
            "no_formula_promoted": True,
        },
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    summary = f"""# P0554 MACS1931 companion audit

The 11 five-root formulas predict companion-to-anchor absolute magnification
ratios from {report['model_predictions']['companion_to_anchor_flux_ratio_range'][0]:.3f}
to {report['model_predictions']['companion_to_anchor_flux_ratio_range'][1]:.3f}.
Published MACS1931 family-2 matches within the frozen 0.476751-arcsecond radius:
**{report['catalog_audit']['companion_predictions_matching_published_family2']}**.

The formal F160W aperture audit finds
**{report['formal_photometry']['companion_predictions_with_formal_source']}** of
{len(companions)} companion coordinates above the five-sigma diagnostic threshold
and **{report['formal_photometry']['formal_blank_rejection_candidates_before_visual_audit']}**
formal blank-rejection candidates. These are not final classifications: the
multi-arcsecond lens residuals, foreground light, correlated drizzle noise, and
single-band identity ambiguity require visual and multi-band follow-up. No
formula is promoted.
"""
    (output / protocol["outputs"]["summary"]).write_text(summary, encoding="utf-8")
    print(json.dumps(json_safe(report), indent=2), flush=True)


if __name__ == "__main__":
    main()
