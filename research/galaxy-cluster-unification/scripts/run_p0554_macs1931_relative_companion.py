#!/usr/bin/env python3
"""Audit anchor-registered P0554 companion positions in MACS1931 F160W."""

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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0554_local_cross_domain_sensitivity import json_safe, sha256  # noqa: E402
from run_p0554_macs1931_companion_audit import (  # noqa: E402
    aperture_photometry,
    assign_position_groups,
    display_normalization,
    load_drizzled_wcs,
    model_to_sky,
    nearest_catalog_fields,
    pixel_scales_arcsec,
)


def make_overview(data, wcs, audit, catalog, observed_2c, settings, output):
    scale = float(np.sqrt(np.prod(pixel_scales_arcsec(wcs))))
    size = int(math.ceil(2.0 * float(settings["overview_half_width_arcsec"]) / scale))
    position = wcs.world_to_pixel_values(float(observed_2c.ra_deg), float(observed_2c.dec_deg))
    cutout = Cutout2D(data, position, (size, size), wcs=wcs, mode="trim")
    fig = plt.figure(figsize=(10, 9), constrained_layout=True)
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
    ax.scatter(
        audit.root_ra_deg,
        audit.root_dec_deg,
        transform=transform,
        marker="x",
        color="gold",
        s=85,
        label="anchor-registered companions",
    )
    for row in audit.itertuples(index=False):
        ax.text(
            row.root_ra_deg,
            row.root_dec_deg,
            str(row.position_group),
            transform=transform,
            color="gold",
            fontsize=8,
        )
    ax.set_title("MACS1931 F160W: companion predictions after registering anchor to 2c")
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
        row = audit[audit.variant_id.eq(variant_id)].iloc[0]
        panels.append(
            {
                "label": f"{variant_id}\nregistered companion, group {int(row.position_group)}",
                "ra": float(row.root_ra_deg),
                "dec": float(row.root_dec_deg),
            }
        )
    fig = plt.figure(figsize=(15, 4), constrained_layout=True)
    scale = float(np.sqrt(np.prod(pixel_scales_arcsec(wcs))))
    pixels = int(math.ceil(2.0 * float(settings["cutout_half_width_arcsec"]) / scale))
    for index, panel in enumerate(panels, start=1):
        position = wcs.world_to_pixel_values(panel["ra"], panel["dec"])
        cutout = Cutout2D(data, position, (pixels, pixels), wcs=wcs, mode="trim")
        ax = fig.add_subplot(1, 4, index, projection=cutout.wcs)
        ax.imshow(cutout.data, origin="lower", cmap="gray", norm=display_normalization(cutout.data))
        x, y = cutout.wcs.world_to_pixel_values(panel["ra"], panel["dec"])
        ax.scatter([x], [y], marker="+", color="red", s=95, lw=1.5)
        ax.set_title(panel["label"], fontsize=9)
        ax.coords[0].set_ticklabel_visible(False)
        ax.coords[1].set_ticklabel_visible(False)
        ax.coords[0].set_axislabel("")
        ax.coords[1].set_axislabel("")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def make_all_cutouts(data, wcs, audit, settings, output):
    ordered = audit.sort_values("pair_separation_arcsec")
    fig = plt.figure(figsize=(15, 11), constrained_layout=True)
    scale = float(np.sqrt(np.prod(pixel_scales_arcsec(wcs))))
    pixels = int(math.ceil(2.0 * float(settings["cutout_half_width_arcsec"]) / scale))
    for index, row in enumerate(ordered.itertuples(index=False), start=1):
        position = wcs.world_to_pixel_values(row.root_ra_deg, row.root_dec_deg)
        cutout = Cutout2D(data, position, (pixels, pixels), wcs=wcs, mode="trim")
        ax = fig.add_subplot(3, 4, index, projection=cutout.wcs)
        ax.imshow(cutout.data, origin="lower", cmap="gray", norm=display_normalization(cutout.data))
        x, y = cutout.wcs.world_to_pixel_values(row.root_ra_deg, row.root_dec_deg)
        ax.scatter([x], [y], marker="+", color="red", s=90, lw=1.4)
        ax.set_title(
            f"{row.variant_id}\nsep={row.pair_separation_arcsec:.2f} arcsec, S/N={row.formal_snr:.1f}",
            fontsize=8,
        )
        ax.coords[0].set_ticklabel_visible(False)
        ax.coords[1].set_ticklabel_visible(False)
        ax.coords[0].set_axislabel("")
        ax.coords[1].set_axislabel("")
    fig.suptitle("Post-audit diagnostic: every anchor-registered companion position", fontsize=14)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    config_path = ROOT / "configs" / "p0554_macs1931_relative_companion_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if not protocol["status"].startswith("frozen_"):
        raise RuntimeError("relative companion protocol is not frozen")
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    absolute = pd.read_csv(ROOT / protocol["inputs"]["absolute_root_audit"])
    assignments = pd.read_csv(ROOT / protocol["inputs"]["global_assignments"])
    catalog = pd.read_csv(ROOT / protocol["inputs"]["image_catalog"])
    catalog = catalog[catalog.system.str.contains("1931")].copy()
    observed_2c = catalog[catalog.image_id.eq("2c")].iloc[0]
    observed_xy = assignments[assignments.image_id.eq("2c")][
        ["observed_x_arcsec", "observed_y_arcsec"]
    ].drop_duplicates().iloc[0].to_numpy(float)

    rows = []
    for variant_id, block in absolute.groupby("variant_id"):
        pair = block.set_index("root_role")
        anchor = pair.loc["anchor"]
        companion = pair.loc["companion"]
        displacement = np.asarray(
            [
                companion.root_x_arcsec - anchor.root_x_arcsec,
                companion.root_y_arcsec - anchor.root_y_arcsec,
            ],
            dtype=float,
        )
        registered = observed_xy + displacement
        ra, dec = model_to_sky(registered[0], registered[1], protocol["astrometry"])
        rows.append(
            {
                "variant_id": variant_id,
                "root_role": "companion",
                "root_x_arcsec": float(registered[0]),
                "root_y_arcsec": float(registered[1]),
                "root_ra_deg": float(ra),
                "root_dec_deg": float(dec),
                "anchor_translation_x_arcsec": float(observed_xy[0] - anchor.root_x_arcsec),
                "anchor_translation_y_arcsec": float(observed_xy[1] - anchor.root_y_arcsec),
                "pair_separation_arcsec": float(np.linalg.norm(displacement)),
                "pair_parities": f"{anchor.parity};{companion.parity}",
                "predicted_companion_to_anchor_flux_ratio": float(
                    companion.predicted_companion_to_anchor_flux_ratio
                ),
                "expected_companion_snr_from_2c": float(companion.expected_companion_snr_from_2c),
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

    photometry_rows = []
    radius = float(protocol["astrometry"]["published_catalog_match_radius_arcsec"])
    for row in audit.itertuples(index=False):
        photometry_rows.append(
            {
                **aperture_photometry(
                    data,
                    weight,
                    wcs,
                    row.root_ra_deg,
                    row.root_dec_deg,
                    protocol["photometry"],
                ),
                **nearest_catalog_fields(
                    row,
                    catalog,
                    protocol["astrometry"]["model_center_dec_deg"],
                    radius,
                ),
            }
        )
    audit = pd.concat([audit.reset_index(drop=True), pd.DataFrame(photometry_rows)], axis=1)
    settings = protocol["photometry"]
    audit["formal_source_at_position"] = (
        audit.formal_snr >= float(settings["formal_detection_snr"])
    ) & (
        audit.valid_weight_fraction >= float(settings["minimum_valid_weight_fraction"])
    )
    audit["formal_blank"] = (
        audit.expected_companion_snr_from_2c
        >= float(settings["strong_expected_companion_snr"])
    ) & (
        audit.predicted_companion_to_anchor_flux_ratio
        >= float(settings["minimum_predicted_companion_to_anchor_flux_ratio"])
    ) & (
        audit.formal_snr < float(settings["formal_detection_snr"])
    ) & (
        audit.valid_weight_fraction >= float(settings["minimum_valid_weight_fraction"])
    )
    annotations_path = output / "visual_assessment.json"
    visual = None
    if annotations_path.exists():
        visual = json.loads(annotations_path.read_text(encoding="utf-8"))
        annotation_frame = pd.DataFrame(visual["annotations"]).rename(
            columns={
                "classification": "visual_classification",
                "note": "visual_note",
            }
        )
        if set(annotation_frame.variant_id) != set(audit.variant_id):
            raise RuntimeError("visual annotation variant coverage changed")
        audit = audit.merge(annotation_frame, on="variant_id", validate="one_to_one")
    audit.to_csv(output / protocol["outputs"]["registered_audit"], index=False)
    group_summary = (
        audit.groupby("position_group", as_index=False)
        .agg(
            formula_positions=("variant_id", "size"),
            variants=("variant_id", lambda values: ";".join(sorted(values))),
            mean_pair_separation_arcsec=("pair_separation_arcsec", "mean"),
            mean_ra_deg=("root_ra_deg", "mean"),
            mean_dec_deg=("root_dec_deg", "mean"),
            median_formal_snr=("formal_snr", "median"),
            minimum_nearest_catalog_distance_arcsec=("nearest_catalog_distance_arcsec", "min"),
            any_catalogued_family2_match=("catalogued_family2_match", "any"),
            any_formal_source_at_position=("formal_source_at_position", "any"),
            all_formal_blank=("formal_blank", "all"),
        )
        .merge(
            groups[["position_group", "maximum_internal_separation_arcsec"]],
            on="position_group",
            validate="one_to_one",
        )
    )
    group_summary.to_csv(output / protocol["outputs"]["position_groups"], index=False)
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
    make_all_cutouts(
        data,
        wcs,
        audit,
        protocol["display_and_grouping"],
        output / "macs1931_registered_all_cutouts_diagnostic.png",
    )

    visual_counts = (
        audit.visual_classification.value_counts().to_dict()
        if visual is not None
        else {}
    )
    report = {
        "report_version": "P0554-MACS1931-RELATIVE-COMPANION-RESULTS-0.1.0",
        "status": "complete_with_manual_visual_audit" if visual is not None else "pixel_audit_complete_visual_interpretation_required",
        "protocol": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": sha256(config_path),
        },
        "visual_annotation": {
            "path": str(annotations_path.relative_to(ROOT)),
            "sha256": sha256(annotations_path),
            "counts": visual_counts,
        } if visual is not None else None,
        "coverage": {
            "variants": len(audit),
            "registered_position_groups": int(group_summary.position_group.nunique()),
            "published_family2_images": int(catalog.family_id.eq(2).sum()),
        },
        "registered_predictions": {
            "pair_separation_range_arcsec": [
                float(audit.pair_separation_arcsec.min()),
                float(audit.pair_separation_arcsec.max()),
            ],
            "companion_to_anchor_flux_ratio_range": [
                float(audit.predicted_companion_to_anchor_flux_ratio.min()),
                float(audit.predicted_companion_to_anchor_flux_ratio.max()),
            ],
            "catalogued_family2_matches": int(audit.catalogued_family2_match.astype(bool).sum()),
            "formal_sources_at_registered_positions": int(
                audit.formal_source_at_position.astype(bool).sum()
            ),
            "formal_blanks": int(audit.formal_blank.astype(bool).sum()),
            "formal_snr_range": [float(audit.formal_snr.min()), float(audit.formal_snr.max())],
        },
        "verdict": {
            "published_catalog_confirms_registered_companion": bool(
                audit.catalogued_family2_match.astype(bool).any()
            ),
            "all_registered_positions_formally_blank": bool(audit.formal_blank.astype(bool).all()),
            "visual_audit_complete": visual is not None,
            "plausible_centered_uncatalogued_counterimages": int(
                visual_counts.get("plausible_uncatalogued_counterimage", 0)
            ),
            "variants_with_clean_blank_companion": int(
                visual_counts.get("clean_blank", 0)
            ),
            "variants_with_contaminated_inconclusive_position": int(
                visual_counts.get("neighbor_contaminated_nonmatching", 0)
            ),
            "registered_extra_pair_supported_by_f160w": bool(
                visual_counts.get("plausible_uncatalogued_counterimage", 0) > 0
            ),
            "no_formula_promoted": True,
        },
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    summary = f"""# P0554 MACS1931 anchor-registered companion audit

After translating each formula's anchor root onto observed image 2c, the
opposite-parity companion lies {audit.pair_separation_arcsec.min():.3f}--
{audit.pair_separation_arcsec.max():.3f} arcseconds away. Published family-2
matches: **{int(audit.catalogued_family2_match.astype(bool).sum())}**. Formal
F160W sources at the 11 registered coordinates:
**{int(audit.formal_source_at_position.astype(bool).sum())}**; formal blanks:
**{int(audit.formal_blank.astype(bool).sum())}**. The manual visual audit finds
**{int(visual_counts.get('clean_blank', 0))}** clean blanks,
**{int(visual_counts.get('neighbor_contaminated_nonmatching', 0))}**
contaminated/inconclusive location, and
**{int(visual_counts.get('plausible_uncatalogued_counterimage', 0))}** plausible
centered uncatalogued counterimages. No formula is promoted.
"""
    (output / protocol["outputs"]["summary"]).write_text(summary, encoding="utf-8")
    print(json.dumps(json_safe(report), indent=2), flush=True)


if __name__ == "__main__":
    main()
