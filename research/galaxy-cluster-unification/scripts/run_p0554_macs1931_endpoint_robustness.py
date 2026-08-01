#!/usr/bin/env python3
"""Stress-test the mild Subaru count excess at the MACS1931 halo endpoint."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import theilslopes


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0554_macs1931_subaru_endpoint import (  # noqa: E402
    density_score,
    load_halo,
    project,
    read_known_members,
    read_photoz,
    rotate,
    selections,
    source_weights,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sky_coordinates(ra, dec, center_dec):
    cosine = math.cos(math.radians(float(center_dec)))
    return np.column_stack([np.asarray(ra, float) * cosine, np.asarray(dec, float)]) * 3600.0


def crossmatch(left_ra, left_dec, right_ra, right_dec, center_dec):
    left = sky_coordinates(left_ra, left_dec, center_dec)
    right = sky_coordinates(right_ra, right_dec, center_dec)
    return cKDTree(right).query(left, k=1)


def fit_color_locus(catalog, member_indices, color_config):
    pairs = [("B", "V"), ("V", "RC"), ("RC", "IC"), ("IC", "Z")]
    fits = []
    residual_z = np.full((len(catalog), len(pairs)), np.nan)
    magnitude = catalog.IC.to_numpy(float)
    for feature_index, (blue, red) in enumerate(pairs):
        blue_mag = catalog[blue].to_numpy(float)
        red_mag = catalog[red].to_numpy(float)
        color = blue_mag - red_mag
        valid = (
            np.isfinite(color)
            & (blue_mag > 0.0)
            & (red_mag > 0.0)
            & np.isfinite(magnitude)
            & (magnitude > 0.0)
        )
        train = np.zeros(len(catalog), dtype=bool)
        train[np.asarray(member_indices, dtype=int)] = True
        train &= valid
        if int(train.sum()) < int(color_config["minimum_training_members_per_color"]):
            raise RuntimeError(f"too few member counterparts to train {blue}-{red}")
        slope, intercept, _, _ = theilslopes(color[train], magnitude[train])
        member_residual = color[train] - (intercept + slope * magnitude[train])
        median = float(np.median(member_residual))
        mad = float(np.median(np.abs(member_residual - median)))
        scale = max(1.4826 * mad, float(color_config["minimum_residual_scale_mag"]))
        residual_z[valid, feature_index] = (
            color[valid] - (intercept + slope * magnitude[valid]) - median
        ) / scale
        fits.append(
            {
                "feature": f"{blue}-{red}",
                "training_members": int(train.sum()),
                "slope_per_ic_mag": float(slope),
                "intercept_mag": float(intercept),
                "residual_center_mag": median,
                "residual_scale_mag": scale,
            }
        )
    valid_count = np.sum(np.isfinite(residual_z), axis=1)
    rms = np.sqrt(
        np.nansum(np.square(residual_z), axis=1) / np.maximum(valid_count, 1)
    )
    passes = (
        valid_count >= int(color_config["minimum_valid_colors_per_source"])
    ) & (rms <= float(color_config["maximum_rms_standardized_residual"]))
    return passes, rms, pd.DataFrame(fits)


def score_mask(catalog, position, halo, mask, parent, protocol, selection_kind, weight_kind):
    block = catalog.loc[mask]
    source_position = position[mask]
    weight = source_weights(block, parent, weight_kind)
    width = float(protocol["endpoint_test"]["kernel_width_kpc"])
    real = density_score(source_position, weight, halo, width)
    values = []
    null_rows = []
    for angle in protocol["endpoint_test"]["rotation_angles_deg"]:
        value = density_score(source_position, weight, rotate(halo, angle), width)
        values.append(value)
        null_rows.append(
            {
                "selection_kind": selection_kind,
                "weight_kind": weight_kind,
                "rotation_angle_deg": float(angle),
                "density_score": value,
            }
        )
    values = np.asarray(values)
    null_median = float(np.median(values))
    return {
        "selection_kind": selection_kind,
        "weight_kind": weight_kind,
        "selected_objects": int(mask.sum()),
        "real_density_score": real,
        "null_median_density_score": null_median,
        "density_ratio": real / max(null_median, np.finfo(float).tiny),
        "rotation_p_value": float((1 + np.sum(values >= real)) / (1 + len(values))),
    }, null_rows


def main():
    config_path = ROOT / "configs/p0554_macs1931_endpoint_robustness_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    parent_path = ROOT / protocol["inputs"]["parent_protocol"]
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    parent_report = json.loads(
        (ROOT / protocol["inputs"]["parent_report"]).read_text(encoding="utf-8")
    )
    if parent_report["protocol"]["sha256"] != sha256(parent_path):
        raise RuntimeError("parent endpoint report does not match its protocol")
    legacy_provenance = json.loads(
        (ROOT / protocol["inputs"]["legacy_provenance"]).read_text(encoding="utf-8-sig")
    )
    legacy_path = ROOT / protocol["inputs"]["legacy_catalog"]
    if sha256(legacy_path) != legacy_provenance["catalog"]["sha256"]:
        raise RuntimeError("Legacy catalog differs from acquisition provenance")

    catalog = read_photoz(ROOT / protocol["inputs"]["subaru_catalog"], parent["catalog_columns"])
    settings = parent["coordinate_and_target"]
    position = project(catalog.RA, catalog.Dec, settings)
    parent_tight = selections(catalog, parent)["photoz_tight"]
    legacy = pd.read_csv(legacy_path)
    legacy_distance, legacy_index = crossmatch(
        catalog.RA, catalog.Dec, legacy.ra, legacy.dec, settings["center_dec_deg"]
    )
    legacy_matched = legacy_distance <= float(protocol["crossmatch"]["subaru_to_legacy_radius_arcsec"])
    legacy_type = np.full(len(catalog), "UNMATCHED", dtype=object)
    legacy_type[legacy_matched] = legacy.iloc[legacy_index[legacy_matched]].type.astype(str).to_numpy()
    legacy_gaia = np.full(len(catalog), np.nan)
    legacy_gaia[legacy_matched] = pd.to_numeric(
        legacy.iloc[legacy_index[legacy_matched]].gaia_phot_g_mean_mag, errors="coerce"
    ).to_numpy(float)
    legacy_star = legacy_matched & ((legacy_type == "PSF") | (np.isfinite(legacy_gaia) & (legacy_gaia > 0.0)))
    legacy_extended = legacy_matched & np.isin(
        legacy_type, protocol["crossmatch"]["legacy_extended_types"]
    )

    known = read_known_members(ROOT / protocol["inputs"]["known_member_catalog"])
    member_distance, member_index = crossmatch(
        known.ra, known.dec, catalog.RA, catalog.Dec, settings["center_dec_deg"]
    )
    member_indices = member_index[
        member_distance <= float(protocol["crossmatch"]["known_member_to_subaru_radius_arcsec"])
    ]
    color_pass, color_rms, color_fits = fit_color_locus(
        catalog, member_indices, protocol["color_locus"]
    )
    catalog["legacy_match"] = legacy_matched
    catalog["legacy_type"] = legacy_type
    catalog["legacy_gaia_g"] = legacy_gaia
    catalog["cluster_color_rms"] = color_rms

    masks = {
        "photoz_tight_parent": parent_tight,
        "tight_legacy_star_veto": parent_tight & ~legacy_star,
        "tight_legacy_extended_confirmation": parent_tight & legacy_extended,
        "tight_cluster_color_locus": parent_tight & color_pass,
        "tight_star_veto_and_color_locus": parent_tight & ~legacy_star & color_pass,
    }
    _, _, halo, _ = load_halo(parent)
    score_rows = []
    null_rows = []
    for selection_kind, mask in masks.items():
        for weight_kind in protocol["endpoint_test"]["weights"]:
            row, null = score_mask(
                catalog, position, halo, mask, parent, protocol, selection_kind, weight_kind
            )
            score_rows.append(row)
            null_rows.extend(null)
    scores = pd.DataFrame(score_rows)
    nulls = pd.DataFrame(null_rows)

    magnitude_rows = []
    robust_mask = masks["tight_star_veto_and_color_locus"]
    for low, high in protocol["endpoint_test"]["magnitude_bins_ic"]:
        mask = robust_mask & (catalog.IC >= float(low)) & (catalog.IC < float(high))
        if int(mask.sum()) == 0:
            continue
        row, _ = score_mask(
            catalog,
            position,
            halo,
            mask,
            parent,
            protocol,
            f"IC_{low:g}_{high:g}",
            "unit_count",
        )
        row["ic_low"] = float(low)
        row["ic_high"] = float(high)
        magnitude_rows.append(row)
    magnitude_scores = pd.DataFrame(magnitude_rows)

    audit = []
    for kind, mask in masks.items():
        audit.append(
            {
                "selection_kind": kind,
                "selected_objects": int(mask.sum()),
                "inside_1200kpc": int((mask & (np.linalg.norm(position, axis=1) <= 1200.0)).sum()),
                "matched_to_legacy": int((mask & legacy_matched).sum()),
                "legacy_psf_or_gaia_star": int((mask & legacy_star).sum()),
            }
        )
    audit = pd.DataFrame(audit)
    robust = scores[
        (scores.selection_kind == "tight_star_veto_and_color_locus")
        & (scores.weight_kind == "unit_count")
    ].iloc[0]
    robust_pass = bool(
        robust.rotation_p_value <= 0.05 and robust.density_ratio >= 1.5
    )
    outcome = "count_excess_survives_robustness" if robust_pass else "count_excess_not_robust"

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output / protocol["outputs"]["selection_audit"], index=False)
    color_fits.to_csv(output / protocol["outputs"]["color_locus_fits"], index=False)
    scores.to_csv(output / protocol["outputs"]["endpoint_scores"], index=False)
    nulls.to_csv(output / protocol["outputs"]["rotation_nulls"], index=False)
    magnitude_scores.to_csv(output / protocol["outputs"]["magnitude_scores"], index=False)
    report = {
        "report_version": "P0554-MACS1931-ENDPOINT-ROBUSTNESS-RESULTS-0.1.0",
        "status": "complete",
        "outcome": outcome,
        "protocol": {
            "path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(config_path),
        },
        "coverage": {
            "subaru_rows": len(catalog),
            "subaru_objects_matched_to_legacy": int(legacy_matched.sum()),
            "subaru_objects_vetoed_as_legacy_psf_or_gaia": int(legacy_star.sum()),
            "published_members_matched_to_subaru": len(member_indices),
        },
        "selection_audit": audit.to_dict("records"),
        "robust_endpoint_test": robust.to_dict(),
        "robustness_gate_passed": robust_pass,
        "all_endpoint_scores": scores.to_dict("records"),
        "magnitude_scores": magnitude_scores.to_dict("records"),
        "interpretation": (
            "The mild count excess remains after the predeclared star and color filters, but this spent follow-up still cannot establish a baryonic subgroup or a gravity-routing law."
            if robust_pass
            else "The mild count excess does not survive the combined star-veto and cluster-color robustness gate, so it should not be treated as a secure baryonic subgroup."
        ),
        "limits": protocol["limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    count_scores = scores[scores.weight_kind == "unit_count"]
    axes[0].barh(count_scores.selection_kind, count_scores.density_ratio, color="tab:blue")
    axes[0].axvline(1.5, color="black", linestyle="--", label="frozen robustness ratio")
    axes[0].set(xlabel="real / median rotation density", title="Endpoint excess under contamination controls")
    axes[0].legend()
    axes[1].bar(
        magnitude_scores.apply(lambda row: f"{row.ic_low:g}-{row.ic_high:g}", axis=1),
        magnitude_scores.density_ratio,
        color="tab:orange",
    )
    axes[1].axhline(1.0, color="black", linewidth=1)
    axes[1].set(xlabel="IC magnitude bin", ylabel="density ratio", title="Which magnitudes carry the robust selection?")
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)
    summary = (
        "# MACS1931 endpoint robustness\n\n"
        f"Outcome: **{outcome}**. After the combined star veto and known-member color-locus cut, "
        f"the count density ratio is {robust.density_ratio:.3f} with rotation p={robust.rotation_p_value:.4f}.\n"
    )
    (output / protocol["outputs"]["summary"]).write_text(summary, encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
