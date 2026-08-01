#!/usr/bin/env python3
"""Test whether the off-center MACS1931 halo overlaps wide-field galaxies."""

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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.halo_backtrack import component_samples, thin_bayes_chain  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def project(ra, dec, settings):
    cosine = math.cos(math.radians(float(settings["center_dec_deg"])))
    x = (np.asarray(ra, dtype=float) - float(settings["center_ra_deg"])) * 3600.0 * cosine
    y = (np.asarray(dec, dtype=float) - float(settings["center_dec_deg"])) * 3600.0
    return np.column_stack([x, y]) * float(settings["angular_scale_kpc_per_arcsec"])


def good_extended(catalog, quality):
    result = catalog.type.astype(str).isin(quality["extended_types"]).to_numpy()
    result &= catalog.maskbits.fillna(-1).to_numpy(int) <= int(quality["maximum_allowed_maskbits"])
    minimum = float(quality["minimum_signal_to_noise_in_each_required_band"])
    for band in quality["required_bands"]:
        flux = catalog[f"flux_{band}"].to_numpy(float)
        inverse_variance = catalog[f"flux_ivar_{band}"].to_numpy(float)
        result &= np.isfinite(flux) & np.isfinite(inverse_variance)
        result &= flux > 0.0
        result &= flux * np.sqrt(np.maximum(inverse_variance, 0.0)) >= minimum
    return result


def selections(catalog, protocol):
    base = good_extended(catalog, protocol["quality"])
    z = float(protocol["coordinate_and_matching"]["cluster_redshift"])
    median = catalog.z_phot_median_i.to_numpy(float)
    l68 = catalog.z_phot_l68_i.to_numpy(float)
    u68 = catalog.z_phot_u68_i.to_numpy(float)
    l95 = catalog.z_phot_l95_i.to_numpy(float)
    u95 = catalog.z_phot_u95_i.to_numpy(float)
    return {
        "photoz68_overlap": base & np.isfinite(median) & (median >= 0.20) & (median <= 0.50) & (l68 <= z) & (u68 >= z),
        "photoz_median_005": base & np.isfinite(median) & (np.abs(median - z) <= 0.05),
        "photoz95_overlap": base & np.isfinite(median) & (median >= 0.10) & (median <= 0.60) & (l95 <= z) & (u95 >= z),
        "extended_no_redshift": base,
    }


def weights(catalog, quality, kind):
    if kind == "unit_count":
        return np.ones(len(catalog), dtype=float)
    if kind == "capped_dereddened_z_flux":
        flux = catalog.flux_z.to_numpy(float) / np.maximum(
            catalog.mw_transmission_z.to_numpy(float), np.finfo(float).tiny
        )
        cap = float(np.quantile(flux[np.isfinite(flux) & (flux > 0.0)], float(quality["flux_weight_upper_quantile_cap"])))
        return np.clip(np.maximum(flux, 0.0), 0.0, cap)
    raise ValueError(kind)


def posterior_mean_density(source_position, source_weight, halo_position, width_kpc):
    difference = halo_position[:, None, :] - source_position[None, :, :]
    kernel = np.exp(-0.5 * np.sum(np.square(difference), axis=2) / float(width_kpc) ** 2)
    normalized = source_weight / np.sum(source_weight)
    return float(np.mean(kernel @ normalized))


def rotate(points, angle_deg):
    angle = math.radians(float(angle_deg))
    matrix = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    return np.asarray(points) @ matrix.T


def read_known_members(path):
    rows = []
    for line in Path(path).read_text(encoding="ascii").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        rows.append({"source_id": fields[0], "ra": float(fields[1]), "dec": float(fields[2]), "magnitude": float(fields[6])})
    return pd.DataFrame(rows)


def main():
    config_path = ROOT / "configs/p0554_macs1931_wide_field_analysis_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    acquisition = json.loads((ROOT / protocol["inputs"]["acquisition_protocol"]).read_text(encoding="utf-8"))
    provenance = json.loads((ROOT / protocol["inputs"]["provenance"]).read_text(encoding="utf-8-sig"))
    catalog_path = ROOT / protocol["inputs"]["catalog"]
    if sha256(catalog_path) != provenance["catalog"]["sha256"]:
        raise RuntimeError("wide-field catalog hash differs from acquisition provenance")
    catalog = pd.read_csv(catalog_path)
    if len(catalog) != int(acquisition["field"]["preflight_object_count"]):
        raise RuntimeError("wide-field catalog row count changed")
    settings = protocol["coordinate_and_matching"]
    catalog_position = project(catalog.ra, catalog.dec, settings)
    selected = selections(catalog, protocol)

    known = read_known_members(ROOT / protocol["inputs"]["known_member_catalog"])
    known_position = project(known.ra, known.dec, settings)
    cosine = math.cos(math.radians(float(settings["center_dec_deg"])))
    catalog_sky = np.column_stack([catalog.ra.to_numpy(float) * cosine, catalog.dec.to_numpy(float)]) * 3600.0
    known_sky = np.column_stack([known.ra.to_numpy(float) * cosine, known.dec.to_numpy(float)]) * 3600.0
    distance, index = cKDTree(catalog_sky).query(known_sky, k=1)
    match_gate = float(settings["known_member_crossmatch_radius_arcsec"])
    match_rows = []
    for known_index, (separation, catalog_index) in enumerate(zip(distance, index, strict=True)):
        matched = separation <= match_gate
        row = {
            "source_id": known.iloc[known_index].source_id,
            "known_x_kpc": known_position[known_index, 0],
            "known_y_kpc": known_position[known_index, 1],
            "matched": matched,
            "separation_arcsec": separation,
            "ls_id": int(catalog.iloc[catalog_index].ls_id) if matched else None,
            "ls_type": catalog.iloc[catalog_index].type if matched else None,
        }
        for kind, mask in selected.items():
            row[f"passes_{kind}"] = bool(mask[catalog_index]) if matched else False
        match_rows.append(row)
    matches = pd.DataFrame(match_rows)

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    audit_row = {
        "raw_catalog_objects": len(catalog),
        "extended_objects": int(catalog.type.astype(str).isin(protocol["quality"]["extended_types"]).sum()),
        "extended_clean_objects": int(selected["extended_no_redshift"].sum()),
        "objects_with_usable_r_flux": int((catalog.flux_r > 0.0).sum()),
        "objects_with_usable_i_flux": int((catalog.flux_i > 0.0).sum()),
        "objects_with_usable_z_flux": int((catalog.flux_z > 0.0).sum()),
        "objects_with_usable_photoz_i": int((catalog.z_phot_median_i >= 0.0).sum()),
    }
    pd.DataFrame([audit_row]).to_csv(
        output / protocol["outputs"]["catalog_audit"], index=False
    )
    matches.to_csv(output / protocol["outputs"]["known_member_matches"], index=False)
    if int(selected[protocol["density_test"]["primary_selection"]].sum()) == 0:
        empty_columns = ["selection_kind", "weight_kind", "kernel_width_kpc"]
        pd.DataFrame(columns=list(catalog.columns) + ["x_kpc", "y_kpc", "selection_kind"]).to_csv(
            output / protocol["outputs"]["selected_sources"], index=False
        )
        pd.DataFrame(columns=empty_columns).to_csv(
            output / protocol["outputs"]["density_scores"], index=False
        )
        pd.DataFrame(columns=empty_columns + ["rotation_angle_deg"]).to_csv(
            output / protocol["outputs"]["rotation_nulls"], index=False
        )
        report = {
            "report_version": "P0554-MACS1931-WIDE-FIELD-RESULTS-0.1.0",
            "status": "input_inadequate_no_density_score",
            "protocol": {"path": str(config_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(config_path)},
            "coverage": audit_row,
            "known_members": len(known),
            "known_members_crossmatched_within_1arcsec": int(matches.matched.sum()),
            "reason": "This DR10 patch contains g-band-only photometry: every r/i/z flux is zero and every i-band photo-z is the -99 sentinel. The frozen multi-band/photo-z selection is therefore empty.",
            "decision": "Do not replace the failed selection with g-only morphology in this star-crowded field. Preserve the catalog for provenance and seek Subaru, VISTA, DES, or spectroscopic wider-field membership data.",
            "density_claim_made": False,
            "limits": protocol["limits"],
        }
        (output / protocol["outputs"]["report"]).write_text(
            json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
        )
        figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        labels = ["g", "r", "i", "z", "photo-z"]
        values = [
            int((catalog.flux_g > 0.0).sum()),
            audit_row["objects_with_usable_r_flux"],
            audit_row["objects_with_usable_i_flux"],
            audit_row["objects_with_usable_z_flux"],
            audit_row["objects_with_usable_photoz_i"],
        ]
        axis.bar(labels, values, color=["tab:green", "0.7", "0.7", "0.7", "0.7"])
        axis.set(ylabel="catalog objects with usable value", title="MACS1931 Legacy DR10 input adequacy")
        figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
        plt.close(figure)
        summary = (
            "# MACS1931 wide-field counterpart test\n\n"
            "No density score was run. This DR10 patch is g-band-only: all r/i/z fluxes are zero and all i-band photo-z values are the -99 sentinel. A g-only morphology selection is not credible in this crowded stellar field.\n"
        )
        (output / protocol["outputs"]["summary"]).write_text(summary, encoding="utf-8")
        print(json.dumps(json_safe(report), indent=2), flush=True)
        return

    halo_protocol = json.loads((ROOT / protocol["inputs"]["halo_backtrack_protocol"]).read_text(encoding="utf-8"))
    system = next(row for row in halo_protocol["systems"] if row["label"] == "MACS1931")
    chain = ROOT / halo_protocol["inputs"]["chain_root"] / system["chain_directory"] / "bayes.dat"
    headers, samples, _ = thin_bayes_chain(chain, int(settings["posterior_samples"]))
    components = component_samples(headers, samples, float(settings["angular_scale_kpc_per_arcsec"]))
    halo = np.column_stack([
        components[int(settings["halo_object_id"])]["x_kpc"],
        components[int(settings["halo_object_id"])]["y_kpc"],
    ])

    score_rows = []
    null_rows = []
    selected_rows = []
    for selection_kind, mask in selected.items():
        block = catalog.loc[mask].copy()
        position = catalog_position[mask]
        block["x_kpc"] = position[:, 0]
        block["y_kpc"] = position[:, 1]
        block["selection_kind"] = selection_kind
        selected_rows.append(block)
        for weight_kind in protocol["density_test"]["weights"]:
            source_weight = weights(block, protocol["quality"], weight_kind)
            for width in protocol["density_test"]["kernel_widths_kpc"]:
                real = posterior_mean_density(position, source_weight, halo, float(width))
                null_values = []
                for angle in protocol["density_test"]["rotation_angles_deg"]:
                    value = posterior_mean_density(position, source_weight, rotate(halo, angle), float(width))
                    null_values.append(value)
                    null_rows.append({
                        "selection_kind": selection_kind,
                        "weight_kind": weight_kind,
                        "kernel_width_kpc": float(width),
                        "rotation_angle_deg": float(angle),
                        "posterior_mean_density": value,
                    })
                null_array = np.asarray(null_values)
                score_rows.append({
                    "selection_kind": selection_kind,
                    "weight_kind": weight_kind,
                    "kernel_width_kpc": float(width),
                    "selected_source_count": len(block),
                    "real_posterior_mean_density": real,
                    "rotation_null_median": float(np.median(null_array)),
                    "density_ratio_vs_null_median": real / max(float(np.median(null_array)), np.finfo(float).tiny),
                    "one_sided_rotation_p": float((1 + np.sum(null_array >= real)) / (1 + len(null_array))),
                    "rotation_percentile": float(np.mean(null_array <= real)),
                })
    scores = pd.DataFrame(score_rows)
    nulls = pd.DataFrame(null_rows)
    selected_table = pd.concat(selected_rows, ignore_index=True)
    primary = scores[
        scores.selection_kind.eq(protocol["density_test"]["primary_selection"])
        & scores.weight_kind.eq(protocol["density_test"]["primary_weight"])
        & scores.kernel_width_kpc.eq(float(protocol["density_test"]["primary_kernel_width_kpc"]))
    ].iloc[0]

    primary_mask = selected[protocol["density_test"]["primary_selection"]]
    primary_position = catalog_position[primary_mask]
    halo_median = np.median(halo, axis=0)
    nearest_distance, nearest_index = cKDTree(primary_position).query(halo_median, k=min(10, len(primary_position)))
    nearest_distance = np.atleast_1d(nearest_distance)
    nearest_index = np.atleast_1d(nearest_index)
    primary_catalog = catalog.loc[primary_mask].reset_index(drop=True)
    nearest = [
        {
            "rank": rank,
            "ls_id": int(primary_catalog.iloc[source_index].ls_id),
            "distance_to_median_halo_kpc": float(distance_value),
            "z_phot_median_i": float(primary_catalog.iloc[source_index].z_phot_median_i),
            "type": str(primary_catalog.iloc[source_index].type),
        }
        for rank, (distance_value, source_index) in enumerate(zip(nearest_distance, nearest_index, strict=True), start=1)
    ]
    field = acquisition["field"]
    corners = project(
        [field["ra_min_deg"], field["ra_min_deg"], field["ra_max_deg"], field["ra_max_deg"]],
        [field["dec_min_deg"], field["dec_max_deg"], field["dec_min_deg"], field["dec_max_deg"]],
        settings,
    )
    rotated = np.vstack([rotate(halo, angle) for angle in protocol["density_test"]["rotation_angles_deg"]])
    footprint_gate = (
        (rotated[:, 0] >= corners[:, 0].min())
        & (rotated[:, 0] <= corners[:, 0].max())
        & (rotated[:, 1] >= corners[:, 1].min())
        & (rotated[:, 1] <= corners[:, 1].max())
    )
    report = {
        "report_version": "P0554-MACS1931-WIDE-FIELD-RESULTS-0.1.0",
        "status": "complete",
        "protocol": {"path": str(config_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(config_path)},
        "coverage": {
            "raw_catalog_objects": len(catalog),
            "known_members": len(known),
            "known_members_crossmatched_within_1arcsec": int(matches.matched.sum()),
            "primary_selected_sources": int(primary.selected_source_count),
            "rotation_nulls": int(len(nulls)),
            "all_rotated_halo_samples_inside_catalog_footprint": bool(np.all(footprint_gate)),
        },
        "known_member_recovery": {
            kind: float(matches.loc[matches.matched, f"passes_{kind}"].mean())
            for kind in selected
        },
        "primary_density_test": primary.to_dict(),
        "all_density_tests": scores.to_dict("records"),
        "nearest_primary_candidates_to_halo_median": nearest,
        "halo_posterior": {
            "median_x_kpc": float(halo_median[0]),
            "median_y_kpc": float(halo_median[1]),
            "radial_distance_kpc": float(np.linalg.norm(halo_median)),
        },
        "interpretation_limit": "This tests a galaxy counterpart to a model-dependent halo position. It does not test whether gravity traveled there.",
        "limits": protocol["limits"],
    }
    selected_table.to_csv(output / protocol["outputs"]["selected_sources"], index=False)
    scores.to_csv(output / protocol["outputs"]["density_scores"], index=False)
    nulls.to_csv(output / protocol["outputs"]["rotation_nulls"], index=False)
    (output / protocol["outputs"]["report"]).write_text(json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    axes[0].scatter(catalog_position[:, 0], catalog_position[:, 1], s=1, color="0.8", alpha=0.25, label="all catalog")
    axes[0].scatter(primary_position[:, 0], primary_position[:, 1], s=9, c="tab:blue", alpha=0.75, label="photo-z candidates")
    axes[0].scatter(known_position[:, 0], known_position[:, 1], s=18, facecolors="none", edgecolors="tab:orange", linewidths=0.7, label="known central members")
    axes[0].scatter(halo[:, 0], halo[:, 1], s=5, color="crimson", alpha=0.12, label="halo posterior")
    axes[0].scatter([halo_median[0]], [halo_median[1]], marker="x", s=80, color="crimson", linewidths=2)
    axes[0].set(xlabel="east x (kpc)", ylabel="north y (kpc)", aspect="equal", title="Wide-field candidate galaxies and southern halo")
    axes[0].legend(fontsize=8, loc="upper right")
    primary_null = nulls[
        nulls.selection_kind.eq(protocol["density_test"]["primary_selection"])
        & nulls.weight_kind.eq(protocol["density_test"]["primary_weight"])
        & nulls.kernel_width_kpc.eq(float(protocol["density_test"]["primary_kernel_width_kpc"]))
    ]
    axes[1].hist(primary_null.posterior_mean_density, bins=15, color="0.65", edgecolor="white")
    axes[1].axvline(float(primary.real_posterior_mean_density), color="crimson", linewidth=2, label="actual halo angle")
    axes[1].set(xlabel="posterior-mean galaxy density", ylabel="rotated halo angles", title="Radius-preserving angular control")
    axes[1].legend()
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)
    summary = [
        "# MACS1931 wide-field counterpart test",
        "",
        f"The primary selection contains **{int(primary.selected_source_count)}** plausible sources. The galaxy density at the southern halo is **{float(primary.density_ratio_vs_null_median):.2f}x** the median rotated-halo density, with one-sided rotation p=**{float(primary.one_sided_rotation_p):.3f}**.",
        f"The nearest primary candidate is **{nearest[0]['distance_to_median_halo_kpc']:.1f} kpc** from the posterior median halo center.",
        "",
        "This identifies whether the halo has an obvious wider-field galaxy counterpart; it does not validate gravity routing.",
    ]
    (output / protocol["outputs"]["summary"]).write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps(json_safe(report), indent=2), flush=True)


if __name__ == "__main__":
    main()
