#!/usr/bin/env python3
"""Audit the southern MACS1931 lens-halo endpoint with Subaru photo-z data."""

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
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0554_halo_backtrack_capacity import solve_checked  # noqa: E402
from voidscreen.gravity_flow_inverse import weighted_quantile  # noqa: E402
from voidscreen.halo_backtrack import (  # noqa: E402
    component_samples,
    posterior_component_destinations,
    thin_bayes_chain,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def read_photoz(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        names=columns,
        engine="c",
    )
    if frame.empty or frame.shape[1] != len(columns):
        raise RuntimeError("Subaru BPZ catalog did not parse with the frozen 27-column schema")
    return frame


def base_quality(catalog: pd.DataFrame, protocol: dict) -> np.ndarray:
    quality = protocol["quality"]
    result = np.ones(len(catalog), dtype=bool)
    if quality["require_positive_segment_area"]:
        result &= catalog.area.to_numpy(float) > 0.0
    result &= catalog.nfdet.to_numpy(float) >= int(quality["minimum_detected_filters"])
    result &= catalog.nfobs.to_numpy(float) >= int(quality["minimum_observed_filters"])
    result &= catalog.odds.to_numpy(float) >= float(quality["minimum_odds"])
    lo, hi = map(float, quality["ic_magnitude_range"])
    ic = catalog.IC.to_numpy(float)
    result &= np.isfinite(ic) & (ic >= lo) & (ic <= hi)
    result &= np.isfinite(catalog.zb.to_numpy(float))
    return result


def selections(catalog: pd.DataFrame, protocol: dict) -> dict[str, np.ndarray]:
    base = base_quality(catalog, protocol)
    zc = float(protocol["coordinate_and_target"]["cluster_redshift"])
    zb = catalog.zb.to_numpy(float)
    scale = 1.0 + zc
    return {
        "photoz_tight": base & (np.abs(zb - zc) <= 0.05 * scale),
        "photoz_medium": base & (np.abs(zb - zc) <= 0.10 * scale),
        "photoz_95_overlap": (
            base
            & (np.abs(zb - zc) <= 0.15 * scale)
            & (catalog.zbmin.to_numpy(float) <= zc)
            & (catalog.zbmax.to_numpy(float) >= zc)
        ),
    }


def source_weights(frame: pd.DataFrame, protocol: dict, kind: str) -> np.ndarray:
    if kind == "unit_count":
        return np.ones(len(frame), dtype=float)
    if kind == "capped_ic_luminosity":
        magnitude = frame.IC.to_numpy(float)
        weight = np.power(10.0, -0.4 * (magnitude - np.nanmedian(magnitude)))
        cap = float(
            np.quantile(
                weight[np.isfinite(weight) & (weight > 0.0)],
                float(protocol["quality"]["luminosity_weight_upper_quantile_cap"]),
            )
        )
        return np.minimum(weight, cap)
    raise ValueError(kind)


def density_score(source_position, weight, halo_position, width_kpc):
    difference = halo_position[:, None, :] - source_position[None, :, :]
    kernel = np.exp(-0.5 * np.sum(np.square(difference), axis=2) / float(width_kpc) ** 2)
    return float(np.mean(kernel @ np.asarray(weight, dtype=float)))


def rotate(points, angle_deg):
    angle = math.radians(float(angle_deg))
    matrix = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    return np.asarray(points, dtype=float) @ matrix.T


def read_known_members(path: Path) -> pd.DataFrame:
    rows = []
    for line in Path(path).read_text(encoding="ascii").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        rows.append(
            {
                "source_id": fields[0],
                "ra": float(fields[1]),
                "dec": float(fields[2]),
                "magnitude": float(fields[6]),
            }
        )
    return pd.DataFrame(rows)


def load_halo(protocol: dict):
    parent = json.loads(
        (ROOT / protocol["inputs"]["halo_backtrack_protocol"]).read_text(encoding="utf-8")
    )
    system = next(row for row in parent["systems"] if row["label"] == "MACS1931")
    chain = ROOT / parent["inputs"]["chain_root"] / system["chain_directory"] / "bayes.dat"
    settings = protocol["coordinate_and_target"]
    headers, samples, chain_rows = thin_bayes_chain(chain, int(settings["posterior_samples"]))
    components = component_samples(
        headers, samples, float(settings["angular_scale_kpc_per_arcsec"])
    )
    halo = np.column_stack(
        [
            components[int(settings["halo_object_id"])]["x_kpc"],
            components[int(settings["halo_object_id"])]["y_kpc"],
        ]
    )
    return parent, components, halo, chain_rows


def make_target(parent, components, target_kind, grid_spacing_kpc=None):
    target_spec = next(
        row
        for row in parent["halo_target"]["target_variants"]
        if row["target_kind"] == target_kind
    )
    settings = parent["halo_target"]
    spacing = float(
        settings["grid_spacing_kpc"] if grid_spacing_kpc is None else grid_spacing_kpc
    )
    axis = np.arange(
        float(settings["grid_min_kpc"]),
        float(settings["grid_max_kpc"]) + 0.5 * spacing,
        spacing,
    )
    return posterior_component_destinations(
        components,
        axis,
        width_mode=target_spec["width_mode"],
        width_kpc=target_spec["width_kpc"],
        weight_mode=target_spec["weight_mode"],
        maximum_radius_kpc=settings["maximum_radius_kpc"],
        minimum_relative_density=settings["minimum_relative_component_density"],
    )


def main():
    config_path = ROOT / "configs/p0554_macs1931_subaru_endpoint_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    provenance = json.loads(
        (ROOT / protocol["inputs"]["provenance"]).read_text(encoding="utf-8-sig")
    )
    product = next(
        row for row in provenance["products"] if row["role"] == "bpz_photometric_redshifts"
    )
    catalog_path = ROOT / protocol["inputs"]["photoz_catalog"]
    if sha256(catalog_path) != product["sha256"]:
        raise RuntimeError("Subaru BPZ catalog hash differs from acquisition provenance")

    catalog = read_photoz(catalog_path, protocol["catalog_columns"])
    settings = protocol["coordinate_and_target"]
    catalog_position = project(catalog.RA, catalog.Dec, settings)
    catalog["x_kpc"] = catalog_position[:, 0]
    catalog["y_kpc"] = catalog_position[:, 1]
    selected = selections(catalog, protocol)
    parent, components, halo, chain_rows = load_halo(protocol)
    halo_median = np.median(halo, axis=0)

    # The Subaru footprint is not perfectly rectangular, so this is a deliberately
    # conservative observed-row bounding-box check, supplemented by nfobs cuts.
    ra_halo = (
        halo[:, 0]
        / (
            3600.0
            * math.cos(math.radians(float(settings["center_dec_deg"])))
            * float(settings["angular_scale_kpc_per_arcsec"])
        )
        + float(settings["center_ra_deg"])
    )
    dec_halo = (
        halo[:, 1] / (3600.0 * float(settings["angular_scale_kpc_per_arcsec"]))
        + float(settings["center_dec_deg"])
    )
    inside = (
        (ra_halo >= catalog.RA.min())
        & (ra_halo <= catalog.RA.max())
        & (dec_halo >= catalog.Dec.min())
        & (dec_halo <= catalog.Dec.max())
    )
    inside_fraction = float(np.mean(inside))
    if inside_fraction < float(
        settings["minimum_posterior_samples_inside_catalog_footprint_fraction"]
    ):
        raise RuntimeError("too much of the halo posterior lies outside the catalog bounds")

    known = read_known_members(ROOT / protocol["inputs"]["known_member_catalog"])
    known_position = project(known.ra, known.dec, settings)
    cosine = math.cos(math.radians(float(settings["center_dec_deg"])))
    catalog_sky = np.column_stack(
        [catalog.RA.to_numpy(float) * cosine, catalog.Dec.to_numpy(float)]
    ) * 3600.0
    known_sky = np.column_stack(
        [known.ra.to_numpy(float) * cosine, known.dec.to_numpy(float)]
    ) * 3600.0
    separation, nearest_index = cKDTree(catalog_sky).query(known_sky, k=1)
    matches = []
    for known_index, (distance_arcsec, catalog_index) in enumerate(
        zip(separation, nearest_index, strict=True)
    ):
        matched = distance_arcsec <= float(settings["known_member_crossmatch_radius_arcsec"])
        row = {
            "source_id": known.iloc[known_index].source_id,
            "known_x_kpc": known_position[known_index, 0],
            "known_y_kpc": known_position[known_index, 1],
            "matched": matched,
            "separation_arcsec": distance_arcsec,
            "subaru_id": int(catalog.iloc[catalog_index].id) if matched else None,
        }
        for selection_kind, mask in selected.items():
            row[f"passes_{selection_kind}"] = bool(mask[catalog_index]) if matched else False
        matches.append(row)
    matches = pd.DataFrame(matches)

    score_rows = []
    null_rows = []
    candidate_rows = []
    selected_rows = []
    endpoint = protocol["endpoint_test"]
    for selection_kind, mask in selected.items():
        block = catalog.loc[mask].copy()
        selected_rows.append(block.assign(selection_kind=selection_kind))
        source_position = block[["x_kpc", "y_kpc"]].to_numpy(float)
        for weight_kind in endpoint["weights"]:
            weight = source_weights(block, protocol, weight_kind)
            for width in endpoint["kernel_widths_kpc"]:
                real = density_score(source_position, weight, halo, width)
                null = []
                for angle in endpoint["rotation_angles_deg"]:
                    value = density_score(source_position, weight, rotate(halo, angle), width)
                    null.append(value)
                    null_rows.append(
                        {
                            "selection_kind": selection_kind,
                            "weight_kind": weight_kind,
                            "kernel_width_kpc": float(width),
                            "rotation_angle_deg": float(angle),
                            "density_score": value,
                        }
                    )
                null = np.asarray(null, dtype=float)
                p_value = float((1 + np.sum(null >= real)) / (1 + len(null)))
                null_median = float(np.median(null))
                distances = np.linalg.norm(source_position - halo_median[None, :], axis=1)
                row = {
                    "selection_kind": selection_kind,
                    "weight_kind": weight_kind,
                    "kernel_width_kpc": float(width),
                    "selected_objects": len(block),
                    "real_density_score": real,
                    "null_median_density_score": null_median,
                    "density_ratio": real / max(null_median, np.finfo(float).tiny),
                    "rotation_p_value": p_value,
                    "nearest_selected_distance_kpc": float(np.min(distances)),
                }
                for radius in endpoint["aperture_radii_kpc"]:
                    row[f"objects_within_{int(radius)}kpc"] = int(np.sum(distances <= radius))
                score_rows.append(row)

        # Candidate rankings depend only on geometry, not the luminosity weighting.
        difference = halo[:, None, :] - source_position[None, :, :]
        kernel = np.mean(
            np.exp(-0.5 * np.sum(np.square(difference), axis=2) / 100.0**2), axis=0
        )
        order = np.argsort(kernel)[::-1]
        for rank, index in enumerate(order[:50], start=1):
            source = block.iloc[index]
            candidate_rows.append(
                {
                    "selection_kind": selection_kind,
                    "rank": rank,
                    "subaru_id": int(source.id),
                    "ra_deg": float(source.RA),
                    "dec_deg": float(source.Dec),
                    "x_kpc": float(source.x_kpc),
                    "y_kpc": float(source.y_kpc),
                    "distance_to_halo_median_kpc": float(
                        np.linalg.norm(source_position[index] - halo_median)
                    ),
                    "posterior_mean_w100_kernel": float(kernel[index]),
                    "IC": float(source.IC),
                    "zb": float(source.zb),
                    "zbmin": float(source.zbmin),
                    "zbmax": float(source.zbmax),
                    "odds": float(source.odds),
                    "nfdet": int(source.nfdet),
                }
            )

    scores = pd.DataFrame(score_rows)
    nulls = pd.DataFrame(null_rows)
    candidates = pd.DataFrame(candidate_rows)
    selected_frame = pd.concat(selected_rows, ignore_index=True)

    # Re-run the spent inverse with the wide-field primary selection. This is a
    # diagnostic of source-catalog truncation, never a forward validation.
    backtrack = protocol["descriptive_backtrack"]
    capacity = json.loads(
        (ROOT / protocol["inputs"]["capacity_protocol"]).read_text(encoding="utf-8")
    )
    destination, destination_weight, component_ids, _ = make_target(
        parent,
        components,
        backtrack["target_kind"],
        backtrack["transport_grid_spacing_kpc"],
    )
    primary_mask = selected[endpoint["primary_selection"]]
    transport_sources = catalog.loc[primary_mask].copy()
    transport_sources = transport_sources[
        np.hypot(transport_sources.x_kpc, transport_sources.y_kpc)
        <= float(backtrack["source_aperture_kpc"])
    ].reset_index(drop=True)
    source_position = transport_sources[["x_kpc", "y_kpc"]].to_numpy(float)
    distance_matrix = np.linalg.norm(
        destination[None, :, :] - source_position[:, None, :], axis=2
    )
    transport_rows = []
    origin_rows = []
    halo_id = int(settings["halo_object_id"])
    for weight_kind in backtrack["source_weights"]:
        weight = source_weights(transport_sources, protocol, weight_kind)
        weight /= np.sum(weight)
        plan, audit, iterations = solve_checked(
            capacity,
            source_position,
            weight,
            destination,
            destination_weight,
            float(backtrack["capacity_multiplier"]),
        )
        component_flow = np.sum(plan[:, component_ids == halo_id], axis=1)
        order = np.argsort(component_flow)[::-1]
        transport_rows.append(
            {
                "selection_kind": endpoint["primary_selection"],
                "weight_kind": weight_kind,
                "source_count": len(transport_sources),
                "capacity_multiplier": float(backtrack["capacity_multiplier"]),
                "solver_iterations": iterations,
                "mean_path_kpc": float(np.sum(plan * distance_matrix)),
                "median_path_kpc": weighted_quantile(distance_matrix, plan, 0.5),
                "p90_path_kpc": weighted_quantile(distance_matrix, plan, 0.9),
                "rms_transport_kpc": float(
                    np.sqrt(np.sum(plan * np.square(distance_matrix)))
                ),
                "southern_halo_top_origin_id": int(transport_sources.iloc[order[0]].id),
                "southern_halo_top_origin_distance_kpc": float(
                    np.linalg.norm(source_position[order[0]] - halo_median)
                ),
                "southern_halo_top_origin_fraction": float(
                    component_flow[order[0]] / np.sum(component_flow)
                ),
                **audit,
            }
        )
        for rank, index in enumerate(order[:20], start=1):
            source = transport_sources.iloc[index]
            origin_rows.append(
                {
                    "weight_kind": weight_kind,
                    "origin_rank": rank,
                    "subaru_id": int(source.id),
                    "x_kpc": float(source.x_kpc),
                    "y_kpc": float(source.y_kpc),
                    "IC": float(source.IC),
                    "zb": float(source.zb),
                    "distance_to_halo_median_kpc": float(
                        np.linalg.norm(source_position[index] - halo_median)
                    ),
                    "fraction_of_southern_halo_inflow": float(
                        component_flow[index] / np.sum(component_flow)
                    ),
                }
            )
    transports = pd.DataFrame(transport_rows)
    origins = pd.DataFrame(origin_rows)

    primary = scores[
        (scores.selection_kind == endpoint["primary_selection"])
        & (scores.weight_kind == endpoint["primary_weight"])
        & (scores.kernel_width_kpc == float(endpoint["primary_kernel_width_kpc"]))
    ].iloc[0]
    gate = endpoint["counterpart_gate"]
    counterpart = bool(
        primary.rotation_p_value <= float(gate["maximum_primary_p_value"])
        and primary.density_ratio >= float(gate["minimum_primary_density_ratio"])
        and primary.objects_within_200kpc
        >= int(gate["minimum_selected_objects_within_200kpc_of_posterior_median"])
    )
    outcome = (
        "significant_baryonic_counterpart_candidate"
        if counterpart
        else "no_frozen_significant_baryonic_counterpart"
    )

    audit = {
        "catalog_rows": len(catalog),
        "ra_range_deg": [float(catalog.RA.min()), float(catalog.RA.max())],
        "dec_range_deg": [float(catalog.Dec.min()), float(catalog.Dec.max())],
        "positive_area_rows": int((catalog.area > 0).sum()),
        "base_quality_rows": int(base_quality(catalog, protocol).sum()),
        "selection_counts": {kind: int(mask.sum()) for kind, mask in selected.items()},
        "halo_posterior_samples": len(halo),
        "halo_posterior_inside_catalog_bounds_fraction": inside_fraction,
        "known_members": len(known),
        "known_members_crossmatched_within_1arcsec": int(matches.matched.sum()),
        "known_member_primary_selection_recovery": int(
            matches[f"passes_{endpoint['primary_selection']}"] .sum()
        ),
    }

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([audit]).to_csv(output / protocol["outputs"]["catalog_audit"], index=False)
    matches.to_csv(output / protocol["outputs"]["known_member_matches"], index=False)
    selected_frame.to_csv(output / protocol["outputs"]["selected_sources"], index=False)
    scores.to_csv(output / protocol["outputs"]["endpoint_scores"], index=False)
    nulls.to_csv(output / protocol["outputs"]["rotation_nulls"], index=False)
    candidates.to_csv(output / protocol["outputs"]["candidate_counterparts"], index=False)
    transports.to_csv(output / protocol["outputs"]["transport_scores"], index=False)
    origins.to_csv(output / protocol["outputs"]["transport_origins"], index=False)

    previous = json.loads(
        (ROOT / protocol["inputs"]["member_aperture_report"]).read_text(encoding="utf-8")
    )
    previous_q4 = previous["qcap4_aperture_effect"]
    report = {
        "report_version": "P0554-MACS1931-SUBARU-ENDPOINT-RESULTS-0.1.0",
        "status": "complete",
        "outcome": outcome,
        "protocol": {
            "path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(config_path),
        },
        "catalog": audit,
        "halo": {
            "chain_rows": chain_rows,
            "posterior_median_x_kpc": float(halo_median[0]),
            "posterior_median_y_kpc": float(halo_median[1]),
            "posterior_median_radius_kpc": float(np.linalg.norm(halo_median)),
        },
        "primary_endpoint_test": primary.to_dict(),
        "frozen_counterpart_gate_passed": counterpart,
        "all_endpoint_scores": scores.to_dict("records"),
        "transport": {
            "previous_full_published_member_qcap4_rms_kpc": float(
                previous_q4["rms_route_450kpc"]
            ),
            "subaru_scores": transports.to_dict("records"),
        },
        "interpretation": (
            "The frozen endpoint test finds a statistically unusual concentration of plausible cluster-redshift galaxies at the second-halo position. This would make an omitted baryonic subgroup the conservative explanation before nonlocal gravity routing."
            if counterpart
            else "The frozen endpoint test does not find a sufficiently unusual concentration of plausible cluster-redshift galaxies at the second-halo position. The endpoint remains interesting, but catalog incompleteness and halo-model uncertainty prevent a gravity-routing claim."
        ),
        "limits": protocol["limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    primary_sources = catalog.loc[primary_mask]
    axes[0, 0].scatter(
        primary_sources.x_kpc,
        primary_sources.y_kpc,
        s=8,
        alpha=0.45,
        color="tab:blue",
        label="Subaru photo-z tight",
    )
    axes[0, 0].scatter(
        known_position[:, 0], known_position[:, 1], s=14, facecolors="none", edgecolors="0.4", label="published members"
    )
    axes[0, 0].scatter(halo[:, 0], halo[:, 1], s=7, alpha=0.2, color="tab:red", label="halo posterior")
    axes[0, 0].scatter(*halo_median, marker="x", s=90, color="black", label="halo median")
    axes[0, 0].set(
        xlim=(-1300, 1300), ylim=(-1300, 1300), xlabel="east x (kpc)", ylabel="north y (kpc)", title="Wide-field candidate members and halo endpoint"
    )
    axes[0, 0].set_aspect("equal")
    axes[0, 0].legend(fontsize=8, loc="upper right")

    primary_null = nulls[
        (nulls.selection_kind == endpoint["primary_selection"])
        & (nulls.weight_kind == endpoint["primary_weight"])
        & (nulls.kernel_width_kpc == float(endpoint["primary_kernel_width_kpc"]))
    ]
    axes[0, 1].hist(primary_null.density_score, bins=15, color="0.7", edgecolor="white")
    axes[0, 1].axvline(primary.real_density_score, color="tab:red", linewidth=2, label="real endpoint")
    axes[0, 1].set(
        xlabel="posterior-mean density score",
        ylabel="same-radius rotations",
        title=f"Frozen primary: ratio={primary.density_ratio:.2f}, p={primary.rotation_p_value:.3f}",
    )
    axes[0, 1].legend()

    recovery = [
        int(matches[f"passes_{kind}"].sum()) for kind in selected
    ]
    axes[1, 0].bar(list(selected), recovery, color=["tab:blue", "tab:orange", "tab:green"])
    axes[1, 0].axhline(len(known), color="black", linestyle="--", label=f"all known ({len(known)})")
    axes[1, 0].set(ylabel="published members recovered", title="Independent selection audit")
    axes[1, 0].tick_params(axis="x", rotation=18)
    axes[1, 0].legend()

    axes[1, 1].bar(
        transports.weight_kind,
        transports.rms_transport_kpc,
        color=["tab:purple", "tab:brown"],
    )
    axes[1, 1].axhline(
        float(previous_q4["rms_route_450kpc"]),
        color="black",
        linestyle="--",
        label="published-member inverse",
    )
    axes[1, 1].set(ylabel="RMS inverse route (kpc)", title="Spent inverse sensitivity")
    axes[1, 1].tick_params(axis="x", rotation=15)
    axes[1, 1].legend()
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)

    summary = (
        "# MACS1931 Subaru endpoint audit\n\n"
        f"Outcome: **{outcome}**. The frozen primary density ratio is {primary.density_ratio:.3f} "
        f"with rotation p={primary.rotation_p_value:.4f}; {int(primary.objects_within_200kpc)} selected "
        "objects lie within 200 kpc of the posterior median.\n\n"
        + report["interpretation"]
        + "\n"
    )
    (output / protocol["outputs"]["summary"]).write_text(summary, encoding="utf-8")
    print(json.dumps(json_safe(report), indent=2), flush=True)


if __name__ == "__main__":
    main()
