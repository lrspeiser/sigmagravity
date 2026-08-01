"""Backtrack baryonic origins of RELICS lensing-excess morphology."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, maximum_filter


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_gravity_arc_fresh_sample import (  # noqa: E402
    build_source_context,
    target_from_path,
)
from run_gravity_arc_tomography import deposit_points, normalized_in_aperture  # noqa: E402
from voidscreen.gravity_flow_inverse import (  # noqa: E402
    coarsen_destination,
    local_projection_excess,
    map_similarity,
    off_plane_arc_length,
    rasterize_transport_paths,
    solve_transport,
    source_route_table,
    transport_diagnostics,
    weighted_quantile,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def destination_variants(target, baryon_template, aperture):
    full = normalized_in_aperture(target, aperture)
    half, half_fit = local_projection_excess(
        target, baryon_template, aperture, scale_fraction=0.5
    )
    excess, excess_fit = local_projection_excess(
        target, baryon_template, aperture, scale_fraction=1.0
    )
    return {
        "full_kappa": (full, {
            "fitted_local_projection": 0.0,
            "applied_local_projection": 0.0,
            "positive_target_weight_removed": 0.0,
            "positive_residual_weight_before_renormalization": 1.0,
        }),
        "half_local_subtracted": (half, half_fit),
        "local_projected_excess": (excess, excess_fit),
    }


def dominant_route_rows(
    system,
    target_kind,
    plan,
    source_positions,
    source_weights,
    destination_positions,
    *,
    limit=250,
):
    flat = plan.ravel()
    order = np.argsort(flat)[::-1][: int(limit)]
    source_index, destination_index = np.unravel_index(order, plan.shape)
    cumulative = np.cumsum(flat[order])
    rows = []
    for rank, (i, j, weight, accumulated) in enumerate(
        zip(source_index, destination_index, flat[order], cumulative, strict=True),
        start=1,
    ):
        delta = destination_positions[j] - source_positions[i]
        rows.append(
            {
                "system": system,
                "target_kind": target_kind,
                "route_rank": rank,
                "source_index": int(i),
                "source_x_kpc": float(source_positions[i, 0]),
                "source_y_kpc": float(source_positions[i, 1]),
                "source_weight": float(source_weights[i]),
                "destination_x_kpc": float(destination_positions[j, 0]),
                "destination_y_kpc": float(destination_positions[j, 1]),
                "route_weight": float(weight),
                "cumulative_listed_weight": float(accumulated),
                "projected_length_kpc": float(np.linalg.norm(delta)),
            }
        )
    return rows


def peak_origin_rows(
    system,
    target_kind,
    target,
    x_grid,
    y_grid,
    radius_grid,
    plan,
    destination_positions,
    source_positions,
    source_weights,
):
    peaks = target == maximum_filter(target, size=11, mode="constant")
    peaks &= radius_grid <= 250.0
    indices = np.argwhere(peaks & (target > 0.0))
    indices = sorted(indices, key=lambda item: target[tuple(item)], reverse=True)[:5]
    rows = []
    for peak_rank, (iy, ix) in enumerate(indices, start=1):
        position = np.array([x_grid[iy, ix], y_grid[iy, ix]])
        near = np.linalg.norm(destination_positions - position[None, :], axis=1) <= 25.0
        contribution = np.sum(plan[:, near], axis=1)
        total = float(np.sum(contribution))
        for origin_rank, source_index in enumerate(np.argsort(contribution)[::-1][:5], start=1):
            rows.append(
                {
                    "system": system,
                    "target_kind": target_kind,
                    "peak_rank": peak_rank,
                    "peak_x_kpc": float(position[0]),
                    "peak_y_kpc": float(position[1]),
                    "peak_target_weight_within_25kpc": total,
                    "origin_rank": origin_rank,
                    "source_index": int(source_index),
                    "source_x_kpc": float(source_positions[source_index, 0]),
                    "source_y_kpc": float(source_positions[source_index, 1]),
                    "source_weight": float(source_weights[source_index]),
                    "fraction_of_peak_inflow": float(
                        contribution[source_index] / max(total, np.finfo(float).tiny)
                    ),
                    "source_to_peak_kpc": float(
                        np.linalg.norm(source_positions[source_index] - position)
                    ),
                }
            )
    return rows


def radial_shuffle_nulls(
    system,
    target_kind,
    source_positions,
    source_weights,
    destination_positions,
    destination_weights,
    *,
    entropy_length_kpc,
    count,
    seed,
    iterations,
    tolerance,
):
    rng = np.random.default_rng(int(seed))
    radius = np.linalg.norm(source_positions, axis=1)
    rows = []
    for shuffle_index in range(int(count)):
        angle = rng.uniform(0.0, 2.0 * np.pi, size=len(radius))
        shuffled = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
        plan, _ = solve_transport(
            shuffled,
            source_weights,
            destination_positions,
            destination_weights,
            entropy_length_kpc=entropy_length_kpc,
            iterations=iterations,
            tolerance=tolerance,
        )
        displacement = destination_positions[None, :, :] - shuffled[:, None, :]
        rms = float(np.sqrt(np.sum(plan * np.sum(np.square(displacement), axis=2))))
        source_normalized = source_weights / np.sum(source_weights)
        destination_normalized = destination_weights / np.sum(destination_weights)
        rows.append(
            {
                "system": system,
                "target_kind": target_kind,
                "shuffle_index": shuffle_index,
                "rms_transport_kpc": rms,
                "source_marginal_max_error": float(
                    np.max(np.abs(np.sum(plan, axis=1) - source_normalized))
                ),
                "target_marginal_max_error": float(
                    np.max(np.abs(np.sum(plan, axis=0) - destination_normalized))
                ),
            }
        )
    return rows


def make_figure(contexts, products, output):
    figure, axes = plt.subplots(len(contexts), 5, figsize=(16, 3.1 * len(contexts)))
    for row, context in enumerate(contexts):
        label = context.label
        mask = np.abs(context.axis_kpc) <= 310.0
        extent = (-310.0, 310.0, -310.0, 310.0)
        panels = [
            (products[label]["baryon"], "F160W member light"),
            (products[label]["lenstool_target"], "Lenstool excess"),
            (products[label]["lenstool_path"], "Lenstool min-path density"),
            (products[label]["glafic_target"], "GLAFIC excess"),
            (products[label]["glafic_path"], "GLAFIC min-path density"),
        ]
        for column, (image, title) in enumerate(panels):
            axis = axes[row, column]
            cropped = image[np.ix_(mask, mask)]
            axis.imshow(
                np.log1p(cropped / max(np.quantile(cropped, 0.98), np.finfo(float).tiny)),
                origin="lower",
                extent=extent,
                cmap="magma" if column in (0, 1, 3) else "viridis",
                interpolation="nearest",
            )
            sizes = 5.0 + 45.0 * np.sqrt(
                context.soft_weights / np.max(context.soft_weights)
            )
            axis.scatter(
                context.positions[:, 0],
                context.positions[:, 1],
                s=sizes,
                facecolors="none",
                edgecolors="white",
                linewidths=0.45,
                alpha=0.65,
            )
            axis.set(xlim=(-300, 300), ylim=(-300, 300), aspect="equal")
            if row == 0:
                axis.set_title(title, fontsize=10)
            if column == 0:
                axis.set_ylabel(f"{label}\ny (kpc)", fontsize=8)
            else:
                axis.set_yticklabels([])
            if row == len(contexts) - 1:
                axis.set_xlabel("x (kpc)")
            else:
                axis.set_xticklabels([])
    figure.suptitle(
        "Baryon-to-lensing-excess inverse transport (projected minimum paths; not measured trajectories)",
        y=0.999,
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def write_summary(report, output):
    aggregate = report["aggregate_primary_inverse"]
    nulls = report["radial_shuffle_control"]
    agreement = report["cross_method_agreement"]
    lines = [
        "# Gravity-flow inverse tomography",
        "",
        "## Bottom line",
        "",
        report["bottom_line"],
        "",
        "## Primary route scale",
        "",
        f"- Lenstool median cluster-level path: {aggregate['lenstool_median_of_median_path_kpc']:.1f} kpc.",
        f"- GLAFIC median cluster-level path: {aggregate['glafic_median_of_median_path_kpc']:.1f} kpc.",
        f"- Lenstool median inward-direction cosine: {aggregate['lenstool_median_cos_inward']:.3f}.",
        f"- GLAFIC median inward-direction cosine: {aggregate['glafic_median_cos_inward']:.3f}.",
        "",
        "## Controls",
        "",
        f"The real angular galaxy layout has a shorter RMS route than the median radial-angle shuffle in {nulls['systems_real_better_than_shuffle_median_lenstool']}/10 Lenstool maps and {nulls['systems_real_better_than_shuffle_median_glafic']}/10 GLAFIC maps.",
        f"Median path-map agreement between reconstruction methods is Pearson {agreement['median_path_map_pearson']:.3f} and JS {agreement['median_path_map_JS']:.4f}.",
        "",
        "## Interpretation boundary",
        "",
        "This identifies minimum-cost baryonic origins and projected destination couplings. It does not observe gravity lines, determine their off-plane height, or measure the absolute field multiplier. See `report.json`, `route_statistics.csv`, `dominant_routes.csv`, `peak_origins.csv`, and the figure for the complete outputs.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    protocol_path = ROOT / "configs" / "gravity_flow_inverse_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    acquisition = json.loads(
        (ROOT / protocol["inputs"]["acquisition_protocol"]).read_text(encoding="utf-8")
    )
    analysis = json.loads(
        (ROOT / protocol["inputs"]["analysis_protocol"]).read_text(encoding="utf-8")
    )
    audit = json.loads((ROOT / protocol["inputs"]["input_audit"]).read_text(encoding="utf-8"))
    if not audit["coverage_gate_passed"]:
        raise RuntimeError("fresh-sample geometry audit did not pass")
    settings = acquisition["spatial_preprocessing"]
    transport = protocol["inverse_transport"]
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    sources = pd.read_csv(ROOT / protocol["inputs"]["sources"])
    system_audit = pd.read_csv(ROOT / protocol["inputs"]["systems"]).set_index("system")
    raw = ROOT / acquisition["acquisition"]["output_directory"]

    route_rows = []
    source_rows = []
    dominant_rows = []
    peak_rows = []
    null_rows = []
    off_plane_rows = []
    method_rows = []
    path_maps = {}
    products = {}
    contexts = []
    primary_by_system = {}
    factor = int(round(transport["target_grid_spacing_kpc"] / settings["grid_spacing_kpc"]))

    for system_index, system in enumerate(acquisition["systems"]):
        label = system["label"]
        context, world = build_source_context(
            system, system_audit.loc[label], sources, settings
        )
        contexts.append(context)
        baryon = deposit_points(
            context, context.positions, context.soft_weights, width_kpc=20.0
        )
        baryon = normalized_in_aperture(baryon, context.aperture)
        models = {model["method"]: model for model in system["models"]}
        lenstool_dir = raw / "models" / system["slug"] / "lenstool"
        range_paths = sorted((lenstool_dir / "range").glob("*_kappa.fits"))
        target_sum = np.zeros_like(context.x_grid)
        for path in range_paths:
            target_sum += target_from_path(path, world, context, settings)
        lenstool_target = normalized_in_aperture(target_sum / len(range_paths), context.aperture)
        glafic = models["glafic"]
        glafic_path = raw / "models" / system["slug"] / "glafic" / glafic["best_filename"]
        glafic_target = target_from_path(glafic_path, world, context, settings)
        targets = {
            "lenstool_ensemble_mean": lenstool_target,
            "glafic_best": glafic_target,
        }
        center = np.sum(context.positions * context.soft_weights[:, None], axis=0)
        products[label] = {"baryon": baryon}
        primary_by_system[label] = {}

        for method_index, (target_kind, target) in enumerate(targets.items()):
            variants = destination_variants(target, baryon, context.aperture)
            for destination_kind, (destination, projection) in variants.items():
                positions, weights, _ = coarsen_destination(
                    destination,
                    context.x_grid,
                    context.y_grid,
                    factor=factor,
                    radius_kpc=settings["common_radius_kpc"],
                )
                for entropy_length in transport["entropy_lengths_kpc"]:
                    plan, _ = solve_transport(
                        context.positions,
                        context.soft_weights,
                        positions,
                        weights,
                        entropy_length_kpc=entropy_length,
                        iterations=transport["iterations"],
                        tolerance=transport["tolerance"],
                    )
                    diagnostics = transport_diagnostics(
                        plan,
                        context.positions,
                        context.soft_weights,
                        positions,
                        weights,
                        baryonic_center=center,
                    )
                    route_rows.append(
                        {
                            "system": label,
                            "target_kind": target_kind,
                            "destination_kind": destination_kind,
                            "entropy_length_kpc": float(entropy_length),
                            **projection,
                            **diagnostics,
                        }
                    )
                    if (
                        destination_kind == "local_projected_excess"
                        and float(entropy_length)
                        == float(transport["primary_entropy_length_kpc"])
                    ):
                        path_map = rasterize_transport_paths(
                            plan,
                            context.positions,
                            positions,
                            context.axis_kpc,
                            samples_per_path=transport["path_samples"],
                            retained_weight=transport["path_map_retained_transport_weight"],
                        )
                        key = f"{system['slug']}__{target_kind}"
                        path_maps[key] = path_map
                        primary_by_system[label][target_kind] = {
                            "target": destination,
                            "plan": plan,
                            "positions": positions,
                            "weights": weights,
                            "path_map": path_map,
                            "diagnostics": diagnostics,
                        }
                        for row in source_route_table(
                            plan,
                            context.positions,
                            context.soft_weights,
                            positions,
                            baryonic_center=center,
                        ):
                            source_rows.append(
                                {
                                    "system": label,
                                    "target_kind": target_kind,
                                    **row,
                                }
                            )
                        dominant_rows.extend(
                            dominant_route_rows(
                                label,
                                target_kind,
                                plan,
                                context.positions,
                                context.soft_weights,
                                positions,
                            )
                        )
                        peak_rows.extend(
                            peak_origin_rows(
                                label,
                                target_kind,
                                destination,
                                context.x_grid,
                                context.y_grid,
                                context.radius_grid,
                                plan,
                                positions,
                                context.positions,
                                context.soft_weights,
                            )
                        )
                        displacement = positions[None, :, :] - context.positions[:, None, :]
                        distance = np.linalg.norm(displacement, axis=2)
                        for height_ratio in transport["off_plane_height_ratios_for_length_only"]:
                            length = off_plane_arc_length(distance, height_ratio)
                            off_plane_rows.append(
                                {
                                    "system": label,
                                    "target_kind": target_kind,
                                    "height_to_projected_distance_ratio": float(height_ratio),
                                    "mean_arc_length_kpc": float(np.sum(plan * length)),
                                    "median_arc_length_kpc": weighted_quantile(length, plan, 0.5),
                                    "p90_arc_length_kpc": weighted_quantile(length, plan, 0.9),
                                }
                            )
                        null_rows.extend(
                            radial_shuffle_nulls(
                                label,
                                target_kind,
                                context.positions,
                                context.soft_weights,
                                positions,
                                weights,
                                entropy_length_kpc=entropy_length,
                                count=protocol["controls"]["radial_angle_shuffles"],
                                seed=(
                                    protocol["controls"]["shuffle_seed"]
                                    + 1000 * system_index
                                    + 100 * method_index
                                ),
                                iterations=protocol["controls"]["control_iterations"],
                                tolerance=protocol["controls"]["control_tolerance"],
                            )
                        )
            print(f"{label}: completed inverse routes for {target_kind}", flush=True)

        left = primary_by_system[label]["lenstool_ensemble_mean"]
        right = primary_by_system[label]["glafic_best"]
        similarity = map_similarity(left["path_map"], right["path_map"])
        method_rows.append(
            {
                "system": label,
                "path_map_JS": similarity["jensen_shannon"],
                "path_map_Pearson": similarity["pearson"],
                "delta_median_path_kpc_glafic_minus_lenstool": (
                    right["diagnostics"]["median_path_kpc"]
                    - left["diagnostics"]["median_path_kpc"]
                ),
                "delta_cos_inward_glafic_minus_lenstool": (
                    right["diagnostics"]["mean_cos_inward"]
                    - left["diagnostics"]["mean_cos_inward"]
                ),
            }
        )
        products[label].update(
            {
                "lenstool_target": left["target"],
                "lenstool_path": gaussian_filter(left["path_map"], sigma=1.0),
                "glafic_target": right["target"],
                "glafic_path": gaussian_filter(right["path_map"], sigma=1.0),
            }
        )

    route_frame = pd.DataFrame(route_rows)
    source_frame = pd.DataFrame(source_rows)
    dominant_frame = pd.DataFrame(dominant_rows)
    peak_frame = pd.DataFrame(peak_rows)
    null_frame = pd.DataFrame(null_rows)
    off_plane_frame = pd.DataFrame(off_plane_rows)
    method_frame = pd.DataFrame(method_rows)
    route_frame.to_csv(output / protocol["outputs"]["route_statistics"], index=False)
    source_frame.to_csv(output / protocol["outputs"]["source_routes"], index=False)
    dominant_frame.to_csv(output / protocol["outputs"]["dominant_routes"], index=False)
    peak_frame.to_csv(output / protocol["outputs"]["peak_origins"], index=False)
    null_frame.to_csv(output / protocol["outputs"]["null_costs"], index=False)
    off_plane_frame.to_csv(output / protocol["outputs"]["off_plane_lengths"], index=False)
    method_frame.to_csv(output / protocol["outputs"]["method_agreement"], index=False)
    np.savez_compressed(output / protocol["outputs"]["path_maps"], **path_maps)

    primary = route_frame[
        route_frame.destination_kind.eq("local_projected_excess")
        & route_frame.entropy_length_kpc.eq(transport["primary_entropy_length_kpc"])
    ]
    null_summary = []
    for (system, target_kind), block in null_frame.groupby(["system", "target_kind"]):
        observed = float(
            primary[
                primary.system.eq(system) & primary.target_kind.eq(target_kind)
            ].rms_transport_kpc.iloc[0]
        )
        values = block.rms_transport_kpc.to_numpy(float)
        null_summary.append(
            {
                "system": system,
                "target_kind": target_kind,
                "observed_rms_kpc": observed,
                "shuffle_median_rms_kpc": float(np.median(values)),
                "improvement_over_shuffle_median_fraction": float(
                    1.0 - observed / np.median(values)
                ),
                "one_sided_permutation_p": float(
                    (1 + np.sum(values <= observed)) / (1 + len(values))
                ),
            }
        )
    null_summary_frame = pd.DataFrame(null_summary)
    maximum_control_error = float(null_frame.source_marginal_max_error.max())
    if maximum_control_error > float(
        protocol["controls"]["maximum_control_source_marginal_error"]
    ):
        raise RuntimeError(
            f"control transport marginal error {maximum_control_error:.3g} exceeds gate"
        )
    null_summary_frame.to_csv(output / "radial_shuffle_summary.csv", index=False)

    lenstool_primary = primary[primary.target_kind.eq("lenstool_ensemble_mean")]
    glafic_primary = primary[primary.target_kind.eq("glafic_best")]
    report = {
        "report_version": protocol["protocol_version"],
        "status": "completed descriptive ten-cluster baryon-to-lensing-excess inverse tomography",
        "protocol_sha256": sha256(protocol_path),
        "coverage": {
            "systems": len(contexts),
            "hard_photoz_baryonic_sources": int(audit["totals"]["hard_photoz_members_300kpc"]),
            "lensing_reconstruction_methods": 2,
            "destination_definitions": 3,
            "entropy_scales": len(transport["entropy_lengths_kpc"]),
            "transport_solutions": int(len(route_frame) + len(null_frame)),
            "dominant_routes_saved": int(len(dominant_frame)),
        },
        "aggregate_primary_inverse": {
            "lenstool_median_of_median_path_kpc": float(lenstool_primary.median_path_kpc.median()),
            "glafic_median_of_median_path_kpc": float(glafic_primary.median_path_kpc.median()),
            "lenstool_median_p90_path_kpc": float(lenstool_primary.p90_path_kpc.median()),
            "glafic_median_p90_path_kpc": float(glafic_primary.p90_path_kpc.median()),
            "lenstool_median_cos_inward": float(lenstool_primary.mean_cos_inward.median()),
            "glafic_median_cos_inward": float(glafic_primary.mean_cos_inward.median()),
            "lenstool_median_fraction_ending_inward": float(lenstool_primary.fraction_ending_inward.median()),
            "glafic_median_fraction_ending_inward": float(glafic_primary.fraction_ending_inward.median()),
            "maximum_source_marginal_error": float(primary.source_marginal_max_error.max()),
            "maximum_target_marginal_error": float(primary.target_marginal_max_error.max()),
        },
        "radial_shuffle_control": {
            "maximum_source_marginal_error": maximum_control_error,
            "systems_real_better_than_shuffle_median_lenstool": int(
                np.sum(
                    null_summary_frame.target_kind.eq("lenstool_ensemble_mean")
                    & (null_summary_frame.improvement_over_shuffle_median_fraction > 0.0)
                )
            ),
            "systems_real_better_than_shuffle_median_glafic": int(
                np.sum(
                    null_summary_frame.target_kind.eq("glafic_best")
                    & (null_summary_frame.improvement_over_shuffle_median_fraction > 0.0)
                )
            ),
            "lenstool_median_improvement_fraction": float(
                null_summary_frame[
                    null_summary_frame.target_kind.eq("lenstool_ensemble_mean")
                ].improvement_over_shuffle_median_fraction.median()
            ),
            "glafic_median_improvement_fraction": float(
                null_summary_frame[
                    null_summary_frame.target_kind.eq("glafic_best")
                ].improvement_over_shuffle_median_fraction.median()
            ),
            "systems_p_le_0_05_lenstool": int(
                np.sum(
                    null_summary_frame[
                        null_summary_frame.target_kind.eq("lenstool_ensemble_mean")
                    ].one_sided_permutation_p
                    <= 0.05
                )
            ),
            "systems_p_le_0_05_glafic": int(
                np.sum(
                    null_summary_frame[
                        null_summary_frame.target_kind.eq("glafic_best")
                    ].one_sided_permutation_p
                    <= 0.05
                )
            ),
        },
        "cross_method_agreement": {
            "median_path_map_JS": float(method_frame.path_map_JS.median()),
            "median_path_map_pearson": float(method_frame.path_map_Pearson.median()),
            "range_path_map_pearson": [
                float(method_frame.path_map_Pearson.min()),
                float(method_frame.path_map_Pearson.max()),
            ],
        },
        "bottom_line": "The inversion gives a reproducible baryonic origin-to-excess coupling and a family of possible return arcs. Its scientific value depends on cross-method and angular-null stability; it remains a descriptive representation, not evidence that gravity physically traversed those paths.",
        "claim_limits": protocol["limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    make_figure(contexts, products, output / protocol["outputs"]["figure"])
    write_summary(report, output / protocol["outputs"]["summary"])
    print(json.dumps(json_safe(report), indent=2))


if __name__ == "__main__":
    main()
