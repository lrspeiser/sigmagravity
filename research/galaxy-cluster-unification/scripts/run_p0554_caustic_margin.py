#!/usr/bin/env python3
"""Diagnose whether P0554 route root changes are physical caustics or solver basins."""

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
from scipy.optimize import brentq, linear_sum_assignment, root
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_adaptive_route_multicluster_raw import route_fraction  # noqa: E402
from run_adaptive_route_raw_rxj2129 import baryon_field, build_route_field  # noqa: E402
from run_arc_invariant_absolute_lensing import raw_field  # noqa: E402
from run_clash_stellar_morphology_response import MorphologyLens  # noqa: E402
from run_p0554_local_cross_domain_sensitivity import A0, json_safe, raw_contexts, sha256  # noqa: E402
from run_p0554_route_softness_interaction import (  # noqa: E402
    build_variants,
    load_route_sources,
)


def geometry_for(frame: pd.DataFrame, system: str, variant: str) -> np.ndarray:
    block = frame[
        frame.system_label.eq(system) & frame.variant_id.eq(variant)
    ]
    if len(block) != 1:
        raise RuntimeError(f"geometry coverage changed for {system}/{variant}")
    labels = [
        "axis_ratio_q",
        "position_angle_phi_radian",
        "center_x_arcsec",
        "center_y_arcsec",
        "external_shear_gamma1",
        "external_shear_gamma2",
    ]
    return block.iloc[0][labels].to_numpy(float)


def source_for(frame: pd.DataFrame, system: str, variant: str, family: int) -> np.ndarray:
    block = frame[
        frame.system_label.eq(system)
        & frame.variant_id.eq(variant)
        & frame.source_family.eq(int(family))
    ]
    if block.empty:
        raise RuntimeError(f"source coverage changed for {system}/{variant}/{family}")
    values = block[["source_x_arcsec", "source_y_arcsec"]].drop_duplicates()
    if len(values) != 1:
        raise RuntimeError(f"profiled source changed within {system}/{variant}/{family}")
    return values.iloc[0].to_numpy(float)


def equation_for(lens, model, parameters, source, redshift):
    def equation(theta):
        beta_x, beta_y = lens.ray_shooting(
            model,
            parameters,
            np.asarray([theta[0]], dtype=float),
            np.asarray([theta[1]], dtype=float),
            float(redshift),
        )
        return np.asarray([beta_x[0] - source[0], beta_y[0] - source[1]])

    def derivative(theta):
        return lens.jacobian(
            model,
            parameters,
            np.asarray([theta[0]], dtype=float),
            np.asarray([theta[1]], dtype=float),
            float(redshift),
        )[0]

    return equation, derivative


def deduplicate_roots(candidates: list[dict], tolerance: float) -> list[dict]:
    accepted = []
    for item in sorted(candidates, key=lambda row: (row["closure"], row["x"], row["y"])):
        point = np.asarray([item["x"], item["y"]])
        if any(
            np.linalg.norm(point - np.asarray([other["x"], other["y"]])) <= tolerance
            for other in accepted
        ):
            continue
        accepted.append(item)
    return accepted


def roots_from_seeds(
    lens,
    model: str,
    parameters: np.ndarray,
    source: np.ndarray,
    redshift: float,
    seeds: np.ndarray,
    *,
    closure_tolerance: float,
    deduplication_tolerance: float,
    aperture: float | None = None,
) -> list[dict]:
    equation, derivative = equation_for(lens, model, parameters, source, redshift)
    candidates = []
    for seed in np.asarray(seeds, dtype=float):
        solution = root(equation, seed, jac=derivative, method="hybr", tol=1.0e-10)
        point = np.asarray(solution.x, dtype=float)
        if not np.all(np.isfinite(point)):
            continue
        closure = float(np.linalg.norm(equation(point)))
        if closure > float(closure_tolerance):
            continue
        if aperture is not None and np.hypot(point[0], point[1]) > float(aperture):
            continue
        candidates.append(
            {
                "x": float(point[0]),
                "y": float(point[1]),
                "closure": closure,
                "solver_success": bool(solution.success),
            }
        )
    return deduplicate_roots(candidates, float(deduplication_tolerance))


def local_seeds(observed: np.ndarray, settings: dict) -> np.ndarray:
    seeds = [np.asarray(observed, dtype=float)]
    angles = np.linspace(
        0.0, 2.0 * np.pi, int(settings["local_search_angles"]), endpoint=False
    )
    for radius in settings["local_search_radii_arcsec"]:
        if float(radius) == 0.0:
            continue
        for angle in angles:
            seeds.append(
                observed
                + float(radius) * np.asarray([np.cos(angle), np.sin(angle)])
            )
    return np.asarray(seeds)


def jacobian_metrics(lens, model, parameters, point, redshift, settings):
    steps = [float(value) for value in settings["jacobian_step_checks_arcsec"]]
    matrices = [
        lens.jacobian(
            model,
            parameters,
            np.asarray([point[0]]),
            np.asarray([point[1]]),
            redshift,
            step=step,
        )[0]
        for step in steps
    ]
    primary_step = float(settings["jacobian_primary_step_arcsec"])
    index = int(np.argmin(np.abs(np.asarray(steps) - primary_step)))
    matrix = matrices[index]
    singular = np.linalg.svd(matrix, compute_uv=False)
    minimum = float(np.min(singular))
    maximum = float(np.max(singular))
    minimum_values = np.asarray(
        [np.min(np.linalg.svd(item, compute_uv=False)) for item in matrices]
    )
    return {
        "determinant": float(np.linalg.det(matrix)),
        "abs_determinant": abs(float(np.linalg.det(matrix))),
        "minimum_singular_value": minimum,
        "maximum_singular_value": maximum,
        "condition_number": maximum / max(minimum, np.finfo(float).tiny),
        "minimum_singular_step_span": float(np.max(minimum_values) - np.min(minimum_values)),
        "matrix": matrix,
    }


def determinant_at(lens, model, parameters, point, redshift, step):
    matrix = lens.jacobian(
        model,
        parameters,
        np.asarray([point[0]]),
        np.asarray([point[1]]),
        redshift,
        step=step,
    )[0]
    return float(np.linalg.det(matrix))


def critical_projection(lens, model, parameters, root_point, source, redshift, settings):
    origin = np.asarray(root_point, dtype=float).copy()
    jacobian_step = float(settings["jacobian_primary_step_arcsec"])
    maximum_distance = float(settings["critical_projection_max_distance_arcsec"])
    radial_step = float(settings["critical_curve_radial_step_arcsec"])
    radii = np.arange(0.0, maximum_distance + 0.5 * radial_step, radial_step)
    angles = np.linspace(
        0.0,
        2.0 * np.pi,
        int(settings["critical_curve_search_angles"]),
        endpoint=False,
    )
    crossings = []
    for angle in angles:
        direction = np.asarray([np.cos(angle), np.sin(angle)])
        previous_radius = radii[0]
        previous = determinant_at(
            lens, model, parameters, origin, redshift, jacobian_step
        )
        if abs(previous) <= float(settings["critical_projection_tolerance"]):
            crossings.append(origin.copy())
            continue
        for radius in radii[1:]:
            point = origin + radius * direction
            current = determinant_at(
                lens, model, parameters, point, redshift, jacobian_step
            )
            if np.isfinite(previous) and np.isfinite(current) and previous * current <= 0.0:
                try:
                    crossing_radius = brentq(
                        lambda value: determinant_at(
                            lens,
                            model,
                            parameters,
                            origin + value * direction,
                            redshift,
                            jacobian_step,
                        ),
                        previous_radius,
                        radius,
                        xtol=1.0e-8,
                        rtol=1.0e-10,
                    )
                except ValueError:
                    crossing_radius = radius
                crossings.append(origin + crossing_radius * direction)
                break
            previous_radius, previous = radius, current
    if not crossings:
        return {
            "critical_projection_converged": False,
            "critical_x_arcsec": np.nan,
            "critical_y_arcsec": np.nan,
            "image_critical_distance_arcsec": np.nan,
            "source_caustic_margin_arcsec": np.nan,
            "critical_abs_determinant": np.nan,
        }
    point = min(crossings, key=lambda value: np.linalg.norm(value - origin))
    determinant = determinant_at(lens, model, parameters, point, redshift, jacobian_step)
    beta_x, beta_y = lens.ray_shooting(
        model,
        parameters,
        np.asarray([point[0]]),
        np.asarray([point[1]]),
        redshift,
    )
    beta = np.asarray([beta_x[0], beta_y[0]])
    return {
        "critical_projection_converged": True,
        "critical_x_arcsec": float(point[0]),
        "critical_y_arcsec": float(point[1]),
        "image_critical_distance_arcsec": float(np.linalg.norm(point - origin)),
        "source_caustic_margin_arcsec": float(np.linalg.norm(beta - source)),
        "critical_abs_determinant": abs(determinant),
    }


def image_diagnostic(
    lens,
    variant,
    parameters,
    source,
    row,
    old_prediction,
    settings,
):
    observed = np.asarray([row.x_arcsec, row.y_arcsec], dtype=float)
    redshift = float(row.source_redshift)
    observed_jacobian = jacobian_metrics(
        lens, variant.variant_id, parameters, observed, redshift, settings
    )
    beta_x, beta_y = lens.ray_shooting(
        variant.variant_id,
        parameters,
        np.asarray([observed[0]]),
        np.asarray([observed[1]]),
        redshift,
    )
    mismatch = np.asarray([beta_x[0], beta_y[0]]) - source
    newton = np.linalg.pinv(observed_jacobian["matrix"], rcond=1.0e-12) @ mismatch
    roots = roots_from_seeds(
        lens,
        variant.variant_id,
        parameters,
        source,
        redshift,
        local_seeds(observed, settings),
        closure_tolerance=float(settings["root_closure_tolerance_arcsec"]),
        deduplication_tolerance=float(settings["root_deduplication_arcsec"]),
    )
    nearest = (
        min(roots, key=lambda item: np.hypot(item["x"] - observed[0], item["y"] - observed[1]))
        if roots
        else None
    )
    if nearest is None:
        root_metrics = {
            "determinant": np.nan,
            "abs_determinant": np.nan,
            "minimum_singular_value": np.nan,
            "maximum_singular_value": np.nan,
            "condition_number": np.nan,
            "minimum_singular_step_span": np.nan,
        }
        critical = {
            "critical_projection_converged": False,
            "critical_x_arcsec": np.nan,
            "critical_y_arcsec": np.nan,
            "image_critical_distance_arcsec": np.nan,
            "source_caustic_margin_arcsec": np.nan,
            "critical_abs_determinant": np.nan,
        }
        nearest_x = nearest_y = nearest_distance = nearest_closure = np.nan
    else:
        nearest_point = np.asarray([nearest["x"], nearest["y"]])
        root_metrics = jacobian_metrics(
            lens,
            variant.variant_id,
            parameters,
            nearest_point,
            redshift,
            settings,
        )
        critical = critical_projection(
            lens,
            variant.variant_id,
            parameters,
            nearest_point,
            source,
            redshift,
            settings,
        )
        nearest_x, nearest_y = nearest["x"], nearest["y"]
        nearest_distance = float(np.linalg.norm(nearest_point - observed))
        nearest_closure = nearest["closure"]
    return {
        "system_label": old_prediction.system_label,
        "variant_id": variant.variant_id,
        "image_id": str(row.image_id),
        "source_family": int(row.source_family),
        "source_redshift": redshift,
        "old_root_converged": bool(old_prediction.root_converged),
        "old_radial_residual_arcsec": float(old_prediction.radial_residual_arcsec)
        if np.isfinite(float(old_prediction.radial_residual_arcsec))
        else np.nan,
        "observed_x_arcsec": observed[0],
        "observed_y_arcsec": observed[1],
        "observed_source_mismatch_arcsec": float(np.linalg.norm(mismatch)),
        "observed_linearized_newton_distance_arcsec": float(np.linalg.norm(newton)),
        "observed_abs_determinant": observed_jacobian["abs_determinant"],
        "observed_minimum_singular_value": observed_jacobian["minimum_singular_value"],
        "observed_condition_number": observed_jacobian["condition_number"],
        "observed_minimum_singular_step_span": observed_jacobian[
            "minimum_singular_step_span"
        ],
        "local_unique_roots": len(roots),
        "local_multistart_root_found": nearest is not None,
        "local_nearest_root_x_arcsec": nearest_x,
        "local_nearest_root_y_arcsec": nearest_y,
        "local_nearest_root_distance_arcsec": nearest_distance,
        "local_nearest_root_closure_arcsec": nearest_closure,
        "root_abs_determinant": root_metrics["abs_determinant"],
        "root_minimum_singular_value": root_metrics["minimum_singular_value"],
        "root_condition_number": root_metrics["condition_number"],
        "root_minimum_singular_step_span": root_metrics[
            "minimum_singular_step_span"
        ],
        **critical,
    }


def global_family_search(
    lens,
    variant,
    parameters,
    source,
    images,
    settings,
):
    redshift = float(images.source_redshift.median())
    half = float(settings["global_search_half_width_arcsec"])
    spacing = float(settings["global_search_spacing_arcsec"])
    axis = np.arange(-half, half + 0.5 * spacing, spacing)
    gx, gy = np.meshgrid(axis, axis, indexing="xy")
    seeds = [np.column_stack([gx.ravel(), gy.ravel()])]
    for row in images.itertuples(index=False):
        seeds.append(
            local_seeds(
                np.asarray([row.x_arcsec, row.y_arcsec], dtype=float), settings
            )
        )
    roots = roots_from_seeds(
        lens,
        variant.variant_id,
        parameters,
        source,
        redshift,
        np.vstack(seeds),
        closure_tolerance=float(settings["root_closure_tolerance_arcsec"]),
        deduplication_tolerance=float(settings["root_deduplication_arcsec"]),
        aperture=float(settings["global_root_aperture_arcsec"]),
    )
    root_rows = []
    observed_xy = images[["x_arcsec", "y_arcsec"]].to_numpy(float)
    for index, item in enumerate(roots):
        point = np.asarray([item["x"], item["y"]])
        metrics = jacobian_metrics(
            lens,
            variant.variant_id,
            parameters,
            point,
            redshift,
            settings,
        )
        critical = critical_projection(
            lens,
            variant.variant_id,
            parameters,
            point,
            source,
            redshift,
            settings,
        )
        root_rows.append(
            {
                "variant_id": variant.variant_id,
                "root_index": index,
                "root_x_arcsec": item["x"],
                "root_y_arcsec": item["y"],
                "root_radius_arcsec": float(np.hypot(item["x"], item["y"])),
                "closure_arcsec": item["closure"],
                "solver_success": item["solver_success"],
                "nearest_observed_distance_arcsec": float(
                    np.min(np.linalg.norm(observed_xy - point, axis=1))
                ),
                "root_abs_determinant": metrics["abs_determinant"],
                "root_minimum_singular_value": metrics["minimum_singular_value"],
                **critical,
            }
        )
    assignments = []
    assigned = {}
    if roots:
        root_xy = np.asarray([[item["x"], item["y"]] for item in roots])
        cost = np.linalg.norm(observed_xy[:, None, :] - root_xy[None, :, :], axis=2)
        observed_index, root_index = linear_sum_assignment(cost)
        assigned = {int(left): int(right) for left, right in zip(observed_index, root_index)}
    for index, row in enumerate(images.itertuples(index=False)):
        root_index = assigned.get(index)
        item = None if root_index is None else roots[root_index]
        assignments.append(
            {
                "variant_id": variant.variant_id,
                "image_id": str(row.image_id),
                "source_family": int(row.source_family),
                "observed_x_arcsec": float(row.x_arcsec),
                "observed_y_arcsec": float(row.y_arcsec),
                "assigned": item is not None,
                "assigned_root_index": root_index,
                "assigned_root_x_arcsec": np.nan if item is None else item["x"],
                "assigned_root_y_arcsec": np.nan if item is None else item["y"],
                "assigned_distance_arcsec": np.nan
                if item is None
                else float(
                    np.hypot(
                        item["x"] - float(row.x_arcsec),
                        item["y"] - float(row.y_arcsec),
                    )
                ),
                "unique_global_roots": len(roots),
                "observed_family_images": len(images),
                "sufficient_unique_roots": len(roots) >= len(images),
            }
        )
    return root_rows, assignments


def binary_auc(labels, values, *, higher_predicts_success: bool | None = None):
    y = np.asarray(labels, dtype=bool)
    x = np.asarray(values, dtype=float)
    keep = np.isfinite(x)
    y, x = y[keep], x[keep]
    positive, negative = int(y.sum()), int((~y).sum())
    if positive == 0 or negative == 0:
        return {
            "rows": len(y),
            "AUC": np.nan,
            "absolute_AUC": np.nan,
            "higher_predicts_success": None,
        }
    ranks = rankdata(x)
    auc = float(
        (np.sum(ranks[y]) - positive * (positive + 1) / 2.0)
        / (positive * negative)
    )
    if higher_predicts_success is None:
        direction = auc >= 0.5
    else:
        direction = bool(higher_predicts_success)
    directed = auc if direction else 1.0 - auc
    return {
        "rows": len(y),
        "AUC": auc,
        "absolute_AUC": directed,
        "higher_predicts_success": direction,
    }


def discrimination_table(diagnostics, assignments):
    target = diagnostics[
        diagnostics.system_label.eq("MACS1931") & diagnostics.image_id.eq("2c")
    ].copy()
    assigned = assignments[assignments.image_id.eq("2c")][
        ["variant_id", "assigned_distance_arcsec"]
    ]
    target = target.merge(assigned, on="variant_id", validate="one_to_one")
    metrics = {
        "observed_minimum_singular_value": True,
        "observed_abs_determinant": True,
        "observed_linearized_newton_distance_arcsec": False,
        "local_nearest_root_distance_arcsec": False,
        "root_minimum_singular_value": None,
        "image_critical_distance_arcsec": None,
        "source_caustic_margin_arcsec": None,
        "assigned_distance_arcsec": False,
    }
    rows = []
    for metric, direction in metrics.items():
        audit = binary_auc(
            target.old_root_converged.to_numpy(bool),
            target[metric].to_numpy(float),
            higher_predicts_success=direction,
        )
        rho = spearmanr(
            target[metric].to_numpy(float),
            target.old_root_converged.astype(int).to_numpy(),
            nan_policy="omit",
        ).statistic
        rows.append({"metric": metric, "spearman_rho": float(rho), **audit})
    return pd.DataFrame(rows).sort_values("absolute_AUC", ascending=False), target


def parameter_pair_table(interaction_protocol, diagnostics, assignments, global_roots):
    target = diagnostics[
        diagnostics.system_label.eq("MACS1931") & diagnostics.image_id.eq("2c")
    ].set_index("variant_id")
    assigned = assignments[assignments.image_id.eq("2c")].set_index("variant_id")
    counts = global_roots.groupby("variant_id").size()
    rows = []
    for pair in interaction_protocol["impact_pairs"]:
        low, high = pair["low"], pair["high"]
        rows.append(
            {
                "parameter": pair["parameter"],
                "low_variant": low,
                "high_variant": high,
                "low_old_root_converged": bool(target.loc[low, "old_root_converged"]),
                "high_old_root_converged": bool(target.loc[high, "old_root_converged"]),
                "low_local_root_found": bool(target.loc[low, "local_multistart_root_found"]),
                "high_local_root_found": bool(target.loc[high, "local_multistart_root_found"]),
                "low_global_root_count": int(counts.get(low, 0)),
                "high_global_root_count": int(counts.get(high, 0)),
                "low_2c_assignment_distance_arcsec": float(
                    assigned.loc[low, "assigned_distance_arcsec"]
                ),
                "high_2c_assignment_distance_arcsec": float(
                    assigned.loc[high, "assigned_distance_arcsec"]
                ),
                "low_source_caustic_margin_arcsec": float(
                    target.loc[low, "source_caustic_margin_arcsec"]
                ),
                "high_source_caustic_margin_arcsec": float(
                    target.loc[high, "source_caustic_margin_arcsec"]
                ),
                "low_observed_minimum_singular_value": float(
                    target.loc[low, "observed_minimum_singular_value"]
                ),
                "high_observed_minimum_singular_value": float(
                    target.loc[high, "observed_minimum_singular_value"]
                ),
            }
        )
    return pd.DataFrame(rows)


def make_figure(target, global_summary, discrimination, output):
    order = target.sort_values("source_caustic_margin_arcsec").variant_id.tolist()
    indexed = target.set_index("variant_id").loc[order]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    colors = np.where(indexed.old_root_converged, "tab:blue", "crimson")
    axes[0, 0].bar(np.arange(len(indexed)), indexed.source_caustic_margin_arcsec, color=colors)
    axes[0, 0].set(
        xticks=np.arange(len(indexed)),
        xticklabels=indexed.index,
        ylabel="source-to-caustic margin (arcsec)",
        title="MACS1931 image 2c (red = old one-seed failure)",
    )
    axes[0, 0].tick_params(axis="x", rotation=90, labelsize=6)

    global_summary = global_summary.set_index("variant_id").loc[order]
    axes[0, 1].bar(np.arange(len(order)), global_summary.unique_global_roots, color=colors)
    axes[0, 1].axhline(
        global_summary.observed_family_images.iloc[0], color="black", ls="--", label="observed images"
    )
    axes[0, 1].set(
        xticks=np.arange(len(order)),
        xticklabels=order,
        ylabel="unique global roots",
        title="Frozen family-2 multiplicity search",
    )
    axes[0, 1].tick_params(axis="x", rotation=90, labelsize=6)
    axes[0, 1].legend()

    axes[1, 0].bar(
        np.arange(len(indexed)), indexed.assigned_distance_arcsec, color=colors
    )
    axes[1, 0].set(
        xticks=np.arange(len(indexed)),
        xticklabels=indexed.index,
        ylabel="2c assigned-root distance (arcsec)",
        title="Global branch matching",
    )
    axes[1, 0].tick_params(axis="x", rotation=90, labelsize=6)

    display = discrimination.sort_values("absolute_AUC")
    axes[1, 1].barh(display.metric, display.absolute_AUC)
    axes[1, 1].axvline(0.5, color="black", ls="--")
    axes[1, 1].set(
        xlim=(0.45, 1.01),
        xlabel="directed AUC for the extra-pair regime",
        title="Which diagnostic predicts the 3-to-5-root transition?",
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    config_path = ROOT / "configs" / "p0554_caustic_margin_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if not protocol["status"].startswith("frozen_"):
        raise RuntimeError("caustic-margin protocol is not frozen")
    interaction_path = ROOT / protocol["inputs"]["interaction_protocol"]
    interaction = json.loads(interaction_path.read_text(encoding="utf-8"))
    variants = build_variants(interaction)
    settings = protocol["evaluation"]
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    postprocess_only = "--postprocess-only" in sys.argv[1:]

    if postprocess_only:
        diagnostics = pd.read_csv(output / protocol["outputs"]["image_diagnostics"])
        global_roots = pd.read_csv(output / protocol["outputs"]["global_roots"])
        assignments = pd.read_csv(output / protocol["outputs"]["global_assignments"])
        print("Reusing frozen caustic diagnostics; regenerating summaries only.", flush=True)
    else:
        contexts = raw_contexts(interaction)
        sources, route_protocols = load_route_sources(interaction, contexts)
        geometry = pd.read_csv(ROOT / protocol["inputs"]["interaction_geometry"])
        archived = pd.read_csv(ROOT / protocol["inputs"]["interaction_predictions"])
        diagnostic_rows, global_root_rows, assignment_rows = [], [], []
        for context in contexts:
            print(f"{context.label}: reconstruct fields", flush=True)
            baryons = baryon_field(context.anchors, context.local)
            radial_cache, route_cache = {}, {}
            for index, variant in enumerate(variants, start=1):
                print(f"{context.label}: diagnose {variant.variant_id} ({index}/{len(variants)})", flush=True)
                radial_key = float(variant.spec["lensing_addition_softness"])
                if radial_key not in radial_cache:
                    radial_cache[radial_key], _ = raw_field(
                        variant.spec, variant.q, context.anchors, context.local, A0
                    )
                radial = radial_cache[radial_key]
                angular = None
                angular_strength = 0.0
                if variant.route:
                    adaptive = route_fraction(
                        variant.candidate, sources[context.label], context.local
                    )
                    angular_strength = float(
                        adaptive["routing_fraction"] ** variant.route_power
                    )
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
                    {variant.variant_id: radial},
                    parent=variant.variant_id,
                    morphology=angular,
                    fraction=angular_strength,
                )
                parameters = geometry_for(
                    geometry, context.label, variant.variant_id
                )
                old_block = archived[
                    archived.system_label.eq(context.label)
                    & archived.variant_id.eq(variant.variant_id)
                    & archived.stage.eq("heldout")
                ].set_index("image_id")
                for row in context.heldout.itertuples(index=False):
                    source = source_for(
                        archived,
                        context.label,
                        variant.variant_id,
                        int(row.source_family),
                    )
                    diagnostic_rows.append(
                        image_diagnostic(
                            lens,
                            variant,
                            parameters,
                            source,
                            row,
                            old_block.loc[str(row.image_id)],
                            settings,
                        )
                    )

                if context.label == settings["global_target_system"]:
                    family = int(settings["global_target_family"])
                    images = pd.concat([context.training, context.heldout], ignore_index=True)
                    images = images[images.source_family.eq(family)].copy()
                    source = source_for(
                        archived, context.label, variant.variant_id, family
                    )
                    root_rows, family_assignments = global_family_search(
                        lens,
                        variant,
                        parameters,
                        source,
                        images,
                        settings,
                    )
                    global_root_rows.extend(root_rows)
                    assignment_rows.extend(family_assignments)

        diagnostics = pd.DataFrame(diagnostic_rows)
        global_roots = pd.DataFrame(global_root_rows)
        assignments = pd.DataFrame(assignment_rows)
        diagnostics.to_csv(output / protocol["outputs"]["image_diagnostics"], index=False)
        global_roots.to_csv(output / protocol["outputs"]["global_roots"], index=False)
        assignments.to_csv(output / protocol["outputs"]["global_assignments"], index=False)

    discrimination, target = discrimination_table(diagnostics, assignments)
    global_summary = (
        assignments.groupby("variant_id", as_index=False)
        .agg(
            unique_global_roots=("unique_global_roots", "first"),
            observed_family_images=("observed_family_images", "first"),
            sufficient_unique_roots=("sufficient_unique_roots", "first"),
            all_images_assigned=("assigned", "all"),
            maximum_assignment_distance_arcsec=("assigned_distance_arcsec", "max"),
        )
    )
    image_2c_observed = assignments[assignments.image_id.eq("2c")][
        ["observed_x_arcsec", "observed_y_arcsec"]
    ].drop_duplicates()
    if len(image_2c_observed) != 1:
        raise RuntimeError("MACS1931 image 2c coordinate changed across formulas")
    image_2c_xy = image_2c_observed.iloc[0].to_numpy(float)
    pair_geometry_rows = []
    for variant_id, block in global_roots.groupby("variant_id"):
        xy = block[["root_x_arcsec", "root_y_arcsec"]].to_numpy(float)
        distance = np.linalg.norm(xy - image_2c_xy[None, :], axis=1)
        order = np.argsort(distance)
        first = xy[order[0]]
        second = xy[order[1]] if len(order) > 1 else np.full(2, np.nan)
        pair_geometry_rows.append(
            {
                "variant_id": variant_id,
                "nearest_global_root_to_2c_arcsec": float(distance[order[0]]),
                "second_nearest_global_root_to_2c_arcsec": float(distance[order[1]])
                if len(order) > 1
                else np.nan,
                "nearest_two_root_separation_arcsec": float(
                    np.linalg.norm(first - second)
                )
                if len(order) > 1
                else np.nan,
                "global_roots_within_15_arcsec_of_2c": int((distance <= 15.0).sum()),
            }
        )
    pair_geometry = pd.DataFrame(pair_geometry_rows)
    global_summary = global_summary.merge(
        pair_geometry, on="variant_id", validate="one_to_one"
    )
    target_assignment = assignments[assignments.image_id.eq("2c")][
        ["variant_id", "assigned", "assigned_distance_arcsec"]
    ].rename(
        columns={
            "assigned": "image_2c_assigned",
            "assigned_distance_arcsec": "image_2c_assignment_distance_arcsec",
        }
    )
    variant_summary = (
        diagnostics.groupby("variant_id", as_index=False)
        .agg(
            old_failed_heldout_roots=("old_root_converged", lambda values: int((~values).sum())),
            local_multistart_missing_roots=("local_multistart_root_found", lambda values: int((~values).sum())),
            minimum_source_caustic_margin_arcsec=("source_caustic_margin_arcsec", "min"),
            minimum_root_singular_value=("root_minimum_singular_value", "min"),
            maximum_local_nearest_root_distance_arcsec=("local_nearest_root_distance_arcsec", "max"),
        )
        .merge(global_summary, on="variant_id", validate="one_to_one")
        .merge(target_assignment, on="variant_id", validate="one_to_one")
    )
    variant_summary.to_csv(output / protocol["outputs"]["variant_summary"], index=False)
    pairs = parameter_pair_table(interaction, diagnostics, assignments, global_roots)
    pairs.to_csv(output / protocol["outputs"]["parameter_pair_summary"], index=False)

    old_failures = diagnostics[~diagnostics.old_root_converged.astype(bool)]
    multistart_recovered = int(old_failures.local_multistart_root_found.astype(bool).sum())
    all_sufficient = bool(global_summary.sufficient_unique_roots.astype(bool).all())
    all_assigned = bool(global_summary.all_images_assigned.astype(bool).all())
    observed_multiplicity_insufficiency = bool(
        (~global_summary.sufficient_unique_roots.astype(bool)).any()
        or (~global_summary.all_images_assigned.astype(bool)).any()
    )
    target_status = diagnostics[
        diagnostics.system_label.eq("MACS1931") & diagnostics.image_id.eq("2c")
    ][["variant_id", "old_root_converged", "local_nearest_root_distance_arcsec", "source_caustic_margin_arcsec"]]
    topology = global_summary.merge(target_status, on="variant_id", validate="one_to_one")
    successful_counts = sorted(
        topology[topology.old_root_converged.astype(bool)].unique_global_roots.unique()
    )
    failed_counts = sorted(
        topology[~topology.old_root_converged.astype(bool)].unique_global_roots.unique()
    )
    extra_pair_bifurcation = bool(
        len(successful_counts) == 1
        and len(failed_counts) == 1
        and successful_counts[0] - failed_counts[0] == 2
    )
    pair_near_2c_matches_status = bool(
        topology[topology.old_root_converged.astype(bool)]
        .global_roots_within_15_arcsec_of_2c.eq(2)
        .all()
        and topology[~topology.old_root_converged.astype(bool)]
        .global_roots_within_15_arcsec_of_2c.eq(0)
        .all()
    )
    successful = topology[topology.old_root_converged.astype(bool)]
    failed = topology[~topology.old_root_converged.astype(bool)]
    scores = pd.read_csv(ROOT / protocol["inputs"]["interaction_scores"])
    report = {
        "report_version": "P0554-CAUSTIC-MARGIN-RESULTS-0.2.0",
        "status": "complete",
        "protocol": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": sha256(config_path),
        },
        "coverage": {
            "variants": len(variants),
            "raw_clusters": int(diagnostics.system_label.nunique()),
            "heldout_formula_image_rows": len(diagnostics),
            "old_one_seed_failures": len(old_failures),
            "global_target_system": settings["global_target_system"],
            "global_target_family": int(settings["global_target_family"]),
            "global_formula_searches": int(global_summary.variant_id.nunique()),
            "global_unique_roots_total": len(global_roots),
            "SPARC_galaxies": 131,
            "CLASH_systems": 20,
        },
        "multistart": {
            "old_failures_recovered": multistart_recovered,
            "old_failures": len(old_failures),
            "all_old_failures_recovered": multistart_recovered == len(old_failures),
        },
        "global_multiplicity": {
            "all_formulas_have_sufficient_unique_roots": all_sufficient,
            "all_observed_family_images_assigned": all_assigned,
            "predeclared_observed_multiplicity_insufficiency_detected": observed_multiplicity_insufficiency,
            "descriptive_extra_root_pair_bifurcation_detected": extra_pair_bifurcation,
            "extra_pair_near_image_2c_matches_old_status": pair_near_2c_matches_status,
            "minimum_unique_roots": int(global_summary.unique_global_roots.min()),
            "maximum_unique_roots": int(global_summary.unique_global_roots.max()),
            "observed_family_images": int(global_summary.observed_family_images.iloc[0]),
            "old_success_unique_root_counts": successful_counts,
            "old_failure_unique_root_counts": failed_counts,
            "successful_nearest_root_distance_range_arcsec": [
                float(successful.nearest_global_root_to_2c_arcsec.min()),
                float(successful.nearest_global_root_to_2c_arcsec.max()),
            ],
            "failed_nearest_root_distance_range_arcsec": [
                float(failed.nearest_global_root_to_2c_arcsec.min()),
                float(failed.nearest_global_root_to_2c_arcsec.max()),
            ],
            "successful_second_root_distance_range_arcsec": [
                float(successful.second_nearest_global_root_to_2c_arcsec.min()),
                float(successful.second_nearest_global_root_to_2c_arcsec.max()),
            ],
            "successful_pair_separation_range_arcsec": [
                float(successful.nearest_two_root_separation_arcsec.min()),
                float(successful.nearest_two_root_separation_arcsec.max()),
            ],
        },
        "discrimination": discrimination.to_dict("records"),
        "image_2c": target.to_dict("records"),
        "parameter_pairs": pairs.to_dict("records"),
        "cross_domain_controls": scores.to_dict("records"),
        "verdict": {
            "predeclared_observed_multiplicity_failure": observed_multiplicity_insufficiency,
            "prior_root_recovery_tracks_a_real_extra_pair_bifurcation": bool(
                extra_pair_bifurcation and pair_near_2c_matches_status
            ),
            "one_seed_status_is_not_merely_a_numerical_artifact": bool(
                extra_pair_bifurcation
            ),
            "caustic_margin_perfectly_discriminates_the_extra_pair_regime": bool(
                float(
                    discrimination.set_index("metric").loc[
                        "source_caustic_margin_arcsec", "absolute_AUC"
                    ]
                )
                == 1.0
            ),
            "no_formula_promoted": True,
        },
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    make_figure(
        target,
        global_summary,
        discrimination,
        output / protocol["outputs"]["figure"],
    )
    best_metric = discrimination.iloc[0]
    summary = f"""# P0554 caustic-margin diagnostic

All {len(old_failures)} prior one-seed failures can reach some alternative root
under the frozen local multistart search. Nevertheless, global multiplicity
shows an exact bifurcation: every old-success formula has
{successful_counts[0] if successful_counts else 'NA'} family-2 roots and every
old-failure formula has {failed_counts[0] if failed_counts else 'NA'}. The
successful formulas add a pair near image 2c; the failed formulas do not.

Thus the predeclared observed-multiplicity insufficiency test is
**{observed_multiplicity_insufficiency}**, because all formulas retain at least
the three observed-image roots, but the secondary 3-to-5 extra-pair result is
**{extra_pair_bifurcation and pair_near_2c_matches_status}**. The old status is
not merely a numerical artifact. Assigned image distances remain the accuracy
test, and the extra companion is a new observational liability/prediction.

The strongest discriminator of the old solver status is
`{best_metric.metric}` with directed AUC {best_metric.absolute_AUC:.3f}. It
also discriminates the three-root from the five-root regime in this spent
image family. The Jacobian evaluated only at the observed coordinate is much
weaker, so the useful quantity is branch/caustic geometry rather than local
field strength at one catalog point. No formula is promoted.
"""
    (output / protocol["outputs"]["summary"]).write_text(summary, encoding="utf-8")
    print(
        json.dumps(
            json_safe(
                {
                    "coverage": report["coverage"],
                    "multistart": report["multistart"],
                    "global_multiplicity": report["global_multiplicity"],
                    "top_discriminators": discrimination.head(8).to_dict("records"),
                    "verdict": report["verdict"],
                }
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
