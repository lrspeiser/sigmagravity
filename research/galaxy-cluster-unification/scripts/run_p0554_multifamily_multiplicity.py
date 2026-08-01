#!/usr/bin/env python3
"""Count missing and surplus image roots across every raw-cluster source family."""

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
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_adaptive_route_multicluster_raw import route_fraction  # noqa: E402
from run_adaptive_route_raw_rxj2129 import baryon_field, build_route_field  # noqa: E402
from run_arc_invariant_absolute_lensing import raw_field  # noqa: E402
from run_clash_stellar_morphology_response import MorphologyLens  # noqa: E402
from run_p0554_caustic_margin import geometry_for, roots_from_seeds, source_for  # noqa: E402
from run_p0554_local_cross_domain_sensitivity import A0, json_safe, raw_contexts, sha256  # noqa: E402
from run_p0554_route_softness_interaction import build_variants, load_route_sources  # noqa: E402


def search_seeds(images, settings):
    half = float(settings["global_search_half_width_arcsec"])
    spacing = float(settings["global_grid_spacing_arcsec"])
    offset = float(settings["interlaced_grid_offset_arcsec"])
    axis = np.arange(-half, half + 0.5 * spacing, spacing)
    shifted = np.arange(-half + offset, half - offset + 0.5 * spacing, spacing)
    grids = []
    for values in (axis, shifted):
        gx, gy = np.meshgrid(values, values, indexing="xy")
        grids.append(np.column_stack([gx.ravel(), gy.ravel()]))
    angles = np.linspace(0.0, 2.0 * np.pi, int(settings["local_seed_angles"]), endpoint=False)
    for row in images.itertuples(index=False):
        observed = np.asarray([row.x_arcsec, row.y_arcsec], dtype=float)
        local = [observed]
        for radius in settings["local_seed_radii_arcsec"]:
            if float(radius) == 0.0:
                continue
            local.extend(
                observed + float(radius) * np.asarray([np.cos(angle), np.sin(angle)])
                for angle in angles
            )
        grids.append(np.asarray(local, dtype=float))
    return np.vstack(grids)


def classify_family(
    lens,
    variant,
    parameters,
    source,
    images,
    settings,
    system_label,
):
    redshift = float(images.source_redshift.median())
    roots = roots_from_seeds(
        lens,
        variant.variant_id,
        parameters,
        source,
        redshift,
        search_seeds(images, settings),
        closure_tolerance=float(settings["root_closure_tolerance_arcsec"]),
        deduplication_tolerance=float(settings["root_deduplication_arcsec"]),
        aperture=float(settings["global_root_aperture_arcsec"]),
    )
    root_xy = np.asarray([[item["x"], item["y"]] for item in roots], dtype=float)
    observed_xy = images[["x_arcsec", "y_arcsec"]].to_numpy(float)
    determinants = np.asarray([], dtype=float)
    if len(roots):
        jacobians = lens.jacobian(
            variant.variant_id,
            parameters,
            root_xy[:, 0],
            root_xy[:, 1],
            redshift,
            step=float(settings["jacobian_step_arcsec"]),
        )
        determinants = np.linalg.det(jacobians)
    magnifications = np.divide(
        1.0,
        determinants,
        out=np.full_like(determinants, np.nan),
        where=determinants != 0.0,
    )
    assigned_observed, assigned_roots = np.asarray([], int), np.asarray([], int)
    if len(roots):
        cost = np.linalg.norm(observed_xy[:, None, :] - root_xy[None, :, :], axis=2)
        assigned_observed, assigned_roots = linear_sum_assignment(cost)
    assignment_map = {
        int(observed_index): int(root_index)
        for observed_index, root_index in zip(assigned_observed, assigned_roots)
    }
    root_assignment_map = {
        int(root_index): int(observed_index)
        for observed_index, root_index in zip(assigned_observed, assigned_roots)
    }
    assignment_rows = []
    distances = []
    for observed_index, image in enumerate(images.itertuples(index=False)):
        root_index = assignment_map.get(observed_index)
        if root_index is None:
            distance = np.nan
            root_x = root_y = signed_mu = np.nan
        else:
            distance = float(np.linalg.norm(observed_xy[observed_index] - root_xy[root_index]))
            root_x, root_y = root_xy[root_index]
            signed_mu = magnifications[root_index]
            distances.append(distance)
        assignment_rows.append(
            {
                "system_label": system_label,
                "variant_id": variant.variant_id,
                "source_family": int(image.source_family),
                "image_id": str(image.image_id),
                "observed_x_arcsec": float(image.x_arcsec),
                "observed_y_arcsec": float(image.y_arcsec),
                "assigned": root_index is not None,
                "assigned_root_index": root_index,
                "assigned_root_x_arcsec": root_x,
                "assigned_root_y_arcsec": root_y,
                "assigned_distance_arcsec": distance,
                "assigned_signed_magnification": signed_mu,
            }
        )
    assigned_abs_mu = np.abs(magnifications[assigned_roots]) if len(assigned_roots) else np.asarray([])
    faintest_assigned = float(np.nanmin(assigned_abs_mu)) if len(assigned_abs_mu) else np.nan
    relative_threshold = float(settings.get("potentially_observable_surplus_fraction", 0.25))
    threshold = relative_threshold * faintest_assigned if np.isfinite(faintest_assigned) else np.nan
    unassigned = sorted(set(range(len(roots))) - set(int(value) for value in assigned_roots))
    observable_surplus = [
        index
        for index in unassigned
        if np.isfinite(threshold) and abs(magnifications[index]) >= threshold
    ]
    root_rows = []
    for index, item in enumerate(roots):
        observed_index = root_assignment_map.get(index)
        root_rows.append(
            {
                "system_label": system_label,
                "variant_id": variant.variant_id,
                "source_family": int(images.source_family.iloc[0]),
                "source_redshift": redshift,
                "root_index": index,
                "root_x_arcsec": item["x"],
                "root_y_arcsec": item["y"],
                "root_radius_arcsec": float(np.hypot(item["x"], item["y"])),
                "closure_arcsec": item["closure"],
                "signed_jacobian_determinant": float(determinants[index]),
                "signed_magnification": float(magnifications[index]),
                "parity": "positive" if determinants[index] > 0 else "negative",
                "assigned": observed_index is not None,
                "assigned_image_id": None
                if observed_index is None
                else str(images.iloc[observed_index].image_id),
                "relative_to_faintest_assigned_abs_magnification": float(
                    abs(magnifications[index]) / faintest_assigned
                )
                if np.isfinite(faintest_assigned) and faintest_assigned > 0
                else np.nan,
                "potentially_observable_surplus": index in observable_surplus,
            }
        )
    observed_count = len(images)
    root_count = len(roots)
    if root_count < observed_count:
        classification = "missing_multiplicity"
    elif root_count == observed_count:
        classification = "exact_multiplicity"
    elif observable_surplus:
        classification = "potentially_observable_surplus"
    else:
        classification = "demagnified_only_surplus"
    summary = {
        "system_label": system_label,
        "variant_id": variant.variant_id,
        "source_family": int(images.source_family.iloc[0]),
        "source_redshift": redshift,
        "observed_images": observed_count,
        "global_roots": root_count,
        "surplus_roots": max(0, root_count - observed_count),
        "missing_roots": max(0, observed_count - root_count),
        "unassigned_roots": len(unassigned),
        "potentially_observable_surplus_roots": len(observable_surplus),
        "faintest_assigned_abs_magnification": faintest_assigned,
        "observable_surplus_abs_magnification_threshold": threshold,
        "all_observed_images_assigned": len(assignment_map) == observed_count,
        "assignment_RMS_arcsec": float(np.sqrt(np.mean(np.square(distances))))
        if len(distances) == observed_count
        else np.nan,
        "assignment_max_arcsec": float(np.max(distances)) if distances else np.nan,
        "multiplicity_classification": classification,
    }
    return root_rows, assignment_rows, summary


def make_figure(family_summary, variant_summary, output):
    variants = variant_summary.variant_id.tolist()
    classes = [
        "missing_multiplicity",
        "exact_multiplicity",
        "demagnified_only_surplus",
        "potentially_observable_surplus",
    ]
    colors = ["crimson", "tab:green", "goldenrod", "tab:purple"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    bottom = np.zeros(len(variants))
    indexed = variant_summary.set_index("variant_id").loc[variants]
    for classification, color in zip(classes, colors):
        values = indexed[f"families_{classification}"].to_numpy(float)
        axes[0].bar(variants, values, bottom=bottom, label=classification.replace("_", " "), color=color)
        bottom += values
    axes[0].set(ylabel="source families", title="Global image-multiplicity classifications")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend(fontsize=8)

    pivot = family_summary.pivot_table(
        index=["system_label", "source_family"],
        columns="variant_id",
        values="potentially_observable_surplus_roots",
        aggfunc="first",
    ).loc[:, variants]
    image = axes[1].imshow(pivot.to_numpy(float), aspect="auto", cmap="magma", vmin=0)
    axes[1].set(
        xticks=np.arange(len(variants)),
        xticklabels=variants,
        yticks=np.arange(len(pivot)),
        yticklabels=[f"{system} F{family}" for system, family in pivot.index],
        title="Potentially observable unassigned roots",
    )
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].tick_params(axis="y", labelsize=7)
    fig.colorbar(image, ax=axes[1], label="surplus roots")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def markdown_table(frame):
    display = frame.copy()
    for column in display.select_dtypes(include="number").columns:
        display[column] = display[column].map(
            lambda value: f"{value:.3f}" if isinstance(value, float) else str(value)
        )
    header = "| " + " | ".join(display.columns) + " |"
    separator = "|" + "|".join("---" for _ in display.columns) + "|"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def write_summary(protocol, output, family_summary, variant_summary):
    table = markdown_table(
        variant_summary[
            [
                "variant_id",
                "families_missing_multiplicity",
                "families_exact_multiplicity",
                "families_demagnified_only_surplus",
                "families_potentially_observable_surplus",
                "potentially_observable_surplus_roots",
                "equal_family_assignment_RMS_arcsec",
            ]
        ]
    )
    summary = f"""# P0554 multifamily multiplicity audit

The frozen audit searched {len(family_summary)} formula-family combinations
across 27 source families and five raw clusters. Root multiplicity is separated
from detectability: surplus roots below one quarter of the faintest assigned
image magnification are tracked separately from potentially observable surplus.

{table}

No formula is promoted. Published catalogs can omit images and the relative
magnification threshold is only a screening proxy.
"""
    (output / protocol["outputs"]["summary"]).write_text(summary, encoding="utf-8")


def threshold_sensitivity(roots):
    rows = []
    for threshold in (0.10, 0.25, 0.50, 1.00):
        for (variant_id, system_label, family), block in roots.groupby(
            ["variant_id", "system_label", "source_family"]
        ):
            count = int(
                (
                    ~block.assigned.astype(bool)
                    & block.relative_to_faintest_assigned_abs_magnification.ge(threshold)
                ).sum()
            )
            rows.append(
                {
                    "relative_magnification_threshold": threshold,
                    "variant_id": variant_id,
                    "system_label": system_label,
                    "source_family": int(family),
                    "potentially_observable_surplus_roots": count,
                }
            )
    detail = pd.DataFrame(rows)
    return (
        detail.groupby(["relative_magnification_threshold", "variant_id"], as_index=False)
        .agg(
            families_with_potentially_observable_surplus=(
                "potentially_observable_surplus_roots",
                lambda values: int((values > 0).sum()),
            ),
            potentially_observable_surplus_roots=(
                "potentially_observable_surplus_roots",
                "sum",
            ),
        )
    )


def main():
    config_path = ROOT / "configs" / "p0554_multifamily_multiplicity_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if not protocol["status"].startswith("frozen_"):
        raise RuntimeError("multifamily multiplicity protocol is not frozen")
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    if "--postprocess-only" in sys.argv[1:]:
        roots = pd.read_csv(output / protocol["outputs"]["global_roots"])
        family_summary = pd.read_csv(output / protocol["outputs"]["family_summary"])
        variant_summary = pd.read_csv(output / protocol["outputs"]["variant_summary"])
        report = json.loads((output / protocol["outputs"]["report"]).read_text(encoding="utf-8"))
        sensitivity = threshold_sensitivity(roots)
        sensitivity.to_csv(output / "threshold_sensitivity.csv", index=False)
        report["report_version"] = "P0554-MULTIFAMILY-MULTIPLICITY-RESULTS-0.2.0"
        report["descriptive_post_primary_threshold_sensitivity"] = sensitivity.to_dict("records")
        report["verdict"]["route_has_more_surplus_roots_than_baseline_at_every_checked_threshold"] = bool(
            all(
                block.set_index("variant_id").loc[
                    "route_parent", "potentially_observable_surplus_roots"
                ]
                > block.set_index("variant_id").loc[
                    "baseline", "potentially_observable_surplus_roots"
                ]
                for _, block in sensitivity.groupby("relative_magnification_threshold")
            )
        )
        (output / protocol["outputs"]["report"]).write_text(
            json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
        )
        write_summary(protocol, output, family_summary, variant_summary)
        print(
            json.dumps(
                json_safe(
                    {
                        "coverage": report["coverage"],
                        "variant_summary": report["variant_summary"],
                        "verdict": report["verdict"],
                    }
                ),
                indent=2,
            ),
            flush=True,
        )
        return
    interaction = json.loads(
        (ROOT / protocol["inputs"]["interaction_protocol"]).read_text(encoding="utf-8")
    )
    all_variants = {variant.variant_id: variant for variant in build_variants(interaction)}
    variants = [all_variants[variant_id] for variant_id in protocol["variants"]]
    contexts = raw_contexts(interaction)
    sources, route_protocols = load_route_sources(interaction, contexts)
    geometry = pd.read_csv(ROOT / protocol["inputs"]["interaction_geometry"])
    archived = pd.read_csv(ROOT / protocol["inputs"]["interaction_predictions"])
    settings = protocol["evaluation"]
    root_rows, assignment_rows, summary_rows = [], [], []
    for context in contexts:
        images = pd.concat([context.training, context.heldout], ignore_index=True)
        baryons = baryon_field(context.anchors, context.local)
        radial_cache, route_cache = {}, {}
        print(f"{context.label}: {images.source_family.nunique()} families", flush=True)
        for variant in variants:
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
                {variant.variant_id: radial},
                parent=variant.variant_id,
                morphology=angular,
                fraction=angular_strength,
            )
            parameters = geometry_for(geometry, context.label, variant.variant_id)
            print(f"  {variant.variant_id}", flush=True)
            for family, family_images in images.groupby("source_family", sort=True):
                source = source_for(
                    archived,
                    context.label,
                    variant.variant_id,
                    int(family),
                )
                roots, assignments, summary = classify_family(
                    lens,
                    variant,
                    parameters,
                    source,
                    family_images,
                    settings,
                    context.label,
                )
                root_rows.extend(roots)
                assignment_rows.extend(assignments)
                summary_rows.append(summary)
    roots = pd.DataFrame(root_rows)
    assignments = pd.DataFrame(assignment_rows)
    family_summary = pd.DataFrame(summary_rows)
    roots.to_csv(output / protocol["outputs"]["global_roots"], index=False)
    assignments.to_csv(output / protocol["outputs"]["assignments"], index=False)
    family_summary.to_csv(output / protocol["outputs"]["family_summary"], index=False)
    sensitivity = threshold_sensitivity(roots)
    sensitivity.to_csv(output / "threshold_sensitivity.csv", index=False)

    rows = []
    classes = [
        "missing_multiplicity",
        "exact_multiplicity",
        "demagnified_only_surplus",
        "potentially_observable_surplus",
    ]
    for variant_id, block in family_summary.groupby("variant_id"):
        row = {
            "variant_id": variant_id,
            "families": len(block),
            "observed_images": int(block.observed_images.sum()),
            "global_roots": int(block.global_roots.sum()),
            "surplus_roots": int(block.surplus_roots.sum()),
            "missing_roots": int(block.missing_roots.sum()),
            "potentially_observable_surplus_roots": int(
                block.potentially_observable_surplus_roots.sum()
            ),
            "families_all_observed_assigned": int(block.all_observed_images_assigned.astype(bool).sum()),
            "equal_family_assignment_RMS_arcsec": float(block.assignment_RMS_arcsec.mean()),
        }
        for classification in classes:
            row[f"families_{classification}"] = int(
                block.multiplicity_classification.eq(classification).sum()
            )
        rows.append(row)
    order = {variant_id: index for index, variant_id in enumerate(protocol["variants"])}
    variant_summary = pd.DataFrame(rows).sort_values(
        "variant_id", key=lambda values: values.map(order)
    )
    variant_summary.to_csv(output / protocol["outputs"]["variant_summary"], index=False)
    make_figure(family_summary, variant_summary, output / protocol["outputs"]["figure"])

    baseline = family_summary[family_summary.variant_id.eq("baseline")][
        ["system_label", "source_family", "multiplicity_classification", "potentially_observable_surplus_roots"]
    ].rename(
        columns={
            "multiplicity_classification": "baseline_classification",
            "potentially_observable_surplus_roots": "baseline_observable_surplus_roots",
        }
    )
    changes = family_summary.merge(
        baseline, on=["system_label", "source_family"], validate="many_to_one"
    )
    changes = changes[~changes.variant_id.eq("baseline")]
    changes["classification_changed"] = (
        changes.multiplicity_classification != changes.baseline_classification
    )
    changes["observable_surplus_delta"] = (
        changes.potentially_observable_surplus_roots
        - changes.baseline_observable_surplus_roots
    )
    scores = pd.read_csv(ROOT / protocol["inputs"]["interaction_scores"])
    outside_macs1931_family2 = family_summary[
        ~(
            family_summary.system_label.eq("MACS1931")
            & family_summary.source_family.eq(2)
        )
    ]
    report = {
        "report_version": "P0554-MULTIFAMILY-MULTIPLICITY-RESULTS-0.2.0",
        "status": "complete",
        "protocol": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": sha256(config_path),
        },
        "coverage": {
            "variants": len(variants),
            "systems": len(contexts),
            "source_families": int(family_summary[["system_label", "source_family"]].drop_duplicates().shape[0]),
            "formula_family_searches": len(family_summary),
            "published_images": int(
                family_summary[family_summary.variant_id.eq("baseline")].observed_images.sum()
            ),
            "accepted_global_roots": len(roots),
        },
        "variant_summary": variant_summary.to_dict("records"),
        "changes_from_baseline": (
            changes.groupby("variant_id", as_index=False)
            .agg(
                families_changed_classification=("classification_changed", "sum"),
                observable_surplus_root_delta=("observable_surplus_delta", "sum"),
                families_with_more_observable_surplus=("observable_surplus_delta", lambda values: int((values > 0).sum())),
                families_with_less_observable_surplus=("observable_surplus_delta", lambda values: int((values < 0).sum())),
            )
            .to_dict("records")
        ),
        "descriptive_post_primary_threshold_sensitivity": sensitivity.to_dict("records"),
        "cross_domain_controls": scores[scores.variant_id.isin(protocol["variants"])].to_dict("records"),
        "verdict": {
            "potentially_observable_surplus_occurs_outside_MACS1931_family2": bool(
                outside_macs1931_family2.potentially_observable_surplus_roots.gt(0).any()
            ),
            "any_variant_has_exact_multiplicity_for_every_family": bool(
                variant_summary.families_exact_multiplicity.eq(27).any()
            ),
            "observable_surplus_is_a_recurring_issue": bool(
                variant_summary.potentially_observable_surplus_roots.max() > 2
            ),
            "route_has_more_surplus_roots_than_baseline_at_every_checked_threshold": bool(
                all(
                    block.set_index("variant_id").loc[
                        "route_parent", "potentially_observable_surplus_roots"
                    ]
                    > block.set_index("variant_id").loc[
                        "baseline", "potentially_observable_surplus_roots"
                    ]
                    for _, block in sensitivity.groupby("relative_magnification_threshold")
                )
            ),
            "no_formula_promoted": True,
        },
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    write_summary(protocol, output, family_summary, variant_summary)
    print(json.dumps(json_safe({"coverage": report["coverage"], "variant_summary": report["variant_summary"], "verdict": report["verdict"]}), indent=2), flush=True)


if __name__ == "__main__":
    main()
