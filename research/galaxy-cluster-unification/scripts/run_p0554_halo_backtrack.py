#!/usr/bin/env python3
"""Backtrack published cluster-halo posterior locations to baryonic sources."""

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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0554_all_baryon_route_screen import (  # noqa: E402
    prepare_hst_map,
    prepare_xray_maps,
)
from run_p0554_local_cross_domain_sensitivity import (  # noqa: E402
    json_safe,
    raw_contexts,
)
from run_p0554_route_softness_interaction import load_route_sources  # noqa: E402
from voidscreen.gravity_flow_inverse import (  # noqa: E402
    off_plane_arc_length,
    rasterize_transport_paths,
    solve_transport,
    source_route_table,
    transport_diagnostics,
)
from voidscreen.halo_backtrack import (  # noqa: E402
    coarsen_source_map,
    component_samples,
    halo_assignment,
    posterior_component_destinations,
    thin_bayes_chain,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def converged_transport(protocol, source_positions, source_weights, destination_positions, destination_weights, entropy_length_kpc, *, control=False):
    """Retry difficult Sinkhorn cases until the saved marginal gate is met."""
    settings = protocol["inverse_transport"]
    gate = float(
        protocol["controls"]["maximum_saved_marginal_error"]
        if control
        else settings["maximum_saved_marginal_error"]
    )
    attempts = [int(settings["iterations"])] + [
        int(value) for value in settings["adaptive_retry_iterations"]
    ]
    best = None
    best_error = np.inf
    for iterations in attempts:
        plan, cost = solve_transport(
            source_positions,
            source_weights,
            destination_positions,
            destination_weights,
            entropy_length_kpc=float(entropy_length_kpc),
            iterations=iterations,
            tolerance=float(settings["tolerance"]),
        )
        source = source_weights / np.sum(source_weights)
        target = destination_weights / np.sum(destination_weights)
        error = max(
            float(np.max(np.abs(np.sum(plan, axis=1) - source))),
            float(np.max(np.abs(np.sum(plan, axis=0) - target))),
        )
        if error < best_error:
            best = (plan, cost, iterations, error)
            best_error = error
        if error <= gate:
            return plan, cost, iterations, error
    raise RuntimeError(
        f"transport marginal error {best_error:.6g} exceeds gate {gate:.6g} "
        f"after {best[2]} iterations"
    )


def component_summary_rows(label, components, chain_rows, retained_rows, aperture_kpc):
    rows = []
    for object_id, values in components.items():
        row = {
            "system_label": label,
            "halo_id": int(object_id),
            "chain_rows": int(chain_rows),
            "retained_posterior_rows": int(retained_rows),
            "posterior_center_fraction_inside_target_aperture": float(
                np.mean(
                    np.hypot(values["x_kpc"], values["y_kpc"])
                    <= float(aperture_kpc)
                )
            ),
            "maximum_retained_posterior_center_radius_kpc": float(
                np.max(np.hypot(values["x_kpc"], values["y_kpc"]))
            ),
        }
        for key in ("x_kpc", "y_kpc", "core_kpc", "sigma_km_s", "emass", "theta_deg"):
            array = np.asarray(values[key], dtype=float)
            row[f"{key}_q16"] = float(np.quantile(array, 0.16))
            row[f"{key}_median"] = float(np.median(array))
            row[f"{key}_q84"] = float(np.quantile(array, 0.84))
        rows.append(row)
    return rows


def source_variants(protocol, all_baryon, acquisition, reused, context, members, map_axis):
    scale = float(
        context.local["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"]
    )
    known_images = pd.concat([context.training, context.heldout], ignore_index=True)
    star, _ = prepare_hst_map(
        all_baryon, acquisition, reused, context, known_images, map_axis
    )
    _, gas, _ = prepare_xray_maps(all_baryon, acquisition, context, map_axis)
    settings = protocol["baryonic_sources"]
    kwargs = {
        "factor": int(settings["continuous_map_block_factor"]),
        "maximum_radius_kpc": float(settings["continuous_map_maximum_radius_kpc"]),
        "retained_weight": float(settings["continuous_map_retained_weight"]),
    }
    star_position, star_weight = coarsen_source_map(star, map_axis, scale, **kwargs)
    gas_position, gas_weight = coarsen_source_map(gas, map_axis, scale, **kwargs)
    star_norm = star / np.sum(star)
    gas_norm = gas / np.sum(gas)
    equal_position, equal_weight = coarsen_source_map(
        0.5 * star_norm + 0.5 * gas_norm, map_axis, scale, **kwargs
    )
    member_position = members[["x_arcsec", "y_arcsec"]].to_numpy(float) * scale
    member_weight = members["base_weight"].to_numpy(float)
    member_weight /= np.sum(member_weight)
    return {
        "member_catalog_light": (member_position, member_weight),
        "continuous_f160w_light": (star_position, star_weight),
        "xray_emissivity_proxy": (gas_position, gas_weight),
        "equal_f160w_xray_morphology": (equal_position, equal_weight),
    }, star, gas


def target_variants(protocol, components, axis):
    settings = protocol["halo_target"]
    result = {}
    for variant in settings["target_variants"]:
        result[variant["target_kind"]] = posterior_component_destinations(
            components,
            axis,
            width_mode=variant["width_mode"],
            width_kpc=float(variant["width_kpc"]),
            weight_mode=variant["weight_mode"],
            maximum_radius_kpc=float(settings["maximum_radius_kpc"]),
            minimum_relative_density=float(settings["minimum_relative_component_density"]),
        )
    return result


def requested_solutions(protocol):
    primary_source = protocol["baryonic_sources"]["primary_source"]
    primary_target = protocol["halo_target"]["primary_target"]
    primary_entropy = float(protocol["inverse_transport"]["primary_entropy_length_kpc"])
    output = {
        (primary_source, row["target_kind"], primary_entropy)
        for row in protocol["halo_target"]["target_variants"]
    }
    output |= {
        (source, primary_target, primary_entropy)
        for source in protocol["baryonic_sources"]["source_variants"]
    }
    output |= {
        (primary_source, primary_target, float(entropy))
        for entropy in protocol["inverse_transport"]["entropy_sensitivity_kpc"]
    }
    return sorted(output)


def radial_shuffle_rows(
    protocol,
    label,
    system_index,
    source_positions,
    source_weights,
    destination_positions,
    destination_weights,
    baryonic_center,
):
    control = protocol["controls"]
    inverse = protocol["inverse_transport"]
    rng = np.random.default_rng(int(control["shuffle_seed"]) + 1009 * system_index)
    radius = np.linalg.norm(source_positions, axis=1)
    rows = []
    for shuffle_index in range(int(control["radial_angle_shuffles"])):
        angle = rng.uniform(0.0, 2.0 * np.pi, len(radius))
        shuffled = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
        plan, _, used_iterations, solver_error = converged_transport(
            protocol,
            shuffled,
            source_weights,
            destination_positions,
            destination_weights,
            float(inverse["primary_entropy_length_kpc"]),
            control=True,
        )
        stats = transport_diagnostics(
            plan,
            shuffled,
            source_weights,
            destination_positions,
            destination_weights,
            baryonic_center=baryonic_center,
        )
        rows.append(
            {
                "system_label": label,
                "shuffle_index": shuffle_index,
                "rms_transport_kpc": stats["rms_transport_kpc"],
                "source_marginal_max_error": stats["source_marginal_max_error"],
                "target_marginal_max_error": stats["target_marginal_max_error"],
                "solver_iterations": used_iterations,
                "solver_max_marginal_error": solver_error,
            }
        )
    return rows


def make_figure(contexts, products, output, target_limit):
    figure, axes = plt.subplots(len(contexts), 4, figsize=(14, 3.2 * len(contexts)))
    for row, context in enumerate(contexts):
        item = products[context.label]
        scale = item["scale"]
        map_extent = (-60.0 * scale, 60.0 * scale, -60.0 * scale, 60.0 * scale)
        target_extent = (-target_limit, target_limit, -target_limit, target_limit)
        panels = [
            (item["star"], map_extent, "F160W launch proxy", "magma"),
            (item["gas"], map_extent, "X-ray sensitivity proxy", "magma"),
            (item["target"], target_extent, "posterior halo arrivals", "inferno"),
            (item["paths"], target_extent, "least-distance paths", "viridis"),
        ]
        for column, (image, extent, title, cmap) in enumerate(panels):
            axis = axes[row, column]
            positive = image[image > 0.0]
            scale_value = (
                float(np.quantile(positive, 0.9)) if len(positive) else 1.0
            )
            axis.imshow(
                np.log1p(np.maximum(image, 0.0) / max(scale_value, np.finfo(float).tiny)),
                origin="lower",
                extent=extent,
                cmap=cmap,
                interpolation="nearest",
            )
            axis.scatter(
                item["member_positions"][:, 0],
                item["member_positions"][:, 1],
                s=7.0,
                facecolors="none",
                edgecolors="white",
                linewidths=0.45,
                alpha=0.65,
            )
            axis.scatter(
                item["halo_centers"][:, 0],
                item["halo_centers"][:, 1],
                marker="x",
                s=45,
                color="cyan",
                linewidths=1.2,
            )
            axis.set(
                xlim=(-0.98 * target_limit, 0.98 * target_limit),
                ylim=(-0.98 * target_limit, 0.98 * target_limit),
                aspect="equal",
            )
            if row == 0:
                axis.set_title(title, fontsize=10)
            if column == 0:
                axis.set_ylabel(f"{context.label}\ny (kpc)")
            else:
                axis.set_yticklabels([])
            if row == len(contexts) - 1:
                axis.set_xlabel("x (kpc)")
            else:
                axis.set_xticklabels([])
    figure.suptitle(
        "Backtracking baryonic launch points to published halo arrivals\n"
        "(minimum projected transport; cyan crosses are posterior halo centers)",
        y=0.999,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def summarize(protocol, config_path, output):
    routes = pd.read_csv(output / protocol["outputs"]["route_statistics"])
    nulls = pd.read_csv(output / protocol["outputs"]["radial_shuffle_nulls"])
    primary_source = protocol["baryonic_sources"]["primary_source"]
    primary_target = protocol["halo_target"]["primary_target"]
    primary_entropy = float(protocol["inverse_transport"]["primary_entropy_length_kpc"])
    primary = routes[
        routes.source_kind.eq(primary_source)
        & routes.target_kind.eq(primary_target)
        & routes.entropy_length_kpc.eq(primary_entropy)
    ].set_index("system_label")
    controls = []
    for label, row in primary.iterrows():
        values = nulls[nulls.system_label.eq(label)].rms_transport_kpc.to_numpy(float)
        real = float(row.rms_transport_kpc)
        controls.append(
            {
                "system_label": label,
                "real_rms_transport_kpc": real,
                "null_median_rms_transport_kpc": float(np.median(values)),
                "improvement_fraction_vs_null_median": 1.0 - real / float(np.median(values)),
                "one_sided_permutation_p": float((1 + np.sum(values <= real)) / (1 + len(values))),
            }
        )
    control_table = pd.DataFrame(controls)
    target_rows = routes[
        routes.source_kind.eq(primary_source)
        & routes.entropy_length_kpc.eq(primary_entropy)
    ]
    source_rows = routes[
        routes.target_kind.eq(primary_target)
        & routes.entropy_length_kpc.eq(primary_entropy)
    ]
    report = {
        "report_version": "P0554-HALO-BACKTRACK-RESULTS-0.1.0",
        "status": "complete",
        "protocol": {
            "path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(config_path),
        },
        "coverage": {
            "systems": int(primary.index.nunique()),
            "cluster_scale_halos": int(
                pd.read_csv(output / protocol["outputs"]["halo_posterior_summary"])[
                    ["system_label", "halo_id"]
                ].drop_duplicates().shape[0]
            ),
            "transport_solutions": int(len(routes)),
            "radial_angle_controls": int(len(nulls)),
            "maximum_source_marginal_error": float(routes.source_marginal_max_error.max()),
            "maximum_target_marginal_error": float(routes.target_marginal_max_error.max()),
            "minimum_posterior_center_fraction_inside_target_aperture": float(
                pd.read_csv(output / protocol["outputs"]["halo_posterior_summary"])
                .posterior_center_fraction_inside_target_aperture.min()
            ),
        },
        "primary_inverse": {
            "median_of_system_median_path_kpc": float(primary.median_path_kpc.median()),
            "median_of_system_p90_path_kpc": float(primary.p90_path_kpc.median()),
            "median_mean_cos_inward": float(primary.mean_cos_inward.median()),
            "median_fraction_ending_inward": float(primary.fraction_ending_inward.median()),
            "per_system": primary.reset_index().to_dict("records"),
        },
        "radial_angle_control": {
            "systems_real_shorter_than_null_median": int(
                (control_table.improvement_fraction_vs_null_median > 0.0).sum()
            ),
            "systems_permutation_p_le_0_05": int(
                (control_table.one_sided_permutation_p <= 0.05).sum()
            ),
            "median_improvement_fraction_vs_null": float(
                control_table.improvement_fraction_vs_null_median.median()
            ),
            "per_system": control_table.to_dict("records"),
        },
        "target_sensitivity": target_rows.groupby("target_kind")[
            "rms_transport_kpc"
        ].median().sort_values().to_dict(),
        "source_sensitivity": source_rows.groupby("source_kind")[
            "rms_transport_kpc"
        ].median().sort_values().to_dict(),
        "bottom_line": (
            "The inverse now makes the proposed backtracking concrete: every published "
            "cluster-scale halo arrival is assigned to observed baryonic launch points "
            "with posterior uncertainty and conservation of normalized flow. The paths "
            "remain a descriptive least-distance attribution, not observed gravity lines."
        ),
        "promotion": {
            "new_field_law_promoted": False,
            "reason": "The arrival targets were inferred from the same lens data under a standard Lenstool model; only a later baryon-only forward prediction on unused raw observables can test a law.",
        },
        "limits": protocol["limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# P0554 halo-arrival backtracking",
        "",
        "## Outcome",
        "",
        report["bottom_line"],
        "",
        f"The primary inverse has a median cluster-level path of **{report['primary_inverse']['median_of_system_median_path_kpc']:.1f} kpc** and median cluster-level 90th percentile of **{report['primary_inverse']['median_of_system_p90_path_kpc']:.1f} kpc**.",
        f"The actual galaxy angles produce a shorter route than the radius-preserving null median in **{report['radial_angle_control']['systems_real_shorter_than_null_median']}/5** systems; **{report['radial_angle_control']['systems_permutation_p_le_0_05']}/5** reach one-sided p <= 0.05.",
        "",
        "## Meaning",
        "",
        "The tables answer which baryonic galaxy is the cheapest origin for each modeled halo component and how long the projected reassignment must be. They do not show that gravity traveled along those paths, determine an off-plane arc height, or determine the missing-gravity amplitude.",
        "",
        "The next defensible step is to compress any repeatable source-to-arrival pattern into a formula that uses baryons alone, freeze it, and predict unused raw multiple-image positions without reading their halo model.",
    ]
    (output / protocol["outputs"]["summary"]).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return report


def main():
    config_path = ROOT / "configs/p0554_halo_backtrack_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if not protocol["status"].startswith("frozen_"):
        raise RuntimeError("halo-backtrack protocol is not frozen")
    interaction = json.loads(
        (ROOT / protocol["inputs"]["interaction_protocol"]).read_text(encoding="utf-8")
    )
    all_baryon = json.loads(
        (ROOT / protocol["inputs"]["all_baryon_screen_protocol"]).read_text(encoding="utf-8")
    )
    acquisition = json.loads(
        (ROOT / all_baryon["inputs"]["acquisition_protocol"]).read_text(encoding="utf-8")
    )
    reused = json.loads(
        (ROOT / all_baryon["inputs"]["reused_hst_protocol"]).read_text(encoding="utf-8")
    )
    contexts = raw_contexts(interaction)
    members_by_label, _ = load_route_sources(interaction, contexts)
    system_config = {row["label"]: row for row in protocol["systems"]}
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    map_settings = all_baryon["map_construction"]
    map_axis = np.arange(
        float(map_settings["axis_min_arcsec"]),
        float(map_settings["axis_max_arcsec"])
        + 0.5 * float(map_settings["grid_spacing_arcsec"]),
        float(map_settings["grid_spacing_arcsec"]),
    )
    halo_settings = protocol["halo_target"]
    target_axis = np.arange(
        float(halo_settings["grid_min_kpc"]),
        float(halo_settings["grid_max_kpc"])
        + 0.5 * float(halo_settings["grid_spacing_kpc"]),
        float(halo_settings["grid_spacing_kpc"]),
    )
    primary_source = protocol["baryonic_sources"]["primary_source"]
    primary_target = protocol["halo_target"]["primary_target"]
    primary_entropy = float(protocol["inverse_transport"]["primary_entropy_length_kpc"])
    inverse = protocol["inverse_transport"]
    chain_root = ROOT / protocol["inputs"]["chain_root"]

    halo_rows = []
    route_rows = []
    member_rows = []
    inflow_rows = []
    null_rows = []
    length_rows = []
    path_maps = {}
    products = {}

    for system_index, context in enumerate(contexts):
        label = context.label
        print(f"{label}: loading baryons and halo posterior", flush=True)
        scale = float(
            context.local["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"]
        )
        sources, star, gas = source_variants(
            protocol,
            all_baryon,
            acquisition,
            reused,
            context,
            members_by_label[label],
            map_axis,
        )
        chain = chain_root / system_config[label]["chain_directory"] / "bayes.dat"
        headers, samples, chain_rows = thin_bayes_chain(
            chain, int(halo_settings["posterior_samples_per_system"])
        )
        components = component_samples(headers, samples, scale)
        halo_rows.extend(
            component_summary_rows(
                label,
                components,
                chain_rows,
                len(samples),
                float(halo_settings["maximum_radius_kpc"]),
            )
        )
        targets = target_variants(protocol, components, target_axis)
        baryonic_center = np.sum(
            sources[primary_source][0] * sources[primary_source][1][:, None], axis=0
        )
        primary_plan = None
        primary_destination = None
        primary_component_ids = None
        primary_target_image = None

        for source_kind, target_kind, entropy_length in requested_solutions(protocol):
            print(f"  {source_kind} -> {target_kind}, entropy={entropy_length:g}", flush=True)
            source_positions, source_weights = sources[source_kind]
            destination_positions, destination_weights, component_ids, maps = targets[target_kind]
            plan, _, used_iterations, solver_error = converged_transport(
                protocol,
                source_positions,
                source_weights,
                destination_positions,
                destination_weights,
                entropy_length,
            )
            stats = transport_diagnostics(
                plan,
                source_positions,
                source_weights,
                destination_positions,
                destination_weights,
                baryonic_center=baryonic_center,
            )
            route_rows.append(
                {
                    "system_label": label,
                    "source_kind": source_kind,
                    "target_kind": target_kind,
                    "entropy_length_kpc": entropy_length,
                    "source_count": len(source_positions),
                    "destination_count": len(destination_positions),
                    "halo_count": len(components),
                    "solver_iterations": used_iterations,
                    "solver_max_marginal_error": solver_error,
                    **stats,
                }
            )
            is_primary = (
                source_kind == primary_source
                and target_kind == primary_target
                and entropy_length == primary_entropy
            )
            if is_primary:
                primary_plan = plan
                primary_destination = (destination_positions, destination_weights)
                primary_component_ids = component_ids
                primary_target_image = sum(maps.values())
                path_map = rasterize_transport_paths(
                    plan,
                    source_positions,
                    destination_positions,
                    target_axis,
                    samples_per_path=int(inverse["path_samples"]),
                    retained_weight=float(inverse["path_map_retained_transport_weight"]),
                )
                path_maps[label] = path_map
                route_table = source_route_table(
                    plan,
                    source_positions,
                    source_weights,
                    destination_positions,
                    baryonic_center=baryonic_center,
                )
                member_table = members_by_label[label].reset_index(drop=True)
                for member, route in zip(member_table.itertuples(index=False), route_table, strict=True):
                    member_rows.append(
                        {
                            "system_label": label,
                            "source_id": member.source_id,
                            **route,
                        }
                    )
                conditional, component_marginal = halo_assignment(plan, component_ids)
                object_ids = np.unique(component_ids)
                source_mass = np.sum(plan, axis=1)
                flows = conditional * source_mass[:, None]
                for halo_column, object_id in enumerate(object_ids):
                    order = np.argsort(flows[:, halo_column])[::-1]
                    for rank, source_index in enumerate(order, start=1):
                        inflow_rows.append(
                            {
                                "system_label": label,
                                "halo_id": int(object_id),
                                "origin_rank": rank,
                                "source_index": int(source_index),
                                "source_id": member_table.iloc[source_index].source_id,
                                "source_x_kpc": float(source_positions[source_index, 0]),
                                "source_y_kpc": float(source_positions[source_index, 1]),
                                "source_weight": float(source_weights[source_index]),
                                "flow_weight": float(flows[source_index, halo_column]),
                                "fraction_of_source_sent_to_halo": float(
                                    conditional[source_index, halo_column]
                                ),
                                "fraction_of_halo_inflow_from_source": float(
                                    flows[source_index, halo_column]
                                    / max(component_marginal[halo_column], np.finfo(float).tiny)
                                ),
                            }
                        )
                displacement = destination_positions[None, :, :] - source_positions[:, None, :]
                projected_distance = np.linalg.norm(displacement, axis=2)
                for ratio in inverse["off_plane_height_ratios_for_length_only"]:
                    arc_length = off_plane_arc_length(projected_distance, float(ratio))
                    length_rows.append(
                        {
                            "system_label": label,
                            "height_to_projected_distance_ratio": float(ratio),
                            "mean_arc_length_kpc": float(np.sum(plan * arc_length)),
                            "rms_arc_length_kpc": float(
                                np.sqrt(np.sum(plan * np.square(arc_length)))
                            ),
                        }
                    )

        if primary_plan is None or primary_destination is None or primary_component_ids is None:
            raise RuntimeError(f"primary inverse missing for {label}")
        null_rows.extend(
            radial_shuffle_rows(
                protocol,
                label,
                system_index,
                sources[primary_source][0],
                sources[primary_source][1],
                primary_destination[0],
                primary_destination[1],
                baryonic_center,
            )
        )
        halo_centers = np.array(
            [
                [np.median(values["x_kpc"]), np.median(values["y_kpc"])]
                for values in components.values()
            ]
        )
        products[label] = {
            "scale": scale,
            "star": star,
            "gas": gas,
            "target": primary_target_image / np.sum(primary_target_image),
            "paths": path_maps[label],
            "member_positions": sources[primary_source][0],
            "halo_centers": halo_centers,
        }

    pd.DataFrame(halo_rows).to_csv(
        output / protocol["outputs"]["halo_posterior_summary"], index=False
    )
    pd.DataFrame(route_rows).to_csv(
        output / protocol["outputs"]["route_statistics"], index=False
    )
    pd.DataFrame(member_rows).to_csv(
        output / protocol["outputs"]["member_routes"], index=False
    )
    pd.DataFrame(inflow_rows).to_csv(
        output / protocol["outputs"]["halo_inflows"], index=False
    )
    pd.DataFrame(null_rows).to_csv(
        output / protocol["outputs"]["radial_shuffle_nulls"], index=False
    )
    pd.DataFrame(length_rows).to_csv(
        output / protocol["outputs"]["off_plane_lengths"], index=False
    )
    np.savez_compressed(output / protocol["outputs"]["path_maps"], **path_maps)
    make_figure(
        contexts,
        products,
        output / protocol["outputs"]["figure"],
        float(halo_settings["maximum_radius_kpc"]),
    )
    report = summarize(protocol, config_path, output)
    print(json.dumps(json_safe(report), indent=2), flush=True)


if __name__ == "__main__":
    main()
