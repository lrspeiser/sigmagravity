#!/usr/bin/env python3
"""Relax the all-sources-must-route assumption in the P0554 halo inverse."""

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

from run_p0554_local_cross_domain_sensitivity import json_safe, raw_contexts  # noqa: E402
from run_p0554_route_softness_interaction import load_route_sources  # noqa: E402
from voidscreen.gravity_flow_inverse import weighted_quantile  # noqa: E402
from voidscreen.halo_backtrack import (  # noqa: E402
    component_samples,
    posterior_component_destinations,
    solve_capacity_transport,
    thin_bayes_chain,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def solve_checked(protocol, source_position, source_weight, destination_position, destination_weight, multiplier):
    best = None
    for iterations in protocol["iterations"]:
        plan, audit = solve_capacity_transport(
            source_position,
            source_weight,
            destination_position,
            destination_weight,
            capacity_multiplier=float(multiplier),
            entropy_length_kpc=float(protocol["entropy_length_kpc"]),
            iterations=int(iterations),
        )
        best = (plan, audit, int(iterations))
        if (
            audit["target_marginal_max_error"]
            <= float(protocol["maximum_target_marginal_error"])
            and audit["maximum_source_capacity_excess"]
            <= float(protocol["maximum_source_capacity_excess"])
        ):
            return best
    raise RuntimeError(f"capacity transport failed numerical gates: {best[1]}")


def main():
    config_path = ROOT / "configs/p0554_halo_backtrack_capacity_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    parent = json.loads((ROOT / protocol["parent_protocol"]).read_text(encoding="utf-8"))
    parent_report = json.loads((ROOT / protocol["parent_report"]).read_text(encoding="utf-8"))
    if parent_report["protocol"]["sha256"] != sha256(ROOT / protocol["parent_protocol"]):
        raise RuntimeError("parent inverse report does not match its protocol")
    interaction = json.loads(
        (ROOT / parent["inputs"]["interaction_protocol"]).read_text(encoding="utf-8")
    )
    contexts = raw_contexts(interaction)
    members_by_label, _ = load_route_sources(interaction, contexts)
    systems = {row["label"]: row for row in parent["systems"]}
    target_spec = next(
        row for row in parent["halo_target"]["target_variants"]
        if row["target_kind"] == protocol["target_kind"]
    )
    halo_settings = parent["halo_target"]
    axis = np.arange(
        float(halo_settings["grid_min_kpc"]),
        float(halo_settings["grid_max_kpc"]) + 0.5 * float(halo_settings["grid_spacing_kpc"]),
        float(halo_settings["grid_spacing_kpc"]),
    )
    route_rows = []
    origin_rows = []

    for context in contexts:
        label = context.label
        print(label, flush=True)
        scale = float(
            context.local["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"]
        )
        members = members_by_label[label].reset_index(drop=True)
        source_position = members[["x_arcsec", "y_arcsec"]].to_numpy(float) * scale
        source_weight = members.base_weight.to_numpy(float)
        source_weight /= np.sum(source_weight)
        chain = (
            ROOT / parent["inputs"]["chain_root"]
            / systems[label]["chain_directory"] / "bayes.dat"
        )
        headers, samples, _ = thin_bayes_chain(
            chain, int(halo_settings["posterior_samples_per_system"])
        )
        components = component_samples(headers, samples, scale)
        destination_position, destination_weight, component_ids, _ = (
            posterior_component_destinations(
                components,
                axis,
                width_mode=target_spec["width_mode"],
                width_kpc=float(target_spec["width_kpc"]),
                weight_mode=target_spec["weight_mode"],
                maximum_radius_kpc=float(halo_settings["maximum_radius_kpc"]),
                minimum_relative_density=float(
                    halo_settings["minimum_relative_component_density"]
                ),
            )
        )
        displacement = destination_position[None, :, :] - source_position[:, None, :]
        distance = np.linalg.norm(displacement, axis=2)
        for multiplier in protocol["capacity_multipliers"]:
            print(f"  q_cap={multiplier:g}", flush=True)
            plan, audit, iterations = solve_checked(
                protocol,
                source_position,
                source_weight,
                destination_position,
                destination_weight,
                multiplier,
            )
            source_outflow = np.sum(plan, axis=1)
            source_capacity = float(multiplier) * source_weight
            active = source_outflow > 1e-6
            normalized_outflow = source_outflow / np.sum(source_outflow)
            route_rows.append(
                {
                    "system_label": label,
                    "capacity_multiplier": float(multiplier),
                    "solver_iterations": iterations,
                    "mean_path_kpc": float(np.sum(plan * distance)),
                    "median_path_kpc": weighted_quantile(distance, plan, 0.5),
                    "p90_path_kpc": weighted_quantile(distance, plan, 0.9),
                    "rms_transport_kpc": float(
                        np.sqrt(np.sum(plan * np.square(distance)))
                    ),
                    "active_source_count": int(np.sum(active)),
                    "effective_origin_count": float(
                        1.0 / np.sum(np.square(normalized_outflow))
                    ),
                    "top_source_flow_fraction": float(np.max(normalized_outflow)),
                    "median_active_capacity_used_fraction": float(
                        np.median(
                            source_outflow[active]
                            / np.maximum(source_capacity[active], np.finfo(float).tiny)
                        )
                    ),
                    **audit,
                }
            )
            for object_id in np.unique(component_ids):
                component_flow = np.sum(plan[:, component_ids == object_id], axis=1)
                component_total = float(np.sum(component_flow))
                order = np.argsort(component_flow)[::-1]
                center = np.array(
                    [
                        np.median(components[int(object_id)]["x_kpc"]),
                        np.median(components[int(object_id)]["y_kpc"]),
                    ]
                )
                for rank, source_index in enumerate(order, start=1):
                    origin_rows.append(
                        {
                            "system_label": label,
                            "capacity_multiplier": float(multiplier),
                            "halo_id": int(object_id),
                            "origin_rank": rank,
                            "source_id": members.iloc[source_index].source_id,
                            "source_x_kpc": float(source_position[source_index, 0]),
                            "source_y_kpc": float(source_position[source_index, 1]),
                            "source_light_weight": float(source_weight[source_index]),
                            "flow_weight": float(component_flow[source_index]),
                            "fraction_of_halo_inflow": float(
                                component_flow[source_index]
                                / max(component_total, np.finfo(float).tiny)
                            ),
                            "source_to_posterior_median_halo_center_kpc": float(
                                np.linalg.norm(source_position[source_index] - center)
                            ),
                        }
                    )

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    routes = pd.DataFrame(route_rows)
    origins = pd.DataFrame(origin_rows)
    routes.to_csv(output / protocol["outputs"]["route_statistics"], index=False)
    origins.to_csv(output / protocol["outputs"]["halo_origins"], index=False)
    top = origins[origins.origin_rank.eq(1) & origins.capacity_multiplier.ge(2.0)]
    stability = []
    for (label, halo_id), group in top.groupby(["system_label", "halo_id"]):
        stability.append(
            {
                "system_label": label,
                "halo_id": int(halo_id),
                "same_top_origin_at_q2_q4_q8": group.source_id.nunique() == 1,
                "top_origin_ids": ",".join(group.sort_values("capacity_multiplier").source_id.astype(str)),
                "median_top_origin_fraction": float(group.fraction_of_halo_inflow.median()),
                "median_top_origin_distance_kpc": float(
                    group.source_to_posterior_median_halo_center_kpc.median()
                ),
            }
        )
    stability_table = pd.DataFrame(stability)
    q1 = routes[routes.capacity_multiplier.eq(1.0)].set_index("system_label")
    parent_primary = pd.DataFrame(parent_report["primary_inverse"]["per_system"]).set_index(
        "system_label"
    )
    q1_difference = np.max(
        np.abs(q1.rms_transport_kpc - parent_primary.rms_transport_kpc)
    )
    aggregate = routes.groupby("capacity_multiplier").agg(
        median_system_rms_kpc=("rms_transport_kpc", "median"),
        median_effective_origin_count=("effective_origin_count", "median"),
        median_top_source_flow_fraction=("top_source_flow_fraction", "median"),
    ).reset_index()
    report = {
        "report_version": "P0554-HALO-BACKTRACK-CAPACITY-RESULTS-0.1.0",
        "status": "complete",
        "protocol": {"path": str(config_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(config_path)},
        "coverage": {
            "systems": int(routes.system_label.nunique()),
            "cluster_scale_halos": int(origins[["system_label", "halo_id"]].drop_duplicates().shape[0]),
            "capacity_solutions": int(len(routes)),
            "maximum_target_marginal_error": float(routes.target_marginal_max_error.max()),
            "maximum_source_capacity_excess": float(routes.maximum_source_capacity_excess.max()),
            "q1_maximum_RMS_difference_from_parent_kpc": float(q1_difference),
        },
        "aggregate_by_capacity": aggregate.to_dict("records"),
        "origin_stability": {
            "halos_with_same_top_origin_at_q2_q4_q8": int(stability_table.same_top_origin_at_q2_q4_q8.sum()),
            "halos_total": int(len(stability_table)),
            "per_halo": stability_table.to_dict("records"),
        },
        "interpretation": (
            "Relaxing the all-sources-must-route assumption tests which baryons are economical origins. "
            "It does not identify a physical response amplitude or predict a destination from baryons alone."
        ),
        "limits": protocol["limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for label, group in routes.groupby("system_label"):
        axes[0].plot(group.capacity_multiplier, group.rms_transport_kpc, marker="o", label=label)
        axes[1].plot(group.capacity_multiplier, group.effective_origin_count, marker="o", label=label)
    axes[0].set(xscale="log", xticks=protocol["capacity_multipliers"], xlabel="source-capacity multiplier", ylabel="RMS route (kpc)")
    axes[1].set(xscale="log", xticks=protocol["capacity_multipliers"], xlabel="source-capacity multiplier", ylabel="effective number of origins")
    axes[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes[1].legend(fontsize=8)
    figure.suptitle("Halo backtracking when unused baryonic launch capacity may remain local")
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)
    summary = [
        "# Capacity-relaxed halo backtracking",
        "",
        report["interpretation"],
        "",
        f"The same top origin survives q_cap=2,4,8 for **{report['origin_stability']['halos_with_same_top_origin_at_q2_q4_q8']}/{report['origin_stability']['halos_total']}** modeled halos.",
        f"The q_cap=1 implementation reproduces the balanced parent RMS to within **{q1_difference:.3g} kpc**.",
        "",
        "These are origin-attribution sensitivities, not fitted constants or a forward gravity law.",
    ]
    (output / protocol["outputs"]["summary"]).write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps(json_safe(report), indent=2), flush=True)


if __name__ == "__main__":
    main()
