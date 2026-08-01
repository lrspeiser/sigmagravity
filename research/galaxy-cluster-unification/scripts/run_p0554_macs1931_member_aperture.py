#!/usr/bin/env python3
"""Measure MACS1931 halo-backtrack sensitivity to the member aperture."""

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


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_members(path, parent):
    rows = []
    for line in Path(path).read_text(encoding="ascii").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        rows.append({"source_id": fields[0], "ra": float(fields[1]), "dec": float(fields[2]), "magnitude": float(fields[6])})
    frame = pd.DataFrame(rows)
    center_ra = 292.9570833333
    center_dec = -26.5756166667
    scale = 4.996269752867861
    frame["x_kpc"] = (frame.ra - center_ra) * 3600.0 * math.cos(math.radians(center_dec)) * scale
    frame["y_kpc"] = (frame.dec - center_dec) * 3600.0 * scale
    frame["radius_kpc"] = np.hypot(frame.x_kpc, frame.y_kpc)
    frame["base_weight"] = np.power(10.0, -0.4 * (frame.magnitude - frame.magnitude.min()))
    return frame, scale


def main():
    config_path = ROOT / "configs/p0554_macs1931_member_aperture_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    capacity = json.loads((ROOT / protocol["parent_capacity_protocol"]).read_text(encoding="utf-8"))
    parent = json.loads((ROOT / capacity["parent_protocol"]).read_text(encoding="utf-8"))
    members, scale = read_members(ROOT / protocol["member_catalog"], parent)
    system = next(row for row in parent["systems"] if row["label"] == "MACS1931")
    chain = ROOT / parent["inputs"]["chain_root"] / system["chain_directory"] / "bayes.dat"
    headers, samples, _ = thin_bayes_chain(chain, int(parent["halo_target"]["posterior_samples_per_system"]))
    components = component_samples(headers, samples, scale)
    target_spec = next(row for row in parent["halo_target"]["target_variants"] if row["target_kind"] == capacity["target_kind"])
    settings = parent["halo_target"]
    axis = np.arange(settings["grid_min_kpc"], settings["grid_max_kpc"] + 0.5 * settings["grid_spacing_kpc"], settings["grid_spacing_kpc"])
    destination, destination_weight, component_ids, _ = posterior_component_destinations(
        components,
        axis,
        width_mode=target_spec["width_mode"],
        width_kpc=target_spec["width_kpc"],
        weight_mode=target_spec["weight_mode"],
        maximum_radius_kpc=settings["maximum_radius_kpc"],
        minimum_relative_density=settings["minimum_relative_component_density"],
    )
    score_rows = []
    origin_rows = []
    halo_id = 2
    halo_center = np.array([np.median(components[halo_id]["x_kpc"]), np.median(components[halo_id]["y_kpc"])])
    for aperture in protocol["apertures_kpc"]:
        selected = members[members.radius_kpc <= float(aperture)].copy().reset_index(drop=True)
        source = selected[["x_kpc", "y_kpc"]].to_numpy(float)
        source_weight = selected.base_weight.to_numpy(float)
        source_weight /= np.sum(source_weight)
        distance = np.linalg.norm(destination[None, :, :] - source[:, None, :], axis=2)
        for multiplier in protocol["capacity_multipliers"]:
            plan, audit, iterations = solve_checked(capacity, source, source_weight, destination, destination_weight, multiplier)
            component_flow = np.sum(plan[:, component_ids == halo_id], axis=1)
            order = np.argsort(component_flow)[::-1]
            top = int(order[0])
            score_rows.append({
                "aperture_kpc": float(aperture),
                "member_count": len(selected),
                "capacity_multiplier": float(multiplier),
                "solver_iterations": iterations,
                "mean_path_kpc": float(np.sum(plan * distance)),
                "median_path_kpc": weighted_quantile(distance, plan, 0.5),
                "p90_path_kpc": weighted_quantile(distance, plan, 0.9),
                "rms_transport_kpc": float(np.sqrt(np.sum(plan * np.square(distance)))),
                "southern_halo_top_origin_id": selected.iloc[top].source_id,
                "southern_halo_top_origin_distance_kpc": float(np.linalg.norm(source[top] - halo_center)),
                "southern_halo_top_origin_fraction": float(component_flow[top] / np.sum(component_flow)),
                **audit,
            })
            for rank, index in enumerate(order[:10], start=1):
                origin_rows.append({
                    "aperture_kpc": float(aperture),
                    "capacity_multiplier": float(multiplier),
                    "origin_rank": rank,
                    "source_id": selected.iloc[index].source_id,
                    "source_x_kpc": float(source[index, 0]),
                    "source_y_kpc": float(source[index, 1]),
                    "distance_to_halo_median_kpc": float(np.linalg.norm(source[index] - halo_center)),
                    "fraction_of_southern_halo_inflow": float(component_flow[index] / np.sum(component_flow)),
                })
    scores = pd.DataFrame(score_rows)
    origins = pd.DataFrame(origin_rows)
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / protocol["outputs"]["scores"], index=False)
    origins.to_csv(output / protocol["outputs"]["halo_origins"], index=False)
    parent_row = scores[(scores.aperture_kpc == 300.0) & (scores.capacity_multiplier == 4.0)].iloc[0]
    full_row = scores[(scores.aperture_kpc == 450.0) & (scores.capacity_multiplier == 4.0)].iloc[0]
    report = {
        "report_version": "P0554-MACS1931-MEMBER-APERTURE-RESULTS-0.1.0",
        "status": "complete",
        "protocol": {"path": str(config_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(config_path)},
        "coverage": {"published_members_total": len(members), "maximum_member_radius_kpc": float(members.radius_kpc.max()), "solutions": len(scores)},
        "qcap4_aperture_effect": {
            "member_count_300kpc": int(parent_row.member_count),
            "member_count_450kpc": int(full_row.member_count),
            "rms_route_300kpc": float(parent_row.rms_transport_kpc),
            "rms_route_450kpc": float(full_row.rms_transport_kpc),
            "rms_improvement_fraction": float(1.0 - full_row.rms_transport_kpc / parent_row.rms_transport_kpc),
            "top_origin_300kpc": str(parent_row.southern_halo_top_origin_id),
            "top_origin_450kpc": str(full_row.southern_halo_top_origin_id),
            "top_origin_distance_300kpc": float(parent_row.southern_halo_top_origin_distance_kpc),
            "top_origin_distance_450kpc": float(full_row.southern_halo_top_origin_distance_kpc),
        },
        "scores": scores.to_dict("records"),
        "interpretation": "Restoring published members diagnoses how much of the long path was manufactured by the earlier analysis aperture; the remaining gap still requires wider membership data or a genuinely nonlocal explanation.",
        "limits": protocol["limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for multiplier, group in scores.groupby("capacity_multiplier"):
        axis.plot(group.aperture_kpc, group.rms_transport_kpc, marker="o", label=f"q_cap={multiplier:g}")
    axis.set(xlabel="member aperture (kpc)", ylabel="RMS route (kpc)", title="MACS1931 source-catalog aperture sensitivity")
    axis.legend()
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)
    summary = (
        "# MACS1931 member-aperture sensitivity\n\n"
        f"At q_cap=4, restoring all 120 published members changes the RMS route by {100*report['qcap4_aperture_effect']['rms_improvement_fraction']:+.1f}% and moves the dominant southern-halo origin from {report['qcap4_aperture_effect']['top_origin_distance_300kpc']:.1f} to {report['qcap4_aperture_effect']['top_origin_distance_450kpc']:.1f} kpc from the halo median.\n"
    )
    (output / protocol["outputs"]["summary"]).write_text(summary, encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
