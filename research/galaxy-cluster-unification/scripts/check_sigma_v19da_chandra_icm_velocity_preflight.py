#!/usr/bin/env python3
"""Build the target-sealed V19DA Chandra velocity-region preflight.

This program may read only source-side spatial products, CSV metadata, file
sizes, and parent reports.  It deliberately never opens a source/background
PHA payload, an ARF/RMF payload, a fitted redshift, or a lensing/halo product.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19da_chandra_icm_velocity_preregistration.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19da_chandra_icm_velocity_preflight"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def local_path(relative: str) -> Path:
    return ROOT / relative


def wsl_unc(path: str, distribution: str) -> Path:
    if not path.startswith("/"):
        raise RuntimeError(f"external archive path is not absolute: {path}")
    return Path(f"\\\\wsl.localhost\\{distribution}\\{path.lstrip('/').replace('/', chr(92))}")


def validate_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"required input is absent: {path}")
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(f"hash mismatch for {path}: {observed} != {expected}")


def adjacency(binmap: np.ndarray, admitted: set[int]) -> dict[int, dict[int, int]]:
    graph: dict[int, dict[int, int]] = {value: {} for value in admitted}
    boundary: dict[tuple[int, int], int] = defaultdict(int)
    for left, right in ((binmap[:, :-1], binmap[:, 1:]), (binmap[:-1, :], binmap[1:, :])):
        different = left != right
        for first, second in zip(left[different], right[different], strict=True):
            first_id = int(first)
            second_id = int(second)
            if first_id in admitted and second_id in admitted:
                boundary[tuple(sorted((first_id, second_id)))] += 1
    for (first, second), pixels in boundary.items():
        graph[first][second] = pixels
        graph[second][first] = pixels
    return graph


def graph_component_sizes(graph: dict[int, dict[int, int]]) -> list[int]:
    unseen = set(graph)
    sizes: list[int] = []
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        stack = [seed]
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for neighbor in graph[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def deterministic_merge(
    statistics: pd.DataFrame,
    graph: dict[int, dict[int, int]],
    target_net_counts: float,
) -> list[dict[str, Any]]:
    admitted = statistics.loc[statistics["valid"].astype(bool)].copy()
    rows = {int(row.bin_id): row for row in admitted.itertuples(index=False)}
    active = set(rows)
    members = {key: {key} for key in rows}
    net = {key: float(row.net_counts) for key, row in rows.items()}
    science = {key: float(row.science_counts) for key, row in rows.items()}
    background = {key: float(row.scaled_background_counts) for key, row in rows.items()}
    pixels = {key: int(row.pixels) for key, row in rows.items()}
    working_graph = {key: dict(value) for key, value in graph.items()}

    while True:
        candidates = sorted(
            (net[node], node)
            for node in active
            if net[node] < target_net_counts and working_graph[node]
        )
        if not candidates:
            break
        _weight, first = candidates[0]
        surface_brightness = net[first] / pixels[first]
        second = min(
            working_graph[first],
            key=lambda node: (
                -working_graph[first][node],
                abs(surface_brightness - net[node] / pixels[node]),
                node,
            ),
        )
        keep = min(first, second)
        neighbors = (set(working_graph[first]) | set(working_graph[second])) - {first, second}
        new_edges = {
            node: working_graph[first].get(node, 0) + working_graph[second].get(node, 0)
            for node in neighbors
        }
        new_members = members[first] | members[second]
        new_net = net[first] + net[second]
        new_science = science[first] + science[second]
        new_background = background[first] + background[second]
        new_pixels = pixels[first] + pixels[second]

        for neighbor in neighbors:
            working_graph[neighbor].pop(first, None)
            working_graph[neighbor].pop(second, None)
            working_graph[neighbor][keep] = new_edges[neighbor]
        for node in (first, second):
            active.discard(node)
            working_graph.pop(node, None)
            members.pop(node, None)
            net.pop(node, None)
            science.pop(node, None)
            background.pop(node, None)
            pixels.pop(node, None)
        active.add(keep)
        working_graph[keep] = new_edges
        members[keep] = new_members
        net[keep] = new_net
        science[keep] = new_science
        background[keep] = new_background
        pixels[keep] = new_pixels

    output: list[dict[str, Any]] = []
    for group_id, root_id in enumerate(sorted(active), start=1):
        member_ids = sorted(members[root_id])
        output.append(
            {
                "group_id": group_id,
                "root_bin_id": root_id,
                "member_bin_ids": ";".join(str(value) for value in member_ids),
                "member_bin_count": len(member_ids),
                "net_counts_0p5_7_keV": net[root_id],
                "science_counts_0p5_7_keV": science[root_id],
                "scaled_background_counts_0p5_7_keV": background[root_id],
                "pixels": pixels[root_id],
                "meets_target": net[root_id] >= target_net_counts,
                "connected_by_construction": True,
            }
        )
    return output


def audit_external_products(config: dict[str, Any], product_index: pd.DataFrame) -> dict[str, Any]:
    distribution = config["runtime"]["wsl_distribution"]
    roles = (
        ("source_pha_name", "source_pha_bytes"),
        ("background_pha_name", "background_pha_bytes"),
        ("arf_name", "arf_bytes"),
        ("rmf_name", "rmf_bytes"),
    )
    checked = 0
    missing: list[str] = []
    wrong_size: list[str] = []
    for row in product_index.itertuples(index=False):
        directory = wsl_unc(str(row.cell_directory), distribution) / "products"
        for name_column, size_column in roles:
            path = directory / str(getattr(row, name_column))
            checked += 1
            if not path.is_file():
                missing.append(str(path))
                continue
            expected = int(getattr(row, size_column))
            if path.stat().st_size != expected:
                wrong_size.append(str(path))
    return {
        "products_checked_by_metadata_only": checked,
        "missing_products": missing,
        "wrong_size_products": wrong_size,
        "all_products_present_with_frozen_sizes": not missing and not wrong_size,
        "payload_hashes_reused_from_validated_parent_not_recomputed": True,
    }


def build(config_path: Path, output: Path, check_external: bool = True) -> dict[str, Any]:
    config = load_json(config_path)
    if config.get("freeze_state") != "frozen_before_any_v19da_pha_or_redshift_payload_access":
        raise RuntimeError("V19DA config is not frozen at the required target-sealed boundary")

    for item in config["inputs"].values():
        if not isinstance(item, dict) or "path" not in item or "sha256" not in item:
            continue
        validate_hash(local_path(item["path"]), item["sha256"])

    v19m = load_json(local_path(config["inputs"]["v19m_report"]["path"]))
    v19cx = load_json(local_path(config["inputs"]["v19cx_report"]["path"]))
    if v19m.get("status") != "both_adaptive_thermodynamic_region_gates_passed":
        raise RuntimeError("V19M source-region parent no longer passes")
    if v19cx.get("status") != "unified_spectral_combination_commissioning_gate_failed":
        raise RuntimeError("V19CX terminal parent status changed")
    if v19cx.get("validated_response_cells") != 5082:
        raise RuntimeError("V19CX validated response-cell count changed")
    if v19cx.get("full_494_region_combination_and_fit_authorized") is not False:
        raise RuntimeError("V19CX historical failed closure was altered")

    validated = pd.read_csv(local_path(config["inputs"]["validated_cell_index"]["path"]))
    products = pd.read_csv(local_path(config["inputs"]["unified_product_index"]["path"]))
    expected_columns = set(config["metadata_contract"]["required_product_index_columns"])
    if not expected_columns.issubset(products.columns):
        raise RuntimeError("unified product index is missing required metadata columns")
    if len(validated) != 5082 or len(products) != 5082:
        raise RuntimeError("validated response indexes no longer contain 5,082 rows")
    if products["cell_name"].duplicated().any() or validated["cell_name"].duplicated().any():
        raise RuntimeError("response index contains duplicate cell names")
    if set(products["cell_name"]) != set(validated["cell_name"]):
        raise RuntimeError("validated indexes disagree on cell identity")

    region_rows: list[dict[str, Any]] = []
    region_summary: dict[str, Any] = {}
    for cluster, cluster_config in config["clusters"].items():
        statistics_path = local_path(cluster_config["region_statistics"]["path"])
        binmap_path = local_path(cluster_config["binmap"]["path"])
        statistics = pd.read_csv(statistics_path)
        binmap = np.asarray(fits.getdata(binmap_path), dtype=int)
        admitted = set(statistics.loc[statistics["valid"].astype(bool), "bin_id"].astype(int))
        graph = adjacency(binmap, admitted)
        components = graph_component_sizes(graph)
        if components != [len(admitted)]:
            raise RuntimeError(f"{cluster} admitted-bin graph is not one connected component: {components}")
        indexed_bins = set(validated.loc[validated["cluster"].eq(cluster), "bin_id"].astype(int))
        if indexed_bins != admitted:
            raise RuntimeError(f"{cluster} response bins differ from admitted spatial bins")

        cluster_summary: dict[str, Any] = {
            "admitted_bins": len(admitted),
            "adjacency_edges": sum(len(value) for value in graph.values()) // 2,
            "connected_components": components,
            "validated_cells": int(validated["cluster"].eq(cluster).sum()),
            "branches": {},
        }
        for branch, target in config["regionization"]["net_count_targets_0p5_7_keV"].items():
            merged = deterministic_merge(statistics, graph, float(target))
            used = [int(value) for row in merged for value in str(row["member_bin_ids"]).split(";")]
            unique_use = len(used) == len(set(used)) == len(admitted) and set(used) == admitted
            if not unique_use or not all(row["meets_target"] for row in merged):
                raise RuntimeError(f"{cluster} {branch} merge did not conserve bins or meet its count target")
            minimum_regions = int(config["regionization"]["minimum_regions_by_branch"][branch])
            for row in merged:
                region_rows.append({"cluster": cluster, "branch": branch, **row})
            cluster_summary["branches"][branch] = {
                "target_net_counts": target,
                "regions": len(merged),
                "minimum_regions_gate": minimum_regions,
                "minimum_net_counts": min(row["net_counts_0p5_7_keV"] for row in merged),
                "median_net_counts": float(np.median([row["net_counts_0p5_7_keV"] for row in merged])),
                "maximum_net_counts": max(row["net_counts_0p5_7_keV"] for row in merged),
                "all_bins_used_once": unique_use,
                "all_regions_connected": True,
                "passes_region_count_gate": len(merged) >= minimum_regions,
            }
        region_summary[cluster] = cluster_summary

    external = (
        audit_external_products(config, products)
        if check_external
        else {
            "products_checked_by_metadata_only": 0,
            "all_products_present_with_frozen_sizes": None,
            "external_check_explicitly_skipped": True,
        }
    )
    gates = {
        "both_parent_states_match": True,
        "validated_indexes_have_5082_unique_matching_cells": True,
        "both_admitted_bin_graphs_are_connected": True,
        "every_merge_branch_uses_every_admitted_bin_once": all(
            branch["all_bins_used_once"]
            for cluster in region_summary.values()
            for branch in cluster["branches"].values()
        ),
        "every_merge_branch_meets_count_and_region_gates": all(
            branch["minimum_net_counts"] >= branch["target_net_counts"]
            and branch["passes_region_count_gate"]
            for cluster in region_summary.values()
            for branch in cluster["branches"].values()
        ),
        "external_products_present_with_frozen_sizes": external[
            "all_products_present_with_frozen_sizes"
        ]
        is True,
        "no_spectral_or_gravity_target_opened": True,
    }
    decision = (
        "passed_target_sealed_chandra_velocity_preflight"
        if all(gates.values())
        else "failed_target_sealed_chandra_velocity_preflight"
    )
    report = {
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "decision": decision,
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "parent_status": {"v19m": v19m["status"], "v19cx": v19cx["status"]},
        "region_summary": region_summary,
        "external_archive_audit": external,
        "gates": gates,
        "access_audit": {
            "source_or_background_pha_payload_opened": False,
            "arf_or_rmf_payload_opened": False,
            "channel_energy_or_line_centroid_opened": False,
            "temperature_abundance_or_redshift_fit_opened": False,
            "published_abell2146_velocity_outcome_opened": False,
            "lensing_halo_gravity_action_or_formula_payload_opened": False,
            "spatial_binmap_and_broad_count_statistics_opened": True,
            "external_science_products_checked_by_name_and_size_only": check_external,
        },
        "authorization": {
            "build_combined_ungrouped_source_background_and_response_products": decision.startswith("passed"),
            "open_bullet_development_spectra_before_the_combiner_is_hash_frozen": False,
            "open_abell2146_transfer_spectra_before_bullet_reproduction_passes": False,
            "derive_or_tune_a_gravity_formula": False,
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(region_rows).to_csv(output / "frozen_region_groups.csv", index=False)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-external", action="store_true")
    args = parser.parse_args()
    report = build(args.config, args.output, check_external=not args.skip_external)
    print(json.dumps({"decision": report["decision"], "gates": report["gates"]}, indent=2))


if __name__ == "__main__":
    main()
