#!/usr/bin/env python3
"""Combine the frozen V19DA Bullet velocity regions without spectral fitting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_sigma_v19cw_observation_hierarchy_equivalence as v19cw

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19db_bullet_velocity_combination.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19db_bullet_velocity_combination"
DEFAULT_SCRATCH = Path("/home/henry/sigma-v19db-bullet-velocity-combination/v100")
AUTHORIZED_STATUS = "bullet_primary_velocity_region_combination_passed"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_parent(item: dict[str, Any]) -> Path:
    path = ROOT / item["path"]
    if not path.is_file() or sha256(path) != item["sha256"]:
        raise RuntimeError(f"V19DB frozen parent changed: {path}")
    return path


def validate_frozen(config: dict[str, Any]) -> dict[str, Path]:
    expected_state = "frozen_after_committed_v19da_preflight_before_any_v19db_pha_payload_access"
    if config.get("freeze_state") != expected_state:
        raise RuntimeError("V19DB config is not frozen at the payload-blind boundary")
    implementation = config["implementation"]
    if implementation["runner"] != Path(__file__).resolve().relative_to(ROOT).as_posix():
        raise RuntimeError("V19DB config names another runner")
    if implementation["runner_sha256"] != sha256(Path(__file__).resolve()):
        raise RuntimeError("V19DB runner changed after its payload-blind hash freeze")
    parents = {
        key: validate_parent(item)
        for key, item in config["parents"].items()
        if isinstance(item, dict) and "path" in item
    }
    preflight = load_json(parents["v19da_preflight_report"])
    if preflight.get("decision") != config["parents"]["v19da_preflight_report"]["required_decision"]:
        raise RuntimeError("V19DA source-only preflight is not a terminal pass")
    equivalence = load_json(parents["v19cw_report"])
    if equivalence.get("status") != config["parents"]["v19cw_report"]["required_status"]:
        raise RuntimeError("V19CW hierarchy-equivalence parent is not a terminal pass")
    auth = config["authorization"]
    sealed = (
        auth["open_bullet_primary_pha_and_response_after_this_combiner_is_hash_frozen"]
        and auth["combine_bullet_primary_8000_regions"]
        and not auth["combine_bullet_robustness_or_obsid554"]
        and not auth["open_abell2146"]
        and not auth["fit_temperature_abundance_redshift_or_velocity"]
        and not auth["open_lensing_halo_or_gravity_payload"]
        and not auth["derive_or_change_action_or_gravity_constants"]
    )
    if not sealed:
        raise RuntimeError("V19DB authorization boundary is open")
    hierarchy = config["hierarchy"]
    if hierarchy != {
        "partition_key": "obsid",
        "first_level_method": "sum",
        "final_method": "sum",
        "bscale_method": "asca",
        "exp_origin": "pha",
        "intermediate_rmf_threshold": 0.0,
        "final_rmf_threshold": 1e-06,
        "reason": hierarchy["reason"],
    }:
        raise RuntimeError("V19DB hierarchy changed")
    return parents


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def cell_key(row: dict[str, Any]) -> tuple[str, int, int, int]:
    return (str(row["cluster"]), int(row["bin_id"]), int(row["obsid"]), int(row["ccd_id"]))


def build_plan(config: dict[str, Any], parents: dict[str, Path]) -> list[dict[str, Any]]:
    workload = config["workload"]
    groups = [
        row
        for row in read_csv(parents["frozen_region_groups"])
        if row["cluster"] == workload["cluster"] and row["branch"] == workload["branch"]
    ]
    if len(groups) != int(workload["regions"]):
        raise RuntimeError("V19DB frozen region count changed")
    bin_to_group: dict[int, int] = {}
    members_by_group: dict[int, list[int]] = {}
    for group in groups:
        group_id = int(group["group_id"])
        members = [int(value) for value in group["member_bin_ids"].split(";")]
        members_by_group[group_id] = members
        for bin_id in members:
            if bin_id in bin_to_group:
                raise RuntimeError(f"V19DB source bin {bin_id} occurs in multiple regions")
            bin_to_group[bin_id] = group_id
    if len(bin_to_group) != int(workload["source_bins"]):
        raise RuntimeError("V19DB does not partition all source bins")

    primary_obsids = {int(value) for value in workload["obsids"]}
    excluded_obsids = {int(value) for value in workload["excluded_primary_obsids"]}
    unified_rows = read_csv(parents["unified_product_index"])
    selected = [
        row
        for row in unified_rows
        if row["cluster"] == workload["cluster"] and int(row["obsid"]) in primary_obsids
    ]
    if len(selected) != int(workload["response_cells"]):
        raise RuntimeError("V19DB primary response-cell count changed")
    if any(int(row["obsid"]) in excluded_obsids for row in selected):
        raise RuntimeError("V19DB admitted an excluded observation")
    keys = [cell_key(row) for row in selected]
    if len(set(keys)) != len(keys):
        raise RuntimeError("V19DB selected duplicate response cells")

    validated_rows = {cell_key(row): row for row in read_csv(parents["validated_cell_index"])}
    plan_by_group: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        bin_id = int(row["bin_id"])
        if bin_id not in bin_to_group:
            raise RuntimeError(f"V19DB product index contains unassigned bin {bin_id}")
        key = cell_key(row)
        if key not in validated_rows:
            raise RuntimeError(f"V19DB validated index lacks {key}")
        validated = validated_rows[key]
        if validated["source_pha_sha256"] != row["source_pha_sha256"]:
            raise RuntimeError(f"V19DB source hash disagreement for {row['cell_name']}")
        product_root = Path(row["cell_directory"]) / "products"
        products = {
            "source": product_root / row["source_pha_name"],
            "background": product_root / row["background_pha_name"],
            "arf": product_root / row["arf_name"],
            "rmf": product_root / row["rmf_name"],
        }
        expected = {
            "source": (int(row["source_pha_bytes"]), row["source_pha_sha256"]),
            "background": (int(row["background_pha_bytes"]), row["background_pha_sha256"]),
            "arf": (int(row["arf_bytes"]), row["arf_sha256"]),
            "rmf": (int(row["rmf_bytes"]), row["rmf_sha256"]),
        }
        plan_by_group[bin_to_group[bin_id]].append(
            {
                "cluster": row["cluster"],
                "bin_id": bin_id,
                "obsid": int(row["obsid"]),
                "ccd_id": int(row["ccd_id"]),
                "cell_name": row["cell_name"],
                "source_pha_total_counts": int(validated["source_pha_total_counts"]),
                "products": products,
                "expected": expected,
            }
        )
    plan: list[dict[str, Any]] = []
    for group in sorted(groups, key=lambda row: int(row["group_id"])):
        group_id = int(group["group_id"])
        cells = sorted(plan_by_group[group_id], key=lambda row: (row["obsid"], row["bin_id"], row["ccd_id"]))
        observed_bins = {int(row["bin_id"]) for row in cells}
        if observed_bins != set(members_by_group[group_id]):
            raise RuntimeError(f"V19DB region {group_id} cell/bin membership changed")
        plan.append(
            {
                "group_id": group_id,
                "root_bin_id": int(group["root_bin_id"]),
                "member_bin_ids": members_by_group[group_id],
                "net_counts_0p5_7_keV": float(group["net_counts_0p5_7_keV"]),
                "cells": cells,
            }
        )
    assigned = [cell["cell_name"] for region in plan for cell in region["cells"]]
    if len(assigned) != int(workload["response_cells"]) or len(set(assigned)) != len(assigned):
        raise RuntimeError("V19DB plan does not partition all primary cells exactly once")
    return plan


def validate_products(cells: list[dict[str, Any]]) -> None:
    for cell in cells:
        for role, path in cell["products"].items():
            size, digest = cell["expected"][role]
            if not path.is_file() or path.stat().st_size != size or sha256(path) != digest:
                raise RuntimeError(f"V19DB changed {role} product: {path}")


def pha_total_counts(path: Path) -> int:
    with fits.open(path, memmap=False) as hdus:
        counts = np.asarray(hdus["SPECTRUM"].data["COUNTS"], dtype=np.int64)
    return int(counts.sum())


def pha_links(path: Path) -> dict[str, str]:
    with fits.open(path, memmap=False) as hdus:
        header = hdus["SPECTRUM"].header
        return {key: str(header[key]) for key in ("BACKFILE", "ANCRFILE", "RESPFILE")}


def snapshot_product(source: Path, destination: Path, role: str) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(destination) != sha256(source):
            raise RuntimeError(f"V19DB existing frozen snapshot changed: {destination}")
    else:
        shutil.copy2(source, destination)
    return {
        "role": role,
        "path": destination.relative_to(ROOT).as_posix(),
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    }


def hierarchical_combine_region(
    region: dict[str, Any], config: dict[str, Any], scratch: Path, output: Path
) -> dict[str, Any]:
    group_id = int(region["group_id"])
    label = f"BULLET_primary8000_region{group_id:03d}"
    work = scratch / label
    env = v19cw.inherited_spectra.isolated_environment(os.environ, work / "pfiles", work / "tmp")
    cells = region["cells"]
    validate_products(cells)
    by_obsid: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        by_obsid[int(cell["obsid"])].append(cell)
    first_level: list[dict[str, Any]] = []
    for obsid in sorted(by_obsid):
        group = by_obsid[obsid]
        group_work = work / f"obs{obsid}"
        pha = v19cw.pha_only_combine(
            f"{label}_obs{obsid}", [row["products"]["source"] for row in group], group_work, config, env
        )
        response = v19cw.add_response(
            f"{label}_obs{obsid}",
            [row["products"]["source"] for row in group],
            [row["products"]["arf"] for row in group],
            [row["products"]["rmf"] for row in group],
            group_work,
            float(config["hierarchy"]["intermediate_rmf_threshold"]),
            env,
        )
        first_level.append({"obsid": obsid, "cells": len(group), "pha": pha, "response": response})
    expected_obsids = [int(value) for value in config["workload"]["obsids"]]
    if [row["obsid"] for row in first_level] != sorted(expected_obsids):
        raise RuntimeError(f"V19DB region {group_id} lacks a primary observation")
    final_work = work / "final"
    final_pha = v19cw.pha_only_combine(
        label, [row["pha"]["source"] for row in first_level], final_work, config, env
    )
    final_response = v19cw.add_response(
        label,
        [row["pha"]["source"] for row in first_level],
        [row["response"]["arf"] for row in first_level],
        [row["response"]["rmf"] for row in first_level],
        final_work,
        float(config["hierarchy"]["final_rmf_threshold"]),
        env,
    )
    links = v19cw.set_links(
        final_pha["source"], final_pha["background"], final_response["arf"], final_response["rmf"]
    )
    expected_links = {
        "BACKFILE": final_pha["background"].name,
        "ANCRFILE": final_response["arf"].name,
        "RESPFILE": final_response["rmf"].name,
    }
    expected_counts = sum(int(row["source_pha_total_counts"]) for row in cells)
    combined_counts = pha_total_counts(final_pha["source"])
    if combined_counts != expected_counts or links != expected_links:
        raise RuntimeError(f"V19DB region {group_id} count conservation or link gate failed")
    snapshot_root = output / "frozen_products" / label
    products = [
        snapshot_product(final_pha["source"], snapshot_root / final_pha["source"].name, "ungrouped_source_spectrum"),
        snapshot_product(final_pha["background"], snapshot_root / final_pha["background"].name, "background_spectrum"),
        snapshot_product(final_response["arf"], snapshot_root / final_response["arf"].name, "source_arf"),
        snapshot_product(final_response["rmf"], snapshot_root / final_response["rmf"].name, "source_rmf"),
    ]
    snapshot_source = ROOT / products[0]["path"]
    if pha_links(snapshot_source) != expected_links:
        raise RuntimeError(f"V19DB region {group_id} frozen PHA links changed")
    return {
        "group_id": group_id,
        "label": label,
        "root_bin_id": int(region["root_bin_id"]),
        "member_bins": len(region["member_bin_ids"]),
        "cells": len(cells),
        "obsid_cell_counts": {str(obsid): len(by_obsid[obsid]) for obsid in sorted(by_obsid)},
        "expected_full_pha_source_counts": expected_counts,
        "combined_full_pha_source_counts": combined_counts,
        "full_pha_count_conservation_exact": combined_counts == expected_counts,
        "links_exact": links == expected_links,
        "products": products,
        "scratch_products": {
            "source": str(final_pha["source"]),
            "background": str(final_pha["background"]),
            "arf": str(final_response["arf"]),
            "rmf": str(final_response["rmf"]),
        },
    }


def direct_combine_pilot(
    region: dict[str, Any], config: dict[str, Any], scratch: Path
) -> dict[str, Path]:
    label = f"BULLET_primary8000_region{int(region['group_id']):03d}_direct"
    work = scratch / "equivalence" / label
    env = v19cw.inherited_spectra.isolated_environment(os.environ, work / "pfiles", work / "tmp")
    stack = work / "source_spectra.lis"
    stack.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(str(row["products"]["source"]) for row in region["cells"]) + "\n"
    if stack.exists() and stack.read_text(encoding="utf-8") != content:
        raise RuntimeError("V19DB direct-pilot source stack changed")
    stack.write_text(content, encoding="utf-8")
    outroot = work / label
    products = {
        "source": work / f"{label}_src.pi",
        "background": work / f"{label}_bkg.pi",
        "arf": work / f"{label}_src.arf",
        "rmf": work / f"{label}_src.rmf",
    }
    command = [
        "combine_spectra",
        f"src_spectra=@{stack}",
        f"outroot={outroot}",
        f"method={config['hierarchy']['final_method']}",
        f"bscale_method={config['hierarchy']['bscale_method']}",
        f"exp_origin={config['hierarchy']['exp_origin']}",
        "clobber=no",
        "verbose=1",
        "mode=h",
    ]
    v19cw.inherited_spectra.run_step(command, work / "logs" / "combine_spectra.log", list(products.values()), env)
    return products


def compare_pilot(
    region: dict[str, Any], hierarchical: dict[str, Any], config: dict[str, Any], scratch: Path
) -> dict[str, Any]:
    direct = direct_combine_pilot(region, config, scratch)
    candidate = {key: Path(value) for key, value in hierarchical["scratch_products"].items()}
    source = v19cw.pha_comparison(direct["source"], candidate["source"], grouped=False)
    background = v19cw.pha_comparison(direct["background"], candidate["background"], grouped=False)
    arf = v19cw.arf_comparison(direct["arf"], candidate["arf"])
    rmf, matrices = v19cw.rmf_comparison(direct["rmf"], candidate["rmf"])
    folds = v19cw.folded_comparison(direct["arf"], candidate["arf"], matrices[0], matrices[1])
    tolerance = float(config["gates"]["direct_hierarchical_pilot_forward_fold_relative_l1_at_most"])
    gates = {
        "source_counts_exact": source["counts_exact"],
        "source_exposure_exact_to_1e_12": source["exposure_relative_difference"] <= 1e-12,
        "background_relative_l1_at_most_1e_12": background["counts_relative_l1_difference"] <= 1e-12,
        "arf_energy_grid_exact": arf["energy_grid_exact"],
        "rmf_energy_and_channel_grids_exact": rmf["energy_and_channel_grids_exact"],
        "every_forward_fold_relative_l1_at_most_1e_8": all(
            item["relative_l1_difference"] <= tolerance for item in folds.values()
        ),
    }
    return {
        "group_id": int(region["group_id"]),
        "selection_rule": config["workload"]["pilot_selection_rule"],
        "source": source,
        "background": background,
        "arf": arf,
        "rmf": rmf,
        "forward_folds": folds,
        "gates": gates,
        "passed": all(gates.values()),
    }


def write_manifest(records: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    path = output / "combined_region_manifest.csv"
    fields = [
        "group_id", "label", "root_bin_id", "member_bins", "cells",
        "expected_full_pha_source_counts", "combined_full_pha_source_counts",
        "full_pha_count_conservation_exact", "links_exact",
        "source_path", "source_sha256", "background_path", "background_sha256",
        "arf_path", "arf_sha256", "rmf_path", "rmf_sha256",
    ]
    role_columns = {
        "ungrouped_source_spectrum": "source",
        "background_spectrum": "background",
        "source_arf": "arf",
        "source_rmf": "rmf",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in sorted(records, key=lambda row: int(row["group_id"])):
            row = {key: record[key] for key in fields[:9]}
            for product in record["products"]:
                prefix = role_columns[product["role"]]
                row[f"{prefix}_path"] = product["path"]
                row[f"{prefix}_sha256"] = product["sha256"]
            writer.writerow(row)
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "rows": len(records),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def execute(config: dict[str, Any], output: Path, scratch: Path) -> dict[str, Any]:
    parents = validate_frozen(config)
    plan = build_plan(config, parents)
    pilot_group = int(config["workload"]["region_equivalence_pilot_group_id"])
    pilot_region = next(region for region in plan if int(region["group_id"]) == pilot_group)
    pilot_record = hierarchical_combine_region(pilot_region, config, scratch, output)
    pilot = compare_pilot(pilot_region, pilot_record, config, scratch)
    if not pilot["passed"]:
        raise RuntimeError("V19DB direct-versus-hierarchical regional response pilot failed")
    records = [pilot_record]
    remaining = [region for region in plan if int(region["group_id"]) != pilot_group]
    workers = int(config["runtime"]["parallel_regions"])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(hierarchical_combine_region, region, config, scratch, output): int(region["group_id"])
            for region in remaining
        }
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda row: int(row["group_id"]))
    manifest = write_manifest(records, output)
    cells = [cell["cell_name"] for region in plan for cell in region["cells"]]
    obsid_counts = Counter(cell["obsid"] for region in plan for cell in region["cells"])
    gates = {
        "43_regions_combined": len(records) == int(config["workload"]["regions"]),
        "366_bins_partitioned_exactly_once": len({bin_id for region in plan for bin_id in region["member_bin_ids"]}) == int(config["workload"]["source_bins"]),
        "3483_cells_partitioned_exactly_once": len(cells) == len(set(cells)) == int(config["workload"]["response_cells"]),
        "all_regions_conserve_full_pha_counts_exactly": all(row["full_pha_count_conservation_exact"] for row in records),
        "all_region_links_exact": all(row["links_exact"] for row in records),
        "all_regions_have_four_frozen_products": all(len(row["products"]) == 4 for row in records),
        "direct_hierarchical_pilot_passed": pilot["passed"],
    }
    clean_records = []
    for record in records:
        clean = dict(record)
        clean.pop("scratch_products")
        clean_records.append(clean)
    return {
        "status": AUTHORIZED_STATUS if all(gates.values()) else "bullet_primary_velocity_region_combination_gate_failed",
        "plan": {
            "regions": len(plan),
            "source_bins": len({bin_id for region in plan for bin_id in region["member_bin_ids"]}),
            "response_cells": len(cells),
            "obsid_cell_counts": {str(key): obsid_counts[key] for key in sorted(obsid_counts)},
        },
        "equivalence_pilot": pilot,
        "combined_region_manifest": manifest,
        "regions": clean_records,
        "gates": gates,
        "bullet_spectral_fitting_authorized": all(gates.values()),
    }


def preflight(config: dict[str, Any]) -> dict[str, Any]:
    parents = validate_frozen(config)
    plan = build_plan(config, parents)
    return {
        "status": "v19db_payload_blind_execution_plan_passed",
        "regions": len(plan),
        "source_bins": len({bin_id for region in plan for bin_id in region["member_bin_ids"]}),
        "response_cells": sum(len(region["cells"]) for region in plan),
        "pha_or_response_payload_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    output = args.output.resolve()
    config = load_json(config_path)
    try:
        result = preflight(config) if args.preflight_only else execute(config, output, args.scratch.resolve())
    except Exception as exc:  # noqa: BLE001
        result = {
            "status": "v19db_execution_failed_closed",
            "execution_exception": f"{type(exc).__name__}: {exc}",
            "gates": {"execution_completed": False},
            "bullet_spectral_fitting_authorized": False,
        }
    report = {
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        **result,
        "temperature_abundance_redshift_or_velocity_fitted": False,
        "abell2146_payload_opened": False,
        "lensing_halo_gravity_or_action_payload_opened": False,
    }
    report_path = output / ("preflight_report.json" if args.preflight_only else "report.json")
    atomic_json(report_path, report)
    print(json.dumps({key: report.get(key) for key in ("status", "execution_exception")}, indent=2, sort_keys=True))
    required = "v19db_payload_blind_execution_plan_passed" if args.preflight_only else AUTHORIZED_STATUS
    if report["status"] != required:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
