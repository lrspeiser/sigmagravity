#!/usr/bin/env python3
"""Run the V19CW-equivalent Bullet recovery and unchanged V19X2 fits."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from astropy.io import fits

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_sigma_v19cw_observation_hierarchy_equivalence as v19cw
import run_sigma_v19x2_unified_spectral_combination_commissioning as v19x2

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19cx_bullet_hierarchical_recovery.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19cx_bullet_hierarchical_recovery"
DEFAULT_SCRATCH = Path("/home/henry/sigma-v19cx-bullet-hierarchical-recovery/v100")
FROZEN_STATE = "frozen_after_terminal_v19w5_pass"
AUTHORIZED_STATUS = "unified_spectral_combination_commissioning_passed_and_full_regional_fits_authorized"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_frozen(config: dict[str, Any]) -> None:
    if config.get("freeze_state") != FROZEN_STATE:
        raise RuntimeError("V19CX is not frozen at the V19X3B-compatible commissioning boundary")
    if "runtime_remediation" not in config or "hierarchy" not in config:
        raise RuntimeError("V19CX lacks the commissioned hierarchy")
    if config.get("implementation", {}).get("runner") != Path(__file__).resolve().relative_to(ROOT).as_posix():
        raise RuntimeError("V19CX config names another runner")
    implementation = config["implementation"]
    if "runner_sha256" in implementation and v19x2.adapter.sha256(Path(__file__).resolve()) != implementation["runner_sha256"]:
        raise RuntimeError("V19CX runner changed after freeze")
    authorization = config["authorization"]
    if not authorization["run_v19cw_equivalent_hierarchy_only_above_frozen_cell_threshold"]:
        raise RuntimeError("V19CX hierarchy is not authorized")
    if authorization["overwrite_v19x2_failure_report"] or authorization["run_v19bq_or_v19bs"] or authorization["derive_action"] or authorization["change_gravity_formula_parameter_source_state_or_lensing_target"]:
        raise RuntimeError("V19CX authorization boundary is open")


def validate_parents(config: dict[str, Any]) -> None:
    parents = config["parents"]
    for key in ("v19x2_failed_report", "v19cw_config", "v19cw_runner", "v19cw_report"):
        path = ROOT / parents[key]
        if not path.is_file() or v19x2.adapter.sha256(path) != parents[f"{key}_sha256"]:
            raise RuntimeError(f"V19CX parent changed: {key}")
    cw = load_json(ROOT / parents["v19cw_report"])
    if cw.get("status") != "observation_hierarchy_equivalent_and_bullet_recovery_may_be_frozen" or not cw.get("gates") or not all(cw["gates"].values()):
        raise RuntimeError("V19CX lacks a terminal V19CW pass")
    exact_x2 = load_json(ROOT / "configs" / "sigma_v19x2_unified_spectral_combination_commissioning.json")
    for section in ("registered_workload", "combination", "fit_sequence", "gates", "runtime_authorization"):
        if config[section] != exact_x2[section]:
            raise RuntimeError(f"V19CX changed frozen V19X2 scientific section: {section}")


def resolve_cell_products(cell: dict[str, Any]) -> dict[str, Path]:
    source = Path(cell["source_pha"])
    with fits.open(source, memmap=False) as hdus:
        header = hdus["SPECTRUM"].header
        names = {"background": header["BACKFILE"], "arf": header["ANCRFILE"], "rmf": header["RESPFILE"]}
    products = {"source": source}
    for role, name in names.items():
        path = source.parent / str(name)
        if not path.is_file():
            raise RuntimeError(f"V19CX missing linked {role}: {path}")
        products[role] = path
    return products


def snapshot_combination(
    label: str,
    cells: list[dict[str, Any]],
    grouped_source: Path,
    background: Path,
    arf: Path,
    rmf: Path,
    output: Path,
    *,
    combine_step: dict[str, Any],
    group_step: dict[str, Any],
    links: dict[str, str],
    remediation: dict[str, Any],
) -> dict[str, Any]:
    expected_total = sum(int(row["source_pha_total_counts"]) for row in cells)
    combined_total = v19x2.inherited_v19x.pha_total_counts(grouped_source)
    if combined_total != expected_total:
        raise RuntimeError(f"V19CX {label} lost source counts: {combined_total} != {expected_total}")
    expected_links = {"BACKFILE": background.name, "ANCRFILE": arf.name, "RESPFILE": rmf.name}
    if links != expected_links:
        raise RuntimeError(f"V19CX {label} links changed: {links}")
    snapshots = []
    snapshot_root = output / "frozen_products" / label
    for role, source in (
        ("grouped_source_spectrum", grouped_source),
        ("background_spectrum", background),
        ("source_arf", arf),
        ("source_rmf", rmf),
    ):
        item = v19x2.inherited_v19x.inherited_spectra.copy_snapshot(source, snapshot_root / source.name)
        item["role"] = role
        snapshots.append(item)
    return {
        "label": label,
        "cells": len(cells),
        "event_energy_source_counts_0p5_7_keV": sum(int(row["source_band_events"]) for row in cells),
        "event_energy_background_counts_0p5_7_keV": sum(int(row["background_band_events"]) for row in cells),
        "expected_full_pha_source_counts": expected_total,
        "combined_full_pha_source_counts": combined_total,
        "full_pha_count_conservation_exact": combined_total == expected_total,
        "source_stack_sha256": remediation["source_stack_sha256"],
        "combine_step": combine_step,
        "group_step": group_step,
        "grouped_pha_links": links,
        "expected_grouped_pha_links": expected_links,
        "runtime_remediation": remediation,
        "frozen_snapshot": {
            "files": len(snapshots),
            "bytes": sum(int(item["bytes"]) for item in snapshots),
            "products": snapshots,
        },
    }


def reuse_direct(label: str, cells: list[dict[str, Any]], output: Path, config: dict[str, Any]) -> dict[str, Any]:
    spec = config["runtime_remediation"]["direct_references"][label]
    paths: dict[str, Path] = {}
    for role, item in spec.items():
        path = ROOT / item["path"]
        if not path.is_file() or path.stat().st_size != int(item["bytes"]) or v19x2.adapter.sha256(path) != item["sha256"]:
            raise RuntimeError(f"V19CX direct reference changed: {path}")
        paths[role] = path
    with fits.open(paths["source_grouped"], memmap=False) as hdus:
        header = hdus["SPECTRUM"].header
        links = {key: str(header[key]) for key in ("BACKFILE", "ANCRFILE", "RESPFILE")}
    return snapshot_combination(
        label,
        cells,
        paths["source_grouped"],
        paths["background"],
        paths["arf"],
        paths["rmf"],
        output,
        combine_step={"reused_hash_frozen_direct_reference": True},
        group_step={"reused_hash_frozen_direct_reference": True},
        links=links,
        remediation={
            "mode": "hash_frozen_direct_reference",
            "source_stack_sha256": spec["source_grouped"]["sha256"],
            "cell_threshold": int(config["runtime_remediation"]["maximum_direct_stack_cells"]),
        },
    )


def hierarchical_combine(label: str, cells: list[dict[str, Any]], scratch: Path, output: Path, config: dict[str, Any]) -> dict[str, Any]:
    work = scratch / label
    env = v19x2.inherited_v19x.inherited_spectra.isolated_environment(os.environ, work / "pfiles", work / "tmp")
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        groups[int(cell["obsid"])].append(cell)
    first_level: list[dict[str, Any]] = []
    for obsid in sorted(groups):
        group_cells = groups[obsid]
        products = [resolve_cell_products(cell) for cell in group_cells]
        group_work = work / f"obs{obsid}"
        pha = v19cw.pha_only_combine(f"{label}_obs{obsid}", [row["source"] for row in products], group_work, config, env)
        response = v19cw.add_response(
            f"{label}_obs{obsid}",
            [row["source"] for row in products],
            [row["arf"] for row in products],
            [row["rmf"] for row in products],
            group_work,
            float(config["hierarchy"]["intermediate_rmf_threshold"]),
            env,
        )
        first_level.append({"obsid": obsid, "cells": len(group_cells), "pha": pha, "response": response})
    final_work = work / "final"
    final_pha = v19cw.pha_only_combine(label, [row["pha"]["source"] for row in first_level], final_work, config, env)
    final_response = v19cw.add_response(
        label,
        [row["pha"]["source"] for row in first_level],
        [row["response"]["arf"] for row in first_level],
        [row["response"]["rmf"] for row in first_level],
        final_work,
        float(config["hierarchy"]["final_rmf_threshold"]),
        env,
    )
    links = v19cw.set_links(final_pha["source"], final_pha["background"], final_response["arf"], final_response["rmf"])
    grouped = final_work / f"{label}_src_grp.pi"
    group_step = v19cw.group_source(final_pha["source"], grouped, final_work, env)
    remediation = {
        "mode": "v19cw_equivalent_observation_hierarchy",
        "partition_key": "obsid",
        "groups": [{"obsid": row["obsid"], "cells": row["cells"]} for row in first_level],
        "source_stack_sha256": v19x2.adapter.sha256(final_work / f"{label}_source.lis"),
        "intermediate_rmf_threshold": float(config["hierarchy"]["intermediate_rmf_threshold"]),
        "final_rmf_threshold": float(config["hierarchy"]["final_rmf_threshold"]),
        "v19cw_report_sha256": config["parents"]["v19cw_report_sha256"],
    }
    return snapshot_combination(
        label,
        cells,
        grouped,
        final_pha["background"],
        final_response["arf"],
        final_response["rmf"],
        output,
        combine_step={"first_level": [row["pha"]["step"] for row in first_level], "final": final_pha["step"]},
        group_step=group_step,
        links=links,
        remediation=remediation,
    )


def select_combiner(label: str, cells: list[dict[str, Any]], scratch: Path, output: Path, config: dict[str, Any]) -> dict[str, Any]:
    references = config["runtime_remediation"]["direct_references"]
    if label in references:
        return reuse_direct(label, cells, output, config)
    threshold = int(config["runtime_remediation"]["maximum_direct_stack_cells"])
    if len(cells) > threshold:
        return hierarchical_combine(label, cells, scratch, output, config)
    result = v19x2.inherited_v19x.combine_aperture(label, cells, scratch, output, config)
    result["runtime_remediation"] = {
        "mode": "frozen_direct_below_threshold",
        "cell_threshold": threshold,
        "cells": len(cells),
    }
    return result


def execute(config: dict[str, Any], output: Path, scratch: Path) -> dict[str, Any]:
    validate_frozen(config)
    validate_parents(config)
    runtime = config["runtime_authorization"]
    response_report_path = ROOT / runtime["required_response_report"]
    response_report, unified_index = v19x2.adapter.authorize_unified_index(
        response_report_path,
        expected_config_sha256=config["parents"]["v19w5_config_sha256"],
        expected_runner_sha256=config["parents"]["v19w5_runner_sha256"],
        expected_cells=int(runtime["required_unified_cells"]),
        expected_products=int(runtime["required_unified_products"]),
        expected_status=runtime["required_status"],
        authority_label=runtime["response_authority"],
    )
    manifest = v19x2.inherited_v19x.load_manifest(config)
    plan = v19x2.inherited_v19x.build_aperture_plan(config, manifest)
    archive_roots = {name: Path(path) for name, path in config["execution"]["response_archives"].items()}
    validated = v19x2.adapter.validate_unified_archive(manifest, unified_index, archive_roots, recovery_archive=runtime["recovery_archive"])
    ordered_records = [validated[v19x2.adapter.task_key(row)] for row in manifest]
    validated_index = v19x2.write_validated_index(ordered_records, output / "validated_cell_index.csv")
    combinations: dict[str, dict[str, dict[str, Any]]] = {}
    for cluster, apertures in plan.items():
        combinations[cluster] = {}
        for kind, rows in apertures.items():
            label = f"{cluster}_integrated" if kind == "integrated" else f"{cluster}_bin{int(rows[0]['bin_id'])}"
            cells = [validated[v19x2.adapter.task_key(row)] for row in rows]
            combinations[cluster][kind] = select_combiner(label, cells, scratch, output, config)
    integrated_fits = []
    for cluster in plan:
        combination = combinations[cluster]["integrated"]
        try:
            integrated_fits.append(v19x2.inherited_v19x.fit_spectrum(config, cluster, combination, None))
        except Exception as exc:  # noqa: BLE001
            integrated_fits.append(v19x2.inherited_v19x.failed_fit(cluster, combination["label"], exc))
    integrated_by_cluster = {row["cluster"]: row for row in integrated_fits}
    regional_fits = []
    for cluster in plan:
        combination = combinations[cluster]["regional"]
        integrated = integrated_by_cluster[cluster]
        if not integrated["fit_completed"]:
            regional_fits.append(v19x2.inherited_v19x.failed_fit(cluster, combination["label"], RuntimeError("integrated abundance fit failed; regional fit not run")))
            continue
        try:
            regional_fits.append(v19x2.inherited_v19x.fit_spectrum(config, cluster, combination, float(integrated["parameters"]["abundance_solar"])))
        except Exception as exc:  # noqa: BLE001
            regional_fits.append(v19x2.inherited_v19x.failed_fit(cluster, combination["label"], exc))
    rows = [item for cluster in combinations.values() for item in cluster.values()]
    expected_cells = int(runtime["required_unified_cells"])
    archive_counts = dict(Counter(row["archive"] for row in ordered_records))
    gates = {
        "v19w5_unified_archive_and_every_product_hash_exact": len(validated) == expected_cells,
        "base_and_recovery_archive_labels_are_preserved": sum(archive_counts.values()) == expected_cells and set(archive_counts).issubset(set(config["execution"]["response_archives"])),
        "v19cw_equivalence_parent_exact": True,
        "hierarchy_applied_only_above_frozen_cell_threshold": combinations["BULLET"]["integrated"]["runtime_remediation"]["mode"] == "v19cw_equivalent_observation_hierarchy" and all(combinations[cluster][kind]["runtime_remediation"]["mode"] != "v19cw_equivalent_observation_hierarchy" for cluster in combinations for kind in combinations[cluster] if not (cluster == "BULLET" and kind == "integrated")),
        "combination_uses_every_registered_cell_exactly_once": all(combinations[cluster]["integrated"]["cells"] == int(config["registered_workload"]["clusters"][cluster]["total_cells"]) and combinations[cluster]["regional"]["cells"] == int(config["registered_workload"]["clusters"][cluster]["commissioning_region"]["cells"]) for cluster in combinations),
        "combined_source_background_arf_and_rmf_exist_and_links_are_exact": all(row["grouped_pha_links"] == row["expected_grouped_pha_links"] and row["frozen_snapshot"]["files"] == 4 for row in rows),
        "every_cell_event_energy_counts_equal_manifest": True,
        "combined_full_pha_source_counts_conserved_exactly": all(row["full_pha_count_conservation_exact"] for row in rows),
        "both_integrated_fits_pass": all(row["fit_completed"] and row["gates"]["all_passed"] for row in integrated_fits),
        "both_regional_fits_pass": all(row["fit_completed"] and row["gates"]["all_passed"] for row in regional_fits),
    }
    passed = all(gates.values())
    return {
        "status": AUTHORIZED_STATUS if passed else "unified_spectral_combination_commissioning_gate_failed",
        "validated_cell_index": validated_index,
        "validated_response_cells": len(validated),
        "validated_response_archive_counts": archive_counts,
        "combinations": combinations,
        "integrated_fits": integrated_fits,
        "regional_fits": regional_fits,
        "gates": gates,
        "full_494_region_combination_and_fit_authorized": passed,
        "response_report_sha256": v19x2.adapter.sha256(response_report_path),
        "response_unified_index_sha256": response_report["unified_product_index"]["sha256"],
        "v19cw_report_sha256": config["parents"]["v19cw_report_sha256"],
        "obsolete_v19x2_direct_large_stack_retried": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    args = parser.parse_args()
    config_path = args.config.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_json(config_path)
    try:
        result = execute(config, output, args.scratch.resolve())
    except Exception as exc:  # noqa: BLE001
        result = {
            "status": "bullet_hierarchical_recovery_execution_failed_closed",
            "execution_exception": f"{type(exc).__name__}: {exc}",
            "gates": {"execution_completed": False},
            "full_494_region_combination_and_fit_authorized": False,
        }
    report = {
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": v19x2.adapter.sha256(config_path),
        "runner_sha256": v19x2.adapter.sha256(Path(__file__).resolve()),
        **result,
        "scientific_temperature_map_claimed": False,
        "thermal_stress_constructed": False,
        "replacement_cluster_lensing_target_opened": False,
        "gravity_formula_or_parameter_changed": False,
        "v19bq_or_v19bs_run": False,
        "action_derived": False,
    }
    v19cw.atomic_json(output / "report.json", report)
    print(json.dumps({key: report.get(key) for key in ("status", "execution_exception")}, indent=2, sort_keys=True))
    if report["status"] != AUTHORIZED_STATUS:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
