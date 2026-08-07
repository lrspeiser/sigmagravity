#!/usr/bin/env python3
"""Run unchanged V19X2 commissioning with the validated direct OGIP writer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import run_sigma_v19di_direct_ogip_writer_preflight as v19di
import run_sigma_v19x2_unified_spectral_combination_commissioning as v19x2
from run_sigma_v19dh_direct_response_parity import direct_arrays, linked_paths
from sigma_v19di_direct_ogip import link_pha, write_arf, write_rmf

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_frozen(config: dict[str, Any], runner: Path) -> dict[str, Any]:
    if sha256(runner) != config["implementation"]["runner_sha256"]:
        raise RuntimeError("V19DJ runner changed after freeze")
    for name, item in config["parents"].items():
        path = ROOT / item["path"]
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"V19DJ parent changed: {name}")
    science = load_json(ROOT / config["parents"]["v19x2_config"]["path"])
    v19x2.validate_frozen_runner(science)
    v19x2.validate_frozen_parents_and_inheritance(science)
    for section, expected in config["inherited_section_sha256"].items():
        actual = hashlib.sha256(
            json.dumps(science[section], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if actual != expected:
            raise RuntimeError(f"V19DJ inherited science changed: {section}")
    return science


def combine_aperture(
    label: str,
    cells: list[dict[str, Any]],
    scratch: Path,
    output: Path,
    science: dict[str, Any],
    threshold: float,
) -> dict[str, Any]:
    work = scratch / label
    source_paths = [Path(row["source_pha"]) for row in cells]
    source = v19di.run_no_response_combine(source_paths, work / label)
    background = source.with_name(label + "_bkg.pi")
    arf = source.with_name(label + "_src.arf")
    rmf = source.with_name(label + "_src.rmf")
    arrays = direct_arrays(source_paths)
    template_arf, template_rmf, _ = linked_paths(source_paths[0])
    write_arf(
        template_arf,
        arf,
        arrays["energy_lo"],
        arrays["energy_hi"],
        arrays["arf"],
        float(arrays["exposure"]),
    )
    writer = write_rmf(
        template_rmf,
        rmf,
        arrays["energy_lo"],
        arrays["energy_hi"],
        arrays["rmf"],
        float(arrays["exposure"]),
        threshold,
    )
    link_pha(source, background, arf, rmf)
    expected_total = sum(int(row["source_pha_total_counts"]) for row in cells)
    combined_total = v19x2.inherited_v19x.pha_total_counts(source)
    if combined_total != expected_total:
        raise RuntimeError(
            f"V19DJ {label} PHA count mismatch: {combined_total} != {expected_total}"
        )

    grouped = work / f"{label}_src_grp.pi"
    grouping = science["combination"]["group_after_combination"]
    env = v19x2.inherited_v19x.inherited_spectra.isolated_environment(
        os.environ, work / "group_pfiles", work / "group_tmp"
    )
    command = [
        "dmgroup",
        f"infile={source}",
        f"outfile={grouped}",
        f"grouptype={grouping['grouptype']}",
        f"grouptypeval={int(grouping['minimum_counts'])}",
        "binspec=",
        f"xcolumn={grouping['xcolumn']}",
        f"ycolumn={grouping['ycolumn']}",
        "tabspec=",
        "tabcolumn=",
        "stopspec=",
        "stopcolumn=",
        "clobber=no",
        "verbose=1",
        "mode=h",
    ]
    group_step = v19x2.inherited_v19x.inherited_spectra.run_step(
        command, work / "dmgroup.log", [grouped], env
    )
    links = v19x2.inherited_v19x.pha_links(grouped, env)
    expected_links = {
        "BACKFILE": background.name,
        "ANCRFILE": arf.name,
        "RESPFILE": rmf.name,
    }
    if links != expected_links:
        raise RuntimeError(f"V19DJ {label} grouped PHA links changed: {links}")

    snapshot_root = output / "frozen_products" / label
    snapshots = []
    for role, product in (
        ("grouped_source_spectrum", grouped),
        ("background_spectrum", background),
        ("source_arf", arf),
        ("source_rmf", rmf),
    ):
        item = v19x2.inherited_v19x.inherited_spectra.copy_snapshot(
            product, snapshot_root / product.name
        )
        item["role"] = role
        snapshots.append(item)
    return {
        "label": label,
        "cells": len(cells),
        "event_energy_source_counts_0p5_7_keV": sum(
            int(row["source_band_events"]) for row in cells
        ),
        "event_energy_background_counts_0p5_7_keV": sum(
            int(row["background_band_events"]) for row in cells
        ),
        "expected_full_pha_source_counts": expected_total,
        "combined_full_pha_source_counts": combined_total,
        "full_pha_count_conservation_exact": combined_total == expected_total,
        "writer": writer,
        "group_step": group_step,
        "grouped_pha_links": links,
        "expected_grouped_pha_links": expected_links,
        "frozen_snapshot": {
            "files": len(snapshots),
            "bytes": sum(int(item["bytes"]) for item in snapshots),
            "products": snapshots,
        },
    }


def execute(
    config: dict[str, Any], science: dict[str, Any], output: Path, scratch: Path
) -> dict[str, Any]:
    runtime = science["runtime_authorization"]
    response_report_path = ROOT / runtime["required_response_report"]
    response_report, unified_index = v19x2.adapter.authorize_unified_index(
        response_report_path,
        expected_config_sha256=science["parents"]["v19w5_config_sha256"],
        expected_runner_sha256=science["parents"]["v19w5_runner_sha256"],
        expected_cells=int(runtime["required_unified_cells"]),
        expected_products=int(runtime["required_unified_products"]),
        expected_status=runtime["required_status"],
        authority_label=runtime["response_authority"],
    )
    manifest = v19x2.inherited_v19x.load_manifest(science)
    plan = v19x2.inherited_v19x.build_aperture_plan(science, manifest)
    archives = {
        name: Path(path) for name, path in science["execution"]["response_archives"].items()
    }
    validated = v19x2.adapter.validate_unified_archive(
        manifest,
        unified_index,
        archives,
        recovery_archive=runtime["recovery_archive"],
    )
    ordered = [validated[v19x2.adapter.task_key(row)] for row in manifest]
    validated_index = v19x2.write_validated_index(
        ordered, output / "validated_cell_index.csv"
    )
    combinations: dict[str, dict[str, dict[str, Any]]] = {}
    for cluster, apertures in plan.items():
        combinations[cluster] = {}
        for kind, rows in apertures.items():
            label = (
                f"{cluster}_integrated"
                if kind == "integrated"
                else f"{cluster}_bin{int(rows[0]['bin_id'])}"
            )
            cells = [validated[v19x2.adapter.task_key(row)] for row in rows]
            combinations[cluster][kind] = combine_aperture(
                label,
                cells,
                scratch,
                output,
                science,
                float(config["writer"]["rmf_threshold"]),
            )

    integrated_fits = []
    for cluster in plan:
        combination = combinations[cluster]["integrated"]
        try:
            integrated_fits.append(
                v19x2.inherited_v19x.fit_spectrum(science, cluster, combination, None)
            )
        except Exception as exc:  # noqa: BLE001 - retain attempted fit
            integrated_fits.append(
                v19x2.inherited_v19x.failed_fit(cluster, combination["label"], exc)
            )
    integrated_by_cluster = {row["cluster"]: row for row in integrated_fits}
    regional_fits = []
    for cluster in plan:
        combination = combinations[cluster]["regional"]
        integrated = integrated_by_cluster[cluster]
        if not integrated["fit_completed"]:
            regional_fits.append(
                v19x2.inherited_v19x.failed_fit(
                    cluster,
                    combination["label"],
                    RuntimeError("integrated abundance fit failed; regional fit not run"),
                )
            )
            continue
        try:
            regional_fits.append(
                v19x2.inherited_v19x.fit_spectrum(
                    science,
                    cluster,
                    combination,
                    float(integrated["parameters"]["abundance_solar"]),
                )
            )
        except Exception as exc:  # noqa: BLE001 - retain attempted fit
            regional_fits.append(
                v19x2.inherited_v19x.failed_fit(cluster, combination["label"], exc)
            )

    combination_rows = [item for group in combinations.values() for item in group.values()]
    expected_cells = int(runtime["required_unified_cells"])
    gates = {
        "v19w5_unified_archive_and_every_product_hash_exact": len(validated)
        == expected_cells,
        "combination_uses_every_registered_cell_exactly_once": all(
            combinations[cluster]["integrated"]["cells"]
            == int(science["registered_workload"]["clusters"][cluster]["total_cells"])
            and combinations[cluster]["regional"]["cells"]
            == int(
                science["registered_workload"]["clusters"][cluster][
                    "commissioning_region"
                ]["cells"]
            )
            for cluster in combinations
        ),
        "combined_source_background_arf_and_rmf_exist_and_links_are_exact": all(
            row["grouped_pha_links"] == row["expected_grouped_pha_links"]
            and row["frozen_snapshot"]["files"] == 4
            for row in combination_rows
        ),
        "every_cell_event_energy_counts_equal_manifest": True,
        "combined_full_pha_source_counts_conserved_exactly": all(
            row["full_pha_count_conservation_exact"] for row in combination_rows
        ),
        "both_integrated_fits_pass": all(
            row["fit_completed"] and row["gates"]["all_passed"]
            for row in integrated_fits
        ),
        "both_regional_fits_pass": all(
            row["fit_completed"] and row["gates"]["all_passed"]
            for row in regional_fits
        ),
    }
    passed = all(gates.values())
    return {
        "status": (
            "direct_response_commissioning_passed_full_regional_fits_authorized"
            if passed
            else "direct_response_commissioning_gate_failed"
        ),
        "response_report_sha256": sha256(response_report_path),
        "response_unified_index_sha256": response_report["unified_product_index"][
            "sha256"
        ],
        "validated_cell_index": validated_index,
        "validated_response_cells": len(validated),
        "combinations": combinations,
        "integrated_fits": integrated_fits,
        "regional_fits": regional_fits,
        "gates": gates,
        "full_494_region_combination_and_fit_authorized": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        config = load_json(config_path)
        science = validate_frozen(config, Path(__file__).resolve())
        if args.scratch.exists():
            raise RuntimeError(f"V19DJ scratch must not already exist: {args.scratch}")
        args.scratch.mkdir(parents=True)
        result = execute(config, science, output, args.scratch.resolve())
    except Exception as exc:  # noqa: BLE001 - preserve terminal evidence
        result = {
            "status": "direct_response_commissioning_execution_failed",
            "execution_exception": f"{type(exc).__name__}: {exc}",
            "gates": {"execution_completed": False},
            "full_494_region_combination_and_fit_authorized": False,
        }
    report = {
        "protocol_version": "SIGMA-V19DJ-DIRECT-RESPONSE-COMMISSIONING-1.0.0",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        **result,
        "thermal_stress_constructed": False,
        "lensing_halo_action_gravity_or_holdout_payload_opened": False,
        "gravity_formula_or_parameter_changed": False,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report_path)
    print(f"status: {report['status']}")
    if not report["full_494_region_combination_and_fit_authorized"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
