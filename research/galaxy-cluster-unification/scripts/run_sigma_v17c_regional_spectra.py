#!/usr/bin/env python3
"""Extract and combine every frozen regional Sigma v17C Chandra spectrum.

This stage is deliberately authorization-gated by the independently written
integrated-temperature report.  Creating this runner does not authorize its
execution: both cluster-wide fits must pass the frozen v17C gates first.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_sigma_v17c_integrated_spectra import (
    DEFAULT_ASTROMETRY,
    DEFAULT_CLEANING,
    DEFAULT_CONFIG,
    DEFAULT_HI4PI,
    DEFAULT_REDUCTION,
    DEFAULT_REGIONS,
    DEFAULT_REPRO,
    DEFAULT_RESTORATION,
    DEFAULT_SCRATCH,
    DEFAULT_VISUAL_AUDIT,
    ROOT,
    celestial_coordinate_chip,
    event_count,
    event_reference_coordinate,
    find_product,
    frozen_region_files,
    isolated_environment,
    prepare_background_geometry,
    prepare_translated_fov,
    run_step,
    sha256,
    verify_blanksky_scaling,
)

DEFAULT_INTEGRATED_SPECTRA = (
    ROOT / "results" / "sigma_v17c_integrated_spectra" / "report.json"
)
DEFAULT_INTEGRATED_TEMPERATURES = (
    ROOT / "results" / "sigma_v17c_integrated_temperatures" / "report.json"
)
DEFAULT_RESPONSE_SUPPORT = (
    ROOT / "configs" / "sigma_v17c_regional_response_support.json"
)
DEFAULT_RESPONSE_SUPPORT_REPORT = (
    ROOT / "results" / "sigma_v17c_regional_response_support" / "report.json"
)
DEFAULT_RUNTIME_RESPONSE_SUPPORT = (
    ROOT / "configs" / "sigma_v17c_regional_runtime_response_support.json"
)
DEFAULT_RUNTIME_RESPONSE_SUPPORT_REPORT = (
    ROOT / "results" / "sigma_v17c_regional_runtime_response_support" / "report.json"
)
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v17c_regional_spectra"
REGIONAL_RUNTIME_TMP_ROOT = Path("/tmp/sv17c")
MAX_REGIONAL_RUNTIME_TMP_BYTES = 64

OFF_CCD_RESPONSE_REASON = "event_mean_response_reference_maps_off_selected_ccd"
MISSING_CCD_SUPPORT_REASON = (
    "event_mean_response_reference_lacks_source_background_ccd_support"
)
EMPTY_CALIBRATED_RESPONSE_REASON = (
    "ciao_empty_calibrated_response_domain_after_valid_spectra"
)


def response_support_skip_marker(task: dict[str, Any]) -> Path:
    return task["log"].with_suffix(".response_support_skip.json")


def load_response_support_skip(task: dict[str, Any]) -> dict[str, Any] | None:
    marker = response_support_skip_marker(task)
    if not marker.is_file():
        return None
    record = json.loads(marker.read_text(encoding="utf-8"))
    identity = {
        "cluster": task["cluster"],
        "region_id": int(task["region_id"]),
        "obsid": int(task["obsid"]),
        "ccd_id": int(task["ccd_id"]),
    }
    if any(record.get(key) != value for key, value in identity.items()):
        raise RuntimeError(f"response-support marker identity changed: {marker}")
    if record.get("reason") != EMPTY_CALIBRATED_RESPONSE_REASON:
        raise RuntimeError(f"response-support marker reason changed: {marker}")
    for product in record.get("quarantined_partial_products", []):
        path = Path(product["path"])
        if not path.is_file() or path.stat().st_size != product["bytes"]:
            raise RuntimeError(f"quarantined response-support product changed: {path}")
        if sha256(path) != product["sha256"]:
            raise RuntimeError(f"quarantined response-support hash changed: {path}")
    if any(
        path.exists()
        for path in (
            task["source_pha"],
            task["background_pha"],
            task["arf"],
            task["rmf"],
        )
    ):
        raise RuntimeError(f"response-support marker has live extraction products: {marker}")
    return {
        **record,
        "marker": str(marker),
        "marker_sha256": sha256(marker),
        "reused": True,
    }


def classify_and_quarantine_empty_response_support(
    task: dict[str, Any],
) -> dict[str, Any] | None:
    products = {
        "source_pha": task["source_pha"],
        "background_pha": task["background_pha"],
        "arf": task["arf"],
        "rmf": task["rmf"],
    }
    signature = {
        name: path.is_file() and path.stat().st_size > 0
        for name, path in products.items()
    }
    log = task["log"]
    log_text = log.read_text(encoding="utf-8") if log.is_file() else ""
    required_fragments = (
        "Extracting src spectra",
        "Extracting bkg spectra",
        "ERROR max() iterable argument is empty",
    )
    matches = (
        signature
        == {
            "source_pha": True,
            "background_pha": True,
            "arf": False,
            "rmf": False,
        }
        and all(fragment in log_text for fragment in required_fragments)
    )
    if not matches:
        return None
    if task.get("allow_empty_calibrated_response_skip") is not True:
        raise RuntimeError("empty calibrated-response skip is not frozen")
    quarantine = task["source_pha"].parent.parent / "response_support_quarantine"
    quarantine.mkdir(parents=True, exist_ok=True)
    quarantined = []
    for name in ("source_pha", "background_pha"):
        source = products[name]
        destination = quarantine / source.name
        if destination.exists():
            raise RuntimeError(f"response-support quarantine collision: {destination}")
        source.replace(destination)
        quarantined.append(
            {
                "kind": name,
                "path": str(destination),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )
    record = {
        "cluster": task["cluster"],
        "region_id": int(task["region_id"]),
        "obsid": int(task["obsid"]),
        "ccd_id": int(task["ccd_id"]),
        "reason": EMPTY_CALIBRATED_RESPONSE_REASON,
        "source_band_events": int(task["source_band_events"]),
        "background_band_events": int(task["background_band_events"]),
        "response_reference": task["response_reference"],
        "product_signature": signature,
        "required_log_fragments": list(required_fragments),
        "log": str(log),
        "log_sha256": sha256(log),
        "quarantined_partial_products": quarantined,
        "reused": False,
    }
    marker = response_support_skip_marker(task)
    record["marker"] = str(marker)
    marker.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**record, "marker_sha256": sha256(marker)}


def region_id(path: Path) -> int:
    return int(path.stem.rsplit("_", maxsplit=1)[1])


def copy_snapshot(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256(source)
    if destination.exists():
        if sha256(destination) != digest:
            raise RuntimeError(f"existing regional snapshot changed: {destination}")
        reused = True
    else:
        shutil.copy2(source, destination)
        reused = False
    return {
        "relative_path": destination.relative_to(ROOT).as_posix(),
        "bytes": destination.stat().st_size,
        "sha256": digest,
        "reused": reused,
    }


def execute_regional_cell(task: dict, scratch: Path, namespace: str) -> dict[str, Any]:
    obsid = int(task["obsid"])
    chip = int(task["ccd_id"])
    rid = int(task["region_id"])
    runtime_tmp = (
        REGIONAL_RUNTIME_TMP_ROOT
        / namespace
        / task["cluster"]
        / f"r{rid:03d}"
        / f"{obsid}c{chip}"
    )
    if len(os.fsencode(runtime_tmp)) > MAX_REGIONAL_RUNTIME_TMP_BYTES:
        raise RuntimeError(f"regional runtime path is not AF_UNIX-safe: {runtime_tmp}")
    existing_skip = load_response_support_skip(task)
    if existing_skip is not None:
        return {"response_support_skip": existing_skip}
    env = isolated_environment(
        os.environ,
        scratch
        / f"pfiles_{namespace}"
        / "regional"
        / task["cluster"]
        / f"region_{rid:03d}"
        / f"{obsid}_ccd{chip}",
        runtime_tmp,
    )
    try:
        step = run_step(
            task["command"],
            task["log"],
            [task["source_pha"], task["background_pha"], task["arf"], task["rmf"]],
            env,
        )
    except RuntimeError:
        skip = classify_and_quarantine_empty_response_support(task)
        if skip is None:
            raise
        return {"response_support_skip": skip}
    scaling = verify_blanksky_scaling(
        task["source_pha"],
        task["background_pha"],
        float(task["bkgscale_value"]),
        env,
    )
    return {
        "cluster": task["cluster"],
        "region_id": rid,
        "obsid": obsid,
        "ccd_id": chip,
        "source_band_events": int(task["source_band_events"]),
        "background_band_events": int(task["background_band_events"]),
        "response_reference": task["response_reference"],
        "source_spectrum": str(task["source_pha"]),
        "source_spectrum_sha256": sha256(task["source_pha"]),
        "background_spectrum": str(task["background_pha"]),
        "background_spectrum_sha256": sha256(task["background_pha"]),
        "arf_sha256": sha256(task["arf"]),
        "rmf_sha256": sha256(task["rmf"]),
        "blanksky_scaling": scaling,
        "translated_fov": task["translated_fov"],
        "runtime_tmp": str(runtime_tmp),
        "step": step,
    }


def observation_contexts(
    cluster_name: str,
    config: dict,
    region_row: dict,
    astrometry_rows: dict[int, dict],
    repro_rows: dict[int, dict],
    cleaning_rows: dict[int, dict],
    restoration: dict,
    work: Path,
    env: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    contexts = []
    geometry_records = []
    for obsid in config["clusters"][cluster_name]["obsids"]:
        astrometry_row = astrometry_rows[obsid]
        repro_row = repro_rows[obsid]
        cleaning_row = cleaning_rows[obsid]
        observation = next(
            item for item in region_row["observation_steps"] if int(item["obsid"]) == obsid
        )
        science = Path(observation["science_reprojected"])
        background = Path(observation["blanksky_reprojected"])
        aspect = Path(astrometry_row["application"]["corrected_aspect_list"])
        corrected_background = (
            work / "background_geometry" / f"acisf{obsid}_blanksky_geometry.fits"
        )
        geometry_record = prepare_background_geometry(
            science,
            background,
            corrected_background,
            env,
        )
        geometry_records.append(geometry_record)
        mask = find_product(repro_row, "_msk1.fits")
        badpix = find_product(repro_row, "_repro_bpix1.fits")
        source_fov = find_product(repro_row, "_repro_fov1.fits")
        expected_fov_hash = next(
            item["sha256"]
            for item in repro_row["products"]
            if item["relative_path"].endswith("_repro_fov1.fits")
        )
        if obsid == 12260 and sha256(source_fov) != expected_fov_hash:
            if sha256(source_fov) != restoration["restored_sha256"]:
                raise RuntimeError("documented ObsID 12260 FOV restoration hash mismatch")
            expected_fov_hash = restoration["restored_sha256"]
        translated_fov = work / "fov" / f"acisf{obsid}_gaia_fov1.fits"
        fov_record = prepare_translated_fov(
            source_fov,
            translated_fov,
            astrometry_row,
            expected_fov_hash,
            env,
        )
        contexts.append(
            {
                "obsid": obsid,
                "science": science,
                "background": corrected_background,
                "aspect": aspect,
                "mask": mask,
                "badpix": badpix,
                "translated_fov": translated_fov,
                "translated_fov_record": fov_record,
                "blanksky_scaling": cleaning_row["blanksky_scaling"],
            }
        )
    return contexts, geometry_records


def plan_region(
    cluster_name: str,
    path: Path,
    contexts: list[dict[str, Any]],
    response_support: dict[str, Any],
    work: Path,
    env: dict[str, str],
    runtime_response_support: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rid = region_id(path)
    region_work = work / f"region_{rid:03d}"
    individual = region_work / "individual"
    logs = region_work / "logs"
    individual.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    tasks = []
    skipped = []
    allowed_response_skips = set(
        response_support["admission_rule"]["allowed_calibrated_response_skip_reasons"]
    )
    allow_empty_response_skip = (
        runtime_response_support is not None
        and runtime_response_support["admission_rule"]["allowed_reason"]
        == EMPTY_CALIBRATED_RESPONSE_REASON
    )
    for context in contexts:
        obsid = int(context["obsid"])
        science = context["science"]
        background = context["background"]
        translated_fov = context["translated_fov"]
        for key, bkgscale_value in sorted(context["blanksky_scaling"].items()):
            chip = int(key.removeprefix("BKGSCAL"))
            source_filter = (
                f"{science}[ccd_id={chip}]"
                f"[sky=region({translated_fov})][sky=region({path})]"
            )
            background_filter = (
                f"{background}[ccd_id={chip}]"
                f"[sky=region({translated_fov})][sky=region({path})]"
            )
            source_events = event_count(source_filter + "[energy=500:7000]", env)
            background_events = event_count(background_filter + "[energy=500:7000]", env)
            if source_events == 0:
                skipped.append(
                    {
                        "region_id": rid,
                        "obsid": obsid,
                        "ccd_id": chip,
                        "reason": "zero_source_band_events_after_frozen_filters",
                        "background_band_events": background_events,
                    }
                )
                continue
            response_reference = event_reference_coordinate(source_filter, science, env)
            if response_reference["events"] != source_events:
                raise RuntimeError(
                    f"response-reference event count mismatch for {cluster_name} "
                    f"region {rid} ObsID {obsid} CCD {chip}"
                )
            if response_reference["dmcoords_chip_id"] != chip:
                if OFF_CCD_RESPONSE_REASON not in allowed_response_skips:
                    raise RuntimeError("off-CCD response skip is not frozen")
                skipped.append(
                    {
                        "region_id": rid,
                        "obsid": obsid,
                        "ccd_id": chip,
                        "reason": OFF_CCD_RESPONSE_REASON,
                        "source_band_events": source_events,
                        "background_band_events": background_events,
                        "response_reference": response_reference,
                    }
                )
                continue
            source_chip = celestial_coordinate_chip(
                science,
                context["aspect"],
                response_reference["ra_deg"],
                response_reference["dec_deg"],
                env,
            )
            background_chip = celestial_coordinate_chip(
                background,
                context["aspect"],
                response_reference["ra_deg"],
                response_reference["dec_deg"],
                env,
            )
            if source_chip != chip or background_chip != chip:
                if MISSING_CCD_SUPPORT_REASON not in allowed_response_skips:
                    raise RuntimeError("source/background CCD support skip is not frozen")
                skipped.append(
                    {
                        "region_id": rid,
                        "obsid": obsid,
                        "ccd_id": chip,
                        "reason": MISSING_CCD_SUPPORT_REASON,
                        "source_band_events": source_events,
                        "background_band_events": background_events,
                        "response_reference": response_reference,
                        "science_aspect_chip_id": source_chip,
                        "background_aspect_chip_id": background_chip,
                    }
                )
                continue
            response_reference["science_aspect_chip_id"] = source_chip
            response_reference["background_aspect_chip_id"] = background_chip
            outroot = individual / f"acisf{obsid}_ccd{chip}_region{rid:03d}"
            source_pha = outroot.with_suffix(".pi")
            background_pha = outroot.with_name(outroot.name + "_bkg.pi")
            arf = outroot.with_suffix(".arf")
            rmf = outroot.with_suffix(".rmf")
            command = [
                "specextract",
                f"infile={source_filter}",
                f"outroot={outroot}",
                f"bkgfile={background_filter}",
                f"asp=@{context['aspect']}",
                f"mskfile={context['mask']}",
                f"badpixfile={context['badpix']}",
                "dafile=CALDB",
                "bkgresp=no",
                "weight=yes",
                "weight_rmf=yes",
                "resp_pos=CENTROID",
                f"refcoord={response_reference['ra_deg']:.14f},{response_reference['dec_deg']:.14f}",
                "correctpsf=no",
                "combine=no",
                "grouptype=NONE",
                "binspec=NONE",
                "bkg_grouptype=NONE",
                "bkg_binspec=NONE",
                "energy=0.3:11.0:0.01",
                "energy_wmap=500:7000",
                "binwmap=det=8",
                "binarfwmap=1",
                "parallel=no",
                "nproc=1",
                "clobber=no",
                "verbose=1",
                "mode=h",
            ]
            tasks.append(
                {
                    "cluster": cluster_name,
                    "region_id": rid,
                    "region_path": str(path),
                    "obsid": obsid,
                    "ccd_id": chip,
                    "source_band_events": source_events,
                    "background_band_events": background_events,
                    "response_reference": response_reference,
                    "source_pha": source_pha,
                    "background_pha": background_pha,
                    "arf": arf,
                    "rmf": rmf,
                    "bkgscale_value": float(bkgscale_value),
                    "translated_fov": context["translated_fov_record"],
                    "allow_empty_calibrated_response_skip": allow_empty_response_skip,
                    "command": command,
                    "log": logs / f"{obsid}_ccd{chip}_specextract.log",
                }
            )
    return tasks, skipped


def combine_region(
    cluster_name: str,
    path: Path,
    extracted: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    work: Path,
    output: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    rid = region_id(path)
    region_work = work / f"region_{rid:03d}"
    logs = region_work / "logs"
    source_spectra = [Path(row["source_spectrum"]) for row in extracted]
    if not source_spectra:
        raise RuntimeError(f"{cluster_name} region {rid} produced no spectra")
    stack = region_work / "source_spectra.lis"
    content = "\n".join(str(item) for item in source_spectra) + "\n"
    if stack.exists() and stack.read_text(encoding="utf-8") != content:
        raise RuntimeError(f"existing regional stack differs: {stack}")
    stack.write_text(content, encoding="utf-8")
    combined_root = region_work / f"{cluster_name}_region_{rid:03d}"
    combined_source = combined_root.with_name(combined_root.name + "_src.pi")
    combined_background = combined_root.with_name(combined_root.name + "_bkg.pi")
    combined_arf = combined_root.with_name(combined_root.name + "_src.arf")
    combined_rmf = combined_root.with_name(combined_root.name + "_src.rmf")
    combine_command = [
        "combine_spectra",
        f"src_spectra=@{stack}",
        f"outroot={combined_root}",
        "method=sum",
        "bscale_method=asca",
        "exp_origin=pha",
        "clobber=no",
        "verbose=1",
        "mode=h",
    ]
    combine_step = run_step(
        combine_command,
        logs / "combine_spectra.log",
        [combined_source, combined_background, combined_arf, combined_rmf],
        env,
    )
    grouped = region_work / f"{cluster_name}_region_{rid:03d}_src_grp.pi"
    group_command = [
        "dmgroup",
        f"infile={combined_source}",
        f"outfile={grouped}",
        "grouptype=NUM_CTS",
        "grouptypeval=25",
        "binspec=",
        "xcolumn=CHANNEL",
        "ycolumn=COUNTS",
        "tabspec=",
        "tabcolumn=",
        "stopspec=",
        "stopcolumn=",
        "clobber=no",
        "verbose=1",
        "mode=h",
    ]
    group_step = run_step(group_command, logs / "dmgroup.log", [grouped], env)
    snapshot_root = (
        output / "frozen_regional_spectra" / cluster_name / f"region_{rid:03d}"
    )
    snapshots = []
    for role, source in (
        ("grouped_source_spectrum", grouped),
        ("background_spectrum", combined_background),
        ("source_arf", combined_arf),
        ("source_rmf", combined_rmf),
        ("spectral_region", path),
    ):
        record = copy_snapshot(source, snapshot_root / source.name)
        record["role"] = role
        snapshots.append(record)
    return {
        "region_id": rid,
        "source_region": str(path),
        "source_region_sha256": sha256(path),
        "extracted_cells": len(extracted),
        "skipped_cells": skipped,
        "source_band_events": sum(row["source_band_events"] for row in extracted),
        "background_band_events": sum(
            row["background_band_events"] for row in extracted
        ),
        "extractions": extracted,
        "combined": {
            "source_spectra": len(source_spectra),
            "stack": str(stack),
            "combine_step": combine_step,
            "group_step": group_step,
        },
        "frozen_snapshot": {
            "files": len(snapshots),
            "bytes": sum(item["bytes"] for item in snapshots),
            "products": snapshots,
        },
    }


def build_cluster(
    cluster_name: str,
    config: dict,
    response_support: dict[str, Any],
    runtime_response_support: dict[str, Any],
    region_row: dict,
    astrometry_rows: dict[int, dict],
    repro_rows: dict[int, dict],
    cleaning_rows: dict[int, dict],
    restoration: dict,
    scratch: Path,
    output: Path,
) -> dict[str, Any]:
    namespace = config["execution"]["work_namespace"]
    work = scratch / namespace / "regional" / cluster_name
    work.mkdir(parents=True, exist_ok=True)
    planning_env = isolated_environment(
        os.environ,
        scratch / f"pfiles_{namespace}" / "planning" / "regional" / cluster_name,
        scratch / f"tmp_{namespace}" / "planning" / "regional" / cluster_name,
    )
    contexts, geometry_records = observation_contexts(
        cluster_name,
        config,
        region_row,
        astrometry_rows,
        repro_rows,
        cleaning_rows,
        restoration,
        work,
        planning_env,
    )
    paths = frozen_region_files(region_row)
    all_tasks = []
    skipped_by_region: dict[int, list[dict[str, Any]]] = {}
    for index, path in enumerate(paths, start=1):
        tasks, skipped = plan_region(
            cluster_name,
            path,
            contexts,
            response_support,
            work,
            planning_env,
            runtime_response_support,
        )
        all_tasks.extend(tasks)
        skipped_by_region[region_id(path)] = skipped
        print(
            f"{cluster_name}: planned region {index}/{len(paths)} "
            f"with {len(tasks)} extraction cells",
            flush=True,
        )
    worker_limit = int(config["execution"]["external_parallel_cells"])
    workers = min(worker_limit, len(all_tasks))
    if workers < 1:
        raise RuntimeError(f"{cluster_name} produced no regional extraction tasks")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(execute_regional_cell, task, scratch, namespace)
            for task in all_tasks
        ]
        outcomes = [future.result() for future in futures]
    extracted = []
    for outcome in outcomes:
        if "response_support_skip" in outcome:
            skip = outcome["response_support_skip"]
            skipped_by_region.setdefault(int(skip["region_id"]), []).append(skip)
        else:
            extracted.append(outcome)
    extracted_by_region: dict[int, list[dict[str, Any]]] = {}
    for row in extracted:
        extracted_by_region.setdefault(int(row["region_id"]), []).append(row)
    combine_env = isolated_environment(
        os.environ,
        scratch / f"pfiles_{namespace}" / "combine" / "regional" / cluster_name,
        scratch / f"tmp_{namespace}" / "combine" / "regional" / cluster_name,
    )
    region_results = []
    for index, path in enumerate(paths, start=1):
        rid = region_id(path)
        result = combine_region(
            cluster_name,
            path,
            extracted_by_region.get(rid, []),
            skipped_by_region[rid],
            work,
            output,
            combine_env,
        )
        region_results.append(result)
        print(f"{cluster_name}: combined region {index}/{len(paths)}", flush=True)
    return {
        "cluster": cluster_name,
        "frozen_regions": len(paths),
        "planned_extraction_cells": len(all_tasks),
        "extracted_cells": len(extracted),
        "skipped_cells": sum(len(items) for items in skipped_by_region.values()),
        "background_geometry": geometry_records,
        "execution": {
            "work_namespace": namespace,
            "external_parallel_cells": worker_limit,
        },
        "regions": region_results,
    }


def validate_authorization(
    config_path: Path,
    config: dict,
    integrated_spectra_path: Path,
    integrated_spectra: dict,
    integrated_temperatures_path: Path,
    integrated_temperatures: dict,
) -> None:
    config_hash = sha256(config_path)
    if integrated_spectra["status"] != (
        "both_frozen_integrated_spectra_extracted_combined_and_grouped"
    ):
        raise RuntimeError("integrated spectral extraction is incomplete")
    if integrated_temperatures["status"] != "both_integrated_temperature_gates_passed":
        raise RuntimeError("both integrated temperature gates have not passed")
    if integrated_temperatures.get("regional_fit_authorized") is not True:
        raise RuntimeError("regional spectral extraction is not authorized")
    if integrated_spectra["protocol_version"] != config["protocol_version"]:
        raise RuntimeError("integrated spectra and regional protocols differ")
    if integrated_temperatures["protocol_version"] != config["protocol_version"]:
        raise RuntimeError("integrated temperatures and regional protocols differ")
    if integrated_spectra["config_sha256"] != config_hash:
        raise RuntimeError("config changed after integrated spectral extraction")
    if integrated_temperatures["config_sha256"] != config_hash:
        raise RuntimeError("config changed after integrated temperature fitting")
    if integrated_temperatures["spectral_extraction_report_sha256"] != sha256(
        integrated_spectra_path
    ):
        raise RuntimeError("integrated temperature fit used another extraction report")
    if integrated_temperatures_path == integrated_spectra_path:
        raise RuntimeError("authorization reports unexpectedly share a path")


def validate_response_support(
    support_path: Path,
    support: dict[str, Any],
    config_path: Path,
    integrated_spectra_path: Path,
    integrated_spectra: dict[str, Any],
    integrated_temperatures_path: Path,
    integrated_temperatures: dict[str, Any],
) -> None:
    expected_status = (
        "frozen after both integrated-temperature gates passed and before any "
        "regional spectrum, regional temperature, thermal-stress map, v17 inverse "
        "coefficient, or v17 lensing score existed"
    )
    if support.get("status") != expected_status or not support_path.is_file():
        raise RuntimeError("regional response-support protocol is not frozen")
    parents = support["parents"]
    checks = (
        ("spectral_protocol_sha256", config_path),
        ("integrated_spectra_report_sha256", integrated_spectra_path),
        ("integrated_temperatures_report_sha256", integrated_temperatures_path),
    )
    for key, path in checks:
        if parents.get(key) != sha256(path):
            raise RuntimeError(f"regional response-support parent changed: {key}")
    required = support["required_upstream_state"]
    if integrated_spectra.get("status") != required["integrated_spectra_status"]:
        raise RuntimeError("response-support protocol lacks integrated spectra")
    if integrated_temperatures.get("status") != required[
        "integrated_temperatures_status"
    ]:
        raise RuntimeError("response-support protocol lacks integrated temperatures")
    if integrated_temperatures.get("regional_fit_authorized") is not required[
        "regional_fit_authorized"
    ]:
        raise RuntimeError("response-support protocol is not authorized")
    if integrated_temperatures.get("lensing_target_opened") is not False:
        raise RuntimeError("response-support protocol was frozen after target access")
    expected_reasons = {OFF_CCD_RESPONSE_REASON, MISSING_CCD_SUPPORT_REASON}
    reasons = set(
        support["admission_rule"]["allowed_calibrated_response_skip_reasons"]
    )
    if reasons != expected_reasons:
        raise RuntimeError("regional response-support reasons changed")
    integrity = support["integrity"]
    if any(
        integrity[key]
        for key in (
            "regional_spectrum_existed_at_freeze",
            "regional_temperature_existed_at_freeze",
            "thermal_stress_constructed_at_freeze",
            "lensing_target_opened",
            "scientific_threshold_changed",
            "gravity_parameter_changed",
            "core_v17c_protocol_changed",
        )
    ):
        raise RuntimeError("regional response-support integrity boundary changed")


def validate_runtime_response_support(
    runtime_path: Path,
    runtime_support: dict[str, Any],
    runtime_report_path: Path,
    runtime_report: dict[str, Any],
    config_path: Path,
    integrated_spectra_path: Path,
    integrated_spectra: dict[str, Any],
    integrated_temperatures_path: Path,
    integrated_temperatures: dict[str, Any],
    preflight_path: Path,
    preflight_report_path: Path,
) -> None:
    expected_status = (
        "frozen after 193 valid AS295 extraction cells and one response-only "
        "failure, but before any regional spectrum report, regional temperature, "
        "thermal-stress map, inverse coefficient, or lensing target access"
    )
    if runtime_support.get("status") != expected_status or not runtime_path.is_file():
        raise RuntimeError("regional runtime response-support protocol is not frozen")
    parents = runtime_support["parents"]
    checks = (
        ("spectral_protocol_sha256", config_path),
        ("integrated_spectra_report_sha256", integrated_spectra_path),
        ("integrated_temperatures_report_sha256", integrated_temperatures_path),
        ("preflight_response_support_sha256", preflight_path),
        ("preflight_response_support_report_sha256", preflight_report_path),
    )
    for key, path in checks:
        if parents.get(key) != sha256(path):
            raise RuntimeError(f"runtime response-support parent changed: {key}")
    required = runtime_support["required_upstream_state"]
    if integrated_spectra.get("status") != required["integrated_spectra_status"]:
        raise RuntimeError("runtime response support lacks integrated spectra")
    if integrated_temperatures.get("status") != required[
        "integrated_temperatures_status"
    ]:
        raise RuntimeError("runtime response support lacks integrated temperatures")
    if integrated_temperatures.get("regional_fit_authorized") is not required[
        "regional_fit_authorized"
    ]:
        raise RuntimeError("runtime response support is not authorized")
    if integrated_temperatures.get("lensing_target_opened") is not False:
        raise RuntimeError("runtime response support was frozen after target access")
    rule = runtime_support["admission_rule"]
    if rule.get("allowed_reason") != EMPTY_CALIBRATED_RESPONSE_REASON:
        raise RuntimeError("runtime response-support reason changed")
    expected_fragments = {
        "Extracting src spectra",
        "Extracting bkg spectra",
        "ERROR max() iterable argument is empty",
    }
    if set(rule.get("required_log_fragments", [])) != expected_fragments:
        raise RuntimeError("runtime response-support signature changed")
    integrity = runtime_support["integrity"]
    if integrity.get("completed_valid_as295_cells_at_freeze") != 193:
        raise RuntimeError("runtime response-support completion boundary changed")
    if any(
        integrity[key]
        for key in (
            "regional_spectrum_report_existed_at_freeze",
            "regional_temperature_existed_at_freeze",
            "thermal_stress_constructed_at_freeze",
            "lensing_target_opened",
            "scientific_threshold_changed",
            "gravity_parameter_changed",
            "core_v17c_protocol_changed",
            "preflight_response_support_protocol_changed",
        )
    ):
        raise RuntimeError("runtime response-support integrity boundary changed")
    if runtime_report.get("status") != (
        "runtime_empty_response_support_signature_reproduced_and_frozen"
    ):
        raise RuntimeError("runtime response-support report status changed")
    if runtime_report.get("config_sha256") != sha256(runtime_path):
        raise RuntimeError("runtime response-support report used another config")
    discovery = runtime_report["discovery"]
    diagnostic = ROOT / discovery["diagnostic_log"]
    if not diagnostic.is_file() or discovery["diagnostic_log_sha256"] != sha256(
        diagnostic
    ):
        raise RuntimeError("runtime response-support diagnostic changed")
    if runtime_report["decision"].get("lensing_target_opened") is not False:
        raise RuntimeError("runtime response-support report opened the target")
    if not runtime_report_path.is_file():
        raise RuntimeError("runtime response-support report is absent")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reduction", type=Path, default=DEFAULT_REDUCTION)
    parser.add_argument("--regions", type=Path, default=DEFAULT_REGIONS)
    parser.add_argument("--visual-audit", type=Path, default=DEFAULT_VISUAL_AUDIT)
    parser.add_argument("--hi4pi", type=Path, default=DEFAULT_HI4PI)
    parser.add_argument("--astrometry", type=Path, default=DEFAULT_ASTROMETRY)
    parser.add_argument("--repro", type=Path, default=DEFAULT_REPRO)
    parser.add_argument("--cleaning", type=Path, default=DEFAULT_CLEANING)
    parser.add_argument("--restoration", type=Path, default=DEFAULT_RESTORATION)
    parser.add_argument("--integrated-spectra", type=Path, default=DEFAULT_INTEGRATED_SPECTRA)
    parser.add_argument(
        "--integrated-temperatures", type=Path, default=DEFAULT_INTEGRATED_TEMPERATURES
    )
    parser.add_argument(
        "--response-support", type=Path, default=DEFAULT_RESPONSE_SUPPORT
    )
    parser.add_argument(
        "--response-support-report",
        type=Path,
        default=DEFAULT_RESPONSE_SUPPORT_REPORT,
    )
    parser.add_argument(
        "--runtime-response-support",
        type=Path,
        default=DEFAULT_RUNTIME_RESPONSE_SUPPORT,
    )
    parser.add_argument(
        "--runtime-response-support-report",
        type=Path,
        default=DEFAULT_RUNTIME_RESPONSE_SUPPORT_REPORT,
    )
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    paths = {
        "config": args.config.resolve(),
        "reduction_config": args.reduction.resolve(),
        "regions": args.regions.resolve(),
        "visual_audit": args.visual_audit.resolve(),
        "hi4pi_provenance": args.hi4pi.resolve(),
        "astrometry": args.astrometry.resolve(),
        "repro": args.repro.resolve(),
        "cleaning": args.cleaning.resolve(),
        "restoration": args.restoration.resolve(),
        "integrated_spectra": args.integrated_spectra.resolve(),
        "integrated_temperatures": args.integrated_temperatures.resolve(),
        "response_support": args.response_support.resolve(),
        "response_support_report": args.response_support_report.resolve(),
        "runtime_response_support": args.runtime_response_support.resolve(),
        "runtime_response_support_report": args.runtime_response_support_report.resolve(),
    }
    loaded = {
        name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()
    }
    config = loaded["config"]
    regions = loaded["regions"]
    validate_authorization(
        paths["config"],
        config,
        paths["integrated_spectra"],
        loaded["integrated_spectra"],
        paths["integrated_temperatures"],
        loaded["integrated_temperatures"],
    )
    validate_response_support(
        paths["response_support"],
        loaded["response_support"],
        paths["config"],
        paths["integrated_spectra"],
        loaded["integrated_spectra"],
        paths["integrated_temperatures"],
        loaded["integrated_temperatures"],
    )
    validate_runtime_response_support(
        paths["runtime_response_support"],
        loaded["runtime_response_support"],
        paths["runtime_response_support_report"],
        loaded["runtime_response_support_report"],
        paths["config"],
        paths["integrated_spectra"],
        loaded["integrated_spectra"],
        paths["integrated_temperatures"],
        loaded["integrated_temperatures"],
        paths["response_support"],
        paths["response_support_report"],
    )
    if regions["status"] != "both_clusters_passed_frozen_temperature_region_gate":
        raise RuntimeError("temperature-region gate has not passed")
    worker_limit = int(config["execution"]["external_parallel_cells"])
    if not 1 <= worker_limit <= 4:
        raise RuntimeError(f"invalid external cell worker limit: {worker_limit}")
    for key, path_key in (
        ("reduction_config_sha256", "reduction_config"),
        ("temperature_region_report_sha256", "regions"),
        ("spatial_visual_audit_sha256", "visual_audit"),
        ("hi4pi_provenance_sha256", "hi4pi_provenance"),
        ("response_commissioning_restoration_sha256", "restoration"),
    ):
        if config["parents"][key] != sha256(paths[path_key]):
            raise RuntimeError(f"frozen parent hash mismatch: {key}")
    astrometry_rows = {
        int(row["obsid"]): row for row in loaded["astrometry"]["observations"]
    }
    repro_rows = {int(row["obsid"]): row for row in loaded["repro"]["observations"]}
    cleaning_rows = {
        int(row["obsid"]): row for row in loaded["cleaning"]["observations"]
    }
    region_rows = {row["cluster"]: row for row in regions["clusters"]}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    clusters = []
    for cluster_name in config["clusters"]:
        clusters.append(
            build_cluster(
                cluster_name,
                config,
                loaded["response_support"],
                loaded["runtime_response_support"],
                region_rows[cluster_name],
                astrometry_rows,
                repro_rows,
                cleaning_rows,
                loaded["restoration"],
                args.scratch.resolve(),
                output,
            )
        )
    report = {
        "status": "both_frozen_regional_spectra_extracted_combined_and_grouped",
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(paths["config"]),
        "integrated_spectra_report_sha256": sha256(paths["integrated_spectra"]),
        "integrated_temperatures_report_sha256": sha256(
            paths["integrated_temperatures"]
        ),
        "response_support_config_sha256": sha256(paths["response_support"]),
        "response_support_report_sha256": sha256(paths["response_support_report"]),
        "runtime_response_support_config_sha256": sha256(
            paths["runtime_response_support"]
        ),
        "runtime_response_support_report_sha256": sha256(
            paths["runtime_response_support_report"]
        ),
        "clusters": clusters,
        "regional_temperature_fit_authorized": True,
        "thermal_stress_constructed": False,
        "lensing_target_opened": False,
    }
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(report_path)


if __name__ == "__main__":
    main()
