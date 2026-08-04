#!/usr/bin/env python3
"""Run the frozen hierarchical Gaia/Chandra astrometry repair for Sigma v17A2."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pycrates
import run_sigma_v17a_chandra_astrometry as direct

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v17a2_hierarchical_astrometry.json"
DEFAULT_PARENT = ROOT / "results" / "sigma_v17a_chandra_astrometry" / "report.json"
DEFAULT_CLEANING = ROOT / "results" / "sigma_v17a_chandra_cleaning" / "report.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v17a2_hierarchical_astrometry"


def selected_catalog(
    source: Path,
    output: Path,
    log: Path,
    env: dict[str, str],
    minimum_significance: float = 3.0,
) -> dict:
    return direct.run_step(
        [
            "dmcopy",
            f"{source}[SRC_SIGNIFICANCE>={minimum_significance},NET_COUNTS>0]",
            str(output),
            "clobber=no",
            "mode=h",
        ],
        log,
        [output],
        env,
    )


def update_catalog_wcs(
    catalog: Path,
    transform: Path,
    wcs_image: Path,
    marker: Path,
    log: Path,
    env: dict[str, str],
) -> dict:
    command = [
        "wcs_update",
        f"infile={catalog}",
        "outfile=",
        f"transformfile={transform}",
        f"wcsfile={wcs_image}",
        "clobber=no",
        "verbose=1",
        "mode=h",
    ]
    if marker.exists():
        if not log.exists() or marker.read_text(encoding="utf-8").strip() != direct.sha256(catalog):
            raise RuntimeError(f"invalid corrected-catalog marker: {marker}")
        return {"command": command, "reused": True, "log": str(log)}
    completed = subprocess.run(command, check=False, capture_output=True, text=True, env=env)
    log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"catalog WCS update failed; see {log}")
    marker.write_text(direct.sha256(catalog) + "\n", encoding="utf-8")
    return {"command": command, "reused": False, "log": str(log)}


def make_anchor(
    cluster: str,
    obsid: int,
    parent: dict,
    cleaning_row: dict,
    scratch: Path,
) -> tuple[dict, Path]:
    parent_row = next(
        row
        for row in parent["observations"]
        if row["cluster"] == cluster and int(row["obsid"]) == obsid
    )
    if not all(parent_row["gates"].values()):
        raise RuntimeError(f"selected anchor did not pass the direct Gaia gate: {cluster} {obsid}")
    work = scratch / "astrometry_v102" / cluster / str(obsid)
    logs = work / "logs"
    work.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    env = direct.isolated_environment(
        os.environ,
        scratch / "pfiles_astrometry_v102" / cluster / str(obsid),
        scratch / "tmp_astrometry_v102" / cluster / str(obsid),
    )
    source = Path(parent_row["source_catalog"])
    corrected = work / "reference_xray_sources_gaia.fits"
    selection_step = selected_catalog(source, corrected, logs / "select_anchor_sources.log", env)
    update_step = update_catalog_wcs(
        corrected,
        Path(parent_row["transform"]),
        Path(parent_row["wcs_image"]),
        work / ".reference_catalog_wcs_updated",
        logs / "wcs_update_reference_catalog.log",
        env,
    )
    result = {
        **parent_row,
        "stage": "Gaia_anchor",
        "work": str(work),
        "clean_event": cleaning_row["clean_event"],
        "blanksky_event": cleaning_row["blanksky_event"],
        "anchor_catalog": str(corrected),
        "anchor_catalog_sha256": direct.sha256(corrected),
        "hierarchical_steps": {
            "source_selection": selection_step,
            "catalog_wcs_update": update_step,
        },
    }
    return result, corrected


def fit_relative(
    row: dict,
    anchor_obsid: int,
    anchor_catalog: Path,
    config: dict,
    scratch: Path,
) -> dict:
    cluster = row["cluster"]
    obsid = int(row["obsid"])
    work = scratch / "astrometry_v102" / cluster / str(obsid)
    logs = work / "logs"
    work.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    env = direct.isolated_environment(
        os.environ,
        scratch / "pfiles_astrometry_v102" / cluster / str(obsid),
        scratch / "tmp_astrometry_v102" / cluster / str(obsid),
    )
    clean_event = Path(row["clean_event"])
    source = clean_event.parent / "source_detect_b2" / "sources.fits"
    wcs_image = clean_event.parent / "source_detect_b2" / "initial_0.5-7.0_thresh.img"
    selected = work / "xray_sources_selected.fits"
    select_step = selected_catalog(source, selected, logs / "select_xray_sources.log", env)

    rule = config["relative_matching"]
    transform = work / "relative_translation.xform.fits"
    match_step = direct.run_step(
        [
            "wcs_match",
            f"infile={selected}",
            f"refsrcfile={anchor_catalog}",
            f"outfile={transform}",
            f"wcsfile={wcs_image}",
            f"radius={rule['initial_radius_arcsec']}",
            f"residlim={rule['residual_limit_arcsec']}",
            f"residtype={rule['residual_type']}",
            f"residfac={rule['residual_factor']}",
            f"method={rule['method']}",
            "clobber=no",
            "verbose=1",
            "mode=h",
        ],
        logs / "wcs_match.log",
        [transform],
        env,
    )
    stats_path = work / "relative_match_statistics.fits"
    parse_step = direct.run_step(
        [
            "parse_wcs_match_log",
            f"infile={selected}",
            f"refsrcfile={anchor_catalog}",
            f"logfile={logs / 'wcs_match.log'}",
            f"outfile={stats_path}",
            "clobber=no",
            "verbose=1",
            "mode=h",
        ],
        logs / "parse_wcs_match_log.log",
        [stats_path],
        env,
    )
    stats = direct.match_statistics(stats_path, env)
    values = direct.transform_values(transform)
    gates = {
        "minimum_final_source_pairs": (
            stats["included_pairs"] >= rule["minimum_final_source_pairs"]
        ),
        "maximum_final_radial_rms": (
            stats["included_rms_recomputed_arcsec"] <= rule["maximum_final_radial_rms_arcsec"]
        ),
        "maximum_individual_residual": (
            stats["included_max_recomputed_arcsec"] <= rule["residual_limit_arcsec"]
        ),
        "translation_only_matrix": (
            abs(values["a11"] - 1.0) < 1e-12
            and abs(values["a22"] - 1.0) < 1e-12
            and abs(values["a12"]) < 1e-12
            and abs(values["a21"]) < 1e-12
        ),
    }
    return {
        "cluster": cluster,
        "obsid": obsid,
        "stage": "relative_to_Gaia_anchor",
        "anchor_obsid": anchor_obsid,
        "clean_event": str(clean_event),
        "blanksky_event": row["blanksky_event"],
        "source_catalog": str(source),
        "wcs_image": str(wcs_image),
        "anchor_catalog": str(anchor_catalog),
        "anchor_catalog_sha256": direct.sha256(anchor_catalog),
        "selected_xray_sources": len(pycrates.read_file(str(selected)).get_column("RA").values),
        "transform": str(transform),
        "transform_sha256": direct.sha256(transform),
        "transform_values": values,
        "match_statistics": stats,
        "match_statistics_path": str(stats_path),
        "gates": gates,
        "steps": {"source_selection": select_step, "match": match_step, "parse": parse_step},
        "work": str(work),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--cleaning", type=Path, default=DEFAULT_CLEANING)
    parser.add_argument("--scratch", type=Path, default=Path("/home/henry/sigma-v17a-chandra"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config_path = args.config.resolve()
    parent_path = args.parent.resolve()
    cleaning_path = args.cleaning.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    cleaning = json.loads(cleaning_path.read_text(encoding="utf-8"))
    if direct.sha256(parent_path) != config["parent_direct_report_sha256"]:
        raise RuntimeError("the parent direct-astrometry failure report changed after freeze")
    if parent["status"] != config["parent_failure"]["formal_status"]:
        raise RuntimeError("the parent report is not the frozen failed direct gate")

    references = config["reference_selection"]["resolved_reference_obsids"]
    for cluster, resolved in references.items():
        rows = [row for row in cleaning["observations"] if row["cluster"] == cluster]
        expected = min(
            rows, key=lambda row: (-float(row["clean_exposure_seconds"]), int(row["obsid"]))
        )["obsid"]
        if int(resolved) != int(expected):
            raise RuntimeError(f"resolved reference violates the frozen rule for {cluster}")

    results = []
    for cluster, anchor_obsid_value in references.items():
        anchor_obsid = int(anchor_obsid_value)
        anchor_cleaning = next(
            row
            for row in cleaning["observations"]
            if row["cluster"] == cluster and int(row["obsid"]) == anchor_obsid
        )
        anchor, anchor_catalog = make_anchor(
            cluster, anchor_obsid, parent, anchor_cleaning, args.scratch.resolve()
        )
        results.append(anchor)
        stats = anchor["match_statistics"]
        print(
            f"{cluster} {anchor_obsid} Gaia anchor: {stats['included_pairs']} pairs, "
            f"RMS={stats['included_rms_recomputed_arcsec']:.4f} arcsec",
            flush=True,
        )
        for row in cleaning["observations"]:
            if row["cluster"] != cluster or int(row["obsid"]) == anchor_obsid:
                continue
            try:
                result = fit_relative(
                    row, anchor_obsid, anchor_catalog, config, args.scratch.resolve()
                )
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
                result = {
                    "cluster": cluster,
                    "obsid": int(row["obsid"]),
                    "stage": "relative_to_Gaia_anchor",
                    "anchor_obsid": anchor_obsid,
                    "error": f"{type(error).__name__}: {error}",
                    "gates": {"match_execution": False},
                    "work": str(
                        args.scratch.resolve() / "astrometry_v102" / cluster / str(row["obsid"])
                    ),
                }
                print(f"{cluster} {row['obsid']}: {error}", flush=True)
            else:
                stats = result["match_statistics"]
                print(
                    f"{cluster} {row['obsid']} relative to {anchor_obsid}: "
                    f"{stats['included_pairs']} pairs, "
                    f"RMS={stats['included_rms_recomputed_arcsec']:.4f} arcsec",
                    flush=True,
                )
            results.append(result)

    failed = [
        {
            "cluster": row["cluster"],
            "obsid": row["obsid"],
            "stage": row["stage"],
            "gates": row["gates"],
            "error": row.get("error"),
        }
        for row in results
        if not all(row["gates"].values())
    ]
    if not failed:
        for row in results:
            row["application"] = direct.apply_observation(
                row, args.scratch.resolve(), Path(row["work"])
            )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": (
            "all_frozen_observations_hierarchically_registered_to_Gaia_DR3"
            if not failed
            else "frozen_hierarchical_astrometric_gate_failed"
        ),
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": direct.sha256(config_path),
        "parent_direct_report_sha256": direct.sha256(parent_path),
        "cleaning_report_sha256": direct.sha256(cleaning_path),
        "observation_count": len(results),
        "reference_obsids": references,
        "failed_observations": failed,
        "observations": sorted(results, key=lambda row: (row["cluster"], row["obsid"])),
        "all_hierarchical_gates_passed": not failed,
        "transforms_applied": not failed,
        "registered_event_images_inspected": False,
        "lensing_target_opened": False,
        "temperature_map_constructed": False,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report_path)
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
