#!/usr/bin/env python3
"""Run frozen hierarchical Gaia/Chandra registration for Sigma v19G."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import sigma_v19f_chandra_common as common

ROOT = common.ROOT
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19g_gaia_hierarchical_astrometry.json"
DEFAULT_ACQUISITION = (
    ROOT / "results" / "sigma_v19g_gaia_acquisition" / "provenance.json"
)
DEFAULT_CLEANING = ROOT / "results" / "sigma_v19f_chandra_cleaning" / "report.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19g_chandra_astrometry"
DEFAULT_SCRATCH = Path("/home/henry/sigma-v19f-chandra")


def load_ciao_modules():
    """Load pycrates-dependent helpers only inside the CIAO runtime."""
    import run_sigma_v17a2_hierarchical_astrometry as hierarchical
    import run_sigma_v17a_chandra_astrometry as direct

    return hierarchical, direct


def validate(
    config_path: Path,
    acquisition_path: Path,
    cleaning_path: Path,
) -> tuple[dict, dict, dict]:
    config = common.load_json(config_path)
    acquisition = common.load_json(acquisition_path)
    cleaning = common.load_json(cleaning_path)
    expected_status = (
        "frozen before any v19 Gaia query, X-ray/Gaia cross-match, relative X-ray "
        "cross-match, transform application, registered science-image inspection, "
        "shock fit, source construction, or replacement-cluster lensing access"
    )
    if config["status"] != expected_status:
        raise RuntimeError("v19G astrometry protocol is not frozen")
    common.validate_parent_hashes(config)
    if acquisition["config_sha256"] != common.sha256(config_path):
        raise RuntimeError("v19G Gaia acquisition used another protocol")
    if acquisition["lensing_target_opened"] is not False:
        raise RuntimeError("v19G Gaia acquisition opened a lensing target")
    if acquisition["xray_source_crossmatch_run"] is not False:
        raise RuntimeError("v19G Gaia acquisition was not target blind")
    if common.sha256(cleaning_path) != config["parents"]["cleaning_report_sha256"]:
        raise RuntimeError("v19G cleaning report differs from its frozen parent")
    if cleaning["lensing_target_opened"] is not False:
        raise RuntimeError("v19G cleaning opened a lensing target")
    if cleaning["event_images_visually_inspected"] is not False:
        raise RuntimeError("v19G cleaning inspected a science image")
    if cleaning["observation_count"] != 20:
        raise RuntimeError("v19G requires all 20 cleaned observations")
    if config["matching"]["method"] != "trans":
        raise RuntimeError("v19G Gaia matching is not translation only")
    if config["relative_matching"]["method"] != "trans":
        raise RuntimeError("v19G relative matching is not translation only")
    return config, acquisition, cleaning


def reference_row(config: dict, cleaning: dict, cluster: str) -> dict:
    rows = [row for row in cleaning["observations"] if row["cluster"] == cluster]
    expected = min(
        rows,
        key=lambda row: (-float(row["clean_exposure_seconds"]), int(row["obsid"])),
    )
    declared = config["reference_selection"]["resolved_from_frozen_cleaning_report"][
        cluster
    ]
    if int(expected["obsid"]) != int(declared["obsid"]):
        raise RuntimeError(f"v19G reference rule changed for {cluster}")
    if abs(
        float(expected["clean_exposure_seconds"])
        - float(declared["clean_exposure_seconds"])
    ) > 1e-9:
        raise RuntimeError(f"v19G reference exposure changed for {cluster}")
    return expected


def failed_row(cluster: str, obsid: int, stage: str, error: Exception) -> dict:
    return {
        "cluster": cluster,
        "obsid": obsid,
        "stage": stage,
        "error": f"{type(error).__name__}: {error}",
        "gates": {"match_execution": False},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--acquisition", type=Path, default=DEFAULT_ACQUISITION)
    parser.add_argument("--cleaning", type=Path, default=DEFAULT_CLEANING)
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config_path = args.config.resolve()
    acquisition_path = args.acquisition.resolve()
    cleaning_path = args.cleaning.resolve()
    config, acquisition, cleaning = validate(
        config_path, acquisition_path, cleaning_path
    )
    hierarchical, direct = load_ciao_modules()
    scratch = args.scratch.resolve()
    results = []

    for cluster in ("BULLET", "ABELL2146"):
        anchor_cleaning = reference_row(config, cleaning, cluster)
        anchor_obsid = int(anchor_cleaning["obsid"])
        try:
            absolute = direct.fit_observation(
                anchor_cleaning,
                config,
                acquisition,
                scratch,
            )
            if not all(absolute["gates"].values()):
                raise RuntimeError(f"absolute Gaia gates failed: {absolute['gates']}")
            anchor, anchor_catalog = hierarchical.make_anchor(
                cluster,
                anchor_obsid,
                {"observations": [absolute]},
                anchor_cleaning,
                scratch,
            )
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
            results.append(failed_row(cluster, anchor_obsid, "Gaia_anchor", error))
            print(f"{cluster} {anchor_obsid} Gaia anchor failed: {error}", flush=True)
            continue

        results.append(anchor)
        stats = anchor["match_statistics"]
        print(
            f"{cluster} {anchor_obsid} Gaia anchor: "
            f"{stats['included_pairs']} pairs, "
            f"RMS={stats['included_rms_recomputed_arcsec']:.4f} arcsec",
            flush=True,
        )
        for row in cleaning["observations"]:
            if row["cluster"] != cluster or int(row["obsid"]) == anchor_obsid:
                continue
            obsid = int(row["obsid"])
            try:
                result = hierarchical.fit_relative(
                    row,
                    anchor_obsid,
                    anchor_catalog,
                    config,
                    scratch,
                )
            except (
                OSError,
                RuntimeError,
                ValueError,
                subprocess.SubprocessError,
            ) as error:
                result = failed_row(
                    cluster, obsid, "relative_to_Gaia_anchor", error
                )
                print(f"{cluster} {obsid} relative match failed: {error}", flush=True)
            else:
                stats = result["match_statistics"]
                print(
                    f"{cluster} {obsid} relative to {anchor_obsid}: "
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
    if not failed and len(results) == 20:
        for row in results:
            row["application"] = direct.apply_observation(
                row,
                scratch,
                Path(row["work"]),
            )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    passed = not failed and len(results) == 20
    report = {
        "status": (
            "all_frozen_v19g_observations_hierarchically_registered_to_Gaia_DR3"
            if passed
            else "frozen_v19g_hierarchical_astrometric_gate_failed"
        ),
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": common.sha256(config_path),
        "gaia_acquisition_sha256": common.sha256(acquisition_path),
        "cleaning_report_sha256": common.sha256(cleaning_path),
        "observation_count": len(results),
        "reference_obsids": {
            cluster: int(config["clusters"][cluster]["reference_obsid"])
            for cluster in config["clusters"]
        },
        "failed_observations": failed,
        "observations": sorted(
            results, key=lambda row: (row["cluster"], int(row["obsid"]))
        ),
        "all_hierarchical_gates_passed": passed,
        "transforms_applied": passed,
        "registered_science_images_inspected": False,
        "shock_front_fitted": False,
        "source_constructed": False,
        "lensing_target_opened": False,
    }
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(report_path)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
