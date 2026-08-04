#!/usr/bin/env python3
"""Run the frozen Gaia DR3 translation-only registration for Sigma v17A."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pycrates

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v17a_gaia_astrometry.json"
DEFAULT_REDUCTION = ROOT / "configs" / "sigma_v17a_chandra_reduction.json"
DEFAULT_ACQUISITION = (
    ROOT / "results" / "sigma_v17a_gaia_astrometry_acquisition" / "provenance.json"
)
DEFAULT_CLEANING = ROOT / "results" / "sigma_v17a_chandra_cleaning" / "report.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v17a_chandra_astrometry"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def isolated_environment(base: os._Environ[str], pfiles: Path, tmp: Path) -> dict[str, str]:
    env = dict(base)
    pfiles.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    existing = env.get("PFILES", "")
    system = existing.split(";", maxsplit=1)[1] if ";" in existing else existing
    env["PFILES"] = f"{pfiles};{system}" if system else str(pfiles)
    env["ASCDS_WORK_PATH"] = str(tmp)
    env["TMPDIR"] = str(tmp)
    return env


def run_step(
    command: list[str],
    log_path: Path,
    expected: list[Path],
    env: dict[str, str],
) -> dict:
    present = [path.is_file() for path in expected]
    if all(present):
        if not log_path.exists():
            raise RuntimeError(f"complete output lacks required log: {log_path}")
        return {"command": command, "reused": True, "log": str(log_path)}
    if any(present):
        raise RuntimeError(f"partial output exists for step log {log_path}")
    result = subprocess.run(command, check=False, capture_output=True, text=True, env=env)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"command failed; see {log_path}")
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError(f"command did not create expected files: {missing}")
    return {"command": command, "reused": False, "log": str(log_path)}


def dmkeypar(path: Path, key: str, env: dict[str, str]) -> str:
    result = subprocess.run(
        ["dmkeypar", str(path), key, "echo+"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    value = result.stdout.strip()
    if not value:
        raise RuntimeError(f"missing {key} in {path}")
    return value


def observation_epoch(event: Path, env: dict[str, str]) -> tuple[str, str, float]:
    start_text = dmkeypar(event, "DATE-OBS", env)
    end_text = dmkeypar(event, "DATE-END", env)
    start = datetime.fromisoformat(start_text)
    end = datetime.fromisoformat(end_text)
    midpoint = start + (end - start) / 2
    unix_seconds = midpoint.timestamp()
    julian_date = 2_440_587.5 + unix_seconds / 86_400.0
    julian_year = 2000.0 + (julian_date - 2_451_545.0) / 365.25
    return start_text, end_text, julian_year


def optional_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def load_gaia(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty Gaia catalog: {path}")
    return rows


def propagated_reference(rows: list[dict], epoch: float) -> list[tuple[float, float, float, float]]:
    propagated = []
    for row in rows:
        ra = float(row["ra"])
        dec = float(row["dec"])
        ref_epoch = float(row["ref_epoch"])
        ra_error_mas = float(row["ra_error"])
        dec_error_mas = float(row["dec_error"])
        pmra = optional_float(row["pmra"])
        pmdec = optional_float(row["pmdec"])
        pmra_error = optional_float(row["pmra_error"])
        pmdec_error = optional_float(row["pmdec_error"])
        elapsed = epoch - ref_epoch
        cos_dec = math.cos(math.radians(dec))
        if pmra is not None and pmdec is not None:
            ra += pmra * elapsed / (3_600_000.0 * cos_dec)
            dec += pmdec * elapsed / 3_600_000.0
            if pmra_error is not None:
                ra_error_mas = math.hypot(ra_error_mas, pmra_error * elapsed)
            if pmdec_error is not None:
                dec_error_mas = math.hypot(dec_error_mas, pmdec_error * elapsed)
        ra_error_deg = ra_error_mas / (3_600_000.0 * cos_dec)
        dec_error_deg = dec_error_mas / 3_600_000.0
        propagated.append((ra, ra_error_deg, dec, dec_error_deg))
    return propagated


def write_reference(path: Path, rows: list[tuple[float, float, float, float]]) -> None:
    content = "#RA RA_ERR DEC DEC_ERR\n" + "".join(
        f"{ra:.12f} {ra_error:.12g} {dec:.12f} {dec_error:.12g}\n"
        for ra, ra_error, dec, dec_error in rows
    )
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"existing epoch reference differs: {path}")
        return
    path.write_text(content, encoding="utf-8")


def transform_values(path: Path) -> dict[str, float]:
    crate = pycrates.read_file(str(path))
    return {
        name: float(crate.get_column(name).values[0])
        for name in ("a11", "a12", "a21", "a22", "t1", "t2")
    }


def match_statistics(path: Path, env: dict[str, str]) -> dict:
    crate = pycrates.read_file(str(path))
    included = np.asarray(crate.get_column("INCLUDE").values, dtype=bool)
    residual = np.asarray(crate.get_column("XFORM_RESIDUAL_RSS").values, dtype=float)
    if not included.any():
        raise RuntimeError(f"no source pairs survived in {path}")
    return {
        "candidate_pairs": len(included),
        "included_pairs": int(included.sum()),
        "rms_before_arcsec": float(dmkeypar(path, "RMS_R_B", env)),
        "rms_after_arcsec": float(dmkeypar(path, "RMS_R_A", env)),
        "max_after_arcsec": float(dmkeypar(path, "MAX_R_A", env)),
        "included_rms_recomputed_arcsec": float(np.sqrt(np.mean(residual[included] ** 2))),
        "included_max_recomputed_arcsec": float(np.max(residual[included])),
    }


def inventory(directory: Path) -> list[dict]:
    return [
        {
            "relative_path": path.relative_to(directory).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(item for item in directory.rglob("*") if item.is_file())
    ]


def fit_observation(
    row: dict,
    config: dict,
    acquisition: dict,
    scratch: Path,
) -> dict:
    cluster = row["cluster"]
    obsid = int(row["obsid"])
    work = scratch / "astrometry_v101" / cluster / str(obsid)
    logs = work / "logs"
    work.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    env = isolated_environment(
        os.environ,
        scratch / "pfiles_astrometry" / cluster / str(obsid),
        scratch / "tmp_astrometry" / cluster / str(obsid),
    )
    event = Path(row["clean_event"])
    source_catalog = event.parent / "source_detect_b2" / "sources.fits"
    wcs_image = event.parent / "source_detect_b2" / "initial_0.5-7.0_thresh.img"
    selected = work / "xray_sources_selected.fits"
    selection = config["xray_source_selection"]
    selection_filter = (
        f"{source_catalog}[SRC_SIGNIFICANCE>={selection['minimum_source_significance']},"
        f"NET_COUNTS>{selection['minimum_net_counts_exclusive']}]"
    )
    dmcopy_step = run_step(
        ["dmcopy", selection_filter, str(selected), "clobber=no", "mode=h"],
        logs / "select_xray_sources.log",
        [selected],
        env,
    )

    start, end, epoch = observation_epoch(event, env)
    gaia_record = next(item for item in acquisition["records"] if item["cluster"] == cluster)
    gaia_path = ROOT / gaia_record["relative_path"]
    if sha256(gaia_path) != gaia_record["sha256"]:
        raise RuntimeError(f"Gaia catalog hash mismatch: {gaia_path}")
    reference = work / "gaia_dr3_observation_epoch.dat"
    write_reference(reference, propagated_reference(load_gaia(gaia_path), epoch))

    matching = config["matching"]
    transform = work / "gaia_translation.xform.fits"
    match_step = run_step(
        [
            "wcs_match",
            f"infile={selected}",
            f"refsrcfile={reference}",
            f"outfile={transform}",
            f"wcsfile={wcs_image}",
            f"radius={matching['initial_radius_arcsec']}",
            f"residlim={matching['residual_limit_arcsec']}",
            f"residtype={matching['residual_type']}",
            f"residfac={matching['residual_factor']}",
            f"method={matching['method']}",
            "clobber=no",
            "verbose=1",
            "mode=h",
        ],
        logs / "wcs_match.log",
        [transform],
        env,
    )
    stats_path = work / "gaia_match_statistics.fits"
    parse_step = run_step(
        [
            "parse_wcs_match_log",
            f"infile={selected}",
            f"refsrcfile={reference}",
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
    stats = match_statistics(stats_path, env)
    gates = {
        "minimum_final_source_pairs": (
            stats["included_pairs"] >= matching["minimum_final_source_pairs"]
        ),
        "maximum_final_radial_rms": (
            stats["included_rms_recomputed_arcsec"] <= matching["maximum_final_radial_rms_arcsec"]
        ),
        "maximum_individual_residual": (
            stats["included_max_recomputed_arcsec"] <= matching["residual_limit_arcsec"]
        ),
    }
    values = transform_values(transform)
    gates["translation_only_matrix"] = (
        abs(values["a11"] - 1.0) < 1e-12
        and abs(values["a22"] - 1.0) < 1e-12
        and abs(values["a12"]) < 1e-12
        and abs(values["a21"]) < 1e-12
    )
    return {
        "cluster": cluster,
        "obsid": obsid,
        "clean_event": str(event),
        "blanksky_event": row["blanksky_event"],
        "source_catalog": str(source_catalog),
        "wcs_image": str(wcs_image),
        "observation_start": start,
        "observation_end": end,
        "observation_epoch_jyear": epoch,
        "gaia_catalog": str(gaia_path),
        "gaia_catalog_sha256": gaia_record["sha256"],
        "epoch_reference": str(reference),
        "epoch_reference_sha256": sha256(reference),
        "selected_xray_sources": len(pycrates.read_file(str(selected)).get_column("RA").values),
        "transform": str(transform),
        "transform_sha256": sha256(transform),
        "transform_values": values,
        "match_statistics": stats,
        "match_statistics_path": str(stats_path),
        "gates": gates,
        "steps": {"source_selection": dmcopy_step, "match": match_step, "parse": parse_step},
        "work": str(work),
    }


def apply_observation(result: dict, scratch: Path, output_work: Path | None = None) -> dict:
    cluster = result["cluster"]
    obsid = result["obsid"]
    work = output_work if output_work is not None else Path(result["work"])
    logs = work / "logs"
    env = isolated_environment(
        os.environ,
        scratch / "pfiles_astrometry" / cluster / str(obsid),
        scratch / "tmp_astrometry" / cluster / str(obsid),
    )
    transform = Path(result["transform"])
    wcs_image = Path(result["wcs_image"])
    repro_dir = scratch / "repro" / cluster / str(obsid)
    asols = sorted(repro_dir.glob("pcadf*_asol1.fits"))
    if not asols:
        raise RuntimeError(f"no aspect solution found for {cluster} {obsid}: {repro_dir}")
    corrected_asols = []
    asol_steps = []
    for asol in asols:
        corrected = work / asol.name.replace("_asol1.fits", "_gaia_asol1.fits")
        step = run_step(
            [
                "wcs_update",
                f"infile={asol}",
                f"outfile={corrected}",
                f"transformfile={transform}",
                f"wcsfile={wcs_image}",
                "clobber=no",
                "verbose=1",
                "mode=h",
            ],
            logs / f"wcs_update_{asol.stem}.log",
            [corrected],
            env,
        )
        corrected_asols.append(corrected)
        asol_steps.append(step)
    asol_list = work / "corrected_asol1.lis"
    list_content = "\n".join(str(path) for path in corrected_asols) + "\n"
    if asol_list.exists() and asol_list.read_text(encoding="utf-8") != list_content:
        raise RuntimeError(f"existing corrected aspect list differs: {asol_list}")
    if not asol_list.exists():
        asol_list.write_text(list_content, encoding="utf-8")
    asol_header = ",".join(path.name for path in corrected_asols)

    event_steps = []
    corrected_events = {}
    for label, source in (
        ("science", Path(result["clean_event"])),
        ("blanksky", Path(result["blanksky_event"])),
    ):
        corrected = work / f"acisf{obsid}_{label}_gaia_evt.fits"
        copy_step = run_step(
            ["dmcopy", str(source), str(corrected), "opt=all", "clobber=no", "mode=h"],
            logs / f"copy_{label}_event.log",
            [corrected],
            env,
        )
        update_log = logs / f"wcs_update_{label}_event.log"
        marker = work / f".{label}_event_wcs_updated"
        if marker.exists():
            if not update_log.exists():
                raise RuntimeError(f"event update marker lacks log: {marker}")
            marker_hash = marker.read_text(encoding="utf-8").strip()
            if marker_hash != sha256(corrected):
                raise RuntimeError(f"event update marker hash differs: {marker}")
            current_asol_header = dmkeypar(corrected, "ASOLFILE", env)
            repaired_header = current_asol_header != asol_header
            if repaired_header:
                subprocess.run(
                    [
                        "dmhedit",
                        str(corrected),
                        "file=",
                        "op=add",
                        "key=ASOLFILE",
                        f"value={asol_header}",
                        "mode=h",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    env=env,
                )
                marker.write_text(sha256(corrected) + "\n", encoding="utf-8")
            update_step = {
                "reused": True,
                "ASOLFILE_header_repaired": repaired_header,
                "log": str(update_log),
            }
        else:
            command = [
                "wcs_update",
                f"infile={corrected}",
                "outfile=",
                f"transformfile={transform}",
                f"wcsfile={wcs_image}",
                "clobber=no",
                "verbose=1",
                "mode=h",
            ]
            completed = subprocess.run(
                command, check=False, capture_output=True, text=True, env=env
            )
            update_log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
            if completed.returncode != 0:
                raise RuntimeError(f"event WCS update failed; see {update_log}")
            subprocess.run(
                [
                    "dmhedit",
                    str(corrected),
                    "file=",
                    "op=add",
                    "key=ASOLFILE",
                    f"value={asol_header}",
                    "mode=h",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            marker.write_text(sha256(corrected) + "\n", encoding="utf-8")
            update_step = {"command": command, "reused": False, "log": str(update_log)}
        corrected_events[label] = {
            "source": str(source),
            "source_sha256": sha256(source),
            "path": str(corrected),
            "sha256": sha256(corrected),
            "asolfile_header": dmkeypar(corrected, "ASOLFILE", env),
        }
        event_steps.append({"label": label, "copy": copy_step, "update": update_step})

    products = inventory(work)
    return {
        "corrected_aspects": [
            {"path": str(path), "sha256": sha256(path)} for path in corrected_asols
        ],
        "corrected_aspect_list": str(asol_list),
        "corrected_events": corrected_events,
        "steps": {"aspect_updates": asol_steps, "event_updates": event_steps},
        "product_files": len(products),
        "product_bytes": sum(row["bytes"] for row in products),
        "products": products,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reduction", type=Path, default=DEFAULT_REDUCTION)
    parser.add_argument("--acquisition", type=Path, default=DEFAULT_ACQUISITION)
    parser.add_argument("--cleaning", type=Path, default=DEFAULT_CLEANING)
    parser.add_argument("--scratch", type=Path, default=Path("/home/henry/sigma-v17a-chandra"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config_path = args.config.resolve()
    reduction_path = args.reduction.resolve()
    acquisition_path = args.acquisition.resolve()
    cleaning_path = args.cleaning.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    reduction = json.loads(reduction_path.read_text(encoding="utf-8"))
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    cleaning = json.loads(cleaning_path.read_text(encoding="utf-8"))
    if config["parent_reduction_protocol"] != str(reduction_path.relative_to(ROOT).as_posix()):
        raise RuntimeError("astrometry protocol does not identify this reduction protocol")
    if acquisition["config_sha256"] != sha256(config_path):
        raise RuntimeError("Gaia acquisition does not match the astrometry protocol")
    if cleaning["protocol_version"] != reduction["protocol_version"]:
        raise RuntimeError("cleaning and reduction protocols differ")

    results = []
    for row in cleaning["observations"]:
        try:
            result = fit_observation(row, config, acquisition, args.scratch.resolve())
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
            result = {
                "cluster": row["cluster"],
                "obsid": int(row["obsid"]),
                "error": f"{type(error).__name__}: {error}",
                "gates": {"match_execution": False},
            }
            results.append(result)
            print(
                f"{result['cluster']} {result['obsid']}: match execution failed: {error}",
                flush=True,
            )
            continue
        results.append(result)
        stats = result["match_statistics"]
        print(
            f"{result['cluster']} {result['obsid']}: "
            f"{stats['included_pairs']} pairs, "
            f"RMS={stats['included_rms_recomputed_arcsec']:.4f} arcsec",
            flush=True,
        )

    failed = [
        {
            "cluster": row["cluster"],
            "obsid": row["obsid"],
            "gates": row["gates"],
            "error": row.get("error"),
        }
        for row in results
        if not all(row["gates"].values())
    ]
    if not failed:
        for row in results:
            row["application"] = apply_observation(row, args.scratch.resolve())

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": (
            "all_frozen_observations_registered_to_Gaia_DR3"
            if not failed
            else "frozen_Gaia_DR3_astrometric_gate_failed"
        ),
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "reduction_config_sha256": sha256(reduction_path),
        "gaia_acquisition_sha256": sha256(acquisition_path),
        "cleaning_report_sha256": sha256(cleaning_path),
        "observation_count": len(results),
        "failed_observations": failed,
        "observations": results,
        "all_absolute_gates_passed": not failed,
        "relative_registration_checked": False,
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
