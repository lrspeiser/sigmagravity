#!/usr/bin/env python3
"""Extract and combine the frozen integrated Sigma v17C Chandra spectra."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v17c_spectral_temperature.json"
DEFAULT_REDUCTION = ROOT / "configs" / "sigma_v17a_chandra_reduction.json"
DEFAULT_REGIONS = ROOT / "results" / "sigma_v17b_temperature_regions" / "report.json"
DEFAULT_VISUAL_AUDIT = (
    ROOT / "results" / "sigma_v17b_temperature_regions" / "audit" / "visual_audit.json"
)
DEFAULT_HI4PI = ROOT / "results" / "sigma_v17c_hi4pi_acquisition" / "provenance.json"
DEFAULT_ASTROMETRY = ROOT / "results" / "sigma_v17a2_hierarchical_astrometry" / "report.json"
DEFAULT_REPRO = ROOT / "results" / "sigma_v17a_chandra_repro" / "report.json"
DEFAULT_CLEANING = ROOT / "results" / "sigma_v17a_chandra_cleaning" / "report.json"
DEFAULT_RESTORATION = ROOT / "results" / "sigma_v17c_response_commissioning" / "restoration.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v17c_integrated_spectra"
DEFAULT_SCRATCH = Path("/home/henry/sigma-v17a-chandra")


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


def run_step(command: list[str], log: Path, expected: list[Path], env: dict[str, str]) -> dict:
    present = [path.is_file() for path in expected]
    if all(present):
        if not log.is_file():
            raise RuntimeError(f"complete outputs lack log: {log}")
        return {"command": command, "reused": True, "log": str(log)}
    if any(present):
        raise RuntimeError(f"partial outputs for {command[0]}: {expected}")
    log.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, check=False, capture_output=True, text=True, env=env)
    log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"{command[0]} failed; inspect {log}")
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError(f"{command[0]} did not create {missing}; inspect {log}")
    return {"command": command, "reused": False, "log": str(log)}


def command_text(command: list[str], env: dict[str, str]) -> str:
    return subprocess.run(
        command, check=True, capture_output=True, text=True, env=env
    ).stdout.strip()


def header_number(path: Path, keyword: str, env: dict[str, str]) -> float:
    return float(command_text(["dmkeypar", str(path), keyword, "echo+"], env))


def event_count(virtual_file: str, env: dict[str, str]) -> int:
    value = command_text(["dmlist", virtual_file, "counts"], env)
    return int(value.split()[0])


def event_reference_coordinate(
    source_filter: str,
    science: Path,
    env: dict[str, str],
) -> dict:
    """Return an outcome-blind ICRS response location on the filtered CCD."""
    import numpy as np
    import pycrates

    virtual_file = source_filter + "[energy=500:7000][cols x,y]"
    crate = pycrates.read_file(virtual_file)
    x = np.asarray(crate.get_column("x").values, dtype=float)
    y = np.asarray(crate.get_column("y").values, dtype=float)
    if x.size == 0 or y.size != x.size:
        raise RuntimeError(f"cannot determine response reference from {virtual_file}")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise RuntimeError(f"non-finite response-reference coordinates in {virtual_file}")
    mean_x = float(x.mean())
    mean_y = float(y.mean())
    subprocess.run(
        ["punlearn", "dmcoords"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    command = [
        "dmcoords",
        f"infile={science}",
        "option=sky",
        f"x={mean_x:.14f}",
        f"y={mean_y:.14f}",
        "celfmt=deg",
        "verbose=0",
        "mode=h",
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"dmcoords failed for response reference in {science}: "
            f"{completed.stdout}{completed.stderr}"
        )
    ra = float(command_text(["pget", "dmcoords", "ra"], env))
    dec = float(command_text(["pget", "dmcoords", "dec"], env))
    chip_id = int(float(command_text(["pget", "dmcoords", "chip_id"], env)))
    if not all(np.isfinite(value) for value in (mean_x, mean_y, ra, dec)):
        raise RuntimeError(f"non-finite converted response reference for {virtual_file}")
    return {
        "selection": virtual_file,
        "events": int(x.size),
        "mean_sky_x": mean_x,
        "mean_sky_y": mean_y,
        "ra_deg": ra,
        "dec_deg": dec,
        "dmcoords_chip_id": chip_id,
        "command": command,
    }


def find_product(repro_row: dict, suffix: str) -> Path:
    matches = [
        Path(repro_row["output_directory"]) / item["relative_path"]
        for item in repro_row["products"]
        if item["relative_path"].endswith(suffix)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"ObsID {repro_row['obsid']} expected one {suffix} product, found {matches}"
        )
    return matches[0]


def frozen_region_files(cluster: dict) -> list[Path]:
    products = [
        item for item in cluster["frozen_snapshot"]["products"] if item["role"] == "spectral_region"
    ]
    paths = []
    for item in products:
        path = ROOT / item["relative_path"]
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise RuntimeError(f"frozen spectral region changed: {path}")
        paths.append(path)
    return sorted(paths, key=lambda path: int(path.stem.rsplit("_", maxsplit=1)[1]))


def write_integrated_region(paths: list[Path], destination: Path) -> dict:
    lines = ["# Region file format: CIAO version 1.0"]
    for path in paths:
        lines.extend(
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")
        )
    content = "\n".join(lines) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"existing integrated region differs: {destination}")
        reused = True
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        reused = False
    return {
        "path": str(destination),
        "sha256": sha256(destination),
        "bytes": destination.stat().st_size,
        "source_region_files": len(paths),
        "reused": reused,
    }


def prepare_translated_fov(
    source: Path,
    destination: Path,
    astrometry_row: dict,
    expected_source_hash: str,
    env: dict[str, str],
) -> dict:
    marker = destination.with_suffix(".wcs_update.json")
    if destination.exists() or marker.exists():
        if not destination.is_file() or not marker.is_file():
            raise RuntimeError(f"partial translated FOV products for {destination}")
        record = json.loads(marker.read_text(encoding="utf-8"))
        if sha256(destination) != record["translated_sha256"]:
            raise RuntimeError(f"translated FOV changed: {destination}")
        return {**record, "reused": True}
    if sha256(source) != expected_source_hash:
        raise RuntimeError(f"source FOV hash mismatch: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    before = sha256(destination)
    command = [
        "wcs_update",
        f"infile={destination}",
        "outfile=",
        f"transformfile={astrometry_row['transform']}",
        f"wcsfile={astrometry_row['wcs_image']}",
        "clobber=no",
        "verbose=1",
        "mode=h",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True, env=env)
    log = destination.with_suffix(".wcs_update.log")
    log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"wcs_update failed; inspect {log}")
    after = sha256(destination)
    if after == before:
        raise RuntimeError(f"translated FOV was not changed: {destination}")
    record = {
        "source": str(source),
        "source_sha256": before,
        "translated": str(destination),
        "translated_sha256": after,
        "transform_sha256": sha256(Path(astrometry_row["transform"])),
        "command": command,
        "log": str(log),
        "log_sha256": sha256(log),
        "reused": False,
    }
    marker.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def prepare_background_geometry(
    source: Path,
    background: Path,
    destination: Path,
    env: dict[str, str],
) -> dict:
    """Copy blank-sky events and replace only stale detector-geometry metadata."""
    marker = destination.with_suffix(".geometry.json")
    if destination.exists() or marker.exists():
        if not destination.is_file() or not marker.is_file():
            raise RuntimeError(f"partial blank-sky geometry products for {destination}")
        record = json.loads(marker.read_text(encoding="utf-8"))
        if sha256(destination) != record["corrected_sha256"]:
            raise RuntimeError(f"blank-sky geometry copy changed: {destination}")
        if sha256(background) != record["source_sha256"]:
            raise RuntimeError(f"blank-sky source changed: {background}")
        return {**record, "reused": True}

    keys = [
        "OBS_ID",
        "DETNAM",
        "SIM_X",
        "SIM_Y",
        "SIM_Z",
        "RA_PNT",
        "DEC_PNT",
        "ROLL_PNT",
        "RA_NOM",
        "DEC_NOM",
        "ROLL_NOM",
        "DY_AVG",
        "DZ_AVG",
        "DTH_AVG",
    ]
    string_keys = {"OBS_ID", "DETNAM"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(background, destination)
    edit_list = destination.with_suffix(".geometry.lis")
    lines = ["#add"]
    copied_values = {}
    for key in keys:
        value = command_text(["dmkeypar", str(source), key, "echo+"], env)
        copied_values[key] = value
        rendered = f"'{value}'" if key in string_keys else value
        lines.append(f"{key} = {rendered}")
    edit_list.write_text("\n".join(lines) + "\n", encoding="ascii")
    command = [
        "dmhedit",
        f"infile={destination}",
        f"filelist={edit_list}",
        "operation=",
        "key=",
        "value=",
        "verbose=0",
        "mode=h",
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    log = destination.with_suffix(".geometry.log")
    log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"blank-sky geometry correction failed; inspect {log}")
    source_rows = event_count(str(background), env)
    corrected_rows = event_count(str(destination), env)
    if corrected_rows != source_rows:
        raise RuntimeError(
            f"blank-sky row count changed: {corrected_rows} vs {source_rows}"
        )
    record = {
        "source": str(background),
        "source_sha256": sha256(background),
        "science_geometry_source": str(source),
        "science_geometry_source_sha256": sha256(source),
        "corrected": str(destination),
        "corrected_sha256": sha256(destination),
        "source_rows": source_rows,
        "corrected_rows": corrected_rows,
        "copied_keywords": copied_values,
        "edit_list": str(edit_list),
        "edit_list_sha256": sha256(edit_list),
        "command": command,
        "log": str(log),
        "log_sha256": sha256(log),
        "reused": False,
    }
    marker.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def celestial_coordinate_chip(
    event_file: Path,
    aspect: Path,
    ra: float,
    dec: float,
    env: dict[str, str],
) -> int:
    subprocess.run(
        ["punlearn", "dmcoords"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    completed = subprocess.run(
        [
            "dmcoords",
            f"infile={event_file}",
            f"asolfile=@{aspect}",
            "option=cel",
            f"ra={ra:.14f}",
            f"dec={dec:.14f}",
            "celfmt=deg",
            "verbose=0",
            "mode=h",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"dmcoords celestial validation failed for {event_file}: "
            f"{completed.stdout}{completed.stderr}"
        )
    return int(float(command_text(["pget", "dmcoords", "chip_id"], env)))


def verify_blanksky_scaling(
    source_pha: Path,
    background_pha: Path,
    bkgscale: float,
    env: dict[str, str],
) -> dict:
    source_exposure = header_number(source_pha, "EXPOSURE", env)
    background_exposure = header_number(background_pha, "EXPOSURE", env)
    source_backscal = header_number(source_pha, "BACKSCAL", env)
    background_backscal = header_number(background_pha, "BACKSCAL", env)
    source_areascal = header_number(source_pha, "AREASCAL", env)
    background_areascal = header_number(background_pha, "AREASCAL", env)
    expected_background_areascal = 1.0 / bkgscale
    effective_scale = (
        source_exposure
        / background_exposure
        * source_backscal
        / background_backscal
        * source_areascal
        / background_areascal
    )
    areascal_relative_error = abs(background_areascal / expected_background_areascal - 1.0)
    if areascal_relative_error > 1e-6:
        raise RuntimeError(
            f"blank-sky AREASCAL mismatch for {background_pha}: "
            f"{background_areascal} vs {expected_background_areascal}"
        )
    return {
        "BKGSCALn": bkgscale,
        "source_exposure": source_exposure,
        "background_exposure": background_exposure,
        "source_BACKSCAL": source_backscal,
        "background_BACKSCAL": background_backscal,
        "source_AREASCAL": source_areascal,
        "background_AREASCAL": background_areascal,
        "expected_background_AREASCAL": expected_background_areascal,
        "areascal_relative_error": areascal_relative_error,
        "effective_background_scale": effective_scale,
    }


def execute_extraction_cell(task: dict, scratch: Path, namespace: str) -> dict:
    """Run one independent response extraction in a cell-private CIAO environment."""
    obsid = int(task["obsid"])
    chip = int(task["ccd_id"])
    env = isolated_environment(
        os.environ,
        scratch
        / f"pfiles_{namespace}"
        / "integrated"
        / task["cluster"]
        / f"{obsid}_ccd{chip}",
        scratch
        / f"tmp_{namespace}"
        / "integrated"
        / task["cluster"]
        / f"{obsid}_ccd{chip}",
    )
    step = run_step(
        task["command"],
        task["log"],
        [task["source_pha"], task["background_pha"], task["arf"], task["rmf"]],
        env,
    )
    scaling = verify_blanksky_scaling(
        task["source_pha"],
        task["background_pha"],
        float(task["bkgscale_value"]),
        env,
    )
    return {
        "obsid": obsid,
        "ccd_id": chip,
        "source_band_events": task["source_band_events"],
        "background_band_events": task["background_band_events"],
        "response_reference": task["response_reference"],
        "source_spectrum": str(task["source_pha"]),
        "source_spectrum_sha256": sha256(task["source_pha"]),
        "background_spectrum": str(task["background_pha"]),
        "background_spectrum_sha256": sha256(task["background_pha"]),
        "arf_sha256": sha256(task["arf"]),
        "rmf_sha256": sha256(task["rmf"]),
        "blanksky_scaling": scaling,
        "translated_fov": task["translated_fov"],
        "step": step,
    }


def copy_snapshot(source: Path, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256(source)
    if destination.exists():
        if sha256(destination) != digest:
            raise RuntimeError(f"existing integrated snapshot changed: {destination}")
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


def build_cluster(
    cluster_name: str,
    config: dict,
    region_row: dict,
    astrometry_rows: dict[int, dict],
    repro_rows: dict[int, dict],
    cleaning_rows: dict[int, dict],
    restoration: dict,
    scratch: Path,
    output: Path,
) -> dict:
    namespace = config["execution"]["work_namespace"]
    work = scratch / namespace / "integrated" / cluster_name
    logs = work / "logs"
    individual = work / "individual"
    for path in (work, logs, individual):
        path.mkdir(parents=True, exist_ok=True)
    env = isolated_environment(
        os.environ,
        scratch / f"pfiles_{namespace}" / "planning" / "integrated" / cluster_name,
        scratch / f"tmp_{namespace}" / "planning" / "integrated" / cluster_name,
    )
    regions = frozen_region_files(region_row)
    integrated_region = work / "integrated_frozen_regions.reg"
    integrated_record = write_integrated_region(regions, integrated_region)
    extracted = []
    skipped = []
    extraction_tasks = []
    background_geometry_records = []

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
        corrected_background = work / "background_geometry" / f"acisf{obsid}_blanksky_geometry.fits"
        background_geometry = prepare_background_geometry(
            science,
            background,
            corrected_background,
            env,
        )
        background_geometry_records.append(background_geometry)
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
        for key, bkgscale_value in sorted(cleaning_row["blanksky_scaling"].items()):
            chip = int(key.removeprefix("BKGSCAL"))
            source_filter = (
                f"{science}[ccd_id={chip}]"
                f"[sky=region({translated_fov})][sky=region({integrated_region})]"
            )
            background_filter = (
                f"{corrected_background}[ccd_id={chip}]"
                f"[sky=region({translated_fov})][sky=region({integrated_region})]"
            )
            source_band_events = event_count(source_filter + "[energy=500:7000]", env)
            background_band_events = event_count(background_filter + "[energy=500:7000]", env)
            if source_band_events == 0:
                skipped.append(
                    {
                        "obsid": obsid,
                        "ccd_id": chip,
                        "reason": "zero_source_band_events_after_frozen_filters",
                        "background_band_events": background_band_events,
                    }
                )
                continue
            response_reference = event_reference_coordinate(source_filter, science, env)
            if response_reference["events"] != source_band_events:
                raise RuntimeError(
                    f"response-reference event count mismatch for ObsID {obsid} "
                    f"CCD {chip}: {response_reference['events']} vs {source_band_events}"
                )
            if response_reference["dmcoords_chip_id"] != chip:
                raise RuntimeError(
                    f"response reference for ObsID {obsid} CCD {chip} maps to "
                    f"CCD {response_reference['dmcoords_chip_id']}"
                )
            source_coordinate_chip = celestial_coordinate_chip(
                science,
                aspect,
                response_reference["ra_deg"],
                response_reference["dec_deg"],
                env,
            )
            background_coordinate_chip = celestial_coordinate_chip(
                corrected_background,
                aspect,
                response_reference["ra_deg"],
                response_reference["dec_deg"],
                env,
            )
            if source_coordinate_chip != chip or background_coordinate_chip != chip:
                raise RuntimeError(
                    f"response reference for ObsID {obsid} CCD {chip} maps to "
                    f"science/background CCD {source_coordinate_chip}/{background_coordinate_chip}"
                )
            response_reference["science_aspect_chip_id"] = source_coordinate_chip
            response_reference["background_aspect_chip_id"] = background_coordinate_chip
            outroot = individual / f"acisf{obsid}_ccd{chip}_integrated"
            source_pha = outroot.with_suffix(".pi")
            background_pha = outroot.with_name(outroot.name + "_bkg.pi")
            arf = outroot.with_suffix(".arf")
            rmf = outroot.with_suffix(".rmf")
            command = [
                "specextract",
                f"infile={source_filter}",
                f"outroot={outroot}",
                f"bkgfile={background_filter}",
                f"asp=@{aspect}",
                f"mskfile={mask}",
                f"badpixfile={badpix}",
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
            extraction_tasks.append(
                {
                    "cluster": cluster_name,
                    "obsid": obsid,
                    "ccd_id": chip,
                    "source_band_events": source_band_events,
                    "background_band_events": background_band_events,
                    "response_reference": response_reference,
                    "source_pha": source_pha,
                    "background_pha": background_pha,
                    "arf": arf,
                    "rmf": rmf,
                    "bkgscale_value": float(bkgscale_value),
                    "translated_fov": fov_record,
                    "command": command,
                    "log": logs / f"{obsid}_ccd{chip}_specextract.log",
                }
            )

    if extraction_tasks:
        worker_limit = int(config["execution"]["external_parallel_cells"])
        workers = min(worker_limit, len(extraction_tasks))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(execute_extraction_cell, task, scratch, namespace)
                for task in extraction_tasks
            ]
            extracted = [future.result() for future in futures]
    source_spectra = [Path(row["source_spectrum"]) for row in extracted]
    if not source_spectra:
        raise RuntimeError(f"{cluster_name} produced no integrated spectra")
    spectra_list = work / "integrated_source_spectra.lis"
    content = "\n".join(str(path) for path in source_spectra) + "\n"
    if spectra_list.exists() and spectra_list.read_text(encoding="utf-8") != content:
        raise RuntimeError(f"existing source-spectrum stack differs: {spectra_list}")
    spectra_list.write_text(content, encoding="utf-8")
    combined_root = work / f"{cluster_name}_integrated"
    combined_source = combined_root.with_name(combined_root.name + "_src.pi")
    combined_background = combined_root.with_name(combined_root.name + "_bkg.pi")
    combined_arf = combined_root.with_name(combined_root.name + "_src.arf")
    combined_rmf = combined_root.with_name(combined_root.name + "_src.rmf")
    combine_command = [
        "combine_spectra",
        f"src_spectra=@{spectra_list}",
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
    grouped = work / f"{cluster_name}_integrated_src_grp.pi"
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
    group_step = run_step(
        group_command,
        logs / "dmgroup.log",
        [grouped],
        env,
    )
    snapshot_root = output / "frozen_integrated_spectra" / cluster_name
    snapshots = []
    for role, source in (
        ("grouped_source_spectrum", grouped),
        ("background_spectrum", combined_background),
        ("source_arf", combined_arf),
        ("source_rmf", combined_rmf),
        ("integrated_region", integrated_region),
    ):
        record = copy_snapshot(source, snapshot_root / source.name)
        record["role"] = role
        snapshots.append(record)
    return {
        "cluster": cluster_name,
        "integrated_region": integrated_record,
        "extracted_cells": len(extracted),
        "skipped_cells": skipped,
        "source_band_events": sum(row["source_band_events"] for row in extracted),
        "background_band_events": sum(row["background_band_events"] for row in extracted),
        "background_geometry": background_geometry_records,
        "extractions": extracted,
        "execution": {
            "work_namespace": namespace,
            "external_parallel_cells": int(config["execution"]["external_parallel_cells"]),
            "planned_cells": len(extraction_tasks),
        },
        "combined": {
            "source_spectra": len(source_spectra),
            "stack": str(spectra_list),
            "combine_step": combine_step,
            "group_step": group_step,
        },
        "frozen_snapshot": {
            "files": len(snapshots),
            "bytes": sum(row["bytes"] for row in snapshots),
            "products": snapshots,
        },
    }


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
    }
    loaded = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    config = loaded["config"]
    regions = loaded["regions"]
    worker_limit = int(config["execution"]["external_parallel_cells"])
    if not 1 <= worker_limit <= 4:
        raise RuntimeError(f"invalid external cell worker limit: {worker_limit}")
    if regions["status"] != "both_clusters_passed_frozen_temperature_region_gate":
        raise RuntimeError("temperature-region gate has not passed")
    for key, path_key in (
        ("reduction_config_sha256", "reduction_config"),
        ("temperature_region_report_sha256", "regions"),
        ("spatial_visual_audit_sha256", "visual_audit"),
        ("hi4pi_provenance_sha256", "hi4pi_provenance"),
        ("response_commissioning_restoration_sha256", "restoration"),
    ):
        if config["parents"][key] != sha256(paths[path_key]):
            raise RuntimeError(f"frozen parent hash mismatch: {key}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    astrometry_rows = {int(row["obsid"]): row for row in loaded["astrometry"]["observations"]}
    repro_rows = {int(row["obsid"]): row for row in loaded["repro"]["observations"]}
    cleaning_rows = {int(row["obsid"]): row for row in loaded["cleaning"]["observations"]}
    region_rows = {row["cluster"]: row for row in regions["clusters"]}
    clusters = []
    for cluster_name in config["clusters"]:
        result = build_cluster(
            cluster_name,
            config,
            region_rows[cluster_name],
            astrometry_rows,
            repro_rows,
            cleaning_rows,
            loaded["restoration"],
            args.scratch.resolve(),
            output,
        )
        clusters.append(result)
        print(
            f"{cluster_name}: combined {result['extracted_cells']} ObsID/CCD spectra "
            f"with {result['source_band_events']} source-band events",
            flush=True,
        )
    report = {
        "status": "both_frozen_integrated_spectra_extracted_combined_and_grouped",
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(paths["config"]),
        "temperature_region_report_sha256": sha256(paths["regions"]),
        "spatial_visual_audit_sha256": sha256(paths["visual_audit"]),
        "hi4pi_provenance_sha256": sha256(paths["hi4pi_provenance"]),
        "astrometry_report_sha256": sha256(paths["astrometry"]),
        "repro_report_sha256": sha256(paths["repro"]),
        "cleaning_report_sha256": sha256(paths["cleaning"]),
        "restoration_report_sha256": sha256(paths["restoration"]),
        "clusters": clusters,
        "temperature_or_abundance_fit_run": False,
        "thermal_stress_constructed": False,
        "lensing_target_opened": False,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
