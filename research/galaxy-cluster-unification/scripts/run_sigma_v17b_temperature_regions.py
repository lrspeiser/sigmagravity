#!/usr/bin/env python3
"""Build frozen common Chandra contour bins for the Sigma v17 stress test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pycrates

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v17b_temperature_regions.json"
DEFAULT_REDUCTION = ROOT / "configs" / "sigma_v17a_chandra_reduction.json"
DEFAULT_ASTROMETRY = ROOT / "results" / "sigma_v17a2_hierarchical_astrometry" / "report.json"
DEFAULT_CLEANING = ROOT / "results" / "sigma_v17a_chandra_cleaning" / "report.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v17b_temperature_regions"
DEFAULT_SCRATCH = Path("/home/henry/sigma-v17a-chandra")
DEFAULT_CONTBIN = Path("/home/henry/sigma-v17a-tools/contbin")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_file(source: Path, destination: Path, role: str) -> dict:
    """Copy a frozen small product once, or verify an identical existing copy."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(source)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise RuntimeError(f"frozen snapshot size differs: {destination}")
        if sha256(destination) != source_hash:
            raise RuntimeError(f"frozen snapshot hash differs: {destination}")
        reused = True
    else:
        shutil.copy2(source, destination)
        reused = False
    try:
        relative_path = destination.relative_to(ROOT).as_posix()
    except ValueError:
        relative_path = str(destination)
    return {
        "role": role,
        "relative_path": relative_path,
        "bytes": destination.stat().st_size,
        "sha256": source_hash,
        "reused": reused,
    }


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
        raise RuntimeError(f"partial outputs exist for {log}")
    result = subprocess.run(command, check=False, capture_output=True, text=True, env=env)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"command failed; see {log}")
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError(f"command did not create expected outputs: {missing}")
    return {"command": command, "reused": False, "log": str(log)}


def write_exact(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"existing deterministic file differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def dmcoords(
    image: Path, option: str, values: dict[str, float], env: dict[str, str]
) -> dict[str, float]:
    command = [
        "dmcoords",
        str(image),
        f"option={option}",
        "celfmt=deg",
        "verbose=0",
        "mode=h",
        *(f"{key}={value:.12f}" for key, value in values.items()),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, env=env)
    outputs = {}
    for key in ("x", "y", "logicalx", "logicaly", "ra", "dec"):
        result = subprocess.run(
            ["pget", "dmcoords", key],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        outputs[key] = float(result.stdout.strip())
    return outputs


def exact_sky_grid(image: Path, env: dict[str, str]) -> tuple[str, dict[str, float]]:
    subprocess.run(
        ["get_sky_limits", str(image), "verbose=0", "mode=h"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    result = subprocess.run(
        ["pget", "get_sky_limits", "xygrid"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    grid = result.stdout.strip()
    xpart, ypart = grid.split(",")
    xfields = xpart.split(":")
    yfields = ypart.split(":")
    if len(xfields) != 3 or len(yfields) != 3:
        raise RuntimeError(f"unexpected get_sky_limits grid: {grid}")
    return grid, {
        "xlo": float(xfields[0]),
        "xhi": float(xfields[1]),
        "ylo": float(yfields[0]),
        "yhi": float(yfields[1]),
    }


def image_values(path: Path) -> np.ndarray:
    return np.asarray(pycrates.read_file(str(path)).get_image().values, dtype=float)


def image_expression(
    inputs: list[Path], output: Path, expression: str, log: Path, env: dict[str, str]
) -> dict:
    return run_step(
        [
            "dmimgcalc",
            f"infile={','.join(str(path) for path in inputs)}",
            "infile2=none",
            f"outfile={output}",
            f"operation=imgout={expression}",
            "clobber=no",
            "verbose=1",
            "mode=h",
        ],
        log,
        [output],
        env,
    )


def weighted_image_sum(
    inputs: list[Path],
    weights: list[float],
    output: Path,
    prefix: str,
    work: Path,
    logs: Path,
    env: dict[str, str],
) -> dict:
    if len(inputs) != len(weights) or not inputs:
        raise RuntimeError("weighted image sum has inconsistent or empty inputs")
    terms = []
    term_steps = []
    for index, (path, weight) in enumerate(zip(inputs, weights, strict=True)):
        term = work / f"{prefix}_term_{index:03d}.img"
        term_steps.append(
            image_expression(
                [path],
                term,
                f"img1*{weight:.17g}",
                logs / f"{prefix}_term_{index:03d}.log",
                env,
            )
        )
        terms.append(term)
    reduction_steps = []
    level = 0
    while len(terms) > 1:
        next_terms = []
        pair_count = (len(terms) + 1) // 2
        for index in range(pair_count):
            left = terms[2 * index]
            if 2 * index + 1 >= len(terms):
                next_terms.append(left)
                continue
            right = terms[2 * index + 1]
            destination = (
                output if len(terms) == 2 else work / f"{prefix}_level_{level:02d}_{index:03d}.img"
            )
            reduction_steps.append(
                image_expression(
                    [left, right],
                    destination,
                    "img1+img2",
                    logs / f"{prefix}_level_{level:02d}_{index:03d}.log",
                    env,
                )
            )
            next_terms.append(destination)
        terms = next_terms
        level += 1
    if terms[0] != output:
        reduction_steps.append(
            run_step(
                ["dmcopy", str(terms[0]), str(output), "clobber=no", "mode=h"],
                logs / f"{prefix}_final_copy.log",
                [output],
                env,
            )
        )
    return {"term_steps": term_steps, "reduction_steps": reduction_steps}


def centroid(
    science: np.ndarray,
    background: np.ndarray,
    exposure: np.ndarray,
    initial_x: float,
    initial_y: float,
    radius_pixels: float,
    relative_exposure_minimum: float,
    tolerance: float,
    maximum_iterations: int,
) -> tuple[float, float, list[dict]]:
    yy, xx = np.indices(science.shape, dtype=float)
    xx += 1.0
    yy += 1.0
    valid_exposure = (
        np.isfinite(exposure)
        & (exposure >= relative_exposure_minimum * float(np.nanmax(exposure)))
        & (exposure > 0)
    )
    weights = np.zeros_like(science, dtype=float)
    weights[valid_exposure] = np.maximum(
        (science[valid_exposure] - background[valid_exposure]) / exposure[valid_exposure],
        0.0,
    )
    current_x = initial_x
    current_y = initial_y
    history = []
    for iteration in range(1, maximum_iterations + 1):
        aperture = (xx - current_x) ** 2 + (yy - current_y) ** 2 <= radius_pixels**2
        selected = aperture & valid_exposure & np.isfinite(weights) & (weights > 0)
        total = float(np.sum(weights[selected]))
        if total <= 0:
            raise RuntimeError("centroid aperture has no positive net exposure-corrected flux")
        next_x = float(np.sum(xx[selected] * weights[selected]) / total)
        next_y = float(np.sum(yy[selected] * weights[selected]) / total)
        displacement = math.hypot(next_x - current_x, next_y - current_y)
        history.append(
            {
                "iteration": iteration,
                "logical_x": next_x,
                "logical_y": next_y,
                "displacement_output_pixels": displacement,
                "positive_weight_pixels": int(np.count_nonzero(selected)),
                "weight_sum": total,
            }
        )
        current_x, current_y = next_x, next_y
        if displacement < tolerance:
            return current_x, current_y, history
    raise RuntimeError("diffuse-emission centroid did not converge")


def inventory(directory: Path) -> list[dict]:
    return [
        {
            "relative_path": path.relative_to(directory).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(item for item in directory.rglob("*") if item.is_file())
    ]


def support_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern} in {directory}, found {matches}")
    return matches[0]


def build_cluster(
    cluster: str,
    config: dict,
    reduction: dict,
    astrometry: dict,
    cleaning: dict,
    scratch: Path,
    contbin_root: Path,
) -> dict:
    work = scratch / "temperature_regions_v17b_v102" / cluster
    events_dir = work / "events"
    images_dir = work / "images"
    regions_dir = work / "regions"
    logs = work / "logs"
    for path in (events_dir, images_dir, regions_dir, logs):
        path.mkdir(parents=True, exist_ok=True)
    env = isolated_environment(
        os.environ,
        scratch / "pfiles_temperature_regions" / cluster,
        scratch / "tmp_temperature_regions" / cluster,
    )
    rows = sorted(
        (row for row in astrometry["observations"] if row["cluster"] == cluster),
        key=lambda row: int(row["obsid"]),
    )
    cleaning_rows = {int(row["obsid"]): row for row in cleaning["observations"]}
    reference_obsid = int(config["registered_input"]["reference_obsids"][cluster])
    reference_row = next(row for row in rows if int(row["obsid"]) == reference_obsid)
    reference_event = Path(reference_row["application"]["corrected_events"]["science"]["path"])

    coord = config["coordinates"]["clusters"][cluster]
    initial = dmcoords(
        reference_event,
        "cel",
        {"ra": coord["initial_center_ra_deg"], "dec": coord["initial_center_dec_deg"]},
        env,
    )
    native_pixel = float(config["coordinates"]["native_pixel_arcsec"])
    half_native = math.ceil(
        config["coordinates"]["grid_half_width_kpc"]
        / coord["kpc_per_arcsec_Planck18"]
        / native_pixel
    )
    binsize = int(config["coordinates"]["image_binsize_native_pixels"])
    xlo = math.floor(initial["x"] - half_native) + 0.5
    ylo = math.floor(initial["y"] - half_native) + 0.5
    nx = math.ceil((2 * half_native) / binsize)
    ny = nx
    xhi = xlo + nx * binsize
    yhi = ylo + ny * binsize
    xygrid = f"{xlo}:{xhi}:{binsize},{ylo}:{yhi}:{binsize}"

    reproj_science = []
    reproj_blanksky = []
    asol_files = []
    badpix_files = []
    mask_files = []
    observation_steps = []
    for row in rows:
        obsid = int(row["obsid"])
        cleaning_row = cleaning_rows[obsid]
        source_region = (
            Path(cleaning_row["clean_event"]).parent
            / "source_detect_b2"
            / "point_sources_expanded.reg"
        )
        science = Path(row["application"]["corrected_events"]["science"]["path"])
        blanksky = Path(row["application"]["corrected_events"]["blanksky"]["path"])
        science_nosrc = events_dir / f"acisf{obsid}_science_nosrc_evt.fits"
        blanksky_nosrc = events_dir / f"acisf{obsid}_blanksky_nosrc_evt.fits"
        science_mask_step = run_step(
            [
                "dmcopy",
                f"{science}[exclude sky=region({source_region})]",
                str(science_nosrc),
                "clobber=no",
                "mode=h",
            ],
            logs / f"{obsid}_science_source_exclusion.log",
            [science_nosrc],
            env,
        )
        blanksky_mask_step = run_step(
            [
                "dmcopy",
                f"{blanksky}[exclude sky=region({source_region})]",
                str(blanksky_nosrc),
                "clobber=no",
                "mode=h",
            ],
            logs / f"{obsid}_blanksky_source_exclusion.log",
            [blanksky_nosrc],
            env,
        )
        science_reproj = events_dir / f"acisf{obsid}_science_nosrc_reproj_evt.fits"
        blanksky_reproj = events_dir / f"acisf{obsid}_blanksky_nosrc_reproj_evt.fits"
        science_reproj_step = run_step(
            [
                "reproject_events",
                f"infile={science_nosrc}",
                f"outfile={science_reproj}",
                f"match={reference_event}",
                "aspect=none",
                "random=-1",
                "clobber=no",
                "verbose=1",
                "mode=h",
            ],
            logs / f"{obsid}_science_reproject.log",
            [science_reproj],
            env,
        )
        blanksky_reproj_step = run_step(
            [
                "reproject_events",
                f"infile={blanksky_nosrc}",
                f"outfile={blanksky_reproj}",
                f"match={reference_event}",
                "aspect=none",
                f"random={obsid}",
                "clobber=no",
                "verbose=1",
                "mode=h",
            ],
            logs / f"{obsid}_blanksky_reproject.log",
            [blanksky_reproj],
            env,
        )
        reproj_science.append(science_reproj)
        reproj_blanksky.append(blanksky_reproj)
        corrected_aspects = [Path(item["path"]) for item in row["application"]["corrected_aspects"]]
        if len(corrected_aspects) != 1:
            raise RuntimeError(f"flux_obs support expects one aspect for ObsID {obsid}")
        asol_files.append(corrected_aspects[0])
        repro_dir = scratch / "repro" / cluster / str(obsid)
        badpix_files.append(support_file(repro_dir, "*repro_bpix1.fits"))
        mask_files.append(support_file(repro_dir, "*_msk1.fits"))
        observation_steps.append(
            {
                "obsid": obsid,
                "source_region": str(source_region),
                "science_reprojected": str(science_reproj),
                "blanksky_reprojected": str(blanksky_reproj),
                "steps": {
                    "science_source_exclusion": science_mask_step,
                    "blanksky_source_exclusion": blanksky_mask_step,
                    "science_reprojection": science_reproj_step,
                    "blanksky_reprojection": blanksky_reproj_step,
                },
            }
        )

    science_list = work / "science_reprojected.lis"
    asol_list = work / "corrected_aspects.lis"
    badpix_list = work / "badpix.lis"
    mask_list = work / "detector_masks.lis"
    write_exact(science_list, "\n".join(str(path) for path in reproj_science) + "\n")
    write_exact(asol_list, "\n".join(str(path) for path in asol_files) + "\n")
    write_exact(badpix_list, "\n".join(str(path) for path in badpix_files) + "\n")
    write_exact(mask_list, "\n".join(str(path) for path in mask_files) + "\n")

    flux_dir = images_dir / "flux"
    flux_dir.mkdir(parents=True, exist_ok=True)
    science_counts = flux_dir / "0.5-7.0_thresh.img"
    exposure_map = flux_dir / "0.5-7.0_thresh.expmap"
    flux_step = run_step(
        [
            "flux_obs",
            f"infiles=@{science_list}",
            f"outroot={flux_dir}/",
            "bands=0.5:7.0:2.3",
            f"xygrid={xygrid}",
            f"asolfiles=@{asol_list}",
            f"badpixfiles=@{badpix_list}",
            f"maskfiles=@{mask_list}",
            "background=none",
            "parallel=no",
            "cleanup=yes",
            "clobber=no",
            "verbose=1",
            "mode=h",
        ],
        logs / "flux_obs.log",
        [science_counts, exposure_map],
        env,
    )
    actual_xygrid, actual_grid = exact_sky_grid(science_counts, env)

    background_images = []
    background_scales = []
    for path, row in zip(reproj_blanksky, rows, strict=True):
        obsid = int(row["obsid"])
        scales = cleaning_rows[obsid]["blanksky_scaling"]
        for keyword, value in sorted(scales.items()):
            chip = int(keyword.removeprefix("BKGSCAL"))
            image = images_dir / f"acisf{obsid}_ccd{chip}_blanksky_counts.img"
            run_step(
                [
                    "dmcopy",
                    f"{path}[ccd_id={chip},energy=500:7000][bin x={actual_xygrid.split(',')[0]},y={actual_xygrid.split(',')[1]}]",
                    str(image),
                    "clobber=no",
                    "mode=h",
                ],
                logs / f"{obsid}_ccd{chip}_blanksky_image.log",
                [image],
                env,
            )
            background_images.append(image)
            background_scales.append(float(value))
    if not background_images:
        raise RuntimeError("no blank-sky chip images were produced")

    scaled_background = images_dir / "scaled_blanksky_counts.img"
    background_variance = images_dir / "scaled_blanksky_variance.img"
    background_step = weighted_image_sum(
        background_images,
        background_scales,
        scaled_background,
        "scaled_background",
        images_dir,
        logs,
        env,
    )
    variance_step = weighted_image_sum(
        background_images,
        [scale * scale for scale in background_scales],
        background_variance,
        "background_variance",
        images_dir,
        logs,
        env,
    )
    noise_map = images_dir / "poisson_noise.img"
    noise_step = image_expression(
        [science_counts, background_variance],
        noise_map,
        "sqrt(fabs(img1+img2))",
        logs / "noise_map.log",
        env,
    )

    initial_image = dmcoords(
        science_counts,
        "cel",
        {"ra": coord["initial_center_ra_deg"], "dec": coord["initial_center_dec_deg"]},
        env,
    )
    radius_output_pixels = (
        config["coordinates"]["analysis_radius_kpc"]
        / coord["kpc_per_arcsec_Planck18"]
        / config["coordinates"]["output_pixel_arcsec"]
    )
    centroid_config = config["centroid"]
    center_x, center_y, center_history = centroid(
        image_values(science_counts),
        image_values(scaled_background),
        image_values(exposure_map),
        initial_image["logicalx"],
        initial_image["logicaly"],
        radius_output_pixels,
        config["science_image"]["minimum_relative_exposure"],
        centroid_config["convergence_output_pixels"],
        centroid_config["maximum_iterations"],
    )
    final_center = dmcoords(
        science_counts,
        "logical",
        {"logicalx": center_x, "logicaly": center_y},
        env,
    )

    exposure_threshold = config["science_image"]["minimum_relative_exposure"] * float(
        np.nanmax(image_values(exposure_map))
    )
    radius_native_pixels = (
        config["coordinates"]["analysis_radius_kpc"]
        / coord["kpc_per_arcsec_Planck18"]
        / native_pixel
    )
    aperture_exposure = images_dir / "aperture_exposure.img"
    aperture_step = run_step(
        [
            "dmcopy",
            f"{exposure_map}[sky=circle({final_center['x']},{final_center['y']},{radius_native_pixels})][opt full]",
            str(aperture_exposure),
            "clobber=no",
            "mode=h",
        ],
        logs / "aperture_exposure.log",
        [aperture_exposure],
        env,
    )
    analysis_mask = images_dir / "analysis_mask.img"
    # CIAO 4.18 dmimgcalc cannot emit a Boolean image directly: casting a
    # comparison produces a null 0-D dataset.  This numerical Heaviside is the
    # same frozen >= threshold, with an inclusion epsilon far below real4
    # exposure-map resolution so exact-threshold pixels remain included.
    threshold_epsilon = max(abs(exposure_threshold) * 1e-12, np.finfo(float).eps)
    threshold_delta = f"(img1-{exposure_threshold:.17g}+{threshold_epsilon:.17g})"
    mask_step = image_expression(
        [aperture_exposure],
        analysis_mask,
        f"(short)((1+({threshold_delta}/fabs({threshold_delta})))/2)",
        logs / "analysis_mask.log",
        env,
    )
    mask_values = image_values(analysis_mask)
    if not np.array_equal(np.unique(np.nan_to_num(mask_values, nan=0.0)), [0.0, 1.0]):
        raise RuntimeError("analysis mask is not binary")

    binning = config["contour_binning"]
    contbin = contbin_root / "contbin"
    make_regions = contbin_root / "make_region_files"
    if sha256(contbin) != binning["contbin_sha256"]:
        raise RuntimeError("contbin executable hash mismatch")
    if sha256(make_regions) != binning["make_region_files_sha256"]:
        raise RuntimeError("make_region_files executable hash mismatch")
    binned_image = images_dir / "contbin_binned_counts.fits"
    sn_image = images_dir / "contbin_signal_to_noise.fits"
    binmap = images_dir / "contbin_binmap.fits"
    contbin_step = run_step(
        [
            str(contbin),
            str(science_counts),
            f"--bg={scaled_background}",
            f"--expmap={exposure_map}",
            f"--bgexpmap={exposure_map}",
            f"--noisemap={noise_map}",
            f"--mask={analysis_mask}",
            f"--sn={binning['target_signal_to_noise']}",
            f"--smoothsn={binning['smoothing_signal_to_noise']}",
            "--constrainfill",
            f"--constrainval={binning['geometric_constraint_factor']}",
            f"--out={binned_image}",
            f"--outsn={sn_image}",
            f"--outbinmap={binmap}",
        ],
        logs / "contbin.log",
        [binned_image, sn_image, binmap],
        env,
    )
    region_files = sorted(regions_dir.glob("*.reg"))
    region_log = logs / "make_region_files.log"
    if region_files:
        make_region_step = {"reused": True, "log": str(region_log)}
    else:
        completed = subprocess.run(
            [
                str(make_regions),
                f"--minx={actual_grid['xlo']}",
                f"--miny={actual_grid['ylo']}",
                f"--bin={binsize}",
                f"--outdir={regions_dir}/",
                str(binmap),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        region_log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(f"make_region_files failed; see {region_log}")
        region_files = sorted(regions_dir.glob("*.reg"))
        if not region_files:
            raise RuntimeError("make_region_files produced no regions")
        make_region_step = {"reused": False, "log": str(region_log)}

    source = image_values(science_counts)
    background = image_values(scaled_background)
    variance = image_values(background_variance)
    bins = image_values(binmap)
    mask = np.nan_to_num(image_values(analysis_mask), nan=0.0) > 0
    region_stats = []
    for bin_id in sorted(int(value) for value in np.unique(bins[mask]) if value >= 0):
        selected = mask & (bins == bin_id)
        source_counts = float(np.sum(source[selected]))
        background_counts = float(np.sum(background[selected]))
        background_variance_counts = float(np.sum(variance[selected]))
        net_counts = source_counts - background_counts
        noise = math.sqrt(max(source_counts + background_variance_counts, 0.0))
        signal_to_noise = net_counts / noise if noise > 0 else float("nan")
        source_fraction = net_counts / source_counts if source_counts > 0 else float("nan")
        gates = {
            "target_signal_to_noise": signal_to_noise >= binning["target_signal_to_noise"],
            "minimum_net_counts": net_counts >= binning["minimum_net_counts_per_region"],
            "minimum_source_fraction": source_fraction >= binning["minimum_source_fraction"],
        }
        region_stats.append(
            {
                "bin_id": bin_id,
                "pixels": int(np.count_nonzero(selected)),
                "science_counts": source_counts,
                "scaled_background_counts": background_counts,
                "background_variance": background_variance_counts,
                "net_counts": net_counts,
                "signal_to_noise": signal_to_noise,
                "source_fraction": source_fraction,
                "valid": all(gates.values()),
                **{f"gate_{key}": value for key, value in gates.items()},
            }
        )
    stats_path = work / "region_statistics.csv"
    fieldnames = list(region_stats[0]) if region_stats else []
    with stats_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(region_stats)
    valid_regions = sum(row["valid"] for row in region_stats)
    cluster_gates = {
        "minimum_total_regions": len(region_stats) >= binning["minimum_valid_regions"],
        "minimum_valid_regions": valid_regions >= binning["minimum_valid_regions"],
        "centroid_converged": center_history[-1]["displacement_output_pixels"]
        < centroid_config["convergence_output_pixels"],
    }
    products = inventory(work)
    return {
        "cluster": cluster,
        "reference_obsid": reference_obsid,
        "requested_xygrid": xygrid,
        "xygrid": actual_xygrid,
        "grid": {**actual_grid, "binsize": binsize},
        "initial_center": initial_image,
        "final_center": final_center,
        "centroid_history": center_history,
        "analysis_radius_output_pixels": radius_output_pixels,
        "exposure_threshold": exposure_threshold,
        "observation_steps": observation_steps,
        "background_chip_images": len(background_images),
        "background_scales": background_scales,
        "images": {
            "science_counts": str(science_counts),
            "exposure_map": str(exposure_map),
            "scaled_background": str(scaled_background),
            "background_variance": str(background_variance),
            "noise_map": str(noise_map),
            "analysis_mask": str(analysis_mask),
            "binned_image": str(binned_image),
            "signal_to_noise": str(sn_image),
            "binmap": str(binmap),
        },
        "region_statistics": str(stats_path),
        "region_count": len(region_stats),
        "valid_region_count": valid_regions,
        "minimum_net_counts": min(row["net_counts"] for row in region_stats),
        "minimum_signal_to_noise": min(row["signal_to_noise"] for row in region_stats),
        "minimum_source_fraction": min(row["source_fraction"] for row in region_stats),
        "gates": cluster_gates,
        "steps": {
            "flux_obs": flux_step,
            "scaled_background": background_step,
            "background_variance": variance_step,
            "noise_map": noise_step,
            "aperture_exposure": aperture_step,
            "analysis_mask": mask_step,
            "contbin": contbin_step,
            "make_region_files": make_region_step,
        },
        "product_files": len(products),
        "product_bytes": sum(row["bytes"] for row in products),
        "products": products,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reduction", type=Path, default=DEFAULT_REDUCTION)
    parser.add_argument("--astrometry", type=Path, default=DEFAULT_ASTROMETRY)
    parser.add_argument("--cleaning", type=Path, default=DEFAULT_CLEANING)
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--contbin", type=Path, default=DEFAULT_CONTBIN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config_path = args.config.resolve()
    reduction_path = args.reduction.resolve()
    astrometry_path = args.astrometry.resolve()
    cleaning_path = args.cleaning.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    reduction = json.loads(reduction_path.read_text(encoding="utf-8"))
    astrometry = json.loads(astrometry_path.read_text(encoding="utf-8"))
    cleaning = json.loads(cleaning_path.read_text(encoding="utf-8"))
    if astrometry["status"] != ("all_frozen_observations_hierarchically_registered_to_Gaia_DR3"):
        raise RuntimeError("hierarchical astrometry has not passed")
    if (
        config["coordinates"]["analysis_radius_kpc"]
        != reduction["common_map"]["analysis_radius_kpc"]
    ):
        raise RuntimeError("temperature-region aperture differs from reduction freeze")

    clusters = []
    for cluster in config["coordinates"]["clusters"]:
        result = build_cluster(
            cluster,
            config,
            reduction,
            astrometry,
            cleaning,
            args.scratch.resolve(),
            args.contbin.resolve(),
        )
        clusters.append(result)
        print(
            f"{cluster}: {result['valid_region_count']}/{result['region_count']} valid "
            f"regions; min net={result['minimum_net_counts']:.1f}, "
            f"min S/N={result['minimum_signal_to_noise']:.2f}",
            flush=True,
        )

    failed = [row["cluster"] for row in clusters if not all(row["gates"].values())]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for row in clusters:
        cluster_output = output / "frozen_region_products" / row["cluster"]
        snapshot = []
        for role, source_text in sorted(row["images"].items()):
            source = Path(source_text)
            snapshot.append(
                snapshot_file(
                    source,
                    cluster_output / "images" / f"{role}{source.suffix}",
                    role,
                )
            )
        statistics = Path(row["region_statistics"])
        snapshot.append(
            snapshot_file(
                statistics,
                cluster_output / "regions" / "region_statistics.csv",
                "region_statistics",
            )
        )
        for source in sorted((statistics.parent / "regions").glob("*.reg")):
            snapshot.append(
                snapshot_file(
                    source,
                    cluster_output / "regions" / source.name,
                    "spectral_region",
                )
            )
        row["frozen_snapshot"] = {
            "files": len(snapshot),
            "bytes": sum(item["bytes"] for item in snapshot),
            "products": snapshot,
        }
    report = {
        "status": (
            "both_clusters_passed_frozen_temperature_region_gate"
            if not failed
            else "frozen_temperature_region_gate_failed"
        ),
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "reduction_config_sha256": sha256(reduction_path),
        "astrometry_report_sha256": sha256(astrometry_path),
        "cleaning_report_sha256": sha256(cleaning_path),
        "failed_clusters": failed,
        "clusters": clusters,
        "event_images_visually_inspected": False,
        "temperature_map_constructed": False,
        "lensing_target_opened": False,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report_path)
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
