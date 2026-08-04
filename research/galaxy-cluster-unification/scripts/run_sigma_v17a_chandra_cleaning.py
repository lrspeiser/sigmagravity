#!/usr/bin/env python3
"""Run the frozen point-source, flare, and blank-sky cleaning for Sigma v17A."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import pycrates

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v17a_chandra_reduction.json"
DEFAULT_REPRO = ROOT / "results" / "sigma_v17a_chandra_repro" / "report.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v17a_chandra_cleaning"
ELLIPSE_RE = re.compile(
    r"^ellipse\((?P<x>[^,]+),(?P<y>[^,]+),(?P<a>[^,]+),"
    r"(?P<b>[^,]+),(?P<angle>[^)]+)\)$"
)


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
            log_path.write_text("reused complete output\n", encoding="utf-8")
        return {"command": command, "reused": True, "log": str(log_path)}
    if any(present):
        raise RuntimeError(f"partial output exists for step log {log_path}")
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"command failed; see {log_path}")
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError(f"command did not create expected files: {missing}")
    return {"command": command, "reused": False, "log": str(log_path)}


def dmkeypar(path: Path, key: str, env: dict[str, str]) -> str | None:
    result = subprocess.run(
        ["dmkeypar", str(path), key, "echo+"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode:
        return None
    value = result.stdout.strip()
    return value or None


def dmlist_count(path: Path, env: dict[str, str]) -> int:
    result = subprocess.run(
        ["dmlist", str(path), "counts"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return int(result.stdout.split()[0])


def expanded_point_source_regions(
    catalog: Path,
    wavdetect_regions: Path,
    output: Path,
    expansion: float = 1.5,
    minimum_significance: float = 3.0,
) -> dict:
    crate = pycrates.read_file(str(catalog))
    significance = crate.get_column("SRC_SIGNIFICANCE").values
    net_counts = crate.get_column("NET_COUNTS").values
    regions = [
        line.strip()
        for line in wavdetect_regions.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(regions) != len(significance):
        raise RuntimeError("wavdetect catalog/region row count mismatch")

    selected = []
    for line, sig, net in zip(regions, significance, net_counts, strict=True):
        if float(sig) < minimum_significance or float(net) <= 0:
            continue
        match = ELLIPSE_RE.match(line)
        if match is None:
            raise RuntimeError(f"unsupported wavdetect region: {line}")
        values = match.groupdict()
        selected.append(
            "ellipse("
            f"{values['x']},{values['y']},"
            f"{float(values['a']) * expansion:.8f},"
            f"{float(values['b']) * expansion:.8f},"
            f"{values['angle']})"
        )
    if not selected:
        raise RuntimeError("no significant point-source regions were detected")
    output.write_text("\n".join(selected) + "\n", encoding="utf-8")
    return {
        "wavdetect_sources": len(regions),
        "selected_sources": len(selected),
        "minimum_significance": minimum_significance,
        "ellipse_expansion": expansion,
        "region_sha256": sha256(output),
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


def process_observation(row: dict, scratch: Path, config: dict) -> dict:
    cluster = row["cluster"]
    obsid = int(row["obsid"])
    repro_dir = Path(row["output_directory"])
    event = Path(row["event"]["path"])
    work = scratch / "clean" / cluster / str(obsid)
    detect = work / "source_detect_b2"
    logs = work / "logs"
    detect.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    env = isolated_environment(
        os.environ,
        scratch / "pfiles_clean" / cluster / str(obsid),
        scratch / "tmp_clean" / cluster / str(obsid),
    )

    point_config = config["point_sources"]
    bands = point_config["detection_image_energy_keV"]
    outroot = detect / "initial"
    image = detect / "initial_0.5-7.0_thresh.img"
    expmap = detect / "initial_0.5-7.0_thresh.expmap"
    psfmap = detect / "initial_0.5-7.0_thresh.psfmap"
    flux = detect / "initial_0.5-7.0_flux.img"
    fov = next(iter(repro_dir.glob("*repro_fov1.fits")))
    flux_step = run_step(
        [
            "fluximage",
            str(event),
            str(outroot),
            f"bands={bands[0]}:{bands[1]}:2.3",
            f"binsize={point_config['image_binsize_native_pixels']}",
            "psfecf=0.9",
            "parallel=no",
            "cleanup=yes",
            "clobber=no",
            "verbose=1",
            "mode=h",
        ],
        logs / "fluximage.log",
        [image, expmap, psfmap, flux],
        env,
    )

    catalog = detect / "sources.fits"
    wav_regions = detect / "sources.reg"
    wav_step = run_step(
        [
            "wavdetect",
            f"infile={image}",
            f"outfile={catalog}",
            f"scellfile={detect / 'scell.fits'}",
            f"imagefile={detect / 'recon.fits'}",
            f"defnbkgfile={detect / 'background.fits'}",
            f"expfile={expmap}",
            f"psffile={psfmap}",
            "scales=" + " ".join(str(value) for value in point_config["wavdetect_scales_pixels"]),
            f"regfile={wav_regions}",
            f"sigthresh={point_config['wavdetect_sigthresh']}",
            "clobber=no",
            "verbose=1",
            "mode=h",
        ],
        logs / "wavdetect.log",
        [catalog, wav_regions],
        env,
    )
    expanded_regions = detect / "point_sources_expanded.reg"
    if expanded_regions.exists():
        expanded_regions.unlink()
    source_summary = expanded_point_source_regions(
        catalog,
        wav_regions,
        expanded_regions,
        minimum_significance=float(point_config["minimum_source_significance"]),
    )

    flare = config["flare_filtering"]
    source_excluded_event = work / f"acisf{obsid}_nosrc_evt2.fits"
    source_mask_step = run_step(
        [
            "dmcopy",
            f"infile={event}[exclude sky=region({expanded_regions})]",
            f"outfile={source_excluded_event}",
            "clobber=no",
            "verbose=1",
            "mode=h",
        ],
        logs / "source_mask.log",
        [source_excluded_event],
        env,
    )
    lightcurve = work / "flare_lightcurve.fits"
    event_filter = (
        f"{source_excluded_event}"
        f"[energy={int(flare['lightcurve_energy_keV'][0] * 1000)}:"
        f"{int(flare['lightcurve_energy_keV'][1] * 1000)}]"
        f"[bin time=::{flare['time_bin_seconds']}]"
    )
    lc_step = run_step(
        [
            "dmextract",
            f"infile={event_filter}",
            f"outfile={lightcurve}",
            "opt=ltc1",
            "clobber=no",
            "verbose=1",
            "mode=h",
        ],
        logs / "lightcurve.log",
        [lightcurve],
        env,
    )

    gti = work / "flare_clean.gti"
    deflare_step = run_step(
        [
            "deflare",
            f"infile={lightcurve}",
            f"outfile={gti}",
            "method=sigma",
            "nsigma=3",
            "minlength=1",
            "plot=no",
            "verbose=1",
            "mode=h",
        ],
        logs / "deflare.log",
        [gti],
        env,
    )

    clean_event = work / f"acisf{obsid}_flareclean_evt2.fits"
    copy_step = run_step(
        [
            "dmcopy",
            f"infile={event}[@{gti}]",
            f"outfile={clean_event}",
            "clobber=no",
            "verbose=1",
            "mode=h",
        ],
        logs / "flare_filter.log",
        [clean_event],
        env,
    )
    original_exposure = float(dmkeypar(event, "EXPOSURE", env) or "nan")
    clean_exposure = float(dmkeypar(clean_event, "EXPOSURE", env) or "nan")
    retained_fraction = clean_exposure / original_exposure
    if retained_fraction < float(flare["minimum_retained_fraction"]):
        raise RuntimeError(f"{cluster}/{obsid} failed the flare retained-fraction gate")

    asol_list = next(iter(repro_dir.glob("*asol1.lis")))
    blanksky = work / f"acisf{obsid}_blanksky_evt.fits"
    background = config["background"]
    bkg_band = background["normalization_energy_keV"]
    blanksky_step = run_step(
        [
            "blanksky",
            f"evtfile={clean_event}",
            f"outfile={blanksky}",
            f"asolfile=@{asol_list}",
            f"weight_method={background['weight_method']}",
            f"bkgparams=[energy={int(bkg_band[0] * 1000)}:{int(bkg_band[1] * 1000)}]",
            f"random={obsid}",
            "clobber=no",
            "verbose=1",
            "mode=h",
        ],
        logs / "blanksky.log",
        [blanksky],
        env,
    )
    bkgscales = {f"BKGSCAL{chip}": dmkeypar(blanksky, f"BKGSCAL{chip}", env) for chip in range(10)}
    bkgscales = {key: value for key, value in bkgscales.items() if value is not None}
    if not bkgscales:
        raise RuntimeError(f"blank-sky scaling keywords are absent for {cluster}/{obsid}")

    products = inventory(work)
    return {
        "cluster": cluster,
        "obsid": obsid,
        "repro_event": str(event),
        "repro_event_sha256": sha256(event),
        "fov": str(fov),
        "steps": {
            "fluximage": flux_step,
            "wavdetect": wav_step,
            "source_mask": source_mask_step,
            "lightcurve": lc_step,
            "deflare": deflare_step,
            "flare_filter": copy_step,
            "blanksky": blanksky_step,
        },
        "point_sources": source_summary,
        "original_event_rows": dmlist_count(event, env),
        "clean_event": str(clean_event),
        "clean_event_rows": dmlist_count(clean_event, env),
        "original_exposure_seconds": original_exposure,
        "clean_exposure_seconds": clean_exposure,
        "retained_exposure_fraction": retained_fraction,
        "blanksky_event": str(blanksky),
        "blanksky_event_rows": dmlist_count(blanksky, env),
        "blanksky_scaling": bkgscales,
        "products": products,
        "product_files": len(products),
        "product_bytes": sum(product["bytes"] for product in products),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repro", type=Path, default=DEFAULT_REPRO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--jobs", type=int)
    args = parser.parse_args()

    config_path = args.config.resolve()
    repro_path = args.repro.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    repro = json.loads(repro_path.read_text(encoding="utf-8"))
    if repro["config_sha256"] != sha256(config_path):
        raise RuntimeError("reprocessed observations do not match the frozen protocol")
    if repro["observation_count"] != 11:
        raise RuntimeError("all 11 observations must be reprocessed before cleaning")

    scratch = args.scratch.resolve()
    if str(scratch) != repro["scratch"]:
        raise RuntimeError("cleaning scratch root must match the repro provenance")
    jobs = args.jobs or int(config["event_reprocessing"]["parallel_observations"])
    if jobs < 1 or jobs > 2:
        raise ValueError("the frozen v17A protocol permits one or two parallel observations")

    completed = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(process_observation, row, scratch, config): (
                row["cluster"],
                row["obsid"],
            )
            for row in repro["observations"]
        }
        for future in as_completed(futures):
            cluster, obsid = futures[future]
            completed.append(future.result())
            print(f"cleaned {cluster}/{obsid}", flush=True)
    completed.sort(key=lambda row: (row["cluster"], row["obsid"]))

    report = {
        "status": "all_frozen_chandra_observations_flare_cleaned_with_blanksky",
        "generated_utc": datetime.now(UTC).isoformat(),
        "protocol_version": config["protocol_version"],
        "config_sha256": sha256(config_path),
        "repro_report_sha256": sha256(repro_path),
        "scratch": str(scratch),
        "jobs": jobs,
        "observations": completed,
        "observation_count": len(completed),
        "clean_exposure_seconds": sum(row["clean_exposure_seconds"] for row in completed),
        "minimum_retained_exposure_fraction": min(
            row["retained_exposure_fraction"] for row in completed
        ),
        "product_files": sum(row["product_files"] for row in completed),
        "product_bytes": sum(row["product_bytes"] for row in completed),
        "astrometry_completed": False,
        "event_images_visually_inspected": False,
        "lensing_target_opened": False,
        "temperature_map_constructed": False,
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "observation_count": report["observation_count"],
                "clean_exposure_seconds": report["clean_exposure_seconds"],
                "minimum_retained_exposure_fraction": report["minimum_retained_exposure_fraction"],
                "product_files": report["product_files"],
                "product_bytes": report["product_bytes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
