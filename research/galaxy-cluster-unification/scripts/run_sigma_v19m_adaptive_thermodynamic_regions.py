#!/usr/bin/env python3
"""Build the frozen V19M adaptive thermodynamic regions under CIAO/WSL."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pycrates

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19m_adaptive_thermodynamic_regions.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19m_adaptive_thermodynamic_regions"
DEFAULT_SCRATCH = Path("/home/henry/sigma-v19m-thermodynamics/regions_v100")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_parent_hashes(config: dict[str, Any]) -> None:
    parents = config["parents"]
    for key, value in parents.items():
        if key.endswith("_sha256"):
            continue
        expected = parents.get(f"{key}_sha256")
        if expected is None:
            continue
        path = ROOT / value
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"V19M parent hash mismatch: {path}")


def run_step(
    command: list[str],
    log: Path,
    expected: list[Path],
    cwd: Path | None = None,
) -> dict[str, Any]:
    present = [path.is_file() for path in expected]
    if all(present):
        if not log.is_file():
            raise RuntimeError(f"complete outputs lack log: {log}")
        return {"command": command, "log": str(log), "reused": True}
    if any(present):
        raise RuntimeError(f"partial outputs exist for {log}")
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True, cwd=cwd
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"command failed; see {log}")
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError(f"command did not create expected products: {missing}")
    return {"command": command, "log": str(log), "reused": False}


def image_values(path: Path) -> np.ndarray:
    return np.asarray(pycrates.read_file(str(path)).get_image().values, dtype=float)


def product(cluster: dict[str, Any], role: str) -> tuple[Path, dict[str, Any]]:
    rows = [row for row in cluster["frozen_snapshot"]["products"] if row["role"] == role]
    if len(rows) != 1:
        raise RuntimeError(f"expected one {role} product for {cluster['cluster']}")
    row = rows[0]
    path = ROOT / row["relative_path"]
    if path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
        raise RuntimeError(f"frozen V19H product changed: {path}")
    return path, row


def snapshot(source: Path, destination: Path, role: str) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256(source)
    if destination.exists():
        if sha256(destination) != digest:
            raise RuntimeError(f"existing snapshot differs: {destination}")
        reused = True
    else:
        shutil.copy2(source, destination)
        reused = False
    return {
        "role": role,
        "relative_path": destination.relative_to(ROOT).as_posix(),
        "bytes": destination.stat().st_size,
        "sha256": digest,
        "reused": reused,
    }


def write_region_statistics(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_cluster(
    config: dict[str, Any], source_row: dict[str, Any], scratch: Path, output: Path
) -> dict[str, Any]:
    cluster = source_row["cluster"]
    work = scratch / cluster
    images = work / "images"
    regions = work / "regions"
    logs = work / "logs"
    for path in (images, regions, logs):
        path.mkdir(parents=True, exist_ok=True)

    counts, _ = product(source_row, "broad_counts")
    background, _ = product(source_row, "broad_scaled_background")
    variance, _ = product(source_row, "broad_background_variance")
    exposure, _ = product(source_row, "broad_exposure")
    mask, _ = product(source_row, "analysis_mask")

    noise = images / "poisson_noise.img"
    noise_step = run_step(
        [
            "dmimgcalc",
            f"infile={counts},{variance}",
            "infile2=none",
            f"outfile={noise}",
            "operation=imgout=sqrt(fabs(img1+img2))",
            "clobber=no",
            "verbose=1",
            "mode=h",
        ],
        logs / "noise_map.log",
        [noise],
    )

    runtime = config["runtime"]
    contbin_root = Path(runtime["contbin_root"])
    contbin = contbin_root / "contbin"
    make_regions = contbin_root / "make_region_files"
    if sha256(contbin) != runtime["contbin_sha256"]:
        raise RuntimeError("V19M contbin executable hash mismatch")
    if sha256(make_regions) != runtime["make_region_files_sha256"]:
        raise RuntimeError("V19M make_region_files executable hash mismatch")

    rules = config["spatial_binning"]
    binned = images / "contbin_binned_counts.fits"
    signal_to_noise = images / "contbin_signal_to_noise.fits"
    binmap = images / "contbin_binmap.fits"
    contbin_step = run_step(
        [
            str(contbin),
            str(counts),
            f"--bg={background}",
            f"--expmap={exposure}",
            f"--bgexpmap={exposure}",
            f"--noisemap={noise}",
            f"--mask={mask}",
            f"--sn={rules['target_signal_to_noise']}",
            f"--smoothsn={rules['smoothing_signal_to_noise']}",
            "--constrainfill",
            f"--constrainval={rules['geometric_constraint_factor']}",
            f"--out={binned}",
            f"--outsn={signal_to_noise}",
            f"--outbinmap={binmap}",
        ],
        logs / "contbin.log",
        [binned, signal_to_noise, binmap],
        cwd=work,
    )

    grid = source_row["grid"]
    region_files = sorted(regions.glob("*.reg"))
    region_log = logs / "make_region_files.log"
    if region_files:
        make_region_step = {"log": str(region_log), "reused": True}
    else:
        completed = subprocess.run(
            [
                str(make_regions),
                f"--minx={grid['xlo']}",
                f"--miny={grid['ylo']}",
                f"--bin={grid['binsize']}",
                f"--outdir={regions}/",
                str(binmap),
            ],
            check=False,
            capture_output=True,
            text=True,
            cwd=work,
        )
        region_log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(f"make_region_files failed; see {region_log}")
        region_files = sorted(regions.glob("*.reg"))
        if not region_files:
            raise RuntimeError("make_region_files produced no regions")
        make_region_step = {"log": str(region_log), "reused": False}

    science_values = image_values(counts)
    background_values = image_values(background)
    variance_values = image_values(variance)
    bin_values = image_values(binmap)
    mask_values = np.nan_to_num(image_values(mask), nan=0.0) > 0.5
    stats = []
    valid_ids = []
    for bin_id in sorted(
        int(value) for value in np.unique(bin_values[mask_values]) if value >= 0
    ):
        selected = mask_values & (bin_values == bin_id)
        source_counts = float(np.sum(science_values[selected]))
        background_counts = float(np.sum(background_values[selected]))
        background_variance = float(np.sum(variance_values[selected]))
        net_counts = source_counts - background_counts
        noise_counts = math.sqrt(max(source_counts + background_variance, 0.0))
        sn = net_counts / noise_counts if noise_counts > 0.0 else float("nan")
        source_fraction = net_counts / source_counts if source_counts > 0.0 else float("nan")
        gates = {
            "target_signal_to_noise": sn >= float(rules["target_signal_to_noise"]),
            "minimum_net_counts": net_counts
            >= float(rules["minimum_net_counts_per_region"]),
            "minimum_source_fraction": source_fraction
            >= float(rules["minimum_source_fraction"]),
        }
        valid = all(gates.values())
        if valid:
            valid_ids.append(bin_id)
        stats.append(
            {
                "bin_id": bin_id,
                "pixels": int(np.count_nonzero(selected)),
                "science_counts": source_counts,
                "scaled_background_counts": background_counts,
                "background_variance": background_variance,
                "net_counts": net_counts,
                "signal_to_noise": sn,
                "source_fraction": source_fraction,
                "valid": valid,
                **{f"gate_{key}": value for key, value in gates.items()},
            }
        )

    region_by_id = {int(path.stem.rsplit("_", 1)[1]): path for path in region_files}
    missing = sorted(set(valid_ids) - set(region_by_id))
    if missing:
        raise RuntimeError(f"valid V19M bins lack region files: {missing}")

    frozen = output / "frozen_region_products" / cluster
    snapshots = [
        snapshot(noise, frozen / "images" / "noise_map.img", "noise_map"),
        snapshot(binned, frozen / "images" / "binned_image.fits", "binned_image"),
        snapshot(
            signal_to_noise,
            frozen / "images" / "signal_to_noise.fits",
            "signal_to_noise",
        ),
        snapshot(binmap, frozen / "images" / "binmap.fits", "binmap"),
    ]
    stats_path = work / "region_statistics.csv"
    write_region_statistics(stats_path, stats)
    snapshots.append(
        snapshot(
            stats_path,
            frozen / "regions" / "region_statistics.csv",
            "region_statistics",
        )
    )
    for bin_id in valid_ids:
        source = region_by_id[bin_id]
        snapshots.append(
            snapshot(source, frozen / "regions" / source.name, "spectral_region")
        )

    gates = {
        "minimum_valid_regions": len(valid_ids)
        >= int(rules["minimum_valid_regions_per_cluster"]),
        "all_admitted_regions_pass_every_gate": all(
            row["valid"]
            and row["gate_target_signal_to_noise"]
            and row["gate_minimum_net_counts"]
            and row["gate_minimum_source_fraction"]
            for row in stats
            if row["bin_id"] in valid_ids
        ),
    }
    return {
        "cluster": cluster,
        "region_count": len(stats),
        "valid_region_count": len(valid_ids),
        "valid_region_ids": valid_ids,
        "minimum_signal_to_noise": float(min(row["signal_to_noise"] for row in stats)),
        "minimum_net_counts": float(min(row["net_counts"] for row in stats)),
        "minimum_source_fraction": float(min(row["source_fraction"] for row in stats)),
        "gates": gates,
        "steps": {
            "noise_map": noise_step,
            "contbin": contbin_step,
            "make_region_files": make_region_step,
        },
        "products": snapshots,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    validate_parent_hashes(config)
    source_report = load_json(ROOT / config["parents"]["source_map_report"])
    source_rows = {row["cluster"]: row for row in source_report["clusters"]}
    clusters = [
        build_cluster(config, source_rows[name], args.scratch.resolve(), args.output.resolve())
        for name in config["sample"]["clusters"]
    ]
    passed = all(all(row["gates"].values()) for row in clusters)
    report = {
        "status": (
            "both_adaptive_thermodynamic_region_gates_passed"
            if passed
            else "adaptive_thermodynamic_region_gate_failed"
        ),
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "clusters": clusters,
        "regional_spectral_extraction_authorized": passed,
        "post_hash_visual_audit_run": False,
        "spectrum_or_response_constructed": False,
        "temperature_density_mach_or_speed_fitted": False,
        "lensing_target_opened": False,
        "gravity_formula_or_parameter_changed": False,
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"status: {report['status']}")
    for row in clusters:
        print(
            f"{row['cluster']}: {row['valid_region_count']}/{row['region_count']} "
            "regions admitted"
        )
    print(f"report: {report_path}")
    print(f"sha256: {sha256(report_path)}")


if __name__ == "__main__":
    main()
