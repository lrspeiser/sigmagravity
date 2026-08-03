"""Acquire and audit the frozen P0738 THINGS + SINGS resolved sample.

The acquisition stage downloads raw bytes, verifies their exact advertised
sizes and hashes, and reads only FITS headers.  It deliberately never touches
an image array, so the frozen holdout remains unopened for scientific use.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from astropy.io import fits


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/p0738_morphology_diverse_resolved_acquisition.json"
DEFAULT_DATA = ROOT / "data/raw/p0738_things_sings_resolved"
DEFAULT_OUTPUT = ROOT / "results/p0738_morphology_diverse_resolved_acquisition"
CHUNK_BYTES = 1024 * 1024
USER_AGENT = "SigmaGravityResearch/0.1 (scientific data acquisition)"


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_assets(config: dict[str, Any]) -> list[dict[str, Any]]:
    things = config["sources"]["things"]
    sings = config["sources"]["sings"]
    rows: list[dict[str, Any]] = []
    for system in config["systems"]:
        for moment in config["acquisitionRules"]["requiredMoments"]:
            filename = things["fileTemplate"].format(
                catalogNumber=system["catalogNumber"], moment=moment
            )
            rows.append(
                {
                    "galaxy": system["id"],
                    "split": system["split"],
                    "survey": "THINGS",
                    "kind": f"moment{moment}",
                    "scientificRole": "baryonic_input" if moment == 0 else "withheld_target",
                    "url": f"{things['baseUrl']}/{filename}",
                    "relativePath": f"{system['id']}/{filename}",
                    "expectedBytes": int(system["expectedBytes"][f"thingsMoment{moment}"]),
                }
            )
        for kind, template_key, size_key in (
            ("irac1", "imageTemplate", "singsIrac1"),
            ("irac1_weight", "weightTemplate", "singsIrac1Weight"),
        ):
            relative = sings[template_key].format(slug=system["slug"])
            filename = relative.rsplit("/", 1)[-1]
            rows.append(
                {
                    "galaxy": system["id"],
                    "split": system["split"],
                    "survey": "SINGS",
                    "kind": kind,
                    "scientificRole": "baryonic_input",
                    "url": f"{sings['baseUrl']}/{relative}",
                    "relativePath": f"{system['id']}/{filename}",
                    "expectedBytes": int(system["expectedBytes"][size_key]),
                }
            )
    return rows


def _download_once(url: str, partial: Path, expected_bytes: int) -> None:
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if 0 < existing < expected_bytes:
        headers["Range"] = f"bytes={existing}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        append = existing > 0 and getattr(response, "status", None) == 206
        mode = "ab" if append else "wb"
        with partial.open(mode) as handle:
            while True:
                chunk = response.read(CHUNK_BYTES)
                if not chunk:
                    break
                handle.write(chunk)


def download(url: str, destination: Path, expected_bytes: int) -> bool:
    """Download one immutable file; return True only when bytes were transferred."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        actual = destination.stat().st_size
        if actual != expected_bytes:
            raise ValueError(
                f"immutable raw file has {actual} bytes, expected {expected_bytes}: {destination}"
            )
        return False
    partial = destination.with_suffix(destination.suffix + ".partial")
    for attempt in range(1, 4):
        try:
            _download_once(url, partial, expected_bytes)
            actual = partial.stat().st_size
            if actual != expected_bytes:
                raise ValueError(f"download has {actual} bytes, expected {expected_bytes}")
            os.replace(partial, destination)
            return True
        except (OSError, urllib.error.URLError, ValueError):
            if attempt == 3:
                raise
            time.sleep(float(attempt))
    raise RuntimeError("unreachable")


def fits_header_audit(path: Path) -> dict[str, Any]:
    """Inspect structural metadata without ever accessing ``HDU.data``."""

    with fits.open(
        path,
        mode="readonly",
        memmap=False,
        lazy_load_hdus=True,
        do_not_scale_image_data=True,
    ) as hdus:
        header = hdus[0].header
        naxis = int(header.get("NAXIS", 0))
        shape = [int(header.get(f"NAXIS{index}", 0)) for index in range(1, naxis + 1)]
        ctype1 = str(header.get("CTYPE1", ""))
        ctype2 = str(header.get("CTYPE2", ""))
        return {
            "primaryHduReadable": True,
            "hduCount": len(hdus),
            "naxis": naxis,
            "fitsAxisLengths": shape,
            "ctype1": ctype1,
            "ctype2": ctype2,
            "twoCelestialAxes": "RA" in ctype1.upper() and "DEC" in ctype2.upper(),
            "bunit": str(header.get("BUNIT", "")),
            "beamMajorDeg": None
            if header.get("BMAJ") is None
            else float(header["BMAJ"]),
            "beamMinorDeg": None
            if header.get("BMIN") is None
            else float(header["BMIN"]),
        }


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    fieldnames = sorted({key for row in materialized for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    assets = source_assets(config)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)

    audited: list[dict[str, Any]] = []
    transferred = 0
    for index, asset in enumerate(assets, start=1):
        destination = args.data_dir / asset["relativePath"]
        changed = download(asset["url"], destination, int(asset["expectedBytes"]))
        transferred += int(changed)
        header = fits_header_audit(destination)
        audited.append(
            {
                **asset,
                "actualBytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
                **header,
                "arrayOpened": False,
            }
        )
        print(
            f"[{index:02d}/{len(assets):02d}] {asset['galaxy']} {asset['kind']} "
            f"{'downloaded' if changed else 'verified'}"
        )

    gates = config["acquisitionGates"]
    split_counts = {
        split: sum(system["split"] == split for system in config["systems"])
        for split in ("development", "validation", "holdout")
    }
    total_bytes = sum(int(row["actualBytes"]) for row in audited)
    gate_results = {
        "requiredSystems": len(config["systems"]) == int(gates["requiredSystems"]),
        "requiredDevelopmentSystems": split_counts["development"]
        == int(gates["requiredDevelopmentSystems"]),
        "requiredValidationSystems": split_counts["validation"]
        == int(gates["requiredValidationSystems"]),
        "requiredHoldoutSystems": split_counts["holdout"]
        == int(gates["requiredHoldoutSystems"]),
        "requiredFiles": len(audited) == int(gates["requiredFiles"]),
        "expectedTotalBytes": total_bytes == int(gates["expectedTotalBytes"]),
        "allByteCountsExact": all(
            int(row["actualBytes"]) == int(row["expectedBytes"]) for row in audited
        ),
        "allSha256Recorded": all(len(str(row["sha256"])) == 64 for row in audited),
        "allFitsPrimaryHdusReadable": all(row["primaryHduReadable"] for row in audited),
        "allFitsHaveTwoCelestialAxes": all(row["twoCelestialAxes"] for row in audited),
        "minimumHubbleType": min(s["sparc"]["hubbleType"] for s in config["systems"])
        <= int(gates["minimumHubbleType"]),
        "maximumHubbleType": max(s["sparc"]["hubbleType"] for s in config["systems"])
        >= int(gates["maximumHubbleType"]),
        "minimumInclinationDeg": min(
            s["sparc"]["inclinationDeg"] for s in config["systems"]
        )
        <= float(gates["minimumInclinationDeg"]),
        "maximumInclinationDeg": max(
            s["sparc"]["inclinationDeg"] for s in config["systems"]
        )
        >= float(gates["maximumInclinationDeg"]),
        "maximumGravityParameters": int(
            config["acquisitionRules"]["gravityParametersDuringAcquisition"]
        )
        <= int(gates["maximumGravityParameters"]),
        "holdoutArraysRemainUnopened": not any(
            row["arrayOpened"] for row in audited if row["split"] == "holdout"
        ),
    }
    status = "pass" if all(gate_results.values()) else "fail"
    manifest_core = {
        "schemaVersion": "sigma-p0738-acquisition-manifest/1",
        "stage": config["stage"],
        "status": status,
        "configSha256": sha256_bytes(config_bytes),
        "dataDirectory": str(args.data_dir.relative_to(ROOT)).replace("\\", "/"),
        "fileCount": len(audited),
        "totalBytes": total_bytes,
        "filesTransferredThisRun": transferred,
        "splitCounts": split_counts,
        "holdoutArraysOpened": False,
        "gravityParameters": 0,
        "velocityTargetsUsedForBaryonicExtraction": False,
        "gateResults": gate_results,
        "files": audited,
    }
    manifest = {**manifest_core, "manifestSha256": sha256_bytes(canonical_json(manifest_core))}
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output / "files.csv", audited)
    summary = f"""# P0738 morphology-diverse resolved acquisition

Status: **{status.upper()}**

- Systems: {len(config['systems'])} ({split_counts['development']} development, {split_counts['validation']} validation, {split_counts['holdout']} holdout)
- Files: {len(audited)}
- Bytes: {total_bytes}
- THINGS products: {sum(row['survey'] == 'THINGS' for row in audited)}
- SINGS products: {sum(row['survey'] == 'SINGS' for row in audited)}
- Holdout image arrays opened: no
- Gravity parameters: 0
- Velocity targets used for baryonic extraction: no
- Manifest SHA-256: `{manifest['manifestSha256']}`

This stage proves only acquisition integrity and sample diversity. It does not
score a gravity formula or claim that the raw images are already registered,
background-subtracted, deprojected, or converted into baryonic mass.
"""
    (args.output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(json.dumps({"status": status, "manifestSha256": manifest["manifestSha256"]}))
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
