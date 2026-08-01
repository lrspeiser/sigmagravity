#!/usr/bin/env python3
"""Hash and validate the permitted P0633 LITTLE THINGS baryonic inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.galaxy_maps import aips_clean_beam_degrees

DEFAULT_CONFIG = ROOT / "configs" / "p0636_little_things_baryon_acquisition.json"
DEFAULT_OUTPUT = ROOT / "results" / "p0636_little_things_baryon_acquisition"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def product_spec(target: dict) -> list[dict]:
    archive = target["archive_directory"]
    prefix = target["optical_prefix"]
    return [
        {
            "role": "H_I_moment_0",
            "filename": target["hi_filename"],
            "url": f"https://things.cv.nrao.edu/littlethings/{archive}/HI/{target['hi_filename']}",
        },
        {
            "role": "B_band",
            "filename": f"{prefix}b.fits",
            "url": f"https://science.nrao.edu/science/surveys/littlethings/data/{archive}/{prefix}b.fits",
        },
        {
            "role": "V_band",
            "filename": f"{prefix}v.fits",
            "url": f"https://science.nrao.edu/science/surveys/littlethings/data/{archive}/{prefix}v.fits",
        },
        {
            "role": "UBV_calibration",
            "filename": f"{prefix}_ubvcalib.txt",
            "url": f"https://science.nrao.edu/science/surveys/littlethings/data/{archive}/{prefix}_ubvcalib.txt",
        },
    ]


def audit(config: dict) -> dict:
    raw = ROOT / config["raw_directory"]
    forbidden = tuple(fragment.lower() for fragment in config["forbidden_filename_fragments"])
    rows = []
    errors = []
    for target in config["targets"]:
        directory = raw / target["id"]
        expected = product_spec(target)
        expected_names = {product["filename"] for product in expected}
        actual_names = {path.name for path in directory.iterdir()} if directory.exists() else set()
        if actual_names != expected_names:
            errors.append(
                f"{target['id']}: expected {sorted(expected_names)}, found {sorted(actual_names)}"
            )
        for product in expected:
            path = directory / product["filename"]
            lower = path.name.lower()
            if any(fragment in lower for fragment in forbidden):
                errors.append(f"{target['id']}: forbidden product name {path.name}")
                continue
            if not path.exists():
                continue
            row = {
                "galaxy": target["id"],
                **product,
                "relative_path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "valid": True,
            }
            if path.suffix.lower() == ".fits":
                try:
                    with fits.open(path, memmap=True) as hdul:
                        data = np.squeeze(hdul[0].data)
                        row["shape"] = "x".join(str(value) for value in data.shape)
                        row["finite"] = bool(np.isfinite(data).all())
                        if data.ndim != 2 or not row["finite"]:
                            raise ValueError("FITS image is not a finite 2D image")
                        if product["role"] == "H_I_moment_0":
                            beam = aips_clean_beam_degrees(hdul[0].header)
                            row["beam_major_deg"] = beam[0]
                            row["beam_minor_deg"] = beam[1]
                            row["beam_position_angle_deg"] = beam[2]
                            row["unit"] = str(hdul[0].header.get("BUNIT", ""))
                            if "JY/B*M/S" not in row["unit"].upper():
                                raise ValueError(f"unexpected moment-0 unit {row['unit']}")
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    row["valid"] = False
                    errors.append(f"{target['id']}/{path.name}: {exc}")
            else:
                calibration = path.read_text(encoding="utf-8", errors="replace")
                if not re.search(r"begin\s+BFIT", calibration) or not re.search(
                    r"begin\s+VFIT", calibration
                ):
                    row["valid"] = False
                    errors.append(f"{target['id']}/{path.name}: missing B/V calibration fits")
            rows.append(row)
    frame = pd.DataFrame(rows)
    return {
        "status": "ready" if not errors else "input_failure",
        "protocol_version": config["protocol_version"],
        "targets": len(config["targets"]),
        "products": len(rows),
        "total_bytes": int(frame["bytes"].sum()) if not frame.empty else 0,
        "all_products_valid": bool(not frame.empty and frame["valid"].all()),
        "errors": errors,
        "sealed_state": config["sealed_state"],
        "P0633_target_observables_opened": False,
        "products_detail": rows,
    }


def write_outputs(report: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "provenance.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(report["products_detail"]).to_csv(output / "provenance.csv", index=False)
    summary = f"""# P0636 LITTLE THINGS baryonic-input acquisition

- Status: **{report['status'].upper()}**
- Targets: {report['targets']}
- Permitted products: {report['products']}
- Bytes: {report['total_bytes']}
- All products valid: `{str(report['all_products_valid']).lower()}`
- Sealed target observables opened: `{str(report['P0633_target_observables_opened']).lower()}`

Only H I moment-0, B, V, and UBV calibration products are present. Kinematic
cubes, moment-1, moment-2, circular-velocity products, and P0633 target scores
remain sealed.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    if config.get("status") != "baryonic_inputs_only_before_candidate_lock":
        raise RuntimeError("P0636 acquisition config is not recognized")
    report = audit(config)
    report["config_sha256"] = sha256(args.config.resolve())
    write_outputs(report, args.output.resolve())
    print(
        json.dumps(
            {key: report[key] for key in ("status", "targets", "products", "total_bytes", "errors")},
            indent=2,
        )
    )
    if report["status"] != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
