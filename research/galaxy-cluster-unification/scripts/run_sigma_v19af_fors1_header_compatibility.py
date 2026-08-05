#!/usr/bin/env python3
"""Audit frozen FORS1 compressed payloads through primary headers only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from astropy.io import fits


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19af_fors1_header_compatibility.json"
FITS_BLOCK = 2880
FITS_CARD = 80


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def windows_path_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive
    if len(drive) != 2 or drive[1] != ":":
        raise RuntimeError(f"V19AF requires a drive-letter Windows path: {resolved}")
    relative = resolved.as_posix()[3:]
    return f"/mnt/{drive[0].lower()}/{relative}"


def decompress_unix_compress(
    source: Path, target: Path, *, distro: str, executable: str
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "wsl.exe",
        "-d",
        distro,
        "--",
        executable,
        "-dc",
        "--",
        windows_path_to_wsl(source),
    ]
    with target.open("wb") as output:
        result = subprocess.run(command, stdout=output, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        target.unlink(missing_ok=True)
        raise RuntimeError(
            f"V19AF decompression failed for {source.name}: "
            + result.stderr.decode("utf-8", "replace")
        )


def read_primary_header_bytes(path: Path) -> bytes:
    blocks: list[bytes] = []
    with path.open("rb") as handle:
        while True:
            block = handle.read(FITS_BLOCK)
            if len(block) != FITS_BLOCK:
                raise RuntimeError(f"truncated FITS primary header: {path}")
            blocks.append(block)
            cards = [block[index : index + FITS_CARD] for index in range(0, FITS_BLOCK, FITS_CARD)]
            if any(card[:8].decode("ascii", "strict").strip() == "END" for card in cards):
                return b"".join(blocks)
            if len(blocks) > 100:
                raise RuntimeError(f"unreasonably long FITS primary header: {path}")


def parse_primary_header(raw: bytes) -> fits.Header:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise RuntimeError("FITS primary header is not ASCII") from exc
    return fits.Header.fromstring(text, sep="")


def normalized_header_value(header: fits.Header, keyword: str) -> Any:
    value = header.get(keyword)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeError(f"nonfinite FITS header value for {keyword}")
        return value
    return str(value)


def header_cards(raw: bytes) -> list[str]:
    cards = [
        raw[index : index + FITS_CARD].decode("ascii", "strict")
        for index in range(0, len(raw), FITS_CARD)
    ]
    end = next(index for index, card in enumerate(cards) if card[:8].strip() == "END")
    return cards[: end + 1]


def expected_primary_size(header: fits.Header, header_bytes: int) -> int:
    naxis = int(header["NAXIS"])
    pixels = 1
    for axis in range(1, naxis + 1):
        pixels *= int(header[f"NAXIS{axis}"])
    data_bytes = pixels * (abs(int(header["BITPIX"])) // 8)
    padded_data = ((data_bytes + FITS_BLOCK - 1) // FITS_BLOCK) * FITS_BLOCK
    return header_bytes + padded_data


def validate_config(config_path: Path, config: dict[str, Any]) -> dict[str, str]:
    if config["status"] != "frozen_before_decompression_or_any_fits_header_or_pixel_access":
        raise RuntimeError("V19AF protocol is not frozen")
    runner = ROOT / config["implementation"]["runner"]
    if sha256(runner) != config["implementation"]["runner_sha256"]:
        raise RuntimeError("frozen V19AF runner hash mismatch")
    hashes = {"config": sha256(config_path), "runner": sha256(runner)}
    for artifact in config["parent_artifacts"]:
        path = ROOT / artifact["path"]
        actual = sha256(path)
        if actual != artifact["sha256"]:
            raise RuntimeError(f"V19AF parent hash mismatch: {artifact['path']}")
        hashes[artifact["path"]] = actual
    authorization = config["authorization"]
    prohibited = (
        "persist_decompressed_pixel_payload",
        "interpret_or_measure_pixel_values",
        "inspect_member_or_candidate_cutouts",
        "fit_photometry_or_counterparts",
        "infer_stellar_mass_or_current",
        "read_lensing_or_halo_payload",
        "change_gravity_physics",
        "open_holdout",
    )
    if any(authorization[name] for name in prohibited):
        raise RuntimeError("V19AF authorizes a prohibited action")
    return hashes


def run(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_json(config_path)
    input_hashes = validate_config(config_path, config)
    parent_report = load_json(ROOT / config["inputs"]["acquisition_report"])
    files = parent_report["files"]
    if len(files) != int(config["gates"]["exact_input_files"]):
        raise RuntimeError("V19AF input file count changed")
    if not parent_report["gates"]["all_acquisition_gates_pass"]:
        raise RuntimeError("V19AF parent acquisition did not pass")

    records: list[dict[str, Any]] = []
    header_payloads: list[dict[str, Any]] = []
    decompression = config["decompression"]
    with tempfile.TemporaryDirectory(prefix="sigma_v19af_") as temporary:
        temporary_root = Path(temporary)
        for index, source_record in enumerate(files, start=1):
            source = ROOT / source_record["path"]
            if sha256(source) != source_record["sha256"]:
                raise RuntimeError(f"V19AF compressed source hash changed: {source}")
            target = temporary_root / f"{index:03d}.fits"
            decompress_unix_compress(
                source,
                target,
                distro=decompression["wsl_distribution"],
                executable=decompression["executable"],
            )
            raw_header = read_primary_header_bytes(target)
            header = parse_primary_header(raw_header)
            for keyword in config["header"]["required_primary_keywords"]:
                if keyword not in header:
                    raise RuntimeError(f"V19AF missing required header {keyword}: {source.name}")
            if header["SIMPLE"] is not True or int(header["NAXIS"]) != 2:
                raise RuntimeError(f"V19AF source is not a simple 2D primary image: {source.name}")
            if int(header["NAXIS1"]) <= 0 or int(header["NAXIS2"]) <= 0:
                raise RuntimeError(f"V19AF invalid image dimensions: {source.name}")
            expected_size = expected_primary_size(header, len(raw_header))
            decompressed_size = target.stat().st_size
            if decompressed_size < expected_size:
                raise RuntimeError(f"V19AF decompressed FITS is truncated: {source.name}")
            selected = {
                keyword: normalized_header_value(header, keyword)
                for keyword in config["header"]["candidate_detector_signature_keywords"]
            }
            cards = header_cards(raw_header)
            record = {
                "dp_id": source_record["dp_id"],
                "role": source_record["role"],
                "filter_path": source_record["filter_path"],
                "compressed_path": source_record["path"],
                "compressed_sha256": source_record["sha256"],
                "decompressed_bytes": decompressed_size,
                "decompressed_sha256": sha256(target),
                "primary_header_bytes": len(raw_header),
                "primary_header_sha256": hashlib.sha256(raw_header).hexdigest(),
                "primary_card_count_through_end": len(cards),
                "expected_minimum_primary_hdu_bytes": expected_size,
                "trailing_bytes_after_primary_hdu": decompressed_size - expected_size,
                "bitpix": int(header["BITPIX"]),
                "naxis1": int(header["NAXIS1"]),
                "naxis2": int(header["NAXIS2"]),
                "detector_signature_candidates": selected,
                "pixel_values_interpreted": False,
            }
            records.append(record)
            header_payloads.append(
                {
                    "dp_id": source_record["dp_id"],
                    "primary_header_sha256": record["primary_header_sha256"],
                    "cards": cards,
                }
            )
            print(
                f"[{index:02d}/{len(files):02d}] {source_record['role']} "
                f"{source_record['dp_id']}: {record['naxis1']}x{record['naxis2']} header",
                flush=True,
            )

    candidate_keywords = config["header"]["candidate_detector_signature_keywords"]
    active_keywords: list[str] = []
    partially_present: list[str] = []
    for keyword in candidate_keywords:
        values = [row["detector_signature_candidates"][keyword] for row in records]
        if all(value is None for value in values):
            continue
        if any(value is None for value in values):
            partially_present.append(keyword)
        else:
            active_keywords.append(keyword)
    signatures = {
        json.dumps(
            {
                "BITPIX": row["bitpix"],
                "NAXIS1": row["naxis1"],
                "NAXIS2": row["naxis2"],
                **{
                    keyword: row["detector_signature_candidates"][keyword]
                    for keyword in active_keywords
                },
            },
            sort_keys=True,
        )
        for row in records
    }
    role_counts = {
        role: sum(row["role"] == role for row in records)
        for role in ("science", "bias", "flat")
    }
    gates = {
        "exact_input_file_count": len(records) == int(config["gates"]["exact_input_files"]),
        "exact_role_counts": role_counts == config["gates"]["exact_role_counts"],
        "every_compressed_hash_matches_acquisition": True,
        "every_primary_hdu_is_simple_2d": all(
            row["naxis1"] > 0 and row["naxis2"] > 0 for row in records
        ),
        "no_partially_present_detector_signature_keyword": not partially_present,
        "one_exact_detector_signature": len(signatures) == 1,
        "every_decompressed_primary_hdu_complete": all(
            row["trailing_bytes_after_primary_hdu"] >= 0 for row in records
        ),
        "no_decompressed_payload_persisted": True,
        "no_pixel_value_interpreted": True,
        "no_member_photometry_mass_lensing_or_gravity": True,
    }
    gates["all_header_compatibility_gates_pass"] = all(gates.values())
    outputs = config["outputs"]
    headers_path = ROOT / outputs["headers"]
    report_path = ROOT / outputs["report"]
    headers_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    headers_path.write_text(
        json.dumps(header_payloads, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report = {
        "report_version": "SIGMA-V19AF-FORS1-HEADER-COMPATIBILITY-1.0.0",
        "status": "passed_header_compatibility_without_pixel_interpretation"
        if gates["all_header_compatibility_gates_pass"]
        else "failed_header_compatibility",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config": config_path.relative_to(ROOT).as_posix(),
        "input_hashes": input_hashes,
        "role_counts": role_counts,
        "records": records,
        "active_detector_signature_keywords": active_keywords,
        "partially_present_detector_signature_keywords": partially_present,
        "unique_detector_signatures": sorted(signatures),
        "gates": gates,
        "outputs": {
            "headers": headers_path.relative_to(ROOT).as_posix(),
            "headers_sha256": sha256(headers_path),
        },
        "claim_boundary": config["claim_boundary"],
        "decompressed_payload_persisted": False,
        "pixel_values_interpreted": False,
        "member_or_candidate_cutout_inspected": False,
        "photometry_or_counterpart_fitted": False,
        "stellar_mass_or_current_inferred": False,
        "lensing_or_halo_payload_opened": False,
        "gravity_formula_or_parameter_changed": False,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    report = run(args.config)
    print(
        json.dumps(
            {
                "status": report["status"],
                "role_counts": report["role_counts"],
                "active_detector_signature_keywords": report[
                    "active_detector_signature_keywords"
                ],
                "gates": report["gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
