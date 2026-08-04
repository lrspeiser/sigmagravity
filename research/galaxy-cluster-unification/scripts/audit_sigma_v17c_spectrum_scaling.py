#!/usr/bin/env python3
"""Independently audit a completed Sigma v17C source/background PHA pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def card_value(card: str) -> str:
    """Return the FITS value field without a trailing comment."""
    raw = card[10:].rstrip()
    if raw.startswith("'"):
        index = 1
        while index < len(raw):
            if raw[index] == "'":
                if index + 1 < len(raw) and raw[index + 1] == "'":
                    index += 2
                    continue
                return raw[1:index].replace("''", "'").strip()
            index += 1
        raise RuntimeError(f"unterminated FITS string card: {card}")
    return raw.split("/", maxsplit=1)[0].strip()


def fits_header(path: Path, extension: str = "SPECTRUM") -> dict[str, str]:
    """Read one FITS extension header without loading its table payload."""
    with path.open("rb") as handle:
        while True:
            header: dict[str, str] = {}
            saw_card = False
            while True:
                block = handle.read(2880)
                if not block:
                    if saw_card:
                        raise RuntimeError(f"truncated FITS header in {path}")
                    raise RuntimeError(f"FITS extension {extension} absent from {path}")
                if len(block) != 2880:
                    raise RuntimeError(f"invalid FITS block length in {path}")
                saw_card = True
                end_found = False
                for offset in range(0, 2880, 80):
                    card = block[offset : offset + 80].decode("ascii")
                    key = card[:8].strip()
                    if key == "END":
                        end_found = True
                        break
                    if card[8:10] == "= ":
                        header[key] = card_value(card)
                if end_found:
                    break

            if header.get("EXTNAME", "").strip() == extension:
                return header

            bitpix = abs(int(header.get("BITPIX", "0")))
            naxis = int(header.get("NAXIS", "0"))
            elements = 0 if naxis == 0 else 1
            for axis in range(1, naxis + 1):
                elements *= int(header[f"NAXIS{axis}"])
            pcount = int(header.get("PCOUNT", "0"))
            gcount = int(header.get("GCOUNT", "1"))
            data_bytes = (bitpix // 8 * elements + pcount) * gcount
            handle.seek(data_bytes + (-data_bytes) % 2880, os.SEEK_CUR)


def numeric(header: dict[str, str], key: str) -> float:
    try:
        value = float(header[key].replace("D", "E"))
    except (KeyError, ValueError) as exc:
        raise RuntimeError(f"missing or non-numeric FITS keyword {key}") from exc
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite FITS keyword {key}")
    return value


def frozen_bkgscale(cleaning_report: Path, obsid: int, ccd_id: int) -> tuple[float, str]:
    report = json.loads(cleaning_report.read_text(encoding="utf-8"))
    matches = [row for row in report["observations"] if int(row["obsid"]) == obsid]
    if len(matches) != 1:
        raise RuntimeError(f"expected one cleaning row for ObsID {obsid}, found {len(matches)}")
    key = f"BKGSCAL{ccd_id}"
    try:
        value = float(matches[0]["blanksky_scaling"][key])
    except KeyError as exc:
        raise RuntimeError(f"cleaning report lacks {key} for ObsID {obsid}") from exc
    return value, key


def relative_error(observed: float, expected: float) -> float:
    return abs(observed / expected - 1.0)


def sherpa_background_scale(source_pha: Path) -> float:
    """Return the scale used by Sherpa; requires execution in the CIAO environment."""
    from sherpa.astro import ui

    ui.clean()
    ui.load_pha(str(source_pha))
    return float(ui.get_bkg_scale())


def audit(args: argparse.Namespace) -> dict[str, Any]:
    source_pha = args.source_pha.resolve()
    background_pha = args.background_pha.resolve()
    arf = args.arf.resolve() if args.arf else source_pha.with_suffix(".arf")
    rmf = args.rmf.resolve() if args.rmf else source_pha.with_suffix(".rmf")
    correction_log = background_pha.with_suffix(".areascal.log")
    required_files = [source_pha, background_pha, arf, rmf, correction_log]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing completed-cell products: {missing}")

    source = fits_header(source_pha)
    background = fits_header(background_pha)
    bkgscale, bkgscale_key = frozen_bkgscale(
        args.cleaning_report.resolve(), args.obsid, args.ccd_id
    )
    source_exposure = numeric(source, "EXPOSURE")
    background_exposure = numeric(background, "EXPOSURE")
    source_backscal = numeric(source, "BACKSCAL")
    background_backscal = numeric(background, "BACKSCAL")
    source_areascal = numeric(source, "AREASCAL")
    background_areascal = numeric(background, "AREASCAL")
    effective_scale = (
        source_exposure
        / background_exposure
        * source_backscal
        / background_backscal
        * source_areascal
        / background_areascal
    )
    expected_background_areascal = (
        source_exposure
        / background_exposure
        * source_backscal
        / background_backscal
        * source_areascal
        / bkgscale
    )

    pointers = {
        "BACKFILE": {"observed": source.get("BACKFILE"), "expected": background_pha.name},
        "ANCRFILE": {"observed": source.get("ANCRFILE"), "expected": arf.name},
        "RESPFILE": {"observed": source.get("RESPFILE"), "expected": rmf.name},
    }
    for item in pointers.values():
        item["basename_matches"] = (
            Path(str(item["observed"])).name == str(item["expected"])
        )

    tolerance = float(args.relative_tolerance)
    effective_error = relative_error(effective_scale, bkgscale)
    areascal_error = relative_error(background_areascal, expected_background_areascal)
    sherpa_scale = sherpa_background_scale(source_pha) if args.check_sherpa else None
    sherpa_error = (
        relative_error(sherpa_scale, bkgscale) if sherpa_scale is not None else None
    )
    checks = {
        "effective_scale_matches_frozen_BKGSCALn": effective_error <= tolerance,
        "background_AREASCAL_matches_reconstruction": areascal_error <= tolerance,
        "source_product_pointers_match": all(
            bool(item["basename_matches"]) for item in pointers.values()
        ),
        "correction_completed": correction_log.is_file(),
        "sherpa_scale_matches_frozen_BKGSCALn": (
            sherpa_error <= tolerance if sherpa_error is not None else None
        ),
    }
    required_checks = [value for value in checks.values() if value is not None]
    report = {
        "protocol_version": "SIGMA-V17C-SPECTRUM-SCALING-AUDIT-1.0.0",
        "generated_utc": datetime.now(UTC).isoformat(),
        "status": "passed" if all(required_checks) else "failed",
        "obsid": args.obsid,
        "ccd_id": args.ccd_id,
        "relative_tolerance": tolerance,
        "frozen_scaling_source": {
            "cleaning_report": str(args.cleaning_report.resolve()),
            "cleaning_report_sha256": sha256(args.cleaning_report.resolve()),
            "keyword": bkgscale_key,
            "BKGSCALn": bkgscale,
        },
        "headers": {
            "source_EXPOSURE": source_exposure,
            "background_EXPOSURE": background_exposure,
            "source_BACKSCAL": source_backscal,
            "background_BACKSCAL": background_backscal,
            "source_AREASCAL": source_areascal,
            "background_AREASCAL": background_areascal,
        },
        "reconstructed": {
            "expected_background_AREASCAL": expected_background_areascal,
            "background_AREASCAL_relative_error": areascal_error,
            "effective_background_scale": effective_scale,
            "effective_scale_relative_error_from_BKGSCALn": effective_error,
            "sherpa_get_bkg_scale": sherpa_scale,
            "sherpa_scale_relative_error_from_BKGSCALn": sherpa_error,
        },
        "pointers": pointers,
        "checks": checks,
        "files": {
            path.name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in required_files
        },
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pha", type=Path, required=True)
    parser.add_argument("--background-pha", type=Path, required=True)
    parser.add_argument("--arf", type=Path)
    parser.add_argument("--rmf", type=Path)
    parser.add_argument("--cleaning-report", type=Path, required=True)
    parser.add_argument("--obsid", type=int, required=True)
    parser.add_argument("--ccd-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relative-tolerance", type=float, default=1e-6)
    parser.add_argument("--check-sherpa", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
