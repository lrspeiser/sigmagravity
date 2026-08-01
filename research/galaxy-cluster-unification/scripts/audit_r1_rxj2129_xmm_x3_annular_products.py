#!/usr/bin/env python3
"""Audit frozen RX J2129 X3 annular products and prefit count adequacy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits


PROJECT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT / "configs/r1_rxj2129_xmm_x3_annular_protocol.json"
X2_REPORT_PATH = PROJECT / "results/r1_rxj2129_xmm_event_processing/report.json"
SOURCE_MASK_MANIFEST_PATH = (
    PROJECT / "data/derived/r1_rxj2129_xmm_x3_source_mask_manifest.json"
)
LEDGER_PATH = PROJECT / "data/derived/r1_rxj2129_xmm_x3_annular_count_ledger.csv"
MANIFEST_PATH = PROJECT / "data/derived/r1_rxj2129_xmm_x3_annular_product_manifest.json"
REPORT_PATH = PROJECT / "results/r1_rxj2129_xmm_x3_annular_products/report.json"
LINUX_ROOT = Path(
    "/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/"
    "x3/annular_products"
)
WINDOWS_ROOT = Path(
    "//wsl.localhost/Ubuntu-24.04/home/henry/.local/share/sigmagravity-xmm/"
    "work/rxj2129/0093030201/x3/annular_products"
)
INSTRUMENTS = {
    "MOS2": {
        "prefix": "mos2S002",
        "spectra_task": "mosspectra",
        "background_task": "mosback",
        "spectra_marker": ".mosspectra_complete",
        "background_marker": ".mosback_complete",
        "oot_file": None,
    },
    "pn": {
        "prefix": "pnS003",
        "spectra_task": "pnspectra",
        "background_task": "pnback",
        "spectra_marker": ".pnspectra_complete",
        "background_marker": ".pnback_complete",
        "oot_file": "pnS003-fovt-oot.pi",
    },
}


def parse_args() -> argparse.Namespace:
    default_root = WINDOWS_ROOT if os.name == "nt" else LINUX_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--product-root",
        type=Path,
        default=Path(os.environ.get("SIGMAGRAVITY_XMM_X3_ROOT", default_root)),
    )
    parser.add_argument("--ledger", type=Path, default=LEDGER_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def region_mask_audit(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "expected_rows": int(expected["derived_rows"]),
        "expected_sha256": expected["derived_sha256"],
    }
    if not path.is_file():
        result["passed"] = False
        return result
    with fits.open(path, memmap=False) as hdul:
        result["has_REGION_extension"] = "REGION" in [hdu.name for hdu in hdul]
        result["rows"] = (
            len(hdul["REGION"].data) if result["has_REGION_extension"] else None
        )
    result["sha256"] = sha256(path)
    result["passed"] = (
        result["has_REGION_extension"]
        and result["rows"] == result["expected_rows"]
        and result["sha256"] == result["expected_sha256"]
    )
    return result


def read_spectrum(path: Path) -> dict[str, Any]:
    with fits.open(path, memmap=False) as hdul:
        hdu = hdul["SPECTRUM"]
        names = set(hdu.columns.names)
        if "COUNTS" in names:
            values = np.asarray(hdu.data["COUNTS"], dtype=float)
            kind = "COUNTS"
        elif "RATE" in names:
            values = np.asarray(hdu.data["RATE"], dtype=float)
            kind = "RATE"
        else:
            raise ValueError(f"{path} has neither COUNTS nor RATE")
        return {
            "channel": np.asarray(hdu.data["CHANNEL"], dtype=int),
            "values": values,
            "kind": kind,
            "exposure_s": float(hdu.header["EXPOSURE"]),
            "backscal": float(hdu.header["BACKSCAL"]),
        }


def counts_in_channels(spec: dict[str, Any], channels: tuple[int, int]) -> float:
    selected = (spec["channel"] >= channels[0]) & (spec["channel"] <= channels[1])
    total = float(np.sum(spec["values"][selected]))
    return total * spec["exposure_s"] if spec["kind"] == "RATE" else total


def task_ended(log_text: str, task: str) -> bool:
    return re.search(rf"{task}\s+\({task}-[^)]+\).*\sended:", log_text) is not None


def task_errors(log_text: str, task: str) -> list[str]:
    del task
    return re.findall(r"^\*\*.*:\s+error\b.*$", log_text, flags=re.MULTILINE)


def warning_summary(log_text: str) -> dict[str, Any]:
    codes = re.findall(r"warning \(([^)]+)\)", log_text)
    return {"count": len(codes), "codes": dict(sorted(Counter(codes).items()))}


def fits_product_audit(path: Path, kind: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else 0,
    }
    if not result["exists"] or result["bytes"] <= 0:
        result["passed"] = False
        return result
    with fits.open(path, memmap=False) as hdul:
        extensions = [hdu.name for hdu in hdul]
        result["extensions"] = extensions
        if kind == "spectrum":
            required = {"SPECTRUM"}
            if required.issubset(extensions):
                spectrum = hdul["SPECTRUM"]
                values = (
                    np.asarray(spectrum.data["COUNTS"], dtype=float)
                    if "COUNTS" in spectrum.columns.names
                    else np.asarray(spectrum.data["RATE"], dtype=float)
                )
                result["exposure_s"] = float(spectrum.header.get("EXPOSURE", 0.0))
                result["backscal"] = float(spectrum.header.get("BACKSCAL", 0.0))
                result["finite_values"] = bool(np.isfinite(values).all())
                result["value_sum"] = float(np.sum(values))
                numeric_passed = (
                    result["exposure_s"] > 0
                    and result["backscal"] > 0
                    and result["finite_values"]
                    and result["value_sum"] > 0
                )
            else:
                numeric_passed = False
        elif kind == "rmf":
            required = {"MATRIX", "EBOUNDS"}
            numeric_passed = required.issubset(extensions) and (
                len(hdul["MATRIX"].data) > 0 and len(hdul["EBOUNDS"].data) > 0
            )
        elif kind == "arf":
            required = {"SPECRESP"}
            if required.issubset(extensions):
                response = np.asarray(hdul["SPECRESP"].data["SPECRESP"], dtype=float)
                result["finite_response"] = bool(np.isfinite(response).all())
                result["response_sum_cm2"] = float(np.sum(response))
                result["positive_response_channels"] = int(np.count_nonzero(response > 0))
                numeric_passed = (
                    result["finite_response"]
                    and result["response_sum_cm2"] > 0
                    and result["positive_response_channels"] > 0
                )
            else:
                numeric_passed = False
        else:
            raise ValueError(kind)
        result["passed"] = required.issubset(extensions) and numeric_passed
    result["sha256"] = sha256(path)
    return result


def audit_instrument_products(
    directory: Path,
    instrument: str,
    spec: dict[str, Any],
    inner_detector: float,
    outer_detector: float,
    expected_masks: dict[str, Any],
    expected_pn_quadrants: str | None,
) -> dict[str, Any]:
    prefix = spec["prefix"]
    spectra_log_path = directory / f"{spec['spectra_task']}.log"
    background_log_path = directory / f"{spec['background_task']}.log"
    region_path = directory / "annulus_region.txt"
    required = [
        spec["spectra_marker"],
        spec["background_marker"],
        f"{prefix}-fovt.pi",
        f"{prefix}-bkg.pi",
        f"{prefix}.rmf",
        f"{prefix}.arf",
        "srcdet.fits",
        "srcsky.fits",
        "source_mask_rows.txt",
        "annulus_region.txt",
        f"{spec['spectra_task']}.log",
        f"{spec['background_task']}.log",
    ]
    if spec["oot_file"]:
        required.append(spec["oot_file"])
        required.append("pn_quadrants.txt")
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        return {"passed": False, "missing_products": missing}

    spectra_log = spectra_log_path.read_text(errors="replace")
    background_log = background_log_path.read_text(errors="replace")
    mask_rows = int((directory / "source_mask_rows.txt").read_text().strip())
    expected_mask_rows = int(expected_masks["detector"]["derived_rows"])
    mask_row_declaration_valid = (
        mask_rows == expected_mask_rows
        and int(expected_masks["sky"]["derived_rows"]) == expected_mask_rows
    )
    source_removal_mode_valid = (
        (mask_rows > 0 and "withsrcrem=yes" in spectra_log and "srcdet.fits" in spectra_log)
        or (mask_rows == 0 and "withsrcrem=no" in spectra_log)
    )
    mask_audits = {
        "detector": region_mask_audit(directory / "srcdet.fits", expected_masks["detector"]),
        "sky": region_mask_audit(directory / "srcsky.fits", expected_masks["sky"]),
    }
    if instrument == "pn":
        declared_quadrants = (directory / "pn_quadrants.txt").read_text().strip()
        expected_sas_quadrants = " ".join(
            "yes" if flag == "T" else "no"
            for flag in str(expected_pn_quadrants).split()
        )
        pn_quadrant_declaration_valid = (
            declared_quadrants == expected_pn_quadrants
            and f"quads='{expected_sas_quadrants}'" in spectra_log
            and f"quads='{expected_sas_quadrants}'" in background_log
        )
        pn_badpixel_resolution_valid = "badpixelresolution=1" in spectra_log
    else:
        declared_quadrants = None
        pn_quadrant_declaration_valid = True
        pn_badpixel_resolution_valid = True
    region_text = region_path.read_text().strip()
    numbers = [float(item) for item in re.findall(r"-?[0-9]+(?:\.[0-9]+)?", region_text)]
    region_valid = (
        region_text.count("circle(") == 2
        and len(numbers) == 6
        and math.isclose(numbers[2], outer_detector, abs_tol=1e-8)
        and math.isclose(numbers[5], inner_detector, abs_tol=1e-8)
    )
    products = {
        "source_spectrum": fits_product_audit(
            directory / f"{prefix}-fovt.pi", "spectrum"
        ),
        "QPB_spectrum": fits_product_audit(
            directory / f"{prefix}-bkg.pi", "spectrum"
        ),
        "RMF": fits_product_audit(directory / f"{prefix}.rmf", "rmf"),
        "ARF": fits_product_audit(directory / f"{prefix}.arf", "arf"),
    }
    if spec["oot_file"]:
        products["OOT_spectrum"] = fits_product_audit(
            directory / str(spec["oot_file"]), "spectrum"
        )
    spectra_ended = task_ended(spectra_log, spec["spectra_task"])
    background_ended = task_ended(background_log, spec["background_task"])
    spectra_error_records = task_errors(spectra_log, spec["spectra_task"])
    background_error_records = task_errors(background_log, spec["background_task"])
    spectra_warning_summary = warning_summary(spectra_log)
    background_warning_summary = warning_summary(background_log)
    passed = (
        region_valid
        and spectra_ended
        and background_ended
        and not spectra_error_records
        and not background_error_records
        and all(product["passed"] for product in products.values())
        and mask_row_declaration_valid
        and source_removal_mode_valid
        and all(mask["passed"] for mask in mask_audits.values())
        and pn_quadrant_declaration_valid
        and pn_badpixel_resolution_valid
        and "annulus_region.txt" in spectra_log
    )
    return {
        "passed": passed,
        "missing_products": [],
        "region_valid": region_valid,
        "region_expression_sha256": hashlib.sha256(
            (region_text + "\n").encode()
        ).hexdigest(),
        "spectra_task_ended_normally": spectra_ended,
        "background_task_ended_normally": background_ended,
        "spectra_task_error_records": spectra_error_records,
        "background_task_error_records": background_error_records,
        "spectra_task_warning_summary": spectra_warning_summary,
        "background_task_warning_summary": background_warning_summary,
        "source_mask_rows": mask_rows,
        "source_mask_row_declaration_valid": mask_row_declaration_valid,
        "source_removal_mode_valid": source_removal_mode_valid,
        "source_masks": mask_audits,
        "declared_pn_quadrants": declared_quadrants,
        "pn_quadrant_declaration_valid": pn_quadrant_declaration_valid,
        "pn_badpixel_resolution_valid": pn_badpixel_resolution_valid,
        "products": products,
    }


def count_instrument(
    directory: Path,
    instrument: str,
    spec: dict[str, Any],
    channels: tuple[int, int],
    multiplier: float,
    oot_scale: float,
) -> dict[str, float]:
    prefix = spec["prefix"]
    source = read_spectrum(directory / f"{prefix}-fovt.pi")
    observed_before_oot = counts_in_channels(source, channels)
    oot_counts = 0.0
    if spec["oot_file"]:
        oot = read_spectrum(directory / str(spec["oot_file"]))
        oot_counts = counts_in_channels(oot, channels)
    observed = observed_before_oot - oot_scale * oot_counts
    qpb = read_spectrum(directory / f"{prefix}-bkg.pi")
    qpb_counts = counts_in_channels(qpb, channels)
    conservative_qpb = multiplier * qpb_counts
    return {
        "observed_counts_before_OOT_subtraction": observed_before_oot,
        "OOT_scale": oot_scale,
        "OOT_counts_before_scaling": oot_counts,
        "observed_counts": observed,
        "ESAS_QPB_counts": qpb_counts,
        "conservative_QPB_multiplier": multiplier,
        "conservative_QPB_counts": conservative_qpb,
        "net_counts": observed - conservative_qpb,
        "variance_proxy": max(observed, 0.0) + max(conservative_qpb, 0.0),
    }


def write_ledger(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    protocol = json.loads(PROTOCOL_PATH.read_text())
    x2_report = json.loads(X2_REPORT_PATH.read_text())
    source_mask_manifest = json.loads(SOURCE_MASK_MANIFEST_PATH.read_text())
    assert x2_report["gates"]["R1B3_XMM_X2_flare_background_gate_passed"] is True
    assert x2_report["passing_instruments"] == ["MOS2", "pn"]
    assert source_mask_manifest["gates"]["X3_target_aware_source_mask_gate_passed"] is True

    geometry = protocol["fixed_geometry"]
    adequacy = protocol["prefit_adequacy"]
    channels = tuple(protocol["extraction"]["PI_channels_inclusive"])
    edges_kpc = geometry["radial_edges_kpc"]
    edges_detector = geometry["radial_edges_detector_units"]
    ids = geometry["annulus_ids"]
    multipliers = adequacy["conservative_QPB_multiplier"]
    oot_scale = float(protocol["extraction"]["pn"]["OOT_scale"])

    annuli: dict[str, Any] = {}
    ledger_rows: list[dict[str, Any]] = []
    for index, annulus_id in enumerate(ids):
        instrument_results: dict[str, Any] = {}
        combined_net = 0.0
        combined_variance = 0.0
        for instrument, spec in INSTRUMENTS.items():
            directory = args.product_root / annulus_id / instrument
            product_audit = audit_instrument_products(
                directory,
                instrument,
                spec,
                float(edges_detector[index]),
                float(edges_detector[index + 1]),
                source_mask_manifest["annular_compact_masks"][annulus_id]["products"][instrument],
                (
                    protocol["extraction"]["pn"]["active_quadrants_by_annulus"][annulus_id]
                    if instrument == "pn"
                    else None
                ),
            )
            if product_audit["passed"]:
                counts = count_instrument(
                    directory,
                    instrument,
                    spec,
                    channels,
                    float(multipliers[instrument]),
                    oot_scale if instrument == "pn" else 0.0,
                )
                combined_net += counts["net_counts"]
                combined_variance += counts["variance_proxy"]
            else:
                counts = None
            instrument_results[instrument] = {
                "product_audit": product_audit,
                "counts": counts,
            }
        signal_to_noise = (
            combined_net / math.sqrt(combined_variance)
            if combined_variance > 0
            else float("nan")
        )
        all_products_passed = all(
            value["product_audit"]["passed"] for value in instrument_results.values()
        )
        net_positive = combined_net > 0
        snr_passed = signal_to_noise >= float(
            geometry["minimum_signal_to_noise_each_annulus"]
        )
        passed = all_products_passed and net_positive and snr_passed
        annuli[annulus_id] = {
            "radial_range_kpc": [edges_kpc[index], edges_kpc[index + 1]],
            "radial_range_detector_units": [
                edges_detector[index],
                edges_detector[index + 1],
            ],
            "instrument_results": instrument_results,
            "combined_net_counts": combined_net,
            "combined_variance_proxy": combined_variance,
            "combined_signal_to_noise": signal_to_noise,
            "all_products_passed": all_products_passed,
            "net_counts_positive": net_positive,
            "signal_to_noise_gate_passed": snr_passed,
            "passed": passed,
        }
        mos2_counts = instrument_results["MOS2"]["counts"] or {}
        pn_counts = instrument_results["pn"]["counts"] or {}
        ledger_rows.append(
            {
                "annulus_id": annulus_id,
                "inner_kpc": edges_kpc[index],
                "outer_kpc": edges_kpc[index + 1],
                "MOS2_observed_counts": mos2_counts.get("observed_counts"),
                "MOS2_ESAS_QPB_counts": mos2_counts.get("ESAS_QPB_counts"),
                "MOS2_conservative_QPB_counts": mos2_counts.get(
                    "conservative_QPB_counts"
                ),
                "MOS2_net_counts": mos2_counts.get("net_counts"),
                "pn_observed_counts_after_OOT": pn_counts.get("observed_counts"),
                "pn_ESAS_QPB_counts": pn_counts.get("ESAS_QPB_counts"),
                "pn_conservative_QPB_counts": pn_counts.get(
                    "conservative_QPB_counts"
                ),
                "pn_net_counts": pn_counts.get("net_counts"),
                "combined_net_counts": combined_net,
                "combined_signal_to_noise": signal_to_noise,
                "passed": passed,
            }
        )

    passing_ids = [name for name, result in annuli.items() if result["passed"]]
    total_net = float(sum(annuli[name]["combined_net_counts"] for name in passing_ids))
    annulus_count_gate = len(passing_ids) >= int(geometry["minimum_accepted_annuli"])
    total_count_gate = total_net >= float(geometry["minimum_total_net_counts"])
    stage_passed = annulus_count_gate and total_count_gate
    generated_utc = datetime.now(timezone.utc).isoformat()

    manifest = {
        "manifest_version": "R1B3-RXJ2129-XMM-X3-annular-products-0.2",
        "generated_utc": generated_utc,
        "protocol": PROTOCOL_PATH.relative_to(PROJECT).as_posix(),
        "source_mask_manifest": SOURCE_MASK_MANIFEST_PATH.relative_to(PROJECT).as_posix(),
        "external_product_root": str(args.product_root),
        "annuli": annuli,
        "passing_annuli": passing_ids,
        "gates": {
            "minimum_annulus_count_gate_passed": annulus_count_gate,
            "minimum_total_net_count_gate_passed": total_count_gate,
            "X3_annular_product_adequacy_gate_passed": stage_passed,
            "X3_gas_likelihood_gate_passed": False,
        },
    }
    report = {
        "report_version": "R1B3-RXJ2129-XMM-X3-annular-products-0.2",
        "generated_utc": generated_utc,
        "stage": "X3_annular_count_response_adequacy",
        "status": "pass" if stage_passed else "fail",
        "outcome": (
            f"{len(passing_ids)} of {len(ids)} immutable annuli pass; passing-annulus "
            f"combined net counts = {total_net:.3f}."
        ),
        "passing_instruments": ["MOS2", "pn"],
        "passing_annuli": passing_ids,
        "minimum_accepted_annuli": geometry["minimum_accepted_annuli"],
        "passing_annulus_combined_net_counts": total_net,
        "minimum_total_net_counts": geometry["minimum_total_net_counts"],
        "minimum_signal_to_noise_each_annulus": geometry[
            "minimum_signal_to_noise_each_annulus"
        ],
        "gates": manifest["gates"],
        "artifacts": {
            "count_ledger": args.ledger.relative_to(PROJECT).as_posix(),
            "product_manifest": args.manifest.relative_to(PROJECT).as_posix(),
        },
        "authorization": {
            "freeze_XMM_specific_gas_likelihood_protocol": stage_passed,
            "fit_temperature_or_density": False,
            "infer_gas_mass": False,
            "infer_dynamical_or_Weyl_response": False,
            "fit_new_force_or_action": False,
        },
    }
    write_ledger(args.ledger, ledger_rows)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not stage_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
