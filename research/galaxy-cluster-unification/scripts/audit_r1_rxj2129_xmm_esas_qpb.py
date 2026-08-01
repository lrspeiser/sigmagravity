#!/usr/bin/env python3
"""Audit RX J2129 ESAS QPB products and evaluate the frozen corner/FWC gate."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from scipy.stats import gamma


PROJECT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT / "configs/r1_rxj2129_xmm_background_mask_protocol.json"
OUTPUT_PATH = PROJECT / "results/r1_rxj2129_xmm_event_processing/qpb_background_audit.json"
LINUX_ESAS_ROOT = Path(
    "/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/"
    "x2b/background/esas_full_fov"
)
WINDOWS_ESAS_ROOT = Path(
    "//wsl.localhost/Ubuntu-24.04/home/henry/.local/share/sigmagravity-xmm/"
    "work/rxj2129/0093030201/x2b/background/esas_full_fov"
)
INSTRUMENTS = {
    "MOS1": {
        "prefix": "mos1S001",
        "task": "mosback",
        "marker": ".mosback_complete",
        "science": "mos1S001-fovt.pi",
        "background": "mos1S001-bkg.pi",
        "image": "mos1S001-bkgimdet-500-7000.fits",
        "rmf": "mos1S001.rmf",
        "arf": "mos1S001.arf",
        "sectors": [2, 3, 4, 5, 6, 7],
    },
    "MOS2": {
        "prefix": "mos2S002",
        "task": "mosback",
        "marker": ".mosback_complete",
        "science": "mos2S002-fovt.pi",
        "background": "mos2S002-bkg.pi",
        "image": "mos2S002-bkgimdet-500-7000.fits",
        "rmf": "mos2S002.rmf",
        "arf": "mos2S002.arf",
        "sectors": [2, 3, 4, 6, 7],
    },
    "pn": {
        "prefix": "pnS003",
        "task": "pnback",
        "marker": ".pnback_complete",
        "science": "pnS003-fovt.pi",
        "background": "pnS003-bkg.pi",
        "image": "pnS003-bkgimdet-500-7000.fits",
        "rmf": "pnS003.rmf",
        "arf": "pnS003.arf",
        "sectors": [1, 2, 3, 4],
    },
}


def parse_args() -> argparse.Namespace:
    default_root = WINDOWS_ESAS_ROOT if os.name == "nt" else LINUX_ESAS_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--esas-root",
        type=Path,
        default=Path(os.environ.get("SIGMAGRAVITY_XMM_ESAS_ROOT", default_root)),
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def spectrum(path: Path) -> dict[str, Any]:
    with fits.open(path, memmap=False) as hdul:
        hdu = hdul["SPECTRUM"]
        names = set(hdu.columns.names)
        channel = np.asarray(hdu.data["CHANNEL"], dtype=int)
        if "COUNTS" in names:
            values = np.asarray(hdu.data["COUNTS"], dtype=float)
            value_kind = "COUNTS"
        elif "RATE" in names:
            values = np.asarray(hdu.data["RATE"], dtype=float)
            value_kind = "RATE"
        else:
            raise ValueError(f"{path} has neither COUNTS nor RATE")
        return {
            "channel": channel,
            "values": values,
            "value_kind": value_kind,
            "exposure_s": float(hdu.header["EXPOSURE"]) if "EXPOSURE" in hdu.header else None,
            "backscal": float(hdu.header["BACKSCAL"]),
            "areascale": float(hdu.header.get("AREASCAL", 1.0)),
            "region_row_counts": [
                len(item.data)
                for item in hdul
                if item.name.startswith("REG") and item.data is not None
            ],
        }


def band_counts(spec: dict[str, Any], low_channel: int, high_channel: int) -> float:
    selected = (spec["channel"] >= low_channel) & (spec["channel"] <= high_channel)
    total = float(np.sum(spec["values"][selected]))
    if spec["value_kind"] == "RATE":
        if spec["exposure_s"] is None:
            raise ValueError("RATE spectrum is missing EXPOSURE")
        total *= spec["exposure_s"]
    return total


def interval_mass(shape: float, rate: float, lower: float, upper: float) -> float:
    """Stable Gamma probability for [lower, upper]."""
    cdf_lower = float(gamma.cdf(lower, a=shape, scale=1.0 / rate))
    cdf_upper = float(gamma.cdf(upper, a=shape, scale=1.0 / rate))
    if cdf_lower > 0.5:
        return max(
            0.0,
            float(gamma.sf(lower, a=shape, scale=1.0 / rate))
            - float(gamma.sf(upper, a=shape, scale=1.0 / rate)),
        )
    return max(0.0, cdf_upper - cdf_lower)


def posterior_summary(
    observed_counts: float,
    expected_counts_at_unit_scale: float,
    bounds: tuple[float, float],
) -> dict[str, Any]:
    """Conditional Jeffreys-Poisson posterior for a multiplicative scale."""
    shape = observed_counts + 0.5
    rate = expected_counts_at_unit_scale
    lower, upper = bounds
    edge_width = 0.02 * (upper - lower)
    denominator = interval_mass(shape, rate, lower, upper)
    if denominator <= 0.0:
        edge_mass = None
        posterior_pass = False
    else:
        low_edge = interval_mass(shape, rate, lower, lower + edge_width)
        high_edge = interval_mass(shape, rate, upper - edge_width, upper)
        edge_mass = (low_edge + high_edge) / denominator
        posterior_pass = edge_mass < 0.05
    return {
        "model": "Jeffreys-Poisson conditional on the ESAS FWC template",
        "shape": shape,
        "rate": rate,
        "mean": shape / rate,
        "sd": math.sqrt(shape) / rate,
        "central_95_interval": [
            float(gamma.ppf(0.025, a=shape, scale=1.0 / rate)),
            float(gamma.ppf(0.975, a=shape, scale=1.0 / rate)),
        ],
        "outer_edge_width": edge_width,
        "posterior_mass_in_outermost_2_percent_of_either_bound": edge_mass,
        "posterior_bound_rule_passed": posterior_pass,
    }


def parse_exposures(instrument: str, log_text: str) -> dict[int, tuple[float, float]]:
    if instrument.startswith("MOS"):
        pattern = re.compile(
            r"Chip\s+(\d+)\s+exposure\s*=\s*([0-9.]+)\s+exposure\s*=\s*([0-9.]+)"
        )
    else:
        pattern = re.compile(
            r"Quadrant\s+(\d+)\s+exposure=\s*([0-9.]+)\s+exposure\s*=\s*([0-9.]+)"
        )
    parsed: dict[int, tuple[float, float]] = {}
    for sector, fwc_exposure, observation_exposure in pattern.findall(log_text):
        parsed[int(sector)] = (float(fwc_exposure), float(observation_exposure))
    return parsed


def validate_product_set(directory: Path, spec: dict[str, Any], log_text: str) -> dict[str, Any]:
    task = spec["task"]
    science = spectrum(directory / spec["science"])
    background = spectrum(directory / spec["background"])
    with fits.open(directory / spec["image"], memmap=False) as hdul:
        image = np.asarray(hdul[0].data, dtype=float)
    with fits.open(directory / "srcdet.fits", memmap=False) as hdul:
        detector_mask_rows = len(hdul[1].data)
    with fits.open(directory / "srcsky.fits", memmap=False) as hdul:
        sky_mask_rows = len(hdul[1].data)
    required = [
        spec["marker"],
        spec["science"],
        spec["background"],
        spec["image"],
        spec["rmf"],
        spec["arf"],
        "srcdet.fits",
        "srcsky.fits",
    ]
    missing = [name for name in required if not (directory / name).is_file()]
    nonempty = all((directory / name).stat().st_size > 0 for name in required if name != spec["marker"])
    science_valid = (
        science["exposure_s"] is not None
        and science["exposure_s"] > 0
        and science["backscal"] > 0
        and np.all(np.isfinite(science["values"]))
        and np.all(science["values"] >= 0)
    )
    background_valid = (
        background["exposure_s"] is not None
        and background["exposure_s"] > 0
        and background["backscal"] > 0
        and np.all(np.isfinite(background["values"]))
        and np.all(background["values"] >= 0)
    )
    image_valid = image.size > 0 and np.all(np.isfinite(image)) and np.all(image >= 0)
    error_records = re.findall(rf"^\*\*\s+{task}:\s+error\b.*$", log_text, flags=re.MULTILINE)
    ended_normally = re.search(rf"{task}\s+\({task}-[^)]+\).*\sended:", log_text) is not None
    warning_summary = {
        key: int(value)
        for key, value in re.findall(r"warning\s+(\w+)\s+silently occurred\s+(\d+)\s+times", log_text)
    }
    passed = (
        not missing
        and nonempty
        and science_valid
        and background_valid
        and image_valid
        and detector_mask_rows == 87
        and sky_mask_rows == 87
        and ended_normally
        and not error_records
    )
    return {
        "passed": passed,
        "missing_products": missing,
        "all_required_nonempty": nonempty,
        "task_ended_normally": ended_normally,
        "task_error_records": error_records,
        "science_spectrum_valid": science_valid,
        "background_spectrum_valid": background_valid,
        "background_image_valid": bool(image_valid),
        "background_image_sum": float(np.sum(image)),
        "source_mask_rows": {"detector": detector_mask_rows, "sky": sky_mask_rows},
        "science_region_row_count_range": [
            min(science["region_row_counts"]),
            max(science["region_row_counts"]),
        ],
        "calibration_intermediate_metadata_warning_summary": warning_summary,
        "warning_classification": (
            "Nonfatal: missing OBS_ID/EXPIDSTR/REVOLUT/SUBMODE warnings occur on selected "
            "FWC calibration intermediates; the final science and QPB products pass identity-"
            "independent exposure, area, finiteness, and completion checks."
        ),
    }


def audit_instrument(
    esas_root: Path,
    instrument: str,
    spec: dict[str, Any],
    bounds: tuple[float, float],
    channels: tuple[int, int],
) -> dict[str, Any]:
    directory = esas_root / instrument
    log_path = directory / f"{spec['task']}.log"
    log_text = log_path.read_text(errors="replace")
    product_audit = validate_product_set(directory, spec, log_text)
    exposures = parse_exposures(instrument, log_text)
    oot_scale = 0.0
    if instrument == "pn":
        scales = {
            float(item)
            for item in re.findall(r"Observation OOT Scale Factor\s*:\s*([0-9.]+)", log_text)
        }
        if len(scales) != 1:
            raise ValueError(f"Expected one unique pn OOT scale, found {sorted(scales)}")
        oot_scale = scales.pop()

    sectors: dict[str, Any] = {}
    total_observed = 0.0
    total_expected = 0.0
    failure_reasons: list[str] = []
    for sector in spec["sectors"]:
        if sector not in exposures:
            raise ValueError(f"Missing exposure summary for {instrument} sector {sector}")
        fwc_exposure, logged_observation_exposure = exposures[sector]
        prefix = spec["prefix"]
        if instrument.startswith("MOS"):
            observed = spectrum(directory / f"{prefix}-corccd{sector}.pi")
            fwc = spectrum(directory / f"{prefix}-corfwcccd{sector}.pi")
            observed_counts = band_counts(observed, *channels)
            fwc_counts = band_counts(fwc, *channels)
        else:
            observed = spectrum(directory / f"{prefix}-corq{sector}.pi")
            observed_oot = spectrum(directory / f"{prefix}-corootq{sector}.pi")
            fwc = spectrum(directory / f"{prefix}-corfwcq{sector}.pi")
            fwc_oot = spectrum(directory / f"{prefix}-corfwcootq{sector}.pi")
            observed_raw = band_counts(observed, *channels)
            observed_oot_counts = band_counts(observed_oot, *channels)
            fwc_raw = band_counts(fwc, *channels)
            fwc_oot_counts = band_counts(fwc_oot, *channels)
            observed_counts = observed_raw - oot_scale * observed_oot_counts
            fwc_counts = fwc_raw - oot_scale * fwc_oot_counts
        if observed["exposure_s"] is None:
            raise ValueError(f"Missing observation exposure for {instrument} sector {sector}")
        expected = fwc_counts * (
            observed["exposure_s"]
            * observed["backscal"]
            / (fwc_exposure * fwc["backscal"])
        )
        scale = observed_counts / expected
        posterior = posterior_summary(observed_counts, expected, bounds)
        scale_passed = bounds[0] < scale < bounds[1]
        sector_passed = scale_passed and posterior["posterior_bound_rule_passed"]
        if not sector_passed:
            failure_reasons.append(f"sector_{sector}_FWC_corner_scale_gate_failed")
        sector_result: dict[str, Any] = {
            "observed_corner_counts": observed_counts,
            "FWC_corner_counts": fwc_counts,
            "observation_exposure_s": observed["exposure_s"],
            "logged_observation_exposure_s": logged_observation_exposure,
            "FWC_exposure_s": fwc_exposure,
            "observation_backscal": observed["backscal"],
            "FWC_backscal": fwc["backscal"],
            "expected_observation_counts_at_unit_FWC_scale": expected,
            "FWC_corner_scale": scale,
            "scale_inside_open_interval": scale_passed,
            "posterior": posterior,
            "passed": sector_passed,
        }
        if instrument == "pn":
            sector_result["OOT_scale"] = oot_scale
            sector_result["observed_corner_counts_before_OOT_subtraction"] = observed_raw
            sector_result["observed_OOT_corner_counts_before_scaling"] = observed_oot_counts
            sector_result["FWC_corner_counts_before_OOT_subtraction"] = fwc_raw
            sector_result["FWC_OOT_corner_counts_before_scaling"] = fwc_oot_counts
        sectors[str(sector)] = sector_result
        total_observed += observed_counts
        total_expected += expected

    pooled_scale = total_observed / total_expected
    pooled_posterior = posterior_summary(total_observed, total_expected, bounds)
    sector_gate_passed = not failure_reasons
    passed = product_audit["passed"] and sector_gate_passed
    if not product_audit["passed"]:
        failure_reasons.insert(0, "ESAS_product_integrity_gate_failed")
    return {
        "passed": passed,
        "failure_reasons": failure_reasons,
        "product_audit": product_audit,
        "sectors": sectors,
        "pooled_diagnostic": {
            "observed_corner_counts": total_observed,
            "expected_observation_counts_at_unit_FWC_scale": total_expected,
            "FWC_corner_scale": pooled_scale,
            "posterior": pooled_posterior,
            "passed_but_not_used_to_override_a_sector_failure": (
                bounds[0] < pooled_scale < bounds[1]
                and pooled_posterior["posterior_bound_rule_passed"]
            ),
        },
    }


def main() -> None:
    args = parse_args()
    protocol = json.loads(PROTOCOL_PATH.read_text())
    bounds = tuple(protocol["background"]["FWC_corner_scale_allowed_open_interval"])
    channels = (101, 1400)
    results = {
        name: audit_instrument(args.esas_root, name, spec, bounds, channels)
        for name, spec in INSTRUMENTS.items()
    }
    passing = [name for name, result in results.items() if result["passed"]]
    minimum = int(protocol["background"]["minimum_instruments_after_background_gate"])
    minimum_gate = len(passing) >= minimum
    report = {
        "report_version": "R1B3-RXJ2129-XMM-X2b2-FWC-corner-1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL_PATH.relative_to(PROJECT).as_posix(),
        "stage": "X2b2_FWC_corner_subgate",
        "status": "pass_minimum_instruments" if minimum_gate else "fail_minimum_instruments",
        "method": {
            "energy_band_eV": [500, 7000],
            "PI_channels_inclusive": list(channels),
            "scale_definition": (
                "Observed corner counts divided by FWC corner counts transferred to the same "
                "exposure and BACKSCAL; pn applies the ESAS OOT factor to both spectra."
            ),
            "instrument_rule": (
                "Every included outer MOS CCD or pn quadrant must pass. The pooled value is a "
                "diagnostic and cannot hide a detector-sector failure."
            ),
            "allowed_open_interval": list(bounds),
            "posterior_bound_rule": protocol["background"]["posterior_bound_rule"],
        },
        "instrument_results": results,
        "passing_instruments": passing,
        "excluded_at_FWC_corner_subgate": [name for name in results if name not in passing],
        "minimum_passing_instruments": minimum,
        "minimum_instrument_gate_passed": minimum_gate,
        "outcome": (
            f"FWC/corner subgate retains {', '.join(passing)}; local outer-annulus transfer "
            "testing is authorized, but full X2 remains false."
            if minimum_gate
            else "Fewer than two instruments pass the FWC/corner subgate; X2b2 fails."
        ),
        "authorization": {
            "run_frozen_local_outer_annulus_transfer_audit": minimum_gate,
            "claim_full_X2_pass": False,
            "construct_X3_annular_gas_likelihood": False,
            "fit_temperature_or_density": False,
            "infer_dynamical_or_Weyl_response": False,
            "fit_new_force_or_action": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=json_default) + "\n")
    print(json.dumps(report, indent=2, default=json_default))
    if not minimum_gate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
