#!/usr/bin/env python3
"""Evaluate the frozen RX J2129 local outer-annulus QPB-transfer gate."""

from __future__ import annotations

import argparse
import hashlib
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
OUTPUT_PATH = PROJECT / "results/r1_rxj2129_xmm_event_processing/outer_annulus_transfer_audit.json"
LINUX_ROOT = Path(
    "/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/"
    "x2b/background/esas_outer_annulus"
)
WINDOWS_ROOT = Path(
    "//wsl.localhost/Ubuntu-24.04/home/henry/.local/share/sigmagravity-xmm/"
    "work/rxj2129/0093030201/x2b/background/esas_outer_annulus"
)
INSTRUMENTS = {
    "MOS2": {
        "task": "mosback",
        "spectra_task": "mosspectra",
        "spectra_marker": ".mosspectra_outer_complete",
        "background_marker": ".mosback_outer_complete",
        "prefix": "mos2S002",
        "hard_band_eV": [9500.0, 11500.0],
    },
    "pn": {
        "task": "pnback",
        "spectra_task": "pnspectra",
        "spectra_marker": ".pnspectra_outer_complete",
        "background_marker": ".pnback_outer_complete",
        "prefix": "pnS003",
        "hard_band_eV": [10000.0, 12000.0],
    },
}


def parse_args() -> argparse.Namespace:
    default_root = WINDOWS_ROOT if os.name == "nt" else LINUX_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outer-root",
        type=Path,
        default=Path(os.environ.get("SIGMAGRAVITY_XMM_OUTER_ROOT", default_root)),
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


def counts_in_band(spec: dict[str, Any], channels: tuple[int, int]) -> float:
    selected = (spec["channel"] >= channels[0]) & (spec["channel"] <= channels[1])
    total = float(np.sum(spec["values"][selected]))
    return total * spec["exposure_s"] if spec["kind"] == "RATE" else total


def stable_interval_mass(shape: float, rate: float, lower: float, upper: float) -> float:
    cdf_lower = float(gamma.cdf(lower, a=shape, scale=1.0 / rate))
    cdf_upper = float(gamma.cdf(upper, a=shape, scale=1.0 / rate))
    if cdf_lower > 0.5:
        return max(
            0.0,
            float(gamma.sf(lower, a=shape, scale=1.0 / rate))
            - float(gamma.sf(upper, a=shape, scale=1.0 / rate)),
        )
    return max(0.0, cdf_upper - cdf_lower)


def posterior(observed: float, model: float, bounds: tuple[float, float]) -> dict[str, Any]:
    shape = observed + 0.5
    rate = model
    low, high = bounds
    edge_width = 0.02 * (high - low)
    denominator = stable_interval_mass(shape, rate, low, high)
    if denominator <= 0:
        edge_mass = None
        passed = False
    else:
        edge_mass = (
            stable_interval_mass(shape, rate, low, low + edge_width)
            + stable_interval_mass(shape, rate, high - edge_width, high)
        ) / denominator
        passed = edge_mass < 0.05
    return {
        "model": "Jeffreys-Poisson conditional on the ESAS QPB spectrum",
        "shape": shape,
        "rate": rate,
        "mean": shape / rate,
        "sd": math.sqrt(shape) / rate,
        "central_95_interval": [
            float(gamma.ppf(0.025, a=shape, scale=1.0 / rate)),
            float(gamma.ppf(0.975, a=shape, scale=1.0 / rate)),
        ],
        "posterior_mass_in_outermost_2_percent_of_either_bound": edge_mass,
        "posterior_bound_rule_passed": passed,
    }


def task_audit(directory: Path, spec: dict[str, Any]) -> dict[str, Any]:
    spectra_log = (directory / f"{spec['spectra_task']}.log").read_text(errors="replace")
    background_log = (directory / f"{spec['task']}.log").read_text(errors="replace")
    spectra_errors = re.findall(
        rf"^\*\*\s+{spec['spectra_task']}:\s+error\b.*$", spectra_log, flags=re.MULTILINE
    )
    background_errors = re.findall(
        rf"^\*\*\s+{spec['task']}:\s+error\b.*$", background_log, flags=re.MULTILINE
    )
    spectra_ended = re.search(
        rf"{spec['spectra_task']}\s+\({spec['spectra_task']}-[^)]+\).*\sended:",
        spectra_log,
    ) is not None
    background_ended = re.search(
        rf"{spec['task']}\s+\({spec['task']}-[^)]+\).*\sended:", background_log
    ) is not None
    region_text = (directory / "outer_region.txt").read_text().strip()
    region_numbers = [float(item) for item in re.findall(r"-?[0-9]+(?:\.[0-9]+)?", region_text)]
    region_valid = (
        region_text.count("circle(") == 2
        and len(region_numbers) == 6
        and math.isclose(region_numbers[2], 4820.21694730, abs_tol=1e-6)
        and math.isclose(region_numbers[5], 3481.26779527, abs_tol=1e-6)
    )
    required = [
        spec["spectra_marker"],
        spec["background_marker"],
        f"{spec['prefix']}-fovt.pi",
        f"{spec['prefix']}-bkg.pi",
        f"{spec['prefix']}.rmf",
        f"{spec['prefix']}.arf",
        "srcdet.fits",
        "srcsky.fits",
        "outer_region.txt",
    ]
    if spec["prefix"] == "pnS003":
        required.append("pnS003-fovt-oot.pi")
    missing = [item for item in required if not (directory / item).is_file()]
    passed = (
        not missing
        and spectra_ended
        and background_ended
        and not spectra_errors
        and not background_errors
        and region_valid
        and "outer_region.txt" in spectra_log
        and "srcdet.fits" in spectra_log
    )
    return {
        "passed": passed,
        "missing_products": missing,
        "spectra_task_ended_normally": spectra_ended,
        "background_task_ended_normally": background_ended,
        "spectra_task_error_records": spectra_errors,
        "background_task_error_records": background_errors,
        "frozen_region_expression_valid": region_valid,
        "region_expression_sha256": hashlib.sha256((region_text + "\n").encode()).hexdigest(),
    }


def audit_instrument(
    root: Path,
    instrument: str,
    spec: dict[str, Any],
    bounds: tuple[float, float],
    minimum_counts: int,
) -> dict[str, Any]:
    directory = root / instrument
    integrity = task_audit(directory, spec)
    low_eV, high_eV = spec["hard_band_eV"]
    channels = (int(low_eV // 5) + 1, int(high_eV // 5))
    observed_spec = read_spectrum(directory / f"{spec['prefix']}-fovt.pi")
    observed_before_oot = counts_in_band(observed_spec, channels)
    oot_scale = 0.0
    oot_counts = 0.0
    if instrument == "pn":
        background_log = (directory / "pnback.log").read_text(errors="replace")
        scales = {
            float(item)
            for item in re.findall(
                r"Observation OOT Scale Factor\s*:\s*([0-9.]+)", background_log
            )
        }
        if len(scales) != 1:
            raise ValueError(f"Expected one unique pn OOT scale, found {sorted(scales)}")
        oot_scale = scales.pop()
        oot_spec = read_spectrum(directory / "pnS003-fovt-oot.pi")
        oot_counts = counts_in_band(oot_spec, channels)
    observed = observed_before_oot - oot_scale * oot_counts
    model_spec = read_spectrum(directory / f"{spec['prefix']}-bkg.pi")
    model = counts_in_band(model_spec, channels)
    if observed <= 0 or model <= 0:
        raise ValueError(f"Non-positive transfer counts for {instrument}: {observed=}, {model=}")
    transfer_scale = observed / model
    post = posterior(observed, model, bounds)
    count_gate = observed >= minimum_counts
    scale_gate = bounds[0] < transfer_scale < bounds[1]
    passed = integrity["passed"] and count_gate and scale_gate and post["posterior_bound_rule_passed"]
    return {
        "passed": passed,
        "failure_reasons": [
            reason
            for condition, reason in (
                (integrity["passed"], "ESAS_outer_annulus_product_integrity_gate_failed"),
                (count_gate, "minimum_observed_hard_band_counts_failed"),
                (scale_gate, "outer_annulus_transfer_scale_interval_failed"),
                (post["posterior_bound_rule_passed"], "posterior_boundary_mass_gate_failed"),
            )
            if not condition
        ],
        "product_audit": integrity,
        "hard_band_eV": spec["hard_band_eV"],
        "PI_channels_inclusive": list(channels),
        "observed_counts_before_OOT_subtraction": observed_before_oot,
        "OOT_scale": oot_scale,
        "OOT_counts_before_scaling": oot_counts,
        "observed_counts": observed,
        "ESAS_model_QPB_counts": model,
        "outer_annulus_transfer_scale": transfer_scale,
        "minimum_count_gate_passed": count_gate,
        "scale_inside_open_interval": scale_gate,
        "posterior": post,
    }


def main() -> None:
    args = parse_args()
    protocol = json.loads(PROTOCOL_PATH.read_text())
    background = protocol["background"]
    definition = background["local_outer_annulus_transfer_scale_definition"]
    bounds = tuple(background["outer_annulus_transfer_scale_allowed_open_interval"])
    minimum_counts = int(definition["minimum_observed_hard_band_counts"])
    results = {
        instrument: audit_instrument(args.outer_root, instrument, spec, bounds, minimum_counts)
        for instrument, spec in INSTRUMENTS.items()
    }
    passing = [name for name, result in results.items() if result["passed"]]
    minimum_instruments = int(background["minimum_instruments_after_background_gate"])
    gate_passed = len(passing) >= minimum_instruments
    report = {
        "report_version": "R1B3-RXJ2129-XMM-X2b2-outer-annulus-1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL_PATH.relative_to(PROJECT).as_posix(),
        "stage": "X2b2_local_outer_annulus_transfer_subgate",
        "status": "pass" if gate_passed else "fail",
        "frozen_region": background["local_outer_annulus"],
        "method": definition,
        "allowed_open_interval": list(bounds),
        "instrument_results": results,
        "passing_instruments": passing,
        "minimum_passing_instruments": minimum_instruments,
        "full_X2b2_background_gate_passed": gate_passed,
        "authorization": {
            "set_full_X2_gate_true": gate_passed,
            "construct_X3_annular_count_response_products": gate_passed,
            "fit_temperature_or_density": False,
            "infer_dynamical_or_Weyl_response": False,
            "fit_new_force_or_action": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=json_default) + "\n")
    print(json.dumps(report, indent=2, default=json_default))
    if not gate_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
