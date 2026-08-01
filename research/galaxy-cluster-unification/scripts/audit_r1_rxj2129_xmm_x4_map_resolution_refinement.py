"""Audit promoted 920 versus sqrt(2)-refined 1302 a04->a04 X4 ARFs."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/r1_rxj2129_xmm_gas_likelihood_protocol.json"
BASELINE_ROOT = Path(
    "//wsl.localhost/Ubuntu-24.04/home/henry/.local/share/sigmagravity-xmm/"
    "work/rxj2129/0093030201/x4/map_resolution_convergence_v0_1"
)
REFINEMENT_ROOT = Path(
    "//wsl.localhost/Ubuntu-24.04/home/henry/.local/share/sigmagravity-xmm/"
    "work/rxj2129/0093030201/x4/map_resolution_convergence_v0_2"
)
X3_ROOT = Path(
    "//wsl.localhost/Ubuntu-24.04/home/henry/.local/share/sigmagravity-xmm/"
    "work/rxj2129/0093030201/x3/annular_products"
)
REPORT = ROOT / "results/r1_rxj2129_xmm_x4_map_resolution_convergence/report.json"
FATAL_PATTERN = re.compile(
    r"\*\* .*: error|detmapXBoundsExceeded|detmapYBoundsExceeded|zeroSumDetmap",
    re.IGNORECASE,
)


def _axis(header: fits.Header, axis: int, size: int) -> np.ndarray:
    return float(header[f"CRVAL{axis}L"]) + (
        np.arange(size, dtype=float) + 1.0 - float(header[f"CRPIX{axis}L"])
    ) * float(header[f"CDELT{axis}L"])


def _map(path: Path, region_path: Path) -> dict[str, Any]:
    with fits.open(path, memmap=True) as hdul:
        image = np.asarray(hdul[0].data, dtype=float)
        header = hdul[0].header
        x = _axis(header, 1, image.shape[1])
        y = _axis(header, 2, image.shape[0])
    match = re.search(
        r"circle\(([-+0-9.eE]+),([-+0-9.eE]+),([-+0-9.eE]+)\).*"
        r"circle\(([-+0-9.eE]+),([-+0-9.eE]+),([-+0-9.eE]+)\)",
        region_path.read_text(encoding="utf-8"),
    )
    if match is None:
        raise ValueError("Could not parse frozen a04 detector annulus")
    cx, cy, outer, cx2, cy2, inner = map(float, match.groups())
    radius2 = (y[:, None] - cy) ** 2 + (x[None, :] - cx) ** 2
    pixels = int(
        np.count_nonzero((radius2 <= outer**2) & (radius2 > inner**2))
    )
    pixel_x = abs(float(header["CDELT1L"]))
    pixel_y = abs(float(header["CDELT2L"]))
    coverage = bool(
        x.min() <= -25000
        and x.max() >= 25000
        and y.min() <= -25000
        and y.max() >= 25000
    )
    valid = bool(
        np.isfinite(image).all()
        and np.all(image == 1.0)
        and np.allclose([cx, cy], [cx2, cy2], atol=1e-9)
        and coverage
        and pixels >= 301
        and pixel_x <= 80.0
        and pixel_y <= 80.0
    )
    return {
        "path": str(path),
        "shape": list(image.shape),
        "pixel_size_detector_units": [pixel_x, pixel_y],
        "maximum_allowed_pixel_size_detector_units": 80.0,
        "pixel_size_gate_passed": bool(pixel_x <= 80.0 and pixel_y <= 80.0),
        "coverage_gate_passed": coverage,
        "a04_region_pixels": pixels,
        "minimum_required_region_pixels": 301,
        "valid": valid,
    }


def _arf(path: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    with fits.open(path, memmap=True) as hdul:
        data = hdul["SPECRESP"].data
        lo = np.asarray(data["ENERG_LO"], dtype=float)
        hi = np.asarray(data["ENERG_HI"], dtype=float)
        response = np.asarray(data["SPECRESP"], dtype=float)
    science = (hi > 0.7) & (lo < 7.0)
    valid = bool(
        len(response) > 0
        and np.isfinite(lo).all()
        and np.isfinite(hi).all()
        and np.isfinite(response).all()
        and np.all(hi > lo)
        and np.all(response >= 0)
        and np.all(response[science] > 0)
    )
    return {
        "path": str(path),
        "rows": int(len(response)),
        "science_band_rows": int(science.sum()),
        "science_band_sum_cm2": float(response[science].sum()),
        "valid": valid,
    }, lo, hi, response


def _log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    fatal = FATAL_PATTERN.findall(text)
    energy = re.findall(r"EnergyOutsideValidityRange\).*?E=([^ ]+) eV", text)
    valid = bool(
        re.search(r"arfgen .* ended:", text)
        and not fatal
        and (not energy or all(float(value) == 0.0 for value in energy))
    )
    return {
        "path": str(path),
        "fatal_records": fatal,
        "displayed_zero_energy_warning_count": len(energy),
        "energy_warnings_confined_to_zero_eV": bool(
            not energy or all(float(value) == 0.0 for value in energy)
        ),
        "valid": valid,
    }


def _instrument(instrument: str, lower: str, gate: dict[str, Any]) -> dict[str, Any]:
    baseline_directory = BASELINE_ROOT / instrument
    refined_directory = REFINEMENT_ROOT / instrument
    region = X3_ROOT / f"a04_175_275kpc/{instrument}/annulus_region.txt"
    maps = {
        "baseline": _map(baseline_directory / f"{lower}_uniform_920.fits", region),
        "refined": _map(refined_directory / f"{lower}_uniform_1302.fits", region),
    }
    baseline_audit, base_lo, base_hi, baseline = _arf(
        baseline_directory / f"{instrument}_a04_from_a04_cross_920.arf"
    )
    refined_audit, ref_lo, ref_hi, refined = _arf(
        refined_directory / f"{instrument}_a04_from_a04_cross_1302.arf"
    )
    common_grid = bool(np.array_equal(base_lo, ref_lo) and np.array_equal(base_hi, ref_hi))
    science = (ref_hi > 0.7) & (ref_lo < 7.0)
    fractional = np.abs(baseline[science] - refined[science]) / refined[science]
    differences = {
        "band_integrated_fractional_change": float(abs(baseline[science].sum() / refined[science].sum() - 1.0)),
        "median_fit_band_fractional_change": float(np.median(fractional)),
        "p95_fit_band_fractional_change": float(np.percentile(fractional, 95)),
        "maximum_fit_band_fractional_change": float(fractional.max()),
    }
    thresholds = {
        "band_integrated_fractional_change": float(gate["maximum_band_integrated_fractional_change"]),
        "median_fit_band_fractional_change": float(gate["maximum_median_fit_band_fractional_change"]),
        "p95_fit_band_fractional_change": float(gate["maximum_95th_percentile_fit_band_fractional_change"]),
    }
    comparison_passed = bool(all(differences[name] <= value for name, value in thresholds.items()))
    logs = {
        "baseline": _log(baseline_directory / "arfgen_cross_920.log"),
        "refined": _log(refined_directory / "arfgen_cross_1302.log"),
    }
    completion_markers = {
        "baseline_map": (
            baseline_directory / f".{lower}_uniform_920_complete"
        ).is_file(),
        "baseline_arf": (baseline_directory / ".cross_920_complete").is_file(),
        "refined_map": (
            refined_directory / f".{lower}_uniform_1302_complete"
        ).is_file(),
        "refined_arf": (refined_directory / ".cross_1302_complete").is_file(),
        "refined_instrument": (
            refined_directory / ".instrument_convergence_complete"
        ).is_file(),
    }
    passed = bool(
        all(item["valid"] for item in maps.values())
        and baseline_audit["valid"]
        and refined_audit["valid"]
        and common_grid
        and all(item["valid"] for item in logs.values())
        and maps["refined"]["pixel_size_detector_units"][0]
        < maps["baseline"]["pixel_size_detector_units"][0]
        and comparison_passed
        and all(completion_markers.values())
    )
    return {
        "status": "pass" if passed else "fail",
        "maps": maps,
        "arfs": {"baseline": baseline_audit, "refined": refined_audit},
        "common_energy_grid": common_grid,
        "differences": differences,
        "thresholds": thresholds,
        "comparison_passed": comparison_passed,
        "logs": logs,
        "completion_markers": completion_markers,
        "passed": passed,
    }


def audit() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    gate = protocol["X4_response_calibration"]["detector_maps"]["resolution_convergence_gate"]
    if (gate["baseline_requested_dimension"], gate["refined_requested_dimension"]) != (920, 1302):
        raise ValueError("Protocol does not authorize the promoted 920-to-1302 comparison")
    instruments = {"MOS2": _instrument("MOS2", "mos2", gate), "pn": _instrument("pn", "pn", gate)}
    root_marker = (REFINEMENT_ROOT / ".map_resolution_convergence_complete").is_file()
    passed = bool(
        all(item["passed"] for item in instruments.values()) and root_marker
    )
    report = {
        "report_version": "R1B3-RXJ2129-XMM-X4-map-resolution-0.2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if passed else "fail",
        "representative_pair": gate["representative_pair"],
        "baseline_requested_dimension": 920,
        "refined_requested_dimension": 1302,
        "instruments": instruments,
        "completion_marker_present": root_marker,
        "gates": {"X4_map_resolution_convergence_passed": passed},
        "authorization": {
            "construct_full_X4_at_baseline_resolution": passed,
            "fit_temperature_or_density": False,
            "infer_gas_mass": False,
            "infer_dynamical_or_Weyl_response": False,
            "fit_new_force_or_action": False,
        },
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    report = audit()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
