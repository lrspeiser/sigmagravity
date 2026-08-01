"""Audit the quarantined MOS2 X4 direct, cross-region, and central-source interface."""

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
MOS2_ROOT = Path(
    "//wsl.localhost/Ubuntu-24.04/home/henry/.local/share/sigmagravity-xmm/"
    "work/rxj2129/0093030201/x4/interface_v0_2_mos2_a01_from_a02"
)
PN_ROOT = Path(
    "//wsl.localhost/Ubuntu-24.04/home/henry/.local/share/sigmagravity-xmm/"
    "work/rxj2129/0093030201/x4/interface_v0_1_pn_a01_from_a02"
)
X3_ROOT = Path(
    "//wsl.localhost/Ubuntu-24.04/home/henry/.local/share/sigmagravity-xmm/"
    "work/rxj2129/0093030201/x3/annular_products"
)
REPORT = ROOT / "results/r1_rxj2129_xmm_x4_response_interface/report.json"
FATAL_PATTERN = re.compile(
    r"\*\* .*: error|detmapXBoundsExceeded|detmapYBoundsExceeded|zeroSumDetmap",
    re.IGNORECASE,
)


def _linear_axis(header: fits.Header, axis: int, size: int) -> np.ndarray:
    # SAS writes angular primary WCS plus the detector-unit "L" alternate WCS.
    # The frozen annular radii and arfgen detector-map bounds use the latter.
    suffix = "L" if header.get(f"CTYPE{axis}L") in {"DETX", "DETY"} else ""
    crval = float(header[f"CRVAL{axis}{suffix}"])
    crpix = float(header[f"CRPIX{axis}{suffix}"])
    cdelt = float(header[f"CDELT{axis}{suffix}"])
    return crval + (np.arange(size, dtype=float) + 1.0 - crpix) * cdelt


def _detector_map_audit(path: Path, region_path: Path) -> dict[str, Any]:
    with fits.open(path, memmap=True) as hdul:
        image = np.asarray(hdul[0].data, dtype=float)
        header = hdul[0].header
        x = _linear_axis(header, 1, image.shape[1])
        y = _linear_axis(header, 2, image.shape[0])

    region = region_path.read_text(encoding="utf-8")
    match = re.search(
        r"circle\(([-+0-9.eE]+),([-+0-9.eE]+),([-+0-9.eE]+)\).*"
        r"circle\(([-+0-9.eE]+),([-+0-9.eE]+),([-+0-9.eE]+)\)",
        region,
    )
    if match is None:
        raise ValueError("Could not parse frozen a01 detector annulus")
    center_x, center_y, outer, second_x, second_y, inner = map(float, match.groups())
    same_center = bool(np.allclose([center_x, center_y], [second_x, second_y], atol=1e-9))
    xx, yy = np.meshgrid(x, y)
    radius = np.hypot(xx - center_x, yy - center_y)
    a01_pixels = int(np.count_nonzero((radius <= outer) & (radius > inner)))
    finite = bool(np.isfinite(image).all())
    uniform = bool(finite and np.all(image == 1.0))
    bounds = {
        "x_min_center": float(x.min()),
        "x_max_center": float(x.max()),
        "y_min_center": float(y.min()),
        "y_max_center": float(y.max()),
    }
    coverage = bool(
        bounds["x_min_center"] <= -25000
        and bounds["x_max_center"] >= 25000
        and bounds["y_min_center"] <= -25000
        and bounds["y_max_center"] >= 25000
    )
    return {
        "shape": list(image.shape),
        "finite": finite,
        "uniform_unit_weight": uniform,
        "bounds_detector_units": bounds,
        "coverage_gate_passed": coverage,
        "frozen_annulus_center_consistent": same_center,
        "a01_map_pixel_count": a01_pixels,
        "minimum_required_output_pixels": 301,
        "output_pixel_gate_passed": bool(a01_pixels >= 301),
    }


def _rmf_audit(path: Path) -> dict[str, Any]:
    with fits.open(path, memmap=False) as hdul:
        matrix = hdul["MATRIX"].data
        ebounds = hdul["EBOUNDS"].data
        energy_lo = np.asarray(matrix["ENERG_LO"], dtype=float)
        energy_hi = np.asarray(matrix["ENERG_HI"], dtype=float)
        values = np.concatenate(
            [np.asarray(row, dtype=float).ravel() for row in matrix["MATRIX"]]
        )
        channels = np.asarray(ebounds["CHANNEL"], dtype=float)
    valid = bool(
        len(matrix) > 0
        and len(ebounds) > 0
        and np.isfinite(energy_lo).all()
        and np.isfinite(energy_hi).all()
        and np.all(energy_hi > energy_lo)
        and np.isfinite(values).all()
        and np.all(values >= 0)
        and np.any(values > 0)
        and np.isfinite(channels).all()
    )
    return {
        "matrix_rows": int(len(matrix)),
        "channel_rows": int(len(ebounds)),
        "positive_matrix_elements": int(np.count_nonzero(values > 0)),
        "matrix_sum": float(values.sum()),
        "valid": valid,
    }


def _arf_audit(path: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    with fits.open(path, memmap=True) as hdul:
        data = hdul["SPECRESP"].data
        energy_lo = np.asarray(data["ENERG_LO"], dtype=float)
        energy_hi = np.asarray(data["ENERG_HI"], dtype=float)
        response = np.asarray(data["SPECRESP"], dtype=float)
    science = (energy_hi > 0.7) & (energy_lo < 7.0)
    valid = bool(
        len(response) > 0
        and np.isfinite(energy_lo).all()
        and np.isfinite(energy_hi).all()
        and np.isfinite(response).all()
        and np.all(energy_hi > energy_lo)
        and np.all(response >= 0)
        and np.any(response[science] > 0)
    )
    audit = {
        "rows": int(len(response)),
        "science_band_rows": int(science.sum()),
        "science_band_positive_rows": int(np.count_nonzero(response[science] > 0)),
        "science_band_response_sum_cm2": float(response[science].sum()),
        "science_band_response_max_cm2": float(response[science].max()),
        "valid": valid,
    }
    return audit, energy_lo, energy_hi, response


def _log_audit(path: Path, task: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    fatal = FATAL_PATTERN.findall(text)
    warnings = re.findall(r"\*\* [^\n]*: warning \(([^)]+)\)", text)
    energy_lines = re.findall(
        r"EnergyOutsideValidityRange\).*?E=([^ ]+) eV", text
    )
    energy_warning_valid = bool(
        not energy_lines or all(float(value) == 0.0 for value in energy_lines)
    )
    normal_end = bool(re.search(rf"{re.escape(task)} .* ended:", text))
    return {
        "normal_end": normal_end,
        "fatal_records": fatal,
        "warning_codes": sorted(set(warnings)),
        "displayed_zero_energy_warning_count": len(energy_lines),
        "energy_warnings_confined_to_zero_eV": energy_warning_valid,
        "valid": bool(normal_end and not fatal and energy_warning_valid),
    }


def _instrument_audit(
    external_root: Path,
    prefix: str,
    region_path: Path,
    invalid_predecessor_root: str | None = None,
) -> dict[str, Any]:
    required = {
        "marker": external_root / ".interface_complete",
        "detmap": external_root / f"{prefix.lower()}_uniform_detmap.fits",
        "rmf": external_root / f"{prefix}_output_a01.rmf",
        "direct": external_root / f"{prefix}_output_a01_direct.arf",
        "cross": external_root / f"{prefix}_output_a01_from_input_a02_cross.arf",
        "central": external_root / f"{prefix}_output_a01_from_central_source50.arf",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {prefix} X4 interface products: {missing}")

    detmap = _detector_map_audit(required["detmap"], region_path)
    rmf = _rmf_audit(required["rmf"])
    arfs: dict[str, dict[str, Any]] = {}
    grids: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    responses: dict[str, np.ndarray] = {}
    for name in ("direct", "cross", "central"):
        arf, lo, hi, response = _arf_audit(required[name])
        arfs[name] = arf
        grids[name] = (lo, hi)
        responses[name] = response
    common_grid = bool(
        np.array_equal(grids["direct"][0], grids["cross"][0])
        and np.array_equal(grids["direct"][1], grids["cross"][1])
        and np.array_equal(grids["direct"][0], grids["central"][0])
        and np.array_equal(grids["direct"][1], grids["central"][1])
    )
    science = (grids["direct"][1] > 0.7) & (grids["direct"][0] < 7.0)
    direct = responses["direct"][science]
    cross = responses["cross"][science]
    central = responses["central"][science]
    coupling = {
        "cross_to_direct_integrated_ratio": float(cross.sum() / direct.sum()),
        "central_to_direct_integrated_ratio": float(central.sum() / direct.sum()),
    }

    logs = {
        "rmfgen": _log_audit(external_root / "rmfgen.log", "rmfgen"),
        "direct": _log_audit(external_root / "arfgen_direct.log", "arfgen"),
        "cross": _log_audit(external_root / "arfgen_cross.log", "arfgen"),
        "central": _log_audit(external_root / "arfgen_central.log", "arfgen"),
    }
    passed = bool(
        detmap["finite"]
        and detmap["uniform_unit_weight"]
        and detmap["coverage_gate_passed"]
        and detmap["frozen_annulus_center_consistent"]
        and detmap["output_pixel_gate_passed"]
        and rmf["valid"]
        and all(item["valid"] for item in arfs.values())
        and common_grid
        and all(item["valid"] for item in logs.values())
        and coupling["cross_to_direct_integrated_ratio"] > 0
        and coupling["central_to_direct_integrated_ratio"] > 0
    )
    return {
        "status": "pass" if passed else "fail",
        "external_root": str(external_root),
        "invalid_predecessor_root": invalid_predecessor_root,
        "detector_map": detmap,
        "rmf": rmf,
        "arfs": arfs,
        "common_energy_grid": common_grid,
        "science_band_coupling": coupling,
        "logs": logs,
        "passed": passed,
    }


def audit() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["authorization"]["fit_X5_temperature_or_density_before_X4_pass"]:
        raise ValueError("Interface protocol cannot authorize an X5 gas fit")
    instruments = {
        "MOS2": _instrument_audit(
            MOS2_ROOT,
            "MOS2",
            X3_ROOT / "a01_010_050kpc/MOS2/annulus_region.txt",
            (
                "/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/"
                "0093030201/x4/interface_v0_1_mos2_a01_from_a02"
            ),
        ),
        "pn": _instrument_audit(
            PN_ROOT,
            "pn",
            X3_ROOT / "a01_010_050kpc/pn/annulus_region.txt",
        ),
    }
    passed = bool(all(item["passed"] for item in instruments.values()))
    report = {
        "report_version": "R1B3-RXJ2129-XMM-X4-interface-0.2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if passed else "fail",
        "scope": "MOS2 and pn output a01, diffuse input a02, and central source50",
        "instruments": instruments,
        "gates": {"X4_response_interface_passed": passed},
        "authorization": {
            "scale_interface_to_full_X4_product_set": passed,
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
