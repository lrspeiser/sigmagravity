"""Audit the complete frozen MOS2+pn X4 response package and write its manifest."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/r1_rxj2129_xmm_gas_likelihood_protocol.json"
CONVERGENCE = ROOT / "results/r1_rxj2129_xmm_x4_map_resolution_convergence/report.json"
PRODUCTION_RUNNER = ROOT / "scripts/run_r1_rxj2129_xmm_x4_responses.sh"
EXTERNAL_ROOT = Path(
    "//wsl.localhost/Ubuntu-24.04/home/henry/.local/share/sigmagravity-xmm/"
    "work/rxj2129/0093030201/x4/cross_region_responses"
)
X3_ROOT = Path(
    "//wsl.localhost/Ubuntu-24.04/home/henry/.local/share/sigmagravity-xmm/"
    "work/rxj2129/0093030201/x3/annular_products"
)
REPORT = ROOT / "results/r1_rxj2129_xmm_x4_responses/report.json"
MANIFEST = ROOT / "data/derived/r1_rxj2129_xmm_x4_response_manifest.json"
ANNULI = (
    "a01_010_050kpc",
    "a02_050_100kpc",
    "a03_100_175kpc",
    "a04_175_275kpc",
    "a05_275_380kpc",
    "a06_380_500kpc",
)
INSTRUMENTS = {
    "MOS2": {"lower": "mos2", "fits_identity": "EMOS2"},
    "pn": {"lower": "pn", "fits_identity": "EPN"},
}
FATAL_PATTERN = re.compile(
    r"\*\* .*: error|detmapXBoundsExceeded|detmapYBoundsExceeded|zeroSumDetmap",
    re.IGNORECASE,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _product(path: Path, kind: str, **identities: Any) -> dict[str, Any]:
    return {
        "kind": kind,
        **identities,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _axis(header: fits.Header, axis: int, size: int) -> np.ndarray:
    return float(header[f"CRVAL{axis}L"]) + (
        np.arange(size, dtype=float) + 1.0 - float(header[f"CRPIX{axis}L"])
    ) * float(header[f"CDELT{axis}L"])


def _region_geometry(path: Path) -> tuple[float, float, float, float]:
    match = re.search(
        r"circle\(([-+0-9.eE]+),([-+0-9.eE]+),([-+0-9.eE]+)\).*"
        r"circle\(([-+0-9.eE]+),([-+0-9.eE]+),([-+0-9.eE]+)\)",
        path.read_text(encoding="utf-8"),
    )
    if match is None:
        raise ValueError(f"Could not parse frozen detector annulus: {path}")
    cx, cy, outer, cx2, cy2, inner = map(float, match.groups())
    if not np.allclose([cx, cy], [cx2, cy2], atol=1e-9):
        raise ValueError(f"Frozen annulus centers differ: {path}")
    return cx, cy, inner, outer


def _map_audit(
    path: Path,
    requested_dimension: int,
    region_paths: dict[str, Path],
) -> dict[str, Any]:
    with fits.open(path, memmap=True) as hdul:
        image = np.asarray(hdul[0].data)
        header = hdul[0].header
        x = _axis(header, 1, image.shape[1])
        y = _axis(header, 2, image.shape[0])
        finite = bool(np.isfinite(image).all())
        unit = bool(finite and np.all(image == 1.0))
        pixel_x = abs(float(header["CDELT1L"]))
        pixel_y = abs(float(header["CDELT2L"]))
    coverage = bool(
        x.min() <= -25000
        and x.max() >= 25000
        and y.min() <= -25000
        and y.max() >= 25000
    )
    counts: dict[str, int] = {}
    for annulus, region_path in region_paths.items():
        cx, cy, inner, outer = _region_geometry(region_path)
        radius2 = (y[:, None] - cy) ** 2 + (x[None, :] - cx) ** 2
        counts[annulus] = int(
            np.count_nonzero((radius2 <= outer**2) & (radius2 > inner**2))
        )
    pixel_gate = bool(pixel_x <= 80.0 and pixel_y <= 80.0)
    region_gate = bool(counts and min(counts.values()) >= 301)
    valid = bool(finite and unit and coverage and pixel_gate and region_gate)
    return {
        "path": str(path),
        "requested_dimension": requested_dimension,
        "realized_shape": list(image.shape),
        "pixel_size_detector_units": [pixel_x, pixel_y],
        "maximum_allowed_pixel_size_detector_units": 80.0,
        "pixel_size_gate_passed": pixel_gate,
        "uniform_unit_weight": unit,
        "coverage_gate_passed": coverage,
        "annular_pixel_counts": counts,
        "minimum_required_annular_pixels": 301,
        "annular_pixel_gate_passed": region_gate,
        "valid": valid,
    }


def _rmf_audit(path: Path, expected_identity: str) -> dict[str, Any]:
    with fits.open(path, memmap=False) as hdul:
        matrix = hdul["MATRIX"].data
        ebounds = hdul["EBOUNDS"].data
        header = hdul["MATRIX"].header
        lo = np.asarray(matrix["ENERG_LO"], dtype=float)
        hi = np.asarray(matrix["ENERG_HI"], dtype=float)
        values = np.concatenate(
            [np.asarray(row, dtype=float).ravel() for row in matrix["MATRIX"]]
        )
    identity = header.get("INSTRUME") == expected_identity
    valid = bool(
        len(matrix) > 0
        and len(ebounds) > 0
        and identity
        and np.isfinite(lo).all()
        and np.isfinite(hi).all()
        and np.all(hi > lo)
        and np.isfinite(values).all()
        and np.all(values >= 0)
        and np.any(values > 0)
    )
    return {
        "matrix_rows": int(len(matrix)),
        "channel_rows": int(len(ebounds)),
        "fits_instrument": header.get("INSTRUME"),
        "identity_gate_passed": identity,
        "positive_matrix_elements": int(np.count_nonzero(values > 0)),
        "matrix_sum": float(values.sum()),
        "valid": valid,
    }


def _arf_audit(
    path: Path, expected_identity: str
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    with fits.open(path, memmap=True) as hdul:
        data = hdul["SPECRESP"].data
        header = hdul["SPECRESP"].header
        lo = np.asarray(data["ENERG_LO"], dtype=float)
        hi = np.asarray(data["ENERG_HI"], dtype=float)
        response = np.asarray(data["SPECRESP"], dtype=float)
    science = (hi > 0.7) & (lo < 7.0)
    identity = header.get("INSTRUME") == expected_identity
    positive_fit_band = bool(science.any() and np.all(response[science] > 0))
    valid = bool(
        len(response) > 0
        and identity
        and np.isfinite(lo).all()
        and np.isfinite(hi).all()
        and np.isfinite(response).all()
        and np.all(hi > lo)
        and np.all(response >= 0)
        and positive_fit_band
    )
    return {
        "rows": int(len(response)),
        "science_band_rows": int(science.sum()),
        "science_band_positive_rows": int(np.count_nonzero(response[science] > 0)),
        "science_band_sum_cm2": float(response[science].sum()),
        "science_band_max_cm2": float(response[science].max()),
        "fits_instrument": header.get("INSTRUME"),
        "identity_gate_passed": identity,
        "all_fit_band_rows_positive": positive_fit_band,
        "valid": valid,
    }, lo, hi, response


def _log_audit(path: Path, task: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    fatal = FATAL_PATTERN.findall(text)
    warning_codes = sorted(set(re.findall(r"\*\* [^\n]*: warning \(([^)]+)\)", text)))
    energy = re.findall(r"EnergyOutsideValidityRange\).*?E=([^ ]+) eV", text)
    zero_only = bool(not energy or all(float(value) == 0.0 for value in energy))
    normal_end = bool(re.search(rf"{re.escape(task)} .* ended:", text))
    return {
        "path": str(path),
        "normal_end": normal_end,
        "fatal_records": fatal,
        "warning_codes": warning_codes,
        "displayed_zero_energy_warning_count": len(energy),
        "energy_warnings_confined_to_zero_eV": zero_only,
        "valid": bool(normal_end and not fatal and zero_only),
    }


def _instrument(
    instrument: str,
    specification: dict[str, str],
    dimensions: dict[str, int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lower = specification["lower"]
    identity = specification["fits_identity"]
    products: list[dict[str, Any]] = []
    maps: dict[str, Any] = {}
    for dimension in sorted(set(dimensions.values()), reverse=True):
        annuli = [annulus for annulus in ANNULI if dimensions[annulus] == dimension]
        path = EXTERNAL_ROOT / "detmaps" / instrument / f"{lower}_uniform_{dimension}.fits"
        marker = path.parent / f".{lower}_uniform_{dimension}_complete"
        region_paths = {
            annulus: X3_ROOT / annulus / instrument / "annulus_region.txt"
            for annulus in annuli
        }
        maps[str(dimension)] = {
            **_map_audit(path, dimension, region_paths),
            "completion_marker_present": marker.is_file(),
        }
        maps[str(dimension)]["valid"] = bool(
            maps[str(dimension)]["valid"] and marker.is_file()
        )
        products.append(_product(path, "detector_map", instrument=instrument, requested_dimension=dimension))

    outputs: dict[str, Any] = {}
    cross_matrix: list[list[float]] = []
    for output in ANNULI:
        directory = EXTERNAL_ROOT / instrument / output
        rmf_path = directory / f"{instrument}_{output}.rmf"
        direct_path = directory / f"{instrument}_{output}_direct.arf"
        central_path = directory / f"{instrument}_{output}_from_central_source50.arf"
        rmf = _rmf_audit(rmf_path, identity)
        direct, direct_lo, direct_hi, direct_response = _arf_audit(direct_path, identity)
        central, central_lo, central_hi, central_response = _arf_audit(central_path, identity)
        science = (direct_hi > 0.7) & (direct_lo < 7.0)
        common_grid = bool(
            np.array_equal(direct_lo, central_lo)
            and np.array_equal(direct_hi, central_hi)
        )
        cross: dict[str, Any] = {}
        cross_row: list[float] = []
        logs = {
            "rmf": _log_audit(directory / "rmfgen.log", "rmfgen"),
            "direct": _log_audit(directory / "arfgen_direct.log", "arfgen"),
            "central": _log_audit(directory / "arfgen_central.log", "arfgen"),
        }
        products.extend(
            [
                _product(rmf_path, "rmf", instrument=instrument, output_annulus=output),
                _product(direct_path, "direct_arf", instrument=instrument, output_annulus=output),
                _product(central_path, "central_source_arf", instrument=instrument, output_annulus=output, source_id=50),
            ]
        )
        for input_annulus in ANNULI:
            path = directory / f"{instrument}_{output}_from_{input_annulus}_cross.arf"
            audit, lo, hi, response = _arf_audit(path, identity)
            grid = bool(np.array_equal(direct_lo, lo) and np.array_equal(direct_hi, hi))
            ratio = float(response[science].sum() / direct_response[science].sum())
            map_dimension = max(dimensions[output], dimensions[input_annulus])
            audit.update(
                {
                    "common_output_energy_grid": grid,
                    "fit_band_integrated_ratio_to_direct": ratio,
                    "requested_map_dimension": map_dimension,
                    "valid": bool(audit["valid"] and grid and ratio > 0),
                }
            )
            cross[input_annulus] = audit
            cross_row.append(ratio)
            logs[f"cross_from_{input_annulus}"] = _log_audit(
                directory / f"arfgen_cross_from_{input_annulus}.log", "arfgen"
            )
            products.append(
                _product(
                    path,
                    "cross_region_arf",
                    instrument=instrument,
                    output_annulus=output,
                    input_annulus=input_annulus,
                    requested_map_dimension=map_dimension,
                )
            )
        cross_matrix.append(cross_row)
        markers = {
            "rmf": (directory / ".rmf_complete").is_file(),
            "direct": (directory / ".direct_complete").is_file(),
            "central": (directory / ".central_complete").is_file(),
            "all_cross": all(
                (directory / f".cross_from_{annulus}_complete").is_file()
                for annulus in ANNULI
            ),
            "output": (directory / ".output_annulus_complete").is_file(),
        }
        central_ratio = float(
            central_response[science].sum() / direct_response[science].sum()
        )
        passed = bool(
            rmf["valid"]
            and direct["valid"]
            and central["valid"]
            and common_grid
            and central_ratio > 0
            and all(item["valid"] for item in cross.values())
            and all(item["valid"] for item in logs.values())
            and all(markers.values())
        )
        outputs[output] = {
            "status": "pass" if passed else "fail",
            "rmf": rmf,
            "direct_arf": direct,
            "central_source_arf": central,
            "central_to_direct_integrated_ratio": central_ratio,
            "cross_region_arfs": cross,
            "common_direct_central_energy_grid": common_grid,
            "logs": logs,
            "completion_markers": markers,
            "passed": passed,
        }

    file_counts = {
        "rmfs": len(list((EXTERNAL_ROOT / instrument).glob("*/*.rmf"))),
        "direct_arfs": len(list((EXTERNAL_ROOT / instrument).glob("*/*_direct.arf"))),
        "central_source_arfs": len(list((EXTERNAL_ROOT / instrument).glob("*/*_from_central_source50.arf"))),
        "cross_region_arfs": len(list((EXTERNAL_ROOT / instrument).glob("*/*_cross.arf"))),
    }
    expected_counts = {
        "rmfs": 6,
        "direct_arfs": 6,
        "central_source_arfs": 6,
        "cross_region_arfs": 36,
    }
    markers = {
        "instrument": (EXTERNAL_ROOT / instrument / ".instrument_complete").is_file(),
        "root": (EXTERNAL_ROOT / ".x4_response_products_complete").is_file(),
    }
    passed = bool(
        all(item["valid"] for item in maps.values())
        and all(item["passed"] for item in outputs.values())
        and file_counts == expected_counts
        and all(markers.values())
    )
    return {
        "status": "pass" if passed else "fail",
        "detector_maps": maps,
        "outputs": outputs,
        "cross_to_direct_integrated_ratio_matrix": {
            "row_order_output_annuli": list(ANNULI),
            "column_order_input_annuli": list(ANNULI),
            "values": cross_matrix,
        },
        "file_counts": file_counts,
        "expected_file_counts": expected_counts,
        "completion_markers": markers,
        "passed": passed,
    }, products


def audit() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    convergence = json.loads(CONVERGENCE.read_text(encoding="utf-8"))
    if convergence["gates"]["X4_map_resolution_convergence_passed"] is not True:
        raise ValueError("Full X4 cannot be audited before map-resolution convergence")
    dimensions = {
        key: int(value)
        for key, value in protocol["X4_response_calibration"]["detector_maps"][
            "production_requested_dimensions_by_annulus"
        ].items()
    }
    if tuple(dimensions) != ANNULI:
        raise ValueError("Protocol annulus order differs from the frozen X4 order")
    instruments: dict[str, Any] = {}
    products: list[dict[str, Any]] = []
    for instrument, specification in INSTRUMENTS.items():
        audit_record, instrument_products = _instrument(
            instrument, specification, dimensions
        )
        instruments[instrument] = audit_record
        products.extend(instrument_products)
    counts = {
        "rmfs": sum(item["file_counts"]["rmfs"] for item in instruments.values()),
        "direct_arfs": sum(item["file_counts"]["direct_arfs"] for item in instruments.values()),
        "central_source_arfs": sum(item["file_counts"]["central_source_arfs"] for item in instruments.values()),
        "cross_region_arfs": sum(item["file_counts"]["cross_region_arfs"] for item in instruments.values()),
    }
    expected = {
        "rmfs": 12,
        "direct_arfs": 12,
        "central_source_arfs": 12,
        "cross_region_arfs": 72,
    }
    passed = bool(
        all(item["passed"] for item in instruments.values()) and counts == expected
    )
    generated = datetime.now(timezone.utc).isoformat()
    inputs = {
        "protocol": _input_record(PROTOCOL),
        "map_convergence": _input_record(CONVERGENCE),
        "production_runner": _input_record(PRODUCTION_RUNNER),
        "audit_implementation": _input_record(Path(__file__).resolve()),
    }
    manifest = {
        "manifest_version": "R1B3-RXJ2129-XMM-X4-products-0.2-input-bound",
        "generated_utc": generated,
        "inputs": inputs,
        "external_root": str(EXTERNAL_ROOT),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "convergence_report": str(CONVERGENCE.relative_to(ROOT)),
        "annulus_order": list(ANNULI),
        "production_requested_dimensions_by_annulus": dimensions,
        "product_counts": counts,
        "products": products,
    }
    report = {
        "report_version": "R1B3-RXJ2129-XMM-X4-products-0.2-input-bound",
        "generated_utc": generated,
        "inputs": inputs,
        "status": "pass" if passed else "fail",
        "instruments": instruments,
        "product_counts": counts,
        "expected_product_counts": expected,
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "gates": {"X4_response_products_passed": passed},
        "authorization": {
            "construct_X5_joint_likelihood_scaffold": passed,
            "fit_temperature_or_density": False,
            "infer_gas_mass": False,
            "infer_dynamical_or_Weyl_response": False,
            "fit_new_force_or_action": False,
        },
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    report = audit()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
