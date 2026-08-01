#!/usr/bin/env python3
"""Audit A1689 individual calibrated 2-D GMOS frames before sky modeling."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import astrodata
import gemini_instruments  # noqa: F401 - registers Gemini AstroData classes
import numpy as np
from astropy.io import fits


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/r1_a1689_gmos_reduction_covariance_protocol.json"
CAL_REPORT = ROOT / "results/r1_a1689_gmos_calibrations/report.json"
PRODUCTS = ROOT / "data/derived/r1_a1689_gmos_reconstruction/science_cal2d"
REPORT = ROOT / "results/r1_a1689_gmos_science_cal2d/report.json"
COSMIC_RAY_BIT = np.uint16(8)

EXPECTED_HISTORY = [
    "prepare", "addDQ", "addVAR", "overscanCorrect", "biasCorrect",
    "ADUToElectrons", "addVAR", "attachWavelengthSolution", "flatCorrect",
    "QECorrect", "flagCosmicRays", "distortionCorrect", "writeOutputs",
]
FORBIDDEN_HISTORY = {
    "findApertures", "skyCorrectFromSlit", "adjustWCSToReference",
    "resampleToCommonFrame", "scaleCountsToReference", "stackFrames",
    "traceApertures", "extractSpectra", "fluxCalibrate",
}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest().upper()


def text_column(values: np.ndarray) -> list[str]:
    return [
        value.decode(errors="replace").strip() if isinstance(value, bytes) else str(value).strip()
        for value in values
    ]


def inspect_product(science: str, bias: str, flat: str, arc: str, bpm: str) -> dict:
    stem = Path(science).stem
    matches = sorted(PRODUCTS.glob(f"{stem}*_cal2d.fits"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one calibrated product for {science}; found {len(matches)}")
    product = matches[0]
    with fits.open(product, memmap=False) as hdul:
        history = text_column(hdul["HISTORY"].data["primitive"])
        provenance = text_column(hdul["PROVENANCE"].data["filename"])
        sci_hdus = [hdu for hdu in hdul if hdu.name == "SCI"]
        var_hdus = [hdu for hdu in hdul if hdu.name == "VAR"]
        dq_hdus = [hdu for hdu in hdul if hdu.name == "DQ"]
        if not (len(sci_hdus) == len(var_hdus) == len(dq_hdus) == 1):
            raise RuntimeError(f"{product.name}: expected one mosaicked SCI/VAR/DQ triplet")
        sci = np.asarray(sci_hdus[0].data)
        var = np.asarray(var_hdus[0].data)
        dq = np.asarray(dq_hdus[0].data)
        history_gate = history == EXPECTED_HISTORY
        provenance_expected = {science, bpm, bias, flat, arc}
        provenance_gate = set(provenance) == provenance_expected
        structure_gate = bool(
            sci.shape == var.shape == dq.shape
            and np.isfinite(sci).all()
            and np.isfinite(var).all()
            and np.all(var >= 0)
            and np.issubdtype(dq.dtype, np.integer)
        )
        cosmic_pixels = int(np.count_nonzero(dq & COSMIC_RAY_BIT))
        no_forbidden = not FORBIDDEN_HISTORY.intersection(history)

    ad = astrodata.open(product)
    wcs_frames = list(ad[0].wcs.available_frames)
    # distortionCorrect collapses the intermediate frame into the output grid;
    # its execution is retained in HISTORY while the serialized final WCS has
    # the expected pixels -> world frames.
    wcs_gate = wcs_frames == ["pixels", "world"] and len(ad) == 1
    return {
        "science": science,
        "product": str(product.relative_to(ROOT)).replace("\\", "/"),
        "product_sha256": sha256(product),
        "shape": list(sci.shape),
        "history": history,
        "history_matches_frozen_recipe": history_gate,
        "forbidden_sky_stack_or_extraction_primitives_absent": no_forbidden,
        "provenance_filenames": provenance,
        "exact_frozen_calibration_provenance": provenance_gate,
        "finite_sci_nonnegative_variance_and_integer_dq": structure_gate,
        "positive_variance_fraction": float(np.mean(var > 0)),
        "unmasked_fraction": float(np.mean(dq == 0)),
        "cosmic_ray_flagged_pixels": cosmic_pixels,
        "wcs_frames": wcs_frames,
        "single_interpolation_distortion_mosaic_present": wcs_gate,
        "passed": bool(
            history_gate and no_forbidden and provenance_gate and structure_gate
            and cosmic_pixels > 0 and wcs_gate
        ),
    }


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    calibration_report = json.loads(CAL_REPORT.read_text(encoding="utf-8"))
    mapping = config["raw_inputs"]["science_to_flat_arc_mapping"]
    bias_by_science = {
        science: (
            "N20090615S0531_bias.fits"
            if science.startswith("N20090615") else "N20090621S0193_bias.fits"
        )
        for science in mapping
    }
    bpm = config["raw_inputs"]["bad_pixel_mask"]["selected_filename"]
    rows = [
        inspect_product(
            science,
            bias_by_science[science],
            f"{Path(flat_arc[0]).stem}_flat.fits",
            f"{Path(flat_arc[1]).stem}_arc.fits",
            bpm,
        )
        for science, flat_arc in mapping.items()
    ]
    gate = bool(
        calibration_report["gates"]["P2a_calibration_products_gate_passed"]
        and len(rows) == config["calibration_acceptance"]["science_2d_products_required"]
        and all(row["passed"] for row in rows)
    )
    report = {
        "report_version": "R1B1-A1689-GMOS-science-cal2d-0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_blind": True,
        "scope": "individual_calibrated_2d_before_frozen_sky_and_centroid_audit",
        "products": rows,
        "gates": {
            "P2a_calibration_products_gate_passed": calibration_report["gates"]["P2a_calibration_products_gate_passed"],
            "P2b_individual_calibrated_2d_gate_passed": gate,
            "P2_calibrated_2d_sky_centroid_coverage_gate_passed": False,
            "P3_profile_covariance_gate_passed": False,
            "gravity_response_fit_authorized": False,
        },
        "authorization": {
            "fit_frozen_continuum_centroid_and_sky_models": gate,
            "fit_stellar_kinematics": False,
            "infer_gravity_response": False,
            "fit_new_force_or_action": False,
        },
        "next_action": "If P2b passes, fit the shared continuum-only center, apply the three preregistered sky-window variants per exposure, and audit centroid range, coverage, and sky residuals before any pPXF call.",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
