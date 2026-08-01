#!/usr/bin/env python3
"""Extract the nine frozen A1689 signed spectra and apply only the S/N gate."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import medfilt


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/r1_a1689_gmos_reduction_covariance_protocol.json"
COMBINATION_REPORT = ROOT / "results/r1_a1689_gmos_combination/report.json"
OUTPUT = ROOT / "data/derived/r1_a1689_gmos_reconstruction/a1689_signed_spectra.npz"
LEDGER = ROOT / "data/derived/r1_a1689_gmos_signed_spectra.csv"
REPORT = ROOT / "results/r1_a1689_gmos_signed_spectra/report.json"


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    combination = json.loads(COMBINATION_REPORT.read_text(encoding="utf-8"))
    if not combination["authorization"]["fit_frozen_nine_signed_stellar_kinematic_bins"]:
        raise RuntimeError("P2 did not authorize signed-spectrum extraction")
    product = np.load(ROOT / combination["output"])
    wavelength = np.asarray(product["wavelength_angstrom"], dtype=float)
    offset = np.asarray(product["signed_offset_arcsec"], dtype=float)
    data = np.asarray(product["combined_science_electron"], dtype=float)
    variance = np.asarray(product["combined_variance_electron2"], dtype=float)
    dq = np.asarray(product["combined_dq"], dtype=np.uint16)
    edges = config["spatial_extraction"]["signed_bin_edges_arcsec"]
    min_fraction = config["spatial_extraction"]["minimum_valid_pixel_fraction_per_bin"]
    min_snr = config["spatial_extraction"]["minimum_median_signal_to_noise_per_angstrom_per_signed_bin"]
    fit_range = config["stellar_kinematic_fit"]["observed_fit_range_angstrom"]
    wave_step = float(np.median(np.diff(wavelength)))

    spectra = []
    variances = []
    masks = []
    rows = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        spatial = (offset >= lower) & ((offset < upper) if index < len(edges) - 1 else (offset <= upper))
        values = data[spatial]
        errors = variance[spatial]
        flags = dq[spatial]
        valid = np.isfinite(values) & np.isfinite(errors) & (errors > 0) & (flags == 0)
        count = valid.sum(axis=0)
        enough = count >= min_fraction * np.count_nonzero(spatial)
        spectrum = np.where(valid, values, 0.0).sum(axis=0)
        summed_variance = np.where(valid, errors, 0.0).sum(axis=0)
        noise = np.sqrt(np.maximum(summed_variance, 0.0))
        local = medfilt(noise, kernel_size=21)
        ratio = np.divide(noise, local, out=np.full_like(noise, np.inf), where=local > 0)
        finite_ratio = np.isfinite(ratio) & enough
        ratio_center = float(np.median(ratio[finite_ratio]))
        ratio_sigma = float(1.4826 * np.median(np.abs(ratio[finite_ratio] - ratio_center)))
        high_variance = ratio > ratio_center + 6.0 * max(ratio_sigma, 0.02)
        sky_line = (wavelength >= 5569.0) & (wavelength <= 5585.0)
        fit_window = (wavelength >= fit_range[0]) & (wavelength <= fit_range[1])
        retained = enough & ~high_variance & ~sky_line & (summed_variance > 0)
        snr_pixels = retained & fit_window
        snr = float(np.median(spectrum[snr_pixels] / np.sqrt(summed_variance[snr_pixels]) / np.sqrt(wave_step)))
        passed = bool(snr >= min_snr)
        spectra.append(spectrum)
        variances.append(summed_variance)
        masks.append(retained)
        rows.append({
            "signed_bin": index,
            "lower_arcsec": lower,
            "upper_arcsec": upper,
            "spatial_rows": int(np.count_nonzero(spatial)),
            "minimum_valid_spatial_rows_per_retained_wavelength": int(np.min(count[retained])),
            "retained_wavelength_pixels": int(np.count_nonzero(retained)),
            "baseline_fit_wavelength_pixels": int(np.count_nonzero(snr_pixels)),
            "median_signal_to_noise_per_angstrom": snr,
            "snr_gate_passed": passed,
        })

    spectra_array = np.asarray(spectra)
    variance_array = np.asarray(variances)
    mask_array = np.asarray(masks)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT,
        wavelength_angstrom=wavelength,
        signed_bin_edges_arcsec=np.asarray(edges),
        spectrum_electron=spectra_array.astype(np.float32),
        variance_electron2=variance_array.astype(np.float32),
        retained_mask=mask_array,
    )
    ledger = pd.DataFrame(rows)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(LEDGER, index=False)
    passing = int(ledger["snr_gate_passed"].sum())
    gate = passing >= config["profile_acceptance"]["minimum_finite_signed_bins"]
    report = {
        "report_version": "R1B1-A1689-GMOS-signed-spectra-0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_blind": True,
        "signed_bins": rows,
        "passing_signed_bins": passing,
        "required_passing_signed_bins": config["profile_acceptance"]["minimum_finite_signed_bins"],
        "outputs": {
            "spectra": str(OUTPUT.relative_to(ROOT)).replace("\\", "/"),
            "ledger": str(LEDGER.relative_to(ROOT)).replace("\\", "/"),
        },
        "gates": {
            "P2_calibrated_2d_sky_centroid_coverage_gate_passed": True,
            "P3a_signed_spectrum_snr_gate_passed": bool(gate),
            "P3_profile_covariance_gate_passed": False,
            "gravity_response_fit_authorized": False,
        },
        "authorization": {
            "run_frozen_baseline_ppxf_on_retained_signed_bins": bool(gate),
            "infer_gravity_response": False,
            "fit_lens_mass_model": False,
            "fit_new_force_or_action": False,
        },
        "next_action": (
            "Run the frozen baseline XSL pPXF fit on retained bins, retaining every failed fit and without changing masks or edges."
            if gate else
            "Keep A1689 geometry-only because fewer than seven signed spectra pass S/N; do not merge bins after seeing the result."
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
