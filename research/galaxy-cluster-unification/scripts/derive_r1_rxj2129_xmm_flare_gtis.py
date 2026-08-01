#!/usr/bin/env python3
"""Apply the frozen RX J2129 EPIC flare rule and write deterministic GTIs."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits


PROJECT = Path(__file__).resolve().parents[1]
ANALYSIS = Path(
    "/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/analysis"
)
DERIVED = PROJECT / "data/derived/r1_rxj2129_xmm_x2"
DIAGNOSTICS_PATH = DERIVED / "flare_diagnostics.json"
BIN_SECONDS = 100.0
MIN_LIVE_FRACTION = 0.5
MAX_ITERATIONS = 10
MAD_SCALE = 1.4826
SIGMA_CLIP = 2.5

INSTRUMENTS = {
    "MOS1": {"rate": "MOS1_high_energy_rate.ds", "ceiling": 0.35},
    "MOS2": {"rate": "MOS2_high_energy_rate.ds", "ceiling": 0.35},
    "pn": {"rate": "pn_high_energy_rate.ds", "ceiling": 0.40},
}


def minimum_detector_live_fraction(
    times: np.ndarray, hdus: fits.HDUList
) -> np.ndarray:
    """Return the conservative minimum STDGTI overlap fraction per time bin."""
    bin_starts = times - BIN_SECONDS / 2
    bin_stops = times + BIN_SECONDS / 2
    fractions: list[np.ndarray] = []
    for hdu in hdus:
        if not hdu.name.startswith("STDGTI"):
            continue
        detector_fraction = np.zeros(times.size, dtype=float)
        starts = np.asarray(hdu.data["START"], dtype=float)
        stops = np.asarray(hdu.data["STOP"], dtype=float)
        for start, stop in zip(starts, stops, strict=True):
            detector_fraction += np.maximum(
                0.0, np.minimum(bin_stops, stop) - np.maximum(bin_starts, start)
            ) / BIN_SECONDS
        fractions.append(np.clip(detector_fraction, 0.0, 1.0))
    if not fractions:
        raise RuntimeError("rate product has no detector STDGTI extensions")
    return np.min(np.vstack(fractions), axis=0)


def merged_intervals(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = times - BIN_SECONDS / 2
    stops = times + BIN_SECONDS / 2
    merged_starts: list[float] = []
    merged_stops: list[float] = []
    for start, stop in zip(starts, stops, strict=True):
        if merged_stops and math.isclose(start, merged_stops[-1], abs_tol=1e-6):
            merged_stops[-1] = float(stop)
        else:
            merged_starts.append(float(start))
            merged_stops.append(float(stop))
    return np.asarray(merged_starts), np.asarray(merged_stops)


def write_gti(path: Path, starts: np.ndarray, stops: np.ndarray) -> None:
    primary = fits.PrimaryHDU()
    created = datetime.now(timezone.utc).isoformat()
    primary.header["CREATOR"] = "derive_r1_rxj2129_xmm_flare_gtis.py"
    primary.header["DATE"] = created
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="START", format="D", unit="s", array=starts),
            fits.Column(name="STOP", format="D", unit="s", array=stops),
        ],
        name="STDGTI",
    )
    table.header["HDUCLASS"] = "OGIP"
    table.header["HDUCLAS1"] = "GTI"
    table.header["TIMEUNIT"] = "s"
    table.header["TIMESYS"] = "TT"
    table.header["MJDREF"] = 50814.0
    table.header["CREATOR"] = "derive_r1_rxj2129_xmm_flare_gtis.py"
    table.header["DATE"] = created
    fits.HDUList([primary, table]).writeto(path, overwrite=True, checksum=True)


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    output: dict[str, object] = {
        "version": "R1B3-RXJ2129-XMM-X2a-flare-0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "configs/r1_rxj2129_xmm_event_processing_protocol.json",
        "rule": {
            "time_bin_seconds": BIN_SECONDS,
            "minimum_bin_live_fraction": MIN_LIVE_FRACTION,
            "sigma_clip": SIGMA_CLIP,
            "MAD_scale": MAD_SCALE,
            "maximum_iterations": MAX_ITERATIONS,
            "manual_edits": False,
        },
        "instruments": {},
    }

    for label, spec in INSTRUMENTS.items():
        rate_path = ANALYSIS / str(spec["rate"])
        with fits.open(rate_path, memmap=True) as hdus:
            rate_hdu = hdus["RATE"]
            names = set(rate_hdu.columns.names)
            required = {"TIME", "RATE"}
            if not required.issubset(names):
                raise RuntimeError(f"{rate_path}: missing {sorted(required - names)}")
            times = np.asarray(rate_hdu.data["TIME"], dtype=float)
            rates = np.asarray(rate_hdu.data["RATE"], dtype=float)
            fracexp = minimum_detector_live_fraction(times, hdus)

        finite_live = (
            np.isfinite(times)
            & np.isfinite(rates)
            & np.isfinite(fracexp)
            & (fracexp >= MIN_LIVE_FRACTION)
        )
        keep = finite_live.copy()
        history: list[dict[str, object]] = []
        for iteration in range(1, MAX_ITERATIONS + 1):
            selected_rates = rates[keep]
            if selected_rates.size == 0:
                raise RuntimeError(f"{label}: no live rate bins remain")
            median = float(np.median(selected_rates))
            mad = float(np.median(np.abs(selected_rates - median)))
            robust_limit = (
                float("inf")
                if mad == 0
                else median + SIGMA_CLIP * MAD_SCALE * mad
            )
            threshold = min(float(spec["ceiling"]), robust_limit)
            new_keep = finite_live & (rates <= threshold)
            history.append(
                {
                    "iteration": iteration,
                    "input_bins": int(keep.sum()),
                    "median_counts_per_second": median,
                    "MAD_counts_per_second": mad,
                    "robust_upper_limit_counts_per_second": (
                        None if not math.isfinite(robust_limit) else robust_limit
                    ),
                    "applied_upper_limit_counts_per_second": threshold,
                    "retained_bins": int(new_keep.sum()),
                }
            )
            if np.array_equal(new_keep, keep):
                keep = new_keep
                break
            keep = new_keep
        else:
            raise RuntimeError(f"{label}: flare membership did not converge")

        gti_starts, gti_stops = merged_intervals(times[keep])
        gti_path = DERIVED / f"{label}_flare_gti.fits"
        write_gti(gti_path, gti_starts, gti_stops)
        cleaned_live_seconds = float(np.sum(fracexp[keep] * BIN_SECONDS))
        all_live_seconds = float(np.sum(fracexp[finite_live] * BIN_SECONDS))

        csv_path = DERIVED / f"{label}_high_energy_rate.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["time_s", "rate_counts_per_s", "fractional_exposure", "live_bin", "retained"]
            )
            for time, rate, fraction, live, retained in zip(
                times, rates, fracexp, finite_live, keep, strict=True
            ):
                writer.writerow(
                    [
                        f"{time:.6f}",
                        f"{rate:.12g}" if math.isfinite(rate) else "nan",
                        f"{fraction:.12g}" if math.isfinite(fraction) else "nan",
                        int(live),
                        int(retained),
                    ]
                )

        output["instruments"][label] = {
            "rate_product": str(rate_path),
            "rate_bins": int(times.size),
            "finite_bins_with_minimum_live_fraction": int(finite_live.sum()),
            "retained_bins": int(keep.sum()),
            "rejected_bins": int(finite_live.sum() - keep.sum()),
            "final_rate_limit_counts_per_second": history[-1][
                "applied_upper_limit_counts_per_second"
            ],
            "fixed_ceiling_counts_per_second": spec["ceiling"],
            "all_eligible_bin_live_seconds": all_live_seconds,
            "retained_bin_live_seconds": cleaned_live_seconds,
            "retained_fraction_of_eligible_bin_live_seconds": (
                cleaned_live_seconds / all_live_seconds if all_live_seconds else 0.0
            ),
            "gti_intervals": int(gti_starts.size),
            "gti_path": str(gti_path.relative_to(PROJECT)),
            "rate_csv": str(csv_path.relative_to(PROJECT)),
            "iterations": history,
        }

    DIAGNOSTICS_PATH.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
