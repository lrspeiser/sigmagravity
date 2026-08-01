#!/usr/bin/env python3
"""Audit whether public Chandra events can support a frozen RX J2129 gas likelihood."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.io.fits import Header
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "r1_rxj2129_chandra_event_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def event_wcs(header) -> WCS:
    result = Header()
    result["NAXIS"] = 2
    mapping = {
        "CTYPE1": "TCTYP11",
        "CTYPE2": "TCTYP12",
        "CRVAL1": "TCRVL11",
        "CRVAL2": "TCRVL12",
        "CRPIX1": "TCRPX11",
        "CRPIX2": "TCRPX12",
        "CDELT1": "TCDLT11",
        "CDELT2": "TCDLT12",
    }
    for output, source in mapping.items():
        result[output] = header[source]
    result["RADESYS"] = header.get("RADESYS", "ICRS")
    return WCS(result)


def read_observation(path: Path, center_ra: float, center_dec: float) -> dict:
    with fits.open(path, memmap=False) as hdul:
        events = hdul["EVENTS"]
        required = {"x", "y", "energy"}
        columns = set(events.columns.names)
        valid = required.issubset(columns) and len(events.data) > 0
        wcs = event_wcs(events.header)
        center_x, center_y = wcs.all_world2pix([[center_ra, center_dec]], 1)[0]
        pixel_arcsec = abs(float(events.header["TCDLT11"])) * 3600
        dx = -(events.data["x"].astype(float) - center_x) * pixel_arcsec
        dy = (events.data["y"].astype(float) - center_y) * pixel_arcsec
        radius = np.hypot(dx, dy)
        energy_kev = events.data["energy"].astype(float) / 1000
        return {
            "obsid": int(events.header["OBS_ID"]),
            "exposure_ks": float(events.header["EXPOSURE"]) / 1000,
            "event_count": int(len(events.data)),
            "pixel_arcsec": pixel_arcsec,
            "pointing_ra_deg": float(events.header["RA_PNT"]),
            "pointing_dec_deg": float(events.header["DEC_PNT"]),
            "valid": bool(valid),
            "dx_arcsec": dx,
            "dy_arcsec": dy,
            "radius_arcsec": radius,
            "energy_kev": energy_kev,
        }


def band_mask(energy_kev: np.ndarray, limits: list[float]) -> np.ndarray:
    return (energy_kev >= limits[0]) & (energy_kev <= limits[1])


def main() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if cfg["status"] != "frozen_before_combined_event_adequacy_evaluation":
        raise RuntimeError("Refusing to execute an unfrozen event-adequacy protocol")
    provenance = json.loads((ROOT / cfg["inputs"]["archive_provenance"]).read_text(encoding="utf-8-sig"))
    provenance_by_path = {record["local_path"]: record for record in provenance["records"]}
    center = cfg["frozen_center"]
    input_keys = ["obsid_9370_evt2", "obsid_552_evt2"]
    observations = []
    hashes_verified = []
    for key in input_keys:
        relative = cfg["inputs"][key]
        path = ROOT / relative
        record = provenance_by_path[relative]
        hashes_verified.append(
            path.stat().st_size == int(record["size_bytes"])
            and sha256(path) == record["sha256"].upper()
        )
        observations.append(read_observation(path, center["ra_deg"], center["dec_deg"]))

    bands = cfg["bands_keV"]
    edges = np.asarray(cfg["radial_edges_arcsec"], dtype=float)
    ledger = []
    combined_counts = {name: np.zeros(len(edges) - 1, dtype=int) for name in bands}
    for observation in observations:
        for band_name, limits in bands.items():
            mask = band_mask(observation["energy_kev"], limits)
            counts, _ = np.histogram(observation["radius_arcsec"][mask], bins=edges)
            combined_counts[band_name] += counts
            for index, count in enumerate(counts):
                ledger.append(
                    {
                        "scope": f"obsid_{observation['obsid']}",
                        "obsid": observation["obsid"],
                        "exposure_ks": observation["exposure_ks"],
                        "band": band_name,
                        "inner_arcsec": edges[index],
                        "outer_arcsec": edges[index + 1],
                        "events": int(count),
                    }
                )

    soft = combined_counts["soft_imaging"]
    areas = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    bg_lo, bg_hi = cfg["background_screen"]["annulus_arcsec"]
    bg_indices = np.where((edges[:-1] >= bg_lo) & (edges[1:] <= bg_hi))[0]
    background_density = float(soft[bg_indices].sum() / areas[bg_indices].sum())
    expected_background = background_density * areas
    net_soft = soft - expected_background
    source_fraction = np.divide(
        net_soft, soft, out=np.full_like(net_soft, np.nan, dtype=float), where=soft > 0
    )
    for band_name, counts in combined_counts.items():
        for index, count in enumerate(counts):
            record = {
                "scope": "combined",
                "obsid": "9370+552",
                "exposure_ks": sum(item["exposure_ks"] for item in observations),
                "band": band_name,
                "inner_arcsec": edges[index],
                "outer_arcsec": edges[index + 1],
                "events": int(count),
            }
            if band_name == "soft_imaging":
                record.update(
                    {
                        "conservative_expected_background_events": expected_background[index],
                        "conservative_net_events": net_soft[index],
                        "conservative_source_fraction": source_fraction[index],
                    }
                )
            ledger.append(record)

    soft_dx = []
    soft_dy = []
    for observation in observations:
        use = band_mask(observation["energy_kev"], bands["soft_imaging"])
        aperture = use & (
            observation["radius_arcsec"] <= cfg["center_diagnostics"]["centroid_aperture_arcsec"]
        )
        soft_dx.append(observation["dx_arcsec"][aperture])
        soft_dy.append(observation["dy_arcsec"][aperture])
    soft_dx = np.concatenate(soft_dx)
    soft_dy = np.concatenate(soft_dy)
    centroid = np.array([np.mean(soft_dx), np.mean(soft_dy)])
    centroid_offset = float(np.linalg.norm(centroid))

    pixel = float(cfg["center_diagnostics"]["peak_histogram_pixel_arcsec"])
    aperture = float(cfg["center_diagnostics"]["centroid_aperture_arcsec"])
    map_edges = np.arange(-aperture - pixel / 2, aperture + pixel, pixel)
    image, y_edges, x_edges = np.histogram2d(soft_dy, soft_dx, bins=[map_edges, map_edges])
    smoothed = gaussian_filter(
        image, sigma=float(cfg["center_diagnostics"]["peak_gaussian_sigma_pixels"])
    )
    peak_row, peak_col = np.unravel_index(np.argmax(smoothed), smoothed.shape)
    peak = np.array(
        [
            (x_edges[peak_col] + x_edges[peak_col + 1]) / 2,
            (y_edges[peak_row] + y_edges[peak_row + 1]) / 2,
        ]
    )
    peak_offset = float(np.linalg.norm(peak))

    thresholds = cfg["advance_thresholds"]
    inner = np.where(edges[1:] <= 5.0)[0]
    checks = {
        "public_evt2_observations": len(observations) == thresholds["public_evt2_observations"],
        "combined_exposure": sum(item["exposure_ks"] for item in observations)
        >= thresholds["minimum_combined_exposure_ks"],
        "combined_soft_events_inside_5arcsec": int(soft[inner].sum())
        >= thresholds["minimum_combined_soft_events_inside_5arcsec"],
        "combined_soft_events_each_inner_annulus": bool(
            np.all(soft[inner] >= thresholds["minimum_combined_soft_events_each_of_four_inner_annuli"])
        ),
        "conservative_source_fraction_each_inner_annulus": bool(
            np.all(
                source_fraction[inner]
                >= thresholds["minimum_conservative_source_fraction_each_inner_annulus"]
            )
        ),
        "soft_centroid_offset": centroid_offset
        <= thresholds["maximum_soft_centroid_offset_arcsec"],
        "fits_headers_and_event_columns_valid": bool(all(item["valid"] for item in observations)),
        "evt2_hashes_verified": bool(all(hashes_verified)),
        "no_gas_density_or_mass_inferred": True,
        "gravity_or_independent_lens_residual_used": False,
    }
    positive_gate_keys = [
        key for key in checks if key != "gravity_or_independent_lens_residual_used"
    ]
    passed = bool(all(checks[key] for key in positive_gate_keys))

    ledger_path = ROOT / cfg["outputs"]["event_ledger"]
    report_path = ROOT / cfg["outputs"]["report"]
    diagnostic_path = ROOT / cfg["outputs"]["diagnostic"]
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ledger).to_csv(ledger_path, index=False)

    fig, (ax_map, ax_profile) = plt.subplots(1, 2, figsize=(11, 4.8))
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    ax_map.imshow(smoothed, origin="lower", extent=extent, cmap="magma")
    ax_map.scatter(0, 0, marker="+", s=100, color="cyan", label="frozen BCG center")
    ax_map.scatter(*centroid, marker="x", s=70, color="white", label="soft centroid")
    ax_map.scatter(*peak, marker="o", facecolors="none", edgecolors="lime", label="soft peak")
    ax_map.set_xlabel("east offset (arcsec)")
    ax_map.set_ylabel("north offset (arcsec)")
    ax_map.set_title("Combined 0.7-2.0 keV event map")
    ax_map.legend(fontsize=7)
    centers = (edges[:-1] + edges[1:]) / 2
    ax_profile.step(centers, soft, where="mid", label="observed soft events")
    ax_profile.step(centers, expected_background, where="mid", label="outer-annulus upper bound")
    ax_profile.axvline(5, color="0.4", linestyle="--", label="dynamics support")
    ax_profile.set_yscale("log")
    ax_profile.set_xlabel("radius (arcsec)")
    ax_profile.set_ylabel("events per frozen annulus")
    ax_profile.set_title("Raw count adequacy only")
    ax_profile.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(diagnostic_path, dpi=180)
    plt.close(fig)

    report = {
        "protocol_version": cfg["protocol_version"],
        "status": "raw_chandra_event_adequacy_passed" if passed else "raw_chandra_event_adequacy_failed",
        "archive_provenance_version": provenance["provenance_version"],
        "archive_file_records": len(provenance["records"]),
        "evt2_hashes_verified": bool(all(hashes_verified)),
        "observations": [
            {
                key: value
                for key, value in observation.items()
                if key
                not in {"dx_arcsec", "dy_arcsec", "radius_arcsec", "energy_kev"}
            }
            for observation in observations
        ],
        "combined_exposure_ks": float(sum(item["exposure_ks"] for item in observations)),
        "combined_soft_events_inside_5arcsec": int(soft[inner].sum()),
        "four_inner_soft_event_counts": soft[inner].astype(int).tolist(),
        "four_inner_conservative_source_fractions": source_fraction[inner].tolist(),
        "outer_annulus_soft_event_surface_density_per_arcsec2": background_density,
        "soft_center": {
            "centroid_east_arcsec": float(centroid[0]),
            "centroid_north_arcsec": float(centroid[1]),
            "centroid_offset_arcsec": centroid_offset,
            "smoothed_peak_east_arcsec": float(peak[0]),
            "smoothed_peak_north_arcsec": float(peak[1]),
            "smoothed_peak_offset_arcsec": peak_offset,
        },
        "checks": checks,
        "raw_event_adequacy_gate_pass": passed,
        "machine_readable_published_gas_profile_still_missing": True,
        "raw_public_data_shortfall": False,
        "gas_density_or_mass_inferred": False,
        "gravity_or_independent_lens_residual_used": False,
        "strict_r1_ready": False,
        "limitations": [
            "The area-scaled 40-60 arcsec annulus is only a conservative contamination screen; the final model requires reprojected blank-sky events and a soft-Galactic nuisance.",
            "Raw counts are not exposure corrected, PSF corrected, deprojected, or converted to electron density or gas mass.",
            "The sub-arcsecond innermost dynamics radius requires a forward PSF model and may remain non-identifiable even when the total count gate passes.",
            "A current CIAO/CALDB reduction and response audit must be frozen before any spectral or gas-profile fit.",
        ],
        "outputs": cfg["outputs"],
        "next_action": (
            "Freeze and execute the two-ObsID CIAO/CALDB response, blank-sky, PSF, spectral, and projected-density likelihood; keep gas mass and gravity fitting disabled until that protocol exists."
            if passed
            else "Do not attempt a four-bin gas reconstruction from these events; preserve the shortfall and continue the replacement-host inventory."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
