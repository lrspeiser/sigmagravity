#!/usr/bin/env python3
"""Audit the frozen CIAO/CALDB RX J2129 reduction without fitting gas physics."""

from __future__ import annotations

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


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "r1_rxj2129_chandra_reduction_protocol.json"


def find_one(folder: Path, pattern: str) -> Path:
    matches = sorted(folder.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern!r} in {folder}; found {len(matches)}")
    return matches[0]


def event_wcs(header: Header) -> WCS:
    column_by_name = {
        str(header[key]).strip().lower(): int(key.removeprefix("TTYPE"))
        for key in header
        if key.startswith("TTYPE")
    }
    try:
        x_column = column_by_name["x"]
        y_column = column_by_name["y"]
    except KeyError as error:
        raise RuntimeError("EVENTS table has no x/y coordinate columns") from error
    result = Header()
    result["NAXIS"] = 2
    mapping = {
        "CTYPE1": f"TCTYP{x_column}",
        "CTYPE2": f"TCTYP{y_column}",
        "CRVAL1": f"TCRVL{x_column}",
        "CRVAL2": f"TCRVL{y_column}",
        "CRPIX1": f"TCRPX{x_column}",
        "CRPIX2": f"TCRPX{y_column}",
        "CDELT1": f"TCDLT{x_column}",
        "CDELT2": f"TCDLT{y_column}",
    }
    for output, source in mapping.items():
        result[output] = header[source]
    result["RADESYS"] = header.get("RADESYS", "ICRS")
    return WCS(result)


def read_events(path: Path, center_ra: float, center_dec: float) -> dict:
    with fits.open(path, memmap=False) as hdul:
        events = hdul["EVENTS"]
        header = events.header
        primary = hdul[0].header
        celestial_wcs = event_wcs(header)
        center_x, center_y = celestial_wcs.all_world2pix(
            [[center_ra, center_dec]], 1
        )[0]
        pixel_arcsec = abs(float(celestial_wcs.wcs.cdelt[0])) * 3600
        dx = -(np.asarray(events.data["x"], dtype=float) - center_x) * pixel_arcsec
        dy = (np.asarray(events.data["y"], dtype=float) - center_y) * pixel_arcsec
        data = {
            "radius_arcsec": np.hypot(dx, dy),
            "energy_kev": np.asarray(events.data["energy"], dtype=float) / 1000,
            "ccd_id": np.asarray(events.data["ccd_id"], dtype=int),
            "exposure_ks": float(header.get("EXPOSURE", primary.get("EXPOSURE"))) / 1000,
            "ascds_version": str(header.get("ASCDSVER", primary.get("ASCDSVER", ""))),
            "caldb_version": str(header.get("CALDBVER", primary.get("CALDBVER", ""))),
            "header": header.copy(),
            "primary_header": primary.copy(),
        }
    return data


def bkg_scale_by_ccd(background: dict) -> dict[int, float]:
    scales = {}
    for ccd in np.unique(background["ccd_id"]):
        key = f"BKGSCAL{ccd}"
        value = background["header"].get(
            key, background["primary_header"].get(key, np.nan)
        )
        scales[int(ccd)] = float(value)
    if not scales or not np.all(np.isfinite(list(scales.values()))):
        raise RuntimeError("Missing finite BKGSCALn keywords in blank-sky event file")
    return scales


def center_image_value(path: Path, ra_deg: float, dec_deg: float) -> float:
    with fits.open(path, memmap=False) as hdul:
        image = np.asarray(hdul[0].data, dtype=float)
        wcs = WCS(hdul[0].header).celestial
        x, y = wcs.world_to_pixel_values(ra_deg, dec_deg)
        column = int(np.rint(x))
        row = int(np.rint(y))
        if row < 0 or column < 0 or row >= image.shape[0] or column >= image.shape[1]:
            return float("nan")
        return float(image[row, column])


def validate_spectrum(root: Path) -> dict:
    pha = root.with_suffix(".pi")
    arf = root.with_suffix(".arf")
    rmf = root.with_suffix(".rmf")
    bkg = root.parent / f"{root.name}_bkg.pi"
    required = [pha, arf, rmf, bkg]
    files_exist = all(path.exists() and path.stat().st_size > 0 for path in required)
    if not files_exist:
        return {"valid": False, "counts": 0, "files": [str(p.relative_to(ROOT)) for p in required]}
    with fits.open(pha, memmap=False) as hdul:
        counts = int(np.asarray(hdul["SPECTRUM"].data["COUNTS"], dtype=np.int64).sum())
    with fits.open(arf, memmap=False) as hdul:
        response = np.asarray(hdul["SPECRESP"].data["SPECRESP"], dtype=float)
        arf_valid = bool(np.isfinite(response).all() and np.any(response > 0))
    with fits.open(rmf, memmap=False) as hdul:
        matrix = np.asarray(hdul["MATRIX"].data["MATRIX"])
        rmf_valid = bool(len(matrix) > 0)
    return {
        "valid": bool(files_exist and counts > 0 and arf_valid and rmf_valid),
        "counts": counts,
        "arf_valid": arf_valid,
        "rmf_valid": rmf_valid,
        "files": [str(p.relative_to(ROOT)) for p in required],
    }


def main() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if cfg["status"] != "frozen_before_ciao_reprocessing_or_calibrated_product_inspection":
        raise RuntimeError("Refusing to audit against an unfrozen reduction protocol")
    adequacy = json.loads(
        (ROOT / cfg["inputs"]["event_adequacy_report"]).read_text(encoding="utf-8")
    )
    if adequacy["raw_event_adequacy_gate_pass"] is not True:
        raise RuntimeError("Raw event adequacy did not authorize this reduction")

    center = cfg["frozen_center"]
    out_root = ROOT / cfg["outputs"]["reduction_root"]
    thresholds = cfg["reduction_advance_thresholds"]
    original_exposure = {
        int(item["obsid"]): float(item["exposure_ks"])
        for item in adequacy["observations"]
    }
    edges = np.asarray(
        [item[0] for item in thresholds["comparison_annuli_arcsec"]]
        + [thresholds["comparison_annuli_arcsec"][-1][1]],
        dtype=float,
    )

    observations = []
    ledger_rows = []
    for obsid in cfg["inputs"]["obsids"]:
        products = out_root / str(obsid) / "products"
        spectra = out_root / str(obsid) / "spectra"
        clean_path = products / f"{obsid}_clean_evt2.fits"
        blank_path = products / f"{obsid}_blanksky_evt.fits"
        clean = read_events(clean_path, center["ra_deg"], center["dec_deg"])
        blank = read_events(blank_path, center["ra_deg"], center["dec_deg"])
        scales = bkg_scale_by_ccd(blank)
        source_soft = (clean["energy_kev"] >= 0.7) & (clean["energy_kev"] <= 2.0)
        blank_soft = (blank["energy_kev"] >= 0.7) & (blank["energy_kev"] <= 2.0)
        source_counts, _ = np.histogram(clean["radius_arcsec"][source_soft], bins=edges)
        scaled_background = np.zeros(len(edges) - 1, dtype=float)
        for ccd, scale in scales.items():
            use = blank_soft & (blank["ccd_id"] == ccd)
            counts, _ = np.histogram(blank["radius_arcsec"][use], bins=edges)
            scaled_background += counts * scale
        net_counts = source_counts - scaled_background
        net_rate = net_counts / clean["exposure_ks"]
        for index in range(len(edges) - 1):
            ledger_rows.append(
                {
                    "obsid": obsid,
                    "inner_arcsec": edges[index],
                    "outer_arcsec": edges[index + 1],
                    "source_soft_counts": int(source_counts[index]),
                    "scaled_blank_sky_soft_counts": scaled_background[index],
                    "net_soft_counts": net_counts[index],
                    "net_soft_rate_per_ks": net_rate[index],
                }
            )

        soft_expmap = find_one(products, f"{obsid}_soft*thresh.expmap")
        psfmap = products / f"{obsid}_soft_r90_psfmap.fits"
        global_spectrum = validate_spectrum(spectra / f"{obsid}_global_60arcsec")
        diagnostic_spectra = {
            name: validate_spectrum(spectra / f"{obsid}_{name}")
            for name in (
                "annulus_0_5arcsec",
                "annulus_5_15arcsec",
                "annulus_15_30arcsec",
                "annulus_30_60arcsec",
            )
        }
        observations.append(
            {
                "obsid": obsid,
                "original_exposure_ks": original_exposure[obsid],
                "retained_exposure_ks": clean["exposure_ks"],
                "retained_exposure_fraction": clean["exposure_ks"] / original_exposure[obsid],
                "ciao_version_header": clean["ascds_version"],
                "caldb_version_header": clean["caldb_version"],
                "blank_sky_scales_by_ccd": {str(k): v for k, v in scales.items()},
                "center_exposure_map_value": center_image_value(
                    soft_expmap, center["ra_deg"], center["dec_deg"]
                ),
                "center_r90_psf_arcsec": center_image_value(
                    psfmap, center["ra_deg"], center["dec_deg"]
                ),
                "global_spectrum": global_spectrum,
                "diagnostic_spectra": diagnostic_spectra,
                "source_soft_counts_inner": source_counts.astype(int).tolist(),
                "scaled_blank_sky_soft_counts_inner": scaled_background.tolist(),
                "net_soft_rates_per_ks_inner": net_rate.tolist(),
            }
        )

    ledger = pd.DataFrame(ledger_rows)
    rates = {
        obsid: ledger.loc[ledger["obsid"] == obsid, "net_soft_rate_per_ks"].to_numpy()
        for obsid in cfg["inputs"]["obsids"]
    }
    exp = {item["obsid"]: item["retained_exposure_ks"] for item in observations}
    pooled = (rates[552] * exp[552] + rates[9370] * exp[9370]) / (exp[552] + exp[9370])
    fractional_difference = np.divide(
        np.abs(rates[552] - rates[9370]),
        pooled,
        out=np.full_like(pooled, np.inf),
        where=pooled > 0,
    )
    for index, value in enumerate(fractional_difference):
        use = (ledger["inner_arcsec"] == edges[index]) & (
            ledger["outer_arcsec"] == edges[index + 1]
        )
        ledger.loc[use, "leave_one_observation_fractional_rate_difference"] = value

    scale_limits = thresholds["allowed_blank_sky_particle_scale_interval"]
    checks = {
        "software_versions": all(
            item["ciao_version_header"].startswith("4.18")
            and item["caldb_version_header"].startswith("4.12.4")
            for item in observations
        ),
        "both_observations_reprocessed": len(observations) == 2,
        "retained_exposure_fraction_each_observation": all(
            item["retained_exposure_fraction"]
            >= thresholds["minimum_retained_exposure_fraction_each_observation"]
            for item in observations
        ),
        "combined_retained_exposure": sum(
            item["retained_exposure_ks"] for item in observations
        )
        >= thresholds["minimum_combined_retained_exposure_ks"],
        "blank_sky_particle_scales": all(
            scale_limits[0] <= value <= scale_limits[1]
            for item in observations
            for value in item["blank_sky_scales_by_ccd"].values()
        ),
        "finite_positive_center_exposure": all(
            np.isfinite(item["center_exposure_map_value"])
            and item["center_exposure_map_value"] > 0
            for item in observations
        ),
        "center_r90_psf": all(
            np.isfinite(item["center_r90_psf_arcsec"])
            and item["center_r90_psf_arcsec"]
            <= thresholds["maximum_center_r90_psf_arcsec_each_observation"]
            for item in observations
        ),
        "valid_global_responses": all(
            item["global_spectrum"]["valid"] for item in observations
        ),
        "combined_global_counts": sum(
            item["global_spectrum"]["counts"] for item in observations
        )
        >= thresholds["minimum_combined_global_0p7_7keV_events"],
        "inner_soft_observation_compatibility": bool(
            np.all(
                fractional_difference
                <= thresholds["maximum_leave_one_observation_inner_soft_fractional_difference"]
            )
        ),
        "no_gas_density_mass_or_gravity_fit": True,
    }
    passed = bool(all(checks.values()))

    ledger_path = ROOT / cfg["outputs"]["audit_ledger"]
    report_path = ROOT / cfg["outputs"]["audit_report"]
    diagnostic_path = ROOT / cfg["outputs"]["diagnostic"]
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(ledger_path, index=False)

    centers = (edges[:-1] + edges[1:]) / 2
    fig, (ax_rates, ax_psf) = plt.subplots(1, 2, figsize=(10.5, 4.5))
    for item in observations:
        ax_rates.plot(
            centers,
            item["net_soft_rates_per_ks_inner"],
            marker="o",
            label=f"ObsID {item['obsid']}",
        )
    ax_rates.set_xlabel("radius (arcsec)")
    ax_rates.set_ylabel("blank-sky-scaled net soft events ks$^{-1}$")
    ax_rates.set_title("Leave-one-observation reduction screen")
    ax_rates.legend()
    ax_psf.bar(
        [str(item["obsid"]) for item in observations],
        [item["center_r90_psf_arcsec"] for item in observations],
    )
    ax_psf.axhline(
        thresholds["maximum_center_r90_psf_arcsec_each_observation"],
        color="tab:red",
        linestyle="--",
        label="frozen maximum",
    )
    ax_psf.set_xlabel("ObsID")
    ax_psf.set_ylabel("center 90% PSF radius (arcsec)")
    ax_psf.set_title("PSF resolvability screen")
    ax_psf.legend()
    fig.tight_layout()
    fig.savefig(diagnostic_path, dpi=180)
    plt.close(fig)

    report = {
        "protocol_version": cfg["protocol_version"],
        "status": "calibrated_reduction_gate_passed" if passed else "calibrated_reduction_gate_failed",
        "observations": observations,
        "combined_retained_exposure_ks": float(
            sum(item["retained_exposure_ks"] for item in observations)
        ),
        "combined_global_0p7_7keV_counts": int(
            sum(item["global_spectrum"]["counts"] for item in observations)
        ),
        "inner_soft_leave_one_observation_fractional_differences": fractional_difference.tolist(),
        "maximum_inner_soft_leave_one_observation_fractional_difference": float(
            np.max(fractional_difference)
        ),
        "checks": checks,
        "calibrated_reduction_gate_pass": passed,
        "gas_density_or_mass_inferred": False,
        "gravity_or_independent_lens_residual_used": False,
        "gas_profile_fit_authorized": passed,
        "weyl_or_dynamical_response_authorized": False,
        "strict_r1_ready": False,
        "limitations": [
            "The diagnostic annular spectra are reduction products, not four independently fitted temperature measurements.",
            "The blank-sky products model particle and average sky background; the later likelihood still requires a soft-Galactic nuisance and must preserve source/background Poisson counts.",
            "A circular CALDB PSF-radius map is only a resolvability screen; the later spatial likelihood must forward-model each observation rather than coadd the event images.",
            "Passing this gate authorizes a separately frozen gas likelihood only. It does not establish a gas density, gas mass, Weyl response, dynamical response, or modified-gravity result.",
        ],
        "outputs": cfg["outputs"],
        "next_action": (
            "Freeze a one-component cusped-beta projected spatial-plus-global-spectral Poisson model, with per-observation exposure, PSF, particle background, and soft-Galactic nuisance, before fitting any gas density or mass."
            if passed
            else "Keep the gas profile blocked and diagnose only the failed frozen reduction checks; do not tune the gate on these outputs."
        ),
    }
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
