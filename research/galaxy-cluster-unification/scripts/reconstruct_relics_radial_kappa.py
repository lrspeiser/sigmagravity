from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS


ROOT = Path(__file__).resolve().parents[1]


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _radial_profile(
    path: Path,
    edges_arcsec: np.ndarray,
    *,
    center_ra_deg: float | None,
    center_dec_deg: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    header = fits.getheader(path)
    data = np.asarray(fits.getdata(path), dtype=float)
    y, x = np.indices(data.shape, dtype=float)
    if center_ra_deg is None or center_dec_deg is None:
        center_x = float(header["CRPIX1"]) - 1.0
        center_y = float(header["CRPIX2"]) - 1.0
    else:
        center_x, center_y = WCS(header).all_world2pix(center_ra_deg, center_dec_deg, 0)
    dx = (x - center_x) * abs(float(header["CDELT1"])) * 3600.0
    dy = (y - center_y) * abs(float(header["CDELT2"])) * 3600.0
    radius = np.hypot(dx, dy)
    values = []
    pixels = []
    for low, high in zip(edges_arcsec[:-1], edges_arcsec[1:], strict=True):
        mask = (radius >= low) & (radius < high) & np.isfinite(data)
        if not mask.any():
            raise ValueError(f"{path}: annulus {low}-{high} arcsec has no finite pixels")
        values.append(float(data[mask].mean()))
        pixels.append(int(mask.sum()))
    return np.asarray(values), np.asarray(pixels)


def reconstruct(
    *,
    input_dir: Path,
    dynamics_path: Path,
    center_config_path: Path,
    profile_output: Path,
    covariance_output: Path,
    center_audit_output: Path,
    report_output: Path,
) -> dict:
    systems = {
        "A2537": {"directory": "A2537", "dynamics_source": "Newman2013"},
        "MACS J0417": {"directory": "MACS_J0417", "dynamics_source": "Kaleidoscope2025"},
        "MACS J0949": {"directory": "MACS_J0949", "dynamics_source": "Kaleidoscope2025"},
    }
    center_config = json.loads(center_config_path.read_text(encoding="utf-8"))
    dynamics = pd.read_csv(dynamics_path)
    profile_rows: list[dict] = []
    covariance_rows: list[dict] = []
    center_rows: list[dict] = []
    summaries: dict[str, dict] = {}
    for system, metadata in systems.items():
        center = center_config["systems"][system]
        selected_dynamics = dynamics.loc[
            (dynamics["system"] == system)
            & (dynamics["source_sample"] == metadata["dynamics_source"])
        ]
        if selected_dynamics.empty:
            raise ValueError(f"no dynamics profile for {system}")
        dynamics_min = float(selected_dynamics["bin_min_arcsec"].min())
        dynamics_max = float(selected_dynamics["bin_max_arcsec"].max())

        range_dir = input_dir / metadata["directory"] / "range"
        paths = sorted(range_dir.glob("*_kappa.fits"))
        if len(paths) != 100:
            raise ValueError(f"{system}: expected 100 kappa range maps, found {len(paths)}")
        first_header = fits.getheader(paths[0])
        pixel_scale = abs(float(first_header["CDELT1"])) * 3600.0
        map_ra = float(first_header["CRVAL1"])
        map_dec = float(first_header["CRVAL2"])
        bcg_ra = center["bcg_ra_deg"]
        bcg_dec = center["bcg_dec_deg"]
        if bcg_ra is None or bcg_dec is None:
            center_offset = None
        else:
            dra = (float(bcg_ra) - map_ra) * np.cos(np.deg2rad(float(bcg_dec))) * 3600.0
            ddec = (float(bcg_dec) - map_dec) * 3600.0
            center_offset = float(np.hypot(dra, ddec))
        center_rows.append(
            {
                "system": system,
                "map_reference_ra_deg": map_ra,
                "map_reference_dec_deg": map_dec,
                "published_bcg_ra_deg": bcg_ra,
                "published_bcg_dec_deg": bcg_dec,
                "map_reference_to_bcg_arcsec": center_offset,
                "map_pixel_scale_arcsec": pixel_scale,
                "centering_verified": bool(center["centering_verified"]),
                "source": center["source"],
                "source_note": center["source_note"],
            }
        )
        max_edge = pixel_scale * 20
        edges = np.arange(0.0, max_edge + 0.5 * pixel_scale, pixel_scale)
        profiles = []
        pixel_counts = None
        for path in paths:
            values, counts = _radial_profile(
                path,
                edges,
                center_ra_deg=bcg_ra,
                center_dec_deg=bcg_dec,
            )
            profiles.append(values)
            if pixel_counts is None:
                pixel_counts = counts
            elif not np.array_equal(pixel_counts, counts):
                raise ValueError(f"{system}: inconsistent annulus pixel counts")
        ensemble = np.asarray(profiles)
        covariance = np.cov(ensemble, rowvar=False, ddof=1)
        means = ensemble.mean(axis=0)
        std = ensemble.std(axis=0, ddof=1)
        full_overlap = edges[1:] <= dynamics_max + 1e-12
        any_overlap = (edges[:-1] < dynamics_max) & (edges[1:] > dynamics_min)
        for index, (low, high) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            profile_rows.append(
                {
                    "system": system,
                    "annulus_index": index,
                    "r_min_arcsec": low,
                    "r_max_arcsec": high,
                    "r_mid_arcsec": 0.5 * (low + high),
                    "kappa_mean": means[index],
                    "kappa_std": std[index],
                    "annulus_pixels": int(pixel_counts[index]),
                    "dynamics_r_min_arcsec": dynamics_min,
                    "dynamics_r_max_arcsec": dynamics_max,
                    "annulus_any_dynamics_overlap": bool(any_overlap[index]),
                    "annulus_fully_within_dynamics_support": bool(full_overlap[index]),
                }
            )
        for i in range(len(means)):
            for j in range(len(means)):
                covariance_rows.append(
                    {
                        "system": system,
                        "annulus_i": i,
                        "annulus_j": j,
                        "covariance_kappa": covariance[i, j],
                    }
                )
        fully_overlapping = int(full_overlap.sum())
        summaries[system] = {
            "dynamics_source": metadata["dynamics_source"],
            "dynamics_bins": len(selected_dynamics),
            "dynamics_support_arcsec": [dynamics_min, dynamics_max],
            "lensing_mcmc_maps": len(paths),
            "lensing_annulus_width_arcsec": pixel_scale,
            "lensing_annuli_fully_within_dynamics_support": fully_overlapping,
            "passes_three_overlapping_lensing_annuli": fully_overlapping >= 3,
            "centering_verified": bool(center["centering_verified"]),
            "map_reference_to_bcg_arcsec": center_offset,
            "passes_verified_three_plus_three_overlap": bool(
                center["centering_verified"] and fully_overlapping >= 3
            ),
        }

    profile = pd.DataFrame(profile_rows)
    covariance_frame = pd.DataFrame(covariance_rows)
    center_frame = pd.DataFrame(center_rows)
    profile_output.parent.mkdir(parents=True, exist_ok=True)
    covariance_output.parent.mkdir(parents=True, exist_ok=True)
    center_audit_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(profile_output, index=False)
    covariance_frame.to_csv(covariance_output, index=False)
    center_frame.to_csv(center_audit_output, index=False)
    overlap_passes = sum(
        value["passes_three_overlapping_lensing_annuli"] for value in summaries.values()
    )
    verified_overlap_passes = sum(
        value["passes_verified_three_plus_three_overlap"] for value in summaries.values()
    )
    report = {
        "audit_version": "RELICS-radial-kappa-0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "systems": summaries,
        "summary": {
            "systems_reconstructed": len(summaries),
            "systems_with_three_full_lensing_annuli_inside_dynamics_support": overlap_passes,
            "systems_with_verified_bcg_centered_three_plus_three_overlap": verified_overlap_passes,
            "systems_passing_complete_R1_gate": 0,
        },
        "classification": {
            "covariance": "sample covariance across 100 RELICS Lenstool MCMC range maps",
            "lensing_quantity": "projected convergence kappa scaled to D_ls/D_s=1",
            "theory_dependence": "standard-lens-equation Lenstool reconstruction; not raw likelihood",
            "centering": "published BCG-centered where verified; FITS-reference-centered otherwise",
            "remaining_requirements": (
                "Complete baryonic profiles, joint dynamics covariance, and observable-level lensing "
                "forward models remain missing. A2537 now has verified BCG-centered 3+3 geometric "
                "overlap, but its lensing profiles remain standard-lens-equation Lenstool products and "
                "its later frozen raw-dynamics calibration gate failed; the radial count is also "
                "insufficient for two of three systems."
            ),
        },
        "outputs": {
            "radial_profiles": _display_path(profile_output),
            "radial_covariance": _display_path(covariance_output),
            "lens_center_audit": _display_path(center_audit_output),
        },
    }
    report_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=ROOT / "data" / "raw" / "relics_lens_models"
    )
    parser.add_argument(
        "--dynamics",
        type=Path,
        default=ROOT / "data" / "derived" / "r1_published_bcg_velocity_profiles.csv",
    )
    parser.add_argument(
        "--center-config",
        type=Path,
        default=ROOT / "configs" / "r1_lens_centers.json",
    )
    parser.add_argument(
        "--profile-output",
        type=Path,
        default=ROOT / "data" / "derived" / "relics_radial_kappa_profiles.csv",
    )
    parser.add_argument(
        "--covariance-output",
        type=Path,
        default=ROOT / "data" / "derived" / "relics_radial_kappa_covariance.csv",
    )
    parser.add_argument(
        "--center-audit-output",
        type=Path,
        default=ROOT / "data" / "derived" / "r1_lens_center_audit.csv",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=ROOT / "results" / "relics_radial_kappa" / "report.json",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            reconstruct(
                input_dir=args.input_dir,
                dynamics_path=args.dynamics,
                center_config_path=args.center_config,
                profile_output=args.profile_output,
                covariance_output=args.covariance_output,
                center_audit_output=args.center_audit_output,
                report_output=args.report_output,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
