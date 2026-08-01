from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM


ROOT = Path(__file__).resolve().parents[1]
COSMOLOGY = FlatLambdaCDM(H0=70.0, Om0=0.3)
MAGNITUDE_ERROR = 0.1


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def projected_enclosed_fraction(radius_kpc: np.ndarray, core_kpc: float, cut_kpc: float) -> np.ndarray:
    radius = np.asarray(radius_kpc, dtype=float)
    if not 0 <= core_kpc < cut_kpc:
        raise ValueError("dPIE radii must satisfy 0 <= core < cut")
    return (
        np.sqrt(core_kpc**2 + radius**2)
        - np.sqrt(cut_kpc**2 + radius**2)
        + cut_kpc
        - core_kpc
    ) / (cut_kpc - core_kpc)


def annular_surface_density(
    low_kpc: np.ndarray,
    high_kpc: np.ndarray,
    *,
    total_mass_msun: float,
    core_kpc: float,
    cut_kpc: float,
) -> np.ndarray:
    low = np.asarray(low_kpc, dtype=float)
    high = np.asarray(high_kpc, dtype=float)
    if np.any(high <= low):
        raise ValueError("annulus outer radii must exceed inner radii")
    enclosed_low = projected_enclosed_fraction(low, core_kpc, cut_kpc)
    enclosed_high = projected_enclosed_fraction(high, core_kpc, cut_kpc)
    return total_mass_msun * (enclosed_high - enclosed_low) / (np.pi * (high**2 - low**2))


def _conditional_covariance(
    low_kpc: np.ndarray,
    high_kpc: np.ndarray,
    *,
    total_mass_msun: float,
    core_kpc: float,
    cut_kpc: float,
    cut_error_kpc: float,
) -> np.ndarray:
    nominal = annular_surface_density(
        low_kpc,
        high_kpc,
        total_mass_msun=total_mass_msun,
        core_kpc=core_kpc,
        cut_kpc=cut_kpc,
    )
    sigma_ln_luminosity = np.log(10.0) * MAGNITUDE_ERROR / 2.5
    step = max(1e-4 * cut_kpc, 1e-5)
    low_cut = max(core_kpc + step, cut_kpc - step)
    high_cut = cut_kpc + step
    minus = annular_surface_density(
        low_kpc,
        high_kpc,
        total_mass_msun=total_mass_msun,
        core_kpc=core_kpc,
        cut_kpc=low_cut,
    )
    plus = annular_surface_density(
        low_kpc,
        high_kpc,
        total_mass_msun=total_mass_msun,
        core_kpc=core_kpc,
        cut_kpc=high_cut,
    )
    derivative_cut = (plus - minus) / (high_cut - low_cut)
    return np.outer(nominal, nominal) * sigma_ln_luminosity**2 + np.outer(
        derivative_cut, derivative_cut
    ) * cut_error_kpc**2


def reconstruct(
    *,
    photometry_path: Path,
    dynamics_path: Path,
    relics_profile_path: Path,
    profile_output: Path,
    covariance_output: Path,
    report_output: Path,
) -> dict:
    photometry = pd.read_csv(photometry_path)
    dynamics = pd.read_csv(dynamics_path)
    relics = pd.read_csv(relics_profile_path)
    newman = photometry.loc[photometry["source_sample"] == "Newman2013"].copy()
    if len(newman) != 7 or newman["stellar_m_to_l_v_sps"].isna().any():
        raise ValueError("expected seven normalized Newman BCG light models")

    profile_rows: list[dict] = []
    covariance_rows: list[dict] = []
    system_summaries: dict[str, dict] = {}
    for light in newman.itertuples(index=False):
        kpc_per_arcsec = float(
            COSMOLOGY.kpc_proper_per_arcmin(light.cluster_redshift).value / 60.0
        )
        selected_dynamics = dynamics.loc[
            (dynamics["source_sample"] == "Newman2013")
            & (dynamics["system"] == light.system)
        ].copy()
        grids = {
            "dynamics_bin": selected_dynamics[
                ["bin_min_arcsec", "bin_max_arcsec"]
            ].rename(columns={"bin_min_arcsec": "low", "bin_max_arcsec": "high"})
        }
        if light.system == "A2537":
            selected_relics = relics.loc[relics["system"] == "A2537"]
            grids["relics_reference_annulus"] = selected_relics[
                ["r_min_arcsec", "r_max_arcsec"]
            ].rename(columns={"r_min_arcsec": "low", "r_max_arcsec": "high"})

        total_mass_msun = float(light.stellar_mass_sps_1e11_msun * 1e11)
        grid_counts = {}
        for grid_kind, grid in grids.items():
            low_arcsec = grid["low"].to_numpy(dtype=float)
            high_arcsec = grid["high"].to_numpy(dtype=float)
            low_kpc = low_arcsec * kpc_per_arcsec
            high_kpc = high_arcsec * kpc_per_arcsec
            density = annular_surface_density(
                low_kpc,
                high_kpc,
                total_mass_msun=total_mass_msun,
                core_kpc=float(light.r_core_kpc),
                cut_kpc=float(light.r_cut_kpc),
            )
            covariance = _conditional_covariance(
                low_kpc,
                high_kpc,
                total_mass_msun=total_mass_msun,
                core_kpc=float(light.r_core_kpc),
                cut_kpc=float(light.r_cut_kpc),
                cut_error_kpc=float(light.r_cut_error_kpc),
            )
            enclosed = total_mass_msun * projected_enclosed_fraction(
                high_kpc, float(light.r_core_kpc), float(light.r_cut_kpc)
            )
            grid_counts[grid_kind] = len(grid)
            for index in range(len(grid)):
                profile_rows.append(
                    {
                        "source_sample": "Newman2013",
                        "system": light.system,
                        "grid_kind": grid_kind,
                        "annulus_index": index,
                        "r_min_arcsec": low_arcsec[index],
                        "r_max_arcsec": high_arcsec[index],
                        "r_min_kpc": low_kpc[index],
                        "r_max_kpc": high_kpc[index],
                        "stellar_surface_density_msun_kpc2": density[index],
                        "conditional_std_msun_kpc2": float(np.sqrt(covariance[index, index])),
                        "stellar_mass_enclosed_at_rmax_msun": enclosed[index],
                        "total_stellar_mass_sps_msun": total_mass_msun,
                        "stellar_m_to_l_v_sps": light.stellar_m_to_l_v_sps,
                        "sps_imf": light.sps_imf,
                        "profile_model": "spherical_circularized_dPIE",
                    }
                )
            for i in range(len(grid)):
                for j in range(len(grid)):
                    covariance_rows.append(
                        {
                            "system": light.system,
                            "grid_kind": grid_kind,
                            "annulus_i": i,
                            "annulus_j": j,
                            "conditional_covariance_msun2_kpc4": covariance[i, j],
                        }
                    )
        system_summaries[light.system] = {
            "cluster_redshift": float(light.cluster_redshift),
            "kpc_per_arcsec": kpc_per_arcsec,
            "total_stellar_mass_sps_msun": total_mass_msun,
            "stellar_m_to_l_v_sps": float(light.stellar_m_to_l_v_sps),
            "grids": grid_counts,
        }

    profiles = pd.DataFrame(profile_rows)
    covariance_frame = pd.DataFrame(covariance_rows)
    for output in (profile_output, covariance_output, report_output):
        output.parent.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(profile_output, index=False)
    covariance_frame.to_csv(covariance_output, index=False)
    a2537_overlap = profiles.loc[
        (profiles["system"] == "A2537")
        & (profiles["grid_kind"] == "relics_reference_annulus")
        & (profiles["r_max_arcsec"] <= 3.65 + 1e-12)
    ]
    report = {
        "audit_version": "Newman-BCG-stellar-profile-0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cosmology": {"H0_km_s_Mpc": 70.0, "Omega_m": 0.3, "Omega_Lambda": 0.7},
        "systems": system_summaries,
        "summary": {
            "newman_systems_reconstructed": len(system_summaries),
            "profile_rows": len(profiles),
            "a2537_reference_annuli_inside_dynamics_support": len(a2537_overlap),
            "systems_with_complete_baryonic_profile": 0,
            "systems_passing_complete_R1_gate": 0,
        },
        "classification": {
            "component": "BCG stellar mass only",
            "profile": "analytic dPIE normalized by published rest-frame V luminosity and Chabrier SPS M/L",
            "covariance": (
                "conditional covariance from the published 0.1 mag luminosity uncertainty and r_cut "
                "uncertainty; excludes SPS M/L uncertainty and cross-probe covariance"
            ),
            "not_included": "intracluster gas, intracluster light beyond the one-component BCG fit, and satellite galaxies",
            "r1_status": "partial baryonic input; not a complete forward-model baryonic profile",
            "centering_caveat": (
                "RELICS reference annuli are FITS-reference-centered; exact BCG-centered overlap remains unverified"
            ),
        },
        "outputs": {
            "stellar_profiles": _display_path(profile_output),
            "conditional_covariance": _display_path(covariance_output),
        },
    }
    report_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--photometry",
        type=Path,
        default=ROOT / "data" / "derived" / "r1_published_bcg_photometric_fits.csv",
    )
    parser.add_argument(
        "--dynamics",
        type=Path,
        default=ROOT / "data" / "derived" / "r1_published_bcg_velocity_profiles.csv",
    )
    parser.add_argument(
        "--relics-profiles",
        type=Path,
        default=ROOT / "data" / "derived" / "relics_radial_kappa_profiles.csv",
    )
    parser.add_argument(
        "--profile-output",
        type=Path,
        default=ROOT / "data" / "derived" / "newman_bcg_stellar_profiles.csv",
    )
    parser.add_argument(
        "--covariance-output",
        type=Path,
        default=ROOT / "data" / "derived" / "newman_bcg_stellar_profile_covariance.csv",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=ROOT / "results" / "newman_bcg_stellar_profiles" / "report.json",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            reconstruct(
                photometry_path=args.photometry,
                dynamics_path=args.dynamics,
                relics_profile_path=args.relics_profiles,
                profile_output=args.profile_output,
                covariance_output=args.covariance_output,
                report_output=args.report_output,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
