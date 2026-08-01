from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18
from astropy.io import fits
import astropy.units as u

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.build_manga_bcg_table import parse_table
from voidscreen.unified import C_M_S, G_SI, M_SUN_KG


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strings(values) -> np.ndarray:
    return np.asarray(
        [value.decode().strip() if isinstance(value, bytes) else str(value).strip() for value in values]
    )


def _load_bcg_coordinates(bcg_tex: Path, drpall_path: Path) -> pd.DataFrame:
    bcg = parse_table(bcg_tex)
    with fits.open(drpall_path, memmap=True) as hdus:
        table = hdus["MANGA"].data
        plateifu = _strings(table["plateifu"])
        rows = pd.DataFrame(
            {
                "plateifu": plateifu,
                "ra_deg": np.asarray(table["objra"], dtype=float),
                "dec_deg": np.asarray(table["objdec"], dtype=float),
                "drpall_z": np.asarray(table["z"], dtype=float),
            }
        )
    selected = bcg[["plateifu", "redshift"]].merge(
        rows, on="plateifu", how="left", validate="one_to_one"
    )
    if selected[["ra_deg", "dec_deg", "drpall_z"]].isna().any().any():
        missing = selected.loc[selected["ra_deg"].isna(), "plateifu"].tolist()
        raise ValueError(f"MaNGA DRPall is missing BCGs: {missing}")
    coordinates = SkyCoord(
        ra=selected["ra_deg"].to_numpy() * u.deg,
        dec=selected["dec_deg"].to_numpy() * u.deg,
        frame="icrs",
    )
    selected["galactic_l_deg"] = coordinates.galactic.l.deg
    selected["in_erass1_public_western_hemisphere"] = selected["galactic_l_deg"].between(
        180.0, 360.0, inclusive="left"
    )
    selected["table_minus_drpall_z"] = selected["redshift"] - selected["drpall_z"]
    return selected


def _load_erass(path: Path) -> pd.DataFrame:
    with fits.open(path, memmap=True) as hdus:
        table = hdus["Joined"].data
        return pd.DataFrame(
            {
                "erass_name": _strings(table["NAME"]),
                "erass_ra_deg": np.asarray(table["RA_XFIT"], dtype=float),
                "erass_dec_deg": np.asarray(table["DEC_XFIT"], dtype=float),
                "erass_z": np.asarray(table["BEST_Z"], dtype=float),
                "erass_z_err": np.asarray(table["BEST_ZERR"], dtype=float),
                "erass_ext_like": np.asarray(table["EXT_LIKE"], dtype=float),
                "erass_pcont": np.asarray(table["PCONT"], dtype=float),
                "erass_mgas500_1e11_msun": np.asarray(table["MGAS500"], dtype=float),
                "erass_m500_1e13_msun": np.asarray(table["M500"], dtype=float),
                "erass_r500_kpc": np.asarray(table["R500"], dtype=float),
                "erass_l500_1e42_erg_s": np.asarray(table["L500"], dtype=float),
            }
        )


def _reverse_selection_feasibility(
    drpall_path: Path, erass: pd.DataFrame, *, maximum_delta_z_factor: float
) -> dict[str, dict[str, int]]:
    with fits.open(drpall_path, memmap=True) as hdus:
        table = hdus["MANGA"].data
        manga = pd.DataFrame(
            {
                "plateifu": _strings(table["plateifu"]),
                "ra_deg": np.asarray(table["objra"], dtype=float),
                "dec_deg": np.asarray(table["objdec"], dtype=float),
                "redshift": np.asarray(table["z"], dtype=float),
            }
        )
    manga_coordinates = SkyCoord(
        ra=manga["ra_deg"].to_numpy() * u.deg,
        dec=manga["dec_deg"].to_numpy() * u.deg,
        frame="icrs",
    )
    cluster_coordinates = SkyCoord(
        ra=erass["erass_ra_deg"].to_numpy() * u.deg,
        dec=erass["erass_dec_deg"].to_numpy() * u.deg,
        frame="icrs",
    )
    cluster_index, separation, _ = manga_coordinates.match_to_catalog_sky(
        cluster_coordinates
    )
    redshift = manga["redshift"].to_numpy(dtype=float)
    cluster_redshift = erass["erass_z"].to_numpy(dtype=float)[cluster_index]
    projected_kpc = separation.arcminute * Planck18.kpc_proper_per_arcmin(
        np.maximum(redshift, 0.001)
    ).value
    redshift_match = np.abs(redshift - cluster_redshift) <= maximum_delta_z_factor * (
        1.0 + redshift
    )
    positive_gas = (
        erass["erass_mgas500_1e11_msun"].to_numpy(dtype=float)[cluster_index] > 0.0
    )
    output = {}
    for radius_kpc in (50, 100, 200):
        selected = redshift_match & (projected_kpc <= radius_kpc)
        output[f"within_{radius_kpc}_kpc"] = {
            "manga_galaxies": int(selected.sum()),
            "unique_erass_clusters": int(np.unique(cluster_index[selected]).size),
            "manga_galaxies_with_positive_catalog_gas_scale": int(
                (selected & positive_gas).sum()
            ),
        }
    return output


def _crossmatch(
    bcg: pd.DataFrame,
    erass: pd.DataFrame,
    *,
    maximum_projected_kpc: float,
    maximum_delta_z_factor: float,
) -> pd.DataFrame:
    cluster_coordinates = SkyCoord(
        ra=erass["erass_ra_deg"].to_numpy() * u.deg,
        dec=erass["erass_dec_deg"].to_numpy() * u.deg,
        frame="icrs",
    )
    records = []
    for row in bcg.itertuples(index=False):
        coordinate = SkyCoord(ra=row.ra_deg * u.deg, dec=row.dec_deg * u.deg, frame="icrs")
        separation = coordinate.separation(cluster_coordinates)
        angular_scale = Planck18.kpc_proper_per_arcmin(row.redshift).value
        projected_kpc = separation.arcminute * angular_scale
        delta_z = np.abs(erass["erass_z"].to_numpy() - row.redshift)
        allowed_delta_z = maximum_delta_z_factor * (1.0 + row.redshift)
        eligible = (projected_kpc <= maximum_projected_kpc) & (delta_z <= allowed_delta_z)
        if np.any(eligible):
            eligible_indices = np.flatnonzero(eligible)
            selected_index = eligible_indices[np.argmin(projected_kpc[eligible])]
            match = erass.iloc[selected_index]
            record = match.to_dict()
            record.update(
                {
                    "erass_matched": True,
                    "erass_separation_arcmin": float(separation.arcminute[selected_index]),
                    "erass_projected_separation_kpc": float(projected_kpc[selected_index]),
                    "erass_delta_z": float(erass.iloc[selected_index]["erass_z"] - row.redshift),
                }
            )
        else:
            record = {column: np.nan for column in erass.columns}
            record.update(
                {
                    "erass_matched": False,
                    "erass_separation_arcmin": np.nan,
                    "erass_projected_separation_kpc": np.nan,
                    "erass_delta_z": np.nan,
                }
            )
        record.update(row._asdict())
        records.append(record)
    output = pd.DataFrame(records)
    valid_gas = (
        output["erass_matched"]
        & (output["erass_mgas500_1e11_msun"] > 0.0)
        & (output["erass_r500_kpc"] > 0.0)
    )
    output["erass_has_gas_scale"] = valid_gas
    gas_mass_kg = output["erass_mgas500_1e11_msun"] * 1e11 * M_SUN_KG
    radius_m = output["erass_r500_kpc"] * 3.085677581491367e19
    output["catalog_gas_chi_scale_g_m_over_r500_c2"] = np.where(
        valid_gas, G_SI * gas_mass_kg / radius_m / C_M_S**2, np.nan
    )
    output["catalog_gas_potential_speed_km_s"] = np.where(
        valid_gas, np.sqrt(G_SI * gas_mass_kg / radius_m) / 1000.0, np.nan
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit independent host-environment catalog coverage for the 50 MaNGA BCGs."
    )
    parser.add_argument(
        "--bcg-tex",
        type=Path,
        default=ROOT / "data" / "raw" / "manga_bcg_tian2024" / "RAR_BCG.tex",
    )
    parser.add_argument(
        "--drpall",
        type=Path,
        default=ROOT / "data" / "raw" / "manga_dr17" / "drpall-v3_1_1.fits",
    )
    parser.add_argument(
        "--erass",
        type=Path,
        default=ROOT
        / "data"
        / "raw"
        / "erass1_clusters"
        / "erass1cl_main_v3.2.fits",
    )
    parser.add_argument(
        "--gates", type=Path, default=ROOT / "configs" / "theory_stage_gates.json"
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "data" / "derived" / "bcg_environment_catalog_coverage.csv",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=ROOT / "data" / "derived" / "bcg_environment_catalog_coverage_report.json",
    )
    parser.add_argument("--maximum-projected-kpc", type=float, default=500.0)
    parser.add_argument("--maximum-delta-z-factor", type=float, default=0.01)
    args = parser.parse_args()

    gates = json.loads(args.gates.read_text(encoding="utf-8"))
    bcg = _load_bcg_coordinates(args.bcg_tex, args.drpall)
    erass = _load_erass(args.erass)
    matched = _crossmatch(
        bcg,
        erass,
        maximum_projected_kpc=args.maximum_projected_kpc,
        maximum_delta_z_factor=args.maximum_delta_z_factor,
    )
    footprint = matched["in_erass1_public_western_hemisphere"]
    matches = matched["erass_matched"]
    gas = matched["erass_has_gas_scale"]
    host_minimum = gates["stage_4_independent_environment"]["host_profile_systems_min"]
    coverage_minimum = gates["stage_4_independent_environment"][
        "host_profile_coverage_fraction_min"
    ]
    footprint_count = int(footprint.sum())
    footprint_gas_count = int((footprint & gas).sum())
    report = {
        "status": "completed public host-catalog coverage audit",
        "match_rule_frozen_for_this_audit": {
            "maximum_projected_separation_kpc": args.maximum_projected_kpc,
            "maximum_abs_delta_z": f"{args.maximum_delta_z_factor}*(1+z_bcg)",
            "tie_break": "smallest projected separation",
            "erass_public_footprint": "western Galactic hemisphere, 180 <= l < 360 degrees",
        },
        "inputs": {
            "bcg_tex_sha256": _sha256(args.bcg_tex),
            "drpall_sha256": _sha256(args.drpall),
            "erass_sha256": _sha256(args.erass),
        },
        "coverage": {
            "bcg_systems": len(matched),
            "drpall_coordinate_matches": int(matched["ra_deg"].notna().sum()),
            "erass_public_footprint_systems": footprint_count,
            "erass_matches_all_sky_sample": int(matches.sum()),
            "erass_matches_with_gas_scale": int(gas.sum()),
            "erass_footprint_matches_with_gas_scale": footprint_gas_count,
            "erass_footprint_gas_scale_fraction": (
                float(footprint_gas_count / footprint_count) if footprint_count else 0.0
            ),
        },
        "reverse_selection_feasibility": {
            "description": (
                "MaNGA galaxies near eRASS1 X-ray centers with the same redshift rule; "
                "these are candidates, not yet verified BCGs or central galaxies."
            ),
            "counts": _reverse_selection_feasibility(
                args.drpall,
                erass,
                maximum_delta_z_factor=args.maximum_delta_z_factor,
            ),
        },
        "catalog_gas_scale": {
            "definition": "G*Mgas500/(R500*c^2); scale diagnostic, not a central gas potential profile",
            "chi_quantiles": {
                key: float(value)
                for key, value in matched.loc[
                    gas, "catalog_gas_chi_scale_g_m_over_r500_c2"
                ]
                .quantile([0.1, 0.5, 0.9])
                .rename(index={0.1: "p10", 0.5: "median", 0.9: "p90"})
                .items()
            },
            "potential_speed_km_s_quantiles": {
                key: float(value)
                for key, value in matched.loc[gas, "catalog_gas_potential_speed_km_s"]
                .quantile([0.1, 0.5, 0.9])
                .rename(index={0.1: "p10", 0.5: "median", 0.9: "p90"})
                .items()
            },
        },
        "stage_4_coverage_gate": {
            "host_profile_system_count_passes": int(gas.sum()) >= host_minimum,
            "selected_public_footprint_coverage_passes": (
                footprint_gas_count / footprint_count >= coverage_minimum
                if footprint_count
                else False
            ),
            "passes_all": int(gas.sum()) >= host_minimum
            and (
                footprint_gas_count / footprint_count >= coverage_minimum
                if footprint_count
                else False
            ),
        },
        "guardrail": (
            "eRASS1 catalog MGAS500 and R500 are catalog-level scale estimates, not resolved "
            "central gas profiles. They can establish coverage and order of magnitude only."
        ),
    }
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    matched.sort_values("plateifu").to_csv(args.output_csv, index=False)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
