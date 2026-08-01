from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.build_manga_bcg_table import parse_table
from voidscreen.bcg_bridge import (
    acceleration_from_log_mass,
    calibrate_log_offset,
    mfl_log_acceleration_at_radius,
    nsa_sersic_log_mass_within_radius,
    physical_radius_kpc,
)
from voidscreen.data import KPC_M
from voidscreen.theory import H7S_MODEL_NAME, fit_h7s, h7s_acceleration
from voidscreen.unified import (
    A0_M_S2,
    C_M_S,
    G_SI,
    M_SUN_KG,
    load_clash_acceleration_frame,
    load_sparc_acceleration_frame,
    rar_acceleration,
)

FIXED_RAR = "fixed_galaxy_rar"
CLUSTER_RAR = "cluster_scale_rar"
H7S_HOST = "H7s_standard_mu_cosmic_baryon_host_r200"
H7S_ERASS_MEDIAN_GAS = "H7s_standard_mu_erass_median_gas_host_r200"
H7S_ERASS_P90_GAS = "H7s_standard_mu_erass_p90_gas_host_r200"


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


def _load_jam(path: Path) -> pd.DataFrame:
    with fits.open(path, memmap=True) as hdus:
        base = hdus[1].data
        mfl = hdus[2].data
        return pd.DataFrame(
            {
                "plateifu": _strings(base["plateifu"]),
                "mangaid": _strings(base["mangaid"]),
                "ra_deg": np.asarray(base["obj_ra"], dtype=float),
                "dec_deg": np.asarray(base["obj_dec"], dtype=float),
                "redshift": np.asarray(base["z"], dtype=float),
                "distance_mpc": np.asarray(base["DA"], dtype=float),
                "mge_re_arcsec": np.asarray(base["Re_arcsec_MGE"], dtype=float),
                "nsa_sersic_re_arcsec": np.asarray(
                    base["nsa_sersic_th50"], dtype=float
                ),
                "nsa_sersic_n": np.asarray(base["nsa_sersic_n"], dtype=float),
                "nsa_log_mass_h_minus_2_msun": np.asarray(
                    base["nsa_sersic_mass"], dtype=float
                ),
                "jam_quality": np.asarray(base["Qual"], dtype=int),
                "jam_log_mtotal_re_msun": np.asarray(mfl["log_Mt_Re"], dtype=float),
                "jam_total_density_slope_re": np.asarray(mfl["Gt_Re"], dtype=float),
            }
        )


def _load_gema(path: Path) -> pd.DataFrame:
    with fits.open(path, memmap=True) as hdus:
        group = hdus["DR17_param_groups"].data
        lss = hdus["DR17_param_LSS"].data
        group_frame = pd.DataFrame(
            {
                "mangaid": _strings(group["mangaid"]),
                "gema_brightest_group_flag": np.asarray(group["BG"], dtype=int),
                "gema_most_massive_group_flag": np.asarray(group["MG"], dtype=int),
                "gema_group_size": np.asarray(group["GroupSize"], dtype=int),
                "gema_group_tidal_strength": np.asarray(group["Q_group"], dtype=float),
            }
        )
        lss_frame = pd.DataFrame(
            {
                "mangaid": _strings(lss["mangaid"]),
                "gema_log_halo_mass_msun": np.asarray(lss["mh"], dtype=float),
                "gema_tidal_eigenvalue_1": np.asarray(lss["t1"], dtype=float),
                "gema_tidal_eigenvalue_2": np.asarray(lss["t2"], dtype=float),
                "gema_tidal_eigenvalue_3": np.asarray(lss["t3"], dtype=float),
            }
        )
    return group_frame.merge(lss_frame, on="mangaid", how="left", validate="one_to_one")


def _load_spiders(path: Path) -> pd.DataFrame:
    with fits.open(path, memmap=True) as hdus:
        table = hdus[1].data
        return pd.DataFrame(
            {
                "spiders_id": _strings(table["CLUS_ID"]),
                "spiders_ra_opt_deg": np.asarray(table["RA_OPT"], dtype=float),
                "spiders_dec_opt_deg": np.asarray(table["DEC_OPT"], dtype=float),
                "spiders_redshift": np.asarray(table["SCREEN_CLUZSPEC"], dtype=float),
                "spiders_xray_luminosity_erg_s": np.asarray(
                    table["LX0124"], dtype=float
                ),
                "spiders_xray_luminosity_error_erg_s": np.asarray(
                    table["ELX"], dtype=float
                ),
                "spiders_r200_deg": np.asarray(table["R200C_DEG"], dtype=float),
                "spiders_richness": np.asarray(table["LAMBDA_CHISQ_OPT"], dtype=float),
                "spiders_red_sequence_members": np.asarray(table["NMEM"], dtype=int),
                "spiders_velocity_dispersion_km_s": np.asarray(
                    table["SCREEN_CLUVDISP_BEST"], dtype=float
                ),
                "spiders_status": _strings(table["STATUS"]),
            }
        )


def _erass_gas_fraction_summary(path: Path) -> dict[str, float | int]:
    with fits.open(path, memmap=True) as hdus:
        table = hdus["Joined"].data
        fraction = np.asarray(table["FGAS500"], dtype=float)
        mass = np.asarray(table["M500"], dtype=float)
        gas_mass = np.asarray(table["MGAS500"], dtype=float)
        radius = np.asarray(table["R500"], dtype=float)
    valid = (
        np.isfinite(fraction)
        & (fraction > 0.0)
        & (fraction < 1.0)
        & (mass > 0.0)
        & (gas_mass > 0.0)
        & (radius > 0.0)
    )
    values = fraction[valid]
    return {
        "systems": int(valid.sum()),
        "p10": float(np.quantile(values, 0.1)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
    }


def _valid_jam(frame: pd.DataFrame, minimum_quality: int) -> np.ndarray:
    finite_columns = [
        "redshift",
        "distance_mpc",
        "mge_re_arcsec",
        "nsa_sersic_re_arcsec",
        "nsa_sersic_n",
        "nsa_log_mass_h_minus_2_msun",
        "jam_log_mtotal_re_msun",
        "jam_total_density_slope_re",
    ]
    finite = np.isfinite(frame[finite_columns]).all(axis=1)
    positive = (
        (frame["redshift"] > 0.0)
        & (frame["distance_mpc"] > 0.0)
        & (frame["mge_re_arcsec"] > 0.0)
        & (frame["nsa_sersic_re_arcsec"] > 0.0)
        & (frame["nsa_sersic_n"] > 0.0)
        & frame["nsa_log_mass_h_minus_2_msun"].between(5.0, 14.0)
        & frame["jam_log_mtotal_re_msun"].between(5.0, 14.0)
    )
    return finite & positive & (frame["jam_quality"] >= minimum_quality)


def _build_matches(
    jam: pd.DataFrame,
    gema: pd.DataFrame,
    spiders: pd.DataFrame,
    selection: dict,
) -> pd.DataFrame:
    unique_jam = jam.sort_values(
        ["mangaid", "jam_quality", "mge_re_arcsec"],
        ascending=[True, False, False],
    ).drop_duplicates("mangaid")
    candidates = gema.merge(unique_jam, on="mangaid", how="inner", validate="one_to_one")
    candidates = candidates[
        (candidates["gema_brightest_group_flag"] == selection["gema_brightest_group_flag"])
        & _valid_jam(candidates, selection["jam_minimum_visual_quality"])
    ].copy()
    candidates.reset_index(drop=True, inplace=True)

    candidate_coordinates = SkyCoord(
        ra=candidates["ra_deg"].to_numpy() * u.deg,
        dec=candidates["dec_deg"].to_numpy() * u.deg,
        frame="icrs",
    )
    cluster_coordinates = SkyCoord(
        ra=spiders["spiders_ra_opt_deg"].to_numpy() * u.deg,
        dec=spiders["spiders_dec_opt_deg"].to_numpy() * u.deg,
        frame="icrs",
    )
    cluster_index, separation, _ = candidate_coordinates.match_to_catalog_sky(
        cluster_coordinates
    )
    cluster_rows = spiders.iloc[cluster_index].reset_index(drop=True)
    projected_kpc = separation.arcminute * Planck18.kpc_proper_per_arcmin(
        candidates["redshift"].to_numpy()
    ).value
    delta_z = (
        cluster_rows["spiders_redshift"].to_numpy()
        - candidates["redshift"].to_numpy()
    )
    maximum_delta_z = selection["spiders_maximum_abs_delta_z_factor"] * (
        1.0 + candidates["redshift"].to_numpy()
    )
    eligible = (
        (projected_kpc <= selection["spiders_maximum_projected_separation_kpc"])
        & (np.abs(delta_z) <= maximum_delta_z)
        & (cluster_rows["spiders_xray_luminosity_erg_s"].to_numpy() > 0.0)
        & (cluster_rows["spiders_r200_deg"].to_numpy() > 0.0)
    )
    matched = pd.concat(
        [
            candidates.loc[eligible].reset_index(drop=True),
            cluster_rows.loc[eligible].reset_index(drop=True),
        ],
        axis=1,
    )
    matched["spiders_projected_separation_kpc"] = projected_kpc[eligible]
    matched["spiders_delta_z"] = delta_z[eligible]
    matched = matched.sort_values(
        [
            "spiders_id",
            "jam_quality",
            "spiders_projected_separation_kpc",
            "mge_re_arcsec",
        ],
        ascending=[True, False, True, False],
    ).drop_duplicates("spiders_id")
    matched.reset_index(drop=True, inplace=True)
    scale = Planck18.kpc_proper_per_arcmin(
        matched["spiders_redshift"].to_numpy()
    ).value
    matched["spiders_r200_kpc"] = matched["spiders_r200_deg"] * 60.0 * scale
    return matched


def _raw_proxy_at_tian_radius(frame: pd.DataFrame, hubble_h: float) -> pd.DataFrame:
    output = frame.copy()
    mge_re_kpc = physical_radius_kpc(output["distance_mpc"], output["mge_re_arcsec"])
    nsa_re_kpc = physical_radius_kpc(
        output["distance_mpc"], output["nsa_sersic_re_arcsec"]
    )
    output["raw_proxy_log_gobs"] = mfl_log_acceleration_at_radius(
        output["jam_log_mtotal_re_msun"],
        mge_re_kpc,
        output["radius_kpc"],
        output["jam_total_density_slope_re"],
    )
    log_mbar = nsa_sersic_log_mass_within_radius(
        output["nsa_log_mass_h_minus_2_msun"],
        output["radius_kpc"],
        nsa_re_kpc,
        output["nsa_sersic_n"],
        hubble_h=hubble_h,
    )
    output["raw_proxy_log_gbar"] = np.log10(
        acceleration_from_log_mass(log_mbar, output["radius_kpc"])
    )
    return output


def _calibrate_proxy(
    tian: pd.DataFrame,
    jam: pd.DataFrame,
    test_plateifus: set[str],
    *,
    minimum_quality: int,
    hubble_h: float,
) -> tuple[dict, pd.DataFrame]:
    calibration = tian[~tian["plateifu"].isin(test_plateifus)].merge(
        jam.drop(columns="redshift"),
        on="plateifu",
        how="inner",
        validate="one_to_one",
    )
    calibration = calibration[_valid_jam(calibration, minimum_quality)].copy()
    calibration = _raw_proxy_at_tian_radius(calibration, hubble_h)
    gobs = calibrate_log_offset(
        calibration["raw_proxy_log_gobs"], calibration["log_gobs"]
    )
    gbar = calibrate_log_offset(
        calibration["raw_proxy_log_gbar"], calibration["log_gbar"]
    )
    calibration["calibrated_proxy_log_gobs"] = (
        calibration["raw_proxy_log_gobs"] + gobs["offset_dex"]
    )
    calibration["calibrated_proxy_log_gbar"] = (
        calibration["raw_proxy_log_gbar"] + gbar["offset_dex"]
    )
    return {"gobs": gobs, "gbar": gbar}, calibration


def _measurement_sample(
    matched: pd.DataFrame,
    tian: pd.DataFrame,
    calibration: dict,
    *,
    hubble_h: float,
    erass_gas_fraction: dict,
) -> pd.DataFrame:
    direct_columns = [
        "plateifu",
        "radius_kpc",
        "log_gbar",
        "err_log_gbar",
        "log_gobs",
        "err_log_gobs",
    ]
    sample = matched.merge(
        tian[direct_columns], on="plateifu", how="left", validate="one_to_one"
    )
    direct = sample["log_gobs"].notna()
    proxy = ~direct
    sample["measurement_source"] = np.where(
        direct, "Tian2024_direct", "DynPop_NSA_calibrated_proxy"
    )
    mge_re_kpc = physical_radius_kpc(sample["distance_mpc"], sample["mge_re_arcsec"])
    nsa_re_kpc = physical_radius_kpc(
        sample["distance_mpc"], sample["nsa_sersic_re_arcsec"]
    )
    sample.loc[proxy, "radius_kpc"] = mge_re_kpc[proxy]
    raw_log_gobs = np.log10(
        acceleration_from_log_mass(sample["jam_log_mtotal_re_msun"], mge_re_kpc)
    )
    proxy_log_mass = nsa_sersic_log_mass_within_radius(
        sample["nsa_log_mass_h_minus_2_msun"],
        mge_re_kpc,
        nsa_re_kpc,
        sample["nsa_sersic_n"],
        hubble_h=hubble_h,
    )
    raw_log_gbar = np.log10(acceleration_from_log_mass(proxy_log_mass, mge_re_kpc))
    sample.loc[proxy, "log_gobs"] = (
        raw_log_gobs[proxy] + calibration["gobs"]["offset_dex"]
    )
    sample.loc[proxy, "log_gbar"] = (
        raw_log_gbar[proxy] + calibration["gbar"]["offset_dex"]
    )
    sample.loc[proxy, "err_log_gobs"] = max(
        calibration["gobs"]["rms_residual_dex"], 0.05
    )
    sample.loc[proxy, "err_log_gbar"] = max(
        calibration["gbar"]["rms_residual_dex"], 0.05
    )
    sample["gbar_m_s2"] = np.power(10.0, sample["log_gbar"])
    sample["gobs_m_s2"] = np.power(10.0, sample["log_gobs"])
    sample["bcg_baryonic_chi"] = (
        sample["gbar_m_s2"] * sample["radius_kpc"] * KPC_M / C_M_S**2
    )
    r200_m = sample["spiders_r200_kpc"].to_numpy(dtype=float) * KPC_M
    critical_density = Planck18.critical_density(
        sample["spiders_redshift"].to_numpy(dtype=float)
    ).to_value(u.kg / u.m**3)
    m200_kg = (4.0 * np.pi / 3.0) * 200.0 * critical_density * np.power(r200_m, 3)
    cosmic_baryon_fraction = float(Planck18.Ob0 / Planck18.Om0)
    host_baryon_mass_kg = cosmic_baryon_fraction * m200_kg
    sample["spiders_m200_from_r200_msun"] = m200_kg / M_SUN_KG
    sample["host_cosmic_baryon_mass_msun"] = host_baryon_mass_kg / M_SUN_KG
    total_mass_chi = G_SI * m200_kg / r200_m / C_M_S**2
    sample["host_cosmic_baryon_chi_r200"] = cosmic_baryon_fraction * total_mass_chi
    sample["host_erass_median_gas_chi_r200"] = (
        erass_gas_fraction["median"] * total_mass_chi
    )
    sample["host_erass_p90_gas_chi_r200"] = (
        erass_gas_fraction["p90"] * total_mass_chi
    )
    sample["combined_bcg_host_chi"] = (
        sample["bcg_baryonic_chi"] + sample["host_cosmic_baryon_chi_r200"]
    )
    return sample


def _predict(
    name: str,
    gbar: np.ndarray,
    radius_kpc: np.ndarray,
    vector,
    host_chi: np.ndarray,
) -> np.ndarray:
    if name == FIXED_RAR:
        return rar_acceleration(gbar, A0_M_S2)
    if name == CLUSTER_RAR:
        return rar_acceleration(gbar, 2.0e-9)
    if name == H7S_MODEL_NAME:
        chi = gbar * radius_kpc * KPC_M / C_M_S**2
        return h7s_acceleration(gbar, chi, vector)
    if name in (H7S_HOST, H7S_ERASS_MEDIAN_GAS, H7S_ERASS_P90_GAS):
        chi = gbar * radius_kpc * KPC_M / C_M_S**2 + host_chi
        return h7s_acceleration(gbar, chi, vector)
    raise ValueError(f"unknown model: {name}")


def _score(sample: pd.DataFrame, vector) -> tuple[pd.DataFrame, dict]:
    pieces = []
    epsilon = 1e-5
    gbar = sample["gbar_m_s2"].to_numpy(dtype=float)
    radius = sample["radius_kpc"].to_numpy(dtype=float)
    zero_host = np.zeros(len(sample), dtype=float)
    host_chi_by_model = {
        FIXED_RAR: zero_host,
        CLUSTER_RAR: zero_host,
        H7S_MODEL_NAME: zero_host,
        H7S_HOST: sample["host_cosmic_baryon_chi_r200"].to_numpy(dtype=float),
        H7S_ERASS_MEDIAN_GAS: sample[
            "host_erass_median_gas_chi_r200"
        ].to_numpy(dtype=float),
        H7S_ERASS_P90_GAS: sample["host_erass_p90_gas_chi_r200"].to_numpy(
            dtype=float
        ),
    }
    for name, host_chi in host_chi_by_model.items():
        output = sample.copy()
        predicted = _predict(name, gbar, radius, vector, host_chi)
        upper = _predict(name, gbar * np.exp(epsilon), radius, vector, host_chi)
        lower = _predict(name, gbar * np.exp(-epsilon), radius, vector, host_chi)
        slope = (np.log(upper) - np.log(lower)) / (2.0 * epsilon)
        output["model"] = name
        output["predicted_log_gobs"] = np.log10(predicted)
        output["model_log_slope_gbar"] = slope
        output["sigma_residual_dex"] = np.sqrt(
            np.square(output["err_log_gobs"])
            + np.square(slope * output["err_log_gbar"])
        )
        output["residual_dex"] = output["predicted_log_gobs"] - output["log_gobs"]
        output["chi2_term"] = np.square(
            output["residual_dex"] / output["sigma_residual_dex"]
        )
        pieces.append(output)
    predictions = pd.concat(pieces, ignore_index=True)
    metrics = {}
    for name, model_frame in predictions.groupby("model", sort=False):
        sources = {}
        for source, frame in model_frame.groupby("measurement_source", sort=False):
            sources[str(source)] = _metrics(frame)
        record = _metrics(model_frame)
        record["by_measurement_source"] = sources
        metrics[str(name)] = record
    return predictions, metrics


def _metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    residual = frame["residual_dex"].to_numpy(dtype=float)
    return {
        "systems": len(frame),
        "chi2_per_point": float(frame["chi2_term"].mean()),
        "rms_dex": float(np.sqrt(np.mean(np.square(residual)))),
        "median_abs_dex": float(np.median(np.abs(residual))),
        "mean_residual_dex": float(np.mean(residual)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the independent SPIDERS-MaNGA BCG bridge and score frozen H7s."
    )
    parser.add_argument(
        "--registry", type=Path, default=ROOT / "configs" / "bcg_bridge_sample.json"
    )
    parser.add_argument(
        "--gates", type=Path, default=ROOT / "configs" / "theory_stage_gates.json"
    )
    parser.add_argument(
        "--jam",
        type=Path,
        default=ROOT / "data" / "raw" / "manga_dynpop" / "SDSSDR17_MaNGA_JAM.fits",
    )
    parser.add_argument(
        "--gema",
        type=Path,
        default=ROOT / "data" / "raw" / "manga_gema_dr17" / "GEMA_2.0.2.fits",
    )
    parser.add_argument(
        "--spiders",
        type=Path,
        default=ROOT
        / "data"
        / "raw"
        / "spiders_clusters"
        / "catCluster-SPIDERS_RASS_CLUS-v3.0.fits",
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
        "--tian",
        type=Path,
        default=ROOT / "data" / "raw" / "manga_bcg_tian2024" / "RAR_BCG.tex",
    )
    parser.add_argument("--sparc", type=Path, default=ROOT / "data" / "raw" / "sparc")
    parser.add_argument(
        "--clash",
        type=Path,
        default=ROOT / "data" / "raw" / "clash_tian2020" / "fig2.dat",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "bcg_bridge_sample"
    )
    parser.add_argument(
        "--sample-output",
        type=Path,
        default=ROOT / "data" / "derived" / "bcg_bridge_sample.csv",
    )
    parser.add_argument("--starts", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    gates = json.loads(args.gates.read_text(encoding="utf-8"))
    selection = registry["catalog_selection"]
    hubble_h = registry["acceleration_proxy"]["nsa_mass_h"]
    jam = _load_jam(args.jam)
    gema = _load_gema(args.gema)
    spiders = _load_spiders(args.spiders)
    erass_gas_fraction = _erass_gas_fraction_summary(args.erass)
    tian = parse_table(args.tian)
    matched = _build_matches(jam, gema, spiders, selection)
    test_plateifus = set(matched["plateifu"])
    calibration, calibration_frame = _calibrate_proxy(
        tian,
        jam,
        test_plateifus,
        minimum_quality=selection["jam_minimum_visual_quality"],
        hubble_h=hubble_h,
    )
    sample = _measurement_sample(
        matched,
        tian,
        calibration,
        hubble_h=hubble_h,
        erass_gas_fraction=erass_gas_fraction,
    )

    galaxy = load_sparc_acceleration_frame(args.sparc)
    cluster = load_clash_acceleration_frame(args.clash)
    fit = fit_h7s(galaxy, cluster, starts=args.starts, seed=args.seed)
    predictions, metrics = _score(sample, fit.vector)

    host_gate = gates["stage_4_independent_environment"]
    host_complete = (
        (sample["spiders_xray_luminosity_erg_s"] > 0.0)
        & (sample["spiders_r200_kpc"] > 0.0)
    )
    coverage_fraction = float(host_complete.mean())
    sample_gate = {
        "systems": len(sample),
        "systems_minimum": host_gate["host_profile_systems_min"],
        "systems_pass": len(sample) >= host_gate["host_profile_systems_min"],
        "host_xray_coverage_fraction": coverage_fraction,
        "coverage_minimum": host_gate["host_profile_coverage_fraction_min"],
        "coverage_pass": coverage_fraction
        >= host_gate["host_profile_coverage_fraction_min"],
    }
    sample_gate["passes_all"] = bool(
        sample_gate["systems_pass"] and sample_gate["coverage_pass"]
    )
    continue_gate = host_gate["continue_gate"]
    success_gate = host_gate["scientific_success_gate"]
    score_gates = {}
    for model_name in (
        H7S_MODEL_NAME,
        H7S_ERASS_MEDIAN_GAS,
        H7S_ERASS_P90_GAS,
        H7S_HOST,
    ):
        candidate_metrics = metrics[model_name]
        score_gate = {
            "continue": {
                "chi2_per_point": candidate_metrics["chi2_per_point"]
                <= continue_gate["bcg_chi2_per_point_max"],
                "absolute_mean_residual": abs(candidate_metrics["mean_residual_dex"])
                <= continue_gate["bcg_absolute_mean_residual_dex_max"],
            },
            "scientific_success": {
                "chi2_per_point": candidate_metrics["chi2_per_point"]
                <= success_gate["bcg_chi2_per_point_max"],
                "rms": candidate_metrics["rms_dex"]
                <= success_gate["bcg_rms_dex_max"],
                "absolute_mean_residual": abs(candidate_metrics["mean_residual_dex"])
                <= success_gate["bcg_absolute_mean_residual_dex_max"],
            },
        }
        for record in score_gate.values():
            record["passes_all"] = all(record.values())
        score_gates[model_name] = score_gate

    args.output.mkdir(parents=True, exist_ok=True)
    args.sample_output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.sample_output, index=False)
    predictions.to_csv(args.output / "predictions.csv", index=False)
    calibration_frame.to_csv(args.output / "proxy_calibration.csv", index=False)
    report = {
        "status": "completed frozen SPIDERS-MaNGA BCG bridge and host-scale controls",
        "inputs": {
            "registry_sha256": _sha256(args.registry),
            "jam_sha256": _sha256(args.jam),
            "gema_sha256": _sha256(args.gema),
            "spiders_sha256": _sha256(args.spiders),
            "erass_sha256": _sha256(args.erass),
            "tian_sha256": _sha256(args.tian),
        },
        "selection": selection,
        "sample": {
            "systems": len(sample),
            "unique_spiders_hosts": int(sample["spiders_id"].nunique()),
            "direct_tian_accelerations": int(
                (sample["measurement_source"] == "Tian2024_direct").sum()
            ),
            "calibrated_dynpop_proxies": int(
                (sample["measurement_source"] == "DynPop_NSA_calibrated_proxy").sum()
            ),
            "jam_quality_counts": {
                str(key): int(value)
                for key, value in sample["jam_quality"].value_counts().sort_index().items()
            },
            "host_xray_coverage_fraction": coverage_fraction,
            "host_cosmic_baryon_chi_r200": {
                "p10": float(sample["host_cosmic_baryon_chi_r200"].quantile(0.1)),
                "median": float(sample["host_cosmic_baryon_chi_r200"].median()),
                "p90": float(sample["host_cosmic_baryon_chi_r200"].quantile(0.9)),
            },
            "projected_separation_kpc": {
                "median": float(sample["spiders_projected_separation_kpc"].median()),
                "maximum": float(sample["spiders_projected_separation_kpc"].max()),
            },
        },
        "disjoint_proxy_calibration": {
            "test_plateifus_used": False,
            "calibration_systems": len(calibration_frame),
            "metrics": calibration,
        },
        "frozen_h7s_development_fit": {
            "SPARC_systems": int(galaxy["system"].nunique()),
            "CLASH_systems": int(cluster["system"].nunique()),
            "parameters": fit.parameters,
            "raw_vector": fit.vector.tolist(),
            "optimizer_success": fit.success,
            "starts": fit.starts,
            "bcg_values_used": False,
        },
        "metrics": metrics,
        "host_completion": registry["host_completion"],
        "external_erass_gas_fraction": erass_gas_fraction,
        "gate_audit": {"sample": sample_gate, "scores": score_gates},
        "interpretation_guardrails": [
            "This is a new, disjoint BCG bridge sample except for direct Tian test values; no BCG value entered the H7s theory fit.",
            "The DynPop/NSA proxy offsets were calibrated only on Tian systems absent from the test sample.",
            "JAM quality zero systems are retained with the empirical proxy-calibration scatter propagated as measurement uncertainty.",
            "The host-completed score uses a zero-fit cosmic-baryon potential scale derived from SPIDERS R200; it does not fit a host normalization to BCG accelerations.",
            "The eRASS median and p90 gas-fraction controls are derived from 10,440 independent catalog systems and likewise use no BCG acceleration.",
            "SPIDERS R200 is inferred from X-ray scaling and is not a resolved gas density profile, so even a passing score would license a resolved-profile test rather than complete Stage 4.",
        ],
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
