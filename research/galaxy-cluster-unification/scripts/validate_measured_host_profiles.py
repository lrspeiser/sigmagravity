from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.bcg_bridge import physical_radius_kpc
from voidscreen.host_profiles import (
    nfw_overdensity_conversion,
    potential_chi_from_mass,
    sersic_deprojected_potential_factor,
    spherical_profile_potential_factor,
    truncated_nfw_potential_factor,
    vikhlinin_density_shape,
)
from voidscreen.theory import h7s_acceleration


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentiles(values: np.ndarray) -> dict[str, float]:
    quantiles = np.percentile(np.asarray(values, dtype=float), [5, 16, 50, 84, 95])
    return dict(zip(("p05", "p16", "median", "p84", "p95"), map(float, quantiles)))


def _metrics(residual: np.ndarray, sigma: np.ndarray) -> dict[str, float | int]:
    return {
        "systems": residual.size,
        "chi2_per_point": float(np.mean(np.square(residual / sigma))),
        "rms_dex": float(np.sqrt(np.mean(np.square(residual)))),
        "median_abs_dex": float(np.median(np.abs(residual))),
        "mean_residual_dex": float(np.mean(residual)),
    }


def _score(
    sample: pd.DataFrame,
    vector: np.ndarray,
    bcg_chi: np.ndarray,
    host_chi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    gbar = sample["gbar_m_s2"].to_numpy(dtype=float)
    gobs = sample["log_gobs"].to_numpy(dtype=float)
    epsilon = 1e-5
    combined = bcg_chi + host_chi
    predicted = h7s_acceleration(gbar, combined, vector)
    upper = h7s_acceleration(
        gbar * np.exp(epsilon), bcg_chi * np.exp(epsilon) + host_chi, vector
    )
    lower = h7s_acceleration(
        gbar * np.exp(-epsilon), bcg_chi * np.exp(-epsilon) + host_chi, vector
    )
    slope = (np.log(upper) - np.log(lower)) / (2.0 * epsilon)
    residual = np.log10(predicted) - gobs
    sigma = np.sqrt(
        np.square(sample["err_log_gobs"].to_numpy(dtype=float))
        + np.square(slope * sample["err_log_gbar"].to_numpy(dtype=float))
    )
    return predicted, residual, sigma, _metrics(residual, sigma)


def _fit_erass_gas_fraction(path: Path, config: dict) -> tuple[dict, pd.DataFrame]:
    with fits.open(path, memmap=True) as hdus:
        data = hdus[1].data
        mass = np.asarray(data["M500"], dtype=float) * 1e13
        fraction = np.asarray(data["FGAS500"], dtype=float)
    valid = (
        np.isfinite(mass)
        & np.isfinite(fraction)
        & (mass > 0.0)
        & (fraction > 0.0)
        & (fraction < 0.3)
    )
    log_mass = np.log10(mass[valid] / 1e14)
    log_fraction = np.log10(fraction[valid])
    low, high = np.percentile(log_mass, config["log_mass_percentile_range"])
    edges = np.linspace(low, high, int(config["equal_log_mass_bins"]) + 1)
    rows = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (log_mass >= lower) & (log_mass < upper)
        if np.any(selected):
            rows.append(
                {
                    "log10_m500_over_1e14": float(np.median(log_mass[selected])),
                    "median_log10_fgas500": float(np.median(log_fraction[selected])),
                    "systems": int(np.sum(selected)),
                }
            )
    bins = pd.DataFrame(rows)
    coefficients, covariance = np.polyfit(
        bins["log10_m500_over_1e14"],
        bins["median_log10_fgas500"],
        1,
        cov=True,
    )
    slope, intercept = map(float, coefficients)
    residual = log_fraction - (intercept + slope * log_mass)
    scatter = float((np.percentile(residual, 84) - np.percentile(residual, 16)) / 2.0)
    return (
        {
            "catalog_systems": int(np.sum(valid)),
            "slope": slope,
            "intercept_at_1e14_msun": intercept,
            "coefficient_covariance_slope_intercept": covariance.tolist(),
            "robust_scatter_dex": scatter,
        },
        bins,
    )


def _load_gas_profile_factors(
    sample: pd.DataFrame,
    r500_kpc: np.ndarray,
    path: Path,
) -> tuple[pd.DataFrame, np.ndarray]:
    profiles = pd.read_csv(path)
    required = ("R500", "alpha", "beta", "rc", "rs", "eps")
    profiles = profiles.dropna(subset=list(required)).reset_index(drop=True)
    factors = np.empty((len(sample), len(profiles)), dtype=float)
    radius_ratio = sample["radius_kpc"].to_numpy(dtype=float) / r500_kpc
    for host_index, host_radius in enumerate(radius_ratio):
        for profile_index, row in enumerate(profiles.itertuples(index=False)):
            density = lambda radius, row=row: vikhlinin_density_shape(
                radius,
                alpha=float(row.alpha),
                beta=float(row.beta),
                core_over_r500=float(row.rc / row.R500),
                steepening_over_r500=float(row.rs / row.R500),
                epsilon=float(row.eps),
            )
            factors[host_index, profile_index] = spherical_profile_potential_factor(
                density, float(host_radius)
            )
    return profiles, factors


def _gate(metrics: dict, gate: dict) -> dict[str, bool]:
    result = {
        "chi2_per_point": metrics["chi2_per_point"] <= gate["bcg_chi2_per_point_max"],
        "absolute_mean_residual": abs(metrics["mean_residual_dex"])
        <= gate["bcg_absolute_mean_residual_dex_max"],
    }
    if "bcg_rms_dex_max" in gate:
        result["rms"] = metrics["rms_dex"] <= gate["bcg_rms_dex_max"]
    result["passes_all"] = bool(all(result.values()))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen Stage 4 host-profile test.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "measured_host_profile_validation.json",
    )
    parser.add_argument(
        "--gates", type=Path, default=ROOT / "configs" / "theory_stage_gates.json"
    )
    parser.add_argument(
        "--sample", type=Path, default=ROOT / "data" / "derived" / "bcg_bridge_sample.csv"
    )
    parser.add_argument(
        "--prior-report",
        type=Path,
        default=ROOT / "results" / "bcg_bridge_sample" / "report.json",
    )
    parser.add_argument(
        "--coverage-report",
        type=Path,
        default=ROOT / "results" / "measured_host_profiles" / "coverage_report.json",
    )
    parser.add_argument(
        "--erass",
        type=Path,
        default=ROOT / "data" / "raw" / "erass1_clusters" / "erass1cl_main_v3.2.fits",
    )
    parser.add_argument(
        "--gas-profiles",
        type=Path,
        default=ROOT
        / "data"
        / "raw"
        / "cluster_gas_profiles"
        / "elkholy2015"
        / "clusters.csv",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "measured_host_profiles"
    )
    parser.add_argument(
        "--sample-output",
        type=Path,
        default=ROOT / "data" / "derived" / "measured_host_profile_sample.csv",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    gates = json.loads(args.gates.read_text(encoding="utf-8"))[
        "stage_4_independent_environment"
    ]
    prior = json.loads(args.prior_report.read_text(encoding="utf-8"))
    coverage = json.loads(args.coverage_report.read_text(encoding="utf-8"))
    if not coverage["passes_profile_constrained_count_gate"]:
        raise RuntimeError("the frozen profile-constrained coverage gate did not pass")
    vector = np.asarray(prior["frozen_h7s_development_fit"]["raw_vector"], dtype=float)
    sample = pd.read_csv(args.sample)
    if len(sample) != int(config["sample"]["frozen_systems"]):
        raise RuntimeError("the frozen BCG bridge system count changed")

    gas_relation, gas_bins = _fit_erass_gas_fraction(args.erass, config["gas_mass"])
    concentration = float(config["mass_conversion"]["fiducial_c200"])
    radius_ratio, mass_ratio = nfw_overdensity_conversion(concentration)
    r500_kpc = sample["spiders_r200_kpc"].to_numpy(dtype=float) * radius_ratio
    m500_msun = sample["spiders_m200_from_r200_msun"].to_numpy(dtype=float) * mass_ratio
    gas_profiles, gas_factor_matrix = _load_gas_profile_factors(
        sample, r500_kpc, args.gas_profiles
    )
    gas_factor = np.median(gas_factor_matrix, axis=1)
    log_fgas = gas_relation["intercept_at_1e14_msun"] + gas_relation["slope"] * np.log10(
        m500_msun / 1e14
    )
    fgas = np.clip(
        np.power(10.0, log_fgas),
        config["gas_mass"]["fgas_floor"],
        config["gas_mass"]["fgas_ceiling"],
    )
    gas_mass_msun = fgas * m500_msun
    gas_chi = potential_chi_from_mass(gas_mass_msun, r500_kpc, gas_factor)

    satellite_config = config["satellite_stars"]
    log_satellite_mass = satellite_config["normalization"] + satellite_config["slope"] * (
        np.log10(m500_msun) - 14.5
    )
    satellite_mass_msun = np.power(10.0, log_satellite_mass)
    satellite_concentration = r500_kpc / (1000.0 * satellite_config["scale_radius_mpc"])
    satellite_factor = truncated_nfw_potential_factor(satellite_concentration, 0.0)
    satellite_chi = potential_chi_from_mass(
        satellite_mass_msun, r500_kpc, satellite_factor
    )

    sersic_re_kpc = physical_radius_kpc(
        sample["distance_mpc"], sample["nsa_sersic_re_arcsec"]
    )
    bcg_factor = sersic_deprojected_potential_factor(
        sample["radius_kpc"].to_numpy(dtype=float) / sersic_re_kpc,
        np.clip(sample["nsa_sersic_n"].to_numpy(dtype=float), 0.3, 10.0),
    )
    bcg_enclosed_chi = sample["bcg_baryonic_chi"].to_numpy(dtype=float)
    bcg_chi = bcg_enclosed_chi * bcg_factor
    host_chi = gas_chi + satellite_chi
    predicted, residual, sigma, point_metrics = _score(sample, vector, bcg_chi, host_chi)

    profile_sample = sample.copy()
    profile_sample["r500_kpc"] = r500_kpc
    profile_sample["m500_msun"] = m500_msun
    profile_sample["fgas500_profile_constrained"] = fgas
    profile_sample["gas_mass500_msun"] = gas_mass_msun
    profile_sample["gas_profile_factor"] = gas_factor
    profile_sample["gas_chi"] = gas_chi
    profile_sample["satellite_mass500_msun"] = satellite_mass_msun
    profile_sample["satellite_profile_factor"] = satellite_factor
    profile_sample["satellite_chi"] = satellite_chi
    profile_sample["bcg_profile_factor"] = bcg_factor
    profile_sample["bcg_complete_chi"] = bcg_chi
    profile_sample["total_profile_chi"] = bcg_chi + host_chi
    profile_sample["predicted_log_gobs"] = np.log10(predicted)
    profile_sample["residual_dex"] = residual
    profile_sample["sigma_residual_dex"] = sigma

    source_metrics = {}
    for source, group in profile_sample.groupby("measurement_source", sort=False):
        indices = group.index.to_numpy()
        source_metrics[str(source)] = _metrics(residual[indices], sigma[indices])
    point_metrics["by_measurement_source"] = source_metrics

    monte_carlo = config["monte_carlo"]
    realizations = int(monte_carlo["realizations"])
    rng = np.random.default_rng(int(monte_carlo["seed"]))
    concentrations = rng.uniform(
        config["mass_conversion"]["c200_min"],
        config["mass_conversion"]["c200_max"],
        size=realizations,
    )
    conversions = np.asarray([nfw_overdensity_conversion(value) for value in concentrations])
    mc_r500 = conversions[:, 0, None] * sample["spiders_r200_kpc"].to_numpy()[None, :]
    mc_m500 = (
        conversions[:, 1, None]
        * sample["spiders_m200_from_r200_msun"].to_numpy()[None, :]
    )
    coefficient_mean = np.asarray(
        [gas_relation["slope"], gas_relation["intercept_at_1e14_msun"]]
    )
    coefficient_covariance = np.asarray(
        gas_relation["coefficient_covariance_slope_intercept"]
    )
    gas_coefficients = rng.multivariate_normal(
        coefficient_mean, coefficient_covariance, size=realizations
    )
    mc_log_fgas = gas_coefficients[:, 1, None] + gas_coefficients[:, 0, None] * np.log10(
        mc_m500 / 1e14
    )
    mc_log_fgas += rng.normal(
        0.0, gas_relation["robust_scatter_dex"], size=(realizations, len(sample))
    )
    mc_fgas = np.clip(
        np.power(10.0, mc_log_fgas),
        config["gas_mass"]["fgas_floor"],
        config["gas_mass"]["fgas_ceiling"],
    )
    profile_draws = rng.integers(0, len(gas_profiles), size=(realizations, len(sample)))
    mc_gas_factor = gas_factor_matrix[np.arange(len(sample))[None, :], profile_draws]
    mc_gas_chi = potential_chi_from_mass(mc_fgas * mc_m500, mc_r500, mc_gas_factor)

    mc_satellite_normalization = rng.normal(
        satellite_config["normalization"],
        satellite_config["normalization_sigma"],
        size=realizations,
    )
    mc_satellite_slope = rng.normal(
        satellite_config["slope"], satellite_config["slope_sigma"], size=realizations
    )
    mc_log_satellite_mass = mc_satellite_normalization[:, None] + mc_satellite_slope[
        :, None
    ] * (np.log10(mc_m500) - 14.5)
    mc_log_satellite_mass += rng.normal(
        0.0,
        satellite_config["intrinsic_scatter_dex"],
        size=(realizations, len(sample)),
    )
    satellite_scale_radius = np.clip(
        rng.normal(
            satellite_config["scale_radius_mpc"],
            satellite_config["scale_radius_sigma_mpc"],
            size=realizations,
        ),
        *satellite_config["scale_radius_bounds_mpc"],
    )
    mc_satellite_concentration = mc_r500 / (1000.0 * satellite_scale_radius[:, None])
    mc_satellite_factor = truncated_nfw_potential_factor(mc_satellite_concentration, 0.0)
    mc_satellite_chi = potential_chi_from_mass(
        np.power(10.0, mc_log_satellite_mass), mc_r500, mc_satellite_factor
    )

    exterior_chi = bcg_enclosed_chi * (bcg_factor - 1.0)
    mc_bcg_chi = bcg_enclosed_chi[None, :] + exterior_chi[None, :] * np.power(
        10.0,
        rng.normal(
            0.0,
            config["bcg_stars"]["exterior_log_scatter_dex"],
            size=(realizations, len(sample)),
        ),
    )
    mc_host_chi = mc_gas_chi + mc_satellite_chi
    gbar = np.broadcast_to(
        sample["gbar_m_s2"].to_numpy(dtype=float)[None, :], mc_host_chi.shape
    )
    epsilon = 1e-5
    mc_predicted = h7s_acceleration(gbar, mc_bcg_chi + mc_host_chi, vector)
    mc_upper = h7s_acceleration(
        gbar * np.exp(epsilon), mc_bcg_chi * np.exp(epsilon) + mc_host_chi, vector
    )
    mc_lower = h7s_acceleration(
        gbar * np.exp(-epsilon), mc_bcg_chi * np.exp(-epsilon) + mc_host_chi, vector
    )
    mc_slope = (np.log(mc_upper) - np.log(mc_lower)) / (2.0 * epsilon)
    mc_residual = np.log10(mc_predicted) - sample["log_gobs"].to_numpy()[None, :]
    mc_sigma = np.sqrt(
        np.square(sample["err_log_gobs"].to_numpy()[None, :])
        + np.square(mc_slope * sample["err_log_gbar"].to_numpy()[None, :])
    )
    mc_chi2 = np.mean(np.square(mc_residual / mc_sigma), axis=1)
    mc_rms = np.sqrt(np.mean(np.square(mc_residual), axis=1))
    mc_mean = np.mean(mc_residual, axis=1)
    continue_gate = gates["continue_gate"]
    science_gate = gates["scientific_success_gate"]
    mc_continue = (mc_chi2 <= continue_gate["bcg_chi2_per_point_max"]) & (
        np.abs(mc_mean) <= continue_gate["bcg_absolute_mean_residual_dex_max"]
    )
    mc_science = (
        (mc_chi2 <= science_gate["bcg_chi2_per_point_max"])
        & (mc_rms <= science_gate["bcg_rms_dex_max"])
        & (np.abs(mc_mean) <= science_gate["bcg_absolute_mean_residual_dex_max"])
    )
    point_gates = {
        "continue": _gate(point_metrics, continue_gate),
        "scientific_success": _gate(point_metrics, science_gate),
    }
    continue_probability = float(np.mean(mc_continue))
    science_probability = float(np.mean(mc_science))
    robust_continue = bool(
        point_gates["continue"]["passes_all"]
        and continue_probability >= monte_carlo["continue_pass_probability_min"]
    )
    robust_science = bool(
        point_gates["scientific_success"]["passes_all"]
        and science_probability >= monte_carlo["scientific_pass_probability_min"]
    )

    args.output.mkdir(parents=True, exist_ok=True)
    args.sample_output.parent.mkdir(parents=True, exist_ok=True)
    profile_sample.to_csv(args.sample_output, index=False)
    gas_bins.to_csv(args.output / "erass_gas_fraction_bins.csv", index=False)
    pd.DataFrame(
        {
            "chi2_per_point": mc_chi2,
            "rms_dex": mc_rms,
            "mean_residual_dex": mc_mean,
            "passes_continue": mc_continue,
            "passes_scientific_success": mc_science,
        }
    ).to_csv(args.output / "monte_carlo_metrics.csv", index=False)
    report = {
        "status": "completed frozen Stage 4 measured/profile-constrained host validation",
        "inputs": {
            "config_sha256": _sha256(args.config),
            "gates_sha256": _sha256(args.gates),
            "sample_sha256": _sha256(args.sample),
            "prior_report_sha256": _sha256(args.prior_report),
            "coverage_report_sha256": _sha256(args.coverage_report),
            "erass_sha256": _sha256(args.erass),
            "gas_profiles_sha256": _sha256(args.gas_profiles),
        },
        "sample": {
            "systems": len(sample),
            "profile_constrained_systems": len(sample),
            "direct_xray_systems": coverage["coverage"]["direct_xray_union_hosts"],
            "independent_satellite_catalog_systems": coverage["coverage"][
                "independent_satellite_catalog_union_hosts"
            ],
            "minimum_required": config["sample"]["required_systems"],
            "passes_coverage": len(sample) >= config["sample"]["required_systems"],
            "bcg_residual_used_for_selection": False,
        },
        "frozen_h7s_parameters": prior["frozen_h7s_development_fit"],
        "gas_fraction_relation": gas_relation,
        "profile_inputs": {
            "published_gas_profiles": len(gas_profiles),
            "gas_profile_population_potential_factor": _percentiles(gas_factor_matrix.ravel()),
            "host_fiducial_gas_potential_factor": _percentiles(gas_factor),
            "bcg_potential_factor": _percentiles(bcg_factor),
            "satellite_potential_factor": _percentiles(satellite_factor),
            "fgas500": _percentiles(fgas),
        },
        "point_metrics": point_metrics,
        "point_gates": point_gates,
        "uncertainty_propagation": {
            "realizations": realizations,
            "seed": monte_carlo["seed"],
            "chi2_per_point": _percentiles(mc_chi2),
            "rms_dex": _percentiles(mc_rms),
            "mean_residual_dex": _percentiles(mc_mean),
            "continue_pass_probability": continue_probability,
            "scientific_success_pass_probability": science_probability,
            "required_continue_probability": monte_carlo[
                "continue_pass_probability_min"
            ],
            "required_scientific_probability": monte_carlo[
                "scientific_pass_probability_min"
            ],
        },
        "decision": {
            "robust_continue": robust_continue,
            "robust_scientific_success": robust_science,
            "stage_4_mechanism_supported": robust_continue,
            "next_action": (
                config["decision"]["science_success"]
                if robust_continue
                else config["decision"]["continue_failure"]
            ),
            "stage_3_identifiability_failure_still_active": True,
        },
        "guardrails": [
            "No BCG acceleration or residual set a host mass, profile, or normalization.",
            "The H7s parameters are copied unchanged from the SPARC+CLASH development fit.",
            "All 34 frozen systems are scored; none is removed using its residual.",
            "Gas and satellite mass outside R500 is excluded.",
            "Passing Stage 4 does not override the Stage 3 hard-bound failure.",
            config["honesty_note"],
        ],
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
