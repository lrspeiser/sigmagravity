"""Propagate the RX J2129 off-center satellite stellar-force likelihood."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.constants import G
import astropy.units as u


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/r1_rxj2129_satellite_force_protocol.json"
G_M_SUN_KPC2 = G.to_value(u.m**3 / (u.kg * u.s**2)) * u.Msun.to(u.kg) / u.kpc.to(u.m) ** 2


def _resolve(path: str) -> Path:
    return ROOT / path


def _kernels(
    candidates: pd.DataFrame,
    radii_kpc: np.ndarray,
    scale_kpc_arcsec: float,
    angles: np.ndarray,
    scale_multiplier: float,
    center_ra: float,
    center_dec: float,
) -> np.ndarray:
    cosine = np.cos(np.deg2rad(center_dec))
    sx = (candidates["ra_deg"].to_numpy() - center_ra) * cosine * 3600.0 * scale_kpc_arcsec
    sy = (candidates["dec_deg"].to_numpy() - center_dec) * 3600.0 * scale_kpc_arcsec
    size_arcsec = np.maximum(
        0.065,
        np.sqrt(candidates["a"].to_numpy() * candidates["b"].to_numpy()) * 0.065,
    )
    softening = size_arcsec * scale_kpc_arcsec * scale_multiplier
    kernels = np.empty((len(candidates), len(radii_kpc), len(angles)), dtype=float)
    center_distance2 = sx**2 + sy**2 + softening**2
    center_ax = G_M_SUN_KPC2 * sx / center_distance2**1.5
    center_ay = G_M_SUN_KPC2 * sy / center_distance2**1.5
    for radial_index, radius in enumerate(radii_kpc):
        px = radius * np.cos(angles)
        py = radius * np.sin(angles)
        dx = sx[:, None] - px[None, :]
        dy = sy[:, None] - py[None, :]
        distance2 = dx**2 + dy**2 + softening[:, None] ** 2
        ax = G_M_SUN_KPC2 * dx / distance2**1.5 - center_ax[:, None]
        ay = G_M_SUN_KPC2 * dy / distance2**1.5 - center_ay[:, None]
        inward_x = -np.cos(angles)
        inward_y = -np.sin(angles)
        kernels[:, radial_index, :] = ax * inward_x[None, :] + ay * inward_y[None, :]
    return kernels


def _draw_weights(
    candidates: pd.DataFrame,
    probabilities: np.ndarray,
    draws: int,
    mass_sigma_dex: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    bootstrap_rows = rng.integers(0, probabilities.shape[0], size=draws)
    selected_probabilities = probabilities[bootstrap_rows]
    membership = rng.random(selected_probabilities.shape) < selected_probabilities
    log_mass = candidates["log10_stellar_mass_msun"].to_numpy()[None, :]
    mass = np.power(10.0, log_mass + rng.normal(0.0, mass_sigma_dex, size=membership.shape))
    return membership * mass, bootstrap_rows


def _evaluate(weights: np.ndarray, kernels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = weights @ kernels.reshape(kernels.shape[0], -1)
    total = total.reshape(weights.shape[0], kernels.shape[1], kernels.shape[2])
    radial_mean = total.mean(axis=2)
    anisotropic_rms = total.std(axis=2, ddof=1)
    return radial_mean, anisotropic_rms


def _write_covariance(path: Path, covariance: np.ndarray) -> None:
    labels = [f"bin_{index + 1}" for index in range(covariance.shape[0])]
    frame = pd.DataFrame(covariance, columns=labels)
    frame.insert(0, "row", labels)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _plot(profile: pd.DataFrame, report: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    axes[0].errorbar(
        profile["radius_arcsec"],
        profile["satellite_radial_acceleration_mean_m_s2"],
        yerr=profile["satellite_radial_acceleration_sigma_m_s2"],
        marker="o",
        label="posterior radial mean",
    )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set(xlabel="radius (arcsec)", ylabel="inward satellite acceleration (m s$^{-2}$)")
    axes[1].plot(
        profile["radius_arcsec"],
        profile["anisotropic_radial_rms_median_m_s2"],
        marker="o",
        label="inner posterior anisotropic RMS",
    )
    axes[1].plot(
        profile["radius_arcsec"],
        profile["outer_tidal_upper_bound_m_s2"],
        marker="s",
        label="outer worst-case tidal bound",
    )
    axes[1].set(xlabel="radius (arcsec)", ylabel="acceleration scale (m s$^{-2}$)", yscale="log")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("RX J2129 off-center satellite stellar-force likelihood")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def propagate(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    authorization = config["authorization"]
    if not authorization["off_center_mass_acceleration_likelihood"]:
        raise ValueError("satellite force propagation is not authorized")
    if authorization["lens_residual_read"] or authorization["gravity_response_fit"]:
        raise ValueError("satellite force protocol cannot read a residual")
    membership_report = json.loads(
        _resolve(config["inputs"]["membership_report"]).read_text(encoding="utf-8")
    )
    if not membership_report["membership_probability_gate_pass"]:
        raise ValueError("membership probability gate failed")
    candidates = pd.read_csv(_resolve(config["inputs"]["membership_likelihood"]))
    probability_product = np.load(
        _resolve(config["inputs"]["membership_probability_bootstrap"])
    )
    probability_ids = probability_product["clash_ids"].astype(str)
    if not np.array_equal(probability_ids, candidates["clash_id"].astype(str).to_numpy()):
        raise ValueError("membership bootstrap candidate order mismatch")
    probabilities = probability_product["membership_probability"]
    dynamics = pd.read_csv(_resolve(config["inputs"]["dynamics_bins"]))
    radii_arcsec = 0.5 * (
        dynamics["semimajor_min_arcsec"].to_numpy()
        + dynamics["semimajor_max_arcsec"].to_numpy()
    )
    scale = config["geometry"]["angular_scale_kpc_per_arcsec"]
    radii_kpc = radii_arcsec * scale
    angles = np.linspace(
        0.0,
        2.0 * np.pi,
        config["geometry"]["azimuth_samples_per_radius"],
        endpoint=False,
    )
    weights, bootstrap_rows = _draw_weights(
        candidates,
        probabilities,
        config["monte_carlo"]["draws"],
        config["galaxy_profile"]["mass_uncertainty_dex"],
        config["monte_carlo"]["random_seed"],
    )
    evaluations: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for multiplier in config["galaxy_profile"]["scale_sensitivity_multipliers"]:
        kernels = _kernels(
            candidates,
            radii_kpc,
            scale,
            angles,
            multiplier,
            config["geometry"]["center_ra_deg"],
            config["geometry"]["center_dec_deg"],
        )
        evaluations[float(multiplier)] = _evaluate(weights, kernels)
    baseline_radial, baseline_anisotropic = evaluations[1.0]
    covariance = np.cov(baseline_radial, rowvar=False, ddof=1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    bcg = pd.read_csv(_resolve(config["inputs"]["published_bcg_profile"]))
    bcg_acceleration = bcg["bcg_acceleration_m_s2"].to_numpy()
    scale_shifts = np.max(
        np.vstack(
            [
                np.abs(evaluations[multiplier][0].mean(axis=0) - baseline_radial.mean(axis=0))
                for multiplier in evaluations
                if multiplier != 1.0
            ]
        ),
        axis=0,
    )
    full_candidates = pd.read_csv(_resolve(config["inputs"]["full_candidate_ledger"]))
    outer = full_candidates[
        full_candidates["separation_arcsec"]
        > config["geometry"]["inner_membership_radius_arcsec"]
    ]
    outer_radius_kpc = outer["separation_arcsec"].to_numpy() * scale
    outer_mass_upper = outer["stellar_mass_upper_0p30dex_msun"].to_numpy()
    outer_bound = np.asarray(
        [
            np.sum(
                2.0
                * G_M_SUN_KPC2
                * outer_mass_upper
                * radius
                / np.maximum(outer_radius_kpc - radius, 1e-6) ** 3
            )
            for radius in radii_kpc
        ]
    )
    profile = pd.DataFrame(
        {
            "bin": dynamics["bin"].to_numpy(),
            "radius_arcsec": radii_arcsec,
            "radius_kpc": radii_kpc,
            "satellite_radial_acceleration_mean_m_s2": baseline_radial.mean(axis=0),
            "satellite_radial_acceleration_sigma_m_s2": baseline_radial.std(axis=0, ddof=1),
            "satellite_radial_acceleration_p16_m_s2": np.quantile(baseline_radial, 0.16, axis=0),
            "satellite_radial_acceleration_p84_m_s2": np.quantile(baseline_radial, 0.84, axis=0),
            "anisotropic_radial_rms_median_m_s2": np.median(baseline_anisotropic, axis=0),
            "maximum_plummer_scale_shift_m_s2": scale_shifts,
            "maximum_plummer_scale_shift_fraction_of_bcg": scale_shifts / bcg_acceleration,
            "outer_tidal_upper_bound_m_s2": outer_bound,
            "outer_tidal_upper_bound_fraction_of_bcg": outer_bound / bcg_acceleration,
        }
    )
    finite = bool(np.isfinite(profile.select_dtypes(include=[np.number])).all().all())
    psd = bool(np.allclose(covariance, covariance.T) and eigenvalues.min() >= -1e-30)
    thresholds = config["advance_thresholds"]
    checks = {
        "inner_candidates": len(candidates) == thresholds["inner_candidates"],
        "monte_carlo_draws": len(baseline_radial) == thresholds["monte_carlo_draws"],
        "posterior_profile_finite": finite,
        "four_by_four_covariance_symmetric_positive_semidefinite": psd,
        "maximum_plummer_scale_shift_fraction_of_bcg_each_bin": bool(
            (profile["maximum_plummer_scale_shift_fraction_of_bcg"]
             <= thresholds["maximum_plummer_scale_shift_fraction_of_bcg_each_bin"]).all()
        ),
        "outer_tidal_bound_fraction_of_bcg_each_bin": bool(
            (profile["outer_tidal_upper_bound_fraction_of_bcg"]
             <= thresholds["outer_tidal_bound_fraction_of_bcg_each_bin"]).all()
        ),
        "membership_bootstrap_correlation_preserved": True,
        "gravity_or_lens_residual_used": False,
    }
    gate_pass = all(
        checks[key] == thresholds[key]
        for key in thresholds
        if key in {"membership_bootstrap_correlation_preserved", "gravity_or_lens_residual_used"}
    ) and all(
        bool(value)
        for key, value in checks.items()
        if key not in {"membership_bootstrap_correlation_preserved", "gravity_or_lens_residual_used"}
    )
    profile_path = _resolve(config["outputs"]["profile"])
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(profile_path, index=False)
    _write_covariance(_resolve(config["outputs"]["covariance"]), covariance)
    np.savez_compressed(
        _resolve(config["outputs"]["draws"]),
        radial_acceleration_m_s2=baseline_radial,
        anisotropic_radial_rms_m_s2=baseline_anisotropic,
        membership_bootstrap_rows=bootstrap_rows,
    )
    report = {
        "protocol_version": config["protocol_version"],
        "status": (
            "off_center_satellite_stellar_force_likelihood_complete"
            if gate_pass
            else "inner_satellite_likelihood_complete_strict_outer_or_sensitivity_gate_failed"
        ),
        "gravity_or_lens_residual_read": False,
        "inner_candidates": int(len(candidates)),
        "outer_candidates_in_worst_case_bound": int(len(outer)),
        "monte_carlo_draws": int(len(baseline_radial)),
        "covariance_eigenvalues": eigenvalues.tolist(),
        "maximum_plummer_scale_shift_fraction_of_bcg": float(
            profile["maximum_plummer_scale_shift_fraction_of_bcg"].max()
        ),
        "maximum_outer_tidal_bound_fraction_of_bcg": float(
            profile["outer_tidal_upper_bound_fraction_of_bcg"].max()
        ),
        "checks": checks,
        "satellite_force_gate_pass": gate_pass,
        "lens_member_dark_subhalo_likelihood_complete": False,
        "strict_r1_ready": False,
        "outputs": config["outputs"],
        "next_action": (
            "Carry the numeric satellite stellar-force covariance into the baryonic package; separately freeze the lens member-subhalo scaling likelihood."
            if gate_pass
            else "Retain the numeric inner satellite likelihood, record the failed outer or scale gate, and do not declare the full satellite term complete."
        ),
    }
    _plot(profile, report, _resolve(config["outputs"]["diagnostic"]))
    report_path = _resolve(config["outputs"]["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(propagate(args.config), indent=2))


if __name__ == "__main__":
    main()
