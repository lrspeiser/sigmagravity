#!/usr/bin/env python3
"""Test a minimal time-dependent, weak-field metric completion of Sigma gravity."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_sigma_field_exploration import run_diagnostic_lensing  # noqa: E402
from voidscreen.covariant_closure import (  # noqa: E402
    aqual_characteristics,
    sigma_metric_lensing_acceleration,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def interpolate_log(radius, values, target) -> np.ndarray:
    radius = np.asarray(radius, dtype=float)
    values = np.asarray(values, dtype=float)
    target = np.asarray(target, dtype=float)
    return np.exp(np.interp(np.log(target), np.log(radius), np.log(values)))


def load_candidate_profiles(protocol: dict) -> pd.DataFrame:
    settings = protocol["base_candidate"]
    table = pd.read_csv(ROOT / settings["profile"])
    selected = table[table["model"].eq(settings["model"])].copy()
    if set(selected["domain"]) != {"galaxy_archetype", "RXJ2129"}:
        raise RuntimeError("candidate profile domains changed")
    if selected[["Sigma", "gbar_m_s2", "gpred_m_s2"]].isna().any().any():
        raise RuntimeError("candidate profile has missing fields")
    return selected.sort_values(["domain", "radius_kpc"]).reset_index(drop=True)


def load_cluster_target(protocol: dict) -> pd.DataFrame:
    target = pd.read_csv(
        ROOT / protocol["inputs"]["cluster_radial_target"],
        sep=r"\s+",
        names=["system", "radius_kpc", "log_gbar", "log_gobs", "err_gbar", "err_gobs"],
    )
    selected = target[target["system"] == "RXJ2129"].copy()
    if len(selected) != 5:
        raise RuntimeError("RXJ2129 radial target changed")
    return selected.sort_values("radius_kpc").reset_index(drop=True)


def characteristic_table(profiles: pd.DataFrame, protocol: dict) -> pd.DataFrame:
    settings = protocol["base_candidate"]
    records = []
    for domain, group in profiles.groupby("domain", sort=True):
        group = group.sort_values("radius_kpc")
        characteristics = aqual_characteristics(
            group["gpred_m_s2"].to_numpy(float),
            group["Sigma"].to_numpy(float),
            a0_m_s2=float(settings["a0_m_s2"]),
            activation=float(settings["activation"]),
            eta=float(settings["eta"]),
        )
        for index, row in enumerate(group.itertuples(index=False)):
            records.append(
                {
                    "domain": domain,
                    "radius_kpc": float(row.radius_kpc),
                    "Sigma": float(row.Sigma),
                    "gbar_m_s2": float(row.gbar_m_s2),
                    "gdyn_m_s2": float(row.gpred_m_s2),
                    "mu_time_kinetic": float(
                        characteristics["mu_time_kinetic"][index]
                    ),
                    "parallel_gradient_coefficient": float(
                        characteristics["parallel_gradient_coefficient"][index]
                    ),
                    "parallel_speed_squared_over_c2": float(
                        characteristics["parallel_speed_squared_over_c2"][index]
                    ),
                    "perpendicular_speed_squared_over_c2": float(
                        characteristics["perpendicular_speed_squared_over_c2"][index]
                    ),
                }
            )
    return pd.DataFrame(records)


def scan_metric_slip(
    cluster: pd.DataFrame,
    target: pd.DataFrame,
    protocol: dict,
) -> pd.DataFrame:
    target_radius = target["radius_kpc"].to_numpy(float)
    target_gobs = np.power(10.0, target["log_gobs"].to_numpy(float))
    records = []
    for zeta in protocol["metric_slip_scan"]["zeta"]:
        lensing = sigma_metric_lensing_acceleration(
            cluster["gbar_m_s2"].to_numpy(float),
            cluster["gpred_m_s2"].to_numpy(float),
            cluster["Sigma"].to_numpy(float),
            zeta=float(zeta),
        )
        sampled = interpolate_log(
            cluster["radius_kpc"].to_numpy(float), lensing, target_radius
        )
        residual = np.log10(sampled / target_gobs)
        at_100 = interpolate_log(
            cluster["radius_kpc"].to_numpy(float), lensing, [100.0]
        )[0]
        gdyn_100 = interpolate_log(
            cluster["radius_kpc"].to_numpy(float),
            cluster["gpred_m_s2"].to_numpy(float),
            [100.0],
        )[0]
        records.append(
            {
                "zeta": float(zeta),
                "radial_cluster_RMSE_dex": float(np.sqrt(np.mean(residual**2))),
                "radial_cluster_mean_log_residual_dex": float(np.mean(residual)),
                "lensing_to_dynamics_at_100kpc": float(at_100 / gdyn_100),
                "minimum_lensing_acceleration_m_s2": float(np.min(lensing)),
            }
        )
    return pd.DataFrame(records)


def lens_profile_for_zeta(cluster: pd.DataFrame, zeta: float) -> pd.DataFrame:
    lensing = sigma_metric_lensing_acceleration(
        cluster["gbar_m_s2"].to_numpy(float),
        cluster["gpred_m_s2"].to_numpy(float),
        cluster["Sigma"].to_numpy(float),
        zeta=float(zeta),
    )
    return pd.DataFrame(
        {
            "domain": "RXJ2129",
            "radius_kpc": cluster["radius_kpc"].to_numpy(float),
            "gSigma_m_s2": lensing,
        }
    )


def run_raw_lensing_candidates(
    cluster: pd.DataFrame,
    selected_zeta: float,
    protocol: dict,
) -> tuple[pd.DataFrame, dict]:
    candidates = {
        "conformal": -2.0,
        "zero_slip": 0.0,
        "radial_selected": float(selected_zeta),
    }
    prediction_tables = []
    summaries = {}
    for label, zeta in candidates.items():
        print(f"raw lensing closure {label}: zeta={zeta}", flush=True)
        profile = lens_profile_for_zeta(cluster, zeta)
        predictions, summary = run_diagnostic_lensing(
            pd.Series({"zeta": zeta}), protocol, profile
        )
        predictions["closure"] = label
        predictions["zeta"] = zeta
        prediction_tables.append(predictions)
        summaries[label] = {"zeta": zeta, **summary}
    return pd.concat(prediction_tables, ignore_index=True), summaries


def score_or_inf(summary: dict) -> float:
    value = summary["heldout"]["exact_radial_RMS_arcsec"]
    return float(value) if value is not None else float("inf")


def make_figure(
    characteristics: pd.DataFrame,
    zeta_scan: pd.DataFrame,
    profiles: pd.DataFrame,
    target: pd.DataFrame,
    selected_zeta: float,
    raw_summaries: dict,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), constrained_layout=True)

    ax = axes[0, 0]
    for domain, color in [("galaxy_archetype", "#1b9e77"), ("RXJ2129", "#377eb8")]:
        group = characteristics[characteristics["domain"] == domain]
        ax.semilogx(
            group["radius_kpc"],
            np.sqrt(group["parallel_speed_squared_over_c2"]),
            color=color,
            label=domain.replace("_", " "),
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="metric light speed")
    ax.set(
        title="Naive scalar characteristic speed",
        xlabel="radius (kpc)",
        ylabel="parallel scalar speed / c",
    )
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(zeta_scan["zeta"], zeta_scan["radial_cluster_RMSE_dex"], marker="o")
    ax.axvline(selected_zeta, color="#d95f02", linestyle="--", label="radial selection")
    ax.axvline(0.0, color="black", linestyle=":", label="zero slip")
    ax.set(
        title="Universal metric-slip scan",
        xlabel="zeta",
        ylabel="RX J2129 radial RMSE (dex)",
    )
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    cluster = profiles[profiles["domain"] == "RXJ2129"].sort_values("radius_kpc")
    radius = cluster["radius_kpc"].to_numpy(float)
    ax.loglog(radius, cluster["gbar_m_s2"], color="grey", label="baryons")
    for label, zeta, color in [
        ("conformal", -2.0, "#7570b3"),
        ("zero slip", 0.0, "#1b9e77"),
        (f"selected zeta={selected_zeta:g}", selected_zeta, "#d95f02"),
    ]:
        lensing = sigma_metric_lensing_acceleration(
            cluster["gbar_m_s2"], cluster["gpred_m_s2"], cluster["Sigma"], zeta=zeta
        )
        ax.loglog(radius, lensing, color=color, label=label)
    ax.errorbar(
        target["radius_kpc"],
        np.power(10.0, target["log_gobs"]),
        yerr=(
            np.power(10.0, target["log_gobs"])
            * np.log(10.0)
            * target["err_gobs"]
        ),
        fmt="ko",
        label="radial lens target",
    )
    ax.set(
        title="Dynamics-to-lensing metric closure",
        xlabel="radius (kpc)",
        ylabel="effective acceleration (m/s²)",
    )
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    labels = ["conformal", "zero_slip", "radial_selected", "compact_halo_reference"]
    values = [score_or_inf(raw_summaries[label]) for label in labels[:3]]
    values.append(float(raw_summaries["compact_halo_reference_heldout_RMS_arcsec"]))
    display = [value if np.isfinite(value) else 6.0 for value in values]
    bars = ax.bar(
        [label.replace("_", " ") for label in labels],
        display,
        color=["#7570b3", "#1b9e77", "#d95f02", "#4d4d4d"],
    )
    for bar, value in zip(bars, values):
        label = f"{value:.2f}" if np.isfinite(value) else "failed roots"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="1-arcsec goal")
    ax.tick_params(axis="x", rotation=15)
    ax.set(title="Raw heldout image positions", ylabel="radial RMS (arcsec)")

    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs" / "sigma_covariant_weak_field_protocol.json",
    )
    args = parser.parse_args()
    config_path = args.protocol.resolve()
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_covariant_characteristic_or_metric_slip_scores":
        raise RuntimeError("covariant weak-field protocol was not frozen before scoring")

    profiles = load_candidate_profiles(protocol)
    target = load_cluster_target(protocol)
    characteristics = characteristic_table(profiles, protocol)
    cluster = profiles[profiles["domain"] == "RXJ2129"].sort_values("radius_kpc")
    zeta_scan = scan_metric_slip(cluster, target, protocol)
    selected = zeta_scan.loc[zeta_scan["radial_cluster_RMSE_dex"].idxmin()]
    selected_zeta = float(selected["zeta"])
    raw_predictions, raw_summaries = run_raw_lensing_candidates(
        cluster, selected_zeta, protocol
    )

    raw_reference_path = ROOT / protocol["inputs"]["raw_lensing_reference_report"]
    raw_reference = json.loads(raw_reference_path.read_text(encoding="utf-8"))
    compact_halo_rms = float(
        raw_reference["model_scores"]["GR_plus_cluster_halo"]["heldout"][
            "exact_radial_RMS_arcsec"
        ]
    )
    raw_summaries["compact_halo_reference_heldout_RMS_arcsec"] = compact_halo_rms
    selected_raw_rms = score_or_inf(raw_summaries["radial_selected"])
    zero_slip_raw_rms = score_or_inf(raw_summaries["zero_slip"])

    mathematical = protocol["mathematical_gates"]
    minimum_time_kinetic = float(characteristics["mu_time_kinetic"].min())
    minimum_parallel_gradient = float(
        characteristics["parallel_gradient_coefficient"].min()
    )
    maximum_speed_squared = float(
        characteristics["parallel_speed_squared_over_c2"].max()
    )
    time_kinetic_pass = bool(minimum_time_kinetic > 0.0)
    gradient_pass = bool(minimum_parallel_gradient > 0.0)
    causal_cone_pass = bool(
        maximum_speed_squared
        <= float(mathematical["maximum_characteristic_speed_squared_over_c2"])
    )
    metric_action_derived = False
    mathematical_pass = bool(
        time_kinetic_pass and gradient_pass and causal_cone_pass and metric_action_derived
    )

    observational = protocol["observational_gates"]
    scan_values = list(map(float, protocol["metric_slip_scan"]["zeta"]))
    selected_not_boundary = selected_zeta not in {min(scan_values), max(scan_values)}
    radial_pass = bool(
        float(selected["radial_cluster_RMSE_dex"])
        <= float(observational["radial_cluster_RMSE_dex_max"])
    )
    raw_absolute_pass = bool(
        selected_raw_rms <= float(observational["raw_heldout_RMS_arcsec_max"])
    )
    raw_improvement_pass = bool(selected_raw_rms < zero_slip_raw_rms)
    halo_ratio = selected_raw_rms / compact_halo_rms
    halo_ratio_pass = bool(
        halo_ratio <= float(observational["raw_heldout_RMS_to_compact_halo_ratio_max"])
    )
    observational_pass = bool(
        radial_pass
        and raw_absolute_pass
        and raw_improvement_pass
        and halo_ratio_pass
        and selected_not_boundary
    )

    if observational_pass and not mathematical_pass:
        verdict = (
            "The metric-slip idea is observationally useful in this spent one-cluster diagnostic, "
            "but the naive Lorentz-scalar completion is not a viable relativistic theory. "
            "A dynamical vector or another causal metric sector is required before the "
            "lensing gain has theoretical meaning."
        )
    elif not observational_pass and not mathematical_pass:
        verdict = (
            "The minimal scalar-only completion fails both as a causal relativistic theory "
            "and as a sufficient raw-lensing repair. Do not promote the slip ansatz without "
            "a different covariant mechanism."
        )
    elif observational_pass:
        verdict = (
            "The prototype passes this limited gateway, but still requires derivation of "
            "the physical metric "
            "and independent galaxy-lens and cluster tests."
        )
    else:
        verdict = (
            "The prototype is mathematically acceptable in this audit but does not solve "
            "the lensing data."
        )

    output = ROOT / "results" / "sigma_covariant_weak_field"
    output.mkdir(parents=True, exist_ok=True)
    characteristics.to_csv(output / "characteristics.csv", index=False)
    zeta_scan.to_csv(output / "zeta_scan.csv", index=False)
    raw_predictions.to_csv(output / "raw_lensing_predictions.csv", index=False)
    make_figure(
        characteristics,
        zeta_scan,
        profiles,
        target,
        selected_zeta,
        raw_summaries,
        output / "sigma_covariant_weak_field.png",
    )

    domain_characteristics = {}
    for domain, group in characteristics.groupby("domain", sort=True):
        domain_characteristics[domain] = {
            "minimum_mu_time_kinetic": float(group["mu_time_kinetic"].min()),
            "minimum_parallel_gradient_coefficient": float(
                group["parallel_gradient_coefficient"].min()
            ),
            "maximum_parallel_speed_over_c": float(
                np.sqrt(group["parallel_speed_squared_over_c2"].max())
            ),
            "fraction_of_profile_superluminal_by_more_than_1e_6": float(
                np.mean(group["parallel_speed_squared_over_c2"] > 1.000001)
            ),
        }

    action_reference_path = ROOT / protocol["inputs"]["action_reference_report"]
    report = {
        "report_version": protocol["protocol_version"],
        "status": "completed covariant weak-field gateway test",
        "protocol": {
            "path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(config_path),
            "freeze_status": protocol["status"],
        },
        "candidate": protocol["base_candidate"],
        "covariant_prototype": protocol["covariant_prototype"],
        "mathematical_health": {
            "minimum_mu_time_kinetic": minimum_time_kinetic,
            "minimum_parallel_gradient_coefficient": minimum_parallel_gradient,
            "maximum_parallel_speed_squared_over_c2": maximum_speed_squared,
            "maximum_parallel_speed_over_c": float(np.sqrt(maximum_speed_squared)),
            "canonical_Sigma_speed_over_c": 1.0,
            "positive_time_kinetic_pass": time_kinetic_pass,
            "positive_parallel_gradient_pass": gradient_pass,
            "same_or_narrower_than_metric_light_cone_pass": causal_cone_pass,
            "full_metric_slip_action_derived": metric_action_derived,
            "all_mathematical_gates_pass": mathematical_pass,
            "by_domain": domain_characteristics,
        },
        "radial_metric_slip_selection": {
            "selected_zeta": selected_zeta,
            "selected_zeta_is_scan_boundary": not selected_not_boundary,
            "selected_radial_cluster_RMSE_dex": float(
                selected["radial_cluster_RMSE_dex"]
            ),
            "selected_radial_typical_factor": float(
                10.0 ** float(selected["radial_cluster_RMSE_dex"])
            ),
            "selected_lensing_to_dynamics_at_100kpc": float(
                selected["lensing_to_dynamics_at_100kpc"]
            ),
            "zeta_parameters_selected_from_raw_images": 0,
            "zeta_parameters_selected_from_RXJ2129_radial_lensing": 1,
        },
        "raw_lensing": raw_summaries,
        "comparisons": {
            "selected_vs_zero_slip_heldout_RMS_ratio": selected_raw_rms
            / zero_slip_raw_rms,
            "selected_vs_compact_halo_heldout_RMS_ratio": halo_ratio,
            "gravity_or_lensing_amplitudes_fit_to_raw_images": 0,
        },
        "gate_audit": {
            "radial_cluster_RMSE_pass": radial_pass,
            "raw_heldout_below_1_arcsec_pass": raw_absolute_pass,
            "raw_heldout_improves_zero_slip_pass": raw_improvement_pass,
            "raw_heldout_within_compact_halo_ratio_pass": halo_ratio_pass,
            "selected_zeta_not_scan_boundary_pass": selected_not_boundary,
            "all_observational_gates_pass": observational_pass,
            "all_mathematical_and_observational_gates_pass": bool(
                mathematical_pass and observational_pass
            ),
        },
        "concepts_learned": {
            "verdict": verdict,
            "time": (
                "A canonical Box(Sigma) term supplies ordinary time propagation for Sigma, "
                "but the nonlinear AQUAL sector determines a separate effective causal cone."
            ),
            "curvature": (
                "The zeta closure explicitly separates the time potential Phi governing stars "
                "from the spatial potential Psi needed for lensing."
            ),
            "next_theory_requirement": (
                "Derive the slip and causal cone from a scalar-vector-tensor or generalized "
                "Einstein-Aether action; do not treat zeta as a final empirical multiplier."
            ),
        },
        "scope": protocol["scope"],
        "primary_sources": protocol["primary_sources"],
        "input_hashes": {
            "candidate_profile": sha256(ROOT / protocol["base_candidate"]["profile"]),
            "cluster_radial_target": sha256(
                ROOT / protocol["inputs"]["cluster_radial_target"]
            ),
            "raw_lensing_reference_report": sha256(raw_reference_path),
            "action_reference_report": sha256(action_reference_path),
        },
        "outputs": protocol["outputs"],
    }
    (output / "report.json").write_text(
        json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_safe(report["mathematical_health"]), indent=2), flush=True)
    print(json.dumps(json_safe(report["radial_metric_slip_selection"]), indent=2), flush=True)
    print(json.dumps(json_safe(report["gate_audit"]), indent=2), flush=True)
    print(verdict, flush=True)


if __name__ == "__main__":
    main()
