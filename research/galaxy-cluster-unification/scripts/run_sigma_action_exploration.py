#!/usr/bin/env python3
"""Compare compact action-motivated force laws for the screened Sigma field."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from run_sigma_field_exploration import (  # noqa: E402
    KPC_M,
    cluster_solution,
    fixed_rar_enhancement,
    galaxy_solution,
    interpolate_log_radius,
    json_safe,
    log_slope,
    run_diagnostic_lensing,
)
from voidscreen.raw_lensing import loglog_interpolate_with_tails  # noqa: E402
from voidscreen.sigma_actions import (  # noqa: E402
    conformal_symmetron_acceleration,
    refracted_aqual_acceleration,
)


COLORS = {
    "conformal_symmetron": "#7570B3",
    "sigma_gated_AQUAL": "#1B9E77",
    "sigma_refracted_AQUAL": "#D95F02",
}
LABELS = {
    "conformal_symmetron": "conformal symmetron",
    "sigma_gated_AQUAL": "Sigma-gated AQUAL",
    "sigma_refracted_AQUAL": "Sigma-refracted AQUAL",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predictions_for_row(row: pd.Series, protocol: dict, galaxy_cache, cluster_cache):
    key = (float(row["log10_rho_s_g_cm3"]), float(row["L_Sigma_kpc"]))
    galaxy_radius, _, galaxy_sigma, galaxy_gbar = galaxy_cache[key]
    tian, _, _, cluster_radius, _, cluster_sigma = cluster_cache[key]
    target_radius = tian["radius_kpc"].to_numpy(float)
    target_gbar = np.power(10.0, tian["log_gbar"].to_numpy(float))
    cluster_gbar = loglog_interpolate_with_tails(
        cluster_radius, target_radius, target_gbar, outer_slope=-2.0
    )
    model = row["model"]
    if model == "conformal_symmetron":
        galaxy_prediction = conformal_symmetron_acceleration(
            galaxy_gbar,
            galaxy_radius,
            galaxy_sigma.field,
            alpha=10.0 ** float(row["log10_alpha"]),
        )
        cluster_prediction = conformal_symmetron_acceleration(
            cluster_gbar,
            cluster_radius,
            cluster_sigma.field,
            alpha=10.0 ** float(row["log10_alpha"]),
        )
    elif model == "sigma_gated_AQUAL":
        galaxy_prediction = refracted_aqual_acceleration(
            galaxy_gbar,
            galaxy_sigma.field,
            a0_m_s2=float(protocol["fixed_constants"]["a0_m_s2"]),
            activation=float(row["activation_lambda"]),
        )
        cluster_prediction = refracted_aqual_acceleration(
            cluster_gbar,
            cluster_sigma.field,
            a0_m_s2=float(protocol["fixed_constants"]["a0_m_s2"]),
            activation=float(row["activation_lambda"]),
        )
    elif model == "sigma_refracted_AQUAL":
        galaxy_prediction = refracted_aqual_acceleration(
            galaxy_gbar,
            galaxy_sigma.field,
            a0_m_s2=float(protocol["fixed_constants"]["a0_m_s2"]),
            eta=float(row["eta"]),
        )
        cluster_prediction = refracted_aqual_acceleration(
            cluster_gbar,
            cluster_sigma.field,
            a0_m_s2=float(protocol["fixed_constants"]["a0_m_s2"]),
            eta=float(row["eta"]),
        )
    else:
        raise ValueError(f"unknown model {model}")
    return (
        galaxy_radius,
        galaxy_sigma.field,
        galaxy_gbar,
        galaxy_prediction,
        tian,
        cluster_radius,
        cluster_sigma.field,
        cluster_gbar,
        cluster_prediction,
    )


def score_predictions(values, protocol: dict) -> dict:
    (
        galaxy_radius,
        _,
        galaxy_gbar,
        galaxy_prediction,
        tian,
        cluster_radius,
        _,
        cluster_gbar,
        cluster_prediction,
    ) = values
    settings = protocol["tests"]["spherical_galaxy"]
    low, high = map(float, settings["score_radii_kpc"])
    far_low, far_high = map(float, settings["far_slope_radii_kpc"])
    use = (galaxy_radius >= low) & (galaxy_radius <= high)
    a0 = float(protocol["fixed_constants"]["a0_m_s2"])
    comparison = galaxy_gbar * fixed_rar_enhancement(galaxy_gbar, a0)
    galaxy_rmse = float(
        np.sqrt(np.mean(np.log10(galaxy_prediction[use] / comparison[use]) ** 2))
    )
    speed = np.sqrt(galaxy_prediction * galaxy_radius * KPC_M) / 1000.0
    target_radius = tian["radius_kpc"].to_numpy(float)
    target_prediction = interpolate_log_radius(
        cluster_radius, cluster_prediction, target_radius
    )
    target_observed = np.power(10.0, tian["log_gobs"].to_numpy(float))
    residual = np.log10(target_prediction / target_observed)
    cluster_rmse = float(np.sqrt(np.mean(residual**2)))
    return {
        "galaxy_RAR_RMSE_dex_5_50kpc": galaxy_rmse,
        "galaxy_velocity_log_slope_10_50kpc": log_slope(
            galaxy_radius, speed, 10.0, 50.0
        ),
        "galaxy_velocity_log_slope_100_250kpc": log_slope(
            galaxy_radius, speed, far_low, far_high
        ),
        "galaxy_enhancement_at_20kpc": float(
            interpolate_log_radius(
                galaxy_radius, galaxy_prediction / galaxy_gbar, np.array([20.0])
            )[0]
        ),
        "RXJ2129_derived_field_RMSE_dex": cluster_rmse,
        "RXJ2129_mean_log_residual_dex": float(np.mean(residual)),
        "RXJ2129_enhancement_at_100kpc": float(
            interpolate_log_radius(
                cluster_radius, cluster_prediction / cluster_gbar, np.array([100.0])
            )[0]
        ),
        "joint_descriptive_score_dex": float(
            np.sqrt(0.5 * (galaxy_rmse**2 + cluster_rmse**2))
        ),
    }


def run_grids(protocol: dict):
    environment = protocol["environment_grid"]
    galaxy_cache = {}
    cluster_cache = {}
    records = []
    for log_rho in environment["log10_rho_s_g_cm3"]:
        for length in environment["L_Sigma_kpc"]:
            key = (float(log_rho), float(length))
            galaxy_cache[key] = galaxy_solution(protocol, 10.0 ** float(log_rho), float(length))
            cluster_cache[key] = cluster_solution(10.0 ** float(log_rho), float(length))
            common = {
                "log10_rho_s_g_cm3": float(log_rho),
                "L_Sigma_kpc": float(length),
            }
            for log_alpha in protocol["models"]["conformal_symmetron"]["log10_alpha"]:
                row = pd.Series(
                    {
                        **common,
                        "model": "conformal_symmetron",
                        "log10_alpha": float(log_alpha),
                        "activation_lambda": math.nan,
                        "eta": math.nan,
                    }
                )
                values = predictions_for_row(row, protocol, galaxy_cache, cluster_cache)
                if np.all(values[3] > 0.0) and np.all(values[8] > 0.0):
                    records.append({**row.to_dict(), **score_predictions(values, protocol)})
            for activation in protocol["models"]["sigma_gated_AQUAL"]["lambda"]:
                row = pd.Series(
                    {
                        **common,
                        "model": "sigma_gated_AQUAL",
                        "log10_alpha": math.nan,
                        "activation_lambda": float(activation),
                        "eta": 0.0,
                    }
                )
                values = predictions_for_row(row, protocol, galaxy_cache, cluster_cache)
                records.append({**row.to_dict(), **score_predictions(values, protocol)})
            for eta in protocol["models"]["sigma_refracted_AQUAL"]["eta"]:
                row = pd.Series(
                    {
                        **common,
                        "model": "sigma_refracted_AQUAL",
                        "log10_alpha": math.nan,
                        "activation_lambda": 1.0,
                        "eta": float(eta),
                    }
                )
                values = predictions_for_row(row, protocol, galaxy_cache, cluster_cache)
                records.append({**row.to_dict(), **score_predictions(values, protocol)})
    table = pd.DataFrame(records).sort_values(
        ["model", "joint_descriptive_score_dex"]
    ).reset_index(drop=True)
    return table, galaxy_cache, cluster_cache


def best_rows(table: pd.DataFrame) -> pd.DataFrame:
    indices = table.groupby("model")["joint_descriptive_score_dex"].idxmin()
    return table.loc[indices].sort_values("joint_descriptive_score_dex").reset_index(drop=True)


def build_profiles(best: pd.DataFrame, protocol: dict, galaxy_cache, cluster_cache):
    frames = []
    a0 = float(protocol["fixed_constants"]["a0_m_s2"])
    for row in best.itertuples(index=False):
        series = pd.Series(row._asdict())
        values = predictions_for_row(series, protocol, galaxy_cache, cluster_cache)
        gr, gs, gb, gp, tian, cr, cs, cb, cp = values
        frames.append(
            pd.DataFrame(
                {
                    "model": row.model,
                    "domain": "galaxy_archetype",
                    "radius_kpc": gr,
                    "Sigma": gs,
                    "gbar_m_s2": gb,
                    "gpred_m_s2": gp,
                    "comparison_g_m_s2": gb * fixed_rar_enhancement(gb, a0),
                }
            )
        )
        frames.append(
            pd.DataFrame(
                {
                    "model": row.model,
                    "domain": "RXJ2129",
                    "radius_kpc": cr,
                    "Sigma": cs,
                    "gbar_m_s2": cb,
                    "gpred_m_s2": cp,
                    "comparison_g_m_s2": loglog_interpolate_with_tails(
                        cr,
                        tian["radius_kpc"].to_numpy(float),
                        np.power(10.0, tian["log_gobs"].to_numpy(float)),
                    ),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def make_figure(
    table: pd.DataFrame,
    best: pd.DataFrame,
    profiles: pd.DataFrame,
    lens_predictions: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), constrained_layout=True)
    galaxy = profiles[profiles["domain"] == "galaxy_archetype"]
    first = galaxy[galaxy["model"] == best.iloc[0]["model"]]
    radius = first["radius_kpc"].to_numpy(float)
    axes[0, 0].semilogx(
        radius,
        np.sqrt(first["gbar_m_s2"] * radius * KPC_M) / 1000.0,
        color="grey",
        label="Newtonian baryons",
    )
    axes[0, 0].semilogx(
        radius,
        np.sqrt(first["comparison_g_m_s2"] * radius * KPC_M) / 1000.0,
        color="black",
        linestyle="--",
        label="RAR comparison",
    )
    for model, group in galaxy.groupby("model"):
        radius = group["radius_kpc"].to_numpy(float)
        speed = np.sqrt(group["gpred_m_s2"] * radius * KPC_M) / 1000.0
        axes[0, 0].semilogx(radius, speed, color=COLORS[model], label=LABELS[model])
    axes[0, 0].set(xlim=(1, 300), xlabel="radius (kpc)", ylabel="circular speed (km/s)")
    axes[0, 0].set_title("Galaxy: best universal row per action")
    axes[0, 0].legend(fontsize=7)

    cluster = profiles[profiles["domain"] == "RXJ2129"]
    first = cluster[cluster["model"] == best.iloc[0]["model"]]
    axes[0, 1].loglog(first["radius_kpc"], first["gbar_m_s2"], color="grey", label="baryons")
    axes[0, 1].loglog(
        first["radius_kpc"], first["comparison_g_m_s2"], color="black", linestyle="--", label="derived target"
    )
    for model, group in cluster.groupby("model"):
        axes[0, 1].loglog(group["radius_kpc"], group["gpred_m_s2"], color=COLORS[model], label=LABELS[model])
    axes[0, 1].set(xlim=(10, 1000), ylim=(2e-13, 8e-10), xlabel="radius (kpc)", ylabel="acceleration (m/s²)")
    axes[0, 1].set_title("RX J2129 transfer")
    axes[0, 1].legend(fontsize=7)

    for model, group in table.groupby("model"):
        axes[0, 2].scatter(
            group["galaxy_RAR_RMSE_dex_5_50kpc"],
            group["RXJ2129_derived_field_RMSE_dex"],
            s=12,
            alpha=0.35,
            color=COLORS[model],
            label=LABELS[model],
        )
    for row in best.itertuples(index=False):
        axes[0, 2].scatter(
            row.galaxy_RAR_RMSE_dex_5_50kpc,
            row.RXJ2129_derived_field_RMSE_dex,
            s=75,
            marker="*",
            color=COLORS[row.model],
            edgecolor="black",
        )
    axes[0, 2].set(xlabel="galaxy RMSE (dex)", ylabel="cluster RMSE (dex)")
    axes[0, 2].set_title("Universal-setting tradeoff")
    axes[0, 2].legend(fontsize=7)

    for model, group in table.groupby("model"):
        axes[1, 0].scatter(
            group["galaxy_velocity_log_slope_100_250kpc"],
            group["joint_descriptive_score_dex"],
            s=12,
            alpha=0.35,
            color=COLORS[model],
            label=LABELS[model],
        )
    axes[1, 0].axvline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set(xlabel="far velocity slope", ylabel="joint descriptive RMSE (dex)")
    axes[1, 0].set_title("Flatness versus galaxy/cluster closeness")

    for model, group in galaxy.groupby("model"):
        axes[1, 1].semilogx(group["radius_kpc"], group["Sigma"], color=COLORS[model], label=LABELS[model])
    axes[1, 1].set(xlim=(0.1, 300), ylim=(-0.02, 1.02), xlabel="radius (kpc)", ylabel="Sigma")
    axes[1, 1].set_title("Environmental field chosen by each action")
    axes[1, 1].legend(fontsize=7)

    heldout = lens_predictions[
        lens_predictions["stage"].eq("heldout")
        & lens_predictions["candidate_selection"].eq("joint")
    ]
    axes[1, 2].scatter(heldout["observed_x_arcsec"], heldout["observed_y_arcsec"], color="black", label="observed")
    converged = heldout["root_converged"].astype(bool)
    axes[1, 2].scatter(
        heldout.loc[converged, "predicted_x_arcsec"],
        heldout.loc[converged, "predicted_y_arcsec"],
        marker="x",
        color=COLORS["sigma_refracted_AQUAL"],
        label="refracted AQUAL",
    )
    for row in heldout[converged].itertuples(index=False):
        axes[1, 2].plot(
            [row.observed_x_arcsec, row.predicted_x_arcsec],
            [row.observed_y_arcsec, row.predicted_y_arcsec],
            color=COLORS["sigma_refracted_AQUAL"],
            alpha=0.5,
        )
    axes[1, 2].set_aspect("equal")
    axes[1, 2].set(xlabel="east offset (arcsec)", ylabel="north offset (arcsec)")
    axes[1, 2].set_title("TeVeS-inspired zero-slip diagnostic")
    axes[1, 2].legend(fontsize=7)

    fig.suptitle("Action-motivated screened Sigma force laws", fontsize=15)
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
        default=ROOT / "configs" / "sigma_action_exploration_protocol.json",
    )
    args = parser.parse_args()
    config_path = args.protocol.resolve()
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / "results" / "sigma_action_exploration"
    output.mkdir(parents=True, exist_ok=True)

    table, galaxy_cache, cluster_cache = run_grids(protocol)
    best = best_rows(table)
    print(best.to_string(index=False), flush=True)
    profiles = build_profiles(best, protocol, galaxy_cache, cluster_cache)

    lens_profile = profiles[
        profiles["model"].eq("sigma_refracted_AQUAL")
        & profiles["domain"].eq("RXJ2129")
    ].copy()
    lens_profile["gSigma_m_s2"] = lens_profile["gpred_m_s2"]
    lens_profile["domain"] = "RXJ2129"
    joint_lens_predictions, joint_lens_summary = run_diagnostic_lensing(
        best[best["model"] == "sigma_refracted_AQUAL"].iloc[0],
        protocol,
        lens_profile,
    )
    joint_lens_predictions["candidate_selection"] = "joint"

    refracted_rows = table[table["model"] == "sigma_refracted_AQUAL"]
    cluster_best = refracted_rows.loc[
        refracted_rows["RXJ2129_derived_field_RMSE_dex"].idxmin()
    ]
    cluster_best_profile = build_profiles(
        pd.DataFrame([cluster_best]), protocol, galaxy_cache, cluster_cache
    )
    cluster_lens_profile = cluster_best_profile[
        cluster_best_profile["domain"] == "RXJ2129"
    ].copy()
    cluster_lens_profile["gSigma_m_s2"] = cluster_lens_profile["gpred_m_s2"]
    cluster_lens_predictions, cluster_lens_summary = run_diagnostic_lensing(
        cluster_best, protocol, cluster_lens_profile
    )
    cluster_lens_predictions["candidate_selection"] = "cluster_target_only"
    lens_predictions = pd.concat(
        [joint_lens_predictions, cluster_lens_predictions], ignore_index=True
    )
    print("joint lensing", joint_lens_summary, flush=True)
    print("cluster-target-only lensing", cluster_lens_summary, flush=True)

    table.to_csv(output / "parameter_grid.csv", index=False)
    best.to_csv(output / "best_rows.csv", index=False)
    profiles.to_csv(output / "radial_profiles.csv", index=False)
    lens_predictions.to_csv(output / "raw_lensing_predictions.csv", index=False)
    make_figure(
        table,
        best,
        profiles,
        lens_predictions,
        output / "sigma_action_exploration.png",
    )

    earlier = json.loads(
        (ROOT / "results" / "sigma_field_exploration" / "report.json").read_text(
            encoding="utf-8"
        )
    )["best_descriptive_grid_row"]
    report = {
        "report_version": protocol["protocol_version"],
        "status": "completed action-motivated exploratory comparison",
        "protocol": {
            "path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(config_path),
            "freeze_status": protocol["status"],
        },
        "design_rule": protocol["design_rule"],
        "model_rows": table.groupby("model").size().to_dict(),
        "best_joint_rows": {
            row.model: row._asdict() for row in best.itertuples(index=False)
        },
        "earlier_bounded_permittivity_reference": earlier,
        "raw_lensing_diagnostics": {
            "joint_row": {
                "parameter_row": best[
                    best["model"] == "sigma_refracted_AQUAL"
                ].iloc[0].to_dict(),
                "scores": joint_lens_summary,
            },
            "cluster_derived_target_only_row": {
                "parameter_row": cluster_best.to_dict(),
                "scores": cluster_lens_summary,
                "warning": "Included to diagnose parameter tension; not a universal galaxy/cluster solution.",
            },
        },
        "interpretation": {
            "symmetron_identification": "The Sigma equation matches the normalized static symmetron effective-potential equation.",
            "conformal_symmetron_photon_result": "A conformal matter coupling changes massive-particle motion but not null paths directly; it is not assigned enhanced lensing.",
            "AQUAL_action": "Both AQUAL rows use a mu function that is the X derivative of an explicit free function.",
            "weak_backreaction_only": True,
            "covariant_completion": False,
            "independent_cluster_validation": False,
            "gravity_or_lensing_amplitude_fit_to_images": False,
        },
        "primary_sources": protocol["primary_sources"],
        "outputs": {
            "grid": "results/sigma_action_exploration/parameter_grid.csv",
            "best_rows": "results/sigma_action_exploration/best_rows.csv",
            "profiles": "results/sigma_action_exploration/radial_profiles.csv",
            "raw_lensing": "results/sigma_action_exploration/raw_lensing_predictions.csv",
            "figure": "results/sigma_action_exploration/sigma_action_exploration.png",
        },
    }
    (output / "report.json").write_text(
        json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
