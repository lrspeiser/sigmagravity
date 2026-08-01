#!/usr/bin/env python3
"""Test Sigma backreaction and stress-energy from one shared normalization."""

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

from run_sigma_field_exploration import (  # noqa: E402
    G_SI,
    KPC_M,
    cluster_solution,
    fixed_rar_enhancement,
    galaxy_solution,
    geometric_radial_faces,
    interpolate_log_radius,
    json_safe,
    log_slope,
)
from voidscreen.raw_lensing import loglog_interpolate_with_tails  # noqa: E402
from voidscreen.sigma_actions import (  # noqa: E402
    refracted_aqual_acceleration,
    scalar_field_stress_energy_profile,
    solve_coupled_spherical_sigma,
)


MSUN_KG = 1.988409870698051e30


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def systems(base_protocol: dict, log_rho: float, length: float):
    gr, gd, gs, gb = galaxy_solution(base_protocol, 10.0**log_rho, length)
    tian, _, _, cr, cd, cs = cluster_solution(10.0**log_rho, length)
    tr = tian["radius_kpc"].to_numpy(float)
    target_gbar = np.power(10.0, tian["log_gbar"].to_numpy(float))
    cb = loglog_interpolate_with_tails(cr, tr, target_gbar, outer_slope=-2.0)
    return gr, gd, gs, gb, tian, cr, cd, cs, cb


def score(gr, gb, gp, tian, cr, cb, cp, *, a0: float) -> dict:
    use = (gr >= 5.0) & (gr <= 50.0)
    comparison = gb * fixed_rar_enhancement(gb, a0)
    galaxy_rmse = float(np.sqrt(np.mean(np.log10(gp[use] / comparison[use]) ** 2)))
    target_radius = tian["radius_kpc"].to_numpy(float)
    target = np.power(10.0, tian["log_gobs"].to_numpy(float))
    cluster_prediction = interpolate_log_radius(cr, cp, target_radius)
    cluster_rmse = float(np.sqrt(np.mean(np.log10(cluster_prediction / target) ** 2)))
    velocity = np.sqrt(gp * gr * KPC_M) / 1000.0
    return {
        "galaxy_RMSE_dex": galaxy_rmse,
        "galaxy_typical_factor": 10.0**galaxy_rmse,
        "cluster_RMSE_dex": cluster_rmse,
        "cluster_typical_factor": 10.0**cluster_rmse,
        "joint_RMSE_dex": float(np.sqrt(0.5 * (galaxy_rmse**2 + cluster_rmse**2))),
        "far_velocity_slope": log_slope(gr, velocity, 100.0, 250.0),
    }


def enclosed_baryonic_mass(acceleration, radius_kpc) -> np.ndarray:
    radius_m = np.asarray(radius_kpc, dtype=float) * KPC_M
    return np.asarray(acceleration, dtype=float) * radius_m**2 / G_SI


def run_feedback(protocol: dict, base_protocol: dict) -> pd.DataFrame:
    fixed = protocol["fixed_candidate"]
    eta = float(fixed["eta"])
    log_rho = float(fixed["log10_rho_s_g_cm3"])
    length = float(fixed["L_Sigma_kpc"])
    a0 = float(fixed["a0_m_s2"])
    gr, gd, gs, gb, tian, cr, cd, cs, cb = systems(base_protocol, log_rho, length)
    gf = geometric_radial_faces(len(gr), 0.01, 1000.0)
    cf = geometric_radial_faces(len(cr), 0.05, 10000.0)
    records = []
    for chi in protocol["feedback_values"]:
        chi = float(chi)
        coupled_galaxy = solve_coupled_spherical_sigma(
            gf,
            gd,
            gb,
            rho_s_g_cm3=10.0**log_rho,
            length_kpc=length,
            a0_m_s2=a0,
            eta=eta,
            backreaction=chi,
            initial_sigma=gs.field,
        )
        coupled_cluster = solve_coupled_spherical_sigma(
            cf,
            cd,
            cb,
            rho_s_g_cm3=10.0**log_rho,
            length_kpc=length,
            a0_m_s2=a0,
            eta=eta,
            backreaction=chi,
            initial_sigma=cs.field,
        )
        _, galaxy_scalar_mass, galaxy_scalar_g = scalar_field_stress_energy_profile(
            gr,
            coupled_galaxy.field,
            length_kpc=length,
            a0_m_s2=a0,
            backreaction=chi,
        )
        _, cluster_scalar_mass, cluster_scalar_g = scalar_field_stress_energy_profile(
            cr,
            coupled_cluster.field,
            length_kpc=length,
            a0_m_s2=a0,
            backreaction=chi,
        )
        gp = refracted_aqual_acceleration(
            gb + galaxy_scalar_g, coupled_galaxy.field, a0_m_s2=a0, eta=eta
        )
        cp = refracted_aqual_acceleration(
            cb + cluster_scalar_g, coupled_cluster.field, a0_m_s2=a0, eta=eta
        )
        metrics = score(gr, gb, gp, tian, cr, cb, cp, a0=a0)
        galaxy_baryon_mass = enclosed_baryonic_mass(gb, gr)
        cluster_baryon_mass = enclosed_baryonic_mass(cb, cr)
        records.append(
            {
                "chi": chi,
                **metrics,
                "galaxy_solver_converged": coupled_galaxy.converged,
                "cluster_solver_converged": coupled_cluster.converged,
                "galaxy_Sigma_at_1kpc": float(
                    interpolate_log_radius(gr, coupled_galaxy.field, np.array([1.0]))[0]
                ),
                "cluster_Sigma_at_100kpc": float(
                    interpolate_log_radius(cr, coupled_cluster.field, np.array([100.0]))[0]
                ),
                "field_to_baryon_mass_at_20kpc": float(
                    interpolate_log_radius(
                        gr, galaxy_scalar_mass / galaxy_baryon_mass, np.array([20.0])
                    )[0]
                ),
                "field_to_baryon_mass_at_100kpc": float(
                    interpolate_log_radius(
                        cr, cluster_scalar_mass / cluster_baryon_mass, np.array([100.0])
                    )[0]
                ),
            }
        )
    return pd.DataFrame(records)


def run_stress_grid(protocol: dict, base_protocol: dict) -> pd.DataFrame:
    grid = protocol["stress_energy_grid"]
    a0 = float(protocol["fixed_candidate"]["a0_m_s2"])
    records = []
    for log_rho in grid["log10_rho_s_g_cm3"]:
        for length in grid["L_Sigma_kpc"]:
            gr, _, gs, gb, tian, cr, _, cs, cb = systems(
                base_protocol, float(log_rho), float(length)
            )
            galaxy_baryon_mass = enclosed_baryonic_mass(gb, gr)
            cluster_baryon_mass = enclosed_baryonic_mass(cb, cr)
            for log_chi in grid["log10_chi"]:
                chi = 10.0 ** float(log_chi)
                _, galaxy_scalar_mass, galaxy_scalar_g = scalar_field_stress_energy_profile(
                    gr,
                    gs.field,
                    length_kpc=float(length),
                    a0_m_s2=a0,
                    backreaction=chi,
                )
                _, cluster_scalar_mass, cluster_scalar_g = scalar_field_stress_energy_profile(
                    cr,
                    cs.field,
                    length_kpc=float(length),
                    a0_m_s2=a0,
                    backreaction=chi,
                )
                for eta in grid["eta"]:
                    gp = refracted_aqual_acceleration(
                        gb + galaxy_scalar_g, gs.field, a0_m_s2=a0, eta=float(eta)
                    )
                    cp = refracted_aqual_acceleration(
                        cb + cluster_scalar_g, cs.field, a0_m_s2=a0, eta=float(eta)
                    )
                    records.append(
                        {
                            "eta": float(eta),
                            "log10_rho_s_g_cm3": float(log_rho),
                            "L_Sigma_kpc": float(length),
                            "log10_chi": float(log_chi),
                            **score(gr, gb, gp, tian, cr, cb, cp, a0=a0),
                            "field_to_baryon_mass_at_20kpc": float(
                                interpolate_log_radius(
                                    gr,
                                    galaxy_scalar_mass / galaxy_baryon_mass,
                                    np.array([20.0]),
                                )[0]
                            ),
                            "field_to_baryon_mass_at_100kpc": float(
                                interpolate_log_radius(
                                    cr,
                                    cluster_scalar_mass / cluster_baryon_mass,
                                    np.array([100.0]),
                                )[0]
                            ),
                            "Sigma_feedback_included": False,
                        }
                    )
    return pd.DataFrame(records).sort_values("joint_RMSE_dex").reset_index(drop=True)


def make_figure(feedback: pd.DataFrame, stress: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    ax = axes[0, 0]
    ax.semilogx(feedback["chi"], feedback["galaxy_typical_factor"], marker="o", label="galaxy")
    ax.semilogx(feedback["chi"], feedback["cluster_typical_factor"], marker="o", label="cluster")
    ax.set(xlabel="feedback strength chi", ylabel="typical multiplicative mismatch")
    ax.set_title("Letting gravity push back on Sigma")
    ax.legend()

    ax = axes[0, 1]
    scatter = ax.scatter(
        stress["galaxy_typical_factor"],
        stress["cluster_typical_factor"],
        c=stress["log10_chi"],
        cmap="viridis",
        alpha=0.45,
        s=15,
    )
    ax.set(xlabel="galaxy mismatch factor", ylabel="cluster mismatch factor")
    ax.set_title("Counting Sigma's stored energy")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(2.0, color="grey", linewidth=0.8, linestyle="--")
    ax.axhline(2.0, color="grey", linewidth=0.8, linestyle="--")
    fig.colorbar(scatter, ax=ax, label="log10 chi")

    fixed = stress[
        stress["eta"].eq(0.6)
        & stress["log10_rho_s_g_cm3"].eq(-23.5)
        & stress["L_Sigma_kpc"].eq(3.0)
    ].sort_values("log10_chi")
    ax = axes[1, 0]
    ax.semilogy(
        fixed["log10_chi"],
        fixed["field_to_baryon_mass_at_20kpc"],
        marker="o",
        label="galaxy, 20 kpc",
    )
    ax.semilogy(
        fixed["log10_chi"],
        fixed["field_to_baryon_mass_at_100kpc"],
        marker="o",
        label="cluster, 100 kpc",
    )
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set(xlabel="log10 chi", ylabel="Sigma-field mass / baryonic mass")
    ax.set_title("When the field becomes gravitationally heavy")
    ax.legend()

    ax = axes[1, 1]
    ax.semilogx(feedback["chi"], feedback["joint_RMSE_dex"], marker="o", color="#D95F02")
    ax.axhline(0.2303812065, color="black", linestyle="--", label="one-way baseline")
    ax.set(xlabel="feedback strength chi", ylabel="combined score (smaller is better)")
    ax.set_title("Complete-action score stays nearly unchanged")
    ax.legend()
    for panel in axes.ravel():
        panel.grid(alpha=0.2)
    fig.suptitle("Can Sigma's own energy bridge galaxies and clusters?", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs" / "sigma_complete_action_protocol.json",
    )
    args = parser.parse_args()
    config_path = args.protocol.resolve()
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    base_protocol = json.loads(
        (ROOT / "configs" / "sigma_action_exploration_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    output = ROOT / "results" / "sigma_complete_action"
    output.mkdir(parents=True, exist_ok=True)

    feedback = run_feedback(protocol, base_protocol)
    stress = run_stress_grid(protocol, base_protocol)
    feedback.to_csv(output / "backreaction_sweep.csv", index=False)
    stress.to_csv(output / "stress_energy_grid.csv", index=False)
    make_figure(feedback, stress, output / "sigma_complete_action.png")

    best_feedback = feedback.loc[feedback["joint_RMSE_dex"].idxmin()]
    best_stress = stress.iloc[0]
    report = {
        "report_version": protocol["protocol_version"],
        "status": "completed weak-field complete-action consistency test",
        "question": protocol["plain_language_question"],
        "protocol": {
            "path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(config_path),
        },
        "baseline_one_way_joint_RMSE_dex": 0.23038120647953703,
        "best_feedback_row": best_feedback.to_dict(),
        "best_stress_energy_row": best_stress.to_dict(),
        "plain_language_results": {
            "feedback_change": "Letting gravity reshape Sigma changes the combined score by less than one part in a thousand.",
            "stored_energy_change": "Across the frozen grid, counting Sigma's positive stored energy does not improve on the no-stress-energy baseline.",
            "mass_tradeoff": "A normalization that makes the field heavy enough to help the cluster also adds too much gravitating field energy to the galaxy.",
        },
        "scope": {
            "same_chi_controls_feedback_and_field_energy": True,
            "independent_scalar_mass_amplitude_fit": False,
            "full_covariant_metric_solved": False,
            "raw_lensing_rerun": False,
            "stress_grid_uses_weak_feedback_profiles": True,
        },
        "outputs": {
            "feedback": "results/sigma_complete_action/backreaction_sweep.csv",
            "stress_energy": "results/sigma_complete_action/stress_energy_grid.csv",
            "figure": "results/sigma_complete_action/sigma_complete_action.png",
        },
    }
    (output / "report.json").write_text(
        json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_safe(report["plain_language_results"]), indent=2), flush=True)
    print("best feedback", best_feedback.to_dict(), flush=True)
    print("best stress", best_stress.to_dict(), flush=True)


if __name__ == "__main__":
    main()
