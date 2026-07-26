"""Generate the evidence-aligned figures for the Frontiers resubmission.

The script only reads frozen, machine-readable results already produced by the
revision analyses.  It does not refit any model or alter production outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent.parent / "figures"

SPARC = ROOT / "research" / "sparc_statistical_validation" / "results"
AUDIT = ROOT / "research" / "reviewer_derivation_audit" / "results"
FOX = ROOT / "data" / "clusters" / "fox2022_sigma_results.csv"


BLUE = "#0072B2"
ORANGE = "#E69F00"
PURPLE = "#6A3D9A"
SKY = "#56B4E9"
BLACK = "#222222"
GRAY = "#777777"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "savefig.dpi": 360,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_1_sparc() -> None:
    galaxies = pd.read_csv(SPARC / "per_galaxy_primary.csv")
    nuisance = pd.read_csv(SPARC / "nuisance_grid.csv")
    summary = json.loads((SPARC / "bootstrap_summary.json").read_text())
    delta = galaxies["delta_sigma_minus_mond_kms"].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(11.7, 3.6))

    markers = {1: "o", 2: "s", 3: "^"}
    colors = {1: BLUE, 2: ORANGE, 3: PURPLE}
    for quality in sorted(galaxies["quality"].unique()):
        subset = galaxies[galaxies["quality"] == quality]
        axes[0].scatter(
            subset["rms_mond_kms"],
            subset["rms_sigma_kms"],
            s=23,
            marker=markers[int(quality)],
            color=colors[int(quality)],
            alpha=0.78,
            edgecolor="white",
            linewidth=0.35,
            label=f"SPARC quality {int(quality)}",
        )
    limit = 1.05 * max(galaxies["rms_mond_kms"].max(), galaxies["rms_sigma_kms"].max())
    axes[0].plot([0, limit], [0, limit], "--", color=BLACK, linewidth=1.2)
    axes[0].set(xlabel="MOND RMS [km s$^{-1}$]", ylabel=r"$\Sigma$-Gravity RMS [km s$^{-1}$]")
    axes[0].legend(frameon=False, loc="upper left")

    axes[1].hist(delta, bins=22, color=BLUE, alpha=0.78, edgecolor="white")
    mean_delta = summary["sigma_minus_mond"]["mean"]
    axes[1].axvline(0, linestyle="--", color=BLACK, linewidth=1.2, label="equal RMS")
    axes[1].axvline(mean_delta, color=ORANGE, linewidth=2.3, label=f"mean = {mean_delta:.3f}")
    axes[1].set(
        xlabel=r"$\Sigma$ RMS $-$ MOND RMS [km s$^{-1}$]",
        ylabel="Galaxies",
    )
    axes[1].legend(frameon=False)

    nuisance_delta = nuisance["mean_delta_sigma_minus_mond_kms"].to_numpy()
    axes[2].hist(nuisance_delta, bins=14, color=ORANGE, alpha=0.78, edgecolor="white")
    axes[2].axvline(0, linestyle="--", color=BLACK, linewidth=1.2)
    axes[2].axvline(mean_delta, color=BLUE, linewidth=2.3, label="locked analysis")
    axes[2].set(
        xlabel="Mean paired contrast [km s$^{-1}$]",
        ylabel="Frozen nuisance configurations",
    )
    axes[2].legend(frameon=False)

    fig.tight_layout()
    save(fig, "figure_1_sparc_paired")


def figure_2_clusters() -> None:
    fox = pd.read_csv(FOX)
    clash = pd.read_csv(AUDIT / "tian_submitted_residuals.csv")
    clash = clash[~clash["overlaps_fox_calibration"].astype(bool)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.7))

    x = fox["MSL_200"] / 1.0e14
    y = fox["M_sigma"] / 1.0e14
    yerr = fox["MSL_err"] / 1.0e14
    axes[0].errorbar(
        x,
        y,
        xerr=yerr,
        fmt="o",
        ms=4.4,
        color=BLUE,
        ecolor=SKY,
        elinewidth=0.8,
        capsize=1.5,
        alpha=0.85,
    )
    limit = 1.08 * max(x.max(), y.max())
    axes[0].plot([0, limit], [0, limit], "--", color=BLACK, linewidth=1.2)
    axes[0].set(
        xlabel=r"Fox lensing mass [$10^{14}\,M_\odot$]",
        ylabel=r"Calibrated $\Sigma$ effective mass [$10^{14}\,M_\odot$]",
        xlim=(0, limit),
        ylim=(0, limit),
    )
    axes[0].text(
        0.04,
        0.95,
        "Calibration sample (N=42)",
        transform=axes[0].transAxes,
        va="top",
        fontweight="bold",
    )

    radii = sorted(clash["radius_kpc"].unique())
    rng = np.random.default_rng(20260718)
    jitter = np.exp(rng.normal(0.0, 0.018, len(clash)))
    axes[1].scatter(
        clash["radius_kpc"] * jitter,
        clash["ratio_predicted_observed"],
        s=21,
        facecolor="none",
        edgecolor=BLUE,
        linewidth=0.8,
        alpha=0.52,
        label="cluster measurements",
    )
    medians = clash.groupby("radius_kpc")["ratio_predicted_observed"].median()
    axes[1].plot(
        medians.index,
        medians.values,
        "o-",
        color=ORANGE,
        markeredgecolor=BLACK,
        markeredgewidth=0.4,
        label="median by radius",
    )
    axes[1].axhline(1, linestyle="--", color=BLACK, linewidth=1.2)
    axes[1].set_xscale("log")
    axes[1].set_xticks([14.3, 30, 100, 200, 400, 600])
    axes[1].set_xticklabels(["14", "30", "100", "200", "400", "600"])
    axes[1].set(
        xlabel="Projected radius [kpc]",
        ylabel="Predicted / observed acceleration",
    )
    axes[1].text(
        0.04,
        0.95,
        "No-refit CLASH check\n(17 clusters; 73 points)",
        transform=axes[1].transAxes,
        va="top",
        fontweight="bold",
    )
    axes[1].legend(frameon=False, loc="lower right")

    fig.tight_layout()
    save(fig, "figure_2_cluster_roles")


def figure_3_qumond() -> None:
    data = pd.read_csv(AUDIT / "qumond_axisymmetric_residuals.csv")
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    styles = {
        "F574-2": (BLUE, "o"),
        "UGC05716": (ORANGE, "s"),
        "NGC3741": (PURPLE, "^"),
    }
    for galaxy, subset in data.groupby("galaxy"):
        color, marker = styles.get(galaxy, (GRAY, "o"))
        ax.plot(
            subset["radius_over_Rdisk"],
            100 * subset["algebraic_relative_error"],
            marker=marker,
            markersize=4,
            color=color,
            label=galaxy,
        )
    ax.axhline(0, linestyle="--", color=BLACK, linewidth=1.2)
    ax.set(
        xlabel=r"Radius / $R_{\rm disk}$",
        ylabel="Algebraic minus numerical acceleration [%]",
    )
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    save(fig, "figure_3_qumond_approximation")


def figure_4_counterrotation() -> None:
    before = pd.read_csv(AUDIT / "counterrotation_smd_before.csv")
    after = pd.read_csv(AUDIT / "counterrotation_smd_after.csv")
    readiness = json.loads((AUDIT / "counterrotation_readiness.json").read_text())
    features = before["feature"].tolist()
    labels = {
        "log_stellar_mass": r"$\log M_\star$",
        "log_Re_kpc": r"$\log R_e$",
        "sersic_n": "Sérsic $n$",
        "axis_ratio": "axis ratio",
        "inclination_deg": "inclination",
        "redshift": "redshift",
        "jam_quality": "JAM fit quality",
        "jam_chi2_dof": "JAM fit quality",
    }
    y = np.arange(len(features))

    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.8), gridspec_kw={"width_ratios": [1.45, 1]})
    axes[0].scatter(before["absolute_smd"], y, marker="o", color=GRAY, label="before matching")
    axes[0].scatter(after["absolute_smd"], y, marker="s", color=BLUE, label="after matching")
    for i, (x0, x1) in enumerate(zip(before["absolute_smd"], after["absolute_smd"])):
        axes[0].plot([x0, x1], [i, i], color="#BBBBBB", linewidth=1, zorder=0)
    axes[0].axvline(0.1, linestyle="--", color=BLACK, linewidth=1.2, label="balance threshold")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([labels.get(f, f) for f in features])
    axes[0].invert_yaxis()
    axes[0].set(xlabel="Absolute standardized mean difference")
    axes[0].legend(frameon=False, loc="lower right")

    effect = readiness["secondary_JAM_NFW_fdm_comparison"]
    estimate = effect["mean_matched_difference_case_minus_control"]
    low, high = effect["bootstrap_95_percent_interval"]
    axes[1].errorbar(
        estimate,
        0,
        xerr=[[estimate - low], [high - estimate]],
        fmt="o",
        color=ORANGE,
        ecolor=ORANGE,
        markersize=7,
        capsize=5,
        linewidth=2,
    )
    axes[1].axvline(0, linestyle="--", color=BLACK, linewidth=1.2)
    axes[1].set(
        xlabel=r"Matched $\Delta f_{\rm DM}(<R_e)$",
        xlim=(-0.075, 0.065),
        yticks=[],
    )
    axes[1].text(
        0.5,
        0.82,
        f"{estimate:+.4f}\n95% CI [{low:+.4f}, {high:+.4f}]",
        transform=axes[1].transAxes,
        ha="center",
        va="center",
    )
    axes[1].text(
        0.5,
        0.12,
        "JAM/NFW-derived secondary outcome",
        transform=axes[1].transAxes,
        ha="center",
        va="center",
        fontsize=8.5,
    )

    fig.tight_layout()
    save(fig, "figure_4_counterrotation_matched")


def main() -> None:
    style()
    figure_1_sparc()
    figure_2_clusters()
    figure_3_qumond()
    figure_4_counterrotation()
    print(f"Wrote revision figures to {OUT}")


if __name__ == "__main__":
    main()
