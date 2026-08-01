from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--responses",
        type=Path,
        default=ROOT / "results" / "nbp0_morphology_sweep" / "responses.csv",
    )
    parser.add_argument(
        "--empirical-report",
        type=Path,
        default=ROOT / "results" / "nbp0_sparc_morphology_test" / "report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "nbp0_morphology_summary.png",
    )
    args = parser.parse_args()
    frame = pd.read_csv(args.responses)
    empirical = json.loads(args.empirical_report.read_text(encoding="utf-8"))
    radii = np.asarray([1.0, 2.2, 4.0, 6.0, 8.0])
    labels = [str(radius).replace(".", "p") for radius in radii]

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), constrained_layout=True)

    oat = frame.loc[
        (frame["case_family"] == "one_at_a_time")
        & (frame["varied_parameter"] == "stellar_bulge_fraction")
    ].sort_values("stellar_bulge_fraction")
    for radius in (1.0, 4.0, 6.0, 8.0):
        label = str(radius).replace(".", "p")
        axes[0].plot(
            oat["stellar_bulge_fraction"],
            oat[f"enhancement_R{label}"],
            marker="o",
            label=f"R = {radius:g} Rd",
        )
    axes[0].set(
        xlabel="stellar bulge fraction",
        ylabel="modified / Newtonian radial acceleration",
        title="A. Baseline one-at-a-time response",
    )
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.2)

    paired = frame.loc[frame["case_family"] == "paired_environment"]
    fractions = []
    meaningful = []
    for label in labels:
        wide = paired.pivot(
            index="pair_id", columns="morphology_member", values=f"enhancement_R{label}"
        )
        difference = wide["disk"] - wide["bulge"]
        fractions.append(float(np.mean(difference > 0.0)))
        meaningful.append(
            float(np.mean(difference > 0.01 * np.abs(wide["bulge"])))
        )
    axes[1].plot(radii, fractions, marker="o", label="disk > bulge")
    axes[1].plot(
        radii, meaningful, marker="s", label="disk > bulge by at least 1%"
    )
    axes[1].axhline(0.8, color="black", linestyle="--", linewidth=1, label="frozen gate")
    axes[1].set(
        xlabel="radius / Rd",
        ylabel="fraction of 128 matched environments",
        ylim=(0.0, 1.02),
        title="B. The morphology ordering changes with radius",
    )
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.2)

    sensitivity = pd.DataFrame(empirical["mass_to_light_sensitivity_results"])
    disk_values = sorted(sensitivity["disk_mass_to_light"].unique())
    bulge_values = sorted(sensitivity["bulge_mass_to_light"].unique())
    matrix = sensitivity.pivot(
        index="disk_mass_to_light",
        columns="bulge_mass_to_light",
        values="relative_structure_to_morphology_RMSE_improvement",
    ).loc[disk_values, bulge_values]
    image = axes[2].imshow(
        100.0 * matrix.to_numpy(),
        origin="lower",
        cmap="RdBu_r",
        vmin=-10.0,
        vmax=10.0,
        aspect="auto",
    )
    for row in range(len(disk_values)):
        for column in range(len(bulge_values)):
            axes[2].text(
                column,
                row,
                f"{100.0 * matrix.iloc[row, column]:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
            )
    axes[2].set_xticks(range(len(bulge_values)), [f"{value:.1f}" for value in bulge_values])
    axes[2].set_yticks(range(len(disk_values)), [f"{value:.1f}" for value in disk_values])
    axes[2].set(
        xlabel="bulge mass-to-light ratio",
        ylabel="disk mass-to-light ratio",
        title="C. Held-out RMSE change from morphology",
    )
    figure.colorbar(image, ax=axes[2], label="relative improvement (%)", shrink=0.85)

    figure.suptitle(
        "NBP0-M1: scalar permittivity responds to shape but fails the predicted disk ordering",
        fontsize=13,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
