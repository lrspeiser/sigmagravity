from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_triaxial_memory import (
    centered_axis,
    gaussian_mixture_density,
    integrated_response,
    spectral_tidal_memory,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def maximum_factor_range(
    ratios: pd.DataFrame,
    varied: str,
    fixed: list[str],
) -> float:
    factors = []
    for _, group in ratios.groupby(fixed):
        if group[varied].nunique() < 2:
            continue
        values = group.cluster_to_galaxy_response_ratio
        factors.append(float(values.max() / values.min()))
    return max(factors, default=1.0)


def high_ratio_neighbor_fraction(ratios: pd.DataFrame, row: pd.Series) -> float:
    powers = sorted(ratios.screen_power.unique())
    lengths = sorted(ratios.memory_length.unique())
    widths = sorted(ratios.analysis_half_width.unique())
    neighbors = []
    for field, values in (
        ("screen_power", powers),
        ("memory_length", lengths),
        ("analysis_half_width", widths),
    ):
        index = values.index(row[field])
        for neighbor_index in (index - 1, index + 1):
            if not 0 <= neighbor_index < len(values):
                continue
            mask = (ratios.total_mass == row.total_mass) & (ratios.screen_order == row.screen_order)
            for other in {"screen_power", "memory_length", "analysis_half_width"} - {field}:
                mask &= ratios[other] == row[other]
            mask &= ratios[field] == values[neighbor_index]
            neighbors.extend(ratios.loc[mask, "cluster_to_galaxy_response_ratio"].tolist())
    toggled = ratios[
        (ratios.total_mass == row.total_mass)
        & (ratios.screen_order != row.screen_order)
        & (ratios.screen_power == row.screen_power)
        & (ratios.memory_length == row.memory_length)
        & (ratios.analysis_half_width == row.analysis_half_width)
    ]
    neighbors.extend(toggled.cluster_to_galaxy_response_ratio.tolist())
    if not neighbors:
        return 0.0
    return float(np.mean(np.asarray(neighbors) >= 10.0))


def plot_diagnostics(output: Path, ratios: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    colors = {0.3: "#2a6fbb", 1.0: "#8a3ffc", 3.0: "#d1495b"}
    for column, order in enumerate(("before_memory", "after_memory")):
        subset = ratios[(ratios.screen_order == order) & (ratios.analysis_half_width == 2.0)]
        for mass, group in subset.groupby("total_mass"):
            for power, power_group in group.groupby("screen_power"):
                axes[0, column].plot(
                    power_group.memory_length,
                    power_group.cluster_to_galaxy_response_ratio,
                    marker="o",
                    color=colors[float(mass)],
                    alpha=0.35 + 0.08 * float(power),
                    label=f"M={mass:g}, p={power:g}",
                )
        axes[0, column].axhline(10.0, color="black", linestyle="--", linewidth=0.8)
        axes[0, column].set_xscale("log")
        axes[0, column].set_yscale("log")
        axes[0, column].set_xlabel("memory length / fixture scale")
        axes[0, column].set_ylabel("distributed / compact response")
        axes[0, column].set_title(order.replace("_", " "))
        axes[0, column].grid(alpha=0.2)
    axes[0, 0].legend(fontsize=6, ncols=3, loc="lower left")

    for column, mass in enumerate((1.0, 3.0)):
        subset = ratios[
            (ratios.total_mass == mass)
            & (ratios.screen_power == 4.0)
            & (ratios.memory_length == 1.0)
        ]
        for order, group in subset.groupby("screen_order"):
            axes[1, column].plot(
                group.analysis_half_width,
                group.cluster_to_galaxy_response_ratio,
                marker="o",
                label=order.replace("_", " "),
            )
        axes[1, column].axhline(10.0, color="black", linestyle="--", linewidth=0.8)
        axes[1, column].set_yscale("log")
        axes[1, column].set_xlabel("analysis half-width / fixture scale")
        axes[1, column].set_ylabel("distributed / compact response")
        axes[1, column].set_title(f"Scored-volume sensitivity, M={mass:g}")
        axes[1, column].grid(alpha=0.2)
        axes[1, column].legend()
    figure.suptitle("Sigma v3D post-failure diagnostics", fontsize=14)
    figure.savefig(output / "post_failure_diagnostics.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose the frozen Sigma v3D failure.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v3d_post_failure_diagnostics.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v3d_post_failure_diagnostics",
    )
    args = parser.parse_args()
    diagnostic = json.loads(args.config.read_text(encoding="utf-8"))
    parent_path = ROOT / diagnostic["parent_protocol"]
    if sha256(parent_path) != diagnostic["parent_config_sha256"]:
        raise RuntimeError("parent v3D protocol hash no longer matches the frozen failure")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    fixture = parent["dimensionless_fixture"]
    axis = centered_axis(int(diagnostic["grid_points"]), float(fixture["box_half_width_L_sigma"]))
    spacing = float(axis[1] - axis[0])
    densities: dict[tuple[str, float], np.ndarray] = {}
    for system in ("galaxy", "cluster"):
        for mass in map(float, diagnostic["mass_normalizations"]):
            densities[system, mass] = gaussian_mixture_density(
                axis,
                fixture[f"{system}_components"],
                total_mass=mass,
            )

    records = []
    for order in diagnostic["screen_orders"]:
        for power in map(float, diagnostic["screen_powers"]):
            for length in map(float, diagnostic["memory_lengths_L_sigma_fixture"]):
                for mass in map(float, diagnostic["mass_normalizations"]):
                    for system in ("galaxy", "cluster"):
                        field = spectral_tidal_memory(
                            densities[system, mass],
                            spacing=spacing,
                            gravitational_constant=float(fixture["G"]),
                            a_sigma=float(fixture["a_sigma"]),
                            memory_length=length,
                            screen_power=power,
                            screen_order=order,
                        )
                        for half_width in map(
                            float, diagnostic["analysis_half_widths_L_sigma_fixture"]
                        ):
                            records.append(
                                {
                                    "system": system,
                                    "total_mass": mass,
                                    "screen_order": order,
                                    "screen_power": power,
                                    "memory_length": length,
                                    "analysis_half_width": half_width,
                                    "integrated_response": integrated_response(
                                        field.bounded_potential,
                                        axis,
                                        analysis_half_width=half_width,
                                    ),
                                }
                            )
    responses = pd.DataFrame.from_records(records)
    args.output.mkdir(parents=True, exist_ok=True)
    responses.to_csv(args.output / "diagnostic_responses.csv", index=False)
    key = [
        "total_mass",
        "screen_order",
        "screen_power",
        "memory_length",
        "analysis_half_width",
    ]
    pivot = responses.pivot(index=key, columns="system", values="integrated_response")
    ratios = pivot.reset_index()
    ratios["cluster_to_galaxy_response_ratio"] = ratios.cluster / ratios.galaxy
    ratios.to_csv(args.output / "diagnostic_ratios.csv", index=False)

    dimensions = {
        "screen": (
            "screen_power",
            ["total_mass", "screen_order", "memory_length", "analysis_half_width"],
        ),
        "memory": (
            "memory_length",
            ["total_mass", "screen_order", "screen_power", "analysis_half_width"],
        ),
        "ordering": (
            "screen_order",
            ["total_mass", "screen_power", "memory_length", "analysis_half_width"],
        ),
        "volume": (
            "analysis_half_width",
            ["total_mass", "screen_order", "screen_power", "memory_length"],
        ),
    }
    sensitivities = {
        name: maximum_factor_range(ratios, varied, fixed)
        for name, (varied, fixed) in dimensions.items()
    }
    best = ratios.loc[ratios.cluster_to_galaxy_response_ratio.idxmax()].copy()
    high_rows = ratios[ratios.cluster_to_galaxy_response_ratio >= 10.0].copy()
    if not high_rows.empty:
        high_rows["neighbor_high_fraction"] = high_rows.apply(
            lambda row: high_ratio_neighbor_fraction(ratios, row), axis=1
        )
        maximum_neighbor_fraction = float(high_rows.neighbor_high_fraction.max())
    else:
        maximum_neighbor_fraction = 0.0
    dominant = max(sensitivities, key=sensitivities.get)
    report = {
        "protocol_id": diagnostic["protocol_id"],
        "status": diagnostic["status"],
        "parent_config_sha256": diagnostic["parent_config_sha256"],
        "diagnostic_config_sha256": sha256(args.config),
        "combinations": len(ratios),
        "maximum_cluster_to_galaxy_response_ratio": float(best.cluster_to_galaxy_response_ratio),
        "best_settings": {
            name: (str(best[name]) if name == "screen_order" else float(best[name])) for name in key
        },
        "fraction_combinations_at_least_ten": float(
            np.mean(ratios.cluster_to_galaxy_response_ratio >= 10.0)
        ),
        "maximum_high_ratio_neighbor_fraction": maximum_neighbor_fraction,
        "maximum_factor_sensitivity": sensitivities,
        "dominant_sensitivity": dominant,
        "material_sensitivities": {
            name: bool(value >= 3.0) for name, value in sensitivities.items()
        },
        "fragile_large_separation": bool(not high_rows.empty and maximum_neighbor_fraction < 0.5),
        "frozen_v3d_decision_changed": False,
        "decision": "use_for_mechanism_selection_only",
        "raw_holdout_opened": False,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_diagnostics(args.output, ratios)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
