#!/usr/bin/env python3
"""Ablate real-map morphology and thickness on project-spent DDO154."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_p0635_ddo154_map_commissioning import radial_circular_speed, score_curve

from voidscreen.data import load_curves
from voidscreen.field_solvers import (
    simple_mond_acceleration,
    solve_aqual,
    solve_newtonian,
    solve_qumond,
    surface_density_to_volume,
)

DEFAULT_CONFIG = ROOT / "configs" / "p0635_ddo154_map_commissioning.json"
DEFAULT_MAPS = ROOT / "results" / "p0635_ddo154_map_commissioning" / "baryonic_maps.npz"
DEFAULT_OUTPUT = ROOT / "results" / "p0635_ddo154_map_geometry_sensitivity"
SPARC = ROOT / "data" / "raw" / "sparc"


def axisymmetrize(surface_density: np.ndarray, axis_kpc: np.ndarray) -> np.ndarray:
    spacing = float(np.diff(axis_kpc)[0])
    x, y = np.meshgrid(axis_kpc, axis_kpc, indexing="ij")
    radial_bin = np.rint(np.hypot(x, y) / spacing).astype(int)
    result = np.zeros_like(surface_density)
    for index in np.unique(radial_bin):
        mask = radial_bin == index
        result[mask] = float(np.mean(surface_density[mask]))
    source_mass = float(np.sum(surface_density))
    result *= source_mass / float(np.sum(result))
    return result


def build_density(
    gas: np.ndarray,
    stars: np.ndarray,
    axis: np.ndarray,
    gas_height: float,
    stellar_height: float,
) -> np.ndarray:
    return surface_density_to_volume(
        gas, axis, scale_height=gas_height
    ) + surface_density_to_volume(stars, axis, scale_height=stellar_height)


def evaluate_variant(
    name: str,
    density: np.ndarray,
    axis: np.ndarray,
    gravity: float,
    a0: float,
    sparc_curve,
    *,
    run_aqual: bool,
) -> tuple[list[dict], list[pd.DataFrame]]:
    spacing = float(np.diff(axis)[0])
    newtonian = solve_newtonian(density, spacing, gravitational_constant=gravity)
    qumond = solve_qumond(density, spacing, a0=a0, gravitational_constant=gravity)
    solutions = [("newtonian", newtonian), ("QUMOND", qumond)]
    if run_aqual:
        solutions.append(
            (
                "AQUAL",
                solve_aqual(
                    density,
                    spacing,
                    a0=a0,
                    gravitational_constant=gravity,
                    residual_tolerance=1e-5,
                    maximum_nonlinear_iterations=100,
                    damping=0.5,
                ),
            )
        )
    rows = []
    frames = []
    for law, solution in solutions:
        curve = radial_circular_speed(solution, axis)
        curve.insert(0, "law", law)
        curve.insert(0, "variant", name)
        frames.append(curve)
        score = score_curve(
            curve["radius_kpc"].to_numpy(),
            curve["circular_speed_km_s"].to_numpy(),
            sparc_curve.radius_kpc,
            sparc_curve.velocity_observed_kms,
            sparc_curve.velocity_error_kms,
        )
        rows.append(
            {
                "variant": name,
                "law": law,
                **score,
                "normalized_residual_RMS": solution.normalized_residual_rms,
                "converged": solution.converged,
                "nonlinear_iterations": solution.nonlinear_iterations,
            }
        )
    algebraic = frames[0].copy()
    algebraic["law"] = "algebraic_simple_MOND"
    g_newton = algebraic["inward_acceleration_km2_s2_kpc"].to_numpy()
    g_mond = simple_mond_acceleration(g_newton, a0)
    algebraic["inward_acceleration_km2_s2_kpc"] = g_mond
    algebraic["circular_speed_km_s"] = np.sqrt(
        algebraic["radius_kpc"].to_numpy() * g_mond
    )
    frames.append(algebraic)
    score = score_curve(
        algebraic["radius_kpc"].to_numpy(),
        algebraic["circular_speed_km_s"].to_numpy(),
        sparc_curve.radius_kpc,
        sparc_curve.velocity_observed_kms,
        sparc_curve.velocity_error_kms,
    )
    rows.append(
        {
            "variant": name,
            "law": "algebraic_simple_MOND",
            **score,
            "normalized_residual_RMS": 0.0,
            "converged": True,
            "nonlinear_iterations": 0,
        }
    )
    return rows, frames


def run(config: dict, maps_path: Path, output: Path) -> dict:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    with np.load(maps_path) as maps:
        axis = maps["axis_kpc"].astype(float)
        gas = maps["gas_surface_density_solar_kpc2"].astype(float)
        stars = maps["stellar_surface_density_solar_kpc2"].astype(float)
    gas_axisymmetric = axisymmetrize(gas, axis)
    stars_axisymmetric = axisymmetrize(stars, axis)
    constants = config["field_laws"]
    gravity = float(constants["gravitational_constant_kpc_km2_s2_per_solar_mass"])
    a0 = float(constants["a0_km2_s2_per_kpc"])
    baseline_gas_height = float(config["grid"]["gas_scale_height_kpc"])
    baseline_stellar_height = float(config["grid"]["stellar_scale_height_kpc"])
    empty_stars = np.zeros_like(stars)
    variants = [
        (
            "lumpy_razor_thin",
            build_density(gas, stars, axis, 0.0, 0.0),
            True,
        ),
        (
            "lumpy_baseline_thickness",
            build_density(
                gas, stars, axis, baseline_gas_height, baseline_stellar_height
            ),
            False,
        ),
        (
            "lumpy_thick",
            build_density(gas, stars, axis, 0.60, 0.90),
            False,
        ),
        (
            "axisymmetric_baseline_thickness",
            build_density(
                gas_axisymmetric,
                stars_axisymmetric,
                axis,
                baseline_gas_height,
                baseline_stellar_height,
            ),
            True,
        ),
        (
            "lumpy_gas_only",
            build_density(gas, empty_stars, axis, baseline_gas_height, 0.0),
            False,
        ),
    ]
    sparc_curve = next(
        curve for curve in load_curves(SPARC) if curve.metadata.name == "DDO154"
    )
    rows = []
    frames = []
    for name, density, run_aqual in variants:
        variant_rows, variant_frames = evaluate_variant(
            name,
            density,
            axis,
            gravity,
            a0,
            sparc_curve,
            run_aqual=run_aqual,
        )
        rows.extend(variant_rows)
        frames.extend(variant_frames)
    scores = pd.DataFrame(rows)
    curves = pd.concat(frames, ignore_index=True)
    scores.to_csv(output / "geometry_scores.csv", index=False)
    curves.to_csv(output / "geometry_curves.csv", index=False)

    pivot = scores.pivot(index="variant", columns="law", values="RMSE_km_s")
    diagnostics = {
        "QUMOND_axisymmetry_RMSE_change_km_s": float(
            pivot.loc["axisymmetric_baseline_thickness", "QUMOND"]
            - pivot.loc["lumpy_baseline_thickness", "QUMOND"]
        ),
        "QUMOND_razor_minus_baseline_RMSE_km_s": float(
            pivot.loc["lumpy_razor_thin", "QUMOND"]
            - pivot.loc["lumpy_baseline_thickness", "QUMOND"]
        ),
        "QUMOND_thick_minus_baseline_RMSE_km_s": float(
            pivot.loc["lumpy_thick", "QUMOND"]
            - pivot.loc["lumpy_baseline_thickness", "QUMOND"]
        ),
        "QUMOND_remove_stars_RMSE_change_km_s": float(
            pivot.loc["lumpy_gas_only", "QUMOND"]
            - pivot.loc["lumpy_baseline_thickness", "QUMOND"]
        ),
        "axisymmetric_QUMOND_minus_algebraic_RMSE_km_s": float(
            pivot.loc["axisymmetric_baseline_thickness", "QUMOND"]
            - pivot.loc["axisymmetric_baseline_thickness", "algebraic_simple_MOND"]
        ),
    }
    report = {
        "status": "complete" if bool(scores["converged"].all()) else "solver_failure",
        "galaxy": "DDO154",
        "variants": [name for name, _, _ in variants],
        "scores": rows,
        "diagnostics": diagnostics,
        "runtime_seconds": float(time.perf_counter() - started),
        "claim_boundary": "Exploratory commissioning ablation on a project-spent galaxy; no P0633 target was opened and no candidate parameter was selected.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    figure, axis_plot = plt.subplots(figsize=(9.0, 5.0))
    shown = scores.loc[scores["law"].isin(["algebraic_simple_MOND", "QUMOND", "AQUAL"])]
    order = [name for name, _, _ in variants]
    x = np.arange(len(order), dtype=float)
    for offset, law, color in (
        (-0.22, "algebraic_simple_MOND", "#999933"),
        (0.0, "QUMOND", "#228833"),
        (0.22, "AQUAL", "#CC6677"),
    ):
        subset = shown.loc[shown["law"].eq(law)].set_index("variant")
        values = [subset.loc[name, "RMSE_km_s"] if name in subset.index else np.nan for name in order]
        axis_plot.bar(x + offset, values, width=0.21, label=law.replace("_", " "), color=color)
    axis_plot.set_xticks(x, [name.replace("_", "\n") for name in order], fontsize=8)
    axis_plot.set_ylabel("DDO154 rotation-curve RMSE (km/s)")
    axis_plot.set_title("Real-map geometry sensitivity; lower is better")
    axis_plot.grid(axis="y", alpha=0.25)
    axis_plot.legend()
    figure.tight_layout()
    figure.savefig(output / "geometry_sensitivity.png", dpi=180)
    plt.close(figure)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--maps", type=Path, default=DEFAULT_MAPS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    report = run(config, args.maps.resolve(), args.output.resolve())
    print(json.dumps({"status": report["status"], "diagnostics": report["diagnostics"]}, indent=2))
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
