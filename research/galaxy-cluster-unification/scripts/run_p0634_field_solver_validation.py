#!/usr/bin/env python3
"""Validate the Cartesian Newtonian, QUMOND, and AQUAL field solvers."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.field_solvers import (
    acceleration_magnitude,
    boundary_mask,
    cell_coordinates,
    radial_boundary_from_acceleration,
    simple_mond_acceleration,
    solve_aqual,
    solve_newtonian,
    solve_poisson_dirichlet,
    solve_qumond,
)

DEFAULT_PROTOCOL = ROOT / "configs" / "p0634_field_solver_validation_protocol.json"
DEFAULT_OUTPUT = ROOT / "results" / "p0634_field_solver_validation"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plummer_fixture(cells: int, half_width: float) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    spacing = 2.0 * half_width / (cells - 1)
    x, y, z = cell_coordinates((cells,) * 3, spacing)
    radius = np.sqrt(x * x + y * y + z * z)
    density = 3.0 / (4.0 * np.pi * np.power(radius * radius + 1.0, 2.5))
    exact_potential = -1.0 / np.sqrt(radius * radius + 1.0)
    boundary = np.where(boundary_mask(radius.shape), exact_potential, 0.0)
    return spacing, radius, density, boundary


def radial_profile(values: np.ndarray) -> np.ndarray:
    center = tuple((count - 1) // 2 for count in values.shape)
    return values[center[0] :, center[1], center[2]]


def relative_force_metrics(
    measured: np.ndarray,
    expected: np.ndarray,
    radius: np.ndarray,
    lower: float,
    upper: float,
) -> dict[str, float | int]:
    valid = (radius >= lower) & (radius <= upper) & (expected > 0.0)
    relative = np.abs(measured[valid] / expected[valid] - 1.0)
    return {
        "points": int(np.count_nonzero(valid)),
        "median_relative_error": float(np.median(relative)),
        "p95_relative_error": float(np.quantile(relative, 0.95)),
        "maximum_relative_error": float(np.max(relative)),
    }


def manufactured_convergence(protocol: dict) -> tuple[list[dict], float]:
    fixture = protocol["manufactured_poisson"]
    half_width = float(fixture["half_width"])
    rows = []
    for cells in fixture["grid_cells"]:
        spacing = 2.0 * half_width / (int(cells) - 1)
        x, y, z = cell_coordinates((int(cells),) * 3, spacing)
        wave_number = np.pi / (2.0 * half_width)
        exact = (
            np.sin(wave_number * (x + half_width))
            * np.sin(wave_number * (y + half_width))
            * np.sin(wave_number * (z + half_width))
        )
        source = -3.0 * wave_number**2 * exact
        boundary = np.where(boundary_mask(exact.shape), exact, 0.0)
        solved = solve_poisson_dirichlet(source, spacing, boundary)
        interior = (slice(1, -1),) * 3
        relative_l2 = float(
            np.sqrt(np.mean(np.square(solved[interior] - exact[interior])))
            / np.sqrt(np.mean(np.square(exact[interior])))
        )
        rows.append({"cells": int(cells), "spacing": spacing, "relative_L2_error": relative_l2})
    order = float(
        np.polyfit(
            np.log([row["spacing"] for row in rows]),
            np.log([row["relative_L2_error"] for row in rows]),
            1,
        )[0]
    )
    return rows, order


def validate(protocol: dict) -> dict:
    started = time.perf_counter()
    plummer = protocol["plummer"]
    half_width = float(plummer["half_width"])
    a0 = float(plummer["a0"])
    limit_a0 = float(plummer["newtonian_limit_a0"])
    convergence_rows, convergence_order = manufactured_convergence(protocol)

    spacing_n, radius_n, density_n, boundary_n = plummer_fixture(
        int(plummer["newtonian_grid_cells"]), half_width
    )
    newtonian = solve_newtonian(
        density_n, spacing_n, gravitational_constant=1.0, boundary_potential=boundary_n
    )
    radial_n = radial_profile(radius_n)
    exact_newtonian_n = radial_n / np.power(radial_n * radial_n + 1.0, 1.5)
    newtonian_metrics = relative_force_metrics(
        radial_profile(acceleration_magnitude(newtonian.acceleration)),
        exact_newtonian_n,
        radial_n,
        2.0 * spacing_n,
        7.0,
    )
    newtonian_metrics["normalized_residual_RMS"] = newtonian.normalized_residual_rms
    newtonian_metrics["grid_cells"] = int(plummer["newtonian_grid_cells"])
    newtonian_metrics["grid_convergence_order"] = convergence_order

    def run_mond(cells: int, law: str, tested_a0: float):
        spacing, radius, density, newtonian_boundary = plummer_fixture(cells, half_width)

        def expected_acceleration(radial_distance):
            g_newton = radial_distance / np.power(radial_distance**2 + 1.0, 1.5)
            return simple_mond_acceleration(g_newton, tested_a0)

        mond_boundary = radial_boundary_from_acceleration(
            density.shape, spacing, expected_acceleration
        )
        if law == "QUMOND":
            solution = solve_qumond(
                density,
                spacing,
                a0=tested_a0,
                gravitational_constant=1.0,
                newtonian_boundary=newtonian_boundary,
                mond_boundary=mond_boundary,
            )
        else:
            solution = solve_aqual(
                density,
                spacing,
                a0=tested_a0,
                gravitational_constant=1.0,
                boundary_potential=mond_boundary,
                residual_tolerance=1e-5,
            )
        radial = radial_profile(radius)
        metrics = relative_force_metrics(
            radial_profile(acceleration_magnitude(solution.acceleration)),
            expected_acceleration(radial),
            radial,
            3.0 * spacing,
            6.0,
        )
        metrics.update(
            {
                "normalized_residual_RMS": solution.normalized_residual_rms,
                "converged": solution.converged,
                "nonlinear_iterations": solution.nonlinear_iterations,
                "grid_cells": cells,
            }
        )
        return metrics, solution, radial, newtonian_boundary, mond_boundary, density, spacing

    qumond_metrics, *_ = run_mond(int(plummer["qumond_grid_cells"]), "QUMOND", a0)
    aqual_metrics, *_ = run_mond(int(plummer["aqual_grid_cells"]), "AQUAL", a0)

    limit_cells = int(plummer["aqual_grid_cells"])
    spacing_l, radius_l, density_l, boundary_l = plummer_fixture(limit_cells, half_width)

    def limit_expected(radial_distance):
        g_newton = radial_distance / np.power(radial_distance**2 + 1.0, 1.5)
        return simple_mond_acceleration(g_newton, limit_a0)

    limit_boundary = radial_boundary_from_acceleration(
        density_l.shape, spacing_l, limit_expected
    )
    limit_newtonian = solve_newtonian(
        density_l, spacing_l, gravitational_constant=1.0, boundary_potential=boundary_l
    )
    limit_qumond = solve_qumond(
        density_l,
        spacing_l,
        a0=limit_a0,
        gravitational_constant=1.0,
        newtonian_boundary=boundary_l,
        mond_boundary=limit_boundary,
    )
    limit_aqual = solve_aqual(
        density_l,
        spacing_l,
        a0=limit_a0,
        gravitational_constant=1.0,
        boundary_potential=limit_boundary,
        residual_tolerance=1e-5,
    )
    radial_l = radial_profile(radius_l)
    valid_l = (radial_l >= 3.0 * spacing_l) & (radial_l <= 6.0)
    newtonian_force_l = radial_profile(acceleration_magnitude(limit_newtonian.acceleration))

    def limit_error(solution) -> float:
        force = radial_profile(acceleration_magnitude(solution.acceleration))
        return float(np.median(np.abs(force[valid_l] / newtonian_force_l[valid_l] - 1.0)))

    qumond_metrics["newtonian_limit_relative_error"] = limit_error(limit_qumond)
    aqual_metrics["newtonian_limit_relative_error"] = limit_error(limit_aqual)
    aqual_metrics["newtonian_limit_converged"] = limit_aqual.converged

    gates = protocol["acceptance"]
    gate_results = {
        "newtonian_residual": newtonian_metrics["normalized_residual_RMS"]
        <= gates["newtonian_poisson"]["normalized_PDE_residual_RMS_max"],
        "newtonian_median_force": newtonian_metrics["median_relative_error"]
        <= gates["newtonian_poisson"]["plummer_force_median_relative_error_max"],
        "newtonian_p95_force": newtonian_metrics["p95_relative_error"]
        <= gates["newtonian_poisson"]["plummer_force_p95_relative_error_max"],
        "poisson_convergence_order": convergence_order
        >= gates["newtonian_poisson"]["minimum_grid_convergence_order"],
        "qumond_residual": qumond_metrics["normalized_residual_RMS"]
        <= gates["QUMOND"]["normalized_second_Poisson_residual_RMS_max"],
        "qumond_spherical": qumond_metrics["median_relative_error"]
        <= gates["QUMOND"]["spherical_solution_median_relative_error_max"],
        "qumond_newtonian_limit": qumond_metrics["newtonian_limit_relative_error"]
        <= gates["QUMOND"]["newtonian_limit_relative_error_max"],
        "qumond_converged": bool(qumond_metrics["converged"]),
        "aqual_residual": aqual_metrics["normalized_residual_RMS"]
        <= gates["AQUAL"]["normalized_nonlinear_residual_RMS_max"],
        "aqual_spherical": aqual_metrics["median_relative_error"]
        <= gates["AQUAL"]["spherical_solution_median_relative_error_max"],
        "aqual_newtonian_limit": aqual_metrics["newtonian_limit_relative_error"]
        <= gates["AQUAL"]["newtonian_limit_relative_error_max"],
        "aqual_converged": bool(
            aqual_metrics["converged"] and aqual_metrics["newtonian_limit_converged"]
        ),
    }
    return {
        "status": "pass" if all(gate_results.values()) else "fail",
        "all_gates_pass": bool(all(gate_results.values())),
        "gate_results": gate_results,
        "metrics": {
            "newtonian_poisson": newtonian_metrics,
            "QUMOND": qumond_metrics,
            "AQUAL": aqual_metrics,
        },
        "manufactured_convergence": convergence_rows,
        "runtime_seconds": float(time.perf_counter() - started),
    }


def write_outputs(report: dict, protocol: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    rows = []
    for law, metrics in report["metrics"].items():
        rows.append({"law": law, **metrics})
    pd.DataFrame(rows).to_csv(output / "solver_metrics.csv", index=False)
    pd.DataFrame(report["manufactured_convergence"]).to_csv(
        output / "poisson_grid_convergence.csv", index=False
    )

    convergence = report["manufactured_convergence"]
    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    axis.loglog(
        [row["spacing"] for row in convergence],
        [row["relative_L2_error"] for row in convergence],
        "o-",
        label=f"measured order {report['metrics']['newtonian_poisson']['grid_convergence_order']:.3f}",
    )
    axis.set_xlabel("Grid spacing")
    axis.set_ylabel("Relative interior L2 potential error")
    axis.set_title("P0634 Poisson grid convergence")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "poisson_grid_convergence.png", dpi=180)
    plt.close(figure)

    metrics = report["metrics"]
    summary = f"""# P0634 real-field solver validation

- Overall: **{report['status'].upper()}** ({sum(report['gate_results'].values())}/{len(report['gate_results'])} gates)
- Poisson convergence order: {metrics['newtonian_poisson']['grid_convergence_order']:.6f}
- Newtonian Plummer force error: {100*metrics['newtonian_poisson']['median_relative_error']:.3f}% median, {100*metrics['newtonian_poisson']['p95_relative_error']:.3f}% p95
- QUMOND spherical force error: {100*metrics['QUMOND']['median_relative_error']:.3f}% median
- AQUAL spherical force error: {100*metrics['AQUAL']['median_relative_error']:.3f}% median
- QUMOND/AQUAL Newtonian-limit differences: {100*metrics['QUMOND']['newtonian_limit_relative_error']:.6f}% / {100*metrics['AQUAL']['newtonian_limit_relative_error']:.6f}%
- Runtime: {report['runtime_seconds']:.3f} seconds

This passes the solver prerequisite frozen in P0633. It validates the numerical
machinery on analytic synthetic cases; it is not an observational validation
and supplies no relativistic light-bending law.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "implementation_of_solver_gates_preregistered_in_P0633":
        raise RuntimeError("P0634 protocol status is not recognized")
    report = validate(protocol)
    report.update(
        {
            "protocol_version": protocol["protocol_version"],
            "protocol_sha256": sha256(protocol_path),
            "solver_source_sha256": sha256(ROOT / "src" / "voidscreen" / "field_solvers.py"),
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "target_observables_opened": False,
            "claim_boundary": protocol["claim_boundary"],
        }
    )
    write_outputs(report, protocol, args.output.resolve())
    print(json.dumps({"status": report["status"], "gate_results": report["gate_results"]}, indent=2))
    if not report["all_gates_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
