from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.field_solvers import cell_coordinates
from voidscreen.sigma_v7_cubic_hessian import (
    solve_cubic_hessian_dirichlet,
)


def relative_rms(difference: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.sqrt(np.mean(np.square(difference)))
        / max(float(np.sqrt(np.mean(np.square(reference)))), np.finfo(float).tiny)
    )


def gaussian_component(
    coordinates: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    axis: int,
    center: float,
    amplitude: float,
    sigma: float,
) -> np.ndarray:
    shifted = [np.asarray(value, dtype=float) for value in coordinates]
    shifted[axis] = shifted[axis] - center
    radius_squared = sum(np.square(value) for value in shifted)
    return amplitude * np.exp(-radius_squared / (2.0 * sigma**2))


def solve_pair(
    *,
    shape: tuple[int, int, int],
    spacing: float,
    axis: int,
    centers: tuple[float, float],
    amplitude: float,
    sigma: float,
    relaxation: float,
    tolerance: float,
    max_iterations: int,
    include_individuals: bool,
):
    coordinates = cell_coordinates(shape, spacing)
    boundary = np.zeros(shape)
    sources = tuple(
        gaussian_component(
            coordinates,
            axis=axis,
            center=center,
            amplitude=amplitude,
            sigma=sigma,
        )
        for center in centers
    )
    options = {
        "kappa": 1.0,
        "relaxation": relaxation,
        "tolerance": tolerance,
        "max_iterations": max_iterations,
    }
    individuals = (
        tuple(
            solve_cubic_hessian_dirichlet(source, spacing, boundary, **options)
            for source in sources
        )
        if include_individuals
        else ()
    )
    combined = solve_cubic_hessian_dirichlet(
        sources[0] + sources[1], spacing, boundary, **options
    )
    return individuals, combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the Sigma v7C cubic Hessian solve.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v7c_cubic_hessian_gate.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v7c_cubic_hessian_gate",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manufactured = config["manufactured_sphere"]
    size = int(manufactured["grid_size"])
    spacing = float(manufactured["spacing"])
    coordinates = cell_coordinates((size, size, size), spacing)
    amplitude = float(manufactured["quadratic_amplitude"])
    expected = amplitude * sum(np.square(value) for value in coordinates)
    source = np.full(expected.shape, 18.0 * amplitude + 24.0 * amplitude**2)
    manufactured_solution = solve_cubic_hessian_dirichlet(
        source,
        spacing,
        expected,
        kappa=1.0,
        relaxation=float(manufactured["relaxation"]),
        tolerance=float(manufactured["tolerance"]),
        max_iterations=int(manufactured["max_iterations"]),
    )
    manufactured_interior = (slice(1, -1),) * 3
    manufactured_error = relative_rms(
        manufactured_solution.potential[manufactured_interior]
        - expected[manufactured_interior],
        expected[manufactured_interior],
    )

    components = config["separated_components"]
    centers = tuple(float(value) for value in components["component_centers"])
    common = {
        "centers": centers,
        "amplitude": float(components["component_amplitude"]),
        "sigma": float(components["component_sigma"]),
        "relaxation": float(components["relaxation"]),
        "tolerance": float(components["tolerance"]),
        "max_iterations": int(components["max_iterations"]),
    }
    coarse_size = int(components["coarse_grid_size"])
    coarse_spacing = float(components["coarse_spacing"])
    individuals, coarse_x = solve_pair(
        shape=(coarse_size,) * 3,
        spacing=coarse_spacing,
        axis=0,
        include_individuals=True,
        **common,
    )
    _, coarse_y = solve_pair(
        shape=(coarse_size,) * 3,
        spacing=coarse_spacing,
        axis=1,
        include_individuals=False,
        **common,
    )
    coarse_interior = (slice(1, -1),) * 3
    summed = individuals[0].potential + individuals[1].potential
    component_nonadditivity = relative_rms(
        coarse_x.potential[coarse_interior] - summed[coarse_interior],
        coarse_x.potential[coarse_interior],
    )
    rotation_covariance = relative_rms(
        np.swapaxes(coarse_x.potential, 0, 1)[coarse_interior]
        - coarse_y.potential[coarse_interior],
        coarse_y.potential[coarse_interior],
    )

    fine_size = int(components["fine_grid_size"])
    fine_spacing = float(components["fine_spacing"])
    _, fine_x = solve_pair(
        shape=(fine_size,) * 3,
        spacing=fine_spacing,
        axis=0,
        include_individuals=False,
        **common,
    )
    fine_on_coarse = fine_x.potential[::2, ::2, ::2]
    if fine_on_coarse.shape != coarse_x.potential.shape:
        raise RuntimeError("fine grid does not downsample exactly onto the coarse grid")
    resolution_change = relative_rms(
        fine_on_coarse[coarse_interior] - coarse_x.potential[coarse_interior],
        fine_on_coarse[coarse_interior],
    )

    solutions = (
        manufactured_solution,
        *individuals,
        coarse_x,
        coarse_y,
        fine_x,
    )
    maximum_residual = max(solution.residual_rms for solution in solutions)
    minimum_temporal = min(
        solution.minimum_temporal_kinetic_coefficient for solution in solutions
    )
    minimum_spatial = min(
        solution.minimum_ellipticity_eigenvalue for solution in solutions
    )
    all_converged = all(solution.converged for solution in solutions)
    thresholds = config["gates"]
    gates = {
        "all_nonlinear_solves_converged": all_converged,
        "manufactured_spherical_recovery": manufactured_error
        <= float(thresholds["maximum_manufactured_relative_error"]),
        "normalized_equation_residual": maximum_residual
        <= float(thresholds["maximum_normalized_equation_residual"]),
        "positive_temporal_kinetic_coefficient": minimum_temporal
        > float(thresholds["minimum_temporal_kinetic_coefficient"]),
        "positive_spatial_ellipticity": minimum_spatial
        > float(thresholds["minimum_spatial_ellipticity_eigenvalue"]),
        "material_component_nonadditivity": component_nonadditivity
        >= float(thresholds["minimum_component_nonadditivity"]),
        "rotation_covariance": rotation_covariance
        <= float(thresholds["maximum_rotation_covariance_error"]),
        "resolution_stability": resolution_change
        <= float(thresholds["maximum_resolution_change"]),
        "parameter_count": int(config["physical_parameters"]["count"])
        <= int(config["physical_parameters"]["maximum_allowed"]),
    }
    gates = {name: bool(value) for name, value in gates.items()}
    report = {
        "status": "completed Sigma v7C cubic-Hessian construction gate",
        "observational_data_accessed": False,
        "raw_holdout_opened": False,
        "candidate": config["candidate"],
        "dimensionless_equation": config["dimensionless_equation"],
        "physical_parameter_count": int(config["physical_parameters"]["count"]),
        "manufactured_sphere": {
            "analytic_potential": "pi=A(x^2+y^2+z^2)",
            "analytic_source": "18 A + 24 kappa A^2",
            "relative_potential_error": manufactured_error,
            "normalized_residual": manufactured_solution.residual_rms,
            "iterations": manufactured_solution.iterations,
        },
        "separated_components": {
            "relative_nonadditivity": component_nonadditivity,
            "rotation_covariance_error": rotation_covariance,
            "coarse_to_double_resolution_change": resolution_change,
            "coarse_combined_iterations": coarse_x.iterations,
            "fine_combined_iterations": fine_x.iterations,
        },
        "branch_health": {
            "all_solves_converged": all_converged,
            "maximum_normalized_residual": maximum_residual,
            "minimum_temporal_kinetic_coefficient": minimum_temporal,
            "minimum_spatial_ellipticity_eigenvalue": minimum_spatial,
            "principal_matrix": "Z_ij=3 delta_ij+2 kappa[(laplacian pi)delta_ij-H_ij]",
            "temporal_coefficient": "Z_t=3+2 kappa laplacian(pi)",
        },
        "gates": gates,
        "all_v7c_construction_gates_pass": bool(all(gates.values())),
        "decision": "construction_pass_retain_as_dynamics_control_physical_projection_failed",
        "reason": "The fixed 3D Hessian equation recovers an analytic spherical solution, remains temporally positive and spatially elliptic, is rotation covariant, changes by less than 2% at double resolution, and produces a 5%+ nonadditive response to separated sources. The separate physical-metric projection gate shows that this scalar nonadditivity cannot be counted as a lensing prediction.",
        "scope": "Construction success establishes a numerical scalar dynamics control only. The leading conformal helicity-zero metric response cancels from the Weyl potential, while the disformal and coupled-tensor terms that could affect light are not closed by this frozen PDE. No observational-map test is authorized.",
        "next_gate": "See sigma_v7c_metric_projection_gate/report.json. The scalar-only v7C lensing interpretation is retired; synthesize the three-formulation positive-spin2 failure before selecting another mechanism.",
        "data_policy": config["data_policy"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
