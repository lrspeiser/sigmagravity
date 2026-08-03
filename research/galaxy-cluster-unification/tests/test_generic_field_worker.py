from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from voidscreen.generic_field_worker import (
    _finite_volume_divergence_gradient,
    evaluate_field_expression,
    solve_field_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def manufactured_manifest(dimensions: int, *, variable_coefficient: bool = False) -> dict:
    coordinate = "cartesian_2d" if dimensions == 2 else "cartesian_3d"
    lhs = {"op": "laplacian", "args": [{"field": "u"}]}
    parameters = {}
    if variable_coefficient:
        parameters["k"] = {"unit": "1", "value": 2.5, "scope": "universal"}
        lhs = {
            "op": "divergence",
            "args": [{
                "op": "multiply",
                "args": [{"parameter": "k"}, {"op": "gradient", "args": [{"field": "u"}]}],
            }],
        }
    return {
        "schemaVersion": "sigma-field-model/1",
        "name": "Manufactured field",
        "modelClass": "stationary_elliptic",
        "geometry": {
            "coordinateSystem": coordinate,
            "dimensions": dimensions,
            "domain": {"lengthUnit": "m"},
        },
        "fields": {
            "forcing": {"rank": "scalar", "role": "source", "unit": "1/s^2", "datasetKey": "forcing"},
            "u": {"rank": "scalar", "role": "solved", "unit": "m^2/s^2", "boundary": {"type": "dirichlet", "value": 0.0}},
        },
        "parameters": parameters,
        "equations": [{"id": "manufactured", "kind": "equality", "lhs": lhs, "rhs": {"field": "forcing"}}],
        "observables": [{
            "id": "gradient",
            "target": "diagnostic",
            "rank": "vector",
            "unit": "m/s^2",
            "expression": {"op": "gradient", "args": [{"field": "u"}]},
        }],
        "dataRequirements": [{"key": "forcing", "rank": "scalar", "unit": "1/s^2"}],
        "solver": {"family": "finite_volume_elliptic", "relativeTolerance": 1e-10, "maxIterations": 8, "damping": 1.0},
        "parameterPolicy": {"mode": "universal_fixed", "perObjectParameters": []},
    }


@pytest.mark.parametrize("dimensions,cells", [(2, 33), (3, 17)])
def test_manufactured_sine_solution_in_two_and_three_dimensions(dimensions: int, cells: int):
    axes = [np.linspace(0.0, 1.0, cells) for _ in range(dimensions)]
    mesh = np.meshgrid(*axes, indexing="ij")
    expected = np.prod([np.sin(np.pi * coordinate) for coordinate in mesh], axis=0)
    forcing = -(dimensions * np.pi**2) * expected
    solution = solve_field_manifest(
        manufactured_manifest(dimensions),
        {"forcing": forcing},
        1.0 / (cells - 1),
    )
    relative_error = np.linalg.norm(solution.fields["u"] - expected) / np.linalg.norm(expected)
    assert solution.converged
    assert relative_error < (0.004 if dimensions == 2 else 0.015)
    assert max(solution.equation_residuals.values()) <= 1e-10
    assert solution.metadata["dimensions"] == dimensions
    assert len(solution.observables["gradient"]) == dimensions


def test_variable_coefficient_is_read_from_expression_not_theory_name():
    cells = 29
    axes = [np.linspace(0.0, 1.0, cells) for _ in range(2)]
    x, y = np.meshgrid(*axes, indexing="ij")
    expected = np.sin(np.pi * x) * np.sin(np.pi * y)
    coefficient = 2.5
    forcing = -coefficient * 2.0 * np.pi**2 * expected
    solution = solve_field_manifest(
        manufactured_manifest(2, variable_coefficient=True),
        {"forcing": forcing},
        1.0 / (cells - 1),
    )
    relative_error = np.linalg.norm(solution.fields["u"] - expected) / np.linalg.norm(expected)
    assert solution.converged
    assert relative_error < 0.005
    assert max(solution.equation_residuals.values()) <= 1e-10
    assert solution.metadata["engine"] == "generic-divergence-field-worker-v1"


def test_small_update_alone_cannot_claim_convergence():
    manifest = manufactured_manifest(2)
    manifest["solver"] = {
        "family": "finite_volume_elliptic",
        "relativeTolerance": 2.0,
        "residualTolerance": 1e-12,
        "maxIterations": 1,
        "damping": 0.5,
    }
    forcing = np.ones((9, 9), dtype=float)
    solution = solve_field_manifest(manifest, {"forcing": forcing}, 1.0)
    assert solution.maximum_relative_update <= 2.0
    assert max(solution.equation_residuals.values()) > 1e-12
    assert not solution.converged


def test_linearized_initialization_and_iteration_limit_are_disclosed() -> None:
    manifest = manufactured_manifest(2)
    manifest["solver"].update(
        {
            "initialization": "linearized_unit_coefficient",
            "maxIterations": 250,
        }
    )
    forcing = np.ones((9, 9), dtype=float)
    solution = solve_field_manifest(manifest, {"forcing": forcing}, 1.0)
    assert solution.converged
    assert solution.metadata["initialization"] == "linearized_unit_coefficient"
    assert solution.metadata["requested_maximum_iterations"] == 250
    assert solution.metadata["executed_maximum_iterations"] == 200
    assert solution.metadata["maximum_iterations_limited_by_worker"]


def test_newton_krylov_recovers_a_nonlinear_manufactured_field() -> None:
    cells = 17
    axis = np.linspace(0.0, 1.0, cells)
    spacing = float(axis[1] - axis[0])
    x, y = np.meshgrid(axis, axis, indexing="ij")
    expected = np.sin(np.pi * x) * np.sin(np.pi * y)
    gradient = np.gradient(expected, spacing, spacing, edge_order=2)
    coefficient = 1.0 + 0.25 * np.sqrt(sum(np.square(value) for value in gradient))
    forcing, _scale = _finite_volume_divergence_gradient(
        expected, coefficient, [spacing, spacing], coefficient_floor=1e-8
    )
    manifest = manufactured_manifest(2)
    manifest["parameters"] = {
        "beta": {"unit": "1", "value": 0.25, "scope": "universal"}
    }
    manifest["equations"][0]["lhs"] = {
        "op": "divergence",
        "args": [
            {
                "op": "multiply",
                "args": [
                    {
                        "op": "add",
                        "args": [
                            {"const": 1.0},
                            {
                                "op": "multiply",
                                "args": [
                                    {"parameter": "beta"},
                                    {
                                        "op": "norm",
                                        "args": [
                                            {
                                                "op": "gradient",
                                                "args": [{"field": "u"}],
                                            }
                                        ],
                                    },
                                ],
                            },
                        ],
                    },
                    {"op": "gradient", "args": [{"field": "u"}]},
                ],
            }
        ],
    }
    manifest["solver"].update(
        {
            "initialization": "linearized_unit_coefficient",
            "nonlinearMethod": "newton_krylov",
            "maxIterations": 80,
            "residualTolerance": 1e-8,
            "lineSearch": "armijo",
            "picardWarmupIterations": 5,
            "picardWarmupDamping": 0.2,
        }
    )
    solution = solve_field_manifest(manifest, {"forcing": forcing}, spacing)
    relative_error = np.linalg.norm(solution.fields["u"] - expected) / np.linalg.norm(
        expected
    )
    assert solution.converged
    assert relative_error < 1e-8
    assert solution.metadata["nonlinear_method"] == "newton_krylov"
    assert solution.metadata["krylov_method"] == "lgmres"
    assert solution.metadata["picard_warmup_iterations"] == 5
    assert solution.metadata["picard_warmup_damping"] == 0.2
    assert solution.iterations > 5


def test_zero_source_harmonic_boundary_has_a_well_scaled_residual():
    cells = 17
    axis = np.linspace(-1.0, 1.0, cells)
    expected = np.broadcast_to(axis[:, None], (cells, cells)).copy()
    solution = solve_field_manifest(
        manufactured_manifest(2),
        {"forcing": np.zeros_like(expected)},
        float(axis[1] - axis[0]),
        boundary_values={"u": expected},
    )
    assert solution.converged
    assert max(solution.equation_residuals.values()) <= 1e-10
    assert np.allclose(solution.fields["u"], expected, rtol=0.0, atol=1e-12)


def test_worker_rejects_unsupported_coordinate_system():
    manifest = manufactured_manifest(2)
    manifest["geometry"]["coordinateSystem"] = "axisymmetric_cylindrical"
    with pytest.raises(ValueError, match="cartesian_2d"):
        solve_field_manifest(manifest, {"forcing": np.zeros((9, 9))}, 1.0)


def test_published_refracted_gravity_tree_runs_without_a_theory_specific_branch():
    path = ROOT / "hosted-simulator" / "examples" / "models" / "refracted-gravity.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["geometry"]["coordinateSystem"] = "cartesian_2d"
    manifest["geometry"]["dimensions"] = 2
    manifest["solver"].update({"maxIterations": 4, "damping": 1.0})
    cells = 17
    coordinates = np.linspace(-4.0, 4.0, cells)
    x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
    density = 1.0e-27 + 2.0e-21 * np.exp(-(x**2 + y**2))
    solution = solve_field_manifest(
        manifest,
        {"baryon_density": density},
        0.5 * 3.085677581491367e19,
    )
    assert solution.converged
    assert solution.iterations == 2
    assert np.all(np.isfinite(solution.fields["Phi"]))
    assert len(solution.observables["massive_tracer_acceleration"]) == 2
    assert solution.metadata["engine"] == "generic-divergence-field-worker-v1"


def test_explicit_zero_vector_limit_resolves_singular_isotropic_flux() -> None:
    zeros = np.zeros((9, 9), dtype=float)
    expression = {
        "op": "multiply_zero_vector_limit",
        "args": [
            {
                "op": "divide",
                "args": [
                    {"const": 1.0},
                    {
                        "op": "norm",
                        "args": [
                            {"op": "gradient", "args": [{"field": "potential"}]}
                        ],
                    },
                ],
            },
            {"op": "gradient", "args": [{"field": "potential"}]},
        ],
    }
    flux = evaluate_field_expression(
        expression,
        fields={"potential": zeros},
        parameters={},
        spacing=[1.0, 1.0],
    )
    assert isinstance(flux, tuple)
    assert all(np.array_equal(component, zeros) for component in flux)


def test_published_qumond_tree_runs_without_a_theory_specific_branch() -> None:
    path = ROOT / "hosted-simulator" / "examples" / "models" / "qumond.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["geometry"]["coordinateSystem"] = "cartesian_2d"
    manifest["geometry"]["dimensions"] = 2
    manifest["solver"].update({"maxIterations": 8, "damping": 1.0})
    cells = 17
    coordinates = np.linspace(-4.0, 4.0, cells)
    x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
    density = 1.0e-27 + 2.0e-21 * np.exp(-(x**2 + y**2))
    solution = solve_field_manifest(
        manifest,
        {"baryon_density": density},
        0.5 * 3.085677581491367e19,
    )
    assert solution.converged
    assert solution.iterations == 2
    assert all(np.all(np.isfinite(value)) for value in solution.fields.values())
    assert len(solution.observables["massive_tracer_acceleration"]) == 2
    assert solution.metadata["engine"] == "generic-divergence-field-worker-v1"


def test_nonlocal_convolution_is_a_physical_linear_integral_without_wraparound() -> None:
    cells = 9
    spacing = [2.0, 3.0]
    cell_area = spacing[0] * spacing[1]
    center = cells // 2
    kernel = np.zeros((cells, cells), dtype=float)
    kernel[center, center] = 2.0
    kernel[center + 1, center] = 0.5
    kernel[center, center - 1] = 0.25

    centered_delta = np.zeros_like(kernel)
    centered_delta[center, center] = 1.0 / cell_area
    centered = evaluate_field_expression(
        {
            "op": "convolution",
            "args": [{"field": "source"}, {"field": "kernel"}],
        },
        fields={"source": centered_delta, "kernel": kernel},
        parameters={},
        spacing=spacing,
    )
    assert np.allclose(centered, kernel, rtol=0.0, atol=1e-12)

    corner_delta = np.zeros_like(kernel)
    corner_delta[0, 0] = 1.0 / cell_area
    corner = evaluate_field_expression(
        {
            "op": "convolution",
            "args": [{"field": "source"}, {"field": "kernel"}],
        },
        fields={"source": corner_delta, "kernel": kernel},
        parameters={},
        spacing=spacing,
    )
    assert abs(float(corner[-1, -1])) < 1e-12
    assert np.isclose(corner[0, 0], kernel[center, center], rtol=0.0, atol=1e-12)


def test_nonlocal_worker_requires_declared_convolution_semantics() -> None:
    path = ROOT / "hosted-simulator" / "examples" / "models" / "nonlocal-response.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    del manifest["solver"]["nonlocalBoundary"]
    source = np.zeros((5, 5, 5), dtype=float)
    kernel = np.zeros_like(source)
    kernel[2, 2, 2] = 1.0
    with pytest.raises(
        ValueError,
        match="convolution requires solver.nonlocalBoundary='zero_padded'",
    ):
        solve_field_manifest(
            manifest,
            {"baryon_density": source, "response_kernel": kernel},
            1.0,
        )


def test_published_nonlocal_response_runs_without_a_theory_specific_branch() -> None:
    path = ROOT / "hosted-simulator" / "examples" / "models" / "nonlocal-response.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["solver"].update({"maxIterations": 4, "damping": 1.0})
    cells = 9
    spacing = 0.5 * 3.085677581491367e19
    coordinates = (np.arange(cells) - cells // 2) * spacing
    x, y, z = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    radius_squared = x**2 + y**2 + z**2
    density = 2.0e-21 * np.exp(-radius_squared / (2.0 * spacing**2))
    kernel = np.exp(-radius_squared / (2.0 * (1.5 * spacing) ** 2))
    kernel /= float(np.sum(kernel) * spacing**3)

    solution = solve_field_manifest(
        manifest,
        {"baryon_density": density, "response_kernel": kernel},
        spacing,
    )
    assert solution.converged
    assert np.all(np.isfinite(solution.fields["Phi"]))
    assert np.linalg.norm(solution.fields["Phi"]) > 0.0
    assert len(solution.observables["massive_tracer_acceleration"]) == 3
    metadata = solution.metadata["nonlocal_convolution"]
    assert metadata == {
        "boundary": "zero_padded",
        "mode": "linear_same",
        "kernel_origin": "centered_sample",
        "measure": "physical_volume",
        "cell_volume": metadata["cell_volume"],
        "periodic_wraparound": False,
        "automatic_kernel_normalization": False,
    }
    assert np.isclose(metadata["cell_volume"], spacing**3, rtol=1e-15)
