from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import j0, jn_zeros

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


def periodic_fft_manifest(
    dimensions: int, *, zero_mode_policy: str = "require_zero_mean"
) -> dict:
    manifest = manufactured_manifest(dimensions)
    manifest["name"] = "Manufactured periodic FFT field"
    manifest["fields"]["u"]["boundary"] = {"type": "periodic"}
    manifest["solver"] = {
        "family": "fft_poisson",
        "relativeTolerance": 1e-12,
        "residualTolerance": 1e-12,
        "maxIterations": 1,
        "periodicZeroMode": zero_mode_policy,
        "potentialGauge": "zero_mean",
        "zeroModeTolerance": 1e-12,
    }
    return manifest


@pytest.mark.parametrize(
    "shape,spacing,modes",
    [
        ((24, 20), (0.3, 0.7), (2, 3)),
        ((12, 10, 8), (0.25, 0.4, 0.8), (1, 2, 3)),
    ],
)
def test_periodic_fft_poisson_recovers_anisotropic_fourier_modes(
    shape: tuple[int, ...],
    spacing: tuple[float, ...],
    modes: tuple[int, ...],
) -> None:
    axes = [
        np.arange(count, dtype=float) * step
        for count, step in zip(shape, spacing, strict=True)
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    lengths = [count * step for count, step in zip(shape, spacing, strict=True)]
    factors = [
        np.sin(2.0 * np.pi * mode * coordinate / length)
        for mode, coordinate, length in zip(modes, mesh, lengths, strict=True)
    ]
    expected = np.prod(factors, axis=0)
    k_squared = sum(
        (2.0 * np.pi * mode / length) ** 2
        for mode, length in zip(modes, lengths, strict=True)
    )
    forcing = -k_squared * expected

    solution = solve_field_manifest(
        periodic_fft_manifest(len(shape)),
        {"forcing": forcing},
        spacing,
    )

    relative_error = np.linalg.norm(solution.fields["u"] - expected) / np.linalg.norm(expected)
    assert solution.converged
    assert solution.iterations == 1
    assert solution.maximum_relative_update == 0.0
    assert relative_error < 2e-12
    assert max(solution.equation_residuals.values()) < 2e-13
    metadata = solution.metadata["fft_poisson"]
    assert metadata["operator_convention"] == "continuum_fourier_laplacian"
    assert metadata["domain_lengths"] == pytest.approx(lengths)
    assert metadata["zero_mode_policy"] == "require_zero_mean"
    diagnostic = metadata["equations"][0]
    assert abs(diagnostic["potential_mean"]) < 1e-14
    assert diagnostic["energy_balance_relative_error"] < 2e-13
    assert diagnostic["relative_imaginary_leakage"] < 2e-13

    expected_first_gradient = (
        (2.0 * np.pi * modes[0] / lengths[0])
        * np.cos(2.0 * np.pi * modes[0] * mesh[0] / lengths[0])
        * np.prod(factors[1:], axis=0)
    )
    gradient_error = np.linalg.norm(
        solution.observables["gradient"][0] - expected_first_gradient
    ) / np.linalg.norm(expected_first_gradient)
    assert gradient_error < 2e-12


def test_periodic_fft_zero_mode_subtraction_is_explicit_and_reported() -> None:
    shape = (18, 14)
    spacing = (0.4, 0.7)
    x, y = np.meshgrid(
        np.arange(shape[0]) * spacing[0],
        np.arange(shape[1]) * spacing[1],
        indexing="ij",
    )
    expected = np.sin(2.0 * np.pi * x / (shape[0] * spacing[0])) * np.cos(
        4.0 * np.pi * y / (shape[1] * spacing[1])
    )
    k_squared = (2.0 * np.pi / (shape[0] * spacing[0])) ** 2 + (
        4.0 * np.pi / (shape[1] * spacing[1])
    ) ** 2
    offset = 7.5
    forcing = -k_squared * expected + offset

    solution = solve_field_manifest(
        periodic_fft_manifest(2, zero_mode_policy="subtract_mean"),
        {"forcing": forcing},
        spacing,
    )

    assert solution.converged
    assert np.linalg.norm(solution.fields["u"] - expected) / np.linalg.norm(expected) < 2e-12
    diagnostic = solution.metadata["fft_poisson"]["equations"][0]
    assert diagnostic["removed_source_mean"] == pytest.approx(offset)
    assert abs(diagnostic["effective_source_integral"]) < 1e-10
    assert abs(diagnostic["raw_source_integral"]) > 1.0


def test_periodic_fft_resolution_sensitivity_is_spectral_for_resolved_mode() -> None:
    errors = []
    amplitudes = []
    for cells in (12, 24, 48):
        spacing = 2.0 / cells
        axis = np.arange(cells, dtype=float) * spacing
        x, y = np.meshgrid(axis, axis, indexing="ij")
        expected = np.sin(2.0 * np.pi * x / 2.0) * np.cos(
            4.0 * np.pi * y / 2.0
        )
        k_squared = (2.0 * np.pi / 2.0) ** 2 + (4.0 * np.pi / 2.0) ** 2
        solution = solve_field_manifest(
            periodic_fft_manifest(2),
            {"forcing": -k_squared * expected},
            spacing,
        )
        errors.append(
            np.linalg.norm(solution.fields["u"] - expected)
            / np.linalg.norm(expected)
        )
        amplitudes.append(float(np.max(solution.fields["u"])))
    assert max(errors) < 2e-12
    assert max(amplitudes) - min(amplitudes) < 2e-12


def test_periodic_fft_even_grid_nyquist_policy_is_real_and_audited() -> None:
    shape = (16, 14)
    spacing = (0.3, 0.5)
    i = np.arange(shape[0], dtype=float)[:, None]
    y = np.arange(shape[1], dtype=float)[None, :] * spacing[1]
    expected = np.power(-1.0, i) * np.cos(
        2.0 * np.pi * y / (shape[1] * spacing[1])
    )
    k_squared = (np.pi / spacing[0]) ** 2 + (
        2.0 * np.pi / (shape[1] * spacing[1])
    ) ** 2
    solution = solve_field_manifest(
        periodic_fft_manifest(2),
        {"forcing": -k_squared * expected},
        spacing,
    )
    assert np.linalg.norm(solution.fields["u"] - expected) / np.linalg.norm(
        expected
    ) < 2e-12
    assert np.max(np.abs(solution.observables["gradient"][0])) < 1e-12
    metadata = solution.metadata["fft_poisson"]
    assert metadata["first_derivative_nyquist_policy"] == (
        "zero_for_real_nodal_derivative"
    )
    assert metadata["equations"][0]["energy_balance_relative_error"] < 2e-13


def test_periodic_fft_rejects_a_nonzero_mean_without_subtraction_policy() -> None:
    with pytest.raises(ValueError, match="not solvable on a periodic domain"):
        solve_field_manifest(
            periodic_fft_manifest(2),
            {"forcing": np.ones((9, 11), dtype=float)},
            (0.5, 0.8),
        )


def test_periodic_fft_rejects_nonperiodic_or_coupled_contracts() -> None:
    forcing = np.zeros((9, 9), dtype=float)
    manifest = periodic_fft_manifest(2)
    manifest["fields"]["u"]["boundary"] = {"type": "dirichlet", "value": 0.0}
    with pytest.raises(ValueError, match="requires a periodic boundary"):
        solve_field_manifest(manifest, {"forcing": forcing}, 1.0)

    manifest = periodic_fft_manifest(2)
    manifest["equations"][0]["rhs"] = {
        "op": "add",
        "args": [{"field": "forcing"}, {"field": "u"}],
    }
    with pytest.raises(ValueError, match="requires independent right-hand sides"):
        solve_field_manifest(manifest, {"forcing": forcing}, 1.0)


def test_periodic_fft_rejects_boundary_value_overrides() -> None:
    with pytest.raises(ValueError, match="cannot accept a boundary-value override"):
        solve_field_manifest(
            periodic_fft_manifest(2),
            {"forcing": np.zeros((9, 9), dtype=float)},
            1.0,
            boundary_values={"u": 0.0},
        )


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


def _axisymmetric_bessel_case(cells: int):
    manifest = manufactured_manifest(2)
    manifest["geometry"]["coordinateSystem"] = "axisymmetric_cylindrical"
    axis = np.linspace(0.0, 1.0, cells)
    spacing = float(axis[1] - axis[0])
    radius, vertical = np.meshgrid(axis, axis, indexing="ij")
    radial_mode = float(jn_zeros(0, 1)[0])
    expected = j0(radial_mode * radius) * np.sin(np.pi * vertical)
    forcing = -(radial_mode**2 + np.pi**2) * expected
    solution = solve_field_manifest(
        manifest,
        {"forcing": forcing},
        spacing,
        grid_geometry={"axisOrder": ["r", "z"], "origin": [0.0, 0.0]},
    )
    error = np.linalg.norm(solution.fields["u"] - expected) / np.linalg.norm(expected)
    return solution, error


def test_axisymmetric_bessel_solution_has_a_regular_axis() -> None:
    solution, relative_error = _axisymmetric_bessel_case(49)
    radial_acceleration, _vertical_acceleration = solution.observables["gradient"]
    assert solution.converged
    assert relative_error < 0.002
    assert max(solution.equation_residuals.values()) <= 1e-10
    assert np.array_equal(radial_acceleration[0, :], np.zeros(49))
    assert solution.metadata["axisymmetric_cylindrical"] == {
        "axis_order": ["r", "z"],
        "origin": [0.0, 0.0],
        "radial_axis_index": 0,
        "vertical_axis_index": 1,
        "axis_boundary": "zero_radial_flux_regularity",
        "outer_boundaries": "declared_dirichlet_or_isolated_approximation",
    }


def test_axisymmetric_bessel_solution_is_second_order_under_refinement() -> None:
    _coarse_solution, coarse_error = _axisymmetric_bessel_case(25)
    _fine_solution, fine_error = _axisymmetric_bessel_case(49)
    assert coarse_error / fine_error > 3.5


def test_axisymmetric_spatially_variable_coefficient_recovers_discrete_solution() -> None:
    cells = 33
    axis = np.linspace(0.0, 1.0, cells)
    spacing = float(axis[1] - axis[0])
    radius, vertical = np.meshgrid(axis, axis, indexing="ij")
    radial_mode = float(jn_zeros(0, 1)[0])
    expected = j0(radial_mode * radius) * np.sin(np.pi * vertical)
    coefficient = 1.0 + 0.5 * radius**2 + 0.2 * np.cos(np.pi * vertical)
    forcing, _flux_scale = _finite_volume_divergence_gradient(
        expected,
        coefficient,
        [spacing, spacing],
        coefficient_floor=1e-8,
        coordinate_system="axisymmetric_cylindrical",
    )
    manifest = manufactured_manifest(2)
    manifest["geometry"]["coordinateSystem"] = "axisymmetric_cylindrical"
    manifest["fields"]["permittivity"] = {
        "rank": "scalar",
        "role": "source",
        "unit": "1",
        "datasetKey": "permittivity",
    }
    manifest["dataRequirements"].append(
        {"key": "permittivity", "rank": "scalar", "unit": "1"}
    )
    manifest["equations"][0]["lhs"] = {
        "op": "divergence",
        "args": [
            {
                "op": "multiply",
                "args": [
                    {"field": "permittivity"},
                    {"op": "gradient", "args": [{"field": "u"}]},
                ],
            }
        ],
    }
    solution = solve_field_manifest(
        manifest,
        {"forcing": forcing, "permittivity": coefficient},
        spacing,
        grid_geometry={"axisOrder": ["r", "z"], "origin": [0.0, 0.0]},
    )
    relative_error = np.linalg.norm(solution.fields["u"] - expected) / np.linalg.norm(
        expected
    )
    assert solution.converged
    assert relative_error < 1e-11
    assert max(solution.equation_residuals.values()) <= 1e-10


@pytest.mark.parametrize(
    "grid_geometry,match",
    [
        (None, "requires grid geometry"),
        ({"axisOrder": ["z", "r"], "origin": [0.0, 0.0]}, "axisOrder"),
        ({"axisOrder": ["r", "z"], "origin": [1.0, 0.0]}, "radial origin"),
    ],
)
def test_axisymmetric_worker_rejects_ambiguous_grid_geometry(grid_geometry, match):
    manifest = manufactured_manifest(2)
    manifest["geometry"]["coordinateSystem"] = "axisymmetric_cylindrical"
    with pytest.raises(ValueError, match=match):
        solve_field_manifest(
            manifest,
            {"forcing": np.zeros((9, 9))},
            1.0,
            grid_geometry=grid_geometry,
        )


def test_axisymmetric_worker_rejects_cartesian_convolution_semantics() -> None:
    manifest = manufactured_manifest(2)
    manifest["geometry"]["coordinateSystem"] = "axisymmetric_cylindrical"
    manifest["observables"].append(
        {
            "id": "invalid_cartesian_convolution",
            "target": "diagnostic",
            "rank": "scalar",
            "unit": "1",
            "expression": {
                "op": "convolution",
                "args": [{"field": "forcing"}, {"field": "forcing"}],
            },
        }
    )
    manifest["solver"].update(
        {
            "family": "nonlocal_elliptic",
            "nonlocalBoundary": "zero_padded",
            "convolutionMode": "linear_same",
            "kernelOrigin": "centered_sample",
            "convolutionMeasure": "physical_volume",
        }
    )
    with pytest.raises(ValueError, match="explicit cylindrical kernel"):
        solve_field_manifest(
            manifest,
            {"forcing": np.zeros((9, 9))},
            1.0,
            grid_geometry={"axisOrder": ["r", "z"], "origin": [0.0, 0.0]},
        )


def test_worker_rejects_unsupported_coordinate_system():
    manifest = manufactured_manifest(2)
    manifest["geometry"]["coordinateSystem"] = "spherical_4d"
    with pytest.raises(ValueError, match="supports cartesian_2d"):
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


def test_published_two_potential_tree_separates_photons_and_matter() -> None:
    path = ROOT / "hosted-simulator" / "examples" / "models" / "two-potential.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    cells = 9
    spacing = 0.5 * 3.085677581491367e19
    coordinates = (np.arange(cells) - cells // 2) * spacing
    x, y, z = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    radius_squared = x**2 + y**2 + z**2
    density = 2.0e-21 * np.exp(-radius_squared / (2.0 * spacing**2))

    solution = solve_field_manifest(
        manifest,
        {"baryon_density": density},
        spacing,
    )

    psi = solution.fields["Psi"]
    phi = solution.fields["Phi"]
    matter = solution.observables["massive_tracer_acceleration"]
    photons = solution.observables["photon_lensing_acceleration"]
    assert solution.converged
    assert np.linalg.norm(psi) > 0.0
    assert np.linalg.norm(phi - 1.5 * psi) / np.linalg.norm(phi) < 1e-12
    for photon_component, matter_component in zip(photons, matter, strict=True):
        denominator = max(float(np.linalg.norm(photon_component)), np.finfo(float).tiny)
        assert np.linalg.norm(photon_component - 1.25 * matter_component) / denominator < 1e-12
    assert solution.metadata["solver_family"] == "coupled_elliptic"
    assert solution.metadata["equation_count"] == 2
    assert solution.metadata["solved_field_count"] == 2
    assert solution.metadata["multi_field_update_scheme"] == "sequential_gauss_seidel"
    assert solution.metadata["engine"] == "generic-divergence-field-worker-v1"


def test_two_fields_can_feed_back_into_each_other_without_theory_specific_code() -> None:
    cells = 33
    axis = np.linspace(0.0, 1.0, cells)
    spacing = float(axis[1] - axis[0])
    x, y = np.meshgrid(axis, axis, indexing="ij")
    expected_u = np.sin(np.pi * x) * np.sin(np.pi * y)
    expected_v = 0.5 * np.sin(2.0 * np.pi * x) * np.sin(np.pi * y)
    coupling = 0.25
    forcing_u = -2.0 * np.pi**2 * expected_u - coupling * expected_v
    forcing_v = -5.0 * np.pi**2 * expected_v - coupling * expected_u
    manifest = {
        "schemaVersion": "sigma-field-model/1",
        "name": "Coupled manufactured fields",
        "modelClass": "stationary_elliptic",
        "geometry": {
            "coordinateSystem": "cartesian_2d",
            "dimensions": 2,
            "domain": {"lengthUnit": "m"},
        },
        "fields": {
            "forcing_u": {"rank": "scalar", "role": "source", "unit": "1/s^2", "datasetKey": "forcing_u"},
            "forcing_v": {"rank": "scalar", "role": "source", "unit": "1/s^2", "datasetKey": "forcing_v"},
            "u": {"rank": "scalar", "role": "solved", "unit": "m^2/s^2", "boundary": {"type": "dirichlet", "value": 0.0}},
            "v": {"rank": "scalar", "role": "solved", "unit": "m^2/s^2", "boundary": {"type": "dirichlet", "value": 0.0}},
        },
        "parameters": {
            "lambda": {"unit": "1/m^2", "value": coupling, "scope": "universal"}
        },
        "equations": [
            {
                "id": "u_from_v",
                "kind": "equality",
                "lhs": {"op": "laplacian", "args": [{"field": "u"}]},
                "rhs": {"op": "add", "args": [{"field": "forcing_u"}, {"op": "multiply", "args": [{"parameter": "lambda"}, {"field": "v"}]}]},
            },
            {
                "id": "v_from_u",
                "kind": "equality",
                "lhs": {"op": "laplacian", "args": [{"field": "v"}]},
                "rhs": {"op": "add", "args": [{"field": "forcing_v"}, {"op": "multiply", "args": [{"parameter": "lambda"}, {"field": "u"}]}]},
            },
        ],
        "observables": [],
        "dataRequirements": [
            {"key": "forcing_u", "rank": "scalar", "unit": "1/s^2"},
            {"key": "forcing_v", "rank": "scalar", "unit": "1/s^2"},
        ],
        "solver": {
            "family": "coupled_elliptic",
            "relativeTolerance": 1e-10,
            "residualTolerance": 1e-10,
            "maxIterations": 80,
            "damping": 1.0,
        },
        "parameterPolicy": {"mode": "universal_fixed", "perObjectParameters": []},
    }

    solution = solve_field_manifest(
        manifest,
        {"forcing_u": forcing_u, "forcing_v": forcing_v},
        spacing,
    )

    relative_u = np.linalg.norm(solution.fields["u"] - expected_u) / np.linalg.norm(expected_u)
    relative_v = np.linalg.norm(solution.fields["v"] - expected_v) / np.linalg.norm(expected_v)
    assert solution.converged
    assert relative_u < 0.004
    assert relative_v < 0.005
    assert solution.iterations > 2
    assert solution.metadata["equation_count"] == 2
    assert solution.metadata["solved_field_count"] == 2
    assert solution.metadata["multi_field_update_scheme"] == "sequential_gauss_seidel"


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
