"""Formula-independent field execution for typed field-model manifests.

This module is the local scientific-worker prototype behind the hosted model
contract.  It intentionally executes a small, auditable equation language
instead of dispatching on theory names.  Version one solves scalar elliptic
equations of the forms::

    laplacian(phi) = source
    divergence(coefficient * gradient(phi)) = source

on uniform two- or three-dimensional Cartesian grids and uniform
axisymmetric cylindrical ``(r,z)`` grids. Coefficients and sources are
evaluated from the submitted expression tree, so the same path can represent
Poisson gravity, density-dependent Refracted Gravity, AQUAL-like nonlinear
equations, and coupled QUMOND-like equations. The cylindrical axis is a
regularity boundary with zero radial flux, never a fabricated Dirichlet wall.

The same manifest path also provides a direct periodic FFT Poisson solver on
uniform Cartesian 2-D and 3-D grids.  Its torus zero mode and potential gauge
are explicit model controls; a non-solvable mean source is never silently
discarded unless the confirmed manifest requests mean subtraction.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np
from scipy import sparse
from scipy.optimize import NoConvergence, anderson, newton_krylov
from scipy.signal import fftconvolve
from scipy.sparse.linalg import spsolve

Array = np.ndarray
ExpressionValue = float | Array | tuple[Array, ...]
PREVIEW_MAXIMUM_ITERATIONS = 200


@dataclass(frozen=True)
class GenericFieldSolution:
    fields: dict[str, Array]
    observables: dict[str, ExpressionValue]
    converged: bool
    iterations: int
    maximum_relative_update: float
    equation_residuals: dict[str, float]
    residual_history: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def _spacing(values: float | Sequence[float], dimensions: int) -> tuple[float, ...]:
    if np.isscalar(values):
        result = (float(values),) * dimensions
    else:
        result = tuple(float(value) for value in values)
    if len(result) != dimensions or any(not math.isfinite(value) or value <= 0 for value in result):
        raise ValueError(f"spacing must contain {dimensions} finite positive values")
    return result


def _solution_region(
    coordinate_system: str, shape: tuple[int, ...]
) -> tuple[slice, ...]:
    if coordinate_system == "axisymmetric_cylindrical":
        if len(shape) != 2:
            raise ValueError("axisymmetric_cylindrical requires a 2D (r,z) grid")
        return (slice(0, -1), slice(1, -1))
    return tuple(slice(1, -1) for _ in shape)


def _axisymmetric_grid_metadata(
    grid_geometry: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(grid_geometry, Mapping):
        raise ValueError(
            "axisymmetric_cylindrical execution requires grid geometry with axisOrder=['r','z'] and origin=[0,z0]"
        )
    axis_order = list(grid_geometry.get("axisOrder", []))
    if axis_order != ["r", "z"]:
        raise ValueError("axisymmetric_cylindrical requires axisOrder=['r','z']")
    raw_origin = grid_geometry.get("origin")
    if (
        not isinstance(raw_origin, Sequence)
        or isinstance(raw_origin, (str, bytes))
        or len(raw_origin) != 2
    ):
        raise ValueError("axisymmetric_cylindrical requires origin=[0,z0]")
    origin = [float(value) for value in raw_origin]
    if any(not math.isfinite(value) for value in origin) or origin[0] != 0.0:
        raise ValueError("axisymmetric_cylindrical radial origin must be exactly r=0")
    return {
        "axis_order": axis_order,
        "origin": origin,
        "radial_axis_index": 0,
        "vertical_axis_index": 1,
        "axis_boundary": "zero_radial_flux_regularity",
        "outer_boundaries": "declared_dirichlet_or_isolated_approximation",
    }


def _axisymmetric_gradient(
    field: Array, spacing: Sequence[float]
) -> tuple[Array, Array]:
    values = np.asarray(field, dtype=float)
    if values.ndim != 2:
        raise ValueError("axisymmetric gradient requires a 2D (r,z) scalar field")
    radial, vertical = np.gradient(values, *spacing, edge_order=2)
    radial = np.asarray(radial, dtype=float)
    radial[0, :] = 0.0
    return radial, np.asarray(vertical, dtype=float)


def _axisymmetric_divergence(
    vector: tuple[Array, ...], spacing: Sequence[float]
) -> Array:
    if len(vector) != 2:
        raise ValueError("axisymmetric divergence requires radial and vertical components")
    radial_component = np.asarray(vector[0], dtype=float)
    vertical_component = np.asarray(vector[1], dtype=float)
    if radial_component.shape != vertical_component.shape or radial_component.ndim != 2:
        raise ValueError("axisymmetric vector components must share one 2D (r,z) grid")
    radial_scale = max(
        float(np.max(np.abs(radial_component))), np.finfo(float).tiny
    )
    if float(np.max(np.abs(radial_component[0, :]))) > 1e-10 * radial_scale:
        raise ValueError(
            "axisymmetric radial vector component must vanish at r=0"
        )
    dr, dz = (float(value) for value in spacing)
    radii = np.arange(radial_component.shape[0], dtype=float) * dr
    weighted_radial = radii[:, None] * radial_component
    radial_divergence = np.zeros_like(radial_component)
    radial_derivative = np.gradient(weighted_radial, dr, axis=0, edge_order=2)
    radial_divergence[1:, :] = radial_derivative[1:, :] / radii[1:, None]
    radial_divergence[0, :] = 2.0 * (
        radial_component[1, :] - radial_component[0, :]
    ) / dr
    vertical_divergence = np.gradient(
        vertical_component, dz, axis=1, edge_order=2
    )
    return radial_divergence + vertical_divergence


def _scalar_array(value: ExpressionValue, shape: tuple[int, ...], name: str) -> Array:
    if isinstance(value, tuple):
        raise TypeError(f"{name} must be scalar")
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(shape, float(array), dtype=float)
    if array.shape != shape or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def _binary(operation: str, left: ExpressionValue, right: ExpressionValue) -> ExpressionValue:
    if isinstance(left, tuple) or isinstance(right, tuple):
        if operation not in {"add", "subtract", "multiply", "divide"}:
            raise ValueError(f"{operation} does not accept vector fields")
        if isinstance(left, tuple) and isinstance(right, tuple):
            if len(left) != len(right) or operation in {"multiply", "divide"}:
                raise ValueError(f"{operation} cannot combine these vector fields")
            function = np.add if operation == "add" else np.subtract
            return tuple(function(a, b) for a, b in zip(left, right, strict=True))
        vector = left if isinstance(left, tuple) else right
        scalar = right if isinstance(left, tuple) else left
        if operation == "add":
            return tuple(np.add(component, scalar) for component in vector)
        if operation == "subtract":
            if isinstance(left, tuple):
                return tuple(np.subtract(component, scalar) for component in vector)
            return tuple(np.subtract(scalar, component) for component in vector)
        if operation == "multiply":
            return tuple(np.multiply(component, scalar) for component in vector)
        if isinstance(left, tuple):
            return tuple(np.divide(component, scalar) for component in vector)
        raise ValueError("a scalar cannot be divided by a vector")
    functions = {
        "add": np.add,
        "subtract": np.subtract,
        "multiply": np.multiply,
        "divide": np.divide,
        "min": np.minimum,
        "max": np.maximum,
    }
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        return functions[operation](left, right)


def _multiply_zero_vector_limit(
    left: ExpressionValue, right: ExpressionValue
) -> tuple[Array, ...]:
    """Scale a vector while explicitly defining a singular zero-vector limit.

    This operator is opt-in because an expression such as ``f(|v|) v`` may
    contain a coefficient that diverges at ``v=0``. The manifest author uses
    it only when the complete vector flux is mathematically defined to be zero
    there. Non-finite coefficients at non-zero vectors remain non-finite and
    are rejected by the consuming field equation.
    """

    if isinstance(left, tuple) == isinstance(right, tuple):
        raise ValueError(
            "multiply_zero_vector_limit requires exactly one vector and one scalar"
        )
    vector = left if isinstance(left, tuple) else right
    scalar = right if isinstance(left, tuple) else left
    scalar_values = np.asarray(scalar, dtype=float)
    magnitude_squared = sum(np.square(component) for component in vector)
    zero_limit = (magnitude_squared == 0.0) & ~np.isfinite(scalar_values)
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        products = tuple(np.multiply(component, scalar_values) for component in vector)
    return tuple(np.where(zero_limit, 0.0, component) for component in products)


def _linear_same_convolution(
    field: ExpressionValue,
    kernel: ExpressionValue,
    spacing: Sequence[float],
) -> ExpressionValue:
    """Evaluate a centered nonlocal integral without periodic wraparound.

    The kernel is sampled on the same odd Cartesian grid as the field, with its
    origin at the central sample. ``fftconvolve(..., mode="same")`` computes the
    linear convolution by zero-padding outside the submitted domain. Multiplying
    by the physical cell volume turns the discrete sum into the manifest's
    declared spatial integral.
    """

    if isinstance(kernel, tuple):
        raise TypeError("convolution kernel must be scalar")
    kernel_values = np.asarray(kernel, dtype=float)
    dimensions = len(spacing)
    if kernel_values.ndim != dimensions:
        raise ValueError(
            f"convolution kernel must have {dimensions} spatial dimensions"
        )
    if any(size < 3 or size % 2 == 0 for size in kernel_values.shape):
        raise ValueError(
            "convolution kernel requires at least three odd samples per dimension "
            "so its centered origin is unambiguous"
        )
    if np.any(~np.isfinite(kernel_values)):
        raise ValueError("convolution kernel must be finite")
    cell_volume = float(np.prod(np.asarray(spacing, dtype=float)))

    def apply(values: float | Array) -> Array:
        array = np.asarray(values, dtype=float)
        if array.shape != kernel_values.shape or np.any(~np.isfinite(array)):
            raise ValueError(
                "convolution field and kernel must be finite arrays with matching shapes"
            )
        result = fftconvolve(array, kernel_values, mode="same") * cell_volume
        if result.shape != array.shape or np.any(~np.isfinite(result)):
            raise RuntimeError("convolution returned a non-finite or mis-shaped field")
        return np.asarray(result, dtype=float)

    if isinstance(field, tuple):
        return tuple(apply(component) for component in field)
    return apply(field)


def _periodic_wavenumber_components(
    shape: Sequence[int], spacing: Sequence[float]
) -> tuple[tuple[Array, ...], Array]:
    """Return broadcast wave-number components and ``|k|^2`` for a periodic grid."""

    if len(shape) != len(spacing):
        raise ValueError("periodic wave numbers require one spacing per dimension")
    components: list[Array] = []
    k_squared = np.zeros(tuple(int(value) for value in shape), dtype=float)
    for axis, (count, step) in enumerate(zip(shape, spacing, strict=True)):
        frequency = 2.0 * np.pi * np.fft.fftfreq(int(count), d=float(step))
        broadcast_shape = [1] * len(shape)
        broadcast_shape[axis] = int(count)
        component = frequency.reshape(broadcast_shape)
        components.append(component)
        k_squared += np.square(component)
    return tuple(components), k_squared


def _periodic_first_derivative_components(
    shape: Sequence[int], spacing: Sequence[float]
) -> tuple[Array, ...]:
    """Return real-field derivative wave numbers with even-grid Nyquist modes zeroed."""

    components: list[Array] = []
    for axis, (count, step) in enumerate(zip(shape, spacing, strict=True)):
        frequency = 2.0 * np.pi * np.fft.fftfreq(int(count), d=float(step))
        if int(count) % 2 == 0:
            frequency[int(count) // 2] = 0.0
        broadcast_shape = [1] * len(shape)
        broadcast_shape[axis] = int(count)
        components.append(frequency.reshape(broadcast_shape))
    return tuple(components)


def _periodic_spectral_gradient(
    field: Array, spacing: Sequence[float]
) -> tuple[Array, ...]:
    values = np.asarray(field, dtype=float)
    components = _periodic_first_derivative_components(values.shape, spacing)
    transformed = np.fft.fftn(values)
    return tuple(
        np.asarray(np.fft.ifftn(1j * component * transformed).real, dtype=float)
        for component in components
    )


def _periodic_spectral_divergence(
    vector: tuple[Array, ...], spacing: Sequence[float]
) -> Array:
    if len(vector) != len(spacing):
        raise ValueError("periodic divergence requires one component per dimension")
    shape = np.asarray(vector[0]).shape
    if any(np.asarray(component).shape != shape for component in vector):
        raise ValueError("periodic vector components must share one grid shape")
    components = _periodic_first_derivative_components(shape, spacing)
    transformed = sum(
        1j * wave_number * np.fft.fftn(np.asarray(component, dtype=float))
        for wave_number, component in zip(components, vector, strict=True)
    )
    return np.asarray(np.fft.ifftn(transformed).real, dtype=float)


def _periodic_spectral_laplacian(field: Array, spacing: Sequence[float]) -> Array:
    values = np.asarray(field, dtype=float)
    _components, k_squared = _periodic_wavenumber_components(values.shape, spacing)
    return np.asarray(
        np.fft.ifftn(-k_squared * np.fft.fftn(values)).real,
        dtype=float,
    )


def evaluate_field_expression(
    node: Mapping[str, Any],
    *,
    fields: Mapping[str, Array],
    parameters: Mapping[str, float],
    spacing: Sequence[float],
    coordinate_system: str | None = None,
    differential_scheme: str = "finite_difference",
) -> ExpressionValue:
    """Evaluate a validated field-expression tree using NumPy operations."""

    if "const" in node:
        return float(node["const"])
    if "field" in node:
        try:
            return fields[str(node["field"])]
        except KeyError as error:
            raise ValueError(f"missing field data: {node['field']}") from error
    if "parameter" in node:
        try:
            return float(parameters[str(node["parameter"])] )
        except KeyError as error:
            raise ValueError(f"missing parameter value: {node['parameter']}") from error

    operation = str(node.get("op", ""))
    arguments = node.get("args", [])
    values = [
        evaluate_field_expression(
            argument,
            fields=fields,
            parameters=parameters,
            spacing=spacing,
            coordinate_system=coordinate_system,
            differential_scheme=differential_scheme,
        )
        for argument in arguments
    ]
    if operation in {"add", "subtract", "min", "max", "divide"}:
        return _binary(operation, values[0], values[1])
    if operation == "multiply":
        result = values[0]
        for value in values[1:]:
            result = _binary(operation, result, value)
        return result
    if operation == "multiply_zero_vector_limit":
        if len(values) != 2:
            raise ValueError("multiply_zero_vector_limit requires exactly two arguments")
        return _multiply_zero_vector_limit(values[0], values[1])
    if operation == "negate":
        return tuple(-value for value in values[0]) if isinstance(values[0], tuple) else -values[0]
    if operation == "norm":
        if not isinstance(values[0], tuple):
            return np.abs(values[0])
        return np.sqrt(sum(np.square(component) for component in values[0]))
    if operation == "gradient":
        if isinstance(values[0], tuple):
            raise ValueError("generic worker v1 does not take the gradient of a vector")
        if coordinate_system == "axisymmetric_cylindrical":
            return _axisymmetric_gradient(np.asarray(values[0], dtype=float), spacing)
        if differential_scheme == "periodic_spectral":
            return _periodic_spectral_gradient(np.asarray(values[0], dtype=float), spacing)
        return tuple(np.gradient(values[0], *spacing, edge_order=2))
    if operation == "divergence":
        if not isinstance(values[0], tuple) or len(values[0]) != len(spacing):
            raise ValueError("divergence requires one vector component per grid dimension")
        if coordinate_system == "axisymmetric_cylindrical":
            return _axisymmetric_divergence(values[0], spacing)
        if differential_scheme == "periodic_spectral":
            return _periodic_spectral_divergence(values[0], spacing)
        return sum(
            np.gradient(component, spacing[axis], axis=axis, edge_order=2)
            for axis, component in enumerate(values[0])
        )
    if operation == "laplacian":
        if isinstance(values[0], tuple):
            raise ValueError("generic worker v1 expects a scalar Laplacian")
        if coordinate_system == "axisymmetric_cylindrical":
            result, _flux_scale = _finite_volume_divergence_gradient(
                np.asarray(values[0], dtype=float),
                np.ones_like(values[0], dtype=float),
                spacing,
                coefficient_floor=np.finfo(float).tiny,
                coordinate_system=coordinate_system,
            )
            return result
        if differential_scheme == "periodic_spectral":
            return _periodic_spectral_laplacian(
                np.asarray(values[0], dtype=float), spacing
            )
        result = np.zeros_like(values[0], dtype=float)
        for axis, step in enumerate(spacing):
            first = np.gradient(values[0], step, axis=axis, edge_order=2)
            result += np.gradient(first, step, axis=axis, edge_order=2)
        return result
    if operation == "dot":
        if not isinstance(values[0], tuple) or not isinstance(values[1], tuple):
            raise ValueError("dot requires vectors")
        return sum(a * b for a, b in zip(values[0], values[1], strict=True))
    if operation == "pow":
        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            return np.power(values[0], values[1])
    if operation == "sqrt":
        with np.errstate(invalid="ignore"):
            return np.sqrt(values[0])
    if operation == "exp":
        return np.exp(values[0])
    if operation == "log":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(values[0])
    if operation == "tanh":
        return np.tanh(values[0])
    if operation == "smoothstep":
        clipped = np.clip(values[0], 0.0, 1.0)
        return clipped * clipped * (3.0 - 2.0 * clipped)
    if operation in {"lt", "lte", "gt", "gte"}:
        functions = {"lt": np.less, "lte": np.less_equal, "gt": np.greater, "gte": np.greater_equal}
        return functions[operation](values[0], values[1]).astype(float)
    if operation == "convolution":
        if len(values) != 2:
            raise ValueError("convolution requires exactly two arguments")
        return _linear_same_convolution(values[0], values[1], spacing)
    if operation == "piecewise":
        result = evaluate_field_expression(
            node["otherwise"],
            fields=fields,
            parameters=parameters,
            spacing=spacing,
            coordinate_system=coordinate_system,
            differential_scheme=differential_scheme,
        )
        if isinstance(result, tuple):
            raise ValueError("vector piecewise expressions are not supported in worker v1")
        for branch in reversed(node["branches"]):
            condition = evaluate_field_expression(
                branch["when"],
                fields=fields,
                parameters=parameters,
                spacing=spacing,
                coordinate_system=coordinate_system,
                differential_scheme=differential_scheme,
            )
            value = evaluate_field_expression(
                branch["value"],
                fields=fields,
                parameters=parameters,
                spacing=spacing,
                coordinate_system=coordinate_system,
                differential_scheme=differential_scheme,
            )
            result = np.where(condition, value, result)
        return result
    raise ValueError(f"operator {operation!r} is validated but not executable by worker v1")


def _field_from_gradient(node: Mapping[str, Any]) -> str | None:
    if node.get("op") != "gradient" or len(node.get("args", [])) != 1:
        return None
    child = node["args"][0]
    return str(child["field"]) if isinstance(child, Mapping) and "field" in child else None


def _elliptic_lhs(node: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    if node.get("op") == "laplacian":
        arguments = node.get("args", [])
        if len(arguments) == 1 and isinstance(arguments[0], Mapping) and "field" in arguments[0]:
            return str(arguments[0]["field"]), {"const": 1.0}
    if node.get("op") != "divergence" or len(node.get("args", [])) != 1:
        raise ValueError("worker v1 lhs must be laplacian(field) or divergence(coefficient*gradient(field))")
    flux = node["args"][0]
    direct = _field_from_gradient(flux)
    if direct:
        return direct, {"const": 1.0}
    if not isinstance(flux, Mapping) or flux.get("op") != "multiply":
        raise ValueError("worker v1 divergence lhs requires a scalar coefficient times a gradient")
    factors = list(flux.get("args", []))
    targets = [(index, _field_from_gradient(factor)) for index, factor in enumerate(factors)]
    targets = [(index, target) for index, target in targets if target]
    if len(targets) != 1:
        raise ValueError("worker v1 divergence lhs requires exactly one gradient of a solved field")
    target_index, target = targets[0]
    coefficient_factors = [factor for index, factor in enumerate(factors) if index != target_index]
    coefficient: Mapping[str, Any]
    if len(coefficient_factors) == 1:
        coefficient = coefficient_factors[0]
    else:
        coefficient = {"op": "multiply", "args": coefficient_factors}
    return target, coefficient


def _direct_laplacian_target(node: Mapping[str, Any]) -> str | None:
    """Return the scalar field in an exact ``laplacian(field)`` expression."""

    if node.get("op") != "laplacian" or len(node.get("args", [])) != 1:
        return None
    child = node["args"][0]
    if not isinstance(child, Mapping) or set(child) != {"field"}:
        return None
    return str(child["field"])


def _referenced_field_names(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        names = {str(value["field"])} if "field" in value else set()
        for child in value.values():
            names.update(_referenced_field_names(child))
        return names
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        names: set[str] = set()
        for child in value:
            names.update(_referenced_field_names(child))
        return names
    return set()


def _solve_periodic_fft_poisson_equations(
    *,
    equations: Sequence[Mapping[str, Any]],
    fields: dict[str, Array],
    solved_names: Sequence[str],
    parameters: Mapping[str, float],
    spacing: Sequence[float],
    shape: tuple[int, ...],
    solver: Mapping[str, Any],
    residual_tolerance: float,
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, Any]]:
    """Directly solve independent periodic scalar Poisson equations by FFT."""

    ignored_controls = sorted(
        key
        for key in {
            "damping",
            "coefficientFloor",
            "initialization",
            "nonlinearMethod",
            "lineSearch",
            "andersonAlpha",
            "andersonHistory",
            "andersonRegularization",
            "krylovMethod",
            "krylovInnerIterations",
            "picardWarmupIterations",
            "picardWarmupDamping",
            "nonlocalBoundary",
            "convolutionMode",
            "kernelOrigin",
            "convolutionMeasure",
        }
        if key in solver
    )
    if ignored_controls:
        raise ValueError(
            "fft_poisson cannot declare iterative or nonlocal controls that it "
            f"would ignore: {', '.join(ignored_controls)}"
        )
    zero_mode_policy = str(solver.get("periodicZeroMode", ""))
    if zero_mode_policy not in {"require_zero_mean", "subtract_mean"}:
        raise ValueError(
            "fft_poisson requires solver.periodicZeroMode=require_zero_mean or subtract_mean"
        )
    if solver.get("potentialGauge") != "zero_mean":
        raise ValueError("fft_poisson requires solver.potentialGauge='zero_mean'")
    zero_mode_tolerance = float(solver.get("zeroModeTolerance", math.nan))
    if not math.isfinite(zero_mode_tolerance) or not 0 < zero_mode_tolerance < 1:
        raise ValueError("fft_poisson zeroModeTolerance must lie in (0,1)")
    if int(solver.get("maxIterations", 0)) != 1:
        raise ValueError("fft_poisson is a direct solve and requires maxIterations=1")

    solved_set = set(str(name) for name in solved_names)
    _components, k_squared = _periodic_wavenumber_components(shape, spacing)
    nonzero = k_squared > 0.0
    nonzero_wavenumbers = np.sqrt(k_squared[nonzero])
    cell_volume = float(np.prod(np.asarray(spacing, dtype=float)))
    domain_volume = cell_volume * float(np.prod(shape))
    residuals: dict[str, float] = {}
    equation_diagnostics: list[dict[str, Any]] = []

    for equation in equations:
        equation_id = str(equation["id"])
        target = _direct_laplacian_target(equation["lhs"])
        if target is None:
            raise ValueError(
                f"fft_poisson equation {equation_id} lhs must be exactly laplacian(solved_field)"
            )
        solved_dependencies = sorted(
            _referenced_field_names(equation["rhs"]) & solved_set
        )
        if solved_dependencies:
            raise ValueError(
                "fft_poisson v1 requires independent right-hand sides; "
                f"equation {equation_id} references solved fields: {', '.join(solved_dependencies)}"
            )
        source = _scalar_array(
            evaluate_field_expression(
                equation["rhs"],
                fields=fields,
                parameters=parameters,
                spacing=spacing,
                coordinate_system=(
                    "cartesian_2d" if len(shape) == 2 else "cartesian_3d"
                ),
                differential_scheme="periodic_spectral",
            ),
            shape,
            f"equation {equation_id} rhs",
        )
        source_mean = float(np.mean(source))
        source_rms = float(np.sqrt(np.mean(np.square(source))))
        source_mean_to_rms = abs(source_mean) / max(
            source_rms, np.finfo(float).tiny
        )
        if (
            zero_mode_policy == "require_zero_mean"
            and source_mean_to_rms > zero_mode_tolerance
        ):
            raise ValueError(
                f"fft_poisson equation {equation_id} is not solvable on a periodic domain: "
                f"abs(mean(source))/rms(source)={source_mean_to_rms:.6g} exceeds "
                f"zeroModeTolerance={zero_mode_tolerance:.6g}"
            )

        effective_source = source - source_mean
        source_hat = np.fft.fftn(effective_source)
        potential_hat = np.zeros(shape, dtype=complex)
        potential_hat[nonzero] = -source_hat[nonzero] / k_squared[nonzero]
        potential_complex = np.fft.ifftn(potential_hat)
        maximum_imaginary = float(np.max(np.abs(potential_complex.imag)))
        potential = np.asarray(potential_complex.real, dtype=float)
        potential -= float(np.mean(potential))
        fields[target] = potential

        reconstructed = np.asarray(
            np.fft.ifftn(-k_squared * np.fft.fftn(potential)).real,
            dtype=float,
        )
        residual_field = reconstructed - effective_source
        residual_rms = float(np.sqrt(np.mean(np.square(residual_field))))
        effective_source_rms = float(
            np.sqrt(np.mean(np.square(effective_source)))
        )
        reconstructed_rms = float(
            np.sqrt(np.mean(np.square(reconstructed)))
        )
        relative_residual = residual_rms / max(
            effective_source_rms,
            reconstructed_rms,
            np.finfo(float).tiny,
        )
        residuals[equation_id] = relative_residual

        sample_count = float(np.prod(shape))
        gradient_energy_integral = float(
            np.sum(k_squared * np.square(np.abs(potential_hat)))
            * cell_volume
            / sample_count
        )
        minus_potential_source_integral = float(
            -np.sum(potential * effective_source) * cell_volume
        )
        energy_balance_relative_error = abs(
            gradient_energy_integral - minus_potential_source_integral
        ) / max(
            abs(gradient_energy_integral),
            abs(minus_potential_source_integral),
            np.finfo(float).tiny,
        )
        potential_scale = max(
            float(np.max(np.abs(potential))), np.finfo(float).tiny
        )
        equation_diagnostics.append(
            {
                "equation_id": equation_id,
                "target_field": target,
                "raw_source_mean": source_mean,
                "raw_source_rms": source_rms,
                "raw_source_mean_to_rms": source_mean_to_rms,
                "removed_source_mean": source_mean,
                "raw_source_integral": float(np.sum(source) * cell_volume),
                "effective_source_integral": float(
                    np.sum(effective_source) * cell_volume
                ),
                "potential_mean": float(np.mean(potential)),
                "maximum_imaginary_leakage": maximum_imaginary,
                "relative_imaginary_leakage": maximum_imaginary
                / potential_scale,
                "relative_spectral_residual": relative_residual,
                "gradient_energy_integral": gradient_energy_integral,
                "minus_potential_source_integral": minus_potential_source_integral,
                "energy_balance_relative_error": energy_balance_relative_error,
            }
        )

    metadata = {
        "method": "direct_periodic_spectral_poisson",
        "operator_convention": "continuum_fourier_laplacian",
        "boundary": "periodic_all_axes",
        "zero_mode_policy": zero_mode_policy,
        "zero_mode_tolerance": zero_mode_tolerance,
        "potential_gauge": "zero_mean",
        "differential_observables": "periodic_spectral",
        "first_derivative_nyquist_policy": "zero_for_real_nodal_derivative",
        "cell_volume": cell_volume,
        "domain_volume": domain_volume,
        "domain_lengths": [
            float(count) * float(step)
            for count, step in zip(shape, spacing, strict=True)
        ],
        "nonzero_mode_count": int(np.count_nonzero(nonzero)),
        "minimum_nonzero_wavenumber": float(np.min(nonzero_wavenumbers)),
        "maximum_wavenumber": float(np.max(nonzero_wavenumbers)),
        "equations": equation_diagnostics,
    }
    history = [
        {
            "iteration": 1,
            "maximum_relative_update": 0.0,
            "equation_residuals": dict(residuals),
            "method": "direct_solve",
        }
    ]
    if max(residuals.values(), default=0.0) > residual_tolerance:
        metadata["failure"] = "spectral_residual_exceeds_tolerance"
    return residuals, history, metadata


def _boundary_array(
    value: float | Array,
    shape: tuple[int, ...],
    *,
    coordinate_system: str | None = None,
) -> Array:
    supplied = np.asarray(value, dtype=float)
    if supplied.ndim == 0:
        supplied = np.full(shape, float(supplied), dtype=float)
    if supplied.shape != shape or np.any(~np.isfinite(supplied)):
        raise ValueError(f"boundary must be finite with shape {shape}")
    result = np.zeros(shape, dtype=float)
    for axis in range(len(shape)):
        leading = [slice(None)] * len(shape)
        trailing = [slice(None)] * len(shape)
        leading[axis] = 0
        trailing[axis] = -1
        if not (coordinate_system == "axisymmetric_cylindrical" and axis == 0):
            result[tuple(leading)] = supplied[tuple(leading)]
        result[tuple(trailing)] = supplied[tuple(trailing)]
    return result


def _harmonic_face(left: float, right: float) -> float:
    return 2.0 * left * right / (left + right)


def _solve_axisymmetric_variable_coefficient_dirichlet(
    source: Array,
    coefficient: Array,
    spacing: Sequence[float],
    boundary: float | Array,
    *,
    coefficient_floor: float,
) -> Array:
    """Solve ``1/r d_r(r a d_r u) + d_z(a d_z u) = source``.

    Samples are nodal with ``r_i=i*dr``. The unknown region includes ``r=0``
    and excludes the outer radial and both vertical boundaries. Integrating
    the radial flux over the half control volume at the axis gives the regular
    coefficient ``4*a_(1/2)/dr^2``.
    """

    rhs_field = np.asarray(source, dtype=float)
    scale = np.maximum(np.asarray(coefficient, dtype=float), coefficient_floor)
    if rhs_field.ndim != 2 or min(rhs_field.shape) < 5:
        raise ValueError(
            "axisymmetric source must be a 2D (r,z) grid with at least five cells per axis"
        )
    if scale.shape != rhs_field.shape or np.any(~np.isfinite(rhs_field)) or np.any(~np.isfinite(scale)):
        raise ValueError("source and coefficient must be finite arrays with matching shapes")
    dr, dz = _spacing(spacing, 2)
    boundary_field = _boundary_array(
        boundary,
        rhs_field.shape,
        coordinate_system="axisymmetric_cylindrical",
    )
    radial_count, vertical_count = rhs_field.shape
    unknown_shape = (radial_count - 1, vertical_count - 2)
    unknown_count = int(np.prod(unknown_shape))
    rows: list[int] = []
    columns: list[int] = []
    entries: list[float] = []
    right_hand_side = np.empty(unknown_count, dtype=float)

    def unknown_index(radial_index: int, vertical_index: int) -> int:
        return int(
            np.ravel_multi_index(
                (radial_index, vertical_index - 1), unknown_shape
            )
        )

    for radial_index in range(radial_count - 1):
        radius = radial_index * dr
        for vertical_index in range(1, vertical_count - 1):
            row = unknown_index(radial_index, vertical_index)
            right_hand_side[row] = -rhs_field[radial_index, vertical_index]
            diagonal = 0.0
            center_scale = float(scale[radial_index, vertical_index])

            plus_face = _harmonic_face(
                center_scale, float(scale[radial_index + 1, vertical_index])
            )
            if radial_index == 0:
                plus_conductance = 4.0 * plus_face / dr**2
            else:
                plus_conductance = (
                    (radius + 0.5 * dr) / radius * plus_face / dr**2
                )
            diagonal += plus_conductance
            if radial_index + 1 < radial_count - 1:
                rows.append(row)
                columns.append(unknown_index(radial_index + 1, vertical_index))
                entries.append(-plus_conductance)
            else:
                right_hand_side[row] += (
                    plus_conductance
                    * boundary_field[radial_index + 1, vertical_index]
                )

            if radial_index > 0:
                minus_face = _harmonic_face(
                    center_scale, float(scale[radial_index - 1, vertical_index])
                )
                minus_conductance = (
                    (radius - 0.5 * dr) / radius * minus_face / dr**2
                )
                diagonal += minus_conductance
                rows.append(row)
                columns.append(unknown_index(radial_index - 1, vertical_index))
                entries.append(-minus_conductance)

            for direction in (-1, 1):
                neighbor_vertical = vertical_index + direction
                face_scale = _harmonic_face(
                    center_scale,
                    float(scale[radial_index, neighbor_vertical]),
                )
                conductance = face_scale / dz**2
                diagonal += conductance
                if 0 < neighbor_vertical < vertical_count - 1:
                    rows.append(row)
                    columns.append(
                        unknown_index(radial_index, neighbor_vertical)
                    )
                    entries.append(-conductance)
                else:
                    right_hand_side[row] += (
                        conductance
                        * boundary_field[radial_index, neighbor_vertical]
                    )

            rows.append(row)
            columns.append(row)
            entries.append(diagonal)

    matrix = sparse.csr_matrix(
        (entries, (rows, columns)), shape=(unknown_count, unknown_count)
    )
    interior = spsolve(matrix, right_hand_side)
    if np.any(~np.isfinite(interior)):
        raise RuntimeError("axisymmetric elliptic solve returned non-finite values")
    result = boundary_field.copy()
    result[0:-1, 1:-1] = interior.reshape(unknown_shape)
    return result


def solve_variable_coefficient_dirichlet(
    source: Array,
    coefficient: Array,
    spacing: Sequence[float],
    boundary: float | Array = 0.0,
    *,
    coefficient_floor: float = 1e-8,
    coordinate_system: str | None = None,
) -> Array:
    """Solve ``div(coefficient grad(phi)) = source`` on a supported grid."""

    rhs_field = np.asarray(source, dtype=float)
    scale = np.asarray(coefficient, dtype=float)
    if rhs_field.ndim not in {2, 3} or min(rhs_field.shape) < 5:
        raise ValueError("source must be a 2D or 3D grid with at least five cells per axis")
    if scale.shape != rhs_field.shape or np.any(~np.isfinite(rhs_field)) or np.any(~np.isfinite(scale)):
        raise ValueError("source and coefficient must be finite arrays with matching shapes")
    if not math.isfinite(coefficient_floor) or coefficient_floor <= 0:
        raise ValueError("coefficient_floor must be finite and positive")
    if coordinate_system == "axisymmetric_cylindrical":
        return _solve_axisymmetric_variable_coefficient_dirichlet(
            rhs_field,
            scale,
            spacing,
            boundary,
            coefficient_floor=coefficient_floor,
        )
    if coordinate_system not in {None, "cartesian_2d", "cartesian_3d"}:
        raise ValueError(f"unsupported coordinate system: {coordinate_system}")
    steps = _spacing(spacing, rhs_field.ndim)
    scale = np.maximum(scale, coefficient_floor)
    boundary_field = _boundary_array(boundary, rhs_field.shape)
    interior_shape = tuple(count - 2 for count in rhs_field.shape)
    unknown_count = int(np.prod(interior_shape))
    rows: list[int] = []
    columns: list[int] = []
    entries: list[float] = []
    right_hand_side = np.empty(unknown_count, dtype=float)

    def unknown_index(full_index: tuple[int, ...]) -> int:
        shifted = tuple(value - 1 for value in full_index)
        return int(np.ravel_multi_index(shifted, interior_shape))

    ranges = [range(1, count - 1) for count in rhs_field.shape]
    for full_index in product(*ranges):
        row = unknown_index(full_index)
        right_hand_side[row] = -rhs_field[full_index]
        diagonal = 0.0
        center_scale = scale[full_index]
        for axis, step in enumerate(steps):
            for direction in (-1, 1):
                neighbor = list(full_index)
                neighbor[axis] += direction
                neighbor_index = tuple(neighbor)
                neighbor_scale = scale[neighbor_index]
                face_scale = 2.0 * center_scale * neighbor_scale / (center_scale + neighbor_scale)
                conductance = face_scale / step**2
                diagonal += conductance
                if 0 < neighbor[axis] < rhs_field.shape[axis] - 1:
                    rows.append(row)
                    columns.append(unknown_index(neighbor_index))
                    entries.append(-conductance)
                else:
                    right_hand_side[row] += conductance * boundary_field[neighbor_index]
        rows.append(row)
        columns.append(row)
        entries.append(diagonal)

    matrix = sparse.csr_matrix((entries, (rows, columns)), shape=(unknown_count, unknown_count))
    interior = spsolve(matrix, right_hand_side)
    if np.any(~np.isfinite(interior)):
        raise RuntimeError("elliptic solve returned non-finite values")
    result = boundary_field.copy()
    result[tuple(slice(1, -1) for _ in rhs_field.shape)] = interior.reshape(interior_shape)
    return result


def _relative_update(previous: Array, current: Array) -> float:
    numerator = float(np.sqrt(np.mean(np.square(current - previous))))
    denominator = float(np.sqrt(np.mean(np.square(current))))
    return numerator / max(denominator, np.finfo(float).tiny)


def _finite_volume_divergence_gradient(
    field: Array,
    coefficient: Array,
    spacing: Sequence[float],
    *,
    coefficient_floor: float,
    coordinate_system: str | None = None,
) -> tuple[Array, Array]:
    """Apply the solver stencil and return its local absolute flux scale."""

    values = np.asarray(field, dtype=float)
    scale = np.maximum(np.asarray(coefficient, dtype=float), coefficient_floor)
    if coordinate_system == "axisymmetric_cylindrical":
        if values.ndim != 2 or scale.shape != values.shape:
            raise ValueError(
                "axisymmetric operator requires matching 2D (r,z) fields"
            )
        dr, dz = _spacing(spacing, 2)
        result = np.zeros_like(values)
        flux_scale = np.zeros_like(values)
        radial_count, vertical_count = values.shape
        for radial_index in range(radial_count - 1):
            radius = radial_index * dr
            for vertical_index in range(1, vertical_count - 1):
                center = float(values[radial_index, vertical_index])
                center_scale = float(scale[radial_index, vertical_index])
                plus_face = _harmonic_face(
                    center_scale, float(scale[radial_index + 1, vertical_index])
                )
                if radial_index == 0:
                    plus_flux = (
                        4.0
                        * plus_face
                        * (values[1, vertical_index] - center)
                        / dr**2
                    )
                    radial_value = plus_flux
                    radial_flux_scale = abs(float(plus_flux))
                else:
                    plus_flux = (
                        (radius + 0.5 * dr)
                        / radius
                        * plus_face
                        * (values[radial_index + 1, vertical_index] - center)
                        / dr**2
                    )
                    minus_face = _harmonic_face(
                        center_scale,
                        float(scale[radial_index - 1, vertical_index]),
                    )
                    minus_flux = (
                        (radius - 0.5 * dr)
                        / radius
                        * minus_face
                        * (center - values[radial_index - 1, vertical_index])
                        / dr**2
                    )
                    radial_value = plus_flux - minus_flux
                    radial_flux_scale = abs(float(plus_flux)) + abs(
                        float(minus_flux)
                    )
                vertical_plus_face = _harmonic_face(
                    center_scale, float(scale[radial_index, vertical_index + 1])
                )
                vertical_minus_face = _harmonic_face(
                    center_scale, float(scale[radial_index, vertical_index - 1])
                )
                vertical_plus_flux = (
                    vertical_plus_face
                    * (values[radial_index, vertical_index + 1] - center)
                    / dz**2
                )
                vertical_minus_flux = (
                    vertical_minus_face
                    * (center - values[radial_index, vertical_index - 1])
                    / dz**2
                )
                result[radial_index, vertical_index] = (
                    radial_value + vertical_plus_flux - vertical_minus_flux
                )
                flux_scale[radial_index, vertical_index] = (
                    radial_flux_scale
                    + abs(float(vertical_plus_flux))
                    + abs(float(vertical_minus_flux))
                )
        return result, flux_scale
    if coordinate_system not in {None, "cartesian_2d", "cartesian_3d"}:
        raise ValueError(f"unsupported coordinate system: {coordinate_system}")
    result = np.zeros_like(values)
    flux_scale = np.zeros_like(values)
    interior = tuple(slice(1, -1) for _ in values.shape)
    center = values[interior]
    center_scale = scale[interior]
    for axis, step in enumerate(spacing):
        plus = list(interior)
        minus = list(interior)
        plus[axis] = slice(2, None)
        minus[axis] = slice(None, -2)
        plus_index = tuple(plus)
        minus_index = tuple(minus)
        plus_scale = scale[plus_index]
        minus_scale = scale[minus_index]
        plus_face = 2.0 * center_scale * plus_scale / (center_scale + plus_scale)
        minus_face = 2.0 * center_scale * minus_scale / (center_scale + minus_scale)
        plus_flux = plus_face * (values[plus_index] - center) / float(step) ** 2
        minus_flux = minus_face * (center - values[minus_index]) / float(step) ** 2
        result[interior] += plus_flux - minus_flux
        flux_scale[interior] += np.abs(plus_flux) + np.abs(minus_flux)
    return result, flux_scale


def _equation_residuals(
    equations: Sequence[Mapping[str, Any]],
    fields: Mapping[str, Array],
    parameters: Mapping[str, float],
    spacing: Sequence[float],
    shape: tuple[int, ...],
    *,
    coefficient_floor: float,
    coordinate_system: str,
) -> dict[str, float]:
    residuals: dict[str, float] = {}
    interior = _solution_region(coordinate_system, shape)
    for equation in equations:
        target, coefficient_expression = _elliptic_lhs(equation["lhs"])
        coefficient = _scalar_array(
            evaluate_field_expression(
                coefficient_expression,
                fields=fields,
                parameters=parameters,
                spacing=spacing,
                coordinate_system=coordinate_system,
            ),
            shape,
            f"equation {equation['id']} coefficient",
        )
        left, flux_scale = _finite_volume_divergence_gradient(
            fields[target],
            coefficient,
            spacing,
            coefficient_floor=coefficient_floor,
            coordinate_system=coordinate_system,
        )
        right = _scalar_array(
            evaluate_field_expression(
                equation["rhs"],
                fields=fields,
                parameters=parameters,
                spacing=spacing,
                coordinate_system=coordinate_system,
            ),
            shape,
            f"equation {equation['id']} rhs",
        )
        numerator = float(np.sqrt(np.mean(np.square((left - right)[interior]))))
        rhs_scale = float(np.sqrt(np.mean(np.square(right[interior]))))
        operator_scale = float(np.sqrt(np.mean(np.square(flux_scale[interior]))))
        residuals[str(equation["id"])] = numerator / max(
            rhs_scale, operator_scale, np.finfo(float).tiny
        )
    return residuals


def _solve_nonlinear_root_method(
    *,
    method: str,
    equations: Sequence[Mapping[str, Any]],
    fields: dict[str, Array],
    parameters: Mapping[str, float],
    spacing: Sequence[float],
    shape: tuple[int, ...],
    coefficient_floor: float,
    relative_tolerance: float,
    residual_tolerance: float,
    maximum_iterations: int,
    solver: Mapping[str, Any],
    coordinate_system: str,
    iteration_offset: int = 0,
) -> tuple[int, float, bool, list[dict[str, Any]], dict[str, Any]]:
    """Solve one nonlinear elliptic field through its discrete residual."""

    if len(equations) != 1:
        raise ValueError(f"{method} worker v1 requires exactly one equation")
    equation = equations[0]
    target, coefficient_expression = _elliptic_lhs(equation["lhs"])
    interior = _solution_region(coordinate_system, shape)
    interior_shape = tuple(
        (
            shape[axis] - 1
            if coordinate_system == "axisymmetric_cylindrical" and axis == 0
            else shape[axis] - 2
        )
        for axis in range(len(shape))
    )
    template = fields[target].copy()
    initial = fields[target][interior].copy()
    field_scale = float(np.sqrt(np.mean(np.square(initial))))
    if not math.isfinite(field_scale) or field_scale <= np.finfo(float).tiny:
        field_scale = 1.0
    initial_scaled = (initial / field_scale).ravel()
    previous_callback_field = fields[target].copy()
    history: list[dict[str, Any]] = []

    def physical_field(values: Array) -> Array:
        current = template.copy()
        current[interior] = np.asarray(values, dtype=float).reshape(interior_shape) * field_scale
        return current

    def normalized_residual(values: Array) -> Array:
        fields[target] = physical_field(values)
        coefficient = _scalar_array(
            evaluate_field_expression(
                coefficient_expression,
                fields=fields,
                parameters=parameters,
                spacing=spacing,
                coordinate_system=coordinate_system,
            ),
            shape,
            f"equation {equation['id']} coefficient",
        )
        left, flux_scale = _finite_volume_divergence_gradient(
            fields[target],
            coefficient,
            spacing,
            coefficient_floor=coefficient_floor,
            coordinate_system=coordinate_system,
        )
        right = _scalar_array(
            evaluate_field_expression(
                equation["rhs"],
                fields=fields,
                parameters=parameters,
                spacing=spacing,
                coordinate_system=coordinate_system,
            ),
            shape,
            f"equation {equation['id']} rhs",
        )
        rhs_scale = float(np.sqrt(np.mean(np.square(right[interior]))))
        operator_scale = float(np.sqrt(np.mean(np.square(flux_scale[interior]))))
        scale = max(rhs_scale, operator_scale, np.finfo(float).tiny)
        return ((left - right)[interior] / scale).ravel()

    def callback(values: Array, _residual: Array) -> None:
        nonlocal previous_callback_field
        current = physical_field(values)
        fields[target] = current
        maximum_update = _relative_update(previous_callback_field, current)
        current_residuals = _equation_residuals(
            equations,
            fields,
            parameters,
            spacing,
            shape,
            coefficient_floor=coefficient_floor,
            coordinate_system=coordinate_system,
        )
        history.append(
            {
                "iteration": len(history) + 1,
                "maximum_relative_update": maximum_update,
                "equation_residuals": current_residuals,
            }
        )
        previous_callback_field = current.copy()

    line_search = solver.get("lineSearch", "armijo")
    line_search_value = None if line_search == "none" else str(line_search)
    common = {
        "maxiter": maximum_iterations,
        "f_tol": residual_tolerance * 0.25,
        "x_rtol": relative_tolerance * 0.25,
        "line_search": line_search_value,
        "callback": callback,
    }
    try:
        if method == "anderson":
            final_scaled = anderson(
                normalized_residual,
                initial_scaled,
                alpha=float(solver.get("andersonAlpha", 1.0)),
                M=int(solver.get("andersonHistory", 5)),
                w0=float(solver.get("andersonRegularization", 0.01)),
                **common,
            )
        elif method == "newton_krylov":
            final_scaled = newton_krylov(
                normalized_residual,
                initial_scaled,
                method=str(solver.get("krylovMethod", "lgmres")),
                inner_maxiter=int(solver.get("krylovInnerIterations", 20)),
                **common,
            )
        else:  # pragma: no cover - caller validates the enum
            raise ValueError(f"unsupported nonlinear root method: {method}")
    except NoConvergence as error:
        final_scaled = np.asarray(error.args[0], dtype=float)

    fields[target] = physical_field(np.asarray(final_scaled, dtype=float))
    residuals = _equation_residuals(
        equations,
        fields,
        parameters,
        spacing,
        shape,
        coefficient_floor=coefficient_floor,
        coordinate_system=coordinate_system,
    )
    maximum_update = (
        float(history[-1]["maximum_relative_update"]) if history else math.inf
    )
    converged = (
        maximum_update <= relative_tolerance
        and max(residuals.values()) <= residual_tolerance
    )
    metadata = {
        "root_unknown_scale": field_scale,
        "line_search": line_search,
    }
    if method == "anderson":
        metadata.update(
            {
                "anderson_alpha": float(solver.get("andersonAlpha", 1.0)),
                "anderson_history": int(solver.get("andersonHistory", 5)),
                "anderson_regularization": float(
                    solver.get("andersonRegularization", 0.01)
                ),
            }
        )
    else:
        metadata.update(
            {
                "krylov_method": str(solver.get("krylovMethod", "lgmres")),
                "krylov_inner_iterations": int(
                    solver.get("krylovInnerIterations", 20)
                ),
            }
        )
    for record in history:
        record["iteration"] = int(record["iteration"]) + int(iteration_offset)
    return len(history), maximum_update, converged, history, metadata


def _run_picard_steps(
    *,
    equations: Sequence[Mapping[str, Any]],
    fields: dict[str, Array],
    parameters: Mapping[str, float],
    spacing: Sequence[float],
    shape: tuple[int, ...],
    manifest: Mapping[str, Any],
    boundaries: Mapping[str, float | Array],
    coefficient_floor: float,
    damping: float,
    maximum_iterations: int,
    relative_tolerance: float,
    residual_tolerance: float,
    stop_when_converged: bool,
    coordinate_system: str,
    iteration_offset: int = 0,
) -> tuple[int, float, bool, list[dict[str, Any]]]:
    history: list[dict[str, Any]] = []
    maximum_update = math.inf
    converged = False
    for local_iteration in range(1, maximum_iterations + 1):
        maximum_update = 0.0
        for equation in equations:
            target, coefficient_expression = _elliptic_lhs(equation["lhs"])
            source = _scalar_array(
                evaluate_field_expression(
                    equation["rhs"],
                    fields=fields,
                    parameters=parameters,
                    spacing=spacing,
                    coordinate_system=coordinate_system,
                ),
                shape,
                f"equation {equation['id']} rhs",
            )
            coefficient = _scalar_array(
                evaluate_field_expression(
                    coefficient_expression,
                    fields=fields,
                    parameters=parameters,
                    spacing=spacing,
                    coordinate_system=coordinate_system,
                ),
                shape,
                f"equation {equation['id']} coefficient",
            )
            definition = manifest["fields"][target]
            default_boundary = definition.get("boundary", {}).get("value", 0.0)
            solved = solve_variable_coefficient_dirichlet(
                source,
                coefficient,
                spacing,
                boundaries.get(target, default_boundary),
                coefficient_floor=coefficient_floor,
                coordinate_system=coordinate_system,
            )
            previous = fields[target]
            updated = damping * solved + (1.0 - damping) * previous
            maximum_update = max(maximum_update, _relative_update(previous, updated))
            fields[target] = updated
        current_residuals = _equation_residuals(
            equations,
            fields,
            parameters,
            spacing,
            shape,
            coefficient_floor=coefficient_floor,
            coordinate_system=coordinate_system,
        )
        history.append(
            {
                "iteration": int(iteration_offset) + local_iteration,
                "maximum_relative_update": maximum_update,
                "equation_residuals": current_residuals,
            }
        )
        converged = (
            maximum_update <= relative_tolerance
            and max(current_residuals.values()) <= residual_tolerance
        )
        if stop_when_converged and converged:
            break
    return len(history), maximum_update, converged, history


def _contains_operator(value: Any, operator: str) -> bool:
    if isinstance(value, Mapping):
        return value.get("op") == operator or any(
            _contains_operator(child, operator) for child in value.values()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_operator(child, operator) for child in value)
    return False


def solve_field_manifest(
    manifest: Mapping[str, Any],
    source_fields: Mapping[str, Array],
    spacing: float | Sequence[float],
    *,
    boundary_values: Mapping[str, float | Array] | None = None,
    grid_geometry: Mapping[str, Any] | None = None,
) -> GenericFieldSolution:
    """Execute a validated ``sigma-field-model/1`` manifest on supplied fields."""

    dimensions = int(manifest.get("geometry", {}).get("dimensions", 0))
    if dimensions not in {2, 3}:
        raise ValueError("generic worker supports dimensions=2 or dimensions=3")
    coordinate_system = manifest.get("geometry", {}).get("coordinateSystem")
    if coordinate_system not in {
        "cartesian_2d",
        "cartesian_3d",
        "axisymmetric_cylindrical",
    }:
        raise ValueError(
            "generic worker supports cartesian_2d, cartesian_3d, and axisymmetric_cylindrical"
        )
    if coordinate_system == "axisymmetric_cylindrical" and dimensions != 2:
        raise ValueError("axisymmetric_cylindrical requires dimensions=2")
    axisymmetric_metadata = (
        _axisymmetric_grid_metadata(grid_geometry)
        if coordinate_system == "axisymmetric_cylindrical"
        else {}
    )
    steps = _spacing(spacing, dimensions)
    fields: dict[str, Array] = {}
    shape: tuple[int, ...] | None = None
    for name, definition in manifest.get("fields", {}).items():
        if definition.get("role") != "source":
            continue
        dataset_key = str(definition.get("datasetKey", ""))
        if dataset_key not in source_fields:
            raise ValueError(f"source field {name} requires data key {dataset_key}")
        values = np.asarray(source_fields[dataset_key], dtype=float)
        if values.ndim != dimensions or min(values.shape) < 5 or np.any(~np.isfinite(values)):
            raise ValueError(f"source field {dataset_key} must be a finite {dimensions}D grid")
        if shape is None:
            shape = values.shape
        elif values.shape != shape:
            raise ValueError("all source fields must share one grid shape")
        fields[name] = values
    if shape is None:
        raise ValueError("at least one source field is required")

    parameters = {
        name: float(definition.get("value", definition.get("initial")))
        for name, definition in manifest.get("parameters", {}).items()
    }
    solver = manifest.get("solver", {})
    solver_family = str(solver.get("family", ""))
    boundaries = dict(boundary_values or {})
    solved_names = [
        name for name, definition in manifest.get("fields", {}).items() if definition.get("role") == "solved"
    ]
    for name in solved_names:
        definition = manifest["fields"][name]
        boundary_type = definition.get("boundary", {}).get("type")
        if solver_family == "fft_poisson":
            if coordinate_system not in {"cartesian_2d", "cartesian_3d"}:
                raise ValueError("fft_poisson requires Cartesian 2D or 3D geometry")
            if boundary_type != "periodic":
                raise ValueError(
                    f"fft_poisson requires a periodic boundary for solved field {name}"
                )
            if name in boundaries:
                raise ValueError(
                    f"fft_poisson field {name} cannot accept a boundary-value override"
                )
            fields[name] = np.zeros(shape, dtype=float)
        else:
            if boundary_type not in {"dirichlet", "isolated"}:
                raise ValueError(
                    f"worker v1 cannot execute boundary type {boundary_type!r} for {name}"
                )
            default = definition.get("boundary", {}).get("value", 0.0)
            fields[name] = _boundary_array(
                boundaries.get(name, default),
                shape,
                coordinate_system=coordinate_system,
            )

    equations = list(manifest.get("equations", []))
    targets = [_elliptic_lhs(equation["lhs"])[0] for equation in equations]
    if sorted(targets) != sorted(solved_names):
        raise ValueError("worker v1 requires exactly one elliptic equation per solved field")

    uses_convolution = _contains_operator(
        {
            "equations": manifest.get("equations", []),
            "observables": manifest.get("observables", []),
        },
        "convolution",
    )
    nonlocal_metadata: dict[str, Any] = {}
    if uses_convolution:
        if coordinate_system == "axisymmetric_cylindrical":
            raise ValueError(
                "axisymmetric convolution requires an explicit cylindrical kernel operator; Cartesian linear_same semantics are not valid"
            )
        required_semantics = {
            "family": "nonlocal_elliptic",
            "nonlocalBoundary": "zero_padded",
            "convolutionMode": "linear_same",
            "kernelOrigin": "centered_sample",
            "convolutionMeasure": "physical_volume",
        }
        for key, expected in required_semantics.items():
            if solver.get(key) != expected:
                raise ValueError(
                    f"convolution requires solver.{key}={expected!r}"
                )
        if any(size % 2 == 0 for size in shape):
            raise ValueError(
                "convolution requires an odd grid size in every dimension"
            )
        nonlocal_metadata["nonlocal_convolution"] = {
            "boundary": "zero_padded",
            "mode": "linear_same",
            "kernel_origin": "centered_sample",
            "measure": "physical_volume",
            "cell_volume": float(np.prod(np.asarray(steps, dtype=float))),
            "periodic_wraparound": False,
            "automatic_kernel_normalization": False,
        }
    tolerance = float(solver.get("relativeTolerance", 1e-7))
    residual_tolerance = float(solver.get("residualTolerance", tolerance))
    requested_maximum_iterations = int(solver.get("maxIterations", 80))
    maximum_iterations = min(
        requested_maximum_iterations, PREVIEW_MAXIMUM_ITERATIONS
    )
    damping = float(solver.get("damping", 0.7))
    coefficient_floor = float(solver.get("coefficientFloor", 1e-8))
    initialization = str(solver.get("initialization", "zero"))
    nonlinear_method = str(solver.get("nonlinearMethod", "picard"))
    picard_warmup_iterations = int(solver.get("picardWarmupIterations", 0))
    picard_warmup_damping = float(solver.get("picardWarmupDamping", damping))
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("solver relativeTolerance must be finite and positive")
    if not math.isfinite(residual_tolerance) or residual_tolerance <= 0:
        raise ValueError("solver residualTolerance must be finite and positive")
    if not 0 < damping <= 1:
        raise ValueError("solver damping must lie in (0,1]")
    if initialization not in {"zero", "linearized_unit_coefficient"}:
        raise ValueError(
            "solver initialization must be zero or linearized_unit_coefficient"
        )
    if nonlinear_method not in {"picard", "anderson", "newton_krylov"}:
        raise ValueError(
            "solver nonlinearMethod must be picard, anderson, or newton_krylov"
        )
    if picard_warmup_iterations < 0 or picard_warmup_iterations >= maximum_iterations:
        raise ValueError(
            "solver picardWarmupIterations must be non-negative and below maxIterations"
        )
    if not 0 < picard_warmup_damping <= 1:
        raise ValueError("solver picardWarmupDamping must lie in (0,1]")

    if solver_family == "fft_poisson":
        residuals, history, fft_metadata = _solve_periodic_fft_poisson_equations(
            equations=equations,
            fields=fields,
            solved_names=solved_names,
            parameters=parameters,
            spacing=steps,
            shape=shape,
            solver=solver,
            residual_tolerance=residual_tolerance,
        )
        converged = max(residuals.values(), default=0.0) <= residual_tolerance
        observables = {
            str(observable["id"]): evaluate_field_expression(
                observable["expression"],
                fields=fields,
                parameters=parameters,
                spacing=steps,
                coordinate_system=coordinate_system,
                differential_scheme="periodic_spectral",
            )
            for observable in manifest.get("observables", [])
        }
        return GenericFieldSolution(
            fields={name: fields[name] for name in solved_names},
            observables=observables,
            converged=converged,
            iterations=1,
            maximum_relative_update=0.0,
            equation_residuals=residuals,
            residual_history=tuple(history),
            metadata={
                "engine": "generic-divergence-field-worker-v1",
                "solver_family": solver_family,
                "equation_count": len(equations),
                "solved_field_count": len(solved_names),
                "multi_field_update_scheme": None,
                "dimensions": dimensions,
                "coordinate_system": coordinate_system,
                "shape": shape,
                "spacing": steps,
                "boundary_approximation": None,
                "requested_maximum_iterations": requested_maximum_iterations,
                "executed_maximum_iterations": 1,
                "maximum_iterations_limited_by_worker": False,
                "relative_update_tolerance": tolerance,
                "equation_residual_tolerance": residual_tolerance,
                "fft_poisson": fft_metadata,
            },
        )

    if initialization == "linearized_unit_coefficient":
        for equation in equations:
            target, _coefficient_expression = _elliptic_lhs(equation["lhs"])
            source = _scalar_array(
                evaluate_field_expression(
                    equation["rhs"],
                    fields=fields,
                    parameters=parameters,
                    spacing=steps,
                    coordinate_system=coordinate_system,
                ),
                shape,
                f"equation {equation['id']} rhs",
            )
            definition = manifest["fields"][target]
            default_boundary = definition.get("boundary", {}).get("value", 0.0)
            fields[target] = solve_variable_coefficient_dirichlet(
                source,
                np.ones(shape, dtype=float),
                steps,
                boundaries.get(target, default_boundary),
                coefficient_floor=coefficient_floor,
                coordinate_system=coordinate_system,
            )
    root_metadata: dict[str, Any] = {}
    if nonlinear_method == "picard":
        iteration, maximum_update, converged, history = _run_picard_steps(
            equations=equations,
            fields=fields,
            parameters=parameters,
            spacing=steps,
            shape=shape,
            manifest=manifest,
            boundaries=boundaries,
            coefficient_floor=coefficient_floor,
            damping=damping,
            maximum_iterations=maximum_iterations,
            relative_tolerance=tolerance,
            residual_tolerance=residual_tolerance,
            stop_when_converged=True,
            coordinate_system=coordinate_system,
        )
    else:
        warmup_history: list[dict[str, Any]] = []
        if picard_warmup_iterations:
            _warmup_count, _warmup_update, _warmup_converged, warmup_history = (
                _run_picard_steps(
                    equations=equations,
                    fields=fields,
                    parameters=parameters,
                    spacing=steps,
                    shape=shape,
                    manifest=manifest,
                    boundaries=boundaries,
                    coefficient_floor=coefficient_floor,
                    damping=picard_warmup_damping,
                    maximum_iterations=picard_warmup_iterations,
                    relative_tolerance=tolerance,
                    residual_tolerance=residual_tolerance,
                    stop_when_converged=False,
                    coordinate_system=coordinate_system,
                )
            )
        root_iterations, maximum_update, converged, root_history, root_metadata = (
            _solve_nonlinear_root_method(
                method=nonlinear_method,
                equations=equations,
                fields=fields,
                parameters=parameters,
                spacing=steps,
                shape=shape,
                coefficient_floor=coefficient_floor,
                relative_tolerance=tolerance,
                residual_tolerance=residual_tolerance,
                maximum_iterations=maximum_iterations - picard_warmup_iterations,
                solver=solver,
                coordinate_system=coordinate_system,
                iteration_offset=picard_warmup_iterations,
            )
        )
        history = warmup_history + root_history
        iteration = picard_warmup_iterations + root_iterations

    residuals = _equation_residuals(
        equations,
        fields,
        parameters,
        steps,
        shape,
        coefficient_floor=coefficient_floor,
        coordinate_system=coordinate_system,
    )

    observables = {
        str(observable["id"]): evaluate_field_expression(
            observable["expression"],
            fields=fields,
            parameters=parameters,
            spacing=steps,
            coordinate_system=coordinate_system,
        )
        for observable in manifest.get("observables", [])
    }
    return GenericFieldSolution(
        fields={name: fields[name] for name in solved_names},
        observables=observables,
        converged=converged,
        iterations=iteration,
        maximum_relative_update=maximum_update,
        equation_residuals=residuals,
        residual_history=tuple(history),
        metadata={
            "engine": "generic-divergence-field-worker-v1",
            "solver_family": str(solver.get("family", "")),
            "equation_count": len(equations),
            "solved_field_count": len(solved_names),
            "multi_field_update_scheme": (
                "sequential_gauss_seidel" if len(equations) > 1 else None
            ),
            "dimensions": dimensions,
            "coordinate_system": coordinate_system,
            **(
                {"axisymmetric_cylindrical": axisymmetric_metadata}
                if axisymmetric_metadata
                else {}
            ),
            "shape": shape,
            "spacing": steps,
            "boundary_approximation": "isolated manifests use the supplied or zero far-field Dirichlet boundary",
            "coefficient_floor": coefficient_floor,
            "initialization": initialization,
            "nonlinear_method": nonlinear_method,
            "picard_warmup_iterations": picard_warmup_iterations,
            "picard_warmup_damping": picard_warmup_damping,
            "requested_maximum_iterations": requested_maximum_iterations,
            "executed_maximum_iterations": maximum_iterations,
            "maximum_iterations_limited_by_worker": (
                maximum_iterations != requested_maximum_iterations
            ),
            "preview_worker_maximum_iterations": PREVIEW_MAXIMUM_ITERATIONS,
            "relative_update_tolerance": tolerance,
            "equation_residual_tolerance": residual_tolerance,
            **nonlocal_metadata,
            **root_metadata,
        },
    )
