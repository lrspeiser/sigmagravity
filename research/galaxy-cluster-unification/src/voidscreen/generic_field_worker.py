"""Formula-independent finite-volume execution for typed field-model manifests.

This module is the local scientific-worker prototype behind the hosted model
contract.  It intentionally executes a small, auditable equation language
instead of dispatching on theory names.  Version one solves scalar elliptic
equations of the forms::

    laplacian(phi) = source
    divergence(coefficient * gradient(phi)) = source

on uniform two- or three-dimensional Cartesian grids.  Coefficients and
sources are evaluated from the submitted expression tree, so the same path can
represent Poisson gravity, density-dependent Refracted Gravity, AQUAL-like
nonlinear equations, and coupled QUMOND-like equations.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

Array = np.ndarray
ExpressionValue = float | Array | tuple[Array, ...]


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
    return functions[operation](left, right)


def evaluate_field_expression(
    node: Mapping[str, Any],
    *,
    fields: Mapping[str, Array],
    parameters: Mapping[str, float],
    spacing: Sequence[float],
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
        evaluate_field_expression(argument, fields=fields, parameters=parameters, spacing=spacing)
        for argument in arguments
    ]
    if operation in {"add", "subtract", "min", "max", "divide"}:
        return _binary(operation, values[0], values[1])
    if operation == "multiply":
        result = values[0]
        for value in values[1:]:
            result = _binary(operation, result, value)
        return result
    if operation == "negate":
        return tuple(-value for value in values[0]) if isinstance(values[0], tuple) else -values[0]
    if operation == "norm":
        if not isinstance(values[0], tuple):
            return np.abs(values[0])
        return np.sqrt(sum(np.square(component) for component in values[0]))
    if operation == "gradient":
        if isinstance(values[0], tuple):
            raise ValueError("generic worker v1 does not take the gradient of a vector")
        return tuple(np.gradient(values[0], *spacing, edge_order=2))
    if operation == "divergence":
        if not isinstance(values[0], tuple) or len(values[0]) != len(spacing):
            raise ValueError("divergence requires one vector component per grid dimension")
        return sum(
            np.gradient(component, spacing[axis], axis=axis, edge_order=2)
            for axis, component in enumerate(values[0])
        )
    if operation == "laplacian":
        if isinstance(values[0], tuple):
            raise ValueError("generic worker v1 expects a scalar Laplacian")
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
        return np.power(values[0], values[1])
    if operation == "sqrt":
        return np.sqrt(values[0])
    if operation == "exp":
        return np.exp(values[0])
    if operation == "log":
        return np.log(values[0])
    if operation == "tanh":
        return np.tanh(values[0])
    if operation == "smoothstep":
        clipped = np.clip(values[0], 0.0, 1.0)
        return clipped * clipped * (3.0 - 2.0 * clipped)
    if operation in {"lt", "lte", "gt", "gte"}:
        functions = {"lt": np.less, "lte": np.less_equal, "gt": np.greater, "gte": np.greater_equal}
        return functions[operation](values[0], values[1]).astype(float)
    if operation == "piecewise":
        result = evaluate_field_expression(
            node["otherwise"], fields=fields, parameters=parameters, spacing=spacing
        )
        if isinstance(result, tuple):
            raise ValueError("vector piecewise expressions are not supported in worker v1")
        for branch in reversed(node["branches"]):
            condition = evaluate_field_expression(
                branch["when"], fields=fields, parameters=parameters, spacing=spacing
            )
            value = evaluate_field_expression(
                branch["value"], fields=fields, parameters=parameters, spacing=spacing
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


def _boundary_array(value: float | Array, shape: tuple[int, ...]) -> Array:
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
        result[tuple(leading)] = supplied[tuple(leading)]
        result[tuple(trailing)] = supplied[tuple(trailing)]
    return result


def solve_variable_coefficient_dirichlet(
    source: Array,
    coefficient: Array,
    spacing: Sequence[float],
    boundary: float | Array = 0.0,
    *,
    coefficient_floor: float = 1e-8,
) -> Array:
    """Solve ``div(coefficient grad(phi)) = source`` in two or three dimensions."""

    rhs_field = np.asarray(source, dtype=float)
    scale = np.asarray(coefficient, dtype=float)
    if rhs_field.ndim not in {2, 3} or min(rhs_field.shape) < 5:
        raise ValueError("source must be a 2D or 3D grid with at least five cells per axis")
    if scale.shape != rhs_field.shape or np.any(~np.isfinite(rhs_field)) or np.any(~np.isfinite(scale)):
        raise ValueError("source and coefficient must be finite arrays with matching shapes")
    if not math.isfinite(coefficient_floor) or coefficient_floor <= 0:
        raise ValueError("coefficient_floor must be finite and positive")
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


def _equation_residuals(
    equations: Sequence[Mapping[str, Any]],
    fields: Mapping[str, Array],
    parameters: Mapping[str, float],
    spacing: Sequence[float],
    shape: tuple[int, ...],
) -> dict[str, float]:
    residuals: dict[str, float] = {}
    interior = tuple(slice(1, -1) for _ in shape)
    for equation in equations:
        left = _scalar_array(
            evaluate_field_expression(
                equation["lhs"], fields=fields, parameters=parameters, spacing=spacing
            ),
            shape,
            f"equation {equation['id']} lhs",
        )
        right = _scalar_array(
            evaluate_field_expression(
                equation["rhs"], fields=fields, parameters=parameters, spacing=spacing
            ),
            shape,
            f"equation {equation['id']} rhs",
        )
        numerator = float(np.sqrt(np.mean(np.square((left - right)[interior]))))
        denominator = float(np.sqrt(np.mean(np.square(right[interior]))))
        residuals[str(equation["id"])] = numerator / max(
            denominator, np.finfo(float).tiny
        )
    return residuals


def solve_field_manifest(
    manifest: Mapping[str, Any],
    source_fields: Mapping[str, Array],
    spacing: float | Sequence[float],
    *,
    boundary_values: Mapping[str, float | Array] | None = None,
) -> GenericFieldSolution:
    """Execute a validated ``sigma-field-model/1`` manifest on supplied fields."""

    dimensions = int(manifest.get("geometry", {}).get("dimensions", 0))
    if dimensions not in {2, 3}:
        raise ValueError("generic worker supports Cartesian dimensions=2 or dimensions=3")
    coordinate_system = manifest.get("geometry", {}).get("coordinateSystem")
    if coordinate_system not in {"cartesian_2d", "cartesian_3d"}:
        raise ValueError("generic worker v1 supports cartesian_2d and cartesian_3d")
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
    boundaries = dict(boundary_values or {})
    solved_names = [
        name for name, definition in manifest.get("fields", {}).items() if definition.get("role") == "solved"
    ]
    for name in solved_names:
        definition = manifest["fields"][name]
        boundary_type = definition.get("boundary", {}).get("type")
        if boundary_type not in {"dirichlet", "isolated"}:
            raise ValueError(f"worker v1 cannot execute boundary type {boundary_type!r} for {name}")
        default = definition.get("boundary", {}).get("value", 0.0)
        fields[name] = _boundary_array(boundaries.get(name, default), shape)

    equations = list(manifest.get("equations", []))
    targets = [_elliptic_lhs(equation["lhs"])[0] for equation in equations]
    if sorted(targets) != sorted(solved_names):
        raise ValueError("worker v1 requires exactly one elliptic equation per solved field")

    solver = manifest.get("solver", {})
    tolerance = float(solver.get("relativeTolerance", 1e-7))
    maximum_iterations = min(int(solver.get("maxIterations", 80)), 200)
    damping = float(solver.get("damping", 0.7))
    coefficient_floor = float(solver.get("coefficientFloor", 1e-8))
    if not 0 < damping <= 1:
        raise ValueError("solver damping must lie in (0,1]")
    maximum_update = math.inf
    converged = False
    history: list[dict[str, Any]] = []

    for iteration in range(1, maximum_iterations + 1):
        maximum_update = 0.0
        for equation in equations:
            target, coefficient_expression = _elliptic_lhs(equation["lhs"])
            source = _scalar_array(
                evaluate_field_expression(
                    equation["rhs"], fields=fields, parameters=parameters, spacing=steps
                ),
                shape,
                f"equation {equation['id']} rhs",
            )
            coefficient = _scalar_array(
                evaluate_field_expression(
                    coefficient_expression, fields=fields, parameters=parameters, spacing=steps
                ),
                shape,
                f"equation {equation['id']} coefficient",
            )
            definition = manifest["fields"][target]
            default_boundary = definition.get("boundary", {}).get("value", 0.0)
            solved = solve_variable_coefficient_dirichlet(
                source,
                coefficient,
                steps,
                boundaries.get(target, default_boundary),
                coefficient_floor=coefficient_floor,
            )
            previous = fields[target]
            updated = damping * solved + (1.0 - damping) * previous
            maximum_update = max(maximum_update, _relative_update(previous, updated))
            fields[target] = updated
        current_residuals = _equation_residuals(
            equations, fields, parameters, steps, shape
        )
        history.append(
            {
                "iteration": iteration,
                "maximum_relative_update": maximum_update,
                "equation_residuals": current_residuals,
            }
        )
        if maximum_update <= tolerance:
            converged = True
            break

    residuals = history[-1]["equation_residuals"]

    observables = {
        str(observable["id"]): evaluate_field_expression(
            observable["expression"], fields=fields, parameters=parameters, spacing=steps
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
            "dimensions": dimensions,
            "coordinate_system": coordinate_system,
            "shape": shape,
            "spacing": steps,
            "boundary_approximation": "isolated manifests use the supplied or zero far-field Dirichlet boundary",
            "coefficient_floor": coefficient_floor,
        },
    )
