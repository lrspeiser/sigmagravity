"""Coordinate-safe two-dimensional strong-lensing measurements.

The public contract in this module is deliberately stated in sky coordinates:
array rows are north, array columns are east, and vector components are named
``alpha_east`` and ``alpha_north``.  This prevents the silent x/y ambiguity that
was discovered after the P0708 external prediction lock.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import minimum_filter
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.spatial import cKDTree

C_M_S = 299_792_458.0
RAD_TO_ARCSEC = 206_264.80624709636


class SkyDeflectionField(Protocol):
    """A reduced-deflection field expressed in east/north arcseconds."""

    half_extent_arcsec: float

    def alpha(
        self,
        east_arcsec,
        north_arcsec,
        source_redshift: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(alpha_east, alpha_north)`` in arcseconds."""


class GridSkyDeflectionField:
    """Interpolate a deflection grid with an explicit row/column contract.

    Both deflection arrays must have shape ``(north, east)``.  Their values are
    physical deflections in arcseconds at distance ratio one.  ``distance_ratio``
    converts the source redshift to :math:`D_{ls}/D_s`.
    """

    def __init__(
        self,
        *,
        north_axis_arcsec,
        east_axis_arcsec,
        alpha_east_ratio_one_arcsec,
        alpha_north_ratio_one_arcsec,
        distance_ratio: Callable[[float], float],
    ) -> None:
        north = np.asarray(north_axis_arcsec, dtype=float)
        east = np.asarray(east_axis_arcsec, dtype=float)
        alpha_east = np.asarray(alpha_east_ratio_one_arcsec, dtype=float)
        alpha_north = np.asarray(alpha_north_ratio_one_arcsec, dtype=float)
        expected = (len(north), len(east))
        if north.ndim != 1 or east.ndim != 1 or min(len(north), len(east)) < 3:
            raise ValueError("east and north axes must be one-dimensional grids")
        if np.any(~np.isfinite(north)) or np.any(~np.isfinite(east)):
            raise ValueError("east and north axes must be finite")
        if np.any(np.diff(north) <= 0.0) or np.any(np.diff(east) <= 0.0):
            raise ValueError("east and north axes must be strictly increasing")
        if alpha_east.shape != expected or alpha_north.shape != expected:
            raise ValueError("deflection arrays must have shape (north, east)")
        if np.any(~np.isfinite(alpha_east)) or np.any(~np.isfinite(alpha_north)):
            raise ValueError("deflection arrays must be finite")
        self.north_axis_arcsec = north
        self.east_axis_arcsec = east
        self._distance_ratio = distance_ratio
        self._alpha_east = RegularGridInterpolator(
            (north, east), alpha_east, bounds_error=False, fill_value=np.nan
        )
        self._alpha_north = RegularGridInterpolator(
            (north, east), alpha_north, bounds_error=False, fill_value=np.nan
        )
        self.half_extent_arcsec = float(
            min(np.max(np.abs(north)), np.max(np.abs(east)))
        )

    def alpha(
        self,
        east_arcsec,
        north_arcsec,
        source_redshift: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        east, north = _broadcast_coordinates(east_arcsec, north_arcsec)
        points = np.column_stack([north.ravel(), east.ravel()])
        ratio = float(self._distance_ratio(float(source_redshift)))
        if not np.isfinite(ratio) or ratio <= 0.0:
            raise ValueError("distance ratio must be finite and positive")
        return (
            ratio * self._alpha_east(points).reshape(east.shape),
            ratio * self._alpha_north(points).reshape(east.shape),
        )


class LinearCombinationSkyDeflectionField:
    """A dimensionless linear combination of coordinate-compatible fields."""

    def __init__(
        self,
        fields: tuple[SkyDeflectionField, ...],
        coefficients: tuple[float, ...],
    ) -> None:
        if not fields or len(fields) != len(coefficients):
            raise ValueError("fields and coefficients must be matching nonempty tuples")
        if any(not np.isfinite(coefficient) for coefficient in coefficients):
            raise ValueError("coefficients must be finite")
        self.fields = fields
        self.coefficients = tuple(float(value) for value in coefficients)
        self.half_extent_arcsec = min(field.half_extent_arcsec for field in fields)

    def alpha(
        self,
        east_arcsec,
        north_arcsec,
        source_redshift: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        east, north = _broadcast_coordinates(east_arcsec, north_arcsec)
        alpha_east = np.zeros_like(east)
        alpha_north = np.zeros_like(north)
        for coefficient, field in zip(self.coefficients, self.fields, strict=True):
            field_east, field_north = field.alpha(east, north, source_redshift)
            alpha_east += coefficient * field_east
            alpha_north += coefficient * field_north
        return alpha_east, alpha_north


@dataclass(frozen=True)
class LensInvariants:
    """Local derivatives of the lens mapping in the east/north basis."""

    jacobian: np.ndarray
    convergence: np.ndarray
    shear_1: np.ndarray
    shear_2: np.ndarray
    shear_magnitude: np.ndarray
    rotation: np.ndarray
    determinant: np.ndarray
    minimum_eigenvalue: np.ndarray
    maximum_eigenvalue: np.ndarray
    absolute_magnification: np.ndarray


@dataclass(frozen=True)
class RootSearchResult:
    roots_arcsec: np.ndarray
    closure_arcsec: np.ndarray
    absolute_magnification: np.ndarray


@dataclass(frozen=True)
class RootAssignment:
    pairs: np.ndarray
    rms_arcsec: float
    matched_images: int
    complete: bool


@dataclass(frozen=True)
class SkyPhotonDeflection2D:
    """Photon deflection with an explicit ``(north, east, line-of-sight)`` contract."""

    alpha_east_radian: np.ndarray
    alpha_north_radian: np.ndarray
    alpha_east_arcsec: np.ndarray
    alpha_north_arcsec: np.ndarray
    distance_ratio: float
    zero_slip_multiplier: float


def photon_deflection_sky(
    acceleration_north_east_los: tuple[np.ndarray, np.ndarray, np.ndarray],
    dz: float,
    *,
    distance_ratio: float = 1.0,
    light_speed: float = C_M_S,
) -> SkyPhotonDeflection2D:
    """Integrate a field stored as ``(north, east, line-of-sight)``."""
    if (
        len(acceleration_north_east_los) != 3
        or dz <= 0.0
        or light_speed <= 0.0
        or distance_ratio <= 0.0
    ):
        raise ValueError("sky photon-deflection inputs are invalid")
    if any(np.asarray(component).ndim != 3 for component in acceleration_north_east_los):
        raise ValueError("acceleration components must be three-dimensional")
    north, east, _line_of_sight = acceleration_north_east_los
    multiplier = 2.0 * float(distance_ratio) / float(light_speed) ** 2
    alpha_east = -multiplier * np.trapezoid(east, dx=float(dz), axis=2)
    alpha_north = -multiplier * np.trapezoid(north, dx=float(dz), axis=2)
    return SkyPhotonDeflection2D(
        alpha_east_radian=alpha_east,
        alpha_north_radian=alpha_north,
        alpha_east_arcsec=alpha_east * RAD_TO_ARCSEC,
        alpha_north_arcsec=alpha_north * RAD_TO_ARCSEC,
        distance_ratio=float(distance_ratio),
        zero_slip_multiplier=multiplier,
    )


def _broadcast_coordinates(east_arcsec, north_arcsec) -> tuple[np.ndarray, np.ndarray]:
    east, north = np.broadcast_arrays(
        np.asarray(east_arcsec, dtype=float),
        np.asarray(north_arcsec, dtype=float),
    )
    if np.any(~np.isfinite(east)) or np.any(~np.isfinite(north)):
        raise ValueError("east and north coordinates must be finite")
    return east, north


def ray_shoot(
    field: SkyDeflectionField,
    east_arcsec,
    north_arcsec,
    source_redshift: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map image-plane coordinates to the source plane."""
    east, north = _broadcast_coordinates(east_arcsec, north_arcsec)
    alpha_east, alpha_north = field.alpha(east, north, float(source_redshift))
    return east - alpha_east, north - alpha_north


def lens_jacobian(
    field: SkyDeflectionField,
    east_arcsec,
    north_arcsec,
    source_redshift: float,
    *,
    step_arcsec: float = 0.08,
) -> np.ndarray:
    """Return ``d(beta_east,beta_north)/d(east,north)``."""
    if not np.isfinite(step_arcsec) or step_arcsec <= 0.0:
        raise ValueError("step_arcsec must be finite and positive")
    east, north = _broadcast_coordinates(east_arcsec, north_arcsec)
    b_east_plus, b_north_plus = ray_shoot(
        field, east + step_arcsec, north, source_redshift
    )
    b_east_minus, b_north_minus = ray_shoot(
        field, east - step_arcsec, north, source_redshift
    )
    c_east_plus, c_north_plus = ray_shoot(
        field, east, north + step_arcsec, source_redshift
    )
    c_east_minus, c_north_minus = ray_shoot(
        field, east, north - step_arcsec, source_redshift
    )
    result = np.empty(east.shape + (2, 2), dtype=float)
    result[..., 0, 0] = (b_east_plus - b_east_minus) / (2.0 * step_arcsec)
    result[..., 1, 0] = (b_north_plus - b_north_minus) / (2.0 * step_arcsec)
    result[..., 0, 1] = (c_east_plus - c_east_minus) / (2.0 * step_arcsec)
    result[..., 1, 1] = (c_north_plus - c_north_minus) / (2.0 * step_arcsec)
    return result


def lens_invariants(
    field: SkyDeflectionField,
    east_arcsec,
    north_arcsec,
    source_redshift: float,
    *,
    step_arcsec: float = 0.08,
) -> LensInvariants:
    """Measure convergence, shear, rotation, eigenvalues, and magnification."""
    jacobian = lens_jacobian(
        field,
        east_arcsec,
        north_arcsec,
        source_redshift,
        step_arcsec=step_arcsec,
    )
    a_ee = jacobian[..., 0, 0]
    a_en = jacobian[..., 0, 1]
    a_ne = jacobian[..., 1, 0]
    a_nn = jacobian[..., 1, 1]
    convergence = 1.0 - 0.5 * (a_ee + a_nn)
    shear_1 = 0.5 * (a_nn - a_ee)
    shear_2 = -0.5 * (a_en + a_ne)
    rotation = 0.5 * (a_en - a_ne)
    determinant = a_ee * a_nn - a_en * a_ne
    symmetric = 0.5 * (jacobian + np.swapaxes(jacobian, -1, -2))
    eigenvalues = np.linalg.eigvalsh(symmetric)
    magnification = 1.0 / np.maximum(np.abs(determinant), 1.0e-12)
    return LensInvariants(
        jacobian=jacobian,
        convergence=convergence,
        shear_1=shear_1,
        shear_2=shear_2,
        shear_magnitude=np.hypot(shear_1, shear_2),
        rotation=rotation,
        determinant=determinant,
        minimum_eigenvalue=eigenvalues[..., 0],
        maximum_eigenvalue=eigenvalues[..., 1],
        absolute_magnification=magnification,
    )


def profiled_source(
    field: SkyDeflectionField,
    observed_images_arcsec,
    source_redshift: float,
) -> np.ndarray:
    """Profile one source position as the mean ray-shot image position."""
    observed = np.asarray(observed_images_arcsec, dtype=float)
    if observed.ndim != 2 or observed.shape[1] != 2 or len(observed) == 0:
        raise ValueError("observed images must have shape (n, 2)")
    beta_east, beta_north = ray_shoot(
        field, observed[:, 0], observed[:, 1], source_redshift
    )
    return np.asarray([np.mean(beta_east), np.mean(beta_north)], dtype=float)


def _solve_one_root(
    field: SkyDeflectionField,
    source_arcsec: np.ndarray,
    source_redshift: float,
    start_arcsec: np.ndarray,
    bound_arcsec: float,
    closure_tolerance_arcsec: float,
) -> tuple[np.ndarray | None, float]:
    def equation(theta: np.ndarray) -> np.ndarray:
        beta_east, beta_north = ray_shoot(
            field,
            np.asarray([theta[0]]),
            np.asarray([theta[1]]),
            source_redshift,
        )
        return np.asarray(
            [beta_east[0] - source_arcsec[0], beta_north[0] - source_arcsec[1]]
        )

    start = np.clip(np.asarray(start_arcsec, dtype=float), -bound_arcsec, bound_arcsec)
    result = least_squares(
        equation,
        start,
        bounds=([-bound_arcsec, -bound_arcsec], [bound_arcsec, bound_arcsec]),
        max_nfev=240,
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-12,
    )
    closure = float(np.linalg.norm(equation(result.x)))
    if (
        not result.success
        or not np.all(np.isfinite(result.x))
        or closure > closure_tolerance_arcsec
    ):
        return None, closure
    return np.asarray(result.x, dtype=float), closure


def _deduplicate_roots(
    roots: list[tuple[np.ndarray, float]],
    tolerance_arcsec: float,
) -> list[tuple[np.ndarray, float]]:
    unique: list[tuple[np.ndarray, float]] = []
    for candidate, closure in sorted(roots, key=lambda item: item[1]):
        matches = [
            index
            for index, (root, _existing_closure) in enumerate(unique)
            if float(np.linalg.norm(candidate - root)) <= tolerance_arcsec
        ]
        if not matches:
            unique.append((candidate, closure))
    return sorted(unique, key=lambda item: (float(item[0][0]), float(item[0][1])))


def find_lens_roots(
    field: SkyDeflectionField,
    source_arcsec,
    source_redshift: float,
    *,
    bound_arcsec: float,
    observed_starts_arcsec=None,
    grid_points: int = 161,
    closure_tolerance_arcsec: float = 2.0e-3,
    deduplication_tolerance_arcsec: float = 0.20,
    jacobian_step_arcsec: float = 0.08,
    include_residual_minima: bool = True,
    maximum_residual_minimum_seeds: int = 64,
    supplemental_grid_points: tuple[int, ...] = (81, 161, 241),
) -> RootSearchResult:
    """Find roots using zero-contour crossings plus residual-norm minima.

    The residual-minimum seeds make the search much less sensitive to a narrow
    image basin falling between grid cells.  ``include_residual_minima=False``
    is retained solely to reproduce the legacy P0714 measurement algorithm.
    """
    source = np.asarray(source_arcsec, dtype=float)
    if source.shape != (2,) or np.any(~np.isfinite(source)):
        raise ValueError("source_arcsec must be a finite two-vector")
    if not np.isfinite(bound_arcsec) or bound_arcsec <= 0.0:
        raise ValueError("bound_arcsec must be finite and positive")
    if grid_points < 21:
        raise ValueError("grid_points must be at least 21")
    if maximum_residual_minimum_seeds < 1:
        raise ValueError("maximum_residual_minimum_seeds must be positive")
    if any(points < 21 for points in supplemental_grid_points):
        raise ValueError("supplemental grid sizes must be at least 21")
    grid = np.linspace(-bound_arcsec, bound_arcsec, int(grid_points))
    east, north = np.meshgrid(grid, grid, indexing="xy")
    beta_east, beta_north = ray_shoot(field, east, north, source_redshift)
    residual_east = beta_east - source[0]
    residual_north = beta_north - source[1]

    def straddles(values: np.ndarray) -> np.ndarray:
        corners = [
            values[:-1, :-1],
            values[1:, :-1],
            values[:-1, 1:],
            values[1:, 1:],
        ]
        return (np.minimum.reduce(corners) <= 0.0) & (
            np.maximum.reduce(corners) >= 0.0
        )

    crossing = straddles(residual_east) & straddles(residual_north)
    row, column = np.nonzero(crossing)
    starts = [
        np.asarray(
            [
                0.5 * (grid[east_index] + grid[east_index + 1]),
                0.5 * (grid[north_index] + grid[north_index + 1]),
            ]
        )
        for north_index, east_index in zip(row, column, strict=True)
    ]
    if include_residual_minima:
        minimum_candidates: list[tuple[float, np.ndarray]] = []

        def collect_minima(
            east_axis: np.ndarray,
            north_axis: np.ndarray,
            norm: np.ndarray | None = None,
        ) -> None:
            if norm is None:
                grid_east, grid_north = np.meshgrid(
                    east_axis, north_axis, indexing="xy"
                )
                local_beta_east, local_beta_north = ray_shoot(
                    field, grid_east, grid_north, source_redshift
                )
                norm = np.hypot(
                    local_beta_east - source[0], local_beta_north - source[1]
                )
            finite_norm = np.where(np.isfinite(norm), norm, np.inf)
            local_minimum = finite_norm <= minimum_filter(
                finite_norm,
                size=3,
                mode="constant",
                cval=np.inf,
            )
            minimum_row, minimum_column = np.nonzero(
                local_minimum & np.isfinite(finite_norm)
            )
            minimum_candidates.extend(
                (
                    float(finite_norm[north_index, east_index]),
                    np.asarray(
                        [east_axis[east_index], north_axis[north_index]], dtype=float
                    ),
                )
                for north_index, east_index in zip(
                    minimum_row, minimum_column, strict=True
                )
            )

        collect_minima(grid, grid, np.hypot(residual_east, residual_north))
        # The 65x65 archived maps showed complementary narrow basins at 81, 161,
        # and 241 search nodes.  Use all preregistered floors so production root
        # counts do not depend on a single lucky grid phase.
        for supplemental_points in sorted(set(supplemental_grid_points)):
            if supplemental_points == grid_points:
                continue
            supplemental_axis = np.linspace(
                -bound_arcsec, bound_arcsec, supplemental_points
            )
            supplemental_east, supplemental_north = np.meshgrid(
                supplemental_axis, supplemental_axis, indexing="xy"
            )
            supplemental_beta_east, supplemental_beta_north = ray_shoot(
                field, supplemental_east, supplemental_north, source_redshift
            )
            supplemental_residual_east = supplemental_beta_east - source[0]
            supplemental_residual_north = supplemental_beta_north - source[1]
            supplemental_crossing = straddles(supplemental_residual_east) & straddles(
                supplemental_residual_north
            )
            supplemental_row, supplemental_column = np.nonzero(
                supplemental_crossing
            )
            starts.extend(
                np.asarray(
                    [
                        0.5
                        * (
                            supplemental_axis[east_index]
                            + supplemental_axis[east_index + 1]
                        ),
                        0.5
                        * (
                            supplemental_axis[north_index]
                            + supplemental_axis[north_index + 1]
                        ),
                    ]
                )
                for north_index, east_index in zip(
                    supplemental_row, supplemental_column, strict=True
                )
            )
            collect_minima(
                supplemental_axis,
                supplemental_axis,
                np.hypot(
                    supplemental_residual_east, supplemental_residual_north
                ),
            )
        starts.extend(
            point
            for _residual, point in sorted(
                minimum_candidates,
                key=lambda item: item[0],
            )[:maximum_residual_minimum_seeds]
        )
    if observed_starts_arcsec is not None:
        observed = np.asarray(observed_starts_arcsec, dtype=float)
        if observed.ndim != 2 or observed.shape[1] != 2:
            raise ValueError("observed starts must have shape (n, 2)")
        starts.extend(observed)

    solved: list[tuple[np.ndarray, float]] = []
    for start in starts:
        root, closure = _solve_one_root(
            field,
            source,
            source_redshift,
            start,
            bound_arcsec,
            closure_tolerance_arcsec,
        )
        if root is not None:
            solved.append((root, closure))
    unique = _deduplicate_roots(solved, deduplication_tolerance_arcsec)
    if not unique:
        return RootSearchResult(np.empty((0, 2)), np.empty(0), np.empty(0))
    roots = np.asarray([item[0] for item in unique])
    closures = np.asarray([item[1] for item in unique])
    invariants = lens_invariants(
        field,
        roots[:, 0],
        roots[:, 1],
        source_redshift,
        step_arcsec=jacobian_step_arcsec,
    )
    return RootSearchResult(roots, closures, invariants.absolute_magnification)


def assign_observed_roots(observed_images_arcsec, roots_arcsec) -> RootAssignment:
    """Minimum-cost one-to-one assignment between observed and modeled images."""
    observed = np.asarray(observed_images_arcsec, dtype=float)
    roots = np.asarray(roots_arcsec, dtype=float)
    if observed.ndim != 2 or observed.shape[1] != 2:
        raise ValueError("observed images must have shape (n, 2)")
    if roots.ndim != 2 or roots.shape[1] != 2:
        raise ValueError("roots must have shape (n, 2)")
    if len(roots) == 0:
        return RootAssignment(np.empty((0, 2), dtype=int), float("inf"), 0, False)
    cost = np.linalg.norm(observed[:, None, :] - roots[None, :, :], axis=2)
    observed_index, root_index = linear_sum_assignment(cost)
    pairs = np.column_stack([observed_index, root_index])
    complete = len(pairs) == len(observed)
    rms = (
        float(np.sqrt(np.mean(np.square(cost[observed_index, root_index]))))
        if complete
        else float("inf")
    )
    return RootAssignment(pairs, rms, len(pairs), complete)


def critical_curve_points(
    field: SkyDeflectionField,
    source_redshift: float,
    *,
    bound_arcsec: float,
    grid_points: int = 161,
) -> np.ndarray:
    """Return cell centers where the lens-Jacobian determinant changes sign."""
    if grid_points < 21:
        raise ValueError("grid_points must be at least 21")
    grid = np.linspace(-bound_arcsec, bound_arcsec, int(grid_points))
    east, north = np.meshgrid(grid, grid, indexing="xy")
    step = float(grid[1] - grid[0])
    beta_east, beta_north = ray_shoot(field, east, north, source_redshift)
    dbeta_east_dn, dbeta_east_de = np.gradient(
        beta_east, step, step, edge_order=2
    )
    dbeta_north_dn, dbeta_north_de = np.gradient(
        beta_north, step, step, edge_order=2
    )
    determinant = (
        dbeta_east_de * dbeta_north_dn - dbeta_east_dn * dbeta_north_de
    )
    corners = [
        determinant[:-1, :-1],
        determinant[1:, :-1],
        determinant[:-1, 1:],
        determinant[1:, 1:],
    ]
    sign_change = (np.minimum.reduce(corners) <= 0.0) & (
        np.maximum.reduce(corners) >= 0.0
    )
    row, column = np.nonzero(sign_change)
    return np.column_stack(
        [
            0.5 * (grid[column] + grid[column + 1]),
            0.5 * (grid[row] + grid[row + 1]),
        ]
    )


def symmetric_percentile_distance(
    first_points,
    second_points,
    *,
    percentile: float = 0.95,
) -> float | None:
    """Symmetric nearest-neighbor percentile distance between point clouds."""
    first = np.asarray(first_points, dtype=float)
    second = np.asarray(second_points, dtype=float)
    if len(first) == 0 or len(second) == 0:
        return None
    if first.ndim != 2 or first.shape[1] != 2 or second.ndim != 2 or second.shape[1] != 2:
        raise ValueError("point clouds must have shape (n, 2)")
    if not 0.0 <= percentile <= 1.0:
        raise ValueError("percentile must lie in [0, 1]")
    left = cKDTree(first).query(second)[0]
    right = cKDTree(second).query(first)[0]
    return float(max(np.quantile(left, percentile), np.quantile(right, percentile)))
