"""Finite-volume axisymmetric solvers for morphology-dependent permittivity tests.

The solvers use cell-centered cylindrical coordinates on the R>=0, z>=0
quadrant, impose reflection symmetry at the axis and midplane, and use explicit
Dirichlet conditions at the outer radial and vertical boundaries.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class AxisymmetricGrid:
    radial_cells: int
    vertical_cells: int
    radial_max: float
    vertical_max: float

    def __post_init__(self) -> None:
        if self.radial_cells < 4 or self.vertical_cells < 4:
            raise ValueError("axisymmetric grids require at least four cells per axis")
        if not math.isfinite(self.radial_max) or self.radial_max <= 0.0:
            raise ValueError("radial_max must be finite and positive")
        if not math.isfinite(self.vertical_max) or self.vertical_max <= 0.0:
            raise ValueError("vertical_max must be finite and positive")

    @property
    def radial_step(self) -> float:
        return self.radial_max / self.radial_cells

    @property
    def vertical_step(self) -> float:
        return self.vertical_max / self.vertical_cells

    @property
    def radial_centers(self) -> np.ndarray:
        return (np.arange(self.radial_cells, dtype=float) + 0.5) * self.radial_step

    @property
    def vertical_centers(self) -> np.ndarray:
        return (np.arange(self.vertical_cells, dtype=float) + 0.5) * self.vertical_step

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.radial_centers, self.vertical_centers, indexing="ij")

    def cell_volumes(self) -> np.ndarray:
        """Full 3D volumes represented by quadrant cells, including z reflection."""
        radial_faces = np.arange(self.radial_cells + 1, dtype=float) * self.radial_step
        annular_area = math.pi * (
            np.square(radial_faces[1:]) - np.square(radial_faces[:-1])
        )
        return 2.0 * annular_area[:, None] * self.vertical_step * np.ones(
            (1, self.vertical_cells)
        )


def represented_mass(grid: AxisymmetricGrid, density: np.ndarray) -> float:
    values = _validated_field(grid, density, "density")
    return float(np.sum(values * grid.cell_volumes()))


def normalize_density(
    grid: AxisymmetricGrid, density: np.ndarray, *, target_mass: float = 1.0
) -> np.ndarray:
    values = _validated_field(grid, density, "density")
    if np.any(values < 0.0):
        raise ValueError("density must be nonnegative")
    if not math.isfinite(target_mass) or target_mass <= 0.0:
        raise ValueError("target_mass must be finite and positive")
    current = represented_mass(grid, values)
    if current <= 0.0:
        raise ValueError("density must contain positive mass")
    return values * (target_mass / current)


def double_exponential_density(
    grid: AxisymmetricGrid,
    *,
    mass: float,
    radial_scale: float,
    vertical_scale: float,
) -> np.ndarray:
    """Return rho proportional exp(-R/Rd) sech^2(z/zd), normalized on the grid."""
    if not math.isfinite(radial_scale) or radial_scale <= 0.0:
        raise ValueError("radial_scale must be finite and positive")
    if not math.isfinite(vertical_scale) or vertical_scale <= 0.0:
        raise ValueError("vertical_scale must be finite and positive")
    radial, vertical = grid.mesh()
    density = np.exp(-radial / radial_scale) / np.square(
        np.cosh(np.minimum(vertical / vertical_scale, 350.0))
    )
    return normalize_density(grid, density, target_mass=mass)


def hernquist_density(
    grid: AxisymmetricGrid,
    *,
    mass: float,
    scale_radius: float,
) -> np.ndarray:
    """Return a spherical Hernquist density, cell-sampled and grid-normalized."""
    if not math.isfinite(scale_radius) or scale_radius <= 0.0:
        raise ValueError("scale_radius must be finite and positive")
    radial, vertical = grid.mesh()
    spherical_radius = np.sqrt(np.square(radial) + np.square(vertical))
    density = scale_radius / (
        spherical_radius * np.power(spherical_radius + scale_radius, 3)
    )
    return normalize_density(grid, density, target_mass=mass)


def miyamoto_nagai_density(
    grid: AxisymmetricGrid,
    *,
    mass: float,
    radial_scale: float,
    vertical_scale: float,
) -> np.ndarray:
    """Return the analytic Miyamoto-Nagai density, normalized on the finite grid."""
    if not math.isfinite(radial_scale) or radial_scale < 0.0:
        raise ValueError("radial_scale must be finite and nonnegative")
    if not math.isfinite(vertical_scale) or vertical_scale <= 0.0:
        raise ValueError("vertical_scale must be finite and positive")
    radial, vertical = grid.mesh()
    zeta = np.sqrt(np.square(vertical) + vertical_scale**2)
    shifted = radial_scale + zeta
    numerator = radial_scale * np.square(radial) + (
        radial_scale + 3.0 * zeta
    ) * np.square(shifted)
    denominator = np.power(zeta, 3) * np.power(
        np.square(radial) + np.square(shifted), 2.5
    )
    density = vertical_scale**2 * numerator / denominator
    return normalize_density(grid, density, target_mass=mass)


def miyamoto_nagai_potential(
    radial: np.ndarray | float,
    vertical: np.ndarray | float,
    *,
    mass: float,
    radial_scale: float,
    vertical_scale: float,
    gravitational_constant: float = 1.0,
) -> np.ndarray:
    radial_values = np.asarray(radial, dtype=float)
    vertical_values = np.asarray(vertical, dtype=float)
    zeta = np.sqrt(np.square(vertical_values) + vertical_scale**2)
    return -gravitational_constant * mass / np.sqrt(
        np.square(radial_values) + np.square(radial_scale + zeta)
    )


def _validated_field(
    grid: AxisymmetricGrid, values: np.ndarray, label: str
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    expected = (grid.radial_cells, grid.vertical_cells)
    if array.shape != expected:
        raise ValueError(f"{label} must have shape {expected}")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{label} must be finite")
    return array


def _harmonic(left: float, right: float) -> float:
    return 2.0 * left * right / (left + right)


def _boundary_value(
    boundary: Callable[[float, float], float] | None,
    radial: float,
    vertical: float,
    *,
    mass: float,
    gravitational_constant: float,
    far_permittivity: float,
) -> float:
    if boundary is not None:
        value = float(boundary(radial, vertical))
        if not math.isfinite(value):
            raise ValueError("boundary potential must be finite")
        return value
    distance = math.hypot(radial, vertical)
    return -gravitational_constant * mass / (far_permittivity * distance)


def solve_axisymmetric_potential(
    grid: AxisymmetricGrid,
    density: np.ndarray,
    permittivity: np.ndarray,
    *,
    gravitational_constant: float = 1.0,
    far_permittivity: float | None = None,
    boundary_potential: Callable[[float, float], float] | None = None,
) -> np.ndarray:
    """Solve div(epsilon grad Phi)=4 pi G rho on the reflected quadrant."""
    rho = _validated_field(grid, density, "density")
    epsilon = _validated_field(grid, permittivity, "permittivity")
    if np.any(rho < 0.0):
        raise ValueError("density must be nonnegative")
    if np.any(epsilon <= 0.0):
        raise ValueError("permittivity must be positive")
    if not math.isfinite(gravitational_constant) or gravitational_constant <= 0.0:
        raise ValueError("gravitational_constant must be finite and positive")
    if far_permittivity is None:
        far_permittivity = float(np.min(epsilon))
    if not math.isfinite(far_permittivity) or far_permittivity <= 0.0:
        raise ValueError("far_permittivity must be finite and positive")

    total_mass = represented_mass(grid, rho)
    nr = grid.radial_cells
    nz = grid.vertical_cells
    dr = grid.radial_step
    dz = grid.vertical_step
    radial_faces = np.arange(nr + 1, dtype=float) * dr
    vertical_face_area = 0.5 * (
        np.square(radial_faces[1:]) - np.square(radial_faces[:-1])
    )
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    rhs = np.zeros(nr * nz, dtype=float)

    def flat_index(i: int, j: int) -> int:
        return i * nz + j

    for i in range(nr):
        radial_center = (i + 0.5) * dr
        radial_inner = radial_faces[i]
        radial_outer = radial_faces[i + 1]
        vertical_area = vertical_face_area[i]
        cell_volume_per_radian = vertical_area * dz
        for j in range(nz):
            vertical_center = (j + 0.5) * dz
            index = flat_index(i, j)
            diagonal = 0.0
            rhs[index] = -4.0 * math.pi * gravitational_constant * rho[
                i, j
            ] * cell_volume_per_radian

            if i > 0:
                conductance = (
                    radial_inner
                    * dz
                    * _harmonic(epsilon[i, j], epsilon[i - 1, j])
                    / dr
                )
                diagonal += conductance
                rows.append(index)
                columns.append(flat_index(i - 1, j))
                values.append(-conductance)
            if i + 1 < nr:
                conductance = (
                    radial_outer
                    * dz
                    * _harmonic(epsilon[i, j], epsilon[i + 1, j])
                    / dr
                )
                diagonal += conductance
                rows.append(index)
                columns.append(flat_index(i + 1, j))
                values.append(-conductance)
            else:
                boundary_epsilon = _harmonic(epsilon[i, j], far_permittivity)
                conductance = radial_outer * dz * boundary_epsilon / (0.5 * dr)
                diagonal += conductance
                rhs[index] += conductance * _boundary_value(
                    boundary_potential,
                    grid.radial_max,
                    vertical_center,
                    mass=total_mass,
                    gravitational_constant=gravitational_constant,
                    far_permittivity=far_permittivity,
                )

            if j > 0:
                conductance = (
                    vertical_area
                    * _harmonic(epsilon[i, j], epsilon[i, j - 1])
                    / dz
                )
                diagonal += conductance
                rows.append(index)
                columns.append(flat_index(i, j - 1))
                values.append(-conductance)
            if j + 1 < nz:
                conductance = (
                    vertical_area
                    * _harmonic(epsilon[i, j], epsilon[i, j + 1])
                    / dz
                )
                diagonal += conductance
                rows.append(index)
                columns.append(flat_index(i, j + 1))
                values.append(-conductance)
            else:
                boundary_epsilon = _harmonic(epsilon[i, j], far_permittivity)
                conductance = vertical_area * boundary_epsilon / (0.5 * dz)
                diagonal += conductance
                rhs[index] += conductance * _boundary_value(
                    boundary_potential,
                    radial_center,
                    grid.vertical_max,
                    mass=total_mass,
                    gravitational_constant=gravitational_constant,
                    far_permittivity=far_permittivity,
                )

            rows.append(index)
            columns.append(index)
            values.append(diagonal)

    matrix = sparse.csr_matrix((values, (rows, columns)), shape=(nr * nz, nr * nz))
    potential = spsolve(matrix, rhs)
    if np.any(~np.isfinite(potential)):
        raise RuntimeError("axisymmetric potential solve returned non-finite values")
    return potential.reshape((nr, nz))


def solve_axisymmetric_helmholtz_smoothing(
    grid: AxisymmetricGrid,
    source: np.ndarray,
    smoothing_length: float,
) -> np.ndarray:
    """Solve (1-L^2 Laplacian) X=source with outer X=0 and reflection symmetry."""
    values_source = _validated_field(grid, source, "source")
    if np.any(values_source < 0.0):
        raise ValueError("source must be nonnegative")
    if not math.isfinite(smoothing_length) or smoothing_length < 0.0:
        raise ValueError("smoothing_length must be finite and nonnegative")
    if smoothing_length == 0.0:
        return values_source.copy()

    nr = grid.radial_cells
    nz = grid.vertical_cells
    dr = grid.radial_step
    dz = grid.vertical_step
    radial_faces = np.arange(nr + 1, dtype=float) * dr
    vertical_face_area = 0.5 * (
        np.square(radial_faces[1:]) - np.square(radial_faces[:-1])
    )
    rows: list[int] = []
    columns: list[int] = []
    entries: list[float] = []
    rhs = np.zeros(nr * nz, dtype=float)
    length_squared = smoothing_length**2

    def flat_index(i: int, j: int) -> int:
        return i * nz + j

    for i in range(nr):
        radial_inner = radial_faces[i]
        radial_outer = radial_faces[i + 1]
        vertical_area = vertical_face_area[i]
        cell_volume = vertical_area * dz
        for j in range(nz):
            index = flat_index(i, j)
            diagonal = cell_volume
            rhs[index] = values_source[i, j] * cell_volume
            if i > 0:
                conductance = radial_inner * dz / dr
                diagonal += length_squared * conductance
                rows.append(index)
                columns.append(flat_index(i - 1, j))
                entries.append(-length_squared * conductance)
            if i + 1 < nr:
                conductance = radial_outer * dz / dr
                diagonal += length_squared * conductance
                rows.append(index)
                columns.append(flat_index(i + 1, j))
                entries.append(-length_squared * conductance)
            else:
                diagonal += length_squared * radial_outer * dz / (0.5 * dr)
            if j > 0:
                conductance = vertical_area / dz
                diagonal += length_squared * conductance
                rows.append(index)
                columns.append(flat_index(i, j - 1))
                entries.append(-length_squared * conductance)
            if j + 1 < nz:
                conductance = vertical_area / dz
                diagonal += length_squared * conductance
                rows.append(index)
                columns.append(flat_index(i, j + 1))
                entries.append(-length_squared * conductance)
            else:
                diagonal += length_squared * vertical_area / (0.5 * dz)
            rows.append(index)
            columns.append(index)
            entries.append(diagonal)
    matrix = sparse.csr_matrix((entries, (rows, columns)), shape=(nr * nz, nr * nz))
    smoothed = spsolve(matrix, rhs).reshape((nr, nz))
    if np.any(~np.isfinite(smoothed)) or np.any(smoothed < -1.0e-12):
        raise RuntimeError("Helmholtz smoothing returned a nonphysical field")
    return np.maximum(smoothed, 0.0)


def logistic_permittivity(
    basin_density: np.ndarray,
    *,
    minimum_permittivity: float,
    critical_density: float,
    sharpness: float,
) -> np.ndarray:
    """Map a nonnegative basin-density field to epsilon0<=epsilon<=1."""
    values = np.asarray(basin_density, dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("basin_density must be finite and nonnegative")
    if (
        not math.isfinite(minimum_permittivity)
        or minimum_permittivity <= 0.0
        or minimum_permittivity > 1.0
    ):
        raise ValueError("minimum_permittivity must be in (0,1]")
    if not math.isfinite(critical_density) or critical_density <= 0.0:
        raise ValueError("critical_density must be finite and positive")
    if not math.isfinite(sharpness) or sharpness <= 0.0:
        raise ValueError("sharpness must be finite and positive")
    ratio_power = np.power(values / critical_density, sharpness)
    activation = ratio_power / (1.0 + ratio_power)
    return minimum_permittivity + (1.0 - minimum_permittivity) * activation


def acceleration_components(
    grid: AxisymmetricGrid, potential: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = _validated_field(grid, potential, "potential")
    radial_gradient = np.gradient(values, grid.radial_step, axis=0, edge_order=2)
    vertical_gradient = np.gradient(values, grid.vertical_step, axis=1, edge_order=2)
    return -radial_gradient, -vertical_gradient


def midplane_inward_acceleration(
    grid: AxisymmetricGrid, potential: np.ndarray
) -> np.ndarray:
    radial_acceleration, _ = acceleration_components(grid, potential)
    return -radial_acceleration[:, 0]
