"""Exploratory screened Sigma-field solvers.

The static model is

    div[(1-eta*Sigma**2) grad(Phi)] = 4*pi*G*rho_b
    L**2 laplacian(Sigma) = (rho_b/rho_s - 1)*Sigma + Sigma**3.

It is a Newtonian variational prototype, not a covariant field theory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.integrate import solve_bvp
from scipy.optimize import minimize

from .axisymmetric_permittivity import AxisymmetricGrid


MSUN_G = 1.988409870698051e33
KPC_CM = 3.085677581491367e21
MSUN_KPC3_TO_G_CM3 = MSUN_G / KPC_CM**3


@dataclass(frozen=True)
class SigmaSolve:
    field: np.ndarray
    energy: float
    converged: bool
    iterations: int
    maximum_scaled_gradient: float


def sigma_permittivity(sigma, eta: float) -> np.ndarray:
    """Return epsilon=1-eta*Sigma^2 with its physical bounds enforced."""
    field = np.asarray(sigma, dtype=np.float64)
    if np.any(~np.isfinite(field)) or np.any((field < 0.0) | (field > 1.0)):
        raise ValueError("sigma must be finite and lie in [0, 1]")
    if not math.isfinite(eta) or not 0.0 <= eta < 1.0:
        raise ValueError("eta must be finite and lie in [0, 1)")
    return 1.0 - eta * np.square(field)


def local_sigma_equilibrium(density_g_cm3, rho_s_g_cm3: float) -> np.ndarray:
    """Return the nonnegative homogeneous minimum of the Sigma potential."""
    density = np.asarray(density_g_cm3, dtype=np.float64)
    if np.any(~np.isfinite(density)) or np.any(density < 0.0):
        raise ValueError("density must be finite and nonnegative")
    if not math.isfinite(rho_s_g_cm3) or rho_s_g_cm3 <= 0.0:
        raise ValueError("rho_s_g_cm3 must be finite and positive")
    return np.sqrt(np.maximum(0.0, 1.0 - density / rho_s_g_cm3))


def hernquist_density_g_cm3(radius_kpc, mass_solar: float, scale_kpc: float) -> np.ndarray:
    radius = np.asarray(radius_kpc, dtype=np.float64)
    if np.any(radius <= 0.0) or mass_solar <= 0.0 or scale_kpc <= 0.0:
        raise ValueError("Hernquist radii, mass, and scale must be positive")
    density_msun_kpc3 = (
        mass_solar
        * scale_kpc
        / (2.0 * math.pi * radius * np.power(radius + scale_kpc, 3))
    )
    return density_msun_kpc3 * MSUN_KPC3_TO_G_CM3


def hernquist_enclosed_mass_solar(radius_kpc, mass_solar: float, scale_kpc: float) -> np.ndarray:
    radius = np.asarray(radius_kpc, dtype=np.float64)
    return mass_solar * np.square(radius / (radius + scale_kpc))


def geometric_radial_faces(
    cells: int, radius_min_kpc: float, radius_max_kpc: float
) -> np.ndarray:
    if cells < 16:
        raise ValueError("at least 16 radial cells are required")
    if radius_min_kpc <= 0.0 or radius_max_kpc <= radius_min_kpc:
        raise ValueError("radial limits must be positive and increasing")
    return np.r_[0.0, np.geomspace(radius_min_kpc, radius_max_kpc, cells)]


def radial_cell_centers(radius_faces: np.ndarray) -> np.ndarray:
    faces = np.asarray(radius_faces, dtype=np.float64)
    numerator = 0.75 * (np.power(faces[1:], 4) - np.power(faces[:-1], 4))
    denominator = np.power(faces[1:], 3) - np.power(faces[:-1], 3)
    return numerator / denominator


def _bounded_minimum(
    matrix: sparse.csr_matrix,
    boundary_source: np.ndarray,
    volumes: np.ndarray,
    density_ratio: np.ndarray,
    starts: list[np.ndarray],
) -> SigmaSolve:
    a = density_ratio - 1.0

    def objective(values: np.ndarray) -> tuple[float, np.ndarray]:
        matrix_values = matrix @ values
        energy = (
            0.5 * float(values @ matrix_values)
            - float(boundary_source @ values)
            + float(np.sum(volumes * (0.5 * a * np.square(values) + 0.25 * values**4)))
        )
        gradient = matrix_values - boundary_source + volumes * (a * values + values**3)
        return energy, gradient

    best = None
    for start in starts:
        result = minimize(
            lambda values: objective(values),
            np.clip(np.asarray(start, dtype=np.float64), 0.0, 1.0),
            method="L-BFGS-B",
            jac=True,
            bounds=[(0.0, 1.0)] * len(start),
            options={"maxiter": 1200, "ftol": 1.0e-13, "gtol": 1.0e-9, "maxls": 50},
        )
        if best is None or result.fun < best.fun:
            best = result
    _, gradient = objective(best.x)
    scale = np.maximum(volumes, np.finfo(float).tiny)
    return SigmaSolve(
        field=np.clip(best.x, 0.0, 1.0),
        energy=float(best.fun),
        converged=bool(best.success),
        iterations=int(best.nit),
        maximum_scaled_gradient=float(np.max(np.abs(gradient) / scale)),
    )


def solve_spherical_sigma(
    radius_faces_kpc,
    density_g_cm3,
    *,
    rho_s_g_cm3: float,
    length_kpc: float,
    outer_sigma: float = 1.0,
) -> SigmaSolve:
    """Solve the spherical Sigma Euler equation with a fixed outer boundary.

    Multiple initial branches are tried and the lowest-energy converged,
    nonnegative solution is retained.  This matters for a sufficiently large
    vacuum region, where both the symmetric Sigma=0 solution and a broken
    Sigma>0 solution are stationary points.
    """
    faces = np.asarray(radius_faces_kpc, dtype=np.float64)
    density = np.asarray(density_g_cm3, dtype=np.float64)
    if faces.ndim != 1 or len(faces) != len(density) + 1:
        raise ValueError("radius faces must contain one more value than density")
    if faces[0] != 0.0 or np.any(np.diff(faces) <= 0.0):
        raise ValueError("radius faces must start at zero and strictly increase")
    if np.any(~np.isfinite(density)) or np.any(density < 0.0):
        raise ValueError("density must be finite and nonnegative")
    if rho_s_g_cm3 <= 0.0 or length_kpc < 0.0 or not 0.0 <= outer_sigma <= 1.0:
        raise ValueError("invalid Sigma parameters")
    equilibrium = local_sigma_equilibrium(density, rho_s_g_cm3)
    if length_kpc == 0.0:
        return SigmaSolve(equilibrium, math.nan, True, 0, 0.0)

    centers = radial_cell_centers(faces)
    mesh = np.r_[centers, faces[-1]]
    log_centers = np.log(centers)

    def density_at(radius: np.ndarray) -> np.ndarray:
        return np.interp(np.log(radius), log_centers, density)

    def equation(radius: np.ndarray, values: np.ndarray) -> np.ndarray:
        sigma, derivative = values
        coefficient = density_at(radius) / rho_s_g_cm3 - 1.0
        curvature = (coefficient * sigma + sigma**3) / length_kpc**2
        return np.vstack((derivative, curvature - 2.0 * derivative / radius))

    def boundary(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.array((left[1], right[0] - outer_sigma))

    def initial_values(field: np.ndarray) -> np.ndarray:
        field = np.r_[field, outer_sigma]
        return np.vstack((field, np.gradient(field, mesh)))

    starts = [initial_values(equilibrium)]
    constant = np.full_like(equilibrium, outer_sigma)
    if not np.allclose(constant, equilibrium, rtol=0.0, atol=1.0e-8):
        starts.append(initial_values(constant))
    if outer_sigma == 0.0:
        starts.append(np.zeros((2, len(mesh)), dtype=np.float64))

    def energy(solution) -> float:
        sample_radius = np.geomspace(mesh[0], mesh[-1], max(1000, 2 * len(mesh)))
        sigma, derivative = solution.sol(sample_radius)
        coefficient = density_at(sample_radius) / rho_s_g_cm3 - 1.0
        integrand = sample_radius**2 * (
            0.5 * length_kpc**2 * derivative**2
            + 0.5 * coefficient * sigma**2
            + 0.25 * sigma**4
        )
        return float(np.trapezoid(integrand, sample_radius))

    candidates = []
    for start in starts:
        result = solve_bvp(
            equation,
            boundary,
            mesh,
            start,
            tol=1.0e-4,
            max_nodes=20_000,
            verbose=0,
        )
        if result.success:
            sampled = result.sol(mesh)[0]
            if np.min(sampled) >= -1.0e-5 and np.max(sampled) <= 1.0 + 1.0e-5:
                candidates.append((energy(result), result))
    if not candidates:
        return SigmaSolve(equilibrium, math.nan, False, 0, math.inf)
    best_energy, best = min(candidates, key=lambda item: item[0])
    field = np.clip(best.sol(centers)[0], 0.0, 1.0)
    maximum_residual = float(np.max(best.rms_residuals))
    return SigmaSolve(
        field=field,
        energy=best_energy,
        converged=bool(best.success and maximum_residual <= 2.0e-4),
        iterations=int(best.niter),
        maximum_scaled_gradient=maximum_residual,
    )


def solve_axisymmetric_sigma(
    grid: AxisymmetricGrid,
    density_g_cm3: np.ndarray,
    *,
    rho_s_g_cm3: float,
    length: float,
    outer_sigma: float = 1.0,
) -> SigmaSolve:
    """Minimize the axisymmetric Sigma energy with reflected axis/midplane."""
    density = np.asarray(density_g_cm3, dtype=np.float64)
    expected = (grid.radial_cells, grid.vertical_cells)
    if density.shape != expected or np.any(~np.isfinite(density)) or np.any(density < 0.0):
        raise ValueError(f"density must be a finite nonnegative field of shape {expected}")
    if rho_s_g_cm3 <= 0.0 or length < 0.0 or not 0.0 <= outer_sigma <= 1.0:
        raise ValueError("invalid Sigma parameters")
    equilibrium = local_sigma_equilibrium(density, rho_s_g_cm3)
    if length == 0.0:
        return SigmaSolve(equilibrium, math.nan, True, 0, 0.0)

    nr, nz = expected
    dr, dz = grid.radial_step, grid.vertical_step
    radial_faces = np.arange(nr + 1, dtype=float) * dr
    vertical_face_area = 0.5 * (
        np.square(radial_faces[1:]) - np.square(radial_faces[:-1])
    )
    volumes = vertical_face_area[:, None] * dz * np.ones((1, nz))
    rows: list[int] = []
    columns: list[int] = []
    entries: list[float] = []
    boundary_source = np.zeros(nr * nz, dtype=np.float64)
    length_squared = length**2

    def index(i: int, j: int) -> int:
        return i * nz + j

    def add(i: int, j: int, value: float) -> None:
        rows.append(i)
        columns.append(j)
        entries.append(value)

    diagonal = np.zeros(nr * nz, dtype=np.float64)
    for i in range(nr):
        for j in range(nz):
            current = index(i, j)
            if i + 1 < nr:
                neighbor = index(i + 1, j)
                conductance = length_squared * radial_faces[i + 1] * dz / dr
                diagonal[current] += conductance
                diagonal[neighbor] += conductance
                add(current, neighbor, -conductance)
                add(neighbor, current, -conductance)
            else:
                conductance = length_squared * radial_faces[-1] * dz / (0.5 * dr)
                diagonal[current] += conductance
                boundary_source[current] += conductance * outer_sigma
            if j + 1 < nz:
                neighbor = index(i, j + 1)
                conductance = length_squared * vertical_face_area[i] / dz
                diagonal[current] += conductance
                diagonal[neighbor] += conductance
                add(current, neighbor, -conductance)
                add(neighbor, current, -conductance)
            else:
                conductance = length_squared * vertical_face_area[i] / (0.5 * dz)
                diagonal[current] += conductance
                boundary_source[current] += conductance * outer_sigma
    for current, value in enumerate(diagonal):
        add(current, current, value)
    matrix = sparse.csr_matrix((entries, (rows, columns)), shape=(nr * nz, nr * nz))
    starts = [equilibrium.ravel(), np.zeros(nr * nz), np.ones(nr * nz) * outer_sigma]
    result = _bounded_minimum(
        matrix,
        boundary_source,
        volumes.ravel(),
        (density / rho_s_g_cm3).ravel(),
        starts,
    )
    return SigmaSolve(
        field=result.field.reshape(expected),
        energy=result.energy,
        converged=result.converged,
        iterations=result.iterations,
        maximum_scaled_gradient=result.maximum_scaled_gradient,
    )
