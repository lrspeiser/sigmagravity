"""Composite galaxy sources and response metrics for the NBP0 morphology test."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .axisymmetric_permittivity import (
    AxisymmetricGrid,
    acceleration_components,
    double_exponential_density,
    hernquist_density,
    logistic_permittivity,
    midplane_inward_acceleration,
    solve_axisymmetric_helmholtz_smoothing,
    solve_axisymmetric_potential,
)


@dataclass(frozen=True)
class MorphologyParameters:
    stellar_bulge_fraction: float
    disk_vertical_scale_over_Rdisk: float
    bulge_scale_over_Rdisk: float
    gas_fraction: float
    gas_radial_scale_over_Rdisk: float
    gas_vertical_scale_over_Rdisk: float
    minimum_permittivity: float
    log10_critical_density_dimensionless: float
    sharpness: float
    smoothing_length_over_Rdisk: float

    def __post_init__(self) -> None:
        fractions = (self.stellar_bulge_fraction, self.gas_fraction)
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("bulge and gas fractions must be finite and in [0,1]")
        positive = (
            self.disk_vertical_scale_over_Rdisk,
            self.bulge_scale_over_Rdisk,
            self.gas_radial_scale_over_Rdisk,
            self.gas_vertical_scale_over_Rdisk,
            self.sharpness,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("geometry scales and sharpness must be finite and positive")
        if (
            not math.isfinite(self.minimum_permittivity)
            or self.minimum_permittivity <= 0.0
            or self.minimum_permittivity > 1.0
        ):
            raise ValueError("minimum_permittivity must be in (0,1]")
        if not math.isfinite(self.log10_critical_density_dimensionless):
            raise ValueError("critical-density logarithm must be finite")
        if (
            not math.isfinite(self.smoothing_length_over_Rdisk)
            or self.smoothing_length_over_Rdisk < 0.0
        ):
            raise ValueError("smoothing length must be finite and nonnegative")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def composite_galaxy_density(
    grid: AxisymmetricGrid, parameters: MorphologyParameters
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    stellar_mass = 1.0 - parameters.gas_fraction
    bulge_mass = stellar_mass * parameters.stellar_bulge_fraction
    disk_mass = stellar_mass - bulge_mass
    gas_mass = parameters.gas_fraction
    zero = np.zeros((grid.radial_cells, grid.vertical_cells), dtype=float)
    disk = (
        double_exponential_density(
            grid,
            mass=disk_mass,
            radial_scale=1.0,
            vertical_scale=parameters.disk_vertical_scale_over_Rdisk,
        )
        if disk_mass > 0.0
        else zero.copy()
    )
    bulge = (
        hernquist_density(
            grid, mass=bulge_mass, scale_radius=parameters.bulge_scale_over_Rdisk
        )
        if bulge_mass > 0.0
        else zero.copy()
    )
    gas = (
        double_exponential_density(
            grid,
            mass=gas_mass,
            radial_scale=parameters.gas_radial_scale_over_Rdisk,
            vertical_scale=parameters.gas_vertical_scale_over_Rdisk,
        )
        if gas_mass > 0.0
        else zero.copy()
    )
    components = {"disk": disk, "bulge": bulge, "gas": gas}
    masses = {"disk": disk_mass, "bulge": bulge_mass, "gas": gas_mass}
    return disk + bulge + gas, components, masses


def _interpolate_radial(radius: np.ndarray, values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.interp(targets, radius, values, left=np.nan, right=np.nan)


def _log_slope(radius: np.ndarray, values: np.ndarray, low: float, high: float) -> float:
    valid = (
        (radius >= low)
        & (radius <= high)
        & np.isfinite(values)
        & (values > 0.0)
    )
    if valid.sum() < 3:
        return math.nan
    return float(np.polyfit(np.log(radius[valid]), np.log(values[valid]), 1)[0])


def solve_morphology_response(
    grid: AxisymmetricGrid,
    parameters: MorphologyParameters,
    *,
    response_radii: np.ndarray,
    outer_slope_interval: tuple[float, float] = (4.0, 8.0),
) -> dict[str, object]:
    density, _, masses = composite_galaxy_density(grid, parameters)
    basin = solve_axisymmetric_helmholtz_smoothing(
        grid, density, parameters.smoothing_length_over_Rdisk
    )
    epsilon = logistic_permittivity(
        basin,
        minimum_permittivity=parameters.minimum_permittivity,
        critical_density=10.0 ** parameters.log10_critical_density_dimensionless,
        sharpness=parameters.sharpness,
    )
    newtonian = solve_axisymmetric_potential(grid, density, np.ones_like(density))
    modified = solve_axisymmetric_potential(
        grid,
        density,
        epsilon,
        far_permittivity=parameters.minimum_permittivity,
    )
    radial = grid.radial_centers
    g_newtonian = midplane_inward_acceleration(grid, newtonian)
    g_modified = midplane_inward_acceleration(grid, modified)
    valid_positive = (g_newtonian > 0.0) & (g_modified > 0.0)
    enhancement = np.full_like(g_newtonian, np.nan)
    enhancement[valid_positive] = g_modified[valid_positive] / g_newtonian[valid_positive]
    circular_speed_newtonian = np.sqrt(np.maximum(radial * g_newtonian, 0.0))
    circular_speed_modified = np.sqrt(np.maximum(radial * g_modified, 0.0))

    newtonian_radial_acceleration, newtonian_vertical_acceleration = acceleration_components(
        grid, newtonian
    )
    modified_radial_acceleration, modified_vertical_acceleration = acceleration_components(
        grid, modified
    )
    probe_radial_index = int(np.argmin(np.abs(radial - 4.0)))
    probe_vertical_index = int(np.argmin(np.abs(grid.vertical_centers - 1.0)))
    above_radial = -modified_radial_acceleration[
        probe_radial_index, probe_vertical_index
    ]
    above_vertical = -modified_vertical_acceleration[
        probe_radial_index, probe_vertical_index
    ]
    above_radial_newtonian = -newtonian_radial_acceleration[
        probe_radial_index, probe_vertical_index
    ]
    above_vertical_newtonian = -newtonian_vertical_acceleration[
        probe_radial_index, probe_vertical_index
    ]
    modified_direction_ratio = (
        abs(above_vertical) / abs(above_radial) if above_radial != 0.0 else math.inf
    )
    newtonian_direction_ratio = (
        abs(above_vertical_newtonian) / abs(above_radial_newtonian)
        if above_radial_newtonian != 0.0
        else math.inf
    )
    midplane_enhancement = _interpolate_radial(
        radial, enhancement, np.asarray(response_radii, dtype=float)
    )
    geometry_only_enhancement = (
        parameters.minimum_permittivity * midplane_enhancement
    )
    low, high = outer_slope_interval
    return {
        "parameters": parameters.to_dict(),
        "component_masses": masses,
        "response_radii_over_Rdisk": np.asarray(response_radii, dtype=float).tolist(),
        "midplane_acceleration_enhancement": midplane_enhancement.tolist(),
        "geometry_only_enhancement": geometry_only_enhancement.tolist(),
        "newtonian_speed_log_slope": _log_slope(
            radial, circular_speed_newtonian, low, high
        ),
        "modified_speed_log_slope": _log_slope(
            radial, circular_speed_modified, low, high
        ),
        "outer_speed_slope_change": _log_slope(
            radial, circular_speed_modified, low, high
        )
        - _log_slope(radial, circular_speed_newtonian, low, high),
        "epsilon_minimum_realized": float(np.min(epsilon)),
        "epsilon_maximum_realized": float(np.max(epsilon)),
        "epsilon_midplane_at_response_radii": _interpolate_radial(
            radial, epsilon[:, 0], np.asarray(response_radii, dtype=float)
        ).tolist(),
        "above_plane_probe": {
            "radius_over_Rdisk": float(radial[probe_radial_index]),
            "height_over_Rdisk": float(grid.vertical_centers[probe_vertical_index]),
            "inward_radial_acceleration": float(above_radial),
            "toward_plane_vertical_acceleration": float(above_vertical),
            "newtonian_inward_radial_acceleration": float(above_radial_newtonian),
            "newtonian_toward_plane_vertical_acceleration": float(
                above_vertical_newtonian
            ),
            "absolute_vertical_to_radial_ratio": float(modified_direction_ratio),
            "newtonian_absolute_vertical_to_radial_ratio": float(
                newtonian_direction_ratio
            ),
            "constitutive_direction_ratio_change": float(
                modified_direction_ratio / newtonian_direction_ratio
                if newtonian_direction_ratio != 0.0
                else math.inf
            ),
            "radial_acceleration_enhancement": float(
                above_radial / above_radial_newtonian
                if above_radial_newtonian != 0.0
                else math.inf
            ),
            "vertical_acceleration_enhancement": float(
                above_vertical / above_vertical_newtonian
                if above_vertical_newtonian != 0.0
                else math.inf
            ),
        },
    }
