"""Observation-matched galaxy scenes and a small virtual telescope.

Replica mode intentionally uses the observed rotation curve.  Its purpose is to
verify that a seeded scene can reproduce the data product that describes a real
galaxy.  It is not a gravity test.  Blind-physics mode can reuse the same light
scene but must replace ``observed_velocity_km_s`` with a theory prediction.

The available SPARC inputs are radial 3.6 micron profiles, not raw two-
dimensional images.  Consequently the renderer is an axisymmetric observable-
equivalent reconstruction; it does not invent evidence for bars or spiral arms.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import RotationCurve, parse_rotation_curve, parse_table1
from .sparc_morphology import parse_sparc_metadata


def _finite_1d(values, name: str, *, nonnegative: bool = False) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be a finite one-dimensional array")
    if nonnegative and np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return array


@dataclass(frozen=True)
class LightProfile:
    """Face-on nonparametric disk/bulge surface luminosity profile."""

    radius_kpc: np.ndarray
    disk_lsun_pc2: np.ndarray
    bulge_lsun_pc2: np.ndarray

    def __post_init__(self) -> None:
        radius = _finite_1d(self.radius_kpc, "radius_kpc")
        disk = _finite_1d(self.disk_lsun_pc2, "disk_lsun_pc2", nonnegative=True)
        bulge = _finite_1d(self.bulge_lsun_pc2, "bulge_lsun_pc2", nonnegative=True)
        if len(radius) != len(disk) or len(radius) != len(bulge):
            raise ValueError("light profile arrays have inconsistent lengths")
        order = np.argsort(radius, kind="stable")
        radius, disk, bulge = radius[order], disk[order], bulge[order]
        if np.any(radius < 0.0) or np.any(np.diff(radius) <= 0.0):
            raise ValueError("profile radii must be nonnegative and unique")
        object.__setattr__(self, "radius_kpc", radius)
        object.__setattr__(self, "disk_lsun_pc2", disk)
        object.__setattr__(self, "bulge_lsun_pc2", bulge)

    @property
    def total_lsun(self) -> float:
        return self._integrated_luminosity(self.disk_lsun_pc2 + self.bulge_lsun_pc2)

    @property
    def disk_lsun(self) -> float:
        return self._integrated_luminosity(self.disk_lsun_pc2)

    @property
    def bulge_lsun(self) -> float:
        return self._integrated_luminosity(self.bulge_lsun_pc2)

    def _integrated_luminosity(self, density: np.ndarray) -> float:
        # The interpolation policy holds the innermost measured density to the
        # origin, so the numerical luminosity integral must use the same policy.
        radius_pc = self.radius_kpc * 1000.0
        extended = density
        if radius_pc[0] > 0.0:
            radius_pc = np.concatenate(([0.0], radius_pc))
            extended = np.concatenate(([density[0]], extended))
        return float(2.0 * math.pi * np.trapezoid(extended * radius_pc, radius_pc))

    def interpolate(self, radius_kpc, component: str = "total") -> np.ndarray:
        radius = np.asarray(radius_kpc, dtype=np.float64)
        if component == "disk":
            values = self.disk_lsun_pc2
        elif component == "bulge":
            values = self.bulge_lsun_pc2
        elif component == "total":
            values = self.disk_lsun_pc2 + self.bulge_lsun_pc2
        else:
            raise ValueError(f"unknown light component {component!r}")
        # Linear interpolation preserves sharp measured structure and permits
        # true zero-bulge profiles.  Light outside the observed aperture is not
        # invented.
        return np.interp(radius, self.radius_kpc, values, left=values[0], right=0.0)


@dataclass(frozen=True)
class AngularPhotometry:
    radius_arcsec: np.ndarray
    surface_brightness_mag_arcsec2: np.ndarray
    uncertainty_mag: np.ndarray
    keep_flag: np.ndarray


@dataclass(frozen=True)
class GalaxyReplicaSeed:
    name: str
    distance_mpc: float
    inclination_deg: float
    hubble_type: int
    quality: int
    light: LightProfile
    angular_photometry: AngularPhotometry
    rotation: RotationCurve

    @property
    def bulge_fraction(self) -> float:
        return self.light.bulge_lsun / max(self.light.total_lsun, 1.0e-30)


@dataclass(frozen=True)
class RenderedGalaxy:
    name: str
    x_kpc: np.ndarray
    y_kpc: np.ndarray
    disk_lsun_pc2: np.ndarray
    bulge_lsun_pc2: np.ndarray
    line_of_sight_velocity_km_s: np.ndarray
    finite_velocity: np.ndarray
    apparent_axis_ratio: float

    @property
    def total_lsun_pc2(self) -> np.ndarray:
        return self.disk_lsun_pc2 + self.bulge_lsun_pc2


@dataclass(frozen=True)
class ReplicaParticles:
    name: str
    positions_kpc: np.ndarray
    velocities_km_s: np.ndarray
    luminosities_lsun: np.ndarray
    components: np.ndarray
    fingerprint: str


def parse_light_profile(path: Path) -> LightProfile:
    rows: list[list[float]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Expected three profile columns at {path}:{line_number}")
            try:
                rows.append([float(value) for value in parts[:3]])
            except ValueError as exc:
                raise ValueError(f"Non-numeric light profile at {path}:{line_number}") from exc
    if not rows:
        raise ValueError(f"No light rows found in {path}")
    values = np.asarray(rows, dtype=np.float64)
    # A few official files repeat a rounded radius.  Collapse those rows by
    # their mean surface brightness instead of choosing one arbitrarily.
    unique_radius, inverse = np.unique(values[:, 0], return_inverse=True)
    disk = np.zeros(len(unique_radius), dtype=np.float64)
    bulge = np.zeros(len(unique_radius), dtype=np.float64)
    counts = np.zeros(len(unique_radius), dtype=np.float64)
    np.add.at(disk, inverse, values[:, 1])
    np.add.at(bulge, inverse, values[:, 2])
    np.add.at(counts, inverse, 1.0)
    return LightProfile(unique_radius, disk / counts, bulge / counts)


def parse_angular_photometry(path: Path) -> AngularPhotometry:
    rows: list[list[float]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if (
                not line.strip()
                or line.lstrip().startswith("#")
                or line.lower().lstrip().startswith("radius")
            ):
                continue
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Expected four photometry columns at {path}:{line_number}")
            try:
                rows.append([float(value) for value in parts[:4]])
            except ValueError as exc:
                raise ValueError(f"Non-numeric photometry at {path}:{line_number}") from exc
    if not rows:
        raise ValueError(f"No photometry rows found in {path}")
    values = np.asarray(rows, dtype=np.float64)
    return AngularPhotometry(
        values[:, 0], values[:, 1], values[:, 3], values[:, 2].astype(np.int64)
    )


def load_replica_seed(
    name: str,
    sparc_directory: Path,
    photometric_directory: Path,
    decomposition_directory: Path,
) -> GalaxyReplicaSeed:
    sparc_directory = Path(sparc_directory)
    metadata = parse_table1(sparc_directory / "table1.dat")
    extended = parse_sparc_metadata(sparc_directory / "table1.dat").set_index("galaxy")
    if name not in metadata or name not in extended.index:
        raise KeyError(f"Unknown SPARC galaxy {name}")
    rotation = parse_rotation_curve(
        sparc_directory / "rotmod" / f"{name}_rotmod.dat", metadata[name]
    )
    row = extended.loc[name]
    return GalaxyReplicaSeed(
        name=name,
        distance_mpc=float(row.distance_mpc),
        inclination_deg=float(row.inclination_deg),
        hubble_type=int(row.hubble_type),
        quality=int(row.quality),
        light=parse_light_profile(Path(decomposition_directory) / f"{name}.dens"),
        angular_photometry=parse_angular_photometry(
            Path(photometric_directory) / f"{name}.sfb"
        ),
        rotation=rotation,
    )


def valid_rotation_mask(seed: GalaxyReplicaSeed) -> np.ndarray:
    curve = seed.rotation
    return (
        np.isfinite(curve.radius_kpc)
        & np.isfinite(curve.velocity_observed_kms)
        & np.isfinite(curve.velocity_error_kms)
        & (curve.radius_kpc > 0.0)
        & (curve.velocity_observed_kms > 0.0)
        & (curve.velocity_error_kms > 0.0)
    )


def apparent_disk_axis_ratio(inclination_deg: float, intrinsic_axis_ratio: float = 0.12) -> float:
    cosine = math.cos(math.radians(float(inclination_deg)))
    sine = math.sin(math.radians(float(inclination_deg)))
    return float(math.sqrt(cosine * cosine + intrinsic_axis_ratio**2 * sine * sine))


def _rotation_at(seed: GalaxyReplicaSeed, radius_kpc: np.ndarray) -> np.ndarray:
    mask = valid_rotation_mask(seed)
    curve = seed.rotation
    return np.interp(
        radius_kpc,
        curve.radius_kpc[mask],
        curve.velocity_observed_kms[mask],
        left=curve.velocity_observed_kms[mask][0],
        right=curve.velocity_observed_kms[mask][-1],
    )


def render_replica(
    seed: GalaxyReplicaSeed,
    velocity_radius_kpc,
    circular_velocity_km_s,
    *,
    pixels: int = 257,
    extent_multiplier: float = 1.08,
    intrinsic_disk_axis_ratio: float = 0.12,
) -> RenderedGalaxy:
    """Render a galaxy using an explicitly supplied circular-speed prediction.

    Blind physics tests call this function with theory output.  The API requires
    the velocity anchors explicitly so the observed curve is never an implicit
    fallback.
    """
    if pixels < 33 or pixels % 2 == 0:
        raise ValueError("pixels must be an odd integer of at least 33")
    if extent_multiplier < 1.0:
        raise ValueError("extent_multiplier must be at least one")
    velocity_radius = _finite_1d(velocity_radius_kpc, "velocity_radius_kpc")
    velocity = _finite_1d(circular_velocity_km_s, "circular_velocity_km_s", nonnegative=True)
    if len(velocity_radius) != len(velocity):
        raise ValueError("velocity radius and value arrays have inconsistent lengths")
    order = np.argsort(velocity_radius, kind="stable")
    velocity_radius, velocity = velocity_radius[order], velocity[order]
    if np.any(velocity_radius <= 0.0) or np.any(np.diff(velocity_radius) <= 0.0):
        raise ValueError("velocity radii must be positive and unique")
    extent_anchor = max(
        float(seed.light.radius_kpc[-1]),
        float(velocity_radius[-1]),
    )
    extent = extent_multiplier * extent_anchor
    coordinates = np.linspace(-extent, extent, pixels)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    axis_ratio = apparent_disk_axis_ratio(seed.inclination_deg, intrinsic_disk_axis_ratio)
    disk_radius = np.sqrt(x_grid**2 + (y_grid / axis_ratio) ** 2)
    sky_radius = np.sqrt(x_grid**2 + y_grid**2)
    # Division by projected axis ratio conserves disk luminosity under the
    # sky-plane compression.
    disk = seed.light.interpolate(disk_radius, "disk") / axis_ratio
    bulge = seed.light.interpolate(sky_radius, "bulge")
    rotation = np.interp(
        disk_radius,
        velocity_radius,
        velocity,
        left=velocity[0],
        right=velocity[-1],
    )
    cos_azimuth = np.divide(x_grid, disk_radius, out=np.zeros_like(x_grid), where=disk_radius > 0.0)
    velocity = rotation * math.sin(math.radians(seed.inclination_deg)) * cos_azimuth
    finite_velocity = (disk + bulge) > 0.0
    velocity = np.where(finite_velocity, velocity, np.nan)
    return RenderedGalaxy(
        name=seed.name,
        x_kpc=x_grid,
        y_kpc=y_grid,
        disk_lsun_pc2=disk,
        bulge_lsun_pc2=bulge,
        line_of_sight_velocity_km_s=velocity,
        finite_velocity=finite_velocity,
        apparent_axis_ratio=axis_ratio,
    )


def render_observed_replica(
    seed: GalaxyReplicaSeed,
    *,
    pixels: int = 257,
    extent_multiplier: float = 1.08,
    intrinsic_disk_axis_ratio: float = 0.12,
) -> RenderedGalaxy:
    """Replica-mode wrapper that deliberately uses the observed curve."""
    mask = valid_rotation_mask(seed)
    return render_replica(
        seed,
        seed.rotation.radius_kpc[mask],
        seed.rotation.velocity_observed_kms[mask],
        pixels=pixels,
        extent_multiplier=extent_multiplier,
        intrinsic_disk_axis_ratio=intrinsic_disk_axis_ratio,
    )


def _bilinear_at_major_axis(image: np.ndarray, x_grid: np.ndarray, radii: np.ndarray) -> np.ndarray:
    coordinates = x_grid[0]
    center = image.shape[0] // 2
    # y=0 is exactly represented because grids are required to have odd size.
    return np.interp(radii, coordinates, image[center], left=np.nan, right=np.nan)


def score_replica(seed: GalaxyReplicaSeed, rendered: RenderedGalaxy) -> dict[str, float]:
    """Compare continuous and finite-pixel products with the observed knots.

    The continuous scores test the coordinate/projection transformation at the
    published sampling locations.  Pixelized scores separately expose losses
    caused only by the chosen visualization grid.
    """
    radii = seed.light.radius_kpc
    disk_recovered = (
        _bilinear_at_major_axis(rendered.disk_lsun_pc2, rendered.x_kpc, radii)
        * rendered.apparent_axis_ratio
    )
    bulge_recovered = _bilinear_at_major_axis(rendered.bulge_lsun_pc2, rendered.x_kpc, radii)
    expected = seed.light.disk_lsun_pc2 + seed.light.bulge_lsun_pc2
    recovered = disk_recovered + bulge_recovered
    positive = (expected > 0.0) & (recovered > 0.0) & np.isfinite(recovered)
    pixelized_light_rmse_dex = float(
        np.sqrt(np.mean(np.square(np.log10(recovered[positive] / expected[positive]))))
    )
    continuous_recovered = (
        seed.light.interpolate(radii, "disk") + seed.light.interpolate(radii, "bulge")
    )
    continuous_positive = (expected > 0.0) & (continuous_recovered > 0.0)
    light_rmse_dex = float(
        np.sqrt(
            np.mean(
                np.square(
                    np.log10(
                        continuous_recovered[continuous_positive]
                        / expected[continuous_positive]
                    )
                )
            )
        )
    )

    angular = seed.angular_photometry
    angular_radius_kpc = angular.radius_arcsec * seed.distance_mpc * 1000.0 / 206265.0
    # SPARC uses M_sun(3.6 micron)=3.24 mag and the standard 21.572
    # arcsec-to-parsec surface-brightness conversion.
    angular_density = np.power(
        10.0,
        -0.4 * (angular.surface_brightness_mag_arcsec2 - 3.24 - 21.572),
    )
    angular_recovered = seed.light.interpolate(angular_radius_kpc, "total")
    angular_valid = (
        (angular.keep_flag > 0)
        & (angular_radius_kpc <= seed.light.radius_kpc[-1])
        & (angular_density > 0.0)
        & (angular_recovered > 0.0)
    )
    angular_photometry_rmse_dex = float(
        np.sqrt(
            np.mean(
                np.square(
                    np.log10(
                        angular_recovered[angular_valid] / angular_density[angular_valid]
                    )
                )
            )
        )
    )

    mask = valid_rotation_mask(seed)
    curve = seed.rotation
    velocity_los = _bilinear_at_major_axis(
        rendered.line_of_sight_velocity_km_s, rendered.x_kpc, curve.radius_kpc[mask]
    )
    sine = math.sin(math.radians(seed.inclination_deg))
    velocity_recovered = velocity_los / max(sine, 1.0e-12)
    finite = np.isfinite(velocity_recovered)
    pixelized_rotation_rmse = float(
        np.sqrt(
            np.mean(
                np.square(
                    velocity_recovered[finite] - curve.velocity_observed_kms[mask][finite]
                )
            )
        )
    )
    continuous_velocity = _rotation_at(seed, curve.radius_kpc[mask])
    rotation_rmse = float(
        np.sqrt(np.mean(np.square(continuous_velocity - curve.velocity_observed_kms[mask])))
    )

    coordinate = rendered.x_kpc[0]
    pixel_kpc = float(coordinate[1] - coordinate[0])
    rendered_lsun = float(rendered.total_lsun_pc2.sum() * (pixel_kpc * 1000.0) ** 2)
    light_fractional_error = rendered_lsun / max(seed.light.total_lsun, 1.0e-30) - 1.0
    return {
        "light_rmse_dex": light_rmse_dex,
        "rotation_rmse_km_s": rotation_rmse,
        "angular_photometry_rmse_dex": angular_photometry_rmse_dex,
        "pixelized_light_rmse_dex": pixelized_light_rmse_dex,
        "pixelized_rotation_rmse_km_s": pixelized_rotation_rmse,
        "rendered_luminosity_lsun": rendered_lsun,
        "input_luminosity_lsun": seed.light.total_lsun,
        "total_light_fractional_error": float(light_fractional_error),
        "bulge_fraction": seed.bulge_fraction,
        "apparent_axis_ratio": rendered.apparent_axis_ratio,
        "light_knots": int(positive.sum()),
        "rotation_knots": int(finite.sum()),
        "angular_photometry_knots": int(angular_valid.sum()),
    }


def _sample_component_radii(
    radius_kpc: np.ndarray, density: np.ndarray, count: int
) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.float64)
    dense_radius = np.linspace(0.0, float(radius_kpc[-1]), 16384)
    dense_density = np.interp(dense_radius, radius_kpc, density, left=density[0], right=0.0)
    integrand = np.maximum(dense_density, 0.0) * dense_radius
    increments = 0.5 * (integrand[1:] + integrand[:-1]) * np.diff(dense_radius)
    cdf = np.concatenate(([0.0], np.cumsum(increments)))
    if cdf[-1] <= 0.0:
        return np.empty(0, dtype=np.float64)
    cdf /= cdf[-1]
    quantiles = (np.arange(count, dtype=np.float64) + 0.5) / count
    return np.interp(quantiles, cdf, dense_radius)


def generate_replica_particles(
    seed: GalaxyReplicaSeed, *, particle_count: int = 65536
) -> ReplicaParticles:
    """Create a deterministic luminosity-tracer realization of the replica."""
    if particle_count < 1024:
        raise ValueError("particle_count must be at least 1024")
    disk_light = seed.light.disk_lsun
    bulge_light = seed.light.bulge_lsun
    total = disk_light + bulge_light
    disk_count = int(round(particle_count * disk_light / max(total, 1.0e-30)))
    if disk_light > 0.0:
        disk_count = max(1, disk_count)
    if bulge_light > 0.0:
        disk_count = min(particle_count - 1, disk_count)
    bulge_count = particle_count - disk_count
    golden = math.pi * (3.0 - math.sqrt(5.0))

    disk_radius = _sample_component_radii(
        seed.light.radius_kpc, seed.light.disk_lsun_pc2, disk_count
    )
    disk_angle = golden * np.arange(len(disk_radius))
    disk_position = np.column_stack(
        (disk_radius * np.cos(disk_angle), disk_radius * np.sin(disk_angle), np.zeros(len(disk_radius)))
    )
    disk_speed = _rotation_at(seed, disk_radius)
    disk_velocity = np.column_stack(
        (-disk_speed * np.sin(disk_angle), disk_speed * np.cos(disk_angle), np.zeros(len(disk_radius)))
    )

    # The bulge input is a projected profile.  Sample it in the projected plane
    # rather than claiming a unique 3-D deprojection that the data do not supply.
    bulge_radius = _sample_component_radii(
        seed.light.radius_kpc, seed.light.bulge_lsun_pc2, bulge_count
    )
    bulge_angle = golden * (np.arange(len(bulge_radius)) + len(disk_radius))
    bulge_position = np.column_stack(
        (bulge_radius * np.cos(bulge_angle), bulge_radius * np.sin(bulge_angle), np.zeros(len(bulge_radius)))
    )
    bulge_speed = _rotation_at(seed, bulge_radius)
    bulge_velocity = np.column_stack(
        (-bulge_speed * np.sin(bulge_angle), bulge_speed * np.cos(bulge_angle), np.zeros(len(bulge_radius)))
    )

    positions = np.vstack((disk_position, bulge_position))
    velocities = np.vstack((disk_velocity, bulge_velocity))
    luminosities = np.concatenate(
        (
            np.full(len(disk_radius), disk_light / max(len(disk_radius), 1)),
            np.full(len(bulge_radius), bulge_light / max(len(bulge_radius), 1)),
        )
    )
    components = np.concatenate(
        (np.full(len(disk_radius), "disk", dtype="U6"), np.full(len(bulge_radius), "bulge", dtype="U6"))
    )
    digest = hashlib.sha256()
    for array in (positions, velocities, luminosities):
        digest.update(np.asarray(array).tobytes())
    return ReplicaParticles(seed.name, positions, velocities, luminosities, components, digest.hexdigest())
