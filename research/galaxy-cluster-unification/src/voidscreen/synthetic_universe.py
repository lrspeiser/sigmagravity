"""Fast seeded baryonic scenes and interchangeable gravity forward models.

This module deliberately separates three things that are easy to mix up:

* a baryonic seed, which may use light, gas, distance, and morphology data;
* a gravity law, whose parameters are universal rather than object-specific; and
* a target observation, which must not be consulted while constructing a seed.

The particle scenes are useful for geometry, controlled counterfactuals, and
direct-force verification.  Real-system scoring uses the measured baryonic
force knots as a higher-fidelity radial representation of the same seed.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import qmc

G_SI = 6.67430e-11
KPC_M = 3.085677581491367e19
M_SUN_KG = 1.98847e30
PC_M = KPC_M / 1000.0
A0_M_S2 = 1.2e-10

TRANSPORT_PARAMETER_NAMES = (
    "log10_a0_m_s2",
    "diffuse_amplitude",
    "low_acceleration_power",
    "log10_surface_transition_msun_pc2",
    "surface_power",
    "log10_extent_transition_kpc",
    "return_amplitude",
)

# These are broad, predeclared physical-search limits, not posterior intervals.
TRANSPORT_PARAMETER_BOUNDS = (
    (-12.0, -9.0),
    (0.0, 20.0),
    (0.2, 4.0),
    (-1.0, 4.0),
    (0.2, 4.0),
    (0.0, 3.0),
    (0.0, 30.0),
)


def _as_positive_1d(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2:
        raise ValueError(f"{name} must be a one-dimensional array with at least two values")
    if np.any(~np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must be finite and strictly positive")
    return array


@dataclass(frozen=True)
class RadialBaryonProfile:
    """Baryonic radial force anchors with no dynamical or lensing target."""

    system: str
    radius_kpc: np.ndarray
    gbar_m_s2: np.ndarray

    def __post_init__(self) -> None:
        radius = _as_positive_1d(self.radius_kpc, "radius_kpc")
        gbar = _as_positive_1d(self.gbar_m_s2, "gbar_m_s2")
        if len(radius) != len(gbar):
            raise ValueError("radius and baryonic acceleration lengths differ")
        order = np.argsort(radius, kind="stable")
        radius = radius[order]
        gbar = gbar[order]
        if np.any(np.diff(radius) <= 0.0):
            raise ValueError("profile radii must be unique")
        object.__setattr__(self, "radius_kpc", radius)
        object.__setattr__(self, "gbar_m_s2", gbar)

    def interpolate(self, radius_kpc, *, outer_slope: float = -2.0) -> np.ndarray:
        """Log-log interpolation with finite power-law tails."""
        radius = np.asarray(radius_kpc, dtype=np.float64)
        if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
            raise ValueError("query radii must be finite and positive")
        log_r = np.log(radius)
        anchor_r = np.log(self.radius_kpc)
        anchor_g = np.log(self.gbar_m_s2)
        result = np.interp(log_r, anchor_r, anchor_g)
        inner = log_r < anchor_r[0]
        outer = log_r > anchor_r[-1]
        inner_slope = (anchor_g[1] - anchor_g[0]) / (anchor_r[1] - anchor_r[0])
        result = np.where(inner, anchor_g[0] + inner_slope * (log_r - anchor_r[0]), result)
        result = np.where(outer, anchor_g[-1] + outer_slope * (log_r - anchor_r[-1]), result)
        return np.exp(result)

    @property
    def enclosed_mass_msun(self) -> np.ndarray:
        radius_m = self.radius_kpc * KPC_M
        spherical_equivalent = self.gbar_m_s2 * radius_m**2 / (G_SI * M_SUN_KG)
        return np.maximum.accumulate(spherical_equivalent)

    def mass_fraction_radius(self, fraction: float) -> float:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("mass fraction must lie in (0, 1]")
        mass = self.enclosed_mass_msun
        target = fraction * mass[-1]
        return float(np.interp(target, mass, self.radius_kpc))

    @property
    def total_mass_msun(self) -> float:
        return float(self.enclosed_mass_msun[-1])

    @property
    def r80_kpc(self) -> float:
        return self.mass_fraction_radius(0.8)

    @property
    def r50_kpc(self) -> float:
        return self.mass_fraction_radius(0.5)

    @property
    def mean_surface_density_msun_pc2(self) -> float:
        radius_pc = max(self.r80_kpc * 1000.0, 1.0e-9)
        return self.total_mass_msun / (math.pi * radius_pc**2)

    @property
    def concentration_r50_over_r80(self) -> float:
        return self.r50_kpc / max(self.r80_kpc, 1.0e-12)

    def context(self, reference_radius_kpc: float = 200.0) -> dict[str, float]:
        return {
            "total_mass_msun": self.total_mass_msun,
            "r80_kpc": self.r80_kpc,
            "r50_kpc": self.r50_kpc,
            "concentration_r50_over_r80": self.concentration_r50_over_r80,
            "mean_surface_density_msun_pc2": self.mean_surface_density_msun_pc2,
            "reference_radius_kpc": float(reference_radius_kpc),
            "reference_gbar_m_s2": float(self.interpolate([reference_radius_kpc])[0]),
        }


@dataclass(frozen=True)
class GalaxySeed:
    """Observable baryonic parameters used to instantiate one galaxy."""

    name: str
    profile: RadialBaryonProfile
    disk_mass_msun: float
    bulge_mass_msun: float
    gas_mass_msun: float
    disk_scale_kpc: float
    bulge_scale_kpc: float
    gas_scale_kpc: float
    disk_height_ratio: float = 0.12
    gas_height_ratio: float = 0.04
    bar_strength: float = 0.0
    spiral_strength: float = 0.0
    clumpiness: float = 0.0
    inclination_deg: float = 60.0
    random_seed: int = 0

    def __post_init__(self) -> None:
        values = (
            self.disk_mass_msun,
            self.bulge_mass_msun,
            self.gas_mass_msun,
            self.disk_scale_kpc,
            self.bulge_scale_kpc,
            self.gas_scale_kpc,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in values[:3]):
            raise ValueError("component masses must be finite and nonnegative")
        if any(not np.isfinite(value) or value <= 0.0 for value in values[3:]):
            raise ValueError("component scales must be finite and positive")
        if sum(values[:3]) <= 0.0:
            raise ValueError("a galaxy seed needs positive baryonic mass")


@dataclass(frozen=True)
class ClusterSeed:
    """Baryon-only radial cluster seed plus measured shape controls."""

    name: str
    profile: RadialBaryonProfile
    axis_ratio_y: float = 0.85
    axis_ratio_z: float = 0.70
    member_fraction: float = 0.25
    gas_fraction: float = 0.65
    random_seed: int = 0

    def __post_init__(self) -> None:
        if not 0.1 <= self.axis_ratio_y <= 1.0 or not 0.1 <= self.axis_ratio_z <= 1.0:
            raise ValueError("cluster axis ratios must lie between 0.1 and 1")
        if self.member_fraction < 0.0 or self.gas_fraction < 0.0:
            raise ValueError("cluster fractions must be nonnegative")


@dataclass(frozen=True)
class ParticleScene:
    """One Monte Carlo realization of a baryonic seed."""

    system: str
    positions_kpc: np.ndarray
    masses_msun: np.ndarray
    components: np.ndarray
    seed_fingerprint: str

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions_kpc, dtype=np.float64)
        masses = np.asarray(self.masses_msun, dtype=np.float64)
        components = np.asarray(self.components)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("particle positions must have shape (n, 3)")
        if masses.shape != (len(positions),) or components.shape != (len(positions),):
            raise ValueError("particle arrays have inconsistent lengths")
        if np.any(~np.isfinite(positions)) or np.any(~np.isfinite(masses)):
            raise ValueError("particle scene contains non-finite values")
        if np.any(masses <= 0.0):
            raise ValueError("all particle masses must be positive")

    @property
    def total_mass_msun(self) -> float:
        return float(self.masses_msun.sum())


def stable_seed(label: str, salt: str = "P0630") -> int:
    digest = hashlib.sha256(f"{salt}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**32 - 1)


def _component_counts(masses: Sequence[float], n_particles: int) -> np.ndarray:
    if n_particles < len(masses):
        raise ValueError("particle count is smaller than component count")
    # Copy because callers commonly pass the actual component-mass array.
    # Normalizing a view here would silently replace physical masses by fractions.
    fractions = np.array(masses, dtype=np.float64, copy=True)
    fractions /= fractions.sum()
    raw = fractions * n_particles
    counts = np.floor(raw).astype(int)
    positive = fractions > 0.0
    counts[positive & (counts == 0)] = 1
    difference = n_particles - int(counts.sum())
    if difference > 0:
        order = np.argsort(-(raw - np.floor(raw)))
        for index in order[:difference]:
            counts[index] += 1
    elif difference < 0:
        order = np.argsort(raw - np.floor(raw))
        for index in order:
            if difference == 0:
                break
            if counts[index] > int(positive[index]):
                counts[index] -= 1
                difference += 1
    if counts.sum() != n_particles:
        raise RuntimeError("component allocation failed")
    return counts


def _disk_positions(
    rng: np.random.Generator,
    count: int,
    scale_kpc: float,
    height_kpc: float,
    spiral_strength: float,
) -> np.ndarray:
    radius = rng.gamma(shape=2.0, scale=scale_kpc, size=count)
    phi = rng.uniform(0.0, 2.0 * math.pi, count)
    strength = float(np.clip(spiral_strength, 0.0, 0.8))
    if strength > 0.0:
        # A deterministic displacement creates an m=2 overdensity without
        # changing the radial mass profile used by the force-anchor seed.
        phase = 2.0 * phi - 2.5 * np.log1p(radius / scale_kpc)
        phi = phi + 0.5 * strength * np.sin(phase)
    z = rng.laplace(0.0, max(height_kpc, 1.0e-5), count)
    return np.column_stack([radius * np.cos(phi), radius * np.sin(phi), z])


def _hernquist_positions(
    rng: np.random.Generator, count: int, scale_kpc: float, truncation_scale: float = 50.0
) -> np.ndarray:
    maximum_radius = truncation_scale * scale_kpc
    f_max = (maximum_radius / (maximum_radius + scale_kpc)) ** 2
    root_u = np.sqrt(rng.uniform(0.0, f_max, count))
    radius = scale_kpc * root_u / np.maximum(1.0 - root_u, 1.0e-12)
    cos_theta = rng.uniform(-1.0, 1.0, count)
    sin_theta = np.sqrt(np.maximum(1.0 - cos_theta**2, 0.0))
    phi = rng.uniform(0.0, 2.0 * math.pi, count)
    return np.column_stack(
        [
            radius * sin_theta * np.cos(phi),
            radius * sin_theta * np.sin(phi),
            radius * cos_theta,
        ]
    )


def generate_galaxy_scene(seed: GalaxySeed, n_particles: int = 4096) -> ParticleScene:
    rng = np.random.default_rng(seed.random_seed)
    masses = np.asarray(
        [seed.disk_mass_msun, seed.bulge_mass_msun, seed.gas_mass_msun], dtype=float
    )
    counts = _component_counts(masses, n_particles)
    positions, particle_masses, labels = [], [], []
    specifications = (
        (
            "disk",
            _disk_positions(
                rng,
                counts[0],
                seed.disk_scale_kpc,
                seed.disk_height_ratio * seed.disk_scale_kpc,
                seed.spiral_strength,
            ),
        ),
        ("bulge", _hernquist_positions(rng, counts[1], seed.bulge_scale_kpc)),
        (
            "gas",
            _disk_positions(
                rng,
                counts[2],
                seed.gas_scale_kpc,
                seed.gas_height_ratio * seed.gas_scale_kpc,
                0.5 * seed.spiral_strength,
            ),
        ),
    )
    for index, (label, component_positions) in enumerate(specifications):
        count = counts[index]
        if count == 0:
            continue
        if label == "disk" and seed.bar_strength > 0.0:
            component_positions = component_positions.copy()
            bar = float(np.clip(seed.bar_strength, 0.0, 0.8))
            central = np.hypot(component_positions[:, 0], component_positions[:, 1]) < (
                2.0 * seed.disk_scale_kpc
            )
            component_positions[central, 0] *= 1.0 + bar
            component_positions[central, 1] *= 1.0 - 0.5 * bar
        positions.append(component_positions)
        particle_masses.append(np.full(count, masses[index] / count))
        labels.append(np.full(count, label, dtype="U8"))
    fingerprint = hashlib.sha256(repr(seed).encode("utf-8")).hexdigest()
    return ParticleScene(
        system=seed.name,
        positions_kpc=np.concatenate(positions),
        masses_msun=np.concatenate(particle_masses),
        components=np.concatenate(labels),
        seed_fingerprint=fingerprint,
    )


def generate_cluster_scene(seed: ClusterSeed, n_particles: int = 8192) -> ParticleScene:
    rng = np.random.default_rng(seed.random_seed)
    profile_mass = seed.profile.enclosed_mass_msun
    cumulative = profile_mass / profile_mass[-1]
    draw = rng.uniform(0.0, 1.0, n_particles)
    radius = np.interp(draw, cumulative, seed.profile.radius_kpc)
    cos_theta = rng.uniform(-1.0, 1.0, n_particles)
    sin_theta = np.sqrt(np.maximum(1.0 - cos_theta**2, 0.0))
    phi = rng.uniform(0.0, 2.0 * math.pi, n_particles)
    positions = np.column_stack(
        [
            radius * sin_theta * np.cos(phi),
            seed.axis_ratio_y * radius * sin_theta * np.sin(phi),
            seed.axis_ratio_z * radius * cos_theta,
        ]
    )
    labels = np.full(n_particles, "diffuse", dtype="U8")
    member_count = int(round(seed.member_fraction * n_particles))
    gas_count = int(round(seed.gas_fraction * n_particles))
    labels[: min(member_count, n_particles)] = "member"
    labels[member_count : min(member_count + gas_count, n_particles)] = "gas"
    fingerprint = hashlib.sha256(repr(seed).encode("utf-8")).hexdigest()
    return ParticleScene(
        system=seed.name,
        positions_kpc=positions,
        masses_msun=np.full(n_particles, seed.profile.total_mass_msun / n_particles),
        components=labels,
        seed_fingerprint=fingerprint,
    )


def direct_acceleration_m_s2(
    scene: ParticleScene,
    query_positions_kpc,
    *,
    softening_kpc: float = 0.05,
    query_batch: int = 64,
) -> np.ndarray:
    """Direct vector sum used for low-resolution scene verification."""
    query = np.asarray(query_positions_kpc, dtype=np.float64)
    if query.ndim != 2 or query.shape[1] != 3:
        raise ValueError("queries must have shape (n, 3)")
    if softening_kpc <= 0.0:
        raise ValueError("softening must be positive")
    source = scene.positions_kpc
    mass = scene.masses_msun * M_SUN_KG
    result = np.empty_like(query)
    epsilon2 = (softening_kpc * KPC_M) ** 2
    for start in range(0, len(query), query_batch):
        stop = min(start + query_batch, len(query))
        delta_m = (source[None, :, :] - query[start:stop, None, :]) * KPC_M
        distance2 = np.einsum("qsi,qsi->qs", delta_m, delta_m) + epsilon2
        inverse_cube = np.power(distance2, -1.5)
        result[start:stop] = G_SI * np.einsum("qs,qsi,s->qi", inverse_cube, delta_m, mass)
    return result


def radial_particle_acceleration_m_s2(
    scene: ParticleScene, radius_kpc, *, softening_kpc: float = 0.05
) -> np.ndarray:
    radius = _as_positive_1d(radius_kpc, "radius_kpc")
    query = np.column_stack([radius, np.zeros_like(radius), np.zeros_like(radius)])
    vectors = direct_acceleration_m_s2(scene, query, softening_kpc=softening_kpc)
    return np.maximum(-vectors[:, 0], 0.0)


def fixed_rar_acceleration(gbar_m_s2, a0_m_s2: float = A0_M_S2) -> np.ndarray:
    gbar = np.asarray(gbar_m_s2, dtype=np.float64)
    return gbar / np.maximum(1.0 - np.exp(-np.sqrt(gbar / a0_m_s2)), 1.0e-15)


def simple_mond_acceleration(gbar_m_s2, a0_m_s2: float = A0_M_S2) -> np.ndarray:
    gbar = np.asarray(gbar_m_s2, dtype=np.float64)
    return 0.5 * (gbar + np.sqrt(gbar**2 + 4.0 * a0_m_s2 * gbar))


def transport_acceleration_from_features(
    gbar_m_s2,
    radius_kpc,
    *,
    surface_density_msun_pc2,
    r80_kpc,
    reference_gbar_m_s2,
    parameters: Sequence[float],
    reference_radius_kpc: float = 200.0,
) -> np.ndarray:
    """Continuous diffuse-survival plus extent-return phenomenology.

    The first term increases the fraction of the baryonic field that survives
    in low-acceleration, low-surface-density environments.  The second returns
    a mass-normalized 1/r tail only when the baryonic system is extended.  Every
    coefficient is universal; no value is selected per galaxy or cluster.
    """
    vector = np.asarray(parameters, dtype=np.float64)
    if vector.shape != (len(TRANSPORT_PARAMETER_NAMES),):
        raise ValueError("transport law needs exactly seven universal parameters")
    gbar = np.asarray(gbar_m_s2, dtype=np.float64)
    radius = np.asarray(radius_kpc, dtype=np.float64)
    surface = np.asarray(surface_density_msun_pc2, dtype=np.float64)
    extent = np.asarray(r80_kpc, dtype=np.float64)
    reference = np.asarray(reference_gbar_m_s2, dtype=np.float64)
    gbar, radius, surface, extent, reference = np.broadcast_arrays(
        gbar, radius, surface, extent, reference
    )
    if (
        np.any(~np.isfinite(gbar))
        or np.any(gbar <= 0.0)
        or np.any(radius <= 0.0)
        or np.any(surface <= 0.0)
        or np.any(extent <= 0.0)
        or np.any(reference <= 0.0)
    ):
        raise ValueError("transport inputs must be finite and positive")

    log_a0, amplitude, low_power, log_surface, surface_power, log_extent, return_amp = vector
    a0 = 10.0**log_a0
    surface_transition = 10.0**log_surface
    extent_transition = 10.0**log_extent
    low_gate = np.power(a0 / (a0 + gbar), low_power)
    porous_survival = 1.0 / (
        1.0 + np.power(surface / surface_transition, surface_power)
    )
    # A nonzero floor avoids claiming that high-density matter erases gravity;
    # the fitted amplitude measures an additional transmissive contribution.
    local_extra = gbar * amplitude * low_gate * (0.2 + 0.8 * porous_survival)
    extent_gate = np.square(extent / extent_transition)
    extent_gate = extent_gate / (1.0 + extent_gate)
    return_extra = (
        return_amp
        * extent_gate
        * reference
        * reference_radius_kpc
        / radius
        * low_gate
    )
    predicted = gbar + local_extra + return_extra
    if np.any(~np.isfinite(predicted)) or np.any(predicted <= 0.0):
        raise ValueError("transport law generated invalid acceleration")
    return predicted


def predict_acceleration(
    law: str,
    profile: RadialBaryonProfile,
    radius_kpc=None,
    *,
    parameters: Sequence[float] | None = None,
) -> np.ndarray:
    radius = profile.radius_kpc if radius_kpc is None else np.asarray(radius_kpc, dtype=float)
    gbar = profile.interpolate(radius)
    if law == "baryons":
        return gbar
    if law == "rar":
        return fixed_rar_acceleration(gbar)
    if law == "simple_mond":
        return simple_mond_acceleration(gbar)
    if law != "transport":
        raise ValueError(f"unknown gravity law {law}")
    if parameters is None:
        raise ValueError("transport predictions require universal parameters")
    context = profile.context()
    return transport_acceleration_from_features(
        gbar,
        radius,
        surface_density_msun_pc2=context["mean_surface_density_msun_pc2"],
        r80_kpc=context["r80_kpc"],
        reference_gbar_m_s2=context["reference_gbar_m_s2"],
        parameters=parameters,
    )


def rotation_velocity_km_s(acceleration_m_s2, radius_kpc) -> np.ndarray:
    acceleration = np.asarray(acceleration_m_s2, dtype=np.float64)
    radius = np.asarray(radius_kpc, dtype=np.float64)
    return np.sqrt(np.maximum(acceleration * radius * KPC_M, 0.0)) / 1000.0


def sobol_galaxy_population(size: int, seed: int = 6302026) -> dict[str, np.ndarray]:
    """Generate a structured, cheap population for million-scale response tests."""
    if size <= 0:
        raise ValueError("population size must be positive")
    sampler = qmc.Sobol(d=7, scramble=True, seed=seed)
    # Generate the next power of two and slice.  This preserves Sobol balance
    # properties and avoids silently falling back to a poorly balanced prefix.
    exponent = int(math.ceil(math.log2(size))) if size > 1 else 0
    unit = sampler.random_base2(exponent)[:size]
    log_mass = 7.0 + 5.5 * unit[:, 0]
    gas_fraction = 0.02 + 0.88 * unit[:, 1]
    bulge_fraction = 0.80 * unit[:, 2] ** 2
    log_rd = -0.4 + 0.32 * (log_mass - 8.0) + 0.8 * (unit[:, 3] - 0.5)
    disk_scale = np.power(10.0, log_rd)
    radius = disk_scale * (2.0 + 5.0 * unit[:, 4])
    concentration = 0.25 + 0.65 * unit[:, 5]
    clumpiness = unit[:, 6]
    mass = np.power(10.0, log_mass)
    r80 = disk_scale * (2.4 + 1.6 * gas_fraction + 0.6 * clumpiness)
    surface = mass / (math.pi * np.square(r80 * 1000.0))
    disk_fraction = np.maximum(1.0 - gas_fraction - bulge_fraction, 0.02)
    component_total = disk_fraction + gas_fraction + bulge_fraction
    disk_fraction /= component_total
    gas_fraction /= component_total
    bulge_fraction /= component_total
    disk_enclosed = 1.0 - np.exp(-radius / disk_scale) * (1.0 + radius / disk_scale)
    gas_scale = disk_scale * (1.5 + 1.5 * gas_fraction)
    gas_enclosed = 1.0 - np.exp(-radius / gas_scale) * (1.0 + radius / gas_scale)
    bulge_scale = disk_scale * (0.08 + 0.35 * concentration)
    bulge_enclosed = np.square(radius / (radius + bulge_scale))
    enclosed = mass * (
        disk_fraction * disk_enclosed
        + gas_fraction * gas_enclosed
        + bulge_fraction * bulge_enclosed
    )
    gbar = G_SI * enclosed * M_SUN_KG / np.square(radius * KPC_M)
    reference_radius = np.full(size, 200.0)
    reference_enclosed = mass * (
        disk_fraction
        + gas_fraction
        + bulge_fraction
    )
    reference_gbar = G_SI * reference_enclosed * M_SUN_KG / np.square(
        reference_radius * KPC_M
    )
    return {
        "log10_mass_msun": log_mass,
        "gas_fraction": gas_fraction,
        "bulge_fraction": bulge_fraction,
        "disk_scale_kpc": disk_scale,
        "radius_kpc": radius,
        "r80_kpc": r80,
        "surface_density_msun_pc2": surface,
        "concentration": concentration,
        "clumpiness": clumpiness,
        "gbar_m_s2": gbar,
        "reference_gbar_m_s2": reference_gbar,
    }


def stable_hash_partition(
    labels: Sequence[str],
    *,
    salt: str,
    train_fraction: float,
    development_fraction: float,
) -> dict[str, str]:
    """Target-independent whole-system split with stable cross-platform hashes."""
    if train_fraction <= 0.0 or development_fraction < 0.0:
        raise ValueError("invalid split fractions")
    if train_fraction + development_fraction >= 1.0:
        raise ValueError("split must retain a nonzero holdout fraction")
    partitions = {}
    for label in labels:
        digest = hashlib.sha256(f"{salt}:{label}".encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], "big") / float(2**64)
        if value < train_fraction:
            split = "train"
        elif value < train_fraction + development_fraction:
            split = "development"
        else:
            split = "holdout"
        partitions[str(label)] = split
    return partitions


def parameter_mapping(vector: Sequence[float]) -> dict[str, float]:
    values = np.asarray(vector, dtype=float)
    if values.shape != (len(TRANSPORT_PARAMETER_NAMES),):
        raise ValueError("wrong transport parameter count")
    return dict(zip(TRANSPORT_PARAMETER_NAMES, values.tolist(), strict=True))


def parameter_vector(mapping: Mapping[str, float]) -> np.ndarray:
    return np.asarray([mapping[name] for name in TRANSPORT_PARAMETER_NAMES], dtype=float)
