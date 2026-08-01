"""Mass-conserving angular perturbations for thin-lens diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .raw_lensing import C_M_S, RAD_TO_ARCSEC, RadialDeflectionField

G_SI = 6.67430e-11
MSUN_KG = 1.98847e30
MPC_M = 3.085677581491367e22


@dataclass(frozen=True)
class MemberRedistributionField:
    """A discrete light-traced field minus its exact circular average.

    The individual components are softened point lenses.  Their physical
    deflection is averaged on circles around the declared cluster center and
    subtracted from the discrete field.  The returned contrast consequently
    has zero azimuthally averaged radial deflection on the interpolation grid
    and zero net far-field mass.
    """

    member_x_arcsec: np.ndarray
    member_y_arcsec: np.ndarray
    member_mass_msun: np.ndarray
    lens_angular_diameter_distance_mpc: float
    softening_arcsec: float
    circular_mean: RadialDeflectionField

    @classmethod
    def build(
        cls,
        member_x_arcsec,
        member_y_arcsec,
        normalized_weights,
        *,
        total_mass_msun: float,
        lens_angular_diameter_distance_mpc: float,
        softening_arcsec: float,
        impact_arcsec,
        azimuth_samples: int,
    ) -> MemberRedistributionField:
        x = np.asarray(member_x_arcsec, dtype=float)
        y = np.asarray(member_y_arcsec, dtype=float)
        weights = np.asarray(normalized_weights, dtype=float)
        impact = np.asarray(impact_arcsec, dtype=float)
        if x.ndim != 1 or y.shape != x.shape or weights.shape != x.shape:
            raise ValueError("member coordinates and weights must be matching vectors")
        if len(x) == 0 or np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
            raise ValueError("at least one finite member coordinate is required")
        if np.any(weights < 0.0) or not np.isclose(weights.sum(), 1.0):
            raise ValueError("member weights must be nonnegative and normalized")
        if total_mass_msun <= 0.0 or lens_angular_diameter_distance_mpc <= 0.0:
            raise ValueError("mass and lens distance must be positive")
        if softening_arcsec <= 0.0 or azimuth_samples < 64:
            raise ValueError("positive softening and at least 64 azimuth samples required")
        if np.any(impact <= 0.0) or np.any(np.diff(impact) <= 0.0):
            raise ValueError("impact grid must be positive and increasing")

        provisional = cls(
            x,
            y,
            weights * float(total_mass_msun),
            float(lens_angular_diameter_distance_mpc),
            float(softening_arcsec),
            RadialDeflectionField(impact, np.zeros_like(impact)),
        )
        phi = np.linspace(0.0, 2.0 * np.pi, int(azimuth_samples), endpoint=False)
        cosine = np.cos(phi)
        sine = np.sin(phi)
        radial_mean = np.empty_like(impact)
        # Chunking bounds the temporary (radius, azimuth, member) arrays.
        for start in range(0, len(impact), 32):
            stop = min(start + 32, len(impact))
            radius = impact[start:stop, None]
            sample_x = radius * cosine[None, :]
            sample_y = radius * sine[None, :]
            alpha_x, alpha_y = provisional.discrete_alpha_arcsec(
                sample_x, sample_y, distance_ratio=1.0
            )
            radial_mean[start:stop] = np.mean(
                alpha_x * cosine[None, :] + alpha_y * sine[None, :], axis=1
            )
        # Positive convergence guarantees a nonnegative circular deflection;
        # clipping only removes floating-point noise below zero.
        radial_mean = np.maximum(radial_mean, 0.0)
        circular = RadialDeflectionField(impact, radial_mean / RAD_TO_ARCSEC)
        return cls(
            provisional.member_x_arcsec,
            provisional.member_y_arcsec,
            provisional.member_mass_msun,
            provisional.lens_angular_diameter_distance_mpc,
            provisional.softening_arcsec,
            circular,
        )

    @property
    def total_mass_msun(self) -> float:
        return float(np.sum(self.member_mass_msun))

    @property
    def physical_einstein_area_arcsec2(self) -> np.ndarray:
        distance_m = self.lens_angular_diameter_distance_mpc * MPC_M
        return (
            4.0
            * G_SI
            * self.member_mass_msun
            * MSUN_KG
            / (C_M_S**2 * distance_m)
            * RAD_TO_ARCSEC**2
        )

    def discrete_alpha_arcsec(
        self, x_arcsec, y_arcsec, *, distance_ratio: float
    ) -> tuple[np.ndarray, np.ndarray]:
        if not np.isfinite(distance_ratio) or distance_ratio <= 0.0:
            raise ValueError("distance_ratio must be finite and positive")
        x = np.asarray(x_arcsec, dtype=float)
        y = np.asarray(y_arcsec, dtype=float)
        x, y = np.broadcast_arrays(x, y)
        dx = x[..., None] - self.member_x_arcsec
        dy = y[..., None] - self.member_y_arcsec
        denominator = dx**2 + dy**2 + self.softening_arcsec**2
        coefficient = self.physical_einstein_area_arcsec2 * float(distance_ratio)
        return (
            np.sum(coefficient * dx / denominator, axis=-1),
            np.sum(coefficient * dy / denominator, axis=-1),
        )

    def contrast_alpha_arcsec(
        self, x_arcsec, y_arcsec, *, distance_ratio: float
    ) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x_arcsec, dtype=float)
        y = np.asarray(y_arcsec, dtype=float)
        x, y = np.broadcast_arrays(x, y)
        discrete_x, discrete_y = self.discrete_alpha_arcsec(
            x, y, distance_ratio=distance_ratio
        )
        radius = np.hypot(x, y)
        safe = np.maximum(radius, 1.0e-12)
        circular = self.circular_mean.reduced_alpha_arcsec(
            np.maximum(radius, self.circular_mean.impact_arcsec[0]), distance_ratio
        )
        return discrete_x - circular * x / safe, discrete_y - circular * y / safe


@dataclass(frozen=True)
class RadialEnhancementField:
    """Log-radius interpolation of a positive locked enhancement curve."""

    impact_arcsec: np.ndarray
    enhancement: np.ndarray

    def __post_init__(self) -> None:
        impact = np.asarray(self.impact_arcsec, dtype=float)
        enhancement = np.asarray(self.enhancement, dtype=float)
        if impact.ndim != 1 or enhancement.shape != impact.shape:
            raise ValueError("impact and enhancement must be matching vectors")
        if np.any(impact <= 0.0) or np.any(np.diff(impact) <= 0.0):
            raise ValueError("impact must be positive and increasing")
        if np.any(~np.isfinite(enhancement)) or np.any(enhancement <= 0.0):
            raise ValueError("enhancement must be finite and positive")

    def __call__(self, impact_arcsec) -> np.ndarray:
        radius = np.asarray(impact_arcsec, dtype=float)
        clipped = np.clip(radius, self.impact_arcsec[0], self.impact_arcsec[-1])
        return np.exp(
            np.interp(
                np.log(clipped),
                np.log(self.impact_arcsec),
                np.log(self.enhancement),
            )
        )
