from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .data import KPC_M, PackedDataset

C_M_S = 299_792_458.0


@dataclass(frozen=True)
class TensorDataset:
    galaxy_names: tuple[str, ...]
    galaxy_index: torch.Tensor
    radius_kpc: torch.Tensor
    velocity_observed_kms: torch.Tensor
    velocity_error_kms: torch.Tensor
    velocity_gas_kms: torch.Tensor
    velocity_disk_unit_ml_kms: torch.Tensor
    velocity_bulge_unit_ml_kms: torch.Tensor
    train_mask: torch.Tensor
    distance_fractional_error: torch.Tensor
    inclination_deg: torch.Tensor
    inclination_error_deg: torch.Tensor
    environment_standardized: torch.Tensor

    @classmethod
    def from_packed(
        cls,
        data: PackedDataset,
        device: torch.device,
        dtype: torch.dtype = torch.float64,
    ) -> TensorDataset:
        floating = lambda values: torch.as_tensor(values, dtype=dtype, device=device)
        return cls(
            galaxy_names=data.galaxy_names,
            galaxy_index=torch.as_tensor(data.galaxy_index, dtype=torch.long, device=device),
            radius_kpc=floating(data.radius_kpc),
            velocity_observed_kms=floating(data.velocity_observed_kms),
            velocity_error_kms=floating(data.velocity_error_kms),
            velocity_gas_kms=floating(data.velocity_gas_kms),
            velocity_disk_unit_ml_kms=floating(data.velocity_disk_unit_ml_kms),
            velocity_bulge_unit_ml_kms=floating(data.velocity_bulge_unit_ml_kms),
            train_mask=torch.as_tensor(data.train_mask, dtype=torch.bool, device=device),
            distance_fractional_error=floating(data.distance_fractional_error),
            inclination_deg=floating(data.inclination_deg),
            inclination_error_deg=floating(data.inclination_error_deg),
            environment_standardized=floating(data.environment_standardized),
        )


@dataclass(frozen=True)
class Prediction:
    velocity_predicted_kms: torch.Tensor
    velocity_observed_adjusted_kms: torch.Tensor
    velocity_error_adjusted_kms: torch.Tensor
    velocity_baryonic_kms: torch.Tensor
    baryonic_acceleration_m_s2: torch.Tensor
    predicted_acceleration_m_s2: torch.Tensor
    radius_adjusted_kpc: torch.Tensor
    disk_mass_to_light: torch.Tensor
    bulge_mass_to_light: torch.Tensor
    distance_scale: torch.Tensor
    inclination_adjusted_deg: torch.Tensor


class RotationModel(nn.Module):
    model_name = "base"

    def __init__(
        self,
        n_galaxies: int,
        *,
        disk_ml_prior: float = 0.5,
        bulge_ml_prior: float = 0.7,
        log_ml_prior_sigma: float = 0.25,
    ) -> None:
        super().__init__()
        self.n_galaxies = n_galaxies
        self.disk_ml_prior = float(disk_ml_prior)
        self.bulge_ml_prior = float(bulge_ml_prior)
        self.log_ml_prior_sigma = float(log_ml_prior_sigma)
        self.disk_log_shift = nn.Parameter(torch.zeros(n_galaxies, dtype=torch.float64))
        self.bulge_log_shift = nn.Parameter(torch.zeros(n_galaxies, dtype=torch.float64))
        self.distance_z = nn.Parameter(torch.zeros(n_galaxies, dtype=torch.float64))
        self.inclination_z = nn.Parameter(torch.zeros(n_galaxies, dtype=torch.float64))

    def _effective_inputs(self, data: TensorDataset) -> dict[str, torch.Tensor]:
        idx = data.galaxy_index
        disk_ml_by_galaxy = self.disk_ml_prior * torch.exp(self.disk_log_shift)
        bulge_ml_by_galaxy = self.bulge_ml_prior * torch.exp(self.bulge_log_shift)
        distance_log_scale = torch.clamp(
            self.distance_z * data.distance_fractional_error, min=-1.5, max=1.5
        )
        distance_scale_by_galaxy = torch.exp(distance_log_scale)
        inclination_by_galaxy = torch.clamp(
            data.inclination_deg + self.inclination_z * data.inclination_error_deg,
            min=10.0,
            max=89.5,
        )

        disk_ml = disk_ml_by_galaxy[idx]
        bulge_ml = bulge_ml_by_galaxy[idx]
        distance_scale = distance_scale_by_galaxy[idx]
        inclination_adjusted = inclination_by_galaxy[idx]
        inclination_factor = torch.sin(torch.deg2rad(data.inclination_deg[idx])) / torch.sin(
            torch.deg2rad(inclination_adjusted)
        )

        gas_v2 = torch.sign(data.velocity_gas_kms) * data.velocity_gas_kms.square()
        baryonic_v2 = distance_scale * (
            gas_v2
            + disk_ml * data.velocity_disk_unit_ml_kms.square()
            + bulge_ml * data.velocity_bulge_unit_ml_kms.square()
        )
        baryonic_v2 = torch.clamp(baryonic_v2, min=1e-8)
        baryonic_velocity = torch.sqrt(baryonic_v2)
        radius_adjusted_kpc = data.radius_kpc * distance_scale
        radius_m = radius_adjusted_kpc * KPC_M
        baryonic_acceleration = baryonic_v2 * 1e6 / radius_m
        return {
            "disk_ml": disk_ml,
            "bulge_ml": bulge_ml,
            "distance_scale": distance_scale,
            "inclination_adjusted": inclination_adjusted,
            "baryonic_velocity": baryonic_velocity,
            "baryonic_acceleration": baryonic_acceleration,
            "radius_adjusted_kpc": radius_adjusted_kpc,
            "radius_m": radius_m,
            "observed_adjusted": data.velocity_observed_kms * inclination_factor,
            "error_adjusted": data.velocity_error_kms * inclination_factor,
        }

    def predict_acceleration(
        self, data: TensorDataset, inputs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, data: TensorDataset) -> Prediction:
        inputs = self._effective_inputs(data)
        acceleration = self.predict_acceleration(data, inputs)
        acceleration = torch.clamp(acceleration, min=1e-30)
        velocity = torch.sqrt(acceleration * inputs["radius_m"]) / 1000.0
        return Prediction(
            velocity_predicted_kms=velocity,
            velocity_observed_adjusted_kms=inputs["observed_adjusted"],
            velocity_error_adjusted_kms=inputs["error_adjusted"],
            velocity_baryonic_kms=inputs["baryonic_velocity"],
            baryonic_acceleration_m_s2=inputs["baryonic_acceleration"],
            predicted_acceleration_m_s2=acceleration,
            radius_adjusted_kpc=inputs["radius_adjusted_kpc"],
            disk_mass_to_light=inputs["disk_ml"],
            bulge_mass_to_light=inputs["bulge_ml"],
            distance_scale=inputs["distance_scale"],
            inclination_adjusted_deg=inputs["inclination_adjusted"],
        )

    def prior_penalty(self) -> torch.Tensor:
        ml = 0.5 * (
            (self.disk_log_shift / self.log_ml_prior_sigma).square().sum()
            + (self.bulge_log_shift / self.log_ml_prior_sigma).square().sum()
        )
        geometry = 0.5 * (self.distance_z.square().sum() + self.inclination_z.square().sum())
        return ml + geometry

    def physical_parameters(self) -> dict[str, float | int | bool]:
        return {}

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


class NewtonianModel(RotationModel):
    model_name = "newtonian"

    def predict_acceleration(
        self, data: TensorDataset, inputs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        del data
        return inputs["baryonic_acceleration"]


class RARModel(RotationModel):
    model_name = "rar"

    def __init__(self, *args, rar_acceleration_m_s2: float = 1.2e-10, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rar_acceleration_m_s2 = float(rar_acceleration_m_s2)

    def predict_acceleration(
        self, data: TensorDataset, inputs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        del data
        g_bar = torch.clamp(inputs["baryonic_acceleration"], min=1e-30)
        root = torch.sqrt(g_bar / self.rar_acceleration_m_s2)
        return g_bar / torch.clamp(1.0 - torch.exp(-root), min=1e-12)

    def physical_parameters(self) -> dict[str, float]:
        return {"rar_acceleration_m_s2": self.rar_acceleration_m_s2}


def _logit(value: float) -> float:
    return math.log(value / (1.0 - value))


def radial_potential_depth(
    baryonic_acceleration_m_s2: torch.Tensor,
    radius_m: torch.Tensor,
    galaxy_index: torch.Tensor,
    n_galaxies: int,
) -> torch.Tensor:
    """Integrate |Phi_bar| from each radius to infinity with a Keplerian tail.

    Points must be grouped by galaxy and strictly increasing in radius within
    each galaxy, as guaranteed by ``pack_dataset``. The vectorized trapezoidal
    integral is differentiable with respect to the baryonic acceleration and
    radius, including nuisance-parameter shifts.
    """
    if baryonic_acceleration_m_s2.numel() == 0:
        return baryonic_acceleration_m_s2
    same_galaxy = galaxy_index[:-1] == galaxy_index[1:]
    interval = (
        0.5
        * (baryonic_acceleration_m_s2[:-1] + baryonic_acceleration_m_s2[1:])
        * (radius_m[1:] - radius_m[:-1])
    )
    segments = torch.cat(
        (
            torch.where(same_galaxy, interval, torch.zeros_like(interval)),
            torch.zeros_like(radius_m[-1:]),
        )
    )
    group_integrals = torch.zeros(
        n_galaxies, dtype=segments.dtype, device=segments.device
    ).scatter_add(0, galaxy_index, segments)
    global_before = torch.cumsum(segments, dim=0) - segments
    preceding_groups = torch.cumsum(group_integrals, dim=0) - group_integrals
    inward_integral = global_before - preceding_groups[galaxy_index]
    outward_integral = group_integrals[galaxy_index] - inward_integral

    is_last = torch.cat(
        (
            galaxy_index[:-1] != galaxy_index[1:],
            torch.ones_like(galaxy_index[-1:], dtype=torch.bool),
        )
    )
    tail_by_galaxy = torch.zeros(
        n_galaxies, dtype=radius_m.dtype, device=radius_m.device
    ).scatter_add(
        0,
        galaxy_index[is_last],
        baryonic_acceleration_m_s2[is_last] * radius_m[is_last],
    )
    depth = outward_integral + tail_by_galaxy[galaxy_index]
    return torch.clamp(depth, min=1e-30)


class VoidScreeningModel(RotationModel):
    model_name = "void"

    def __init__(
        self,
        *args,
        fixed_flat_power: bool = False,
        environment_enabled: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.fixed_flat_power = bool(fixed_flat_power)
        self.environment_enabled = bool(environment_enabled)

        self.transition_raw = nn.Parameter(torch.tensor(_logit(0.60), dtype=torch.float64))
        self.amplitude_raw = nn.Parameter(torch.tensor(_logit(4.0 / 6.0), dtype=torch.float64))
        self.width_raw = nn.Parameter(
            torch.tensor(_logit((0.35 - 0.03) / 1.47), dtype=torch.float64)
        )
        if not self.fixed_flat_power:
            p_fraction = (0.5 - 0.05) / 1.45
            self.power_raw = nn.Parameter(torch.tensor(_logit(p_fraction), dtype=torch.float64))
        else:
            self.register_buffer("power_fixed", torch.tensor(0.5, dtype=torch.float64))
        if self.environment_enabled:
            self.environment_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        else:
            self.register_buffer("environment_fixed", torch.tensor(0.0, dtype=torch.float64))

    @property
    def transition_acceleration(self) -> torch.Tensor:
        log10_value = -13.0 + 5.0 * torch.sigmoid(self.transition_raw)
        return torch.pow(
            torch.tensor(10.0, device=log10_value.device, dtype=log10_value.dtype), log10_value
        )

    @property
    def amplitude(self) -> torch.Tensor:
        log10_value = -4.0 + 6.0 * torch.sigmoid(self.amplitude_raw)
        return torch.pow(
            torch.tensor(10.0, device=log10_value.device, dtype=log10_value.dtype), log10_value
        )

    @property
    def transition_width_dex(self) -> torch.Tensor:
        return 0.03 + 1.47 * torch.sigmoid(self.width_raw)

    @property
    def power(self) -> torch.Tensor:
        if self.fixed_flat_power:
            return self.power_fixed
        return 0.05 + 1.45 * torch.sigmoid(self.power_raw)

    @property
    def environment_beta(self) -> torch.Tensor:
        if self.environment_enabled:
            return 5.0 * torch.tanh(self.environment_raw)
        return self.environment_fixed

    def predict_acceleration(
        self, data: TensorDataset, inputs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        g_bar = torch.clamp(inputs["baryonic_acceleration"], min=1e-30)
        transition = self.transition_acceleration
        log_ratio = torch.log10(g_bar) - torch.log10(transition)
        activation = torch.sigmoid(-log_ratio / self.transition_width_dex)
        environment = data.environment_standardized[data.galaxy_index]
        amplitude = self.amplitude * torch.exp(self.environment_beta * environment)
        additional = amplitude * transition * torch.pow(g_bar / transition, self.power) * activation
        return g_bar + additional

    def prior_penalty(self) -> torch.Tensor:
        penalty = super().prior_penalty()
        if self.environment_enabled:
            penalty = penalty + 0.5 * self.environment_beta.square()
        return penalty

    def physical_parameters(self) -> dict[str, float | bool]:
        return {
            "transition_acceleration_m_s2": float(self.transition_acceleration.detach().cpu()),
            "amplitude_A0": float(self.amplitude.detach().cpu()),
            "transition_width_dex": float(self.transition_width_dex.detach().cpu()),
            "power_p": float(self.power.detach().cpu()),
            "environment_beta": float(self.environment_beta.detach().cpu()),
            "fixed_flat_power": self.fixed_flat_power,
            "environment_enabled": self.environment_enabled,
        }


class PotentialScreeningModel(RotationModel):
    """Phenomenological outer force activated by baryonic potential depth."""

    model_name = "potential"

    def __init__(
        self,
        *args,
        acceleration_reference_m_s2: float = 1.2e-10,
        environment_enabled: bool = False,
        boundary_layer_enabled: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.acceleration_reference_m_s2 = float(acceleration_reference_m_s2)
        self.environment_enabled = bool(environment_enabled)
        self.boundary_layer_enabled = bool(boundary_layer_enabled)

        # Registry bounds: chi_t in [1e-10, 1e-5], initialized at 1e-7.
        self.transition_raw = nn.Parameter(torch.tensor(_logit(0.60), dtype=torch.float64))
        self.amplitude_raw = nn.Parameter(torch.tensor(_logit(4.0 / 6.0), dtype=torch.float64))
        self.width_raw = nn.Parameter(
            torch.tensor(_logit((0.35 - 0.03) / 1.47), dtype=torch.float64)
        )
        p_fraction = (0.5 - 0.05) / 1.45
        self.power_raw = nn.Parameter(torch.tensor(_logit(p_fraction), dtype=torch.float64))
        if self.environment_enabled:
            self.environment_shift_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        else:
            self.register_buffer("environment_shift_fixed", torch.tensor(0.0, dtype=torch.float64))
        if self.boundary_layer_enabled:
            self.boundary_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        else:
            self.register_buffer("boundary_fixed", torch.tensor(0.0, dtype=torch.float64))

    @property
    def transition_potential_chi(self) -> torch.Tensor:
        log10_value = -10.0 + 5.0 * torch.sigmoid(self.transition_raw)
        return torch.pow(
            torch.tensor(10.0, device=log10_value.device, dtype=log10_value.dtype), log10_value
        )

    @property
    def amplitude(self) -> torch.Tensor:
        log10_value = -4.0 + 6.0 * torch.sigmoid(self.amplitude_raw)
        return torch.pow(
            torch.tensor(10.0, device=log10_value.device, dtype=log10_value.dtype), log10_value
        )

    @property
    def transition_width_dex(self) -> torch.Tensor:
        return 0.03 + 1.47 * torch.sigmoid(self.width_raw)

    @property
    def power(self) -> torch.Tensor:
        return 0.05 + 1.45 * torch.sigmoid(self.power_raw)

    @property
    def environment_shift_zeta(self) -> torch.Tensor:
        if self.environment_enabled:
            return 1.5 * torch.tanh(self.environment_shift_raw)
        return self.environment_shift_fixed

    @property
    def boundary_kappa(self) -> torch.Tensor:
        if self.boundary_layer_enabled:
            return 2.0 * torch.tanh(self.boundary_raw)
        return self.boundary_fixed

    def predict_acceleration(
        self, data: TensorDataset, inputs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        g_bar = torch.clamp(inputs["baryonic_acceleration"], min=1e-30)
        potential_depth = radial_potential_depth(
            g_bar,
            inputs["radius_m"],
            data.galaxy_index,
            self.n_galaxies,
        )
        chi = torch.clamp(potential_depth / (C_M_S**2), min=1e-30)
        environment = data.environment_standardized[data.galaxy_index]
        log10_transition = torch.log10(self.transition_potential_chi) + (
            self.environment_shift_zeta * environment
        )
        log_ratio = torch.log10(chi) - log10_transition
        activation = torch.sigmoid(-log_ratio / self.transition_width_dex)
        reference = torch.as_tensor(
            self.acceleration_reference_m_s2,
            dtype=g_bar.dtype,
            device=g_bar.device,
        )
        additional = (
            self.amplitude * reference * torch.pow(g_bar / reference, self.power) * activation
        )
        if self.boundary_layer_enabled:
            dscreen_dlnr = (
                activation
                * (1.0 - activation)
                / (self.transition_width_dex * math.log(10.0))
                * inputs["radius_m"]
                * g_bar
                / potential_depth
            )
            additional = additional + self.boundary_kappa * reference * dscreen_dlnr
        return g_bar + additional

    def prior_penalty(self) -> torch.Tensor:
        penalty = super().prior_penalty()
        if self.environment_enabled:
            penalty = penalty + 0.5 * (self.environment_shift_zeta / 0.5).square()
        if self.boundary_layer_enabled:
            penalty = penalty + 0.5 * (self.boundary_kappa / 0.5).square()
        return penalty

    def physical_parameters(self) -> dict[str, float | bool]:
        return {
            "transition_potential_chi": float(self.transition_potential_chi.detach().cpu()),
            "amplitude_A0": float(self.amplitude.detach().cpu()),
            "acceleration_reference_m_s2": self.acceleration_reference_m_s2,
            "transition_width_dex": float(self.transition_width_dex.detach().cpu()),
            "power_p": float(self.power.detach().cpu()),
            "environment_shift_zeta_dex_per_sigma": float(
                self.environment_shift_zeta.detach().cpu()
            ),
            "boundary_kappa": float(self.boundary_kappa.detach().cpu()),
            "environment_enabled": self.environment_enabled,
            "boundary_layer_enabled": self.boundary_layer_enabled,
            "potential_tail": "g_bar(R_max) * R_max",
        }


class NFWModel(RotationModel):
    model_name = "nfw"

    def __init__(self, *args, hubble_km_s_mpc: float = 70.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hubble_km_s_mpc = float(hubble_km_s_mpc)
        self.log_v200 = nn.Parameter(
            torch.full((self.n_galaxies,), math.log(100.0), dtype=torch.float64)
        )
        self.log_concentration = nn.Parameter(
            torch.full((self.n_galaxies,), math.log(10.0), dtype=torch.float64)
        )

    @staticmethod
    def _mass_function(value: torch.Tensor) -> torch.Tensor:
        return torch.log1p(value) - value / (1.0 + value)

    def predict_acceleration(
        self, data: TensorDataset, inputs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        idx = data.galaxy_index
        v200 = torch.exp(self.log_v200)[idx]
        concentration = torch.exp(self.log_concentration)[idx]
        hubble_km_s_kpc = self.hubble_km_s_mpc / 1000.0
        r200_kpc = v200 / (10.0 * hubble_km_s_kpc)
        x = torch.clamp(inputs["radius_adjusted_kpc"] / r200_kpc, min=1e-8)
        halo_v2 = (
            v200.square()
            * self._mass_function(concentration * x)
            / (x * self._mass_function(concentration))
        )
        halo_acceleration = halo_v2 * 1e6 / inputs["radius_m"]
        return inputs["baryonic_acceleration"] + halo_acceleration

    def prior_penalty(self) -> torch.Tensor:
        penalty = super().prior_penalty()
        weak_halo_prior = 0.5 * (
            ((self.log_v200 - math.log(100.0)) / 1.0).square().sum()
            + ((self.log_concentration - math.log(10.0)) / 0.6).square().sum()
        )
        return penalty + weak_halo_prior

    def physical_parameters(self) -> dict[str, float]:
        return {
            "hubble_km_s_mpc": self.hubble_km_s_mpc,
            "median_v200_kms": float(torch.exp(self.log_v200).detach().median().cpu()),
            "median_concentration": float(
                torch.exp(self.log_concentration).detach().median().cpu()
            ),
        }


def build_model(
    name: str,
    data: PackedDataset,
    *,
    disk_ml_prior: float,
    bulge_ml_prior: float,
    log_ml_prior_sigma: float,
    rar_acceleration_m_s2: float,
    hubble_km_s_mpc: float,
    fixed_flat_power: bool = False,
    environment_enabled: bool = False,
    boundary_layer_enabled: bool = False,
) -> RotationModel:
    common = {
        "n_galaxies": data.n_galaxies,
        "disk_ml_prior": disk_ml_prior,
        "bulge_ml_prior": bulge_ml_prior,
        "log_ml_prior_sigma": log_ml_prior_sigma,
    }
    if name == "newtonian":
        return NewtonianModel(**common)
    if name == "rar":
        return RARModel(**common, rar_acceleration_m_s2=rar_acceleration_m_s2)
    if name == "nfw":
        return NFWModel(**common, hubble_km_s_mpc=hubble_km_s_mpc)
    if name == "void":
        return VoidScreeningModel(
            **common,
            fixed_flat_power=fixed_flat_power,
            environment_enabled=environment_enabled,
        )
    if name == "potential":
        return PotentialScreeningModel(
            **common,
            acceleration_reference_m_s2=rar_acceleration_m_s2,
            environment_enabled=environment_enabled,
            boundary_layer_enabled=boundary_layer_enabled,
        )
    raise ValueError(f"Unknown model {name!r}")
