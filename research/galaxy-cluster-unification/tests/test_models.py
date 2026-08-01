import numpy as np
import torch

from voidscreen.data import KPC_M
from voidscreen.models import (
    NewtonianModel,
    PotentialScreeningModel,
    SigmaTransferModel,
    TensorDataset,
    VoidScreeningModel,
    radial_potential_depth,
)


def make_tensor_data(
    *,
    radius_kpc: list[float],
    gas: list[float],
    disk: list[float],
    bulge: list[float],
) -> TensorDataset:
    n = len(radius_kpc)
    floating = lambda value: torch.tensor(value, dtype=torch.float64)
    return TensorDataset(
        galaxy_names=("synthetic",),
        galaxy_index=torch.zeros(n, dtype=torch.long),
        radius_kpc=floating(radius_kpc),
        velocity_observed_kms=floating([100.0] * n),
        velocity_error_kms=floating([2.0] * n),
        velocity_gas_kms=floating(gas),
        velocity_disk_unit_ml_kms=floating(disk),
        velocity_bulge_unit_ml_kms=floating(bulge),
        train_mask=torch.ones(n, dtype=torch.bool),
        distance_fractional_error=floating([0.0]),
        inclination_deg=floating([60.0]),
        inclination_error_deg=floating([0.0]),
        environment_standardized=floating([0.0]),
    )


def test_signed_gas_contribution_is_preserved() -> None:
    data = make_tensor_data(radius_kpc=[1.0], gas=[-10.0], disk=[20.0], bulge=[0.0])
    model = NewtonianModel(n_galaxies=1)
    prediction = model(data)
    # sign(Vgas) Vgas^2 + 0.5 Vdisk^2 = -100 + 200 = 100 (km/s)^2
    assert torch.allclose(
        prediction.velocity_baryonic_kms, torch.tensor([10.0], dtype=torch.float64)
    )


def test_sigma_transfer_matches_declared_enhancement() -> None:
    data = make_tensor_data(
        radius_kpc=[1.0], gas=[0.0], disk=[100.0], bulge=[0.0]
    )
    model = SigmaTransferModel(
        n_galaxies=1, response_amplitude=5.0, g_dagger_m_s2=9.6e-11
    )
    prediction = model(data)
    gbar = prediction.baryonic_acceleration_m_s2
    h = torch.sqrt(torch.tensor(9.6e-11) / gbar) * 9.6e-11 / (9.6e-11 + gbar)
    assert torch.allclose(prediction.predicted_acceleration_m_s2 / gbar, 1.0 + 5.0 * h)


def test_fixed_half_power_has_flat_outer_added_velocity_limit() -> None:
    radius = np.asarray([10.0, 20.0, 40.0, 80.0])
    # Choose Vbar proportional to R^-1/2, hence gbar proportional to R^-2.
    baryonic_velocity = 10.0 / np.sqrt(radius)
    data = make_tensor_data(
        radius_kpc=radius.tolist(),
        gas=[0.0] * len(radius),
        disk=(baryonic_velocity / np.sqrt(0.5)).tolist(),
        bulge=[0.0] * len(radius),
    )
    model = VoidScreeningModel(n_galaxies=1, fixed_flat_power=True)
    prediction = model(data)
    added_v2 = (
        prediction.velocity_predicted_kms.square() - prediction.velocity_baryonic_kms.square()
    )
    relative_range = (added_v2.max() - added_v2.min()) / added_v2.mean()
    assert float(relative_range.detach()) < 0.02


def test_void_activation_grows_as_baryonic_acceleration_falls() -> None:
    data = make_tensor_data(
        radius_kpc=[0.1, 1.0, 10.0, 100.0],
        gas=[0.0] * 4,
        disk=[100.0] * 4,
        bulge=[0.0] * 4,
    )
    model = VoidScreeningModel(n_galaxies=1)
    prediction = model(data)
    fractional_extra = (
        prediction.predicted_acceleration_m_s2 / prediction.baryonic_acceleration_m_s2 - 1.0
    )
    assert torch.all(torch.diff(fractional_extra) > 0.0)


def test_radial_potential_depth_recovers_point_mass_limit() -> None:
    radius = torch.logspace(0.0, 3.0, 2000, dtype=torch.float64) * KPC_M
    gm = torch.tensor(2.5e30, dtype=torch.float64)
    acceleration = gm / radius.square()
    depth = radial_potential_depth(
        acceleration,
        radius,
        torch.zeros(radius.numel(), dtype=torch.long),
        n_galaxies=1,
    )
    expected = gm / radius
    assert torch.allclose(depth, expected, rtol=2e-5, atol=0.0)


def test_potential_screen_activates_toward_shallower_outer_potential() -> None:
    radius = np.geomspace(1.0, 100.0, 60)
    baryonic_velocity = 150.0 / np.sqrt(radius)
    data = make_tensor_data(
        radius_kpc=radius.tolist(),
        gas=[0.0] * len(radius),
        disk=(baryonic_velocity / np.sqrt(0.5)).tolist(),
        bulge=[0.0] * len(radius),
    )
    model = PotentialScreeningModel(n_galaxies=1)
    prediction = model(data)
    fractional_extra = (
        prediction.predicted_acceleration_m_s2 / prediction.baryonic_acceleration_m_s2 - 1.0
    )
    assert torch.all(torch.diff(fractional_extra) > 0.0)


def test_positive_environment_shift_unscreens_at_greater_potential_depth() -> None:
    one = make_tensor_data(
        radius_kpc=[1.0, 3.0, 10.0],
        gas=[0.0] * 3,
        disk=[120.0, 80.0, 45.0],
        bulge=[0.0] * 3,
    )
    data = TensorDataset(
        galaxy_names=("underdense", "overdense"),
        galaxy_index=torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
        radius_kpc=torch.cat((one.radius_kpc, one.radius_kpc)),
        velocity_observed_kms=torch.cat((one.velocity_observed_kms, one.velocity_observed_kms)),
        velocity_error_kms=torch.cat((one.velocity_error_kms, one.velocity_error_kms)),
        velocity_gas_kms=torch.cat((one.velocity_gas_kms, one.velocity_gas_kms)),
        velocity_disk_unit_ml_kms=torch.cat(
            (one.velocity_disk_unit_ml_kms, one.velocity_disk_unit_ml_kms)
        ),
        velocity_bulge_unit_ml_kms=torch.cat(
            (one.velocity_bulge_unit_ml_kms, one.velocity_bulge_unit_ml_kms)
        ),
        train_mask=torch.ones(6, dtype=torch.bool),
        distance_fractional_error=torch.zeros(2, dtype=torch.float64),
        inclination_deg=torch.tensor([60.0, 60.0], dtype=torch.float64),
        inclination_error_deg=torch.zeros(2, dtype=torch.float64),
        environment_standardized=torch.tensor([1.0, -1.0], dtype=torch.float64),
    )
    model = PotentialScreeningModel(n_galaxies=2, environment_enabled=True)
    with torch.no_grad():
        model.environment_shift_raw.fill_(1.0)
    prediction = model(data)
    extra = prediction.predicted_acceleration_m_s2 - prediction.baryonic_acceleration_m_s2
    assert torch.all(extra[:3] > extra[3:])


def test_boundary_layer_has_finite_gradients() -> None:
    data = make_tensor_data(
        radius_kpc=[0.5, 1.0, 2.0, 4.0, 8.0],
        gas=[0.0] * 5,
        disk=[130.0, 110.0, 85.0, 60.0, 40.0],
        bulge=[0.0] * 5,
    )
    model = PotentialScreeningModel(n_galaxies=1, boundary_layer_enabled=True)
    prediction = model(data)
    loss = prediction.velocity_predicted_kms.sum() + model.prior_penalty()
    loss.backward()
    assert model.boundary_raw.grad is not None
    assert torch.isfinite(model.boundary_raw.grad)
    assert torch.isfinite(prediction.predicted_acceleration_m_s2).all()
