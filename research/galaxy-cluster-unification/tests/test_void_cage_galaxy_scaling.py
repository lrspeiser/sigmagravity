from pathlib import Path

import numpy as np

from voidscreen.data import pack_dataset
from voidscreen.galaxy_scaling import (
    catalog_scaled_screened_velocity,
    load_sparc_structural_predictors,
    local_acceleration_screened_velocity,
    normalize_positive_by_training_median,
)
from voidscreen.void_cage import baryonic_velocity_squared


ROOT = Path(__file__).resolve().parents[1]


def test_structural_predictors_are_complete_and_residual_blind() -> None:
    packed = pack_dataset(ROOT / "data" / "raw" / "sparc")
    predictors = load_sparc_structural_predictors(
        ROOT / "data" / "raw" / "sparc" / "table1.dat", packed.galaxy_names
    )
    assert predictors.names == packed.galaxy_names
    assert np.all(predictors.mass_proxy_1e9_msun > 0.0)
    assert np.all(predictors.central_stellar_surface_density_msun_pc2 > 0.0)
    assert np.all(predictors.concentration_rdisk_over_reff > 0.0)


def test_training_normalization_does_not_use_heldout_values() -> None:
    values = np.asarray([1.0, 2.0, 1000.0])
    normalized, median = normalize_positive_by_training_median(
        values, np.asarray([True, True, False])
    )
    assert median == 1.5
    assert np.allclose(normalized[:2], [2.0 / 3.0, 4.0 / 3.0])


def test_local_screening_activates_more_at_low_baryonic_acceleration() -> None:
    packed = pack_dataset(ROOT / "data" / "raw" / "sparc")
    baryonic = baryonic_velocity_squared(packed)
    predicted = local_acceleration_screened_velocity(
        packed,
        baryonic,
        log10_velocity_scale_km_s=2.0,
        log10_gstar_m_s2=-10.0,
        screening_power=1.0,
    )
    added_v2 = predicted**2 - baryonic
    gbar = baryonic * 1e6 / (packed.radius_kpc * 3.085677581491367e19)
    order = np.argsort(gbar)
    assert float(np.mean(added_v2[order[:100]])) > float(
        np.mean(added_v2[order[-100:]])
    )


def test_zero_exponents_nest_catalog_and_environment_controls() -> None:
    packed = pack_dataset(ROOT / "data" / "raw" / "sparc")
    baryonic = baryonic_velocity_squared(packed)
    ones = np.ones(packed.n_galaxies)
    first = catalog_scaled_screened_velocity(
        packed,
        baryonic,
        mass_by_galaxy=np.geomspace(0.1, 10.0, packed.n_galaxies),
        transition_driver_by_galaxy=np.geomspace(0.2, 5.0, packed.n_galaxies),
        log10_velocity_scale_km_s=2.0,
        log10_transition_scale_lengths=0.5,
        mass_amplitude_exponent=0.0,
        transition_exponent=0.0,
        environment_by_galaxy=np.geomspace(0.5, 2.0, packed.n_galaxies),
        environment_exponent=0.0,
    )
    nested = catalog_scaled_screened_velocity(
        packed,
        baryonic,
        mass_by_galaxy=ones,
        transition_driver_by_galaxy=ones,
        log10_velocity_scale_km_s=2.0,
        log10_transition_scale_lengths=0.5,
        mass_amplitude_exponent=0.0,
        transition_exponent=0.0,
    )
    assert np.allclose(first, nested)
