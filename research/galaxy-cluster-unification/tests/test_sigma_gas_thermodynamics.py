from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_gas_thermodynamics import (
    KPC_CM,
    PROTON_MASS_G,
    apec_emission_measure_cm3,
    compression_mach_number,
    shock_speed_km_s,
    temperature_jump_from_mach,
    temperature_jump_mach_number,
    uniform_slab_thermodynamics,
)


def test_apec_normalization_recovers_official_emission_measure_definition() -> None:
    distance = 1.2e27
    redshift = 0.3
    emission_measure = 7.5e67
    norm = emission_measure * 1.0e-14 / (4.0 * np.pi * (distance * (1 + redshift)) ** 2)
    assert apec_emission_measure_cm3(norm, distance, redshift) == pytest.approx(
        emission_measure
    )


def test_uniform_slab_uses_electron_to_hydrogen_ratio_in_numerator() -> None:
    ratio = 1.2
    electron_density = 2.0e-3
    area_kpc2 = 100.0
    depth_kpc = 800.0
    distance = 1.0e27
    redshift = 0.25
    emission_measure = (
        electron_density**2
        / ratio
        * area_kpc2
        * KPC_CM**2
        * depth_kpc
        * KPC_CM
    )
    norm = emission_measure * 1.0e-14 / (4.0 * np.pi * (distance * (1 + redshift)) ** 2)
    state = uniform_slab_thermodynamics(
        norm,
        8.0,
        distance,
        redshift,
        area_kpc2,
        depth_kpc,
        electron_to_hydrogen_ratio=ratio,
    )
    assert state["electron_density_cm3"] == pytest.approx(electron_density)
    expected_surface_g_cm2 = 1.17 * PROTON_MASS_G * electron_density * depth_kpc * KPC_CM
    recovered_surface_g_cm2 = (
        state["gas_surface_density_msun_kpc2"]
        * 1.988409870698051e33
        / KPC_CM**2
    )
    assert recovered_surface_g_cm2 == pytest.approx(expected_surface_g_cm2)


def test_depth_scalings_match_uniform_slab_first_principles() -> None:
    shallow = uniform_slab_thermodynamics(1e-3, 7.0, 1e27, 0.2, 200.0, 100.0)
    deep = uniform_slab_thermodynamics(1e-3, 7.0, 1e27, 0.2, 200.0, 400.0)
    assert deep["electron_density_cm3"] / shallow["electron_density_cm3"] == pytest.approx(
        0.5
    )
    assert deep["gas_surface_density_msun_kpc2"] / shallow[
        "gas_surface_density_msun_kpc2"
    ] == pytest.approx(2.0)
    assert deep["thermal_pressure_erg_cm3"] / shallow[
        "thermal_pressure_erg_cm3"
    ] == pytest.approx(0.5)


def test_temperature_scalings_separate_pressure_entropy_and_sound_speed() -> None:
    cool = uniform_slab_thermodynamics(1e-3, 4.0, 1e27, 0.2, 200.0, 500.0)
    hot = uniform_slab_thermodynamics(1e-3, 16.0, 1e27, 0.2, 200.0, 500.0)
    assert hot["electron_density_cm3"] == pytest.approx(cool["electron_density_cm3"])
    assert hot["thermal_pressure_erg_cm3"] / cool["thermal_pressure_erg_cm3"] == pytest.approx(
        4.0
    )
    assert hot["entropy_proxy_keV_cm2"] / cool["entropy_proxy_keV_cm2"] == pytest.approx(
        4.0
    )
    assert hot["sound_speed_km_s"] / cool["sound_speed_km_s"] == pytest.approx(2.0)


def test_rankine_hugoniot_density_and_temperature_inversions_round_trip() -> None:
    mach = np.array([1.0, 1.5, 3.0, 7.0])
    gamma = 5.0 / 3.0
    compression = (gamma + 1.0) * mach**2 / ((gamma - 1.0) * mach**2 + 2.0)
    assert compression_mach_number(compression) == pytest.approx(mach)
    jump = temperature_jump_from_mach(mach)
    assert temperature_jump_mach_number(jump) == pytest.approx(mach, rel=1e-10)
    assert shock_speed_km_s(mach, 1000.0) == pytest.approx(mach * 1000.0)


def test_invalid_gas_and_shock_states_fail_closed() -> None:
    with pytest.raises(ValueError, match="normalization"):
        apec_emission_measure_cm3(0.0, 1e27, 0.2)
    with pytest.raises(ValueError, match="density_compression"):
        compression_mach_number(4.0)
    with pytest.raises(ValueError, match="temperature_jump"):
        temperature_jump_mach_number(0.9)
