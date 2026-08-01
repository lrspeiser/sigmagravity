from __future__ import annotations

import numpy as np

from scripts.run_cpr0_radial_lensing_bridge import (
    nfw_concentration_from_two_masses,
    overdensity_radius_mpc,
)
from voidscreen.host_profiles import nfw_mass_function


def test_two_mass_nfw_reconstruction_recovers_mass_ratio() -> None:
    mass200 = 1.0e15
    mass500 = 6.8e14
    concentration, r200, r500 = nfw_concentration_from_two_masses(
        mass200, mass500, 0.2
    )
    recovered = nfw_mass_function(concentration * r500 / r200) / nfw_mass_function(
        concentration
    )
    assert 0.0 < concentration < 30.0
    assert np.isclose(recovered, mass500 / mass200)


def test_overdensity_radius_preserves_definition() -> None:
    mass = 5.0e14
    r200 = overdensity_radius_mpc(mass, 200.0, 0.1)
    r500 = overdensity_radius_mpc(mass, 500.0, 0.1)
    assert r200 > r500 > 0.0
