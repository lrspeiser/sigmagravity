import numpy as np

from scripts.run_sparc_density_transfer import hernquist_density_msun_pc3


def test_hernquist_density_is_positive_and_decreases_outward():
    density = hernquist_density_msun_pc3(1.0e11, [1.0, 10.0, 100.0], 2.0)
    assert np.all(density > 0.0)
    assert np.all(np.diff(density) < 0.0)


def test_hernquist_density_handles_absent_bulge():
    density = hernquist_density_msun_pc3([0.0, 1.0e10], [1.0, 1.0], [0.0, 1.0])
    assert density[0] == 0.0
    assert density[1] > 0.0
