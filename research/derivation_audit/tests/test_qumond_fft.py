import numpy as np

from sigma_sprint.qumond_fft import exponential_disk_density, solve_qumond_fft


def test_B_zero_recovers_newtonian_field():
    # Odd grids avoid the unpaired real Nyquist mode in first derivatives.
    density, dx, _ = exponential_disk_density(1e10, 2.0, grid_size=25)
    solution = solve_qumond_fft(density, dx, B=0.0)
    for newton, qumond in zip(solution["g_newton"], solution["g_qumond"]):
        scale = np.max(np.abs(newton))
        assert np.max(np.abs(newton - qumond)) / scale < 1e-11
