from __future__ import annotations

import numpy as np

from voidscreen.inverse_response import (
    analyze_stationary_response,
    convolve_stationary_response,
    fit_stationary_response_kernel,
    radial_angle_shuffle,
)


def anisotropic_source(cells: int, phase: float) -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, cells)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    first = np.exp(-((x - 0.28 * np.cos(phase)) ** 2 / 0.09 + (y + 0.2) ** 2 / 0.18))
    second = 0.7 * np.exp(-((x + 0.35) ** 2 / 0.2 + (y - 0.25 * np.sin(phase)) ** 2 / 0.06))
    arm = 0.2 * np.maximum(1.0 + np.cos(4.0 * np.arctan2(y, x) + phase), 0.0)
    return first + second + arm * np.exp(-2.5 * np.hypot(x, y))


def injected_kernel() -> np.ndarray:
    axis = np.arange(-2.0, 3.0)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    kernel = np.exp(-((x - 0.45) ** 2 / 1.8 + (y + 0.25) ** 2 / 3.0))
    return kernel / np.sum(kernel)


def test_multiple_system_inverse_recovers_injected_kernel_and_amplitude() -> None:
    sources = [anisotropic_source(17, 0.2), anisotropic_source(17, 1.1)]
    truth = injected_kernel()
    amplitude = 1.7
    sigma = 0.004
    rng = np.random.default_rng(9)
    noiseless = [
        amplitude * convolve_stationary_response(source, truth, 1.0)
        for source in sources
    ]
    targets = [target + rng.normal(0.0, sigma, target.shape) for target in noiseless]
    uncertainties = [np.full_like(target, sigma) for target in targets]
    analysis = analyze_stationary_response(
        sources,
        targets,
        1.0,
        truth.shape,
        uncertainties=uncertainties,
        ridge=1.0e-10,
        smoothness=1.0e-8,
        ensemble_size=20,
        ensemble_seed=17,
        null_count=19,
        null_seed=23,
    )
    recovered = analysis.fit.normalized_kernel
    cosine = float(
        np.sum(recovered * truth)
        / (np.linalg.norm(recovered) * np.linalg.norm(truth))
    )
    assert cosine > 0.995
    assert abs(analysis.fit.amplitude / amplitude - 1.0) < 0.02
    assert analysis.fit.aggregate_metrics["r_squared"] > 0.999
    assert analysis.amplitude_interval["lower_2_5"] < amplitude
    assert analysis.amplitude_interval["upper_97_5"] > amplitude
    assert analysis.null_summary["signal_against_null"] is True
    assert analysis.null_summary["permutation_p_value"] == 0.05


def test_null_data_do_not_produce_a_significant_route() -> None:
    sources = [anisotropic_source(15, 0.4), anisotropic_source(15, 1.7)]
    targets = [np.full_like(source, 0.5) for source in sources]
    uncertainties = [np.full_like(source, 0.05) for source in sources]
    analysis = analyze_stationary_response(
        sources,
        targets,
        1.0,
        (5, 5),
        uncertainties=uncertainties,
        ridge=1.0e-6,
        smoothness=1.0e-3,
        ensemble_size=0,
        null_count=19,
        null_seed=31,
    )
    assert analysis.null_summary["signal_against_null"] is False
    assert analysis.fit.aggregate_metrics["r_squared"] < 0.25


def test_parametric_ensemble_has_empirical_amplitude_and_kernel_coverage() -> None:
    sources = [anisotropic_source(13, 0.3), anisotropic_source(13, 1.4)]
    axis = np.arange(-1.0, 2.0)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    truth = np.exp(-((x - 0.2) ** 2 / 1.4 + (y + 0.15) ** 2 / 1.8))
    truth /= np.sum(truth)
    amplitude = 1.4
    sigma = 0.008
    noiseless = [
        amplitude * convolve_stationary_response(source, truth, 1.0)
        for source in sources
    ]
    amplitude_covered = 0
    center_covered = 0
    for trial in range(10):
        rng = np.random.default_rng(100 + trial)
        targets = [
            target + rng.normal(0.0, sigma, target.shape) for target in noiseless
        ]
        analysis = analyze_stationary_response(
            sources,
            targets,
            1.0,
            truth.shape,
            uncertainties=[np.full_like(target, sigma) for target in targets],
            ridge=1.0e-10,
            smoothness=1.0e-8,
            ensemble_size=64,
            ensemble_seed=1000 + trial,
            null_count=0,
            regularization_multipliers=(1.0,),
        )
        amplitude_covered += int(
            analysis.amplitude_interval["lower_2_5"]
            <= amplitude
            <= analysis.amplitude_interval["upper_97_5"]
        )
        center_covered += int(
            analysis.kernel_lower[1, 1]
            <= truth[1, 1]
            <= analysis.kernel_upper[1, 1]
        )
    assert amplitude_covered >= 8
    assert center_covered >= 7


def test_rank_diagnostic_reveals_non_identifiable_constant_source() -> None:
    source = np.ones((13, 13), dtype=float)
    target = np.full_like(source, 2.0)
    mask = np.zeros_like(source, dtype=bool)
    mask[3:-3, 3:-3] = True
    analysis = analyze_stationary_response(
        [source],
        [target],
        1.0,
        (5, 5),
        masks=[mask],
        ridge=1.0e-8,
        smoothness=1.0e-4,
        ensemble_size=0,
        null_count=0,
    )
    assert analysis.fit.identifiability["effective_rank"] == 1
    assert analysis.fit.identifiability["nullity"] == 24
    assert analysis.non_identifiability["non_identifiable"] is True


def test_three_dimensional_impulse_recovers_compact_kernel_exactly() -> None:
    source = np.zeros((7, 7, 7), dtype=float)
    source[3, 3, 3] = 1.0 / 8.0
    kernel = np.arange(1.0, 28.0).reshape(3, 3, 3)
    kernel /= np.sum(kernel) * 8.0
    amplitude = 2.5
    target = amplitude * convolve_stationary_response(source, kernel, [2.0, 2.0, 2.0])
    fit = fit_stationary_response_kernel(
        [source],
        [target],
        [2.0, 2.0, 2.0],
        kernel.shape,
        ridge=0.0,
        smoothness=0.0,
    )
    np.testing.assert_allclose(fit.normalized_kernel, kernel, rtol=1e-10, atol=1e-12)
    assert np.isclose(fit.amplitude, amplitude, rtol=1e-10)


def test_signed_compensated_kernel_uses_l1_amplitude_and_remains_predictive() -> None:
    source = np.zeros((9, 9), dtype=float)
    source[4, 4] = 1.0
    shape = np.array(
        [
            [0.0, -0.125, 0.0],
            [-0.125, 0.5, -0.125],
            [0.0, -0.125, 0.0],
        ]
    )
    assert np.isclose(np.sum(shape), 0.0)
    assert np.isclose(np.sum(np.abs(shape)), 1.0)
    amplitude = 2.25
    target = amplitude * convolve_stationary_response(source, shape, 1.0)
    fit = fit_stationary_response_kernel(
        [source],
        [target],
        1.0,
        shape.shape,
        ridge=0.0,
        smoothness=0.0,
        nonnegative=False,
    )
    np.testing.assert_allclose(fit.normalized_kernel, shape, rtol=1e-10, atol=1e-12)
    assert np.isclose(fit.amplitude, amplitude, rtol=1e-10)
    np.testing.assert_allclose(fit.predictions[0], target, rtol=1e-10, atol=1e-12)


def test_radial_shuffle_preserves_each_shell_but_changes_angles() -> None:
    values = np.arange(81, dtype=float).reshape(9, 9)
    shuffled = radial_angle_shuffle(values, np.random.default_rng(4))
    coordinates = np.indices(values.shape, dtype=float)
    radius = np.rint(
        np.hypot(coordinates[0] - 4.0, coordinates[1] - 4.0)
    ).astype(int)
    for shell in np.unique(radius):
        np.testing.assert_array_equal(
            np.sort(shuffled[radius == shell]), np.sort(values[radius == shell])
        )
    assert not np.array_equal(shuffled, values)
