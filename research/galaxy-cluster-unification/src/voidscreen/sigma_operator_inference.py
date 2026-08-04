from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.signal.windows import tukey


def apodization_window(shape: tuple[int, int], alpha: float) -> np.ndarray:
    """Return a separable Tukey window for a two-dimensional map."""
    if len(shape) != 2 or min(shape) < 4:
        raise ValueError("shape must contain two dimensions of at least four pixels")
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie between zero and one")
    return np.outer(tukey(shape[0], alpha), tukey(shape[1], alpha))


def windowed_fourier(field, window) -> np.ndarray:
    """Subtract the weighted mean, apodize, and return an orthonormal FFT."""
    values = np.asarray(field, dtype=float)
    weights = np.asarray(window, dtype=float)
    if values.shape != weights.shape or values.ndim != 2:
        raise ValueError("field and window must be matching two-dimensional arrays")
    if np.any(~np.isfinite(values)) or np.any(~np.isfinite(weights)):
        raise ValueError("field and window must be finite")
    normalization = float(np.sum(weights))
    if normalization <= 0.0:
        raise ValueError("window must have positive support")
    mean = float(np.sum(weights * values) / normalization)
    return np.fft.fft2((values - mean) * weights, norm="ortho")


def angular_wavenumber_grid(shape: tuple[int, int], spacing: float) -> np.ndarray:
    """Return the radial angular wavenumber in inverse spacing units."""
    if len(shape) != 2 or min(shape) < 4 or spacing <= 0.0:
        raise ValueError("shape and spacing are invalid")
    ky = 2.0 * np.pi * np.fft.fftfreq(shape[0], d=spacing)
    kx = 2.0 * np.pi * np.fft.fftfreq(shape[1], d=spacing)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    return np.hypot(kx_grid, ky_grid)


def wavelength_band_mask(
    wavenumber,
    minimum_wavelength: float,
    maximum_wavelength: float,
) -> np.ndarray:
    """Select modes whose wavelengths are inside the declared physical band."""
    k = np.asarray(wavenumber, dtype=float)
    if (
        np.any(~np.isfinite(k))
        or np.any(k < 0.0)
        or minimum_wavelength <= 0.0
        or maximum_wavelength <= minimum_wavelength
    ):
        raise ValueError("wavenumber and wavelength bounds are invalid")
    return (k >= 2.0 * np.pi / maximum_wavelength) & (
        k <= 2.0 * np.pi / minimum_wavelength
    )


def radial_transfer_spectrum(
    source_channels: dict[str, np.ndarray],
    target_channels: dict[str, np.ndarray],
    wavenumber,
    band,
    *,
    bins: int,
) -> pd.DataFrame:
    """Fit one real isotropic transfer per logarithmic radial wavenumber bin."""
    if source_channels.keys() != target_channels.keys() or not source_channels:
        raise ValueError("source and target channels must have identical nonempty keys")
    k = np.asarray(wavenumber, dtype=float)
    selected = np.asarray(band, dtype=bool)
    if k.shape != selected.shape or bins < 2 or not np.any(selected):
        raise ValueError("wavenumber grid, band, or bin count is invalid")
    lower = float(np.min(k[selected]))
    upper = float(np.max(k[selected]))
    edges = np.geomspace(lower, upper * np.nextafter(1.0, 2.0), bins + 1)
    records: list[dict[str, float | int]] = []
    for index in range(bins):
        in_bin = selected & (k >= edges[index]) & (k < edges[index + 1])
        source_power = 0.0
        target_power = 0.0
        cross = 0.0j
        for name, source_values in source_channels.items():
            source = np.asarray(source_values)
            target = np.asarray(target_channels[name])
            if source.shape != k.shape or target.shape != k.shape:
                raise ValueError("all Fourier channels must match the wavenumber grid")
            source_power += float(np.sum(np.abs(source[in_bin]) ** 2))
            target_power += float(np.sum(np.abs(target[in_bin]) ** 2))
            cross += np.sum(target[in_bin] * np.conjugate(source[in_bin]))
        if not np.any(in_bin) or source_power <= 0.0 or target_power <= 0.0:
            continue
        transfer = float(np.real(cross) / source_power)
        coherence = float(np.abs(cross) ** 2 / (source_power * target_power))
        records.append(
            {
                "bin": index,
                "k_min_per_unit": edges[index],
                "k_max_per_unit": edges[index + 1],
                "k_geometric_per_unit": math.sqrt(edges[index] * edges[index + 1]),
                "wavelength_geometric_unit": 2.0
                * np.pi
                / math.sqrt(edges[index] * edges[index + 1]),
                "modes": int(np.count_nonzero(in_bin)),
                "best_real_transfer": transfer,
                "coherence": coherence,
                "source_power": source_power,
                "target_power": target_power,
                "imaginary_cross_fraction": float(
                    abs(np.imag(cross)) / max(abs(cross), np.finfo(float).tiny)
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def transfer_grid_from_spectrum(
    spectrum: pd.DataFrame,
    wavenumber,
    *,
    outside_value: float = 1.0,
    clip_nonnegative: bool = False,
) -> np.ndarray:
    """Map a radial binned transfer back onto a Fourier grid."""
    k = np.asarray(wavenumber, dtype=float)
    result = np.full_like(k, float(outside_value))
    for row in spectrum.itertuples(index=False):
        value = float(row.best_real_transfer)
        if clip_nonnegative:
            value = max(value, 0.0)
        selected = (k >= float(row.k_min_per_unit)) & (k < float(row.k_max_per_unit))
        result[selected] = value
    return result


def normalized_channel_rmse(
    source_channels: dict[str, np.ndarray],
    target_channels: dict[str, np.ndarray],
    transfer,
    band,
) -> float:
    """Equal-channel normalized Fourier RMSE for a shared real transfer."""
    if source_channels.keys() != target_channels.keys() or not source_channels:
        raise ValueError("source and target channels must have identical nonempty keys")
    response = np.asarray(transfer, dtype=float)
    selected = np.asarray(band, dtype=bool)
    scores = []
    for name, source_values in source_channels.items():
        source = np.asarray(source_values)
        target = np.asarray(target_channels[name])
        if source.shape != response.shape or target.shape != response.shape:
            raise ValueError("channels and transfer must have matching shapes")
        denominator = float(np.sum(np.abs(target[selected]) ** 2))
        if denominator <= 0.0:
            raise ValueError("target channel has no power in the selected band")
        numerator = float(np.sum(np.abs(response[selected] * source[selected] - target[selected]) ** 2))
        scores.append(numerator / denominator)
    return float(math.sqrt(np.mean(scores)))
