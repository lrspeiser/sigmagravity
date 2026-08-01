"""Theory-independent projections of measured X-ray electron-density shells."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


PROTON_G = 1.67262192369e-24
KPC_CM = 3.0856775814913673e21
MSUN_G = 1.98847e33


def _shell_arrays(inner_kpc, outer_kpc, electron_density_cm3):
    inner = np.asarray(inner_kpc, dtype=float)
    outer = np.asarray(outer_kpc, dtype=float)
    density = np.asarray(electron_density_cm3, dtype=float)
    if inner.ndim != 1 or outer.shape != inner.shape or density.shape != inner.shape:
        raise ValueError("shell arrays must be matching vectors")
    if (
        len(inner) == 0
        or np.any(~np.isfinite(inner))
        or np.any(~np.isfinite(outer))
        or np.any(~np.isfinite(density))
        or np.any(inner < 0.0)
        or np.any(outer <= inner)
        or np.any(density < 0.0)
    ):
        raise ValueError("shell geometry and densities must be finite and physical")
    order = np.argsort(inner)
    inner, outer, density = inner[order], outer[order], density[order]
    if np.any(np.abs(inner[1:] - outer[:-1]) > 1.0e-6):
        raise ValueError("shells must be contiguous")
    return inner, outer, density


def projected_gas_surface_density_msun_kpc2(
    radius_kpc,
    inner_kpc,
    outer_kpc,
    electron_density_cm3,
    *,
    mu_e: float = 1.17,
):
    """Project piecewise-constant spherical electron-density shells exactly."""

    if not np.isfinite(mu_e) or mu_e <= 0.0:
        raise ValueError("mu_e must be positive")
    inner, outer, density = _shell_arrays(
        inner_kpc, outer_kpc, electron_density_cm3
    )
    radius = np.asarray(radius_kpc, dtype=float)
    if np.any(~np.isfinite(radius)) or np.any(radius < 0.0):
        raise ValueError("projected radii must be finite and nonnegative")
    flat = radius.ravel()[:, None]
    outer_path = np.sqrt(np.maximum(np.square(outer)[None, :] - np.square(flat), 0.0))
    inner_path = np.sqrt(np.maximum(np.square(inner)[None, :] - np.square(flat), 0.0))
    path_kpc = 2.0 * np.maximum(outer_path - inner_path, 0.0)
    column_electrons_cm2 = np.sum(
        path_kpc * KPC_CM * density[None, :], axis=1
    )
    surface_g_cm2 = float(mu_e) * PROTON_G * column_electrons_cm2
    surface_msun_kpc2 = surface_g_cm2 * KPC_CM**2 / MSUN_G
    return surface_msun_kpc2.reshape(radius.shape)


def enclosed_gas_mass_msun(
    radius_kpc,
    inner_kpc,
    outer_kpc,
    electron_density_cm3,
    *,
    mu_e: float = 1.17,
) -> float:
    """Integrate the same piecewise-constant shells inside a sphere."""

    radius = float(radius_kpc)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("radius must be finite and nonnegative")
    inner, outer, density = _shell_arrays(
        inner_kpc, outer_kpc, electron_density_cm3
    )
    clipped_outer = np.minimum(outer, radius)
    use = clipped_outer > inner
    volume_kpc3 = (4.0 * np.pi / 3.0) * (
        np.power(clipped_outer[use], 3) - np.power(inner[use], 3)
    )
    return float(
        np.sum(volume_kpc3 * KPC_CM**3 * float(mu_e) * PROTON_G * density[use])
        / MSUN_G
    )


def annular_morphology_factor(
    axis_arcsec,
    positive_image,
    *,
    power: float,
    smoothing_sigma_arcsec: float,
    contrast_min: float = 0.25,
    contrast_max: float = 4.0,
):
    """Return a capped angular factor with unit mean in each radial pixel bin."""

    axis = np.asarray(axis_arcsec, dtype=float)
    image = np.maximum(np.asarray(positive_image, dtype=float), 0.0)
    if image.shape != (len(axis), len(axis)) or len(axis) < 3:
        raise ValueError("image must match a nontrivial square axis")
    spacing = float(np.median(np.diff(axis)))
    if spacing <= 0.0 or not np.allclose(np.diff(axis), spacing):
        raise ValueError("axis must be uniformly increasing")
    if power < 0.0 or smoothing_sigma_arcsec <= 0.0:
        raise ValueError("power must be nonnegative and smoothing positive")
    if not (0.0 < contrast_min <= 1.0 <= contrast_max):
        raise ValueError("contrast bounds must bracket one")
    smoothed = gaussian_filter(
        image, float(smoothing_sigma_arcsec) / spacing, mode="nearest"
    )
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    bins = np.floor(np.hypot(xx, yy) / spacing).astype(int)
    factor = np.ones_like(smoothed)
    for index in range(int(bins.max()) + 1):
        mask = bins == index
        mean = float(np.mean(smoothed[mask]))
        if mean <= 0.0 or power == 0.0:
            continue
        local = np.power(np.maximum(smoothed[mask] / mean, 0.0), float(power))
        # Solve for a scale whose clipped values have mean one.  This retains
        # both the hard contrast bounds and the radial surface-density profile.
        low_scale, high_scale = 0.0, 1.0
        while np.mean(np.clip(high_scale * local, contrast_min, contrast_max)) < 1.0:
            high_scale *= 2.0
        for _ in range(64):
            trial = 0.5 * (low_scale + high_scale)
            if np.mean(np.clip(trial * local, contrast_min, contrast_max)) < 1.0:
                low_scale = trial
            else:
                high_scale = trial
        factor[mask] = np.clip(
            0.5 * (low_scale + high_scale) * local,
            float(contrast_min),
            float(contrast_max),
        )
    return factor
