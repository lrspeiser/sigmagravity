"""Padded-grid QUMOND check of the algebraic disk approximation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .model import DEFAULT_G_DAGGER, G_SI, nu

KPC_M = 3.085677581491367e19
MSUN_KG = 1.98847e30


def _spectral_gradient(field_k, wavevectors):
    return [np.fft.ifftn(1j * component * field_k).real for component in wavevectors]


def solve_qumond_fft(rho, dx_m: float, B: float, g_dagger=DEFAULT_G_DAGGER):
    """Solve the two QUMOND Poisson equations on a large periodic padded box."""
    density = np.asarray(rho, dtype=float)
    if density.ndim != 3 or len(set(density.shape)) != 1:
        raise ValueError("rho must be a cubic three-dimensional array")
    n = density.shape[0]
    frequencies = 2.0 * np.pi * np.fft.fftfreq(n, d=dx_m)
    kx, ky, kz = np.meshgrid(frequencies, frequencies, frequencies, indexing="ij")
    k2 = kx**2 + ky**2 + kz**2
    nonzero = k2 > 0
    rho_k = np.fft.fftn(density)
    phi_n_k = np.zeros_like(rho_k, dtype=complex)
    phi_n_k[nonzero] = -4.0 * np.pi * G_SI * rho_k[nonzero] / k2[nonzero]
    grad_phi_n = _spectral_gradient(phi_n_k, (kx, ky, kz))
    g_n = [-component for component in grad_phi_n]
    g_n_magnitude = np.sqrt(sum(component**2 for component in g_n))
    floor = max(float(np.nanmax(g_n_magnitude)) * 1e-12, g_dagger * 1e-15)
    response = nu(np.maximum(g_n_magnitude, floor), B, g_dagger)
    flux_k = [np.fft.fftn(response * component) for component in grad_phi_n]
    source_k = 1j * kx * flux_k[0] + 1j * ky * flux_k[1] + 1j * kz * flux_k[2]
    phi_k = np.zeros_like(source_k, dtype=complex)
    phi_k[nonzero] = -source_k[nonzero] / k2[nonzero]
    grad_phi = _spectral_gradient(phi_k, (kx, ky, kz))
    acceleration = [-component for component in grad_phi]
    return {
        "g_newton": g_n,
        "g_newton_magnitude": g_n_magnitude,
        "g_qumond": acceleration,
        "nu": response,
    }


def exponential_disk_density(
    mass_msun: float,
    rd_kpc: float,
    *,
    grid_size: int = 64,
    half_box_rd: float = 8.0,
    scale_height_rd: float = 0.2,
):
    """Return a normalized exponential-sech^2 axisymmetric disk."""
    half_box_m = half_box_rd * rd_kpc * KPC_M
    dx = 2.0 * half_box_m / grid_size
    coordinate = (np.arange(grid_size) - grid_size // 2) * dx
    x, y, z = np.meshgrid(coordinate, coordinate, coordinate, indexing="ij")
    cylindrical_radius = np.sqrt(x**2 + y**2)
    rd_m = rd_kpc * KPC_M
    height_m = scale_height_rd * rd_m
    density_shape = np.exp(-cylindrical_radius / rd_m) / np.cosh(z / height_m) ** 2
    density = density_shape * (mass_msun * MSUN_KG) / (density_shape.sum() * dx**3)
    return density, dx, coordinate


def compare_axisymmetric_disk(
    mass_msun: float,
    rd_kpc: float,
    *,
    B: float = 1.0,
    grid_size: int = 64,
    galaxy: str = "synthetic",
) -> tuple[dict, pd.DataFrame]:
    density, dx, coordinate = exponential_disk_density(
        mass_msun, rd_kpc, grid_size=grid_size
    )
    solution = solve_qumond_fft(density, dx, B)
    center = grid_size // 2
    x = coordinate[center + 1 :]
    g_newton_inward = -solution["g_newton"][0][center + 1 :, center, center]
    g_exact_inward = -solution["g_qumond"][0][center + 1 :, center, center]
    g_newton_magnitude = solution["g_newton_magnitude"][center + 1 :, center, center]
    g_algebraic = nu(g_newton_magnitude, B) * g_newton_inward
    radius_rd = x / (rd_kpc * KPC_M)
    valid = (
        (radius_rd >= 0.75)
        & (radius_rd <= 5.0)
        & (g_exact_inward > 0)
        & (g_algebraic > 0)
    )
    relative = g_algebraic[valid] / g_exact_inward[valid] - 1.0
    table = pd.DataFrame(
        {
            "galaxy": galaxy,
            "grid_size": grid_size,
            "radius_kpc": x[valid] / KPC_M,
            "radius_over_Rdisk": radius_rd[valid],
            "g_newton": g_newton_inward[valid],
            "g_qumond_exact": g_exact_inward[valid],
            "g_algebraic": g_algebraic[valid],
            "algebraic_relative_error": relative,
        }
    )
    summary = {
        "galaxy": galaxy,
        "mass_msun": float(mass_msun),
        "Rdisk_kpc": float(rd_kpc),
        "B": float(B),
        "grid_size": int(grid_size),
        "n_radial_samples": int(len(relative)),
        "median_absolute_fractional_error": float(np.median(np.abs(relative))),
        "maximum_absolute_fractional_error": float(np.max(np.abs(relative))),
        "mean_signed_fractional_error": float(np.mean(relative)),
    }
    return summary, table


def representative_sparc_disks(true_rdisk_csv, rotmod_directory=None):
    """Select low/median/high surface-density analytic SPARC reconstructions."""
    frame = pd.read_csv(true_rdisk_csv)
    frame = frame[(frame["Rdisk"] > 0) & (frame["L36"] > 0)].copy()
    frame["disk_mass_msun"] = 0.5 * frame["L36"] * 1e9
    frame["gas_mass_msun"] = 1.33 * frame["MHI"].clip(lower=0) * 1e9
    frame["baryonic_mass_msun"] = frame["disk_mass_msun"] + frame["gas_mass_msun"]
    frame["surface_density_proxy"] = frame["baryonic_mass_msun"] / (
        2.0 * np.pi * frame["Rdisk"] ** 2
    )
    frame = frame.sort_values("surface_density_proxy").reset_index(drop=True)
    selected = []
    for label, quantile in (("low", 0.1), ("median", 0.5), ("high", 0.9)):
        index = int(round(quantile * (len(frame) - 1)))
        row = frame.iloc[index]
        selected.append(
            {
                "surface_density_class": label,
                "galaxy": row["Name"],
                "Rdisk_kpc": float(row["Rdisk"]),
                "baryonic_mass_msun": float(row["baryonic_mass_msun"]),
                "surface_density_proxy_msun_kpc2": float(row["surface_density_proxy"]),
                "construction": "analytic exponential disk using catalog M/L=0.5 plus 1.33 M_HI",
            }
        )
    return selected


def run_representative_disks(true_rdisk_csv, *, grid_size=64, B=1.0):
    summaries = []
    tables = []
    selections = representative_sparc_disks(true_rdisk_csv)
    for selected in selections:
        summary, table = compare_axisymmetric_disk(
            selected["baryonic_mass_msun"],
            selected["Rdisk_kpc"],
            B=B,
            grid_size=grid_size,
            galaxy=selected["galaxy"],
        )
        summary.update(selected)
        summaries.append(summary)
        tables.append(table)
    return summaries, pd.concat(tables, ignore_index=True)
