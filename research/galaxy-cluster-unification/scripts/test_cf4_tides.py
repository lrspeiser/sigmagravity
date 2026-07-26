from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from voidscreen.data import KPC_M, pack_dataset
from voidscreen.environment import (
    CF4_H0_KMS_MPC,
    GRID_SPECS,
    density_acceleration_and_tidal_fields,
    load_density_grid,
    sample_density_grid,
)

ROOT = Path(__file__).resolve().parents[1]


def _sample_components(
    field: np.ndarray, points_hmpc: np.ndarray, box_size_hmpc: float
) -> np.ndarray:
    leading_shape = field.shape[:-3]
    sampled = [
        sample_density_grid(field[index], points_hmpc, box_size_hmpc=box_size_hmpc)
        for index in np.ndindex(leading_shape)
    ]
    return np.stack(sampled, axis=1).reshape(len(points_hmpc), *leading_shape)


def _outer_required_acceleration(packed) -> pd.DataFrame:
    gas_v2 = np.sign(packed.velocity_gas_kms) * packed.velocity_gas_kms**2
    baryonic_v2 = (
        gas_v2
        + 0.5 * packed.velocity_disk_unit_ml_kms**2
        + 0.7 * packed.velocity_bulge_unit_ml_kms**2
    )
    baryonic_v2 = np.maximum(baryonic_v2, 1e-8)
    radius_m = packed.radius_kpc * KPC_M
    g_bar = baryonic_v2 * 1e6 / radius_m
    g_observed = packed.velocity_observed_kms**2 * 1e6 / radius_m
    rows = []
    for galaxy_index, galaxy in enumerate(packed.galaxy_names):
        points = np.flatnonzero(packed.galaxy_index == galaxy_index)
        outer = points[-1]
        rows.append(
            {
                "galaxy": galaxy,
                "outer_radius_kpc": packed.radius_kpc[outer],
                "outer_g_bar_m_s2": g_bar[outer],
                "outer_g_observed_m_s2": g_observed[outer],
                "outer_required_extra_m_s2": max(g_observed[outer] - g_bar[outer], 0.0),
            }
        )
    return pd.DataFrame(rows)


def _positive_summary(values: np.ndarray) -> dict[str, float | int]:
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        return {"count": 0}
    return {
        "count": int(positive.size),
        "minimum": float(np.min(positive)),
        "median": float(np.median(positive)),
        "maximum": float(np.max(positive)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-parameter physical CF4 density-deficit tidal sanity check."
    )
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "raw" / "sparc")
    parser.add_argument(
        "--environment-csv",
        type=Path,
        default=ROOT / "data" / "derived" / "void_scores_cf4.csv",
    )
    parser.add_argument("--cf4", type=Path, default=ROOT / "data" / "raw" / "cosmicflows4")
    parser.add_argument("--omega-m", type=float, default=0.3)
    parser.add_argument("--padding-factor", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "cf4_tide_test")
    args = parser.parse_args()

    spec = GRID_SPECS[0]
    delta = load_density_grid(args.cf4, spec)
    acceleration, tidal = density_acceleration_and_tidal_fields(
        delta,
        box_size_hmpc=spec.box_size_hmpc,
        omega_m=args.omega_m,
        padding_factor=args.padding_factor,
    )
    environment = pd.read_csv(args.environment_csv)
    points = environment[["sgx_hmpc", "sgy_hmpc", "sgz_hmpc"]].to_numpy(dtype=float)
    acceleration_at_galaxy = _sample_components(acceleration, points, spec.box_size_hmpc)
    tidal_at_galaxy = _sample_components(tidal, points, spec.box_size_hmpc)
    eigenvalues = np.linalg.eigvalsh(tidal_at_galaxy)

    fixed_radius_m = 10.0 * KPC_M
    spectral_fixed = np.max(np.abs(eigenvalues), axis=1) * fixed_radius_m
    inward_fixed = np.maximum(-eigenvalues[:, 0], 0.0) * fixed_radius_m
    table = environment[["galaxy", "void_score_grouped_64"]].copy()
    table["uniform_field_m_s2"] = np.linalg.norm(acceleration_at_galaxy, axis=1)
    table["tidal_eigenvalue_min_s2"] = eigenvalues[:, 0]
    table["tidal_eigenvalue_mid_s2"] = eigenvalues[:, 1]
    table["tidal_eigenvalue_max_s2"] = eigenvalues[:, 2]
    table["tidal_trace_s2"] = np.trace(tidal_at_galaxy, axis1=1, axis2=2)
    table["tidal_spectral_acceleration_at_10kpc_m_s2"] = spectral_fixed
    table["max_inward_tidal_acceleration_at_10kpc_m_s2"] = inward_fixed

    packed = pack_dataset(args.data, environment_csv=args.environment_csv)
    table = table.merge(_outer_required_acceleration(packed), on="galaxy", how="inner")
    outer_radius_m = table["outer_radius_kpc"].to_numpy(dtype=float) * KPC_M
    table["max_inward_tidal_acceleration_at_outer_radius_m_s2"] = (
        np.maximum(-table["tidal_eigenvalue_min_s2"].to_numpy(dtype=float), 0.0) * outer_radius_m
    )
    required = table["outer_required_extra_m_s2"].to_numpy(dtype=float)
    inward_outer = table["max_inward_tidal_acceleration_at_outer_radius_m_s2"].to_numpy(dtype=float)
    valid = (required > 0.0) & (inward_outer > 0.0)
    table["inward_tide_to_required_ratio"] = np.nan
    table.loc[valid, "inward_tide_to_required_ratio"] = inward_outer[valid] / required[valid]
    table["orders_of_magnitude_shortfall"] = np.nan
    table.loc[valid, "orders_of_magnitude_shortfall"] = np.log10(
        required[valid] / inward_outer[valid]
    )

    underdense = table["void_score_grouped_64"] > 0.0
    grid_path = args.cf4 / spec.filename
    report = {
        "status": "completed zero-parameter ordinary-gravity tide check",
        "mechanism": "T0 CF4 density-deficit peculiar tidal field",
        "fit_parameters": 0,
        "h0_km_s_mpc": CF4_H0_KMS_MPC,
        "omega_m_scale_convention": args.omega_m,
        "padding_factor": args.padding_factor,
        "grid": {
            "file": spec.filename,
            "sha256": hashlib.sha256(grid_path.read_bytes()).hexdigest(),
            "shape": list(spec.shape),
            "box_size_hmpc": spec.box_size_hmpc,
        },
        "galaxies": len(table),
        "uniform_field_note": "Accelerates a galaxy and its contents together; excluded from the internal-force comparison.",
        "comparison_note": "The most compressive eigenvalue is an orientation-optimistic upper bound on an inward disk tide.",
        "uniform_field_m_s2": _positive_summary(table["uniform_field_m_s2"].to_numpy()),
        "spectral_tide_at_10kpc_m_s2": _positive_summary(
            table["tidal_spectral_acceleration_at_10kpc_m_s2"].to_numpy()
        ),
        "maximum_inward_tide_at_outer_radius_m_s2": _positive_summary(inward_outer),
        "outer_required_extra_m_s2": _positive_summary(required),
        "inward_tide_to_required_ratio": _positive_summary(
            table["inward_tide_to_required_ratio"].to_numpy(dtype=float)
        ),
        "orders_of_magnitude_shortfall": _positive_summary(
            table["orders_of_magnitude_shortfall"].to_numpy(dtype=float)
        ),
        "median_tidal_trace_s2": {
            "underdense": float(table.loc[underdense, "tidal_trace_s2"].median()),
            "overdense": float(table.loc[~underdense, "tidal_trace_s2"].median()),
        },
    }

    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / "galaxy_tides.csv", index=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    figure, axis = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    plotted = table.loc[valid]
    axis.scatter(
        plotted["outer_required_extra_m_s2"],
        plotted["max_inward_tidal_acceleration_at_outer_radius_m_s2"],
        c=plotted["void_score_grouped_64"],
        cmap="coolwarm",
        s=24,
        alpha=0.75,
    )
    bounds = [1e-18, 1e-8]
    axis.plot(bounds, bounds, color="black", linewidth=1, linestyle="--", label="equal")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(bounds)
    axis.set_ylim(bounds)
    axis.set_xlabel("Outer acceleration excess (m/s^2)")
    axis.set_ylabel("Maximum inward CF4 tide (m/s^2)")
    axis.legend()
    figure.savefig(args.output / "tide_vs_required.png", dpi=180)
    plt.close(figure)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
