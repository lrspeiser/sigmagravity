"""Audit Solar/cluster selectivity of a pressure-sourced symmetron."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model_s(path: Path, constants: dict[str, float]) -> pd.DataFrame:
    values = np.loadtxt(path)
    frame = pd.DataFrame(
        values,
        columns=[
            "radius_fraction",
            "sound_speed_cm_s",
            "density_g_cm3",
            "pressure_dyn_cm2",
            "gamma_1",
            "temperature_k",
        ],
    )
    frame = frame[(frame.radius_fraction >= 0.0) & (frame.radius_fraction <= 1.0)]
    frame = frame.sort_values("radius_fraction").drop_duplicates("radius_fraction")
    frame["pressure_pa"] = frame.pressure_dyn_cm2 * 0.1
    frame["pressure_mass_density_kg_m3"] = (
        3.0 * frame.pressure_pa / float(constants["speed_of_light_m_s"]) ** 2
    )
    return frame.reset_index(drop=True)


def solar_pressure_compactness(frame: pd.DataFrame, constants: dict[str, float]) -> float:
    radius = frame.radius_fraction.to_numpy(float) * float(constants["solar_radius_m"])
    density = frame.pressure_mass_density_kg_m3.to_numpy(float)
    pressure_mass = np.trapezoid(4.0 * math.pi * radius**2 * density, radius)
    return float(pressure_mass / float(constants["solar_mass_kg"]))


def load_cluster_diagnostics(
    path: Path,
    pressure_compactness: float,
    constants: dict[str, float],
) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        names=[
            "system",
            "radius_kpc",
            "log_g_bar",
            "log_g_total",
            "err_log_g_bar",
            "err_log_g_total",
        ],
    )
    radius_m = frame.radius_kpc.to_numpy(float) * float(constants["metres_per_kpc"])
    g_bar = 10.0 ** frame.log_g_bar.to_numpy(float)
    frame["extra_fraction"] = 10.0 ** (frame.log_g_total - frame.log_g_bar) - 1.0
    frame["pressure_column"] = (
        pressure_compactness * g_bar * radius_m / float(constants["speed_of_light_m_s"]) ** 2
    )
    frame["enclosed_mean_pressure_density_kg_m3"] = (
        pressure_compactness
        * 3.0
        * g_bar
        / (4.0 * math.pi * float(constants["gravitational_constant_m3_kg_s2"]) * radius_m)
    )
    return frame


def solve_optimistic_solar_profile(
    solar: pd.DataFrame,
    critical_density: float,
    range_kpc: float,
    constants: dict[str, float],
    max_step: float,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> dict[str, float | bool]:
    radius = solar.radius_fraction.to_numpy(float)
    density = solar.pressure_mass_density_kg_m3.to_numpy(float)
    density_interpolator = PchipInterpolator(radius, density, extrapolate=False)
    b = float(constants["solar_radius_m"]) / (range_kpc * float(constants["metres_per_kpc"]))
    coefficient = 0.5 * b**2
    epsilon = 1e-8

    def source_density(x: float) -> float:
        return max(float(density_interpolator(min(max(x, 0.0), 1.0))), 0.0)

    def field_rhs(x: float, field: float) -> float:
        return coefficient * ((source_density(x) / critical_density - 1.0) * field + field**3)

    def integrate(central_field: float):
        center_rhs = field_rhs(0.0, central_field)
        initial = [
            central_field + center_rhs * epsilon**2 / 6.0,
            center_rhs * epsilon / 3.0,
        ]
        return solve_ivp(
            lambda x, state: [
                state[1],
                field_rhs(x, state[0]) - 2.0 * state[1] / max(x, 1e-30),
            ],
            (epsilon, 1.0),
            initial,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
            max_step=max_step,
        )

    def boundary_residual(central_field: float) -> float:
        solution = integrate(central_field)
        surface_field, surface_derivative = solution.y[:, -1]
        return float(surface_derivative - (1.0 + b) * (1.0 - surface_field))

    central_field = float(brentq(boundary_residual, 0.0, 1.0, xtol=1e-13))
    solution = integrate(central_field)
    surface_field, surface_derivative = solution.y[:, -1]
    scalar_charge = 1.0 - float(surface_field)

    grid = np.linspace(0.0, 1.0, 20001)
    grid_density = np.maximum(density_interpolator(grid), 0.0)
    if b > 1e-12:
        kernel = np.sinh(b * grid) / b
    else:
        kernel = grid
    unscreened_charge = float(
        math.exp(-b)
        * np.trapezoid(
            grid * coefficient * grid_density / critical_density * kernel,
            grid,
        )
    )
    screening_fraction = scalar_charge / unscreened_charge
    return {
        "solver_success": bool(solution.success),
        "central_field_fraction": central_field,
        "surface_field_fraction": float(surface_field),
        "surface_derivative": float(surface_derivative),
        "boundary_residual": boundary_residual(central_field),
        "scalar_charge": scalar_charge,
        "linear_unscreened_charge": unscreened_charge,
        "screening_fraction": screening_fraction,
        "range_to_solar_radius_ratio": 1.0 / b,
    }


def build_report(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    parent = config["parent"]
    hashes_verified = (
        sha256(ROOT / parent["protocol"]) == parent["sha256"]
        and sha256(ROOT / parent["report"]) == parent["report_sha256"]
        and all(
            sha256(ROOT / entry["path"]) == entry["sha256"] for entry in config["inputs"].values()
        )
    )
    if not hashes_verified:
        raise RuntimeError("v17Q parent or input hash changed")

    constants = config["physical_constants"]
    solar = load_model_s(ROOT / config["inputs"]["solar_model_s"]["path"], constants)
    pi_sun = solar_pressure_compactness(solar, constants)
    parent_report = json.loads((ROOT / parent["report"]).read_text(encoding="utf-8"))
    pi_cluster = float(parent_report["pressure_compactness"]["cluster_most_favorable"])
    cluster = load_cluster_diagnostics(
        ROOT / config["inputs"]["clash_spent_profiles"]["path"],
        pi_cluster,
        constants,
    )

    solar_column = (
        pi_sun
        * float(constants["gravitational_constant_m3_kg_s2"])
        * float(constants["solar_mass_kg"])
        / (float(constants["solar_radius_m"]) * float(constants["speed_of_light_m_s"]) ** 2)
    )
    column_ratio = cluster.pressure_column.to_numpy(float) / solar_column
    required = cluster[
        cluster.extra_fraction >= float(config["gates"]["minimum_cluster_extra_weyl_fraction"])
    ]
    required_ratio = required.pressure_column.to_numpy(float) / solar_column
    column_gate = bool(np.all(required_ratio >= 1.0) and len(required) == len(cluster))
    cassini = float(config["gates"]["maximum_cassini_absolute_gamma_minus_one"])
    gamma_column_lower_bound = 2.0 * pi_sun / pi_cluster

    control = config["optimistic_profile_control"]
    critical_density = float(cluster.enclosed_mean_pressure_density_kg_m3.max())
    profile = solve_optimistic_solar_profile(
        solar,
        critical_density,
        float(control["minimum_vacuum_range_kpc"]),
        constants,
        float(control["radial_max_step"]),
        float(control["relative_tolerance"]),
        float(control["absolute_tolerance"]),
    )
    refined = solve_optimistic_solar_profile(
        solar,
        critical_density,
        float(control["minimum_vacuum_range_kpc"]),
        constants,
        0.5 * float(control["radial_max_step"]),
        float(control["relative_tolerance"]),
        float(control["absolute_tolerance"]),
    )
    resolution_change = abs(
        float(refined["screening_fraction"]) - float(profile["screening_fraction"])
    ) / max(abs(float(refined["screening_fraction"])), 1e-30)
    cluster_strength_alpha_squared = 1.0 / pi_cluster
    gamma_profile_proxy = (
        2.0 * cluster_strength_alpha_squared * pi_sun * float(profile["screening_fraction"])
    )
    required_solar_screening = cassini / (2.0 * cluster_strength_alpha_squared * pi_sun)

    numerics_pass = bool(
        profile["solver_success"]
        and abs(float(profile["boundary_residual"]))
        <= float(control["boundary_residual_tolerance"])
        and resolution_change <= float(control["resolution_change_tolerance"])
    )
    column_pass = gamma_column_lower_bound <= cassini
    profile_pass = gamma_profile_proxy <= cassini
    if column_gate and numerics_pass and not column_pass and not profile_pass:
        outcome = "retire_standard_pressure_symmetron_and_reset_direct_pressure_metric"
    elif column_pass and profile_pass and numerics_pass:
        outcome = "retain_pressure_symmetron_for_full_action_gate"
    else:
        outcome = "audit_inconclusive"

    return {
        "report_version": config["protocol_version"],
        "status": "completed_pressure_symmetron_no_go_audit",
        "generated_utc": datetime.now(UTC).isoformat(),
        "protocol": config_path.relative_to(ROOT).as_posix(),
        "protocol_sha256": sha256(config_path),
        "hashes_verified": hashes_verified,
        "holdout_opened": False,
        "empirical_fit_performed": False,
        "solar_model_s": {
            "radial_points": len(solar),
            "pressure_compactness": pi_sun,
            "central_pressure_pa": float(solar.pressure_pa.iloc[0]),
            "central_pressure_mass_density_kg_m3": float(solar.pressure_mass_density_kg_m3.iloc[0]),
            "pressure_column": solar_column,
        },
        "cluster_spent_diagnostic": {
            "points": len(cluster),
            "points_requiring_order_unity_extra": len(required),
            "pressure_compactness": pi_cluster,
            "minimum_pressure_column": float(cluster.pressure_column.min()),
            "median_pressure_column": float(cluster.pressure_column.median()),
            "maximum_pressure_column": float(cluster.pressure_column.max()),
            "minimum_cluster_to_solar_column_ratio": float(column_ratio.min()),
            "median_cluster_to_solar_column_ratio": float(np.median(column_ratio)),
            "maximum_cluster_to_solar_column_ratio": float(column_ratio.max()),
            "largest_enclosed_mean_pressure_density_kg_m3": critical_density,
        },
        "pressure_column_no_go": {
            "all_required_cluster_columns_at_least_solar": column_gate,
            "gamma_lower_bound": gamma_column_lower_bound,
            "cassini_limit": cassini,
            "excess_factor": gamma_column_lower_bound / cassini,
            "cassini_gate_pass": column_pass,
        },
        "optimistic_model_s_control": {
            **profile,
            "critical_density_kg_m3": critical_density,
            "vacuum_range_kpc": float(control["minimum_vacuum_range_kpc"]),
            "resolution_change": resolution_change,
            "numerics_gate_pass": numerics_pass,
            "required_screening_fraction": required_solar_screening,
            "gamma_proxy_at_cluster_strength_coupling": gamma_profile_proxy,
            "gamma_excess_factor": gamma_profile_proxy / cassini,
            "cassini_gate_pass": profile_pass,
        },
        "selection": {
            "pressure_column_ordering_gate_pass": column_gate,
            "pressure_column_cassini_gate_pass": column_pass,
            "optimistic_profile_cassini_gate_pass": profile_pass,
            "outcome": outcome,
            "same_solar_cluster_gate_failures": [
                "v17G_unscreened_pressure_metric",
                "v17P_conserved_kinetic_flux_screen",
                "v17Q_symmetry_restoring_charge_screen",
            ],
            "mechanism_reset_triggered": outcome.startswith("retire_standard_pressure_symmetron"),
            "next_mechanism": (
                "Reset the direct pressure-only reciprocal metric. The next root action "
                "must not add a fourth pressure screen; it must obtain galaxy dynamics "
                "and lensing from a different baryon-forced field mechanism."
            ),
        },
        "claim_boundary": [
            "The pressure-column theorem applies to the standard symmetron thin-shell ordering, not every possible non-derivative scalar potential.",
            "The CLASH products are spent NFW-deprojected diagnostics, not raw lensing evidence.",
            "The Model S profile is an authoritative Solar-model input; the cluster pressure compactness deliberately assumes the hottest gas and all baryons in that phase, favoring the candidate.",
            "Symmetron screening and conformal-disformal/TeVeS-like metrics are published prior art; the pressure-sourced combination is the tested project hypothesis.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v17q_pressure_symmetron_no_go.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "sigma_v17q_pressure_symmetron_no_go",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = build_report(args.config, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
