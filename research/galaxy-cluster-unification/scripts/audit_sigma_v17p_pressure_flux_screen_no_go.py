"""Audit the conserved-potential obstruction for pressure kinetic screens."""

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
from scipy.integrate import quad
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def solve_flux(source: float, power: int, tolerance: float = 1e-13) -> float:
    """Solve t(1+t**power)=source for the unique nonnegative t."""

    if source <= 0.0:
        return 0.0
    upper = max(1.0, source, 2.0 * source ** (1.0 / (power + 1.0)))
    return float(
        brentq(
            lambda value: value * (1.0 + value**power) - source,
            0.0,
            upper,
            xtol=tolerance,
            rtol=4.0 * np.finfo(float).eps,
        )
    )


def potential_ratio(
    charge: float,
    power: int,
    observer_y: float,
    log_margin: float,
    root_tolerance: float,
    quadrature_relative_tolerance: float,
) -> float:
    """Return U_sigma/U_N from the exact spherical screened acceleration."""

    log_lower = math.log(observer_y)
    log_upper = max(math.log(math.sqrt(max(charge, 1e-30))) + log_margin, log_margin)

    def log_integrand(log_y: float) -> float:
        y = math.exp(log_y)
        field = solve_flux(charge / y**2, power, root_tolerance)
        return field * y

    integral = quad(
        log_integrand,
        log_lower,
        log_upper,
        epsabs=1e-11,
        epsrel=quadrature_relative_tolerance,
        limit=1000,
    )[0]
    linear_tail = charge * math.exp(-log_upper)
    return float(observer_y * (integral + linear_tail))


def load_clash_floor(config: dict[str, Any]) -> dict[str, float | int]:
    entry = config["spent_diagnostic_input"]
    frame = pd.read_csv(
        ROOT / entry["path"],
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
    a_sigma = float(config["numerical_controls"]["a_sigma_m_s2"])
    frame["x"] = 10.0**frame.log_g_bar / a_sigma
    frame["extra_fraction"] = 10.0 ** (frame.log_g_total - frame.log_g_bar) - 1.0
    relevant = frame[
        frame.extra_fraction
        >= float(config["numerical_controls"]["minimum_cluster_extra_fraction"])
    ]
    return {
        "points": len(frame),
        "points_requiring_order_unity_extra": len(relevant),
        "minimum_x_all": float(frame.x.min()),
        "minimum_x_requiring_order_unity_extra": float(relevant.x.min()),
        "minimum_extra_fraction": float(frame.extra_fraction.min()),
    }


def build_report(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    locked_entries = [
        config["parent"],
        config["pressure_archetype"],
        config["retired_derivative_metric_class"],
    ]
    hashes_verified = (
        all(
            sha256(ROOT / entry["protocol"]) == entry["sha256"]
            and sha256(ROOT / entry["report"]) == entry["report_sha256"]
            for entry in locked_entries
        )
        and sha256(ROOT / config["spent_diagnostic_input"]["path"])
        == config["spent_diagnostic_input"]["sha256"]
    )
    if not hashes_verified:
        raise RuntimeError("v17P parent or diagnostic hash changed")

    pressure_report = json.loads(
        (ROOT / config["pressure_archetype"]["report"]).read_text(encoding="utf-8")
    )
    pi_cluster = float(pressure_report["pressure_compactness"]["cluster_most_favorable"])
    pi_sun = float(pressure_report["pressure_compactness"]["solar_conservative_virial"])
    pressure_ratio = pi_sun / pi_cluster

    controls = config["numerical_controls"]
    constants = config["physical_constants"]
    a_sigma = float(controls["a_sigma_m_s2"])
    solar_mond_radius_m = math.sqrt(
        float(constants["gravitational_constant_m3_kg_s2"])
        * float(constants["solar_mass_kg"])
        / a_sigma
    )
    observer_m = float(controls["solar_observer_radius_au"]) * float(constants["metres_per_au"])
    observer_y = observer_m / solar_mond_radius_m
    cluster_x_floor = float(controls["cluster_x_floor"])
    gamma_lower_bound = 2.0 * observer_y * math.sqrt(pressure_ratio * cluster_x_floor)
    cassini = float(config["gates"]["maximum_cassini_absolute_gamma_minus_one"])
    crossing_radius_au = (
        math.sqrt(pressure_ratio / cluster_x_floor)
        * solar_mond_radius_m
        / float(constants["metres_per_au"])
    )

    clash = load_clash_floor(config)
    clash_gate = bool(
        clash["minimum_x_requiring_order_unity_extra"] >= cluster_x_floor
        and clash["points_requiring_order_unity_extra"] == clash["points"]
    )

    representative = []
    maximum_flux_residual = 0.0
    for cluster_x in controls["representative_cluster_x"]:
        cluster_x = float(cluster_x)
        for power in controls["representative_flux_powers"]:
            power = int(power)
            alpha_squared = (1.0 + cluster_x**power) / pi_cluster
            solar_charge = alpha_squared * pi_sun
            source_at_observer = solar_charge / observer_y**2
            field_at_observer = solve_flux(
                source_at_observer, power, float(controls["root_tolerance"])
            )
            flux_residual = abs(
                field_at_observer * (1.0 + field_at_observer**power) - source_at_observer
            ) / max(source_at_observer, 1.0)
            maximum_flux_residual = max(maximum_flux_residual, flux_residual)
            potential = potential_ratio(
                solar_charge,
                power,
                observer_y,
                float(controls["potential_integral_log_upper_margin"]),
                float(controls["root_tolerance"]),
                float(controls["quadrature_relative_tolerance"]),
            )
            curve_bound = 2.0 * observer_y * math.sqrt(pressure_ratio * cluster_x)
            representative.append(
                {
                    "cluster_x": cluster_x,
                    "power": power,
                    "alpha": math.sqrt(alpha_squared),
                    "solar_dimensionless_charge": solar_charge,
                    "solar_local_force_fraction_at_observer": field_at_observer * observer_y**2,
                    "solar_gamma_force_proxy": 2.0 * field_at_observer * observer_y**2,
                    "solar_potential_fraction_at_observer": potential,
                    "solar_gamma_potential_proxy": 2.0 * potential,
                    "analytic_gamma_lower_bound": curve_bound,
                    "potential_bound_verified": bool(2.0 * potential >= curve_bound),
                    "cassini_pass": bool(2.0 * potential <= cassini),
                    "normalized_flux_residual": flux_residual,
                }
            )

    analytic_pass = gamma_lower_bound <= cassini
    representative_pass = all(row["cassini_pass"] for row in representative)
    residual_pass = maximum_flux_residual <= float(config["gates"]["maximum_flux_residual"])
    all_bounds_verified = all(row["potential_bound_verified"] for row in representative)
    if clash_gate and residual_pass and all_bounds_verified and not analytic_pass:
        outcome = "retire_monotone_shift_symmetric_pressure_flux_screen"
    elif analytic_pass and representative_pass and residual_pass:
        outcome = "retain_pressure_flux_screen_for_full_action_gate"
    else:
        outcome = "audit_inconclusive"

    minimum_numeric_gamma = min(row["solar_gamma_potential_proxy"] for row in representative)
    return {
        "report_version": config["protocol_version"],
        "status": "completed_pressure_flux_screen_no_go_audit",
        "generated_utc": datetime.now(UTC).isoformat(),
        "protocol": config_path.relative_to(ROOT).as_posix(),
        "protocol_sha256": sha256(config_path),
        "hashes_verified": hashes_verified,
        "holdout_opened": False,
        "empirical_fit_performed": False,
        "clash_spent_diagnostic": clash,
        "pressure_compactness": {
            "cluster_most_favorable": pi_cluster,
            "solar_conservative": pi_sun,
            "solar_to_cluster_ratio": pressure_ratio,
        },
        "analytic_no_go": {
            "solar_mond_radius_au": solar_mond_radius_m / float(constants["metres_per_au"]),
            "observer_y": observer_y,
            "cluster_x_floor": cluster_x_floor,
            "solar_crossing_radius_at_floor_au": crossing_radius_au,
            "gamma_potential_lower_bound": gamma_lower_bound,
            "cassini_limit": cassini,
            "minimum_excess_factor": gamma_lower_bound / cassini,
            "analytic_gate_pass": analytic_pass,
        },
        "representative_curves": representative,
        "numerics": {
            "maximum_normalized_flux_residual": maximum_flux_residual,
            "flux_residual_gate_pass": residual_pass,
            "all_potential_bounds_verified": all_bounds_verified,
            "minimum_numeric_gamma_potential_proxy": minimum_numeric_gamma,
            "minimum_numeric_excess_factor": minimum_numeric_gamma / cassini,
        },
        "selection": {
            "clash_floor_gate_pass": clash_gate,
            "analytic_cassini_gate_pass": analytic_pass,
            "representative_cassini_gate_pass": representative_pass,
            "outcome": outcome,
            "next_mechanism": (
                "A successor must change the source-integrated Solar scalar charge "
                "with a healthy potential-dependent mechanism, or reset the pressure channel; "
                "another monotone kinetic curve is not authorized."
            ),
        },
        "claim_boundary": [
            "This is a necessary weak-field potential bound, not a complete nonlinear PPN solution.",
            "The CLASH accelerations are spent NFW-deprojected diagnostics, not raw lensing evidence.",
            "The result retires a local conserved monotone kinetic-flux screen for the reciprocal pressure metric, not all pressure-sourced or charge-changing theories.",
            "K-mouflage, AQUAL, TeVeS-like metrics, and conformal-disformal transformations are published prior art.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v17p_pressure_flux_screen_no_go.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "sigma_v17p_pressure_flux_screen_no_go",
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
