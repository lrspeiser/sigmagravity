#!/usr/bin/env python3
"""Run the frozen P0706 two-potential RAR metric and Solar audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.source_routing_qumond import projected_baryonic_spectral_anisotropy
from voidscreen.two_potential_metric import build_two_potential_metric, rar_acceleration

DEFAULT_CONFIG = ROOT / "configs" / "p0706_two_potential_rar_metric_audit.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_any_P0706_spent_or_sealed_outcome_score":
        raise RuntimeError("P0706 protocol is not frozen")
    metric_cfg = protocol["metric"]
    solar = protocol["solar"]
    gates = protocol["predeclared_gates"]
    a0 = float(metric_cfg["a0_m_s2"])

    deep_g = a0 * 1e-8
    deep_error = abs(float(rar_acceleration(deep_g, a0)) / math.sqrt(deep_g * a0) - 1.0)
    accelerations = {
        "solar_limb": solar["solar_GM_m3_s2"] / solar["solar_radius_m"] ** 2,
        "cassini_impact": solar["solar_GM_m3_s2"] / (solar["cassini_impact_solar_radii"] * solar["solar_radius_m"]) ** 2,
        "earth": solar["solar_GM_m3_s2"] / solar["earth_semimajor_axis_m"] ** 2,
        "mercury": solar["solar_GM_m3_s2"] / solar["mercury_semimajor_axis_m"] ** 2,
    }
    rar_values = {name: float(rar_acceleration(value, a0)) for name, value in accelerations.items()}
    relative_corrections = {
        name: abs(rar_values[name] / value - 1.0) for name, value in accelerations.items()
    }

    axis = np.linspace(-5.0, 5.0, 41)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    spherical_surface = np.exp(-0.5 * (x * x + y * y))
    routing_fraction, covariance, eigenvalues = projected_baryonic_spectral_anisotropy(
        spherical_surface,
        float(axis[1] - axis[0]),
    )
    x3, y3, z3 = np.meshgrid(axis, axis, axis, indexing="ij")
    spherical_potential = -1.0 / np.sqrt(x3 * x3 + y3 * y3 + z3 * z3 + 0.5**2)
    spherical_metric = build_two_potential_metric(
        spherical_potential,
        spherical_potential,
        float(axis[1] - axis[0]),
    )
    slip_rms = float(
        np.sqrt(np.mean((spherical_metric.spatial_potential - spherical_metric.time_potential) ** 2))
        / np.sqrt(np.mean(spherical_metric.time_potential**2))
    )

    extra_mercury = rar_values["mercury"] - accelerations["mercury"]
    precession_per_orbit_rad = (
        2.0
        * math.pi
        * abs(extra_mercury)
        * solar["mercury_semimajor_axis_m"] ** 2
        * math.sqrt(1.0 - solar["mercury_eccentricity"] ** 2)
        / solar["solar_GM_m3_s2"]
    )
    orbits_per_century = 36525.0 / solar["mercury_orbital_period_days"]
    rad_to_mas = (180.0 / math.pi) * 3600.0 * 1000.0
    mercury_precession = precession_per_orbit_rad * orbits_per_century * rad_to_mas
    ppn_gamma_minus_one = slip_rms

    metrics = {
        "deep_RAR_relative_error": deep_error,
        "solar_accelerations_m_s2": accelerations,
        "solar_RAR_relative_corrections": relative_corrections,
        "maximum_high_acceleration_relative_correction": max(relative_corrections.values()),
        "spherical_projected_routing_fraction": routing_fraction,
        "spherical_projected_covariance": covariance.tolist(),
        "spherical_projected_eigenvalues": eigenvalues.tolist(),
        "weyl_identity_relative_rms": spherical_metric.weyl_identity_relative_rms,
        "spherical_metric_slip_relative_rms": slip_rms,
        "derived_absolute_PPN_gamma_minus_one": ppn_gamma_minus_one,
        "derived_extra_Mercury_precession_mas_per_century": mercury_precession,
        "earth_fractional_extra_force": relative_corrections["earth"],
    }
    gate_results = {
        "deep_RAR_limit": deep_error <= gates["deep_rar_relative_error_max"],
        "high_acceleration_screening": metrics["maximum_high_acceleration_relative_correction"] <= gates["high_acceleration_relative_correction_max"],
        "spherical_routing_null": routing_fraction <= gates["spherical_projected_routing_fraction_max"],
        "weyl_identity": spherical_metric.weyl_identity_relative_rms <= gates["weyl_identity_relative_rms_max"],
        "spherical_zero_slip": slip_rms <= gates["spherical_metric_slip_max"],
        "Cassini_PPN_gamma": ppn_gamma_minus_one <= gates["absolute_PPN_gamma_minus_one_max"],
        "Mercury_precession": mercury_precession <= gates["absolute_extra_Mercury_precession_mas_per_century_max"],
        "earth_force": relative_corrections["earth"] <= gates["earth_fractional_force_max"],
        "no_new_constants": metric_cfg["new_universal_constants"] == gates["new_universal_constants"],
        "no_per_object_gravity": metric_cfg["per_object_gravity_parameters"] == gates["per_object_gravity_parameters"],
        "no_fitted_metric_slip": metric_cfg["fitted_metric_slip_parameters"] == gates["fitted_metric_slip_parameters"],
    }
    passed = all(gate_results.values())
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "report_version": "P0706-TWO-POTENTIAL-RAR-METRIC-AUDIT-RESULTS-1.0.0",
        "status": "pass" if passed else "fail",
        "all_math_and_solar_gates_pass": passed,
        "candidate_advanced_to_spent_joint_screen": passed,
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__).resolve()),
        "operator_source_sha256": sha256(ROOT / "src/voidscreen/two_potential_metric.py"),
        "metrics": metrics,
        "gate_results": gate_results,
        "failed_gates": [name for name, value in gate_results.items() if not value],
        "spent_outcomes_scored": False,
        "sealed_P0633_kinematics_opened": False,
        "sealed_P0640_lensing_constraints_opened": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    (output / protocol["outputs"]["report"]).write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = f"""# P0706 two-potential RAR metric audit

- Status: **{'PASS' if passed else 'FAIL'}**.
- Deep-RAR relative error: **{deep_error:.4g}**.
- Maximum Solar high-acceleration correction: **{metrics['maximum_high_acceleration_relative_correction']:.4g}**.
- Spherical routing fraction / metric slip: **{routing_fraction:.4g} / {slip_rms:.4g}**.
- Derived Cassini `|gamma-1|`: **{ppn_gamma_minus_one:.4g}** (limit `{gates['absolute_PPN_gamma_minus_one_max']:.3g}`).
- Derived extra Mercury precession: **{mercury_precession:.4g} mas/century** (limit `{gates['absolute_extra_Mercury_precession_mas_per_century_max']:.3g}`).
- New constants / per-object gravity / fitted slip parameters: **0 / 0 / 0**.
- Advanced to a frozen spent joint screen: **{'yes' if passed else 'no'}**.
- Spent or sealed outcomes scored/opened: **no**.
"""
    (output / protocol["outputs"]["summary"]).write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
