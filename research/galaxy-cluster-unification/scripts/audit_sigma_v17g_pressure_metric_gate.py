"""Audit the unscreened pressure-only reciprocal one-metric completion."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cluster_pressure_compactness(
    temperature_kev: float,
    mean_molecular_weight: float,
    *,
    joule_per_kev: float,
    proton_mass_kg: float,
    speed_of_light_m_s: float,
) -> float:
    """Return 3 p/(rho c^2) for an isothermal nonrelativistic ideal gas."""

    return (
        3.0
        * temperature_kev
        * joule_per_kev
        / (mean_molecular_weight * proton_mass_kg * speed_of_light_m_s**2)
    )


def virial_pressure_compactness(
    mass_kg: float,
    radius_m: float,
    virial_coefficient: float,
    *,
    gravitational_constant_m3_kg_s2: float,
    speed_of_light_m_s: float,
) -> float:
    """Return the declared lower virial estimate for integral(3p dV)/(Mc^2)."""

    return (
        virial_coefficient
        * gravitational_constant_m3_kg_s2
        * mass_kg
        / (radius_m * speed_of_light_m_s**2)
    )


def minimum_alpha_for_fraction(required_fraction: float, compactness: float) -> float:
    return math.sqrt(required_fraction / compactness)


def gamma_deviation(alpha: float, compactness: float) -> float:
    return 2.0 * alpha**2 * compactness


def maximum_alpha_from_cassini(cassini_bound: float, compactness: float) -> float:
    return math.sqrt(cassini_bound / (2.0 * compactness))


def extra_weyl_fraction(alpha: float, compactness: float) -> float:
    return alpha**2 * compactness


def build_report(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    constants = config["physical_constants"]
    cluster = config["favorable_cluster_bound"]
    solar = config["solar_bound"]
    temperatures = cluster["temperature_keV_range"]

    cluster_compactness = {
        f"{temperature:g}_keV": cluster_pressure_compactness(
            temperature,
            cluster["mean_molecular_weight"],
            joule_per_kev=constants["joule_per_keV"],
            proton_mass_kg=constants["proton_mass_kg"],
            speed_of_light_m_s=constants["speed_of_light_m_s"],
        )
        for temperature in temperatures
    }
    most_favorable_pi = max(cluster_compactness.values())
    solar_pi = virial_pressure_compactness(
        solar["mass_kg"],
        solar["radius_m"],
        solar["conservative_virial_coefficient"],
        gravitational_constant_m3_kg_s2=constants["gravitational_constant_m3_kg_s2"],
        speed_of_light_m_s=constants["speed_of_light_m_s"],
    )
    required_fraction = cluster["minimum_required_extra_Weyl_fraction"]
    cassini_bound = solar["cassini_absolute_gamma_minus_one_max"]
    alpha_cluster = minimum_alpha_for_fraction(required_fraction, most_favorable_pi)
    solar_gamma_at_cluster_alpha = gamma_deviation(alpha_cluster, solar_pi)
    alpha_cassini = maximum_alpha_from_cassini(cassini_bound, solar_pi)
    cluster_fraction_at_cassini = extra_weyl_fraction(alpha_cassini, most_favorable_pi)

    minimum_length_m = (
        solar["minimum_nonzero_propagation_length_kpc"] * constants["metres_per_kpc"]
    )
    maximum_solar_radius_m = (
        solar["maximum_solar_control_radius_au"] * constants["metres_per_au"]
    )
    minimum_yukawa_transmission = math.exp(-maximum_solar_radius_m / minimum_length_m)

    cassini_at_cluster_alpha_pass = solar_gamma_at_cluster_alpha <= cassini_bound
    cluster_at_cassini_alpha_pass = cluster_fraction_at_cassini >= required_fraction
    advance = cassini_at_cluster_alpha_pass and cluster_at_cassini_alpha_pass

    return {
        "report_version": config["protocol_version"],
        "status": "passed" if advance else "failed_pre_fit",
        "generated_utc": datetime.now(UTC).isoformat(),
        "protocol": config_path.relative_to(ROOT).as_posix(),
        "protocol_sha256": sha256(config_path),
        "astronomical_target_opened": False,
        "empirical_fit_performed": False,
        "source_limit": {
            "dust_source_at_alpha_equals_beta": 0.0,
            "isotropic_pressure_source": "J_X=3 alpha p",
            "massive_particle_potential": "Psi=U_N",
            "weyl_potential": "W=U_N-c^2 alpha X/2",
            "extra_Weyl_fraction": "f_W=alpha^2 Pi_p",
            "PPN_gamma_deviation": "abs(gamma-1)=2 alpha^2 Pi_p",
        },
        "pressure_compactness": {
            "cluster_by_temperature": cluster_compactness,
            "cluster_most_favorable": most_favorable_pi,
            "solar_conservative_virial": solar_pi,
        },
        "reciprocal_bounds": {
            "minimum_alpha_for_cluster_gate": alpha_cluster,
            "solar_gamma_deviation_at_cluster_alpha": solar_gamma_at_cluster_alpha,
            "solar_gamma_excess_factor_over_Cassini": solar_gamma_at_cluster_alpha
            / cassini_bound,
            "maximum_Cassini_safe_alpha": alpha_cassini,
            "maximum_cluster_extra_Weyl_fraction_at_Cassini_alpha": (
                cluster_fraction_at_cassini
            ),
            "cluster_fraction_shortfall_factor": required_fraction
            / cluster_fraction_at_cassini,
            "ratio_is_independent_of_alpha": True,
        },
        "range_control": {
            "minimum_nonzero_length_kpc": solar[
                "minimum_nonzero_propagation_length_kpc"
            ],
            "maximum_solar_control_radius_au": solar["maximum_solar_control_radius_au"],
            "minimum_Yukawa_transmission_over_solar_control": minimum_yukawa_transmission,
            "maximum_fractional_range_suppression_over_solar_control": (
                1.0 - minimum_yukawa_transmission
            ),
            "finite_range_can_supply_solar_screening": False,
        },
        "gates": {
            "cluster_strength_at_Cassini_alpha_pass": cluster_at_cassini_alpha_pass,
            "Cassini_at_cluster_strength_alpha_pass": cassini_at_cluster_alpha_pass,
            "same_alpha_passes_both": advance,
            "advance": advance,
        },
        "decision": {
            "canonical_nonzero_length_pressure_metric_completion": (
                "advance" if advance else "reject_without_additional_screening"
            ),
            "failure_class": "Solar_System_metric_slip" if not advance else None,
            "scope": (
                "This rejects only the unscreened propagating alpha=beta reciprocal "
                "pressure channel. It does not reject measured random stress as the "
                "halo-scale source, a separately normalized algebraic source-local "
                "completion, or a derived nonlinear high-field screen."
            ),
            "next_action_if_v17f_nonzero_length_passes": (
                "Derive a healthy nonlinear kinetic screen before any holdout; the "
                "screen should share an existing universal acceleration scale if "
                "possible rather than add an object-dependent constant."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v17g_pressure_metric_gate.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "sigma_v17g_pressure_metric_gate",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = build_report(args.config, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
