from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v7_vainshtein import (
    audit_spherical_vainshtein_carrier,
    carrier_range_for_transition_m,
)

M_SUN_KG = 1.98847e30
KPC_M = 3.085677581491367e19


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the spherical Sigma v7B screen.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v7b_spherical_vainshtein_gate.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v7b_spherical_vainshtein_gate",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    range_scan = config["range_scan"]
    mixing_scan = config["mixing_scan"]
    stress = config["no_label_density_stress_test"]
    protected = stress["protected_system"]
    target = stress["target_system"]
    ranges = np.geomspace(
        float(range_scan["minimum_kpc"]) * KPC_M,
        float(range_scan["maximum_kpc"]) * KPC_M,
        int(range_scan["points"]),
    )
    angles = np.linspace(
        float(mixing_scan["minimum_rad"]),
        float(mixing_scan["maximum_rad"]),
        int(mixing_scan["points"]),
    )
    protected_mass = float(protected["mass_solar"]) * M_SUN_KG
    protected_radius = float(protected["radius_kpc"]) * KPC_M
    target_mass = float(target["mass_solar"]) * M_SUN_KG
    target_radius = float(target["radius_kpc"]) * KPC_M
    audit = audit_spherical_vainshtein_carrier(
        carrier_ranges_m=ranges,
        mixing_angles_rad=angles,
        protected_mass_kg=protected_mass,
        protected_radius_m=protected_radius,
        target_mass_kg=target_mass,
        target_radius_m=target_radius,
        required_lensing_enhancement=float(
            config["gates"]["minimum_useful_lensing_enhancement"]
        ),
    )
    construction_gates = dict(audit["gates"])
    construction_gates["equal_density_identity"] = (
        float(
            audit["equal_density_stress_test"][
                "maximum_relative_screening_coordinate_difference"
            ]
        )
        <= float(config["gates"]["maximum_equal_density_screening_residual"])
    )
    construction_gates["parameter_count"] = (
        int(config["physical_parameters"]["count"])
        <= int(config["physical_parameters"]["maximum_allowed"])
    )
    construction_gates = {
        name: bool(value) for name, value in construction_gates.items()
    }
    protected_transition = carrier_range_for_transition_m(
        protected_radius, protected_mass
    )
    target_transition = carrier_range_for_transition_m(target_radius, target_mass)
    report = {
        "status": "completed Sigma v7B spherical Vainshtein gate",
        "observational_data_accessed": False,
        "raw_holdout_opened": False,
        "candidate": config["candidate"],
        "physical_parameter_count": int(config["physical_parameters"]["count"]),
        "analytic_scaling": {
            "vainshtein_radius": "r_V=(2 G M L^2/c^2)^(1/3)",
            "screening_coordinate": "r/r_V proportional to (r^3/M)^(1/3) L^(-2/3)",
            "transition_mean_density": "rho_bar=3 c^2/(8 pi G L^2)",
        },
        "stress_test": {
            "protected_system": protected,
            "target_system": target,
            "protected_transition_range_kpc": protected_transition / KPC_M,
            "target_transition_range_kpc": target_transition / KPC_M,
            **audit["equal_density_stress_test"],
        },
        "amplitude_bounds": {
            "maximum_unscreened_lensing_enhancement": float(
                audit["maximum_unscreened_lensing_enhancement"]
            ),
            "maximum_unscreened_dynamics_enhancement": float(
                audit["maximum_unscreened_dynamics_enhancement"]
            ),
            "minimum_useful_lensing_enhancement": float(
                config["gates"]["minimum_useful_lensing_enhancement"]
            ),
        },
        "gates": construction_gates,
        "all_v7b_gates_pass": bool(all(construction_gates.values())),
        "decision": "retire_spherical_v7B_as_unifying_carrier_control",
        "reason": "The healthy spherical screen depends only on M/r^3 and cannot distinguish equal-density disk and strong-lens archetypes without an object label. Independently, the one-metric bimetric mixing coefficients cap the fully unscreened lensing enhancement at 1.5, below the conservative factor-3 carrier target.",
        "scope": "This rejects the spherical transition control. It does not reject a full three-dimensional multi-source Vainshtein solve, whose Hessian nonlinearity may respond to component overlap and orientation rather than only enclosed mean density.",
        "next_mechanism_requirement": "Derive and solve the parameter-frozen three-dimensional helicity-0 Hessian equation from the ghost-free decoupling limit. Advance only if its branch is elliptic and positive and its response to separated baryonic components is not reducible to the spherical density screen.",
        "data_policy": config["data_policy"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
