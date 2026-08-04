from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_causal_polarization import (
    local_transport_eigenvalues,
    maximum_characteristic_speed,
    transition_bandpass,
)
from voidscreen.sigma_nonmetricity import (
    nonminimal_scalar_weak_laplacians,
    weyl_trace_nonmetricity,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the theory-only Sigma v5A action.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v5a_causal_polarization_action_audit.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v5a_causal_polarization_action_audit",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    scan = config["analytic_scan"]
    gates_config = config["gates"]
    ratio = np.geomspace(
        float(scan["nonmetricity_ratio_min"]),
        float(scan["nonmetricity_ratio_max"]),
        int(scan["points"]),
    )

    rng = np.random.default_rng(int(scan["random_seed"]))
    grad_psi = rng.normal(size=(4000, 3))
    grad_phi = rng.normal(size=(4000, 3))
    expected_weyl = 4.0 * np.sum(np.square(grad_psi + grad_phi), axis=1)
    measured_weyl = weyl_trace_nonmetricity(grad_psi, grad_phi)
    weyl_error = float(np.max(np.abs(measured_weyl - expected_weyl)))

    baryonic = rng.normal(size=4000)
    scalar = rng.normal(size=4000)
    plain_response = nonminimal_scalar_weak_laplacians(baryonic, scalar)
    plain_weyl_error = float(
        np.max(np.abs(plain_response["photon_weyl"] - baryonic))
    )

    acceleration = np.asarray(scan["acceleration_limit_values"], dtype=float)
    source = transition_bandpass(acceleration)
    transport_records = []
    minimum_kinetic = np.inf
    maximum_speed = 0.0
    for alpha in scan["anisotropy_values"]:
        for orientation in ("spacelike", "timelike"):
            eigenvalues = local_transport_eigenvalues(
                ratio, float(alpha), orientation=orientation
            )
            local_minimum = min(
                float(np.min(eigenvalues["time_magnitude"])),
                float(np.min(eigenvalues["parallel_spatial"])),
                float(np.min(eigenvalues["transverse_spatial"])),
            )
            local_speed = float(
                np.max(
                    maximum_characteristic_speed(
                        ratio, float(alpha), orientation=orientation
                    )
                )
            )
            minimum_kinetic = min(minimum_kinetic, local_minimum)
            maximum_speed = max(maximum_speed, local_speed)
            transport_records.append(
                {
                    "alpha_sigma": float(alpha),
                    "orientation": orientation,
                    "minimum_kinetic_eigenvalue": local_minimum,
                    "maximum_characteristic_speed_over_c": local_speed,
                }
            )

    singular_control = local_transport_eigenvalues(
        ratio,
        float(scan["singular_control_anisotropy"]),
        orientation="spacelike",
    )
    singular_control_minimum = min(
        float(np.min(singular_control["parallel_spatial"])),
        float(np.min(singular_control["transverse_spatial"])),
    )

    constants = len(config["physical_constants"])
    gates = {
        "weyl_trace_identity": weyl_error
        <= float(gates_config["maximum_weyl_trace_identity_error"]),
        "plain_nonminimal_scalar_lensing_null": plain_weyl_error
        <= float(gates_config["maximum_plain_nonminimal_scalar_weyl_error"]),
        "transition_peak": abs(source[1] - gates_config["required_transition_source_at_x_1"])
        <= 1.0e-15,
        "low_field_flat_limit": source[0]
        <= float(gates_config["maximum_transition_source_at_x_1e_minus_5"]),
        "high_field_solar_limit": source[2]
        <= float(gates_config["maximum_transition_source_at_x_1e5"]),
        "conditioned_local_kinetic_matrix": minimum_kinetic
        >= float(gates_config["minimum_kinetic_eigenvalue"]),
        "causal_local_scalar_cone": maximum_speed
        <= float(gates_config["maximum_characteristic_speed"]) + 1.0e-14,
        "parameter_economy": constants
        <= int(gates_config["maximum_universal_constant_count"]),
    }
    gates = {name: bool(value) for name, value in gates.items()}
    report = {
        "status": "completed theory-only Sigma v5A causal-polarization screen",
        "observational_data_accessed": False,
        "manufactured_checks": {
            "weyl_trace_identity_maximum_absolute_error": weyl_error,
            "plain_nonminimal_scalar_weyl_maximum_absolute_error": plain_weyl_error,
            "transition_bandpass": {
                str(value): float(result)
                for value, result in zip(acceleration, source, strict=True)
            },
            "minimum_scanned_kinetic_eigenvalue": minimum_kinetic,
            "singular_control_minimum_spatial_eigenvalue": singular_control_minimum,
            "maximum_scanned_characteristic_speed_over_c": maximum_speed,
            "universal_constant_count": constants,
        },
        "transport_scan": transport_records,
        "gates": gates,
        "all_current_theory_screen_gates_pass": bool(all(gates.values())),
        "unresolved_before_empirical_fit": config["unresolved_before_empirical_fit"],
        "decision": "continue_covariant_derivation_no_empirical_fit",
        "data_policy": config["data_policy"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
