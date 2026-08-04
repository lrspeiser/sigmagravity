from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v12a_general_covector import GeneralCovectorBackground
from voidscreen.sigma_v12a_reduced_energy import (
    common_time_energy_row,
    constrained_modal_energy_spectrum,
    scan_common_time_energy,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit constraint-solved modal energies for Sigma v12A."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v12a_reduced_energy.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v12a_reduced_energy",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fixed = config["fixed_values"]
    k_b = float(fixed["selected_k_b"])
    k_2 = float(fixed["selected_k_2"])
    strength = float(fixed["selected_orientation_strength"])
    tilt = float(fixed["sentinel_tilt_magnitude"])
    wave_number = float(fixed["wave_number_ratio"])
    growth_tolerance = float(fixed["principal_growth_threshold"])
    frequency_threshold = float(fixed["minimum_frequency_fraction"])
    energy_tolerance = float(fixed["relative_energy_tolerance"])

    flat_control = constrained_modal_energy_spectrum(
        GeneralCovectorBackground(
            scalar_covector=(1.0, 0.0, 0.0, 0.0),
            aether_spatial_covector=(1.0e-5, 0.0, 0.0),
            orientation_strength=strength,
            k_b=k_b,
            k_2=k_2,
        ),
        wave_number_ratio=wave_number,
        minimum_frequency_fraction=0.05,
        relative_energy_tolerance=energy_tolerance,
    )
    common_time_scan = scan_common_time_energy(
        k_b=k_b,
        k_2=k_2,
        orientation_strength=strength,
        tilt_magnitude=tilt,
        boost_velocities=tuple(
            float(value) for value in fixed["common_time_boost_velocities"]
        ),
        wave_angles_degrees=tuple(
            float(value)
            for value in fixed["common_time_wave_angles_degrees"]
        ),
        wave_number_ratio=wave_number,
        principal_growth_threshold=growth_tolerance,
        minimum_frequency_fraction=frequency_threshold,
        relative_energy_tolerance=energy_tolerance,
    )

    convergence = [
        common_time_energy_row(
            k_b=k_b,
            k_2=k_2,
            orientation_strength=strength,
            tilt_magnitude=tilt,
            boost_velocity=float(fixed["convergence_boost_velocity"]),
            wave_angles_degrees=tuple(
                float(value)
                for value in fixed["convergence_wave_angles_degrees"]
            ),
            wave_number_ratio=float(convergence_wave),
            principal_growth_threshold=growth_tolerance,
            minimum_frequency_fraction=frequency_threshold,
            relative_energy_tolerance=energy_tolerance,
        )
        for convergence_wave in fixed["convergence_wave_numbers"]
    ]

    orientation_scans = [
        {
            "orientation_strength": float(candidate_strength),
            **scan_common_time_energy(
                k_b=k_b,
                k_2=k_2,
                orientation_strength=float(candidate_strength),
                tilt_magnitude=tilt,
                boost_velocities=tuple(
                    float(value)
                    for value in fixed["orientation_scan_boost_velocities"]
                ),
                wave_angles_degrees=tuple(
                    float(value)
                    for value in fixed["orientation_scan_wave_angles_degrees"]
                ),
                wave_number_ratio=wave_number,
                principal_growth_threshold=growth_tolerance,
                minimum_frequency_fraction=frequency_threshold,
                relative_energy_tolerance=energy_tolerance,
            ),
        }
        for candidate_strength in fixed["orientation_strength_scan"]
    ]

    aether_rest_velocity = tilt / math.sqrt(1.0 + tilt**2)
    aest_parameter_rows = []
    for candidate_k_b in fixed["aest_k_b_values"]:
        for candidate_k_2 in fixed["aest_k_2_values"]:
            candidate_k_b = float(candidate_k_b)
            candidate_k_2 = float(candidate_k_2)
            scalar_speed_squared = (2.0 - candidate_k_b) / (
                candidate_k_2 * candidate_k_b
            )
            if not 0.0 < scalar_speed_squared <= 1.0:
                continue
            row = common_time_energy_row(
                k_b=candidate_k_b,
                k_2=candidate_k_2,
                orientation_strength=0.0,
                tilt_magnitude=tilt,
                boost_velocity=aether_rest_velocity,
                wave_angles_degrees=tuple(
                    float(value)
                    for value in fixed["aest_rest_frame_wave_angles_degrees"]
                ),
                wave_number_ratio=wave_number,
                principal_growth_threshold=growth_tolerance,
                minimum_frequency_fraction=frequency_threshold,
                relative_energy_tolerance=energy_tolerance,
            )
            aest_parameter_rows.append(
                {
                    "flat_scalar_speed_squared": scalar_speed_squared,
                    **row,
                }
            )

    selected_best = common_time_scan["best_maximin_time"]
    if selected_best is None:
        raise RuntimeError("selected v12A row has no kinematically valid common time")
    maximum_identity_residual = max(
        float(candidate["maximum_canonical_krein_identity_residual"])
        for candidate in common_time_scan["candidates"]
        if candidate["common_time_kinematically_valid"]
    )
    maximum_polynomial_residual = max(
        float(candidate["maximum_polynomial_residual"])
        for candidate in common_time_scan["candidates"]
        if candidate["common_time_kinematically_valid"]
    )
    inherited_scan = next(
        row for row in orientation_scans if row["orientation_strength"] == 0.0
    )
    verification_gates = {
        "flat_finite_frequency_control_positive": bool(
            flat_control["all_identified_finite_mode_energies_positive"]
        ),
        "finite_descriptor_constraints_solved": all(
            candidate["root_structure_preserved_all_directions"]
            for candidate in common_time_scan["candidates"]
            if candidate["common_time_kinematically_valid"]
        ),
        "canonical_energy_matches_krein_derivative": maximum_identity_residual
        <= float(fixed["canonical_krein_identity_tolerance"]),
        "modal_euler_residuals_controlled": maximum_polynomial_residual
        <= float(fixed["polynomial_residual_tolerance"]),
        "negative_mode_persists_across_wave_numbers": all(
            float(row["minimum_normalized_energy_all_directions"])
            < -energy_tolerance
            for row in convergence
        ),
        "negative_mode_present_without_dhost_coupling": bool(
            inherited_scan["best_maximin_time"] is not None
            and float(
                inherited_scan["best_maximin_time"][
                    "minimum_normalized_energy_all_directions"
                ]
            )
            < -energy_tolerance
        ),
        "orientation_strength_scan_has_no_energy_rescue": not any(
            bool(row["any_common_time_all_directions_positive_energy"])
            for row in orientation_scans
        ),
        "aest_parameter_rest_frame_scan_has_no_energy_rescue": not any(
            bool(row["all_directions_positive_energy"])
            for row in aest_parameter_rows
            if row["common_time_kinematically_valid"]
        ),
    }
    physical_energy_gates = {
        "selected_common_time_has_bounded_positive_modal_energy": bool(
            common_time_scan["any_common_time_all_directions_positive_energy"]
        ),
        "selected_best_maximin_energy_strictly_positive": float(
            selected_best["minimum_normalized_energy_all_directions"]
        )
        > energy_tolerance,
    }
    report = {
        "status": "Sigma v12A constraint-solved common-time modal-energy gate",
        "protocol_status": config["protocol_status"],
        "selected_row": {
            "k_b": k_b,
            "k_2": k_2,
            "orientation_strength": strength,
            "tilt_magnitude": tilt,
        },
        "energy_identity": {
            "canonical": "E=u^dagger(omega^2 K-B)u/4",
            "krein": "E=omega u^dagger(2 omega K-i C)u/4",
            "maximum_identity_residual": maximum_identity_residual,
        },
        "flat_control": flat_control,
        "common_time_scan": common_time_scan,
        "wave_number_convergence": convergence,
        "orientation_strength_scans": orientation_scans,
        "aest_parameter_rest_frame_rows": aest_parameter_rows,
        "verification_gates": verification_gates,
        "all_verification_gates_pass": bool(all(verification_gates.values())),
        "physical_energy_gates": physical_energy_gates,
        "all_physical_energy_gates_pass": bool(all(physical_energy_gates.values())),
        "selected_v12a_row_rejected_before_data": not bool(
            all(physical_energy_gates.values())
        ),
        "v12a_formulation_rejected_before_data": bool(
            all(verification_gates.values())
            and not all(physical_energy_gates.values())
        ),
        "post_v12_reset_material_formulation_failure_count": 1,
        "three_failure_mechanism_reset_triggered": False,
        "theory_viable": False,
        "observational_data_accessed": False,
        "raw_holdout_opened": False,
        "decision": (
            "Retire exact v12A before observations under the project's strict bounded-energy "
            "gate. The finite descriptor modes solve the quadratic constraints, canonical "
            "and Krein energies agree, and no scanned common time, DHOST strength, or AeST "
            "base pair removes the negative-energy oscillator. Count this as the first "
            "material formulation failure after the v12 mechanism reset, not as three."
        ),
        "interpretation": (
            "The failing oscillator deforms the inherited AeST zero/Jeans sector. It has a "
            "positive instantaneous kinetic contribution but a more negative potential "
            "contribution and a negative Krein signature. Published AeST work treats the "
            "aligned low-momentum mode as potentially Jeans-like; this project intentionally "
            "uses the stricter requirement of a bounded physical Hamiltonian on every "
            "claimed background."
        ),
        "scope_limit": config["scope_limit"],
        "next_formulation_requirement": (
            "A materially distinct successor must remove or constrain the AeST zero/Jeans "
            "sector at the action level while preserving one matter metric, Class-Ia or "
            "equivalent degeneracy, luminal tensors, no free dust state, and at most five "
            "universal constants. Do not tune v12A further or open observations."
        ),
        "data_policy": config["data_policy"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
