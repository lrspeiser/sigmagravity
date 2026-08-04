from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v6_metric_memory import (
    detuned_massive_memory_step_response,
    detuned_static_tensor_transfer,
    repeated_massless_memory_step_response,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reject the v6B repeated pole and select the detuned v6C memory."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v6c_detuned_memory_selection.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v6c_detuned_memory_selection",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    audit = config["mode_audit"]
    wavenumber = float(audit["wavenumber"])
    early_time = (
        np.pi / 2.0 + 2.0 * np.pi * int(audit["peak_index_early"])
    ) / wavenumber
    late_time = (
        np.pi / 2.0 + 2.0 * np.pi * int(audit["peak_index_late"])
    ) / wavenumber
    early_response = abs(float(repeated_massless_memory_step_response(early_time, wavenumber)))
    late_response = abs(float(repeated_massless_memory_step_response(late_time, wavenumber)))
    secular_ratio = late_response / early_response

    end_time = float(audit["periods_for_bounded_scan"]) * 2.0 * np.pi / wavenumber
    time = np.linspace(0.0, end_time, int(audit["samples_for_bounded_scan"]))
    detuned_records = []
    maximum_bound_ratio = 0.0
    for mass_ratio_value in audit["memory_mass_over_wavenumber"]:
        mass_ratio = float(mass_ratio_value)
        memory_mass = mass_ratio * wavenumber
        response = detuned_massive_memory_step_response(time, wavenumber, memory_mass)
        omega_squared = wavenumber**2 + memory_mass**2
        analytic_bound = 2.0 / omega_squared + 2.0 / memory_mass**2
        measured_maximum = float(np.max(np.abs(response)))
        bound_ratio = measured_maximum / analytic_bound
        maximum_bound_ratio = max(maximum_bound_ratio, bound_ratio)
        detuned_records.append(
            {
                "memory_mass_over_wavenumber": mass_ratio,
                "measured_maximum_absolute_response": measured_maximum,
                "analytic_absolute_bound": analytic_bound,
                "measured_to_bound_ratio": bound_ratio,
            }
        )

    wave = np.geomspace(1.0e-12, 1.0e12, 4000)
    transfer = detuned_static_tensor_transfer(wave, wavenumber)
    high_slice = slice(-500, None)
    high_slope = float(
        np.max(np.abs(np.gradient(np.log(transfer[high_slice]), np.log(wave[high_slice]))))
    )
    thresholds = config["gates"]
    gates = {
        "v6B_repeated_pole_secular_growth_detected": secular_ratio
        >= float(thresholds["minimum_v6b_late_to_early_peak_ratio"]),
        "v6C_detuned_responses_are_bounded": maximum_bound_ratio
        <= float(thresholds["maximum_detuned_response_over_analytic_bound"]),
        "v6C_static_transfer_is_bounded": float(np.min(transfer))
        >= float(thresholds["minimum_static_transfer"])
        and float(np.max(transfer)) <= float(thresholds["maximum_static_transfer"]),
        "v6C_static_transfer_has_no_UV_growth": high_slope
        <= float(thresholds["maximum_high_wavenumber_log_slope"]),
        "parameter_count": int(config["physical_parameters"]["count"])
        <= int(config["physical_parameters"]["maximum_allowed"]),
    }
    gates = {name: bool(value) for name, value in gates.items()}
    report = {
        "status": "completed Sigma v6B secular rejection and v6C pre-data selection",
        "observational_data_accessed": False,
        "raw_holdout_opened": False,
        "v6B": {
            "decision": "retire_exact_second_massless_inverse",
            "early_peak_time": early_time,
            "late_peak_time": late_time,
            "early_peak_absolute_response": early_response,
            "late_peak_absolute_response": late_response,
            "late_to_early_peak_ratio": secular_ratio,
            "analytic_response": "S[(1-cos kt)/k^2-t sin(kt)/(2k)]",
            "reason": "The repeated retarded pole produces a linearly growing oscillatory memory. Bounded coherence hides the amplitude divergence only by eventually saturating and erasing morphology contrast.",
        },
        "v6C": {
            "decision": "advance_detuned_massive_memory_to_complete_CTP_variation",
            "envelope": config["v6c_envelope"],
            "physical_parameters": config["physical_parameters"],
            "detuned_mode_records": detuned_records,
            "maximum_measured_to_analytic_bound_ratio": maximum_bound_ratio,
            "static_transfer": "k^2/(k^2+m_sigma^2)",
            "static_transfer_minimum": float(np.min(transfer)),
            "static_transfer_maximum": float(np.max(transfer)),
            "maximum_high_wavenumber_log_slope": high_slope,
            "why_it_evades_v6B": "m_sigma>0 moves the second response pole from omega^2=k^2 to omega^2=k^2+m_sigma^2, eliminating the resonance while retaining a causal, source-forced, bounded static orientation response.",
        },
        "gates": gates,
        "all_v6C_selection_gates_pass": bool(all(gates.values())),
        "not_yet_demonstrated": [
            "complete closed-time-path influence action",
            "diffeomorphism Ward identity",
            "positive spectrum on nonzero memory backgrounds",
            "acceptable preferred-time and FLRW behavior",
            "derived spherical MOND and lensing equations",
            "whether one universal L_sigma transfers between galaxies and clusters",
            "any observational performance"
        ],
        "data_policy": config["data_policy"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
