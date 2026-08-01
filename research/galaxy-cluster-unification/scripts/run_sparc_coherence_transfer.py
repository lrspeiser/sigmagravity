#!/usr/bin/env python3
"""Test morphology-based coherence leakage for the RAR/coherence/RG survivor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_sparc_density_transfer import (
    attach_surface_brightness,
    construct_density,
    metrics,
    sha256,
    velocity_prediction,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs" / "sparc_coherence_transfer_protocol.json"
    )
    parser.add_argument(
        "--candidate-report", type=Path, default=ROOT / "results" / "rar_coherence_rg_sweep" / "report.json"
    )
    parser.add_argument(
        "--predictions", type=Path, default=ROOT / "results" / "initial_comparison" / "rar" / "predictions.csv"
    )
    parser.add_argument(
        "--galaxy-parameters", type=Path, default=ROOT / "results" / "initial_comparison" / "rar" / "galaxy_parameters.csv"
    )
    parser.add_argument(
        "--morphology", type=Path, default=ROOT / "data" / "derived" / "nbp0_sparc_morphology.csv"
    )
    parser.add_argument("--sparc", type=Path, default=ROOT / "data" / "raw" / "sparc")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "sparc_coherence_transfer"
    )
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate_report.read_text(encoding="utf-8"))
    candidate_model = protocol.get("candidate_model", "RAR_coherence_gated_RG")
    parameter_record = candidate["full_sample_descriptive_fits"][candidate_model][
        "parameters"
    ]
    parameters = [
        parameter_record["epsilon_0"],
        parameter_record["log10_rho_c_g_cm3"],
        parameter_record["Q"],
    ]

    raw_predictions = pd.read_csv(args.predictions)
    raw_predictions = raw_predictions[raw_predictions["split"] == "outer_holdout"].copy()
    galaxy = pd.read_csv(args.galaxy_parameters)
    morphology = pd.read_csv(args.morphology)
    frame = raw_predictions.merge(galaxy, on="galaxy", how="left", validate="many_to_one")
    frame = frame.merge(morphology, on="galaxy", how="left", validate="many_to_one")
    frame = attach_surface_brightness(frame, args.sparc)
    if len(frame) != 968 or frame["galaxy"].nunique() != 131:
        raise ValueError("coherence transfer lost the frozen SPARC sample")

    mappings = {
        "C_unity": np.ones(len(frame)),
        "C_one_minus_baryonic_BT": np.clip(
            1.0 - frame["baryonic_bulge_fraction"].fillna(0.0).to_numpy(dtype=float),
            0.0,
            1.0,
        ),
        "C_one_minus_stellar_BT": np.clip(
            1.0 - frame["stellar_bulge_fraction"].fillna(0.0).to_numpy(dtype=float),
            0.0,
            1.0,
        ),
    }
    gas_force = np.square(frame["gas_velocity_component_km_s"].to_numpy(dtype=float))
    disk_force = (
        frame["disk_mass_to_light"].to_numpy(dtype=float)
        * np.square(frame["disk_velocity_unit_ml_km_s"].to_numpy(dtype=float))
    )
    bulge_force = (
        frame["bulge_mass_to_light"].to_numpy(dtype=float)
        * np.square(frame["bulge_velocity_unit_ml_km_s"].to_numpy(dtype=float))
    )
    positive_force = gas_force + disk_force + bulge_force
    mappings["C_one_minus_local_bulge_force"] = np.clip(
        1.0 - np.divide(
            bulge_force,
            positive_force,
            out=np.zeros_like(bulge_force),
            where=positive_force > 0.0,
        ),
        0.0,
        1.0,
    )
    sensitivity = protocol["density_sensitivity"]
    scenarios = []
    for disk_hz in sensitivity["disk_hz_over_Rdisk"]:
        for divisor in sensitivity["gas_RHI_divisor"]:
            for gas_hz in sensitivity["gas_hz_over_Rgas"]:
                scenarios.append(
                    (
                        f"diskhz_{disk_hz}_rhidiv_{divisor}_gashz_{gas_hz}",
                        float(disk_hz),
                        float(divisor),
                        float(gas_hz),
                    )
                )
    primary_values = list(map(float, sensitivity["primary"]))
    primary_name = (
        f"diskhz_{primary_values[0]}_rhidiv_{primary_values[1]}_gashz_{primary_values[2]}"
    )
    rar_prediction = frame["velocity_predicted_kms"].to_numpy(dtype=float)
    constants = {
        "rar_acceleration_m_s2": 1.2e-10,
        "coherence_gate_power": float(protocol.get("coherence_gate_power", 2.0)),
    }
    results = {}
    primary_frame = None
    for name, disk_hz, divisor, gas_hz in scenarios:
        constructed = construct_density(
            frame,
            disk_hz_over_scale=disk_hz,
            gas_rhi_divisor=divisor,
            gas_hz_over_scale=gas_hz,
            include_gas=True,
        )
        record = {"mappings": {}}
        for mapping, coherence in mappings.items():
            predicted = velocity_prediction(
                constructed,
                candidate_model,
                parameters,
                constants,
                coherence=coherence,
            )
            record["mappings"][mapping] = metrics(
                constructed, predicted, rar_prediction
            )
            if name == primary_name:
                constructed[f"coherence_{mapping}"] = coherence
                constructed[f"predicted_{mapping}_km_s"] = predicted
        results[name] = record
        if name == primary_name:
            primary_frame = constructed

    rar_metrics = metrics(frame, rar_prediction, rar_prediction)
    gates = protocol["advance_gate"]
    gate_audit = {}
    for mapping, record in results[primary_name]["mappings"].items():
        gate_audit[mapping] = {
            "RMSE": record["RMSE_km_s"]
            <= gates["outer_RMSE_relative_to_RAR_max"] * rar_metrics["RMSE_km_s"],
            "chi2": record["chi2_per_point"]
            <= gates["outer_chi2_per_point_relative_to_RAR_max"]
            * rar_metrics["chi2_per_point"],
            "extra_velocity": abs(record["median_extra_velocity_vs_RAR_km_s"])
            <= gates["median_absolute_extra_velocity_km_s_max"],
        }
        gate_audit[mapping]["passes_all"] = all(gate_audit[mapping].values())

    sensitivity_ranges = {}
    for mapping in mappings:
        values = [record["mappings"][mapping] for record in results.values()]
        sensitivity_ranges[mapping] = {
            key: [float(min(row[key] for row in values)), float(max(row[key] for row in values))]
            for key in ("RMSE_km_s", "chi2_per_point", "median_extra_velocity_vs_RAR_km_s")
        }
    report = {
        "status": "completed morphology-coherence SPARC transfer",
        "inputs": {
            "protocol_sha256": sha256(args.protocol),
            "candidate_report_sha256": sha256(args.candidate_report),
            "predictions_sha256": sha256(args.predictions),
            "galaxy_parameters_sha256": sha256(args.galaxy_parameters),
            "morphology_sha256": sha256(args.morphology),
        },
        "candidate_parameters": parameter_record,
        "sample": {"galaxies": 131, "outer_points": 968},
        "fixed_RAR": rar_metrics,
        "primary_scenario": primary_name,
        "primary": results[primary_name],
        "sensitivity_ranges": sensitivity_ranges,
        "gate_audit": gate_audit,
        "claim_boundary": protocol["claim_boundary"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    assert primary_frame is not None
    primary_frame.to_csv(args.output / "primary_predictions.csv", index=False)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"primary": report["primary"], "gate_audit": gate_audit}, indent=2))


if __name__ == "__main__":
    main()
