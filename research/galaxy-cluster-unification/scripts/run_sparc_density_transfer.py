#!/usr/bin/env python3
"""Transfer the RAR/RG bridge fits to SPARC under declared density models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.data import KPC_M
from voidscreen.phenomenology import response_enhancement
from voidscreen.sparc_morphology import parse_sparc_profile

M_SUN_G = 1.988409870698051e33
PC_CM = 3.085677581491367e18
MSUN_PC3_TO_G_CM3 = M_SUN_G / PC_CM**3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hernquist_density_msun_pc3(
    total_mass_msun, radius_kpc, scale_kpc
) -> np.ndarray:
    mass = np.asarray(total_mass_msun, dtype=float)
    radius = np.asarray(radius_kpc, dtype=float)
    scale = np.asarray(scale_kpc, dtype=float)
    mass, radius, scale = np.broadcast_arrays(mass, radius, scale)
    valid = (
        np.isfinite(mass)
        & np.isfinite(radius)
        & np.isfinite(scale)
        & (mass > 0.0)
        & (radius > 0.0)
        & (scale > 0.0)
    )
    density = np.zeros_like(radius)
    density[valid] = (
        mass[valid]
        * scale[valid]
        / (2.0 * math.pi * radius[valid] * np.power(radius[valid] + scale[valid], 3))
        / 1.0e9
    )
    return density


def attach_surface_brightness(frame: pd.DataFrame, sparc_dir: Path) -> pd.DataFrame:
    blocks = []
    for galaxy, block in frame.groupby("galaxy", sort=False):
        profile = parse_sparc_profile(sparc_dir / "rotmod" / f"{galaxy}_rotmod.dat")
        lookup = pd.DataFrame(
            {
                "radius_catalog_kpc": profile.radius_kpc,
                "disk_surface_brightness": profile.disk_surface_brightness,
                "bulge_surface_brightness": profile.bulge_surface_brightness,
                "gas_velocity_component_km_s": profile.gas_velocity_km_s,
                "disk_velocity_unit_ml_km_s": profile.disk_velocity_unit_ml_km_s,
                "bulge_velocity_unit_ml_km_s": profile.bulge_velocity_unit_ml_km_s,
            }
        )
        joined = block.merge(
            lookup, on="radius_catalog_kpc", how="left", validate="many_to_one"
        )
        blocks.append(joined)
    output = pd.concat(blocks, ignore_index=True)
    required = [
        "disk_surface_brightness",
        "bulge_surface_brightness",
        "gas_velocity_component_km_s",
        "disk_velocity_unit_ml_km_s",
        "bulge_velocity_unit_ml_km_s",
    ]
    if output[required].isna().any().any():
        raise ValueError("failed to match SPARC surface-brightness rows")
    return output.sort_values(["galaxy_index", "radius_catalog_kpc"]).reset_index(drop=True)


def construct_density(
    frame: pd.DataFrame,
    *,
    disk_hz_over_scale: float,
    gas_rhi_divisor: float,
    gas_hz_over_scale: float,
    include_gas: bool,
) -> pd.DataFrame:
    if min(disk_hz_over_scale, gas_rhi_divisor, gas_hz_over_scale) <= 0.0:
        raise ValueError("density geometry factors must be positive")
    output = frame.copy()
    distance = output["distance_scale"].to_numpy(dtype=float)
    disk_half_thickness_pc = (
        disk_hz_over_scale
        * output["disk_scale_kpc"].to_numpy(dtype=float)
        * distance
        * 1000.0
    )
    disk_density = (
        output["disk_mass_to_light"].to_numpy(dtype=float)
        * output["disk_surface_brightness"].to_numpy(dtype=float)
        / (2.0 * disk_half_thickness_pc)
    )
    bulge_mass = (
        output["bulge_luminosity_fit_solar"].fillna(0.0).to_numpy(dtype=float)
        * output["bulge_mass_to_light"].to_numpy(dtype=float)
        * distance**2
    )
    bulge_scale = (
        output["bulge_scale_fit_kpc"].fillna(0.0).to_numpy(dtype=float) * distance
    )
    bulge_density = hernquist_density_msun_pc3(
        bulge_mass,
        output["radius_adjusted_kpc"].to_numpy(dtype=float),
        bulge_scale,
    )
    gas_density = np.zeros(len(output))
    if include_gas:
        rhi = output["HI_radius_kpc"].to_numpy(dtype=float)
        valid = np.isfinite(rhi) & (rhi > 0.0)
        gas_scale_kpc = np.zeros(len(output))
        gas_scale_kpc[valid] = rhi[valid] / gas_rhi_divisor * distance[valid]
        gas_mass = (
            1.33
            * output["HI_mass_billion_solar"].to_numpy(dtype=float)
            * 1.0e9
            * distance**2
        )
        surface_msun_kpc2 = np.zeros(len(output))
        surface_msun_kpc2[valid] = (
            gas_mass[valid]
            / (2.0 * math.pi * np.square(gas_scale_kpc[valid]))
            * np.exp(
                -output.loc[valid, "radius_adjusted_kpc"].to_numpy(dtype=float)
                / gas_scale_kpc[valid]
            )
        )
        gas_density[valid] = (
            surface_msun_kpc2[valid]
            / 1.0e6
            / (2.0 * gas_hz_over_scale * gas_scale_kpc[valid] * 1000.0)
        )
    output["disk_density_g_cm3"] = disk_density * MSUN_PC3_TO_G_CM3
    output["bulge_density_g_cm3"] = bulge_density * MSUN_PC3_TO_G_CM3
    output["gas_density_g_cm3"] = gas_density * MSUN_PC3_TO_G_CM3
    output["local_density_g_cm3"] = np.maximum(
        output[
            ["disk_density_g_cm3", "bulge_density_g_cm3", "gas_density_g_cm3"]
        ].sum(axis=1),
        1.0e-35,
    )
    return output


def velocity_prediction(
    frame: pd.DataFrame, model: str, parameters, constants: dict, *, coherence=0.0
) -> np.ndarray:
    gbar = frame["g_bar_m_s2"].to_numpy(dtype=float)
    enhancement = response_enhancement(
        model,
        gbar,
        frame["local_density_g_cm3"].to_numpy(dtype=float),
        frame["radius_adjusted_kpc"].to_numpy(dtype=float),
        parameters,
        rar_acceleration_m_s2=float(constants.get("rar_acceleration_m_s2", 1.2e-10)),
        fixed_gate_log10_phi_c=float(constants.get("fixed_gate_log10_phi_c", -6.3)),
        fixed_gate_sharpness=float(constants.get("fixed_gate_sharpness", 4.0)),
        coherence=coherence,
        coherence_gate_power=float(constants.get("coherence_gate_power", 2.0)),
    )
    acceleration = gbar * enhancement
    return np.sqrt(acceleration * frame["radius_adjusted_kpc"].to_numpy(dtype=float) * KPC_M) / 1000.0


def metrics(frame: pd.DataFrame, prediction: np.ndarray, rar_prediction: np.ndarray) -> dict:
    observed = frame["velocity_observed_adjusted_kms"].to_numpy(dtype=float)
    sigma = frame["velocity_error_total_kms"].to_numpy(dtype=float)
    residual = prediction - observed
    extra = prediction - rar_prediction
    return {
        "points": len(frame),
        "galaxies": int(frame["galaxy"].nunique()),
        "chi2_per_point": float(np.mean(np.square(residual / sigma))),
        "RMSE_km_s": float(np.sqrt(np.mean(np.square(residual)))),
        "MAE_km_s": float(np.mean(np.abs(residual))),
        "mean_standardized_residual": float(np.mean(residual / sigma)),
        "median_extra_velocity_vs_RAR_km_s": float(np.median(extra)),
        "p95_extra_velocity_vs_RAR_km_s": float(np.percentile(extra, 95.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs" / "sparc_density_transfer_protocol.json"
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
        "--rescue-report", type=Path, default=ROOT / "results" / "rar_rg_rescue_sweep" / "report.json"
    )
    parser.add_argument(
        "--protected-report", type=Path, default=ROOT / "results" / "rar_fixed_potential_gate_sweep" / "report.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "sparc_density_transfer"
    )
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    rescue = json.loads(args.rescue_report.read_text(encoding="utf-8"))
    protected = json.loads(args.protected_report.read_text(encoding="utf-8"))
    predictions = pd.read_csv(args.predictions)
    predictions = predictions[predictions["split"] == "outer_holdout"].copy()
    galaxy = pd.read_csv(args.galaxy_parameters)
    morphology = pd.read_csv(args.morphology)
    frame = predictions.merge(galaxy, on="galaxy", how="left", validate="many_to_one")
    frame = frame.merge(morphology, on="galaxy", how="left", validate="many_to_one")
    frame = attach_surface_brightness(frame, args.sparc)
    if len(frame) != 968 or frame["galaxy"].nunique() != 131:
        raise ValueError("SPARC transfer did not preserve the frozen outer sample")

    rar_prediction = frame["velocity_predicted_kms"].to_numpy(dtype=float)
    models = {
        "RAR_RG_additive": rescue["full_sample_descriptive_fits"]["RAR_RG_additive"]["parameters"],
        "RAR_RG_product": rescue["full_sample_descriptive_fits"]["RAR_RG_product"]["parameters"],
        "RAR_potential_gated_RG": rescue["full_sample_descriptive_fits"]["RAR_potential_gated_RG"]["parameters"],
        "RAR_fixed_potential_gated_RG": protected["full_sample_descriptive_fits"]["RAR_fixed_potential_gated_RG"]["parameters"],
    }
    constants = {
        "rar_acceleration_m_s2": 1.2e-10,
        "fixed_gate_log10_phi_c": -6.3,
        "fixed_gate_sharpness": 4.0,
    }
    density = protocol["density_model"]
    scenarios = [("stellar_only", 0.2, 3.2, 0.1, False)]
    for disk_hz in density["disk_hz_over_Rdisk_sensitivity"]:
        for divisor in density["gas_RHI_divisor_sensitivity"]:
            for gas_hz in density["gas_hz_over_Rgas_sensitivity"]:
                scenarios.append(
                    (
                        f"diskhz_{disk_hz}_rhidiv_{divisor}_gashz_{gas_hz}",
                        float(disk_hz),
                        float(divisor),
                        float(gas_hz),
                        True,
                    )
                )

    results = {}
    saved_primary = None
    primary_name = "diskhz_0.2_rhidiv_3.2_gashz_0.1"
    for name, disk_hz, divisor, gas_hz, include_gas in scenarios:
        constructed = construct_density(
            frame,
            disk_hz_over_scale=disk_hz,
            gas_rhi_divisor=divisor,
            gas_hz_over_scale=gas_hz,
            include_gas=include_gas,
        )
        record = {
            "density_log10_range": list(
                map(
                    float,
                    [
                        np.log10(constructed["local_density_g_cm3"]).min(),
                        np.log10(constructed["local_density_g_cm3"]).max(),
                    ],
                )
            ),
            "fixed_RAR": metrics(constructed, rar_prediction, rar_prediction),
            "models": {},
        }
        for model, parameter_record in models.items():
            parameter_names = (
                ["epsilon_0", "log10_rho_c_g_cm3", "Q", "log10_phi_c", "k_phi"]
                if model == "RAR_potential_gated_RG"
                else ["epsilon_0", "log10_rho_c_g_cm3", "Q"]
            )
            values = [parameter_record[key] for key in parameter_names]
            predicted = velocity_prediction(constructed, model, values, constants)
            record["models"][model] = metrics(constructed, predicted, rar_prediction)
            if name == primary_name:
                constructed[f"predicted_{model}_km_s"] = predicted
        results[name] = record
        if name == primary_name:
            saved_primary = constructed

    primary = results[primary_name]
    rar_metrics = primary["fixed_RAR"]
    gates = protocol["advance_gate"]
    gate_audit = {}
    for model, record in primary["models"].items():
        gate_audit[model] = {
            "RMSE": record["RMSE_km_s"]
            <= gates["outer_RMSE_relative_to_RAR_max"] * rar_metrics["RMSE_km_s"],
            "chi2": record["chi2_per_point"]
            <= gates["outer_chi2_per_point_relative_to_RAR_max"]
            * rar_metrics["chi2_per_point"],
            "extra_velocity": abs(record["median_extra_velocity_vs_RAR_km_s"])
            <= gates["median_absolute_extra_velocity_km_s_max"],
        }
        gate_audit[model]["passes_all"] = all(gate_audit[model].values())

    report = {
        "status": "completed fixed SPARC density-transfer sensitivity",
        "inputs": {
            "protocol_sha256": sha256(args.protocol),
            "predictions_sha256": sha256(args.predictions),
            "galaxy_parameters_sha256": sha256(args.galaxy_parameters),
            "morphology_sha256": sha256(args.morphology),
            "rescue_report_sha256": sha256(args.rescue_report),
            "protected_report_sha256": sha256(args.protected_report),
        },
        "sample": {"galaxies": 131, "outer_points": 968},
        "primary_scenario": primary_name,
        "primary": primary,
        "sensitivity": results,
        "gate_audit": gate_audit,
        "claim_boundary": protocol["claim_boundary"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    assert saved_primary is not None
    saved_primary.to_csv(args.output / "primary_predictions.csv", index=False)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"primary": primary, "gate_audit": gate_audit}, indent=2))


if __name__ == "__main__":
    main()
